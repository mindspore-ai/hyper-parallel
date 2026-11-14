# PP Pipeline Parallelism Guide

**English** | [简体中文](pipeline_parallel.zh.md)

HyperParallel provides pipeline parallelism, partitioning a model by layers into stages distributed across devices. It supports GPipe, 1F1B, and VPP schedules, with advanced features such as compute-communication overlap (overlap_b_f), PP+FSDP, and P2P prefetch.

## Core Concepts

| Schedule | Description | Bubble ratio |
|----------|------|-------------|
| GPipe | Run all micro-batch forwards before backwards | Larger |
| 1F1B | Alternate one forward and one backward | Smaller |
| VPP (Interleaved 1F1B) | Interleaved assignment with multiple virtual stages per rank | Smallest |
| MPipe (demo) | Split multimodal pipelines into encoder and LLM backbone stages, transposing encoder computation across PP ranks for parallel execution | Eliminates modality bubbles |

## API Overview

| Interface | Description |
|------|------|
| `PipelineStage` | Pipeline stage wrapper supporting dx/dw computation |
| `Schedule1F1B` | 1F1B schedule |
| `ScheduleGPipe` | GPipe schedule |
| `ScheduleInterleaved1F1B` | VPP (interleaved 1F1B) schedule |
| `ScheduleMPipeTranspose` | MPipe Transpose schedule |
| `MetaStep` / `MetaStepType` / `BatchDimSpec` | Scheduling unit abstractions |
| `CommComputeOverlap` | Two-thread compute-communication overlap coordinator |
| `HookCoordinator` / `HookRole` | COMM-first rendezvous primitives |

---

## Basic Usage

### 1. Standard 1F1B Schedule

```python
from hyper_parallel import PipelineStage, Schedule1F1B

# Wrap a partitioned module in PipelineStage.
stage = PipelineStage(split_model, stage_index, stage_num=4)

# Select the pipeline parallel schedule.
schedule = Schedule1F1B(stage, micro_batch_num=8)

# Execute.
x = DTensor.from_local(local_x, x_layout)
losses = schedule.run(x)
```

### 2. VPP (Interleaved 1F1B) Schedule

```python
from hyper_parallel import PipelineStage, ScheduleInterleaved1F1B

# Each rank owns multiple virtual stages in VPP.
schedule = ScheduleInterleaved1F1B(stages, micro_batch_num=8)
losses = schedule.run(*inputs)
```

### 3. MPipe (Transpose) Schedule

```python
from hyper_parallel import PipelineStage, ScheduleMPipeTranspose

# MPipe places an encoder or dataload-only identity on every PP rank.
stage = PipelineStage(stage_module, stage_index=rank, stage_num=pp_size)

schedule = ScheduleMPipeTranspose(
    [stage],
    micro_batch_num=8,
    preprocess_module=preprocess_module,
    num_transpose_layers=transpose_layer_num,
)

losses = schedule.run(local_batch)
```

---

## PP+FSDP/HSDP Integration

Wrap each stage root with `fully_shard()` before constructing `PipelineStage`.
Nested units and `fully_shard([module1, module2])` are supported within that root;
a grouped scheduler is registered once. Ordinary and sharded stages may coexist,
but a plain stage root containing sharded children is rejected. Different stages
must own separate schedulers, parameter wrappers, and mutable parameter storage.
Tied weights across stages use independent copies and `SharedParameterInfo`.

Initialize each local chunk in this order: wrap and materialize parameters on the
application's device, set residency/synchronization and prefetch options, construct
its stage, then construct the schedule. Parameter identity, wrapping, and communication
groups must remain stable after construction.

```python
from hyper_parallel import PipelineStage, ScheduleInterleaved1F1B, fully_shard, init_device_mesh

mesh = init_device_mesh(
    "npu", (pp_size, replicate_size, shard_size),
    mesh_dim_names=("pp", "replicate", "shard"),
)
parameter_mesh = mesh["replicate", "shard"]
stages = []
for stage_index, stage_model in local_model_chunks:
    fully_shard(stage_model, mesh=parameter_mesh)
    stage_model.set_reshard_after_forward(True)
    stage_model.set_reshard_after_backward(True)
    stage_model.set_requires_gradient_sync(True)
    stages.append(PipelineStage(
        stage_model, stage_index, total_stage_num, device=device,
        group=mesh["pp"].get_group(), mesh=mesh["pp"],
    ))
schedule = ScheduleInterleaved1F1B(stages, micro_batch_num=8)
```

For pure FSDP, pass a one-dimensional parameter mesh. The three existing settings
are independent; their selected values give these common configurations:

| Configuration | Reshard after forward | Reshard after backward | Gradient synchronization |
| --- | --- | --- | --- |
| ZeRO1 | `False` | `False` | `False` |
| ZeRO2 | `False` | `False` | `True` |
| ZeRO3 | `True` | `True` | `True` |
| Rebuild parameters, synchronize at the end | `True` | `True` | `False` |

The schedule preserves `fully_shard()` defaults and per-unit user settings, including
settings made with `recurse=False`. It forces gradient synchronization and replica
AllReduce on each stage's last completed backward, determined from completed microbatch
identities rather than their numerical order. Earlier backwards disable replica
AllReduce. Parameters referenced by unfinished split weight backwards or explicitly
retained by an overlap callback temporarily remain unsharded.

Full backwards retain the module backward hooks. During a pipeline run, those hooks
delegate gradient processing to the PP communication controller, which calls the
existing HSDPState preparation and communication primitives. Parameter-owning units
keep the module-level ReduceScatter pipeline; empty units leave their children's tail
communication for later computation or finalization. Standalone FSDP keeps its existing
`post_backward()` path.

Split input backwards retain their graphs without gradient communication; weight
backwards submit a batch of reductions after all weight calculations finish. Parameter
AllGather follows normal module hooks and user-configured prefetch. There are no FSDP
MetaSteps or schedule strategy arguments. After communication completes, the schedule
restores stable parameter shards, original settings, and ordinary backward processing
before shared-parameter gradient synchronization.
Optimizer updates and checkpoint operations belong after a successful `schedule.run()`.
Gradients for `ignored_params` remain the caller's responsibility.

This combination supports GPipe, 1F1B, interleaved 1F1B, and non-reentrant activation
checkpointing. Managed stages require backward execution. Communication fusion,
parameter/optimizer offload, activation swap combined with sharding, MPipe, and
`torch.compile` are unsupported. Ordinary pipeline activation swap and standalone
FSDP/HSDP retain their own supported configurations.

---

## Compute-Communication Overlap (overlap_b_f)

overlap_b_f pairs steady-state BWD and FWD into composite steps. A two-thread coordinator enables physical concurrency between EP A2A and computation:

```python
from hyper_parallel import ScheduleInterleaved1F1B, MetaStepType

# 1. Construct a schedule with B/F overlap and P2P overlap enabled.
schedule = ScheduleInterleaved1F1B(
    stages, micro_batch_num, overlap_p2p=True, overlap_b_f=True,
)

# 2. Attach hooks to every MoE layer.
overlap = CommComputeOverlap()
for i, layer in enumerate(moe_layers):
    layer.experts.dispatch = overlap.wrap_dispatch(layer.experts.dispatch)
    layer.experts.combine = overlap.wrap_combine(
        layer.experts.combine, is_last_layer=(i == len(moe_layers) - 1),
    )

# 3. Register the OVERLAP_B_F callback.
def callback(step, ctx):
    bwd_step, fwd_step = step.sub_steps
    overlap.run(fwd_fn=..., bwd_fn=...)

schedule.register_custom_function(MetaStepType.OVERLAP_B_F, callback)

# 4. Execute.
losses = schedule.run(*inputs)
```

### P2P Prefetch

`overlap_p2p=True` issues inter-stage P2P send/recv one step ahead, naturally overlapping it with computation in the current step:

```python
schedule = ScheduleInterleaved1F1B(
    stages, micro_batch_num, overlap_p2p=True, overlap_b_f=True,
)
```

### batched P2P transport

With overlap_b_f, duplex P2P (batch_isend_irecv) is enabled by default to reduce P2P communication calls:

```python
# Duplex batched P2P is enabled by default.
# forward-side P2P prefetch via batch_isend_irecv
```

---

## PP Activation Swap

Pipeline parallelism can be combined with Activation Swap to offload activations within a stage to CPU:

```python
from hyper_parallel.core.activation_checkpoint import swap_wrapper

# Use swap within a PP stage.
for layer in stage_model.layers:
    stage_model.layers[layer] = swap_wrapper(layer)
```

---

## PipelineStage dx/dw Computation

PipelineStage supports independent computation of dx (input gradients) and dw (weight gradients):

```python
stage = PipelineStage(split_model, stage_index, stage_num=4)
# dx/dw are computed automatically during backward.
```

---

## variable-layer + mixed-recompute

overlap_b_f supports different layer counts across PP stages and mixed recomputation strategies:

```python
# Different stages may contain different numbers of layers.
# Mixed recomputation is handled correctly under overlap_b_f.
```

---

## Performance Recommendations

1. **micro_batch_num**: Typically set to 2×pp_size or 4×pp_size. Increasing it reduces bubbles but increases memory usage.
2. **overlap_b_f**: Enabling it for PP+EP can substantially reduce serialized compute-communication waits.
3. **overlap_p2p**: Enable it together with overlap_b_f to further reduce P2P communication latency.
4. **PP+FSDP**: Sharding parameters inside each stage with FSDP significantly reduces per-device memory usage.
5. **Batch divisibility**: The batch size must be divisible by micro_batch_num.

---

## Further Reading

- [PP overlap_b_f Design](./pipeline_parallel_overlap_b_f.md) — Detailed compute-communication overlap design
- [API Reference](../api/api_reference.md) — Detailed PP interfaces
