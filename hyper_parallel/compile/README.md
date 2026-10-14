# HyperParallel Graph Mode

Graph-mode architecture for automatic parallelization with FSDP.

> **Note**: The `compile/` subpackage relies on a patched autograd engine for
> joint-graph capture (see Limitations). Its `torch.*` imports — including
> `torch.fx.experimental` and `torch._guards` — are therefore intentional and
> carry a `forbidden-backend-import` suppression in
> `.jenkins/check/config/filter_pylint.txt`.

## Core Concept

**User**: Write model code + parallel configuration
**Framework**: Graph capture → DP/PP partitioning → Communication-compute overlap → Execution

## Architecture

### Layer 1: Configuration

```python
from hyper_parallel.compile import GraphParallelPlan, PassConfig

# Configure which modules to mark for FSDP
parallel_plan = GraphParallelPlan()
parallel_plan.fsdp_mark("tok_embeddings")
parallel_plan.fsdp_mark_pattern("layers.*")

# Parallel configuration
pass_config = PassConfig(enable_overlap=True)
```

### Layer 2: Graph Capture

`trace_model_graph` captures forward + backward into a joint FX graph:

- Parameters/buffers are static inputs (placeholders), not `get_attr` nodes
- `torch.autograd.grad` runs inside the traced function
- Uses `FakeTensorMode` + `make_fx` for symbolic tracing

### Layer 3: Pass Pipeline

```text
DeadCodeElimination → CanonicalizeGraph → FSDPPass → PpPass → AutoOverlapPass
```

**FSDPPass** (data-parallel, `PassConfig.dp_mode`):

- Identifies DP parameter placeholders via GraphParallelPlan
- Sinks `all_gather` to each parameter's first forward use (Shard → Replicate)
- Frees the replicated parameter after its last forward read, then re-gathers
  it for the backward and rematerializes any saved forward view of it
  (`reshard_after_forward`), so peak memory tracks the forward working set
- Inserts gradient reduction on gradient outputs — `reduce_scatter`
  (Replicate → Shard) for `"fsdp"` / `"hsdp"`, `all_reduce` on the
  `dp_replicate` axis for `"ddp"` / `"hsdp"`
- Physically shards live model parameters (dim 0) so optimizer is FSDP-agnostic
  (`"ddp"` replicates instead; non-divisible dim-0 params stay replicated
  rather than erroring)

Disable reshard with `PassConfig(fsdp_reshard_after_forward=False)`.

**PpPass** (pipeline-parallel, `PassConfig.pp_enabled`):

- Splits the (post-FSDP) joint graph to this rank's stage at module-FQN
  boundaries, exchanging boundary activations/gradients via P2P
- Installs a self-contained GPipe / 1F1B schedule (`pp_schedule`) as a
  `call_module` stub, so the trainer needs no PP wiring
- v1 is pure-PP (`pp_degree == world_size`, `fsdp_enabled=False`); PP+FSDP
  hybrids require a `mesh_context` carrying a pp dim

**AutoOverlapPass**:

- **Placeholder no-op today**: `enable_overlap` is accepted, but the pass
  does not yet reorder `wait_tensor` nodes. Sinking waits past independent
  compute is planned.

### Layer 4: Execution

```python
from hyper_parallel.compile import GraphTrainer

trainer = GraphTrainer(
    model=model,
    train_fn=train_fn,
    pass_config=pass_config,
    parallel_plan=parallel_plan,
)

# Compile on first batch, then run forward + backward + optimizer
trainer.train(dataloader, max_steps=100, log_interval=10)
```

The compiled graph is a `GraphModule` that:

- Takes sharded parameters as inputs
- Gathers them via AllGather at each step
- Computes forward + backward
- Scatters gradients via ReduceScatter
- Outputs loss + grads

## Usage

```python
from hyper_parallel.compile import GraphParallelPlan, GraphTrainer, PassConfig

# 1. Model
model = Llama3Model(config)

# 2. Parallel plan
parallel_plan = GraphParallelPlan()
parallel_plan.fsdp_mark_pattern("layers.*")

# 3. Trainer
trainer = GraphTrainer(
    model=model,
    train_fn=lambda m, x, y: m(x).loss(y),
    pass_config=PassConfig(enable_overlap=True),
    parallel_plan=parallel_plan,
)

# 4. Training
trainer.train(dataloader, max_steps=1000)
```

## Dynamic input shapes

Static tracing remains the default. Enable symbolic user inputs with one option:

```python
compiler = GraphCompiler(model=model, train_fn=train_fn, dynamic=True)
```

For precise control, pass `dynamic_arg_dims` to `GraphCompiler` or `GraphTrainer`:

```python
compiler = GraphCompiler(
    model=model,
    train_fn=lambda m, x, y: ((m(x) - y) ** 2).mean(),
    dynamic_arg_dims={"x": [0, 1], "y": [0, 1]},
)
# x: [batch, sequence, input_features]; y: [batch, sequence, output_features]
loss, loss_dict = compiler.forward_backward(x=x, y=y)
```

The first call captures one symbolic forward/backward graph. Later batches with
compatible batch/sequence lengths reuse it; gradients still accumulate into the
live parameters. `GraphTrainer` exposes the same options and owns optimizer steps.
Numerical and graph-reuse regressions live in
[the CPU unit tests](../../tests/ut/compile/test_dynamic_shapes.py).

For the existing **BaseTrainer + GraphCompiler** path, configure:

```yaml
compile:
  enabled: true
  use_joint_graph: true
  dynamic_arg_dims:
    model_inputs.input_ids: [0, 1]
    labels: [0, 1]
```

Add paths for other tensors whose sizes vary, such as
`model_inputs.attention_mask: [0, 1]`. Paths refer to the keyword arguments of
`train_fn`; BaseTrainer supplies `model_inputs` and `labels`. Use `dynamic: true`
instead of the mapping to automatically symbolize all user tensor dimensions.

- A mapping takes precedence over `dynamic`; unspecified dimensions stay static.
  An empty mapping selects no dynamic dimensions. Negative dimension indices and
  nested dictionary/list paths (`batch.items.0`) are supported. Registered pytree
  objects can use attribute paths. Invalid paths or dimensions raise `ValueError`.
- Parameters and buffers always keep concrete shapes, including with automatic
  input symbolization. No Dynamo marks are added to the caller's tensors.
- The tracer uses `ShapeEnv` and per-tensor symbolic contexts directly, because
  this path captures through `make_fx`, not Dynamo. Runtime checks enforce traced
  size/stride relationships, input metadata, tensor identity aliases and Python
  constants before executing the joint graph. Shape-dependent Python branches
  are guarded; they do not become arbitrary runtime control flow.
- Use a representative first batch with varying dimensions **greater than 1**.
  PyTorch specializes dimensions of size 0/1 and may constrain dimensions used
  by operators or Python control flow. Inputs outside these constraints raise
  an explanatory error. They are not automatically recompiled: FSDP compilation
  mutates live parameter shards, so blindly retracing would be unsafe.
- First-stage scope: symbolic FX execution, loss and parameter gradients,
  gradient accumulation and optimizer updates. CPU unit tests cover the
  BaseTrainer entry and FSDP graph rewriting with mocked communication.
  The [NPU text example](examples/automodel_text_graph/README.md)
  exercises dynamic token packing through TP2 + FSDP2 with Qwen3-0.6B. Python scalar arguments remain
  constants. Pipeline parallel with dynamic shapes is rejected before tracing.
  Compiled-kernel backends, CUDA/NPU graph capture, serialized graph caches and
  data-dependent output shapes are not added here.

Run the focused checks with:

```bash
python -m pytest tests/ut/compile/test_dynamic_shapes.py -q
```

The approach follows MagiCompiler's separation of dynamic and static dimensions
and reuse of a general symbolic graph, adapted to HyperParallel's joint-graph
tracer. It adds no runtime dependency on the separate MagiCompiler checkout.

## Lazy size specialization

Optionally generate concrete-size FX variants of the general joint graph:

```yaml
compile:
  enabled: true
  use_joint_graph: true
  dynamic: true
  compile_sizes: [116, 120, 127]
  compile_size_input: model_inputs.input_ids
  compile_size_dim: 1
  max_specializations: 8
```

`GraphCompiler` and `GraphTrainer` also accept these four options directly.
`compile_sizes` is opt-in and requires dynamic mode. If `compile_size_input` is
omitted, selection uses the first symbolic axis among flattened user inputs;
`compile_size_dim` only applies to an explicit path and supports negative axes.

The first execution uses the general graph, matching MagiCompiler's warmup
policy. Subsequent configured sizes lazily generate a variant and execute it.
Repeated signatures hit the cache; unconfigured sizes use the general graph.
The cache key includes all user tensor shapes, strides, offsets and metadata,
so another batch dimension at the same sequence length creates a separate
variant. New signatures at `max_specializations` capacity use the general graph.

General guards run before every dispatch, including cache hits. Guard failures
still raise errors; specialization does not expand the valid input domain.
Variants bind pure symbolic shape expressions and regenerate FX code from the
post-pass graph, without recapturing the model or executing tensor/communication
operations during generation. Parameters, token counts and RNG remain live.
This is FX specialization, not an Inductor or NPU kernel compilation backend;
unresolved expressions remain dynamic and no speedup is promised.

Inspect `compiler.specialization_stats` for compilations, cache hits, general and
specialized calls, folded nodes and capacity fallbacks. The snapshot counts the
current compiler instance; variants are not serialized across processes.

```bash
python -m pytest tests/ut/compile/test_size_specialization.py -q
```

See [the implementation walkthrough](../../size_specialization_development.md)
and the [NPU training example](examples/automodel_text_graph/README.md).

## Key Design Decisions

1. **Static Inputs**: Parameters are graph inputs, not `get_attr`. This allows passes to split the graph by reshaping placeholders.

2. **Joint Graph**: Forward + backward captured together via `torch.autograd.grad` inside the traced function.

3. **FSDP-Agnostic Optimizer**: FSDPPass shards the live model's parameters in place. `model.parameters()` returns shards, so optimizer needs no FSDP awareness.

4. **Declarative Sharding**: GraphParallelPlan uses FQN patterns (`layers.*`) instead of imperative module wrapping.

## Parallelism support

| Axis | Status | Notes |
|------|--------|-------|
| FSDP / DDP / HSDP | Implemented | `FSDPPass`; `PassConfig.dp_mode` selects the mode |
| PP | Implemented | `PpPass` + `pp_schedule` (`gpipe` / `1f1b`); pure-PP v1 |
| TP (+ SP / LP) | Via `mesh_context` | TP collectives are baked into automodel boundary forwards; graph mode reuses the mesh |
| EP | Planned | — |

## Limitations

- **Non-divisible parameters are skipped, not errors**: a parameter whose dim 0
  is not divisible by the DP shard degree (or a scalar parameter) is left
  replicated on both the graph and the live model.
- **PP v1 is pure-PP**: `pp_degree == world_size` and `fsdp_enabled=False`.
  PP+FSDP hybrids require a `mesh_context` exposing a pp dim.
- **`AutoOverlapPass` is a placeholder no-op** — see the pass note above.
- **Compilation is not idempotent**: the partitioning passes mutate the live
  model in place, so `GraphCompiler.compile` may only be called once — a
  second call silently re-traces the already-partitioned model and yields an
  incorrect graph (it does not raise; guard the call site yourself).
- **Requires torch with a patched autograd engine** for joint-graph capture.
  On stock torch the backward half is annotated structurally instead (see
  `_annotate_autograd_backward` in `tracer/graph_tracer.py`).
