# HyperParallel Graph Mode

Graph-mode architecture for automatic parallelization with FSDP.

> **Note**: The `compile/` subpackage relies on a patched autograd engine for
> joint-graph capture (see Limitations). Its `torch.*` imports — including
> `torch.fx.experimental` and `torch._guards` — are therefore intentional and
> carry a `forbidden-backend-import` suppression in
> `.jenkins/check/config/filter_pylint.txt`.

## Core Concept

**User**: Write model code + parallel configuration
**Framework**: Graph capture → FSDP partitioning → Communication-compute overlap → Execution

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
DeadCodeElimination → CanonicalizeGraph → FSDPPass → AutoOverlapPass
```

**FSDPPass**:

- Identifies FSDP parameter placeholders via GraphParallelPlan
- Sinks `all_gather` to each parameter's first forward use (Shard → Replicate)
- Frees the replicated parameter after its last forward read, then re-gathers
  it for the backward and rematerializes any saved forward view of it
  (`reshard_after_forward`), so peak memory tracks the forward working set
- Inserts `reduce_scatter` on gradient outputs (Replicate → Shard)
- Physically shards live model parameters (dim 0) so optimizer is FSDP-agnostic

Disable reshard with `PassConfig(fsdp_reshard_after_forward=False)`.

**AutoOverlapPass**:

- Reorders `wait_tensor` nodes for communication-compute overlap

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

Use the standalone graph API to vary batch and sequence dimensions:

```python
from hyper_parallel.compile import GraphCompiler, PassConfig

compiler = GraphCompiler(
    model, train_fn, pass_config=PassConfig(fsdp_enabled=False),
    dynamic_arg_dims={"x": [0, 1], "y": [0, 1]},
)
loss = compiler.forward_backward(x=x, y=y)
```

`train_fn(model, **inputs)` returns a scalar loss tensor. `forward_backward`
returns that loss and accumulates the graph-computed gradients into `param.grad`.
`GraphTrainer` accepts the same dynamic options and manages the optimizer.
This API does not require AutoModel, BaseTrainer or a trainer YAML configuration.

- `dynamic=True` attempts to symbolize all user tensor dimensions. Parameters
  and buffers retain concrete shapes.
- `dynamic_arg_dims` overrides automatic selection, even with `dynamic=False`.
  Unlisted dimensions stay static; an empty mapping selects no dynamic axes.
  Dotted dictionary/list/attribute paths and negative dimension indices work.
- The first call captures one joint forward/backward FX graph. Runtime guards
  check structure, metadata, static dimensions, aliases, strides and traced
  shape branches before later executions. Unsupported inputs raise an error
  before tensor operations or collectives execute; they do not silently retrace.
- Use representative first inputs with dynamic dimensions greater than one.
  Sizes zero/one, shape-dependent Python branches and operator constraints can
  narrow the valid range. Data-dependent shapes and dynamic PP remain outside
  this implementation. FSDP supports resharding and communication overlap.

Run the [loss/gradient example](examples/dynamic_shapes.py), or train the
[small causal language model](examples/dynamic_training.py) with changing
batch/sequence lengths and compare every optimizer step against eager:

```bash
python -m hyper_parallel.compile.examples.dynamic_shapes
python -m hyper_parallel.compile.examples.dynamic_training --device cpu
python -m hyper_parallel.compile.examples.dynamic_training --device npu
```

The training example generates its own token sequences. The same entry supports
FSDP across two NPUs, including the existing reshard-after-forward pass:

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1 python -m torch.distributed.run \
  --standalone --nproc-per-node=2 \
  -m hyper_parallel.compile.examples.dynamic_training --device npu
```

Run CPU UT and distributed regression with:

```bash
python -m pytest tests/ut/compile/test_dynamic_shapes.py -q
python -m pytest tests/torch/compile/test_dynamic_shapes.py -q
```

The implementation uses `ShapeEnv` and `make_fx` directly and has no runtime
MagiCompiler dependency. Execution reuses symbolic FX code; it does not add a
compiled-kernel backend or NPU graph capture.

## Lazy size specialization

Configure hot sequence lengths directly on `GraphCompiler` or `GraphTrainer`:

```python
compiler = GraphCompiler(
    model, train_fn, pass_config=PassConfig(fsdp_enabled=False),
    dynamic_arg_dims={"x": [0, 1], "y": [0, 1]},
    compile_sizes=[5, 7], compile_size_input="x", compile_size_dim=1,
    max_specializations=8,
)
```

The first call executes the general symbolic graph. Later configured sizes
lazily generate concrete-shape FX variants; repeated input signatures hit the
cache, and unconfigured sizes use the general graph. Capacity is bounded:
new signatures at capacity fall back to the general graph.

The selector uses a dotted tensor path and axis (negative axes are supported).
Without an explicit path, it uses the first symbolic user input axis. The cache
key includes every user tensor's shape, stride, storage offset and metadata,
so another batch dimension at the same sequence length needs its own variant.
Tensor values and model weights stay live. General input guards run before
all dispatches, including cache hits.

Specialization folds pure symbolic shape queries and integer arithmetic in the
already transformed graph. It does not execute tensor or communication
operations while generating a variant, repeat model capture or run the FSDP
sharding pass again. This is FX code specialization; no acceleration is promised.
`compile_sizes=None` or `[]` disables it. `specialization_stats` on either the
compiler or trainer exposes generation, hit, fallback and folded-node counters.

```bash
python -m hyper_parallel.compile.examples.size_specialization
python -m hyper_parallel.compile.examples.dynamic_training \
  --device npu --compile-sizes 7 11
```

The training entry also supports FSDP2 with lengths differing between ranks:

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1 python -m torch.distributed.run \
  --standalone --nproc-per-node=2 \
  -m hyper_parallel.compile.examples.dynamic_training \
  --device npu --compile-sizes 7 8 11 12
```

When another job shares the same NPUs, HCCL sockets can use automatically
assigned ports via `HCCL_NPU_SOCKET_PORT_RANGE=auto` and
`HCCL_HOST_SOCKET_PORT_RANGE=auto`; see the
[HCCL environment reference](https://www.hiascend.com/document/detail/zh/canncommercial/850/commlib/hcclug/hcclug_000092.html).

Run [the cache regression tests](../../tests/ut/compile/test_size_specialization.py)
with `python -m pytest tests/ut/compile/test_size_specialization.py -q`.

## Key Design Decisions

1. **Static Inputs**: Parameters are graph inputs, not `get_attr`. This allows passes to split the graph by reshaping placeholders.

2. **Joint Graph**: Forward + backward captured together via `torch.autograd.grad` inside the traced function.

3. **FSDP-Agnostic Optimizer**: FSDPPass shards the live model's parameters in place. `model.parameters()` returns shards, so optimizer needs no FSDP awareness.

4. **Declarative Sharding**: GraphParallelPlan uses FQN patterns (`layers.*`) instead of imperative module wrapping.

## Limitations

- FSDP only (TP/EP/PP planned)
- Parameters must have dim 0 divisible by world_size
- Requires torch with patched autograd engine for joint-graph capture
