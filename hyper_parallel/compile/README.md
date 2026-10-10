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
