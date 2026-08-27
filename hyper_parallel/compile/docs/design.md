# HyperParallel Graph Mode

Graph-mode architecture for automatic parallelization with FSDP.

> **Note**: The `compile/` subpackage is currently torch-only (it relies on a
> patched autograd engine for joint-graph capture; see Limitations). Backend
> imports (`torch.*`) are therefore intentional and tracked as a known
> platform-layering exception pending a future `get_platform()` abstraction for
> graph trace / FSDP-pass APIs.

## Core Concept

**User**: Write model code + parallel configuration
**Framework**: Graph capture → FSDP partitioning → Communication-compute overlap → Execution

## Architecture

### Layer 1: Configuration

```python
from hyper_parallel.compile import ShardingPlan, ParallelConfig

# Configure which modules to wrap with FSDP
sharding_plan = ShardingPlan()
sharding_plan.fsdp_wrap("tok_embeddings")
sharding_plan.fsdp_wrap_pattern("layers.*")

# Parallel configuration
parallel_config = ParallelConfig(enable_overlap=True)
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

- Identifies FSDP parameter placeholders via ShardingPlan
- Inserts `all_gather` after each FSDP parameter (Shard → Replicate)
- Inserts `reduce_scatter` on gradient outputs (Replicate → Shard)
- Physically shards live model parameters (dim 0) so optimizer is FSDP-agnostic

**AutoOverlapPass**:

- Reorders `wait_tensor` nodes for communication-compute overlap

### Layer 4: Execution

```python
from hyper_parallel.compile import GraphTrainer

trainer = GraphTrainer(
    model=model,
    train_fn=train_fn,
    parallel_config=parallel_config,
    sharding_plan=sharding_plan,
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
from hyper_parallel.compile import GraphTrainer, ParallelConfig, ShardingPlan

# 1. Model
model = Llama3Model(config)

# 2. Sharding plan
sharding_plan = ShardingPlan()
sharding_plan.fsdp_wrap_pattern("layers.*")

# 3. Trainer
trainer = GraphTrainer(
    model=model,
    train_fn=lambda m, x, y: m(x).loss(y),
    parallel_config=ParallelConfig(enable_overlap=True),
    sharding_plan=sharding_plan,
)

# 4. Training
trainer.train(dataloader, max_steps=1000)
```

## Key Design Decisions

1. **Static Inputs**: Parameters are graph inputs, not `get_attr`. This allows passes to split the graph by reshaping placeholders.

2. **Joint Graph**: Forward + backward captured together via `torch.autograd.grad` inside the traced function.

3. **FSDP-Agnostic Optimizer**: FSDPPass shards the live model's parameters in place. `model.parameters()` returns shards, so optimizer needs no FSDP awareness.

4. **Declarative Sharding**: ShardingPlan uses FQN patterns (`layers.*`) instead of imperative module wrapping.

## Limitations

- FSDP only (TP/EP/PP planned)
- Parameters must have dim 0 divisible by world_size
- Requires torch with patched autograd engine for joint-graph capture
