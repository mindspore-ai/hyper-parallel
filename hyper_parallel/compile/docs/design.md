# HyperParallel Graph Compile Architecture Design

## Overview

HyperParallel Graph Compile is the graph-mode architecture of HyperParallel, adopting a **declarative parallelism** design philosophy:

- **User Perspective**: Only write single-card model + sharding configuration
- **Framework Responsibilities**: Graph capture → Sharding annotation → Parallel partitioning → Communication-compute overlap → Execution

## Core Design Philosophy

```text
User Input:
  ┌──────────────────┐      ┌────────────────────────┐
  │ Single-card model│  +   │ Parallel config (YAML) │
  │ code (nn.Module) │      │ - fsdp_degree: 8       │
  │                  │      │ - tp_degree: 2         │
  │ No parallel code │      │ - enable_overlap: true │
  └──────────────────┘      └────────────────────────┘

Framework Processing:
  1. Graph Capture: Capture complete single-card model graph (Forward + Backward)
  2. Sharding Annotation: Annotate sharding specs based on ShardingPlan
  3. Parallel Partitioning: Automatic FSDP/TP/EP/PP partitioning in Pass stage
  4. Communication-Compute Overlap: Optimize overlap between communication and computation
```

## Architecture Layers

### Layer 1: User Configuration Layer

**ShardingSpec - Sharding Specification**

```python
ShardingSpec({
    MeshAxisName.TP: spmd.S(0),     # TP dimension shards on tensor dim 0
    MeshAxisName.FSDP: spmd.S(0),   # FSDP dimension shards on tensor dim 0
})
```

**ShardingModuleConfig - Module Sharding Configuration**

```python
ShardingModuleConfig(
    param_shardings={"weight": ShardingSpec.shard(TP, 0)},
    input_src_shardings={"x": sequence_parallel_spec()},
    input_dst_shardings={"x": ShardingSpec.replicate(TP)},
    output_src_sharding=ShardingSpec.partial(TP),
    output_dst_sharding=sequence_parallel_spec(),
)
```

### Layer 2: Graph Capture Layer

```python
def trace_single_card_graph(model, train_fn, sample_input, sample_label):
    """
    Trace single-card model to generate complete computation graph

    Features:
    - No parallel processing, pure single-card semantics
    - Parameters, activations, gradients are all complete tensors
    - Forward + Loss + Backward complete joint graph
    """
```

### Layer 3: Pass Pipeline Layer

**Execution Order**:

```text
[DeadCodeElimination] → Clean up dead code
        ↓
[CanonicalizeGraph] → Canonicalize graph
        ↓
[ShardingPass] → Metadata layer: Annotate sharding specs to node.meta
        ↓
[TPPass] → Execution layer: TP partitioning and communication insertion
        ↓
[EPPass] → Execution layer: EP partitioning and communication insertion
        ↓
[FSDPPass] → Execution layer: FSDP partitioning and communication insertion
        ↓
[PPPass] → Execution layer: PP partitioning and P2P communication
        ↓
[CommReorderPass] → Communication-compute overlap: Reorder communication nodes
        ↓
[OverlapSchedulePass] → Communication-compute overlap: Generate overlap schedule
```

> Note: An optional `InductorPass` for backend compilation exists but is
> currently disabled in `passes/pipeline.py`; the pipeline ends at
> `OverlapSchedulePass` and executes the resulting graph directly.

**Relationship between ShardingPass and TPPass**:

- **ShardingPass** (Metadata layer):
  - Parse ShardingPlan
  - Annotate sharding metadata on graph nodes (no graph modification)
  - Output: `node.meta["sharding_config"]`

- **TPPass** (Execution layer):
  - Read sharding specs annotated by ShardingPass
  - Actually modify graph structure (partition parameters, insert communication)
  - Handle all TP dimension logic

### Layer 4: Graph Execution Layer

The compiled graph is a callable `GraphModule`. On each step `GraphTrainer`
feeds it the live model state in FQN order and consumes its outputs. The
joint graph's parameters/buffers are *static inputs* (leading placeholders),
so execution goes through `run_traced_graph`, which samples them from the
live (FSDP-sharded) model each step and runs the whole fwd+bwd graph under
`torch.no_grad()`:

```python
# GraphTrainer._run_graph
outputs = run_traced_graph(
    self._single_card_graph, self.model, input_batch, label_batch
)
# outputs == [loss] + list(grads)
loss, grads = outputs[0], outputs[1:]
```

The graph itself embeds all parallel behaviour inserted by the passes (FSDP
AllGather of parameters, forward/backward compute, gradient ReduceScatter,
communication-compute overlap). There is no separate runtime object besides
the state-feeding convenience of `run_traced_graph` -- execution is a call
into the graph with the model's parameters threaded through as inputs.

## Sharding Configuration System

### Input/Output Resharding

```text
Input resharding (input_src → input_dst):
  - src: Current sharding of input tensor (for DTensor.from_local)
  - dst: Required sharding for computation (triggers redistribute)

Output resharding (output_src → output_dst):
  - src: Output sharding produced by computation
  - dst: Expected output sharding (triggers redistribute)
```

### Common Sharding Patterns

| Pattern | Param Sharding | Input Sharding | Output Sharding |
|---------|----------------|----------------|-----------------|
| ColwiseParallel | S(0) | - | S(-1) |
| RowwiseParallel | S(1) | - | P → RS/AR |
| SequenceParallel | - | S(1) | S(1) |
| Norm | R/I | SP/I | SP/I |

## Usage Example

```python
from hyper_parallel.compile import GraphTrainer, ParallelConfig
from hyper_parallel.compile.sharding_config import create_sharding_plan_from_yaml

# 1. Create single-card model (no parallel code)
model = Llama3Model(model_spec)

# 2. Configure parallelism
parallel_config = ParallelConfig(
    fsdp_degree=8,
    tp_degree=2,
    enable_overlap=True,
)

# 3. Configure sharding plan (loaded from examples/llama3/config.yaml)
sharding_plan = create_sharding_plan_from_yaml(model_name="llama3")

# 4. Create trainer
trainer = GraphTrainer(
    model=model,
    train_fn=train_fn,
    parallel_config=parallel_config,
    sharding_plan=sharding_plan,
    optimizer_config={"lr": 3e-4},
)

# 5. Training loop
for batch in dataloader:
    loss = trainer.train_step(input_ids, labels)
    trainer.optimizer_step()
```

## Comparison with TorchTitan

| Dimension | TorchTitan | HyperGraph |
|-----------|------------|------------|
| Configuration method | `ShardingConfig` on `Module.Config` | `ShardingPlan` + `ShardingModuleConfig` |
| Application timing | Runtime `parallelize()` | Compile-time Pass Pipeline |
| User interface | Model configuration object | Independent sharding plan object |
| Pattern matching | Python code setup | Supports FQN wildcards |
