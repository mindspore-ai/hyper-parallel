# TP Tensor Parallelism Guide

**English** | [简体中文](tensor_parallel.zh.md)

HyperParallel provides declarative tensor parallelism interfaces compatible with the `torch.distributed.tensor.parallel` API design. It supports ColwiseParallel, RowwiseParallel, SequenceParallel, and Loss Parallel.

## Core Concepts

Tensor parallelism shards model weights and activations across devices. Each device holds only part of the weights and activations, and collective communication coordinates the computation results.

| Strategy | Description | Use case |
|------|------|----------|
| ColwiseParallel | Column sharding (output dimension) | Q/K/V projections and gate/up projections |
| RowwiseParallel | Row sharding (input dimension) | O projection and down projection |
| SequenceParallel | Sequence-dimension sharding | LayerNorm + Dropout |

## API Overview

| Interface | Description |
|------|------|
| `ColwiseParallel` | Column-sharded parallel strategy |
| `RowwiseParallel` | Row-sharded parallel strategy |
| `SequenceParallel` | Sequence parallel strategy |
| `PrepareModuleInput` / `PrepareModuleOutput` / `PrepareModuleInputOutput` | Module input/output preparation hooks |
| `parallelize_module` | Declarative TP application interface |
| `ParallelStyle` | Base class for parallel strategies |

---

## Basic Usage

### 1. Standard TP Configuration for TransformerBlock

```python
from hyper_parallel import ColwiseParallel, RowwiseParallel, parallelize_module, init_device_mesh

tp_mesh = init_device_mesh("npu", (tp_size,), mesh_dim_names=("tp",))

parallelize_module(
    model,
    tp_mesh,
    {
        "attn.q_proj": ColwiseParallel(),
        "attn.k_proj": ColwiseParallel(),
        "attn.v_proj": ColwiseParallel(),
        "attn.o_proj": RowwiseParallel(),
        "mlp.gate_proj": ColwiseParallel(),
        "mlp.up_proj": ColwiseParallel(),
        "mlp.down_proj": RowwiseParallel(),
    },
)
```

### 2. Use fnmatch Glob Patterns

```python
# Use wildcards for multiple TransformerBlock layers.
parallelize_module(
    model,
    tp_mesh,
    {
        "layers.*.attn.q_proj": ColwiseParallel(),
        "layers.*.attn.k_proj": ColwiseParallel(),
        "layers.*.attn.v_proj": ColwiseParallel(),
        "layers.*.attn.o_proj": RowwiseParallel(),
        "layers.*.mlp.gate_proj": ColwiseParallel(),
        "layers.*.mlp.up_proj": ColwiseParallel(),
        "layers.*.mlp.down_proj": RowwiseParallel(),
    },
)
```

---

## ColwiseParallel in Detail

Shard the output dimension of Linear or Embedding across TP ranks:

| Module type | Weight sharding | Bias sharding |
|----------|-------------|-----------|
| Linear | `Shard(0)` — output features | `Shard(0)` |
| Embedding | `Shard(1)` — embedding dimension | N/A |

Parameters:

| Parameter | Type | Default | Description |
|------|------|--------|------|
| `input_layouts` | Placement | `Replicate()` | Input layout |
| `output_layouts` | Placement | `Shard(-1)` | Output layout |
| `use_local_output` | bool | `True` | Whether to convert DTensor to a local tensor |

## RowwiseParallel in Detail

Shard the input dimension of Linear or Embedding across TP ranks:

| Module type | Weight sharding | Bias sharding |
|----------|-------------|-----------|
| Linear | `Shard(1)` — input features | `Replicate()` |
| Embedding | `Shard(0)` — vocabulary dimension | N/A |

Parameters:

| Parameter | Type | Default | Description |
|------|------|--------|------|
| `input_layouts` | Placement | `Shard(-1)` | Input layout |
| `output_layouts` | Placement | `Replicate()` | Output layout |
| `use_local_output` | bool | `True` | Whether to convert DTensor to a local tensor |

---

## Loss Parallel

Loss computation can also be parallelized during TP training, avoiding the need to gather all logits:

```python
# Compute loss in parallel over the TP mesh.
# Logits are already sharded along the TP dimension; compute loss directly on the shards.
```

---

## Typical Combinations

### MLP (Gate / Up / Down)

```text
Input (Replicate) ─┬─► gate_proj (ColwiseParallel) ─► Shard(-1)
                    └─► up_proj   (ColwiseParallel) ─► Shard(-1)
                              ↓ element-wise multiply
                        down_proj (RowwiseParallel) ─► Replicate
```

### Multi-Head Attention

```text
Input (Replicate) ─► Q/K/V projections (ColwiseParallel) ─► Shard(-1)
                              ↓ attention computation
                        Output projection (RowwiseParallel) ─► Replicate
```

---

## Combine with Other Parallel Strategies

TP uses a 1-D DeviceMesh, sliced from a multidimensional mesh for hybrid parallelism:

```python
mesh = init_device_mesh("npu", (dp_size, tp_size), mesh_dim_names=("dp", "tp"))

# Apply TP on the "tp" submesh.
parallelize_module(model, mesh["tp"], tp_plan)

# Apply FSDP on the "dp" submesh.
fully_shard(model, mesh=mesh["dp"])
```

TP hooks do not interfere with FSDP unshard/reshard or pipeline scheduling.

---

## Migrate from PyTorch TP

| PyTorch | HyperParallel | Description |
|---------|---------------|------|
| `from torch.distributed.tensor.parallel import ColwiseParallel` | `from hyper_parallel import ColwiseParallel` | Same class name |
| `from torch.distributed.tensor.parallel import RowwiseParallel` | `from hyper_parallel import RowwiseParallel` | Same class name |
| `parallelize_module(m, mesh, plan)` | `parallelize_module(m, mesh, plan)` | Same signature |
| `style._apply(module, mesh)` | `style.apply(module, mesh)` | No leading underscore |
| `DTensor.redistribute(placements=...)` | `DTensor.redistribute(device_mesh, placements)` | Explicit mesh argument |

---

## Further Reading

- [TP Styles](./tensor_parallel_styles.md) — Detailed documentation for TP styles
- [API Reference](../api/api_reference.md) — Detailed TP interfaces
