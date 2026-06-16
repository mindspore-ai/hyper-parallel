# Tensor Parallel Styles

HyperParallel provides declarative tensor-parallel (TP) styles that can be
composed to shard Transformer modules across devices.  The API is designed to be
compatible with `torch.distributed.tensor.parallel` so that users familiar with
PyTorch can migrate with minimal code changes.

## Quick Start

```python
from hyper_parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
    init_device_mesh,
)

# 1. Create a 1-D TP device mesh
tp_mesh = init_device_mesh("npu", (8,), mesh_dim_names=("tp",))

# 2. Define a model
model = TransformerBlock(...)

# 3. Apply column-wise and row-wise sharding
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

## API Reference

### `ColwiseParallel`

Partition a Linear or Embedding module in a **column-wise** fashion.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_layouts` | `Placement` | `Replicate()` | How to annotate module input |
| `output_layouts` | `Placement` | `Shard(-1)` | Desired output layout |
| `use_local_output` | `bool` | `True` | Convert output DTensor to local tensor |

**Sharding behavior:**

| Module | weight | bias |
|--------|--------|------|
| Linear | `Shard(0)` — split output features | `Shard(0)` |
| Embedding | `Shard(1)` — split embedding dim | N/A |

### `RowwiseParallel`

Partition a Linear or Embedding module in a **row-wise** fashion.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_layouts` | `Placement` | `Shard(-1)` | How to annotate module input |
| `output_layouts` | `Placement` | `Replicate()` | Desired output layout |
| `use_local_output` | `bool` | `True` | Convert output DTensor to local tensor |

**Sharding behavior:**

| Module | weight | bias |
|--------|--------|------|
| Linear | `Shard(1)` — split input features | `Replicate()` |
| Embedding | `Shard(0)` — split vocab dim | N/A |

### `parallelize_module`

```python
parallelize_module(
    module,
    device_mesh,
    parallelize_plan,        # ParallelStyle or Dict[str, ParallelStyle]
    *,
    src_data_rank=0,
)
```

Apply one or more parallel styles to a module tree.  Supports `fnmatch` glob
patterns in dictionary keys (e.g. `"layers.*.mlp.gate_proj"`).

## Typical Compositions

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

## Combining with Other Parallelism

TP styles only accept a 1-D `DeviceMesh`.  For hybrid parallelism, slice a
sub-mesh from a multi-dimensional mesh:

```python
mesh_2d = init_device_mesh(
    "npu", (dp_size, tp_size), mesh_dim_names=("dp", "tp")
)

# TP on the "tp" sub-mesh
parallelize_module(model, mesh_2d["tp"], tp_plan)

# FSDP on the "dp" sub-mesh
fully_shard(model, mesh=mesh_2d["dp"])
```

TP hooks do not interfere with FSDP unshard/reshard or pipeline scheduling.

## Platform Support

The implementation is framework-agnostic.  Module type detection uses
`platform.is_linear_module()` / `platform.is_embedding_module()`, which map to:

| Platform | Linear type | Embedding type |
|----------|-------------|----------------|
| PyTorch | `torch.nn.Linear` | `torch.nn.Embedding` |
| MindSpore | `mindspore.nn.Dense` | `mindspore.nn.Embedding` |

## Migration from PyTorch

| PyTorch | HyperParallel | Notes |
|---------|---------------|-------|
| `from torch.distributed.tensor.parallel import ColwiseParallel` | `from hyper_parallel import ColwiseParallel` | Same class name |
| `from torch.distributed.tensor.parallel import RowwiseParallel` | `from hyper_parallel import RowwiseParallel` | Same class name |
| `parallelize_module(m, mesh, plan)` | `parallelize_module(m, mesh, plan)` | Same signature |
| `style._apply(module, mesh)` | `style.apply(module, mesh)` | No leading underscore |
| `DTensor.redistribute(placements=...)` | `DTensor.redistribute(device_mesh, placements)` | Explicit mesh arg |
