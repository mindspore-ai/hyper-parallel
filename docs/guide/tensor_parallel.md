# TP 张量并行使用指南

HyperParallel 提供声明式张量并行（Tensor Parallel）接口，兼容 `torch.distributed.tensor.parallel` API 设计，支持 ColwiseParallel、RowwiseParallel、SequenceParallel 等并行策略，并支持 Loss Parallel。

## 核心概念

张量并行将模型权重/激活在多个设备上切分，每个设备只持有部分权重和激活，通过集合通信协调计算结果。

| 策略 | 说明 | 适用场景 |
|------|------|----------|
| ColwiseParallel | 列切分（输出维度切分） | Q/K/V projection、gate/up projection |
| RowwiseParallel | 行切分（输入维度切分） | O projection、down projection |
| SequenceParallel | 序列维度切分 | LayerNorm + Dropout |

## 接口概览

| 接口 | 说明 |
|------|------|
| `ColwiseParallel` | 列切分并行策略 |
| `RowwiseParallel` | 行切分并行策略 |
| `SequenceParallel` | 序列并行策略 |
| `PrepareModuleInput` / `PrepareModuleOutput` / `PrepareModuleInputOutput` | 模块输入/输出准备钩子 |
| `parallelize_module` | 声明式 TP 应用接口 |
| `ParallelStyle` | 并行策略基类 |

---

## 基础使用

### 1. TransformerBlock 标准 TP 配置

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

### 2. 使用 fnmatch glob pattern

```python
# 对多层 TransformerBlock 使用通配符
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

## ColwiseParallel 详解

将 Linear 或 Embedding 的输出维度切分到 TP ranks：

| 模块类型 | weight 切分 | bias 切分 |
|----------|-------------|-----------|
| Linear | `Shard(0)` — 切分输出特征 | `Shard(0)` |
| Embedding | `Shard(1)` — 切分 embedding 维 | N/A |

参数：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `input_layouts` | Placement | `Replicate()` | 输入排布 |
| `output_layouts` | Placement | `Shard(-1)` | 输出排布 |
| `use_local_output` | bool | `True` | 是否将 DTensor 转为 local tensor |

## RowwiseParallel 详解

将 Linear 或 Embedding 的输入维度切分到 TP ranks：

| 模块类型 | weight 切分 | bias 切分 |
|----------|-------------|-----------|
| Linear | `Shard(1)` — 切分输入特征 | `Replicate()` |
| Embedding | `Shard(0)` — 切分 vocab 维 | N/A |

参数：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `input_layouts` | Placement | `Shard(-1)` | 输入排布 |
| `output_layouts` | Placement | `Replicate()` | 输出排布 |
| `use_local_output` | bool | `True` | 是否将 DTensor 转为 local tensor |

---

## Loss Parallel

TP 训练中 loss 计算也可以并行化，避免在 loss 计算时需要 gather 全部 logits：

```python
# 在 TP mesh 上并行计算 loss
# logits 已经在 TP 维度上切分，直接计算分片上的 loss
```

---

## 典型组合模式

### MLP (Gate / Up / Down)

```
Input (Replicate) ─┬─► gate_proj (ColwiseParallel) ─► Shard(-1)
                    └─► up_proj   (ColwiseParallel) ─► Shard(-1)
                              ↓ element-wise multiply
                        down_proj (RowwiseParallel) ─► Replicate
```

### Multi-Head Attention

```
Input (Replicate) ─► Q/K/V projections (ColwiseParallel) ─► Shard(-1)
                              ↓ attention computation
                        Output projection (RowwiseParallel) ─► Replicate
```

---

## 与其他并行策略组合

TP 使用 1-D DeviceMesh，在混合并行场景下从多维 mesh 中切片：

```python
mesh = init_device_mesh("npu", (dp_size, tp_size), mesh_dim_names=("dp", "tp"))

# TP 在 "tp" 子 mesh
parallelize_module(model, mesh["tp"], tp_plan)

# FSDP 在 "dp" 子 mesh
fully_shard(model, mesh=mesh["dp"])
```

TP hooks 不干扰 FSDP unshard/reshard 或 pipeline scheduling。

---

## 从 PyTorch TP 迁移

| PyTorch | HyperParallel | 说明 |
|---------|---------------|------|
| `from torch.distributed.tensor.parallel import ColwiseParallel` | `from hyper_parallel import ColwiseParallel` | 类名相同 |
| `from torch.distributed.tensor.parallel import RowwiseParallel` | `from hyper_parallel import RowwiseParallel` | 类名相同 |
| `parallelize_module(m, mesh, plan)` | `parallelize_module(m, mesh, plan)` | 签名相同 |
| `style._apply(module, mesh)` | `style.apply(module, mesh)` | 无前导下划线 |
| `DTensor.redistribute(placements=...)` | `DTensor.redistribute(device_mesh, placements)` | 显式 mesh 参数 |

---

## 更多参考

- [TP Styles 详细文档](./tensor_parallel_styles.md) — 原有的 TP Styles 特性文档
- [API 参考](../api/api_reference.md) — TP 模块接口详细说明