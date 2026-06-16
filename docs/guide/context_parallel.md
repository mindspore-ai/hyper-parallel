# CP 上下文并行使用指南

HyperParallel 提供 Context Parallel（上下文并行）能力，将序列维度切分到多个设备上，支持超长序列训练。提供基础 ContextParallel 和异步 AsyncContextParallel，以及 DSA 系列（基于 Dense Sparse Attention 的 CP 实现）。

## 核心概念

上下文并行将输入序列按长度维度切分到多个设备，每个设备只处理部分序列长度，通过集合通信协调注意力计算结果。适用于长序列训练场景（如 128K+ 序列长度）。

| 实现 | 说明 | 适用场景 |
|------|------|----------|
| ContextParallel | 基础上下文并行 | 通用长序列 |
| AsyncContextParallel | 异步上下文并行 | 需要通信与计算并发 |
| DSAIndexerContextParallel | DSA 索引器 CP | Dense Sparse Attention |
| AsyncDSAIndexerContextParallel | 异步 DSA 索引器 CP | DSA + 通信并发 |
| DSAIndexerLossContextParallel | DSA 索引器 + Loss CP | DSA + Loss 并行 |
| AsyncDSAIndexerLossContextParallel | 异步 DSA 索引器 + Loss CP | DSA + Loss + 通信并发 |
| DSASparseAttentionContextParallel | DSA Sparse Attention CP | 稀疏注意力 |
| AsyncDSASparseAttentionContextParallel | 异步 DSA Sparse Attention CP | 稀疏注意力 + 通信并发 |

## 接口概览

| 接口 | 说明 |
|------|------|
| `ContextParallel` | 基础上下文并行 |
| `AsyncContextParallel` | 异步上下文并行 |
| `DSAIndexerContextParallel` | DSA 索引器上下文并行 |
| `AsyncDSAIndexerContextParallel` | 异步 DSA 索引器上下文并行 |
| `DSAIndexerLossContextParallel` | DSA 索引器 + Loss 上下文并行 |
| `AsyncDSAIndexerLossContextParallel` | 异步 DSA 索引器 + Loss 上下文并行 |
| `DSASparseAttentionContextParallel` | DSA Sparse Attention 上下文并行 |
| `AsyncDSASparseAttentionContextParallel` | 异步 DSA Sparse Attention 上下文并行 |

---

## 基础使用

### 1. ContextParallel 基础用法

```python
from hyper_parallel import ContextParallel, init_device_mesh

cp_mesh = init_device_mesh("npu", (cp_size,), mesh_dim_names=("cp",))
cp = ContextParallel()

# 应用到模型
model.attention = cp.apply(model.attention, cp_mesh)
```

### 2. AsyncContextParallel

```python
from hyper_parallel import AsyncContextParallel, init_device_mesh

cp_mesh = init_device_mesh("npu", (cp_size,), mesh_dim_names=("cp",))
cp = AsyncContextParallel()

model.attention = cp.apply(model.attention, cp_mesh)
```

### 3. DSA 系列 CP

```python
from hyper_parallel import DSAIndexerContextParallel, init_device_mesh

cp_mesh = init_device_mesh("npu", (cp_size,), mesh_dim_names=("cp",))
cp = DSAIndexerContextParallel()

model.attention = cp.apply(model.attention, cp_mesh)
```

---

## TP DTensor local rewrap

CP 支持 TP 场景下 DTensor local rewrap，确保 TP layout 在 CP 内部正确传递：

```python
from hyper_parallel import ContextParallel, init_device_mesh

# 2-D mesh: TP + CP
mesh = init_device_mesh("npu", (cp_size, tp_size), mesh_dim_names=("cp", "tp"))

# CP 在 cp 子 mesh 上应用
cp = ContextParallel()
model.attention = cp.apply(model.attention, mesh["cp"])

# TP layout 在 CP 内部自动 local rewrap
```

---

## 与其他并行策略组合

### CP + TP

```python
mesh = init_device_mesh("npu", (cp_size, tp_size), mesh_dim_names=("cp", "tp"))

# TP 先应用
parallelize_module(model, mesh["tp"], tp_plan)

# CP 再应用
cp = ContextParallel()
model.attention = cp.apply(model.attention, mesh["cp"])
```

### CP + FSDP

```python
mesh = init_device_mesh("npu", (cp_size, dp_size), mesh_dim_names=("cp", "dp"))

# FSDP 在 DP 维度
model = fully_shard(model, mesh=mesh["dp"])

# CP 在 CP 维度
cp = ContextParallel()
model.attention = cp.apply(model.attention, mesh["cp"])
```

### CP + PP + FSDP

```python
mesh = init_device_mesh("npu", (pp_size, dp_size, cp_size), mesh_dim_names=("pp", "dp", "cp"))

# FSDP 在 DP 维度
fully_shard(stage_model, mesh=mesh["dp"])

# CP 在 CP 维度
cp = ContextParallel()
stage_model.attention = cp.apply(stage_model.attention, mesh["cp"])
```

---

## 性能建议

1. **Async CP**：使用 AsyncContextParallel 可实现通信与计算并发，适合长序列场景
2. **DSA 系列**：针对 Dense Sparse Attention 模型的优化，减少不必要的通信
3. **CP + TP local rewrap**：确保 TP layout 正确传递，避免 layout 不一致导致的计算错误
4. **MindSpore 注意**：DSA 系列 async CP 目前仅 PyTorch 后端支持，MindSpore 为 noop 占位