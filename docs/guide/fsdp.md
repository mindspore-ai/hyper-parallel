# HSDP / FSDP 数据并行使用指南

HyperParallel 提供 `fully_shard` 接口实现 HSDP（Hybrid Sharded Data Parallel）和 FSDP（Fully Sharded Data Parallel），支持参数/梯度/优化器状态的分布式切分，显著降低单卡内存占用。

## 核心概念

| 模式 | 说明 | 内存节省 |
|------|------|----------|
| FSDP | 参数/梯度/优化器状态全切分 | 最大 |
| HSDP | 参数/优化器状态切分 + 梯度切分 + overlap | 大 |

## 接口概览

| 接口 | 说明 |
|------|------|
| `fully_shard(model, mesh, ...)` | 对模型应用 FSDP/HSDP 参数切分 |
| `HSDPModule` | HSDP 模块封装 |
| `hsdp_sync_stream` | HSDP 梯度同步流管理 |
| `set_gradient_scaling_factor` | 设置梯度缩放因子 |

---

## 基础使用

### 1. 最小 FSDP 示例

```python
from hyper_parallel import fully_shard, init_device_mesh

mesh = init_device_mesh(device_type="npu", mesh_shape=(dp_size,), mesh_dim_names=("dp",))
model = fully_shard(model, mesh=mesh)

# 正常训练流程
output = model(input)
loss = criterion(output)
loss.backward()
optimizer.step()
optimizer.zero_grad()
```

### 2. 多维 mesh + FSDP

```python
# DP + TP 二维 mesh
mesh = init_device_mesh("npu", (dp_size, tp_size), mesh_dim_names=("dp", "tp"))

# FSDP 在 DP 维度
model = fully_shard(model, mesh=mesh["dp"])
```

### 3. HSDP Overlap 模式

HSDP 全 overlap 模式下，梯度通信与计算并发执行：

```python
model = fully_shard(model, mesh=mesh, overlap=True)
```

### 4. 梯度缩放因子

在混合精度训练或 CP/PP 场景下，可能需要对梯度进行缩放：

```python
from hyper_parallel import fully_shard, init_device_mesh

mesh = init_device_mesh("npu", (dp_size,), mesh_dim_names=("dp",))
model = fully_shard(model, mesh=mesh)

# 设置梯度缩放因子
model.set_gradient_scaling_factor(scale_factor=0.5)
```

---

## fully_shard 参数详解

```python
fully_shard(
    module,                # 要切分的模块
    mesh,                  # DeviceMesh（通常为 DP 维度的子 mesh）
    *,
    reshard_after_forward=True,   # 正向后是否 reshard（节省内存）
    mixed_precision=None,         # 混合精度策略 MixedPrecisionPolicy
    offload_policy=None,          # Offload 策略 OffloadPolicy
    comm_fusion=True,             # 通信融合
    comm_fusion_zero_copy=None,   # 通信融合零拷贝
)
```

---

## 与其他并行策略组合

### FSDP + TP

```python
from hyper_parallel import ColwiseParallel, RowwiseParallel, parallelize_module, init_device_mesh, fully_shard

mesh = init_device_mesh("npu", (dp_size, tp_size), mesh_dim_names=("dp", "tp"))

# 先应用 TP
parallelize_module(model, mesh["tp"], tp_plan)

# 再应用 FSDP
model = fully_shard(model, mesh=mesh["dp"])
```

### FSDP + PP

```python
from hyper_parallel import PipelineStage, Schedule1F1B, fully_shard, init_device_mesh

mesh = init_device_mesh("npu", (pp_size, dp_size), mesh_dim_names=("pp", "dp"))

# 每个 PP stage 内部应用 FSDP
for stage_model in split_models:
    fully_shard(stage_model, mesh=mesh["dp"])

# 创建 PP stage
stage = PipelineStage(stage_model, stage_index, stage_num=pp_size)
```

### FSDP + EP

```python
from hyper_parallel import init_device_mesh, fully_shard
from hyper_parallel.core.expert_parallel import ExpertParallel

mesh = init_device_mesh("npu", (dp_size, ep_size), mesh_dim_names=("dp", "ep"))

# EP 在专家维度
ExpertParallel().apply(moe.experts, mesh["ep"])

# FSDP 在 DP 维度
model = fully_shard(model, mesh=mesh["dp"])
```

---

## 性能建议

1. **reshard_after_forward=True**：正向后 reshard 参数，将完整参数重新切分，大幅节省正向间内存
2. **overlap 模式**：开启 HSDP overlap 可让梯度 all-reduce 与正向计算并发
3. **comm_fusion**：通信融合将多个小通信合并为一个大通信，减少通信次数
4. **混合精度策略**：使用 `MixedPrecisionPolicy` 可在通信中使用低精度（如 FP8），减少通信带宽