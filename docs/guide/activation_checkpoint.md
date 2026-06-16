# Activation Checkpoint / Swap 使用指南

HyperParallel 提供灵活的激活内存优化机制，包括选择性重计算（Activation Checkpoint）和激活 Swap（将中间激活 offload 到 CPU，反向时 prefetch 回来），支持两者协同配置。

## 核心概念

| 机制 | 原理 | 内存节省 | 计算开销 |
|------|------|----------|----------|
| Activation Checkpoint | 正向时不保存中间激活，反向时重计算 | 大（不存储激活） | 有（反向多一次正向计算） |
| Activation Swap | 正向时将激活 offload 到 CPU，反向时 prefetch 回来 | 大（不占用 NPU HBM） | 低（异步 DMA 传输） |
| 协同配置 | 对计算密集层用 checkpoint，对传输密集层用 swap | 最大化 | 最优化 |

## 接口概览

| 接口 | 说明 |
|------|------|
| `checkpoint` | 函数式激活重计算，支持 policy_fn、swap_inputs、context_fn |
| `swap` | 函数式激活 swap，不重计算，仅 offload/prefetch |
| `checkpoint_wrapper` | 模块级 checkpoint 装饰器 |
| `swap_wrapper` | 模块级 swap 装饰器 |
| `swap_tensor_wrapper` | 单 tensor swap 装饰器 |
| `CheckpointPolicy` | 重计算策略枚举：MUST_SAVE / PREFER_SAVE / MUST_RECOMPUTE / PREFER_RECOMPUTE / MUST_SWAP |
| `SwapManager` | Swap 分组管理器（单例） |

---

## 使用方式

### 1. 函数式 checkpoint

```python
from hyper_parallel.core.activation_checkpoint import checkpoint

# 基础使用：对整个函数应用 checkpoint
output = checkpoint(model.layer, x)

# 带 policy_fn：选择性重计算
def my_policy(target):
    if target.is_large_op:
        return CheckpointPolicy.MUST_RECOMPUTE
    return CheckpointPolicy.MUST_SAVE

output = checkpoint(model.layer, x, policy_fn=my_policy)

# 带 swap_inputs：将输入也 swap 到 CPU
output = checkpoint(model.layer, x, swap_inputs=True)
```

### 2. 函数式 swap

```python
from hyper_parallel.core.activation_checkpoint import swap, CheckpointPolicy

# 基础使用：将所有中间激活 swap 到 CPU
output = swap(model.layer, x)

# 带 policy_fn：选择性 swap
def swap_policy(target):
    if target.is_small_op:
        return CheckpointPolicy.MUST_SAVE  # 小 op 不 swap
    return None  # 其余 swap

output = swap(model.layer, x, policy_fn=swap_policy)
```

### 3. 模块级 wrapper

```python
from hyper_parallel.core.activation_checkpoint import checkpoint_wrapper, swap_wrapper

# 用 checkpoint_wrapper 替换模块
model.layer = checkpoint_wrapper(model.layer, policy="full")

# 用 swap_wrapper 替换模块
model.layer = swap_wrapper(model.layer, offload_to="cpu")

# 组合使用：部分层用 checkpoint，部分层用 swap
for i, layer in enumerate(model.layers):
    if i % 2 == 0:
        model.layers[i] = checkpoint_wrapper(layer)
    else:
        model.layers[i] = swap_wrapper(layer)
```

### 4. CheckpointPolicy 精细控制

```python
from hyper_parallel.core.activation_checkpoint import checkpoint, CheckpointPolicy, SwapManager

# MUST_SWAP 策略：配合 SwapManager 使用
def detailed_policy(target):
    if target.name == "large_attention":
        return CheckpointPolicy.MUST_SWAP   # swap 到 CPU
    elif target.name == "small_proj":
        return CheckpointPolicy.MUST_RECOMPUTE  # 重计算
    return CheckpointPolicy.MUST_SAVE  # 保存

manager = SwapManager()
output = checkpoint(model.layer, x, policy_fn=detailed_policy)
```

### 5. Group Swap（批量 swap 融合）

```python
# 开启 group_swap：多个 swap 操作合并为批量 DMA 传输
output = swap(model.layer, x, group_swap=True)
output = checkpoint(model.layer, x, policy_fn=my_policy, group_swap=True)
```

Group Swap 将多个小 tensor 的 DMA 传输合并为一次批量传输，减少 DMA launch overhead，提升 swap 效率。每个 group 限制为 32 MiB，保持 DMA chunk 大粒度同时避免单次传输过大。

### 6. 与 LlamaFactory 集成

```python
# LlamaFactory 已集成 activation recompute & swap
# 参考 examples/ 中的集成示例
```

---

## 与其他并行策略组合

### FSDP + Activation Checkpoint

```python
from hyper_parallel import fully_shard, init_device_mesh
from hyper_parallel.core.activation_checkpoint import checkpoint_wrapper

mesh = init_device_mesh("npu", (dp_size,), mesh_dim_names=("dp",))
model = fully_shard(model, mesh=mesh)

# 对 FSDP 分片后的子模块应用 checkpoint
model.transformer.layer = checkpoint_wrapper(model.transformer.layer)
```

### TP + Activation Swap

```python
from hyper_parallel import ColwiseParallel, RowwiseParallel, parallelize_module, init_device_mesh
from hyper_parallel.core.activation_checkpoint import swap_wrapper

tp_mesh = init_device_mesh("npu", (tp_size,), mesh_dim_names=("tp",))
parallelize_module(model, tp_mesh, tp_plan)

# TP 场景下 swap 更有效：每个 rank 只需 swap 1/tp_size 的激活
model.transformer.layer = swap_wrapper(model.transformer.layer)
```

### PP + Activation Swap

Pipeline Parallel 场景下 Activation Swap 特别有价值：每个 stage 只需保存自己 stage 的激活，swap 后 NPU HBM 几乎只保存当前 micro-batch 的计算状态。

```python
from hyper_parallel import PipelineStage, Schedule1F1B
from hyper_parallel.core.activation_checkpoint import checkpoint_wrapper, swap_wrapper

# PP stage 内部使用 checkpoint + swap
for layer in stage_model.layers:
    stage_model.layers[layer] = swap_wrapper(stage_model.layers[layer])
```

---

## 性能建议

1. **小 tensor 不要 swap**：swap 的 DMA 启动开销对于小 tensor（如 bias）可能超过收益，用 `policy_fn` 过滤
2. **优先对大激活层使用 swap**：Attention 输出、MLP 中间激活等大 tensor 最适合 swap
3. **Group Swap**：开启 `group_swap=True` 可减少 DMA 次数，提升整体效率
4. **Checkpoint vs Swap 选择**：
   - 计算密集层（如 softmax）：用 checkpoint，重计算开销可接受
   - 内存密集层（如大型 MLP）：用 swap，DMA 传输开销更低
5. **协同配置**：混合使用 checkpoint 和 swap 可最大化内存节省并最小化计算/传输开销