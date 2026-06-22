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
from hyper_parallel.core.activation_checkpoint import checkpoint, CheckpointPolicy

# 基础使用：将 model.layer 包装为 checkpoint 区域，x 是传给 model.layer 的输入
output = checkpoint(model.layer, x)

# 带 policy_fn：选择性保存 / 卸载 / 重计算
def my_policy(ctx, op, *args, **kwargs):
    # 返回 MUST_SAVE 表示保存该算子的输出；返回 MUST_SWAP 表示卸载该算子的输出到 HOST 侧；返回 MUST_RECOMPUTE 表示反向时重计算。
    op_name_attr = getattr(op, "name", None)
    op_name = op_name_attr() if callable(op_name_attr) else (op_name_attr or str(op))
    if op_name in {"LayerNormExt", "aten.addmm.default"}:
        return CheckpointPolicy.MUST_SAVE
    return CheckpointPolicy.MUST_RECOMPUTE

output = checkpoint(model.layer, x, policy_fn=my_policy)

# 带 swap_inputs：checkpoint 区域保存的输入也 offload 到 CPU
output = checkpoint(model.layer, x, swap_inputs=True)

# 带 context_fn：为正向和重计算阶段分别提供上下文管理器
def context_fn():
    return forward_context(), recompute_context()

output = checkpoint(model.layer, x, context_fn=context_fn)
```

### 2. 函数式 swap

```python
from hyper_parallel.core.activation_checkpoint import swap, CheckpointPolicy, SwapManager

# 基础使用：将所有中间激活 swap 到 CPU
output = swap(model.layer, x)

# 带 policy_fn：选择性 swap
def swap_policy(target):
    if target.is_small_op:
        return CheckpointPolicy.MUST_SAVE  # 小 tensor 不 swap
    return CheckpointPolicy.MUST_SWAP  # 其余 swap

output = swap(model.layer, x, policy_fn=swap_policy)

# 使用 swap 功能时，需要注册层间的 offload / prefetch 关系（可以不是相邻层）
SwapManager().set_forward_prefetch_layer(model.layers[i], model.layers[i + 1])
```

### 3. 声明式接口

```python
from hyper_parallel.core.activation_checkpoint import (
    CheckpointPolicy,
    SwapManager,
    checkpoint_wrapper,
    swap_wrapper,
)

# 以下示例展示不同用法；不要对同一个模块重复嵌套 wrapper。

# 用 checkpoint_wrapper 替换模块：该模块每次 forward 都会进入 checkpoint 区域
model.layers[0] = checkpoint_wrapper(model.layers[0])

# checkpoint_wrapper 支持与 checkpoint 相同的关键字参数，例如 policy_fn、swap_inputs、group_swap
def recompute_policy(ctx, op, *args, **kwargs):
    return CheckpointPolicy.MUST_RECOMPUTE

model.layers[1] = checkpoint_wrapper(model.layers[1], policy_fn=recompute_policy)

# 用 swap_wrapper 替换模块：模块 forward 中保存的中间激活会按策略 offload 到 CPU
def tensor_swap_policy(tensor):
    if tensor.numel() < 1024:
        return CheckpointPolicy.MUST_SAVE  # 小 tensor 保留在设备侧
    return CheckpointPolicy.MUST_SWAP

model.layers[2] = swap_wrapper(model.layers[2], policy_fn=tensor_swap_policy)

# 组合使用：从尚未包装过的模型开始，对不同层选择一种 wrapper
for i, layer in enumerate(model.layers):
    if i % 2 == 0:
        model.layers[i] = checkpoint_wrapper(layer)
    else:
        model.layers[i] = swap_wrapper(layer)

# 使用 swap 功能时，需要注册层间的 offload / prefetch 关系（可以不是相邻层）
SwapManager().set_forward_prefetch_layer(model.layers[i], model.layers[i + 1])

# checkpoint_wrapper 和 swap_wrapper 既可以包裹 nn.Module / Cell，也可以包裹普通 func。
# 包裹 func 时，返回值仍然是可调用对象，调用方式与原函数保持一致。
def block_func(x):
    return model.block(x)

ckpt_block_func = checkpoint_wrapper(block_func)
swap_block_func = swap_wrapper(block_func)

ckpt_output = ckpt_block_func(x)
swap_output = swap_block_func(x)
```

### 4. 单 tensor 卸载

`swap_tensor_wrapper` 适用于只想卸载某几个明确的大激活 tensor。它需要在模块的 `forward` / `construct` 内部调用，会把其中符合条件的 tensor 注册到当前 swap group 中，等待后续 offload / prefetch 调度。

```python
import torch.nn as nn

from hyper_parallel.core.activation_checkpoint import SwapManager, swap_tensor_wrapper

class TransformerBlock(nn.Module):
    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        attn_out = swap_tensor_wrapper(attn_out, tag="attn_out")

        x = x + attn_out
        x = self.norm1(x)
        x = swap_tensor_wrapper(x, tag="norm1_out")

        x = x + self.ffn(x)
        return self.norm2(x)

# 使用 swap_tensor_wrapper 时，同样需要注册层间的 offload / prefetch 关系（可以不是相邻层）
SwapManager().set_forward_prefetch_layer(model.layers[i], model.layers[i + 1])
```

---

## 与其他并行策略组合

### FSDP + Activation Checkpoint / Activation Swap

```python
from hyper_parallel import fully_shard, init_device_mesh
from hyper_parallel.core.activation_checkpoint import checkpoint_wrapper

mesh = init_device_mesh("npu", (dp_size,), mesh_dim_names=("dp",))

# 先对需要的子模块应用 checkpoint_wrapper 或 swap_wrapper，使 wrapper 位于 FSDP 边界内部
model.transformer.layer = checkpoint_wrapper(model.transformer.layer)

# 再应用 FSDP，让 FSDP 包住已经 checkpoint 的模块
model = fully_shard(model, mesh=mesh)
```

### TP + Activation Checkpoint / Activation Swap

```python
from hyper_parallel import ColwiseParallel, RowwiseParallel, parallelize_module, init_device_mesh
from hyper_parallel.core.activation_checkpoint import SwapManager, swap_wrapper

tp_mesh = init_device_mesh("npu", (tp_size,), mesh_dim_names=("tp",))

# 先应用 TP
parallelize_module(model, tp_mesh, tp_plan)

# 再对需要的层或子模块应用 checkpoint_wrapper 或 swap_wrapper
for i, layer in enumerate(model.layers):
    model.layers[i].attn = swap_wrapper(layer.attn)

# 使用 swap 时，需要注册层间的 offload / prefetch 关系（可以不是相邻层）
SwapManager().set_forward_prefetch_layer(model.layers[i], model.layers[i + 1])
```

---

## 性能建议

1. **小 tensor 不要 swap**：swap 的 DMA 启动开销对于小 tensor（如 bias）可能超过收益，用 `policy_fn` 过滤
2. **优先对大激活层使用 swap**：Attention 输出、MLP 中间激活等大 tensor 最适合 swap
3. **Checkpoint vs Swap 选择**：
   - 计算密集层（如 softmax）：用 checkpoint，重计算开销可接受
   - 内存密集层（如大型 MLP）：用 swap，DMA 传输开销更低
4. **协同配置**：混合使用 checkpoint 和 swap 可最大化内存节省并最小化计算/传输开销
