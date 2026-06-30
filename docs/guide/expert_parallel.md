# EP 专家并行使用指南

HyperParallel 提供 Expert Parallel（专家并行）能力，将 MoE 模型的专家层分布到多个设备上。支持标准 all-to-all EP、TP-only EP、EP+TP 二维并行，以及完整的 MoE 构建模块和负载均衡机制。

## 核心概念

专家并行将 MoE 模型中的多个专家分布到不同设备，通过 all-to-all 通信实现 token 的分发和聚合。

| 策略 | 说明 | Mesh 需求 |
|------|------|-----------|
| ExpertParallel | 标准 all-to-all EP | 1-D ("ep",) |
| TensorParallel | TP-only 权重切分 | 1-D ("tp",) |
| ExpertTensorParallel | EP+TP 二维并行 | 2-D ("ep", "tp") |

## 接口概览

**并行策略接口：**

| 接口 | 说明 |
|------|------|
| `ExpertParallel` | 标准 all-to-all EP |
| `TensorParallel`（EP 内） | TP-only 权重切分 |
| `ExpertTensorParallel` | EP+TP 二维并行 |

**MoE 构建模块：**

| 接口 | 说明 |
|------|------|
| `FeedForward` | SwiGLU feed-forward 共享专家 |
| `GroupedExperts` | 批量专家计算 |
| `TokenChoiceTopKRouter` | Top-k token 路由器 |
| `MoE` | MoE 顶层编排器 |
| `update_expert_bias` | 负载均衡 bias 更新 |

---

## 基础使用

### 1. 标准 EP

```python
from hyper_parallel import init_device_mesh
from hyper_parallel.core.expert_parallel import MoE, ExpertParallel

# 1. 构建 MoE 和 EP mesh
moe = MoE(dim=4096, hidden_dim=14336, num_experts=8, top_k=2)
ep_mesh = init_device_mesh("npu", (8,), mesh_dim_names=("ep",))

# 2. 在 EP mesh 上切分专家
ExpertParallel().apply(moe.experts, ep_mesh)

# 3. 正向传递（dispatch 和 combine 透明）
output = moe(x)  # x: [batch, seq_len, dim]
```

### 2. EP + TP 二维并行

```python
from hyper_parallel.core.expert_parallel import ExpertTensorParallel

mesh = init_device_mesh("npu", (ep_size, tp_size), mesh_dim_names=("ep", "tp"))

# EP+TP 在二维 mesh 上
ExpertTensorParallel().apply(moe.experts, mesh["ep", "tp"])
```

---

## MoE 构建模块

### GroupedExperts

批量专家计算模块，支持三种 forward path：

| Path | 后端 | 触发条件 |
|------|------|----------|
| `_run_experts_for_loop` | All | `use_grouped_mm=False`（默认） |
| `_run_experts_grouped_mm` | CUDA | `use_grouped_mm=True` |
| `_run_experts_grouped_mm_npu` | Ascend NPU | `use_grouped_mm=True` |

```python
from hyper_parallel.core.expert_parallel import GroupedExperts

experts = GroupedExperts(dim=4096, hidden_dim=14336, num_experts=8, use_grouped_mm=True)
output = experts(x, num_tokens_per_expert, scores=None)
```

### TokenChoiceTopKRouter

```python
from hyper_parallel.core.expert_parallel import TokenChoiceTopKRouter

router = TokenChoiceTopKRouter(
    dim=4096,
    num_experts=8,
    top_k=2,
    score_func="sigmoid",  # "sigmoid" 或 "softmax"
    num_expert_groups=None,
    num_limited_groups=None,
    route_scale=1.0,
)
top_scores, selected_experts, num_tokens_per_expert = router(x)
```

### MoE

```python
from hyper_parallel.core.expert_parallel import MoE

moe = MoE(
    dim=4096,
    hidden_dim=14336,
    num_experts=8,
    top_k=2,
    score_before_experts=True,  # True: pre-scale input; False: internal weighting
    load_balance_coeff=0.01,    # 启用 aux loss 时设置
    use_grouped_mm=False,
)
output = moe(x)  # x: [batch, seq_len, dim]
```

---

## 负载均衡

### expert_bias + aux_loss

```python
moe = MoE(dim=4096, hidden_dim=14336, num_experts=8, top_k=2, load_balance_coeff=0.01)

# 训练循环中，优化器 step 后更新 expert_bias
from hyper_parallel.core.expert_parallel import update_expert_bias

update_expert_bias(moe, lr=1e-3, num_recomputations=1)

# aux loss 自动注入到 router 权重梯度
# scalar loss 值可通过 output._load_balance_loss 获取
loss_value = output._load_balance_loss
```

---

## MoE+EP token dispatch 解耦

token permutation 和 dispatch 从 forward 中解耦，支持更灵活的组合：

```python
# 解耦后的 dispatch 支持 PP+EP 多 micro-batch 状态安全
# 使用 DispatchContext 防止实例共享问题
```

---

## 与其他并行策略组合

```python
mesh = init_device_mesh("npu", (dp_size, ep_size, tp_size), mesh_dim_names=("dp", "ep", "tp"))

# EP+TP 在专家维度
ExpertTensorParallel().apply(moe.experts, mesh["ep", "tp"])

# FSDP 在 DP 维度
fully_shard(model, mesh=mesh["dp"])
```

EP hooks 不干扰 FSDP unshard/reshard 或 pipeline micro-batch scheduling。

---

## 更多参考

- [API 参考](../api/api_reference.md) — EP 模块接口详细说明
