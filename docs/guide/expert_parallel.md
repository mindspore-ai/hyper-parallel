# EP Expert Parallelism Guide

**English** | [简体中文](expert_parallel.zh.md)

HyperParallel provides expert parallelism to distribute MoE expert layers across devices. It supports standard all-to-all EP, TP-only EP, two-dimensional EP+TP, and a complete set of MoE building blocks and load balancing mechanisms.

## Core Concepts

Expert parallelism distributes MoE experts across devices and uses all-to-all communication to dispatch and combine tokens.

| Strategy | Description | Mesh requirement |
|------|------|-----------|
| ExpertParallel | Standard all-to-all EP | 1-D ("ep",) |
| TensorParallel | TP-only weight sharding | 1-D ("tp",) |
| ExpertTensorParallel | Two-dimensional EP+TP | 2-D ("ep", "tp") |

## API Overview

**Parallel strategy interfaces:**

| Interface | Description |
|------|------|
| `ExpertParallel` | Standard all-to-all EP |
| `TensorParallel` (within EP) | TP-only weight sharding |
| `ExpertTensorParallel` | Two-dimensional EP+TP |

**MoE building blocks:**

| Interface | Description |
|------|------|
| `FeedForward` | Shared SwiGLU feed-forward expert |
| `GroupedExperts` | Batched expert computation |
| `TokenChoiceTopKRouter` | Top-k token router |
| `MoE` | Top-level MoE orchestrator |
| `update_expert_bias` | Bias update for load balancing |

---

## Basic Usage

### 1. Standard EP

```python
from hyper_parallel import init_device_mesh
from hyper_parallel.core.expert_parallel import MoE, ExpertParallel

# 1. Build the MoE and EP mesh.
moe = MoE(dim=4096, hidden_dim=14336, num_experts=8, top_k=2)
ep_mesh = init_device_mesh("npu", (8,), mesh_dim_names=("ep",))

# 2. Shard experts over the EP mesh.
ExpertParallel().apply(moe.experts, ep_mesh)

# 3. Forward pass with transparent dispatch and combine.
output = moe(x)  # x: [batch, seq_len, dim]
```

### 2. Two-Dimensional EP + TP

```python
from hyper_parallel.core.expert_parallel import ExpertTensorParallel

mesh = init_device_mesh("npu", (ep_size, tp_size), mesh_dim_names=("ep", "tp"))

# Apply EP+TP on a two-dimensional mesh.
ExpertTensorParallel().apply(moe.experts, mesh["ep", "tp"])
```

---

## MoE Building Blocks

### GroupedExperts

Batched expert computation with three forward paths:

| Path | Backend | Trigger |
|------|------|----------|
| `_run_experts_for_loop` | All | `use_grouped_mm=False` (default) |
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
    score_func="sigmoid",  # "sigmoid" or "softmax"
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
    load_balance_coeff=0.01,    # Set when enabling auxiliary loss.
    use_grouped_mm=False,
)
output = moe(x)  # x: [batch, seq_len, dim]
```

---

## Load Balancing

### expert_bias + aux_loss

```python
moe = MoE(dim=4096, hidden_dim=14336, num_experts=8, top_k=2, load_balance_coeff=0.01)

# In the training loop, update expert_bias after the optimizer step.
from hyper_parallel.core.expert_parallel import update_expert_bias

update_expert_bias(moe, lr=1e-3, num_recomputations=1)

# Auxiliary loss is injected automatically into router weight gradients.
# Access the scalar loss through output._load_balance_loss.
loss_value = output._load_balance_loss
```

---

## Decoupled MoE+EP Token Dispatch

Token permutation and dispatch are decoupled from forward for more flexible composition:

```python
# Decoupled dispatch provides safe state handling for multiple PP+EP micro-batches.
# DispatchContext prevents unintended sharing between instances.
```

---

## Combine with Other Parallel Strategies

```python
mesh = init_device_mesh("npu", (dp_size, ep_size, tp_size), mesh_dim_names=("dp", "ep", "tp"))

# Apply EP+TP on the expert dimensions.
ExpertTensorParallel().apply(moe.experts, mesh["ep", "tp"])

# Apply FSDP on the DP dimension.
fully_shard(model, mesh=mesh["dp"])
```

EP hooks do not interfere with FSDP unshard/reshard or pipeline micro-batch scheduling.

---

## Further Reading

- [API Reference](../api/api_reference.md) — Detailed EP interfaces
