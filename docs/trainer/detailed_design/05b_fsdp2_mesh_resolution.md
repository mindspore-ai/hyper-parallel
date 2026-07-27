# 05b：FSDP2 mesh 解析——从全量 DeviceMesh 到 fully_shard 通信域

> **状态**：2026-07-24 定稿
> **篇幅**：~200 行 → 这是设计 doc，不包含完整实现代码；代码见 `hyper_models/components/distributed/fsdp2.py`。

## 1. 问题

`ShardingPlanner.plan()` 和 `apply_sharding_plan()` 操作的是**全量 DeviceMesh**（含 `dp*` / `tp` / `cp` / `ep` / `pp` 所有轴）。但 `fully_shard`（`hyper_parallel.core.fully_shard`）**只能接收 1D 或 2D 的纯 FSDP mesh**（[hyperlink]( ../../../hyper_parallel/platform/torch/fully_shard/scheduler.py#L72-L80)）：

```python
# fully_shard 内部 mesh 校验：
if self.mesh.ndim == 1:
    self.mesh_info = FSDPMeshInfo(mesh=self.mesh, shard_mesh_dim=0)
elif self.mesh.ndim == 2:
    self.mesh_info = HSDPMeshInfo(mesh=self.mesh, shard_mesh_dim=1, replicate_mesh_dim=0)
else:
    raise ValueError("only supports 1D or 2D meshes")
```

如果把 `(dp=4, tp=2)` 的全量 mesh 直接传入，ndim=2 → FSDP 会把 dim 0 当 replicate、dim 1 当 shard——但 dim 1 实际上是 **tp 维**（张量并行，每个 rank 持不同参数 shard），语义完全错误。

因此需要一个**FSDP mesh 解析层**，从全量 mesh 中提取"权重通信域"。

## 2. 核心概念：权重通信域

### 2.1 定义

**权重通信域** = 持有**同一参数 shard**、但处理**不同数据**的所有 rank 组成的通信域。

在这个域内，FSDP 需要做两件事：
- **参数分片**（shard dimension）：把同一份参数切分到域内各 rank，节省显存
- **梯度同步**（replicate dimension）：域内各 rank 的梯度需要 all-reduce（因为它们处理不同数据，梯度不同）

### 2.2 哪些轴属于权重通信域

| 轴名 | 是否属于 FSDP 域 | 原因 |
|------|:---:|------|
| `dp*`（dp / dp_replicate / dp_shard / dp_cp ...） | ✅ 是 | 数据并行——不同 rank 处理不同 batch/sample |
| `cp` | ✅ 是 | 上下文并行——不同 rank 处理不同序列 chunk，参数相同但梯度不同 |
| `tp` | ❌ 否 | 张量并行——每个 rank 持参数的**不同 shard**，梯度已经是 partial |
| `pp` | ❌ 否 | 流水线并行——不同 stage 持有不同层 |
| `ep`（old-style，mesh 原生轴） | ❌ 否 | 专家并行——不同 rank 持有不同 expert，ep 轴已在 DTensor layout 中 |
| `edp`（D-10 派生 expert mesh 的轴） | ✅ 是 | 专家数据并行——同组 EP rank 处理不同数据，见 §5 |

### 2.3 公式

```
FSDP mesh axes = {all mesh axes} − {tp, pp, ep(old-style)}
```

从全量 DeviceMesh 中提取：

```python
def _resolve_fsdp_mesh(device_mesh):
    fsdp_axes = [name for name in device_mesh.mesh_dim_names
                 if name not in ("tp", "pp", "ep")]
    return device_mesh[tuple(fsdp_axes)] if len(fsdp_axes) > 1 else device_mesh[fsdp_axes[0]]
```

## 3. ShardingPlanner 与 HSDP：dp 轴对 Planner 完全透明

一个常见疑问：HSDP 场景下，不同 `dp_replicate` group 的 rank 看到的 rank_list 不同（例如 `dp_rep=0` 的 rank 是 `{0..7}`，`dp_rep=1` 的 rank 是 `{8..15}`），是否需要为每个 replicate 域分别传不同的 mesh 或局部 rank_list 给 ShardingPlanner？

**答案：不需要。** ShardingPlanner 全程操作全量 mesh，`dp*` 轴被自动过滤，不参与 spec 推导。

### 3.1 DeviceMesh.__getitem__ 的动态解析机制

DeviceMesh 的子 mesh 切片是**按 rank 坐标动态解析**的，不是静态的全量切片：

```
全量 mesh: shape=(2,4,2), dims=("dp_replicate", "dp_shard", "tp")
           rank_list = (0..15)

rank (0,0,0) → mesh["tp"] ──→ tp 子 group {0,1}   ← 和 rank (0,0,1) 同 group
rank (0,0,1) → mesh["tp"] ──→ tp 子 group {0,1}   ← 同一个 group
rank (1,0,0) → mesh["tp"] ──→ tp 子 group {8,9}   ← 不同的 group（不同 dp_rep）
rank (1,0,1) → mesh["tp"] ──→ tp 子 group {8,9}
```

每个 rank 调用 `mesh["tp"]` 拿到的都是**自己所在的 tp 子 group**，rank_list 自动正确——DeviceMesh 内部按当前 rank 的坐标在对应的非 tp 轴上固定到当前值、tp 轴全展开。

### 3.2 Planner 全程操作全量 mesh，dp 轴被过滤

[`_build_mesh_dim_names`]( ../../../hyper_models/components/distributed/sharding_planner.py#L287-L298) 只保留 `("tp", "cp", "ep")` 中 size > 1 的轴：

```python
dtensor_axes = ("tp", "cp", "ep")          # dp* 不在 DTensor 管理范围
active = {"tp"}                             # tp_size=2
# mesh_names = ("dp_replicate", "dp_shard", "tp")
# dp_replicate, dp_shard 不在 dtensor_axes → 被过滤
# → return ("tp",)                          # plan.mesh_dim_names 只含 tp
```

Planner 产出的 plan 是**全局唯一**的：`plan.mesh_dim_names = ("tp",)`，spec 中 placement 的维度与 plan 对齐。所有 rank 共用同一份 plan，不需要也不能为不同的 HSDP replicate group 生成不同的 plan。

### 3.3 apply 阶段：每个 rank 用本地 tp 子 mesh 执行同一份 plan

```python
# sharding_applier.py
full_mesh = mesh           # 保留全量（D-10 EP 派生 expert_mesh 需要）
mesh = _get_active_mesh(mesh, plan.mesh_dim_names)  # → mesh["tp"]
# rank (0,0,0): tp_mesh = sub-mesh of {0,1}
# rank (1,0,0): tp_mesh = sub-mesh of {8,9}
_shard_module_params(module, spec.params, mesh, plan.mesh_dim_names)
```

每个 rank 在自己的 tp 子 mesh 上执行 `distribute_tensor`。虽然 rank 不同，但拿到的 tp 子 group 内部的相对坐标是一致的（tp_rank=0 或 tp_rank=1）。

### 3.4 FSDP2Manager 从全量 mesh 提取 FSDP 域

```python
fsdp_mesh = _resolve_fsdp_mesh(device_mesh)
# fsdp_axes = ["dp_replicate", "dp_shard"]  (tp 被排除)
# → mesh[("dp_replicate", "dp_shard")] → shape (2, 4), 2D
fully_shard(model, mesh=fsdp_mesh)
# → HSDPMeshInfo(shard_mesh_dim=1, replicate_mesh_dim=0)
```

FSDP2Manager 和 ShardingPlanner 的职责完全分离：Planner 只关心 DTensor 管理的轴（tp/cp/ep）；FSDP2Manager 只关心全量 mesh 中排除 DTensor 轴后的 dp*+cp 域。

### 3.5 端到端流程

```
全量 DeviceMesh
shape=(2,4,2), dims=("dp_replicate","dp_shard","tp"), rank_list=(0..15)
        │
        ├─→ ShardingPlanner.plan(model, mesh, tp_size=2)
        │     _build_mesh_dim_names: ("dp_replicate","dp_shard","tp") → ("tp",)
        │     plan 全局唯一，所有 rank 共享
        │     spec: params["q_proj.weight"] = {TP: Shard(0)}
        │
        ├─→ apply_sharding_plan(model, plan, mesh)
        │     每个 rank 本地 mesh["tp"] → distribute_tensor
        │     rank(0,0,0): tp_group={0,1}, tp_rank=0
        │     rank(1,0,0): tp_group={8,9}, tp_rank=0   ← tp_rank 相同
        │
        └─→ FSDP2Manager.parallelize(model, plan)
              fsdp_mesh = mesh[("dp_replicate","dp_shard")] → shape (2,4)
              fully_shard(model, mesh=fsdp_mesh)
              → unified = concat((2,4), (2,)) = (2,4,2)
                 dim0(dp_rep)=replicate, dim1(dp_shard)=shard, dim2(tp)=DTensor
```

**不需要也不应该**对每个 HSDP replicate 域分别调 ShardingPlanner 或传不同的 rank_list。Planner 推导的是 TP/CP/EP 层面的参数分片策略，与 HSDP 的数据并行分组结构完全正交。

## 4. 各并行组合的 FSDP mesh 解析

### 4.1 仅 FSDP（无 TP/CP/EP）

```
全量 mesh: (dp,)  无 mesh_dim_names 或仅 dp 轴
FSDP mesh: device_mesh 本身（1D）
fully_shard(model, mesh=device_mesh)
```

### 4.2 TP + FSDP

```
全量 mesh: (dp=4, tp=2), dims=("dp", "tp")
FSDP mesh: device_mesh["dp"] → shape (4,), 1D

fully_shard(model, mesh=device_mesh["dp"])
```

`fully_shard` 内部 unified mesh = `concat((4,), (2,))` = `(4, 2)`（FSDP prefix + DTensor suffix）。

- TP-Shard(0) 参数：`[Replicate, Shard(0)]` → FSDP shard 覆盖 dim 0 → `[StridedShard(0, split_factor=2), Shard(0)]`
- TP-Replicate 参数：`[Replicate, Replicate]` → `[Shard(0), Replicate]`

### 4.3 TP + CP + FSDP

```
全量 mesh: (dp=4, tp=2, cp=2), dims=("dp", "tp", "cp")
FSDP mesh: device_mesh[("dp", "cp")] → shape (4, 2), 2D

fully_shard(model, mesh=device_mesh[("dp", "cp")])
# ndim=2 → HSDPMeshInfo(shard_mesh_dim=1, replicate_mesh_dim=0)
```

**为什么 CP 要纳入 FSDP 域**：CP 不切参数（CP 维参数恒 `Replicate()`），但 CP rank 处理不同的序列 chunk → 梯度不同 → 需要在这个域内做梯度同步。

`fully_shard` 内部 unified mesh = `concat((4,2), (2,))` = `(4, 2, 2)`：
- dim 0 (dp): replicate → 梯度 all-reduce
- dim 1 (cp): shard → FSDP 参数分片（节省显存）
- dim 2 (tp): DTensor layout（`Shard(0)` 或 `Replicate()`）

Unified mesh 的 gradient all-reduce group 由所有 `Replicate()` placement 的轴决定（[param.py:L251-258]( ../../../hyper_parallel/platform/torch/fully_shard/param.py#L251-L258)）：

| 参数类型 | Unified placement | 梯度 all-reduce 域 |
|---|---|---|
| TP-Shard(0) | `[Replicate, StridedShard(0, factor=tp), Shard(0)]` | dp（dim 0），tp_grad_info 标记 Shard → FSDP skip |
| TP-Replicate | `[Replicate, Shard(0), Replicate]` | dp + tp（dim 0, 2 均为 Replicate） |

### 4.4 HSDP + TP

```
全量 mesh: (dp_rep=2, dp_shard=4, tp=2), dims=("dp_replicate", "dp_shard", "tp")
FSDP mesh: device_mesh[("dp_replicate", "dp_shard")] → shape (2, 4), 2D

fully_shard(model, mesh=device_mesh[("dp_replicate", "dp_shard")])
# dim 0 (dp_replicate): replicate, dim 1 (dp_shard): shard
```

### 4.5 old-style EP（mesh 含 "ep" 轴）+ TP + FSDP

```
全量 mesh: (dp=4, tp=2, ep=2), dims=("dp", "tp", "ep")
FSDP mesh: device_mesh["dp"] → shape (4,), 1D
# "ep" 被排除——ep 轴已在 DTensor layout 中管理

fully_shard(model, mesh=device_mesh["dp"])
```

**为什么 old-style EP 的 `ep` 轴要排除**：expert 参数 placement 为 `{TP: Shard(…), EP: Shard(0)}`，EP Shard(0) 意味着不同 EP rank 持有不同 expert 的完整权重。EP rank 间不共享参数 → 不应该在同一个 FSDP 分片组内。

## 5. D-10 TP-extend-EP：双 FSDP 域

### 5.1 背景

D-10 模式下，mesh **不含 `"ep"` 轴**。`apply_sharding_plan` 在运行时从全 dense 区域派生 `expert_mesh (edp, ep)`（[sharding_applier.py:_build_expert_mesh]( ../../../hyper_models/components/distributed/sharding_applier.py#L231-L237)）：

```python
# 全量 mesh: (dp=4, tp=2), ep_size=4
# D = dp × tp = 8
# expert_mesh: (edp=2, ep=4), dims=("edp", "ep")
#   ep 组（a2a 通信域）: {0,1,2,3} / {4,5,6,7}
#   edp 组（专家数据并行）: {0,4} / {1,5} / {2,6} / {3,7}
```

### 5.2 两个独立的 FSDP 域

| 参数类别 | DTensor mesh | FSDP mesh | 说明 |
|---|---|---|---|
| 密集参数（non-expert） | `mesh["tp"]` (1D) | `mesh["dp"]` (1D) | 从主 mesh 提取 dp 子 mesh |
| 专家参数（expert weights） | `expert_mesh` (2D: edp, ep) | `expert_mesh["edp"]` (1D) | 从派生 expert_mesh 提取 edp 子 mesh |

### 5.3 实现

```python
def _parallelize_with_ep(model, plan, fsdp_mesh):
    # 1. 重建 expert_mesh（apply 阶段构建但未持久化到 plan）
    expert_mesh = _build_expert_mesh_from_plan(plan, device_mesh)  # (edp, ep)

    # 2. 密集参数：FSDP over dp+cp 域
    fully_shard(model, mesh=fsdp_mesh)
    # tp_grad_info 会将 expert 参数标记为 Shard → 此调用自动 skip expert 参数

    # 3. 专家参数：对每个 MoE 块单独 FSDP over edp 域
    edp_mesh = expert_mesh["edp"]
    for moe_module in _collect_moe_modules(model, plan):
        fully_shard(moe_module, mesh=edp_mesh)
```

### 5.4 专家参数的 unified mesh

专家参数在 `apply_sharding_plan` 后被 `distribute_tensor` 到 `expert_mesh (edp, ep)` 上，placement 为 `{EP: Shard(0)}`（仅在 expert 维切分）。

`fully_shard(moe_block, mesh=edp_mesh)` 内部的 unified mesh：

```
unified = concat(edp_mesh(edp,), expert_mesh(edp, ep))
        = (edp,) + (edp, ep)  ← DeviceMesh.concatenate 会自动处理重叠轴
```

`edp` 维在 unified mesh 中作为 FSDP shard 维出现，expert 权重原来只在 ep 维 Shard(0)，加上 edp 维的 FSDP 分片后形成 `[Shard(0)_fsdp, Shard(0)_ep]`——expert 权重同时在 edp（数据并行）和 ep（专家并行）两个维度上切分。

## 6. `fully_shard` 内部：`DTENSOR_UNIFIED` 模式

当 `fully_shard` 同时收到显式 FSDP mesh 和已有 DTensor 布局的参数时，进入 `DTENSOR_UNIFIED` 模式（[param.py:L185-198]( ../../../hyper_parallel/platform/torch/fully_shard/param.py#L185-L198)）：

### 6.1 Unified mesh 构造

```python
# param.py:_get_base_spmd_placements
self._spmd_mesh = DeviceMesh.concatenate([self.mesh_info.mesh, self._orig_dtensor_mesh])
dp_prefix_placements = tuple(Replicate() for _ in range(self.mesh_info.mesh.ndim))
return dp_prefix_placements + tuple(self._orig_dtensor_placements)
```

FSDP mesh **前缀**到 DTensor mesh 前面：
- FSDP 维度初始 placement = `Replicate()`（FSDP 不加 DTensor shard）
- DTensor 维度保持原 placement（来自 `apply_sharding_plan`）

### 6.2 FSDP shard 覆盖

```python
# param.py:_apply_data_parallel_placements
placements[shard_mesh_dim] = shard_placement  # 覆盖 FSDP shard 维
# 如果 DTensor 维也在同一 tensor dim 上 shard → StridedShard
```

### 6.3 梯度同步 group 推导

```python
# param.py:_build_layout_driven_group_info
group_axes = [axis for axis, placement in enumerate(placements)
              if placement.is_replicate()]
group_axes = [axis for axis in group_axes if axis != shard_mesh_dim]
```

梯度 all-reduce group = unified mesh 上所有 `Replicate()` placement 对应的轴（排除 FSDP 自身的 shard 维）。这意味着 FSDP 的梯度同步 group 自然包含 dp 轴（初始 Replicate）和 DTensor 中的 Replicate 轴——**不需要手动计算通信域**。

## 7. FSDP2Manager 接口

```python
class FSDP2Manager:
    def __init__(self, config: FSDP2Config, mesh_context):
        self.config = config
        self.device_mesh = mesh_context.device_mesh  # 全量 DeviceMesh

    def parallelize(
        self,
        model: nn.Module,              # apply_sharding_plan 之后的模型
        tp_shard_plan: ShardingPlan,    # planner 输出（检测 D-10 EP）
        tp_grad_info: dict | None,      # （待接线）TP 梯度旁路信息
    ) -> nn.Module:
```

调用方（`infrastructure.py`）无需感知 mesh 解析细节：

```python
# infrastructure.py 中现有代码无需修改：
fsdp2_manager = _instantiate_fsdp2(config=strategy_cfg, mesh_context=mesh)
model = fsdp2_manager.parallelize(model, tp_shard_plan=plan, tp_grad_info=tp_grad_info)
```

## 8. 总结表

| 并行组合 | 全量 mesh dims | FSDP mesh | FSDP mesh 维数 |
|---|---|---|---|
| 仅 FSDP | `(dp,)` | `device_mesh` | 1D |
| TP + FSDP | `("dp", "tp")` | `mesh["dp"]` | 1D |
| TP + CP + FSDP | `("dp", "tp", "cp")` | `mesh[("dp","cp")]` | 2D (HSDP) |
| HSDP + TP | `("dp_rep","dp_shard","tp")` | `mesh[("dp_rep","dp_shard")]` | 2D (HSDP) |
| Old-style EP + TP + FSDP | `("dp","tp","ep")` | `mesh["dp"]` | 1D |
| D-10 EP + TP + FSDP | `("dp","tp")` | 密集: `mesh["dp"]`，专家: `expert_mesh["edp"]` | 1D × 2 |

**统一规则**：传给 `fully_shard` 的 mesh = 全量 mesh 中**权重的数据并行通信域**。从全量 mesh_dim_names 中排除 `tp`/`pp`/`ep`(old-style)，保留所有 `dp*` 和 `cp` 轴。
