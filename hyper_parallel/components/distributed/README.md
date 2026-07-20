# components/distributed — 双模式 DTensor 并行策略

> 设计文档：`docs/detailed_design/05_dual_mode_dtensor_parallel_strategy.md`
> 开发方案：`docs/dev_plan_05_dual_mode_dtensor.md`

独立可用的 DTensor 分片组件：**ShardingPlanner 自动推导** + **ShardingApplier 双模式应用**，
零依赖 `recipes/` / `_transformers/` / `models/` / `datasets/`（见 `test_s5_zero_dep_lint.py`）。

## 快速开始

```python
from hyper_parallel.core.dtensor.device_mesh import init_device_mesh
from hyper_parallel.components.distributed import ShardingPlanner, apply_sharding_plan

mesh = init_device_mesh("cpu", (4,), mesh_dim_names=("tp",))
planner = ShardingPlanner()
plan = planner.plan(model, mesh, tp_size=4)          # 编译期推导（6-phase）
model, tp_grad_info = apply_sharding_plan(model, plan, mesh)   # production 应用
# validate 校验：apply_sharding_plan(model, plan, mesh, validate_mode=True)
```

独立示例（gloo/CPU 可跑）：`examples/distributed_only_tp.py`。

## 双模式语义

| | production | validate |
|---|---|---|
| 参数 | build 期永久解包为 plain local tensor | 保持 DTensor |
| 前向 | 纯 local tensor + PrecompiledBoundary | DTensor dispatch 传播 + out_src/out_dst 校验 |
| 反向 | local autograd（梯度落 local 分片） | local autograd（同左，05 §1.0） |
| tp_grad_info | 返回（供 FSDP2 fully_shard） | None |

**架构约束（双模式等价可达 kernel 级精度）**：凡 DTensor dispatch 隐含或无法表达
数据相关逻辑的模块（embedding mask、attention K/V gather、MoE all-to-all），两模式
必须用**同一份 local-region wrapper** 显式重建该逻辑，区域内计算路径逐指令一致。

## validate 模式的校验豁免/声明式清单（D-01''/D-03'）

- **attention（CP>1）**：out_src 为声明式——CP wrapper 出口按声明 `from_local`
  重包装（区域内 SDPA 对 K/V 做显式 all-gather，dispatch 无法派生该语义）。
- **MoE（`_use_local_map`）**：out_src 为声明式——all-to-all 的数据相关性使
  placement 无法派生；in 契约仍由 boundary 正常校验。
- 其余模块（embed/norm/mlp/lm_head）：out_src 由 DTensor dispatch 派生校验（核心校验）。

## 关键设计修订（相对 05 文档初稿）

| # | 内容 |
|---|------|
| D-01'' | validate 的 CP 与 production 注入**同一个** all-gather wrapper（否决 ring/dispatcher） |
| D-02 | production embed 注入 Megatron 风格 masked embedding wrapper（解包后 vocab mask 丢失） |
| D-03' | MoE 统一走 local region（前向-only，无反向缝合） |
| D-04 | CP causal mask：is_causal 且 q_len≠kv_len 时替换为 offset-aware 显式 mask |
| D-05 | embed 的 CP 契约：batch 已被数据管道 CP 切分 → in/out CP 维 Shard(1) |
| D-06 | MLP/MoE 的 CP 维全程 Shard(1)（pointwise，TP×CP 布局一致性） |
| D-07 | lm_head 的 CP 维 Shard(1)（R8：boundary CP 维恒 identity；loss 在本地 chunk 计算） |
| D-08 | MOE_EXPERT 的 TP placement 按参数 ndim 感知（3D [E,out,in]：colwise=Shard(1)、rowwise=Shard(2)） |

## 目录

```
sharding_config.py    # MeshAxisName/NamedPlacement/ShardingPlan/ModuleShardingSpec
                      #   /ShardingTemplate/TEMPLATES/PlacementMismatchError
param_role.py         # ParamRole(13) + ParameterClassifier + 默认命名规则
sharding_planner.py   # ShardingPlanner 6-phase + ARCH_OVERRIDES + SPECIAL_HANDLERS
sharding_applier.py   # apply_sharding_plan + Phase 0/A/B/C/D + 五路 forward 包装
precompiled_boundary.py # PrecompiledBoundary/RedistOp/_classify_collective
tp_grad.py            # build_tp_grad_info + tied 归一化
cp_utils.py           # flex_cp_allgather + shard_batch_for_cp + _shard_seq_lens_for_cp
local_region.py       # DTensor→local→DTensor 局部区域（前向-only）
sharding/apply.py     # _local_params_context/_set_param_by_path（canonical）
testing/grad_equiv.py # M_D.15a 梯度等价工具
```

## 测试

```bash
python -m pytest tests/components/distributed/ -q
```

单进程用例直接跑；多进程用例经 `run_dist`（spawn + gloo/CPU，macOS 可跑），
覆盖 TP/CP/EP 及两两组合的 plan golden、production 数值（vs 单卡参考）、
validate 校验与双模式等价（S5.3 梯度 / S5.4 输出）。
