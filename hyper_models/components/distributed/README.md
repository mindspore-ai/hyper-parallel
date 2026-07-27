# components/distributed — 双模式 DTensor 并行策略

> 使用教程：`docs/components_distributed_tutorial.md`
> 设计文档：`docs/detailed_design/05_dual_mode_dtensor_parallel_strategy.md`
> 开发方案：`docs/dev_plan_05_dual_mode_dtensor.md`

独立可用的 DTensor 分片组件：**ShardingPlanner 自动推导** + **ShardingApplier 双模式应用**，
零依赖 `recipes/` / `_transformers/` / `models/` / `datasets/`（见 `test_s5_zero_dep_lint.py`）。

## 快速开始

```python
from hyper_parallel.core.dtensor.device_mesh import init_device_mesh
from hyper_models.components.distributed import ShardingPlanner, apply_sharding_plan

mesh = init_device_mesh("cpu", (4,), mesh_dim_names=("tp",))
planner = ShardingPlanner()
plan = planner.plan(model, mesh, tp_size=4)          # 编译期推导（6-phase）
model, tp_grad_info = apply_sharding_plan(model, plan, mesh)   # production 应用
# validate 校验：apply_sharding_plan(model, plan, mesh, validate_mode=True)
```

独立示例（gloo/CPU 可跑）：`examples/distributed/`（tp/cp/ep/自定义模块五例）。

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
- **MoE（`use_local_map`）**：out_src 为声明式——all-to-all 的数据相关性使
  placement 无法派生；in 契约仍由 boundary 正常校验。
- 其余模块（embed/norm/mlp/lm_head）：out_src 由 DTensor dispatch 派生校验（核心校验）。

## 自定义 wrapper 接口速查（详见 05 §8.6）

经 `ShardingPlanner(plan_overrides={fqn: spec})` 的 spec 字段声明；用户代码
恒工作在 local tensor 世界，DTensor↔local 缝合由框架负责。

| 字段 | 签名 | 做什么 | 何时用 | 生效方式 |
|---|---|---|---|---|
| `use_local_map` | `bool` | **纯门控**：模块自身 forward 即数据相关逻辑 → 走 local-region 骨架（解析链末位来源） | 自研模块 forward 内含数据相关逻辑（如自带 a2a 的 EP-aware MoE） | 解析链环 3 |
| `local_compute_fn` | `fn(module, *args, **kw) -> Tensor` | 替换骨架内的计算函数（骨架的边界缝合/双模式不变） | 自研 MoE 注入自定义 dispatch（自定义 router/expert 布局/DeepEP） | 解析链环 1，**不改写任何字段** |
| `inner_target` | `str`（属性名/`"self"`） | **纯位置**：指定 inner attention 子模块（自动定位失败 fail-fast 时的指定入口） | 自研 attention 符合 (q,k,v) 或 HF 原语约定，但自动定位失败/有歧义 | target 链环 1，不改写任何字段 |
| `inner_wrapper` | `str`（注册表名）或 `fn(target, cp_mesh) -> None` | **纯行为**：str 显式固定内置 CP wrapper（`"sdpa_qkv"/"sdpa_hf"/"flex_qkv"/"flex_hf"`，注册表 `CP_WRAPPER_REGISTRY` 开放注册）；callable 全自定义整体接管 | 启发式会猜错行为时 str 固定；内置四路覆盖不到（如内部直调 flash_attn_varlen）时 callable | wrapper 链环 1-2，不改写任何字段 |

> **内置 wrapper 说明**：EP 侧内置 `_hf_native_ep_compute`（planner 识别 HF 原生
> MoE 且 ep_size>1 时经 `_ep_size>0` 意图自动注入，local 链环 2）；CP 侧内置
> 注册表四路（K/V all-gather + D-04 causal 修正 + 双模式容错，缺省按启发式
> 2×2 分派，结果见 apply 日志与 `spec._resolved_inner_wrapper`，可用 str 固定）。
> 门控均为**派生语义**：`_resolve_local_compute_fn` / `_resolve_inner_wrapper`
> 解析非 None 即注入——声明互不嵌套（05 §4.4.2/§4.4.3/§8.6）。

## 关键设计修订（相对 05 文档初稿）

| # | 内容 |
|---|------|
| D-01'' | validate 的 CP 与 production 注入**同一个** all-gather wrapper（否决 ring/dispatcher） |
| D-02 | production embed 注入 Megatron 风格 masked embedding wrapper（解包后 vocab mask 丢失） |
| D-03' | MoE 统一走 local region（前向-only，无反向缝合） |
| D-04 | CP causal mask：is_causal 且 CP 激活（cp_size>1）时替换为 offset-aware 显式 mask |
| D-05 | embed 的 CP 契约：batch 已被数据管道 CP 切分 → in/out CP 维 Shard(1) |
| D-06 | MLP/MoE 的 CP 维全程 Shard(1)（pointwise，TP×CP 布局一致性） |
| D-07 | lm_head 的 CP 维 Shard(1)（R8：boundary CP 维恒 identity；loss 在本地 chunk 计算） |
| D-08 | MOE_EXPERT 的 TP placement 按参数 ndim 感知（3D [E,out,in]：colwise=Shard(1)、rowwise=Shard(2)） |
| D-09 | HF 原生 MoE EP 直通（05 §6.4.7）：planner 识别 per-expert/batched 布局并生成 EP 元数据（`_ep_stack`/`_moe_router`/`_ep_size`），Phase A 前置 `_stack_moe_experts` 堆叠；`_hf_native_ep_compute` 经 local 链环 2 注入；a2a 按后端分派（NCCL/HCCL 不等长 `all_to_all` 零填充；gloo pad-to-max `all_to_all_single`） |
| D-10 | TP-extend-EP（05 §6.4.8）：`ep_size` 即扩展 EP 组大小（无单独 etp 配置）；全 dense 区域重分区为派生 expert mesh `(edp, ep)`（EP 组 = flatten 连续 ep_size 个 rank，先跨完 TP 组再向 dp/cp 扩展）；expert 权重仅 `{EP: Shard(0)}`（每 rank 持完整 expert，无第二轴）；通信流与 Megatron `MoEAlltoAllTokenDispatcher` 同构，无 all_gather/reduce_scatter |
| D-11 | fused batched expert 布局（HF 2025 重构后：`gate_up_proj [E,2I,H]` + `down_proj [E,H,I]`）天生 stacked 无需堆叠，直接标 `{EP: Shard(0)}`，计算侧 chunk 出 gate/up |
| D-12 | inner-wrap 双解析链（`_resolve_inner_target`/`_resolve_inner_wrapper`）：target 定位 fail-fast + `CP_WRAPPER_REGISTRY` 注册表（str 固定/启发式分派/发火检测/日志与 `_resolved_inner_wrapper` 回写），门控派生不改写标记 |
| D-13 | local-region compute_fn 单一解析链（`_resolve_local_compute_fn`）：`local_compute_fn` > EP 注入意图 > `use_local_map` 纯门控；骨架门控为解析结果的派生（非存储 bool），声明互不嵌套 |
| D-14 | DeepSeek MLA 支持（v2/v3）：新增 `ParamRole.REPLICATED`（全维 Replicate，仅 `ARCH_OVERRIDES` 指派）——q_a/kv_a 下投影全复制、q_b/kv_b 上投影按 head 维 COLWISE，MLA attention 与标准 attention 模板同构（已用 transformers 仓真实 DeepseekV2/V3 模型验证 SKIP=0、attention 边界带 `cp_attn`） |
| D-15 | Phase 5 链式契约比较由 `PlacementMismatchError` 降级为 `logger.warning`：该比较是 placement 值相等、无 shape 感知，边上 reshape/transpose 的合法场景（如 `[B,S,H]` Shard(1) fold 成 `[B*S,H]` Shard(0)）必然不等，报错会误杀合法配置；声明正确性由 validate 模式（DTensor dispatch + 数值对拍）兜底。填充缺省 in_src 与 `_is_terminal` 标记不变 |
| D-16 | plan_overrides 嵌套 spec fail-fast（`_check_no_nested_overrides`）：override fqn 不得是派生边界（或另一 override）的祖先/后代，命中即 ValueError——边界假设扁平链（Phase 5 的 out_dst→in_src 参照只在模块出口成立，嵌套时内层实际看到祖先 in_dst），且同一参数会被切两次（production 静默错）；同树只支持同 fqn 替换 |
| D-17 | TP 本地头数改写（`head_count.py`，AutoModel 同款语义）：兼容 HF 显式 `num_heads` reshape 写法——q/k/v colwise `Shard(0)` 的模块，凡前向看到 local tensor（production 全量 / validate 仅 local-region）即把缓存头数属性（`num_heads`/`num_attention_heads`/`n_heads` 等 Q 侧 7 名 + `num_key_value_heads`/`num_kv_heads`/`kv_heads` KV 侧 3 名，清单来自 transformers 全库调研）整除为本地值；不改 config/head_dim/num_key_value_groups，幂等（原值存 `_hp_full_head_counts`）；validate 普通 boundary 不改写（DTensor 全局逻辑形状下显式头数天然正确） |

## 目录

```
sharding_config.py    # MeshAxisName/NamedPlacement/ShardingPlan/ModuleShardingSpec
                      #   /ShardingTemplate/TEMPLATES/PlacementMismatchError
param_role.py         # ParamRole(14) + ParameterClassifier + 默认命名规则
sharding_planner.py   # ShardingPlanner 6-phase + ARCH_OVERRIDES + SPECIAL_HANDLERS
sharding_applier.py   # apply_sharding_plan + Phase 0/A/B/C/D + 五路 forward 包装
precompiled_boundary.py # PrecompiledBoundary/RedistOp/_classify_collective
tp_grad.py            # build_tp_grad_info + tied 归一化
head_count.py         # D-17 TP 本地头数改写（显式 num_heads 写法兼容）
cp_utils.py           # flex_cp_allgather + shard_batch_for_cp + _shard_seq_lens_for_cp
ep_utils.py           # _ep_all_to_all 后端分派 + MOE_ROUTER_ADAPTERS + _hf_native_ep_compute（D-09/D-10）
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
