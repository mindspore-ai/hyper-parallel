## 1. 模块职责

定义 hyper_parallel 的 DTensor 分片配置体系——从 YAML 配置到运行时 DTensor 分片的完整链路。核心差异化能力：**ShardingPlanner 自动推导**替代 AutoModel 的手写 `parallelize_fn`（~400 行/模型 → ~20 行）。

### 核心文件

| 文件 | 职责 |
|------|------|
| `hyper_models/components/distributed/sharding_config.py` | `ShardingPlan` / `ModuleShardingSpec` / `NamedPlacement` 数据模型 |
| `hyper_models/components/distributed/sharding_planner.py` | `ShardingPlanner`：6-phase I/O 契约推导管线 |
| `hyper_models/components/distributed/sharding_applier.py` | `apply_sharding_plan`：双模式应用（validate / production） |
| `hyper_models/components/distributed/precompiled_boundary.py` | `PrecompiledBoundary` / `RedistOp`：编译期通信规划 |
| `hyper_models/components/distributed/param_role.py` | `ParamRole` 枚举 + `ParameterClassifier` |
| `hyper_models/components/distributed/sharding/apply.py` | `_local_params_context` / 路径工具 / `_stack_moe_experts`（D-09b 堆叠） |
| `hyper_models/components/distributed/cp_utils.py` | `flex_cp_allgather`（autograd 版 CP K/V all-gather）+ `shard_batch_for_cp` |
| `hyper_models/components/distributed/ep_utils.py` | `_hf_native_ep_compute` + `MOE_ROUTER_ADAPTERS`（D-09/D-10 EP 直通） |
| `hyper_models/components/distributed/local_region.py` | `local_region`：DTensor↔local 区域缝合（validate / 独立使用） |
| `hyper_models/components/distributed/tp_grad.py` | `build_tp_grad_info`：TP 梯度同步信息（§6.7.1） |
| `hyper_models/components/distributed/testing/grad_equiv.py` | 双模式梯度等价工具（§5.5） |

### 涉及删除的旧代码

| 旧代码 | 替代方案 |
|--------|---------|
| `hyper_parallel/models/*/parallelize.py`（每模型 ~400 行） | `ARCH_OVERRIDES`（~20 行）+ ShardingPlanner 自动推导 |
| `hyper_parallel/core/tensor_parallel/style.py` — `ParallelStyle` 子类 | `ShardingTemplate` + `ModuleShardingSpec` 声明式 |

### 1.1 独立使用：不依赖训练流程

`hyper_models/components/distributed/` 下的所有模块**零依赖**于 `recipes/` 和 `_transformers/`。任何 HF 模型都可以直接使用 ShardingPlanner + apply_sharding_plan（以下示例只使用当前已实现并导出的符号，见 `hyper_models/components/distributed/__init__.py`）：

```python
from transformers import AutoModelForCausalLM
from hyper_parallel.core.dtensor.device_mesh import init_device_mesh
from hyper_models.components.distributed import (
    ShardingPlanner, apply_sharding_plan,
)

# 1. 加载任意 HF 模型
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-4B")

# 2. 构建 mesh（TP=4；mesh 构建的训练流程集成属 06 文档范围）
mesh = init_device_mesh("npu", (4,), mesh_dim_names=("tp",))

# 3. 自动推导分片策略（零模型代码改动）
planner = ShardingPlanner()
plan = planner.plan(model, mesh, tp_size=4)

# 4. 应用分片（生产模式：零 DTensor dispatch；validate_mode=True 走校验模式）
model, tp_grad_info = apply_sharding_plan(model, plan, mesh)

# 5. 用任意框架训练（PyTorch Lightning、HF Trainer、手写循环...）
```

---

## 2. 总入口调用时序：从 `ShardingPlanner` 到运行时 DTensor

双模式 DTensor 在 `_build_model()` 内分两步执行——**编译期规划**（sharding_planner.plan）和**运行时应用**（apply_sharding_plan）。以下是从 `main()` 到 DTensor 分片完成的完整调用链路：

```
main() → recipe.setup(cfg)                                              # 01_hf_compatibility_layer.md §4
└─③.4 model = HyperAutoModelForCausalLM.from_pretrained(cfg.model, distributed_setup=...)  # 01 §6（编号以 01 §4.1 为准）
    └─ _build_model(...)                                             # 01 §6.3
            │
            ├─③.4.2 instantiate_infrastructure(distributed_setup, device)  # 01 §8
            │   └─ sharding_planner = ShardingPlanner()                    # hyper_parallel 核心
            │
            ├─③.4.5.2 _init_model() → meta device 空壳模型                # 01 §7
            │
            ├─③.4.5.7 plan = sharding_planner.plan(model, mesh, ...)       # ★ 编译期规划 → §3
            │   │                                                           # └─ 详见 §3
            │   ├─ Phase 1: ParameterClassifier.classify(model)             # §3.2: ParamRole 分类
            │   │   ├─ 命名规则匹配（按参数名后缀识别角色）
            │   │   └─ 架构规则覆盖（ARCH_OVERRIDES）
            │   │
            │   ├─ Phase 2: BoundaryGrouper.group(model)                   # §3.3: 模块边界分组
            │   │   └─ 每个 transformer layer → attention + mlp + norm 边界
            │   │
            │   ├─ Phase 3: SemanticRoleInference.infer(boundary, roles)   # §3.4: 语义角色推断
            │   │   └─ 从参数角色推断模块类型（attention / mlp / norm / embed / lm_head / moe_mlp / moe_gate）
            │   │
            │   ├─ Phase 4: TemplateLookup.lookup(boundary_type)           # §3.5: 查 ShardingTemplate
            │   │   ├─ attention → {SP, non-SP} × {CP, non-CP} 模板
            │   │   ├─ mlp       → {SP, non-SP} × {CP, non-CP} 模板
            │   │   ├─ norm      → Replicate 模板（TP/CP 双维度）
            │   │   ├─ embed/lm_head → Shard(0)/Shard(1) 模板（TP/CP 双维度）
            │   │   ├─ moe_mlp   → EP Shard(0) + TP Colwise/Rowwise + local_map 模板
            │   │   └─ moe_gate  → EP redistribute + TP Replicate 模板
            │   │   └─ _build_spec_from_template(template, param_roles)    # §3.5.6: 填充 ModuleShardingSpec
            │   │       ├─ COLWISE role  → {TP: template.colwise_placement, CP: Replicate()}
            │   │       ├─ ROWWISE role  → {TP: template.rowwise_placement, CP: Replicate()}
            │   │       ├─ NORM role     → {TP: template.norm_placement, CP: Replicate()}
            │   │       ├─ MOE_EXPERT role → {EP: Shard(0)}（D-10 TP-extend-EP：HF 原生
            │   │       │   MoE 的 expert 仅 expert 维切分，无 TP 键，见 §6.4.8；
            │   │       │   pre-stacked EP-aware 模块沿用 D-08 ndim 感知的 TP 规则）
            │   │       └─ I/O 契约: Template.sp_in_src/dst/out_src/dst → spec（含 TP+CP+EP）
            │   │
            │   ├─ Phase 5: ChainPropagator.propagate(specs)               # §3.6: 链式传播
            │   │   ├─ Scenario 1: 填充下游模块缺失的 in_src (A.out_dst → B.in_src)
            │   │   ├─ Scenario 2: 检测模板错误 (A.out_dst ≠ B.in_src)
            │   │   ├─ Scenario 3: 处理首/尾模块 (dataloader → embed, lm_head → loss)
            │   │   └─ Scenario 4: 自定义模块插入（reshape 边界限制）
            │   │
            │   └─ Phase 6: SpecialHandler.apply(special_params)           # §3.7: 特殊参数处理
            │       └─ 例如: gated_delta_tp_shard, fused_qkv 合并等
            │
            └─③.4.5.8 apply_sharding_plan(model, plan, mesh, validate_mode)  # ★ 运行时应用 → §4
                │                                                              # └─ 详见 §4
                ├─ Phase A: _shard_params(model, plan)                        # §4.2: 参数 → DTensor
                │   ├─ for each spec in plan.modules:
                │   │   ├─ [D-09b] spec._ep_stack 非空 → _stack_moe_experts 前置堆叠
                │   │   │       （per-expert 2D → stacked 3D [E, ...]，§6.4.7）
                │   │   ├─ [D-10] spec._ep_size > 0 → 按名分流：experts.* 在派生
                │   │   │       expert mesh (edp, ep) 上 {EP: Shard(0)} 分片，
                │   │   │       其余 dense 参数走主 mesh（§6.4.8）
                │   │   ├─ for param_name, placements in spec.params:
                │   │   │       distribute_tensor(param, mesh, placements)
                │   │   │       # TP: Shard(0)/Shard(1), CP: Replicate(), EP: Shard(0)
                │   │   └─ EP 参数分片由 spec.params 的 {EP: Shard(0)} placement
                │   │       统一管（_shard_module_params 内置，不再调 ExpertParallel._apply）
                │   │
                │   └─ PEFT 参数特殊处理（LoRA 权重不参与 DTensor 分片）
                │   注：EP token dispatcher（DeepEP/UCCL-EP）的初始化不在本层。
                │       规划未实现——D-09/D-10 后 HF 原生 MoE 由 wrapper 注入
                │       _hf_native_ep_compute（§6.4.7/§6.4.8），无需此注入；
                │       仅自研 EP-aware 模块需要时由模型侧自行初始化（见 §6.4.3）
                │
                ├─ Phase B: PrecompiledBoundary.build(spec, mesh)             # §4.3: 编译期通信计划
                │   │                                                            # └─ 详见 §4.3
                │   ├─ 输入: ModuleShardingSpec(in_src, in_dst, out_src, out_dst)
                │   │        每个 placement 包含 {TP: ..., CP: ..., EP: ...}
                │   ├─ 分析: 按 mesh 维度逐维度比较 src vs dst placement
                │   │   ├─ TP 维度: Shard(1)→Replicate → all_gather
                │   │   ├─ CP 维度: Shard(1)→Replicate → all_gather
                │   │   ├─ EP 维度: Replicate→Shard(0) → redistribute
                │   │   └─ identity 维度: 跳过（零开销）
                │   └─ 输出: PrecompiledBoundary(in_plan=[...], out_plan=[...])
                │       └─ 所有非 identity 操作统一用 DTensor.redistribute()
                │
                └─ Phase C: _wrap_forward(model, boundaries, validate_mode)   # §4.4: forward 包装
                    │
                    ├─ if validate_mode:                                       # §5.3: 校验模式 forward
                    │   ├─ 输入: DTensor（完整放置信息）
                    │   ├─ forward 内部: DTensor 传播 → 记录实际 out_src
                    │   └─ 校验: assert actual_out_src == spec.out_src
                    │            assert actual_out_dst == spec.out_dst（仅终端模块）
                    │
                    ├─ elif spec.use_local_map (MoE EP):                      # §4.4.3: EP local_map 模式
                    │   ├─ boundary.redistribute_inputs(args, kwargs)           # PrecompiledBoundary 入口
                    │   ├─ _local_params_context(module)                          # build期一次性 unpack: DTensor→local（永久替换，在fully_shard前调用）
                    │   │   └─ original_forward(*args, **kwargs)  (params already local)
                    │   │       └─ all-to-all dispatch → expert compute → all-to-all combine
                    │   │          (纯 local tensor, 零 DTensor overhead)
                    │   ├─ output = DTensor.from_local(output, mesh, out_src)   # local→DTensor
                    │   └─ boundary.redistribute_outputs(output)                # PrecompiledBoundary 出口
                    │
                    └─ else (生产模式):                                         # §4.4.1: 标准生产模式
                        ├─ boundary.redistribute_inputs(args, kwargs)           # PrecompiledBoundary 入口
                        │   └─ 同时处理 TP+CP+EP 多维度 redistribution
                        ├─ _local_params_context(module)                        # build期一次性 unpack: DTensor→local（永久替换，在 fully_shard 之前调用，forward 内直接使用 local params）
                        │   └─ original_forward(*args, **kwargs)  (params already local)
                        │       ├─ [if CP enabled] CP inner attention 通信      # §4.4.2
                        │       │   └─ K/V all-gather 在 SDPA/FlexAttention 内部
                        │       └─ 纯 local tensor 计算（零 DTensor dispatch）
                        └─ boundary.redistribute_outputs(output)                # PrecompiledBoundary 出口

CP inner attention 的 forward 替换在 Phase C 的__init__阶段完成（非每次 forward 调用），与 PrecompiledBoundary 一样是编译期确定的:
  └─ sharding_applier._wrap_cp_inner_attention(attn_module, cp_mesh, spec=spec, mesh=mesh,
                                               mesh_dim_names=mesh_dim_names)    # §4.4.2
      ├─ 调用时机: Phase C 执行时，检测到 cp_size > 1 且模块为 attention
      │   （D-01''：production 与 validate 注入**同一个** wrapper——SDPA dispatch
      │    不会对 CP Shard(1) 的 K/V 做 all-gather，见 §4.4.2 实现说明）
      ├─ SDPA 路径: _wrap_sdpa_for_cp() → 替换 inner_attention.forward
      │   └─ 新 forward 中: flex_cp_allgather(K, V) → 本地 Q chunk SDPA → 分片输出
      │      （validate：入口 unwrap DTensor + 临时解包参数，出口按 out_src 声明重包装）
      └─ FlexAttention 路径: _wrap_flex_attn_for_cp() → 替换 inner_attention.forward
          └─ 新 forward 中: flex_cp_allgather(K, V) → attention → 返回分片输出

EP MoE 的 forward 包装在 Phase C 执行时，检测到 spec.use_local_map:
  └─ sharding_applier._wrap_local_region_forward(module, boundary, spec, mesh, mesh_dim_names,
                                                 validate_mode=..., compute_fn=...)  # §4.4.3
      └─ 包装后的 forward: boundary入口 → local_map(ctx) → DTensor.from_local → boundary出口
```

**与 01、03 文档的时序衔接**：

```
main()                                           # 01 §4
└─③.4 model = from_pretrained()
    └─ _build_model()
        ├─③.4.5.2 _init_model()                  # 01 §7: meta device 空壳
        ├─③.4.5.7 sharding_planner.plan()         # 本文档 §3: 编译期规划 ★
        └─③.4.5.8 apply_sharding_plan()           # 本文档 §4: 运行时应用 ★
            ├─ Phase A: _shard_params              # §4.2
            ├─ Phase B: PrecompiledBoundary.build  # §4.3
            └─ Phase C: _wrap_forward              # §4.4
                ├─ _wrap_production_forward        # §4.4.1 (标准 TP)
                ├─ _wrap_cp_inner_attention        # §4.4.2 (CP attention)
                ├─ _wrap_local_region_forward               # §4.4.3 (EP local_map)
                └─ _wrap_validate_forward          # §5.3 (校验模式)

── 运行时使用（训练循环中）──

④ run_train_validation_loop()                    # 03_training_loop.md §6（编号以 01 §4.1 为准）
└─④.1.2 _forward_backward_step()                 # 03_training_loop.md §8
    └─ model(**batch)
        ├─ PrecompiledBoundary.redistribute_inputs(x)   # 本文档 §4.3: TP+CP+EP 多维度通信
        ├─ _local_params_context:  # build-time one-shot (params permanently unpacked, called before fully_shard)
        │   ├─ [if CP] CP K/V all-gather (inner attention 内部)  # §4.4.2
        │   ├─ [if EP] all-to-all dispatch/combine               # §4.4.3
        │   └─ module.forward(x_local)                            # 纯 local tensor 计算
        └─ PrecompiledBoundary.redistribute_outputs(y)   # 本文档 §4.3: TP+CP+EP 多维度通信
```
## 3. 并行配置的数据结构

> **调用位置**: 时序树 sharding_planner.plan — ShardingPlanner 的输出格式 + ShardingApplier 的输入

### 3.1 ShardingPlan：模型级分片计划

```python
# hyper_models/components/distributed/sharding_config.py

@dataclass
class ShardingPlan:
    """一个模型的完整分片计划。"""
    # {module_fqn: ModuleShardingSpec} — 只包含 is_boundary=True 的模块
    modules: dict[str, "ModuleShardingSpec"] = field(default_factory=dict)
    # 全局开关
    sequence_parallel: bool = True
    loss_parallel: bool = False

    # 特殊参数处理器: {module_fqn.param_name: handler_name}
    special_handlers: dict[str, str] = field(default_factory=dict)

    # mesh 维度名（与 DeviceMesh.mesh_dim_names 一致）
    mesh_dim_names: tuple[str, ...] = ()

    # tied-weight 对：[(fqn_a, fqn_b)]，共享存储的参数（如 embed_tokens.weight <-> lm_head.weight）。
    # 由 ShardingPlanner 从模型 weight tying 检测填入，供 build_tp_grad_info 归一化 tp_placement。
    tied_pairs: list[tuple[str, str]] = field(default_factory=list)
```

### 3.2 ModuleShardingSpec：单模块分片规格

```python
@dataclass
class ModuleShardingSpec:
    """单个模块的完整 DTensor 契约。

    四个 Placement 字段构成完整的 I/O 契约——运行时不做推断，直接按声明执行：

      in_src:  输入到达模块边界时的 placement（从上游模块的输出或 dataloader 来）
      in_dst:  模块内部计算需要的 placement（如果不等于 in_src，触发通信）
      out_src: 模块内部计算自然产生的 placement（由 DTensor 策略传播决定，校验模式使用）
      out_dst: 下游模块期望的 placement（如果不等于 out_src，触发通信）

    每个 placement 都是 NamedPlacement = dict[MeshAxisName, Placement]，
    声明在所有活跃的 mesh 维度（TP, CP, EP）上的 placement。

    例 — Llama self_attn, TP=4, CP=2, SP=true:
        ModuleShardingSpec(
            params={
                "q_proj.weight": {TP: Shard(0), CP: Replicate()},
                "k_proj.weight": {TP: Shard(0), CP: Replicate()},
                "v_proj.weight": {TP: Shard(0), CP: Replicate()},
                "o_proj.weight": {TP: Shard(1), CP: Replicate()},
            },
            in_src={"hidden_states": {TP: Shard(1), CP: Shard(1)}},       # 从 SP+CP norm 来
            in_dst={"hidden_states": {TP: Replicate(), CP: Shard(1)}},    # 只 all-gather TP；CP 维 K/V all-gather 在 inner attention
            out_src={TP: Partial(), CP: Shard(1)},                        # 本地 Q+全局 K/V → 输出仅覆盖本地 Q 段 → CP Shard(1)
            out_dst={TP: Shard(1), CP: Shard(1)},                          # reduce-scatter(TP) → SP+CP；CP 维 identity
        )
    """
    # ── 参数分片：子模块路径 → {MeshAxis: Placement} ──
    params: dict[str, NamedPlacement] = field(default_factory=dict)

    # ── 输入契约（必填字段） ──
    in_src: dict[str, NamedPlacement] = field(default_factory=dict)
    in_dst: dict[str, NamedPlacement] = field(default_factory=dict)

    # ── 输出契约 ──
    # out_src/out_dst: dict[str, NamedPlacement]（与 in_src/in_dst 对称），支持返回 tuple 的多输出模块。
    # 单输出模块使用 {"output": NamedPlacement}（或简写为单 key dict）。
    # out_src=None: 不做 src 校验（仅对比 out_dst），或者模块输出不是 DTensor
    # out_dst=None: 输出不需要 redistribution（identity 路径）
    out_src: dict[str, NamedPlacement] | None = None
    out_dst: dict[str, NamedPlacement] | None = None
    # out_names: 多输出模块（返回 tuple）的输出名顺序，用于把 out_src/out_dst 的
    # key 映射到 tuple 位置（RedistOp.arg_index）。缺省时按 out_src 的 key 顺序。
    # 例：attention 返回 (hidden_states, present_kv) → out_names=["hidden_states", "present_kv"]。
    out_names: list[str] | None = None
    # 注：§3.4 / §6 中的示例为简洁起见用 `out_src={TP: Partial(), ...}` 标量写法
    # 表示单输出模块，等价于 `out_src={"output": {TP: Partial(), ...}}`。
    # 标量简写会在归一化阶段（_normalize_out_fields，见 §3.5 _build_spec_from_template）
    # 包装为 {"output": ...}，与 _compile_output_plan 的 dict 契约对齐。

    # ── 边界标记 ──
    is_boundary: bool = True

    # ── 结构标记（用户可配置） ──
    use_local_map: bool = False   # 数据相关模块（MoE 等）: 走 local-region wrapper（用户可配置，见 §3.6.7）
    # CP 自定义入口（用户可配置，2026-07-21，见 §4.4.2）：
    inner_target: Optional[str] = None    # 纯位置：显式指定 inner attention 属性名（"self"=模块本身）
    inner_wrapper: Optional[Union[str, Callable]] = None  # 纯行为：注册表名（"sdpa_qkv"等）显式固定内置 CP wrapper，或自定义 callable fn(target, cp_mesh)
    # local-region 自定义计算（用户可配置，2026-07-21，见 §4.4.3）：
    local_compute_fn: Optional[Callable] = None  # 自定义 compute_fn fn(module, *args, **kwargs)，声明即生效（门控由解析链派生，无需设 use_local_map）

    # ── 内部标记（由 ShardingPlanner 自动设置） ──
    _is_terminal: bool = False  # 链式传播时自动标记
    _needs_cp_attn: bool = False  # attention 模块: inner attention 需要 CP-aware forward 替换
    # HF 原生 MoE 的 EP 直通（D-09/D-10，见 §6.4.7/§6.4.8）：
    _ep_stack: dict[str, list[str]] = {}  # stacked 相对路径 → 源参数相对路径（按 expert idx 排序）
    _moe_router: str = "default"          # MOE_ROUTER_ADAPTERS 的 adapter 名
    _ep_size: int = 0                     # D-10：扩展 EP 组大小（>0 时 MoE 走 SP-in identity + 派生 expert mesh）
```

### 3.2.1 NamedPlacement 的物理含义

`NamedPlacement = dict[MeshAxisName, Placement]`。理解其物理含义的关键规则：

> **Key 是 mesh 维度名，Value 中的 `Shard(N)` 的 N 是 tensor 维度索引。**

```
{TP: Shard(0)}
  ↑                  ↑
  mesh 维度名       沿 tensor 第 0 轴切分

解读: 在 TP 这组 ranks 上，tensor 沿 dim 0 切分。TP=4 → 每个 rank 持有 1/4

{TP: Shard(1)}
  TP 这组 ranks 上，沿 tensor dim 1 切分（对 activation [B,S,H] 就是沿序列切 → SP）

{TP: Replicate()}
  TP 这组 ranks 上，每个 rank 持有完整副本

{TP: Shard(0), EP: Replicate()}
  TP 维度: 沿 tensor dim 0 切分
  EP 维度: 全复制（每个 EP rank 持有完整副本）

{TP: Shard(1), CP: Shard(1), EP: Replicate()}
  TP 维度: 沿序列切（SP）
  CP 维度: 沿序列切（CP）
  EP 维度: 全复制
```

**对于权重** `[H_out, H_in]`：
- `{TP: Shard(0), CP: Replicate()}` → rank i 持有 `[H_out/tp, H_in]`（Colwise 分片）
- `{TP: Shard(1), CP: Replicate()}` → rank i 持有 `[H_out, H_in/tp]`（Rowwise 分片）

**对于激活** `[B, S, H]`：
- `{TP: Shard(1)}` → 沿 S 切分（Sequence Parallel）
- `{TP: Shard(-1)}` → 沿 H 切分（Column-wise 输出）
- `{CP: Shard(1)}` → CP 也沿 S 切分（不同的 rank 组，语义不同）

**对于 MoE expert 权重** `[num_experts, H_out, H_in]`：
- `{EP: Shard(0), TP: Shard(0), CP: Replicate()}` → `[n_experts/ep, H_out/tp, H_in]`

### 3.3 `is_boundary` 的作用与设置者

| | `is_boundary=True` | `is_boundary=False` |
|---|---|---|
| **行为** | 包装 forward + 构建 PrecompiledBoundary | 只做参数分片，不独立包装 forward |
| **谁创建** | ShardingPlanner 自动生成（全部为 True）；用户手动注入时默认 True | 仅用户手动注入时显式指定 |
| **何时使用** | 所有通信边界模块 | 已合并到父边界的子模块，但用户想单独声明其 params（罕见） |

ShardingPlanner 从不创建 `is_boundary=False` 的 spec。核心规则：

```python
for module_fqn, spec in plan.modules.items():
    if not spec.is_boundary:   # ← 跳过 forward 包装
        continue
    module = _resolve_module(model, module_fqn)
    boundary = PrecompiledBoundary(spec, mesh, mesh_dim_names)

    # 门控派生（非存储 bool，§4.4.2/§4.4.3 解析链）：
    if cp_mesh is not None and cp_mesh.size() > 1:      # CP inner attention
        _wrap_cp_inner_attention(module, cp_mesh, spec=spec, mesh=mesh,
                                 mesh_dim_names=mesh_dim_names, direct=False)
    compute_fn = _resolve_local_compute_fn(
        module, spec, mesh, mesh_dim_names, expert_mesh)
    if compute_fn is not None:              # local region（MoE EP / 用户 compute_fn）
        _wrap_local_region_forward(module, boundary, spec, mesh, mesh_dim_names,
                                   validate_mode=..., compute_fn=compute_fn)
    else:                                    # 标准 TP/CP
        _wrap_production_forward(module, boundary)
```

### 3.4 标准 Transformer 的 ShardingPlan 生成结果

以 Llama decoder layer 0 为例，TP=4，CP=2，SP=true：

```python
plan.modules = {
    # ── 边界 1: embed_tokens ──
    # 修订 D-05（见 §3.5 Step 2.6）：CP>1 时 embed 的 in_src/in_dst/out_src CP 维
    # 为 Shard(1) 而非模板字面量的 Replicate——batch 已被 CP 数据管道
    #（shard_batch_for_cp，§6.3.4）按 CP 切好，声明 Replicate 会让 boundary
    # 把已切分的 chunk 再 scatter 一次（序列被切两次）。
    "model.embed_tokens": ModuleShardingSpec(
        params={"weight": {TP: Shard(0), CP: Replicate()}},
        in_src={"input": {TP: Replicate(), CP: Shard(1)}},
        in_dst={"input": {TP: Replicate(), CP: Shard(1)}},     # identity
        out_src={TP: Partial(), CP: Shard(1)},
        out_dst={TP: Shard(1), CP: Shard(1)},                  # reduce-scatter → SP+CP
    ),

    # ── 边界 2: input_layernorm ──
    "model.layers.0.input_layernorm": ModuleShardingSpec(
        params={"weight": {TP: Replicate(), CP: Replicate()}},
        in_src={"hidden_states": {TP: Shard(1), CP: Shard(1)}},
        in_dst={"hidden_states": {TP: Shard(1), CP: Shard(1)}},   # identity
        out_src={TP: Shard(1), CP: Shard(1)},
        out_dst={TP: Shard(1), CP: Shard(1)},                      # identity
    ),

    # ── 边界 3: self_attn ──
    "model.layers.0.self_attn": ModuleShardingSpec(
        params={
            "q_proj.weight": {TP: Shard(0), CP: Replicate()},
            "k_proj.weight": {TP: Shard(0), CP: Replicate()},
            "v_proj.weight": {TP: Shard(0), CP: Replicate()},
            "o_proj.weight": {TP: Shard(1), CP: Replicate()},
        },
        in_src={"hidden_states": {TP: Shard(1), CP: Shard(1)}},
        # CP 维保持 Shard(1)：只 all-gather TP，CP 维的 K/V all-gather
        # 交给 inner attention wrapper 在 SDPA/FlexAttention 内部完成（§4.4.2）。
        in_dst={"hidden_states": {TP: Replicate(), CP: Shard(1)}},
        out_src={TP: Partial(), CP: Shard(1)},   # 本地 Q 段输出 → CP Shard(1)
        out_dst={TP: Shard(1), CP: Shard(1)},
    ),
}
```

**关键观察**：
- 所有 placement 都**完整声明**了 TP+CP+EP（此处 EP 未启用故省略）
- `in_src ≠ in_dst` → PrecompiledBoundary 生成通信 op（**仅 all-gather on TP**；CP 维保持 Shard(1)，K/V all-gather 由 inner attention wrapper 在 forward 内部完成，见 §4.4.2/§6.3.3）
- `out_src ≠ out_dst` → PrecompiledBoundary 生成通信 op（reduce-scatter on TP）；CP 维 out_src=Shard(1) 与 out_dst=Shard(1) identity，boundary 不做 CP 出口通信
- TP 与 CP 的边界通信职责**不对称**：TP 在 boundary 做 all-gather/reduce-scatter；CP 的序列维 all-gather 仅发生在 attention 内部（K/V，§4.4.2），boundary 层 CP 维全程 identity（out_src=out_dst=Shard(1)）



---

### 3.5 ShardingTemplate: CP/EP dimensions in I/O Template

#### ShardingTemplate Data Structure

```python
@dataclass
class ShardingTemplate:
    """Semantic role -> placement template.

    Each field declares placements on ALL active mesh dimensions (TP+CP+EP).
    Aligned with Titan SpmdLayout: dict[MeshAxisName, Placement].

    -- Parameter sharding rules --
    colwise_placement:   COLWISE role params on TP axis.
                        e.g. Shard(0) -> weight [H_out, H_in] -> [H_out/tp, H_in].
    rowwise_placement:   ROWWISE role params on TP axis.
                        e.g. Shard(1) -> weight [H_out, H_in] -> [H_out, H_in/tp].
    norm_placement:      NORM role params on TP+CP axes.
                        e.g. Replicate() -> full [H] on each TP/CP rank.
    moe_expert_placement: MOE_EXPERT role params on EP axis.
                        e.g. Shard(0) -> expert params shard along expert dim.

    -- I/O contract (SP/non-SP, each containing TP+CP+EP three dimensions) --
    sp_in_src:    SP mode, input placement at module boundary.
                 e.g. {"hidden_states": {TP: Shard(1), CP: Shard(1), EP: Replicate()}}
    sp_in_dst:    SP mode, desired input placement for compute.
                 e.g. {"hidden_states": {TP: Replicate(), CP: Replicate(), EP: Replicate()}}
    sp_out_src:   SP mode, natural output placement (DTensor dispatch decides).
                 e.g. {TP: Partial(), CP: Replicate(), EP: Replicate()}
    sp_out_dst:   SP mode, downstream expected placement.
                 e.g. {TP: Shard(1), CP: Shard(1), EP: Replicate()}

    nosp_in_src / nosp_in_dst / nosp_out_src / nosp_out_dst:
                 non-SP mode corresponding placements. Used when SP is off.

    -- Special flags --
    use_local_map:   MoE module: forward needs DTensor->local->DTensor (EP dispatch/combine).
    needs_cp_attn:   CP module: inner attention needs CP-aware forward replacement.
    """

    # Parameter sharding rules
    colwise_placement: Placement = Shard(0)     # TP axis
    rowwise_placement: Placement = Shard(1)     # TP axis
    norm_placement: Placement = Replicate()      # TP+CP axes
    moe_expert_placement: Placement = Shard(0)   # EP axis

    # CP axis: ALL parameters are Replicate() (CP only shards activations)

    # SP mode I/O (complete TP+CP+EP three dimensions)
    sp_in_src: NamedPlacement = field(default_factory=dict)
    sp_in_dst: NamedPlacement = field(default_factory=dict)
    sp_out_src: NamedPlacement | None = None
    sp_out_dst: NamedPlacement | None = None

    # non-SP mode I/O
    nosp_in_src: NamedPlacement = field(default_factory=dict)
    nosp_in_dst: NamedPlacement = field(default_factory=dict)
    nosp_out_src: NamedPlacement | None = None
    nosp_out_dst: NamedPlacement | None = None

    # Special flags
    use_local_map: bool = False     # MoE EP: forward needs local_map
    needs_cp_attn: bool = False     # CP: inner attention needs CP-aware forward
```

**CP/EP dimension rules**:

Each template placement field MUST include declarations for all active mesh dimensions:
1. **TP axis**: determined by template type (Colwise/Rowwise/Replicate/Partial)
2. **CP axis**: params always `Replicate()` (CP never shards params), activations `Shard(1)` or `Replicate()`
3. **EP axis**: non-MoE modules `Replicate()`; MoE experts `Shard(0)`

ShardingPlanner filters unused dimensions based on actual `mesh_dim_names`.

#### Complete Template Enumeration

```python
TEMPLATES: dict[str, ShardingTemplate] = {
    # -- Attention (self_attn: q/k/v Colwise + o Rowwise) --
    "attention": ShardingTemplate(
        colwise_placement=Shard(0),          # q/k/v: [H/tp, H]
        rowwise_placement=Shard(1),          # o: [H, H/tp]
        # SP: all-gather(TP only) -> compute -> reduce-scatter(TP) + CP reshard
        # CP 维 in_dst 保持 Shard(1)：K/V all-gather 由 inner attention wrapper
        # 在 SDPA/FlexAttention 内部完成（needs_cp_attn=True），不在 boundary 层。
        sp_in_src={"hidden_states": {TP: Shard(1), CP: Shard(1), EP: Replicate()}},
        sp_in_dst={"hidden_states": {TP: Replicate(), CP: Shard(1), EP: Replicate()}},
        sp_out_src={TP: Partial(), CP: Shard(1), EP: Replicate()},   # 本地 Q 段输出 → CP Shard(1)
        sp_out_dst={TP: Shard(1), CP: Shard(1), EP: Replicate()},
        # non-SP: compute -> all-reduce(TP)
        nosp_in_src={"hidden_states": {TP: Replicate(), CP: Replicate(), EP: Replicate()}},
        nosp_in_dst={"hidden_states": {TP: Replicate(), CP: Replicate(), EP: Replicate()}},
        nosp_out_src={TP: Partial(), CP: Replicate(), EP: Replicate()},
        nosp_out_dst={TP: Replicate(), CP: Replicate(), EP: Replicate()},
        needs_cp_attn=True,                  # CP: inject CP-aware inner attention
    ),

    # -- MLP (gate/up Colwise + down Rowwise) --
    # 修订 D-06：MLP 的 CP 维全程 Shard(1)（pointwise，CP 无需 boundary 通信）。
    # 若 in_dst CP=Replicate，TP×CP 下全序列 reduce-scatter 会产生与
    # embed/attention（cp-major）不一致的 tp-major 序列布局（见 §12）。
    "mlp": ShardingTemplate(
        colwise_placement=Shard(0),
        rowwise_placement=Shard(1),
        sp_in_src={"hidden_states": {TP: Shard(1), CP: Shard(1), EP: Replicate()}},
        sp_in_dst={"hidden_states": {TP: Replicate(), CP: Shard(1), EP: Replicate()}},
        sp_out_src={TP: Partial(), CP: Shard(1), EP: Replicate()},
        sp_out_dst={TP: Shard(1), CP: Shard(1), EP: Replicate()},
        nosp_in_src={"hidden_states": {TP: Replicate(), CP: Replicate(), EP: Replicate()}},
        nosp_in_dst={"hidden_states": {TP: Replicate(), CP: Replicate(), EP: Replicate()}},
        nosp_out_src={TP: Partial(), CP: Replicate(), EP: Replicate()},
        nosp_out_dst={TP: Replicate(), CP: Replicate(), EP: Replicate()},
    ),

    # -- Norm (RMSNorm/LayerNorm: weight fully replicated, zero communication) --
    "norm": ShardingTemplate(
        norm_placement=Replicate(),
        sp_in_src={"hidden_states": {TP: Shard(1), CP: Shard(1), EP: Replicate()}},
        sp_in_dst={"hidden_states": {TP: Shard(1), CP: Shard(1), EP: Replicate()}},  # identity
        sp_out_src={TP: Shard(1), CP: Shard(1), EP: Replicate()},
        sp_out_dst={TP: Shard(1), CP: Shard(1), EP: Replicate()},                    # identity
        nosp_in_src={"hidden_states": {TP: Replicate(), CP: Replicate(), EP: Replicate()}},
        nosp_in_dst={"hidden_states": {TP: Replicate(), CP: Replicate(), EP: Replicate()}},
        nosp_out_src={TP: Replicate(), CP: Replicate(), EP: Replicate()},
        nosp_out_dst={TP: Replicate(), CP: Replicate(), EP: Replicate()},
    ),

    # -- Embedding (Rowwise: weight Shard(0), output Partial -> SP+CP) --
    # 修订 D-05：CP>1 时 embed 的 in/out CP 维为 Shard(1)（而非 Replicate）——
    # CP 数据管道（shard_batch_for_cp，§6.3.4）已把 input_ids 按 CP 切好，
    # 若按 Replicate 声明，boundary 会把已切分的 chunk 再 scatter 一次
    # （序列被切两次）。该调整在 _build_spec_from_template 中按 has_cp 应用，
    # 模板字面量保留 CP: Replicate 作为无 CP 轴时的默认（会被过滤）。
    "embed": ShardingTemplate(
        colwise_placement=Shard(0),          # weight: [V/tp, H]
        sp_in_src={"input": {TP: Replicate(), CP: Replicate(), EP: Replicate()}},
        sp_in_dst={"input": {TP: Replicate(), CP: Replicate(), EP: Replicate()}},
        sp_out_src={TP: Partial(), CP: Replicate(), EP: Replicate()},
        sp_out_dst={TP: Shard(1), CP: Shard(1), EP: Replicate()},
        nosp_in_src={"input": {TP: Replicate(), CP: Replicate(), EP: Replicate()}},
        nosp_in_dst={"input": {TP: Replicate(), CP: Replicate(), EP: Replicate()}},
        nosp_out_src={TP: Partial(), CP: Replicate(), EP: Replicate()},
        nosp_out_dst={TP: Replicate(), CP: Replicate(), EP: Replicate()},
    ),

    # -- LM Head (Colwise: weight Shard(0), output Shard(-1)) --
    # 修订 D-07：lm_head 的 CP 维全程 Shard(1)（R8 统一——boundary 层 CP 维恒
    # identity，CP 序列 all-gather 仅发生在 attention 内部 K/V）。CP 下
    # lm_head 在本地 CP chunk 上计算 logits/loss（Megatron CP 标准做法），
    # 不做 CP gather。
    "lm_head": ShardingTemplate(
        colwise_placement=Shard(0),          # weight: [V/tp, H]
        sp_in_src={"hidden_states": {TP: Shard(1), CP: Shard(1), EP: Replicate()}},
        sp_in_dst={"hidden_states": {TP: Replicate(), CP: Shard(1), EP: Replicate()}},
        sp_out_src={TP: Shard(-1), CP: Shard(1), EP: Replicate()},
        sp_out_dst={TP: Shard(-1), CP: Shard(1), EP: Replicate()},   # loss_parallel=true default; overridden in _build_spec_from_template when loss_parallel=false
        nosp_in_src={"hidden_states": {TP: Replicate(), CP: Replicate(), EP: Replicate()}},
        nosp_in_dst={"hidden_states": {TP: Replicate(), CP: Replicate(), EP: Replicate()}},
        nosp_out_src={TP: Shard(-1), CP: Replicate(), EP: Replicate()},
        nosp_out_dst={TP: Replicate(), CP: Replicate(), EP: Replicate()},
    ),

    # -- MoE Gate (Router: weight replicated, input all-gather TP, output redistribute -> EP) --
    "moe_gate": ShardingTemplate(
        norm_placement=Replicate(),          # router weight/bias: replicated (TP+CP+EP)
        sp_in_src={"hidden_states": {TP: Shard(1), CP: Shard(1), EP: Replicate()}},
        sp_in_dst={"hidden_states": {TP: Replicate(), CP: Replicate(), EP: Replicate()}},
        sp_out_src={TP: Replicate(), CP: Replicate(), EP: Replicate()},
        sp_out_dst={TP: Replicate(), CP: Replicate(), EP: Shard(0)},
        nosp_in_src={"hidden_states": {TP: Replicate(), CP: Replicate(), EP: Replicate()}},
        nosp_in_dst={"hidden_states": {TP: Replicate(), CP: Replicate(), EP: Replicate()}},
        nosp_out_src={TP: Replicate(), CP: Replicate(), EP: Replicate()},
        nosp_out_dst={TP: Replicate(), CP: Replicate(), EP: Shard(0)},
    ),

    # -- MoE MLP (Dense gate + Routed experts + Optional shared experts) --
    # 修订 D-06：CP 维同 mlp，全程 Shard(1)（pointwise per-token）。
    # 修订 D-08：expert 权重为 batched 3D [E, H_out, H_in] 时，TP 的
    # colwise/rowwise 作用在 +1 维（colwise=Shard(1)、rowwise=Shard(2)），
    # tensor dim 0 的 expert 维归 EP Shard(0)；placement 推断按参数 ndim 感知。
    "moe_mlp": ShardingTemplate(
        colwise_placement=Shard(0),          # expert w1/w3: Colwise on TP
        rowwise_placement=Shard(1),          # expert w2: Rowwise on TP
        norm_placement=Replicate(),          # gate/norm: replicated
        moe_expert_placement=Shard(0),       # expert params: Shard(0) on EP
        sp_in_src={"x_BLD": {TP: Shard(1), CP: Shard(1), EP: Replicate()}},
        sp_in_dst={"x_BLD": {TP: Replicate(), CP: Shard(1), EP: Replicate()}},
        sp_out_src={TP: Partial(), CP: Shard(1), EP: Replicate()},
        sp_out_dst={TP: Shard(1), CP: Shard(1), EP: Replicate()},
        nosp_in_src={"x_BLD": {TP: Replicate(), CP: Replicate(), EP: Replicate()}},
        nosp_in_dst={"x_BLD": {TP: Replicate(), CP: Replicate(), EP: Replicate()}},
        nosp_out_src={TP: Partial(), CP: Replicate(), EP: Replicate()},
        nosp_out_dst={TP: Replicate(), CP: Replicate(), EP: Replicate()},
        use_local_map=True,                  # EP: forward needs local_map
    ),
}
```

**Key change**: Old templates only had `{TP: ...}` placements. New templates have **{TP, CP, EP} three-dimensional declarations** in all I/O fields. This lets PrecompiledBoundary uniformly handle redistribution across all dimensions.

#### Template -> ModuleShardingSpec Mapping

```python
def _build_spec_from_template(self, boundary_fqn, group, template,
                              sequence_parallel, loss_parallel, mesh_dim_names,
                              param_ndims=None):
    has_tp = "tp" in mesh_dim_names
    has_ep = "ep" in mesh_dim_names
    has_cp = "cp" in mesh_dim_names
    # 注：spec.params 中 CP 维恒为 Replicate()（CP 不切参数）；CP 仅影响
    # I/O 契约（模板 sp_* 字段 + Step 2.6 的 D-05 运行时修订）。
    spec = ModuleShardingSpec()

    # Step 1: Fill spec.params by ParamRole
    for param_fqn, role in group:
        param_path = param_fqn[len(boundary_fqn) + 1:]
        ndim = (param_ndims or {}).get(param_fqn, 2)
        if role == ParamRole.COLWISE:
            spec.params[param_path] = _multi_dim(tp=template.colwise_placement if has_tp else None,
                                                  cp=Replicate(), ep=Replicate())
        elif role == ParamRole.ROWWISE:
            spec.params[param_path] = _multi_dim(tp=template.rowwise_placement if has_tp else None,
                                                  cp=Replicate(), ep=Replicate())
        elif role == ParamRole.NORM:
            spec.params[param_path] = _multi_dim(tp=template.norm_placement if has_tp else None,
                                                  cp=Replicate(), ep=Replicate())
        elif role == ParamRole.MOE_GATE:
            spec.params[param_path] = _multi_dim(tp=template.norm_placement if has_tp else None,
                                                  cp=Replicate(), ep=Replicate())
        elif role == ParamRole.MOE_EXPERT:
            # NOTE: 当 has_tp=False 时 tp_p 为 Replicate()（而非 None）。
            # 这与 colwise/rowwise 的 has_tp=False→None（_multi_dim 中过滤掉 TP 键）
            # 不一致。设计选择：MOE_EXPERT 参数的底层 tensor 结构为 [E, H_out, H_in]，
            # 即使 TP 未启用，显式声明 TP: Replicate() 也比完全省略 TP 键更清晰地
            # 表达"此参数在所有 TP rank 上完整复制"的语义，便于 future 当 TP
            # 动态加入时迁移。如果需要严格保持 has_tp=False 时的键一致性，
            # 可改为 tp_p = _moe_expert_tp_placement(...) if has_tp else None。
            # D-08：TP placement 按参数 ndim 感知——3D batched expert 权重
            # [E, H_out, H_in] 的 colwise=Shard(1)、rowwise=Shard(2)（dim 0 是
            # expert 维，归 EP Shard(0)）；2D per-expert 布局沿用标准
            # Shard(0)/Shard(1)（此时 EP 语义不成立，见 §12.2 D-08）。
            tp_p = (_moe_expert_tp_placement(param_path, ndim, template)
                    if has_tp else Replicate())
            spec.params[param_path] = _multi_dim(tp=tp_p,
                                                  cp=Replicate(),
                                                  ep=template.moe_expert_placement if has_ep else None)
        elif role == ParamRole.SHARED_EXPERT:
            # Shared experts: EP 维度全复制（不参与 expert 切分），TP 按 w1/w3(colwise)/w2(rowwise) 切
            tp_p = (_infer_colwise_vs_rowwise(param_path, template))
            spec.params[param_path] = _multi_dim(tp=tp_p if has_tp else None,
                                                  cp=Replicate(), ep=Replicate())
        elif role == ParamRole.EMBED:
            spec.params[param_path] = _multi_dim(tp=template.colwise_placement if has_tp else None,
                                                  cp=Replicate(), ep=Replicate())
        elif role == ParamRole.LM_HEAD:
            spec.params[param_path] = _multi_dim(tp=template.colwise_placement if has_tp else None,
                                                  cp=Replicate(), ep=Replicate())
        elif role == ParamRole.FUSED_QKV:
            spec.params[param_path] = _multi_dim(tp=template.colwise_placement if has_tp else None,
                                                  cp=Replicate(), ep=Replicate())
        elif role == ParamRole.FUSED_GATE_UP:
            spec.params[param_path] = _multi_dim(tp=template.colwise_placement if has_tp else None,
                                                  cp=Replicate(), ep=Replicate())
        elif role == ParamRole.BIAS:
            spec.params[param_path] = _multi_dim(tp=Replicate(), cp=Replicate(), ep=Replicate())
        elif role == ParamRole.REPLICATED:
            spec.params[param_path] = _multi_dim(tp=Replicate(), cp=Replicate(), ep=Replicate())
            # MLA 下投影等（ARCH_OVERRIDES 显式指派）：全维 Replicate
        elif role == ParamRole.SPECIAL:
            pass  # Handled by SpecialHandler in Phase 6
        elif role == ParamRole.SKIP:
            pass  # Frozen / no-shard params — skip

    # Step 2: Select I/O contract based on SP switch（深拷贝模板契约）
    # 必须 copy.deepcopy：§3.6.5 链式传播会就地填充 next_spec.in_src，
    # 直接引用共享模板对象会污染全局 TEMPLATES（后续 spec 与再次 plan() 受影响）。
    if sequence_parallel:
        spec.in_src  = copy.deepcopy(template.sp_in_src)
        spec.in_dst  = copy.deepcopy(template.sp_in_dst)
        spec.out_src = copy.deepcopy(template.sp_out_src)
        spec.out_dst = copy.deepcopy(template.sp_out_dst)
    else:
        spec.in_src  = copy.deepcopy(template.nosp_in_src)
        spec.in_dst  = copy.deepcopy(template.nosp_in_dst)
        spec.out_src = copy.deepcopy(template.nosp_out_src)
        spec.out_dst = copy.deepcopy(template.nosp_out_dst)

    # Step 2.5: lm_head output plan depends on loss_parallel (runtime decision)
    # CP 维恒 Shard(1)（D-07/R8）：CP 下在本地 chunk 上算 loss，不做 CP gather。
    if template is TEMPLATES.get("lm_head"):
        spec.out_dst = _multi_dim(
            tp=Shard(-1) if loss_parallel else Replicate(),
            cp=Shard(1), ep=Replicate(),
        )

    # Step 2.6: embed 的 CP 契约（修订 D-05）：CP 数据管道（shard_batch_for_cp，
    # §6.3.4）已把 input_ids 按 CP 切好——in/out 的 CP 维为 Shard(1) 而非模板
    # 默认的 Replicate，否则 boundary 会把已切分的 chunk 再 scatter 一次。
    if template is TEMPLATES.get("embed") and has_cp and sequence_parallel:
        spec.in_src  = {"input": _multi_dim(tp=Replicate(), cp=Shard(1), ep=Replicate())}
        spec.in_dst  = {"input": _multi_dim(tp=Replicate(), cp=Shard(1), ep=Replicate())}
        spec.out_src = _multi_dim(tp=Partial(), cp=Shard(1), ep=Replicate())

    # Step 3: Transfer special flags
    spec.use_local_map = template.use_local_map
    if template.needs_cp_attn:
        spec._needs_cp_attn = True

    # Step 4: 归一化 out_src/out_dst 标量简写为 dict 契约
    spec = _normalize_out_fields(spec)
    return spec


def _normalize_out_fields(spec):
    """标量简写 {TP: ...} 归一化为 {'output': {TP: ...}}，与 _compile_output_plan 的 dict 契约对齐。

    out_src/out_dst 声明为 dict[str, NamedPlacement] | None，但模板/示例常用
    标量 NamedPlacement 简写（单输出模块）。_compile_output_plan 按 dict 契约逐 key
    编译，遇到标量会 AttributeError。本函数在 spec 构造入口把标量包装成
    {"output": <scalar>}，统一两端契约。
    """
    for attr in ("out_src", "out_dst"):
        val = getattr(spec, attr, None)
        # 检测启发式：若 val 是一个非 None 的 dict，且其任意 value 不是 dict，
        # 则判定为标量 NamedPlacement 简写（如 {TP: Shard(1)}）。真正的 dict 契约
        # 的 value 必定是 dict[str, Placement]（如 {"hidden_states": {TP: Shard(1)}}）。
        # 此启发式无法区分"恰好有一个名为 Shard(0) 的输出模块"这种极端情况，
        # 但实际不存在这种命名约定。
        if val and not all(isinstance(v, dict) for v in val.values()):
            setattr(spec, attr, {"output": val})
    return spec


def _multi_dim(tp=None, cp=None, ep=None):
    """Build multi-dim placement dict, filtering out None dims."""
    result = {}
    if tp is not None: result[MeshAxisName.TP] = tp
    if cp is not None: result[MeshAxisName.CP] = cp
    if ep is not None: result[MeshAxisName.EP] = ep
    return result


# 模块级别名：本文档示例中大量使用裸 {TP: Shard(0), CP: Replicate(), EP: ...}
# 写法，TP/CP/EP 即 MeshAxisName 枚举值的简写别名，统一在此声明一次。
# NOTE: 这些别名必须定义在 TEMPLATES 字典之后——TEMPLATES 内部的 placement
# 字面量在模块加载时求值，此时 TP/CP/EP 尚未绑定，故 TEMPLATES 中使用的是
# 裸 Placement 枚举值（Shard(0)/Replicate()），而非 TP/CP/EP 别名。
# 若将 alias 定义移到 TEMPLATES 之前，可简化 TEMPLATES 内的写法但会增加
# 模块初始化时的依赖顺序约束。
TP = MeshAxisName.TP
CP = MeshAxisName.CP
EP = MeshAxisName.EP


def _infer_colwise_vs_rowwise(param_path: str, template: "ShardingTemplate") -> Placement:
    """根据参数名后缀推断 shared expert 参数的 TP placement。
    w1/w3/gate/up -> colwise(Shard(0)), w2/down -> rowwise(Shard(1))。
    """
    name = param_path.lower()
    if any(k in name for k in ("w2", "down_proj", "down.")):
        return template.rowwise_placement
    return template.colwise_placement


def _moe_expert_tp_placement(param_path: str, ndim: int,
                             template: "ShardingTemplate") -> Placement:
    """MOE_EXPERT 的 TP placement（修订 D-08，按参数 ndim 感知）。

    expert 权重为 batched 3D 布局 [E, H_out, H_in]（ndim>=3）时，tensor dim 0
    是 expert 维（归 EP Shard(0)），TP 的 colwise/rowwise 须作用在 +1 维：
    colwise（切 H_out）→ Shard(1)；rowwise（切 contraction 维 H_in）→ Shard(2)。
    per-expert 2D 布局（experts.N.w1 [H_out, H_in]）沿用标准 Shard(0)/Shard(1)
    ——但此时 EP Shard(0) 会切 H_out，语义不成立：EP 应按"每 rank 持有 expert
    子集"实现（module 级），需 ARCH_OVERRIDES/SpecialHandler，不在模板覆盖范围。
    """
    name = param_path.lower()
    is_rowwise = any(k in name for k in ("w2", "down_proj", "down."))
    if ndim >= 3:
        return Shard(2) if is_rowwise else Shard(1)
    return template.rowwise_placement if is_rowwise else template.colwise_placement
```

#### ParamRole -> Template Field Mapping

```
ParamRole        -> Template field           -> placement value           -> physical meaning
================================================================================================
EMBED            -> colwise_placement        -> {TP: Shard(0)}            weight [V/tp, H]
LM_HEAD          -> colwise_placement        -> {TP: Shard(0)}            weight [V/tp, H]
COLWISE          -> colwise_placement        -> {TP: Shard(0)}            weight [H_out/tp, H_in]
FUSED_QKV        -> colwise_placement        -> {TP: Shard(0)}            weight [3H/tp, H]
FUSED_GATE_UP    -> colwise_placement        -> {TP: Shard(0)}            weight [8H/tp, H]
-----------------------------------------------------------------------------------------------
ROWWISE          -> rowwise_placement        -> {TP: Shard(1)}            weight [H_out, H_in/tp]
-----------------------------------------------------------------------------------------------
NORM             -> norm_placement           -> {TP: Replicate()}         weight [H] replicated
MOE_GATE         -> norm_placement           -> {TP: Replicate()}         router weight replicated
-----------------------------------------------------------------------------------------------
MOE_EXPERT       -> moe_expert_placement     -> {EP: Shard(0)}            expert shard on EP
                   + colwise/rowwise          -> {TP: Shard(0)/(1)}        also shard on TP
                                                （D-08：ndim≥3 时 Shard(1)/Shard(2)）
-----------------------------------------------------------------------------------------------
BIAS             -> (hardcoded)              -> {TP: Replicate()}         bias always replicated
REPLICATED       -> (hardcoded)              -> {TP: Replicate()}         MLA down-proj etc.
                                                                            (ARCH_OVERRIDES only)
SPECIAL          -> (not here)               -> SpecialHandler            Phase 6 handles
SKIP             -> (skip params)            -> --                        frozen/no-shard params
```

**CP dimension rule**: CP **never shards parameters** -- all ParamRoles are `Replicate()` on CP. CP only shards activations (sequence dimension), declared in I/O template fields `sp_in_src/sp_out_dst` as `{CP: Shard(1)}`.


---

> **调用位置**: 时序树 ③.4.5.7 — `sharding_planner.plan(model, mesh, ...)` → `ShardingPlan`

### 3.6 Phase 1-2: Parameter Classification + Boundary Grouping

### 3.6.1 推导不是"从参数推导 I/O"——而是多层协作

仅从参数角色不足以推导 `in_src`/`in_dst`/`out_src`/`out_dst`。完整的推导管线分 **6 个阶段**：

```
Phase 1: 参数角色分类
  named_parameters() → 命名规则匹配 → ParamRole(COLWISE/ROWWISE/NORM/...)

Phase 2: 通信边界分组
  参数 → _find_boundary() → 最近公共父模块 → boundary_groups

Phase 3: 语义角色推断（独立于参数角色！）
  boundary_fqn → 分析 FQN 模式 → boundary_type(ATTENTION/MLP/NORM/EMBED/LM_HEAD)

Phase 4: 模板查表生成 I/O
  boundary_type + sequence_parallel + loss_parallel → in_src, in_dst, out_src, out_dst

Phase 5: 链式传播校验
  上一个边界的 out_dst → 下一个边界的 in_src，自动填充缺省，校验一致

Phase 6: 构建 ModuleShardingSpec
  合并 params + I/O 契约 → ShardingPlan.modules[boundary_fqn]
```

> **命名覆盖缺口教训（D-13，2026-07-21；UT `test_s1_mla_deepseek.py` S1.14）**：
> Phase 1 的默认命名规则覆盖不到时后果不是"少切一个参数"而是**整条边界
> 消失**——DeepSeek MLA 的投影（`q_a_proj`/`q_b_proj`/`kv_a_proj_with_mqa`/
> `kv_b_proj`）不含任何默认规则子串，未覆盖时全部落 SKIP，`self_attn`
> 组只剩 `o_proj`(ROWWISE)，`has_colwise=False` → attention 边界推断
> 失败，MLA 参数**静默全部不分片**（仅 warning），`_needs_cp_attn` 也
> 不置位（CP wrapper 不注入）。修复走**方式 B（ARCH_OVERRIDES 内置
> 条目）**：`q_a`/`kv_a` 下投影 → **REPLICATED**（LoRA rank 维不切，
> latent 在 TP 组内一致），`q_b`/`kv_b` 上投影 → **COLWISE**（head 维
> Shard(0)），`o_proj` 仍 ROWWISE（contract head 维的契约不变）——与
> 标准 attention 模板同构，attention 边界与 CP wrapper 恢复正常推导。
> 键同时注册 architectures 拼写（`"deepseekv2"`/`"deepseekv3"`）与
> model_type 拼写（`"deepseek_v2"`/`"deepseek_v3"`，v2/v3 同构）：
>
> ```python
> _DEEPSEEK_MLA_OVERRIDES = [
>     (["q_a_proj", "kv_a_proj_with_mqa"], ParamRole.REPLICATED),
>     (["q_b_proj", "kv_b_proj"],          ParamRole.COLWISE),
> ]
> ```
>
> 配套新增第 14 个角色 `ParamRole.REPLICATED`（全维 Replicate；**仅经
> ARCH_OVERRIDES 显式指派，默认命名规则不产生该角色**）。

### 3.6.2 Phase 3 详解：语义角色推断

语义角色推断**不依赖参数分类结果**，只看模块完全限定名（FQN）的语义：

```python
# 叶子投影/容器段名守卫：这些段名自身不是边界容器，推断时返回 unknown 继续向上。
_LEAF_SEGMENT_GUARD = frozenset({
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
    "qkv_proj", "fused_qkv", "gate_up_proj", "query_key_value",
    "experts", "shared_experts", "gate", "linear", "proj",
    "fc1", "fc2", "w1", "w2", "w3", "w13", "dense", "dense_h_to_4h", "dense_4h_to_h",
})
_ATTN_PATTERNS = ("attn", "attention")
_MLP_PATTERNS = ("mlp", "ffn", "feed_forward")
_MOE_CONTAINER_PATTERNS = ("mlp", "moe", "moe_block", "moe_layer")


def _infer_boundary_type(self, fqn: str, group: list) -> str:
    """从模块 FQN + 组内参数角色识别语义角色。

    优先级：显式 FQN 模式 > 叶子段守卫 > MoE 角色 > 参数角色组合 > 默认。
    """
    fqn_lower = fqn.lower()
    seg = _last_segment(fqn)   # 末段小写

    # 1. 显式规则（最高优先级，叶模块即边界）
    if _match_any(fqn_lower, ["embed_tokens", "wte", ".embed.", "tok_embeddings",
                              "embed_in", "word_embeddings"]):
        return "embed"
    if _match_any(fqn_lower, ["lm_head", "embed_out", "output_layer"]):
        return "lm_head"
    if _match_any(fqn_lower, ["norm", "layernorm", "rmsnorm", "ln_"]):
        return "norm"
    if _match_any(seg, ["router"]):
        return "moe_gate"

    # 2. 叶子段守卫：投影/expert 叶模块自身不是边界容器
    if seg in _LEAF_SEGMENT_GUARD:
        return "unknown"
    # 数字段守卫：HF per-expert 容器（experts.0..N）不是边界，
    # 参数须向上聚合到 moe 容器（D-09，§6.4.7）
    if seg.isdigit():
        return "unknown"

    # 3. MoE 角色：含 MOE_* 角色的组向上聚合到 moe 容器边界
    roles = {r for _, r in group}
    moe_roles = {ParamRole.MOE_EXPERT, ParamRole.SHARED_EXPERT, ParamRole.MOE_GATE}
    if roles & moe_roles:
        if _match_any(fqn_lower, list(_MOE_CONTAINER_PATTERNS)):
            return "moe_mlp"
        return "unknown"

    # 4. 参数角色组合
    has_colwise = any(r in (ParamRole.COLWISE, ParamRole.FUSED_QKV,
                            ParamRole.FUSED_GATE_UP) for _, r in group)
    has_rowwise = any(r == ParamRole.ROWWISE for _, r in group)
    if has_colwise and has_rowwise:
        if _match_any(fqn_lower, list(_ATTN_PATTERNS)):
            return "attention"
        if _match_any(fqn_lower, list(_MLP_PATTERNS)):
            return "mlp"
        return "attention"  # 默认 attention（更保守的 SP 通信）
    if has_colwise and not has_rowwise:
        # colwise-only：仅命中 MLP 模式才算 mlp（如共享 expert 未与 down 同组），
        # 否则 unknown 继续向上合并——避免 q_proj 叶模块被误判为 mlp 边界。
        if _match_any(fqn_lower, list(_MLP_PATTERNS)):
            return "mlp"
        return "unknown"

    return "unknown"
```

### 3.6.3.1 ParamRole 的桥梁作用

`ParamRole` 是命名规则和 Template 之间的**桥梁**，它连接推导管线的 3 个 Phase：

```
Phase 1: 命名规则 ──→ ParamRole ──→ Phase 2: 边界分组
                         │
                         └──────→ Phase 4: Template 查表时，ParamRole 决定
                                  每个参数在 spec.params 中的 placement
```

**ParamRole 的两个作用**：

1. **参数分组**（Phase 2）：`COLWISE` + `ROWWISE` 参数聚合到同一个父边界（如 `self_attn`），`NORM` 独立为边界
2. **placement 填充**（Phase 4）：Template 根据 ParamRole 填充 `spec.params`——
   `COLWISE → Shard(0)`，`ROWWISE → Shard(1)`，`NORM → Replicate()`

ParamRole **不决定 I/O 契约**（`in_src`/`in_dst`/`out_src`/`out_dst`）——那是由 Template 的
语义角色（attention/mlp/norm/...）决定的。

#### Template → ModuleShardingSpec 的映射机制

关键在于：**`ShardingTemplate` 只知道"每类角色用什么 placement"，不知道"具体哪些参数是哪个角色"。**
后者由 Phase 1 的 `ParamRole` 分类结果提供。`_build_spec_from_template()` 把两者组合起来：

> **规范的 `_build_spec_from_template` 实现见 §3.5 "Template -> ModuleShardingSpec Mapping"。**
> 该版本包含完整的 TP+CP+EP 三维度 placement 处理，且正确定义了 `has_tp`/`has_ep` 等变量（`has_cp` 只用于 D-05 的 embed CP 契约运行时修订——CP 不切参数，spec.params 中 CP 维恒为 Replicate()）。
>
> 以下保留原 Phase 4 章节中的数据流说明和 ParamRole 映射表，用于理解推导逻辑。

**数据流总结**：

```
Phase 1 输出:
  param_roles = {
      "model.layers.0.self_attn.q_proj.weight": ParamRole.COLWISE,
      "model.layers.0.self_attn.k_proj.weight": ParamRole.COLWISE,
      "model.layers.0.self_attn.v_proj.weight": ParamRole.COLWISE,
      "model.layers.0.self_attn.o_proj.weight": ParamRole.ROWWISE,
  }

Phase 2 分组:
  boundary_groups["model.layers.0.self_attn"] = [
      ("model.layers.0.self_attn.q_proj.weight", COLWISE),
      ("model.layers.0.self_attn.k_proj.weight", COLWISE),
      ("model.layers.0.self_attn.v_proj.weight", COLWISE),
      ("model.layers.0.self_attn.o_proj.weight", ROWWISE),
  ]

Phase 3 语义推断:
  boundary_fqn="model.layers.0.self_attn" + 含COLWISE+ROWWISE → boundary_type="attention"

Phase 4 _build_spec_from_template():
  template = TEMPLATES["attention"]
  ↓
  for each (param_fqn, role) in group:
      if role == COLWISE: spec.params["q_proj.weight"] = {TP: template.colwise_placement}
      if role == ROWWISE: spec.params["o_proj.weight"] = {TP: template.rowwise_placement}
  ↓
  spec.in_src  = template.sp_in_src   → {"hidden_states": {TP: Shard(1)}}
  spec.in_dst  = template.sp_in_dst   → {"hidden_states": {TP: Replicate()}}
  spec.out_src = template.sp_out_src  → {TP: Partial()}
  spec.out_dst = template.sp_out_dst  → {TP: Shard(1)}
```

**`colwise_placement` / `rowwise_placement` 的本质**：它们是 Template 中的**规则字段**——
定义"COLWISE 角色的参数统一用 Shard(0)"、"ROWWISE 角色的参数统一用 Shard(1)"。
具体哪些参数是 COLWISE、哪些是 ROWWISE，由 Phase 1 的命名规则 + ParamRole 决定。

#### 完整的 ParamRole → Template 字段映射表

Template 只有 4 个 placement 字段，14 个 ParamRole 枚举值全部映射到这 4 个字段上。
Template **不需要**为每个 Role 设独立字段——多个 Role 共享相同的 placement 规则：

```
ParamRole        → Template 字段           → placement 值        → 物理含义
═══════════════════════════════════════════════════════════════════════════════
EMBED            → colwise_placement        → Shard(0)            weight [V/tp, H]
LM_HEAD          → colwise_placement        → Shard(0)            weight [V/tp, H]
COLWISE          → colwise_placement        → Shard(0)            weight [H_out/tp, H_in]
FUSED_QKV        → colwise_placement        → Shard(0)            weight [3H/tp, H]（后续 SpecialHandler 调整）
FUSED_GATE_UP    → colwise_placement        → Shard(0)            weight [8H/tp, H]（后续 SpecialHandler 调整）
─────────────────────────────────────────────────────────────────────────────────
ROWWISE          → rowwise_placement        → Shard(1)            weight [H_out, H_in/tp]
─────────────────────────────────────────────────────────────────────────────────
NORM             → norm_placement           → Replicate()         weight [H] 全复制
MOE_GATE         → norm_placement           → Replicate()         router weight/bias 全复制
─────────────────────────────────────────────────────────────────────────────────
MOE_EXPERT       → moe_expert_placement     → Shard(0) on EP      expert 参数沿 expert 维切
                   + colwise_placement       → Shard(0) on TP      同时沿 TP 切 hidden 维
─────────────────────────────────────────────────────────────────────────────────
SHARED_EXPERT    → colwise/rowwise (按名)    → {EP: Replicate()}   shared expert 不参与 EP 切分
                   + EP: Replicate()         → TP: Shard(0)/(1)    TP 按 w1/w3(colwise)/w2(rowwise)
─────────────────────────────────────────────────────────────────────────────────
BIAS             → (硬编码)                  → Replicate()         bias 始终全复制
REPLICATED       → (硬编码)                  → Replicate()         MLA 下投影等（D-13；仅 ARCH_OVERRIDES
                                                                    显式指派，默认规则不产生）
SPECIAL          → (不在此处理)               → SpecialHandler     留给 Phase B 自定义
SKIP             → (不加入 params)            → —                  冻结/无需分片的参数
```

**为什么 EMBED 和 LM_HEAD 也用 `colwise_placement`？**
Embedding `[V, H]` Shard(0) → `[V/tp, H]` 和 Colwise Linear `[H_out, H_in]` Shard(0) → `[H_out/tp, H_in]` 都是"沿第一维切"——placement 规则完全相同。只是 I/O 契约不同：
- Embedding 的 I/O 由 `TEMPLATES["embed"]` 的 `sp_in_src`/`sp_in_dst`/... 控制
- LM Head 的 I/O 由 `TEMPLATES["lm_head"]` 的 I/O 字段控制
- Colwise Linear 的 I/O 由所属 boundary 的 Template（如 `TEMPLATES["attention"]`）控制

**ParamRole 只管"参数怎么切"，Template 的 boundary type 管"I/O 怎么走"。两者正交。**

以下给出每种 boundary type 从 Template → `ModuleShardingSpec` 的**完整构造结果**，
以及对应的 PrecompiledBoundary 通信计划。均假设 **TP=4, SP=true**。

#### Attention

```
已知: params = {q_proj:COLWISE, k_proj:COLWISE, v_proj:COLWISE, o_proj:ROWWISE}
     boundary_type = "attention"

构造:
  params:
    q_proj.weight → {TP: Shard(0)}    # Colwise: [H/4, H]，TP 轴沿 tensor dim 0 切
    k_proj.weight → {TP: Shard(0)}
    v_proj.weight → {TP: Shard(0)}
    o_proj.weight → {TP: Shard(1)}    # Rowwise: [H, H/4]，TP 轴沿 tensor dim 1 切

  in_src:  {"hidden_states": {TP: Shard(1)}}       # 从上游 SP norm 来的序列分片
  in_dst:  {"hidden_states": {TP: Replicate()}}     # attention 需要全量序列做 matmul
  out_src: {TP: Partial()}                          # o_proj Rowwise 天然产生 Partial(sum)
  out_dst: {TP: Shard(1)}                           # reduce-scatter → SP，给下游 norm

PrecompiledBoundary:
  in_plan:  [RedistOp("hidden_states", Shard(1)→Replicate, "all_gather")]      ← 1次通信
  out_plan: [RedistOp("output", Partial()→Shard(1), "reduce_scatter")]         ← 1次通信
```

#### MLP

```
已知: params = {gate_proj:COLWISE, up_proj:COLWISE, down_proj:ROWWISE}
     boundary_type = "mlp"

构造:
  params:
    gate_proj.weight → {TP: Shard(0)}
    up_proj.weight   → {TP: Shard(0)}
    down_proj.weight → {TP: Shard(1)}

  in_src:  {"hidden_states": {TP: Shard(1)}}       # 从 post_attn_norm 来（SP）
  in_dst:  {"hidden_states": {TP: Replicate()}}     # gate/up 的 matmul 需要全量
  out_src: {TP: Partial()}                          # down_proj Rowwise → Partial
  out_dst: {TP: Shard(1)}                           # reduce-scatter → SP

PrecompiledBoundary:
  in_plan:  [RedistOp("hidden_states", Shard(1)→Replicate, "all_gather")]
  out_plan: [RedistOp("output", Partial()→Shard(1), "reduce_scatter")]
```

#### Norm

```
已知: params = {weight:NORM}    # RMSNorm / LayerNorm 只有一个 weight
     boundary_type = "norm"

构造:
  params:
    weight → {TP: Replicate()}    # Norm 权重全复制（每个 TP rank 都有完整 [H]）

  in_src:  {"hidden_states": {TP: Shard(1)}}    # 从上游（attn/mlp）SP 输出
  in_dst:  {"hidden_states": {TP: Shard(1)}}    # identity: RMSNorm 可在分片序列上算
  out_src: {TP: Shard(1)}                        # 输出保持 SP（逐元素操作不改 placement）
  out_dst: {TP: Shard(1)}                        # identity

PrecompiledBoundary:
  in_plan:  []   # in_src == in_dst → identity，零 NCCL 调用
  out_plan: []   # out_src == out_dst → identity

注意: 虽然 in_src==in_dst 零通信，但声明仍是必要的——它告诉链式传播
     "我接受 SP 输入，输出 SP"，框架据此校验上下游契约一致。
```

#### Embedding

```
已知: params = {weight:EMBED}
     boundary_type = "embed"

说明: Embedding 的参数分片与 Colwise 相同（Shard(0) 沿词表维度），
     但输入是 token ids 而非 hidden_states，语义不同。

构造:
  params:
    weight → {TP: Shard(0)}    # [V/4, H]，词表沿 dim 0 切

  in_src:  {"input": {TP: Replicate()}}          # token ids 是整数索引，全量
  in_dst:  {"input": {TP: Replicate()}}          # identity
  out_src: {TP: Partial()}                       # Rowwise embedding 天然 Partial
  out_dst: {TP: Shard(1)}                        # reduce-scatter → SP

PrecompiledBoundary:
  in_plan:  []   # identity
  out_plan: [RedistOp("output", Partial()→Shard(1), "reduce_scatter")]
```

#### LM Head

```
已知: params = {weight:LM_HEAD}
     boundary_type = "lm_head"

构造:
  params:
    weight → {TP: Shard(0)}    # Colwise: [V/4, H]，TP 沿 dim 0 切词表

  in_src:  {"hidden_states": {TP: Shard(1)}}         # 从最后一个 norm 来（SP）
  in_dst:  {"hidden_states": {TP: Replicate()}}      # all-gather → 全量序列
  out_src: {TP: Shard(-1)}                           # Colwise 输出沿 vocab 分片
  out_dst: {TP: Shard(-1) if loss_parallel else Replicate()}

PrecompiledBoundary (loss_parallel=false):
  in_plan:  [RedistOp("hidden_states", Shard(1)→Replicate, "all_gather")]
  out_plan: [RedistOp("output", Shard(-1)→Replicate, "all_gather")]

PrecompiledBoundary (loss_parallel=true):
  in_plan:  [RedistOp("hidden_states", Shard(1)→Replicate, "all_gather")]
  out_plan: []   # Shard(-1) 直接给 CrossEntropy loss parallel
```

#### MoE Gate

```
已知: params = {weight:MOE_GATE, bias:MOE_GATE}
     boundary_type = "moe_gate"

说明: Gate/Router 权重必须全复制——所有 rank 需要相同的路由决策。

构造:
  params:
    weight → {TP: Replicate(), EP: Replicate()}    # 全复制
    bias   → {TP: Replicate(), EP: Replicate()}

  in_src:  {"hidden_states": {TP: Shard(1), EP: Replicate()}}      # SP + EP 未分片
  in_dst:  {"hidden_states": {TP: Replicate(), EP: Replicate()}}   # all-gather TP
  out_src: {TP: Replicate(), EP: Replicate()}                      # 路由 logits 全量
  out_dst: {TP: Replicate(), EP: Shard(0)}                         # redistribute → EP

PrecompiledBoundary:
  in_plan:  [RedistOp("hidden_states", {TP:Shard(1)}→{TP:Replicate}, "all_gather")]
  out_plan: [RedistOp("output", {EP:Replicate}→{EP:Shard(0)}, "redistribute")]
```

### 3.6.5 Phase 5 详解：链式传播 —— 填充 + 校验

链式传播处理 4 种场景：

| 场景 | 说明 | 链式传播行为 |
|------|------|-------------|
| **1. 填充缺省 in_src** | 用户手动注入部分声明的 Spec，in_src 为空 | 自动用上一个模块的 out_dst 填充 |
| **2. 首个/末个模块** | embedding 无上游（来自 dataloader），lm_head 无下游 | 首个模块 in_src 必须由模板声明；末个模块 out_dst 无下游校验 |
| **3. 检测模板错误** | 两个模板的 placement 声明不一致 | 编译期报告 mismatch |
| **4. 自定义模块插入** | 用户在两个标准模块间插入自定义模块 | 自动连接契约，校验上下游一致 |

场景 1（填充）的典型例子：用户手动注入了一个只声明 params 和 out_dst 的模块：

```python
# 用户只声明了 out_dst，没填 in_src
plan.modules["model.custom_block"] = ModuleShardingSpec(
    params={"weight": {TP: Shard(0)}},
    in_src={},                                    # ← 空的！
    in_dst={"x": {TP: Replicate()}},
    out_dst={"output": {TP: Shard(1)}},           # per-arg dict
)

# 链式传播：遍历上一个模块的 out_dst keys → 当前模块的 in_src
# 上一个是 "model.layers.0.mlp" → out_dst = {"output": {TP: Shard(1)}}
# → custom_block.in_src["output"] 自动填充 = {TP: Shard(1)}
```

场景 3（检测模板错误）的例子：

```python
# 假设 attention 模板错误地写成了 out_dst=Replicate
# 但下游 norm 模板声明 in_src=Shard(1)
# → 链式传播发现: Replicate ≠ Shard(1)
# → 报告: "placement mismatch: attn.out_dst ≠ norm.in_src"
# → 编译期捕获，而非运行时追查
```

**注意**：对于模板齐全的标准模型（90% 场景），链式传播**主要起校验作用**——所有 in_src 已被模板声明，链式传播验证相邻模块的契约自洽。

```python
def _chain_propagate_and_validate(
    self, plan: ShardingPlan, model: nn.Module
) -> ShardingPlan:
    """链式传播：填充缺省 in_src + 校验相邻模块契约一致性 + _is_terminal 标记。

    匹配规则（对早版按名配对的修订——模板 in_src key 与上游 out_dst key 可能
    不同名，如 attention out "output" vs moe_mlp in "x_BLD"）：
    - 双方都恰好 1 个 entry 时按"唯一 arg"配对（名字无关）；
    - 否则按 key 名配对；
    - next.in_src 整体为空时，用上游唯一 out_dst 值填充其 in_dst 声明的 key。
    """
    sorted_fqns = self._topological_sort_by_forward_order(
        list(plan.modules.keys()), model
    )

    non_terminal: set = set()
    for i in range(len(sorted_fqns) - 1):
        curr_fqn, next_fqn = sorted_fqns[i], sorted_fqns[i + 1]
        curr_spec = plan.modules[curr_fqn]
        next_spec = plan.modules[next_fqn]
        if curr_spec.out_dst is None:
            continue

        pairs = self._pair_contracts(curr_spec.out_dst, next_spec)
        for out_key, in_key in pairs:
            out_placement = curr_spec.out_dst[out_key]
            if in_key is None:
                continue
            non_terminal.add(curr_fqn)   # out_dst 被下游引用
            declared = next_spec.in_src.get(in_key)
            if not declared:
                # 场景 1：填充缺省
                next_spec.in_src[in_key] = out_placement
                continue
            # 场景 3：校验一致性
            next_in = tuple(resolve_placements(declared, plan.mesh_dim_names))
            curr_out = tuple(resolve_placements(out_placement, plan.mesh_dim_names))
            if next_in != curr_out:
                raise PlacementMismatchError(
                    f"{curr_fqn} → {next_fqn}", curr_out, next_in, "chain")

    # _is_terminal 标记：out_dst 未被任何下游 in_src 引用 → terminal
    #（按链式相邻关系判定——不做跨模块 placement 值相等匹配，避免 lm_head 的
    #  Replicate out_dst 被 embed 的 Replicate in_src 误引用。）
    for fqn, spec in plan.modules.items():
        spec._is_terminal = fqn not in non_terminal
    return plan


@staticmethod
def _pair_contracts(out_dst, next_spec):
    """产出 (out_key, in_key|None) 配对：单 entry 名字无关配对，否则按名配对。"""
    in_keys = list(next_spec.in_src.keys()) or list(next_spec.in_dst.keys())
    if len(out_dst) == 1 and len(in_keys) <= 1:
        out_key = next(iter(out_dst))
        in_key = in_keys[0] if in_keys else None
        return [(out_key, in_key)]
    pairs = []
    for out_key in out_dst:
        pairs.append((out_key, out_key if out_key in next_spec.in_src else None))
    return pairs
```

### 3.6.6 Planner 完整入口

```python
class ShardingPlanner:

    def __init__(self, plan_overrides: dict[str, ModuleShardingSpec] | None = None):
        self._classifier = ParameterClassifier(arch_overrides=ARCH_OVERRIDES)
        self._templates = TEMPLATES
        self._special_handler_patterns = dict(_SPECIAL_HANDLER_PATTERNS)
        # 用户手写 spec：Phase 4.5 合并（§3.6.7），在链式传播之前生效
        self._plan_overrides = dict(plan_overrides or {})

    def plan(
        self,
        model: nn.Module,
        mesh: DeviceMesh,
        *,
        tp_size: int = 1,
        cp_size: int = 1,
        ep_size: int = 1,
        sequence_parallel: bool = True,
        loss_parallel: bool = False,
    ) -> ShardingPlan:
        arch = self._get_architecture(model)
        mesh_dim_names = self._build_mesh_dim_names(mesh, tp_size, cp_size, ep_size)
        # D-10 TP-extend-EP（§6.4.8）：ep_size 即扩展 EP 组大小（a2a 通信域，
        # 由 TP 组向相邻 dp/cp rank 扩展）；校验在 _mark_hf_native_moe 实际
        # 命中 HF 原生 MoE 时进行
        ep_extend = ep_size if ep_size > 1 else 0

        # Phase 1: 参数角色分类
        param_roles = self._classify_all_params(model, arch)

        # Phase 2: 通信边界分组
        boundary_groups = self._group_by_boundary(param_roles)

        # Phase 3+4: 语义推断 + 模板填充 I/O
        param_ndims = {name: p.ndim for name, p in model.named_parameters()}
        plan = ShardingPlan(
            mesh_dim_names=mesh_dim_names,
            sequence_parallel=sequence_parallel,
            loss_parallel=loss_parallel,
        )
        inferred_templates: dict[str, ShardingTemplate] = {}
        for boundary_fqn, group in boundary_groups.items():
            boundary_type = self._infer_boundary_type(boundary_fqn, group)
            template = self._templates.get(boundary_type)
            if template is None:
                logger.warning("No template for boundary_type=%s at %s", boundary_type, boundary_fqn)
                continue

            spec = self._build_spec_from_template(
                boundary_fqn, group, template,
                sequence_parallel, loss_parallel, mesh_dim_names,
                param_ndims=param_ndims,
            )
            if spec is not None:
                if boundary_type == "moe_mlp":
                    # D-09/D-11 堆叠/batched 布局识别 + D-10 TP-extend-EP 标记
                    #（_ep_stack/_moe_router/_ep_size + SP-in identity 契约，§6.4.7/§6.4.8）
                    self._mark_hf_native_moe(
                        spec, group, boundary_fqn, template, mesh_dim_names, arch,
                        ep_extend=ep_extend, mesh=mesh, model=model,
                        param_ndims=param_ndims)
                plan.modules[boundary_fqn] = spec
                inferred_templates[boundary_fqn] = template

        # Phase 4.5: 用户 plan_overrides 合并（§3.6.7）——须在 Phase 5 之前，
        # 覆盖 spec 仍参与相邻契约校验与 _is_terminal 标记
        self._merge_plan_overrides(plan, model, inferred_templates)

        # Phase 5: 链式传播校验
        plan = self._chain_propagate_and_validate(plan, model)

        # Phase 6: 特殊参数处理
        plan.special_handlers = self._collect_special_handlers(param_roles)

        # tied-weight 检测（embed <-> lm_head 共享存储）
        plan.tied_pairs = self._detect_tied_pairs(model)

        return plan

    # ── Planner 内部辅助方法签名 ──

    def _get_architecture(self, model: nn.Module) -> str:
        """检测模型架构名（如 llama/qwen2/mixtral），用于选择 ARCH_OVERRIDES。

        优先级：``config.architectures[0]`` > ``config.model_type`` > 类名启发式。
        全部小写化并去 ``ForCausalLM`` / ``ForConditionalGeneration`` 等后缀，
        得到如 ``"llama"``、``"qwen2"``、``"mixtral"`` 的 canonical 架构名。
        """
        cfg = getattr(model, "config", None)
        arch_str = None
        # 1. HF config.architectures（如 ["Qwen2ForCausalLM"]）
        archs = getattr(cfg, "architectures", None)
        if archs:
            arch_str = archs[0]
        # 2. 回退 config.model_type（如 "qwen2"）
        if not arch_str:
            arch_str = getattr(cfg, "model_type", None)
        # 3. 回退类名
        if not arch_str:
            arch_str = type(model).__name__

        s = arch_str.lower()
        for suffix in ("forcausallm", "forconditionalgeneration",
                       "forsequenceclassification", "forimagetexttotext"):
            if s.endswith(suffix):
                s = s[: -len(suffix)]
        return s

    def _build_mesh_dim_names(
        self, mesh: DeviceMesh, tp_size: int, cp_size: int, ep_size: int,
    ) -> tuple[str, ...]:
        """从 mesh 和并行规模构建实际启用的 mesh_dim_names 元组。

        以 ``mesh.mesh_dim_names`` 为权威顺序，过滤出 DTensor 管理的轴
        （tp/cp/ep）。DP/PP 轴不在 DTensor 管理范围（由 FSDP2/PP runtime 管），
        故不纳入。若 mesh 未声明 mesh_dim_names，则按 (tp, cp, ep) 顺序补全。
        """
        mesh_names = tuple(getattr(mesh, "mesh_dim_names", ()) or ())
        dtensor_axes = ("tp", "cp", "ep")
        active = {ax for ax, sz in (("tp", tp_size), ("cp", cp_size), ("ep", ep_size))
                  if sz and sz > 1}
        if mesh_names:
            return tuple(n for n in mesh_names if n in dtensor_axes and n in active)
        # 回退：按固定顺序输出启用的轴
        return tuple(ax for ax in dtensor_axes if ax in active)

    def _classify_all_params(
        self, model: nn.Module, arch: str,
    ) -> dict[str, ParamRole]:
        """Phase 1：委托 ParameterClassifier 分类（规则见 param_role.py）。

        规则来源（优先级递减）：
          1. ``ARCH_OVERRIDES[arch]`` —— 显式 (fqn 模式, ParamRole) 覆盖
             （已内置 DeepSeek MLA 条目：q_a/kv_a 下投影 → REPLICATED，
             q_b/kv_b 上投影 → COLWISE，v2/v3 两种拼写均注册）；
          2. ``_build_default_rules()`` 返回的默认后缀规则
             （``list[tuple[list[str], ParamRole]]``，按顺序首匹配）；
          3. 命中不到 → ``ParamRole.SKIP``（不分片，原样保留）。

        注意：``ln`` 子串规则易误伤 ``linear``/``kernel``，默认规则用更精确的
        ``norm``/``layernorm``/``rmsnorm``/``ln_`` 前缀匹配（见 param_role.py
        ``_build_default_rules``）。
        """
        return self._classifier.classify(model, arch)

    def _group_by_boundary(
        self, param_roles: dict[str, ParamRole],
    ) -> dict[str, list[tuple[str, ParamRole]]]:
        """Phase 2：两趟分组（修正早版"单参数临时 group 推断"会把 q_proj 叶
        模块误判为 mlp 边界的缺陷）：

        趟 1：按直属模块 FQN 分组（去掉 leaf 参数名）。
        趟 2：工作队列深度优先——组内角色齐全时做边界推断；unknown 则把整组
              参数向上合并到父模块并入队（父模块更浅、必然后处理；兄弟模块的
              参数先合并齐备再推断，避免 q_proj 单独被误判）。回溯到根仍
              unknown 归入参数所在模块（后续无模板命中 → warning 跳过）。
        """
        own: dict[str, list] = {}
        for fqn, role in param_roles.items():
            module_fqn = ".".join(fqn.split(".")[:-1])
            own.setdefault(module_fqn, []).append((fqn, role))

        merged = {mfqn: list(params) for mfqn, params in own.items()}
        pending = sorted(merged.keys(), key=lambda f: f.count("."), reverse=True)
        consumed, groups = set(), {}
        i = 0
        while i < len(pending):
            mfqn = pending[i]; i += 1
            if mfqn in consumed:
                continue
            params = merged.get(mfqn, [])
            if self._infer_boundary_type(mfqn, params) != "unknown":
                groups[mfqn] = params
            else:
                parent = mfqn.rsplit(".", 1)[0] if "." in mfqn else ""
                if parent:
                    if parent not in merged:
                        merged[parent] = []
                        pending.append(parent)   # 父模块更浅，尾部入队即可
                    merged[parent].extend(params)
                else:
                    origin = ".".join(params[0][0].split(".")[:-1]) if params else mfqn
                    groups.setdefault(origin, params)
            consumed.add(mfqn)
        return groups

    def _topological_sort_by_forward_order(
        self, fqns: list[str], model: nn.Module,
    ) -> list[str]:
        """按 forward 执行顺序（= 子模块注册顺序）排序 FQN 列表。

        遍历 ``model.modules()``（PyTorch 保证返回顺序为注册/forward 调用顺序），
        过滤出在 ``fqns`` 中的条目。未命中的 FQN 追加到末尾（保守处理，
        并 ``logger.warning`` 提示，便于发现注册顺序与 forward 不一致的模型）。
        ModuleList / 手动注册 / skip-connection 均按注册顺序处理；若模型在
        forward 中乱序调用子模块，需通过 ``ARCH_OVERRIDES`` 显式声明。
        """
        fqn_set = set(fqns)
        ordered: list[str] = []
        seen: set[str] = set()
        for name, _module in model.named_modules():
            if name in fqn_set and name not in seen:
                ordered.append(name)
                seen.add(name)
        # 未命中（注册名与传入 FQN 不一致）—— 追加并告警
        missing = fqn_set - seen
        if missing:
            logger.warning(
                "_topological_sort_by_forward_order: %d FQN 未在 named_modules "
                "中命中，追加到末尾: %s", len(missing), sorted(missing)[:5],
            )
            ordered.extend(sorted(missing))
        return ordered

    def _collect_special_handlers(
        self, param_roles: dict[str, ParamRole],
    ) -> dict[str, str]:
        """Phase 6：收集所有 SPECIAL 角色参数，映射到 handler 名。

        映射规则：若参数名命中某 SpecialHandler 注册的模式（如 ``gated_delta``
        → ``"gated_delta_tp_shard"``），则用该 handler；否则默认 ``"default"``。
        返回的 ``{fqn: handler_name}`` 供 ShardingApplier Phase B 查表调用。
        """
        result: dict[str, str] = {}
        for fqn, role in param_roles.items():
            if role != ParamRole.SPECIAL:
                continue
            handler_name = "default"
            for pattern, hname in self._special_handler_patterns.items():
                if _match_any(fqn.lower(), [pattern.lower()]):
                    handler_name = hname
                    break
            result[fqn] = handler_name
        return result
```

#### 链式传播的局限性：reshape / reduce 边界

链式传播假设相邻模块的 **tensor 维度索引与逻辑轴的对应关系一致**。
当模块间存在更改 tensor shape 的操作时，`Shard(N)` 可能指向不同的逻辑维度：

```
模块 A 输出: [B, S/tp, H], out_dst={TP: Shard(1)}   ← Shard(1) = 沿 S 切
    ↓ reshape: [B, S, H] → [B, H, S]
模块 B 输入: [B, H, S/tp], in_src={TP: Shard(1)}    ← Shard(1) = 沿 H 切！

链式传播校验: Shard(1) == Shard(1) → "通过" ✅
实际语义:  沿 S 切 ≠ 沿 H 切 → 逻辑错误 ❌
```

**这不是链式传播的问题，而是 `Shard(N)` 本身无法表达逻辑轴的固有限制。**
`Shard(N)` 只关心 tensor 的第 N 维，不关心这一维代表什么。

**处理方式**：

| 场景 | 处理 |
|------|------|
| **标准 Transformer**（embed → layers → norm → lm_head） | activation 始终 `[B, S, H]`，Shard(1) 始终是序列维。链式传播完全有效。 |
| **reshape 边界**（如 ViT 的 patch embedding、Qwen VL 的 mRoPE 3D→2D） | 用户必须**显式声明** reshape 后模块的 `in_src`，不依赖链式传播自动填充。 |
| **reduce 边界**（如从 `[B, S, H]` pool 到 `[B, H]`） | `Shard(1)` 从序列维变为隐藏维，用户必须显式声明新的 `in_src`。 |

**实践中**：链式传播对 95% 的 decoder 层间传播是正确且有用的（校验模板声明一致性）。
在 reshape/reduce 边界处，用户通过显式声明 `in_src` 来覆盖自动填充。
链式传播并非"没有必要"——它把运行时追查到 placement 不匹配的 bug 提前到编译期捕获。

### 3.6.7 Phase 4.5：用户 `plan_overrides` 合并（手写 spec 的一等注入路径）

> 对应 §8.5 方式 D。实现：`ShardingPlanner._merge_plan_overrides`；
> UT：`test_s1_plan_overrides.py` + `test_dist_s5_plan_overrides.py`。

**动机**：§8.4 方式 C 要求用户绕开 planner 手工构建整个 `ShardingPlan`，或
`plan()` 返回后再打补丁——前者丢失模板推导，后者丢失 Phase 5 的链式契约校验
与 `_is_terminal` 标记（补丁 spec 与上下游契约不一致时要等运行时 RedistOp
执行才暴露）。`plan_overrides` 把手写 spec 的合并提前到 **Phase 5 之前**，
使覆盖 spec 与推导 spec 走完全相同的校验路径：

```python
planner = ShardingPlanner(plan_overrides={
    "model.layers.0.self_attn": ModuleShardingSpec(
        params={
            "wq.weight": {TP: Shard(0)}, "wk.weight": {TP: Shard(0)},
            "wv.weight": {TP: Shard(0)}, "wo.weight": {TP: Shard(1)},
        },
        # 多输入模块：契约 key 直接写真实签名参数名（forward(self, attn_bias, x)）
        in_src={"x": {TP: Shard(1)}},
        in_dst={"x": {TP: Replicate()}},
        out_src={TP: Partial()},        # 标量简写，合并时自动归一化为 {"output": ...}
        out_dst={TP: Shard(1)},
    ),
})
plan = planner.plan(model, mesh, tp_size=2)
```

**合并语义**（`_merge_plan_overrides`，在 Phase 3+4 循环之后、Phase 5 之前执行）：

| 规则 | 行为 |
|------|------|
| fqn 命中 planner 已生成的 spec | **整体替换**（用户 spec 为权威），记录日志 |
| fqn 未命中（漏识别/无模板/无参数容器） | **插入**，照常参与拓扑排序与链式传播 |
| 结构标记 `use_local_map` / `_needs_cp_attn` | `use_local_map` 为**公开可配置字段**（2026-07-21 起，自研数据相关模块可显式置 True 走 local-region wrapper）；推断模板为 True 时**强制置位**（MoE all-to-all、CP K/V all-gather 缺失会导致**数值错误但不报错**，不允许借覆盖关闭）；`_needs_cp_attn` 保持内部字段 |
| inner-wrap 自定义入口 `inner_target` / `inner_wrapper` | **公开可配置字段**（2026-07-21 起，见 §4.4.2）：target 定位 / 内置注册表名或自定义 callable；**不改写任何标记**——inner-wrap 门控由 `_resolve_inner_wrapper` 解析链派生 |
| local-region 自定义计算 `local_compute_fn` | **公开可配置字段**（2026-07-21 起，见 §4.4.3）：自定义 local-region compute_fn；声明即生效——骨架门控由 `_resolve_local_compute_fn` 解析链派生，**不改写 `use_local_map`** |
| `out_src`/`out_dst` 标量简写 | 合并时调用 `_normalize_out_fields` 归一化 |
| `_is_terminal` | 一律由 Phase 5 统一标记，用户预设值被覆盖 |
| 对象隔离 | **深拷贝**用户 spec——chain 传播会就地改 `in_src`，plan() 可重复调用，不污染调用方持有的对象 |
| fqn 未命中 `named_modules`（拼写错误） | **fail-fast `ValueError`**（显式输入不容忍静默丢弃） |
| 值非 `ModuleShardingSpec` | `TypeError` |

**关键性质**：覆盖 spec 与上下游的契约冲突在 `plan()` 内即抛
`PlacementMismatchError`（与推导 spec 相同的校验时机）；CP>1 时被覆盖的
attention 模块无需手写 `_needs_cp_attn=True`——模板补齐保证 D-01'' 的
CP wrapper 注入不遗漏。

**与方式 B/C 的分工**：命名非标准 → 方式 B（`ARCH_OVERRIDES`）；整模型绕开
planner → 方式 C（§8.4）；**个别模块**的契约/参数分片需要定制（多输入契约
key、特殊通信、reshape 边界的显式 `in_src`）→ 方式 D（本节）。

---



---

## 4. ShardingApplier: Apply Sharding at Runtime

> Call site: apply_sharding_plan() runtime application -- param sharding + PrecompiledBoundary + forward wrapping

### 4.1 核心入口

```python
# hyper_models/components/distributed/sharding_applier.py

def apply_sharding_plan(
    model: nn.Module | list[nn.Module],
    plan: ShardingPlan,
    mesh: DeviceMesh,
    *,
    validate_mode: bool = False,
) -> tuple[nn.Module | list[nn.Module], dict | None]:
    """对任意 nn.Module（或 PP 多 part 列表）应用 ShardingPlan，启用双模式 DTensor。

    返回 (model, tp_grad_info)：
    - production 模式下，Phase C 入口调用一次 `_local_params_context` 把 DTensor 参数
      永久解包为 plain local tensor，并构造 tp_grad_info 供 fully_shard 使用；
    - validate 模式下不解包（参数保持 DTensor），tp_grad_info 为 None。

    `mesh` 为包含 TP 维度的 DeviceMesh；`build_tp_grad_info` 取其 TP 子 mesh。
    """
    mesh_dim_names = plan.mesh_dim_names
    # 活跃子 mesh：planner 会剔除 size=1 轴（plan.mesh_dim_names），但传入的
    # mesh 可能仍含这些轴——placements 按 plan.mesh_dim_names 解析，维度数
    # 必须与 mesh 对齐，否则 distribute_tensor 会静默错轴分片。
    full_mesh = mesh   # D-10：派生 expert mesh 需要全 dense 区域（含 dp/cp 轴）
    mesh = _get_active_mesh(mesh, mesh_dim_names)
    tp_mesh = _get_tp_submesh(mesh, mesh_dim_names)
    models = model if isinstance(model, list) else [model]

    # D-10：任一 spec 开启 TP-extend-EP 时构建派生 expert mesh（全 dense
    # 区域重分区，expert 参数分片与 region 内通信组均使用；§6.4.8）
    ep_size = next((getattr(s, "_ep_size", 0) for s in plan.modules.values()
                    if getattr(s, "_ep_size", 0)), 0)
    expert_mesh = (_build_expert_mesh(full_mesh, full_mesh.mesh_dim_names, ep_size)
                   if ep_size else None)

    # ====== Phase 0: 归一化 out_src/out_dst 标量简写为 dict 契约 ======
    # 覆盖用户注入路径（§8.4 方式 C 手动声明的 spec 可能用标量简写），
    # 避免 _compile_output_plan 遇到标量 NamedPlacement 时 AttributeError。
    # _build_spec_from_template 已在 planner 内部调用过一次，此处对全部 spec
    #（含用户注入）做幂等归一化，保证下游统一 dict 契约。
    for spec in plan.modules.values():
        _normalize_out_fields(spec)

    # ====== Phase A: 参数分片 ======
    for part in models:
        for module_fqn, spec in plan.modules.items():
            module = _resolve_module(part, module_fqn)
            # D-09b：HF 原生 MoE 的 per-expert 参数先堆叠为 [E, ...]，
            # 再按 stacked 条目分片（§6.4.7）
            if getattr(spec, "_ep_stack", None):
                _stack_moe_experts(module, spec._ep_stack)
            if getattr(spec, "_ep_size", 0):
                # D-10：expert 参数在派生 expert mesh 上分片（{EP: Shard(0)}，
                # 仅 expert 维切分），其余 dense 参数走主 mesh
                expert_params = {k: v for k, v in spec.params.items()
                                 if k.startswith("experts.")}
                dense_params = {k: v for k, v in spec.params.items()
                                if not k.startswith("experts.")}
                _shard_module_params(module, expert_params, expert_mesh,
                                     expert_mesh.mesh_dim_names)
                _shard_module_params(module, dense_params, mesh, mesh_dim_names)
            else:
                _shard_module_params(module, spec.params, mesh, mesh_dim_names)

    # ====== Phase B: 特殊处理器 ======
    for part in models:
        for param_ref, handler_name in plan.special_handlers.items():
            handler = SPECIAL_HANDLERS.get(handler_name)
            if handler is None:
                logger.warning("SPECIAL_HANDLERS 未注册 handler: %s", handler_name)
                continue
            module_fqn, param_name = param_ref.rsplit(".", 1)
            handler(_resolve_module(part, module_fqn), param_name, mesh)

    # ====== Phase C 入口: build 期一次性解包（接上 _local_params_context 调用链） ======
    # production 下一次性把 DTensor[TP] 参数替换为 local tensor（plain），在 fully_shard 之前。
    # validate 下不解包，参数保持 DTensor 以走 DTensor dispatch 校验。
    tp_grad_info = None
    if not validate_mode:
        tp_grad_records: dict = {}
        for part in models:
            tp_grad_records.update(_local_params_context(part))
        if tp_grad_records and tp_mesh is not None:
            tp_grad_info = build_tp_grad_info(plan, tp_mesh)

    # ====== Phase C: 包装 forward（production/validate/moe/cp/vocab_embed 五路，见 §4.4.2） ======
    for part in models:
        _apply_phase_c(part, plan, mesh, validate_mode, expert_mesh=expert_mesh)

    # ====== Phase D: tied weights ======
    tied_pairs = list(plan.tied_pairs) or detect_tied_weights(models[0])
    for part in models:
        _replicate_tied_weights(part, mesh, tied_pairs)

    return model, tp_grad_info


def _get_tp_submesh(mesh: DeviceMesh, mesh_dim_names: tuple[str, ...]) -> DeviceMesh | None:
    """从 mesh 中提取 TP 子 mesh（"tp" 维存在时），用于 build_tp_grad_info。"""
    if "tp" not in mesh_dim_names:
        return None
    return mesh["tp"]  # DeviceMesh 支持按维度名取子 mesh


def _get_cp_submesh(mesh: DeviceMesh, mesh_dim_names: tuple[str, ...]) -> DeviceMesh | None:
    """从 mesh 中提取 CP 子 mesh（"cp" 维存在时），用于 _wrap_cp_inner_attention。"""
    if "cp" not in mesh_dim_names:
        return None
    return mesh["cp"]
```

### 4.2 Phase A: 参数分片

> **对应时序**: ③.4.5.8 Phase A — `distribute_tensor()` → DTensor

```python
def _shard_module_params(
    module: nn.Module,
    param_specs: dict[str, NamedPlacement],
    mesh: DeviceMesh,
    mesh_dim_names: tuple[str, ...],
) -> None:
    """distribute_tensor() 转换参数为 DTensor。

    - meta tensor → DTensor: DTensor._local_tensor 仍为 meta（零显存）
      → 等待后续 to_empty() 材质化 + 权重加载填充
    - real tensor → DTensor: 物理切分，每个 rank 持有 local shard
    """
    for param_path, named in param_specs.items():
        param = _get_attr_by_path(module, param_path)
        placements = tuple(resolve_placements(named, mesh_dim_names))
        if not placements:
            continue  # 无活跃 DTensor 轴（全部 size 1）——无需分片

        if isinstance(param, DTensor):
            if tuple(param.placements) != placements:
                raise PlacementMismatchError(...)
            continue

        src = param.data if hasattr(param, 'data') else param
        dt = distribute_tensor(src, mesh, placements)
        requires_grad = getattr(param, 'requires_grad', True)
        _set_param_by_path(module, param_path,
                           nn.Parameter(dt, requires_grad=requires_grad))
```


---

### 4.3 Phase B: PrecompiledBoundary

#### 4.3.1 RedistOp：单个通信操作

```python
@dataclass
class RedistOp:
    """一个预编译的 redistribute 操作。

    collective_type 的用途：
    - "identity": 跳过通信（零开销）
    - "all_gather" / "reduce_scatter" / "all_reduce" / "redistribute":
      调试 + profiling 用途；实际通信统一走 DTensor.redistribute()
    """
    arg_name: str
    arg_index: int | None
    mesh: DeviceMesh
    src_placements: tuple[Placement, ...]
    dst_placements: tuple[Placement, ...]
    collective_type: str  # 调试标签，非通信路径选择

    def execute(self, tensor: torch.Tensor, *,
                as_dtensor: bool = False) -> torch.Tensor:
        """执行通信。

        所有非 identity 路径统一走 DTensor.redistribute()。
        DTensor 内部根据 (src, dst) placement 自动选择最优 NCCL collective。

        Args:
            tensor: 输入 local tensor
            as_dtensor: True → 返回 DTensor（校验模式），False → 返回 local tensor
        """
        if self.collective_type == "identity":
            if isinstance(tensor, DTensor):
                # validate（as_dtensor=True）保持 DTensor；production 返回
                # local——identity op 的输入可能来自 local region 的
                # from_local 重包装（MoE/CP wrapper），boundary 出口必须解包，
                # 否则 DTensor 泄漏到下游产生 mixed dispatch（§12.4.1）。
                return tensor if as_dtensor else tensor.to_local()
            if as_dtensor:
                return DTensor.from_local(
                    tensor, self.mesh, list(self.src_placements), run_check=False
                )
            return tensor

        # 统一路径：零拷贝包装 → redistribute → 可选 to_local
        if isinstance(tensor, DTensor):
            dt = tensor
        else:
            dt = DTensor.from_local(
                tensor, self.mesh, list(self.src_placements), run_check=False
            )
        dt = dt.redistribute(
            placements=list(self.dst_placements), async_op=False
        )
        return dt if as_dtensor else dt.to_local()


def _get_arg(args, kwargs, name, idx, default=None):
    if name in kwargs: return kwargs[name]
    if idx is not None and idx < len(args): return args[idx]
    return default

def _set_arg(args, kwargs, name, idx, value):
    if name in kwargs: kwargs[name] = value; return args, kwargs
    if idx is not None and idx < len(args):
        args = list(args); args[idx] = value; return tuple(args), kwargs
    kwargs[name] = value; return args, kwargs
```

#### 4.3.2 Why use unified `DTensor.redistribute()` instead of explicit collectives?

**结论：统一走 `redistribute()` 是最优选择。** 原因：

1. **PyTorch 内部已做最优选择**：`DTensor.redistribute()` 根据 `(src_placements, dst_placements)` 自动分派到正确的 NCCL collective（`Shard→Replicate` 走 all-gather，`Partial→Shard` 走 reduce-scatter 等），不需要手动判断。

2. **显式 collective 无性能收益**：`DTensor.from_local(run_check=False)` 和 `to_local()` 都是零拷贝（寄存器级操作），redistribute 内部直接调用 NCCL kernel，无额外 host 开销。

3. **`collective_type` 字段的真实用途**：

```python
# 用途 1: 调试日志
for op in boundary.in_plan:
    if op.collective_type != "identity":
        logger.debug(
            "[%s] %s: %s → %s (%s)",
            module_name, op.arg_name,
            op.src_placements, op.dst_placements, op.collective_type,
        )

# 用途 2: Profiling
with torch.profiler.record_function(f"boundary_{op.collective_type}"):
    result = op.execute(tensor)

# 用途 3: 未来平台特定优化（如 NPU 的 fused all-gather+matmul 指令）
if op.collective_type == "all_gather" and _platform_has_fused_ag_matmul():
    return _fused_all_gather_matmul(tensor, ...)
# 当前统一走 DTensor.redistribute()
```

#### 4.3.3 PrecompiledBoundary Compilation Logic

```python
class PrecompiledBoundary:

    def __init__(self, spec: ModuleShardingSpec, mesh: DeviceMesh,
                 mesh_dim_names: tuple[str, ...]):
        self.in_plan = self._compile_input_plan(spec, mesh, mesh_dim_names)
        self.out_plan = self._compile_output_plan(spec, mesh, mesh_dim_names)

    def _compile_input_plan(self, spec, mesh, mesh_dim_names) -> list[RedistOp]:
        """从 in_src → in_dst 编译输入通信计划（sorted 遍历保证确定性顺序）。"""
        plan = []
        all_names = set(spec.in_src.keys()) | set(spec.in_dst.keys())

        for name in sorted(all_names):
            src_named = spec.in_src.get(name, {})
            dst_named = spec.in_dst.get(name, {})

            src_p = tuple(resolve_placements(src_named, mesh_dim_names))
            dst_p = tuple(resolve_placements(dst_named, mesh_dim_names))

            plan.append(RedistOp(
                arg_name=name,
                arg_index=None,
                mesh=mesh,
                src_placements=src_p,
                dst_placements=dst_p,
                collective_type=_classify_collective(src_p, dst_p),
            ))
        return plan

    def _compile_output_plan(self, spec, mesh, mesh_dim_names) -> list[RedistOp]:
        """从 out_src → out_dst 编译输出通信计划（per-arg dict，支持多输出模块）。

        out_src/out_dst 均为 dict[str, NamedPlacement]，按 key 逐个编译。
        如果 out_src 为 None，则不编译（输出不需要通信，或者模块输出不是 DTensor）。
        如果 out_dst 为 None，则不编译（identity 路径）。

        多输出（模块返回 tuple）映射：RedistOp.arg_index 记录该输出在 tuple 中的
        位置。位置来源优先级：(1) spec.out_names（显式声明的输出名顺序）；
        (2) 否则按 out_src 的 key 顺序作为 tuple 索引。单输出模块 arg_index=0。
        """
        if spec.out_src is None or spec.out_dst is None:
            return []

        out_names = getattr(spec, "out_names", None) or list(spec.out_src.keys())
        name_to_idx = {name: i for i, name in enumerate(out_names)}

        plan = []
        all_names = set(spec.out_src.keys()) | set(spec.out_dst.keys())
        for name in sorted(all_names):
            src_named = spec.out_src.get(name, {})
            dst_named = spec.out_dst.get(name, {})
            src_p = tuple(resolve_placements(src_named, mesh_dim_names))
            dst_p = tuple(resolve_placements(dst_named, mesh_dim_names))
            if src_p == dst_p:
                continue  # identity，不需要通信
            plan.append(RedistOp(
                arg_name=name,
                arg_index=name_to_idx.get(name, 0),
                mesh=mesh,
                src_placements=src_p,
                dst_placements=dst_p,
                collective_type=_classify_collective(src_p, dst_p),
            ))
        return plan

    def redistribute_inputs(self, args, kwargs, *, as_dtensor=False):
        """执行输入重分布。as_dtensor=True → 返回 DTensor（校验模式）。

        arg 未在 args/kwargs 中找到（None）时跳过该 op——如 embed 的
        in_src key "input" 与实际 kwargs 名 "input_ids" 不同名且 identity。
        """
        for op in self.in_plan:
            arg = _get_arg(args, kwargs, op.arg_name, op.arg_index, default=None)
            if arg is None:
                continue
            result = op.execute(arg, as_dtensor=as_dtensor)
            args, kwargs = _set_arg(args, kwargs, op.arg_name, op.arg_index, result)
        return args, kwargs

    def redistribute_outputs(self, outputs, *, as_dtensor_input=False):
        """执行输出重分布。支持单输出（Tensor）与多输出（tuple/list[Tensor]）。

        as_dtensor_input=True → 输入已是 DTensor（校验模式）。
        多输出按 op.arg_index（来自 spec.out_names 或 out_src key 顺序）索引
        outputs tuple，逐个执行 redistribute；返回与输入同构（单值或 tuple）。
        """
        is_tuple = isinstance(outputs, (tuple, list))
        outputs_list = list(outputs) if is_tuple else [outputs]
        for op in self.out_plan:
            idx = op.arg_index if op.arg_index is not None else 0
            if idx >= len(outputs_list):
                # 模块未返回该命名输出（如 present_kv 在推理时省略）。
                logger.warning(
                    "PrecompiledBoundary: out_plan expects output '%s' at index %d, "
                    "but module returned only %d outputs. Skipping redistribution for this output.",
                    op.arg_name, idx, len(outputs_list)
                )
                continue
            tensor = outputs_list[idx]
            if tensor is None:
                continue
            # as_dtensor_input=True（validate）→ 保持 DTensor 供 out_dst 校验；
            # 否则返回 local（production / 边界最终出口）。
            outputs_list[idx] = op.execute(tensor, as_dtensor=as_dtensor_input)
        return tuple(outputs_list) if is_tuple else outputs_list[0]


def resolve_placements(
    named: dict[str, Placement],
    mesh_dim_names: tuple[str, ...],
) -> list[Placement]:
    """Arrange placements in mesh_dim_names order, fill missing axes with Replicate()."""
    return [named.get(axis, Replicate()) for axis in mesh_dim_names]


def _classify_collective(src, dst) -> str:
    """从 placement 推导通信类型（调试/profiling 用途）。

    只比较有差异的维度——identity 维（如 attention 的 CP 维 Shard(1)→Shard(1)）
    不参与分类，使 TP 维 Shard→Replicate 正确归类为 all_gather（§12.3.5）。
    """
    if tuple(src) == tuple(dst):
        return "identity"
    diff_src = tuple(s for s, d in zip(src, dst) if s != d)
    diff_dst = tuple(d for s, d in zip(src, dst) if s != d)

    has_shard_src = any(isinstance(p, Shard) for p in diff_src)
    has_partial_src = any(isinstance(p, Partial) for p in diff_src)
    has_shard_dst = any(isinstance(p, Shard) for p in diff_dst)
    all_replicate_dst = all(
        not isinstance(p, (Shard, Partial)) for p in diff_dst
    )

    if has_partial_src and has_shard_dst:
        return "reduce_scatter"
    if has_partial_src and all_replicate_dst:
        return "all_reduce"
    if has_shard_src and all_replicate_dst:
        return "all_gather"
    return "redistribute"
```

#### 4.3.4 Compile-time vs Runtime Comparison

```python
# ── 当前（dmodule.Module 运行时模式）：每次 forward 都有判断开销 ──
def forward_with_redistribution(*args, **kwargs):
    args, kwargs = self._redistribute_inputs(tp_mesh, mesh_axis_names, sc, args, kwargs)
    # ↑ 内部：
    #   for name, value in new_kwargs.items():
    #       if not platform.is_tensor(value): continue         ← 条件判断
    #       if src_named is None and dst_named is None: continue ← 条件判断
    #       if not isinstance(value, DTensor):                  ← 条件判断
    #           resolve_placements()                             ← 每次解析
    #       if placement differs: redistribute()                 ← 条件通信
    outputs = fn(*args, **kwargs)
    return self._redistribute_outputs(...)

# ── 新方案：PrecompiledBoundary 在 plan 阶段构建好，运行时零判断 ──
def production_forward(*args, **kwargs):
    args, kwargs = self._boundary.redistribute_inputs(args, kwargs)
    # ↑ for op in self.in_plan: op.execute(value)  ← 直接执行，零判断
    outputs = original_forward(*args, **kwargs)
    return self._boundary.redistribute_outputs(outputs)
```

---



### 4.4 Phase C: Forward Wrapping: 生产模式 forward 包装

> **对应时序**: ③.4.5.8 Phase C — `_wrap_forward()`

#### 4.4.1 标准生产模式：`_wrap_production_forward` + Phase C 辅助函数

本节含 Phase C 入口的 build 期一次性解包（`_local_params_context`）、
路径式参数/模块定位辅助函数，以及标准生产模式 wrapper `_wrap_production_forward`。

```python
# hyper_models/components/distributed/sharding/apply.py
def _local_params_context(model: nn.Module) -> dict[str, tuple[Placement, ...]]:
    """build 期一次性解包：把 DTensor[TP] 参数替换为 _local_tensor（plain）。
    在 apply_sharding_plan 的 Phase C 入口、fully_shard 之前调用，永久解包不恢复。

    返回 {fqn: placements} 是解包前的 placement 快照（仅用于诊断/调试）。
    注意 tp_grad_info 的 canonical 数据来源是 ShardingPlan（`build_tp_grad_info(plan, tp_mesh)`），
    而非此返回值——production 模式下 plan 仍保留完整 placement 信息。"""
    tp_grad_records = {}
    for name, param in list(model.named_parameters()):
        if isinstance(param, DTensor):
            tp_grad_records[name] = param.placements
            # 路径式赋值：name 是点分 FQN（如 layers.0.self_attn.q_proj.weight），
            # object.__setattr__(model, name, ...) 只会在 model 上设一个怪属性，
            # 不会替换子模块参数。必须沿路径定位到真正的父模块再赋值。
            _set_param_by_path(model, name, nn.Parameter(
                param.to_local(), requires_grad=param.requires_grad))
    return tp_grad_records


def _set_param_by_path(model: nn.Module, fqn: str, new_param: nn.Parameter) -> None:
    """沿点分 FQN 定位父模块并替换 leaf 参数。"""
    *path, leaf = fqn.split(".")
    obj = model
    for p in path:
        obj = obj[int(p)] if p.isdigit() else getattr(obj, p)
    if hasattr(obj, "register_parameter"):
        obj.register_parameter(leaf, new_param)
    else:
        object.__setattr__(obj, leaf, new_param)


def _get_attr_by_path(model, fqn):
    """与 _set_param_by_path 对称的路径式取属性。"""
    obj = model
    for p in fqn.split("."):
        obj = obj[int(p)] if p.isdigit() else getattr(obj, p)
    return obj


def _resolve_module(model, fqn):
    """按 FQN 取模块（不剥离末段，调用点传模块 FQN）。

    与 _get_attr_by_path 同语义——所有调用点（Phase A/B/C）传入的 fqn 均为
    模块完全限定名（如 `model.layers.0.self_attn`），而非参数 FQN，故不做
    `*path, _ =` 末段剥离。若剥离会错误返回父模块（decoder layer），导致
    后续 _shard_module_params 立即 AttributeError。
    """
    obj = model
    for p in fqn.split("."):
        obj = obj[int(p)] if p.isdigit() else getattr(obj, p)
    return obj


def _is_sdpa_attention(module) -> bool:
    # HF PretrainedConfig 是对象（非 dict），attn_implementation 存于
    # `config._attn_implementation` 属性；NeMo/Megatron config 可能是 dict。
    # 类名用子串匹配：LlamaSdpaAttention / Qwen2SdpaAttention 等均含 "SdpaAttention"。
    cfg = getattr(module, "config", None)
    impl = getattr(cfg, "_attn_implementation", None)
    if impl is None and isinstance(cfg, dict):
        impl = cfg.get("attn_implementation")
    cls_name = type(module).__name__
    return (impl == "sdpa") or ("SdpaAttention" in cls_name)


def _is_flex_attention(module) -> bool:
    cfg = getattr(module, "config", None)
    impl = getattr(cfg, "_attn_implementation", None)
    if impl is None and isinstance(cfg, dict):
        impl = cfg.get("attn_implementation")
    cls_name = type(module).__name__
    return (impl == "flex_attention") or ("FlexAttention" in cls_name)


def _is_hf_style_attention(module) -> bool:
    """判定是否 HF 标准注意力（forward(hidden_states,...)，Q/K/V 投影在 forward 内）。

    HF 的 LlamaSdpaAttention / Qwen2SdpaAttention / LlamaAttention 等把 q/k/v
    投影、RoPE、SDPA/FlexAttention 调用全部封在 forward 内，forward 首参为
    `hidden_states`（而非预切分的 q/k/v）。这类模块需走「原语拦截」wrapper
    （§4.4.2 `_wrap_hf_sdpa_for_cp`/`_wrap_hf_flex_for_cp`），不能复用 NeMo/Megatron 的 (q,k,v) wrapper。

    NeMo/Megatron 的 inner_attention 子模块 forward 取 (q,k,v,...)，且通常不直接
    持有 q_proj/k_proj/v_proj（投影在外层 attention 完成），走 (q,k,v) wrapper。
    """
    # 直接持有 q_proj/k_proj/v_proj → 投影在 forward 内 → HF 风格
    has_proj = (hasattr(module, "q_proj") and hasattr(module, "k_proj")
                and hasattr(module, "v_proj"))
    if not has_proj:
        return False
    # forward 首参为 hidden_states（HF 约定）
    try:
        import inspect
        sig = inspect.signature(module.forward)
        first_param = next(iter(sig.parameters.values()), None)
        return first_param is not None and first_param.name == "hidden_states"
    except (ValueError, TypeError):
        # 签名不可内省（C 扩展/已包装）→ 退化为类名判定
        cls_name = type(module).__name__
        return cls_name.endswith("Attention")


def detect_tied_weights(model) -> list[tuple[str, str]]:
    """检测模型中的 tied-weight 对（共享存储的参数）。

    返回 [(fqn_a, fqn_b)]，典型场景：embed_tokens.weight <-> lm_head.weight。
    从模型自身的 weight tying 配置读取（HF `tie_word_embeddings`）。

    PP constraint: 当启用 Pipeline Parallel 时，embed_tokens 和 lm_head
    通常位于不同的 PP stage（part）。此时 detect_tied_weights 在每个 part
    上独立调用，无法检测跨 stage 的 tied 对。对于 PP 场景，tied_pairs
    应由用户在 ShardingPlan 中显式声明（plan.tied_pairs），而非依赖
    detect_tied_weights 自动检测。
    """
    tied = []
    # HF 标准：model.tie_word_embeddings 时 embed_tokens 与 lm_head 共享
    if getattr(getattr(model, "config", None), "tie_word_embeddings", False):
        embed_fqn = lm_head_fqn = None
        # remove_duplicate=False：tied 参数在 named_parameters 默认去重下
        # 只出现一次，必须显式保留重复项才能发现两端 FQN（§12.3.7）。
        for name, _ in model.named_parameters(remove_duplicate=False):
            if name.endswith("embed_tokens.weight"):
                embed_fqn = name
            elif name.endswith("lm_head.weight"):
                lm_head_fqn = name
        if embed_fqn and lm_head_fqn:
            tied.append((embed_fqn, lm_head_fqn))
    return tied


def _broadcast_tied_param(model, tied_pair: tuple[str, str], mesh: DeviceMesh) -> None:
    """tied-weight 对本 rank 内共享存储（A 端存储为准，B 端共享）。

    实现校准（§12.4.3）：**不做跨 rank 广播**。tied 对（embed/lm_head）同为
    Shard(0) 分片，各 rank 的 local shard 承载不同 vocab 区间——把 rank0 的
    shard 广播给 rank1 会破坏 rank1 的分片。tied 语义要求同一 rank 内两端
    共享物理存储（梯度共享）；分片天然一致（同一 global 来源、同一 placement）。
    """
    fqn_a, fqn_b = tied_pair
    try:
        param_a = _get_attr_by_path(model, fqn_a)
        param_b = _get_attr_by_path(model, fqn_b)
    except AttributeError:
        return
    if param_a is None or param_b is None:
        return
    tensor_a = param_a.to_local() if isinstance(param_a, DTensor) else param_a.data
    # B 与 A 共享存储（tied weight 同一物理参数）
    if isinstance(param_b, DTensor):
        param_b._local_tensor = tensor_a
    else:
        param_b.data = tensor_a


def _replicate_tied_weights(model, mesh, tied_pairs=None):
    """Phase D：tied weights 本 rank 内共享存储。"""
    for tied_pair in (tied_pairs if tied_pairs is not None
                      else detect_tied_weights(model)):
        _broadcast_tied_param(model, tied_pair, mesh)


def _wrap_production_forward(
    module: nn.Module,
    boundary: "PrecompiledBoundary",
) -> None:
    """生产模式：纯 local tensor 计算 + 预编译边界通信。

    _local_params_context(module) 在 wrapping 之前已调用（Phase C 入口），
    参数已永久 unpack 为 plain local tensor。forward 内不再需要 context manager。
    """
    original_forward = module.forward

    def production_forward(*args, **kwargs):
        args, kwargs = boundary.redistribute_inputs(args, kwargs)
        outputs = original_forward(*args, **kwargs)
        outputs = boundary.redistribute_outputs(outputs)
        return outputs

    module.forward = production_forward
```

#### D-02：vocab-parallel embedding masked wrapper（production 专用）

**动机**：embed 边界在 production 下参数已解包为 local tensor——DTensor
dispatch 隐含的 vocab 范围 mask 逻辑随之丢失，HF 原生 `F.embedding` 收到
全局 token id 会索引越界（超出本 rank 的 `[V/tp, H]` shard）。

**契约**：`_apply_phase_c` 对 production 的 embed 边界（`nn.Embedding` +
weight 在 TP 上 `Shard(0)` + TP>1，由 `_is_vocab_parallel_embed` 判定）在
`_wrap_production_forward` 之前注入 `_wrap_vocab_parallel_embedding`
（Megatron 风格 masked embedding）：本地 vocab 区间 `[lo, hi)` 外的 token
置 0、索引减去偏移，输出乘 mask——天然形成 Partial 贡献，boundary 出口
`Partial→Shard(1)` 归约不变。validate 模式参数保持 DTensor，dispatch 自带
mask 语义，无需此 wrapper。UT：`test_dist_s5_vocab_embed.py`。

#### 4.4.2 CP Attention Wrapper: `_wrap_cp_inner_attention`

> **Call site**: Phase C, when `cp_size > 1` 且 `_resolve_inner_wrapper` 解析非 None（门控派生：`inner_target`/`inner_wrapper`/`_needs_cp_attn` 任一声明；`_needs_cp_attn` 由 `ShardingTemplate.needs_cp_attn` 模板置位）

CP attention's internal K/V all-gather cannot go in PrecompiledBoundary (it operates on forward-internal `q/k/v`, not module-input `hidden_states`). Phase C replaces the inner attention's forward.

**Call entry** (in `sharding_applier._apply_phase_c()`，module-level 函数，无 ShardingApplier 类）:

```python
# hyper_models/components/distributed/sharding_applier.py -- Phase C main flow (module-level)
def _apply_phase_c(model, plan, mesh, validate_mode, expert_mesh=None):
    """Phase C: 包装 forward（production/validate/moe/cp/vocab_embed 五路）。

    由 apply_sharding_plan 调用，boundary 在此处 per-module 构建。
    cp_mesh 从 mesh 按 mesh_dim_names 提取（"cp" 维存在时）。
    """
    mesh_dim_names = plan.mesh_dim_names
    cp_mesh = _get_cp_submesh(mesh, mesh_dim_names)  # None if no "cp" dim
    for module_fqn, spec in plan.modules.items():
        if not spec.is_boundary:
            continue
        module = _resolve_module(model, module_fqn)
        boundary = PrecompiledBoundary(spec, mesh, mesh_dim_names)
        _bind_input_indices(boundary, module)   # in_plan arg 绑定 forward 签名位置（§12.3.4）

        # Step 1: CP inner attention wrapper (BEFORE PrecompiledBoundary wrapping)
        # D-01''（对早期设计的证伪）：production 与 validate 注入**同一个**
        # all-gather wrapper。早期版本曾让 validate 跳过 wrapper、"靠 DTensor
        # dispatch 自动处理 K/V all-gather"——但 SDPA dispatch 对 CP Shard(1)
        # 的 K/V 不会 all-gather，会算成局部 attention（数值错误，见本节下文
        # 实现说明）。wrapper 入口容错 DTensor/local，区域内两模式逐指令一致。
        # 门控派生：不在此检查 _needs_cp_attn——cp_mesh 激活即调用，
        # direct=False 表示门控完全由 spec 声明派生（inner_target/
        # inner_wrapper/_needs_cp_attn 任一声明 → _resolve_inner_wrapper
        # 解析非 None 才注入；皆无声明则原样返回）
        if cp_mesh is not None and cp_mesh.size() > 1:
            _wrap_cp_inner_attention(module, cp_mesh, spec=spec, mesh=mesh,
                                     mesh_dim_names=mesh_dim_names, direct=False)

        # Step 2: Forward wrapping
        # local region 路（D-03'）：门控由 compute_fn 解析链派生（非 None 即
        # 走骨架）——用户 local_compute_fn / planner EP 注入意图 /
        # use_local_map 纯门控三来源统一解析，互不嵌套（§4.4.3）
        compute_fn = _resolve_local_compute_fn(
            module, spec, mesh, mesh_dim_names, expert_mesh)
        if compute_fn is not None:
            _wrap_local_region_forward(
                module, boundary, spec, mesh, mesh_dim_names,
                validate_mode=validate_mode, compute_fn=compute_fn)
        elif validate_mode:
            _wrap_validate_forward(module, boundary, spec, mesh, mesh_dim_names)
        else:
            # D-02: production vocab-parallel embedding masked wrapper（见 §4.4 D-02 小节）
            if _is_vocab_parallel_embed(module, spec, tp_mesh):
                _wrap_vocab_parallel_embedding(module, tp_mesh)
            _wrap_production_forward(module, boundary)
```

**Implementation** (reference: Titan `context_parallel.py` `apply_cp_to_forward()`):

```python
def _wrap_cp_inner_attention(attn_module, cp_mesh, *, spec=None, mesh=None,
                             mesh_dim_names=(), direct=True):
    """Inject CP-aware inner forward (compile-time replacement).

    direct=True 表示直接调用即显式意图（测试/手动接入路径）；
    _apply_phase_c 传 direct=False（门控完全由 spec 声明派生）。

    Executed BEFORE PrecompiledBoundary wrapping. CP wrapper at inner attention
    level, PrecompiledBoundary at module boundary level. Combined:
      PrecompiledBoundary.pre_forward -> CP inner attn -> PrecompiledBoundary.post_forward

    **双解析链 + 注册表（2026-07-21 定稿）**：解析（纯函数）与应用分离——
    `_resolve_inner_target`（链 1：位置）+ `_resolve_inner_wrapper`
    （链 2：行为），返回 None 即不注入（**门控派生**，无隐式标记改写）。

    链 1 `_resolve_inner_target`：
    0. spec.inner_target（用户显式：属性名/"self"，未命中 fail-fast）；
    1. inner_attention/attn/attention 属性；2. 类名含 SdpaAttention/以
    Attention 结尾；3. 持 q/k/v_proj 结构兜底。

    链 2 `_resolve_inner_wrapper`（优先级从高到低）：
    1. inner_wrapper 是 Callable → 全自定义（"custom"，整体接管）；
    2. inner_wrapper 是 str → CP_WRAPPER_REGISTRY 注册表查找（未知名
       fail-fast；target 缺失 fail-fast）；
    3. inner_target/_needs_cp_attn 声明 → 启发式 2×2 分派（target 缺失
       fail-fast）；
    4. 皆无 → None。

    **内置注册表 CP_WRAPPER_REGISTRY（四路，开放注册）**：
    - "sdpa_hf"：HF 标准风格（forward(hidden_states,...)，Q/K/V 投影在
      forward 内）→ 原语拦截 F.scaled_dot_product_attention，对入参 K/V
      沿 CP 维 all-gather，复用 HF 投影/RoPE/reshape；**发火检测**——
      未拦到调用即 RuntimeError（启发式误猜不再静默）；
    - "sdpa_qkv"：NeMo/Megatron 风格（forward(q,k,v,...)）→ 显式
      all-gather K/V + D-04 offset causal mask；
    - "flex_hf"/"flex_qkv"：FlexAttention 同构两路（block_mask 需按全局
      kv 长度构建）。
    用户可注册：CP_WRAPPER_REGISTRY["my_flash"] = my_fn →
    inner_wrapper="my_flash" 按名引用。

    **fail-fast（2026-07-21 修订）**：target 自动定位失败抛
    ValueError——缺失 K/V all-gather 是"数值错误但不报错"的静默失败
    （与 use_local_map 强制继承同一原则），用户需经 plan_overrides 指定
    `spec.inner_target`（属性名或 "self"）或提供 `spec.inner_wrapper`。

    **隐式显式化**：注入后 INFO 日志记录 target/wrapper/来源（启发式分派
    会提示可用 str 固定），并回写 spec._resolved_inner_wrapper 供内省。

    **梯度归约**：CP 维的参数梯度由 FSDP2 统一管理（CP 轴属于 FSDP2 reduce 组），
    本 wrapper 不做额外 CP 梯度通信；K/V all-gather 的 backward（reduce-scatter）使
    k_proj/v_proj 梯度跨 CP 聚合，与 FSDP2 的 reduce 组协同（见 03 §10.1 CP 因子）。
    """
    resolved = _resolve_inner_wrapper(
        attn_module, spec, cp_mesh, mesh, mesh_dim_names, direct=direct)
    if resolved is None:
        return  # 门控派生：无声明不注入
    name, target, apply_fn = resolved
    apply_fn()
    if spec is not None:
        spec._resolved_inner_wrapper = name   # 内省回写 + INFO 日志


def _cp_sdpa_call(orig_sdpa, cp_mesh, q, k, v, kwargs):
    """CP-aware SDPA：K/V all-gather + D-04 offset-aware causal mask。"""
    cp_dim = 2  # [B, N, S, H] 布局的序列维
    global_k, global_v = flex_cp_allgather(
        k.contiguous(), v.contiguous(), cp_dim, cp_mesh)
    if kwargs.get("is_causal") and cp_mesh.size() > 1:
        # D-04：按 CP 语义触发（不用 q_len ≠ kv_len 形状比较当代理）。CP 激活时
        # q 是本 rank 的 contiguous chunk、kv 是全量，torch 的 is_causal 在
        # q_len ≠ kv_len 时按左上角对齐（等价于假设 Q 从全局 0 开始），rank>0
        # chunk 掩码错误（G4）→ 替换为按本 rank Q 全局偏移 lo 的显式下三角
        # mask（rank0 lo=0 退化为标准 causal，行为一致）。性能注记：显式
        # attn_mask 使 SDPA flash backend 不可选（回退 mem_efficient/math），
        # CP+causal 路径的正确性优先于此。
        cp_rank = cp_mesh.get_local_rank()
        lo = cp_rank * q.shape[cp_dim]
        kwargs = dict(kwargs)
        kwargs.pop("is_causal")
        kwargs["attn_mask"] = _cp_offset_causal_mask(
            q.shape[cp_dim], global_k.shape[cp_dim], lo, q.device)
    return orig_sdpa(q, global_k, global_v, **kwargs)


def _wrap_sdpa_for_cp(inner_attn, cp_mesh):
    """NeMo/Megatron SDPA 路径：inner_attention.forward(q,k,v,...) → 显式 all-gather K/V。

    双模式共用：q/k/v 为 DTensor 时 unwrap（validate），输出按 q 的
    placements 重包装；local 输入透传（production）。
    不再依赖 DTensor dispatch all-gather K/V（PyTorch SDPA dispatch 对
    Shard(1) K/V 不会 all-gather，会算成局部 attention）——统一显式
    flex_cp_allgather：Q 保持本地（CP Shard(1)），K/V all-gather 为全局。
    """
    original_forward = inner_attn.forward
    # flex_cp_allgather 直接收 cp_mesh，内部用 cp_mesh.get_group() 复用 DeviceMesh
    # 已创建的 CP 通信组——不再依赖 dist._get_process_group_name 私有 API，也不
    # 再每次 new_group（避免 process group 泄露）。

    def cp_forward(q, k, v, **kwargs):
        was_dtensor = isinstance(q, DTensor)
        q_placements = tuple(q.placements) if was_dtensor else None
        mesh = q.device_mesh if was_dtensor else None
        ql, kl, vl = (t.to_local() if isinstance(t, DTensor) else t
                      for t in (q, k, v))
        out = _cp_sdpa_call(
            lambda *a, **kw: original_forward(*a, **kw),
            cp_mesh, ql, kl, vl, kwargs)
        if was_dtensor and isinstance(out, torch.Tensor):
            out = DTensor.from_local(out, mesh, q_placements)
        return out

    inner_attn.forward = cp_forward

def _wrap_flex_attn_for_cp(inner_attn, cp_mesh):
    """NeMo/Megatron FlexAttention 路径：inner_attention.forward(q,k,v,...) → 显式 all-gather K/V。

    flex_cp_allgather 见本节末定义（`hyper_models/components/distributed/cp_utils.py`，
    autograd 版——all-gather 前向 + reduce-scatter 语义反向）。
    约束同 _wrap_hf_flex_for_cp：block_mask 需按全局 kv 长度构建。
    """
    original_forward = inner_attn.forward

    def cp_forward(q, k, v, **kwargs):
        was_dtensor = isinstance(q, DTensor)
        q_placements = tuple(q.placements) if was_dtensor else None
        mesh = q.device_mesh if was_dtensor else None
        ql, kl, vl = (t.to_local() if isinstance(t, DTensor) else t
                      for t in (q, k, v))
        global_k, global_v = flex_cp_allgather(
            kl.contiguous(), vl.contiguous(), 2, cp_mesh)
        out = original_forward(ql, global_k, global_v, **kwargs)
        if was_dtensor and isinstance(out, torch.Tensor):
            out = DTensor.from_local(out, mesh, q_placements)
        return out

    inner_attn.forward = cp_forward


def _wrap_hf_sdpa_for_cp(inner_attn, cp_mesh, *, spec=None, mesh=None,
                         mesh_dim_names=()):
    """HF 标准 SDPA 路径：forward(hidden_states,...) → 原语拦截。

    HF 的 LlamaSdpaAttention/Qwen2SdpaAttention 等在 forward 内做 Q/K/V 投影、RoPE、
    reshape，并直接调 `F.scaled_dot_product_attention(q,k,v,...)`。策略：替换 forward
    为 (hidden_states,...) 签名的 CP 版本，内部临时把 `F.scaled_dot_product_attention`
    替换为 CP-aware 版本——对入参 K/V 沿 CP 维 all-gather 后再调原 SDPA。Q 不 gather
    （保持本地序列块）。原 forward 的投影/RoPE/reshape 全部复用，不重写。

    双模式共用（D-01''）：hidden_states 为 DTensor 时（validate）unwrap +
    临时解包模块参数（_temp_local_params），出口按 spec.out_src 声明重包装；
    local 输入透传（production）。原语拦截为临时全局函数替换（try/finally
    还原），非线程安全；单进程 SPMD 训练下安全（与 TorchTitan CP 实现一致）。
    """
    original_forward = inner_attn.forward
    orig_sdpa = F.scaled_dot_product_attention

    out_src_placements = None
    if spec is not None and spec.out_src:
        _named = next(iter(spec.out_src.values()))
        out_src_placements = tuple(resolve_placements(_named, mesh_dim_names))

    def cp_aware_sdpa(q, k, v, **kwargs):
        return _cp_sdpa_call(orig_sdpa, cp_mesh, q, k, v, kwargs)

    def cp_forward(hidden_states, *args, **kwargs):
        was_dtensor = isinstance(hidden_states, DTensor)
        hs = hidden_states.to_local() if was_dtensor else hidden_states
        F.scaled_dot_product_attention = cp_aware_sdpa
        try:
            if was_dtensor:
                with _temp_local_params(inner_attn):
                    out = original_forward(hs, *args, **kwargs)
            else:
                out = original_forward(hs, *args, **kwargs)
        finally:
            F.scaled_dot_product_attention = orig_sdpa
        if (was_dtensor and out_src_placements is not None
                and not isinstance(out, DTensor) and isinstance(out, torch.Tensor)):
            out = DTensor.from_local(out, mesh, out_src_placements)
        return out

    inner_attn.forward = cp_forward

def _wrap_hf_flex_for_cp(inner_attn, cp_mesh, *, spec=None, mesh=None,
                         mesh_dim_names=()):
    """HF 标准 FlexAttention 路径：forward(hidden_states,...) → 原语拦截 flex_attention。

    同 `_wrap_hf_sdpa_for_cp`，但拦截 `torch.nn.attention.flex_attention.flex_attention`。
    约束：score_mod/block_mask 随 kwargs 原样透传——CP 下 kv_len 从 S/cp 变为
    S，block_mask 必须按**全局 kv 长度**构建（在数据管道/模型侧按全量序列
    构造），否则形状/语义错位。wrapper 不校验此项。
    """
    original_forward = inner_attn.forward
    from torch.nn.attention.flex_attention import flex_attention as _orig_flex

    out_src_placements = None
    if spec is not None and spec.out_src:
        _named = next(iter(spec.out_src.values()))
        out_src_placements = tuple(resolve_placements(_named, mesh_dim_names))

    def cp_aware_flex(q, k, v, **kwargs):
        global_k, global_v = flex_cp_allgather(
            k.contiguous(), v.contiguous(), 2, cp_mesh)
        return _orig_flex(q, global_k, global_v, **kwargs)

    def cp_forward(hidden_states, *args, **kwargs):
        import torch.nn.attention.flex_attention as _flex_mod
        was_dtensor = isinstance(hidden_states, DTensor)
        hs = hidden_states.to_local() if was_dtensor else hidden_states
        _flex_mod.flex_attention = cp_aware_flex
        try:
            if was_dtensor:
                with _temp_local_params(inner_attn):
                    out = original_forward(hs, *args, **kwargs)
            else:
                out = original_forward(hs, *args, **kwargs)
        finally:
            _flex_mod.flex_attention = _orig_flex
        if (was_dtensor and out_src_placements is not None
                and not isinstance(out, DTensor) and isinstance(out, torch.Tensor)):
            out = DTensor.from_local(out, mesh, out_src_placements)
        return out

    inner_attn.forward = cp_forward


def flex_cp_allgather(k, v, cp_dim: int, cp_mesh):
    """All-gather K/V along CP dimension for context parallel attention
    （canonical 实现见 hyper_models/components/distributed/cp_utils.py）。

    Forward: all-gather K and V along cp_dim so each rank has full K/V
    （autograd.Function 实现）。
    Backward: reduce-scatter 语义——梯度跨 rank 求和后取本 rank chunk，
    由 _AllGatherAlongDim autograd.Function 显式实现（plain dist.all_gather
    无 autograd 核，§12.1）。

    Args:
        k: Key tensor, shape [B, N, S_local, H]
        v: Value tensor, shape [B, N, S_local, H]
        cp_dim: Dimension to gather along (typically seq dim, =2)
        cp_mesh: CP 维度的 DeviceMesh。通信组直接取 ``cp_mesh.get_group()``——
            该 group 在 DeviceMesh 构建时已创建并缓存，**此处不得再调
            ``dist.new_group``**（否则每次 forward 泄露一个 process group，
            且新建的全 world group 会忽略 CP 子集语义，导致通信错位）。
    Returns:
        (k_global, v_global): Full K/V tensors along cp_dim
    """
    cp_size = cp_mesh.size()
    if cp_size <= 1:
        return k, v
    group = cp_mesh.get_group()  # 复用 DeviceMesh 已创建的 CP 通信组，零泄露
    return (_AllGatherAlongDim.apply(k, cp_dim, group, cp_size),
            _AllGatherAlongDim.apply(v, cp_dim, group, cp_size))


def _resolve_inner_target(module, spec=None):
    """Locate inner attention sub-module within attention module.

    匹配顺序：
    0. 用户显式指定（spec.inner_target，plan_overrides 入口，2026-07-21 新增）：
       "self" 表示模块本身，否则按属性名取——属性不存在/无 forward 时
       fail-fast（拼写错误不能静默降级为无 CP）。
    1. 显式属性名 inner_attention / attn / attention（NeMo/Megatron 风格）。
    2. HF 标准实现：LlamaSdpaAttention / Qwen2SdpaAttention / LlamaAttention /
       Qwen2Attention 等——这些类本身即 inner attention（无嵌套 inner_attention
       属性），通过类名包含 "SdpaAttention"/"Attention" 子串判定。
    3. 结构判定：模块直接持有 q_proj/k_proj/v_proj 子模块，则视为 inner attention。
    """
    explicit = getattr(spec, "inner_target", None) if spec is not None else None
    if explicit:
        if explicit == "self":
            return module
        inner = getattr(module, explicit, None)
        if inner is not None and hasattr(inner, "forward"):
            return inner
        raise ValueError(f"spec.inner_target={explicit!r} 未命中——检查拼写")
    for name in ("inner_attention", "attn", "attention"):
        inner = getattr(module, name, None)
        if inner is not None and hasattr(inner, "forward"):
            return inner
    # HF 标准模型：attention 模块本身就是 inner attention
    cls_name = type(module).__name__
    if "SdpaAttention" in cls_name or cls_name.endswith("Attention"):
        return module
    # 结构兜底：直接持有 q/k/v 投影
    if (hasattr(module, "q_proj") and hasattr(module, "k_proj")
            and hasattr(module, "v_proj")):
        return module
    return None
```

**Design points**:
1. **Compile-time replacement**: `_wrap_cp_inner_attention` executes in Phase C, one-time (not runtime monkey-patch)
2. **Layered with PrecompiledBoundary**: CP wrapper at inner attention level, PrecompiledBoundary at module boundary level
3. **二分路由 by attention 架构**: HF 标准注意力（forward(hidden_states)）走「原语拦截」wrapper；NeMo/Megatron inner_attention（forward(q,k,v)）走 (q,k,v) wrapper。两条路径**统一显式 all-gather K/V**——SDPA 路径不再依赖 DTensor dispatch（PyTorch SDPA dispatch 对 Shard(1) K/V 不会 all-gather，会算成局部 attention）。
4. **原语拦截非线程安全**: HF 路径临时替换 `F.scaled_dot_product_attention`/`flex_attention`（try/finally 还原），单进程 SPMD 训练下安全；与 TorchTitan CP 实现一致。
5. **梯度归约由 FSDP2 统一管理**: CP 维参数梯度由 FSDP2 reduce 组统一管理（CP 轴属于 FSDP2 reduce 组），本 wrapper 不做额外 CP 梯度通信；K/V all-gather 的 backward（reduce-scatter）使 k_proj/v_proj 梯度跨 CP 聚合，与 FSDP2 协同（见 03 §10.1 CP 因子）。
6. **fail-fast + 双解析链 + 注册表（2026-07-21 定稿）**：target 自动定位失败抛 ValueError（静默缺失 K/V all-gather = 数值错误不报错）；用户经 plan_overrides 的 `spec.inner_target`（纯位置）/ `spec.inner_wrapper`（纯行为：`CP_WRAPPER_REGISTRY` 注册表名 str 显式固定，或自定义 callable 整体接管）。**门控派生**——声明不改写 `_needs_cp_attn`，`_resolve_inner_wrapper` 返回非 None 即注入；缺省按启发式 2×2 分派，结果经 INFO 日志与 `spec._resolved_inner_wrapper` 可观察、可用 str 固定（隐式 = 可命名/可观察/可固定/可报错的缺省值，非黑箱）。
7. **D-04 触发条件按 CP 语义而非形状比较（2026-07-21 修订）**：`_cp_sdpa_call` 中 `is_causal` 的显式 mask 替换条件为 `cp_mesh.size() > 1`——不用 `q_len ≠ kv_len` 当代理（GQA 差异在 head 维不影响序列维，但 cross-attention/KV-cache 的 q_len≠kv_len 与 CP 无关，形状推断会把 `lo = cp_rank × q_len` 的偏移语义用错）。CP 激活时该条件恒真（gather 后 kv_len = cp×q_len），rank0 `lo=0` 显式 mask 退化为标准 causal，行为一致。**性能注记**：显式 `attn_mask` 使 SDPA flash backend 不可选（回退 mem_efficient/math）——CP+causal 路径正确性优先于此，kernel 级优化（cuDNN backend / FlexAttention score_mod）留作后续。
8. **FlexAttention 的 block_mask 全局长度约束**：score_mod/block_mask 随 kwargs 原样透传——CP 下 kv_len 从 S/cp 变为 S，`block_mask` 必须按**全局 kv 长度**构建（在数据管道/模型侧按全量序列构造），否则形状/语义错位，wrapper 不校验此项（THD/varlen 场景应走 block_mask 而非 is_causal）。
9. **发火检测（2026-07-21 定稿）**：`sdpa_hf`/`flex_hf` 原语拦截路在 forward 完成后检查被拦截原语是否真被调用——一次未拦到说明启发式误猜（K/V 未 gather，继续运行是静默数值错误），立即 RuntimeError 并指引显式指定（`inner_wrapper='sdpa_qkv'` 或自定义 callable）。

本节双解析链/启发式判定/注册表/发火检测/D-04 触发条件的单进程覆盖见
`test_s3_inner_attn_detect.py`（S3.2）。

**自定义 inner wrapper 契约（`spec.inner_wrapper` 的 callable 形式；str 形式为注册表名，直接引用内置方案）**：

```python
def my_cp_wrapper(target_module, cp_mesh) -> None:
    \"\"\"就地替换 target_module.forward 使其满足 CP 语义。

    约定：
    - forward 在 local tensor 上执行；K/V 沿序列维 all-gather 用
      cp_utils.flex_cp_allgather（自带 backward reduce-scatter 核）；
    - 入口容错双模式：输入可能是 DTensor（validate）或 local tensor
      （production）——DTensor 时 to_local 计算、出口 from_local 重包装；
    - target_module：spec.inner_target 指定的子模块；未指定且自动定位失败时
      退化为边界模块本身。
    \"\"\"
    orig_forward = target_module.forward

    def cp_forward(q, k, v, **kwargs):
        from hyper_models.components.distributed.cp_utils import flex_cp_allgather
        global_k, global_v = flex_cp_allgather(k.contiguous(), v.contiguous(), 2, cp_mesh)
        return orig_forward(q, global_k, global_v, **kwargs)

    target_module.forward = cp_forward
```

适用场景：内置注册表四路覆盖不到的自研 attention——典型如内部直调
`flash_attn_varlen_func`（THD packed）的模块，原语拦截拦不到、(q,k,v)
签名约定也对不上。若自研模块能整理成 `forward(q, k, v, **kwargs)` 约定
（all-gather 后调原实现），可直接 `inner_wrapper="sdpa_qkv"` 复用内置
方案而无需自定义 callable。自定义 callable 只认 local tensor 时，可改为
给模块声明 `use_local_map`——骨架在模块入口把一切转 local，wrapper 只见
local tensor（validate 下 wrapper 收到 DTensor 会有一次性 WARNING 提示）。

#### 4.4.3 EP MoE Wrapper: `_wrap_local_region_forward`

> **重构注记（2026-07-21）**：原 `_wrap_moe_forward` 已轻量拆分为
> `_resolve_local_compute_fn`（**单一解析链**）+ `_wrap_local_region_forward`
> （与 MoE 无关的通用 local-region 骨架）。
>
> **派生门控（2026-07-21 二次重构）**：模块是否走骨架不再是存储的 bool，
> 而是解析链的结果——`_resolve_local_compute_fn` 返回非 None 即走骨架。
> 链上三个声明来源优先级固定、互不嵌套：
> 1. `spec.local_compute_fn`（用户自定义 compute_fn）；
> 2. `spec._ep_size > 0` 且 expert_mesh 可用（planner 记录的 TP-extend-EP
>    **注入意图**，与用户 fn 对等的显式一环）→ `_hf_native_ep_compute`；
> 3. `spec.use_local_map`（**纯门控**：模块自身 forward 即数据相关逻辑，
>    自给自足）→ `module.forward`；
> 4. 以上皆无 → None（普通模块，走 validate/production 路径）。
>
> 由此 `use_local_map` 回归单一含义（骨架 + 自身 forward，不再 latent
> 携带 EP 替换）；`local_compute_fn` 声明即生效，**不改写
> `use_local_map`**（早期版本在 planner 合并时隐式置位，已移除）。
> UT：`test_s4_local_compute_fn.py`（S4.6，解析链优先级 + 派生门控 +
> 双模式骨架注入）。

> **Call site**: Phase C, when `_resolve_local_compute_fn(...)` 返回非 None（派生门控）

EP MoE modules have internal all-to-all dispatch/combine that runs on local tensors.
`_wrap_local_region_forward` implements the Titan `LocalMapConfig` equivalent: PrecompiledBoundary manages
module I/O redistribution, local_map manages internal DTensor->local->DTensor conversion.

**Call entry**: Same as CP wrapper above (see `_apply_phase_c()` code in section 4.4.2).

**Implementation**:

```python
def _wrap_local_region_forward(module, boundary, spec, mesh, mesh_dim_names,
                               *, validate_mode=False, compute_fn=None):
    """通用 local-region forward wrapper（D-03'，原 _wrap_moe_forward 骨架）。

    结构：boundary 入口 → local region → 按声明 out_src 重包装 → boundary
    出口。适用于一切含数据相关逻辑（DTensor dispatch 无法表达，如 MoE
    all-to-all）的模块（spec.use_local_map=True 时由 _apply_phase_c 注入）。

    production：参数已 build 期永久 unpack，输入为 local（boundary 直通）；
    validate：输入为 DTensor → to_local → 临时解包参数（_temp_local_params）
    → local 计算 → 输出按声明 out_src from_local 重包装（out_src 对数据相关
    模块为声明式校验——all-to-all 的数据相关性使 placement 无法派生，这是
    本质限制）。两模式共用同一份 wrapper 代码（local_region 容错透传语义）。

    compute_fn：区域内实际执行的函数，缺省为模块自身 forward；MoE EP 由
    _resolve_local_compute_fn 注入（与原 forward 无关）。

    Wrapped forward flow:
      1. boundary.redistribute_inputs(args, kwargs)    # TP all-gather (entry)
      2. compute_fn (all-to-all on local tensors)       # EP dispatch/combine
      3. DTensor.from_local(output, mesh, out_src)      # local -> DTensor
      4. boundary.redistribute_outputs(output)          # TP reduce-scatter (exit)
      5. 最终出口恒为 local（out_plan 为空时 Step 3 的 from_local 包装也在此解包）
    """
    original_forward = module.forward
    if compute_fn is None:
        compute_fn = original_forward

    # out_src 为 dict[str, NamedPlacement]（per-arg）；单输出取唯一值
    out_src_placements = None
    if spec.out_src:
        _out_src_named = next(iter(spec.out_src.values()))
        out_src_placements = tuple(resolve_placements(_out_src_named, mesh_dim_names))

    @functools.wraps(original_forward)
    def local_region_forward(*args, **kwargs):
        # Step 1: PrecompiledBoundary entry（如 TP all-gather；identity 直通）
        args, kwargs = boundary.redistribute_inputs(
            args, kwargs, as_dtensor=validate_mode)

        # Step 2: local region —— 数据相关计算（如 EP dispatch/combine）在
        # local tensor 上执行；validate 下 unwrap 输入 + 临时解包 DTensor 参数
        if validate_mode:
            local_args = tuple(
                a.to_local() if isinstance(a, DTensor) else a for a in args)
            local_kwargs = {
                k: (v.to_local() if isinstance(v, DTensor) else v)
                for k, v in kwargs.items()
            }
            with _temp_local_params(module):
                output = compute_fn(*local_args, **local_kwargs)
        else:
            output = compute_fn(*args, **kwargs)

        # Step 3: local -> DTensor (restore DTensor metadata lost in all-to-all)
        if out_src_placements is not None and not isinstance(output, DTensor):
            output = DTensor.from_local(output, mesh, out_src_placements)

        # Step 4: PrecompiledBoundary exit（如 TP reduce-scatter）
        output = boundary.redistribute_outputs(
            output, as_dtensor_input=validate_mode)
        # 边界最终出口恒为 local（out_plan 为空时 Step 3 的 from_local
        # 包装也需要在此解包）
        if isinstance(output, DTensor):
            output = output.to_local()
        return output

    module.forward = local_region_forward
```

**Why EP needs explicit local->DTensor conversion**:

Standard TP/CP modules maintain DTensor propagation through forward (Colwise Shard(0) -> Shard(-1) output,
Rowwise Shard(1) -> Partial output). These placements are auto-derived by DTensor op rules.

EP modules execute all-to-all on local tensors inside forward, breaking DTensor propagation.
Hence `DTensor.from_local(output, mesh, out_src_placements)` explicitly restores DTensor metadata.

This is why `use_local_map=True` is exclusive to MoE templates -- only EP modules need
internal local-tensor communication with explicit DTensor recovery.


---



---

## 5. Validate 模式的 placement 校验

> **调用位置**: 时序树 ③.4.5.8 Phase C validate 分支 — `_wrap_validate_forward()`

### 5.1 核心校验：`out_src`

Validate 模式的核心是**用 DTensor 传播的运行结果，验证参数分片声明是否正确**。

```
DTensor 传播的输出 placement ⇔ 用户声明的 out_src
```

`out_src` 由 DTensor dispatch 规则决定（Colwise 输出 `Shard(-1)`，Rowwise 输出 `Partial`）。
如果参数 placement 声明错误（如把 Rowwise 错误声明为 Colwise），DTensor 传播会推导出不同的
placement，校验立即捕获。

### 5.2 `out_dst` 校验的冗余性分析

```
out_dst 校验: redistribute 后的 placement ≈ 声明的 out_dst
  → 依赖链: out_src 正确 + DTensor.redistribute() 正确（PyTorch 内部保证）

对于中间模块:
  out_dst 校验 ≈ 链式传播校验（下一个模块的 in_src 检查）
  → 冗余

对于末个模块（lm_head）:
  无下游模块做链式传播校验
  → 唯一检查，保留价值
```

**结论**：对于中间模块，`out_dst` 校验是冗余的——链式传播中 `A.out_dst == B.in_src` 提供
等价覆盖。仅对**末端模块**（`_is_terminal=True`）进行 `out_dst` 校验。

`_is_terminal` 由 `ShardingPlanner` 在链式传播时自动标记：如果某模块的 `out_dst` 不
被任何其他模块的 `in_src` 引用，则标记为 terminal。

### 5.3 校验 forward 实现

```python
class PlacementMismatchError(ValueError):
    """DTensor 传播结果与 ModuleShardingSpec 声明不一致。"""
    def __init__(self, module_name: str, expected, actual, stage: str):
        self.module_name = module_name
        self.expected = expected
        self.actual = actual
        self.stage = stage
        super().__init__(
            f"[{module_name}] {stage} placement mismatch:\n"
            f"  Expected (from ShardingConfig.{stage}): {expected}\n"
            f"  Actual   (from DTensor propagation):   {actual}\n"
            f"  → Check the ShardingConfig for this module."
        )


def _wrap_validate_forward(
    module: nn.Module,
    boundary: PrecompiledBoundary,
    spec: ModuleShardingSpec,
    mesh: DeviceMesh,
    mesh_dim_names: tuple[str, ...],
) -> None:
    """校验模式：DTensor 全程传播 → 校验 out_src + 可选的 out_dst（仅末端模块）。

    核心逻辑：
    - 参数保持 DTensor（不进 _local_params_context）
    - 所有 op 走 DTensor dispatch（__torch_function__）→ 自动传播 placement
      （自研 DTensor 前向-only，校验仅覆盖前向 placement 传播；backward 两
      模式同为 local autograd，梯度等价由 testing/grad_equiv.py 保证，§5.5）
    - out_src 校验：原生输出 vs 声明 → 验证参数分片正确性
    - out_dst 校验：仅 _is_terminal 模块 → 防御性检查
    """
    original_forward = module.forward
    module_name = type(module).__name__

    def validate_forward(*args, **kwargs):
        # Step 1: 输入 → DTensor
        args, kwargs = boundary.redistribute_inputs(
            args, kwargs, as_dtensor=True
        )

        # Step 2: 参数保持 DTensor，执行原始 forward
        # self.q_proj.weight 仍是 DTensor(placements=[Shard(0)])
        # → F.linear(DTensor(R), DTensor(S(0))) 触发 DTensor dispatch
        outputs = original_forward(*args, **kwargs)

        # Step 3: 【核心校验】out_src — DTensor 传播原生输出 vs 声明
        # 单/多输出统一走 _validate_outputs（多输出按 spec.out_names 映射
        # tuple 位置，缺省按声明 key 顺序）。
        if spec.out_src is not None:
            _validate_out_src(outputs, spec, mesh_dim_names, module_name)

        # Step 4: redistribute 到 out_dst
        outputs = boundary.redistribute_outputs(outputs, as_dtensor_input=True)

        # Step 5: 【防御性校验】out_dst — 仅末端模块
        # 中间模块的 out_dst 由链式传播校验（A.out_dst == B.in_src）覆盖
        if spec._is_terminal and spec.out_dst is not None:
            _validate_out_dst(outputs, spec, mesh_dim_names, module_name)

        if isinstance(outputs, DTensor):
            outputs = outputs.to_local()
        elif isinstance(outputs, (tuple, list)):
            outputs = tuple(
                t.to_local() if isinstance(t, DTensor) else t for t in outputs
            )
        return outputs

    module.forward = validate_forward


def _validate_out_src(outputs, spec, mesh_dim_names, module_name):
    _validate_outputs(outputs, spec, mesh_dim_names, module_name, "out_src")


def _validate_out_dst(outputs, spec, mesh_dim_names, module_name):
    _validate_outputs(outputs, spec, mesh_dim_names, module_name, "out_dst")


def _normalize_placements_ndim(placements, ndim):
    """Shard(-1) 等负维度按 tensor ndim 归一化（Shard(-1) == Shard(ndim-1)）。"""
    out = []
    for p in placements:
        if isinstance(p, Shard) and p.dim < 0:
            out.append(Shard(p.dim + ndim))
        else:
            out.append(p)
    return tuple(out)


def _validate_outputs(outputs, spec, mesh_dim_names, module_name, stage):
    """单/多输出的 placement 校验（out_src / out_dst 共用）。

    多输出按 spec.out_names（缺省按声明 key 顺序）映射到 tuple 位置；
    未返回/非 DTensor 的输出跳过。比较前对声明与实际 placement 做负维度归一化
    （§12.3.6）。
    """
    declared = getattr(spec, stage)
    if isinstance(outputs, (tuple, list)):
        out_names = getattr(spec, "out_names", None) or list(declared.keys())
        name_to_idx = {name: i for i, name in enumerate(out_names)}
        items = list(outputs)
    else:
        name_to_idx = {name: 0 for name in declared}
        items = [outputs]
    for out_name, expected_named in declared.items():
        idx = name_to_idx.get(out_name)
        if idx is None or idx >= len(items):
            continue  # 模块未返回该命名输出（如推理时省略 present_kv）
        tensor = items[idx]
        if not isinstance(tensor, DTensor):
            continue
        ndim = len(tensor.shape)
        expected = _normalize_placements_ndim(
            tuple(resolve_placements(expected_named, mesh_dim_names)), ndim)
        actual = _normalize_placements_ndim(tuple(tensor.placements), ndim)
        if expected != actual:
            suffix = f"[{out_name}]" if len(declared) > 1 else ""
            raise PlacementMismatchError(
                module_name, expected, actual, f"{stage}{suffix}"
            )
```

### 5.4 校验通过 vs 失败

#### 校验通过 ✅

```
self_attn, TP=4, SP=true:
  params: q_proj{Shard(0)}, o_proj{Shard(1)}
  out_src: {TP: Partial()}

DTensor 传播:
  q = F.linear(R, S(0)) → S(-1)
  o = F.linear(S(-1), S(1)) → Partial()
  → outputs.placements = [Partial()]

out_src 校验: expected=[Partial()] actual=[Partial()] → ✅
（out_dst 校验跳过：self_attn 不是末端模块）
```

#### 校验失败 ❌

```
错误: 把 o_proj 声明为 Shard(0)（应为 Shard(1)）

DTensor 传播:
  o = F.linear(S(-1), S(0)) → 推导出与 Shard(1) 不同的 placement

out_src 校验:
  expected=[Shard(-1)] actual=[Partial()] → ❌ PlacementMismatchError!
  "out_src placement mismatch:
    Expected: [Shard(-1)]
    Actual:   [Partial()]
    → Colwise/Rowwise declaration error in params."
```

### 5.5 正确性保证总结

```
① out_src 校验（每个模块，不可替代）:
   DTensor 传播原生输出 ≈ 声明的 out_src
   → 验证：参数 placement 声明（Colwise/Rowwise）是否正确
   → 捕获：参数分片声明错误

② 链式传播校验（中间模块，覆盖 out_dst）:
   模块 A.out_dst ≈ 模块 B.in_src
   → 验证：相邻模块的 placement 契约对齐
   → 等效覆盖了中间模块的 out_dst 校验

③ out_dst 校验（末端模块，防御性）:
   redistribute 后输出 ≈ 声明的 out_dst
   → 仅对 lm_head 等无下游的模块执行
   → 捕获：末端通信计划错误

production local-tensor forward == validation DTensor forward, if and only if:
1. Same ModuleShardingSpec
2. PrecompiledBoundary communication == DTensor dispatch selected communication
3. local tensor shares storage with DTensor local tensor

参数梯度同步不在 validate forward 覆盖范围内——但注意：自研 DTensor 是
**前向-only**（§1.0），不存在 "DTensor backward" 对照组：production
（FSDP/tp_grad_info 旁路）与 validate（local autograd 直出）的 backward
**均为 local tensor 路径**，双模式梯度等价直接逐参数比较（§12.5）：

```python
# hyper_models/components/distributed/testing/grad_equiv.py（canonical）
def run_one_step(model, input_ids, labels, vocab_size):
    """单步 forward+backward，返回 {param_fqn: grad}（两模式共用同一函数）。"""

def assert_grad_equivalence(prod_grads, val_grads, *, rtol=1e-3, atol=1e-5):
    """双模式梯度逐参数 assert_close（跳过两侧均缺失的参数）。

    TP-Shard 参数：两模式梯度天然是 local shard，逐 rank 相等（免同步）；
    TP-Replicate 参数：两模式梯度同为 Partial 贡献，逐 rank 相等——与单卡
    参考梯度比较前需先经 tp_grad_info 旁路 all-reduce（真实路径由 FSDP2
    fork 的 all_reduce_grad 完成，本模块提供模拟）。
    """

def simulate_tp_replicate_grad_sync(grad, tp_group):
    """模拟 tp_grad_info 旁路：TP-Replicate 参数梯度的 TP all-reduce。"""
```
```

---



---

## 6. 并行策略组合：TP × CP × EP × DP

> **调用位置**: 时序树 ③.4.5.7 — ShardingPlan 中的多 mesh 维度组合

### 6.1 核心原则：CP/EP 与 TP 统一走 PrecompiledBoundary

CP（Context Parallel）和 EP（Expert Parallel）**不是**独立于 TP 的特殊通信机制。**三者共享相同的 DTensor placement + PrecompiledBoundary 范式**：

```
统一范式: in_src -> redistribute -> in_dst -> compute(local) -> out_src -> redistribute -> out_dst

Mesh 维度: {TP: placement, CP: placement, EP: placement}
  - 每个维度独立声明 placement
  - PrecompiledBoundary 按维度编译 RedistOp 序列
  - 运行时: DTensor.redistribute() 在对应 mesh 维度上执行通信
```

**与 Titan `ShardingConfig` 的对齐**：Hyper-Parallel 的 `ModuleShardingSpec` 等价于 Titan 的 `ShardingConfig`，两者都用 `in_src/in_dst/out_src/out_dst` 表达通信契约，用 `params/state_shardings` 表达参数分片，用 `local_map`（Titan）= `_local_params_context`（Hyper-Parallel）表达"DTensor->local 计算->DTensor"的零开销模式。

参考实现：
- Titan `protocols/sharding.py` — `ShardingConfig` 数据模型（与 `ModuleShardingSpec` 同构）
- Titan `models/common/moe_sharding.py` — MoE 的 `ShardingConfig` 填充（EP+TP 组合的完整 placement 声明）
- Titan `models/common/decoder_sharding.py` — dense 层的 `ShardingConfig` 填充（CP+TP 组合）
- AutoModel `components/moe/parallelizer.py` — HF 模型上使能 CP/EP 的 `apply_cp()`/`apply_ep()` 函数

### 6.2 总体架构：4D 并行拓扑

```
+--------------------------------------------------------------+
|                      4D Parallel Topology                     |
|                                                              |
|  FSDP2 manages:  dp_shard, dp_replicate                      |
|    - Applied LAST (after DTensor sharding)                    |
|    - All-gather / reduce-scatter parameters                   |
|                                                              |
|  DTensor + PrecompiledBoundary manages:  TP, CP, EP           |
|    - TP: Shard hidden dim (Shard(0)/Shard(1)/Partial)        |
|    - CP: Shard sequence dim (Shard(1) on CP axis)             |
|    - EP: Shard expert dim (Shard(0) on EP axis)              |
|    - Applied FIRST (before FSDP2)                             |
|    - ALL redistribution goes through PrecompiledBoundary      |
|                                                              |
|  PP (optional): Pipeline Parallel                             |
|    - Inter-layer model splitting                              |
+--------------------------------------------------------------+
```

**Mesh 维度顺序（由内到外）**：`pp -> dp_replicate -> dp_shard -> ep -> cp -> tp`

**完整 Mesh 构建示例** — TP=4, CP=2, EP=4, DP=8 (world_size=256)：

> ⚠️ **D-10 前概念示意**：本示例（及 §6.4.1 / §6.4.2 / §6.4.4 的 EP 拓扑描述——
> EP 编入主 mesh 轴、`moe_mesh = DeviceMesh(("ep",))`、main_mesh 含 "ep" 轴）
> 为 D-10 定稿前的旧口径，仅作概念示意。EP 拓扑以 **§6.4.8（D-10：TP-extend-EP）
> 与 06 §4.5.1** 为准：主 mesh 无 EP 轴、expert mesh 在 apply 期派生、
> MeshContext 无 `moe_mesh` 字段。

```python
# main_mesh: DTensor 直接管理的维度为 TP+CP+EP；此处把 dp_shard 与 cp 合并
# 为单维 "dp_shard_cp" 仅为某些 mesh 构造器的简化写法（dp_shard 与 cp 共享
# 同一 process group 维度时）。规范做法应拆为独立 "dp_shard" 与 "cp" 两维，
# 使 DTensor 管理的轴严格为 TP/CP/EP，DP 由 FSDP2 在 DTensor 之外管理
#（见 §6.2 "DTensor + PrecompiledBoundary manages: TP, CP, EP"）。
# 若实际 mesh 采用 "dp_shard_cp" 合并维，需在 ShardingPlanner 的
# mesh_dim_names 解析中显式声明该维既承载 DP 也承载 CP，且 spec.params 中
# 该维的 placement 由 DP（FSDP）与 CP（DTensor）分工：参数 CP 维恒 Replicate()，
# DP 维由 FSDP layout 接管。本示例以下按合并维展示，生产建议拆分。
main_mesh = DeviceMesh("cuda", shape=(8, 4, 2, 4),
                       mesh_dim_names=("dp_shard_cp", "ep", "cp", "tp"))
# moe_mesh: EP 子 mesh（用于 EP 维度的 communication group）
moe_mesh = DeviceMesh("cuda", shape=(4,), mesh_dim_names=("ep",))

# placement 解析（tensor 维度索引按 mesh_dim_names 顺序）:
# {TP: Shard(0), CP: Replicate(), EP: Shard(0)}
# -> placements = (Shard(0), Replicate(), Shard(0))  对应 ("tp", "cp", "ep")
```

### 6.3 Context Parallel (CP)：序列维度的 TP

#### 6.3.1 CP 的本质

CP 的本质是**在 CP mesh 维度上沿序列维度（dim 1）做 Shard**，与 TP 的 Sequence Parallel（SP）共享完全相同的 placement 语义。差异仅在于**作用在哪个 mesh 维度**：

```
TP 的 SP:  {TP: Shard(1)}  -> 在 TP rank 组内沿序列切分
CP:        {CP: Shard(1)}  -> 在 CP rank 组内沿序列切分
两者组合:  {TP: Shard(1), CP: Shard(1)} -> 同时沿 TP 和 CP 切分序列
```

**为什么需要 CP？** — TP 的 SP 受到 `num_attention_heads % tp_size == 0` 约束，不能无限增大。CP 提供了**正交的**序列切分维度，不受 attention heads 数量的限制。两者组合实现 `序列切分总数 = tp_size * cp_size`。

参考 Titan `decoder_sharding.py` 的 `dense_sequence_parallel_placement()`：当 SP+CP 同时启用时，`SpmdLayout({TP: Shard(1), CP: Shard(1)})` —— 两个维度都沿序列维 Shard。

#### 6.3.2 CP 的 PrecompiledBoundary 表达：TP all-gather，CP 维保持 Shard

CP 在 PrecompiledBoundary 中的表达**与 TP 的 SP 不对称**——TP 维做 `Shard(1) <-> Replicate()` 的 all-gather/reduce-scatter，而 **CP 维的 in_dst 保持 `Shard(1)`**（不做 boundary 层 all-gather）。CP 的 K/V all-gather 延后到 attention 内部由 inner attention wrapper 完成（见 §6.3.3）：

```python
# self_attn, TP=4, CP=2, SP=true
ModuleShardingSpec(
    params={
        "q_proj.weight": {TP: Shard(0), CP: Replicate()},   # CP 不切参数
        "k_proj.weight": {TP: Shard(0), CP: Replicate()},
        "v_proj.weight": {TP: Shard(0), CP: Replicate()},
        "o_proj.weight": {TP: Shard(1), CP: Replicate()},
    },
    in_src={
        "hidden_states": {TP: Shard(1), CP: Shard(1)},      # SP+CP 序列分片
    },
    in_dst={
        # 只 all-gather TP；CP 维保持 Shard(1)，K/V all-gather 交给
        # inner attention wrapper（needs_cp_attn=True，§4.4.2/§6.3.3）。
        "hidden_states": {TP: Replicate(), CP: Shard(1)},
    },
    out_src={TP: Partial(), CP: Shard(1)},                # 本地 Q 段输出 → CP Shard(1)；o_proj Rowwise -> TP Partial
    out_dst={TP: Shard(1), CP: Shard(1)},                    # reduce-scatter(TP) -> SP+CP；CP 维 identity
)

# PrecompiledBoundary 编译结果:
# in_plan:  [RedistOp("hidden_states", TP Shard(1)->TP Replicate, "all_gather")]
#           <- CP 维 Shard(1)->Shard(1) identity，跳过
# out_plan: [RedistOp("output", TP Partial->TP Shard(1), "reduce_scatter")]
#           <- CP 维 out_src=Shard(1) 与 out_dst=Shard(1) identity，跳过；
#              CP 序列维 all-gather 仅发生在 attention 内部 K/V（§4.4.2），
#              boundary 层 CP 维不做出口通信
```

**Norm 模块在 CP 下的行为**：

```python
# post_attention_layernorm, TP=4, CP=2
ModuleShardingSpec(
    params={"weight": {TP: Replicate(), CP: Replicate()}},  # norm weight 全复制
    in_src={"hidden_states": {TP: Shard(1), CP: Shard(1)}},
    in_dst={"hidden_states": {TP: Shard(1), CP: Shard(1)}},  # identity - 零通信
    out_src={TP: Shard(1), CP: Shard(1)},
    out_dst={TP: Shard(1), CP: Shard(1)},                    # identity
)
# -> PrecompiledBoundary: in_plan=[], out_plan=[] (零通信)
#    RMSNorm 是逐元素操作，可在分片序列上直接算
```

#### 6.3.3 CP Attention 内部的通信：K/V all-gather

Attention 模块的 PrecompiledBoundary 只对 `hidden_states` 做 **TP 维的 `Shard->Replicate` all-gather**；**CP 维保持 `Shard(1)`**——进入 forward 时 `hidden_states` 在 CP 维仍是序列分片。这样 CP 的 K/V all-gather 职责完全交给 inner attention wrapper，避免 boundary 层与 inner attention 层重复 all-gather CP 维。

在 forward **内部**，SDPA 需要额外的 CP 通信——K/V 的 all-gather。这与 TP 不同：

- TP: `hidden_states [B, S, H]` all-gather 后，Q/K/V 投影和 SDPA 都在完整序列上进行，无额外通信
- CP: `hidden_states [B, S/tp, H]`（TP 已 gather、CP 仍分片）进入 forward 后，CP 需要在 **SDPA 内部** all-gather K/V（因为从 CP 维度看，每个 rank 的 K/V 只覆盖了部分序列的 attention keys/values）

**这个 K/V all-gather 是否应该由 PrecompiledBoundary 管理？**

不能。原因：
1. PrecompiledBoundary 操作的是**模块入口/出口的 tensor**（`hidden_states` / `output`），不是 forward 内部的中间产物（`q/k/v`）
2. K/V all-gather 的模式与 PrecompiledBoundary 的 `in_src -> in_dst -> compute -> out_src -> out_dst` 范式不匹配——它是 attention 内部的细粒度通信

**Hyper-Parallel 的解决方案**：在 `ShardingApplier` 的 Phase C 中，为 attention 模块注入一个**CP-aware inner attention forward**。这个替换是编译期确定的（模板化生成），不是运行时 monkey-patch：

```python
# _wrap_cp_inner_attention 的 canonical 实现见 §4.4.2：
#   inner_attn = _resolve_inner_target(attn_module)
#   if _is_sdpa_attention(inner_attn):
#       _wrap_sdpa_for_cp(inner_attn, cp_mesh)
# 作用于 inner attention 子模块（而非整个 attention 模块），本节不再重复实现。
```

这个 CP attention wrapper 也是**编译期生成**的（从 `ModuleShardingSpec` 和 `cp_mesh` 推导），与 PrecompiledBoundary 的编译期哲学一致——只是作用在 inner attention 级别（更细粒度），而非模块边界级别。

#### 6.3.4 CP 的数据管道集成

CP 的数据分片发生在数据管道阶段——序列维度的 tensors（input_ids, labels, position_ids）在进入模型前就沿 CP mesh 切分好：

```python
# 参考 Titan context_parallel.py 的 prepare_context_parallel_input()
# 参考 AutoModel cp_utils.py 的 make_cp_batch_and_ctx()

def shard_batch_for_cp(batch, cp_mesh):
    """将 batch 中的序列维度 tensors 沿 CP mesh 切分。

    canonical 实现（05 单一实现，02/03 仅引用契约不重复实现）。
    与 02 collater 产出的真实 THD 契约对齐（放弃自创的 cu_seqlens）：
      - input_ids/labels/position_ids: [B, S] int64
      - seq_lens:        [B, max_num_packs] int64，每行是该样本内各 pack
                         子序列的实际长度，-1000 哨兵填充变长子序列数。
      - seq_lens_padded: [B, max_num_packs] int64，各 pack 子序列含
                         separator/padding 的长度，-1000 哨兵填充。
      - qkv_format: "thd"（透传，本函数不修改）

    CP 切分策略：按 token 区间 [cp_rank*chunk, (cp_rank+1)*chunk) 对
    input_ids/labels/position_ids 切片；对 seq_lens/seq_lens_padded 按
    CP rank 重算——遍历每个样本的 pack 累计偏移，找出与本 rank token 区间
    相交的 pack，截断到本地区间并平移到本地坐标系。输出仍含 qkv_format="thd"。
    """
    cp_size = cp_mesh.size()
    if cp_size <= 1:
        return batch

    cp_rank = cp_mesh.get_local_rank()
    seq_len = batch["input_ids"].shape[1]
    # 序列长度必须能整除 2*cp_size（load balancing 要求）；不足则 pad。
    # 注（G5）：2*cp 约束源自 zigzag/ring 负载均衡方案；本设计采用 all-gather
    # K/V + contiguous chunk（D-01'' 已否决 ring），各 rank Q chunk 等长、
    # FLOPs 天然均衡，该约束冗余但无害——保留实现仅为兼容（cp_utils.py G5）。
    pad_len = (-seq_len) % (cp_size * 2)
    chunk = (seq_len + pad_len) // cp_size
    lo = cp_rank * chunk
    hi = lo + chunk
    slc = slice(lo, hi)

    # 先对序列维 tensors 做 CP 对齐 padding（pad_len>0 时），保证各 rank 切出
    # 等长 chunk。否则最后一个 rank 的 v[..., slc] 因 tensor 只有 seq_len 元素，
    # 会被静默截短为 seq_len-lo，导致 CP 各 rank chunk 不等长 → 通信错位。
    # pad 值: input_ids/attention_mask->0, labels->-100(忽略), position_ids->递增。
    _PAD_VALUE = {
        "labels": -100,
        "input_ids": 0,
        "attention_mask": 0,
    }
    padded = dict(batch)
    if pad_len > 0:
        for k, v in batch.items():
            if k == "qkv_format" or not isinstance(v, torch.Tensor) or v.ndim < 1:
                continue
            if k in ("seq_lens", "seq_lens_padded"):
                continue  # 单独重算，不 pad
            pad_val = _PAD_VALUE.get(k, 0)
            if k == "position_ids":
                # position_ids 递增 pad：接续末值继续递增
                last = v[..., -1:].to(torch.long)
                inc = torch.arange(1, pad_len + 1, device=v.device,
                                   dtype=v.dtype).expand_as(v[..., :1])
                inc = inc.reshape(*([1] * (v.ndim - 1)), pad_len)
                inc = inc.expand(*v.shape[:-1], pad_len) + last
                pad_block = inc
            else:
                shape = list(v.shape)
                shape[-1] = pad_len
                pad_block = torch.full(shape, pad_val, dtype=v.dtype, device=v.device)
            padded[k] = torch.cat([v, pad_block], dim=-1)

    # 普通序列维 tensors 直接按 CP rank 切片（已 pad，各 rank 等长）
    out = {}
    for k, v in padded.items():
        if k in ("seq_lens", "seq_lens_padded"):
            continue  # 单独重算
        if k == "qkv_format":
            out[k] = v  # 透传字符串
        elif isinstance(v, torch.Tensor) and v.ndim >= 1:
            out[k] = v[..., slc]
        else:
            out[k] = v

    # seq_lens / seq_lens_padded 按 CP 分片重算（保留 -1000 哨兵语义）
    if "seq_lens" in batch and "seq_lens_padded" in batch:
        out["seq_lens"], out["seq_lens_padded"] = _shard_seq_lens_for_cp(
            batch["seq_lens"], batch["seq_lens_padded"],
            cp_rank=cp_rank, chunk=chunk,
        )
    return out


def _shard_seq_lens_for_cp(
    seq_lens: torch.Tensor,
    seq_lens_padded: torch.Tensor,
    *,
    cp_rank: int,
    chunk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """对 seq_lens/seq_lens_padded 按 CP 分片重算（与 02 collater 真实契约对齐）。

    输入:
      seq_lens:        [B, max_num_packs] int64，-1000 哨兵填充变长子序列数。
      seq_lens_padded: [B, max_num_packs] int64，同上哨兵填充。
      cp_rank / chunk: 本 rank 负责 token 区间 [cp_rank*chunk, (cp_rank+1)*chunk)。

    输出: (local_seq_lens, local_seq_lens_padded)
      形状 [B, max_local_packs] int64，-1000 哨兵填充。max_local_packs 取
      batch 内各样本落在本 rank 的 pack 数最大值。

    语义: 遍历每个样本的 pack 累计偏移（按 seq_lens_padded 累加），对每个 pack:
      - 完全在 [lo, hi) 内: 本地 seq_lens/seq_lens_padded 原样保留
      - 跨越 lo 或 hi 边界: 截断到 [lo, hi) 区间，按截断后的实际/含 padding
        长度重算（separator token 若被截断则不计入 local seq_lens）
      - 完全在区间外: 跳过
    截断后的长度平移到本地坐标系（local offset = global offset - lo）。
    """
    B, K = seq_lens.shape
    lo = cp_rank * chunk
    hi = lo + chunk
    device = seq_lens.device
    SENTINEL = -1000

    local_lens_b: list[list[int]] = []
    local_lens_padded_b: list[list[int]] = []
    max_local_packs = 0
    for b in range(B):
        row_lens = seq_lens[b].tolist()
        row_padded = seq_lens_padded[b].tolist()
        local_lens: list[int] = []
        local_padded: list[int] = []
        offset = 0  # 全局 token 偏移
        for raw_len, raw_pad in zip(row_lens, row_padded):
            if raw_len == SENTINEL:
                break  # 哨兵之后无 pack
            pack_start = offset
            pack_end = offset + raw_pad  # padded 覆盖 separator
            offset = pack_end
            # 求与 [lo, hi) 的交集
            inter_start = max(pack_start, lo)
            inter_end = min(pack_end, hi)
            if inter_start >= inter_end:
                continue  # 无交集
            # 实际 token（不含 separator）落在区间内的长度
            actual_start = max(pack_start, lo)
            actual_end = min(pack_start + raw_len, hi)
            local_actual = max(actual_end - actual_start, 0)
            local_pad = inter_end - inter_start
            if local_actual > 0 or local_pad > 0:
                local_lens.append(local_actual)
                local_padded.append(local_pad)
        local_lens_b.append(local_lens)
        local_lens_padded_b.append(local_padded)
        max_local_packs = max(max_local_packs, len(local_lens))

    if max_local_packs == 0:
        max_local_packs = 1  # 防止空 tensor

    out_lens = torch.full((B, max_local_packs), SENTINEL,
                          dtype=seq_lens.dtype, device=device)
    out_padded = torch.full((B, max_local_packs), SENTINEL,
                            dtype=seq_lens_padded.dtype, device=device)
    for b in range(B):
        n = len(local_lens_b[b])
        if n > 0:
            out_lens[b, :n] = torch.tensor(local_lens_b[b],
                                            dtype=seq_lens.dtype, device=device)
            out_padded[b, :n] = torch.tensor(local_lens_padded_b[b],
                                              dtype=seq_lens_padded.dtype,
                                              device=device)
    return out_lens, out_padded
```

### 6.4 Expert Parallel (EP)：Expert 维度的 Shard + local_map

#### 6.4.1 EP 的本质

> ⚠️ **D-10 前概念示意**：§6.4.1 / §6.4.2 / §6.4.4 中 EP 作为独立 mesh 轴
> （主 mesh 含 "ep"、`moe_mesh = DeviceMesh(("ep",))`）的描述为 D-10 定稿前
> 旧口径，placement 语义本身仍然成立，但 mesh 拓扑以 **§6.4.8（D-10：
> TP-extend-EP）与 06 §4.5.1** 为准——主 mesh 无 EP 轴、expert mesh 在
> apply 期派生、MeshContext 无 `moe_mesh` 字段。

EP 的本质是**在 EP mesh 维度上沿 expert 维度（dim 0）做 Shard**。与 TP 的 Colwise Shard(0) 共享完全相同的 placement 语义：

```python
# TP Colwise: weight [H_out, H_in] -> Shard(0) on TP -> [H_out/tp, H_in]
# EP Expert:  weight [n_experts, H_out, H_in] -> Shard(0) on EP -> [n_experts/ep, H_out, H_in]
# EP+TP 组合: weight [n_experts, H_out, H_in]
#   -> {EP: Shard(0)} -> [n_experts/ep, H_out, H_in]
#   -> {TP: Shard(0)} -> [n_experts/ep, H_out/tp, H_in]
```

#### 6.4.2 EP 的 PrecompiledBoundary 表达 + local_map

参考 Titan `moe_sharding.py` 的设计：MoE 模块的 `ShardingConfig` 包含完整的 `in_src/in_dst/out_src/out_dst`（与 dense 模块一致），同时通过 `local_map=LocalMapConfig(...)` 标记 forward 需要在 `DTensor->local->DTensor` 模式下运行。

Hyper-Parallel 的等价设计：

```python
# MoE 模块（moe_mlp）, TP=4, EP=4, SP=true
ModuleShardingSpec(
    params={
        # Router/gate: 全复制
        "gate.weight": {TP: Replicate(), EP: Replicate()},
        # Routed experts: EP Shard(0) + TP colwise/rowwise（D-08：3D [E,out,in]
        # 权重的 TP 维按 ndim 平移——colwise=Shard(1)、rowwise=Shard(2)）
        "experts.w1.weight": {EP: Shard(0), TP: Shard(1)},  # colwise on 3D
        "experts.w2.weight": {EP: Shard(0), TP: Shard(2)},  # rowwise on 3D
        "experts.w3.weight": {EP: Shard(0), TP: Shard(1)},  # colwise on 3D
        # Shared experts (optional): 无 EP（仅 TP，2D 标准 colwise/rowwise）
        "shared_experts.w1.weight": {EP: Replicate(), TP: Shard(0)},
        "shared_experts.w2.weight": {EP: Replicate(), TP: Shard(1)},
        "shared_experts.w3.weight": {EP: Replicate(), TP: Shard(0)},
    },
    # 入口: SP 序列分片，EP 维度 Replicate（所有 EP rank 看到相同 tokens）
    # 修订 D-06：CP 维全程 Shard(1)（moe_mlp 是 pointwise per-token，CP 无需
    # boundary 通信；对齐 §3.5 canonical 模板表）。
    in_src={
        "x_BLD": {TP: Shard(1), CP: Shard(1), EP: Replicate()},
    },
    in_dst={
        # all-gather TP（gate 和 shared experts 需要完整 hidden_states）
        # EP 保持 Replicate（gate 需要全量 tokens 做路由决策）
        "x_BLD": {TP: Replicate(), CP: Shard(1), EP: Replicate()},
    },
    # 出口: Experts 计算在 local_map 内部完成，输出为 Partial（Rowwise）
    out_src={TP: Partial(), CP: Shard(1), EP: Replicate()},
    out_dst={
        # reduce-scatter -> SP 分片
        TP: Shard(1),
        CP: Shard(1),
        EP: Replicate(),
    },
    # 标记: 此模块 forward 内部需要 local_map
    # (all-to-all dispatch/combine 在 local tensor 上执行)
    use_local_map=True,
)

# PrecompiledBoundary 编译结果:
# in_plan:  [RedistOp("x_BLD", TP Shard(1)->TP Replicate, "all_gather")]
#           <- CP 维 Shard(1)->Shard(1) identity 跳过；EP 维 Replicate->Replicate 跳过
# out_plan: [RedistOp("output", TP Partial->TP Shard(1), "reduce_scatter")]
#           <- CP/EP 维度均 identity 跳过
```

**local_map 的语义**（参考 Titan `LocalMapConfig`）：

> **规范实现见 §4.4.3 `_wrap_local_region_forward(module, boundary, spec, mesh,
> mesh_dim_names, *, validate_mode=False, compute_fn=None)`（canonical 签名）**。
> 本节仅描述 local_map 语义，不重复实现。`_apply_phase_c`（§4.4.2）即按该签名调用。

local_map 在 `use_local_map=True` 时的 forward 行为：

1. `boundary.redistribute_inputs(args, kwargs)` — PrecompiledBoundary 入口 redistribution（TP all-gather）
2. `original_forward(*args, **kwargs)` — All-to-all dispatch/combine 在 local tensor 上执行（参数已由 `_local_params_context` 在 build 期永久 unpack）
3. `DTensor.from_local(output, mesh, out_src_placements, run_check=False)` — local output 包装回 DTensor
4. `boundary.redistribute_outputs(output)` — PrecompiledBoundary 出口 redistribution（TP reduce-scatter）

**关键**：All-to-All dispatch/combine 发生在 local tensor 上（参数已在 build 期通过 `_local_params_context()` 永久 unpack），不在 PrecompiledBoundary 中。PrecompiledBoundary 只管理**模块入口/出口**的 deterministic redistribution（TP 的 all-gather/reduce-scatter），不管理模块内部的 data-dependent 通信（哪些 token 去哪个 expert 取决于 router 输出，是运行时确定的）。

> **⚠️ 对 HF 单卡 MoE 脚本的含义（2026-07-20 调研确认）**：`_wrap_local_region_forward`
> **不注入** all-to-all——EP 下 MoE 模块必须是 EP-aware 实现（forward 内部自带
> dispatch/combine + `init_token_dispatcher` 钩子）。这与 NeMo Automodel、
> TorchTitan/spmd_types 的结论一致：NeMo `apply_ep` 只做"expert 权重 Shard(0) +
> 挂 token dispatcher"，且硬性要求自研 MoE 模型实现（HF checkpoint 经
> state_dict_adapter 转 grouped 格式），原生 HF 单卡 MoE 脚本不支持直接跑 EP；
> spmd_types/Titan 同样由模型侧 `AllToAllTokenDispatcher` 显式调用通信原语。
> **解除方案见 §6.4.7（D-09）**：planner 堆叠元数据 + Phase A 参数堆叠 +
> wrapper 注入 EP 计算路径，HF 原生 MoE 在 EP>1 下用户零改动可用；
> 仅当模型结构超出 D-09 v1 范围（非 SwiGLU/带 bias 等）时，才需要在模型适配层
> 替换为 EP-aware 实现。

这与 Titan `moe_sharding.py` 的 `local_map` 模式完全一致：`ShardingConfig.local_map=LocalMapConfig(in_grad_placements=(...))`。

#### 6.4.3 EP 参数分片与 token dispatcher 初始化归属

**设计决策**：不保留 `ExpertParallel(ParallelStyle)` 子类（§1 已声明删除旧 `ParallelStyle` 子类，
二者矛盾）。EP 参数分片统一由 `_shard_module_params` 按 `spec.params` 中的 `{EP: Shard(0)}`
placement 处理——与 TP/CP 参数同路径，无需独立 `_apply` 入口。EP token dispatcher
（DeepEP/UCCL-EP 后端）的初始化**不在 ShardingApplier 层**，而由 fsdp2/parallelizer 侧
在 `apply_sharding_plan` 返回后调用，保持 ShardingApplier 只管"参数 DTensor 化 + forward 包装"：

```python
# hyper_models/components/distributed/parallelizer.py（fsdp2/parallelizer 侧，非 ShardingApplier）
# apply_sharding_plan 之后：
# ⚠️ 规划未实现：hyper_models/components/distributed 当前无 init_ep_token_dispatchers /
# init_token_dispatcher 实现。D-09/D-10 后 HF 原生 MoE 由 wrapper 注入
# _hf_native_ep_compute（§6.4.7/§6.4.8），无需此注入；本小节仅为自研
# EP-aware 模块（forward 自带 a2a）预留的模型侧初始化契约。
def init_ep_token_dispatchers(model, ep_mesh):
    """EP token dispatcher 初始化（DeepEP/UCCL-EP 后端）。

    在 apply_sharding_plan 之后、fully_shard 之前调用（位于 apply_model_infrastructure 中）。
    遍历模型中所有
    持有 init_token_dispatcher 的 MoE 模块，注入 EP mesh。
    """
    for module in model.modules():
        if hasattr(module, "init_token_dispatcher"):
            module.init_token_dispatcher(ep_mesh=ep_mesh)
```

**与 §4.1 的关系**：§4.1 的 Phase A 仅调 `_shard_module_params`（参数 DTensor 化），
不调 `ExpertParallel._apply`——EP 参数分片已由 `spec.params` 的 `{EP: Shard(0)}`
placement 在 `_shard_module_params` 内统一完成。token dispatcher 初始化由
parallelizer 侧单独调用，时序在 `apply_sharding_plan` 之后。

#### 6.4.4 EP+TP 组合的 Mesh 拓扑

```
world_size=32, tp=4, ep=4 -> dp=2

main_mesh: DeviceMesh(shape=(2, 4, 4),
                       names=("dp_shard", "ep", "tp"))
moe_mesh:  DeviceMesh(shape=(4,), names=("ep",))

参数视角 - expert weight [n_experts, H_inter, H]:
  EP Shard(0) on ep -> [n_experts/ep, H_inter, H]
  再 TP Shard(0) on tp      -> [n_experts/ep, H_inter/tp, H]

通信视角:
  EP 通信: all-to-all dispatch/combine（local tensor，参数已 build 期永久 unpack）
  TP 通信: all-gather / reduce-scatter（PrecompiledBoundary 入口/出口）
  两者正交 - EP 和 TP 使用不同的 process groups
```

#### 6.4.5 MoE ShardingTemplate 扩充

`moe_gate` / `moe_mlp` 两个模板的 **canonical 定义见 §3.5 "Complete Template
Enumeration"**（含完整 TP+CP+EP 三维声明与 D-06 CP=Shard(1) 修订），
`sharding_config.py` 的 `TEMPLATES` 为唯一实现来源，本节不再重复字面量。

要点回顾：
- `moe_gate`：router weight/bias 全复制；入口 all-gather TP；出口
  `EP: Replicate() → Shard(0)` redistribute 到 EP 分片。
- `moe_mlp`：expert 参数 `{EP: Shard(0)}` + TP 按 D-08 ndim 感知
  （3D batched 布局 colwise=Shard(1)/rowwise=Shard(2)）；CP 维全程
  `Shard(1)`（D-06，pointwise per-token 无需 CP boundary 通信）；
  `use_local_map=True` 走 local-region wrapper（§4.4.3）。
- D-10 TP-extend-EP 路径的契约修订（SP-in identity、expert 无 TP 键）
  在 planner `_mark_hf_native_moe` 中运行时应用（§6.4.8），模板字面量不变。

#### 6.4.6 MoE 模块的 ShardingPlanner 6-Phase 集成

```
Phase 1 (ParameterClassifier):
  "model.layers.*.mlp.experts.*.weight" -> MOE_EXPERT  (命名规则: *.experts.*)
  "model.layers.*.mlp.gate.weight"      -> MOE_GATE     (命名规则: *gate*.weight)
  "model.layers.*.mlp.shared_experts.*" -> SHARED_EXPERT (命名规则: *shared_experts*)

Phase 2 (BoundaryGrouper):
  expert 参数(MOE_EXPERT) -> 聚合到 mlp 边界
  gate 参数(MOE_GATE)     -> 聚合到同一边界
  # gate + experts + shared_experts 共享同一个 MoE mlp 边界

Phase 3 (SemanticRoleInference):
  "model.layers.*.mlp" + 含 MOE_EXPERT 角色 -> boundary_type="moe_mlp"
  "model.layers.*.mlp" + 不含 MOE_EXPERT    -> boundary_type="mlp"

Phase 4 (TemplateLookup):
  TEMPLATES["moe_mlp"] -> 自动填充 TP+EP params + I/O 契约

Phase 5 (ChainPropagate):
  标准流程 -- MoE mlp 的 in_src/out_dst 与前后模块对齐
  注意: MoE 内部 all-to-all 不影响链式传播

Phase 6 (SpecialHandler):
  - gated_delta: 自定义 placement（跳过标准模板）
  - ⚠️ 规划未实现：GroupedExpertsDeepEP（init_token_dispatcher 代码注入）、
    GroupedExpertsTE（skip DTensor wrapping）、shared_expert_gate（额外的
    gate projection）三个 handler 当前不在 SPECIAL_HANDLERS 注册表中——
    注册表仅 gated_delta_tp_shard；D-09/D-10 后 HF 原生 MoE 走
    _hf_native_ep_compute 注入路径，DeepEP 融合 dispatcher 留作后续。

gated_delta Phase 6 代码骨架（SSM/Mamba 类模块的 in_proj/A_log/dt 参数）：

```python
SPECIAL_HANDLERS["gated_delta_tp_shard"] = _shard_gated_delta

def _shard_gated_delta(module, param_name, mesh):
    """gated_delta 模块自定义 TP 分片骨架（SSM/Mamba 类模块）。

    按 SSM head 结构切分而非标准 colwise/rowwise。骨架实现：结构识别与
    标准 Shard(0) 回退；head 对齐的精细切分（in_proj/A_log/dt 按 head 维切）
    留待具体模型接入时补全。
    """
    param = getattr(module, param_name, None)
    if param is None:
        return
    sharded = distribute_tensor(param.data, mesh, [Shard(0)])
    module.register_parameter(param_name, nn.Parameter(sharded))
```
```

#### 6.4.7 HF 原生 MoE 的 EP 直通（D-09，2026-07-20 新增）

> 目标：**HF 单卡 MoE 脚本（per-expert Linear 列表 + 无 all_to_all 的 forward）
> 在 EP>1 下用户零改动可用**。all_to_all 由 `_wrap_local_region_forward` 注入的 EP 计算
> 路径完成，不要求模块持有 `init_token_dispatcher`（§6.4.2 ⚠️ 注释的解除方案）。
> 调研依据：veomni 的「stacked 参数 + kernel 内 a2a + checkpoint converter」
> 三层机制——本设计与其同构，但堆叠在内存中 apply 期完成，无需 codegen 与
> checkpoint 转换。
>
> **实现状态（2026-07-20）**：`_ep_stack` 堆叠链路已实现——planner
> `_mark_hf_native_moe` + 数字段守卫、`_stack_moe_experts`、双路径 a2a
> （`_EPAllToAllUneven`/`_EPAllToAllPadded`）、router adapter。
> **注意（语义修订）**：初版 D-09 的「EP 组同 TP 坐标 + 边界 all-gather +
> `{TP: Shard(1|2), EP: Shard(0)}`」compute 路径已被 **D-10 TP-extend-EP**
> （§6.4.8）取代——HF 原生 MoE 在 ep>1 时恒走 TP-extend-EP 路径
> （ep_size = 扩展 EP 组大小 + SP-in identity 边界 + 全 dense 区域重分区
> (edp, ep) + expert 权重仅 expert 维切分）。
> 本节保留作为堆叠机制（`_ep_stack`/`_stack_moe_experts`/router adapter）的
> 设计记录；用例：`test_s5_hf_native_moe.py`（10 例，堆叠与契约）+
> `test_dist_s6_hf_native_moe.py`（a2a pad 路径 fwd/bwd），TP-extend-EP
> 端到端见 `test_dist_s6_ep_extend.py`。

##### 问题

HF 原生 MoE（Qwen3-MoE/DeepSeek/Mixtral/GLM）的三个特征使现有路径失效：

1. 参数是 per-expert Linear 列表（`mlp.experts.{i}.gate_proj.weight`，2D）——
   模板对 2D expert 权重给 `{EP: Shard(0)}` 会把 `H_out` 维切碎（语义错误，
   `_moe_expert_tp_placement` docstring 已标注"per-expert 2D 布局 EP 语义不成立"）；
2. forward 逐 expert 循环，不含 all_to_all，EP 下计算错误（每 rank 只有
   `E/ep` 个 expert 的权重却按全部 E 个路由）；
3. 无 `init_token_dispatcher` 钩子——§6.4.3 的外部注入路径无从挂载。

##### 总体方案：三层各改一点，语义在 wrapper 闭环

```
Planner (Phase 4 后处理)          Phase A 前置                    Phase C wrapper
────────────────────────          ──────────────                  ───────────────
检测 per-expert 参数        →     _stack_moe_experts:       →    _wrap_local_region_forward 分支:
→ spec.params 换成 stacked 条目    per-expert → torch.stack        spec._ep_stack 非空且 ep>1
  (ndim=3, EP Shard(0))            → [E, ...] holder module         → compute_fn =
→ spec._ep_stack 记录源路径        → 再按 stacked 条目               _hf_native_ep_compute
→ spec._moe_router 选 adapter        distribute_tensor 分片          (all_to_all 在这里)
```

ep=1（mesh 无 ep 轴）时不做任何标记，走原路径（TP-only 下 per-expert 2D
分片语义本来就正确）；pre-stacked MoE（`experts.w1 [E, ...]` 3D，如
TinyMoEMLP、自研 EP-aware 模块）同样走原路径——两条老路径完全不受影响。

##### D-09a：Planner 标记（`_mark_hf_native_moe`，Phase 4 后处理）

对 `boundary_type == "moe_mlp"` 的边界，检查组内 MOE_EXPERT 参数名是否命中
per-expert 模式 `experts.<数字>.<proj>.weight`：

```python
# spec 新增内部字段（与 use_local_map 同族的结构标记）
spec._ep_stack: Dict[str, List[str]] = {}   # stacked 相对路径 → 源参数相对路径（按 expert idx 排序）
spec._moe_router: str = "default"           # router adapter 名（按 arch 查注册表）

# spec.params 改写（对每组 per-expert 参数）：
#   源: experts.0.gate_proj.weight, experts.1.gate_proj.weight, ...  (逐条删除)
#   目标: experts.gate_proj → {TP: Shard(1), CP: Replicate(), EP: Shard(0)}
#   （ndim=3 语义——colwise proj: gate/up/w1/w3 切 H_out=Shard(1)；
#     rowwise proj: down/w2 切 H_in=Shard(2)，即 D-08 规则的直接复用）
```

placement 推导不新增规则：stacked ndim=3 恰好落入 D-08 的
`_moe_expert_tp_placement(ndim>=3)` 分支，planner 只需把 ndim 按
"per-expert ndim + 1" 传入。**D-10 定稿后实际写入的 stacked 条目为
`{CP: Replicate(), EP: Shard(0)}`（无 TP 键、无第二轴，§6.4.8）**——
上述 TP 键写法为 D-09 初版契约，此处保留作设计记录。

##### D-09b：Phase A 前置堆叠（`_stack_moe_experts`，sharding/apply.py）

在 `_shard_module_params` 之前执行（apply_sharding_plan Phase A 内 per-module
前置钩子）：

```python
def _stack_moe_experts(module, ep_stack: Dict[str, List[str]]) -> None:
    """per-expert 参数 → stacked 3D 参数（值保持精确相等，stack 即 concat）。

    - 按源路径取 weight → torch.stack(dim=0)，注册到替换后的 experts holder；
    - 原 ModuleList 整体替换（显存释放）；v1 断言无 bias、源参数 shape 一致；
    - meta tensor 路径同样适用（concat meta，零显存推导）。
    """
    # 例: ep_stack = {"experts.gate_proj": ["experts.0.gate_proj.weight", ...]}
    #   → holder = _StackedExperts(gate_proj=Parameter(stacked [E, I, H]), ...)
    #   → module.experts = holder（替换 ModuleList）
```

堆叠后 `_shard_module_params` 按 stacked 条目正常 `distribute_tensor`
（EP Shard(0) → 每 rank `[E/ep, ...]`；TP 维按 D-08）。

##### D-09c：`_wrap_local_region_forward` 注入 EP 计算（核心）

重构为「边界骨架 + compute_fn」两路共用（validate 的 DTensor unwrap/rewrap
逻辑不变，compute_fn 在 local region 内调用）。compute_fn 不由
`_wrap_local_region_forward` 内部判断，而由 Phase C 的**单一解析链**
`_resolve_local_compute_fn` 派生（§4.4.3；返回非 None 即走骨架——
派生门控）：

```python
# _resolve_local_compute_fn 链环 2（EP 注入意图，sharding_applier.py）：
# 用户 spec.local_compute_fn 优先（链环 1），use_local_map 纯门控兜底（链环 3）
if spec._ep_size > 0 and expert_mesh is not None:   # D-10 注入意图
    router_fn = MOE_ROUTER_ADAPTERS.get(
        spec._moe_router, MOE_ROUTER_ADAPTERS["default"])
    compute_fn = functools.partial(
        _hf_native_ep_compute, module,
        router_fn=router_fn,
        ep_group=expert_mesh.get_group("ep"),       # 扩展 EP 组（a2a 通信域）
        tp_group=tp_group)                          # shared_experts Partial 归约用
else:
    compute_fn = module.forward          # 原路径：EP-aware 模块（自带 all_to_all）
# 之后与原实现相同：boundary 入口 → local region(compute_fn) → 出口重包装/解包
```

`_hf_native_ep_compute`（新文件 `ep_utils.py`）：

```python
def _hf_native_ep_compute(module, hidden_states, *, router_fn, ep_mesh):
    """HF 原生 MoE 的 EP 前向（production/validate 共用，全 local tensor）。

    输入 hidden [B, S, H]（boundary 已 TP all-gather → 全序列/CP chunk）；
    输出 [B, S, H] TP-Partial（rowwise 部分和，交 boundary 出口 reduce-scatter）。
    """
    # 1. 路由（gate 全复制，同 TP/CP 坐标的 EP rank 数据相同 → 路由结果一致）
    topk_idx, topk_w = router_fn(module, hidden_states)      # [T, K]
    # 2. 展开 (T,K) → (T*K,)：dest_rank = expert_idx // e_local
    #    按 (dest_rank, expert_idx) 排序 → send_tokens [N_send, H]、send_counts[ep]
    # 3. counts 交换：all_to_all_single（等长 [ep]，gloo/NCCL 通用）
    # 4. token 交换：_ep_all_to_all（后端分派——NCCL/HCCL 不等长 a2a，
    #    gloo pad-to-max a2a_single，见下方「后端分派」）
    # 5. 本地 expert 计算（按 recv 排序段切分，SwiGLU）：
    #    h_i = silu(x @ w1[i].T) * (x @ w3[i].T);  y_i = h_i @ w2[i].T
    #    TP 下 w1/w3 [E_l, I/tp, H]、w2 [E_l, H, I/tp] → y 为 Partial（H 部分和）
    # 6. 逆交换回源 rank → 逆排序 → index_add 按 topk_w 加权聚合
    # 7. shared_experts（若存在）：原模块本地计算，相加（同为 Partial）
    # 8. return y.view(B, S, H)   —— TP-Partial，boundary 出口 reduce-scatter
```

##### 后端分派：不等长 a2a（NCCL/HCCL 生产）+ pad-to-max（gloo 测试）

实测结论（2026-07-20，本机 gloo）：**gloo 支持等长 `all_to_all_single`，不支持
不等长 `all_to_all`（list 版）**；NCCL/HCCL 两者都支持。因此 a2a 原语按后端
分派——生产（NCCL/HCCL）走不等长 a2a 零填充开销，测试（gloo/CPU）走
pad-to-max + `all_to_all_single`：

```python
def _ep_all_to_all(x, send_counts, recv_counts, group):
    """EP token 交换的统一入口（autograd 可导）。

    send_counts/recv_counts: [ep_size] 各 dest/src rank 的 token 数
    （counts 交换用等长 all_to_all_single，各后端通用）。
    后端分派：NCCL/HCCL → 不等长 all_to_all（按 counts 切分，零填充）；
    其他（gloo 测试路径）→ pad-to-max + all_to_all_single。
    """
    if _backend_supports_uneven_a2a(group):      # nccl / hccl
        return _EPAllToAllUneven.apply(x, send_counts, recv_counts, group)
    return _EPAllToAllPadded.apply(x, send_counts, recv_counts, group)


class _EPAllToAllUneven(torch.autograd.Function):
    """不等长 all_to_all（NCCL/HCCL 生产路径）：按 send/recv counts 切分。

    forward:  split(x, send_counts) → dist.all_to_all(out_list, in_list) → concat
    backward: 交换 send/recv counts 再做一次不等长 all_to_all（a2a 自逆）。
    """

    @staticmethod
    def forward(ctx, x, send_counts, recv_counts, group):
        ctx.send_counts, ctx.recv_counts, ctx.group = send_counts, recv_counts, group
        out = torch.empty(sum(recv_counts), *x.shape[1:], dtype=x.dtype, device=x.device)
        dist.all_to_all(list(out.split(recv_counts)),
                        list(x.split(send_counts)), group=group)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        grad = _EPAllToAllUneven.apply(
            grad_output, ctx.recv_counts, ctx.send_counts, ctx.group)
        return grad, None, None, None


class _EPAllToAllPadded(torch.autograd.Function):
    """pad-to-max + all_to_all_single（gloo 测试路径；NCCL 可用但有带宽浪费）。

    forward:  各 dest chunk pad 到 max(send_counts) → a2a_single → 按 recv_counts unpad
    backward: 按 recv_counts pad → a2a_single（等长自逆）→ 按 send_counts unpad
    """
```

两路径数值语义一致（pad 只加不参与计算的填充行，unpad 后逐位相等），
dispatch/combine 的上层逻辑（排序/分组/index_add）不感知后端差异。
`_backend_supports_uneven_a2a` 按 `dist.get_backend(group)` 判定
（`"nccl"`/`"hccl"` → True）。

##### Router adapter 注册表（ep_utils.py）

路由语义是模型相关的（softmax/sigmoid、top-k 归一化、scaling factor），
expert MLP 结构是统一的（SwiGLU）——前者做成注册表，后者写死：

```python
# {adapter_name: (module, hidden) -> (topk_idx [T,K] int64, topk_w [T,K] float)}
# planner 按 arch 名选 adapter（未注册落 "default"）。arch 名来自
# _get_architecture（config.architectures 小写去后缀，如 "qwen3moe"）；
# 带下划线别名覆盖 config.model_type 路径（如 "qwen3_moe"）。
MOE_ROUTER_ADAPTERS = {
    "default": _softmax_topk_router,      # softmax → topk → 按和归一化（Mixtral/Qwen3 语义）
    "qwen3moe": _topk_router_module,      # gate 为 TopKRouter 模块（HF 2025 重构后）：
    "qwen3_moe": _topk_router_module,     #   forward 直返 (logits, scores [T,K], indices [T,K])
    "mixtral": _topk_router_module,       #   ——取后两个；返回不符时 TypeError 提示换 adapter
    "deepseekv3": _sigmoid_group_router,  # sigmoid + e_score_correction_bias
    "deepseek_v3": _sigmoid_group_router, #   + group-limited topk（n_group/topk_group，
    "glm4moe": _sigmoid_group_router,     #   n_group 缺省或 ≤1 时跳过）+ 可选归一化
    "glm4_moe": _sigmoid_group_router,    #   + routed_scaling_factor
                                          #   （与 HF route_tokens_to_experts 逐步一致）
}

def _softmax_topk_router(module, hidden_states):
    """default adapter：softmax → topk → 按和归一化（Mixtral/Qwen3 语义）。

    top_k 来源：config.num_experts_per_tok / config.top_k / module.top_k（缺省 2）；
    归一化开关：config.norm_topk_prob（缺省 True）。gate/router 属性均不存在
    时 AttributeError 并提示注册自定义 adapter。
    """
    gate = getattr(module, "gate", None) or getattr(module, "router", None)
    cfg = getattr(module, "config", None)
    logits = gate(hidden_states)
    logits = logits.view(-1, logits.shape[-1])
    top_k = (getattr(cfg, "num_experts_per_tok", None)
             or getattr(cfg, "top_k", None)
             or getattr(module, "top_k", 2))
    weights = logits.softmax(-1)
    topk_w, topk_idx = weights.topk(int(top_k), dim=-1)
    if getattr(cfg, "norm_topk_prob", True):
        topk_w = topk_w / topk_w.sum(-1, keepdim=True).clamp_min(1e-20)
    return topk_idx, topk_w
```

`_sigmoid_group_router` 的参数来源为 module 自身属性优先
（n_group/topk_group/top_k/norm_topk_prob/routed_scaling_factor），回落
module.config；选择分 = sigmoid 分 + correction bias，权重分 = 无 bias 的
原始 sigmoid 分（gather 回 topk 位）。planner 按
`arch if arch in MOE_ROUTER_ADAPTERS else "default"` 写入
`spec._moe_router`；注册表开放——用户可注册自定义 adapter
（`MOE_ROUTER_ADAPTERS["my_moe"] = fn` 后 arch 名即命中），或以
`local_compute_fn` 整体替换 EP 计算（§4.4.3/§8.6）。

##### 梯度语义

- a2a 经 `_ep_all_to_all` 后端分派（`_EPAllToAllUneven` / `_EPAllToAllPadded`）
  autograd.Function，反向为反向 a2a（不等长交换 split sizes / 等长自逆）；
  sort/index_add/加权聚合均为标准 autograd；
- stacked 参数梯度落在 local shard（与普通 DTensor 参数一致，交 FSDP 同步）；
- router（Replicate）在各 EP rank 上输入相同 → 梯度相同，由 FSDP/DP 平均；
- 双模式梯度等价性与现有一致（两模式 backward 同为 local autograd，§12.5）。

##### v1 范围与限制

| 项 | 说明 |
|---|---|
| expert 结构 | SwiGLU 三矩阵（gate/up/down 或 w1/w2/w3），无 bias；其余结构报 `NotImplementedError` 并回退提示用 EP-aware 模块 |
| token capacity | 无上限（不丢 token）；容量/丢 token 策略留后续 |
| aux loss / load balance | 不在本层（归 loss 路径；router logits 已由 adapter 可得，后续易加） |
| a2a 通信 | 后端分派：NCCL/HCCL 不等长 a2a（零填充）；gloo 走 pad-to-max `all_to_all_single`（最坏 ep× 带宽浪费，仅测试路径） |
| 整除约束 | E % ep_size == 0（`validate_model_compatibility` 已校验） |
| CP 组合 | 不变——a2a 在 CP 本地 chunk 内进行（x_BLD 契约 CP Shard(1)） |

##### 测试计划

| 层 | 用例 |
|---|---|
| 单进程 | 堆叠 handler（stacked 值 == 原 per-expert 值、原参数移除）；planner stacked keys + ndim=3 placements + `_ep_stack` 元数据；router adapter 数值；不等长 a2a 的 split/拼接逻辑（纯 tensor 操作） |
| 分布式 | `_ep_all_to_all` pad 路径 fwd/bwd 对拍（2 进程 gloo）；uneven 路径 collective 在 NCCL/HCCL 环境验证（gloo 不支持 list 版 a2a）；HF-native toy MoE（experts.0..3 独立 Linear + gate + softmax top-2）**TP=2×EP=2 双模式 e2e vs 单卡参考**；EP=2-only e2e；梯度等价 rtol=1e-3 |
| 回归 | 现有 TinyMoEMLP（pre-stacked）与全部 TP/CP 用例不受影响 |

#### 6.4.8 TP-extend-EP：EP 通信域跨 TP 组（D-10，2026-07-20 设计定稿；与 MindSpeed「TP 扩展 EP」/ Megatron etp=1+ep 跨 TP 同构）

> **定稿语义**：**`ep_size` 即扩展 EP 组大小**（a2a 通信域；无单独 etp
> 配置）。扩展 EP 组 = 全 dense 区域（dp_replicate × dp_cp × tp）flatten
> 后**连续的 ep_size 个 rank**——tp 是最内层轴，因此 EP 组先跨完整个
> TP 组、再向相邻 dp/cp rank 扩展（"TP 扩展 EP"）。expert 权重**仅在
> expert 维 Shard(0)**：扩展 EP 组每个 rank 持 `num_experts/ep_size` 个
> **完整** expert，无 hidden 维第二轴切分，因此计算流中**不存在**
> all_gather/reduce_scatter 对。
>
> **实现状态（2026-07-21）**：planner `_mark_hf_native_moe` 生成
> `{EP: Shard(0)}` 契约（无 TP 键、无第二轴），支持两种 expert 布局——
> per-expert 2D（旧版 HF/自研，`_ep_stack` 堆叠）与 **batched 3D**
> （HF 2025 重构后 `experts.gate_up_proj [E,2I,H]`，天生 stacked 无需
> 堆叠，D-11）；`_expert_mesh_layout`（全 dense 区域 → (edp, ep) 纯
> rank 映射）+ `_build_expert_mesh`；`_hf_native_ep_compute`（router →
> a2a → 本地 SwiGLU（分离/fused 双分支）→ a2a）；router adapter 注册表
> （qwen3moe/mixtral TopKRouter 模块、deepseekv3/glm4moe sigmoid+
> correction bias+group-limited topk）；tp_grad_info 的 expert 梯度
> Shard 标记。用例 `test_s5_hf_native_moe.py`（10 例）+
> `test_s6_ep_extend.py`（6 例）+ `test_dist_s6_ep_extend.py`
> （per-expert 与 batched 两布局各一组 mesh (dp=4,tp=2) ep=4 双模式 e2e）。

##### 框架对照（2026-07-20 源码调研）

| 框架 | TP-extend-EP 支持情况 |
|---|---|
| **MindSpeed**（canonical） | **官方「TP 扩展 EP」特性**：EP 通信域合并 TP 维，expert 仅 expert 维切分 |
| **Megatron-LM** | **可达同构**：`expert_tensor_parallel_size=1` + `expert_model_parallel_size` 跨 TP——expert rank generator 独立网格（parallel_state.py:781-800）使 EP 组 = flatten 连续 ep_size 个 rank（含 TP rank）；expert MLP 的 `tp_group=pg_collection.expt_tp` 大小为 1 → 无内部通信（experts.py:1204）；`MoEAlltoAllTokenDispatcher` 本地 permute（无序列 gather）→ `all_to_all(ep_group)`（:685）→ 本地 GEMM → a2a 返回，**无 AG/RS** |
| **torchtitan** | **同构**：EP "借用" TP 的 rank——dense (dp,tp) flatten 重切为 (efsdp, ep)，expert 权重仅 `{EP: Shard(0)}` |
| NeMo Automodel | EP 组靠 mesh 轴序自然跨 TP 坐标，但 token 路径在 TP 组内复制，expert 梯度需 `expert_tp_replication_factor` 缩放修正 |
| spmd_types | 类型系统可表达 `{EDP, EP}` 嵌套 mesh（`reinterpret_mesh`） |

##### 核心洞察：MoE 的正确边界形态是 SP-in（identity），不是 gather/scatter

MoE 是 per-token 计算，TP 各 rank 持不同序列 chunk 时可以**各自独立**
路由与计算（Megatron MoE 从不 gather 序列）。TP-extend-EP 路径因此把
moe_mlp 的边界契约改为 **identity**（通信全部内聚到 region 内部）：

```
D-09（EP 不跨 TP）：  in Shard(1) →[all-gather]→ Replicate → MoE → Partial →[reduce-scatter]→ Shard(1)
D-10（TP-extend-EP）：in Shard(1) →[identity]→  Shard(1) → MoE(内部 a2a) → Shard(1) →[identity]→ Shard(1)
```

SP-in 下各 TP 坐标处理不同 chunk，expert 不会被重复喂 token（规避
automodel 的 tp 倍梯度复制）。

##### mesh 重分区：全 dense 区域 → (edp, ep=ep_size)

expert 域 = **全 dense 区域**（mesh 非 pp 轴全部 rank =
dp_replicate × dp_cp × tp，记 D）。按 mesh 轴序 row-major flatten 后重切
`(edp = D/ep_size, ep = ep_size)`：

```python
# _expert_mesh_layout（纯映射，单进程可测）
derived = np.array(mesh.rank_list).reshape(mesh.mesh_shape)      # 全 dense
derived = derived.reshape(D // ep_size, ep_size)                 # (edp, ep)
init_device_mesh("npu", derived.shape,
                 mesh_dim_names=("edp", "ep"),
                 rank_list=tuple(derived.flatten()))
# EP 组（内层，a2a 通信域）：flatten 序连续的 ep_size 个 rank——
#   先跨完整个 TP 组，再向相邻 dp/cp rank 扩展
# edp 组（外层）：expert 数据并行度 = D/ep_size
```

**用户示例**（8 卡 mesh (dp=4, tp=2)，ep_size=4）：TP 组
`{0,1}/{2,3}/{4,5}/{6,7}`；扩展 EP 组 `{0,1,2,3}` / `{4,5,6,7}`——跨
2 个 TP 组 × 2 个 dp rank；edp=2。ep_size=2 时 EP 组退化为 TP 组。

校验（`_validate_ep_extend`，仅在实际命中 HF 原生 MoE 时执行——
pre-stacked EP-aware 模块不受此约束）：`ep_size ≤ D` 且整除；
`num_experts % ep_size == 0`（每 rank 持 `num_experts/ep_size` 个完整
expert）；pp>1 暂不支持。

##### 参数分片：派生 mesh 上的 {EP: Shard(0)}（仅 expert 维）

planner 对 TP-extend-EP 路径的 moe_mlp 生成不同契约（同 D-05 的运行时
修订风格）：

```python
# TP-extend-EP 路径的 moe_mlp spec（在 D-09 stacked 布局基础上）：
spec.params = {
    "experts.gate_proj": {EP: Shard(0), CP: Replicate()},
    "experts.up_proj":   {EP: Shard(0), CP: Replicate()},
    "experts.down_proj": {EP: Shard(0), CP: Replicate()},
    "gate.weight":       {TP: Replicate(), CP: Replicate(), EP: Replicate()},
    "shared_experts.*":  {TP: Shard(0|1), CP: Replicate(), EP: Replicate()},
}
spec.in_dst = spec.in_src                     # identity（SP-in）
spec.out_src = spec.out_dst = 同 in_src 布局   # identity（SP-out）
spec._ep_size = ep_size   # 扩展 EP 组大小，apply 据此构建派生 mesh
```

expert 权重**没有 TP 键、没有第二轴**——stacked 3D `[E, H_out, H_in]`
只在 dim 0（expert 维）按扩展 EP 组分片，每 rank 得到
`[num_experts/ep_size, H_out, H_in]` 的**完整 expert 矩阵**。apply 侧
`_shard_module_params` 按参数名分流：`experts.*` 用派生 expert mesh +
`("edp", "ep")` 解析 placements；其余参数（gate/shared_experts/其他
模块）用原 mesh 不变。

##### TP-extend-EP 计算流（`_hf_native_ep_compute`，ep_utils.py）

**与 Megatron AlltoAll dispatcher（etp=1 配置）逐步对齐**（括号内为
Megatron 坐标）：

```python
def _hf_native_ep_compute(module, hidden_states, *, router_fn, ep_group, tp_group=None):
    """SP-in（本地 chunk）→ region 内全部通信 → SP-out（本地 chunk）。

    输入 hidden [B, S/tp, H]（本地序列 chunk，boundary identity）；
    输出 [B, S/tp, H]（complete，边界 identity）。
    """
    # 1. router(本地 chunk) → topk（各 rank chunk 不同 → 无 token 复制；
    #    Megatron router 前向无通信）
    # 2. 按 (dest, expert) 排序 + counts 经 all_to_all_single 交换
    #    （Megatron token_dispatcher.py preprocess 同）
    # 3. a2a over 扩展 EP 组（Megatron :685 all_to_all(ep_group)）——
    #    token 带完整 H，发给目标 expert 的持有 rank
    # 4. 本地 SwiGLU（**完整 expert 权重**，expert MLP 无内部通信——
    #    Megatron expt_tp 大小为 1 同构）→ 输出 complete，无 Partial
    # 5. a2a over 扩展 EP 组返回（Megatron combine :844）→ 逆序加权聚合
    #    → out 本地 chunk
    # 6. shared_experts（若存在）：chunk × TP 分片权重 → Partial →
    #    TP 组 all_reduce（chunk 完整化）→ 相加
    # 7. return out（边界 identity，Shard(1) 直出）
```

##### 与上下游模块的通信与张量排布（端到端视图）

TP-extend-EP 路径与上下游**零通信**：token 经 a2a 去程/返程后严格回到
源 rank，序列 chunk 边界全程不变，residual 与链式契约无需任何重排：

```
post_attention_layernorm  out_dst {TP: Shard(1), CP: Shard(1)}
    │  [B, S/tp, H] 本地 chunk ── boundary identity（无通信）
    ▼
moe_mlp（TP-extend-EP region 内部）：
  ① router(chunk)                     本地 chunk → topk_idx/topk_w [T, K]
  ② a2a over 扩展 EP 组               【脱离序列布局】token-major 缓冲：
                                      本 rank 持有「路由到本 rank 本地
                                      expert」的全部 token（可来自 TP/DP
                                      坐标的其他 rank）
  ③ expert GEMM（完整 expert 权重）    [n_tokens, H] → [n_tokens, H] complete
  ④ a2a over 扩展 EP 组返回            token 严格回到源 rank
  ⑤ 逆序加权聚合                       恢复 [B, S/tp, H] 本地 chunk
  ⑥ shared_experts(chunk) + TP all_reduce（若存在）
    │  [B, S/tp, H] 本地 chunk ── boundary identity（无通信）
    ▼
residual add：hidden = hidden + mlp_out（逐元素，chunk 对齐）
    ▼
下一层 input_layernorm  in_src {TP: Shard(1), CP: Shard(1)}  ✓ 链式契约一致
```

逐步张量排布（设 T=S/tp 为本 rank token 数，ep 为扩展 EP 组大小、
E_l = num_experts/ep 为本地 expert 数、n_i 为 a2a 后本地 expert i 的
token 数）：

| 步骤 | 张量 | 排布（逻辑） | placement 语义 |
|---|---|---|---|
| 入口 | hidden [B, S/tp, H] | 序列 chunk | `{TP: Shard(1), CP: Shard(1)}` |
| ① router | topk_idx/topk_w [B·T, K] | 本地 chunk 的路由 | 无（router 权重 TP Replicate） |
| ② a2a(EP) | tokens [Σ_in, H] | token-major，按目标 expert 排序 | 脱离序列布局（通信原语内） |
| ③ GEMM | out [Σ_in, H] | token 同 ②，**complete**（完整权重） | 无 Partial |
| ④ a2a(EP) 返回 | out [K·T, H] | 本 rank 原始 token（含 K 份展开） | 脱离序列布局（通信原语内） |
| ⑤ 聚合 | out [B, S/tp, H] | 恢复序列 chunk | `{TP: Shard(1), CP: Shard(1)}` |

三个不变量（正确性依赖，测试断言点）：

1. **token 守恒与回程**：a2a 去程/返程在同一扩展 EP 组上互逆，每个
   token 严格回到源 rank——序列 chunk 的边界与顺序全程不变，
   residual add 与下游 in_src 无需任何重排；
2. **布局只在 region 内脱离**：序列布局 ↔ token-major 的转换全部发生在
   region 内部（②~④），边界两侧始终是 `{TP: Shard(1), CP: Shard(1)}`——
   与 D-09 路径相比，少了一对边界 all-gather/reduce_scatter；
3. **CP 组合不变**：扩展 EP 组由同一 pp stage 的 dense rank 连续组成，
   a2a 在本地 chunk 上进行，CP 维契约全程 `Shard(1)`。

##### 梯度语义（与 automodel 的关键差异）

- expert 权重：每 rank 持有不同 expert 的完整矩阵，梯度是**各自完整
  expert 的 local shard**（聚合了扩展 EP 组内全部 rank 的 token），
  **无需 TP 归约**；跨 edp（expert 数据并行）的同步由 FSDP 路径处理
  （Megatron：expert 参数仅在 EDP 组归约且带 `edp_size/dp_size` 缩放
  因子，distributed_data_parallel.py:204——本方案对应语义）；
- router（TP Replicate）：各 rank 在本地 chunk 独立计算，梯度不同——走
  现有 tp_grad_info 的 Replicate 参数梯度同步机制；
- **无复制因子**：SP-in 下各 TP 坐标处理不同 chunk，expert 不被重复喂
  token（对照 automodel 复制路径的 `expert_tp_replication_factor` 缩放）；
- **无 Partial 归约点**：expert GEMM 输出 complete（完整权重），整个
  region 内不存在 all_reduce/all_gather/reduce_scatter。

##### expert 参数两种布局的统一处理（D-11，2026-07-21）

planner `_mark_hf_native_moe` 按参数形态分两类（混合结构不标记并 warning）：

| 布局 | 参数形态 | 来源 | 处理 |
|---|---|---|---|
| **per-expert** | `experts.<idx>.<proj>.weight` × 3E 个 2D | 旧版 HF（≤2025 中）/ 自研模型 | `_ep_stack` 记录堆叠元数据，apply Phase A `_stack_moe_experts` 堆叠为 3D 后分片 |
| **batched** | `experts.gate_up_proj [E,2I,H]` + `experts.down_proj [E,H,I]`（automodel 命名 `gate_and_up_projs/down_projs` 同构） | **当前 HF main**（qwen3_moe/mixtral/deepseek_v3/glm4_moe/qwen2_moe，2025 重构后） | 天生 stacked，`_ep_stack` 留空、跳过堆叠，直接 `{EP: Shard(0)}` 分片 |

配套（batched 布局的计算侧）：

- `_swiglu_weights` 增加 fused 分支：`gate_up_proj` 解析为
  `(fused, None, down)`，`_hf_native_ep_compute` 的 expert GEMM 按
  `chunk(2)` 拆出 gate/up（与 HF `NaiveMoe` forward 逐步一致）；
- router adapter 注册表按 arch 分发：**qwen3moe/mixtral**（TopKRouter
  模块直返 `(logits, scores, indices)`，adapter 取后两个）；
  **deepseekv3/glm4moe**（sigmoid + `e_score_correction_bias` +
  group-limited topk（n_group/topk_group）+ 可选归一化 +
  `routed_scaling_factor`，与 HF `route_tokens_to_experts` 逐步一致）；
  未注册 arch 落 default（Linear gate 的 softmax topk）；
- w1/w2/w3 命名的 pre-stacked 布局**不收** batched 路径——那是 EP-aware
  模块（自身 dispatcher）的约定布局，走原路径（D-09 文档约定不变）；
- v1 不支持 expert bias（命中不标记并 warning）。

##### v1 约束与与 D-09 的分工

| 项 | 说明 |
|---|---|
| 开启方式 | **`ep_size > 1` 即开启**（HF 原生 MoE 恒走 TP-extend-EP 路径）；**`ep_size` 即扩展 EP 组大小**，无单独 etp 配置 |
| 校验 | `ep_size ≤ dp_replicate × dp_cp × tp` 且整除；`num_experts % ep_size == 0`；仅命中 HF 原生 MoE 时校验 |
| expert 结构 | SwiGLU 无 bias：per-expert 2D（`_ep_stack` 堆叠）或 batched 3D fused（无需堆叠，D-11） |
| 派生 mesh | DeviceMesh 任意 rank ndarray 构造（`_expert_mesh_layout` 纯映射 + `_build_expert_mesh`）；EP 组 = flatten 连续 ep_size 个 rank（先跨 TP 组再跨 dp/cp）；edp = D/ep_size |
| pp | v1 不支持 pp>1（按 stage 分 mesh 后调用） |
| DeepSeek 形态 | 细粒度 expert（256+）：大 ep_size 直接扩大 a2a 组与 expert 切分度，单 rank 显存 = num_experts/ep_size 个完整 expert |
| flex dispatcher | DeepEP 类 fused 后端（Megatron :1657-1679 直接用组合组 + fused kernel）留作后续性能路径，v1 用显式原语 |

##### 测试计划

| 层 | 用例 |
|---|---|
| 单进程 | planner TP-extend-EP 契约（in_dst/out identity、expert {EP: S(0)} 无 TP 键、`_ep_size == ep_size`）；**batched 布局契约（D-11：无堆叠、fused 权重解析、qwen3moe/deepseekv3 adapter）**；派生 mesh rank 映射（mesh (dp=4,tp=2)：ep=4 → (edp=2, ep=4)，EP 组 {0,1,2,3}/{4,5,6,7}；ep=2 → EP 组即 TP 组）；ep_size 超界/不整除/num_experts 不整除报错 |
| 分布式（gloo pad 回退） | **ep=4 扩展 EP 组跨 TP×dp 坐标**（mesh (dp=4,tp=2)，8 进程）per-expert 与 **batched** 两布局各一组双模式 e2e vs 单卡；a2a pad 路径 fwd/bwd 对拍（`test_dist_s6_hf_native_moe.py`） |
| 回归 | ep=1 路径（D-09 全部用例）不受影响 |

### 6.5 并行策略冲突检测

实际实现为自由函数 `validate_model_compatibility`（`sharding_planner.py`，
经 `hyper_models/components/distributed/__init__.py` 导出；与 06 的拓扑校验分工——这里
只看模型 config）：

```python
def validate_model_compatibility(
    model, *, tp_size: int = 1, cp_size: int = 1, ep_size: int = 1,
    seq_len: Optional[int] = None,
) -> None:
    config = getattr(model, "config", None)
    if config is None:
        return

    if tp_size > 1:
        # TP 校验：attention heads / kv heads 必须整除 tp（属性缺失时跳过）
        heads = getattr(config, "num_attention_heads", None)
        if heads is not None and heads % tp_size != 0:
            raise ValueError(
                f"num_attention_heads ({heads}) must be divisible by TP ({tp_size})")
        kv_heads = getattr(config, "num_key_value_heads", None)
        if kv_heads is not None and kv_heads % tp_size != 0:
            raise ValueError(
                f"num_key_value_heads ({kv_heads}) must be divisible by TP ({tp_size})")
        # TP+EP 联合校验（expert hidden dim 必须整除 tp；触发条件 tp>1 且 ep>1，
        # 字段为 config.moe_intermediate_size）
        moe_inter = getattr(config, "moe_intermediate_size", None)
        if ep_size > 1 and moe_inter is not None and moe_inter % tp_size != 0:
            raise ValueError(
                f"moe_intermediate_size ({moe_inter}) must be divisible by TP ({tp_size})")

    # CP 校验：seq_len 可选——仅在显式传入时校验；2*cp 约束源自 zigzag/ring
    # 负载均衡方案，all-gather K/V + contiguous chunk 方案下冗余（G5，§6.3.4）。
    if cp_size > 1 and seq_len is not None and seq_len % (cp_size * 2) != 0:
        raise ValueError(
            f"seq_len ({seq_len}) must be divisible by 2*cp ({2 * cp_size})")

    # EP 校验：必须为 MoE 模型且 num_experts 整除 ep
    #（num_experts 或 n_routed_experts 字段）
    if ep_size > 1:
        num_experts = (getattr(config, "num_experts", None)
                       or getattr(config, "n_routed_experts", None) or 0)
        if num_experts <= 0:
            raise ValueError("EP>1 requires MoE model (num_experts > 0)")
        if num_experts % ep_size != 0:
            raise ValueError(
                f"num_experts ({num_experts}) must be divisible by EP ({ep_size})")
```

注意：早版的 TP+CP 联合校验（`seq_len % (cp*tp) == 0`）未保留——SP 与 CP
沿同一序列维独立切分，各自约束足够；D-10 的 `ep_size ≤ dense 区域且整除`
校验在 planner `_validate_ep_extend` 中、实际命中 HF 原生 MoE 时执行
（§6.4.8），不在本函数。

### 6.6 通信职责总结

```
+--------------------------------------------------------------+
|              Communication Responsibility Map                 |
|                                                              |
|  PrecompiledBoundary (compile-time plan, runtime execution):  |
|    FORWARD activation communication ONLY:                     |
|    * TP all-gather     (Shard->Replicate on TP mesh)          |
|    * TP reduce-scatter (Partial->Shard on TP mesh)            |
|    * CP all-gather     (Shard->Replicate on CP mesh)          |
|    * CP reduce-scatter (Replicate->Shard on CP mesh)          |
|    * EP redistribute   (Replicate->Shard on EP mesh)          |
|    * loss_parallel     (Shard(-1) directly to CE loss)        |
|                                                              |
|  Inner Attention Wrapper (compile-time forward replacement):  |
|    * CP K/V all-gather in SDPA/FlexAttention                  |
|    Reference: Titan context_parallel.py apply_cp_to_forward() |
|                                                              |
|  _local_params_context (build-time one-shot unpack):          |
|    * EP all-to-all dispatch (tokens -> expert ranks)          |
|    * EP all-to-all combine  (expert outputs -> token ranks)   |
|    Reference: Titan moe_sharding.py LocalMapConfig            |
|                                                              |
|  FSDP2 / HSDP (DP dimension + gradient sync):                 |
|    * DP all-gather / reduce-scatter (parameters)              |
|    * TP all-reduce (Replicate parameter gradients)            |
|      -> norm/bias/gate params: FSDP layout-driven grad sync   |
|      -> See §6.7 for details on gradient synchronization      |
|    * Mixed precision / CPU offload                            |
+--------------------------------------------------------------+
```


---


---



---


### 6.7 Gradient Synchronization: FSDP handles parameter gradients

DTensor 和 PrecompiledBoundary **仅处理 FORWARD 过程中的 activation 通信**（all-gather / reduce-scatter / redistribute）。所有参数的梯度同步由 FSDP2 / HSDP 负责。本节按代码现状区分两个机制（D-12）：

| 机制 | 位置 | 状态 |
|------|------|------|
| **DTENSOR_UNIFIED layout-driven gradient sync**：`fully_shard` 在参数仍为 DTensor 时记录其原始 mesh/placements（`_orig_dtensor_placements`），从统一 SPMD layout 直接推导梯度归约组 | `core/fully_shard/hsdp_param.py` / `platform/torch/fully_shard/param.py` | **当前已实现**（models/* 的 DTensor + fully_shard 直连路径在用，§6.7.2/§6.7.3） |
| **tp_grad_info**：planner/applier 侧从 ShardingPlan 读出的 `{param_fqn: (tp_placement, tp_mesh)}` 查表 | `hyper_models/components/distributed/tp_grad.py` | **规划项**——`apply_sharding_plan` 已产出并返回，但 `fully_shard` 不接受 `tp_grad_info` 入参，全仓库无消费端（仅测试工具 `testing/grad_equiv.py::simulate_tp_replicate_grad_sync` 模拟该旁路）；fully_shard 消费端接线待落地（§6.7.1） |

#### 6.7.1 build_tp_grad_info：从 ShardingPlan 读取梯度同步信息（规划项：已产出，fully_shard 消费端待接线）

```python
# hyper_models/components/distributed/tp_grad.py
def build_tp_grad_info(
    plan: ShardingPlan, tp_mesh: DeviceMesh,
    *, tied_pairs: list[tuple[str, str]] | None = None,
) -> dict[str, tuple[Placement, DeviceMesh]]:
    """{param_fqn: (tp_placement, tp_mesh)}，tp_placement in {Shard, Replicate}。
    从 ShardingPlan 读取（build 期 _local_params_context 解包前 plan 仍在）。

    tied_pairs: 共享存储的参数对（如 ("embed_tokens.weight", "lm_head.weight")），
        来源 `plan.tied_pairs` 或 `detect_tied_weights(model)`。tied 对必须映射到
        同一 tp_placement——否则 FSDP 会把同一份物理参数当成两个不同 TP 切分,
        梯度同步语义冲突。归一化策略：tied 对 placement 不一致时，取较细的分片
        （Shard 优先于 Replicate），保证两端的 TP all-reduce / reduce-scatter 一致。
    """
    info = {}
    for fqn, spec in plan.modules.items():            # ShardingPlan.modules
        for param_name, named_placement in spec.params.items():  # ModuleShardingSpec.params
            full_fqn = f"{fqn}.{param_name}"
            tp_placement = named_placement.get("tp", Replicate())  # NamedPlacement 是 dict[str, Placement]
            if getattr(spec, "_ep_size", 0) and param_name.startswith("experts."):
                # D-10 TP-extend-EP：expert 权重在派生 expert mesh (edp, ep)
                # 上仅按 expert 维分片（无 TP 键）——梯度为各 rank 不同的
                # local shard（不同 expert + 扩展 EP 组聚合的 token），不做
                # TP 组同步（Shard 标记语义；缺省 Replicate 会导致 FSDP 对
                # 分片梯度错误 all-reduce，§12.3.10c）
                tp_placement = Shard(1)
            info[full_fqn] = (tp_placement, tp_mesh)

    # tied-weight 归一化：embed.weight <-> lm_head.weight 等共享存储参数
    pairs = tied_pairs if tied_pairs is not None else getattr(plan, "tied_pairs", None)
    if pairs:
        for a, b in pairs:
            if a in info and b in info:
                pa, _ = info[a]
                pb, _ = info[b]
                if pa != pb:
                    # 取较细分片（Shard 优先），保证 tied 对梯度同步语义一致
                    norm = pa if isinstance(pa, Shard) else pb
                    info[a] = (norm, tp_mesh)
                    info[b] = (norm, tp_mesh)
    return info
```

**tp_grad_info 的数据来源**：从 ShardingPlan 读取 (`build_tp_grad_info(plan, tp_mesh)`)，而非从 DTensor 的 placement 推导。因为在 `_local_params_context` 解包后 DTensor 元数据已丢失，但 build 期的 `plan.modules[*].spec.params` 仍然保留完整的 placement 信息。

**tied-weight 归一化**：当 embed 与 lm_head 共享权重（weight tying）时，两者必须映射到同一 tp_placement。`build_tp_grad_info` 在构造时对 `plan.tied_pairs`（或显式传入的 `tied_pairs`）做归一化——placement 不一致时取较细分片（Shard 优先于 Replicate），保证 tied 对的 TP 梯度同步（all-reduce / reduce-scatter）语义一致。这是 06 root-unit tied-weights 语义的实现归属（06 仅声明契约，归一化逻辑在此）。

#### 6.7.2 DTENSOR_UNIFIED：`_orig_dtensor_placements` 与 `_get_base_spmd_placements`（已实现）

FSDP 侧不查 tp_grad_info 表——梯度归约布局来自参数自身。`fully_shard` 初始化每个参数时记录其原始 DTensor 元数据（`platform/torch/fully_shard/param.py:149-151`）：

```python
# platform/torch/fully_shard/param.py —— TorchHSDPParamV2.__init__
self._orig_param_is_dtensor = isinstance(param, DTensor)
self._orig_dtensor_mesh = param.device_mesh if self._orig_param_is_dtensor else None
self._orig_dtensor_placements = tuple(param.placements) if self._orig_param_is_dtensor else None
```

`param_mode` 由 `infer_fully_shard_param_mode`（`core/fully_shard/hsdp_utils.py`）推导：参数带 DTensor layout 且 `fully_shard` 传入 DP mesh → **DTENSOR_UNIFIED**；参数为 plain tensor → LOCAL_PARAM；无 mesh → DTENSOR_COMPAT。DTENSOR_UNIFIED 下 `_get_base_spmd_placements`（`core/fully_shard/hsdp_param.py:209`）把 DP/FSDP mesh 前缀拼到参数原 DTensor mesh 之前，形成统一 SPMD layout：

```python
# core/fully_shard/hsdp_param.py: _get_base_spmd_placements（DTENSOR_UNIFIED 分支）
if self.param_mode == FullyShardParamMode.DTENSOR_UNIFIED and self._orig_param_is_dtensor:
    # 保留参数原 distributed layout，DP/FSDP 维作为最外层前缀拼到统一 mesh 上
    self._spmd_mesh = DeviceMesh.concatenate([self.mesh_info.mesh, self._orig_dtensor_mesh])
    dp_prefix_placements = tuple(Replicate() for _ in range(self.mesh_info.mesh.ndim))
    return dp_prefix_placements + tuple(self._orig_dtensor_placements)
```

`DeviceMesh.concatenate` 已实现（`core/dtensor/device_mesh.py:1035`）：沿新的最外层维度拼接多个 DeviceMesh，返回更高维的 layout-backed mesh（如 `dp_mesh(["dp"]) + tp_mesh(["tp"])` → `(["dp","tp"], (dp,tp))`），继承各子 mesh 的进程组元数据。

随后 `_apply_data_parallel_placements` 把 FSDP shard placement 写入 DP 前缀位，得到最终 layout：

- TP-Shard 参数：`[Shard(DP), Shard(N)(TP)]`
- TP-Replicate 参数：`[Shard(DP), Replicate(TP)]`

**生效前提**：`fully_shard` 调用时参数仍是 DTensor。若参数已被解包为 plain local tensor，`_orig_param_is_dtensor=False`，落入 LOCAL_PARAM 模式，统一 layout 与下述 TP 归约组推导均不生效（时序约束见 §7）。

#### 6.7.3 layout-driven 归约组与 unsharded 梯度归一化（已实现）

`_build_layout_driven_group_info`（`core/fully_shard/hsdp_param.py:285`）从最终 SPMD layout 推导 unsharded 梯度的 all-reduce 组——收集全部 Replicate 轴、排除 FSDP shard 轴（该轴走 reduce-scatter）：

```python
# core/fully_shard/hsdp_param.py: _build_layout_driven_group_info
group_axes = [axis for axis, p in enumerate(self._spmd_placements) if p.is_replicate()]
if uses_param_shard and spmd_shard_mesh_dim is not None:
    group_axes = [a for a in group_axes if a != spmd_shard_mesh_dim]   # FSDP shard 轴除外
# group_axes 非空 → 按这些轴建 unsharded 梯度归约组（all_reduce_grad 的通信域）
```

由此，**TP-Replicate 参数（norm / bias / gate / router）的梯度 TP all-reduce 由 layout 直接推导**，无需 tp_grad_info 查表：

- TP-Replicate 参数：最终 layout `[Shard(DP), Replicate(TP)]` → group_axes=[TP 轴] → `all_reduce_grad()` 在 **TP 组**上 all-reduce unsharded 梯度；
- TP-Shard 参数：最终 layout `[Shard(DP), Shard(N)]` → 无剩余 Replicate 轴 → 无额外归约组，梯度天然是各 rank 的 local shard，仅走 DP reduce-scatter。

unsharded 梯度在进入集合通信前经 `_normalize_unsharded_grad_to_local`（torch 侧 `_to_local_unsharded_grad`，`platform/torch/fully_shard/param.py:302`）归一化：

```python
# core/fully_shard/hsdp_param.py: _normalize_unsharded_grad_to_local
def _normalize_unsharded_grad_to_local(self, grad, *, reduce_partial_dtensor=True):
    """把 unsharded 梯度归一化到 fully_shard 集合通信期望的 local tensor。"""
    if not isinstance(grad, DTensor):
        return grad                                  # plain local grad，原样进入集合通信
    if reduce_partial_dtensor and any(p.is_partial() for p in grad.placements):
        grad = grad.reduce_partial()                 # Partial → Replicate：TP 组 all-reduce
    if grad.mesh/placements != (self._orig_dtensor_mesh, self._orig_dtensor_placements):
        grad = grad.redistribute(self._orig_dtensor_mesh,
                                 self._orig_dtensor_placements)   # 归位到参数原 layout
    return grad.to_local()
```

**TP-Replicate 参数（norm / bias / gate / router）的梯度同步**：

这些参数在 TP 维度上是 `Replicate()` —— 每个 TP rank 持有完整副本。Forward 时各 rank 在各自的 TP 局部激活上独立计算，产生的梯度是 Partial 贡献，需要 all-reduce 才能保证一致性。**这个 TP all-reduce 由 FSDP 侧完成**，而非 PrecompiledBoundary：

- `fully_shard()` 初始化时记录 `_orig_dtensor_placements`（§6.7.2），`_build_layout_driven_group_info()` 检测到最终 layout 中 TP 轴为 `Replicate()` 时，把该轴纳入 unsharded 梯度归约组；
- `all_reduce_grad()` 在 backward 中在该组（TP 组）上对 unsharded 梯度执行 all-reduce；若梯度以 Partial DTensor 形式回流（DTensor autograd 路径），先经 `reduce_partial()` 归约（同为 TP 组 all-reduce 语义），再 redistribute 回参数原 layout 并 `to_local()`；
- 整条链路**不经过 tp_grad_info**——归约布局来自参数自身的 `_orig_dtensor_placements`，生效前提是 `fully_shard` 调用时参数仍为 DTensor（§7）。

**TP-Shard 参数（colwise / rowwise）的梯度处理**：

这些参数在 TP 维度上是 `Shard(0)` 或 `Shard(1)` —— 每个 TP rank 持有参数的一个分片。Forward 时各 rank 使用自己的参数分片计算，产生的梯度**已经是正确的 local shard**，无需跨 TP rank 同步（rowwise 输出的 Partial 激活梯度由 PrecompiledBoundary 的 reduce-scatter 在 forward 边界归约，与参数梯度无关）。

**与 HSDP 的复用关系**：

该机制完全复用 HSDP 现有的 `_get_base_spmd_placements()` / `_build_layout_driven_group_info()` / `all_reduce_grad()` / `_normalize_unsharded_grad_to_local()` 基础设施，DTensor 参数经 DTENSOR_UNIFIED 模式接入，无需为 DTensor 新增梯度同步代码路径。

**tp_grad_info 的定位（规划项，D-12）**：

production 路径下 `apply_sharding_plan` 在 build 期把 DTensor 参数永久解包为 plain local tensor（§7），此后 `fully_shard` 无法再经 `_orig_param_is_dtensor` 恢复 TP placement——tp_grad_info 是为该场景预留的显式查表通道（§6.7.1 数据结构不变，含 tied-weight 归一化与 D-10 expert Shard 标记）。其 fully_shard 消费端接线待落地；在此之前，TP-Replicate 参数的梯度 TP all-reduce 在测试中由 `testing/grad_equiv.py::simulate_tp_replicate_grad_sync` 模拟。

**总结**：

```
FORWARD 通信职责:   DTensor + PrecompiledBoundary（activation 通信）
BACKWARD 通信职责:  FSDP2 / HSDP（参数梯度同步）
                   ├─ TP-Replicate 参数: layout-driven TP 组 all-reduce
                   │   （DTENSOR_UNIFIED 已实现；tp_grad_info 通道待接线）
                   └─ TP-Shard 参数:     梯度已是 local shard，无需 TP 同步
```

## 7. 与 FSDP2 的关系

> **调用位置**: 时序树 ③.4.5.11 — `fsdp2_manager.parallelize(model)` 在 ShardingApplier 之后执行（06 侧集成契约；`apply_sharding_plan` ↔ `fully_shard` 的接线当前未落地，见本节"代码现状"）

```
分工边界:
  DTensor 管理的维度: TP, CP, EP
  FSDP2 管理的维度: dp_shard, dp_replicate

梯度归约的时序约束（D-12，以代码为准）:
  fully_shard 的 DTENSOR_UNIFIED 模式在初始化时记录参数的原始 DTensor
  mesh/placements（_orig_param_is_dtensor / _orig_dtensor_placements，§6.7.2），
  梯度归约组从该统一 layout 推导（§6.7.3）——
  **fully_shard 必须在参数仍为 DTensor 时执行**。
  若参数已被解包为 plain local tensor，fully_shard 落入 LOCAL_PARAM 模式，
  统一 layout 与 TP 归约组推导均不生效。

执行顺序（目标时序）:
  1. apply_sharding_plan() Phase A/B → 参数 → DTensor（TP/CP/EP 分片）
  2. fully_shard(layer, mesh=dp_mesh) → 此时参数仍是 DTensor：
     记录 _orig_dtensor_placements，推导 DTENSOR_UNIFIED 统一 layout
     （DeviceMesh.concatenate([dp_mesh, orig_mesh])，已实现）
     与 TP 归约组（layout-driven，§6.7.3）
  3. 训练期 forward：参数以 local tensor 参与计算（to_local 临时视图 /
     FSDP unshard 结果），production wrapper 内为纯 local tensor 计算
     + PrecompiledBoundary 通信
  4. backward：local autograd 直出梯度 → FSDP post-backward：
     _to_local_unsharded_grad（Partial DTensor 梯度 reduce_partial →
     redistribute 回原 layout → to_local）→
     all_reduce_grad（TP-Replicate 参数的 TP 组 all-reduce）→
     reduce_scatter_grad（DP 维 reduce-scatter → local shard）

代码现状（与目标时序的缺口，D-12）:
  - apply_sharding_plan 的 production 路径在 Phase C 入口调用
    _local_params_context——名为 context，实为 build 期一次性**永久**解包
    （非 contextmanager，不恢复；validate 模式的临时解包是另一个设施
    _temp_local_params），发生在 apply_sharding_plan 返回之前、任何
    fully_shard 调用之前，返回的模型参数已是 plain local tensor。
  - apply_sharding_plan 同时构造并返回 tp_grad_info（§6.7.1），但
    fully_shard 不接受 tp_grad_info 入参，全仓库无消费端；
    apply_sharding_plan ↔ fully_shard 的集成（06 fsdp2_manager.parallelize）
    尚未接线。因此按当前代码把两者直接串联会落 LOCAL_PARAM 模式，
    TP-Replicate 参数的 TP 梯度归约缺失（仅测试以
    simulate_tp_replicate_grad_sync 模拟）。集成落地时需二选一：
    (a) 调整时序——fully_shard 先于解包执行（或把解包改为训练期临时机制），
        走已实现的 DTENSOR_UNIFIED 路径；
    (b) 补齐 tp_grad_info 的 fully_shard 消费端（查表通道，§6.7.1）。
  - DTENSOR_UNIFIED 路径本身已在 models/* 的 DTensor + fully_shard 直连
    路径（不经 apply_sharding_plan 解包）中实际使用。

各阶段参数状态 (TP=4, FSDP dp_shard=8):

  Stage 1 — apply_sharding_plan Phase A/B 后:
    weight = DTensor(global_shape=[H, H], placements=[Shard(0)])
    DTensor._local_tensor = [H/4, H]  (local shard)
    参数仍是 DTensor 类型，携带完整的 global_shape + placements 元数据

  Stage 2 — fully_shard 后（参数仍为 DTensor，DTENSOR_UNIFIED）:
    _orig_dtensor_placements = (Shard(0),) 已在初始化时记录
    统一 SPMD layout = [Shard(DP), Shard(0)(TP)]   （TP-Shard 参数）
                     / [Shard(DP), Replicate(TP)]  （TP-Replicate 参数）
    FSDP2 接管 DP 维 all-gather / reduce-scatter 生命周期；
    TP-Replicate 参数的 unsharded 梯度归约组 = TP 组（layout-driven，§6.7.3）

  Stage 3 — 训练期:
    forward 在 local tensor 上计算（to_local 临时视图 / FSDP unshard 结果），
    backward 为 local autograd；梯度归约按"执行顺序"第 4 步完成
    （当前 apply_sharding_plan 的 build 期永久解包与此 Stage 的衔接
    属上述待接线项）

  Validate 模式 (validate_mode=True):
    参数保持为 DTensor（不执行解包）
    Forward 全程走 __torch_dispatch__ 传播 placement
    用于校验 out_src / out_dst 声明的正确性

参数视角 (TP=4, FSDP dp_shard=8):
  weight → distribute_tensor(weight, tp_mesh, [Shard(0)])
         → DTensor(global_shape=[H, H], placements=[Shard(0)])
  fully_shard(self_attn, mesh=dp_mesh)        # 参数仍为 DTensor
         → 记录 _orig_dtensor_placements
         → 统一 mesh = DeviceMesh.concatenate([dp_mesh, tp_mesh])
         → FSDP2 管理 DP 维 all-gather/reduce-scatter + TP 维梯度归约

Production forward + post-backward full timing (FSDP hook + boundary interleaving):

```
─ forward ─────────────────────────────────────────────────────────
  for each boundary module in execution order:
    FSDP2 pre-forward hook:
      unshard(module)                              # DP all-gather → full params
    PrecompiledBoundary.redistribute_inputs(x)     # TP/CP/EP all-gather
    module.forward(x_local)                        # pure local tensor compute
    PrecompiledBoundary.redistribute_outputs(y)    # TP reduce-scatter / CP shard
    FSDP2 post-forward hook:
      reshard(module)                              # DP reduce-scatter → 释放 full params
─ backward ────────────────────────────────────────────────────────
  FSDP2 pre-backward hook:
    rebuild_full_params(module)                    # DP all-gather → full params
  autograd backward:
    compute gradients on full params               # local autograd（双模式同一路径）
  FSDP2 post-backward hook:
    _to_local_unsharded_grad()                     # Partial DTensor grad:
                                                   #   reduce_partial（TP 组 all-reduce）
                                                   #   → redistribute 回 _orig_dtensor_placements
                                                   #   → to_local
    all_reduce_grad()                              # TP-Replicate 参数: unsharded 归约组
                                                   #   （= TP 组，layout-driven）all-reduce
    reduce_scatter_grad()                          # DP: reduce-scatter → local shard
──────────────────────────────────────────────────────────────────
```

Checkpoint (DCP):
  DCP 记录 DTensor 元数据 (global_shape + placements)
  → 跨 TP 配置重分片: TP=4→2, DCP 自动 all-gather + re-shard
  → 跨 DP 配置重分片: 同样处理
```

---



---

## 8. 用户自定义模块的配置方式

> **调用位置**: 时序树 ③.4.5.7 Phase 1 — ParameterClassifier 处理 ARCH_OVERRIDES

### 8.1 三种配置方式

| 方式 | 适用场景 | 用户需要做的 |
|------|---------|-------------|
| **A: 自动推导** | 模块遵循标准 Transformer 命名和结构 | **零代码**；默认规则自动覆盖 |
| **B: 架构规则覆盖** | 模块结构标准但命名非标准 | 注册一条命名规则到 `ARCH_OVERRIDES` |
| **C: 手动声明** | 完全自定义模块（无参数或有特殊通信需求） | 手动构建 `ModuleShardingSpec` 注入 `ShardingPlan` |
| **D: plan_overrides 合并** | 个别模块的契约/分片需定制（多输入契约 key、reshape 边界、特殊通信） | 手写该模块的 `ModuleShardingSpec`，经 `ShardingPlanner(plan_overrides=...)` 在 Phase 5 前合并（§3.6.7、§8.5） |
| **E: 自定义 wrapper 字段** | 模块**内部计算**无法被 dispatch 表达/内置 wrapper 覆盖不到（数据相关逻辑、自研 attention/MoE） | 在方式 C/D 的 spec 上声明 `use_local_map` / `local_compute_fn` / `inner_target` / `inner_wrapper`（§8.6） |

### 8.2 方式 A：自动推导（零代码）

```python
# 用户的模块使用标准 HF 结构
class MyStandardModel(nn.Module):
    def __init__(self, config):
        self.embed_tokens = nn.Embedding(...)    # 标准命名 → EMBED ✅
        self.layers = nn.ModuleList([
            MyDecoderLayer(config) for _ in range(config.num_layers)
        ])
        self.norm = nn.RMSNorm(...)              # 标准命名 → NORM ✅
        self.lm_head = nn.Linear(...)            # 标准命名 → LM_HEAD ✅

class MyDecoderLayer(nn.Module):
    def __init__(self, config):
        self.input_layernorm = nn.RMSNorm(...)            # NORM ✅
        self.self_attn = MyAttention(config)              # 子模块
        self.post_attention_layernorm = nn.RMSNorm(...)   # NORM ✅
        self.mlp = MyMLP(config)                          # 子模块

class MyAttention(nn.Module):
    def __init__(self, config):
        self.q_proj = nn.Linear(...)   # COLWISE ✅
        self.k_proj = nn.Linear(...)   # COLWISE ✅
        self.v_proj = nn.Linear(...)   # COLWISE ✅
        self.o_proj = nn.Linear(...)   # ROWWISE ✅

class MyMLP(nn.Module):
    def __init__(self, config):
        self.gate_proj = nn.Linear(...) # COLWISE ✅
        self.up_proj = nn.Linear(...)   # COLWISE ✅
        self.down_proj = nn.Linear(...) # ROWWISE ✅

# 使用方式——零配置
model = HyperAutoModelForCausalLM.from_pretrained(
    "/path/to/my-model",
    distributed_setup={"tp": 4},
)
# ShardingPlanner 自动完成一切 ✅
```

### 8.3 方式 B：架构规则覆盖（一行注册）

```python
# 用户模块有非标准命名
class MyModel(nn.Module):
    def __init__(self, config):
        self.token_embed = nn.Embedding(...)         # 非标准命名 "token_embed"
        self.blocks = nn.ModuleList([...])
        self.final_ln = nn.RMSNorm(...)              # 非标准命名 "final_ln"
        self.output_head = nn.Linear(...)            # 非标准命名 "output_head"

# 只需注册命名规则（pattern 为小写**子串**匹配，非正则）
# 在 hyper_models/components/distributed/sharding_planner.py 中：
ARCH_OVERRIDES["MyModelForCausalLM"] = [
    ("token_embed.weight",  ParamRole.EMBED),
    ("output_head.weight",  ParamRole.LM_HEAD),
    ("final_ln.weight",     ParamRole.NORM),
]
# 其余自动推导 ✅
# （内置条目见 §3.6.1 D-13：DeepSeek MLA 的 q_a/kv_a→REPLICATED、
#  q_b/kv_b→COLWISE，architectures/model_type 两种拼写均注册）
```

### 8.4 方式 C：手动声明 ModuleShardingSpec

适用于**无权重模块**或**特殊通信需求**的场景：

#### 场景 1：无权重但有通信需求的模块

```python
class CustomCommWrapper(nn.Module):
    """自定义通信包装器：无参数，但需要在模型 forward 中作为通信边界。

    功能：对 hidden_states 做自定义的 all-to-all 重排。
    """
    def __init__(self): ...
    def forward(self, hidden_states):
        # 自定义通信逻辑
        ...

# 手动注入到 ShardingPlan
plan = ShardingPlan(mesh_dim_names=("tp",))

plan.modules["model.custom_comm"] = ModuleShardingSpec(
    params={},                                         # 无参数
    in_src={"hidden_states": {TP: Shard(1)}},
    in_dst={"hidden_states": {TP: Shard(1)}},          # identity（通信在模块内部处理）
    out_src={TP: Shard(1)},
    out_dst={TP: Shard(1)},
    is_boundary=True,                                   # 仍然标记为边界
)

# 应用到模型
apply_sharding_plan(model, plan, mesh)
```

#### 场景 2：MoE Router（无权重需要分片但有 placement 声明）

```python
# MoE router 是一个简单的 nn.Linear + softmax
# 它的 weight/bias 需要全复制（Replicate）才能保证所有 rank 路由一致

plan.modules["model.layers.0.mlp.router"] = ModuleShardingSpec(
    params={
        "weight": {TP: Replicate(), EP: Replicate()},
    },
    in_src={"hidden_states": {TP: Shard(1), EP: Replicate()}},
    in_dst={"hidden_states": {TP: Replicate(), EP: Replicate()}},  # all-gather TP
    out_src={TP: Replicate(), EP: Replicate()},
    out_dst={TP: Replicate(), EP: Shard(0)},  # redistribute 到 EP
    is_boundary=True,
)
```

#### 场景 3：完全自定义的并行模块

```python
# 用户自己设计了一个特殊的 attention variant
# 它用 fused QKV + custom projection

class MyFusedAttention(nn.Module):
    def __init__(self, config):
        self.fused_qkv = nn.Linear(H, 3*H)      # 融合 QKV → FUSED_QKV
        self.custom_proj = nn.Linear(H, H)       # 非标准输出投影

    def forward(self, hidden_states):
        qkv = self.fused_qkv(hidden_states)
        q, k, v = qkv.chunk(3, dim=-1)
        # ... custom attention logic ...
        return self.custom_proj(attn_out)

# 步骤 1: 注册命名规则（小写子串匹配，非正则）
ARCH_OVERRIDES["MyModelForCausalLM"] = [
    ("fused_qkv.weight",    ParamRole.FUSED_QKV),  # 特殊角色
    ("custom_proj.weight",  ParamRole.ROWWISE),
]

# 步骤 2: 如果 FUSED_QKV 需要特殊分片逻辑，注册 SpecialHandler
SPECIAL_HANDLERS["my_fused_qkv_shard"] = _shard_my_fused_qkv

def _shard_my_fused_qkv(module, param_name, mesh):
    """自定义 fused QKV 分片：按 head 维度切分，保证 Q/K/V 各 block 内的 head 完整。"""
    tp_size = mesh.size()
    weight = module.fused_qkv.weight  # [3*H, H]
    n_heads = module.config.num_attention_heads
    head_dim = H // n_heads
    # ... 按 head 分组切片后重新拼接 ...
    module.fused_qkv.weight = nn.Parameter(sharded_weight)
```

### 8.5 方式 D：`plan_overrides` 合并注入（个别模块定制）

方式 C 的增强路径：手写 spec 不绕开 planner，而是经构造函数注入、在
**Phase 5 链式传播之前**合并（语义细节见 §3.6.7）。相比 plan() 后打补丁，
覆盖 spec 仍享受相邻契约校验、`_is_terminal` 标记与结构标记模板补齐。

#### 场景：自研多输入 attention（契约 key 非 hidden_states 且非首个位置参数）

```python
class PanguAttention(nn.Module):
    """forward(self, attn_bias, x, kv_cache=None)——被切张量 x 在位置 1，
    模板默认 key "hidden_states" 签名绑定 miss、位置兜底错绑下标 0。"""

# 推荐用法：先推导一次拿模板填充的 spec，只改需要定制的字段后回注
base_plan = ShardingPlanner().plan(model, mesh, tp_size=2)
overrides = {}
for fqn, spec in base_plan.modules.items():
    if fqn.endswith("attention"):
        spec = copy.deepcopy(spec)
        for attr in ("in_src", "in_dst"):
            d = getattr(spec, attr)
            d["x"] = d.pop("hidden_states")      # 契约 key 对齐真实签名
        overrides[fqn] = spec

plan = ShardingPlanner(plan_overrides=overrides).plan(model, mesh, tp_size=2)
model, tp_grad_info = apply_sharding_plan(model, plan, mesh)
```

完全手写也可以（此时注意：`_needs_cp_attn`/`use_local_map` 会从推断模板
自动补齐，无需也不允许手动关闭；`out_src`/`out_dst` 支持标量简写）：

```python
overrides["layers.0.attention"] = ModuleShardingSpec(
    params={"wq.weight": {TP: Shard(0)}, ..., "wo.weight": {TP: Shard(1)}},
    in_src={"x": {TP: Shard(1)}},
    in_dst={"x": {TP: Replicate()}},
    out_src={TP: Partial()},
    out_dst={TP: Shard(1)},
)
```

与上下游契约冲突（如 `in_src` 声明与上游 `out_dst` 不一致）在 `plan()` 内
即抛 `PlacementMismatchError`，不会延迟到运行时。

### 8.6 自定义 wrapper 接口总览（2026-07-21）

方式 A-D 解决"模块的**契约与分片**如何声明"；本节四个公开字段解决
"模块**内部计算逻辑**无法被 DTensor dispatch 表达、或内置 wrapper
覆盖不到"的场景。全部经 `plan_overrides` 的 spec 字段声明，planner
合并时随深拷贝保留，**不改写任何标记**——门控一律由解析链派生：
`use_local_map`/`local_compute_fn` 汇入 `_resolve_local_compute_fn`
（§4.4.3），`inner_target`/`inner_wrapper` 汇入 `_resolve_inner_target` +
`_resolve_inner_wrapper` 双链（§4.4.2）——声明互不嵌套，用户无需重复配置。

#### 接口一览

| 接口 | 类型/签名 | 做什么 | 典型场景 | 隐含标记 |
|------|-----------|--------|----------|----------|
| `use_local_map` | `bool` | **纯门控**：声明"模块自身 forward 即数据相关逻辑"→ 走 local-region 骨架，region 内计算 = **模块自身 forward**（解析链末位来源） | 自研 EP-aware MoE（a2a 已实现在 forward 内）、一切自带数据相关逻辑的模块 | —（模板推断 True 时强制继承，不允许关闭） |
| `local_compute_fn` | `fn(module, *args, **kwargs) -> Tensor` | **替换骨架内的计算函数**——骨架不变，region 内执行用户函数而非模块 forward | 自研 MoE：forward 是单卡逻辑 + 想注入自己的 dispatch（router 不在 MOE_ROUTER_ADAPTERS / expert 布局非标准 / DeepEP 融合 dispatcher） | —（门控由解析链派生，不改写任何字段） |
| `inner_target` | `str`（属性名 或 `"self"`） | **纯位置**：显式指定 inner attention 子模块，走内置 CP wrapper（K/V all-gather + D-04 causal 修正 + 双模式容错，缺省启发式分派） | 自研 attention 符合两约定之一（见下），但自动定位失败或有歧义 | —（门控派生，不改写标记） |
| `inner_wrapper` | `str`（注册表名）或 `fn(target_module, cp_mesh) -> None` | **纯行为**：str 显式固定内置注册表方案（`"sdpa_qkv"/"sdpa_hf"/"flex_qkv"/"flex_hf"`，`CP_WRAPPER_REGISTRY` 开放注册）；callable **整体接管** | 启发式会猜错行为时 str 固定；内置四路覆盖不到（典型：内部直调 `flash_attn_varlen_func`）时 callable | —（门控派生，不改写标记） |

#### 选择逻辑

```
模块内部计算能被模板/DTensor dispatch 正确表达？
├─ 是 → 零配置（方式 A/B/D 即可）
└─ 否 → 含数据相关逻辑（a2a/自定义 dispatch）？
        ├─ 已实现在模块 forward 内 → use_local_map=True
        ├─ 想自己写、但复用骨架（边界缝合/双模式） → local_compute_fn
        └─ 是 attention 的 K/V gather 问题？
                ├─ 定位失败/有歧义 → inner_target 指定位置
                ├─ 启发式会猜错行为 → inner_wrapper="sdpa_qkv" 等 str 固定
                └─ 内置四路都不适用（flash_attn_varlen 等） → inner_wrapper=callable 整体接管
```

`cp_*` 与 `local_*` **正交可组合**：同一模块可同时声明（Phase C Step 1
注入 CP inner wrapper、Step 2 包装 local region），如自研 MoE 层内嵌
自定义 attention。

#### 契约说明（框架负责什么 / 用户负责什么）

**通用原则**：用户代码永远工作在 **local tensor 世界**——DTensor↔local
的缝合（boundary、unwrap/rewrap、临时参数解包）恒由框架负责，用户代码
不感知双模式；通信组一律从 mesh 取（`cp_mesh.get_group()`、
`expert_mesh.get_group("ep")`），**禁止 `dist.new_group`**（进程组泄露）。

**`use_local_map`**（骨架 = `_wrap_local_region_forward`，§4.4.3）：
- 框架负责：boundary 入口（in_src→in_dst 通信）、validate 下输入
  `to_local` + `_temp_local_params` 临时解包 DTensor 参数、输出按声明
  `out_src` from_local 重包装、boundary 出口、最终解包为 local 返回；
- 用户负责：forward 内在 local tensor 上完成全部数据相关逻辑（含通信）。

**`local_compute_fn`**（骨架同上，仅替换 region 内计算）：
- 签名 `fn(module, *local_args, **local_kwargs) -> Tensor`，框架以
  `functools.partial(fn, module)` 绑定后注入；
- 输入恒为 local tensor（两模式一致），返回单个 local tensor（多输出
  模块暂不支持自定义 compute_fn）；
- 优先级（单一解析链 `_resolve_local_compute_fn`，派生门控）：
  `local_compute_fn` > TP-extend-EP 注入意图（`_ep_size>0` →
  `_hf_native_ep_compute`）> `use_local_map` 门控（模块自身 forward）。
  声明即生效——**无需也不应设置 `use_local_map`**，门控不读存储的 bool。

**`inner_target`**（纯定位提示，零代码）：
- `"self"` 表示模块本身即 inner；否则为子模块属性名（未命中即 fail-fast，
  拼写不容忍静默降级）；
- 指定的子模块必须满足内置两约定之一才能被正确包装：
  1. `forward(q, k, v, **kwargs)`（NeMo 约定）→ `_wrap_sdpa_for_cp`/
     `_wrap_flex_attn_for_cp`，`is_causal` 的 D-04 偏移掩码自动获得；
  2. `forward(hidden_states, ...)` 且内部调用 `F.scaled_dot_product_attention`
     或 `flex_attention`（HF 约定）→ 原语拦截路径。

**`inner_wrapper`**（str 固定 / callable 整体接管，§4.4.2 设计点 6/9）：
- str 形式：`CP_WRAPPER_REGISTRY` 注册表名，显式固定内置方案、跳过
  启发式；用户可 `CP_WRAPPER_REGISTRY["my_flash"] = my_fn` 注册命名方案；
- callable 形式：签名 `fn(target_module, cp_mesh) -> None`，就地替换
  `target_module.forward`；`target_module` = `inner_target` 指定的子模块
  （未指定且自动定位失败时退化为边界模块本身）；
- 用户负责：K/V 沿序列维 all-gather（用 `cp_utils.flex_cp_allgather`，
  自带 backward reduce-scatter 核）、双模式容错（输入可能是 DTensor——
  validate——或 local tensor——production，出口需与输入同构）；
- 框架负责：注入时机（cp_size>1 且解析链命中——`inner_target`/
  `inner_wrapper`/`_needs_cp_attn` 任一声明）与 target 解析。

#### 示例 1：`use_local_map`——自研 EP-aware MoE（a2a 在 forward 内）

```python
class MyEPAwareMoE(nn.Module):
    """forward 内含自己的 all_to_all dispatch/combine（EP-aware）。"""
    def forward(self, hidden_states):
        # local tensor 世界：自己的 a2a + expert 计算 + 归约
        ...

spec = ModuleShardingSpec(
    params={"experts.w1": {EP: Shard(0)}, ...},
    in_src={"hidden_states": {TP: Shard(1), CP: Shard(1)}},
    in_dst={"hidden_states": {TP: Shard(1), CP: Shard(1)}},   # identity
    out_src={TP: Partial(), CP: Shard(1)},
    out_dst={TP: Shard(1), CP: Shard(1)},
    use_local_map=True,        # ← 声明：走骨架，region 内跑自己的 forward
)
plan = ShardingPlanner(plan_overrides={"model.layers.0.moe": spec}).plan(...)
```

#### 示例 2：`local_compute_fn`——自研 MoE 注入自定义 dispatch

```python
def my_moe_compute(module, hidden_states):
    """骨架内执行：自定义 router + DeepEP dispatch + expert 计算。

    输入/输出恒为 local tensor；validate 的参数解包、边界缝合由骨架负责。
    """
    logits, scores, indices = my_custom_router(module.gate, hidden_states)
    dispatched, handle = deepep_dispatch(hidden_states, indices,
                                         group=module._ep_group)
    out = my_experts_forward(module.experts, dispatched)
    combined = deepep_combine(out, handle, group=module._ep_group)
    return (combined * scores).sum(dim=0)

spec = ModuleShardingSpec(
    params={...}, in_src={...}, in_dst={...}, out_src=..., out_dst=...,
    local_compute_fn=my_moe_compute,   # ← 声明即生效（门控派生，无需 use_local_map）
)
```

#### 示例 3：`inner_target`——自研 attention 走内置 CP wrapper

```python
class MyAttention(nn.Module):
    def __init__(self, config):
        self.core_attn = MyCoreAttention(config)   # forward(q, k, v, is_causal=...)
        self.q_proj = self.k_proj = self.v_proj = self.o_proj = nn.Linear(...)
    def forward(self, hidden_states):
        q, k, v = ..., ..., ...
        return self.o_proj(self.core_attn(q, k, v, is_causal=True))

spec = ModuleShardingSpec(
    params={"q_proj.weight": {TP: Shard(0)}, ..., "o_proj.weight": {TP: Shard(1)}},
    in_src={"hidden_states": {TP: Shard(1), CP: Shard(1)}},
    in_dst={"hidden_states": {TP: Replicate(), CP: Shard(1)}},
    out_src={TP: Partial(), CP: Shard(1)},
    out_dst={TP: Shard(1), CP: Shard(1)},
    inner_target="core_attn",   # ← 纯位置声明（门控派生，不改写标记）；
)                           #   K/V all-gather + D-04 causal 修正由内置 wrapper 完成
```

#### 示例 4：`inner_wrapper`——flash_attn_varlen 整体接管

```python
from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_models.components.distributed.cp_utils import flex_cp_allgather

def my_flash_cp_wrapper(target_module, cp_mesh):
    """内部直调 flash_attn_varlen_func 的 attention：整体接管 CP 包装。"""
    original_forward = target_module.forward

    def cp_forward(q, k, v, **kwargs):
        was_dtensor = isinstance(q, DTensor)          # 双模式容错
        q_pl = tuple(q.placements) if was_dtensor else None
        ql, kl, vl = (t.to_local() if isinstance(t, DTensor) else t
                      for t in (q, k, v))
        gk, gv = flex_cp_allgather(kl.contiguous(), vl.contiguous(), 2, cp_mesh)
        out = original_forward(ql, gk, gv, **kwargs)   # varlen kernel 原样调用
        if was_dtensor and isinstance(out, torch.Tensor):
            out = DTensor.from_local(out, q.device_mesh, q_pl)
        return out

    target_module.forward = cp_forward

spec = ModuleShardingSpec(
    ...,
    inner_target="core_attn",              # target = core_attn 子模块
    inner_wrapper=my_flash_cp_wrapper,    # ← 内置 2×2 dispatch 完全跳过
)
```

---



---

## 9. 端到端流程

> 汇总了本文档各阶段的完整调用链（已被 §2 总入口调用时序取代，保留作为补充视角）

```python
# 用户代码
model = HyperAutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B",
    distributed_setup={"tp": 4, "cp": 1, "enable_sequence_parallel": True},
)

# 内部流程：
# ① 分布式初始化 + DeviceMesh 构建
# ② AutoConfig.from_pretrained → hf_config
# ③ meta device 空壳构建（零显存）
# ④ ShardingPlanner.plan(model, mesh) → ShardingPlan
#     Phase 1-2: 参数分类 + 边界分组
#     Phase 3-4: 语义推断 + 模板填充 (in_src, in_dst, out_src, out_dst)
#     Phase 5: 链式传播校验
#     Phase 6: 特殊处理器
# ⑤ apply_sharding_plan(model, plan, mesh, validate_mode=False) → (model, tp_grad_info)
#     Phase A: distribute_tensor() → 参数 DTensor
#     Phase B: 特殊处理器
#     Phase C 入口: _local_params_context() → DTensor 永久 unpack + build_tp_grad_info()
#     Phase C: _apply_phase_c() → forward 包装（CP/MoE/validate/production 四分支）
# ⑥ FSDP2Manager.parallelize() → 在 meta 上 fully_shard（canonical：先于 to_empty/load）
# ⑦ model.to_empty(device) → 材质化 sharded 参数
# ⑧ checkpointer.load_base_model() → 每 rank 独立读 safetensors → 写入本地份
# ⑨ 返回可训练模型
#
# ★ 顺序以 06 §5.2 canonical meta 链路为准：fully_shard(meta) → to_empty → load。
#   （第六轮 P1 修复：旧文本 ⑥to_empty→⑦load→⑧parallelize 与 06 §5.2 相反，
#    已对齐为 parallelize 在 to_empty/load 之前，避免先 load 全量再 shard 的二次显存峰值。）
```

---



---

## 10. 新模型上线流程

以新增 **PanguForCausalLM** 为例：

| 场景 | 需要做的 | 代码量 |
|------|---------|--------|
| 命名完全标准 | 零配置 | 0 行 |
| 命名非标准 | 注册 `ARCH_OVERRIDES` 规则 | ~5 行 |
| 有特殊参数（fused QKV、SSM state 等） | 注册 `SpecialHandler` | ~20 行 |
| 完全自定义模块（无权重/特殊通信） | 手动构建 `ModuleShardingSpec` | ~15 行/模块 |

```bash
# 验证（validate 模式）
# 注：HYPER_VALIDATE_PLACEMENT 环境变量规划未实现——当前开关为 API 参数
# apply_sharding_plan(..., validate_mode=True)（§4.1）
torchrun --nproc_per_node=4 train.py --validate-mode
# → DTensor 双重校验 (out_src + out_dst) 全部通过 ✅

# 训练
torchrun --nproc_per_node=4 train.py
# → 生产模式，零 DTensor dispatch 开销
```

---



---

## 11. 总结

| 层面 | 数据 | 核心职责 |
|------|------|---------|
| **NamedPlacement** | `{TP: Shard(0)}` = TP mesh 轴沿 tensor dim 0 切分 | 声明式 placement，与 tensor shape 直接对应 |
| **ModuleShardingSpec** | `params` + `in_src`/`in_dst` + `out_src`/`out_dst` | 完整 I/O 契约，运行时直接使用，无推断 |
| **is_boundary** | `True`/`False` | 控制是否包装 forward + 构建 PrecompiledBoundary |
| **ShardingTemplate** | 语义角色(attention/mlp/norm/embed/lm_head/moe_gate) → 完整 Spec | 自动填充 I/O 契约，含 params + 四元 placement + 通信计划 |
| **链式传播** | 填充缺省 in_src + 校验 A.out_dst ≈ B.in_src | 自动填充 + 编译期契约一致性校验 |
| **PrecompiledBoundary** | `list[RedistOp]` | 编译期通信计划，运行时零判断 |
| **RedistOp.collective_type** | `str`（调试标签） | 通信统一走 `DTensor.redistribute()`，PyTorch 自动选最优 collective |
| **Validate 模式** | out_src 校验（核心）+ out_dst 校验（仅末端模块）| out_src 不可替代；out_dst 由链式传播覆盖（中间模块） |
| **用户自定义模块** | 3 种方式（自动/规则/手动） | 灵活适配无权重、特殊通信等场景 |

---

## 12. 实现回写（实施校准记录）

> 本节记录 `hyper_models/components/distributed/` 实现阶段对本文档的校准（实现以代码为准，
> 测试基线见 `tests/components/distributed/`）。
> 已在正文中就地修订的位置（§3.5 模板 D-05~D-08、§4.4、§3.6）引用本节编号。

### 12.1 自研 DTensor API 适配（与正文伪代码的签名差异）

正文伪代码按 PyTorch DTensor 风格书写；实现使用自研前向-only DTensor
（`hyper_parallel.core.dtensor`），签名差异如下：

| 伪代码 | 实现 |
|--------|------|
| `DeviceMesh("cpu", shape, mesh_dim_names=...)` | `init_device_mesh(device_type, mesh_shape, mesh_dim_names=...)`（裸构造缺 rank_list，`distribute_tensor` 会失败） |
| `DTensor.from_local(t, mesh, placements, run_check=False)` | `DTensor.from_local(t, mesh, placements)`（无 run_check） |
| `dt.redistribute(placements=..., async_op=False)` | `dt.redistribute(mesh, placements)`（mesh 为第一参数） |
| 直接在传入 mesh 上分发 | `apply_sharding_plan` 入口先取 `_get_active_mesh(mesh, plan.mesh_dim_names)` 活跃子 mesh——planner 剔除 size=1 轴后 placements 元数与 mesh 维数必须对齐，否则 `distribute_tensor` 静默错轴分片（EP 分片曾因此失效） |

另外 `flex_cp_allgather` 的反向由自定义 autograd.Function 显式实现
（all-gather 前向 + reduce-scatter 语义反向）；plain `dist.all_gather` 无 autograd 核。

### 12.2 新增设计修订（实现期发现；D-09 为功能新增）

| # | 问题 | 决策 |
|---|------|------|
| D-02 | production 下 embed 边界参数解包后，DTensor dispatch 隐含的 vocab 范围 mask 逻辑丢失——HF 原生 `F.embedding` 收到全局 token id 会索引越界 | **vocab-parallel embedding masked wrapper（§4.4 D-02 小节）**：`_wrap_vocab_parallel_embedding`（Megatron 风格）——本地 vocab 区间外的 token 置 0、索引减偏移，输出乘 mask 形成天然 Partial 贡献，boundary 出口归约不变；仅 production 注入（`_is_vocab_parallel_embed` 判定：nn.Embedding + TP Shard(0) + TP>1），validate 走 dispatch 无需。UT：`test_dist_s5_vocab_embed.py` |
| D-03' | 原 `_wrap_moe_forward` 与 MoE 强耦合，自研数据相关模块（非 MoE 的 a2a/自定义 dispatch）无法复用骨架 | **local region 泛化（§4.4.3）**：拆分为 `_resolve_local_compute_fn`（compute_fn 三路选择：用户 `local_compute_fn` > TP-extend-EP 注入 `_hf_native_ep_compute` > 模块自身 forward）+ `_wrap_local_region_forward`（与 MoE 无关的通用骨架：`validate_mode`/`compute_fn` 参数、validate 下 `_temp_local_params`、最终出口恒 to_local）；一切 `spec.use_local_map=True` 的模块复用同一骨架 |
| D-05 | embed 模板 CP 契约与 §6.3.4 数据管道矛盾：batch 已被 `shard_batch_for_cp` 按 CP 切分，in/out CP 维若声明 Replicate，boundary 会把已切分的 chunk 再 scatter 一次（序列被切两次） | CP>1 时 embed 的 in_src/in_dst/out_src CP 维 = `Shard(1)`（`_build_spec_from_template` 按 has_cp 应用）；out_dst 不变（TP reduce-scatter，CP identity） |
| D-06 | mlp/moe_mlp 模板 in_dst CP=`Replicate` 会在 TP×CP 下产生 tp-major 序列布局，与 embed/attention 产出的 cp-major 布局不一致（数值错误），且 MLP 是 pointwise 无需 CP 通信 | mlp/moe_mlp 的 CP 维全程 `Shard(1)`（in_dst/out_src 同步修改） |
| D-07 | lm_head 模板 in_dst CP=`Replicate` 违反 R8（boundary 层 CP 维恒 identity——CP 序列 all-gather 仅发生在 attention 内部 K/V） | lm_head 的 CP 维全程 `Shard(1)`；CP 下 lm_head 在本地 CP chunk 上计算 logits/loss（Megatron CP 标准做法），输出为 chunk logits |
| D-08 | 3D expert 权重 `[E, H_out, H_in]` 的 TP placement 按 2D 写的 `Shard(0)/Shard(1)` 会错切 expert 维/H_out 维（数值错误） | MOE_EXPERT 的 TP placement 按参数 ndim 感知：ndim≥3 → colwise=`Shard(1)`、rowwise=`Shard(2)`；ndim=2（per-expert 布局）→ 标准 `Shard(0)/Shard(1)`（此时 EP Shard(0) 语义不成立，EP 需按"每 rank 持 expert 子集"的 module 级实现，归 ARCH_OVERRIDES/SpecialHandler） |
| D-09 | HF 原生 MoE（per-expert 2D Linear 列表、无 all_to_all、无 dispatcher 钩子）在 EP>1 下现有路径全部失效（D-08 已标注 2D EP 语义不成立）；veomni/NeMo 均要求模型侧改造 | **EP 直通方案（§6.4.7）**：planner 检测 per-expert 模式并生成 stacked 元数据（`spec._ep_stack`/`spec._moe_router`）；Phase A 前置 `_stack_moe_experts` 把 per-expert 权重 stack 成 3D（落回 D-08 ndim=3 规则）；`_wrap_local_region_forward` 注入 `_hf_native_ep_compute`（router adapter + 本地 SwiGLU）。a2a 按后端分派：NCCL/HCCL 不等长 all_to_all（零填充），gloo pad-to-max `all_to_all_single`（gloo 实测不支持不等长 list 版） |
| D-10 | D-09 的 EP 组不跨 TP（a2a 仅限同 TP 坐标 rank），EP 规模受 dp 限制；expert 权重若做 hidden 维 TP 切（Megatron ETP）需引入 AG/RS 对与第二轴 mesh | **TP-extend-EP 方案（§6.4.8，与 MindSpeed「TP 扩展 EP」/ Megatron etp=1+ep 跨 TP 同构）**：MoE 边界契约改 SP-in identity（Megatron MoE 本就不 gather）；全 dense 区域 flatten 重分区为派生 expert mesh (edp, ep)——扩展 EP 组 = flatten 连续 ep_size 个 rank（先跨完 TP 组再跨 dp/cp）；expert 权重仅 {EP: Shard(0)}（每 rank 持 num_experts/ep_size 个完整 expert，无第二轴切分）；通信流与 Megatron `MoEAlltoAllTokenDispatcher`（etp=1 配置）逐步对齐：router（本地 chunk）→ a2a（扩展 EP 组）→ 本地 SwiGLU（完整权重，无 Partial）→ a2a 返回，**无 all_gather/reduce_scatter**；SP-in 无 token 复制（规避 automodel 的 tp 倍梯度缩放）；**`ep_size` 即扩展 EP 组大小（无单独 etp 配置），校验 `ep_size ≤ dp_replicate×dp_cp×tp` 且整除、`num_experts % ep_size == 0`** |
| D-12 | §6.7/§7 原叙事的 `tp_grad_info → fully_shard(tp_grad_info=...) → _build_layout_driven_group_info → TP all-reduce` 链路端到端走不通：`fully_shard` 不接受 `tp_grad_info` 入参（全仓库无消费者，仅 `grad_equiv.py` 测试模拟旁路）；且 §7 Stage 2 原称"fully_shard 之前永久解包为 plain tensor"——若真如此 `_orig_param_is_dtensor=False`，DTENSOR_UNIFIED 路径死亡 | **按代码现状重写 §6.7/§7**：FSDP 侧实际机制为 DTENSOR_UNIFIED——`fully_shard` 在参数仍为 DTensor 时记录 `_orig_dtensor_placements`（torch param.py:149-151），`_get_base_spmd_placements` 经 `DeviceMesh.concatenate`（已实现，device_mesh.py:1035）拼统一 mesh，`_build_layout_driven_group_info`（hsdp_param.py:285）从最终 layout 的 Replicate 轴推 TP 归约组，`_normalize_unsharded_grad_to_local` 处理 Partial 梯度回流；tp_grad_info 定位为"planner/applier 已产出、fully_shard 消费端接线待落地"的规划项（数据结构保留）。集成落地二选一：(a) fully_shard 先于 `_local_params_context` 解包执行（当前 apply_sharding_plan 在 Phase C 入口永久解包，需调时序或改临时解包）；(b) 补齐 tp_grad_info 消费端 |
| D-13 | DeepSeek MLA 投影（`q_a_proj`/`q_b_proj`/`kv_a_proj_with_mqa`/`kv_b_proj`）不含任何默认命名规则子串，未覆盖时全部落 SKIP——`self_attn` 组只剩 `o_proj`(ROWWISE)，`has_colwise=False` → attention 边界推断失败，MLA 参数**静默全部不分片**（仅 warning），`_needs_cp_attn` 不置位（CP wrapper 不注入）；教训：命名覆盖缺口的失败模式是"整条边界消失"而非"少切一个参数" | **方式 B 内置条目（§3.6.1 注记）**：`q_a`/`kv_a` 下投影 → REPLICATED（LoRA rank 维不切，latent TP 组内一致），`q_b`/`kv_b` 上投影 → COLWISE（head 维），`o_proj` 仍 ROWWISE（contract head 维契约不变，与标准 attention 模板同构）；配套新增第 14 个角色 `ParamRole.REPLICATED`（全维 Replicate，仅经 ARCH_OVERRIDES 显式指派，默认规则不产生）；键同时注册 architectures 拼写（`"deepseekv2"`/`"deepseekv3"`）与 model_type 拼写（`"deepseek_v2"`/`"deepseek_v3"`，v2/v3 同构）。UT：`test_s1_mla_deepseek.py`（S1.14，含"无覆盖时落 SKIP"的回归保护） |

**被推翻的历史方案（如实补记）**：commit bc749470 曾实现 **ETP 方案**——
`ep_size` 即 ETP 组大小、expert 权重声明 `{EP: Shard(0), ETP: Shard(1/2)}`
（expert 维 + hidden 维双轴切分）、引入 `MeshAxisName.ETP` 独立 mesh 轴。
该方案已被 **D-10 TP-extend-EP 取代**（expert 仅 expert 维切分 +
`spec._ep_size` 元数据，无第二轴、无独立 etp 配置），对应
`test_s6_etp.py` / `test_dist_s6_etp.py` 删除、`test_s6_ep_extend.py` /
`test_dist_s6_ep_extend.py` 新增。另：`MeshAxisName.EP_SHARD` 为该时期的
历史残留枚举值，当前代码无任何使用。

### 12.3 Planner 实现修正

1. **Phase 2 边界分组（§3.6.6 伪代码缺陷修正）**：伪代码对单参数临时 group 做
   边界推断，`q_proj` 叶模块会被"仅 colwise → mlp"规则误判为独立边界。实现改为
   **两趟分组**：趟 1 按直属模块 FQN 分组；趟 2 工作队列深度优先，unknown 则整组
   向上合并到父模块（兄弟参数合并齐备后再推断）。回溯到根仍 unknown 归入参数
   所在模块（无模板命中 → warning 跳过，等价不分片）。
2. **链式传播的名字无关单 entry 配对（§3.6.5 修订）**：模板 in_src key 与上游
   out_dst key 可能不同名（attention 的 `"output"` vs moe_mlp 的 `"x_BLD"`）。
   双方都恰好 1 个 entry 时按"唯一 arg"配对（名字无关），否则按 key 名配对；
   in_src 整体为空时按下游 in_dst 声明的 key 填充。
3. **`_is_terminal` 按链式相邻标记**（不做跨模块 placement 值相等匹配——
   lm_head 的 Replicate out_dst 会被 embed 的 Replicate in_src 误引用）。
4. **`_bind_input_indices`**：PrecompiledBoundary 的 in_plan arg_name 在包装时
   绑定到 forward 签名的 positional 下标（模块间调用多为 positional，kwargs 按名
   查找会 miss）；单输入契约（in_plan 仅 1 op）回退绑定到首个 positional 参数——
   覆盖模板 key（如 `"hidden_states"`）与叶模块签名（`nn.Linear.forward(input)`）
   不同名的场景。`redistribute_inputs` 对未找到的 arg 跳过（不注入 None）。
5. **`_classify_collective` 只比较有差异的维度**：identity 维（如 attention 的
   CP 维 Shard(1)→Shard(1)）不参与分类，TP 维 Shard→Replicate 才能正确归类为
   all_gather。
6. **validate 的 placement 比较前做负维度归一化**：`Shard(-1)` == `Shard(ndim-1)`。
7. **tied 检测需 `named_parameters(remove_duplicate=False)`**：tied 参数在默认
   去重下只出现一次。
8. **新增 Phase 4.5 `plan_overrides` 合并**（2026-07-19，§3.6.7/§8.5 特性）：
   `ShardingPlanner(plan_overrides={fqn: spec})` 把用户手写 spec 在 Phase 5
   之前合并（整体替换/插入），覆盖 spec 照常参与链式契约校验与 `_is_terminal`
   标记；`use_local_map`/`_needs_cp_attn` 从推断模板强制补齐（结构属性，缺失
   导致数值错误，不允许借覆盖关闭）；用户 spec 深拷贝隔离，plan() 可重复调用；
   fqn 未命中 `named_modules` fail-fast 抛 `ValueError`。
   UT：`test_s1_plan_overrides.py`（9 例）+ `test_dist_s5_plan_overrides.py`
   （多输入 attention 双模式 e2e，TP=2）。
9. **数字段守卫（D-09 配套，2026-07-20）**：`_infer_boundary_type` 对纯数字
   末段（`experts.0..N` 等 per-expert 容器）返回 unknown——否则 HF 原生 MoE
   的每个 expert 会被误判为独立 moe_mlp 边界，per-expert 参数无法聚合到 mlp
   边界做 D-09 标记。HF MoE 的完整分组链：`experts.0.gate_proj`（叶守卫）→
   `experts.0`（数字守卫）→ `experts`（叶守卫）→ `mlp`（moe_mlp 边界）。
10. **TP-extend-EP 实现要点（D-10，2026-07-20）**：(a) 派生 expert mesh
    拆分为 `_expert_mesh_layout`（纯 rank 映射，单进程可测）+
    `_build_expert_mesh`（init_device_mesh 任意 rank_list 建组），轴
    (edp, ep)；(b) gloo 实测：不等长 list 版 all_to_all **不支持** →
    a2a 走 pad-to-max + all_to_all_single 回退（NCCL/HCCL 用不等长
    a2a 降低通信量）；(c) tp_grad_info 对 expert 参数打 Shard 标记——
    无 TP 键缺省 Replicate 会导致 FSDP 对分片梯度错误 all-reduce；
    (d) 边界契约以模板 in_src 为准推导 identity（SP → Shard(1)，
    non-SP → Replicate），不硬编码。

### 12.4 Applier 实现修正

1. **`RedistOp.execute` identity 分支**：输入为 DTensor 且 `as_dtensor=False`
   （production）时返回 `to_local()`——MoE/CP local region 出口的 `from_local`
   重包装经 identity boundary 时必须解包，否则 DTensor 泄漏到下游产生
   mixed dispatch。
2. **`redistribute_outputs` 在 validate（as_dtensor_input=True）下保持 DTensor**：
   terminal 模块的 out_dst 校验发生在 redistribute 之后，需要 DTensor 输入。
3. **`_broadcast_tied_param` 不跨 rank 广播**（正文 §4.4 的实现是错误的）：
   tied 对（embed/lm_head）同为 Shard(0) 分片，各 rank 的 local shard 承载不同
   vocab 区间——把 rank0 的 shard 广播给 rank1 会破坏 rank1 的分片。tied 语义要求
   **同一 rank 内**两端共享物理存储（梯度共享），分片天然一致（同一 global 来源、
   同一 placement）。实现改为 rank 内 `param_b.data = param_a 的 local tensor`。
4. **`_temp_local_params`**：validate 的 local region（MoE all-to-all、HF CP
   attention）内部需要 local 参数——region 内临时解包 DTensor 参数、退出恢复
   （DTensor 传播链不断）；production 已在 build 期永久解包，无需此 context。
5. **CP (q,k,v) wrapper 的 mask 契约**：`is_causal` 且 CP 激活（cp_mesh.size()>1，
   2026-07-21 由 q_len≠kv_len 形状比较修订为语义判断，见 §4.4.2 设计点 7）时
   wrapper 将 `is_causal` 替换为 `attn_mask`（D-04 offset-aware mask），要求
   inner forward 接受 `attn_mask` kwarg。
6. **MoE wrapper 边界最终出口恒为 local**（out_plan 为空时 from_local 包装也需
   在出口解包）。
7. **文档勘误（2026-07-20）**：§2 时序图与 §4.4.2 伪代码中"validate 跳过 CP
   wrapper、K/V all-gather 由 DTensor dispatch 自动处理"的描述与 D-01'' 矛盾，
   已就地修正——SDPA dispatch 对 CP Shard(1) 的 K/V 不做 all-gather（会算成
   局部 attention），两模式必须注入同一个显式 all-gather wrapper；validate
   下 MoE 走 `_wrap_local_region_forward(validate_mode=True)` 也在伪代码中补全。

### 12.5 双模式梯度语义（§5.5 补充）

- 两模式 backward 均为 local autograd（§1.0），双模式梯度逐参数相等（S5.3 实测
  rtol=1e-3 通过，覆盖 TP-Shard 与 TP-Replicate 两类参数）。
- **replicated loss 的梯度缩放**：loss 在每个 rank 上对 all_gather 后的完整
  logits 重复计算时，all_gather 的反向（reduce_scatter）把各 rank 相同的梯度流
  求和——分布式梯度 = world_size × 单卡梯度（两模式语义一致，不影响双模式等价；
  真实训练中由 loss_parallel 或 DP 梯度平均吸收该缩放）。
- **EP 组内数据复制的 expert 梯度缩放（D-09 实测）**：EP 组内各 rank 持相同
  batch（无 DP 轴的测试拓扑）时，每个 expert 经 a2a 收到 ep_size 份相同
  token（每 rank 各一份）→ expert 权重梯度 = ep_size × 单卡；router 梯度是
  per-copy 的 → 1×。真实训练中 EP 组内数据由 DP 区分，无此缩放（梯度平均由
  FSDP 处理）。TP-extend-EP（D-10）下同理：扩展 EP 组共享 batch 的测试拓扑中
  expert 梯度按组大小放大（a2a pad 路径 fwd/bwd 对拍见
  `test_dist_s6_hf_native_moe.py`）。
- G4 实测确认：torch `is_causal` 在 q_len≠kv_len 时按**左上角对齐**（等价于假设
  chunk 位于序列开头），rank>0 的 CP chunk 必须走 D-04 的 offset-aware mask。

