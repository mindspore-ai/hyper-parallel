# hyper_models 双模式 DTensor 并行策略 —— 详细设计文档（当前实现版）

> **文档层级（2026-08-10）**：本文是**设计决策**（D-xx）的 canonical 记录；
> 使用/机制语义的现行口径以
> [components_distributed_tutorial.md](guide/trainer/components_distributed_tutorial.md)
> 与 [patch_injection_mechanism.md](patch_injection_mechanism.md) 为唯一权威，
> 说法冲突时以后两者为准。
>
> 代码位置：`hyper_models/components/distributed/`
> 历史设计稿：`docs/trainer.local.bak/detailed_design/05_dual_mode_dtensor_parallel_strategy.md`（下称"05 文档"）。
> 本文档以**当前代码为准**（canonical），覆盖 05 文档定稿后的全部设计修订
> （D-01'' ~ D-17）与"显式注入重构"（explicit-injection rework）；与 05 文档
> 冲突处以本文为准。修订清单见 §13。
>
> 日期：2026-08-07

---

## 1. 设计目标与核心思想

### 1.1 问题定义

对任意 HF 风格模型，零模型代码改动地施加 TP/CP/EP 并行，要求：

1. **编译期推导**：由 `ShardingPlanner` 从模型结构自动推导完整分片计划
   （`ShardingPlan`），不允许运行时逐模块试探；
2. **生产零开销**：production 模式前向不含任何 DTensor dispatch（
   `__torch_function__`）——参数在 build 期一次性解包为 plain local tensor，
   通信由编译期预生成的 `PrecompiledBoundary` 直接执行；
3. **可校验**：validate 模式保持参数为 DTensor，依赖 DTensor dispatch 的
   placement 传播对声明的 I/O 契约做运行期校验，并与 production 做数值对拍；
4. **双模式等价**：两模式区域内计算路径**逐指令一致**（kernel 级等价），
   唯一差异是边界缝合方式；
5. **可扩展**：用户自定义模块（自研 attention / MoE / autograd.Function）经
   声明式接口接入，框架对数据相关逻辑零推导零猜测。

### 1.2 为什么有"双模式"

本仓的 DTensor（`hyper_parallel.core.dtensor`）是**前向-only** 的
placement/dispatch 系统——反向不走 DTensor（无 DTensor autograd），两模式
反向均为 local autograd、梯度直接落 local 分片。因此"校验 placement 传播
正确性"只需要前向。双模式的分工：

| | production | validate |
|---|---|---|
| 参数 | build 期一次性永久解包为 plain local tensor（`_local_params_context`） | 保持 DTensor |
| 前向 | 纯 local tensor + `PrecompiledBoundary` 通信 | DTensor dispatch 传播 + out_src/out_dst 校验 |
| 反向 | local autograd（梯度落 local 分片） | local autograd（同左，05 §1.0） |
| 返回值 | `(model, tp_grad_info)`（供 FSDP2 `fully_shard`） | `(model, None)` |
| 用途 | 训练主路径 | 接入新模型/新配置时的正确性校验与数值对拍 |

### 1.3 架构约束（双模式等价的前提）

凡 DTensor dispatch **隐含或无法表达**数据相关逻辑的模块——embedding 的
vocab mask、attention 的 K/V gather、MoE 的 all-to-all——两种模式必须用
**同一份** local-region wrapper 显式重建该逻辑，保证区域内计算路径逐指令
一致。这衍生出三条硬性修订：

- **D-01''**：validate 的 CP 与 production 注入**同一个** all-gather wrapper
  （否决 ring attention / dispatcher 分离方案）；
- **D-02**：production 的 embed 注入 Megatron 风格 masked embedding wrapper
  （参数解包后 DTensor dispatch 的 vocab mask 语义丢失，必须显式重建）；
- **D-03'**：MoE 统一走 local region（前向-only，无反向缝合——对比 PyTorch
  `local_map` / Titan `LocalMapConfig.in_grad_placements`，那些机制存在是因为
  torch DTensor 有反向语义，本仓没有也不需要）。

### 1.4 组件独立性

`components/distributed` 零依赖 `recipes/` / `_transformers/` / `models/` /
`datasets/`（由 `test_s5_zero_dep_lint.py` 守住），可脱离训练流程独立使用：

```python
from hyper_parallel.core.dtensor.device_mesh import init_device_mesh
from hyper_models.components.distributed import ShardingPlanner, apply_sharding_plan

mesh = init_device_mesh("cpu", (4,), mesh_dim_names=("tp",))
plan = ShardingPlanner().plan(model, mesh, tp_size=4)          # 编译期推导（6-phase）
model, tp_grad_info = apply_sharding_plan(model, plan, mesh)   # production 应用
# validate 校验：apply_sharding_plan(model, plan, mesh, validate_mode=True)
```

---

## 2. 总体架构

```
                 ┌──────────────────── 编译期 ─────────────────────┐
  HF 模型 ──► ShardingPlanner.plan(model, mesh, tp/cp/ep_size, …)
                 │  Phase 1  参数角色分类（ParamRole × 14）
                 │  Phase 2  通信边界分组（两遍法）
                 │  Phase 3  语义角色推断（→ 7 种模板）
                 │  Phase 4  模板填充 spec（含 MoE EP 标记 D-09/D-10）
                 │  Phase 4.5 plan_overrides 合并（merge / insert / glob）
                 │  Phase 5  _is_terminal 标记（D-14：链式传播已删除）
                 │  Phase 6  特殊参数 handler 收集
                 ▼
            ShardingPlan = {module_fqn: ModuleShardingSpec}
                 │  + sequence_parallel / loss_parallel 开关
                 │  + special_handlers / tied_pairs / mesh_dim_names
                 ▼
                 ┌──────────────────── 运行期（apply）─────────────────────┐
  apply_sharding_plan(model, plan, mesh, validate_mode=?)
                 │  前置：_preflight_compute_injection（CP/EP 显式注入守门）
                 │  Phase 0  out_src/out_dst 标量简写规范化
                 │  Phase A  参数分片（distribute_tensor；EP 堆叠/专家 mesh）
                 │  Phase B  特殊参数 handler（SPECIAL_HANDLERS）
                 │  Phase C 入口 production 一次性解包 + build_tp_grad_info
                 │  Phase C  forward 包装（五路，post-order）
                 │  Phase D  tied weights 存储共享
                 ▼
        (model, tp_grad_info)  ──► FSDP2 fully_shard（dp 语义）
```

关键分层：**plan 管"声明"，applier 管"执行"，DTensor 只做通信原语**
（`redistribute`）。运行时不做任何 placement 推导。

---

## 3. 坐标系约定：plan 恒为单个 dp 切片（05 §3.1.1）

- `ShardingPlanner._build_mesh_dim_names` 从 `mesh.mesh_dim_names` 中**只保留
  `tp` / `cp` / `ep` 三个 DTensor 轴**，且丢弃 size=1 的轴；`dp*` / `pp` 轴
  一律剥离。plan 的坐标系 = 单 dp 切片：
  - dp 的**数据切分**由数据管道表达（数据分配）；
  - dp 的**参数/梯度切分**由 FSDP2 `fully_shard` 表达（作用于 apply 之后）；
  - pp 由流水并行运行时表达（PP 场景对每个 single-part 模型分别 plan）。
- `plan_overrides` 中声明任何 DP placement → `_check_overrides_no_dp`
  fail-first（DP key 会在 resolve 时被静默丢弃，必须显式拒绝）。
- `tp_size` / `cp_size` 与 mesh 对应轴尺寸必须相等，否则 fail-first
  （`_validate_dtensor_axes`）；`ep_size` 仅在 mesh 显式含 `"ep"` 轴时校验
  （老式 EP）；D-10 TP-extend-EP 下 ep 组从 dense region **派生**，无 `"ep"`
  轴是预期形态。
- 退化单 rank mesh（全维 size 1，编译期单测常见）跳过校验。

---

## 4. 数据模型（`sharding_config.py`）

### 4.1 基本类型

- **`MeshAxisName`**（str 枚举）：`TP/CP/EP/PP/DP/DP_REPLICATE/DP_SHARD/
  DP_SHARD_CP/DP_CP/EP_SHARD`。str 子类，与 `"tp"` 等 plain string 直接
  可比、可做 dict key。
- **`NamedPlacement = Dict[MeshAxisName, Placement]`**：key 是 mesh 维度名，
  value 中 `Shard(N)` 的 N 是**张量维度下标**。`resolve_placements(named,
  mesh_dim_names)` 按 mesh 轴序排列、缺失轴补 `Replicate()`。
- **YAML placement DSL**：`"replicate" / "partial" / "shard(N)"`（闭集文法，
  `parse_placement` 解析；拼写错误 fail-fast 并列出合法文法）。

### 4.2 `ModuleShardingSpec`：单模块完整 DTensor 契约（05 §3.2）

四个 placement 字段构成完整 I/O 契约，**运行时零推导、严格按声明执行**：

| 字段 | 含义 | 不匹配时 |
|---|---|---|
| `in_src` | 输入到达模块边界时的 placement（来自上游输出/数据管道） | — |
| `in_dst` | 模块内部计算所需的 placement | 触发通信（重分布） |
| `out_src` | 模块内部计算自然产出的 placement | validate 模式核心校验对象 |
| `out_dst` | 下游模块期望的 placement | 触发通信 |

辅助字段：

- `params: Dict[str, NamedPlacement]`——子模块路径 → 参数分片声明；
  `None` = 未声明（merge 时继承），显式 `{}` = "本边界不切参数"（纯 I/O
  缝合公民，如 ViT `params={}` 模式）。
- `out_names: List[str]`——多输出模块的输出名序，映射 `out_src/out_dst`
  的 key 到 tuple 位置（`RedistOp.arg_index`）。
- `is_boundary: bool = True`。
- 单输出模块允许 `out_src/out_dst` 写标量简写 `{TP: …}`，规范化阶段
  （`_normalize_out_fields`）包成 `{"output": …}`。

**"不写继承，写了照办"**（2026-08-05 定稿，仅存在于 override 输入侧）：
`None` 或哨兵 `"auto"` = 继承模板推导值；显式值（含 `{}`）即最终值；哨兵
`"none"` = 显式清空（`params/in_*` → `{}`，`out_*` → `None`）。哨兵在
merge/insert 时解析，**绝不进入 plan 输出**——plan 内的 spec 恒为具体值，
下游消费者零分支。

### 4.3 注入接口字段（用户扩展点，详见 §8）

| 字段 | 类型 | 家族 | 作用 |
|---|---|---|---|
| `region_dispatch` | `Optional[bool]`（注入时必填，无默认） | 伴生声明 | **区域 compute 可否 dispatch 穿透**：`False`=含通信/自定义 kernel → 黑盒托管（模块自身 forward 即数据相关逻辑时即区域 compute）；`True`=纯标准算子 → validate 穿透真校验；不注入无需声明 |
| `local_compute_fn` | callable / 工厂 Target | local-region | 替换骨架内计算函数（骨架边界缝合/双模式不变） |
| `inner_target` | `str`（属性名/`"self"`） | inner-wrap | **纯位置**：指定被替换 forward 的 inner 子模块 |
| `inner_wrapper` | 注册表名 / `@inner_wrapper` callable / Target | inner-wrap | **纯行为**：选定包装方案 |
| `inner_out_src` | `"first_input"` / NamedPlacement / `{name: …}` | inner-wrap | 情形 B（inner 子模块）输出布局**显式声明** |

内部标记（planner/applier 自动写，用户不可配）：`_is_terminal`、
`_needs_cp_attn`（模板识别元数据，**不触发任何注入**）、
`_resolved_inner_wrapper` / `_resolved_inner_target`（applier 回写，供
introspection）、`_ep_stack`（D-09 堆叠元数据）、`_ep_size`（D-10 扩展 EP
组大小）。

### 4.4 `ShardingPlan`：模型级计划（05 §3.1）

```python
@dataclass
class ShardingPlan:
    modules: Dict[str, ModuleShardingSpec]   # 仅含 is_boundary=True 的模块
    sequence_parallel: bool = True           # SP 开关：激活 TP 维 Shard(1)
    loss_parallel: bool = False              # lm_head 输出保持 Shard(-1) 还是 Replicate
    special_handlers: Dict[str, str]         # {param_fqn: handler_name}
    mesh_dim_names: Tuple[str, ...]          # 与 DeviceMesh.mesh_dim_names 一致（仅 tp/cp/ep）
    tied_pairs: List[Tuple[str, str]]        # embed <-> lm_head 共享存储对
```

### 4.5 `ShardingTemplate` 与七角色模板（05 §3.5）

`TEMPLATES` 内置 7 个语义角色模板，每个模板对 TP+CP+EP 三维全声明，
planner 按实际 `mesh_dim_names` 过滤。**横切规则**：CP 维参数恒
`Replicate`（CP 不切参数）；CP 维激活 `Shard(1)`（序列维）或 `Replicate`；
EP 维非 MoE 模块 `Replicate`，MoE 专家 `Shard(0)`。

| 模板 | 参数规则 | SP 模式 I/O 要点 |
|---|---|---|
| `attention` | q/k/v Colwise `Shard(0)`，o Rowwise `Shard(1)` | in `Shard(1)→Replicate`（TP 维 all-gather），CP 维恒 `Shard(1)`；out `Partial→Shard(1)`（reduce-scatter）；`needs_cp_attn=True`（K/V gather 在 inner wrapper 内，不在边界层） |
| `mlp` | gate/up Colwise，down Rowwise | 同上结构；CP 维全程 `Shard(1)`（D-06：pointwise，TP×CP 布局一致性） |
| `norm` | 权重 `Replicate` | 全程 identity，零通信 |
| `embed` | 权重 vocab 维 `Shard(0)` | out `Partial → Shard(1)`（reduce-scatter 到 SP+CP）；D-05：CP 激活时 input 已被数据管道切分，in/out CP 维改写为 `Shard(1)` |
| `lm_head` | 权重 `Shard(0)` | out_src `Shard(-1)`；out_dst 按 `loss_parallel`：true → `Shard(-1)`，false → `Replicate`；CP 维恒 `Shard(1)`（D-07/R8：loss 在本地 chunk 计算，无 CP gather） |
| `moe_gate` | router 权重 `Replicate` | 输出重分布到 EP：`{EP: Shard(0)}` |
| `moe_mlp` | router `Replicate`，专家 EP `Shard(0)` + TP ndim 感知（D-08） | `region_dispatch=False`（模板值——forward 自带 a2a 不可 dispatch；HF 原生布局会被 planner 清除为 None，见 §10）；D-10 下契约改写为 SP-in identity |

**D-08（MOE_EXPERT 的 ndim 感知 TP placement）**：3D batched 专家权重
`[E, H_out, H_in]` 的张量 dim 0 是专家维（归 EP `Shard(0)`），TP 维顺延：
colwise → `Shard(1)`，rowwise → `Shard(2)`；2D per-expert 布局保持
`Shard(0)/Shard(1)`（EP 须以"每 rank 持部分专家"的模块级方式实现，超出
模板覆盖，走 ARCH_OVERRIDES/SpecialHandler）。

---

## 5. ParamRole 与参数分类（`param_role.py`）

`ParamRole` 是命名规则与 `ShardingTemplate` 之间的桥梁（14 个枚举值）：

| 角色 | placement | 说明 |
|---|---|---|
| `COLWISE` | TP `Shard(0)` | q/k/v/gate/up proj |
| `ROWWISE` | TP `Shard(1)` | o/down proj |
| `NORM` | `Replicate` | RMSNorm/LayerNorm |
| `EMBED` / `LM_HEAD` | `Shard(0)`（vocab 维） | |
| `MOE_GATE` | `Replicate` | router |
| `MOE_EXPERT` | EP `Shard(0)` + TP colwise/rowwise（ndim 感知） | routed expert |
| `SHARED_EXPERT` | EP `Replicate` + TP colwise/rowwise | shared expert |
| `FUSED_QKV` / `FUSED_GATE_UP` | `Shard(0)` | 融合权重 |
| `BIAS` | 恒 `Replicate` | |
| `REPLICATED` | 全维 `Replicate` | **仅 ARCH_OVERRIDES 指派**（D-14：MLA q_a/kv_a 下投影） |
| `SPECIAL` | Phase 6 SpecialHandler | gated_delta / a_log / dt_bias（SSM/Mamba 系） |
| `SKIP` | 不进 spec.params | 默认兜底 |

`ParameterClassifier` 优先级：**arch_overrides > 默认命名规则（first match）
> SKIP**。默认规则排序原则：更具体的规则在前（shared_experts 先于
experts；带点 pattern 的 MoE gate 先于裸 "gate"；bias 先于 colwise/rowwise，
否则 `q_proj.bias` 会被 colwise 捕获）。

**ARCH_OVERRIDES**（D-14，DeepSeek MLA，v2/v3 同构）：`q_a_proj` /
`kv_a_proj_with_mqa` 强制 `REPLICATED`（LoRA rank 维不切），`q_b_proj` /
`kv_b_proj` 按 head 维 `COLWISE`——MLA attention 与标准 attention 模板同构
（o_proj rowwise 契约不变）。key 同时注册 architectures 拼写
（`deepseekv3`）与 model_type 拼写（`deepseek_v3`）。

---

## 6. ShardingPlanner：6-phase 推导管线（`sharding_planner.py`）

### Phase 1：参数角色分类
`ParameterClassifier.classify(model, arch)` → `{param_fqn: ParamRole}`。
arch 探测：`config.architectures[0]` > `config.model_type` > 类名，小写并剥
`ForCausalLM` 等后缀。

### Phase 2：通信边界分组（两遍法，修正 05 §3.6.6 伪代码缺陷）
- Pass 1：按属主模块 FQN 分组（剥叶子参数名）；
- Pass 2：深度优先工作队列——组内角色齐全则做边界推断；unknown 则整组
  参数**向上合并**入父模块并入队（父模块更浅、后处理，兄弟模块参数合并
  完整后再推断，避免 q_proj 叶子被单独误判为 mlp 边界）；回溯到根仍
  unknown 则归到参数自身模块（后续无模板匹配 → 警告跳过）。

### Phase 3：语义角色推断（`_infer_boundary_type`）
优先级：显式 FQN pattern（embed/lm_head/norm/router）> 叶子段守卫
（`_LEAF_SEGMENT_GUARD`：q_proj/experts/fc1/w1 等段名自身不是边界容器；
数字段守卫：HF per-expert 容器 `experts.0..N` 不是边界）> MoE 角色聚合
（含 MOE_* 角色的组向上聚合到 moe 容器）> 参数角色组合（colwise+rowwise
→ attention/mlp，缺省 attention——SP 通信更保守）。

### Phase 4：模板填充（`_build_spec_from_template`）
按 ParamRole 填 `spec.params`；按 SP 开关选 sp/nosp 契约（深拷贝，防共享
模板被污染）；lm_head 按 `loss_parallel` 改写 out_dst；embed 按 D-05 改写
CP 契约；`moe_mlp` 额外经 `_mark_hf_native_moe` 做 EP 标记（§10）。

### Phase 4.5：用户 plan_overrides 统一合并（05 §3.6.7，唯一注入/覆写接口）

三种模式：

- **merge**（key 命中推导边界）：契约字段 `None`/`"auto"` 继承、显式值
  （含 `{}`）字段粒度替换、`"none"` 清空；注入字段（`local_compute_fn` /
  `inner_target` / `inner_wrapper` / `region_dispatch` 非 None）恒胜；
  内部标记恒继承。这是 CP/EP 计算注入的声明方式——只写注入字段即继承
  整套推导契约；
- **insert**（精确 key 未命中任何边界）：spec 原样深拷贝插入，必须完整自
  声明（全部未声明 → fail-fast "no template matched"；哨兵非法——没有可
  继承的对象）；D-14 起允许嵌套（祖先/后代 FQN），仅受参数唯一性约束；
- **glob key**（含 `*?[`）：fnmatchcase 合并到每个命中边界（`*` 跨点）；
  命中为零则大声警告；glob 永不 insert。

配套防呆：精确 key 必须在 `named_modules` 中存在（拼写 fail-fast）；
`_validate_override_axes` 对未注册轴名/非 Placement 值 fail-fast（typo 轴会
被 `resolve_placements` 静默忽略，必须拦截）；`_warn_dropped_params` 对
字段粒度替换丢弃的推导分片发 WARNING；用户 spec 对象永不被改写（merge 读、
insert 深拷贝），`plan()` 可重复调用。

### Phase 4.5 之后的 D-14 不变量（05 §13.2/§13.3）

- `_check_full_declaration`：链式填充已删除——声明了 `in_dst` 而 `in_src`
  为空 → fail-fast；
- `_check_param_uniqueness`：每个参数恰好被一个边界分片；任何参数被 ≥2 个
  spec 声明 → fail-fast（production 下双重切分会静默损坏）。

### Phase 5：`_is_terminal` 标记（D-14）
编译期链式传播/校验**整体删除**（spec 完全自声明，validate 模式各模块自证
传播）；仅保留前向序最后一个边界的 `_is_terminal` 标记（用于 validate 的
out_dst 防御性校验）。排序按 `named_modules` 注册序。

### Phase 6：特殊参数 handler 收集
`SPECIAL` 角色参数 → handler 名（`_SPECIAL_HANDLER_PATTERNS` 小写子串匹配，
未注册落 `"default"`）。内置 `gated_delta_tp_shard`（SSM/Mamba 骨架实现：
结构识别 + 标准 `Shard(0)` 兜底，head 对齐细粒度切分待具体模型上线时补全）。

### 其他收尾
- `_warn_uncovered_function_modules`：`FunctionModule` 无边界 spec → 警告
  （不会插入任何通信，DX 守门）；
- `_detect_tied_pairs`：`tie_word_embeddings` 时检测 embed/lm_head 共享存储
  （`named_parameters(remove_duplicate=False)`；PP 跨 stage 无法检测，需用户
  显式声明 `plan.tied_pairs`）。

### 模型侧兼容性校验（`validate_model_compatibility`，05 §6.5）
TP 整除 head 数 / kv head 数 / moe_intermediate_size；CP 下
`seq_len % (2*cp) == 0`；EP 整除 num_experts。

---

## 7. PrecompiledBoundary：编译期通信计划（`precompiled_boundary.py`，05 §4.3）

### 7.1 RedistOp

单条预编译重分布操作：`{arg_name, arg_index, mesh, src_placements,
dst_placements, collective_type}`。

- `collective_type`（`_classify_collective`：identity / all_gather /
  reduce_scatter / all_reduce / redistribute）只是**调试/profiling 标签**，
  只做差异维比较（identity 维不参与分类）；所有非 identity 通信统一走
  `DTensor.redistribute()`（in-house DTensor 内部按 (src, dst) 选最优
  collective）。
- `execute(tensor, as_dtensor=?)`：identity + DTensor 输入时按模式
  保留/解包；非 identity 走"零拷贝 from_local 包装 → redistribute → 按需
  to_local"统一路径。

### 7.2 编译逻辑

- **in_plan**：`in_src → in_dst` 逐参数名编译（identity 维自然编译为透传
  op）；`_bind_input_indices` 在 apply 时把 arg_name 绑定到 forward 签名的
  位置下标（模块间调用多为位置传参），单输入契约回退绑定到第 0 个位置
  参数（覆盖模板 key `hidden_states` vs 叶子签名 `input` 的差异）；
- **out_plan**：`out_src → out_dst`，identity 直接跳过；多输出按
  `out_names`（缺省取 out_src 声明键序）映射 tuple 位置；`out_src=None`
  或 `out_dst=None` → 不编译。

运行期 `redistribute_inputs/outputs` 按序列直跑，**零分支**；arg 未找到
（None）跳过（如 embed 的 in_src key `input` 与实际 kwargs 名 `input_ids`
不同且为 identity）。

---

## 8. 显式注入机制（explicit-injection rework 后的核心制度）

### 8.1 总原则

**框架永不自动选择计算/包装方案。** planner 只负责参数分片与元数据；一切
"数据相关逻辑由谁执行"必须**显式声明**，声明即应用（非门控派生）。两条
正交可组合的注入家族：

```
[local-region 家族] 模块级：骨架不变，内容替换
    骨架 = _wrap_local_region_forward（边界进出缝合 + local 计算
           + validate 双模式容错 to_local/_temp_local_params/from_local）
    门控 = 单一解析链 _resolve_local_compute_fn 的派生（非存储 bool）：
        环 1  local_compute_fn（@local_compute 区域计算工厂，callable 直传
              或 Target 载体 —— 仓内参考 ep_compute.hf_native_ep_compute_fn；
              伴生必填 region_dispatch）
        环 2  region_dispatch=False（模块自身 forward 即数据相关逻辑）
        均无 → None（不走骨架）
[inner-wrap 家族] 子模块级：定位 + 替换 inner forward
    inner_target 回答"换谁"，inner_wrapper 回答"换成什么"
    机制不 CP 门控（声明==应用）；四个仓内参考 CP wrapper 自检要求活跃 cp 轴
```

### 8.2 注入纪律（`injection.py`）

两个模板装饰器在 **import 期即 fail-fast**：

| 装饰器 | 层次 | 必选框架上下文 |
|---|---|---|
| `@local_compute` | 工厂层（local_compute_fn 区域计算工厂，唯一形态） | `mesh/tp_mesh/cp_mesh/ep_mesh`（`module` 可选；用不用随你——统一接口规范） |
| `@inner_wrapper` | 工厂层（inner forward 包装） | 上述 mesh 家族 + `target_module` |

硬性规则：

- 必选上下文缺一不可；上下文参数**不得有默认值**（框架必然按名填充）；
- **禁止 `*args/**kwargs`**——签名必须是显式形参列表（"配置键按名绑定、
  拼写错误不得被静默吞掉"制度的前提）；
- 上下文键是保留名：YAML/Target 配置同名键 → fail-fast
  （`fill_context_kwargs`）；每次填充记 INFO（无静默行为）；
- 其余具名参数是用户配置键，只接受**数据值**——禁止再往注入函数里传函数
  （函数套函数无穷无尽；自定义行为 = 写自己的注入函数，路由/排布写死在
  函数体内）；
- `_check_target_config_keys`：Target 配置的键若不在目标函数显式形参列表
  → fail-fast（防 `**_context` 吞掉拼写错误）。

运行期签名校验（apply 时 fail-fast）：

- `validate_local_compute_signature`：compute fn 入参与原 forward 同名、
  位置序为子序列、forward 必填参数全被接住；
- `validate_wrapped_forward`：inner_wrapper 替换后的 forward 能 dummy-bind
  原 forward 全部入参（替换侧允许 `*args/**kwargs` 宽容透传）。

### 8.3 双模适配器（`_install_inner_adapter`，05 §4.4.2 + D-01''）

安装期解析重包规则，运行期零决策：

- **validate**（任一入参是 DTensor）：DTensor 入参 to_local +
  `_temp_local_params(target)` 临时解包参数 → 调用户 forward → 按声明重包
  回 DTensor（传播链接回，边界校验继续）；
- **production**（无 DTensor 入参）：直通，零转换开销。

重包 placements 来源（框架零推导零猜测）：

- 情形 A（target 是边界模块自身）：边界 `spec.out_src`；
- 情形 B（inner 子模块）：`spec.inner_out_src` 显式声明——哨兵
  `"first_input"`（layout-preserving，仅单输出）/ NamedPlacement /
  `{name: NamedPlacement}`（多输出按声明键序）；未声明 → 安装时 fail-fast。

"真的发生了替换"检测比较底层函数对象（`__func__`），纯探针 wrapper 不会
被误装适配器。

### 8.4 inner_target 定位（`_resolve_inner_target`）

显式 `spec.inner_target`（`"self"` = 模块自身；属性名 → fail-fast 若不存在）
> 显式属性 `inner_attention/attn/attention`（NeMo/Megatron 风格）> HF 标准
（类名含 "SdpaAttention" 或以 "Attention" 结尾 → 模块自身）> 结构兜底
（直持 q/k/v_proj）。只声明 inner_target 不声明 inner_wrapper → fail-fast
（定位不能选方案）。

### 8.5 apply 前置守门（`_preflight_compute_injection`）

任何参数被触碰**之前** fail-fast：

- **CP**：活跃 cp mesh 下 `_needs_cp_attn=True` 的 attention 边界缺
  `inner_wrapper` → 报错并给出 YAML 示例；
- **EP**：`_ep_size>0` 的边界缺 `local_compute_fn` 且非
  `region_dispatch=False` → 报错并列出三条出路（① HF 原生 → Target 注入
  `hf_native_ep_compute_fn`（+ `region_dispatch: false`）；② 自研
  EP-aware → `region_dispatch: false`；③ 自定义 compute）。

---

## 9. ShardingApplier：运行期应用（`sharding_applier.py`，05 §4）

### 9.1 主入口 `apply_sharding_plan(model, plan, mesh, *, validate_mode=False)`

支持单个 `nn.Module` 或 PP parts 列表。流程：

1. **mesh 对齐**：`_get_active_mesh` 取与 `plan.mesh_dim_names` 对齐的活跃
   子 mesh（传入 mesh 可能仍含 size=1 轴；placements 按 plan 坐标系解析，
   维度必须对齐，否则 `distribute_tensor` 会沿错轴切）；
2. **前置守门**：`_preflight_compute_injection`（§8.5）；
3. **专家 mesh 派生**（D-10）：任一 spec 带 `_ep_size` 时，从完整 dense
   region 派生 `(edp, ep)` expert mesh **一次**——它同时是 Phase A 专家参数
   的分片域和 Phase C 注入工厂的 `ep_mesh` 上下文，**a2a 通信域与分片域
   构造上就是同一个对象**（派生过程记 INFO）；
4. **Phase 0**：`out_src/out_dst` 标量简写规范化（幂等，覆盖用户注入路径）；
5. **Phase A 参数分片**：`_ep_stack` 非空先 `_stack_moe_experts` 堆叠
   `[E, ...]`；`_ep_size>0` 时专家参数（`experts.*`）在 expert mesh 上分片、
   其余参数在主 mesh 上分片；`distribute_tensor` 幂等（已是 DTensor 且
   placement 一致则跳过，不一致 → `PlacementMismatchError`）；meta tensor
   零内存路径保留。production 下随 Phase A 做 **D-17 头数改写**（§12.2）；
6. **Phase B 特殊 handler**：按 `plan.special_handlers` 调
   `SPECIAL_HANDLERS[name](module, param_name, mesh)`；
7. **Phase C 入口（production only）**：`_local_params_context` 一次性永久
   解包全部 DTensor 参数为 local tensor，并 `build_tp_grad_info(plan,
   tp_mesh)`（§11）；
8. **Phase C forward 包装**（§9.2）；
9. **Phase D tied weights**：`_broadcast_tied_param`——**rank 内** B 端共享
   A 端存储（`_local_tensor` 或 `.data`）。跨 rank 广播是**错的**：两端通常
   都是 `Shard(0)`，各 rank local 分片是不同 vocab 区间；tied 语义是同一
   rank 内同一物理参数（共享梯度），跨 rank 一致性由同一全局源 + 同一
   placement 自然保证。

### 9.2 Phase C：五路 forward 包装

**D-14 不变量 2**：边界按 post-order 包装（FQN 深度降序）——外层边界的
`local_compute_fn` 可能缓存内层 forward，且不变量 3 的解包作用域排除要求
内层 wrapper 先装好。

每个边界：构造 `PrecompiledBoundary` → `_bind_input_indices` →
Step 1 inner-wrap（`_wrap_inner_attention`，机制不 CP 门控，解析链即门控；
注入后记 INFO 并回写 `_resolved_inner_wrapper/_resolved_inner_target`）→
Step 2 按模式分派：

```
compute_fn = _resolve_local_compute_fn(...)   # 单一解析链（§8.1）
if compute_fn is not None:                    # ── 路 1：local-region 骨架
    validate 下对 local-region 模块补 D-17 头数改写
    _wrap_local_region_forward(..., exclude_subtrees=嵌套边界)
elif validate_mode:                           # ── 路 2：validate 包装
    _wrap_validate_forward(...)
else:                                         # ── 路 3/4：production
    if _is_vocab_parallel_embed(...):         #    路 4：D-02 masked embedding
        _wrap_vocab_parallel_embedding(...)
    _wrap_production_forward(...)             #    路 3：标准 production
```

（第五路 = inner-wrap 家族，与模块级包装正交叠加。）

**路 3 `_wrap_production_forward`**：`boundary.redistribute_inputs →
original_forward → boundary.redistribute_outputs`，纯 local tensor，参数已
在 Phase C 入口永久解包。

**路 2 `_wrap_validate_forward`**：

1. 入口先探测嵌套（任一入参是 DTensor → 来自外层 DTensor 传播边界，D-14
   §13.4；必须在 Step 1 把一切都包成 DTensor **之前**探测）；
2. 输入 `redistribute_inputs(as_dtensor=True)`；
3. 参数保持 DTensor，原 forward 经 `__torch_function__` dispatch 传播；
4. **核心校验 out_src**：dispatch 派生的输出 placement vs 声明
   （`_validate_outputs`：多输出按 out_names 映射、负维规格化
   `Shard(-1)==Shard(ndim-1)`、非 DTensor 输出跳过；不一致 →
   `PlacementMismatchError`）；
5. `redistribute_outputs(as_dtensor_input=True)` 到 out_dst；
6. **防御性校验 out_dst**：仅 `_is_terminal` 模块；
7. 出口解包回 local（与 production 边界输出同构）——但嵌套场景保持
   DTensor，外层 forward 的 dispatch 链不断；最外层边界出口解包。

**路 1 `_wrap_local_region_forward`**（D-03' 通用骨架）：
边界入口（`redistribute_inputs`，validate 下 as_dtensor）→ local 区域
（validate：入参 to_local + `_temp_local_params(module,
exclude=嵌套边界子树)` 临时解包参数——**D-14 不变量 3**：嵌套边界子树
的参数必须保持 DTensor，供内层 validate 孤岛 dispatch 使用）→
`compute_fn(*local_args)` → 按声明 out_src `from_local` 重包（数据相关模块
的 out_src 是**声明式校验**——a2a 的数据相关性使 placement 无法派生，这是
固有局限）→ 边界出口 → 最终恒解包为 local。两模式共享同一份 wrapper 代码。

**路 4 `_wrap_vocab_parallel_embedding`**（D-02）：Megatron 风格 masked
embedding——本地 vocab 区间 `[lo, hi)` 外的 token 置零并平移下标，输出乘
mask，天然成为 Partial 贡献，边界出口 `Partial→Shard(1)` 归约不变。

### 9.3 validate 模式的校验豁免/声明式清单（D-01''/D-03'）

- **attention（CP>1）**：out_src 为声明式——CP wrapper 出口按声明
  `from_local` 重包（区域内 SDPA 对 K/V 做显式 all-gather，dispatch 无法
  派生该语义）；
- **MoE（`region_dispatch=False`）**：out_src 为声明式——all-to-all 的数据相关性
  使 placement 无法派生；in 契约仍由 boundary 正常校验；
- **其余模块**（embed/norm/mlp/lm_head）：out_src 由 DTensor dispatch 派生
  校验（核心校验）。

---

## 10. CP 与 EP 支持

### 10.1 CP（`cp_wrappers.py` / `cp_utils.py`）

**仓内参考四路 wrapper**（`INNER_WRAPPER_REGISTRY`，开放注册），按
模型签名约定 × attention 实现 2×2 分派：

| 注册名 | 签名约定 | 机制 |
|---|---|---|
| `sdpa_qkv` | NeMo `forward(q,k,v,...)` | 显式 K/V all-gather + D-04 mask |
| `sdpa_hf` | HF `forward(hidden_states,...)` | `F.scaled_dot_product_attention` 原语拦截（复用 HF 投影/RoPE），发火检测（未拦截到调用 → RuntimeError） |
| `flex_qkv` | NeMo + FlexAttention | 显式 all-gather；block_mask 须按全局 kv 长度构建 |
| `flex_hf` | HF + FlexAttention | `flex_attention` 原语拦截，同上发火检测 |

原语拦截是临时全局函数替换（try/finally 恢复），非线程安全，SPMD 单进程
训练下安全（与 TorchTitan CP 实现一致）。

**D-04（CP causal mask 修正）**：`is_causal` 且 CP 激活时替换为 offset-aware
显式 mask——CP 下 q 是本 rank 连续 chunk、kv 是全长序列，`q_len != kv_len`
时 torch 的 is_causal 左上对齐（等价假设 Q 从全局 0 开始），rank>0 的 chunk
mask 错误；显式 mask 按本 rank 全局 Q 偏移 lo 构造下三角（rank0 lo=0 退化
为标准 causal）。触发依据是 **CP 语义**而非 shape 比较（GQA 的头数差不影响
序列维；cross-attention/KV-cache 的 q_len≠kv_len 与 CP 无关）。代价：显式
attn_mask 排除 SDPA flash 后端（回退 mem_efficient/math），正确性优先。

**数据管道**（D-05）：`shard_batch_for_cp` 在数据侧沿 CP 切分 batch（与 02
collater 的 THD 契约对齐），`_shard_seq_lens_for_cp` 重算 seq_lens；embed
的 in/out CP 维因此声明为 `Shard(1)`（避免二次切分）。`flex_cp_allgather`
经 `_AllGatherAlongDim` autograd.Function 实现前向 all-gather / 反向
reduce-scatter 语义，复用 `cp_mesh.get_group()`（禁止 new_group）。
G5 备注：`seq_len % (2*cp)` 约束源自 zigzag/ring 负载均衡方案，本设计用
all-gather K/V + 连续 chunk（D-01'' 否决 ring），各 rank Q chunk 等长、
FLOPs 天然均衡，该约束冗余但无害。

### 10.2 EP（D-09/D-10/D-11，`ep_utils.py` / `ep_compute.py`）

**两种 EP 模式，仅由 mesh 是否含 `"ep"` 轴决定**（与参数命名无关）：

- **老式 EP**（mesh 含 `"ep"` 轴）：专家参数 `{TP: Shard(…), EP: Shard(0)}`
  双轴分片；per-expert 2D 布局需先堆叠（`_ep_stack`），batched/custom 布局
  天然 3D 无需处理；不设 `_ep_size`，通信由模块自带 dispatcher 或外部
  `_attach_ep` 负责。
- **D-10 TP-extend-EP**（mesh 无 `"ep"` 轴，HF 原生模型默认路径，
  05 §6.4.8）：`ep_size` 即扩展 EP 组大小（无单独 etp 配置）；全 dense
  region（dp×tp×cp，排除 pp）row-major flatten 后重切为 `(edp, ep)`——
  **EP 组 = flatten 序连续 ep_size 个 rank**（tp 通常最内层，EP 组先跨完
  TP 组再向相邻 dp/cp rank 扩展，与 MindSpeed TP-extend-EP / Megatron
  etp=1 同构；例：mesh (dp=4, tp=2)、ep_size=4 → EP 组 {0,1,2,3}/{4,5,6,7}）。
  专家权重**仅 `{EP: Shard(0)}`**（每 rank 持完整 expert，无第二轴），
  MoE 边界契约改写为 SP-in identity（Megatron MoE 本不 gather，全部通信
  内聚在区域内）。校验：`ep_size` 不超过且整除 dense region、
  `num_experts % ep_size == 0`；pp>1 暂不支持（fail-fast）。
- `_expert_mesh_layout` 是纯映射（不建进程组）；`_build_expert_mesh` 传播
  源 mesh 的 no-backend（meta）模式；`build_expert_mesh` 为公开 introspection
  入口（注入的工厂/wrapper 无需调用——框架统一派生后以 `ep_mesh` 上下文
  传入）。

**专家布局**（只影响堆叠策略，与 EP 模式正交）：

- per-expert：`experts.<idx>.<proj>.weight` → Phase A 前置堆叠为
  `experts.<proj>` 3D；
- batched（D-11，HF 2025 重构后）：`experts.gate_up_proj [E,2I,H]` /
  `down_proj [E,H,I]` 天生 stacked，直接标 `{EP: Shard(0)}`，计算侧 chunk
  出 gate/up；兼容 automodel 命名 `gate_and_up_projs/down_projs`；
- custom（`w1/w2/w3`）：模块作者预堆叠 3D，EP-aware by construction，
  **保留**模板 `region_dispatch=False`。

**HF 原生布局的 region_dispatch 清除**：per-expert/batched 布局的 HF 原生
forward **不是** EP-aware 的——planner 将其 `region_dispatch` 清除为 None，
使模块永远不会在切分后的专家上静默跑原生 forward；必须由
`local_compute_fn` 显式注入并伴生 `region_dispatch=False`（缺失则 apply
前置守门 fail-fast）。

**仓内参考 EP compute**（`ep_compute.hf_native_ep_compute_fn`，`@local_compute`）：
router（本地 chunk）→ a2a dispatch（扩展 EP 组）→ 本地 SwiGLU（完整专家
权重，无内部通信）→ a2a combine → 加权聚合（+ shared_experts，经 tp_group
归约；无 tp 轴时 fail-fast）——与 Megatron `MoEAlltoAllTokenDispatcher`
（expert_tensor_parallel_size=1）同构，无 all_gather/reduce_scatter。router
是注入计算的一部分（注入纪律：框架不决定用户路由；仓内参考默认
`_softmax_topk_router`，`MOE_ROUTER_ADAPTERS` 开放注册其他路由语义）。

**a2a 后端分派**（`_ep_all_to_all`）：NCCL/HCCL 走不等长 a2a
（`_EPAllToAllUneven`，零填充；反向交换 send/recv counts 再来一次——a2a
自逆）；gloo 等不支持 ragged a2a 的后端走 pad-to-max +
`all_to_all_single`（`_EPAllToAllPadded`）。两路数值等价（padding 行不参与
计算）。

---

## 11. 与 FSDP2 的衔接（`tp_grad.py` / Phase C 入口）

production apply 返回 `tp_grad_info: {param_fqn: (tp_placement, tp_mesh)}`，
供 FSDP2 `fully_shard` 决定梯度的 TP 组同步语义：

- 数据源是 **plan**（不是 DTensor——production 下参数已解包，只有 plan
  保留完整 placement 信息）；
- `tp_placement ∈ {Shard, Replicate}`：Shard → 梯度各 rank 不同（需
  reduce-scatter 语义），Replicate → 需 all-reduce；
- **D-10 专家参数**：标 `Shard(1)`——专家梯度是扩展 EP 组上不同专家 +
  不同 token 的本地分片，**不做 TP 组同步**（默认 Replicate 会让 FSDP 错误
  all-reduce 已分片的梯度）；
- **tied 归一化**：tied pair 两端 placement 必须一致；不一致时取更细
  分片（Shard 优先），保证两端 TP all-reduce/reduce-scatter 语义一致。

---

## 12. 特殊机制

### 12.1 FunctionModule（`function_module.py`）

边界机制的作用粒度是 `nn.Module.forward`；自定义 `autograd.Function` 以
`A.apply(...)` 裸调用时对框架不可见（没有 FQN 就没有 spec 挂载点）。
`FunctionModule` 给 Function 一个模块壳：无参数无状态，forward 透传
`fn.apply`，backward 走 Function 自己的静态 backward——壳对 autograd 透明。
契约 key 绑定与所有边界一致（`_bind_input_indices`；单输入契约回退第 0 个
位置参数）；多输入需子类化给出显式签名。自定义 Function 不在 DTensor
dispatch 覆盖范围，必须 `region_dispatch=False`。

### 12.2 D-17：TP 本地头数改写（`head_count.py`）

部分 HF modeling 代码用显式（全局）头数 reshape（`q.view(b, s,
self.num_heads, self.head_dim)`）而非 TP 容忍的 `-1` 写法。q/k/v colwise
`Shard(0)` 后每 rank 只有 `num_heads/tp` 个头，凡前向看到 local tensor 的
模块都需要本地头数（AutoModel 同款语义）。

**双模式规则**——模块的缓存属性仅当其 forward 在当前模式下看到 local
tensor 时才改写：

- production：参数永久解包 → 所有头切分模块都改写（Phase A）；
- validate：普通边界模块跑 DTensor dispatch（全局逻辑形状，显式头数天然
  正确）→ **不改写**；local-region 模块区域内两模式都是 local → validate
  也改写（Phase C）。

属性清单来自 transformers 全库调研（2026-07）：Q 侧 7 名
（`num_heads`×393 / `num_attention_heads`×122 / `n_heads`×50 /
`num_attn_heads` / `n_head` / `heads` / `num_head`）+ KV 侧 3 名
（`num_key_value_heads` / `num_kv_heads` / `kv_heads`）；排除 `head_dim` 类
（头维度，永不被切）与 `num_key_value_groups`（比值，TP 不变量）。
不改 config 对象；幂等（原值存 `module._hp_full_head_counts`，重复调用
no-op）；不整除 → 大声警告并保持原值；MLA 的 `q_b_proj` 纳入 QKV 后缀清单。

### 12.3 local_region（`local_region.py`）

`DTensor → local → DTensor` 局部区域包装器（建于
`core.shard.custom_shard` 骨架之上，三点增强：具名参数绑定对齐
`in_dst` dict 契约、kwargs 原生支持、容错透传）。**服务 validate 模式与
独立使用**（production 的 local region 由 `_wrap_local_region_forward` 承担，
不用本函数）。**无反向缝合**——本仓 DTensor 前向-only，区域内反向即 local
autograd，梯度直落 local 参数分片，与 production 一致（对比 torch
`local_map` 需要 `in_grad_placements` 的原因：torch DTensor 有反向语义）。

---

## 13. 设计修订清单（相对 05 文档初稿，README canonical）

| # | 内容 |
|---|---|
| D-01'' | validate 的 CP 与 production 注入**同一个** all-gather wrapper（否决 ring/dispatcher） |
| D-02 | production embed 注入 Megatron 风格 masked embedding wrapper（解包后 vocab mask 丢失） |
| D-03' | MoE 统一走 local region（前向-only，无反向缝合） |
| D-04 | CP causal mask：is_causal 且 CP 激活（cp_size>1）时替换为 offset-aware 显式 mask |
| D-05 | embed 的 CP 契约：batch 已被数据管道 CP 切分 → in/out CP 维 `Shard(1)` |
| D-06 | MLP/MoE 的 CP 维全程 `Shard(1)`（pointwise，TP×CP 布局一致性） |
| D-07 | lm_head 的 CP 维 `Shard(1)`（R8：boundary CP 维恒 identity；loss 在本地 chunk 计算） |
| D-08 | MOE_EXPERT 的 TP placement 按参数 ndim 感知（3D `[E,out,in]`：colwise=`Shard(1)`、rowwise=`Shard(2)`） |
| D-09 | HF 原生 MoE EP 直通（05 §6.4.7）：planner 识别 per-expert/batched 布局并生成 EP 元数据（`_ep_stack`/`_ep_size`），Phase A 前置 `_stack_moe_experts` 堆叠；a2a 按后端分派 |
| D-10 | TP-extend-EP（05 §6.4.8）：`ep_size` 即扩展 EP 组大小（无单独 etp 配置）；全 dense 区域重分区为派生 expert mesh `(edp, ep)`；专家权重仅 `{EP: Shard(0)}`；通信流与 Megatron `MoEAlltoAllTokenDispatcher` 同构 |
| D-11 | fused batched expert 布局（`gate_up_proj [E,2I,H]` + `down_proj [E,H,I]`）天生 stacked 无需堆叠，直接标 `{EP: Shard(0)}`，计算侧 chunk 出 gate/up |
| D-12 | inner-wrap 双解析链（`_resolve_inner_target`/`_resolve_inner_wrapper`）：target 定位 fail-fast + `INNER_WRAPPER_REGISTRY` 注册表，门控派生不改写标记 |
| D-13 | local-region compute_fn 单一解析链（`_resolve_local_compute_fn`）：`local_compute_fn` > `region_dispatch=False`（模块 forward 即 compute）；骨架门控为解析结果的派生，声明互不嵌套。2026-08-07 起 `use_local_map` 统一为 `region_dispatch`（Optional[bool]，注入时必填无默认）：False=黑盒托管，True=validate 穿透真校验 |
| D-14 | DeepSeek MLA 支持（v2/v3）：新增 `ParamRole.REPLICATED`（仅 `ARCH_OVERRIDES` 指派）；另含 §13 嵌套 spec 与 validate 孤岛定稿（链式传播删除、三大不变量） |
| D-15 | Phase 5 链式契约比较降级为 `logger.warning`（无 shape 感知的值相等比较会误杀 reshape/transpose 合法场景；正确性由 validate 模式兜底）——其后 D-14 进一步将链式传播整体删除 |
| D-16 | plan_overrides 嵌套 spec fail-fast（`_check_no_nested_overrides`）——其后 D-14 放宽为允许嵌套、仅守参数唯一性 |
| D-17 | TP 本地头数改写（`head_count.py`，AutoModel 同款语义；Q 侧 7 名 + KV 侧 3 名；幂等，原值存 `_hp_full_head_counts`；validate 仅 local-region 模块改写） |

另有一次横切的**显式注入重构**（explicit-injection rework）：内置 EP
compute 与 CP wrapper **永不自动注入**——`_needs_cp_attn` 退化为纯元数据、
`_ep_size` 只驱动参数分片与 expert mesh 派生；计算注入必须经
plan_overrides / spec 显式声明，缺失时 apply 前置守门 fail-fast
（`_preflight_compute_injection`）。所有注入函数必须带模板装饰器
（`@local_compute` / `@inner_wrapper`），mesh 家族
上下文全由框架按名填充。

---

## 14. 文件职责总览

```
hyper_models/components/distributed/
├── sharding_config.py      # MeshAxisName/NamedPlacement/ShardingPlan/
│                           #   ModuleShardingSpec/ShardingTemplate/TEMPLATES(7)/
│                           #   PlacementMismatchError/placement DSL 解析
├── param_role.py           # ParamRole(14) + ParameterClassifier + 默认命名规则
├── sharding_planner.py     # ShardingPlanner 6-phase + ARCH_OVERRIDES +
│                           #   SPECIAL_HANDLERS + validate_model_compatibility
├── sharding_applier.py     # apply_sharding_plan + 前置守门 + Phase 0/A/B/C/D +
│                           #   五路 forward 包装 + expert mesh 派生（D-10）
├── precompiled_boundary.py # PrecompiledBoundary/RedistOp/_classify_collective
├── injection.py            # @local_compute/@inner_wrapper +
│                           #   上下文填充纪律 + 运行时签名校验
├── local_region.py         # DTensor→local→DTensor 局部区域（validate/独立使用）
├── cp_wrappers.py          # 仓内参考四路 CP inner wrapper + INNER_WRAPPER_REGISTRY
├── cp_utils.py             # flex_cp_allgather + shard_batch_for_cp + D-04 mask
├── ep_compute.py           # hf_native_ep_compute_fn（@local_compute，公开注入入口）
├── ep_utils.py             # _ep_all_to_all 后端分派 + MOE_ROUTER_ADAPTERS +
│                           #   _hf_native_ep_compute（D-09/D-10）
├── tp_grad.py              # build_tp_grad_info + tied 归一化（D-10 专家语义）
├── head_count.py           # D-17 TP 本地头数改写
├── function_module.py      # FunctionModule（autograd.Function 的模块壳）
├── fsdp2.py                # FSDP2Manager（demo 级集成，识别示例 dummy 模型）
├── config.py               # FSDP2Config/DDPConfig/MixedPrecisionPolicy 等（06 占位）
├── pipelining.py           # AutoPipeline 占位
├── sharding/apply.py       # _local_params_context/_temp_local_params/
│                           #   _stack_moe_experts/_set_param_by_path（canonical）
└── testing/grad_equiv.py   # 梯度等价工具（S5.3/S5.4 双模式对拍）
```

测试：`tests/components/distributed/`，单进程用例直接跑，多进程用例经
`run_dist`（spawn + gloo/CPU，macOS 可跑），覆盖 TP/CP/EP 及两两组合的
plan golden、production 数值（vs 单卡参考）、validate 校验与双模式等价。
独立示例（gloo/CPU 可跑）：`examples/distributed/`（tp/cp/ep/自定义模块五例）。
