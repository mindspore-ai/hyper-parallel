# 双模式 DTensor 组件代码详解说明书

> 范围：`hyper_models/components/distributed/` 全部 12 个源文件（约 3300 行）。
> 读法：本文严格按**调用时序**组织——先 build 期（plan → apply），再运行期（forward 五路包装 + 通信原语）。
> 每个函数给出功能、输入/输出，复杂函数附逐步示例。仅阅读本文即可掌握当前实现。
> 设计文档对应关系：05（双模式并行策略）、06（分布式基础设施）；决策编号 D-xx 见 05 文档。

---

## 0. 总览：文件地图与全局调用时序

### 0.1 文件地图

| 文件 | 职责 | 阶段 |
|---|---|---|
| [sharding_config.py](../../../hyper_models/components/distributed/sharding_config.py) | 数据模型：mesh 轴名、ModuleShardingSpec、ShardingPlan、ShardingTemplate、TEMPLATES | 数据模型 |
| [param_role.py](../../../hyper_models/components/distributed/param_role.py) | ParamRole 枚举 + ParameterClassifier（参数命名规则分类） | build-Phase 1 |
| [sharding_planner.py](../../../hyper_models/components/distributed/sharding_planner.py) | ShardingPlanner 6-phase 推导管线：模型 → ShardingPlan | build |
| [precompiled_boundary.py](../../../hyper_models/components/distributed/precompiled_boundary.py) | RedistOp / PrecompiledBoundary：placement 差异 → 预编译通信序列 | build + runtime |
| [sharding/apply.py](../../../hyper_models/components/distributed/sharding/apply.py) | 路径工具、`_local_params_context`（永久解包）、MoE expert 堆叠 | build |
| [sharding_applier.py](../../../hyper_models/components/distributed/sharding_applier.py) | `apply_sharding_plan`：Phase 0/A/B/C/D 应用计划 + 五路 forward 包装 | build + runtime |
| [tp_grad.py](../../../hyper_models/components/distributed/tp_grad.py) | `build_tp_grad_info`：从 plan 生成 FSDP 梯度旁路信息 | build |
| [head_count.py](../../../hyper_models/components/distributed/head_count.py) | D-17 TP 本地头数改写（兼容显式 `num_heads` reshape 写法） | build |
| [ep_utils.py](../../../hyper_models/components/distributed/ep_utils.py) | EP a2a（后端分派）、router adapter、TP-extend-EP 前向 | runtime |
| [cp_utils.py](../../../hyper_models/components/distributed/cp_utils.py) | CP K/V all-gather、CP 数据切分、offset causal mask | runtime + 数据管道 |
| [local_region.py](../../../hyper_models/components/distributed/local_region.py) | DTensor→local→DTensor 局部区域包装（独立工具，validate 用） | runtime（独立） |
| [testing/grad_equiv.py](../../../hyper_models/components/distributed/testing/grad_equiv.py) | 双模式梯度等价验证工具 | 测试 |
| [__init__.py](../../../hyper_models/components/distributed/__init__.py) | 公共 API 导出 | — |


> 测试基线：全套 300 例位于 [`tests/components/distributed/`](../../../tests/components/distributed/)；[`test_s5_zero_dep_lint.py`](../../../tests/components/distributed/test_s5_zero_dep_lint.py)（S5.5）守护组件零依赖边界（禁止 import recipes/_transformers/models/datasets）。
### 0.2 全局调用时序

```
用户代码
  │
  ├─ build 期 ─────────────────────────────────────────────────────────
  │  1. validate_model_compatibility(model, tp=, cp=, ep=, seq_len=)      [可选预检]
  │  2. plan = ShardingPlanner(plan_overrides={...}).plan(model, mesh,
  │            tp_size=, cp_size=, ep_size=, sequence_parallel=, loss_parallel=)
  │       Phase 1  _classify_all_params        参数 → ParamRole
  │       Phase 2  _group_by_boundary          参数 → 边界模块分组
  │       Phase 3  _infer_boundary_type        组 → 语义角色（模板名）
  │       Phase 4  _build_spec_from_template   模板 → ModuleShardingSpec
  │                └ _mark_hf_native_moe       HF 原生 MoE → stacked/EP 元数据
  │       Phase 4.5 _merge_plan_overrides      用户手写 spec 合并
  │       Phase 5  _chain_propagate_and_validate  相邻契约校验 + terminal 标记
  │       Phase 6  _collect_special_handlers   特殊参数处理器
  │  3. model, tp_grad_info = apply_sharding_plan(model, plan, mesh, validate_mode=)
  │       Phase 0  _normalize_out_fields       标量简写归一化（幂等）
  │       Phase A  _stack_moe_experts → _shard_module_params   参数 → DTensor
  │       Phase B  SPECIAL_HANDLERS            特殊参数处理
  │       Phase C  _local_params_context（production 永久解包）
  │                build_tp_grad_info
  │                _apply_phase_c（五路 forward 包装）
  │       Phase D  _replicate_tied_weights     tied weight 共享存储
  │
  └─ runtime（每次 forward）───────────────────────────────────────────
     边界模块 forward 已被替换为 wrapper：
       production:  boundary.redistribute_inputs → 原 forward(local) → boundary.redistribute_outputs
       validate:    同上，但全程 DTensor 传播 + out_src/out_dst 校验
     特殊区域内聚：
       CP attention: K/V all-gather（flex_cp_allgather）+ offset mask
       MoE/EP:       _hf_native_ep_compute（router → a2a → SwiGLU → a2a）
       vocab embed:  masked embedding（本地词表区间）
```

### 0.3 双模式一句话定义

- **production**：build 期把 DTensor 参数**永久解包**为 local tensor，forward 只跑 local 计算 + PrecompiledBoundary 通信，零 DTensor dispatch 开销；梯度经 `tp_grad_info` 交给 FSDP 旁路。
- **validate**：参数保持 DTensor，forward 走 `__torch_function__` dispatch 传播 placement，在边界校验 `out_src`/`out_dst` 与声明一致；反向两模式同为 local autograd。
- 两模式**数值必须逐位一致**：凡 DTensor dispatch 隐含数据相关逻辑的模块（embedding mask / CP K/V gather / MoE all-to-all），两模式注入**同一个** local-region wrapper 显式重建。

---

## 1. 数据模型（sharding_config.py）

本文件不执行逻辑，只定义贯穿全链路的契约结构。

### 1.1 `MeshAxisName` — mesh 维度名枚举

[sharding_config.py:L41-52](../../../hyper_models/components/distributed/sharding_config.py#L41-L52)

str 枚举：`TP/CP/EP/PP/DP/DP_REPLICATE/DP_SHARD/DP_SHARD_CP/DP_CP/EP_SHARD`。因为是 str 枚举，`MeshAxisName.TP == "tp"`，可直接做 dict key 与字符串比较。[L57-59](../../../hyper_models/components/distributed/sharding_config.py#L57-L59) 定义了 `TP/CP/EP` 简写别名。

### 1.2 `NamedPlacement` — 命名 placement

[sharding_config.py:L63-63](../../../hyper_models/components/distributed/sharding_config.py#L63-L63)

`Dict[MeshAxisName, Placement]`。key 是 mesh 维度名，value 是 `Shard(N)/Replicate()/Partial()`；`Shard(N)` 的 N 是 **tensor 维度**索引（与 mesh 维无关）。

### 1.3 `ModuleShardingSpec` — 单模块完整契约（核心数据结构）

[sharding_config.py:L83-267](../../../hyper_models/components/distributed/sharding_config.py#L83-L267)

四段式 I/O 契约，**运行时不做推断，只按声明执行**：

| 字段 | 类型 | 含义 |
|---|---|---|
| `params` | `Dict[str, NamedPlacement]` | 参数分片：模块内相对路径 → placement，如 `{"q_proj.weight": {TP: Shard(0)}}` |
| `in_src` | `Dict[str, NamedPlacement]` | 输入**到达边界时**的 placement（来自上游 out_dst 或 dataloader） |
| `in_dst` | `Dict[str, NamedPlacement]` | 模块内部计算**需要**的 placement（≠in_src 则入口触发通信） |
| `out_src` | `Optional[Dict[str, NamedPlacement]]` | 模块计算**自然产生**的 placement（validate 模式校验用） |
| `out_dst` | `Optional[Dict[str, NamedPlacement]]` | **下游期望**的 placement（≠out_src 则出口触发通信） |
| `out_names` | `Optional[List[str]]` | 多输出 tuple 的输出名顺序（映射 out_src/out_dst key → tuple 位置） |
| `is_boundary` | `bool` | 是否通信边界（默认 True） |
| `use_local_map` | `bool` | **用户可配置**：模块含数据相关逻辑（DTensor dispatch 无法表达，如 MoE a2a）→ 走 `_wrap_local_region_forward` local region。模板推断 True 时强制继承（不允许借覆盖关闭）；用户可在 plan_overrides 对自研模块显式置 True（05 §3.6.7） |
| `inner_target` | `Optional[str]` | **用户可配置**（05 §4.4.2）：**纯位置**——显式指定 inner attention 属性名（`"self"`=模块本身）；自动定位失败 fail-fast 时的指定入口。门控派生，不改写任何标记 |
| `inner_wrapper` | `Optional[Union[str, Callable]]` | **用户可配置**（05 §4.4.2）：**纯行为**——str 为 `CP_WRAPPER_REGISTRY` 注册表名（`"sdpa_qkv"/"sdpa_hf"/"flex_qkv"/"flex_hf"`，显式固定内置 CP wrapper、跳过启发式）；callable `fn(target, cp_mesh)` 整体接管（自定义 CP wrapper）。门控派生，不改写任何标记 |
| `local_compute_fn` | `Optional[Callable]` | **用户可配置**（05 §4.4.3）：local-region 自定义 compute_fn `fn(module, *args, **kwargs)`——复用 `_wrap_local_region_forward` 骨架（边界缝合+双模式容错）但注入自己的计算（自研 MoE 自定义 router/expert 布局/DeepEP）。声明即生效——骨架门控由解析链派生，**不改写 `use_local_map`** |

> 四个用户扩展点（`use_local_map`/`local_compute_fn`/`inner_target`/`inner_wrapper`）的**完整接口说明、决策树与配置示例**见 05 §8.6「自定义 wrapper 接口总览」。

内部标记（planner 自动设置，用户一般不管）：

| 字段 | 设置者 | 含义 |
|---|---|---|
| `_is_terminal` | Phase 5 | out_dst 未被下游引用 → 末端模块（validate 才做 out_dst 校验） |
| `_needs_cp_attn` | 模板 `needs_cp_attn` | attention：inner-wrap 解析链环 3 的输入（声明"需要 inner-wrap"的结构标记，非门控本身） |
| `_resolved_inner_wrapper` | applier 回写 | 实际注入的 inner wrapper：`"sdpa_qkv"` 等注册表名 / `"custom"` / None（plan 内省） |
| `_ep_stack` | `_mark_hf_native_moe` | `{stacked路径: [per-expert源路径]}`，非空 → Phase A 先堆叠 |
| `_moe_router` | `_mark_hf_native_moe` | router adapter 名（默认 `"default"`） |
| `_ep_size` | `_mark_hf_native_moe` | >0 → TP-extend-EP 路径（= ep_size，扩展 EP 组大小），派生 expert mesh |


> **用例**：[`test_s0_spec_fields.py`](../../../tests/components/distributed/test_s0_spec_fields.py)（S0.2：ShardingPlan/ModuleShardingSpec 字段与 05 §3.1-3.2 对齐）。
### 1.4 `ShardingPlan` — 模型级计划

[sharding_config.py:L271-288](../../../hyper_models/components/distributed/sharding_config.py#L271-L288)

| 字段 | 含义 |
|---|---|
| `modules` | `{module_fqn: ModuleShardingSpec}`，仅含边界模块 |
| `sequence_parallel` / `loss_parallel` | 全局开关（决定模板选 SP 还是 non-SP 契约） |
| `special_handlers` | `{param_fqn: handler_name}`（Phase 6 产出） |
| `mesh_dim_names` | 活跃 DTensor 轴名（剔除 size=1 轴），如 `("tp", "cp")` |
| `tied_pairs` | tied-weight 对 `[(embed_fqn, lm_head_fqn)]` |

### 1.5 `ShardingTemplate` / `TEMPLATES` — 语义角色模板

[sharding_config.py:L375-487](../../../hyper_models/components/distributed/sharding_config.py#L375-L487)、[L273-386](../../../hyper_models/components/distributed/sharding_config.py#L273-L386)

模板按**语义角色**预声明 7 套契约（TP+CP+EP 三维全写，planner 按实际 mesh 过滤）：

| 模板名 | 适用模块 | 关键契约（SP 模式） |
|---|---|---|
| `attention` | q/k/v Colwise + o Rowwise | in: TP Shard(1)→Replicate（入口 all-gather）；out: Partial→Shard(1)（出口 reduce-scatter）；`needs_cp_attn=True` |
| `mlp` | gate/up Colwise + down Rowwise | 同 attention 的 TP 流；CP 维全程 Shard(1)（pointwise 无需 CP 通信，D-06） |
| `norm` | RMSNorm/LayerNorm | 全 identity，零通信 |
| `embed` | nn.Embedding | weight Shard(0)（词表）；out: Partial→Shard(1)（reduce-scatter 到 SP+CP） |
| `lm_head` | 输出头 | weight Shard(0)；out_dst 视 `loss_parallel` 为 Shard(-1) 或 Replicate |
| `moe_gate` | router | 全复制；出口 EP Shard(0) |
| `moe_mlp` | MoE 块 | expert 参数 EP Shard(0)；`use_local_map=True` |


> **用例**：[`test_s1_templates.py`](../../../tests/components/distributed/test_s1_templates.py)（S1.5：7 模板字段完整性）；[`test_s1_sp_loss_matrix.py`](../../../tests/components/distributed/test_s1_sp_loss_matrix.py)（S1.7：SP on/off × loss_parallel 四组合 I/O 契约）。
### 1.6 工具函数

- **`_multi_dim(tp=, cp=, ep=)`** [L183-192](../../../hyper_models/components/distributed/sharding_config.py#L183-L192)：构造 NamedPlacement，自动剔除 None 维（size=1 轴不写 key）。
- **`resolve_placements(named, mesh_dim_names)`** [L195-200](../../../hyper_models/components/distributed/sharding_config.py#L195-L200)：把 NamedPlacement 按 mesh_dim_names 顺序排成 `List[Placement]`，缺失轴补 `Replicate()`。**这是 "named → positional" 的唯一转换点**，与 DeviceMesh 对齐。
- **`_normalize_out_fields(spec)`** [L203-213](../../../hyper_models/components/distributed/sharding_config.py#L203-L213)：标量简写 `{TP: ...}` 归一化为 `{"output": {TP: ...}}`（启发式：dict 的 value 不全是 dict → 判定为简写）。幂等。
- **`_hid(tp_p, cp_p, ep_p)` / `_out(...)`** [L216-223](../../../hyper_models/components/distributed/sharding_config.py#L216-L223)：hidden_states 单输入 / 单输出契约简写。
- **`PlacementMismatchError`** [L66-79](../../../hyper_models/components/distributed/sharding_config.py#L66-L79)：placement 声明与实际不一致的统一异常（validate 校验、参数分片两处抛出；Phase 5 链式比较已于 2026-07-22 降级为 warning，见 §2.7）。

---


> **用例**：[`test_s0_placement_utils.py`](../../../tests/components/distributed/test_s0_placement_utils.py)（S0.3：resolve_placements/_multi_dim/_normalize_out_fields）；[`test_s0_error.py`](../../../tests/components/distributed/test_s0_error.py)（S0.4：PlacementMismatchError 报错信息内容）。
## 2. Build 期 Step 1：`ShardingPlanner.plan()` — 模型 → ShardingPlan

入口：[sharding_planner.py:L177-246](../../../hyper_models/components/distributed/sharding_planner.py#L177-L246)

```python
plan = ShardingPlanner(plan_overrides={"model.layers.0.mlp": my_spec}).plan(
    model, mesh, tp_size=2, cp_size=1, ep_size=2,
    sequence_parallel=True, loss_parallel=False)
```

**输入**：model（任意 HF 风格 nn.Module）、mesh（DeviceMesh）、并行度开关。
**输出**：ShardingPlan。
**内部时序**：先做两件准备工作，然后 6 个 Phase 顺序执行。

### 2.0 准备工作

#### `_get_architecture(model)` — 架构名检测
[sharding_planner.py:L250-272](../../../hyper_models/components/distributed/sharding_planner.py#L250-L272)

优先级：`config.architectures[0]` > `config.model_type` > 类名，小写化并剥离 `ForCausalLM` 等后缀。输出如 `"llama"`、`"qwen3"`。用途：ARCH_OVERRIDES 查表 + MoE router adapter 选择。

#### `_build_mesh_dim_names(mesh, tp, cp, ep)` — 活跃轴过滤
[sharding_planner.py:L274-285](../../../hyper_models/components/distributed/sharding_planner.py#L274-L285)

以 `mesh.mesh_dim_names` 为权威顺序，只保留 `("tp","cp","ep")` 中 size>1 的轴。例：mesh 轴 `("dp","tp")`、tp=2 → 输出 `("tp",)`。**输出写入 `plan.mesh_dim_names`，之后所有 placement 解析都以它为对齐基准。**

#### TP-extend-EP 语义常量
[L192-196](../../../hyper_models/components/distributed/sharding_planner.py#L192-L196)：`ep_extend = ep_size if ep_size > 1 else 0`——**ep_size 即扩展 EP 组大小**（a2a 通信域，由 TP 组向相邻 dp/cp rank 扩展；D-10 TP-extend-EP，无单独 etp 配置）。


> **用例**：[`test_s1_mesh_dims.py`](../../../tests/components/distributed/test_s1_mesh_dims.py)（S1.8：`_build_mesh_dim_names` 活跃轴过滤）；[`test_s1_arch_override.py`](../../../tests/components/distributed/test_s1_arch_override.py)（S1.2：ARCH_OVERRIDES 覆盖优先级 + `_get_architecture`）。
### 2.1 Phase 1：参数角色分类

> lm_head 的 CP 契约（D-07/R8）：CP 维恒 `Shard(1)`——CP 下 loss 在本地 chunk 上计算、不做序列 gather（Megatron CP 标准做法），boundary 层 CP 维恒 identity；`loss_parallel` 只决定 TP 维是 `Shard(-1)` 还是 `Replicate`。

#### `ParamRole` 枚举
[param_role.py:L33-51](../../../hyper_models/components/distributed/param_role.py#L33-L51)

14 个角色：`COLWISE / ROWWISE / NORM / EMBED / LM_HEAD / MOE_GATE / MOE_EXPERT / SHARED_EXPERT / FUSED_QKV / FUSED_GATE_UP / BIAS / REPLICATED / SPECIAL / SKIP`。ParamRole 是命名规则与模板之间的桥梁——它**不决定 I/O 契约**（那是模板的职责），只决定参数分片方式与边界聚合。`REPLICATED`（全维 Replicate）默认命名规则不产生，仅经 `ARCH_OVERRIDES` 显式指派——当前用于 DeepSeek MLA 的 q_a/kv_a 下投影（LoRA rank 维不切，q_b/kv_b 上投影按 head 维 COLWISE）。机理：下投影输出 latent 在 TP 组内完全一致，因此下游 q_b/kv_b COLWISE 的输入契约与标准 attention 相同——整条 MLA attention 与标准 attention 模板同构（此即 05 文档"方式 B"命名的架构覆盖设计选项）。

#### `ParameterClassifier.classify(model, arch)` → `{param_fqn: ParamRole}`
[param_role.py:L101-107](../../../hyper_models/components/distributed/param_role.py#L101-L107)

遍历 `named_parameters()` 逐参数分类。

#### `classify_param(name, overrides)` — 单参数三级匹配
[param_role.py:L109-122](../../../hyper_models/components/distributed/param_role.py#L109-L122)

1. **架构覆盖** `ARCH_OVERRIDES[arch]`（[planner L79-91](../../../hyper_models/components/distributed/sharding_planner.py#L79-L91)：llama/qwen2/qwen3/mixtral 为空表占位；已内置 **DeepSeek MLA** 条目（deepseekv2/v3 两种拼写），q_a/kv_a → `REPLICATED`、q_b/kv_b → `COLWISE`，见 §2.1 枚举注记与 test_s1_mla_deepseek.py）；
2. **默认命名规则**（[param_role.py:L59-83](../../../hyper_models/components/distributed/param_role.py#L59-L83)），按序首匹配——排序原则"更具体在前"：`shared_expert` 先于 `experts`；`.bias` 先于 colwise（否则 `q_proj.bias` 被 q_proj 截获）；
3. 未命中 → `SKIP`。


> **用例**：[`test_s0_param_role.py`](../../../tests/components/distributed/test_s0_param_role.py)（S0.1+S1.1：ParamRole 枚举完备性 + 默认命名规则首匹配/兜底 SKIP）；[`test_s1_mla_deepseek.py`](../../../tests/components/distributed/test_s1_mla_deepseek.py)（S1.14：DeepSeek MLA 覆盖——REPLICATED/COLWISE 指派 + attention 边界端到端）。
### 2.2 Phase 2：通信边界分组 `_group_by_boundary(param_roles)`

[sharding_planner.py:L295-340](../../../hyper_models/components/distributed/sharding_planner.py#L295-L340)

**功能**：把扁平的 `{param_fqn: role}` 聚成 `{boundary_module_fqn: [(param_fqn, role), ...]}`。
**两趟算法**（修正了 05 文档伪代码会把 `q_proj` 叶模块误判为 mlp 边界的缺陷）：

- **趟 1**：按直属模块分组（去掉参数叶名）。`a.b.q_proj.weight` → 组 `a.b.q_proj`。
- **趟 2**：工作队列按深度**从深到浅**处理。每组先做边界推断（Phase 3 同款 `_infer_boundary_type`）：
  - 命中（≠"unknown"）→ 该组即边界，定案；
  - unknown → 整组参数**向上合并**到父模块，父模块入队（父更浅，尾部追加即可）；根仍 unknown → 归入参数所在模块（后续无模板命中 → warning 跳过）。

**示例**（TinyLlama 一层）：

```
趟 1: {".model.layers.0.self_attn.q_proj": [q_proj.weight→COLWISE],
       ".model.layers.0.self_attn.o_proj": [o_proj.weight→ROWWISE],
       ".model.layers.0.mlp.gate_proj":    [gate_proj.weight→COLWISE],
       ".model.layers.0.mlp.down_proj":    [down_proj.weight→ROWWISE],
       ".model.layers.0.input_layernorm":  [weight→NORM]}
趟 2: q_proj 组 → 叶子段守卫 unknown → 并入 ".model.layers.0.self_attn"
      o_proj 组 → 同上并入
      ".model.layers.0.self_attn" 组（现有 COLWISE+ROWWISE）→ "attention" ✓ 定案
      mlp 同理 → "mlp"；input_layernorm → "norm"（趟 1 即命中）
输出: {".model.layers.0.self_attn": [...4 个参数], ".model.layers.0.mlp": [...], ...}
```


> **用例**：[`test_s1_boundary_group.py`](../../../tests/components/distributed/test_s1_boundary_group.py)（S1.3：两趟分组——q_proj 叶守卫向上合并、unknown 回溯到根）。
### 2.3 Phase 3：语义角色推断 `_infer_boundary_type(fqn, group)`

[sharding_planner.py:L344-402](../../../hyper_models/components/distributed/sharding_planner.py#L344-L402)

**输入**：模块 FQN + 组内参数角色列表。**输出**：模板名（`"attention"` 等）或 `"unknown"`。
**优先级**：

1. **显式 FQN 模式**：`embed_tokens/wte/...`→embed；`lm_head/...`→lm_head；`norm/layernorm/...`→norm；叶段 `router`→moe_gate；
2. **叶子段守卫**（[L110-115](../../../hyper_models/components/distributed/sharding_planner.py#L110-L115)）：`q_proj/gate_proj/experts/fc1/...` 等叶段名 → unknown（自身不是边界容器）；
3. **数字段守卫** [L376-377](../../../hyper_models/components/distributed/sharding_planner.py#L376-L377)：段名是纯数字（HF per-expert 容器 `experts.0..N`）→ unknown，参数须向上聚合到 moe 容器（D-09）；
4. **MoE 角色**：组内含 `MOE_EXPERT/SHARED_EXPERT/MOE_GATE` 且 FQN 命中 moe 容器模式 → `moe_mlp`；
5. **参数角色组合**：COLWISE+ROWWISE → attention（FQN 含 mlp 则 mlp；都不含默认 attention——SP 通信更保守）；仅 COLWISE → mlp 或 unknown。


> **用例**：[`test_s1_semantic_infer.py`](../../../tests/components/distributed/test_s1_semantic_infer.py)（S1.4：`_infer_boundary_type` 表驱动——显式模式/叶守卫/数字守卫/MoE 角色/角色组合）。
### 2.4 Phase 4：模板 → spec `_build_spec_from_template(...)`

[sharding_planner.py:L406-468](../../../hyper_models/components/distributed/sharding_planner.py#L406-L468)

**输入**：边界 FQN、参数组、模板、SP/loss_parallel 开关、mesh_dim_names、参数 ndim 表。
**输出**：ModuleShardingSpec（或 None）。

四个 Step：

1. **填 params**：逐参数调用 `_placement_for_role`（见 2.4.1）；
2. **选 I/O 契约**：`sequence_parallel=True` 深拷贝模板 `sp_*` 字段，否则 `nosp_*`（深拷贝防止链式传播改脏共享模板）；
3. **运行时覆盖**：
   - lm_head 的 out_dst 按 `loss_parallel` 实时决定（Shard(-1) vs Replicate）[L444-448](../../../hyper_models/components/distributed/sharding_planner.py#L444-L448)；
   - embed 在 CP+SP 下 in/out 的 CP 维改 Shard(1)（D-05：数据管道已切好 input_ids，不能二次切分）[L455-460](../../../hyper_models/components/distributed/sharding_planner.py#L455-L460)；
4. **特殊标记 + 归一化**：`use_local_map`、`_needs_cp_attn` 从模板拷贝；`_normalize_out_fields` 归一化简写。


> **用例**：[`test_s1_plan_golden.py`](../../../tests/components/distributed/test_s1_plan_golden.py)（S1.12：plan() 主入口 golden diff——tiny_llama SP on/off、tiny_hf_llama、tiny_moe）。
#### 2.4.1 `_placement_for_role(param_path, role, template, has_tp, has_ep, ndim)`

[sharding_planner.py:L471-506](../../../hyper_models/components/distributed/sharding_planner.py#L471-L506)

13 角色 → NamedPlacement 的映射表（CP 恒 Replicate，CP 不切参数）：

| 角色 | TP | EP |
|---|---|---|
| COLWISE/EMBED/LM_HEAD/FUSED_* | `Shard(0)` | Replicate |
| ROWWISE | `Shard(1)` | Replicate |
| NORM/MOE_GATE | Replicate | Replicate |
| MOE_EXPERT | ndim≥3 时 Shard(1)/Shard(2)（D-08，见下）；2D 时 Shard(0)/Shard(1) | `Shard(0)` |
| SHARED_EXPERT | w1/w3→Shard(0)，w2→Shard(1) | Replicate（EP 维全复制） |
| BIAS | Replicate | Replicate |
| SPECIAL/SKIP | —（返回 None，不进 spec.params） | — |

**D-08 ndim 感知**（[`_moe_expert_tp_placement` L148-167](../../../hyper_models/components/distributed/sharding_planner.py#L148-L167)）：stacked expert 权重是 3D `[E, H_out, H_in]`，dim 0 是 expert 维（归 EP Shard(0)），所以 TP 的 colwise/rowwise 要 +1 维：colwise（切 H_out）→ Shard(1)，rowwise（切 H_in）→ Shard(2)。


> **用例**：[`test_s1_role_mapping.py`](../../../tests/components/distributed/test_s1_role_mapping.py)（S1.6：13 角色 → placement 映射 + D-08 ndim=3 平移）。
### 2.5 Phase 4 后处理：`_mark_hf_native_moe(...)`（D-09 堆叠 + D-10 TP-extend-EP 核心）

[sharding_planner.py:L556-653](../../../hyper_models/components/distributed/sharding_planner.py#L556-L653)

**功能**：识别 HF 原生 MoE 的两种 expert 布局（D-09a 堆叠 / D-11 batched），把 spec 改写为（堆叠元数据 +）TP-extend-EP 形式。
**命中条件**：`ep_extend>0`（即 ep_size>1）且组内全部 MOE_EXPERT 参数属于同一布局（`_PER_EXPERT_RE` / `_BATCHED_EXPERT_RE`（[L547-554](../../../hyper_models/components/distributed/sharding_planner.py#L547-L554)），混合不标记并 warning）：
- **per-expert**：`experts.<idx>.<proj>.weight`（旧版 HF / 自研）→ 记录 `_ep_stack` 堆叠元数据；
- **batched**（D-11，当前 HF main）：`experts.gate_up_proj [E,2I,H]` / `down_proj [E,H,I]` 等单属性 3D 参数（ndim≥3）→ 天生 stacked，`_ep_stack` 留空跳过堆叠。
`w1/w2/w3` 命名的 pre-stacked 布局**不收**（EP-aware 模块约定，走自身 dispatcher）；expert bias v1 不支持（不标记并 warning）。

原命中条件（[L547-554](../../../hyper_models/components/distributed/sharding_planner.py#L547-L554)，`^experts\.(\d+)\.([^.]+)\.weight$`）。任一不满足 → 原样返回（pre-stacked 布局走模块自身 dispatcher）。

**命中后做三件事**：

1. **TP-extend-EP 校验** `_validate_ep_extend`（[L534-566](../../../hyper_models/components/distributed/sharding_planner.py#L534-L566)）：
   - pp>1 → NotImplementedError（v1 不支持；救济：按 stage 拆分 mesh 后分别调用）；
   - `ep_size ≤ dense 区域且整除`（dense = mesh 非 pp 轴全部 rank = dp_replicate × dp_cp × tp）；
   - `num_experts % ep_size == 0`（每 rank 持 `num_experts/ep_size` 个完整 expert）。
2. **参数表改写** [L631-642](../../../hyper_models/components/distributed/sharding_planner.py#L631-L642)：per-expert 布局按投影名聚合并删除 per-expert 条目、插入 stacked 条目；batched 布局原地改写——两者最终契约相同：
   ```python
   spec.params["experts.gate_proj"] = {EP: Shard(0)}   # 仅 expert 维切分
   spec.params["experts.up_proj"]   = {EP: Shard(0)}
   spec.params["experts.down_proj"] = {EP: Shard(0)}
   spec._ep_stack["experts.gate_proj"] = ["experts.0.gate_proj.weight", "experts.1.gate_proj.weight", ...]
   ```
   注意 stacked 条目**无 TP 键、无第二轴**——D-10 TP-extend-EP：expert 权重
   仅在 dim 0（expert 维）按扩展 EP 组切分，扩展 EP 组每个 rank 持有
   `num_experts/ep_size` 个**完整** expert 矩阵（与 attention 的 TP 完全解耦，
   也无 Megatron ETP 的 hidden 维切分，因此计算流不需要 all_gather/
   reduce_scatter 对）。与 MindSpeed「TP 扩展 EP」/ Megatron
   `expert_tensor_parallel_size=1` + ep 跨 TP 同构。
3. **边界契约改 SP-in identity** [L644-653](../../../hyper_models/components/distributed/sharding_planner.py#L644-L653)：`in_dst = in_src`，`out_src = out_dst = in_src 的布局`——Megatron MoE 本就不 gather 序列，通信全部内聚到 region 内部（`_hf_native_ep_compute`）。同时记录 `_moe_router`（按 arch 选 adapter，未注册落 `"default"`）和 `_ep_size`。

**示例**（mesh (dp=4,tp=2)、ep=4、num_experts=4，用户示例拓扑）：
- dense = 4×2 = 8，ep=4 ≤ 8 且整除 ✓，num_experts 4 % 4 = 0 ✓ → 每 rank 1 个完整 expert；
- 扩展 EP 组 `{0,1,2,3}` / `{4,5,6,7}`（跨 2 个 TP 组 × 2 个 dp rank）；edp=2；
- `spec._ep_size = 4`，`spec._ep_stack` 含 3 个 stacked 条目（gate/up/down_proj）；
- in_src = in_dst = `{x_BLD: {TP: Shard(1), CP: Shard(1)}}`（SP），out 同布局 → 边界全 identity。

**batched 布局（D-11）同例**：`spec._ep_stack == {}`（无需堆叠），`spec.params` 直接含 `experts.gate_up_proj/down_proj = {EP: Shard(0)}`，`spec._moe_router` 按 arch 命中注册 adapter（如 qwen3moe），其余契约相同。


> **用例**：[`test_s5_hf_native_moe.py`](../../../tests/components/distributed/test_s5_hf_native_moe.py)（planner 标记契约、ep=1/pre-stacked 不标记、堆叠 handler）；[`test_s6_ep_extend.py`](../../../tests/components/distributed/test_s6_ep_extend.py)（TP-extend-EP 契约 + batched 布局契约 + 校验报错）。
### 2.6 Phase 4.5：用户覆盖合并 `_merge_plan_overrides(plan, model, inferred_templates)`

[sharding_planner.py:L655-712](../../../hyper_models/components/distributed/sharding_planner.py#L655-L712)

**功能**：把 `ShardingPlanner(plan_overrides={fqn: spec})` 的用户手写 spec 合并进 plan，**必须在 Phase 5 之前**——覆盖 spec 仍参与链式契约校验与 terminal 标记（比 plan() 返回后打补丁安全）。

**语义**：
- fqn 已命中 planner 生成的 spec → **整体替换**；未命中（漏识别/无参数模块）→ **插入**；
- fqn 必须是 `named_modules` 中真实存在的模块，否则 ValueError；
- **嵌套 fail-fast**（2026-07-22，`_check_no_nested_overrides`）：override fqn 不得是任何派生边界的祖先/后代，override 之间也不得互相嵌套，命中即 ValueError 并给出指引——边界假设扁平链（Phase 5 用前一边界的 out_dst 填充/比对 in_src，该参照只在模块出口成立；嵌套时内层实际看到祖先 in_dst），且同一参数会被切两次（production 静默错）；同树只支持**同 fqn 替换**；
- **结构标记**：`use_local_map` 为公开字段——用户显式置 True（自研数据相关模块）合并后保留；推断模板为 True 时强制置位 [L703-709](../../../hyper_models/components/distributed/sharding_planner.py#L703-L709)——MoE all-to-all 与 CP K/V gather 缺失会导致数值错误，不允许借覆盖关闭；
- **inner-wrap 自定义入口**（2026-07-21）：用户声明 `inner_target`/`inner_wrapper` **不改写任何标记**——inner-wrap 门控由 applier 的 `_resolve_inner_wrapper` 解析链派生（05 §4.4.2）；
- **local-region 自定义计算**（2026-07-21）：用户声明 `local_compute_fn` **不改写任何标记**——骨架门控由 applier 的 `_resolve_local_compute_fn` 解析链派生（05 §4.4.3）；
- 深拷贝用户 spec（plan() 可重复调用，Phase 5 会就地改 in_src，不能污染调用方对象）。


> **用例**：[`test_s1_plan_overrides.py`](../../../tests/components/distributed/test_s1_plan_overrides.py)（S1.13：替换/插入/嵌套 fail-fast 三形态/契约不一致告警/结构标记补齐）；[`test_dist_s5_plan_overrides.py`](../../../tests/components/distributed/test_dist_s5_plan_overrides.py)（S5.7：自研多输入 attention（`(attn_bias, x)` 签名）双模式 e2e——覆盖签名绑定 miss 时覆盖契约 key 的场景）。
### 2.7 Phase 5：链式传播 `_chain_propagate_and_validate(plan, model)`

[sharding_planner.py:L716-773](../../../hyper_models/components/distributed/sharding_planner.py#L716-L773)

**功能**：按 forward 顺序逐对相邻边界模块，填充缺省 in_src + 相邻契约不一致告警 + 标记 `_is_terminal`。

**步骤**：
1. `_topological_sort_by_forward_order`（[L823-841](../../../hyper_models/components/distributed/sharding_planner.py#L823-L841)）：按 `named_modules` 注册顺序排序边界 FQN；
2. 逐对 `(curr, next)`：若 `curr.out_dst` 非空，用 `_pair_contracts` 配对 key（[L809-821](../../../hyper_models/components/distributed/sharding_planner.py#L809-L821)：**双方都恰好 1 个 entry 时按"唯一 arg"配对（名字无关）**——解决 attention 输出名 "output" vs moe_mlp 输入名 "x_BLD" 不同名问题；否则按 key 名配对）；
3. 每对 `(out_key, in_key)`：
   - `next.in_src[in_key]` 缺失 → **用 curr.out_dst 填充**（场景 1）；
   - 已声明 → `resolve_placements` 后逐轴比较，**不一致仅 `logger.warning`（场景 3，2026-07-22 由 `PlacementMismatchError` 降级）**——该比较是 placement 值相等、无 shape 感知，边上 reshape/transpose 的合法场景（如 `[B,S,H]` Shard(1) fold 成 `[B*S,H]` Shard(0)）必然不等，报错会误杀合法配置；声明正确性由 validate 模式（DTensor dispatch + 数值等价）兜 correctness；
   - curr.out_dst 被下游引用 → curr 加入 non_terminal 集合；
4. `_is_terminal = fqn not in non_terminal`——按链式相邻关系判定（不做跨模块 placement 值匹配，避免 lm_head 的 Replicate 被 embed 误引用）。

**示例**（embed → norm0 → attn0 → mlp0 → ... → lm_head）：
```
embed.out_dst = {TP: Shard(1), CP: Shard(1)}
  → norm0.in_src 缺省 → 填充为 {TP: Shard(1), CP: Shard(1)}（与模板一致，校验通过）
attn0.out_dst = {TP: Shard(1), CP: Shard(1)}
  → mlp0.in_src 已声明 {TP: Shard(1), CP: Shard(1)} → resolve 后相等 ✓
lm_head.out_dst 无下游 → lm_head._is_terminal = True
```


> **用例**：[`test_s1_chain_propagate.py`](../../../tests/components/distributed/test_s1_chain_propagate.py)（S1.9：填充缺省/不一致告警/reshape 边合法场景/_is_terminal/拓扑排序）。
### 2.8 Phase 6 + tied 检测

- **`_collect_special_handlers(param_roles)`** [L814-829](../../../hyper_models/components/distributed/sharding_planner.py#L814-L829)：SPECIAL 角色参数 → handler 名（`_SPECIAL_HANDLER_PATTERNS` [L115-119](../../../hyper_models/components/distributed/sharding_planner.py#L115-L119)：`gated_delta/a_log/dt_bias` → `gated_delta_tp_shard`，未注册归 `"default"`）。
- **`_detect_tied_pairs(model)`** [L834-854](../../../hyper_models/components/distributed/sharding_planner.py#L834-L854)：`config.tie_word_embeddings=True` 时用 `named_parameters(remove_duplicate=False)` 找 `embed_tokens.weight` + `lm_head.weight` 两端 FQN（tied 参数默认去重只出现一次，必须关去重）。**PP 场景注意**：tied 对分居不同 stage 时单 part 内无法成对检测，需用户显式声明 `plan.tied_pairs`。


> **用例**：[`test_s1_special_handlers.py`](../../../tests/components/distributed/test_s1_special_handlers.py)（S1.10：SPECIAL 角色收集 + handler 注册表命中/兜底）。
### 2.9 独立预检：`validate_model_compatibility(model, tp=, cp=, ep=, seq_len=)`

[sharding_planner.py:L856-899](../../../hyper_models/components/distributed/sharding_planner.py#L856-L899)

模型 config 侧静态校验（与 06 的拓扑校验分工）：heads/kv_heads % tp、moe_intermediate_size % tp、seq_len % (2·cp)、num_experts % ep。失败抛 ValueError。建议在 plan 之前调用。

> 注（G5 由来）：`seq_len % (2·cp)` 约束源自 zigzag/ring 负载均衡方案；D-01'' 否决 ring 后，all-gather K/V + 连续 chunk 下各 rank 的 Q chunk 等长、FLOPs 天然均衡，该约束已冗余但无害，有意保留。

---


> **用例**：[`test_s1_compat.py`](../../../tests/components/distributed/test_s1_compat.py)（S1.11：heads/moe_intermediate 整除 TP、seq_len 整除 2·cp、num_experts 整除 ep 的报错路径）。
## 3. Build 期 Step 2：`apply_sharding_plan()` — 应用计划

入口：[sharding_applier.py:L74-151](../../../hyper_models/components/distributed/sharding_applier.py#L77-L162)

```python
model, tp_grad_info = apply_sharding_plan(model, plan, mesh, validate_mode=False)
```

**输入**：model（或 PP 多 part 列表）、plan、mesh、validate_mode。
**输出**：`(model, tp_grad_info)`——production 下 tp_grad_info 供 FSDP 梯度旁路；validate 下为 None。

### 3.0 入口准备

- **`_get_active_mesh(mesh, plan.mesh_dim_names)`** [L169-L176](../../../hyper_models/components/distributed/sharding_applier.py#L169-L176)：取与 plan 对齐的活跃子 mesh。plan 剔除了 size=1 轴，但传入 mesh 可能仍含——placement 数必须与 mesh 维数对齐，否则 `distribute_tensor` 静默错轴。**注意 `full_mesh` 在 [L91](../../../hyper_models/components/distributed/sharding_applier.py#L91) 先行保存**——D-10 派生 expert mesh 需要全 dense 区域（含 dp/cp 轴）。
- **TP-extend-EP expert mesh 构建** [L99-L102](../../../hyper_models/components/distributed/sharding_applier.py#L99-L102)：任一 spec 的 `_ep_size>0` → `_build_expert_mesh(full_mesh, ...)`。

#### `_expert_mesh_layout(mesh, mesh_dim_names, ep_size)` — 纯映射（可单测）
[sharding_applier.py:L186-213](../../../hyper_models/components/distributed/sharding_applier.py#L197-L224)

**功能**：计算派生 expert mesh 的 `(shape, dim_names, rank_list)`，不建进程组。
**算法**：全 mesh rank_list 按 mesh 轴序 row-major reshape 成 `mesh_shape` 数组 → flatten → 重切 `(edp = D/ep_size, ep)`。

**示例**（mesh (dp=4, tp=2)、world=8、ep_size=4，用户示例拓扑）：
```
rank_list = [0..7]，reshape (4,2) = [[0,1],[2,3],[4,5],[6,7]]
flatten [0,1,2,3,4,5,6,7] → reshape (edp=2, ep=4) = [[0,1,2,3],[4,5,6,7]]
→ shape (2,4), names ("edp","ep")
EP 组（同行，a2a 通信域）: {0,1,2,3},{4,5,6,7} —— 先跨完 TP 组（{0,1},{2,3}）再跨 dp
edp 组（同列）: {0,4},{1,5},{2,6},{3,7} —— expert 数据并行
```
**为什么这样切**：tp 是 mesh 最内层轴，flatten 连续的 ep_size 个 rank 自然先跨完整个 TP 组、再向相邻 dp/cp rank 扩展——这正是 MindSpeed「TP 扩展 EP」的通信域定义（Megatron etp=1 + ep 跨 TP 同构）。expert 权重仅在该 ep 轴 Shard(0)。

#### `_build_expert_mesh(...)`
[L231-L231](../../../hyper_models/components/distributed/sharding_applier.py#L231-L231)：用 layout 的 rank_list 调 `init_device_mesh` 建真实 DeviceMesh（含进程组）。


> **用例**：[`test_s6_ep_extend.py::test_expert_mesh_layout_mapping`](../../../tests/components/distributed/test_s6_ep_extend.py)（派生 mesh rank 映射：ep=4 → (edp=2,ep=4) EP 组 {0,1,2,3}/{4,5,6,7}；ep=2 → EP 组即 TP 组；不整除报错）。
### 3.1 Phase 0：归一化

[L104-L106](../../../hyper_models/components/distributed/sharding_applier.py#L104-L106)：对所有 spec 再跑一遍 `_normalize_out_fields`（幂等，覆盖用户经 plan_overrides 注入的路径）。

### 3.2 Phase A：参数分片

[L108-L136](../../../hyper_models/components/distributed/sharding_applier.py#L108-L136)：逐模块：

1. **`_ep_stack` 非空 → 先 `_stack_moe_experts`**（HF 原生 MoE 堆叠）；
2. **`_ep_size>0` → 参数分两路**：`experts.*` 走 expert_mesh，其余走主 mesh；否则全部走主 mesh `_shard_module_params`；
3. **production → D-17 头数改写**：`maybe_update_head_counts`（[head_count.py](../../../hyper_models/components/distributed/head_count.py)）——前向跑在永久解包的 local tensor 上，把模块缓存的头数属性改写为本地值（见下）；validate 此处不改。

#### production 头数改写（D-17，`maybe_update_head_counts`）
[sharding_applier.py:L129-L136](../../../hyper_models/components/distributed/sharding_applier.py#L129-L136) + [head_count.py](../../../hyper_models/components/distributed/head_count.py)

**问题**：部分 HF modeling 代码 reshape/split 显式使用全局头数（`q.view(b, s, self.num_heads, self.head_dim)` 而非 TP 容错的 `-1` 写法）。TP colwise 切分后每 rank 本地只有 `num_heads/tp` 个头，前向看到 local tensor 时全局头数必然 shape 错误。
**检测**（`_is_head_sharded`）：spec.params 中 q/k/v 类参数（`q_proj/q_b_proj(MLA)/k_proj/v_proj/qkv_proj/qkv` 的 weight）在 TP 维为 colwise `Shard(0)` → 该模块头维被切。
**改写**（`update_module_head_counts`，AutoModel 同款语义）：就地整除模块**实例属性**——Q 侧 `num_heads/num_attention_heads/n_heads/num_attn_heads/n_head/heads/num_head`，KV 侧 `num_key_value_heads/num_kv_heads/kv_heads`（清单来自 transformers 全库 forward reshape 用法调研，2026-07）；**不改** `config`（head_dim/RoPE 推导不受影响）、`head_dim`（不切维）、`num_key_value_groups`（比值不变量）。幂等：原值存 `module._hp_full_head_counts`，重复 apply 不二次除法；非整除只 WARNING 不改写。
**双模式规则**（关键）：只有"前向看到 local tensor"的模块才改写——production 全量改写；validate 下普通 boundary 模块跑 DTensor dispatch（**全局逻辑形状**，显式全局头数天然正确，改写反而错）不改写，local-region 模块（区域内两模式都是 local tensor）在 Phase C 分支内改写（[sharding_applier.py:L310-L316](../../../hyper_models/components/distributed/sharding_applier.py#L310-L316)）。

> **用例**：[`test_s1_head_count.py`](../../../tests/components/distributed/test_s1_head_count.py)（S1.14：检测/改写/幂等单测）；[`test_dist_s2_head_count.py`](../../../tests/components/distributed/test_dist_s2_head_count.py)（S2.8：显式 `num_heads` attention 在 production / validate boundary（自动推导、零改写）/ validate local-region 三位置数值对齐）。

#### `_stack_moe_experts(module, ep_stack)` — per-expert → stacked 3D
[sharding/apply.py:L102-140](../../../hyper_models/components/distributed/sharding/apply.py#L102-L140)

**输入**：MoE 模块、`{stacked相对路径: [源参数相对路径（按 expert idx 排序）]}`。
**做什么**：
- 逐 stacked 条目：取每个源 weight → `torch.stack(dim=0)` 成 `[E, H_out, H_in]`；
- 源模块带 bias → NotImplementedError（v1 不支持）；
- 创建 `_StackedExperts` holder（[L96-99](../../../hyper_models/components/distributed/sharding/apply.py#L96-L99)，空 nn.Module 容器），注册 stacked Parameter；
- **`setattr` 整体替换原 `experts` ModuleList**——原 per-expert 参数显存释放。

**示例**：
```
之前: mlp.experts = ModuleList([TinyLlamaMLP×4])，参数 experts.0.gate_proj.weight ...
之后: mlp.experts = _StackedExperts()，参数 experts.gate_proj [4, 4H, H]（值=stack，精确相等）
```

**为什么必须堆叠**（核心动机）：双模式 DTensor 的参数契约是**张量级**的
（`params: {路径: placement}`，`Shard(0)` 切的是某个张量的 dim 0）。
per-expert 2D 布局下，`experts.0.gate_proj.weight`、`experts.1.gate_proj.weight`
是 E 个互不相干的张量——"rank0 拿 expert 0、rank1 拿 expert 1"不是对任何
张量的切分，而是模块级分配，placement 语言表达不了（在
`experts.0.gate_proj` 上写 Shard(0) 只会把 expert 0 的矩阵本身切开，语义
全错）。堆叠出 dim 0 = expert 维后，`{EP: Shard(0)}` 一次普通张量分片就
精确表达"每 rank 持 `num_experts/ep_size` 个完整 expert"（local shard
`[E/ep, H_out, H_in]`），validate 传播、placement 校验、production 解包
全部走统一路径，无特例。附带收益：wrapper 计算可用 `w_gate[i]` 索引做
per-expert GEMM（与 Megatron GroupedMLP 布局一致，便于未来换 grouped-GEMM
kernel）；契约条目从 3E 个减为 3 个。代价：checkpoint key 变化
（`experts.N.proj.weight` → `experts.proj`，映射见 04 §7.6）。

#### `_shard_module_params(module, param_specs, mesh, mesh_dim_names)`
[sharding_applier.py:L233-258](../../../hyper_models/components/distributed/sharding_applier.py#L244-L269)

逐参数：`resolve_placements` → `distribute_tensor(param.data, mesh, placements)` → `_set_param_by_path` 替换为 DTensor Parameter。三种情形：
- meta tensor → DTensor（local 仍 meta，零显存路径）；
- real tensor → 物理切分，每 rank 持 local shard；
- 已是 DTensor → placement 一致跳过，不一致抛 PlacementMismatchError（幂等保护）。

#### 路径工具（sharding/apply.py）
- [`_get_attr_by_path` L32-37](../../../hyper_models/components/distributed/sharding/apply.py#L32-L37)：点分 FQN 取属性，数字段走 ModuleList 索引；
- [`_set_param_by_path` L40-54](../../../hyper_models/components/distributed/sharding/apply.py#L40-L54)：定位父模块后 `register_parameter` 替换叶参数（直接 `setattr(model, dotted, ...)` 只会设怪属性）；
- [`_resolve_module` L57-68](../../../hyper_models/components/distributed/sharding/apply.py#L57-L68)：按模块 FQN 取模块（不剥末段）。


> **用例**：[`test_s2_path_utils.py`](../../../tests/components/distributed/test_s2_path_utils.py)（S2.1：三个路径工具）；[`test_dist_s2_shard_params.py`](../../../tests/components/distributed/test_dist_s2_shard_params.py)（S2.2：`_shard_module_params` 分片数值）；[`test_s5_hf_native_moe.py`](../../../tests/components/distributed/test_s5_hf_native_moe.py)（`test_stack_moe_experts`：堆叠值精确相等 + 原参数移除；bias 拒绝）；[`test_dist_s4_ep_shard.py`](../../../tests/components/distributed/test_dist_s4_ep_shard.py)（S4.1：MoE expert EP 切片 + gate 全复制）。
### 3.3 Phase B：特殊参数处理器

[L138-L146](../../../hyper_models/components/distributed/sharding_applier.py#L138-L146)：按 `plan.special_handlers` 调 `SPECIAL_HANDLERS[name](module, param_name, mesh)`。当前唯一实现 `_shard_gated_delta`（[planner L91-106](../../../hyper_models/components/distributed/sharding_planner.py#L91-L106)，SSM/Mamba 类骨架：直接 Shard(0) 回退，head 对齐切分留待具体模型接入）。

### 3.4 Phase C 入口：production 永久解包 + tp_grad_info

[L148-L155](../../../hyper_models/components/distributed/sharding_applier.py#L148-L155)

#### `_local_params_context(model)` → `{fqn: placements}`
[sharding/apply.py:L71-86](../../../hyper_models/components/distributed/sharding/apply.py#L71-L86)

**production 专用**：把所有 DTensor 参数替换为 `param.to_local()` 的 plain Parameter（**零拷贝**，共享存储 data_ptr 相同），永久不恢复。返回解包前 placement 快照（仅诊断用）。

#### `build_tp_grad_info(plan, tp_mesh)` → `{param_fqn: (tp_placement, tp_mesh)}`
[tp_grad.py:L26-57](../../../hyper_models/components/distributed/tp_grad.py#L26-L57)

**为什么从 plan 读而非 DTensor**：production 下参数已被解包，只有 plan 保留完整 placement 信息。
- 普通参数：`named_placement.get("tp", Replicate())`；
- **D-10 TP-extend-EP expert 参数** [L38-44](../../../hyper_models/components/distributed/tp_grad.py#L38-L44)：强制 `Shard(1)`——expert 权重在派生 expert mesh (edp, ep) 上仅按 expert 维分片，梯度是各 rank 不同的 local shard（不同 expert + 扩展 EP 组聚合的 token），**不做 TP 组同步**；缺省 Replicate 会让 FSDP 对分片梯度错误 all-reduce；
- tied 对归一化 [L47-56](../../../hyper_models/components/distributed/tp_grad.py#L47-L56)：两端 placement 不一致时取较细分片（Shard 优先），保证两端 TP 归约语义一致。

**下游用法**：FSDP fully_shard 对 `(Shard, tp_mesh)` 参数跳过 TP 维梯度归约（分片梯度各 rank 不同），对 `(Replicate, tp_mesh)` 参数做 TP all-reduce（复制参数的 Partial 梯度需跨 TP 求和）。


> **用例**：[`test_dist_s2_local_params.py`](../../../tests/components/distributed/test_dist_s2_local_params.py)（S2.8：零拷贝解包 data_ptr 共享 + placement 快照）；[`test_s2_tp_grad_info.py`](../../../tests/components/distributed/test_s2_tp_grad_info.py)（S2.9：Shard/Replicate 标记 + tied 归一化取较细分片）。
### 3.5 Phase C：`_apply_phase_c` — 五路 forward 包装

[sharding_applier.py:L277-L326](../../../hyper_models/components/distributed/sharding_applier.py#L277-L326)

逐边界模块：
1. 构造 `PrecompiledBoundary(spec, mesh, mesh_dim_names)`（见 §4）；
2. `_bind_input_indices(boundary, module)` 签名绑定（见 3.5.1）；
3. **Step 1**：CP>1 → `_wrap_cp_inner_attention(direct=False)`（门控派生：`_resolve_inner_wrapper` 非 None 才注入，见 §5.3）；
4. **Step 2** 按模式分路：

| 条件 | wrapper | 章节 |
|---|---|---|
| `_resolve_local_compute_fn` 非 None（两模式，派生门控；**validate 下此分支先做 D-17 头数改写**，见 §3.2） | `_wrap_local_region_forward(compute_fn=...)` | §5.2 |
| validate 其他 | `_wrap_validate_forward` | §5.1 |
| production 且 vocab-parallel embed | 先 `_wrap_vocab_parallel_embedding` 再 `_wrap_production_forward` | §5.4 |
| production 其他 | `_wrap_production_forward` | §5.1 |


> **用例**：[`test_dist_s2_apply_e2e.py`](../../../tests/components/distributed/test_dist_s2_apply_e2e.py)（S2.11：apply 主入口 TP=2 双模式 e2e）；[`test_dist_s3_phase_c_cp.py`](../../../tests/components/distributed/test_dist_s3_phase_c_cp.py)（S3.5：CP 分支双模式注入同一 wrapper）。
#### 3.5.1 `_bind_input_indices(boundary, module)`
[sharding_applier.py:L322-L349](../../../hyper_models/components/distributed/sharding_applier.py#L322-L349)

**问题**：模块间调用多为 positional（`self.mlp(x)`），而 RedistOp 按 kwargs 名查找会 miss。
**解法**：编译期内省 `module.forward` 签名，把 in_plan 的 `arg_name` 绑定到 positional 下标（运行时 `_get_arg` 先 kwargs 后 args）。
**位置回退** [L345-L349](../../../hyper_models/components/distributed/sharding_applier.py#L345-L349)：in_plan 仅 1 个 op 且签名未命中 → 绑定到首个 positional 参数。覆盖"模板 key `hidden_states` vs 叶模块签名 `nn.Linear.forward(input)` 不同名"。

### 3.6 Phase D：tied weights

[L161-L165](../../../hyper_models/components/distributed/sharding_applier.py#L161-L165) → [`_replicate_tied_weights` L1113-1117](../../../hyper_models/components/distributed/sharding_applier.py#L1131-L1135) → [`_broadcast_tied_param` L1086-1110](../../../hyper_models/components/distributed/sharding_applier.py#L1104-L1128)。

**语义**：tied 对**本 rank 内共享存储**（B 端 `_local_tensor`/`data` 指向 A 端），**绝不跨 rank 广播**——两端同为 Shard(0) 时各 rank 的 local shard 承载不同 vocab 区间，广播 rank0 的 shard 会破坏 rank1 的分片。tied 语义要求的是"同 rank 两端同一物理参数（梯度共享）"，跨 rank 一致性由"同一 global 来源 + 同一 placement"天然保证。

---


> **用例**：[`test_dist_s2_tied.py`](../../../tests/components/distributed/test_dist_s2_tied.py)（S2.10：detect/broadcast/replicate——同 rank 共享存储、不跨 rank 广播）。
## 4. 编译期通信规划：PrecompiledBoundary

[precompiled_boundary.py](../../../hyper_models/components/distributed/precompiled_boundary.py)

**思想**：把 `in_src→in_dst`、`out_src→out_dst` 的 placement 差异在 **build 期编译为 RedistOp 序列**，运行时零判断直接执行。所有非 identity 通信统一走 `DTensor.redistribute()`（自研 DTensor 内部按 (src,dst) 自动选最优 collective）。

### 4.1 `_classify_collective(src, dst)` → str

[precompiled_boundary.py:L42-66](../../../hyper_models/components/distributed/precompiled_boundary.py#L42-L66)

从 placement 对推导通信类型标签（**仅调试/profiling 用**，不影响通信路径）：只比较有差异的轴——identity 轴（如 CP 维 Shard(1)→Shard(1)）不参与分类。

| 差异 pattern | 标签 |
|---|---|
| 完全相同 | `identity` |
| Partial→Shard | `reduce_scatter` |
| Partial→全 Replicate | `all_reduce` |
| Shard→全 Replicate | `all_gather` |
| 其他 | `redistribute` |

### 4.2 `RedistOp` — 单条预编译操作

[precompiled_boundary.py:L92-130](../../../hyper_models/components/distributed/precompiled_boundary.py#L92-L130)

字段：`arg_name / arg_index / mesh / src_placements / dst_placements / collective_type`。

**`execute(tensor, as_dtensor=False)`** [L105-129](../../../hyper_models/components/distributed/precompiled_boundary.py#L105-L129)：
- identity：DTensor 输入按 as_dtensor 决定保持/解包；plain 输入 as_dtensor=True 时 `from_local` 包装，否则直通；
- 非 identity：`from_local`（若需要）→ `redistribute(mesh, dst)` → 按 as_dtensor 返回 DTensor 或 `to_local()`。


> **用例**：[`test_dist_s2_redist_op.py`](../../../tests/components/distributed/test_dist_s2_redist_op.py)（S2.3：`RedistOp.execute` + `_classify_collective` 五组合通信数值，2 进程）。
### 4.3 `PrecompiledBoundary` — 编译与执行

[precompiled_boundary.py:L133-238](../../../hyper_models/components/distributed/precompiled_boundary.py#L133-L238)

**`__init__`** [L136-141](../../../hyper_models/components/distributed/precompiled_boundary.py#L136-L141)：编译 `in_plan` + `out_plan` 两个 RedistOp 列表。

**`_compile_input_plan`** [L145-163](../../../hyper_models/components/distributed/precompiled_boundary.py#L145-L163)：对 `in_src ∪ in_dst` 的每个输入名生成一个 RedistOp（identity 也保留——输入需要 from_local 包装/解包）。

**`_compile_output_plan`** [L165-196](../../../hyper_models/components/distributed/precompiled_boundary.py#L165-L196)：`out_src/out_dst` 任一为 None → 空计划；identity 跳过（**输出侧不编译 identity**——出口张量在 wrapper 层统一处理）；`arg_index` 取 `spec.out_names` 顺序（缺省按 out_src key 顺序）。

**示例**（SP attention，mesh 轴 ("tp","cp")）：
```
in:  src={TP: Shard(1), CP: Shard(1)} → dst={TP: Replicate, CP: Shard(1)}
     → in_plan = [RedistOp("hidden_states", None, [Shard(1),Shard(1)]→[Replicate(),Shard(1)], "all_gather")]
out: src={TP: Partial(), CP: Shard(1)} → dst={TP: Shard(1), CP: Shard(1)}
     → out_plan = [RedistOp("output", 0, [Partial(),Shard(1)]→[Shard(1),Shard(1)], "reduce_scatter")]
```

**`redistribute_inputs(args, kwargs, as_dtensor=False)`** [L200-212](../../../hyper_models/components/distributed/precompiled_boundary.py#L200-L212)：逐 op 用 `_get_arg`（先 kwargs 名、后 arg_index）取参，执行后 `_set_arg` 写回。arg 找不到（None）跳过——如 embed 的 in_src key "input" 与实际 kwargs 名 "input_ids" 不同名且 identity。

**`redistribute_outputs(outputs, as_dtensor_input=False)`** [L215-237](../../../hyper_models/components/distributed/precompiled_boundary.py#L215-L237)：单输出/多输出 tuple 保序处理，按 `arg_index` 定位；`as_dtensor_input=True`（validate）保持 DTensor 供 out_dst 校验，否则返回 local。

---


> **用例**：[`test_s2_boundary_compile.py`](../../../tests/components/distributed/test_s2_boundary_compile.py)（S2.4：identity 跳过/多输出 out_names 映射/None 分支）；[`test_s2_boundary_io.py`](../../../tests/components/distributed/test_s2_boundary_io.py)（S2.5：redistribute_inputs/outputs + `_get_arg`/`_set_arg` kwargs/positional 双通道）；[`test_s4_moe_gate_compile.py`](../../../tests/components/distributed/test_s4_moe_gate_compile.py)（S4.3：moe_gate 模板 out_dst {EP: Shard(0)} 编译）。
## 5. 运行期：五路 forward 包装

### 5.1 普通模块两路

#### `_wrap_production_forward(module, boundary)`
[sharding_applier.py:L334-350](../../../hyper_models/components/distributed/sharding_applier.py#L352-L368)

最薄包装：`boundary.redistribute_inputs → original_forward(local) → boundary.redistribute_outputs`。参数已在 build 期永久解包，纯 local 计算，零 DTensor 开销。

#### `_wrap_validate_forward(module, boundary, spec, mesh, mesh_dim_names)`
[sharding_applier.py:L352-391](../../../hyper_models/components/distributed/sharding_applier.py#L370-L409)

六步：
1. 输入 → DTensor（`redistribute_inputs(as_dtensor=True)`）；
2. 原 forward——参数保持 DTensor，走 `__torch_function__` dispatch **传播 placement**；
3. **【核心校验】out_src**：DTensor 传播原生输出 vs 声明（`_validate_out_src`）；
4. redistribute 到 out_dst；
5. **【防御性校验】out_dst**：仅 `_is_terminal` 模块；
6. 返回 local（与 production 边界输出同构）。

#### `_validate_outputs(outputs, spec, mesh_dim_names, module_name, stage)`
[sharding_applier.py:L417-446](../../../hyper_models/components/distributed/sharding_applier.py#L435-L464)

out_src/out_dst 共用的校验实现：多输出按 `spec.out_names`（缺省声明 key 顺序）映射 tuple 位置；非 DTensor 输出跳过；比较前 `_normalize_placements_ndim`（[L427-L435](../../../hyper_models/components/distributed/sharding_applier.py#L427-L435)）把 `Shard(-1)` 按实际 ndim 归一；不一致抛 `PlacementMismatchError`。


> **用例**：[`test_dist_s2_production_fwd.py`](../../../tests/components/distributed/test_dist_s2_production_fwd.py)（S2.6：production 前向数值）；[`test_dist_s2_validate_fwd.py`](../../../tests/components/distributed/test_dist_s2_validate_fwd.py)（S2.7：正确 plan 全 pass + 错误声明抛 PlacementMismatchError）。
### 5.2 local region 两函数：`_resolve_local_compute_fn` + `_wrap_local_region_forward`

[sharding_applier.py:L521-589](../../../hyper_models/components/distributed/sharding_applier.py#L539-L607)

**进入条件（派生门控，2026-07-21 二次重构）**：`_resolve_local_compute_fn` 返回**非 None 即走骨架**——门控不是存储的 bool，而是解析链结果；三个声明来源（用户 `local_compute_fn` / planner EP 注入意图 / `use_local_map` 纯门控）优先级固定、互不嵌套，`local_compute_fn` **不改写** `use_local_map`。**功能拆分（2026-07-21 轻量重构）**：EP 关注点集中在解析器，wrapper 是与 MoE 无关的通用骨架。

**`_resolve_local_compute_fn`** [L499-L536](../../../hyper_models/components/distributed/sharding_applier.py#L499-L536)（**单一解析链**，返回 None = 不走骨架）：
1. **`spec.local_compute_fn`（用户自定义，05 §4.4.3）** → `functools.partial(local_compute_fn, module)`——自研数据相关模块复用骨架但注入自己的计算（签名 `fn(module, *local_args, **local_kwargs) -> Tensor`，恒 local tensor、无模式感知；典型：自研 MoE 自定义 router/expert 布局/DeepEP dispatcher）。声明即生效，不改写任何字段；
2. `spec._ep_size>0` 且 expert_mesh 存在（**planner EP 注入意图**，与用户 fn 对等的显式一环）→ `functools.partial(_hf_native_ep_compute, module, router_fn=..., ep_group=expert_mesh.get_group("ep"), tp_group=...)`（**注入全部 EP 计算**，原 forward 不再调用；ep_group 即扩展 EP 组）；
3. `spec.use_local_map`（**纯门控**：模块自身 forward 即数据相关逻辑，自给自足）→ 模块自身 forward（EP-aware 模块，all_to_all 在模块内部）；
4. 以上皆无 → `None`（普通模块，走 validate/production 路径）。

**`_wrap_local_region_forward`** [L539-L607](../../../hyper_models/components/distributed/sharding_applier.py#L539-L607)（通用骨架，原 `_wrap_moe_forward`）：boundary 入口 → local region → 按声明 out_src 重包装 → boundary 出口。**两模式共用同一份代码**（local_region 容错透传语义）。

**`local_region_forward` 四步** [L573-L605](../../../hyper_models/components/distributed/sharding_applier.py#L573-L605)：
1. boundary 入口（validate 时 as_dtensor=True；TP-extend-EP 契约是 SP-in identity，通常直通）；
2. local region：validate 下 `to_local` 输入 + `_temp_local_params`（[L477-L496](../../../hyper_models/components/distributed/sharding_applier.py#L477-L496)，临时解包 DTensor 参数、退出恢复，保持 DTensor 传播链不断）；production 直接跑；
3. 输出非 DTensor → 按声明 out_src `DTensor.from_local` 重包装（恢复 all-to-all 打断的元数据；**out_src 对数据相关模块是声明式校验**——a2a 数据相关，placement 无法派生，这是本质限制）；
4. boundary 出口 → 恒解包为 local 返回。


> **用例**：[`test_dist_s4_moe_local_map.py`](../../../tests/components/distributed/test_dist_s4_moe_local_map.py)（S4.2：toy MoE EP=2 输出 vs 单卡）；[`test_dist_s4_moe_validate_region.py`](../../../tests/components/distributed/test_dist_s4_moe_validate_region.py)（S4.4：D-03' validate 下 DTensor 契约缝合不断链）；[`test_dist_s4_tp_ep_e2e.py`](../../../tests/components/distributed/test_dist_s4_tp_ep_e2e.py)（S4.5：TP=2×EP=2 双模式 e2e，4 进程）；[`test_s4_local_compute_fn.py`](../../../tests/components/distributed/test_s4_local_compute_fn.py)（S4.6：`local_compute_fn` 选择器优先级 + production/validate 骨架注入，单进程）。
### 5.3 inner-wrap（CP attention 系列，D-01''/D-04）——双解析链 + 注册表

**架构（2026-07-21 定稿）**：解析（纯函数）与应用分离，门控派生——`_apply_phase_c` Step 1 在 `cp_mesh` 激活时调 `_wrap_cp_inner_attention(direct=False)`，`_resolve_inner_wrapper` 返回 None（无声明）即不注入。**机制是通用的"定位内部子模块 + 替换其 forward"**，CP（K/V all-gather）是当前唯一内置域。

**链 1：`_resolve_inner_target`** [L614-L647](../../../hyper_models/components/distributed/sharding_applier.py#L614-L647)（纯位置）：
**优先级 0——用户显式指定** `spec.inner_target`（属性名/`"self"`，未命中即 fail-fast）→ 显式 `inner_attention/attn/attention` 属性 → 类名含 SdpaAttention/以 Attention 结尾 → 结构兜底（持 q/k/v_proj）。

**链 2：`_resolve_inner_wrapper`** [L718-L779](../../../hyper_models/components/distributed/sharding_applier.py#L718-L779)（纯行为，返回 `(name, target, apply_fn)` 或 None）：
1. `inner_wrapper` 是 **Callable** → 全自定义（`_apply_custom_inner_wrapper` [L689-L715](../../../hyper_models/components/distributed/sharding_applier.py#L689-L715)：应用 + validate 收 DTensor 时一次性 WARNING 提示双模式容错或声明 `use_local_map`）；
2. `inner_wrapper` 是 **str** → `CP_WRAPPER_REGISTRY` 查找（未知名 fail-fast 并列出可用名；target 缺失 fail-fast 提示 `inner_target`）；
3. `inner_target`/`_needs_cp_attn` 声明 → **启发式 2×2 分派**（`_dispatch_builtin_cp_wrapper` [L682-L686](../../../hyper_models/components/distributed/sharding_applier.py#L682-L686)：`_is_hf_style_attention` [L668-L679](../../../hyper_models/components/distributed/sharding_applier.py#L668-L679) 首参名 `hidden_states` → HF 原语拦截路；`_is_sdpa/_is_flex_attention` [L658-L666](../../../hyper_models/components/distributed/sharding_applier.py#L658-L666) 按 `config._attn_implementation`/类名选 SDPA/Flex）；target 缺失 → fail-fast（缺失 K/V all-gather 是静默数值错误）；
4. 皆无 → `None`。

**协调器：`_wrap_cp_inner_attention`** [L782-L820](../../../hyper_models/components/distributed/sharding_applier.py#L782-L820)：解析 → 应用 → **回写 `spec._resolved_inner_wrapper`** + INFO 日志（target/wrapper/来源；启发式分派会提示可用 str 固定）。`direct=True`（默认）供测试/手动接入（直接调用即显式意图）；`_apply_phase_c` 传 `direct=False`。

**内置注册表 `CP_WRAPPER_REGISTRY`** [L1031-L1036](../../../hyper_models/components/distributed/sharding_applier.py#L1031-L1036)（契约 `fn(target, cp_mesh, *, spec, mesh, mesh_dim_names)`，用户可注册命名方案）：

| 注册名 | 函数 | 机制 |
|---|---|---|
| `"sdpa_hf"` | [`_wrap_hf_sdpa_for_cp` L887-945](../../../hyper_models/components/distributed/sharding_applier.py#L905-L963) | 临时全局替换 `F.scaled_dot_product_attention`（try/finally 还原；非线程安全但单进程 SPMD 训练下安全，与 TorchTitan CP 实现一致），复用 HF 投影/RoPE；**发火检测**：未拦到调用即 RuntimeError（启发式误猜不再静默） |
| `"flex_hf"` | [`_wrap_hf_flex_for_cp` L948-1004](../../../hyper_models/components/distributed/sharding_applier.py#L966-L1022) | 同上，拦截 `flex_attention`（`block_mask` 须按**全局 kv 长度**构建；发火检测同） |
| `"sdpa_qkv"` | [`_wrap_sdpa_for_cp` L833-858](../../../hyper_models/components/distributed/sharding_applier.py#L851-L876) | 包装 `forward(q,k,v,...)`：显式 all-gather K/V + D-04 mask 后调原 forward |
| `"flex_qkv"` | [`_wrap_flex_attn_for_cp` L861-884](../../../hyper_models/components/distributed/sharding_applier.py#L879-L902) | 同上（block_mask 约束同 flex_hf） |

四者公共语义：DTensor 输入 unwrap（validate）→ local 计算 → 按 q placements / spec.out_src 重包装；local 输入透传（production）。production 与 validate 注入**同一个** wrapper（D-01''，区域内逐指令一致）。

#### `_cp_sdpa_call(orig_sdpa, cp_mesh, q, k, v, kwargs)` — CP-aware SDPA 核心
[sharding_applier.py:L805-826](../../../hyper_models/components/distributed/sharding_applier.py#L823-L844)

1. `flex_cp_allgather(k, v, cp_dim=2, cp_mesh)` —— K/V 沿序列维 all-gather；
2. **D-04** [L828-L842](../../../hyper_models/components/distributed/sharding_applier.py#L828-L842)：`is_causal` 且 **CP 激活**（`cp_mesh.size() > 1`，2026-07-21 由 `q_len ≠ kv_len` 形状比较修订为语义判断——GQA 差异在 head 维不影响序列维，但 cross-attention/KV-cache 的 q_len≠kv_len 与 CP 无关，形状推断会把 `lo` 偏移语义用错）时，torch 的 is_causal 按左上角对齐（等价于假设 Q chunk 位于序列开头）、对 rank>0 的 Q chunk 掩码错误（G4）→ 替换为 `_cp_offset_causal_mask(q_len, kv_len, lo=cp_rank*q_len)`（[cp_utils.py:L214-224](../../../hyper_models/components/distributed/cp_utils.py#L214-L224)：允许 attend `j <= lo + i`；rank0 lo=0 退化为标准 causal）。性能注记：显式 `attn_mask` 使 SDPA flash backend 不可选（回退 mem_efficient/math），正确性优先。

#### CP 原语（cp_utils.py）

- **`_AllGatherAlongDim`** [cp_utils.py:L34-55](../../../hyper_models/components/distributed/cp_utils.py#L34-L55)：autograd.Function——forward 沿 cp_dim all-gather + 按 rank 序 cat；backward all-reduce 后取本 rank chunk（reduce-scatter 语义。plain `dist.all_gather` 无 autograd 核，必须显式实现）。
- **`flex_cp_allgather(k, v, cp_dim, cp_mesh)`** [L58-78](../../../hyper_models/components/distributed/cp_utils.py#L58-L78)：CP≤1 直通；通信组取 `cp_mesh.get_group()`——**DeviceMesh 构建时已缓存，此处禁调 `dist.new_group`**（否则每次 forward 泄露一个进程组）。
- **`shard_batch_for_cp(batch, cp_mesh)`** [L83-144](../../../hyper_models/components/distributed/cp_utils.py#L83-L144)：数据管道 CP 切分。pad 到 2·cp 倍数（position_ids 递增 pad、labels pad -100）→ 按 `[cp_rank·chunk, (cp_rank+1)·chunk)` 切片；`seq_lens/seq_lens_padded` 走 `_shard_seq_lens_for_cp`（[L150-211](../../../hyper_models/components/distributed/cp_utils.py#L150-L211)：逐 pack 求与本地区间的交集，跨界截断、哨兵 -1000 保留、输出平移到本地坐标系）。


> **用例**：[`test_s3_inner_attn_detect.py`](../../../tests/components/distributed/test_s3_inner_attn_detect.py)（S3.2：`_resolve_inner_target` 定位 + `_resolve_inner_wrapper` 链（无声明→None/启发式/str 注册表含未知名与 target 缺失 fail-fast/用户注册/callable 自定义）+ `_resolved_inner_wrapper` 回写 + 发火检测 + D-04 cp_size 语义条件）；[`test_dist_s3_cp_allgather.py`](../../../tests/components/distributed/test_dist_s3_cp_allgather.py)（S3.1：`flex_cp_allgather` 前向全局一致 + backward==手工 reduce-scatter）；[`test_dist_s3_cp_qkv_wrapper.py`](../../../tests/components/distributed/test_dist_s3_cp_qkv_wrapper.py)（S3.3：NeMo 风格 (q,k,v) wrapper vs 单卡）；[`test_dist_s3_cp_hf_wrapper.py`](../../../tests/components/distributed/test_dist_s3_cp_hf_wrapper.py)（S3.4：HF 原语拦截——G4 causal + 拦截还原 + 双模式同源）；[`test_dist_s5_cp_same_kernel.py`](../../../tests/components/distributed/test_dist_s5_cp_same_kernel.py)（S5.2：D-01'' 双模式 kernel 级一致）；[`test_dist_s3_tp_cp_e2e.py`](../../../tests/components/distributed/test_dist_s3_tp_cp_e2e.py)（S3.8：TP=2×CP=2 e2e + R8 boundary 无 CP 非 identity op，4 进程）；[`test_s3_shard_batch.py`](../../../tests/components/distributed/test_s3_shard_batch.py)（S3.6：`shard_batch_for_cp` 逐 rank 参数化）；[`test_s3_shard_seq_lens.py`](../../../tests/components/distributed/test_s3_shard_seq_lens.py)（S3.7：pack 完全在内/跨界/在外/哨兵/防空）；[`test_s1_plan_overrides.py`](../../../tests/components/distributed/test_s1_plan_overrides.py)（S1.13：`inner_target`/`inner_wrapper`/`local_compute_fn` 声明**不改写**任何标记，门控派生）。
### 5.4 vocab-parallel embedding（D-02）

- [`_is_vocab_parallel_embed` L1025-1032](../../../hyper_models/components/distributed/sharding_applier.py#L1043-L1050)：nn.Embedding + weight 在 TP 上 Shard(0) + TP>1；
- [`_wrap_vocab_parallel_embedding` L1035-1057](../../../hyper_models/components/distributed/sharding_applier.py#L1053-L1075)：Megatron 风格 masked embedding。

**为什么需要**：DTensor dispatch 的 vocab 范围 mask 逻辑在参数解包后丢失——HF 原生 `F.embedding` 收到全局 token id 会索引越界。

**wrapper 逻辑**（本地 vocab 区间 `[lo, hi) = [rank·V_local, (rank+1)·V_local)`）：
```python
mask = (input_ids >= lo) & (input_ids < hi)
local_ids = where(mask, input_ids - lo, 0)
out = original_forward(local_ids) * mask.unsqueeze(-1)   # 区间外贡献置 0
```
输出天然是 Partial 贡献，boundary 出口 Partial→Shard(1) 归约不变。


> **用例**：[`test_dist_s5_vocab_embed.py`](../../../tests/components/distributed/test_dist_s5_vocab_embed.py)（S5.1：token 区间内/区间外 mask 置 0/边界值三分支，rank1 区间 N-变体）。
### 5.5 MoE/EP 计算流（ep_utils.py）

#### 5.5.1 后端分派 all_to_all

[ep_utils.py:L51-145](../../../hyper_models/components/distributed/ep_utils.py#L51-L145)

- **`_ep_all_to_all(x, send_counts, recv_counts, group)`** [L137-145](../../../hyper_models/components/distributed/ep_utils.py#L137-L145)：统一入口。counts 是 `list[int]`（长度 = 组大小，各 dest/src rank 的行数）。NCCL/HCCL → uneven；gloo 等 → padded。
- **`_EPAllToAllUneven`** [L55-77](../../../hyper_models/components/distributed/ep_utils.py#L55-L77)：`split(x, send_counts) → dist.all_to_all(list) → cat`，零填充；backward 交换 send/recv counts 再做一次（a2a 自逆）。
> 后端能力实证（2026-07-20 实测）：gloo 支持等长 `all_to_all_single`（含 int64）但不支持 list-based 不等长 `all_to_all`；NCCL/HCCL 两者皆支持。这是下述两条路径的分派依据。

- **`_EPAllToAllPadded`** [L80-134](../../../hyper_models/components/distributed/ep_utils.py#L80-L134)：各 dest chunk pad 到**全局** max（`all_reduce MAX` 求得——a2a_single 要求各 rank chunk 等长且全局一致）→ `all_to_all_single` → 按 recv_counts unpad；backward 按 recv pad → a2a_single → 按 send unpad。两路径数值语义一致（pad 行不参与计算）。


> **用例**：[`test_dist_s6_hf_native_moe.py`](../../../tests/components/distributed/test_dist_s6_hf_native_moe.py)（pad-to-max a2a fwd/bwd 对拍，2 进程 gloo；uneven 路径 collective 需 NCCL/HCCL 环境）。
#### 5.5.2 Router adapter 注册表

- **`_softmax_topk_router`** [L152-176](../../../hyper_models/components/distributed/ep_utils.py#L152-L176)：default adapter（Linear gate 的 softmax topk + `norm_topk_prob` 归一化）。返回 `(topk_idx [T,K] int64, topk_w [T,K] float)`。
- **`_topk_router_module`** [L179-193](../../../hyper_models/components/distributed/ep_utils.py#L179-L193)：**qwen3moe/mixtral** adapter——gate 为 TopKRouter 模块（HF 2025 重构后），forward 直返 `(logits, scores, indices)`，取后两个。
- **`_sigmoid_group_router`**（参数来源优先级：模块自身属性 > `module.config`——n_group/topk_group/top_k/norm_topk_prob/routed_scaling_factor 均按此顺序取） [L196-241](../../../hyper_models/components/distributed/ep_utils.py#L196-L241)：**deepseekv3/glm4moe** adapter——sigmoid + `e_score_correction_bias` + group-limited topk（n_group/topk_group，缺省跳过）+ 可选归一化 + `routed_scaling_factor`，与 HF `route_tokens_to_experts` 逐步一致。
- **`MOE_ROUTER_ADAPTERS`** [L256-265](../../../hyper_models/components/distributed/ep_utils.py#L256-L265)：planner 按 arch 名选（`_get_architecture` 去后缀形式如 `qwen3moe`，下划线别名覆盖 model_type 路径）；未注册落 default。


> **用例**：[`test_s5_hf_native_moe.py`](../../../tests/components/distributed/test_s5_hf_native_moe.py)（`test_softmax_topk_router`：default adapter 与玩具模型路由语义一致；`test_topk_router_module_adapter`：qwen3moe adapter 取 TopKRouter triple；`test_sigmoid_group_router_adapter`：deepseekv3/glm4moe sigmoid+correction bias+norm+scaling 对手算参考）。
#### 5.5.3 `_swiglu_weights(experts)` — 权重解析
[ep_utils.py:L272-299](../../../hyper_models/components/distributed/ep_utils.py#L272-L299)

三种布局归一为 `(w_gate, w_up, w_down)`：分离命名（`gate_proj/up_proj/down_proj` 或 `w1/w3/w2`）；**fused 布局（D-11）**：`gate_up_proj [E,2I,H]`（或 automodel 命名 `gate_and_up_projs`）+ `down_proj` → 返回 `(fused, None, down)`，`w_up=None` 标记 fused，计算侧 `chunk(2)` 拆 gate/up。缺矩阵 → NotImplementedError。


> **用例**：[`test_s5_hf_native_moe.py`](../../../tests/components/distributed/test_s5_hf_native_moe.py)（`test_swiglu_weights_two_naming_families`：分离两套命名；`test_swiglu_weights_fused_layout`：D-11 fused（gate_up_proj/gate_and_up_projs）→ (fused, None, down)）。
#### 5.5.4 `_hf_native_ep_compute(module, hidden_states, *, router_fn, ep_group, tp_group=None)` — TP-extend-EP 前向（核心中的核心）
[ep_utils.py:L310-396](../../../hyper_models/components/distributed/ep_utils.py#L310-L396)

**功能**：SP-in（本地序列 chunk）→ region 内全部通信 → SP-out（本地 chunk）。通信流与 Megatron `MoEAlltoAllTokenDispatcher`（etp=1 配置）逐步对齐：router（无通信）→ a2a（扩展 EP 组）→ 本地 SwiGLU（完整 expert 权重，无 Partial）→ a2a 返回。**无 all_gather/reduce_scatter**——expert 权重不做 hidden 维切分，region 内不存在任何归约点。

**输入**：`hidden_states [B, S/tp, H]`（本地 chunk）；`ep_group`（扩展 EP 组 = 派生 expert mesh 的 ep 轴，大小 = ep_size，含 TP rank）。
**输出**：`[B, S/tp, H]` complete（边界 identity）。

**逐步展开**（以 mesh (dp=4,tp=2)、ep=4、num_experts=4、top_k=2 为例：扩展 EP 组 {0,1,2,3}，每 rank 持 `e_local=1` 个完整 expert）：

| 步 | 代码 | 做什么 | 张量形状变化（本例） |
|---|---|---|---|
| 0 | [L327-337](../../../hyper_models/components/distributed/ep_utils.py#L327-L337) | 解析权重；`e_global = e_local × ep_size`；`expert_offset = ep_rank × e_local`；x 展平 [T,H] | T = B·S/2 |
| 1 router | [L338-344](../../../hyper_models/components/distributed/ep_utils.py#L338-L344) | 本地 chunk 上路由（**无通信**）；`dest = flat_idx // e_local`（目标 EP rank） | topk_idx [T,2] → flat [2T] |
| 2 排序+counts | [L346-354](../../../hyper_models/components/distributed/ep_utils.py#L346-L354) | 按 `(dest, expert)` 排序得 `perm`；`send_counts = bincount(dest)`；counts 经 `all_to_all_single` 交换得 `recv_counts` | send_x [Σsend, H] |
| 3 a2a dispatch | [L356-359](../../../hyper_models/components/distributed/ep_utils.py#L356-L359) | `_ep_all_to_all` 过扩展 EP 组：token（带完整 H）发往 expert 持有 rank（可跨 TP/dp 坐标）；expert id 同步交换 | recv_x [Σrecv, H] |
| 4 本地 SwiGLU | [L361-376](../../../hyper_models/components/distributed/ep_utils.py#L361-L376) | 按 expert id 排序 → 逐本地 expert SwiGLU（**完整 expert 权重**，输出 complete，**无 Partial**）；fused 布局（`w_up is None`）走 `linear(x, gate_up[i]).chunk(2)` 分支 → 逆排序还原 | y [Σrecv, H] |
| 5 a2a combine | [L381-384](../../../hyper_models/components/distributed/ep_utils.py#L381-L384) | 反向 a2a 回扩展 EP 组 → 逆 `perm` → 按 `topk_w` 加权、按 K 聚合 → reshape | out [B, S/2, H] |
| 6 shared_experts | [L387-395](../../../hyper_models/components/distributed/ep_utils.py#L387-L395) | 若存在：本地 chunk × TP 分片权重 → Partial → **TP 组 all_reduce** 后相加 | — |

**关键不变式**：a2a 去程/返程在同一扩展 EP 组上互逆，每个 token 严格回到源 rank——边界两侧序列 chunk 布局全程不变（§05 文档 §6.4.8 的三个不变量）。

**autograd 闭环**：步骤 3/5 的 a2a 是自定义 autograd.Function（a2a 自逆，反向自动镜像）；步骤 4 是纯 local GEMM，普通 autograd；expert 权重梯度直接落在各 rank 的完整 expert local shard 上（无需 TP 归约，tp_grad_info 标 Shard(1) 防 FSDP 错误 all-reduce）。

---


> **用例**：[`test_dist_s6_ep_extend.py`](../../../tests/components/distributed/test_dist_s6_ep_extend.py)（per-expert 与 batched 两布局各一组 8 进程 e2e：mesh (dp=4,tp=2) ep=4、扩展 EP 组跨 TP×dp 坐标、双模式 vs 单卡）。
## 6. 独立工具

### 6.1 `local_region` — DTensor→local→DTensor 区域包装

[local_region.py:L110-230](../../../hyper_models/components/distributed/local_region.py#L110-L230)

**功能**：把任意 func 包装为局部计算区域（前向 only）。validate 模式的 MoE/CP 场景由 `_wrap_local_region_forward`/CP wrapper 内联手写实现（§5），本函数服务**独立使用**场景。

**参数**：`device_mesh`；`in_placements={arg_name: placements}`（缺省不 redistribute）；`out_placements`（单输出扁平写 `(Partial(), Replicate())`，多输出逐位置写、非 tensor 用 None 占位）；`redistribute_inputs=False`（双模式场景边界通信已由 PrecompiledBoundary 完成）。

**容错透传语义**（三点增强，[L169-223](../../../hyper_models/components/distributed/local_region.py#L169-L223)）：
1. 非 DTensor 输入原样透传（production 已解包场景）；
2. 输出已是 DTensor 不重复包装；
3. 全部输入均非 DTensor → 输出也不包装。

**无反向缝合**：自研 DTensor 是前向-only 系统（不存在 DTensor autograd），区域内部反向就是 local tensor 普通 autograd，梯度直接落在 local 参数分片上，与 production 一致——区别于 PyTorch `local_map`（torch DTensor 有反向语义）。

辅助：[`_bind_arg_names` L63-79](../../../hyper_models/components/distributed/local_region.py#L63-L79)（签名内省，positional→名映射；C 扩展式签名不可内省时回退为空 dict——此时仅 kwargs 传参能命中 `in_placements`，positional 参数全部直通）、[`_normalize_out_placements` L82-107](../../../hyper_models/components/distributed/local_region.py#L82-L107)（扁平/逐输出两种写法归一）。


> **用例**：[`test_local_region.py`](../../../tests/components/distributed/test_local_region.py)（单进程：命名参数绑定、容错透传三分支、out_placements 归一化）。
### 6.2 `testing/grad_equiv` — 双模式梯度等价

[testing/grad_equiv.py](../../../hyper_models/components/distributed/testing/grad_equiv.py)

两模式 backward **同为 local 路径**（无 "DTensor backward" 对照组），梯度等价直接逐参数比较：
- [`run_one_step` L36-46](../../../hyper_models/components/distributed/testing/grad_equiv.py#L36-L46)：单步 forward+backward，返回 `{param_fqn: grad}`；
- [`assert_grad_equivalence` L49-57](../../../hyper_models/components/distributed/testing/grad_equiv.py#L49-L57)：双模式梯度逐参数 assert_close；
- [`simulate_tp_replicate_grad_sync` L60-69](../../../hyper_models/components/distributed/testing/grad_equiv.py#L60-L69)：模拟 tp_grad_info 旁路（TP-Replicate 参数梯度的 TP all-reduce）——真实路径由 FSDP2 fork 完成，此处用于独立验证。

**已知语义**（05 §12.5）：EP 组各 rank 共享同一 batch 时，expert 收到的 token 数是单卡的 world 倍 → expert 梯度 = world_size × 单卡梯度（测试按此断言，非 bug）；router 梯度保持 1×（per-copy）。

---


> **用例**：[`test_dist_s5_grad_equiv.py`](../../../tests/components/distributed/test_dist_s5_grad_equiv.py)（S5.3：双模式梯度等价——TP-Shard 与 TP-Replicate 两类参数）；[`test_dist_s5_mode_equiv.py`](../../../tests/components/distributed/test_dist_s5_mode_equiv.py)（S5.4：validate vs production 输出等价 TP/TP×CP/TP×EP 三组合）。
## 7. 端到端走查：mesh (dp=4,tp=2)、ep=4 的 HF 原生 MoE 模型（用户示例拓扑）

把 §2-§5 串起来（对应测试 `test_dist_s6_ep_extend.py::_worker_ep_extend_e2e`，8 进程）：

```
build:
  planner.plan(model, mesh(dp=4,tp=2), tp=2, ep=4)
    → mesh_dim_names=("tp",)；ep_extend=4
    → mlp 边界推断 "moe_mlp" → _mark_hf_native_moe 命中
      校验: dense=8, ep=4≤8 且整除 ✓, num_experts=4%4=0 ✓ → 每 rank 1 个完整 expert
      spec.params: experts.gate_proj={EP:Shard(0)}, ..., gate.weight={TP:Replicate,...}
      spec._ep_size=4, 边界 identity
  apply_sharding_plan(model, plan, mesh)
    → expert_mesh = (2,4)("edp","ep")：扩展 EP 组 {0,1,2,3}/{4,5,6,7}（跨 TP 组 × dp）
    → Phase A: _stack_moe_experts（ModuleList→[4,...] stacked）
               experts.* 在 expert_mesh 分片（EP Shard(0)：每 rank 1 个完整 expert）
               gate.weight 在主 mesh 分片
    → Phase C(production): _local_params_context 永久解包；
               build_tp_grad_info（experts.* 标 Shard(1)，防 FSDP 错误 all-reduce）
               _resolve_local_compute_fn → compute_fn = _hf_native_ep_compute(ep_group, tp_group)  # 链环 2：EP 注入意图

runtime forward（每层 mlp）:
  hidden [B, S/2, H] ─identity boundary→ _hf_native_ep_compute
    → router(本地 chunk) → a2a(扩展 EP 组 {0,1,2,3}) → SwiGLU(完整 expert)
    → a2a 返回 → 加权聚合 → [B, S/2, H] ─identity boundary→ 下游
  （无 all_gather/reduce_scatter——expert 权重无 hidden 维切分，无 Partial 归约点）
  双模式数值一致：production（local）与 validate（DTensor 传播 + 声明校验）输出逐位相等
```

---

## 8. 速查：常用入口与典型用法

```python
from hyper_models.components.distributed import (
    ShardingPlanner, ModuleShardingSpec, apply_sharding_plan,
    validate_model_compatibility, shard_batch_for_cp,
)
from hyper_models.components.distributed.sharding_config import TP, CP
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate

# 0. 预检
validate_model_compatibility(model, tp_size=2, ep_size=2)

# 1. 自动推导 + 用户覆盖（自研模块）
my_spec = ModuleShardingSpec(
    params={"weight": {TP: Shard(0)}},
    in_src={"hidden_states": {TP: Shard(1)}},
    in_dst={"hidden_states": {TP: Replicate()}},
    out_src={"output": {TP: Shard(1)}},
    out_dst={"output": {TP: Shard(1)}},
)
plan = ShardingPlanner(plan_overrides={"model.custom": my_spec}).plan(
    model, mesh, tp_size=2, ep_size=2)

# 2. 应用（production / validate 仅一个开关之差）
model, tp_grad_info = apply_sharding_plan(model, plan, mesh, validate_mode=False)

# 3. CP 数据管道
batch = shard_batch_for_cp(batch, mesh["cp"])
```
