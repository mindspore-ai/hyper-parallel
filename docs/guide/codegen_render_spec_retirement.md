# 消灭 per-family render_spec：让「输入只有 YAML」成立

> 日期：2026-09-17 · 分支：`codegen_restructure`
> 目标：**移除 `models/<family>/adapter/render_spec.py`，让「原生 HF 模型 + 一份 YAML」成为唯一输入，模型接入不再写模型专属代码。**
> 前置：`docs/guide/codegen_structure_driven_plan.md`（结构驱动方案，本档是它的落地判定清单）。

## 1. 为什么必须删

方案 §0 结论 3 原文：**「YAML 即足够」可达**，条件是——把 EP/codegen 层的选择从「模型名」改回「模块结构」。方案 §2 表格已经把 `render_spec.py` 的命运钉死为 **目标状态「去掉」（内容可从结构推导）**。

当前它还在，是**过渡态**，不是目标态。用户验收标准：

> 我的输入就只有 YAML，不应有任意提前适配模型的代码存在。

## 2. 一个前提要先纠正

「EP 逻辑必须完整内联进产物」**不成立**。`codegen/inline/expansion.py` 的 `_PRIMITIVE_PREFIXES`（collectives / experts / runtime 等框架层）在产物里**只保留 import/call，不内联**。所以「展开到某一层级就让用户去方法里看」是**既有设计**，不是妥协。

但删 render_spec 与「展开深度」**无关**，关键只在：产物里那段 EP forward 指向**框架通用实现**（由结构选中），还是 **per-family 字面量**。红线：

- ❌ 产物留空壳/占位让用户自己补 → 违背 §1.1「产物能真实拉起训练」
- ❌ 每族一个 runner（只是把 body 挪到 `adapter/distributed/*.py`）→ 只是换文件名，`_target_` 仍在，没真删

## 3. 判定清单（每条声明是否已可由结构推导吸收）

判据：render_spec.py 每个字段的**手写值 == 结构推导值**（逐字段等价 → 产物逐字节相等）。任一字段不等 = 未删除合格。

| # | 声明项 | 位置 | 覆盖 | 判定 | 删除所需动作 |
|---|---|---|---|---|---|
| 1 | `meta_normalizers` | 已无此字段 | ✅ 已空 | **已就绪可删** | 无 |
| 2 | EP forward `body_template` | qwen=框架 `templates.py`；deepseek=render_spec 手写字符串 | 🟡 可派生 | **需合并** | 两段合成**一个**通用 EP 模板，差异点（router key / additive shared）由 `MoeStructure(router_kind, shared)` 参数化 |
| 3 | strategy `imports` | 模板 body 依赖 | 🟡 | 随 #2 并入框架模板 | 通用模板固定符号表 + `expansion.py` 兜底 |
| 4 | `snippets`（`PARALLEL_STATE_ACCESSOR`/`TP_OPERATORS_CLASS`） | 框架 `templates.py` | ✅ 已在框架层 | 保留，非残留 | 引用移入通用模板 |
| 5 | `strip_boundary_subpatterns`（`experts.*`,`shared_experts`） | deepseek render_spec | 🟡 可派生 | **需补推导** | `storage==module_list→experts.*`；`shared==additive→shared_experts` |
| 6 | `kind` 族名键（`qwen3_moe_ep_routed_forward`） | 已无 | ✅ 已去族名 | **已完成** | structure-keyed，`kind` 仅作描述性元数据 |
| 7 | `target_class`（`Qwen3MoeSparseMoeBlock`/`DeepseekV3MoE`） | strategy_specs | ✅ 由 archetype 投影 | **已完成** | 由类名改结构描述符（`moe_archetype(*)`） |
| 8 | `external_state_classes` | 已无 | ✅ 由结构推导 | **已完成** | 由注入的 EP archetype `target_class` + 族自身生成的 `GQAAttention` 派生 |
| 9 | `replacement_specs`（RMSNorm/GroupedExperts/GQAAttention 源类→组件） | 已无 | ✅ 已泛化 | **已完成** | `recognition.py` 读工厂源码 AST 识别组件种类，`components.py` 按种类给出 wiring；源类名取自 YAML `module_type` |

## 4. 六步退役路径（从最难到最易）

**Step 1 · 升级结构决策器（地基，非删项）**
`structure.py` 现在只输出 `MoeStructure(router, shared, storage, jitter)`、只认 MoE。升级为同时决策：模块替换集合、external_state、EP/CP 策略、subpattern。此步不改行为，只加推导面。

**Step 2 · 消灭手写 body（#2 #3 #5）**
deepseek 手写 body 与 templates 的 qwen body 合并成**一个框架通用 `moe_ep_forward`**，由 `MoeStructure(router_kind, shared)` 参数化（唯一差异：router key + 是否加 `shared_experts`）。产物 EP forward 只薄薄调用。`imports`/`strip_boundary_subpatterns` 随进模板或由结构派生。→ **策略体字面量消失**。

**Step 3 · 去 kind 族名键（#6）· ✅ 已完成**
strategy 注册表 key 从族名改**结构 key**（`moe_ep_routed`，Qwen3-MoE 与 DeepSeek-V3 共用），`kind` 不再承载族名；`kind` 仅作描述性元数据，源码生成从不派发它（已确认 `pipeline.py`/`strategy_pass.py` 无 `kind` 分支）。→ **族名锚消失**。

**Step 4 · target/external 改 archetype（#7 #8）· ✅ 已完成**
新增框架表 `hyper_parallel/distributed/expert_parallel/archetypes.py`：以 `detect_moe_structure` 指纹（router/shared/storage）为唯一键，持有 `kind` / `target_class` / `strip_boundary_subpatterns` / 运行时 `router_kind` / `shared` / 计算工厂 `_target_`。两族 render_spec 改为**投影**该表（`moe_archetype(*)`、`moe_external_state_classes(*)`），不再手写类名清单；`manager._EP_FACTORY_BY_STRUCTURE` 亦改为投影同一张表（单一事实源）。外部状态类由「被内联的 block 类」结构派生 + 各自替换声明贡献（Qwen 的注意力组件用其自身 `new_ctor`）。→ **类名清单消失**。

**Step 5 · 模块级替换去族化（#9，最硬）· ✅ 已完成**
新增 `hyper_parallel/codegen/inline/components.py`：`rms_norm_replacement` / `grouped_experts_replacement` / `gqa_attention_replacement` 三个框架通用构造器，把组件 wiring（import、ctor 形状 `name`/`wrap_source`、keyword-only 实参、是否保留源类）收敛为框架知识，按**组件种类**参数化。render_spec 只声明「目标路径 → 通用组件 + 匹配的源类」，唯一保留的族内事实是源类名（`Qwen3MoeRMSNorm`）与注意力核入口符号（`run_qwen3_moe_flash_attention`）。→ **替换决策消失**。

**Step 6 · 收尾（#1）· ✅ 已完成**
`meta_normalizers` 已确认全空；qwen 的 `_build_attention_replacement_spec` 与两族 EP/CP 策略声明**全部改由结构推导**：
新增 `codegen/inline/framework_spec.py`（结构解析 + 声明摘要 + 缓存）、`codegen/inline/recognition.py`
（从工厂源码 AST 识别通用组件与核入口，源类名取自 YAML `module_type` 而非手写字面量）、
`codegen/inline/components.py`（`rms_norm` / `grouped_experts` / `gqa_attention` 三个按组件种类参数化的通用构造器）。
`pipeline.py` / `replacement_pass.py` / `strategy_pass.py` 均改为「结构 → 声明」，`manager.py` 的签名改为声明摘要
（`inline_declarations`）并按结构推导 `external_state_classes`。**两个 `render_spec.py` 已删（进回收站）**，
`ModelAdapterSpec.inline_codegen` 字段与两族 registration 的接线一并移除。`tests/codegen` 127 passed。

判据落到「结构推导产物 == 原 render_spec 产物」已验证：两族 EP 策略 / CP 策略 / 三个替换声明 / `external_state_classes`
的输出与原 render_spec 逐一等价（`tests/codegen/test_archetypes.py::TestDeclarationsAreAStructuralProjection`）。

## 5. 验收判据

每步做完：`结构推导产物 == 原 render_spec 产物`（逐字节一致，对齐方案 §8 S3 安全网）。**每条声明翻空一截，全部翻空即删文件**，不硬删、不留空壳。

- ✅ 产物能真实拉起训练（8 卡 loss / grad_norm 与 native 逐 step 对齐）
- ✅ 唯一输入是「原生 HF 模型 + YAML」，无模型专属代码
- ❌ 不允许「产物留空占位让用户自己补」

## 6. 规模与优先起点

6 步即 S2–S5 的真实工作总量。最实惠、纯派生、零行为变化的起点是 **Step 2**：EP 通用 runner + 产物薄 forward，直接消除 #2 #3 #5，且能先用 129 逐 step 对齐验证方向可行。

**建议下一步**：动手 Step 2 的最小 diff（deepseek + qwen 合并成一个 `moe_ep_forward`）→ 129 验证 loss / grad_norm 仍逐 step 对齐。