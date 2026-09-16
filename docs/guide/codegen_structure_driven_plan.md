# Codegen 结构驱动方案

> 日期：2026-09-16 · 分支：`codegen_restructure`
> 目标：让「原生 HF 模型 + 一份 YAML」就能得到可训练的并行产物，模型接入不需要写模型专属代码。

---

## 0. 结论摘要

三条结论，按重要性排列：

1. **框架已经把最难的部分做完了。** 所有高性能替换工厂（`@module_replacement`）**已经在通用层**
   `hyper_parallel/components/modules/`；模型 adapter 里剩下的工厂是**薄包装（Middle Man）**，
   其函数体基本只有一行委托。见 §4.2。

2. **adapter 的 provider 机制只接线了 3/8。** `ModelAdapterSpec` 声明了 8 个 provider，
   真正被读取的只有 `sharding_rules` / `loss` / `inline_codegen`；其余 5 个是空壳，
   所以 YAML 才必须写 `_target_`。见 §4.5。

3. **「YAML 即足够」可达**，条件是：把 EP/codegen 层的选择从「模型名」改回「模块结构」，
   并把匹配结果落到 `generated/` 里的产物（而不是提交进仓库的模型专属代码）。
   见 §5。

---

## 1. 目标（设计原则）

### 1.1 产品目标

支持原生 HuggingFace 生态：原生 HF 模型用 codegen 工具即可注入并行能力，
生成的模型文件向用户展示**完整训练脚本**——包含原生模型代码，以及后续 patch 的模块
（高性能算子、Hyper 的 TP / CP / EP 能力）。

### 1.2 开发原则（本文档必须逐条满足）

| # | 原则 | 落实 |
|---|---|---|
| 1 | YAML 为统一配置入口；`codegen: true/false` 决定编译期是否生成 codegen 模型文件 | 保留；见 §5.1 |
| 2 | codegen 生成 meta，记录哪些模块已被替换、替换源从哪 import，runtime 据此判断 | 保留并扩展；见 §5.4 |
| 3 | `auto_model.py` 用 `is_hf_model` 解析项决定是否走 codegen 产物，三态 `hf`/`custom`/`gen` | **已实现**；见 §4.7 |
| 4 | codegen 不含 mesh / fsdp2；但用 codegen 产物可跳过 sharding_plan 的 apply | **已实现**；见 §4.7 |
| 5 | patch 替换配置在 YAML `plan_overrides` 字段 | 保留；`_target_` 变为可选，见 §7 |

### 1.3 预览样例的定位与验收标准

**`preview_modeling_veomni_style.py` 是「预期形态的示例」，不是「逐字节的生成规范」。**

它说明产物应当**长成什么样**：

- 该替换的模块被替换（`RMSNorm` / `GroupedExperts` / `GQAAttention`）
- TP / CP / EP 的策略插在**语义正确的位置**：attention 内联 TP `all_gather` / `reduce_scatter` 与
  CP 的 K/V all-gather；MoE 内联 EP 的 dispatch → local experts → combine
- HF 原生代码逐字保留，只有 patch 处带标记

它**不**意味着：

- ❌ 产物必须与这个文件逐字节相同
- ❌ 为了得到这个文件而**硬编码匹配**——那正是本方案要消除的「模型名驱动」

**真正的验收标准是功能性的**：

| 层级 | 判据 | 性质 |
|---|---|---|
| **最终验收** | 产物能**真实拉起训练**；8 卡上 loss / grad_norm 与 native 路径逐 step 对齐 | **这是目标** |
| 过渡安全网 | 结构推导出的产物 == 当前手写声明产出的产物（逐字节） | 仅用于证明「两种机制等价」，是**临时回归保护**，不是目标 |

§8 的 S3 用逐字节一致验收，属于后者——它证明的是「结构推导没有改变行为」，
不等于「产物符合某个模板」。产物是**从 HF 源码 + 结构识别推导**出来的，
不是照着预览文件配出来的。

---

## 2. 术语：三个被混用为 "inline" 的东西

命名歧义是本方案前期讨论混乱的根源（包括文档作者自己也混过），先钉死：

| 名称 | 是什么 | 位置 | 命运 |
|---|---|---|---|
| **渲染管道** | replacement pass + strategy pass + patch engine | `hyper_parallel/codegen/inline/`（框架层） | ✅ 保留 |
| **render spec** | 每个模型族手写的源码级声明 | `models/<family>/adapter/render_spec.py` | ❌ 目标状态**去掉**（内容可从结构推导） |
| **「展开」动作** | 把编排函数的函数体内联进产物 | 机制属于渲染管道 | ✅ 保留并强化（见 §6） |

**已完成的重命名**：`adapter/inline.py` → `adapter/render_spec.py`，
其访问函数 `get_inline_spec_bundle()` → `get_render_spec()`。
`hyper_parallel/codegen/inline/` 保持不变（它名副其实——把逻辑内联进产物）。

`ModelAdapterSpec.inline_codegen` 字段名与 `InlineSpecBundle` 类型名保持不变：它们命名的是
**渲染管道**这一特性，不是那个文件。

---

## 3. 现状诊断

### 3.1 三层对照

| 层 | 驱动方式 | 加一个标准模型要写什么 |
|---|---|---|
| sharding（planner） | ✅ **结构驱动** | 无 |
| FSDP / checkpoint | ✅ 结构驱动 | 无 |
| EP archetype 选择 | ❌ **模型名驱动** | 登记 `MOE_ROUTER_ADAPTERS` + 写工厂 |
| codegen 源码渲染 | ❌ **模型名驱动** | 写 `adapter/render_spec.py` |

### 3.2 根因：YAML 的 `_target_` 指向模型专属工厂

`plan_overrides` 里每一条都通过 `_target_` 指向模型专属 Python 工厂：

```yaml
- match: "*.mlp"
  when: ep
  local_compute_fn:
    _target_: hyper_parallel.models.qwen3_moe.adapter.distributed.expert_parallel.qwen3moe_ep_compute_fn
```

这个 dotted path 是「YAML 声明」与「模型专属实现」之间的唯一纽带。它一旦存在：

1. 工厂必须存在 → `adapter/distributed/expert_parallel.py`（手写）
2. 源码渲染规则必须挂在这个 path 上 → `adapter/render_spec.py`（手写）

**渲染管道只是让这个间接层的成本变显眼，本身不是根因。** 同样的工厂在 generic（运行时 hook）
路径下也必需——「YAML 就够」这个前提在渲染管道引入之前就不成立。

### 3.3 分支演化

| | `origin/master` @024bc2ae | `codegen_master` @4810ab25 | `codegen_master_v3` @0b8e31c2 |
|---|---|---|---|
| `hyper_parallel/codegen/` | **不存在** | 32 个文件 | 46 个文件 |
| `inline/` 子目录 | — | ❌ | ✅ 10 个文件 |
| `plan/boundary_forms.py` | — | ❌ | ✅ 351 行 |
| `plan/slim.py` | — | ❌ | ✅ 157 行 |
| `adapter_spec` providers | 7（含 `loss`） | 6 | 8（`+loss +inline_codegen`） |
| 产物形态 | — | generic literal | **双形态** |

**`codegen_master` → `v3` 净增 `+3402 / −366` 行**（`runtime.py` 单项 +746）。
膨胀来源是 v3 在 `codegen_master` 之上**叠加**了第二套产物路径，而不是替换它——
这正是 `codegen_restructure` 分支要消除的（§11）。

---

## 4. 关键证据

### 4.1 sharding 层确实不看模型名

`hyper_parallel/distributed/_builder/planner.py:904-907`，`_mark_hf_native_moe`：

```python
# ``arch`` is part of the dispatch-interface signature shared with the
# other boundary post-processors; the EP mode is decided by the mesh
# and layout alone, so it is intentionally unused here.
_ = arch
```

拿到 `arch` 参数但**故意不用**，靠 `_detect_expert_layout` 识别专家布局
（per-expert / batched / custom），按 mesh + layout 决策。

这就是 `models/qwen2_moe/adapter/registration.py` 只声明 `sharding_rules` 的原因，
其注释写明：「The family otherwise uses HF-native model code with the generic templates.」

### 4.2 【决定性】替换工厂已经在通用层

全仓 `@module_replacement` 的分布：

**通用层 `hyper_parallel/components/modules/`**（实现的家）

| 文件 | 工厂 |
|---|---|
| `rms_norm.py` | 2 个（30, 82） |
| `gqa_attention.py` | 2 个（55, 317） |
| `grouped_experts.py` | 1 个（41） |
| `mla_attention.py` | 1 个（34） |
| `dsa_attention.py` | 2 个（147, 522） |
| `shared_expert.py` | 1 个（79） |
| `swiglu_mlp.py` | 1 个（36） |
| `mhc.py` | 2 个（72, 154） |
| `quantization/modules/` | 2 个 |

**模型 adapter**（薄包装）

| 文件 | 工厂 |
|---|---|
| `qwen3_moe/adapter/replacements.py` | 3 个（39, 58, 116） |
| `deepseek_v3/adapter/replacements.py` | 2 个（55, 69） |

看 qwen3_moe 三个工厂的**函数体**：

```python
@module_replacement
def replace_qwen3_moe_rms_norm(*, module, module_fqn, context):
    from hyper_parallel.components.modules import RMSNorm
    return RMSNorm(module=module, module_fqn=module_fqn, context=context)   # 一行

@module_replacement
def replace_qwen3_moe_grouped_experts(*, module, module_fqn, context):
    _validate_batched_experts_contract(module, module_fqn)   # 结构性校验，与 Qwen3 无关
    from hyper_parallel.components.modules import GroupedExperts
    return GroupedExperts(module=module, module_fqn=module_fqn, context=context)
```

`_validate_batched_experts_contract` 检查的是 `gate_up_proj` / `down_proj` 是否为
`nn.Parameter`——纯结构判断。adapter 自己的 docstring 也写明：

> the NPU kernels live in `functional` — **no second family implementation exists here**

**结论**：「换成哪个类」这个选择完全可由模块结构决定。模型 adapter 里的工厂是
Middle Man——**删除测试通过**：删掉它，复杂度消失（运行时能推导出同样的东西）。

### 4.3 EP archetype 已是结构契约，只是名字带了模型名

`hyper_parallel/distributed/expert_parallel/recipes.py`：

| 函数（行） | `archetype_key` | `expected_attrs`（结构签名） | `combine`（结构差异） |
|---|---|---|---|
| `routed_only_ep_compute_fn` (183) | `routed_only_softmax_topk` | `gate, experts` | identity |
| `mixtral_ep_compute_fn` (208) | `mixtral_topk_router` | `gate, experts` | identity（训练期 router jitter） |
| `qwen2moe_ep_compute_fn` (247) | `qwen2moe_shared_expert_gate` | `gate, experts, shared_expert, shared_expert_gate` | `routed + shared * gate` |
| `deepseekv3_ep_compute_fn` (289) | `deepseekv3_sigmoid_group_shared` | `gate, experts, shared_experts` | `routed + shared` |
| `qwen3moe_ep_compute_fn`（adapter） | `qwen3moe_topk_router` | `gate, experts` | identity |

`expected_attrs` 就是结构签名，`combine` 就是结构差异。**archetype 本身是结构变体，
只是被冠上了模型名。**

### 4.4 `MOE_ROUTER_ADAPTERS` 的 1-to-many 暴露 key 选错了

`hyper_parallel/distributed/expert_parallel/routing.py:129-140`：

```python
MOE_ROUTER_ADAPTERS = {
    "default":   _softmax_topk_router,
    "qwen2moe":  _topk_router_module,      # ┐
    "qwen3moe":  _topk_router_module,      # ├ 三个模型名，同一个函数
    "mixtral":   _topk_router_module,      # ┘
    "deepseekv3": _sigmoid_group_router,   # ┐
    "glm4moe":    _sigmoid_group_router,   # ┘ 两个模型名，同一个函数
}
```

**值全是结构性的，key 却是模型名。** 按结构 key 的判据：

- `_softmax_topk_router` ← `gate` 是普通 `nn.Linear`
- `_topk_router_module` ← `gate` 是带 forward 的 router 模块
- `_sigmoid_group_router` ← `gate` 有 `n_group` / `topk_group` / `routed_scaling_factor`

### 4.5 【决定性】provider 机制只接线了 3/8

grep `get_model_adapter()` 的**全部调用点**，实际被读的 provider：

| provider | 消费者 |
|---|---|
| `sharding_rules` | `distributed/_builder/planner.py:565-568` |
| `loss` | `components/losses/chunked_cross_entropy.py:301-302` |
| `inline_codegen` | `codegen/inline/specs.py:34` |
| `replacements` | ❌ 无（YAML `_target_` 直接指） |
| `attention` | ❌ 无 |
| `checkpoint` | ❌ 无 |
| `context_parallel` | ❌ 无 |
| `expert_parallel` | ❌ 无 |

`models/qwen3_moe/adapter/registration.py:76-85` 明明声明了 `attention` / `context_parallel` /
`expert_parallel` / `replacements`——**声明了，没人读**。

**这解释了为什么 YAML 必须写 `_target_`：provider 那条路根本没通。**
所以「把 YAML 指向改成 provider」这个方向不成立——**provider 机制本身要先接线，或者删掉。**

同类死字段（累计 3 处）：

| 字段 | 状态 |
|---|---|
| `InlineSpecBundle.external_state_classes` | 声明了，无人读（runtime 硬编码 frozenset） |
| `ModelAdapterSpec.attention` 等 5 个 | 声明了，无人读（见上表） |

### 4.6 `attention_interface` 的调用链：硬编码，不经过任何机制

```
① YAML
   _target_: ...qwen3_moe.adapter.replacements.replace_qwen3_moe_flash_attention
                      ↓
② models/qwen3_moe/adapter/replacements.py:58-81
   return GQAAttention(
       module=..., module_fqn=..., context=...,
       attention_interface=run_qwen3_moe_flash_attention,   ← ★ 硬编码在这一行
   )
                      ↓
③ components/modules/gqa_attention.py:167-175
   def __init__(self, *, module, module_fqn="", context=None,
                attention_interface=npu_fusion_attention_forward):   ← 有默认值
       self.attention_interface = attention_interface
                      ↓
④ gqa_attention.py:300 / :533
   attn_output, attn_weights = self.attention_interface(self, query, key, value, ...)
```

**没有任何「模型 → attention interface」的注册或识别机制**，只有 ② 那一行硬编码。

`run_qwen3_moe_flash_attention` 与默认实现的实际差异：

| | 默认 `npu_fusion_attention_forward`<br>`components/functional/npu_fusion_attention.py:224` | Qwen3 覆盖<br>`models/qwen3_moe/adapter/attention.py:55` |
|---|---|---|
| mask 约定转换 | 也做 | 也做（`torch.logical_not`） |
| **`sparse_mode`** | `_attention_options(module, kwargs)` **动态推导** | **硬编码**：mask 为 None → `2`；否则 `0` |
| **compressed causal mask** | 无 | ✅ 2048×2048 缓存 mask（left-up causal 稀疏模式） |
| packed 序列 | ✅ 支持 | ❌ 不支持 |
| layout | 自适应 | 固定 `BNSD` |

唯一实质差异是 **left-up causal 稀疏模式**。这看起来是 **kernel 模式/性能选择**，
不是模型结构必需（mask 转换两边都做，packed 支持反而是 Qwen3 版更弱）。
**待确认**：找写它的人确认当初为何单给 Qwen3 开这个模式。

### 4.7 已经满足的机制（不要重做）

- **三态 backend**：`codegen/modeling_backend.py` 已实现 `hf` / `custom` / `gen` 与优先级解析（原则 3）。
- **跳过 sharding_plan 重算**：`codegen/manager.py:588` 设 `meta.covered["sharding_plan"] = True`；
  `loader._parallelize_from_generated` 以此为闸门（原则 4）。
- **GEN 视为 HF infrastructure**：`auto_model.py:472`
  `infrastructure_is_hf_model = is_hf_model or backend is ModelingBackend.GEN`。
- **meta 记录替换来源**：`compile_overrides_from_meta` 产出 `meta.module_overrides`
  （工厂 dotted path + module_type + fqns），即原则 2 的载体。

---

## 5. 目标形态

### 5.1 YAML 契约

**现在**：

```yaml
- match: "*.mlp"
  when: ep
  local_compute_fn:
    _target_: hyper_parallel.models.qwen3_moe.adapter.distributed.expert_parallel.qwen3moe_ep_compute_fn
    use_grouped_gemm: true
```

**目标**：

```yaml
- match: "*.mlp"
  when: ep
  use_grouped_gemm: true        # 运行期选项留在 YAML
```

`_target_` 不再必需。codegen 在 plan 阶段检查被 `match` 命中的模块结构，决定用哪个策略变体。

### 5.2 结构识别器

```
hyper_parallel/distributed/expert_parallel/
    routing.py       # 现有：router 实现（结构函数）
    structure.py     # 新增：router / archetype 的结构识别
    recipes.py       # 现有：archetype（改名为结构名，见 §5.3）
```

识别规则（示意）：

```python
def detect_router_kind(module) -> str:
    ok(module, "gate")                     # 结构前置条件
    gate = module.gate
    if has_attr(gate, "n_group", "topk_group", "routed_scaling_factor"):
        return "sigmoid_group"
    if is_router_module(gate):             # 带 forward、返回 (logits, weights, indices)
        return "topk_router_module"
    if isinstance(gate, nn.Linear):
        return "softmax_topk"
    raise UnsupportedModuleStructure(...)  # 见 §5.5
```

### 5.3 archetype 去掉模型名

| 现在 | 改为 |
|---|---|
| `routed_only_softmax_topk` | `softmax_topk_router` |
| `qwen3moe_topk_router` / `mixtral_topk_router` | `topk_router_module`（mixtral 的 jitter 作为选项） |
| `qwen2moe_shared_expert_gate` | `topk_router_shared_expert_gate` |
| `deepseekv3_sigmoid_group_shared` | `sigmoid_group_router_shared` |

### 5.4 `generated/` 作为匹配结果的载体（原则 2 的扩展）

**关键：匹配发生在生成期，结果落到 `generated/`；运行期读产物，不再需要 adapter 代码。**

```
生成期：HF 源码 + YAML  →  结构识别  →  generated/
                                        ├─ modeling_X.py       （内联并行的模型）
                                        └─ codegen_meta.json   （匹配结果）
运行期：读 generated/ 里的绑定 → 对通用层实现做绑定
```

生成期的输入是 HF 源码 + YAML（不是产物自己），**因此不构成循环依赖**。

`codegen_meta.json` 的内容要从「指向已提交 adapter 代码」改成「自含的结构描述」：

```json
// 现在
{"factory": "hyper_parallel.models.qwen3_moe.adapter.distributed.expert_parallel.qwen3moe_ep_compute_fn"}

// 目标
{"archetype": "ep_topk_router", "router": "topk_router_module",
 "shared_experts": false, "module_type": "Qwen3MoeSparseMoeBlock", "fqns": ["model.layers.3.mlp"]}
```

### 5.5 识别失败必须硬失败

识别不出结构时**抛错**，不允许猜测、不允许静默回退：

```
UnsupportedModuleStructure: *.mlp matched module with attrs
  ['gate', 'experts', 'expert_gate'] — no EP archetype covers this structure.
  Declare an explicit local_compute_fn._target_ to opt out of structure detection.
```

理由：用错误的结构去并行 ⇒ 静默的数值错误，比报错危险得多。

---

## 6. 展开规则（render spec 的核心）

### 6.1 边界判定规则

> **展开「策略编排」，不展开「框架原语」和「硬件内核」。**

读者的诉求是「打开产物就知道 EP/TP/CP 怎么开的」——那需要看到**编排逻辑**，
但不需要看到 collective 的实现或硬件内核的封装。

| 对象 | 性质 | 展开？ |
|---|---|---|
| `run_qwen3_moe_flash_attention` | **编排**：mask 约定转换 + `sparse_mode` 选择 | ✅ **应该展开**（现在只 import，是错的） |
| `_get_compressed_causal_mask` | 编排（含缓存） | ✅ |
| `_prepare_ep_dispatch` | 编排：算 counts、重排索引 | ⚠️ 边界项，看行长定 |
| `ep_all_to_all` | **原语**：collective 封装 | ❌ |
| `self.experts(...)` → `GroupedExperts.forward` | **模块实现**（数百行） | ❌ |
| `torch_npu.npu_fusion_attention` | **硬件内核** | ❌ |

### 6.2 attention 链能展开到什么程度

`run_qwen3_moe_flash_attention` 的完整依赖闭包只有 3 项，全部可展开：

| 内容 | 行数 | 进产物？ |
|---|---|---|
| `GQAAttention.forward` 里 `self.attention_interface(...)` 调用点 | 1 | ✅ 已在 `QWEN3_GQA_ATTENTION_CLASS` 模板里 |
| `run_qwen3_moe_flash_attention` 函数体 | 59 | ✅ |
| `_get_compressed_causal_mask` | 14 | ✅（闭包依赖） |
| `_COMPRESSED_CAUSAL_MASK_SIZE` / `_COMPRESSED_CAUSAL_MASKS` | 2 | ✅（同上） |
| `import torch_npu` | 1 | ✅ 已在函数体内 |
| **`torch_npu.npu_fusion_attention(...)`** | 1 调用 | ❌ **硬件内核，停在这里** |

**合计约 75 行进产物，展开到「只剩一个硬件内核调用」。**

### 6.3 当前 `render_spec.py` 在这点上是错的

`models/qwen3_moe/adapter/render_spec.py` 对它是：

```python
ImportPatch(
    "hyper_parallel.models.qwen3_moe.adapter.attention",
    ("run_qwen3_moe_flash_attention",),
),
```

**导入而不是展开**，所以产物里它是黑盒——这正是读者不满意的地方。改动很小（见 §6.5）。

### 6.4 用 `inspect.getsource`，不要手抄模板

现状是手写模板常量（`QWEN3_GQA_ATTENTION_CLASS` 是手抄的 110 行）。手抄的代价是**源码漂移**——
上一轮 code review 抓的「最大 Duplicated Code」就是这个。

**更好的做法**：codegen 期用 `inspect.getsource()` 直接取真函数源码。
仓库里已有先例，不是新引入的技术风险：

```
auto_parallel/sapp_nd/memory_estimation/_hook_manager.py:79    source = inspect.getsource(fun)
auto_parallel/sapp_nd/memory_estimation/_func_tracer.py:177    textwrap.dedent(inspect.getsource(fun))
```

收益：**零漂移**（真源码是唯一来源）、**零模板行数**、**通用**（任何函数都能展开）。

### 6.5 展开目标应由框架推导，不由模型文件声明

如果展开目标还要手写清单（例如在 `render_spec.py` 里写 `expand=(...)`），
那只是把 `adapter/render_spec.py` 换了个内容，没去掉它——**与 §0 结论 3 矛盾**。

正确设计：**从策略入口递归内联，在框架层声明一次的 primitives 处停。**

```python
# hyper_parallel/codegen/inline/expansion.py —— 框架层，全局一份
_PRIMITIVE_PREFIXES = (
    "hyper_parallel.distributed.expert_parallel.experts",   # ep_all_to_all 等集体通信原语
    "hyper_parallel.components.functional",                  # NPU 内核封装
    "hyper_parallel.platform",                               # 平台层
    # torch / torch_npu 也是自然停点
)
```

```
从产物里的编排入口出发（如 GQAAttention.forward）
  → 遇到 self.attention_interface → run_qwen3_moe_flash_attention → 展开
  → 它的闭包 _get_compressed_causal_mask → 展开
  → 遇到 torch_npu.npu_fusion_attention → 在 primitives 里 → 停
  → 遇到 ep_all_to_all → 在 primitives 里 → 停
```

这样每个模型的产物都自动带全部编排逻辑，**零声明**；模型间差异只体现在「选中了哪个策略实现」，
而那是结构识别的事（§5.2）。

**过渡期**：若递归展开暂时做不到，可先把 `expand=` 放在 `render_spec.py`，
但必须记进「待消除」清单，不得当成终态。

---

## 7. `_target_` 的两种处理方式对比

### 方案 A：保留为显式逃生口（推荐）

- YAML 默认不写 `_target_`，走结构识别
- 结构不被识别时可显式写 `_target_` 指向自定义工厂

**优点**：达成目标；迁移可**逐模型渐进**；未识别结构有合法前进路径
**缺点**：两条契约路径，测试需同时覆盖；用 `_target_` 时仍需 render spec

### 方案 B：彻底移除 `_target_`

- YAML 只能声明 `match` / `when` / 运行期选项；识别不出即报错

**优点**：契约单一，不可能出现「某个模型偷偷走特殊通道」
**缺点**：迁移是**全有或全无**；遇到真·怪结构唯一出路是改框架代码
（把「配置绕过」的风险换成「无法前进」）

### 对比

| 维度 | A | B |
|---|---|---|
| 达成「YAML 即足够」 | 结构已识别时 ✅ | 全部 ✅ |
| 迁移方式 | 逐模型渐进 | 一次性 |
| 未识别结构的出路 | YAML 显式 `_target_` | 改框架代码 |
| 契约路径数 | 2 | 1 |
| 静默走特殊路径的风险 | 存在（可用 preflight 断言 `_target_` 为空来收敛） | 不存在 |

**推荐 A**，并把 `_target_` 定位为**过渡期逃生口**——用
「`plan_overrides` 中 `_target_` 出现次数」作为可度量的收敛指标，目标为 0。

---

## 8. 落地步骤

| 阶段 | 内容 | 验证 |
|---|---|---|
| S0 | ✅ **已完成**：`adapter/inline.py` → `render_spec.py`（命名消歧） | `pytest tests/codegen` 105 passed |
| S1 | 新增 `structure.py` 结构识别器（router kind + archetype 选择），不改现有调用方 | UT：各结构返回正确变体；未知结构抛错 |
| S2 | `recipes.py` / `routing.py` 的 name-keyed 项改为结构 key，旧 key 保留为薄别名 | 现有测试全绿 + 新 key 的等价性测试 |
| S3 | `manager` 在 `_fill_plan_fields` 阶段：`_target_` 缺失时由结构识别补齐，写入 meta | **Qwen3-MoE / DeepSeek-V3 去掉 `_target_` 后产物逐字节一致** |
| S4 | 新增 `expansion.py`：递归展开 + primitives 停止表；`render_spec.py` 的展开条目改由推导 | 产物含完整编排体；`_target_` 与 `expand` 计数下降 |
| S5 | 接线/删除 §4.5 的 5 个死 provider | `codegen check` |

**S3 的验收是关键**：产物逐字节一致，才能证明「结构推导」与「人工声明」等价。

---

## 9. 风险与不变量

### 9.1 不变量

1. **识别失败必须硬失败**，绝不猜测（§5.5）——安全底线。
2. **YAML 仍是唯一配置入口**（原则 1）；结构识别是补全 YAML 未写明的实现选择，
   不是新增第二个配置源。
3. **`plan_overrides` 仍是 patch 配置字段**（原则 5），字段名与位置不变。
4. **codegen 仍不引入 mesh / fsdp2**（原则 4）。
5. **展开不越过 primitives 边界**（§6.1）——否则产物会退化成整个框架的复制。

### 9.2 风险

| # | 风险 | 应对 |
|---|---|---|
| 1 | 结构识别是启发式，可能误判 | 规则必须保守；不确定即抛错；S3 用逐字节对比守住等价性 |
| 2 | 结构变体爆炸 | 先只覆盖在用的 5 个 archetype，不做投机式泛化 |
| 3 | 递归展开失控（产物膨胀） | primitives 停止表是硬边界；产物行数作为可度量指标 |
| 4 | 两条路径（结构识别 + `_target_`）长期共存 | 以 `_target_` 计数为收敛指标（§7） |
| 5 | 识别器成为新的「必须改框架」点 | 这是 per-structure 而非 per-model，是本质改善 |

---

## 10. 与设计原则的逐条对照

| # | 原则 | 状态 |
|---|---|---|
| 1 | YAML 统一入口，`codegen` 字段决定是否生成 | ✅ 保留不变 |
| 2 | codegen 生成 meta 记录替换与 import 来源 | ✅ 保留，扩展为记录结构变体（§5.4） |
| 3 | `is_hf_model` 三态 `hf`/`custom`/`gen` | ✅ 已实现，本方案不改 |
| 4 | codegen 不含 mesh/fsdp2，但可跳过 sharding_plan apply | ✅ 已实现，本方案不改 |
| 5 | patch 配置在 YAML `plan_overrides` | ✅ 字段保留；`_target_` 由必填变可选（§7） |

---

## 11. 与当前重构（`codegen_restructure` 分支）的关系

| commit | 内容 | 关系 |
|---|---|---|
| 1 | 渲染管道成为唯一产物路径 | 保留；错误信息从「缺 render spec」演进为「结构未识别」（S4） |
| 2 | 删 generic literal 发射器 | 纯删除，无冲突 |
| 3 | 删 runtime boundary/rewrap 一族 | 纯删除，无冲突 |
| — | ~~抽 pattern 常量~~ | **被本方案取代**：不是抽字符串常量，而是结构识别 + 递归展开 |

**顺序**：commits 1–3 与本方案 S1–S2 不冲突，可并行；S3 依赖 S1–S2 稳定，
且必须做逐字节等价验证。

---

## 12. 待决策

1. **`_target_` 取方案 A 还是 B**（§7）——建议 A，并把「`_target_` 计数为 0」作为收敛指标。
2. **`structure.py` 落点**：EP 共享层（本文建议）还是 planner 层（与 `_mark_hf_native_moe` 同处）。
3. **`run_qwen3_moe_flash_attention` 的 left-up causal sparse mode 是必需还是性能选择**（§4.6）
   ——需要找原作者的判断依据。
4. **S1 启动时机**：建议等 commits 1–3 与 129 验证完成后再启动（逐字节等价的基准需要先有可跑通的产物）。
5. **§4.5 的 5 个死 provider**：接线还是删除。
