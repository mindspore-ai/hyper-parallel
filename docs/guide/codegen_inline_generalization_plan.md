# Inline Codegen 泛化改进方案

> **目标**：将当前 inline codegen pipeline 从 Qwen3-MoE 硬编码改造为 adapter 驱动的通用架构，使新增模型适配时无需修改 codegen 核心代码。

> **2026-09-16 进度**：Phase 1 已实现并完成本地回归；129 NPU 验证因 SSH 超时待执行。
> 实现差异与验证记录见 [整改状态](codegen_inline_generalization_status.md)。

---

## 1. 背景与问题

### 1.1 现状

inline codegen pipeline 在 `hyper_parallel/codegen/inline/` 下实现了两阶段源码生成：

1. **Replacement Pass** — 将 HF 原始类构造函数替换为 fused 类（如 `Qwen3MoeAttention` → `GQAAttention`）
2. **Strategy Pass** — 在 forward 方法中内联 TP/CP/EP 通信逻辑

框架骨架（`ir.py` / `patch_engine.py` / `replacement_pass.py` / `yaml_rules.py`）是通用的，但以下 5 个文件包含 Qwen3-MoE 硬编码：

| 文件 | 硬编码内容 | 硬编码类型 |
|------|-----------|-----------|
| `specs.py` | 6 个 Qwen3 target path key | 注册表 |
| `templates.py` | 3 个 Qwen3 专用字符串模板（GQA 类、TP 类、EP forward） | 模板 |
| `strategy_pass.py` | `if spec.kind == "qwen3_moe_ep_routed_forward"` + `class_name="Qwen3MoeSparseMoeBlock"` | if 分支 + 类名 |
| `pipeline.py` | `SUPPORTED_STRATEGY_KINDS = {"qwen3_moe_cp_attention", "qwen3_moe_ep_routed_forward"}` | 白名单 |
| `meta_plan.py` | Qwen3 fused_qkv 参数名还原 | 专用逻辑 |

### 1.2 问题

新增一个模型（如 Llama、Qwen2、DeepSeek-V3）时，需要修改 codegen 核心的 5 个文件，违反了项目已有的"adapter 自注册、核心零修改"原则。

### 1.3 现有基础

项目已有成熟的 model adapter 注册机制：

```
models/adapter_spec.py          → ModelAdapterSpec dataclass（声明 provider 字段）
models/registry.py              → 目录扫描发现 + lazy import 自注册
models/<family>/adapter/
    registration.py             → 自注册 ModelAdapterSpec
    replacements.py             → 运行时模块替换
    attention.py                → attention 计算
    distributed/
        context_parallel.py     → CP wrapper
        expert_parallel.py      → EP compute
```

`ModelAdapterSpec` 已有 `replacements` / `attention` / `context_parallel` / `expert_parallel` 等 lazy provider 字段，但 **没有 inline codegen 的 provider**。inline codegen 完全绕过了这套机制。

---

## 2. 改进方案（三层递进）

### 2.1 第一层：Adapter 声明 inline 能力（消除注册表硬编码）

**目标**：将 `specs.py` 中的 Qwen3 硬编码注册表搬迁到 model adapter，codegen 核心通过 `get_model_adapter()` 动态发现。

#### 2.1.1 扩展 ModelAdapterSpec

```python
# models/adapter_spec.py — 新增字段
@dataclass(frozen=True)
class ModelAdapterSpec:
    architecture: str
    model_type: str
    replacements: Optional[Callable[..., Any]] = None
    attention: Optional[Callable[..., Any]] = None
    checkpoint: Optional[Callable[..., Any]] = None
    context_parallel: Optional[Callable[..., Any]] = None
    expert_parallel: Optional[Callable[..., Any]] = None
    sharding_rules: Optional[Callable[..., Any]] = None
    loss: Optional[Callable[..., Any]] = None
    inline_codegen: Optional[Callable[..., Any]] = None  # NEW
```

#### 2.1.2 新增 InlineSpecBundle 数据结构

```python
# codegen/inline/spec_bundle.py — 新文件
@dataclass(frozen=True)
class InlineSpecBundle:
    """One model family's inline codegen declarations."""
    replacement_specs: dict[str, ReplacementSpec]
    strategy_specs: dict[str, StrategySpec]
    meta_normalizers: list[MetaNormalizer]

@dataclass(frozen=True)
class MetaNormalizer:
    """Declarative meta plan correction for one replacement target."""
    target: str
    param_renames: dict[str, tuple[str, ...]]
```

#### 2.1.3 Qwen3-MoE adapter 声明 inline provider

```python
# models/qwen3_moe/adapter/registration.py — 新增 provider
def _load_inline():
    from hyper_parallel.models.qwen3_moe.adapter import inline  # pylint: disable=C0415
    return inline

QWEN3_MOE_ADAPTER_SPEC = ModelAdapterSpec(
    ...
    inline_codegen=_load_inline,  # NEW
)
```

```python
# models/qwen3_moe/adapter/inline.py — 新文件，搬迁现有 specs.py 内容
from hyper_parallel.codegen.inline.spec_bundle import (
    InlineSpecBundle, MetaNormalizer, ReplacementSpec, StrategySpec, ...
)

REPLACEMENT_SPECS = {
    "...replace_qwen3_moe_rms_norm": ReplacementSpec(...),
    "...replace_qwen3_moe_flash_attention": ReplacementSpec(...),
    "...replace_qwen3_moe_grouped_experts": ReplacementSpec(...),
}
STRATEGY_SPECS = {
    "...qwen3moe_ep_compute_fn": StrategySpec(...),
    "...qwen3_moe_flash_attention_cp_wrapper": StrategySpec(...),
}
META_NORMALIZERS = [
    MetaNormalizer(
        target="...replace_qwen3_moe_flash_attention",
        param_renames={
            "linear_qkv.weight": ("q_proj.weight", "k_proj.weight", "v_proj.weight"),
            "linear_qkv.bias": ("q_proj.bias", "k_proj.bias", "v_proj.bias"),
        },
    ),
]

def get_inline_spec_bundle() -> InlineSpecBundle:
    return InlineSpecBundle(
        replacement_specs=REPLACEMENT_SPECS,
        strategy_specs=STRATEGY_SPECS,
        meta_normalizers=META_NORMALIZERS,
    )
```

#### 2.1.4 specs.py 变为 adapter 发现层

```python
# codegen/inline/specs.py — 从硬编码注册表变为动态发现
def replacement_spec(target: str, model_type: str | None = None) -> ReplacementSpec | None:
    bundle = _get_bundle(model_type)
    if bundle and (spec := bundle.replacement_specs.get(target)):
        return spec
    return None  # 或回退到内置 specs

def strategy_spec(target: str, model_type: str | None = None) -> StrategySpec | None:
    bundle = _get_bundle(model_type)
    if bundle and (spec := bundle.strategy_specs.get(target)):
        return spec
    return None

def _get_bundle(model_type: str | None) -> InlineSpecBundle | None:
    if model_type is None:
        return None
    adapter = get_model_adapter(model_type)
    if adapter and adapter.inline_codegen:
        return adapter.inline_codegen().get_inline_spec_bundle()
    return None
```

**效果**：新增模型只需在 `models/xxx/adapter/inline.py` 声明 specs，codegen 核心**零修改**。

---

### 2.2 第二层：StrategySpec 数据驱动化（消除 if 分支和类名硬编码）

**目标**：让 `StrategySpec` 自己携带目标类名和方法名，`strategy_pass.py` 和 `pipeline.py` 变为纯数据驱动。

#### 2.2.1 增强 StrategySpec

```python
# codegen/inline/spec_bundle.py
@dataclass(frozen=True)
class StrategySpec:
    kind: str
    target_class: str               # NEW: 要 patch 的类名（如 "Qwen3MoeSparseMoeBlock"）
    method: str = "forward"         # NEW: 要替换的方法名
    body_template: str = ""         # NEW: forward body 模板字符串
    imports: tuple[ImportPatch, ...] = ()
```

#### 2.2.2 strategy_pass.py 变为纯数据驱动

```python
# codegen/inline/strategy_pass.py — 无 if 分支
def build_strategy_patches(rules, model_type=None):
    patch_set = InlinePatchSet()
    emitted: set[tuple[str, str]] = set()
    for rule in rules:
        target = rule.local_compute_target or rule.inner_wrapper_target
        spec = strategy_spec(target, model_type)
        if spec is None:
            continue
        key = (spec.target_class, spec.kind)
        if key in emitted:
            continue
        emitted.add(key)
        patch_set.imports.extend(spec.imports)
        patch_set.forward_extracts.append(
            ForwardExtractPatch(
                class_name=spec.target_class,   # 从 spec 读，不再硬编码
                method_name=spec.method,        # 从 spec 读
                body=spec.body_template,        # 从 spec 读
            )
        )
    return patch_set
```

#### 2.2.3 pipeline.py 白名单检查变为动态

```python
# codegen/inline/pipeline.py
def _can_inline(rules, model_type=None) -> bool:
    for rule in rules:
        if rule.replace_target is not None and replacement_spec(rule.replace_target, model_type) is None:
            return False
        target = rule.local_compute_target or rule.inner_wrapper_target
        if target is None:
            continue
        if strategy_spec(target, model_type) is None:
            return False
    return True  # 不再需要 SUPPORTED_STRATEGY_KINDS 白名单
```

#### 2.2.4 meta_plan.py 变为数据驱动

```python
# codegen/inline/meta_plan.py — 从 Qwen3 专用变为通用
def normalize_inline_meta(meta, rules, model_type=None):
    bundle = _get_bundle(model_type)
    if bundle is None:
        return
    for normalizer in bundle.meta_normalizers:
        fqns = _replacement_fqns(rules, normalizer.target)
        if not fqns:
            continue
        _rewrite_param_plan(meta.param_plan, fqns, normalizer.param_renames)
        meta.frozen_sharded_params = _rewrite_frozen_param_names(
            meta.frozen_sharded_params, fqns, normalizer.param_renames,
        )
```

**效果**：`strategy_pass.py` / `pipeline.py` / `meta_plan.py` 完全通用，新模型不需要改这三个文件。

---

### 2.3 第三层：共享 pattern 模板（减少模板重复）

**目标**：将 `templates.py` 中 80% 通用的 GQA attention 逻辑提取为参数化 pattern，model adapter 只提供变量值。

#### 2.3.1 提取共享 pattern

```python
# codegen/inline/patterns.py — 新文件，共享 pattern 模板
GQA_ATTENTION_PATTERN = '''
class {class_name}(nn.Module):
    """Generated GQA attention with visible TP/CP communication."""
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.sliding_window = getattr(config, "sliding_window", None)

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)
        {qk_norm_init}
        self.attention_interface = {attention_interface}

    def forward(self, hidden_states, position_embeddings=None, attention_mask=None, ...):
        ps = get_parallel_state()
        if ps.tp_enabled:
            hidden_states = ps.tp.all_gather(hidden_states, dim=1)
        # ... QKV projection, rotary, CP allgather ...
        attn_output = self.attention_interface(self, query_states, key_states, value_states, ...)
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        if ps.tp_enabled:
            attn_output = ps.tp.reduce_scatter(attn_output, dim=1)
        return attn_output, attn_weights
'''

EP_ROUTED_FORWARD_PATTERN = '''
ps = get_parallel_state()
if not ps.ep_enabled:
    return self._forward_impl(hidden_states)
...
topk_indices, topk_weights = MOE_ROUTER_ADAPTERS["{router_key}"](self, hidden_states)
...
'''
```

#### 2.3.2 Model adapter 只提供变量值

```python
# models/qwen3_moe/adapter/inline.py
from hyper_parallel.codegen.inline.patterns import GQA_ATTENTION_PATTERN, EP_ROUTED_FORWARD_PATTERN

QWEN3_GQA_VARS = {
    "class_name": "GQAAttention",
    "qk_norm_init": (
        "self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)\n"
        "        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)"
    ),
    "attention_interface": "run_qwen3_moe_flash_attention",
}

QWEN3_EP_VARS = {
    "router_key": "qwen3moe",
}
```

#### 2.3.3 不同模型的复用情况

| 模型 | GQA pattern | EP pattern | 自定义模板 |
|------|------------|------------|-----------|
| Qwen3-MoE | ✅ 共享 | ✅ 共享 | 无 |
| Qwen2-MoE | ✅ 共享 | ✅ 共享 | 无 |
| Llama | ✅ 共享（`qk_norm_init=""`） | N/A | 无 |
| DeepSeek-V3 | ❌ MLA 结构不同 | ✅ 共享 | `MLA_ATTENTION_PATTERN` |

**效果**：
- GQA 类模型（Qwen3/Qwen2/Llama）共享一个 pattern，只需 3-5 行变量声明
- 结构独特的模型（DeepSeek MLA）自带专用模板
- `templates.py` 中的 Qwen3 专用常量被移除或变为 pattern 引用

---

## 3. 实施步骤

### Phase 1：第一层 + 第二层（一起做）

> 这两层必须同步实施，因为 `StrategySpec` 数据结构变更需要和 adapter 搬迁一致。

| 步骤 | 文件 | 改动 |
|------|------|------|
| 1 | `codegen/inline/spec_bundle.py` | 新建 `InlineSpecBundle` / `MetaNormalizer` dataclass；增强 `StrategySpec` 加 `target_class` / `method` / `body_template` |
| 2 | `models/adapter_spec.py` | `ModelAdapterSpec` 新增 `inline_codegen` 字段 |
| 3 | `models/qwen3_moe/adapter/inline.py` | 新建，搬迁 `specs.py` 的 `REPLACEMENT_SPECS` / `STRATEGY_SPECS` + `meta_plan.py` 的 Qwen3 常量 |
| 4 | `models/qwen3_moe/adapter/registration.py` | 新增 `_load_inline` provider + 注册 |
| 5 | `codegen/inline/specs.py` | `replacement_spec()` / `strategy_spec()` 加 `model_type` 参数，通过 adapter 动态发现 |
| 6 | `codegen/inline/strategy_pass.py` | 删除 if 分支，改为从 `spec.target_class` / `spec.body_template` 读 |
| 7 | `codegen/inline/pipeline.py` | 删除 `SUPPORTED_STRATEGY_KINDS`，`_can_inline` 改为动态检查 |
| 8 | `codegen/inline/meta_plan.py` | 删除 Qwen3 常量，改为遍历 `bundle.meta_normalizers` |
| 9 | `codegen/inline/templates.py` | 暂时保留，`inline.py` 引用其中的模板字符串 |
| 10 | `codegen/emit/modeling.py` | 调用 `try_render_inline_modeling` 时传入 `model_type` |
| 11 | 验证 | 129 服务器重新生成 + 2221 拓扑 loss 验证 |

### Phase 2：第三层（添加第二个模型时做）

| 步骤 | 文件 | 改动 |
|------|------|------|
| 1 | `codegen/inline/patterns.py` | 新建，从 `templates.py` 提取参数化 pattern |
| 2 | `models/qwen3_moe/adapter/inline.py` | 模板引用改为 `patterns.GQA_ATTENTION_PATTERN` + 变量 |
| 3 | `codegen/inline/templates.py` | 删除或保留为 pattern 的引用 |
| 4 | 验证 | 129 服务器重新生成 + 2221 拓扑 loss 验证 |

---

## 4. 新增模型适配成本对比

| 步骤 | 当前 | 泛化后 |
|------|------|--------|
| 注册 inline specs | 改 `codegen/inline/specs.py` | 在 `models/xxx/adapter/inline.py` 声明 |
| 注册 strategy | 改 `specs.py` + `strategy_pass.py` + `pipeline.py` | 同上，不改 codegen 核心 |
| Meta 修正 | 改 `meta_plan.py` | 在 `inline.py` 声明 `MetaNormalizer` |
| 模板 | 改 `templates.py` | 有现成 pattern：5 行变量；否则写一个模板字符串 |
| **codegen 核心修改** | **5 个文件** | **0 个文件** |

---

## 5. 兼容性与风险

### 5.1 向后兼容

- `model_type=None` 时回退到内置 specs（即当前 Qwen3 硬编码内容），保证已有 YAML 不受影响
- `HYPER_CODEGEN_INLINE_PATCH=1` 环境变量门控不变
- 生成的 modeling 文件内容不变（只是数据来源从硬编码变为 adapter 声明）

### 5.2 风险点

| 风险 | 缓解 |
|------|------|
| adapter 未注册 inline_codegen 时行为变化 | `model_type=None` 回退路径保证现有行为不变 |
| StrategySpec 字段变更导致序列化不兼容 | StrategySpec 不参与序列化，只在内存传递 |
| Phase 1 和 Phase 2 分离导致中间状态 templates.py 冗余 | Phase 1 保留 templates.py 不动，Phase 2 再清理 |
| 129 验证回归 | 每个 Phase 完成后在 129 重新生成 + 4 拓扑 loss 验证 |

### 5.3 不在范围内

- `runtime.py:1077` 的 `_EXTERNAL_STATE_INLINE_CLASSES = frozenset({"GQAAttention", "Qwen3MoeSparseMoeBlock"})` 硬编码 — 建议改为从 generated module 自省（有 `get_parallel_state` 函数的类即需注入），但这是 runtime 层改动，不在本方案范围内
- `emit/modeling.py` 和 `emit/parallel.py` 中的 Qwen3 注释示例 — 通用逻辑的文档示例，不影响功能

---

## 6. 文件变更清单

### Phase 1

| 操作 | 文件 |
|------|------|
| 新建 | `codegen/inline/spec_bundle.py` |
| 新建 | `models/qwen3_moe/adapter/inline.py` |
| 修改 | `models/adapter_spec.py`（+1 字段） |
| 修改 | `models/qwen3_moe/adapter/registration.py`（+1 provider） |
| 修改 | `codegen/inline/specs.py`（加 model_type 参数） |
| 修改 | `codegen/inline/strategy_pass.py`（删 if 分支） |
| 修改 | `codegen/inline/pipeline.py`（删白名单） |
| 修改 | `codegen/inline/meta_plan.py`（数据驱动） |
| 修改 | `codegen/emit/modeling.py`（传 model_type） |
| 不变 | `codegen/inline/ir.py` / `patch_engine.py` / `replacement_pass.py` / `yaml_rules.py` / `templates.py` |

### Phase 2

| 操作 | 文件 |
|------|------|
| 新建 | `codegen/inline/patterns.py` |
| 修改 | `models/qwen3_moe/adapter/inline.py`（引用 patterns） |
| 修改或删除 | `codegen/inline/templates.py` |
