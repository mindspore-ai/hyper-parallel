# Hyper-Parallel HF 兼容层详细设计

> 参考实现：[AutoModel `_transformers/`](../../../auto_model/Automodel/nemo_automodel/_transformers/)
> 上下文设计：[dual_mode_dtensor_parallel_strategy.md](../dual_mode_dtensor_parallel_strategy.md)

---

## 1. 模块职责

提供与 HuggingFace `AutoModel` **完全一致**的 API，用户可以用 `HyperAutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-4B")` 无感启动分布式训练。

### 核心文件

| 文件 | 职责 |
|------|------|
| `_transformers/auto_model.py` | `HyperAutoModel*` 类族 + `from_pretrained`/`from_config` 入口 |
| `_transformers/registry.py` | `MODEL_ARCH_MAPPING` 注册中心 + 懒加载 |
| `_transformers/model_init.py` | HF config 获取 + 模型实例化 + 自定义/HF 路径分发 |
| `_transformers/infrastructure.py` | `instantiate_infrastructure` + `apply_model_infrastructure` 两阶段 |
| `hyper_models/components/distributed/config.py` | `DistributedSetup` — 分布式拓扑容器 → 06_distributed_infrastructure.md §3 |
| `hyper_models/components/distributed/mesh.py` | `MeshContext` / `ParallelismSizes` — DeviceMesh 构建 → 06_distributed_infrastructure.md §3 |
| `hyper_models/components/distributed/fsdp2.py` | `FSDP2Manager` — DP 维度包裹 → 06_distributed_infrastructure.md §4 |
| `hyper_models/components/distributed/dtensor_utils.py` | `_local_params_context` — 生产模式零拷贝 → 06_distributed_infrastructure.md §5 |
| `hyper_models/components/distributed/sharding_planner.py` | **ShardingPlanner** — hyper_parallel 核心差异化能力（AutoModel 无此层） |
| `hyper_models/components/distributed/sharding_applier.py` | ShardingApplier — 将 ShardingPlan 应用到模型 |
| `hyper_models/components/distributed/sharding_config.py` | ShardingConfig / ShardingPlan / ModuleShardingSpec 数据模型 |
| `hyper_models/components/checkpoint/conversion_mapping.py` | HF checkpoint key ↔ 模型 FQN 双向映射 |
| `hyper_models/components/models/common/state_dict_adapter.py` | `StateDictAdapter` 基类 |
| `hyper_models/components/models/common/hf_checkpointing_mixin.py` | `HFCheckpointingMixin` |

### 涉及删除的旧代码

| 旧代码 | 替代方案 |
|--------|---------|
| `hyper_parallel/models/spec/model_spec.py` — `ModelSpec` dataclass | `MODEL_ARCH_MAPPING` + `HyperAutoModel` |
| `hyper_parallel/models/spec/registry.py` — `register_spec()` | `MODEL_ARCH_MAPPING` 的 `OrderedDict` |
| `hyper_parallel/trainer/utils/discovery.py` — `discover_model_spec()` | `MODEL_ARCH_MAPPING` 懒加载 + 查表 |
| `hyper_parallel/trainer/base.py` — `_build_model()` 方法 | 新 `_build_model()` 编排（含 ShardingPlanner） |
| 各模型 `__init__.py` 中的 `register_spec()` 调用 | `ARCH_OVERRIDES` 注册 + `MODEL_ARCH_MAPPING` 条目 |

---

### 1.1 关键架构决策：并行组件与训练流程解耦

Hyper-Parallel 的并行能力（ShardingPlanner、ShardingApplier、PrecompiledBoundary、FSDP2Manager、MeshContext）是**独立可用的组件**，不依赖 Hyper-Parallel 的训练流程。用户可以选择：

```python
# 方式 A：全量训练流程（from_pretrained 一步到位）
model = HyperAutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-4B",
    distributed_setup=DistributedSetup(tp=4, cp=2))
# → 自动完成 ShardingPlanner 推导 + apply_sharding_plan + 权重加载 + FSDP2

# 方式 B：只使用并行能力，自己管理训练循环
# ★ 注意：MeshContext / FSDP2Config / FSDP2Manager 属 06_distributed_infrastructure 范围，
#   当前尚未实现（待 06 落地）；以下示例展示目标 API 形态。
from hyper_models.components.distributed import (
    MeshContext, ShardingPlanner, apply_sharding_plan, FSDP2Manager
)
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-4B")
mesh = MeshContext.build(strategy_config, parallelism_sizes)

plan = ShardingPlanner().plan(model, mesh.device_mesh, tp_size=4)
apply_sharding_plan(model, plan, mesh.device_mesh)

fsdp2 = FSDP2Manager(config, mesh)
model = fsdp2.parallelize(model)
# 现在 model 已分片，可以用任何训练循环（PyTorch Lightning、HF Trainer 等）
```

**设计原则**：`hyper_models/components/distributed/` 下的所有模块**零依赖**于 `recipes/`、`_transformers/`、`hyper_models/components/models/`。它们只依赖 `torch` + `DTensor` + 自身的数据结构（`ShardingPlan`、`ModuleShardingSpec`）。这是与 AutoModel 的关键架构差异——AutoModel 的 `parallelize_fn` 深度嵌入每个模型实现，Hyper-Parallel 的并行组件是**纯外部注入**，完全取代了模型内嵌的 `parallelize_fn`。

### 1.2 迁移路径：`hyper_models/components/` 与现有 `core/` 代码的关系

01 引入的 `hyper_models/components/` 目录（`hyper_models/components/distributed/`、`hyper_models/components/checkpoint/`、`hyper_models/components/models/` 等）与现有 `core/dtensor`、`core/shard`、`dmodule` 并存并逐步收敛，关系如下（详见总计划附录 A.2）：

| 现有代码 | 01 引入的 `hyper_models/components/` 对应物 | 关系 |
|---------|------------------------------|------|
| `core/dtensor/` — DTensor 工具与 placement 推导 | `hyper_models/components/distributed/dtensor_utils.py`（`_local_params_context` 等） | **封装复用**：`hyper_models/components/` 调用 `core/dtensor` 的底层 DTensor API，对外提供面向 ShardingPlan 的语义入口；`core/dtensor` 保持作为底层库不动。 |
| `core/shard/` — 分片应用器与 param 管线 | `hyper_models/components/distributed/sharding_applier.py` + `sharding_planner.py` + `sharding_config.py` | **复用**：底层算子保留为 DTensor dispatch 后端（被 `DTensor.redistribute()` / `distribute_tensor()` 等内部复用，`core/shard/` 不下线）；旧的高层 shard plan 编排 API 被 `ShardingPlanner`/`ShardingApplier` 替代，用户侧只能通过 `hyper_models/components/distributed/` 入口使用（与 06 §1 一致） |
| `dmodule/` — 模型动态包装与 spec 注册 | `_transformers/registry.py` + `hyper_models/components/models/common/` | **替代**：`MODEL_ARCH_MAPPING` + `HyperAutoModel` 取代 `dmodule` 的 `ModelSpec`/`register_spec` 机制（见 §1 "涉及删除的旧代码"）。迁移期 `dmodule` 旧模型按 §12 流程逐个迁入 `hyper_models/components/models/`，迁完即下线 `dmodule` 对应条目。 |

**迁移原则**：
- `core/dtensor` 作为底层 DTensor 能力保留，`hyper_models/components/distributed/` 在其上构建 ShardingPlan 驱动的声明式分片层。
- `core/shard` 属于**复用层**（底层算子保留为 DTensor dispatch 后端，不下线）；`dmodule` 属于**被替代层**。迁移期允许并存，但**新代码必须只走 `hyper_models/components/` 入口**，禁止新增对 `core/shard` 高层编排 API 与 `dmodule` 的直接调用。
- 具体 phase 划分与验收标准见总计划附录 A.2。

---

## 2. ConfigNode 配置系统：从 YAML 到对象

### 2.1 总体处理流程

Hyper-Parallel 的配置系统按以下顺序处理一份 YAML 训练配置：

```
YAML 文件 (train.yaml)
    │
    ▼
① load_yaml_config(path)
    │  yaml.safe_load() → 原始 dict
    │  ConfigNode(dict)  → 递归包装
    ▼
② ConfigNode.__init__()
    │  对每个 key-value 调用 _wrap(k, v):
    │    • dict   → ConfigNode(v)           # 递归嵌套
    │    • list   → [_wrap("", i) for i]    # 递归列表
    │    • _target_ → _resolve_target(v)     # 解析为 callable ★
    │    • *_fn   → _resolve_target(v)      # 解析为 callable ★
    │    • 其他    → translate_value(v)      # 类型转换
    ▼
③ _resolve_target(dotted_path)
    │  "torch.optim.AdamW" → <class 'torch.optim.adamw.AdamW'>
    │  "path/to/file.py:MyClass" → <class 'MyClass'>
    ▼
④ ConfigNode 就绪（所有 _target_ 和 *_fn 已解析为 callable）
    │
    ▼
⑤ cfg.xxx.instantiate(**runtime_kwargs)
    │  func = self._target_              # 已解析的 callable
    │  config_kwargs = {}                # 收集其他属性
    │  对每个 attr（排除 _target_ 等内部 key）:
    │    • ConfigNode 且有 _target_ → v.instantiate()   # ★ 递归！
    │    • ConfigNode 无 _target_      → v.to_dict()
    │    • 普通值                       → 原值
    │  func(*args, **config_kwargs, **runtime_kwargs)
    ▼
Python 对象（模型、优化器、Dataset...）
```

**关键设计决策**：`_target_` 和 `*_fn` 在 **YAML 加载时**（ConfigNode 构造）立即解析为 callable，而不是延迟到 `instantiate()` 时。这意味着：
- `self._target_` 始终是一个已解析的 class/function/method，不是字符串
- 日志和序列化时通过 `_original_strings` 保留原始字符串

---

### 2.2 Step 1: `load_yaml_config` — YAML 文件 → ConfigNode

```python
# hyper_models/components/config/loader.py

def load_yaml_config(path: str | Path) -> ConfigNode:
    """加载 YAML 文件并包装为 ConfigNode。

    这是整个配置系统的入口。流程极简：
    ① yaml.safe_load() 读取文件 → 原生 Python dict
    ② ConfigNode(dict) → 递归包装，立即解析所有 _target_ 和 *_fn
    """
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return ConfigNode(raw)
```

**示例**：对于以下 YAML 文件：

```yaml
recipe: FinetuneRecipe
model:
  _target_: hyper_models.HyperAutoModelForCausalLM.from_pretrained
  pretrained_model_name_or_path: Qwen/Qwen3.5-0.8B
optimizer:
  _target_: torch.optim.AdamW
  lr: 1.0e-4
  weight_decay: 0.01
```

`yaml.safe_load()` 产出的原始 dict 为：

```python
{
    "recipe": "FinetuneRecipe",
    "model": {
        "_target_": "hyper_models.HyperAutoModelForCausalLM.from_pretrained",
        "pretrained_model_name_or_path": "Qwen/Qwen3.5-0.8B",
    },
    "optimizer": {
        "_target_": "torch.optim.AdamW",
        "lr": 1.0e-4,           # PyYAML safe_load 已按 YAML 1.1 规范解析为 float
        "weight_decay": 0.01,   # 同上：数值标量 → float，不是字符串
    },
}
```

注意：`yaml.safe_load()` 会自动解析数值/布尔标量（`1.0e-4` → float、`0.01` → float、
`42` → int、`true` → bool）。只有无法匹配数值/布尔形式的标量（如 `"bfloat16"`、
`"Qwen/Qwen3.5-0.8B"`）才保持为字符串，随后由 `translate_value()` 做进一步转换（§2.3）。

**Canonical 模块位置（裁决）**：`load_yaml_config` 位于 `hyper_models/components/config/loader.py`；
`ConfigNode` 与 `_resolve_target` 的 canonical 位置为 **`hyper_models/components/config/node.py`**
（loader.py 内部 `from .node import ConfigNode`）。02 §10 等其它文档的模块位置表述
以此为准对齐。

然后 `ConfigNode(raw)` 递归包装每个嵌套 dict。

---

### 2.3 Step 2: `ConfigNode.__init__` + `_wrap` — 立即解析（Eager Resolution）

```python
# hyper_models/components/config/node.py（ConfigNode 与 _resolve_target 的 canonical 位置，见 §2.2 末注）

from copy import deepcopy


class _OrigValueStr(str):
    """String wrapper that preserves the original placeholder for safe display.

    The resolved value is the actual string content; the original placeholder
    (with $ENV_VAR references) is stored for to_yaml_dict() safe output.
    """
    def __new__(cls, resolved: str, original: str):
        instance = super().__new__(cls, resolved)
        instance._orig_value = original
        return instance


class ConfigNode:
    """配置节点——属性式访问 + 延迟实例化。"""

    _target_: Optional[Callable] = None    # ★ 注意：类型是 Callable，不是 str
    raise_on_missing_attr: bool = True     # AutoModel 默认 True

    def __init__(self, d: Optional[dict] = None, /, raise_on_missing_attr: bool = True, **kwargs):
        # 保存原始 dict 的深拷贝（用于 checkpoint 恢复）
        self.__dict__["_raw_config"] = deepcopy(d) if d else {}
        # 保存 _target_ 和 *_fn 的原始字符串（用于日志/序列化）
        self.__dict__["_original_strings"]: dict[str, str] = {}
        # ★ 核心：对每个 key-value 调用 _wrap 进行分类处理
        source = {**(d or {}), **kwargs}
        for k, v in source.items():
            self.__dict__[k] = self._wrap(k, v)
        self.raise_on_missing_attr = raise_on_missing_attr

    def _wrap(self, k: str, v: Any) -> Any:
        """对每个 key-value 进行分类处理——这是 ConfigNode 的核心分发逻辑。"""
        if isinstance(v, dict):
            # ── dict → 递归包装为 ConfigNode ──
            return ConfigNode(v)

        elif isinstance(v, list):
            # ── list → 递归包装每个元素 ──
            return [self._wrap("", item) for item in v]

        elif k.endswith("_fn"):
            # ── *_fn → 解析为 callable（详见 §2.8） ──
            # 例如: collate_fn: "my_package.my_collate"
            if isinstance(v, str):
                self._original_strings[k] = v   # 保存原始字符串
            return _resolve_target(v)           # ★ 立即解析为 callable

        elif k == "_target_":
            # ── _target_ → 解析为 callable ★ ──
            # 例如: _target_: "torch.optim.AdamW"
            if isinstance(v, str):
                self._original_strings[k] = v   # 保存原始字符串
            # 注：若 v 本身已是 callable（非字符串），则不设置 _original_strings，
            # get_as_string("_target_") 将回退到 str() 表示
            return _resolve_target(v)           # ★ 立即解析为 callable
            # 此时 self._target_ 已经是 <class 'torch.optim.adamw.AdamW'>，不是字符串！

        else:
            # ── 普通值 → 类型转换 ──
            if isinstance(v, str) and "$" in v:
                # 含环境变量引用 → 解析 + 翻译
                resolved = resolve_yaml_env_vars(v)
                translated = translate_value(resolved)
                # 保留原始占位符用于安全打印（不泄露环境变量值）
                if isinstance(translated, str) and resolved != v:
                    return _OrigValueStr(translated, v)
                return translated
            return translate_value(v)
```

**关键点**：
- `self._target_` 来自 `_wrap("_target_", v)`，其中 `_resolve_target(v)` 将字符串 `"torch.optim.AdamW"` 解析为 `<class 'torch.optim.adamw.AdamW'>`
- `self._original_strings["_target_"]` 保存原始字符串 `"torch.optim.AdamW"`，用于 `__repr__`、`to_yaml_dict()` 等安全输出
- 所有解析发生在 `__init__` 阶段（eager），而非 `instantiate()` 阶段（lazy）

---

### 2.4 Step 3: `_resolve_target` — 字符串 → Callable

```python
def _resolve_target(dotted_path: str) -> Any:
    """将字符串解析为 Python 对象（class / function / method）。

    支持两种形式：

    ① 文件路径:对象名 → "path/to/module.py:MyClass"
       ── 从 .py 文件动态加载模块，获取指定属性
       ── 用于用户自定义组件

    ② 点分隔导入路径 → "torch.optim.AdamW"
       ── 从最长前缀开始尝试 import，逐级 getattr
       ── 用于标准库和第三方库组件
    """
    if not isinstance(dotted_path, str):
        return dotted_path  # 已经是 callable → 透传

    # ── 形式 ①: path/to/file.py:attr ──
    if ":" in dotted_path and not dotted_path.startswith(("http:", "https:")):
        file_path, attr_name = dotted_path.rsplit(":", 1)
        assert file_path.endswith(".py"), \
            f"_resolve_target file-path form requires .py suffix, got: {file_path}"
        module = load_module_from_file(file_path)
        return _safe_getattr(module, attr_name)

    # ── 形式 ②: dotted.import.path ──
    parts = dotted_path.split(".")
    # 从最长前缀开始尝试导入
    for i in range(len(parts), 0, -1):
        module_path = ".".join(parts[:i])
        attr_chain = parts[i:]

        if not _is_allowed_module(module_path):
            continue

        try:
            module = importlib.import_module(module_path)
        except (ImportError, ModuleNotFoundError):
            continue

        # 逐级 getattr
        obj = module
        for attr in attr_chain:
            obj = _safe_getattr(obj, attr)
        return obj

    raise ImportError(
        f"Cannot resolve target '{dotted_path}': module not found "
        f"or not in allowed prefixes {ALLOWED_IMPORT_PREFIXES}"
    )
```

**解析过程示例**：

```
① _resolve_target("torch.optim.AdamW")
   parts = ["torch", "optim", "AdamW"]
   i=3: module_path="torch.optim.AdamW" → ModuleNotFoundError（不是模块）
   i=2: module_path="torch.optim" → importlib.import_module("torch.optim") ✓
        attr_chain=["AdamW"]
        getattr(torch.optim, "AdamW") → <class 'torch.optim.adamw.AdamW'> ✓

② _resolve_target("hyper_models.HyperAutoModelForCausalLM.from_pretrained")
   parts = ["hyper_parallel", "HyperAutoModelForCausalLM", "from_pretrained"]
   i=1: module_path="hyper_parallel" → importlib ✓
        attr_chain=["HyperAutoModelForCausalLM", "from_pretrained"]
        getattr(hyper_parallel, "HyperAutoModelForCausalLM") → <class ...>
        getattr(class, "from_pretrained") → <bound method ...> ✓

③ _resolve_target("my_custom/callback.py:on_step_end")
   检测到 ":" 且非 http → 形式①
   load_module_from_file("my_custom/callback.py") → module
   getattr(module, "on_step_end") → <function on_step_end> ✓
```

---

### 2.5 `translate_value` — 标量值类型转换

```python
def translate_value(v: Any) -> Any:
    """将 YAML 字符串智能转换为 Python 原生类型。

    YAML 的值默认都是字符串（如 "1.0e-4", "true", "None"），
    此函数用 ast.literal_eval 自动转换。
    """
    if not isinstance(v, str):
        return v

    # Fast-path: 特殊符号
    special_symbols = {"none": None, "None": None, "true": True, "True": True,
                       "false": False, "False": False}
    # 注：YAML 通常输出首字母大写的 "True"/"False"/"None"，小写 key 为防御性保留
    if v in special_symbols:
        return special_symbols[v]

    # 防止评估超长字符串
    if len(v) > 1000:
        return v

    try:
        return ast.literal_eval(v)  # "1.0e-4" → 0.0001, "[1,2]" → [1,2]
    except Exception:
        return v  # 解析失败 → 保留原字符串
```

---

### 2.6 完整示例：一份 YAML 的逐函数解析

以下追踪一个简化训练配置从 YAML 文件到 ConfigNode 的**完整过程**，覆盖每个函数的作用。

#### 输入 YAML

```yaml
recipe: FinetuneRecipe
seed: 42
model:
  _target_: hyper_models.HyperAutoModelForCausalLM.from_pretrained
  pretrained_model_name_or_path: Qwen/Qwen3.5-0.8B
  torch_dtype: bfloat16
optimizer:
  _target_: torch.optim.AdamW
  lr: 1.0e-4
  betas: [0.9, 0.95]
  weight_decay: 0.01
dataset:
  _target_: datasets.load_dataset
  path: HuggingFaceFW/fineweb
  streaming: true
  split: train
  tokenizer:
    _target_: transformers.AutoTokenizer.from_pretrained
    pretrained_model_name_or_path: Qwen/Qwen3.5-0.8B
```

#### Step 1: `load_yaml_config("train.yaml")`

```python
with open("train.yaml") as f:
    raw = yaml.safe_load(f)
# raw = {
#     "recipe": "FinetuneRecipe",
#     "seed": 42,                         # YAML 已自动转为 int
#     "model": {
#         "_target_": "hyper_models.HyperAutoModelForCausalLM.from_pretrained",
#         "pretrained_model_name_or_path": "Qwen/Qwen3.5-0.8B",
#         "torch_dtype": "bfloat16",
#     },
#     "optimizer": {
#         "_target_": "torch.optim.AdamW",
#         "lr": 0.0001,                   # YAML 已自动转为 float（1.0e-4）
#         "betas": [0.9, 0.95],           # YAML 已转 float 列表
#         "weight_decay": 0.01,           # YAML 已自动转为 float
#     },
#     "dataset": {
#         "_target_": "datasets.load_dataset",
#         "path": "HuggingFaceFW/fineweb",
#         "streaming": true,              # YAML bool
#         "split": "train",
#         "tokenizer": {
#             "_target_": "transformers.AutoTokenizer.from_pretrained",
#             "pretrained_model_name_or_path": "Qwen/Qwen3.5-0.8B",
#         },
#     },
# }
return ConfigNode(raw)  # → Step 2
```

#### Step 2: `ConfigNode.__init__(raw)`

ConfigNode 对 `raw` 的每个顶层 key 调用 `_wrap(k, v)`：

```python
# self = ConfigNode()  (正在构造中)

# ① _wrap("recipe", "FinetuneRecipe")
#    v 是 str, k 不是 "_target_"/"*_fn", 不含 "$"
#    → translate_value("FinetuneRecipe")
#    → ast.literal_eval("FinetuneRecipe") 失败 → 返回 "FinetuneRecipe"
#    结果: self.recipe = "FinetuneRecipe"

# ② _wrap("seed", 42)
#    v 是 int → translate_value(42) → 42（非 str 直接返回）
#    结果: self.seed = 42

# ③ _wrap("model", {"_target_": "...", ...})
#    v 是 dict → ConfigNode(v)  # ★ 递归创建子 ConfigNode
#    ── 进入子 ConfigNode.__init__({"pretrained_model_name_or_path": ..., "_target_": ..., ...}) ──
#      a) _wrap("_target_", "hyper_models.HyperAutoModelForCausalLM.from_pretrained")
#         k == "_target_" → _resolve_target("hyper_models.HyperAutoModelForCausalLM.from_pretrained")
#           parts = ["hyper_parallel", "HyperAutoModelForCausalLM", "from_pretrained"]
#           i=1: importlib.import_module("hyper_parallel") ✓
#                getattr(mod, "HyperAutoModelForCausalLM") → <class>
#                getattr(class, "from_pretrained") → <bound method>
#         保存: self._original_strings["_target_"] = "hyper_models.HyperAutoModelForCausalLM.from_pretrained"
#         结果: self._target_ = <bound method HyperAutoModelForCausalLM.from_pretrained>
#
#      b) _wrap("pretrained_model_name_or_path", "Qwen/Qwen3.5-0.8B")
#         translate_value("Qwen/Qwen3.5-0.8B") → "Qwen/Qwen3.5-0.8B"
#         结果: self.pretrained_model_name_or_path = "Qwen/Qwen3.5-0.8B"
#
#      c) _wrap("torch_dtype", "bfloat16")
#         translate_value("bfloat16") → "bfloat16"
#         结果: self.torch_dtype = "bfloat16"
#    ── 子 ConfigNode 构造完成 ──
#    结果: self.model = <ConfigNode {
#        _target_: <bound method ...from_pretrained>,
#        pretrained_model_name_or_path: "Qwen/Qwen3.5-0.8B",
#        torch_dtype: "bfloat16",
#    }>

# ④ _wrap("optimizer", {"_target_": "torch.optim.AdamW", ...})
#    v 是 dict → ConfigNode(v)  # ★ 递归
#    ── 子 ConfigNode.__init__ ──
#      a) _wrap("_target_", "torch.optim.AdamW")
#         _resolve_target("torch.optim.AdamW")
#           i=2: importlib.import_module("torch.optim") ✓
#                getattr(torch.optim, "AdamW") → <class 'torch.optim.adamw.AdamW'>
#         结果: self._target_ = <class 'torch.optim.adamw.AdamW'>
#              ★ self._target_ 不是一个字符串！它是一个 Python class 对象！
#
#      b) _wrap("lr", 0.0001)
#         translate_value(0.0001) → 0.0001（非 str 直接返回；float 由 YAML 解析产生）
#         结果: self.lr = 0.0001
#
#      c) _wrap("betas", [0.9, 0.95])
#         v 是 list → [self._wrap("", 0.9), self._wrap("", 0.95)]
#         → [translate_value(0.9), translate_value(0.95)]
#         → [0.9, 0.95]
#         结果: self.betas = [0.9, 0.95]
#
#      d) _wrap("weight_decay", 0.01)
#         translate_value(0.01) → 0.01（非 str 直接返回）
#         结果: self.weight_decay = 0.01
#    ── 子 ConfigNode 构造完成 ──
#    结果: self.optimizer = <ConfigNode {
#        _target_: <class 'torch.optim.adamw.AdamW'>,
#        lr: 0.0001,
#        betas: [0.9, 0.95],
#        weight_decay: 0.01,
#    }>

# ⑤ _wrap("dataset", {"_target_": "datasets.load_dataset", ..., "tokenizer": {...}})
#    v 是 dict → ConfigNode(v)  # ★ 递归
#    ── 子 ConfigNode.__init__ ──
#      a) _wrap("_target_", "datasets.load_dataset")
#         _resolve_target("datasets.load_dataset")
#           i=1: importlib.import_module("datasets") ✓
#                getattr(datasets, "load_dataset") → <function load_dataset>
#         结果: self._target_ = <function datasets.load_dataset>
#
#      b) _wrap("path", "HuggingFaceFW/fineweb")
#         translate_value → "HuggingFaceFW/fineweb"
#
#      c) _wrap("streaming", True)
#         translate_value(True) → True
#
#      d) _wrap("split", "train")
#         translate_value → "train"
#
#      e) _wrap("tokenizer", {"_target_": "transformers.AutoTokenizer.from_pretrained", ...})
#         v 是 dict → ConfigNode(v)  # ★ 三层嵌套！
#         ── 子子 ConfigNode.__init__ ──
#           · _wrap("_target_", "transformers.AutoTokenizer.from_pretrained")
#             _resolve_target → <bound method AutoTokenizer.from_pretrained>
#             结果: self._target_ = <bound method>
#           · _wrap("pretrained_model_name_or_path", "Qwen/Qwen3.5-0.8B")
#             → "Qwen/Qwen3.5-0.8B"
#         ── 子子 ConfigNode 构造完成 ──
#         结果: self.tokenizer = <ConfigNode {
#             _target_: <bound method AutoTokenizer.from_pretrained>,
#             pretrained_model_name_or_path: "Qwen/Qwen3.5-0.8B",
#         }>
#    ── 子 ConfigNode 构造完成 ──
#    结果: self.dataset = <ConfigNode {
#        _target_: <function datasets.load_dataset>,
#        path: "HuggingFaceFW/fineweb",
#        streaming: True,
#        split: "train",
#        tokenizer: <ConfigNode {_target_: <bound method>, ...}>,
#    }>
```

**构造完成后的内存结构**：

```
cfg (ConfigNode)
├── recipe: "FinetuneRecipe"
├── seed: 42
├── model (ConfigNode)
│   ├── _target_: <bound method HyperAutoModelForCausalLM.from_pretrained>
│   ├── pretrained_model_name_or_path: "Qwen/Qwen3.5-0.8B"
│   └── torch_dtype: "bfloat16"
├── optimizer (ConfigNode)
│   ├── _target_: <class 'torch.optim.adamw.AdamW'>
│   ├── lr: 0.0001
│   ├── betas: [0.9, 0.95]
│   └── weight_decay: 0.01
└── dataset (ConfigNode)
    ├── _target_: <function datasets.load_dataset>
    ├── path: "HuggingFaceFW/fineweb"
    ├── streaming: True
    ├── split: "train"
    └── tokenizer (ConfigNode)
        ├── _target_: <bound method AutoTokenizer.from_pretrained>
        └── pretrained_model_name_or_path: "Qwen/Qwen3.5-0.8B"
```

每个 `_target_` 字段的值**已经是解析好的 callable**，不再是字符串。

---

### 2.7 Step 4: `__getattr__` / `get` / `get_as_string` — 属性访问

```python
class ConfigNode:
    def __getattr__(self, key: str) -> Any:
        """属性式访问：cfg.model → 返回子 ConfigNode。

        dunder 方法（如 __setstate__）必须 raise AttributeError，
        否则 copy.deepcopy 等协议会误判。
        """
        if key.startswith("__") and key.endswith("__"):
            raise AttributeError(key)
        try:
            return self.__dict__[key]
        except KeyError:
            if self.__dict__.get("raise_on_missing_attr", True):
                raise AttributeError(key)   # AutoModel 默认：严格模式
            return None                     # 宽松模式

    def get(self, key: str, default: Any = None) -> Any:
        """点分隔路径访问：cfg.get("dataset.tokenizer") → 子 ConfigNode。"""
        parts = key.split(".")
        current = self
        for p in parts:
            if isinstance(current, ConfigNode):
                if p in current.__dict__:
                    current = current.__dict__[p]
                else:
                    return default
            elif isinstance(current, list):
                try:
                    current = current[int(p)]
                except (ValueError, IndexError):
                    return default
            else:
                return default
        return current

    def get_as_string(self, key: str) -> str:
        """获取 _target_ 或 *_fn 的原始字符串（非解析后的 callable）。

        用于日志和 YAML 序列化——显示 "torch.optim.AdamW" 而非
        "<class 'torch.optim.adamw.AdamW'>"。
        """
        if key in self._original_strings:
            return self._original_strings[key]
        # 当 _target_ 是 callable（直接在 YAML 中传入）时，_original_strings 无此 key，
        # 回退到 __dict__ 的 str 表示（如 "<class 'torch.optim.adamw.AdamW'>"）
        return self._original_strings.get(key, str(self.__dict__.get(key, "")))
```

**使用示例**：
```python
cfg.model.pretrained_model_name_or_path   # → "Qwen/Qwen3.5-0.8B"
cfg.get("dataset.tokenizer")              # → <ConfigNode {_target_: <bound method>, ...}>
cfg.optimizer.get_as_string("_target_")   # → "torch.optim.AdamW"（原始字符串！）
```

---

### 2.8 `*_fn` 机制详解：函数作为构造参数

**为什么需要 `*_fn`？**

有些组件的构造函数接受**函数/回调**作为参数（如 `collate_fn`、`loss_fn`、`reward_fn`）。这些函数需要在 YAML 中通过字符串路径引用，在 ConfigNode 构造时解析为 callable，然后在 `instantiate()` 时作为普通 keyword argument 传入。

**规则**：任何以 `_fn` 结尾的 key，其字符串值会在 `_wrap()` 中通过 `_resolve_target()` **立即解析为 callable**，行为与 `_target_` 完全一致。

**示例**：配置一个 DataLoader 的 collate function

```yaml
dataloader:
  _target_: torchdata.stateful_dataloader.StatefulDataLoader
  batch_size: 1
  collate_fn: hyper_models.components.datasets.utils.default_collater
```

解析过程：

```python
# _wrap("collate_fn", "hyper_models.components.datasets.utils.default_collater")
# k == "collate_fn" → k.endswith("_fn") is True!
# → _resolve_target("hyper_models.components.datasets.utils.default_collater")
#   → importlib.import_module("hyper_models.components.datasets.utils")
#   → getattr(module, "default_collater")
#   → <function default_collater>
# self._original_strings["collate_fn"] = "hyper_models.components.datasets.utils.default_collater"
# 结果: self.collate_fn = <function default_collater>   ★ 直接是函数对象！

# 注意：collate_fn 不是 _target_，不会被 instantiate() 调用。
# 它作为普通 kwarg 传给 DataLoader 的构造函数：
#   DataLoader(dataset=..., batch_size=1, collate_fn=<function default_collater>)
```

**`_fn` vs `_target_` 的区别**：

| | `_target_` | `*_fn` |
|------|----------|--------|
| 解析时机 | `_wrap()` 中立即解析 | `_wrap()` 中立即解析 |
| `instantiate()` 行为 | **调用** `_target_(*args, **kwargs)` | 作为 **kwarg 传入** `_target_` |
| 用途 | "这个 ConfigNode 要实例化什么类" | "构造函数的某个参数是什么函数" |
| 例子 | `_target_: torch.optim.AdamW` | `collate_fn: my_package.my_collate` |

---

### 2.9 Step 5: `instantiate()` — ConfigNode → Python 对象

```python
class ConfigNode:
    def instantiate(self, *args: Any, **kwargs: Any) -> Any:
        """调用 _target_ 创建对象。

        这是 ConfigNode 的最终目标——将配置树转换为 Python 对象树。

        流程：
        ① 获取 _target_ callable（已在 __init__ 中解析好）
        ② 遍历 self 的其他属性，收集为 config_kwargs
        ③ 对每个属性值调用 _instantiate_value（递归实例化嵌套 ConfigNode）
        ④ 合并运行时 kwargs（覆盖 config 中的值）
        ⑤ 调用 _target_(*args, **merged_kwargs)
        """
        if self._target_ is None:
            raise AttributeError("No _target_ found to instantiate")

        func = self._target_   # ★ 已经是 callable，不需要 _resolve_target
        # 但为安全起见，AutoModel 还是调了一次 _resolve_target（幂等操作）

        # ── 收集 config_kwargs ──
        config_kwargs = {}
        for k, v in self.__dict__.items():
            # 跳过内部 key
            if k in ("_target_", "raise_on_missing_attr", "_raw_config", "_original_strings"):
                continue
            # 运行时覆盖 → 跳过（节省递归实例化的开销）
            if k in kwargs:
                continue
            if k.endswith("_fn"):
                # ★ *_fn 已经解析为 callable，直接作为 kwarg 传入
                # 例如: collate_fn=<function default_collater>
                config_kwargs[k] = v
            else:
                # ★ 递归实例化嵌套 ConfigNode
                config_kwargs[k] = self._instantiate_value(v)

        # 解析 config_kwargs 中的环境变量（只解析 config 的，不解析 runtime 的）
        config_kwargs = resolve_yaml_env_vars(config_kwargs)

        # runtime kwargs 覆盖 config kwargs
        config_kwargs.update(kwargs)

        try:
            return func(*args, **config_kwargs)
        except Exception as e:
            # 详细的错误信息：展示签名 + 参数
            import inspect
            import sys
            import pprint
            sig = inspect.signature(func)
            safe_kwargs = _redact(config_kwargs)  # 脱敏
            print(f"Instantiation failed for `{func.__name__}`\n"
                  f"Accepted signature: {sig}\n"
                  f"Positional args: {args}\n"
                  f"Keyword args: {pprint.pformat(safe_kwargs)}\n"
                  f"Exception: {e}", file=sys.stderr)
            raise e

    def instantiate_path(self, dotted_path: str, default: Any = None, *args, **kwargs) -> Any:
        """按路径查找并 instantiate，未找到返回 default。

        用于可选配置段（peft、qat 等）：
            peft_config = cfg.instantiate_path("peft")  # None if not configured
        """
        item = self.get(dotted_path, default)
        if item is default:
            return default
        return item.instantiate(*args, **kwargs)
```

**实例化示例**：继续 §2.6 的配置

```python
# ── ① 实例化 tokenizer ──
# cfg.dataset.tokenizer.instantiate()
#
# func = <bound method AutoTokenizer.from_pretrained>
# config_kwargs = {"pretrained_model_name_or_path": "Qwen/Qwen3.5-0.8B"}
# → AutoTokenizer.from_pretrained(pretrained_model_name_or_path="Qwen/Qwen3.5-0.8B")
# → <PreTrainedTokenizerFast>

# ── ② 实例化 dataset ──
# cfg.dataset.instantiate()   # ★ 不传 tokenizer！
#
# func = <function datasets.load_dataset>
# config_kwargs = {
#     "path": "HuggingFaceFW/fineweb",
#     "streaming": True,
#     "split": "train",
#     "tokenizer": <PreTrainedTokenizerFast>   # ← 嵌套 ConfigNode 递归 instantiate 的结果
# }
# ★ 但 datasets.load_dataset 不接受 tokenizer kwarg——
#   02 §4.2 的 signature 守卫（inspect.signature 检查）会在调用前剔除
#   不在 load_dataset 签名中的 kwargs，因此 tokenizer 不会被注入。
# → load_dataset(path="HuggingFaceFW/fineweb", streaming=True, split="train")
# → <Dataset>
# （tokenizer 由 build_dataloader 内部 _build_tokenizer 单独实例化，
#   供后续 map/tokenize 步骤使用，而不是传给 load_dataset）

# ── ③ 实例化 model ──
# cfg.model.instantiate(distributed_setup=<setup>)
#
# func = <bound method HyperAutoModelForCausalLM.from_pretrained>
# config_kwargs = {
#     "pretrained_model_name_or_path": "Qwen/Qwen3.5-0.8B",
#     "torch_dtype": "bfloat16",
# }
# runtime: kwargs = {"distributed_setup": <setup>}
# merged = {**config_kwargs, **kwargs}
# → HyperAutoModelForCausalLM.from_pretrained(
#       pretrained_model_name_or_path="Qwen/Qwen3.5-0.8B",
#       torch_dtype="bfloat16",
#       distributed_setup=<setup>,
#   )
# → <HyperAutoModelForCausalLM>
```

---

### 2.10 `_instantiate_value` — 递归实例化

```python
class ConfigNode:
    def _instantiate_value(self, v: Any) -> Any:
        """递归处理 config_kwargs 中的每个值。

        核心规则：
        - ConfigNode 且有 _target_ → v.instantiate()  ★ 递归实例化
        - ConfigNode 无 _target_      → v.to_dict()     ★ 展开为普通 dict
        - list                         → 递归处理每个元素
        - 叶子值                       → translate_value(resolve_yaml_env_vars(v))
        """
        if isinstance(v, ConfigNode) and v._target_ is not None:
            # ★ 嵌套的 _target_ ConfigNode → 先实例化它！
            # 例如：optimizer ConfigNode 内的 lr_scheduler ConfigNode
            return v.instantiate()
        elif isinstance(v, ConfigNode):
            # 无 _target_ 的 ConfigNode → 展开为普通 dict
            # 例如：model.backend 只是一个配置分组，不需要实例化
            return resolve_yaml_env_vars(v.to_dict())
        elif isinstance(v, list):
            return [self._instantiate_value(item) for item in v]
        else:
            # 叶子值：解析环境变量 + 类型转换
            return translate_value(resolve_yaml_env_vars(v))
```

**递归实例化示例**：

```yaml
optimizer:
  _target_: torch.optim.AdamW
  lr: 1.0e-4
  # 没有嵌套的 _target_，所有值都是叶子 → 直接传入
```

vs

```yaml
# 如果优化器内部有嵌套的 _target_（示意）：
training:
  _target_: hyper_models.TrainingLoop
  optimizer:
    _target_: torch.optim.AdamW     # ← 嵌套 _target_，会先被实例化
    lr: 1.0e-4
```

```python
# cfg.training.instantiate()
# config_kwargs["optimizer"] = self._instantiate_value(optimizer_Confignode)
#   → optimizer_Confignode 有 _target_
#   → optimizer_Confignode.instantiate() → AdamW(lr=0.0001)
# config_kwargs = {"optimizer": <AdamW optimizer>}
# → TrainingLoop(optimizer=<AdamW>)
```

---

### 2.11 序列化：`to_dict` / `to_yaml_dict`

```python
class ConfigNode:
    def to_dict(self) -> dict:
        """递归转换为普通 dict（用于 checkpoint 保存）。

        注意：此方法会丢弃 _target_，不可逆——无法从返回的 dict 还原回 ConfigNode。
        需要保留 _target_ 的场景请使用 to_yaml_dict()。
        """
        return {
            k: self._unwrap(v)
            for k, v in self.__dict__.items()
            if k not in ("_target_", "raise_on_missing_attr", "_raw_config", "_original_strings")
        }

    def _unwrap(self, v: Any) -> Any:
        if isinstance(v, ConfigNode):
            return v.to_dict()
        elif isinstance(v, list):
            return [self._unwrap(item) for item in v]
        else:
            return v

    def to_yaml_dict(self, *, use_orig_values: bool = True, **kwargs) -> dict:
        """转换为 YAML 可序列化的 dict。

        use_orig_values=True 时，_target_ 和 *_fn 输出原始字符串
        （如 "torch.optim.AdamW"），而非 callable 的 repr。
        """
        def _convert(key, value):
            if isinstance(value, ConfigNode):
                return value.to_yaml_dict(use_orig_values=use_orig_values, **kwargs)
            if isinstance(value, list):
                return [_convert(None, v) for v in value]
            # _target_ / *_fn / callable → 原始字符串或 dotted path
            orig_strings = getattr(self, "_original_strings", {})
            if use_orig_values and key in orig_strings:
                return orig_strings[key]
            if callable(value) or inspect.ismethod(value) or inspect.isclass(value):
                return self._to_dotted_path(value)  # 反向转换
            if use_orig_values and hasattr(value, "_orig_value"):
                return getattr(value, "_orig_value")
            return value

        return {
            k: _convert(k, v)
            for k, v in self.__dict__.items()
            if k not in ("raise_on_missing_attr", "_raw_config", "_original_strings")
        }

    def __contains__(self, key: object) -> bool:
        """支持 `key in cfg` —— ConfigNode 不可迭代，按内部键集判定。"""
        return key in self.to_dict()

    def replace(self, **overrides) -> "ConfigNode":
        """不可变更新：基于 to_dict() 取值，应用 overrides 后构造新 ConfigNode。

        用于 `build_validation_dataloader` 等场景需覆盖个别字段（如 packed_sequence_size=0）
        而不污染原 config（见 02 §`build_validation_dataloader`）。等价于
        `ConfigNode({**cfg.to_dict(), **overrides})`，但保留 _target_ 解析语义。
        """
        new_dict = self.to_dict()
        new_dict.update(overrides)
        new_cfg = ConfigNode(new_dict)
        # Preserve _target_ from original ConfigNode (to_dict() intentionally excludes it)
        if self._target_ is not None:
            new_cfg._target_ = self._target_
        if "_target_" in self._original_strings:
            new_cfg._original_strings["_target_"] = self._original_strings["_target_"]
        return new_cfg
```

---

### 2.12 安全模型

```python
# 白名单：只允许从显式列出的顶层包前缀解析 _target_ / *_fn。
# 收敛原因：原 _is_allowed_module 含"已导入模块即放行"（if top_level in sys.modules）
# 与 ENABLE_USER_MODULES 全放行分支，导致任意已 import 的顶层模块都可被
# _resolve_target 解析为 callable，三层安全退化为"已安装即放行"。
# canonical：显式白名单前缀匹配 → 放行；其余 → 阻止。
# 用户扩展须显式注册到 hyper_parallel 命名空间下，或在此处显式追加前缀。
ALLOWED_IMPORT_PREFIXES = (
    "hyper_parallel",     # 框架自身
    "torch",              # torch.optim.AdamW 等
    "transformers",       # AutoTokenizer / AutoModel 兼容入口
    "datasets",           # datasets.load_dataset
    "torchdata",          # StatefulDataLoader
    "torchao",
    "liger_kernel",
)

def _is_allowed_module(module_name: str) -> bool:
    """白名单前缀匹配 → 放行；其余 → 阻止。

    不再提供"已导入模块即放行"与 ENABLE_USER_MODULES 全放行分支，
    避免 sys.modules 状态污染白名单。
    """
    top_level = module_name.split(".", 1)[0]
    return top_level in ALLOWED_IMPORT_PREFIXES

def _is_safe_attr(name: str) -> bool:
    """阻止访问私有/魔术属性。"""
    return not (name.startswith("_") or "__" in name)
```

---

### 2.13 与旧代码的对比

| 旧方式（硬编码） | 新方式（ConfigNode `_target_` IoC） |
|-----------------|------------------------|
| `if model_type == "llama": model = LlamaForCausalLM(config)` | `cfg.model.instantiate()` — YAML 决定类型 |
| `if optim == "adamw": opt = AdamW(params, **kwargs)` | `cfg.optimizer.instantiate(params=params)` |
| `register_spec("qwen3_5", ModelSpec(...))` | YAML 配置 `_target_: ...Qwen3_5ForCausalLM` |
| 新增模型需要修改 Recipe 代码 | 新增模型只需新增 YAML 配置文件 |
| 新增组件需要注册 + if/else 分支 | 新增组件只需在 YAML 中声明 `_target_` |
| `_target_` / `*_fn` 是字符串，运行时才 import | ConfigNode 构造时立即解析，`instantiate()` 零延迟 |
| 配置拼写错误静默返回 None | `raise_on_missing_attr=True` 立即抛出 AttributeError |
---


### 2.14 辅助函数签名

```python
# hyper_models/components/config/_utils.py

def resolve_yaml_env_vars(v: Any) -> Any:
    """解析 YAML 值中的环境变量引用（如 ${VAR_NAME}）。

    在 _wrap() 和 _instantiate_value() 的叶子值路径中被调用，确保 $ENV_VAR
    在配置加载和实例化时都被展开。解析后的值不含 $ 引用时原样返回；
    _OrigValueStr 会保留原始占位符供 to_yaml_dict() 安全输出。
    """
    ...

def load_module_from_file(file_path: str):
    """从 .py 文件动态加载模块（用于用户自定义组件）。

    通过 importlib 从文件路径创建模块对象并执行模块代码，
    返回模块对象供 _resolve_target 的 getattr 链使用。
    """
    ...

def _safe_getattr(obj, attr: str) -> Any:
    """安全 getattr——阻止私有/魔术属性访问。

    在 _resolve_target 的逐级 getattr 链中被调用，
    对私有属性（以 _ 开头）抛出 AttributeError 以维护白名单安全模型。
    """
    ...

def _as_dict(cfg: Any) -> dict:
    """将 ConfigNode 或 Mapping 安全转换为普通 dict。"""
    ...

def _redact(kwargs: dict) -> dict:
    """脱敏关键字参数（隐藏 password/secret 等）。"""
    ...


# hyper_models/components/distributed/fsdp2.py

def _instantiate_fsdp2(*, config, mesh_context) -> "FSDP2Manager | None":
    """根据 strategy config 创建 FSDP2Manager 实例。

    canonical：FSDP2Manager 收 2 参 (config, mesh: MeshContext)，
    内部从 mesh.device_mesh / mesh.device 取出 DeviceMesh 与 device。

    canonical（裁决：以 01 为准）：本工厂的关键字形参名为 `mesh_context`——
    06 §4.1 中 `(config, mesh)` 的表述需向此对齐。
    """
    if config is None:
        return None
    return FSDP2Manager(config=config, mesh=mesh_context)


# hyper_models/components/distributed/pipelining.py

def _instantiate_pipeline(pipeline_config, mesh) -> "AutoPipeline | None":
    """根据 pipeline_config 创建 AutoPipeline 实例（pp_size > 1 时）。

    canonical：2 参签名 (pipeline_config, mesh)，与 §8.2
    `AutoPipeline.__init__(self, pipeline_config, mesh)` 一致（device 由 mesh 内部携带）。
    """
    ...


# hyper_models/components/checkpoint/checkpointing.py

def _load_full_state_dict_into_model(model: nn.Module, state_dict: dict) -> None:
    """将 full state dict 加载到（可能已 DTensor 分片的）模型中，每 rank 独立切分。"""
    ...

def _get_state_dict_adapter(model: nn.Module) -> "StateDictAdapter | None":
    """从模型中提取 StateDictAdapter（查找 _state_dict_adapter 属性）。"""
    ...


# hyper_models/components/models/common/

def _model_name_from_cfg(model_cfg) -> str | None:
    """从模型 ConfigNode 中提取 pretrained_model_name_or_path。"""
    ...

def build_model(
    model_cfg,
    peft_config=None,
    distributed_setup=None,
    **kwargs,
) -> tuple["nn.Module", "OptimizerInit"]:  # 需要 Python 3.12+ 或 from __future__ import annotations
    """高层 build_model 入口——ConfigNode.instantiate() 的快捷方式。

    与 HyperAutoModel.from_pretrained 的职责区分：
    - from_pretrained 是 HF 入口，**返回单 model**（PreTrainedModel）。
    - build_model 是 Recipe 内部编排入口，**返回 (model, optimizer_init)**——
      内部调用 from_pretrained（自定义路径）或 _build_model（HF 原生路径）
      完成 meta→shard→load，并从 ShardingPlan / distributed_setup 导出
      OptimizerInit，供 Recipe.setup() 调用 OptimizerConfig.build(model,
      optimizer_init=...) 时使用，避免 Recipe 重复推导 param 分组与 mesh 信息。

    Returns:
        (model, optimizer_init)
        - model: 已分片、权重已加载的模型（meta→shard→load 完成）
        - optimizer_init: 见 OptimizerInit。
    """
    # ① 先处理 peft_config（供 from_pretrained / _build_model 内部 PEFT 注入）
    if peft_config is None and hasattr(model_cfg, "get"):
        peft_node = model_cfg.get("peft")
        if peft_node is not None:
            peft_config = peft_node.instantiate()

    # ② 调用 HF 兼容入口构建模型（自定义路径走 from_pretrained，含 meta→shard→load）
    #    _target_ 已在 ConfigNode 构造时解析为 HyperAutoModelForCausalLM.from_pretrained
    model = model_cfg.instantiate(
        distributed_setup=distributed_setup,
        peft_config=peft_config,
        **kwargs,
    )

    # ③ 从 distributed_setup / ShardingPlan 导出 OptimizerInit（param 分组、mesh、is_peft）
    #    weight_decay 由 Recipe 侧从 cfg.optimizer 读取后经 OptimizerConfig.build 生效（§3.4）；
    #    此处不臆造 wd 值（默认 0.0 占位，禁止用 True——True 等价于 wd=1.0）。
    optimizer_init = OptimizerInit.from_distributed_setup(
        distributed_setup=distributed_setup,
        model=model,
        peft_config=peft_config,
    )
    return model, optimizer_init


@dataclass
class OptimizerInit:
    """优化器初始化描述——由 build_model 从 distributed_setup / ShardingPlan 导出。

    Recipe.setup() 将其传给 OptimizerConfig.build(model, optimizer_init=...)，
    避免 Recipe 侧重复推导 param 分组与 mesh 信息（与 03 §self.model,
    self.optimizer_init = build_model(...) 对称）。
    """
    # param_groups 分组（decay / no_decay / lora_only 等），由 ShardingPlan 推导
    param_groups: list[dict]
    # DeviceMesh（用于 optimizer state 的 DTensor placement）；04 §5.3 期待 DeviceMesh
    device_mesh: "DeviceMesh | None"
    # 是否为 PEFT 训练（影响 param 过滤 / 可训练参数统计）
    is_peft: bool = False
    # 可选：tp_grad_info（由 apply_sharding_plan 导出，供 FSDP2 / optimizer 复用）
    tp_grad_info: Any = None

    @classmethod
    def from_distributed_setup(
        cls,
        *,
        distributed_setup,
        model: "nn.Module",
        peft_config=None,
        weight_decay: float = 0.0,
    ) -> "OptimizerInit":
        """从 distributed_setup.mesh_context.device_mesh + model 参数推导分组。

        Args:
            weight_decay: 从 optimizer 配置读取的实际 weight_decay 值
                （如 cfg.optimizer.weight_decay），用于 decay 组；no_decay 组恒为 0.0。
                ★ 禁止传 bool——`weight_decay=True` 传入 AdamW 等价于 wd=1.0。
        注：最终分组由 OptimizerConfig.build 内部以 self.weight_decay 重做（§3.4），
        此处 param_groups 为预分组描述，供调用方/调试参考。
        """
        mesh_ctx = getattr(distributed_setup, "mesh_context", None) if distributed_setup else None
        device_mesh = mesh_ctx.device_mesh if mesh_ctx is not None else None
        is_peft = peft_config is not None

        # 简化的 decay/no_decay 分组（完整分组逻辑由 OptimizerConfig.build 内部完成）
        decay_p, no_decay_p = [], []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            (no_decay_p if _is_no_decay(name) else decay_p).append(param)
        param_groups = [
            {"params": decay_p, "weight_decay": weight_decay},   # decay 组：配置的实际 wd 值
            {"params": no_decay_p, "weight_decay": 0.0},          # no_decay 组：恒 0.0
        ]
        return cls(
            param_groups=param_groups,
            device_mesh=device_mesh,
            is_peft=is_peft,
        )


# hyper_models/components/models/common/param_utils.py
def _is_no_decay(name: str) -> bool:
    """判定参数是否应归入 no_decay 组（常规规则：bias 与 1D norm 权重不衰减）。

    被 OptimizerInit.from_distributed_setup 与 AdamWConfig.build 复用（第六轮 P1 修复：
    此前调用点存在但函数未定义）。匹配 "bias"、层归一化权重（weight 且 ndim<=1 或名字含
    "norm"/"ln"/"layernorm"）等常见模式。
    """
    if name.endswith("bias"):
        return True
    if "norm" in name.lower() or "ln" in name.lower() or "layernorm" in name.lower():
        return True
    return False


# hyper_models/components/logging/wandb.py

class WandbConfig:
    @classmethod
    def from_kwargs(cls, **kwargs) -> "WandbConfig | None":
        """从关键字参数创建 WandbConfig 实例。"""
        ...
```

---

## 3. RecipeConfig 类型化桥接

### 3.1 设计动机

ConfigNode 是"弱类型"的——任何 YAML key 都会被接受，且所有值在加载时已解析完成（eager resolution）。但 Recipe 需要**类型安全**：optimizer 的 `lr` 必须是 float，scheduler 的 `warmup_steps` 必须是 int。

`RecipeConfig` 是两者之间的桥接层：

```
YAML ConfigNode（弱类型）  →  RecipeConfig（类型化）  →  Recipe.setup()
     "_target_" 已解析              typed 属性                    .build() / .instantiate()
     任意 key 可访问                类型校验 + 默认值              运行时依赖注入
```

### 3.2 两个关键工具函数

```python
# recipes/_typed_config.py

def _section_kwargs(node: Any) -> dict[str, Any]:
    """提取 config 字段（丢弃 _target_）。

    用于**固定类型**的组件（lr_scheduler, step_scheduler, checkpoint 等）。
    这些组件的类型不需要多态，只需要提取 YAML 中的配置值。
    """
    d = node.to_dict() if hasattr(node, "to_dict") else dict(node)
    d.pop("_target_", None)  # to_dict() 已排除 _target_，此处为幂等安全调用
    return d


def _callable_and_kwargs(cfg: Any) -> tuple[Callable, dict]:
    """从 ConfigNode 中提取 _target_ factory + 剩余 kwargs。

    用于**多态类型**的组件（optimizer, loss_fn 等）。
    这些组件通过 _target_ 支持任意类型。

    支持的 cfg 形态：
    - ConfigNode（有 to_dict） → pop _target_
    - 普通对象 → getattr _target_
    - 直接 callable → (cfg, {})
    """
    if hasattr(cfg, "to_dict") or isinstance(cfg, Mapping):
        cfg_dict = _as_dict(cfg)
        target = cfg_dict.pop("_target_", None)
        if target is not None:
            return target, cfg_dict
    target = getattr(cfg, "_target_", None)
    if target is not None:
        return target, {}
    if callable(cfg):
        return cfg, {}
    if hasattr(cfg, "instantiate"):
        return cfg.instantiate, {}
    raise AttributeError(
        "Config must provide _target_, be callable, or provide instantiate()"
    )
```

### 3.3 RecipeConfig 完整实现

> **Canonical 声明（裁决）**：本节为 `RecipeConfig` 的**唯一 canonical 定义**。
> 03_training_loop.md §5.2 与 04_checkpoint.md §9 中的 `RecipeConfig` 展示均为
> 本节的引用/节选；三处如有不一致，**以本节为准**。

```python
# recipes/_typed_config.py

class RecipeConfig:
    """将 YAML ConfigNode 桥接到强类型配置 Dataclass。

    两类属性：
    - typed（cached_property，返回类型化 Config 实例，拥有 .build() 方法）:
      optimizer, lr_scheduler, step_scheduler, loss_fn, checkpoint, wandb, mlflow
    - untyped（__getattr__ 透传原始 ConfigNode，拥有 .instantiate() 方法）:
      model, dataset, dataloader, peft, packed_sequence 等
    """

    def __init__(self, raw: ConfigNode):
        self._raw = raw

    # ═══════════════════════════════════════════
    # typed 属性：_target_ 提取 + 类型校验 → 返回类型化 Config
    # ═══════════════════════════════════════════

    @cached_property
    def optimizer(self) -> "OptimizerConfig | None":
        """optimizer: 多态类型（通过 _target_ 支持任意优化器）。

        YAML 示例:
            optimizer:
              _target_: torch.optim.AdamW
              lr: 2.0e-4
              weight_decay: 0.1
        """
        from hyper_models.components.optim.optimizer import build_optimizer_config

        node = self._raw.get("optimizer", None)
        if node is None:
            return None
        factory, kwargs = _callable_and_kwargs(node)
        # build_optimizer_config 将 factory（如 torch.optim.AdamW）+ kwargs
        # 归一化为 OptimizerConfig 子类实例（AdamWConfig / OptimizerFromFactoryConfig）
        return build_optimizer_config(factory, kwargs)

    @cached_property
    def lr_scheduler(self) -> "LRSchedulerConfig | None":
        """lr_scheduler: 固定类型（LRSchedulerConfig）。

        YAML 示例:
            lr_scheduler:
              lr_warmup_steps: 100
              lr_decay_style: cosine
              min_lr: 1.0e-6
        """
        node = self._raw.get("lr_scheduler", None)
        return LRSchedulerConfig(**_section_kwargs(node)) if node else None

    @cached_property
    def step_scheduler(self) -> "StepSchedulerConfig":
        """step_scheduler: 固定类型（StepSchedulerConfig）。

        过滤掉运行时参数（local_batch_size, dp_size, dataloader），
        这些由 .build() 的调用者传入。
        """
        node = self._raw.get("step_scheduler", None)
        if node is None:
            return StepSchedulerConfig()
        kwargs = {
            k: v for k, v in _section_kwargs(node).items()
            if k not in ("local_batch_size", "dp_size", "dataloader")
        }
        return StepSchedulerConfig(**kwargs)

    @cached_property
    def loss_fn(self) -> "LossConfig | None":
        """loss_fn: 多态类型（通过 _target_ 支持任意 loss 函数）。

        YAML 示例:
            loss_fn:
              _target_: hyper_models.components.loss.masked_ce.MaskedCrossEntropy
        """
        from hyper_models.components.loss import build_loss_config

        node = self._raw.get("loss_fn", None)
        if node is None:
            return None
        factory, kwargs = _callable_and_kwargs(node)
        return build_loss_config(factory, **kwargs)

    @cached_property
    def checkpoint(self) -> "CheckpointingConfig":
        """checkpoint: 固定类型（CheckpointingConfig）。

        模型派生字段（model_repo_id, model_cache_dir, is_peft）在此注入。
        """
        from hyper_models.components.checkpoint.config import CheckpointingConfig

        node = self._raw.get("checkpoint", None)
        kwargs = _as_dict(node) if node is not None else {}
        kwargs.pop("restore_from", None)  # 由 Recipe 单独处理
        model = self._raw.get("model", None)
        kwargs |= {  # dict union (|=) 需要 Python 3.9+
            "model_repo_id": _model_name_from_cfg(model) if model is not None else None,
            # 自 04 §9 版并入 canonical：从 model 段派生缓存目录
            "model_cache_dir": self._raw.get("model.cache_dir", None),
            "is_peft": bool(self._raw.get("peft", None)),
        }
        return CheckpointingConfig(**kwargs)

    @cached_property
    def wandb(self) -> "WandbConfig | None":
        node = self._raw.get("wandb", None)
        return WandbConfig.from_kwargs(**_section_kwargs(node)) if node else None

    # ═══════════════════════════════════════════
    # untyped 属性：透传原始 ConfigNode
    # ═══════════════════════════════════════════

    def __getattr__(self, name: str) -> Any:
        """所有未显式定义的属性透传到原始 ConfigNode。

        这意味着 cfg.model / cfg.dataset / cfg.dataloader / cfg.peft
        都返回原始 ConfigNode（其 _target_ 已解析为 callable）。
        """
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._raw, name)

    def __contains__(self, key: object) -> bool:
        # ★ ConfigNode 不可迭代，不能直接 `key in self._raw`；改用 to_dict() 取键集。
        return key in self._raw.to_dict()

    def get(self, key: str, default: Any = None) -> Any:
        """点号路径访问，直接透传 `ConfigNode.get`（§2.7）。

        `self._raw` 是 01 §2 的自研 **ConfigNode**（不是 OmegaConf）；
        ConfigNode.get 原生支持点号路径（如 "step_scheduler.local_batch_size"），
        未命中返回 default。
        """
        return self._raw.get(key, default)

    def to_dict(self) -> dict:
        return self._raw.to_dict()
```

### 3.4 为什么需要 `.build()` 而非直接 `.instantiate()`？

**`cfg.optimizer` 返回的是 `OptimizerConfig` 实例——一个类型化的配置 dataclass，不是 `torch.optim.Optimizer`。**

```python
# cfg.optimizer → OptimizerConfig(lr=0.0001, betas=(0.9, 0.95), weight_decay=0.01)
# 这只是一个配置对象！不能用于训练。
```

**`cfg.optimizer.build(model)` 才创建真正的优化器。** 它内部做了四件事：

```python
class AdamWConfig(OptimizerConfig):
    lr: float = 1e-3
    betas: tuple = (0.9, 0.999)
    weight_decay: float = 0.1

    def build(self, model, *, optimizer_init=None, device_mesh=None, is_peft=False):
        """OptimizerConfig → list[torch.optim.Optimizer]

        ① 遍历 model 参数 → 分组（decay / no_decay）
        ② 处理 model.parts（PP 模式下返回多个优化器）
        ③ 用分好的 param_groups + 存储的 kwargs 创建优化器
        ④ 返回 list[Optimizer]（每 part 一个）
        """
        parts = getattr(model, "parts", [model])
        optimizers = []
        for part in parts:
            # ★ 参数分组——这是 .instantiate() 做不到的
            decay_p, no_decay_p = [], []
            for name, param in part.named_parameters():
                if not param.requires_grad:
                    continue
                if _is_no_decay(name):
                    no_decay_p.append(param)
                else:
                    decay_p.append(param)
            param_groups = [
                {"params": decay_p, "weight_decay": self.weight_decay},
                {"params": no_decay_p, "weight_decay": 0.0},
            ]
            # ★ 现在才能创建优化器——param_groups 依赖运行时 model 对象
            optimizers.append(torch.optim.AdamW(
                param_groups, lr=self.lr, betas=self.betas
            ))
        return optimizers
```

**`.instantiate()` 和 `.build()` 的根本区别**：

| | `.instantiate()` | `.build()` |
|------|---------|---------|
| 中间层 | **无**——ConfigNode 就是构造参数 | **有**——Typed Config（`OptimizerConfig` 等） |
| YAML → 构造参数 | 1:1 映射：`{path: "x"}` → `load_dataset(path="x")` | 需要 **transform**：`{lr: 2e-4}` → 先分组参数 → `AdamW(param_groups, lr=2e-4)` |
| 构造参数能否提前确定 | ✅ `tokenizer`、`sampler` 等可以通过 runtime kwargs 传入 | ❌ `model.parameters()` 迭代器无法作为 kwarg 传递 |
| 典型场景 | `load_dataset(path=..., tokenizer=t)` | `AdamW(param_groups, lr=...)` 其中 `param_groups` 是 live 对象 |

**简单说：`.instantiate()` 做的是"把 YAML 的 key-value 原样传给构造函数"；`.build()` 做的是"把 YAML 的 key-value 先转换成构造函数能接受的形态，再传进去"。** 参数分组是这个转换的典型例子。

### 3.5 类型化 Config 的 `.build()` 调用一览

```python
# Recipe.setup() 中的使用

# optimizer: OptimizerConfig → .build(model) → list[Optimizer]
#   内部：遍历 model 参数 → decay/no_decay 分组 → 创建优化器
self.optimizer = self.cfg.optimizer.build(
    model, device_mesh=self.mesh.device_mesh
)

# lr_scheduler: LRSchedulerConfig → .build(optimizer, step_scheduler) → OptimizerParamScheduler
#   内部：从 step_scheduler 推断默认步数 → 创建调度器
self.lr_scheduler = self.cfg.lr_scheduler.build(
    self.optimizer, self.step_scheduler
)

# loss_fn: LossConfig → .build() → nn.Module
#   内部：用 _target_ factory + kwargs 实例化 loss 模块
self.loss_fn = self.cfg.loss_fn.build()

# checkpoint: CheckpointingConfig → .build(dp_rank, ...) → Checkpointer
#   内部：创建 Checkpointer 实例，初始化 StorageWriter/Reader
self.checkpointer = self.cfg.checkpoint.build(
    dp_rank=dp_rank, tp_rank=tp_rank, pp_rank=pp_rank
)

# step_scheduler: StepSchedulerConfig → .build(dataloader, dp_size, local_bs) → StepScheduler
self.step_scheduler = self.cfg.step_scheduler.build(
    self.dataloader, dp_size, local_batch_size
)
```

---

## 4. 总入口调用时序：从 `main()` 到所有组件就绪

下面从 `main()` 开始，逐层展开每一个函数调用，直到模型加载完成、优化器就绪。数字序号表示调用顺序，缩进表示调用深度。

### 4.1 总体调用链

```
main()                                          # recipes/llm/train_ft.py
│
├─① cfg = parse_args_and_load_config()
│   └─ load_yaml_config("train.yaml")           # §2.2
│       └─ yaml.safe_load() → raw dict
│       └─ ConfigNode(raw)                      # §2.3: 递归 _wrap()，即时解析所有 _target_ 和 *_fn
│           ├─ _wrap("model", {...}) → ConfigNode({_target_: <bound method from_pretrained>, ...})
│           ├─ _wrap("optimizer", {...}) → ConfigNode({_target_: <class AdamW>, lr: 0.0001, ...})
│           ├─ _wrap("dataset", {...}) → ConfigNode({_target_: <function load_dataset>, ...})
│           │   └─ _wrap("tokenizer", {...}) → ConfigNode({_target_: <bound method from_pretrained>, ...})
│           └─ ...所有 _target_ 和 *_fn 已解析为 callable
│
├─② cfg = RecipeConfig(cfg)                     # §3: 类型化桥接
│   ├─ cfg.optimizer → OptimizerConfig(lr=0.0001, betas=(0.9,0.95), ...)  # _callable_and_kwargs → build_optimizer_config
│   ├─ cfg.lr_scheduler → LRSchedulerConfig(lr_warmup_steps=100, ...)      # _section_kwargs → LRSchedulerConfig(**kwargs)
│   ├─ cfg.step_scheduler → StepSchedulerConfig(ckpt_every_steps=500, ...) # _section_kwargs → StepSchedulerConfig(**kwargs)
│   ├─ cfg.loss_fn → LossConfig(factory=<class MaskedCrossEntropy>, ...)   # _callable_and_kwargs → build_loss_config
│   ├─ cfg.checkpoint → CheckpointingConfig(checkpoint_dir="outputs/", ...) # direct construction
│   └─ cfg.model / cfg.dataset / cfg.dataloader / cfg.peft → 保留为 ConfigNode（__getattr__ 透传）
│
├─③ recipe = FinetuneRecipe()
│
├─④ recipe.setup(cfg)                           # §4.2: 构建所有训练组件
│   │
│   ├─④.1 initialize_distributed("nccl")                          # 初始化 torch.distributed 进程组 + CUDA device
│   ├─④.2 self.rng = StatefulRNG(seed=cfg.get("seed", 42), ranked=True)
│   ├─④.3 self.distributed_setup = create_distributed_setup_from_config(cfg)  # 从 cfg 构建分布式拓扑 → 06_distributed_infrastructure.md §3
│   │
│   ├─④.3a self.callback_manager = build_callback_manager(                     # 03 §4.2: 混合 Callback 系统——注册 CheckpointCallback、
│   │       cfg, ...)                                                           # EvaluateCallback、LoggingCallback、TqdmCallback 等内置 callback
│   │
│   ├─④.4 self.peft_config = cfg.peft.instantiate() if cfg.get("peft") else None  # ★ PEFT 先实例化
│   │   self.model, self.optimizer_init = build_model(          # §2.14 / §4.2: Recipe 内部编排入口，返回 (model, optimizer_init)
│   │       cfg.model,                                          # 与 03 §5.3 ⑪ / §4.2 口径一致（canonical：build_model）
│   │       peft_config=self.peft_config,
│   │       distributed_setup=self.distributed_setup)
│   │   │
│   │   ├─ build_model 内部①: model = cfg.model.instantiate(    # cfg.model 是 ConfigNode（__getattr__ 透传）
│   │   │       distributed_setup=distributed_setup,
│   │   │       peft_config=peft_config)
│   │   │                                                       # _target_ = <bound method HyperAutoModelForCausalLM.from_pretrained>
│   │   │
│   │   └─ HyperAutoModelForCausalLM.from_pretrained(        # §6.2
│   │           pretrained_model_name_or_path="Qwen/Qwen3.5-0.8B",
│   │           torch_dtype="bfloat16",
│   │           distributed_setup=<DistributedSetup>,
│   │           peft_config=<PeftConfig | None>)
│   │       │
│   │       ├─④.4.1 mesh = distributed_setup.mesh_context                    # 解析分布式拓扑
│   │       │
│   │       ├─④.4.2 sharding_planner, fsdp2_manager, autopipeline           # §8
│   │       │       = instantiate_infrastructure(
│   │       │             distributed_setup=distributed_setup,
│   │       │             device=torch.device("cuda", current_device))
│   │       │   ├─ ShardingPlanner()                                         # hyper_parallel 核心 (05_dual_mode_dtensor §1)
│   │       │   ├─ FSDP2Manager(config, mesh)   if strategy                  # DP 维度 (06_distributed_infrastructure.md §4)，2 参 MeshContext
│   │       │   └─ AutoPipeline(pipeline_config, mesh)     if pp_size > 1    # PP 维度 (本文档 §8.2)
│   │       │
│   │       ├─④.4.3 hf_config = AutoConfig.from_pretrained(                 # 获取 HF 配置
│   │       │       pretrained_model_name_or_path,
│   │       │       attn_implementation="sdpa",
│   │       │       torch_dtype="bfloat16")
│   │       │
│   │       ├─④.4.4 is_hf_model = get_is_hf_model(hf_config)                # §5: 查 MODEL_ARCH_MAPPING
│   │       │   ├─ arch_name = hf_config.architectures[0]                   # e.g. "Qwen3_5ForCausalLM"
│   │       │   └─ _resolve_custom_model_cls(arch_name)                     # 懒加载 import
│   │       │       ├─ 命中 → 返回自定义模型类 → is_hf_model=False
│   │       │       └─ 未命中 → None → is_hf_model=True
│   │       │
│   │       └─④.4.5 model = _build_model(                                   # §6.3: 核心编排
│   │               pretrained_model_name_or_path,
│   │               is_hf_model=False,       # 假设自定义模型
│   │               hf_config=hf_config,
│   │               mesh=mesh,
│   │               sharding_planner=sharding_planner,
│   │               fsdp2_manager=fsdp2_manager,
│   │               torch_dtype=torch_dtype,
│   │               validate_placement=False,
│   │               load_base_model=True)
│   │           │
│   │           ├─④.4.5.1 is_meta_device = (world_size > 1 or not is_hf_model)  # 确定 meta device
│   │           │       init_ctx = (no_init_weights(), init_empty_weights())
│   │           │
│   │           ├─④.4.5.2 with init_ctx:                                        # §7: 模型实例化
│   │           │       is_custom_model, model = _init_model(
│   │           │           cls=HyperAutoModelForCausalLM,
│   │           │           pretrained_model_name_or_path,
│   │           │           hf_config, attn_implementation,
│   │           │           torch_dtype, is_hf_model=False)
│   │           │   ├─ arch_name = "Qwen3_5ForCausalLM"
│   │           │   ├─ model_cls = _resolve_custom_model_cls(arch_name)        # §5: MODEL_ARCH_MAPPING 懒加载
│   │           │   │   → importlib.import_module("...qwen3_5.model")
│   │           │   │   → Qwen3_5ForCausalLM
│   │           │   └─ model = Qwen3_5ForCausalLM(hf_config)                   # meta device 空壳
│   │           │       → 模型结构完整，参数为 meta tensor (零显存)
│   │           │
│   │           ├─④.4.5.3 _apply_peft(model, peft_config)         if peft_config    # LoRA 层注入 (§6.4)
│   │           ├─④.4.5.4 _apply_qat(model, qat_config)           if qat_config
│   │           ├─④.4.5.5 _apply_fp8(model, fp8_config)           if fp8_config
│   │           ├─④.4.5.6 _apply_parameter_freezing(model, freeze_config)            # 参数冻结 (§6.5)
│   │           │
│   │           ├─④.4.5.7 plan = sharding_planner.plan(model, mesh.device_mesh, ...)  # §9: 推导分片策略 → 05_dual_mode_dtensor §5
│   │           │   ├─ ParameterClassifier.classify(model)                             # ParamRole 分类 (05 §3.6.2)
│   │           │   ├─ BoundaryGrouper.group(model)                                    # 模块边界分组
│   │           │   ├─ TemplateLookup.lookup(roles, boundary_types)                    # 查 ShardingTemplate (05 §3.5)
│   │           │   ├─ ChainPropagator.propagate(specs)                                # 链式传播校验 (05 §3.6.5)
│   │           │   └─ → ShardingPlan(module_specs={...})                              # 可序列化中间表示 (05 §3)
│   │           │
│   │           ├─④.4.5.8 apply_sharding_plan(model, plan, mesh.device_mesh, validate_mode=False)  # 应用分片 → 05_dual_mode_dtensor §6
│   │           │   ├─ 生产模式: _local_params_context → 用 DTensor._local_tensor 替换参数
│   │           │   │   → PrecompiledBoundary 预编译通信原语 → 零运行时 dispatch 开销 (05 §7)
│   │           │   │   └─ _local_params_context: DTensor._local_tensor → 零拷贝 (06 §5)
│   │           │   ├─ build_tp_grad_info(plan, tp_mesh) → tp_grad_info（从 ShardingPlan 导出，非 DTensor placement）
│   │           │   └─ 校验模式: DTensor 传播 → assert placements 一致 (05 §8)
│   │           │
│   │           ├─④.4.5.9 torch.compile(model, **compile_config)   if compile_config  # Inductor 编译（fully_shard 之前）
│   │           │
│   │           ├─④.4.5.10 fsdp2_manager.parallelize(model, tp_grad_info=tp_grad_info)  if fsdp2  # FSDP2 在 meta 上包裹（canonical：先于 to_empty/load）
│   │           │
│   │           ├─④.4.5.11 load_base_model(model, device, pretrained_path,            # canonical：④.4.5.11 = load_base_model（以衔接表为准）
│   │           │         │  adapter=_get_state_dict_adapter(model),                  # §10.3: 每 rank 独立加载权重 (04_checkpoint.md §5.3, 5 参 canonical 签名)
│   │           │         │  mesh=mesh.device_mesh)                                   # ★ DeviceMesh：按 TP/DP 读本地份，零 NCCL
│   │           │         └─ 前置动作（同属 ④.4.5.11 一步）：model.to_empty(device=device)   # meta → GPU（物化 sharded 参数），load_base_model 写入前必须先物化
│   │           │
│   │           ├─④.4.5.12 _freeze_non_lora_params(model)              if peft_config   # PEFT 非 LoRA 参数冻结 (§6.4)
│   │           └─ return model                                         # 已分片、权重已加载
│   │   ← build_model 内部②: optimizer_init = OptimizerInit.from_distributed_setup(...)  # 导出 param 分组/mesh
│   │   ← build_model 返回 (model, optimizer_init)                      # → self.model, self.optimizer_init
│   │
│   ├─④.5 self.loss_fn = cfg.loss_fn.build()                                          # typed: LossConfig → nn.Module
│   │   └─ 详见 03_training_loop.md §10
│   │
│   ├─④.6 self.peft_config 已在 ④.4 之前实例化并注入 model（见 ④.4）                 # PEFT 在分片之前注入
│   │
│   ├─④.7 self.checkpointer = cfg.checkpoint.build(dp_rank=..., tp_rank=..., ...)     # typed: CheckpointingConfig → Checkpointer
│   │   └─ 详见 04_checkpoint.md §4/§5
│   │
│   ├─④.8 self.optimizer = cfg.optimizer.build(model, device_mesh=...)                # typed: OptimizerConfig → list[Optimizer]
│   │   ├─ decay/no_decay 参数分组
│   │   ├─ 遍历 model.parts (PP 模式下多 part)
│   │   └─ AdamW(param_groups, lr=2e-4, betas=(0.9,0.95), ...)
│   │
│   ├─④.9 self.dataloader, self.tokenizer = build_dataloader(                         # untyped: → DataLoader + Tokenizer
│   │       cfg.dataset, cfg.dataloader, cfg.model, ...)
│   │   ├─ _build_tokenizer(cfg_model, cfg_ds)
│   │   │   └─ cfg.dataset.tokenizer.instantiate(trust_remote_code=True)
│   │   │       → AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")
│   │   ├─ cfg.dataset.instantiate()                                    # ★ 不传 tokenizer：load_dataset 签名无此参数
│   │   │   → load_dataset(path="HuggingFaceFW/fineweb", split="train", streaming=True)
│   │   │     （tokenizer kwarg 被 02 §4.2 signature 守卫剔除；tokenizer 由 build_dataloader
│   │   │       单独返回，供 map/tokenize 步骤使用）
│   │   ├─ StatefulDistributedSampler(dataset, seed=..., ...)
│   │   └─ StatefulDataLoader(**dl_kwargs, **dl_base_kwargs)   # 02 §3.2 Step 10 直接构造：
│   │       dataset/sampler/collate_fn 由前序步骤注入 dl_kwargs，
│   │       cfg.dataloader 仅提供 num_workers/pin_memory 等通用参数（_target_ 仅作过滤键）
│   │
│   ├─④.10 self.val_dataloaders = build_validation_dataloader(cfg, ...)
│   │
│   ├─④.11 self.step_scheduler = cfg.step_scheduler.build(                             # typed: StepSchedulerConfig → StepScheduler
│   │        self.dataloader, dp_size, local_batch_size)
│   │   └─ 详见 03_training_loop.md §4
│   │
│   ├─④.12 self.lr_scheduler = cfg.lr_scheduler.build(                                 # typed: LRSchedulerConfig → OptimizerParamScheduler
│   │        self.optimizer, self.step_scheduler)
│   │   └─ 详见 03_training_loop.md §9.6
│   │
│   ├─④.13 self.load_checkpoint(cfg.get("checkpoint.restore_from", None))             # 断点续训恢复
│   │   └─ 详见 04_checkpoint.md §8
│   │
│   └─④.14 self.mfu_calc = AutoMFU.from_config(self.model_parts[0])                    # MFU 计算器
│
└─⑤ recipe.run_train_validation_loop()                                                 # 开始训练
    └─ 详见 03_training_loop.md §6/§7/§8
```

**关键时序要点**：

> **PP 说明**：`autopipeline.build(model)`（PP stage 拆分）在 `apply_model_infrastructure()`
> 中**最先执行**（④.4.5.3 之前，PP 未启用时无此步）——裁决以 §8.2 为准，stage 切分必须
> 先于权重加载与 FSDP2 包裹（§8.3 ① / §6.3 Step 3）。时序树中从略。

| 序号 | 操作 | 关键输出 |
|:----:|------|---------|
| ① | YAML 加载 | ConfigNode 树（所有 `_target_` 已解析） |
| ② | RecipeConfig 桥接 | typed config 对象就绪，类型校验完成 |
| ④.4.2 | `instantiate_infrastructure` | ShardingPlanner(05 §3.6) + FSDP2Manager(06 §4) + AutoPipeline(01 §8.2) |
| ④.4.4 | `get_is_hf_model` | MODEL_ARCH_MAPPING 查表 → 自定义/HF 路径判定 |
| ④.4.5.2 | `_init_model` | meta device 空壳模型（零显存） |
| ④.4.5.7 | `sharding_planner.plan()` | ShardingPlan（可序列化分片策略）→ 05 §5 |
| ④.4.5.8 | `apply_sharding_plan()` | DTensor 分片应用（生产/校验双模）+ `_local_params_context` 解包 → 05 §4/§7/§8 |
| ④.4.5.10 | `fsdp2_manager.parallelize()` | FSDP2 在 meta 上包裹（canonical：先于 to_empty/load） |
| ④.4.5.11 | `load_base_model()`（前置 `model.to_empty()` 物化，同属本步） | 每 rank 独立加载权重（5 参 canonical，零 NCCL） |
| ④.3a | `build_callback_manager()` | CallbackManager（含 CheckpointCallback/EvaluateCallback/LoggingCallback/TqdmCallback 等内置 callback）→ 03 §4.2 |
| ④.8 | `cfg.optimizer.build()` | 真正的优化器（参数分组完成）→ 03 §9 |
| ④.9 | `build_dataloader()` | DataLoader + Tokenizer → 02 §3 |
| ④.11 | `cfg.step_scheduler.build()` | StepScheduler → 03 §4 |
| ④.13 | `load_checkpoint()` | 恢复所有组件状态 → 04 §8 |
| ⑤ | `run_train_validation_loop()` | 训练主循环 → 03 §6/§7/§8 |

### 4.2 `setup()` 内部：每条语句的数据来源

以下追踪 `setup()` 中每个关键变量的**完整来源**——它来自哪个 ConfigNode、`_target_` 解析成了什么、`instantiate()` 实际调用了哪个函数：

> 编号约定：本节内部变量标记用 **m1–m9**（m = member/组件变量），与 §4.1 时序树的
> canonical ①–⑤ 编号（①=load_yaml_config … ⑤=instantiate）区分，避免同名歧义。

```python
class FinetuneRecipe(BaseRecipe):
    def setup(self, cfg: RecipeConfig):

        # ═══════════════════════════════════════════════════════
        # m1 model
        # 来源: cfg.model (ConfigNode, __getattr__ 透传)
        # _target_: <bound method HyperAutoModelForCausalLM.from_pretrained>
        # build_model() 是 Recipe 内部编排入口（§6.2），返回 (model, optimizer_init)：
        #   → 内部调用 from_pretrained / _build_model：meta device → ShardingPlanner →
        #     apply_sharding_plan → FSDP2Manager.parallelize → load_base_model
        #   → 从 distributed_setup / ShardingPlan 导出 OptimizerInit（param 分组、mesh）
        # 与 03 §5.3 canonical 完全对齐：`self.model, self.optimizer_init = build_model(...)`
        # ═══════════════════════════════════════════════════════
        # ★ 先实例化 peft_config，再传给 build_model（PEFT 必须在 ShardingPlanner.plan
        #    之前注入，见 §6.4 / §8.3 ②）
        self.mesh = self.distributed_setup.mesh_context
        self.peft_config = cfg.peft.instantiate() if cfg.get("peft") else None
        self.model, self.optimizer_init = build_model(
            cfg.model,
            peft_config=self.peft_config,
            distributed_setup=self.distributed_setup,
        )
        # model_parts：PP 多 stage 时为 list，单 stage 为 [model]（与 03 §5.3 对齐）
        self.model_parts = (
            self.model.parts if hasattr(self.model, "parts") else [self.model]
        )

        # ═══════════════════════════════════════════════════════
        # m2 tokenizer
        # 来源: cfg.dataset.tokenizer (子 ConfigNode, __getattr__ 透传)
        # _target_: <bound method AutoTokenizer.from_pretrained>
        # 获取方式: build_dataloader() 内部调用 _build_tokenizer()
        #   路径 4 (有 _target_):
        #     → AutoTokenizer.from_pretrained(
        #           pretrained_model_name_or_path="Qwen/Qwen3.5-0.8B",
        #           trust_remote_code=True)
        # 返回: PreTrainedTokenizerBase
        # ═══════════════════════════════════════════════════════
        # tokenizer 在 build_dataloader() 内部获取，不直接出现在 setup() 中

        # ═══════════════════════════════════════════════════════
        # m3 ds (Dataset)
        # 来源: cfg.dataset (ConfigNode, __getattr__ 透传)
        # _target_: <function datasets.load_dataset>
        # 获取方式: build_dataloader() 内部
        #   → load_dataset(path="HuggingFaceFW/fineweb", name="sample-10BT",
        #                  split="train", streaming=True)
        #   ★ 不传 tokenizer：datasets.load_dataset 不接受 tokenizer kwarg，
        #     02 §4.2 的 signature 守卫保证其不会被注入。
        # 返回: Dataset (可能是 IterableDataset)
        # ═══════════════════════════════════════════════════════
        # ds 在 build_dataloader() 内部通过 cfg.dataset.instantiate() 获取

        # ═══════════════════════════════════════════════════════
        # m4 sampler
        # 来源: 不由 _target_ 驱动，由 build_dataloader() 内部逻辑决定:
        #   - map-style Dataset → StatefulDistributedSampler(dataset, seed=..., ...)
        #   - MegatronPretraining → create_megatron_sampler(...)
        #   - IterableDataset → 无 sampler
        # ═══════════════════════════════════════════════════════
        # sampler 在 build_dataloader() 内部创建

        # ═══════════════════════════════════════════════════════
        # m5 dataloader
        # 来源: cfg.dataloader (ConfigNode, __getattr__ 透传)
        # _target_: <class StatefulDataLoader>
        # 获取方式: build_dataloader() 末尾
        #   → StatefulDataLoader(dataset=ds, sampler=sampler, batch_size=1, ...)
        # 返回: DataLoader
        # ═══════════════════════════════════════════════════════

        # 实际调用——build_dataloader() 一次返回 (dataloader, tokenizer):
        self.dataloader, self.tokenizer = build_dataloader(
            cfg.dataset, cfg.dataloader, cfg.model, cfg.packed_sequence,
            seed=cfg.get("seed", 42),
            local_batch_size=cfg.get("step_scheduler.local_batch_size", 1),
            ...
        )
        # build_dataloader() 内部做了上述 m2–m5 全部工作

        # ═══════════════════════════════════════════════════════
        # m6 optimizer
        # 来源: cfg.optimizer (RecipeConfig cached_property → OptimizerConfig)
        # 非 ConfigNode！cfg.optimizer 已经是 OptimizerConfig(lr=2e-4, ...)
        # .build(model) 内部:
        #   ① 遍历 model 参数 → decay/no_decay 分组
        #   ② 创建 AdamW(param_groups, lr=2e-4, betas=(0.9, 0.95), ...)
        # 返回: list[torch.optim.Optimizer]
        # ═══════════════════════════════════════════════════════
        self.optimizer = cfg.optimizer.build(
            self.model,
            optimizer_init=self.optimizer_init,
            device_mesh=self.mesh.device_mesh,
        )

        # ═══════════════════════════════════════════════════════
        # m7 loss_fn
        # 来源: cfg.loss_fn (RecipeConfig cached_property → LossConfig)
        # .build() 内部: MaskedCrossEntropy()
        # 返回: nn.Module
        # ═══════════════════════════════════════════════════════
        self.loss_fn = cfg.loss_fn.build()
    # └─ 详见 03_training_loop.md §10

        # ═══════════════════════════════════════════════════════
        # m8 lr_scheduler
        # 来源: cfg.lr_scheduler (RecipeConfig cached_property → LRSchedulerConfig)
        # .build(optimizer, step_scheduler) 内部:
        #   ① 未设置字段从 step_scheduler 推断默认值
        #   ② 创建 OptimizerParamScheduler(optimizer, lr_warmup_steps=100, ...)
        # 返回: OptimizerParamScheduler
        # ═══════════════════════════════════════════════════════
        self.lr_scheduler = cfg.lr_scheduler.build(
            self.optimizer, self.step_scheduler
        )

        # ═══════════════════════════════════════════════════════
        # m9 peft_config 已在 m1 之前实例化并注入 model（见 m1 上方 self.peft_config
        #    赋值，与 §6.4 / §8.3 ② PEFT 注入时序一致）
        # ═══════════════════════════════════════════════════════
```

---

## 5. MODEL_ARCH_MAPPING 注册中心

### 5.1 数据结构

```python
# _transformers/registry.py

"""MODEL_ARCH_MAPPING: HF architectures → Hyper-Parallel 自定义模型实现。

与 HuggingFace 的 MODEL_FOR_CAUSAL_LM_MAPPING_NAMES 类似。
模型类懒加载——只在首次访问时才 import，减少启动时间。
"""

import importlib
import logging
from collections import OrderedDict
from functools import lru_cache
from typing import Optional

logger = logging.getLogger(__name__)

# OrderedDict 保证查找顺序（先注册先匹配）
MODEL_ARCH_MAPPING = OrderedDict([
    # ── Dense 模型 ──
    ("LlamaForCausalLM", (
        "hyper_models.components.models.llama.model",
        "LlamaForCausalLM",
    )),
    ("Qwen3_5ForCausalLM", (
        "hyper_models.components.models.qwen3_5.model",
        "Qwen3_5ForCausalLM",
    )),
    # ── MoE 模型 ──
    ("Qwen3_5MoeForConditionalGeneration", (
        "hyper_models.components.models.qwen3_5_moe.model",
        "Qwen3_5MoeForConditionalGeneration",
    )),
    # ── VLM 模型 ──
    ("Qwen3_5ForConditionalGeneration", (
        "hyper_models.components.models.qwen3_5.model",
        "Qwen3_5ForConditionalGeneration",
    )),
    ("Qwen3VLMoeForConditionalGeneration", (
        "hyper_models.components.models.qwen3_vl_moe.model",
        "Qwen3VLMoeForConditionalGeneration",
    )),
])


@lru_cache(maxsize=128)
def _resolve_custom_model_cls(arch_name: str) -> Optional[type]:
    """从 MODEL_ARCH_MAPPING 懒加载模型类。

    返回 None 表示无自定义实现 → 降级到 HF 原生。
    """
    entry = MODEL_ARCH_MAPPING.get(arch_name)
    if entry is None:
        return None

    module_path, class_name = entry[0], entry[1]
    try:
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        logger.warning(
            "Failed to load custom model %s from %s: %s. Falling back to HF native.",
            class_name, module_path, e,
        )
        return None


def get_is_hf_model(config, force_hf: bool = False) -> bool:
    """判断是否应该使用 HF 原生实现。

    Returns:
        True: 使用 HF 原生 AutoModel.from_pretrained()
        False: 使用 Hyper-Parallel 自定义实现
    """
    if force_hf:
        return True
    architectures = getattr(config, "architectures", []) or []
    arch_name = architectures[0] if architectures else ""
    return _resolve_custom_model_cls(arch_name) is None


def get_architectures(config) -> list[str]:
    """安全获取模型的 architectures 列表。"""
    return getattr(config, "architectures", []) or []


def get_hf_config(
    path: str,
    attn_implementation: str = "sdpa",
    torch_dtype="auto",
    **kwargs,
):
    """封装 AutoConfig.from_pretrained 并统一注入 attn_implementation / dtype。

    与 get_is_hf_model 同模块（registry.py），供 from_pretrained / _init_model
    共用，避免分散调用 AutoConfig 导致 attn_implementation/dtype 不一致。

    Args:
        path: HF hub repo ID 或本地路径。
        attn_implementation: "sdpa" / "flash_attention_2" / "eager"。
        torch_dtype: "auto" 时透传（由下游 dtype_from_str 再解析）。
        **kwargs: 透传给 AutoConfig.from_pretrained（如 trust_remote_code）。
    """
    from transformers import AutoConfig

    config_kwargs = dict(kwargs)
    config_kwargs.setdefault("attn_implementation", attn_implementation)
    if torch_dtype != "auto":
        config_kwargs.setdefault("torch_dtype", torch_dtype)
    return AutoConfig.from_pretrained(path, **config_kwargs)
```

### 5.2 懒加载机制

非本次调用路径的模型类不会 import，减少启动时间：

```python
# 第一次调用 from_pretrained("Qwen/Qwen3.5-4B")
# → MODEL_ARCH_MAPPING["Qwen3_5ForCausalLM"]
# → importlib.import_module("hyper_models.components.models.qwen3_5.model")
# → 返回 Qwen3_5ForCausalLM 类

# from_pretrained("meta-llama/Llama-3.2-1B")
# → MODEL_ARCH_MAPPING["LlamaForCausalLM"]
# → importlib.import_module("hyper_models.components.models.llama.model")
# → 返回 LlamaForCausalLM 类

# from_pretrained("google/gemma-2b") — 无注册
# → _resolve_custom_model_cls("Gemma2ForCausalLM") → None
# → get_is_hf_model() → True → 降级到 HF 原生 AutoModelForCausalLM
```

---

## 6. HyperAutoModel 类族

### 6.1 类层次

```python
# _transformers/auto_model.py

"""
Hyper-Parallel AutoModel 类族 — HuggingFace AutoModel* 的透明代理。

多重继承: _BaseHyperAutoModelClass + HF 的 AutoModelForCausalLM
用户可无感替换 `AutoModel` → `HyperAutoModel`。
"""

import torch
from contextlib import nullcontext
from typing import Optional, Union

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
)
from transformers.initialization import no_init_weights
from transformers.models.auto.auto_factory import _BaseAutoModelClass

from hyper_models.components.distributed.init_utils import get_world_size_safe
from hyper_models.components.utils.model_utils import init_empty_weights
from hyper_models.shared.utils import dtype_from_str


class _BaseHyperAutoModelClass:
    """from_pretrained / from_config 公共逻辑。"""

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *model_args,
        distributed_setup=None,
        backend=None,
        peft_config=None,
        torch_dtype="auto",
        attn_implementation="sdpa",
        force_hf: bool = False,
        validate_placement: bool = False,
        qat_config=None,            # QAT 训练配置
        fp8_config=None,            # FP8 训练配置
        compile_config=None,        # torch.compile 配置
        freeze_config=None,         # 参数冻结配置
        **kwargs,
    ) -> PreTrainedModel:
        """HF 兼容入口。

        内部流程:
            ① resolve_distributed_setup → MeshContext
            ② instantiate_infrastructure → ShardingPlanner + FSDP2Manager
            ③ AutoConfig.from_pretrained → hf_config
            ④ get_is_hf_model → 自定义/HF 路径
            ⑤ _build_model → meta device 构建 + ShardingPlanner + ShardingApplier
            ⑥ Checkpointer.load_base_model → 每 rank 独立加载权重
        """

    @classmethod
    def from_config(
        cls,
        config,
        *model_args,
        distributed_setup=None,
        device_mesh=None,
        backend=None,
        torch_dtype="auto",
        attn_implementation="sdpa",
        **kwargs,
    ) -> PreTrainedModel:
        """从 PretrainedConfig 构建模型（不加载权重，load_base_model=False）。"""

    @classmethod
    def _from_pretrained_parent_class(
        cls,
        pretrained_model_name_or_path: str,
        *model_args,
        config=None,
        torch_dtype="auto",
        attn_implementation="sdpa",
        **kwargs,
    ) -> PreTrainedModel:
        """委托给 HF 父类的 from_pretrained（is_hf_model=True 路径用）。

        HyperAutoModelForCausalLM 的 MRO 中第二个父类即 AutoModelForCausalLM，
        通过 super().__thisclass__.from_pretrained 调用 HF 原生实现，避免
        无限递归回 _BaseHyperAutoModelClass.from_pretrained。
        """
        # 取 MRO 中本类之后的第一个 HF AutoModel 父类（跳过 _BaseHyperAutoModelClass，
        # 否则 MRO[1] 取到 _Base 自身 → 回调 _Base.from_pretrained → 无限递归）
        hf_parent_cls = next(
            c for c in cls.__mro__[1:]
            if not issubclass(c, _BaseHyperAutoModelClass)
        )
        return hf_parent_cls.from_pretrained(
            pretrained_model_name_or_path,
            *model_args,
            config=config,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
            **kwargs,
        )

    @classmethod
    def _build_model(
        cls,
        pretrained_model_name_or_path,
        *model_args,
        is_hf_model,
        hf_config,
        mesh,
        sharding_planner,
        fsdp2_manager,
        autopipeline,
        backend,
        peft_config,
        torch_dtype,
        attn_implementation,
        validate_placement,
        load_base_model,
        distributed_setup=None,
        qat_config=None,
        fp8_config=None,
        compile_config=None,
        freeze_config=None,
        **kwargs,
    ) -> PreTrainedModel:
        """共享的模型构建逻辑 ── 详见 §8.3"""


class HyperAutoModelForCausalLM(_BaseHyperAutoModelClass, AutoModelForCausalLM):
    """Hyper-Parallel CausalLM ── 等价于 AutoModelForCausalLM。"""
    pass


class HyperAutoModelForImageTextToText(_BaseHyperAutoModelClass, AutoModelForImageTextToText):
    """Hyper-Parallel VLM ── 等价于 AutoModelForImageTextToText。"""
    pass


class HyperAutoModelForSequenceClassification(_BaseHyperAutoModelClass, AutoModelForSequenceClassification):
    """Hyper-Parallel SequenceClassification。"""
    pass
```

### 6.2 from_pretrained 完整实现

```python
@classmethod
def from_pretrained(
    cls,
    pretrained_model_name_or_path: str,
    *model_args,
    distributed_setup=None,
    backend=None,
    peft_config=None,
    torch_dtype="auto",
    attn_implementation="sdpa",
    force_hf: bool = False,
    validate_placement: bool = False,
    qat_config=None,            # QAT 训练配置
    fp8_config=None,            # FP8 训练配置
    compile_config=None,        # torch.compile 配置
    freeze_config=None,         # 参数冻结配置
    **kwargs,
) -> PreTrainedModel:
    """
    Args:
        pretrained_model_name_or_path: HF hub repo ID 或本地路径
        distributed_setup: DistributedSetup(topology, strategy, ...)
        backend: BackendConfig(attn="sdpa", linear="torch", ...)
        peft_config: LoRA 等 PEFT 配置
        torch_dtype: "auto" | "bfloat16" | "float16" | torch.dtype
        attn_implementation: "sdpa" | "flash_attention_2" | "eager"
        force_hf: True → 强制使用 HF 原生实现
        validate_placement: True → 校验模式（DTensor 传播），False → 生产模式
        qat_config: QAT 训练配置（透传给 _build_model / apply_model_infrastructure）
        fp8_config: FP8 训练配置（透传给 _build_model / apply_model_infrastructure）
        compile_config: torch.compile 配置（透传给 _build_model / apply_model_infrastructure）
        freeze_config: 参数冻结配置（透传给 _build_model / apply_model_infrastructure）
    """
    # ① 解析分布式 setup
    from hyper_models.components.distributed.config import DistributedSetup

    if distributed_setup is None:
        distributed_setup = DistributedSetup()
    mesh = distributed_setup.mesh_context
    torch_dtype = dtype_from_str(torch_dtype) if torch_dtype != "auto" else torch_dtype

    # ② 实例化基础设施
    from hyper_models._transformers.infrastructure import instantiate_infrastructure

    sharding_planner, fsdp2_manager, autopipeline = instantiate_infrastructure(
        distributed_setup=distributed_setup,
        device=torch.device("cuda", torch.cuda.current_device()),
    )

    # ③ 获取 HF config + 判断路径
    # get_hf_config / get_is_hf_model 均定义在 registry.py（§5），统一 import 来源
    from hyper_models._transformers.registry import get_hf_config, get_is_hf_model

    hf_config = get_hf_config(
        pretrained_model_name_or_path, attn_implementation, torch_dtype, **kwargs
    )
    is_hf_model = get_is_hf_model(hf_config, force_hf)

    # ④ 构建模型（含分布式 + 权重加载）
    return cls._build_model(
        pretrained_model_name_or_path,
        *model_args,
        is_hf_model=is_hf_model,
        hf_config=hf_config,
        mesh=mesh,
        sharding_planner=sharding_planner,
        fsdp2_manager=fsdp2_manager,
        autopipeline=autopipeline,
        backend=backend,
        peft_config=peft_config,
        torch_dtype=torch_dtype,
        attn_implementation=attn_implementation,
        validate_placement=validate_placement,
        load_base_model=True,
        distributed_setup=distributed_setup,
        qat_config=qat_config,
        fp8_config=fp8_config,
        compile_config=compile_config,
        freeze_config=freeze_config,
        **kwargs,
    )
```

### 6.3 _build_model 核心编排

```python
@classmethod
def _build_model(
    cls,
    pretrained_model_name_or_path,
    *model_args,
    is_hf_model,
    hf_config,
    mesh,
    sharding_planner,
    fsdp2_manager,
    autopipeline,
    backend,
    peft_config,
    torch_dtype,
    attn_implementation,
    validate_placement,
    load_base_model,
    distributed_setup=None,
    qat_config=None,
    fp8_config=None,
    compile_config=None,
    freeze_config=None,
    **kwargs,
) -> PreTrainedModel:
    """模型构建 + 分布式应用 + 权重加载的完整编排。"""

    # ── Step 1: 确定 meta device ──
    from hyper_models.components.distributed.init_utils import get_world_size_safe
    from transformers.modeling_utils import ContextManagers

    is_meta_device = (
        get_world_size_safe() > 1 or not is_hf_model
    ) and kwargs.get("quantization_config") is None

    init_ctx = (
        ContextManagers([no_init_weights(), init_empty_weights()])
        if is_meta_device
        else nullcontext()
    )

    # ── Step 2: 构建模型（meta device 或真设备） ──
    from hyper_models._transformers.model_init import _init_model

    with init_ctx:
        is_custom_model, model = _init_model(
            cls,
            pretrained_model_name_or_path,
            hf_config,
            attn_implementation,
            torch_dtype,
            is_hf_model,
            *model_args,
            backend=backend,
            **kwargs,
        )

    # ── Step 3-11: 委托给 apply_model_infrastructure()（详见 §8.3） ──
    # apply_model_infrastructure 位于 hyper_models._transformers.infrastructure，
    # 与本 class（auto_model.py）不同模块，需 local-import（第六轮 P1 修复）。
    from hyper_models._transformers.infrastructure import apply_model_infrastructure
    # canonical 执行顺序（PP 最先——裁决以 §8.2 为准；fully_shard 在 to_empty/load 之前）：
    #   Step 3:  PP 拆分（最先执行：stage 切分必须在权重加载与 FSDP2 包裹之前完成，
    #            否则每 rank 加载全模型权重，PP 失去意义）
    #   Step 4:  PEFT 注入（分片之前）
    #   Step 5:  QAT / FP8（分片之前）
    #   Step 6:  参数冻结（分片之前）
    #   Step 7:  ShardingPlanner.plan() → ShardingPlan
    #   Step 8:  apply_sharding_plan()（含 _local_params_context 解包 → tp_grad_info）
    #   Step 9:  torch.compile
    #   Step 10: FSDP2 包裹（meta 上 fully_shard，先于 to_empty/load）
    #   Step 11: meta→GPU（to_empty 物化 sharded 参数）+ load_base_model（写入本地份）
    #   （原 Step 12 "CP hooks" 已取消——D-01''：CP K/V all-gather 在 Step 8
    #    apply_sharding_plan 内编译期注入，无运行时 hooks，见 §8.3 ⑩）
    #
    model = apply_model_infrastructure(
        model,
        mesh=mesh,
        sharding_planner=sharding_planner,
        fsdp2_manager=fsdp2_manager,
        autopipeline=autopipeline,
        peft_config=peft_config,
        qat_config=qat_config,
        fp8_config=fp8_config,
        freeze_config=freeze_config,
        compile_config=compile_config,
        is_meta_device=is_meta_device,
        is_hf_model=is_hf_model,
        device=torch.device("cuda", torch.cuda.current_device()),
        load_base_model=load_base_model,
        pretrained_path=pretrained_model_name_or_path,
        validate_placement=validate_placement,
    )

    model.train()
    return model
```

**HF 原生路径的权重加载职责澄清（`is_meta_device` 与"双重加载"疑问）**：

`world_size > 1` 时即使 `is_hf_model=True`，`is_meta_device` 也为 True（Step 1）。
此时 `_init_model` 路径 A 的 `_from_pretrained_parent_class` 是在
`no_init_weights() + init_empty_weights()` 上下文内执行的——HF `from_pretrained`
的权重填充被 meta context 短路，产出的仍是 **meta 参数空壳**，并未完成真实权重加载。
因此不存在"双重加载"：

- **meta 路径（`world_size > 1`，或任意 `world_size` 的自定义模型）**：权重加载职责
  **唯一归属于** Step ⑧ 的 `load_base_model()`（每 rank 按 TP/DP 读本地份写入，零 NCCL）。
  `_from_pretrained_parent_class` 在此路径下只负责构建模型结构。`to_empty()` 是
  **必要的**——它把 FSDP2 已在 meta 上包裹的 sharded 参数物化到真实设备，
  `load_base_model` 才有真实存储可写。
- **单卡 HF 原生路径（`world_size == 1` 且 `is_hf_model=True`）**：`is_meta_device=False`，
  `_from_pretrained_parent_class` 在真设备上**直接完成完整权重加载**；Step ⑧ 走
  `elif not is_meta_device` 分支，仅 `model.to(device)`，不再调用 `load_base_model`，
  也不需要 `to_empty()`（参数本就在真设备上）。

### 6.4 PEFT 注入（`_apply_peft`）

PEFT（LoRA/DoRA 等）必须在 DTensor 分片**之前**注入模型。原因是 LoRA 层需要先插入到目标 `nn.Linear` 等模块中，然后 ShardingPlanner 才能对这些新插入的层进行分片规划。

```python
# hyper_models/components/_peft/lora.py

def _apply_peft(model: nn.Module, peft_config: PeftConfig) -> nn.Module:
    """在模型上注入 LoRA 层。

    关键点：
    1. 必须在 ShardingPlanner.plan() 之前调用
    2. LoRA 层插入后，ShardingPlanner 自动处理它们的 placement
    """
    from peft import get_peft_model, LoraConfig

    if isinstance(peft_config, dict):
        peft_config = LoraConfig(**peft_config)

    model = get_peft_model(model, peft_config)

    # 标记：后续 ShardingPlanner 会检测到 lora_A/lora_B 参数
    # 并根据命名规则（colwise/rowwise）自动分配 placement
    return model


def _freeze_non_lora_params(model: nn.Module):
    """冻结所有非 LoRA 参数（在 FSDP2 包裹之后调用）。

    LoRA 训练时只有 lora_A 和 lora_B 需要梯度。
    """
    for name, param in model.named_parameters():
        if "lora_" not in name:
            param.requires_grad = False
```

**关键时序**：PEFT 注入在分片**之前**（§6.3 Step 4 / §8.3 ②），非 LoRA 冻结在 FSDP2 包裹 + 权重加载**之后**（§8.3 ⑨.5，对应 §4 时序树 ④.4.5.12）——因为冻结操作需要遍历已包裹、已物化的参数。

### 6.5 参数冻结（`_apply_parameter_freezing`）

支持通过 `freeze_config` YAML 配置精细控制哪些参数参与训练：

```python
# hyper_models/components/training/freeze.py

@dataclass
class FreezeConfig:
    """参数冻结配置。"""
    freeze_embed: bool = False
    freeze_norm: bool = False
    freeze_lm_head: bool = False
    unfreeze_patterns: list[str] | None = None   # 白名单
    freeze_patterns: list[str] | None = None     # 黑名单


def _apply_parameter_freezing(model: nn.Module, config: FreezeConfig):
    """在 ShardingPlanner.plan() 之前应用参数冻结。

    冻结的参数 requires_grad=False → 被 ShardingPlanner 跳过。
    """
    for name, param in model.named_parameters():
        if config.freeze_embed and "embed" in name:
            param.requires_grad = False
        elif config.freeze_norm and ("norm" in name or "rmsnorm" in name):
            param.requires_grad = False
        elif config.freeze_lm_head and "lm_head" in name:
            param.requires_grad = False

        if config.unfreeze_patterns:
            for pattern in config.unfreeze_patterns:
                if re.match(pattern, name):
                    param.requires_grad = True
                    break
        if config.freeze_patterns:
            for pattern in config.freeze_patterns:
                if re.match(pattern, name):
                    param.requires_grad = False
                    break
```

```yaml
# YAML 配置示例
freeze_config:
  freeze_embed: true
  unfreeze_patterns:
    - ".*task_head.*"   # 只训练 task head
```

### 6.6 QAT / FP8 量化注入（`_apply_qat` / `_apply_fp8`）

```python
# hyper_models/components/training/quantization.py

def _apply_qat(model: nn.Module, qat_config: "QATConfig | dict") -> None:
    """在 ShardingPlanner.plan 之前注入 QAT（量化感知训练）。

    将指定 nn.Linear / nn.LayerNorm 替换为 torchao 量化版本，使 ShardingPlanner
    能对量化后的模块做 placement 推导。in-place 修改 model。

    Args:
        qat_config: QATConfig（dtype="int8"/"fp8"、per-axis、observer 等）或 dict。
    """
    if isinstance(qat_config, dict):
        qat_config = QATConfig(**qat_config)

    from torchao.quantization import quantize_, Int8WeightOnlyConfig, Int8DynamicActivationInt4WeightConfig

    dtype = getattr(qat_config, "dtype", "int8")
    group_size = getattr(qat_config, "group_size", 128)

    if dtype == "int8":
        quant_config = Int8WeightOnlyConfig(group_size=group_size)
    elif dtype == "int4":
        quant_config = Int8DynamicActivationInt4WeightConfig()
    else:
        raise ValueError(f"Unsupported QAT dtype: {dtype}")

    quantize_(model, quant_config)


def _apply_fp8(model: nn.Module, fp8_config: "FP8Config | dict") -> None:
    """在 ShardingPlanner.plan 之前注入 FP8 训练（float8 via torchao）。

    与 _apply_qat 互斥（通常二选一）。in-place 修改 model。

    Args:
        fp8_config: FP8Config（src/dst dtype、filter_fn 等）或 dict。
    """
    if isinstance(fp8_config, dict):
        fp8_config = FP8Config(**fp8_config)

    from torchao.float8 import convert_to_float8_training, Float8LinearConfig

    config = Float8LinearConfig(
        cast_forward_inputs=getattr(fp8_config, "cast_forward_inputs", True),
        cast_forward_weight=getattr(fp8_config, "cast_forward_weight", True),
    )
    convert_to_float8_training(model, config=config)
```

**关键时序**：QAT/FP8 在分片**之前**（§8.3 ③），与 PEFT/参数冻结同阶段——量化后的模块结构改变必须在 ShardingPlanner 看到模型之前完成。

---

## 7. _init_model：自定义 vs HF 路径分发

```python
# _transformers/model_init.py

def _init_model(
    cls,                          # HyperAutoModelForCausalLM 等
    pretrained_model_name_or_path: str,
    hf_config,                    # AutoConfig.from_pretrained() 的结果
    attn_implementation: str,
    torch_dtype,
    is_hf_model: bool,
    *model_args,
    backend=None,                 # ★ keyword-only：位于 *model_args 之后，避免被位置参抢占
    **kwargs,
) -> tuple[bool, PreTrainedModel]:
    """
    Returns:
        (is_custom_model, model)
        is_custom_model=True  → Hyper-Parallel 自定义实现
        is_custom_model=False → HF 原生实现
    """

    architectures = getattr(hf_config, "architectures", []) or []
    arch_name = architectures[0] if architectures else ""

    # ── 路径 A: HF 原生 ──
    if is_hf_model:
        # 使用 HF AutoModelForCausalLM.from_pretrained 加载
        # 在真设备上构建 + 加载权重（如果 from_pretrained 调用）
        model = cls._from_pretrained_parent_class(
            pretrained_model_name_or_path,
            *model_args,
            config=hf_config,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
            **kwargs,
        )

        # 注入 HFCheckpointingMixin（使得 save_pretrained 走 DCP）
        from hyper_models.components.models.common.hf_checkpointing_mixin import HFCheckpointingMixin

        model.__class__ = type(
            f"Hyper{model.__class__.__name__}",
            (HFCheckpointingMixin, model.__class__),
            {},
        )

        return False, model

    # ── 路径 B: 自定义模型（meta device 空壳） ──
    from hyper_models._transformers.registry import _resolve_custom_model_cls

    model_cls = _resolve_custom_model_cls(arch_name)
    if model_cls is None:
        raise ValueError(f"No custom model for {arch_name} and force_hf=False")

    # meta device 构建空壳（零显存）
    # 参数为 meta tensor，等后续 to_empty() + 权重加载
    with torch.device("meta"):
        model = model_cls(
            hf_config,
            *model_args,
            backend=backend,
            **kwargs,
        )

    return True, model
```

---

## 8. instantiate_infrastructure / apply_model_infrastructure

### 8.1 两阶段分离

```python
# _transformers/infrastructure.py

def instantiate_infrastructure(
    *,
    distributed_setup,
    device: torch.device,
) -> tuple:
    """Phase 1: 配置 → 运行时对象（只创建，不应用）。

    Returns:
        (sharding_planner, fsdp2_manager, autopipeline)
    """
    from hyper_models.components.distributed.sharding_planner import ShardingPlanner
    from hyper_models.components.distributed.fsdp2 import FSDP2Manager
    # 辅助工厂定义在 hyper_models/components/distributed/fsdp2.py 与 pipelining.py（§2.14）
    from hyper_models.components.distributed.fsdp2 import _instantiate_fsdp2
    from hyper_models.components.distributed.pipelining import _instantiate_pipeline

    # ShardingPlanner 总是创建（即使 is_hf_model=True，plan() 不会被调用）
    sharding_planner = ShardingPlanner()

    # FSDP2Manager
    mesh = distributed_setup.mesh_context
    # canonical 形参名：mesh_context（以 01 为准，06 §4.1 需向此对齐）
    fsdp2_manager = _instantiate_fsdp2(config=distributed_setup.strategy_config, mesh_context=mesh) if distributed_setup.strategy_config else None

    # AutoPipeline（如果 pp > 1；canonical 2 参签名 (pipeline_config, mesh)，见 §8.2）
    autopipeline = _instantiate_pipeline(distributed_setup.pipeline_config, mesh)

    return sharding_planner, fsdp2_manager, autopipeline
```

### 8.2 AutoPipeline：Pipeline Parallelism

当 `pp_size > 1` 时，AutoPipeline 将模型切分为多个 stage，每个 stage 在独立的 GPU 组上执行。AutoPipeline 在 `apply_model_infrastructure()` 中**最先执行**（在 ShardingPlan 和 FSDP2 之前），因为 PP 切分改变了模型的物理结构。

> **裁决说明**：PP 的执行位置以本节为准——PP stage 切分必须在权重加载
> （`load_base_model`）与 FSDP2 包裹**之前**完成，否则每个 rank 都会加载全模型
> 权重，PP 失去意义。§8.3 与 §6.3 的顺序注释已对齐为本顺序。

```python
# hyper_models/components/distributed/pipelining.py

class AutoPipeline:
    """Pipeline Parallelism 管理器。

    在 instantiate_infrastructure() 中创建，在 apply_model_infrastructure() 中应用。
    """

    def __init__(self, pipeline_config: PipelineConfig, mesh: MeshContext):
        self.config = pipeline_config
        self.mesh = mesh

    def build(self, model: nn.Module, loss_fn: nn.Module | None = None) -> None:
        """将模型切分为 PP stages（in-place 注册到 model.parts）。

        切分策略：
        - 按 transformer layers 均分
        - embed 层在 stage 0
        - lm_head 层在 stage -1
        - 插入 send/recv 通信节点

        注意：不返回 self。调用方保持 model 引用不变（仍为原 nn.Module），
        通过 model.parts 暴露各 stage 子模块，避免 `model = autopipeline.build(model)`
        后 `model.train()` 误调到 AutoPipeline 实例上。
        """
        num_layers = _get_num_transformer_layers(model)
        layers_per_stage = num_layers // self.mesh.pp_size

        stages = []
        for rank in range(self.mesh.pp_size):
            stage = _extract_stage(model, rank * layers_per_stage, (rank+1) * layers_per_stage)
            # 包装为 PipelineStage（插入通信 + 调度逻辑）
            stages.append(PipelineStage(stage, rank, self.mesh.pp_size))

        # in-place 挂载：model.parts → 每个 stage 一个独立 nn.Module
        model.parts = stages
        self.parts = stages


@dataclass
class PipelineConfig:
    """PP 配置。"""
    pp_batch_size: int = 1          # PP microbatch 数量
    pp_microbatch_size: int = 1     # 每个 microbatch 的大小
    loss_fn: nn.Module | None = None
```

### 8.3 apply_model_infrastructure（由 _build_model 调用）

```python
def apply_model_infrastructure(
    model,
    *,
    mesh,
    sharding_planner,
    fsdp2_manager,
    autopipeline,
    peft_config,
    qat_config=None,
    fp8_config=None,
    freeze_config=None,
    compile_config=None,
    is_meta_device,
    is_hf_model,
    device,
    load_base_model,
    pretrained_path,
    validate_placement,
    **kwargs,
):
    """Phase 2: 运行时对象 → 应用到模型。

    规范化执行顺序（canonical：PP 最先——以 §8.2 裁决为准；meta 链路，fully_shard 在 to_empty/load 之前）：
        PP 拆分（最先执行） → PEFT → freeze → plan → apply_sharding_plan(含 _local_params_context 解包)
              → build_tp_grad_info → torch.compile → fully_shard(meta 上包裹 FSDP2)
              → to_empty(meta→GPU 物化 sharded 参数) → load_base_model(写入本地份)
              （原末尾"CP hooks"步骤已取消——D-01''：CP 通信在 apply_sharding_plan
               内编译期注入，见本节 ⑩）

    裁决说明：PP 执行位置曾与 §6.3 / 本节旧版注释不一致（旧版置于 FSDP2/权重加载
    之后）。裁定以 §8.2 为准——AutoPipeline 在 apply_model_infrastructure() 中
    最先执行：PP 切分 stage 必须在权重加载与 FSDP2 包裹之前完成，否则每个 rank
    都会加载全模型权重，PP 失去意义。

    与旧 load→FSDP2 顺序的关键差异：FSDP2 在 meta 上包裹，物化即得到 sharded 参数，
    再由 load_base_model 按 TP/DP 读本地份写入——避免先 load 全量再 shard 的二次显存峰值。
    """
    # 本函数位于 hyper_models._transformers.infrastructure；下列 _apply_* 跨模块，
    # 需 local-import（第六轮 P1 修复：补全调用点 import，避免 NameError）。
    from hyper_models.components._peft.lora import _apply_peft
    from hyper_models.components.training.quantization import _apply_qat, _apply_fp8
    from hyper_models.components.training.freeze import _apply_parameter_freezing

    # ① PP 拆分（最先执行——裁决以 §8.2 为准：先于 PEFT/分片/FSDP2/权重加载）
    #    build() in-place 注册 model.parts，不返回值——model 仍为原 nn.Module
    if autopipeline is not None:
        autopipeline.build(model)

    # ② PEFT 注入（在参数分片之前）
    if peft_config is not None:
        model = _apply_peft(model, peft_config)

    # ③ QAT / FP8（在参数分片之前）
    if qat_config is not None:
        _apply_qat(model, qat_config)

    if fp8_config is not None:
        _apply_fp8(model, fp8_config)

    # ④ 参数冻结（在分片之前）
    if freeze_config is not None:
        _apply_parameter_freezing(model, freeze_config)

    # ⑤ ShardingPlanner.plan() → ShardingPlan（补 sequence_parallel / loss_parallel）
    plan = None
    if not is_hf_model and sharding_planner is not None:
        from hyper_models.components.distributed.sharding_planner import ShardingPlanner

        plan = sharding_planner.plan(
            model,
            mesh.device_mesh,
            tp_size=mesh.tp_size,
            cp_size=mesh.cp_size,
            ep_size=mesh.ep_size,
            sequence_parallel=getattr(mesh, "sequence_parallel", False),
            loss_parallel=getattr(mesh, "loss_parallel", False),
        )

    # ⑥ apply_sharding_plan（DTensor 分片应用 + _local_params_context 一次性解包）
    #    生产模式：返回 (model, tp_grad_info)，tp_grad_info 由 ShardingPlan 导出
    tp_grad_info = None
    if not is_hf_model and plan is not None:
        from hyper_models.components.distributed.sharding_applier import apply_sharding_plan

        model, tp_grad_info = apply_sharding_plan(
            model, plan, mesh.device_mesh,
            validate_mode=validate_placement,
        )

    # ⑦ torch.compile（在 fully_shard 之前；Inductor 在 meta/空参数上追踪计算图）
    if compile_config is not None:
        model = torch.compile(model, **compile_config)

    # ⑧ FSDP2 包裹（canonical：在 meta 上 fully_shard，先于 to_empty/load_base_model）
    #    物化 sharded 参数由 to_empty 完成，load_base_model 再写入本地份。
    if fsdp2_manager is not None:
        model = fsdp2_manager.parallelize(model, tp_grad_info=tp_grad_info)

    # ⑨ meta → GPU（物化 sharded 参数）+ 权重加载（5 参 canonical 签名）
    if is_meta_device and load_base_model:
        # meta → 真实设备（FSDP2 已在 meta 上包裹，物化即得到 sharded 参数）
        model.to_empty(device=device)

        from hyper_models.components.checkpoint.loading import load_base_model
        from hyper_models.components.checkpoint.checkpointing import _get_state_dict_adapter

        load_base_model(
            model,
            device,
            pretrained_path,
            adapter=_get_state_dict_adapter(model),
            mesh=mesh.device_mesh,   # ★ 04 §5.3 期待 DeviceMesh，传 mesh.device_mesh
        )
    elif not is_meta_device and load_base_model:
        model = model.to(device=device)

    # ⑨.5 PEFT 非 LoRA 参数冻结（在 FSDP2 + 权重加载之后；与 §6.4 / §4 ④.4.5.12 一致）
    #    原因：冻结操作需要遍历已 FSDP2 包裹、已物化的参数。
    if peft_config is not None:
        from hyper_models.components._peft.lora import _freeze_non_lora_params

        _freeze_non_lora_params(model)

    # ⑩ CP hooks —— 已取消（D-01''）：CP 的 K/V all-gather 由 ⑥
    #    apply_sharding_plan 在编译期注入 inner attention wrapper
    #    （_wrap_cp_inner_attention → flex_cp_allgather，05 §4.4.2），
    #    无运行时 hooks 步骤；早期草案的 attach_context_parallel_hooks
    #    不存在于代码（03 §7.1 同注）。

    return model
```

---

## 9. ShardingPlanner：hyper_parallel 核心差异化架构

### 9.1 与 AutoModel 的分片机制对比

AutoModel 的模型分片**没有统一的中间表示层**。三种并行策略各自独立处理：

```python
# AutoModel infrastructure.py — 三条独立的并行化路径，无统一规划

# 路径 A: Pipeline Parallelism
if autopipeline is not None:
    _shard_pp(autopipeline, model, loss_fn, parallelize_fn)

# 路径 B: Expert Parallel + FSDP (MoE 模型)
else:
    _shard_ep_fsdp(model, model_wrapper, parallelize_fn, mesh)

# 路径 C: FSDP/DDP only (dense 模型)
# → model_wrapper.parallelize(model) 隐式处理
```

每条路径的 `parallelize_fn` 是针对特定模型的**硬编码函数**（~400 行/模型），包含大量 `if-elif` 分支和手动 placement 声明。

**hyper_parallel 的 ShardingPlanner 解决的核心痛点**：

| 痛点 | AutoModel 现状 | ShardingPlanner 方案 |
|------|--------------|-------------------|
| 模型接入成本 | 每模型 ~400 行 `parallelize.py` | ~20 行 `ARCH_OVERRIDES` + `ShardingTemplate` |
| 分片策略可验证性 | 运行时报错，无法提前校验 | validate 模式：DTensor 传播 → 编译期断言 |
| 策略复用 | 模型间大量重复代码 | `ParamRole` + `ShardingTemplate` 声明式复用 |
| 策略可观测性 | 运行时隐式状态 | `ShardingPlan` / `ModuleShardingSpec` 可序列化中间表示 |

### 9.2 ShardingPlanner 作用域

ShardingPlanner **只覆盖 TP/CP/SP 维度的 DTensor 分片**。PP 和 FSDP2 由专门的组件独立处理：

```
┌─────────────────────────────────────────────────────────┐
│                   apply_model_infrastructure              │
│                                                          │
│  ┌──────────────┐  ┌──────────────────┐  ┌────────────┐ │
│  │ AutoPipeline │  │ ShardingPlanner  │  │ FSDP2Manager│ │
│  │ (PP)         │  │ (TP/CP/SP)       │  │ (DP)        │ │
│  └──────────────┘  └──────────────────┘  └────────────┘ │
│        ↓                  ↓                    ↓         │
│  模型切分+调度     DTensor 分片        FSDP2 包裹     │
└─────────────────────────────────────────────────────────┘
```

这种分层设计与 AutoModel 的三条独立路径构成**替代关系**——ShardingPlanner **取代**了 AutoModel 的 `parallelize_fn`，在此之上增加了一层统一的声明式抽象。

### 9.3 ShardingPlan 的中间表示

**ShardingPlan / ModuleShardingSpec 的 canonical 数据模型定义在 05 §3.1 / §3.2**（字段为 `modules` / `special_handlers` / `tied_pairs` / `is_terminal` 等）。01 不再独立重定义，避免与 05 canonical 产生字段竞争（曾出现的 `module_specs` / `_is_terminal` 与 05 `modules` / `is_terminal` 不兼容问题已通过引用 05 消除）。

01 仅保留以下序列化 / 可观测性补充说明：

- **可序列化**：ShardingPlan 是纯 dataclass + NamedPlacement 枚举，可用 `pickle` / DCP metadata 落盘，供 validate 模式离线复算与跨 rank 一致性校验。
- **可观测性**：`ShardingPlan.to_report()`（05 §3.3）输出人类可读的分片报告，便于调试 `parallelize_fn` 替代效果。

与 DCP 的 DTensor metadata 交互：ShardingPlan 在**编译期**推导 placements，DCP 在**保存时**记录实际的 DTensor 元数据。两者独立但互补——ShardingPlan 是"预期"，DCP metadata 是"实际"。

### 9.4 为什么 AutoModel 不需要 ShardingPlanner

AutoModel 面向的是 **NVIDIA 内部模型生态**——模型数量有限（~20 个），每个模型的 `parallelize_fn` 由专家编写和维护。对于这个规模，硬编码的 parallelize 函数比维护一个通用的 ShardingPlanner 更务实。

hyper_parallel 面向的是 **HF 生态的任意模型**——可能有数千个模型架构。对于这个规模，ShardingPlanner 的自动化推导是不可或缺的。

---

## 10. StateDictAdapter + 权重加载

### 10.1 StateDictAdapter 基类

```python
# hyper_models/components/models/common/state_dict_adapter.py

from abc import ABC, abstractmethod


class StateDictAdapter(ABC):
    """HF checkpoint ↔ 模型内部参数的透明转换。

    子类实现两个方法：
    - from_hf(): 加载时，HF checkpoint key → 模型内部 FQN
    - to_hf():   保存时，模型内部 FQN → HF checkpoint key
    """

    @abstractmethod
    def from_hf(self, hf_state_dict: dict, device_mesh=None, **kwargs) -> dict:
        """HF checkpoint → 模型内部 state dict。"""
        ...

    @abstractmethod
    def to_hf(self, state_dict: dict, **kwargs) -> dict:
        """模型内部 state dict → HF checkpoint。"""
        ...
```

### 10.2 Qwen3.5 示例

```python
# hyper_models/components/models/qwen3_5/state_dict_adapter.py

class Qwen3_5DenseStateDictAdapter(StateDictAdapter):
    """Qwen3.5-Dense 的 key 映射。

    差异来源：
    1. _fp32_params 包装：HF checkpoint 的 linear_attn.A_log
       → 模型内部的 linear_attn._fp32_params.A_log
    2. MTP 重命名：HF 的 mtp.fc.weight
       → NeMo 的 mtp.layers.0.eh_proj.weight
    """

    _FP32_PARAMS_KEYS = {"A_log", "dt_bias"}

    def from_hf(self, hf_state_dict: dict, device_mesh=None, **kwargs) -> dict:
        """加载：HF checkpoint → 模型内部。"""
        result = {}
        for hf_key, tensor in hf_state_dict.items():
            # ── _fp32_params 包装 ──
            for fp32_key in self._FP32_PARAMS_KEYS:
                if f".{fp32_key}" in hf_key:
                    internal_key = hf_key.replace(
                        f".{fp32_key}", f"._fp32_params.{fp32_key}"
                    )
                    result[internal_key] = tensor
                    break
            else:
                # ── MTP 重命名 ──
                if "mtp.fc" in hf_key:
                    internal_key = hf_key.replace("mtp.fc", "mtp.layers.0.eh_proj")
                elif "mtp.norm" in hf_key:
                    internal_key = hf_key.replace(
                        "mtp.norm", "mtp.layers.0.final_layernorm"
                    )
                else:
                    internal_key = hf_key
                result[internal_key] = tensor

        return result

    def to_hf(self, state_dict: dict, **kwargs) -> dict:
        """保存：模型内部 → HF checkpoint。"""
        result = {}
        for internal_key, tensor in state_dict.items():
            # ── _fp32_params 解包装 ──
            if "._fp32_params." in internal_key:
                hf_key = internal_key.replace("._fp32_params.", ".")
            # ── MTP 反向重命名 ──
            elif "mtp.layers.0.eh_proj" in internal_key:
                hf_key = internal_key.replace(
                    "mtp.layers.0.eh_proj", "mtp.fc"
                )
            elif "mtp.layers.0.final_layernorm" in internal_key:
                hf_key = internal_key.replace(
                    "mtp.layers.0.final_layernorm", "mtp.norm"
                )
            else:
                hf_key = internal_key
            result[hf_key] = tensor

        return result
```

### 10.3 每 rank 独立加载权重

`load_base_model` 的 canonical 实现位于 **04_checkpoint.md §5.3**（`hyper_models/components/checkpoint/loading.py`），
01 不再重复定义，仅导入引用并保留签名说明，避免与 04 产生双定义竞争。

```python
# hyper_models/components/checkpoint/checkpointing.py
# 01 仅导入 canonical 实现，不在此文件重新定义 load_base_model。
from hyper_models.components.checkpoint.loading import load_base_model
# 签名（见 04 §5.3）：
#   def load_base_model(
#       model: nn.Module,
#       device: torch.device,
#       path: str,
#       adapter: StateDictAdapter | None = None,
#       mesh: DeviceMesh | None = None,   # ★ DeviceMesh（非 MeshContext），与 04 §5.3 对齐
#   ) -> None
# 语义：每 rank 独立读磁盘 + 独立切 DTensor（零 NCCL），
#       含 _reinit_non_persistent_buffers / ensure_tied_lm_head（04 版）。
#


# _load_hf_checkpoint_preserving_dtype 的 canonical 实现位于 04 §5.3
# （hyper_models/components/checkpoint/loading.py），01 不再重复定义，统一 import 引用：
#   from hyper_models.components.checkpoint.loading import _load_hf_checkpoint_preserving_dtype
# 签名（见 04 §5.3）：
#   def _load_hf_checkpoint_preserving_dtype(model_path: str) -> dict
# 语义：所有 rank 并行读 safetensors（含 index 多 shard 合并），返回全量 CPU state dict。
```

---

## 11. HFCheckpointingMixin

```python
# hyper_models/components/models/common/hf_checkpointing_mixin.py

class HFCheckpointingMixin:
    """提供 HF 兼容的 save_pretrained() 和 load_pretrained()。

    内部使用 Hyper-Parallel 的 Checkpointer（DCP + 异步 + StateDictAdapter）。
    不覆写 state_dict() / load_state_dict()——PyTorch DCP 需要标准 nn.Module 行为。
    """

    _state_dict_adapter: "StateDictAdapter | None" = None

    def save_pretrained(
        self,
        save_directory: str,
        checkpointer: "Checkpointer",
        tokenizer=None,
        **kwargs,
    ) -> None:
        """保存模型为 DCP + 可选 consolidated HF safetensors。"""
        checkpointer.save_model(
            model=self,
            weights_path=save_directory,
            tokenizer=tokenizer,
            **kwargs,
        )

    def load_pretrained(
        self,
        pretrained_model_name_or_path: str,
        checkpointer: "Checkpointer",
        **kwargs,
    ) -> None:
        """加载预训练权重。"""
        checkpointer.load_model(
            model=self,
            model_path=pretrained_model_name_or_path,
            is_init_step=True,
            **kwargs,
        )
```

---

## 12. 新模型上线流程（从旧 ModelSpec 迁移）

### 旧方式

```python
# hyper_parallel/models/qwen3_5/__init__.py（旧）
from hyper_parallel.models.spec import register_spec, ModelSpec

register_spec("qwen3_5", ModelSpec(
    name="qwen3_5",
    build_model_fn=lambda cfg: Qwen3_5ForCausalLM(_build_config(cfg)),
    parallelize_fn=parallelize_qwen3_5,
    pipelining_fn=pipeline_qwen3_5_for_trainer,
    state_dict_adapter=Qwen3_5StateDictAdapter,
    tp_load_transform_fn=qwen3_5_tp_load_transforms,
))
# ↑ 需要删除
```

### 新方式

```python
# 1. registry.py 中添加一行
MODEL_ARCH_MAPPING["Qwen3_5ForCausalLM"] = (
    "hyper_models.components.models.qwen3_5.model",
    "Qwen3_5ForCausalLM",
)

# 2. 如果命名非标准，注册 ARCH_OVERRIDES
# hyper_models/components/distributed/sharding_planner.py
ARCH_OVERRIDES["Qwen3_5ForCausalLM"] = [
    (r"linear_attn\.in_proj_qkv\.weight$", ParamRole.SPECIAL),
    (r"linear_attn\.conv1d\.weight$",       ParamRole.SPECIAL),
    (r"linear_attn\.A_log$",                ParamRole.SPECIAL),
    (r"linear_attn\.in_proj_z\.weight$",    ParamRole.COLWISE),
    (r"linear_attn\.out_proj\.weight$",     ParamRole.ROWWISE),
]

# 3. 如果有特殊分片逻辑，注册 SpecialHandler
SPECIAL_HANDLERS["gated_delta_tp_shard"] = _shard_gated_delta_local_params

# 4. 注册 StateDictAdapter
# hyper_models/components/models/qwen3_5/model.py
class Qwen3_5ForCausalLM(HFCheckpointingMixin, HFQwen3_5ForCausalLM):
    _state_dict_adapter = Qwen3_5DenseStateDictAdapter()
    ...
```
