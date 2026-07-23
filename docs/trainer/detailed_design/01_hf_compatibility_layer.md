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

## 2. 强类型配置解析系统：从 YAML 到 TrainerConfig

> **实现状态**：已实现于 commit `78a79c0f`。
> 与原始设计的关键差异：去除了 `ConfigNode` 弱类型中间容器，统一为强类型 dataclass 解析。
>
> 原始设计中的 `ConfigNode` / `_wrap()` / `translate_value()` / `instantiate()` / `RecipeConfig` 等概念
> **未在代码中实现**，实际采用了直接构造强类型 `TrainerConfig` 的方案。以下描述的是实际实现。

### 2.1 总体处理流程

配置系统的入口为 `parse_training_args()`，按以下顺序处理：

```
命令行参数
    │
    ▼
① parse_training_args()
    │  argparse → 分离 config_file 和 --field=value overrides
    ▼
② yaml.safe_load(path)
    │  → 原始 dict
    ▼
③ resolve_root(raw)
    │  校验 TrainerConfig 一级字段（拒绝未知字段）
    │  对每个一级分组调用 resolve_component()
    │    ├─ 读取 _target_ → import_target() 解析为 callable
    │    ├─ 校验 target 类型与 TrainerConfig 字段类型是否匹配
    │    ├─ 校验 target 参数签名与返回类型
    │    └─ 通过 coerce_value() 做 typed 参数转换
    │  构造 TrainerConfig(**resolved)
    ▼
④ _apply_typed_overrides(config, overrides)
    │  解析 --field=value → yaml.safe_load(value)
    │  通过 _replace_path() 做 typed dotted 路径替换
    │  拒绝未知字段、未选择组件和错误类型
    ▼
TrainerConfig (完全类型化的 dataclass 实例)
```

**核心设计决策**：

| 原始设计（ConfigNode） | 实际实现（强类型） |
|---|---|
| ConfigNode 弱类型容器，接受任意 key | `TrainerConfig` 固定 9 个 dataclass 字段，拒绝未知字段 |
| `_target_` 解析后存储在 ConfigNode 上，延迟到 `instantiate()` 才创建对象 | `_target_` 解析后**立即调用** target 构造类型化 Config 对象 |
| 需要 `RecipeConfig` 做弱类型→强类型桥接 | 不需要桥接层——解析结果直接就是强类型 `TrainerConfig` |
| `_wrap()` / `translate_value()` 做标量转换 | `coerce_value()` 基于 type annotation 做 typed 校验与转换 |
| CLI override 通过 `replace()` 修改 ConfigNode | CLI override 通过 `_replace_path()` 对 dataclass 做 typed 路径替换 |

---

### 2.2 对外接口：`parse_training_args()`

```python
# hyper_models/config/manager.py

def parse_training_args(argv: Sequence[str] | None = None) -> TrainerConfig:
    """解析训练命令行参数，返回强类型 TrainerConfig。

    预期命令行形式：
        train.yaml --accelerator.tp_size=4 --optimizer.lr=0.0003

    Args:
        argv: 显式参数 token 列表（用于测试）。默认从 sys.argv 读取。

    Returns:
        完全解析、类型校验通过的 TrainerConfig 实例。
    """
    parser = argparse.ArgumentParser(description="HyperParallel training config")
    parser.add_argument("config_file", help="Path to the YAML training config")
    args, overrides = parser.parse_known_args(argv)
    return _load_training_config(args.config_file, overrides)
```

使用示例：

```python
from hyper_models.config.manager import parse_training_args

config = parse_training_args()
# config.model         → ModelConfig(name="qwen3_5", ...)
# config.optimizer     → AdamW.Config(lr=0.0001, weight_decay=0.01, ...)
# config.training      → TrainingConfig(max_steps=100, ...)
# config.accelerator   → AcceleratorConfig(tp_size=2, dp_shard_size=4)
```

CLI dotted override 统一使用 `--field=value`：

```text
--training.max_steps=200
--accelerator.tp_size=4
--optimizer.lr=0.0003
```

完整训练命令形式：

```bash
torchrun --nproc_per_node=8 scripts/train_lm.py \
  configs/qwen3_5.yaml \
  --training.max_steps=200 \
  --accelerator.tp_size=4 \
  --optimizer.lr=0.0003
```

---

### 2.3 TrainerConfig：强类型配置树

```python
# hyper_models/trainer/config.py

@dataclass
class TrainerConfig:
    """Resolved component tree; runtime objects are built by the task trainer."""

    model: ModelConfig                                          # 必填，来自 hyper_parallel.trainer.config.ModelConfig
    optimizer: Optional[Optimizer.Config] = None                # 可选组件
    lr_scheduler: Optional[LRScheduler.Config] = None           # 可选组件
    loss: Optional[Loss.Config] = None                          # 可选组件
    training: TrainingConfig = field(default_factory=TrainingConfig)
    accelerator: AcceleratorConfig = field(default_factory=AcceleratorConfig)
    mixed_precision: MixedPrecisionConfig = field(default_factory=MixedPrecisionConfig)
    gradient_checkpointing: GradientCheckpointingConfig = field(default_factory=GradientCheckpointingConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)
```

9 个一级字段：

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `model` | `ModelConfig` | **是** | 模型配置，复用 `hyper_parallel.trainer.config.ModelConfig` |
| `optimizer` | `Optional[Optimizer.Config]` | 否 | 优化器类别 Config |
| `lr_scheduler` | `Optional[LRScheduler.Config]` | 否 | 学习率调度器类别 Config |
| `loss` | `Optional[Loss.Config]` | 否 | Loss 类别 Config |
| `training` | `TrainingConfig` | 否（有默认值） | 训练循环参数 |
| `accelerator` | `AcceleratorConfig` | 否 | 并行拓扑 |
| `mixed_precision` | `MixedPrecisionConfig` | 否 | 混合精度 |
| `gradient_checkpointing` | `GradientCheckpointingConfig` | 否 | activation checkpoint 模式 |
| `debug` | `DebugConfig` | 否 | 调试参数 |

各子 Config 定义：

```python
@dataclass
class TrainingConfig:
    max_steps: int = 100
    global_batch_size: int = 8
    init_device: Literal["meta", "cpu", "cuda", "npu"] = "meta"
    loss_aggregation: Literal["token_weighted", "rank_average"] = "token_weighted"

@dataclass
class AcceleratorConfig:
    dp_shard_size: int = 1
    tp_size: int = 1

@dataclass
class MixedPrecisionConfig:
    enabled: bool = False

@dataclass
class GradientCheckpointingConfig:
    activation_checkpoint: Literal["off", "none", "full", "selective"] = "off"

@dataclass
class DebugConfig:
    check_nan_inf: bool = False
```

`model` 字段仍使用 `hyper_parallel.trainer.config.ModelConfig`（复用现有 `ModelSpec` registry 和 `name` 查找机制），本阶段不修改模型侧构建逻辑。

---

### 2.4 YAML 解析：`resolve_root()` 与 `resolve_component()`

#### 2.4.1 `resolve_root()` — 入口

```python
# hyper_models/config/resolver.py

def resolve_root(raw: object) -> TrainerConfig:
    """校验 YAML 根字段并构造 TrainerConfig。"""

    # 1. 必须是 mapping
    if not isinstance(raw, Mapping):
        raise _fail("$", "YAML root must be a mapping")

    # 2. 拒绝未知一级字段
    root_fields = {field.name: field for field in fields(TrainerConfig)}
    unknown = sorted(set(raw) - set(root_fields))
    if unknown:
        raise _fail("$", f"unknown configuration fields: {unknown}")

    # 3. 检查必填字段
    missing = [
        field.name for field in root_fields.values()
        if field.name not in raw
        and field.default is MISSING
        and field.default_factory is MISSING
    ]
    if missing:
        raise _fail("$", f"missing required configuration fields: {missing}")

    # 4. 对每个一级字段调用 resolve_component()
    root_hints = get_type_hints(TrainerConfig)
    resolved = {
        name: resolve_component(node, expected_type=root_hints[name], path=f"$.{name}")
        for name, node in raw.items()
    }

    # 5. 构造 TrainerConfig
    return TrainerConfig(**resolved)
```

#### 2.4.2 `import_target()` — 解析 `_target_`

```python
def import_target(target_path: str, *, path: str) -> object:
    """将 dotted path 解析为 callable，支持嵌套类如 X.Config。

    示例：
        "hyper_models.components.optim.AdamW.Config" → <class 'AdamW.Config'>
        "hyper_models.trainer.config.TrainingConfig" → <class 'TrainingConfig'>
    """
    parts = target_path.split(".")

    # 从最长前缀开始尝试 import
    for split_at in range(len(parts), 0, -1):
        module_name = ".".join(parts[:split_at])
        try:
            target = importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            if exc.name == module_name or module_name.startswith(f"{exc.name}."):
                continue
            # 模块本身存在、但其内部 import 的依赖缺失：
            # 不透传 ModuleNotFoundError，统一包装为 ConfigResolutionError
            raise _fail(
                path,
                f"target {target_path!r} failed while importing dependency {exc.name!r}",
            ) from exc
        except ImportError as exc:
            raise _fail(path, f"target {target_path!r} could not be imported: {exc}") from exc

        # 逐级 getattr
        for attribute in parts[split_at:]:
            if not hasattr(target, attribute):
                raise _fail(path, f"target {target_path!r} has no attribute {attribute!r}")
            target = getattr(target, attribute)

        if not callable(target):
            raise _fail(path, f"target {target_path!r} is not callable")
        return target

    raise _fail(path, f"target {target_path!r} could not be imported")
```

#### 2.4.3 `resolve_component()` — 类型校验 + 参数校验 + 构造

```python
def resolve_component(node: object, *, expected_type: object, path: str) -> object:
    """解析一个 YAML 一级分组。

    完整流程：
    1. 必须是 mapping + 必须有 _target_
    2. import_target() 解析 _target_ 为 callable
    3. 校验 target 返回类型是否与 expected_type 兼容（通过 _annotation_assignable）
    4. 校验 target 参数签名（拒绝 *args / **kwargs）
    5. 对每个 YAML 参数做 typed 转换（coerce_value）
    6. 调用 target(**normalized_args) 构造 Config 对象
    7. 校验返回值的实际类型
    """
```

关键校验点：

- **target 类型匹配**：`_target_` 指向的 callable 返回类型必须与 `TrainerConfig` 对应字段的类型兼容（如 `optimizer` 字段期望 `Optimizer.Config`，则 target 返回类型必须是 `Optimizer.Config` 的子类）
- **参数签名校验**：拒绝 `*args` 和 `**kwargs` 可变参数，所有参数必须有显式名称
- **类 target 的注解取自 `__init__`**：`_target_` 为类时，参数类型注解通过 `get_type_hints(target.__init__)` 解析——类级注解只覆盖 dataclass 字段，普通类的构造参数注解在 `__init__` 上；从 `__init__` 解析同时保证 `from __future__ import annotations` 模块中未求值的字符串注解也能正确解析
- **参数类型转换**：YAML 中的每个参数值都通过 `coerce_value()` 按 target 参数类型做校验和转换
- **factory 返回类型**：非 class 的 target（如工厂函数）必须声明返回类型注解
- **构造即校验**：target 在解析阶段就被调用，返回的 Config 对象立即通过 `coerce_value()` 做类型校验

#### 2.4.4 `coerce_value()` — typed 值校验与转换

```python
def coerce_value(value: object, annotation: object, *, path: str) -> object:
    """按类型注解校验和转换一个值。

    支持的类型：
    - bool, int, float, str — 严格类型检查（bool 不与 int 混淆）
    - list[T] — 递归校验每个元素
    - tuple[...] — 支持固定长度、可变长度（...）
    - Optional[T] / Union[A, B] — 依次尝试匹配
    - Literal["a", "b"] — 精确值校验
    - 自定义类型（dataclass 等） — isinstance 检查
    """
```

关键特性：

- **bool 严格检查**：`bool` 不与 `int` 混合（`isinstance(True, int)` 为 True，但 `coerce_value` 对 `bool` 注解只接受 `bool` 值）
- **Literal 闭集校验**：`init_device` 只能是 `"meta"`/`"cpu"`/`"cuda"`/`"npu"` 之一，`activation_checkpoint` 只能是 `"off"`/`"none"`/`"full"`/`"selective"` 之一，非法值在启动阶段即报错
- **Literal 布尔词映射（PyYAML 1.1 兼容）**：PyYAML 1.1 将未加引号的 `on`/`off`/`yes`/`no`/`true`/`false` 解析为 `bool`。`_coerce_literal` 收到 `bool` 值时统一映射回 choices 中对应的词（`True` 按 `on`→`yes`→`true` 顺序、`False` 按 `off`→`no`→`false` 顺序，取 choices 中首个命中）；建议用户在 YAML 中为这类值加引号以避免歧义
- **完整字段路径**：所有错误信息都包含完整路径（如 `$.optimizer.lr`、`CLI.training.max_steps`）

---

### 2.5 CLI typed override

```python
# hyper_models/config/manager.py

def _apply_typed_overrides(config: TrainerConfig, overrides: Sequence[str]) -> TrainerConfig:
    """对已解析的 TrainerConfig 应用 typed CLI override。

    规则：
    1. 只能 override 最终 Config Tree 中已存在的字段
    2. value 通过 yaml.safe_load() 解析
    3. 通过 _replace_path() 做 dotted 路径替换，每层都校验类型
    4. 未选择组件（值为 None）的路径报错
    5. 未知字段报错（含模糊匹配建议）
    """
```

`_replace_path()` 的关键行为：

- 对每层 dataclass 校验字段名是否存在（未知字段用 `difflib.get_close_matches` 给出建议）
- 叶子节点通过 `coerce_value()` 按类型注解校验新值
- 中间节点为 `None`（组件未选择）时报错

**标量回退转换**：CLI override 的 value 先经 `yaml.safe_load()` 解析。PyYAML 1.1 不把无小数点的科学计数法（如 `1e-4`）解析为 float——`yaml.safe_load("1e-4")` 得到字符串 `"1e-4"`。因此当目标注解为 `int`/`float`（含 `Optional` 包装）且解析结果为 `str` 时，会先做 `int()`/`float()` 回退转换，`--optimizer.lr=1e-4` 为合法写法；转换失败再按类型错误报错。

---

### 2.6 Configurable 基类与嵌套 Config

所有训练组件（Optimizer、LRScheduler、Loss）继承自 `Configurable` 基类，使用嵌套 `Config` dataclass 定义配置参数。

```python
# hyper_models/config/configurable.py

class Configurable:
    """组件基类——通过嵌套 Config dataclass 声明配置参数。

    Config.build() 构造外层对象；__init_subclass__ 自动绑定 Config._owner。
    """

    @dataclass(kw_only=True, slots=True)
    class Config:
        """配置基类：提供 replace() / to_dict() / traverse() / build()。"""
        _owner: ClassVar[type["Configurable"] | None] = None

        def replace(self, **kwargs) -> "Configurable.Config": ...
        def to_dict(self) -> dict: ...
        def traverse(self, config_cls, *, _prefix="") -> Iterator[...]: ...
        def build(self, **kwargs) -> "Configurable": ...
```

以 `AdamW` 为例：

```python
# hyper_models/components/optim/optimizer.py

class Optimizer(Configurable):
    """优化器组件类别基类。"""

    @dataclass
    class Config(Configurable.Config):
        """优化器槽位接受的配置基类。"""

class AdamW(Optimizer):
    """AdamW 优化器。"""

    @dataclass
    class Config(Optimizer.Config):
        lr: float = 1e-4
        weight_decay: float = 0.01
        betas: tuple[float, float] = (0.9, 0.999)
        eps: float = 1e-8
        foreach: Optional[bool] = None

    def __init__(self, config: "AdamW.Config") -> None:
        self.config = config
```

`CosineWithWarmup` 和 `CausalLMLoss` 同理，分别继承 `LRScheduler.Config` 和 `Loss.Config`。

**组件扩展**：新增实现只需继承对应类别基类并定义自己的 `Config` 子类，不需要修改 `TrainerConfig`。`TrainerConfig` 中 `optimizer` 字段类型为 `Optional[Optimizer.Config]`，任何 `Optimizer.Config` 的子类都能通过类型检查。

---

### 2.7 YAML 示例与完整解析过程

#### 输入 YAML

```yaml
model:
  _target_: hyper_parallel.trainer.config.ModelConfig
  name: qwen3_5
  weights_path: /path/to/weights

training:
  _target_: hyper_models.trainer.config.TrainingConfig
  max_steps: 100
  global_batch_size: 8

accelerator:
  _target_: hyper_models.trainer.config.AcceleratorConfig
  tp_size: 2
  dp_shard_size: 4

optimizer:
  _target_: hyper_models.components.optim.AdamW.Config
  lr: 0.0001
  weight_decay: 0.1

lr_scheduler:
  _target_: hyper_models.components.optim.CosineWithWarmup.Config
  warmup_ratio: 0.05

loss:
  _target_: hyper_models.components.loss.CausalLMLoss.Config
  ignore_index: -100
```

#### 解析过程

```
① yaml.safe_load() → raw dict

② resolve_root(raw)
   │
   ├─ $.model
   │   import_target("hyper_parallel.trainer.config.ModelConfig") → <class ModelConfig>
   │   校验: ModelConfig 是类 → result_type = ModelConfig
   │         _annotation_assignable(ModelConfig, ModelConfig) → True ✓
   │   签名: ModelConfig(name=..., weights_path=...)
   │   coerce_value("qwen3_5", str) → "qwen3_5"
   │   coerce_value("/path/to/weights", str) → "/path/to/weights"
   │   → ModelConfig(name="qwen3_5", weights_path="/path/to/weights")
   │
   ├─ $.training
   │   import_target("hyper_models.trainer.config.TrainingConfig") → <class TrainingConfig>
   │   校验: _annotation_assignable(TrainingConfig, TrainingConfig) → True ✓
   │   → TrainingConfig(max_steps=100, global_batch_size=8)
   │
   ├─ $.accelerator
   │   → AcceleratorConfig(tp_size=2, dp_shard_size=4)
   │
   ├─ $.optimizer
   │   import_target("hyper_models.components.optim.AdamW.Config") → <class AdamW.Config>
   │   校验: _annotation_assignable(AdamW.Config, Optimizer.Config) → True ✓（issubclass）
   │   → AdamW.Config(lr=0.0001, weight_decay=0.1)
   │
   ├─ $.lr_scheduler
   │   → CosineWithWarmup.Config(warmup_ratio=0.05)
   │
   └─ $.loss
       → CausalLMLoss.Config(ignore_index=-100)

③ TrainerConfig(**resolved) → 最终强类型配置树

④ _apply_typed_overrides(config, ["--training.max_steps=200", "--accelerator.tp_size=4"])
   │
   ├─ "training.max_steps" → _replace_path(config, ["training", "max_steps"], 200)
   │   config.training 是 TrainingConfig → 字段 "max_steps" 存在 ✓
   │   coerce_value(200, int) → 200 ✓
   │   → replace(config.training, max_steps=200)
   │
   └─ "accelerator.tp_size" → _replace_path(config, ["accelerator", "tp_size"], 4)
       → replace(config.accelerator, tp_size=4)

最终 TrainerConfig:
  model = ModelConfig(name="qwen3_5", weights_path="/path/to/weights")
  training = TrainingConfig(max_steps=200, global_batch_size=8)
  accelerator = AcceleratorConfig(tp_size=4, dp_shard_size=4)
  optimizer = AdamW.Config(lr=0.0001, weight_decay=0.1)
  lr_scheduler = CosineWithWarmup.Config(warmup_ratio=0.05)
  loss = CausalLMLoss.Config(ignore_index=-100)
```

**关键差异**：与原始 ConfigNode 设计不同，这里 `_target_` 解析后**立即被调用**（如 `AdamW.Config(lr=0.0001, weight_decay=0.1)`），返回的是类型化 Config 对象，而非存储 callable 等待后续 `instantiate()`。整个解析过程是**完全 eager** 的——所有类型校验、参数转换、对象构造都在 `parse_training_args()` 返回之前完成。

---

### 2.8 文件结构总览

| 文件 | 职责 |
|------|------|
| `hyper_models/config/configurable.py` | `Configurable` 基类 + 嵌套 `Config` dataclass：`replace()` / `to_dict()` / `traverse()` / `build()` |
| `hyper_models/config/manager.py` | 公开入口 `parse_training_args()` + CLI override `_apply_typed_overrides()` |
| `hyper_models/config/resolver.py` | `resolve_root()` / `resolve_component()` / `import_target()` / `coerce_value()` — typed 解析核心 |
| `hyper_models/trainer/config.py` | `TrainerConfig` + 子 Config dataclass 定义 |
| `hyper_models/components/optim/optimizer.py` | `Optimizer` 类别 + `AdamW.Config` |
| `hyper_models/components/optim/lr_scheduler.py` | `LRScheduler` 类别 + `CosineWithWarmup.Config` |
| `hyper_models/components/loss/loss.py` | `Loss` 类别 + `CausalLMLoss.Config` |

---

### 2.9 与旧代码的对比

| 旧方式（硬编码） | 新方式（强类型 `_target_` IoC） |
|-----------------|------------------------|
| `if model_type == "llama": model = LlamaForCausalLM(config)` | YAML `_target_` 声明类型，`resolve_component()` 自动校验 |
| `if optim == "adamw": opt = AdamW(params, **kwargs)` | `--optimizer.lr=0.0003` typed CLI override |
| `register_spec("qwen3_5", ModelSpec(...))` | YAML `model:` 段 `_target_: hyper_parallel.trainer.config.ModelConfig`，模型选择走 `ModelConfig.name` + registry |
| 新增模型需要修改 Recipe 代码 | 新增模型只需新增 YAML 配置文件 + `MODEL_ARCH_MAPPING` 条目 |
| 新增组件需要注册 + if/else 分支 | 新增组件继承类别基类 + YAML 中声明 `_target_`，不需要修改 `TrainerConfig` |
| 配置拼写错误静默使用默认值 | 未知字段、未选择组件、错误类型**立即报错**，含完整字段路径 |
| 类型错误在运行时暴露 | 所有类型校验在 `parse_training_args()` 阶段完成 |

---

## 3. 配置解析后的组件构建

> **实现状态**：当前 `AdamW`、`CosineWithWarmup` 和 `CausalLMLoss` 提供 typed Config、组件类别和 `build()` 协议。
> 它们**尚未接入**现有 Trainer 的 optimizer、scheduler 和 loss 构建流程（后续迁移完成）。
> `RecipeConfig` 桥接层**未在代码中实现**——因为 `parse_training_args()` 直接返回强类型 `TrainerConfig`，不需要桥接。

### 3.1 Configurable.build() 协议

`Configurable.Config.build()` 用于从 Config 对象构造运行时组件实例：

```python
# Configurable.Config 提供的基础 build()

def build(self, **kwargs) -> "Configurable":
    """构造 Configurable 子类实例。

    使用 Config._owner 找到外层类，dataclasses.replace 复制配置后传入。
    **kwargs 允许传入不存储在 Config 中的运行时参数（如 model、device_mesh 等）。
    """
    owner_cls = self._owner
    built_config = replace(self)
    if not kwargs:
        return _build_from_config(owner_cls, built_config)
    # 校验 kwargs 不与 config 字段名冲突
    ...
    return _build_from_config(owner_cls, built_config, **kwargs)
```

**Config 绑定校验（`__init_subclass__`）**：`Configurable.__init_subclass__` 在子类定义嵌套 `Config` 时自动绑定 `Config._owner`，并做严格校验：

- 嵌套 `Config` 必须是 `Configurable.Config` 的子类，否则抛 `TypeError`
- 若该 `Config` 已被其他 owner 绑定（别名复用，如 `Config = AdamW.Config`），拒绝重绑定并抛 `TypeError`——应**子类化** Config 而非起别名

### 3.2 为什么需要 `.build()` 分开配置与构造

以优化器为例，参数分组（decay / no_decay）依赖运行时的 `model.parameters()` 迭代器，无法在 YAML 解析阶段确定：

```python
# 典型 build() 实现（规划中，当前未接入 Trainer）
class AdamW(Optimizer):
    def __init__(self, config: AdamW.Config, model=None):
        self.config = config
        if model is not None:
            # 参数分组 → 创建真正的 torch.optim.AdamW
            ...
```

**`.build()` 与直接构造的关系**：

| | YAML 解析阶段（`resolve_component`） | 运行时构造阶段（`build()`） |
|---|---|---|
| 做什么 | Config 对象（纯数据） | 真正的运行时对象（torch.optim.Optimizer 等） |
| 何时 | `parse_training_args()` 内 | `Recipe.setup()` 内 |
| 依赖 | 只有 YAML 中的静态值 | 需要 model、device_mesh 等运行时对象 |
| 典型产物 | `AdamW.Config(lr=0.0001, ...)` | `torch.optim.AdamW(param_groups, lr=0.0001)` |

---

### 3.3 Configurable 辅助功能

#### `replace()` — 不可变更新

```python
cfg = AdamW.Config(lr=0.0001, weight_decay=0.1)
cfg2 = cfg.replace(lr=0.001)  # AdamW.Config(lr=0.001, weight_decay=0.1)
# cfg 不变
```

#### `to_dict()` — 序列化

递归将 Config 树转为 JSON 友好的 dict，跳过 `_` 前缀字段。callable 字段用 `repr()` 转换。

#### `traverse()` — 递归遍历

```python
# 遍历配置树中所有特定类型的 Config 节点
for fqn, config, parent, field_name in root_config.traverse(SomeConfig):
    print(f"Found {fqn}: {config}")
```

支持在遍历中替换节点：`setattr(parent, field_name, new_cfg)` 或 `parent[index] = new_cfg`。

---

## 4. 总入口调用时序：从 `main()` 到所有组件就绪

> **实现状态**：当前已实现的仅为配置解析阶段（步骤① `parse_training_args()`）。
> ③ `recipe.setup(cfg)` 及之后的模型构建、优化器构建、数据加载等编排流程
> 仍为**规划中的目标调用链**，尚未在代码中落地。
>
> 当前 `train_lm.py` 和 `train_vl.py` 继续使用原有 `parse_args(HyperTrainerConfig)` 流程，
> 模型节点继续复用现有 `hyper_parallel.trainer.config.ModelConfig` 和 `ModelSpec` registry。

下面从 `main()` 开始，逐层展开每一个函数调用。数字序号表示调用顺序，缩进表示调用深度。
**粗体 = 已实现**，普通文本 = 规划中。

### 4.1 总体调用链

```
main()                                          # recipes/llm/train_ft.py
│
├─① **cfg = parse_training_args()**                               # **§2.2: 已实现**
│   │   **argparse → 分离 config_file 和 --field=value overrides**
│   └─ **_load_training_config(config_file, overrides)**
│       ├─ **yaml.safe_load() → raw dict**
│       ├─ **resolve_root(raw)**                                  # **§2.4: 已实现**
│       │   ├─ **校验 TrainerConfig 一级字段（拒绝未知字段）**
│       │   ├─ **$.model → resolve_component() → ModelConfig(name=..., ...)**
│       │   ├─ **$.training → resolve_component() → TrainingConfig(max_steps=100, ...)**
│       │   ├─ **$.accelerator → resolve_component() → AcceleratorConfig(tp_size=2, ...)**
│       │   ├─ **$.optimizer → resolve_component() → AdamW.Config(lr=0.0001, ...)**
│       │   ├─ **$.lr_scheduler → resolve_component() → CosineWithWarmup.Config(...)**
│       │   ├─ **$.loss → resolve_component() → CausalLMLoss.Config(...)**
│       │   └─ **→ TrainerConfig(**resolved)**                   # **强类型配置树就绪**
│       │
│       └─ **_apply_typed_overrides(config, overrides)**          # **§2.5: 已实现**
│           ├─ **--training.max_steps=200 → replace(TrainingConfig, max_steps=200)**
│           └─ **--accelerator.tp_size=4 → replace(AcceleratorConfig, tp_size=4)**
│
│   最终 cfg: TrainerConfig = TrainerConfig(
│       model = ModelConfig(name="qwen3_5", weights_path="/path/to/weights"),
│       training = TrainingConfig(max_steps=200, global_batch_size=8),
│       accelerator = AcceleratorConfig(tp_size=4, dp_shard_size=4),
│       optimizer = AdamW.Config(lr=0.0001, weight_decay=0.1),
│       lr_scheduler = CosineWithWarmup.Config(warmup_ratio=0.05),
│       loss = CausalLMLoss.Config(ignore_index=-100),
│   )
│
├─② recipe = FinetuneRecipe()                                    # 规划中
│
├─③ recipe.setup(cfg)                                            # 规划中: 构建所有训练组件
│   │
│   ├─③.1 initialize_distributed("nccl")                          # 初始化 torch.distributed 进程组 + CUDA device
│   ├─③.2 self.rng = StatefulRNG(seed=cfg.training.seed, ranked=True)         # 规划中：seed 待加入 TrainingConfig
│   ├─③.3 self.distributed_setup = create_distributed_setup_from_config(cfg)  # 从 cfg 构建分布式拓扑 → 06_distributed_infrastructure.md §3
│   │
│   ├─③.3a self.callback_manager = build_callback_manager(cfg, ...)           # 03 §4.2: 混合 Callback 系统
│   │
│   ├─③.4 self.model, self.optimizer_init = build_model(cfg.model, peft_config, ...)  # §6.7: Recipe 编排入口（peft_config 规划中：由 YAML peft 段解析后传入）
│   │   self.model, self.optimizer_init = build_model(                        # Recipe 内部编排入口（§6.7）
│   │       cfg.model, peft_config=self.peft_config,
│   │       distributed_setup=self.distributed_setup)
│   │   │
│   │   └─ HyperAutoModelForCausalLM.from_pretrained(          # §6.2
│   │           pretrained_model_name_or_path="Qwen/Qwen3.5-0.8B",
│   │           torch_dtype="bfloat16",
│   │           distributed_setup=<DistributedSetup>,
│   │           peft_config=<PeftConfig | None>)
│   │       │
│   │       ├─③.4.1 mesh = distributed_setup.mesh_context
│   │       ├─③.4.2 sharding_planner, fsdp2_manager, autopipeline           # §8
│   │       │       = instantiate_infrastructure(...)
│   │       ├─③.4.3 hf_config = AutoConfig.from_pretrained(...)
│   │       ├─③.4.4 is_hf_model = get_is_hf_model(hf_config)                # §5: MODEL_ARCH_MAPPING
│   │       └─③.4.5 model = _build_model(...)                               # §6.3: meta→shard→load
│   │           ├─③.4.5.1 is_meta_device = (world_size > 1 or not is_hf_model)  # 确定 meta device
│   │           ├─③.4.5.2 with init_ctx: model = _init_model(...)             # §7: meta device 空壳模型（零显存）
│   │           ├─③.4.5.3 _apply_peft(model, peft_config)        if peft_config   # LoRA 层注入 (§6.4)
│   │           ├─③.4.5.4 _apply_qat(model, qat_config)          if qat_config
│   │           ├─③.4.5.5 _apply_fp8(model, fp8_config)          if fp8_config
│   │           ├─③.4.5.6 _apply_parameter_freezing(model, freeze_config)         # 参数冻结 (§6.5)
│   │           ├─③.4.5.7 plan = sharding_planner.plan(model, mesh.device_mesh, ...)  # §9 → 05 §3.6
│   │           ├─③.4.5.8 apply_sharding_plan(model, plan, mesh.device_mesh, ...)     # DTensor 分片应用 → 05 §4
│   │           ├─③.4.5.9 torch.compile(model, **compile_config) if compile_config    # fully_shard 之前
│   │           ├─③.4.5.10 fsdp2_manager.parallelize(model, ...) if fsdp2             # FSDP2 在 meta 上包裹（canonical：先于 to_empty/load）
│   │           ├─③.4.5.11 load_base_model(model, device, pretrained_path, adapter=..., mesh=...)
│   │           │       └─ 前置动作（同属本步）：model.to_empty(device=device)  # meta → GPU 物化，load 前必须先物化
│   │           ├─③.4.5.12 _freeze_non_lora_params(model)        if peft_config   # PEFT 非 LoRA 参数冻结 (§6.4)
│   │           └─ return model                                # 已分片、权重已加载
│   │   ← build_model 内部②: optimizer_init = OptimizerInit.from_distributed_setup(...)  # §6.7: 导出 param 分组/mesh
│   │   ← build_model 返回 (model, optimizer_init)
│   │
│   ├─③.5 self.loss = cfg.loss.build()                                     # Configurable.build()
│   ├─③.6 self.checkpointer = cfg.checkpoint.build(dp_rank=..., tp_rank=...)  # 规划中：checkpoint 待加入 TrainerConfig → 04 §4/§5
│   ├─③.7 self.optimizer = cfg.optimizer.build(model, device_mesh=...)        # Configurable.build()
│   ├─③.8 self.dataloader, self.tokenizer = build_dataloader(cfg, ...)        # 数据加载 → 02 §3
│   ├─③.9 self.val_dataloaders = build_validation_dataloader(cfg, ...)        # 验证集加载 → 02 §3
│   ├─③.10 self.step_scheduler = cfg.step_scheduler.build(                    # 规划中：step_scheduler 待加入 TrainerConfig
│   │        self.dataloader, dp_size, local_batch_size)                      # → 03 §4
│   ├─③.11 self.lr_scheduler = cfg.lr_scheduler.build(optimizer, step_scheduler)  # Configurable.build()
│   ├─③.12 self.load_checkpoint(...)                                          # 断点续训 → 04 §8
│   └─③.13 self.mfu_calc = AutoMFU.from_config(self.model)                    # MFU 计算器
│
└─④ recipe.run_train_validation_loop()                                         # 开始训练
    └─ 详见 03_training_loop.md §6/§7/§8
```

**关键时序要点**：

> **PP 说明**：`autopipeline.build(model)`（PP stage 拆分）在 `apply_model_infrastructure()`
> 中**最先执行**（③.4.5.3 的 PEFT 注入步骤之前，PP 未启用时无此步）——裁决以 §8.2 为准，stage 切分必须
> 先于权重加载与 FSDP2 包裹（§8.3 ① / §6.3 Step 3）。时序树中从略。

| 序号 | 操作 | 关键输出 |
|:----:|------|---------|
| ① | **YAML 加载** | **`TrainerConfig` 强类型配置树（所有 `_target_` 已解析为类型化 Config 对象）** |
| ③.4.2 | `instantiate_infrastructure` | ShardingPlanner(05 §3.6) + FSDP2Manager(06 §4) + AutoPipeline(01 §8.2) |
| ③.4.4 | `get_is_hf_model` | MODEL_ARCH_MAPPING 查表 → 自定义/HF 路径判定 |
| ③.4.5.2 | `_init_model` | meta device 空壳模型（零显存） |
| ③.4.5.7 | `sharding_planner.plan()` | ShardingPlan（可序列化分片策略）→ 05 §3.6 |
| ③.4.5.8 | `apply_sharding_plan()` | DTensor 分片应用（生产/校验双模）+ `_local_params_context` 解包 → 05 §4/§7/§8 |
| ③.4.5.10 | `fsdp2_manager.parallelize()` | FSDP2 在 meta 上包裹（canonical：先于 to_empty/load） |
| ③.4.5.11 | `load_base_model()`（前置 `model.to_empty()` 物化，同属本步） | 每 rank 独立加载权重（5 参 canonical，零 NCCL） |
| ③.3a | `build_callback_manager()` | CallbackManager（含 CheckpointCallback/EvaluateCallback/LoggingCallback/TqdmCallback 等内置 callback）→ 03 §4.2 |
| ③.6 | `cfg.checkpoint.build()` | Checkpointer → 04 §4/§5 |
| ③.7 | `cfg.optimizer.build()` | 真正的优化器（参数分组完成）→ 03 §9 |
| ③.8 | `build_dataloader()` | DataLoader + Tokenizer → 02 §3 |
| ③.10 | `cfg.step_scheduler.build()` | StepScheduler → 03 §4 |
| ③.12 | `load_checkpoint()` | 恢复所有组件状态 → 04 §8 |
| ④ | `run_train_validation_loop()` | 训练主循环 → 03 §6/§7/§8 |

### 4.2 `setup()` 内部：每条语句的数据来源（规划中）

> **实现状态**：以下为规划中的 `setup()` 实现。当前代码中 optimizer / scheduler / loss
> 的 `build()` 未接入 Trainer，`cfg` 为 `TrainerConfig` 强类型对象。

以下追踪 `setup()` 中每个关键变量的**完整来源**——它来自 `TrainerConfig` 的哪个字段、`_target_` 解析成了什么 Config 类型：

```python
class FinetuneRecipe(BaseRecipe):
    def setup(self, cfg: TrainerConfig):  # cfg 已是强类型 TrainerConfig

        # ═══════════════════════════════════════════════════════
        # m1 model
        # 来源: cfg.model → ModelConfig(name="qwen3_5", weights_path="...")
        # （当前仍通过现有 ModelSpec registry 解析模型类，本阶段不修改模型侧逻辑）
        # build_model() 是 Recipe 内部编排入口（§6.7），返回 (model, optimizer_init)
        # ═══════════════════════════════════════════════════════
        self.mesh = self.distributed_setup.mesh_context
        self.peft_config = peft_config  # PEFT 必须在 ShardingPlanner.plan 之前注入
                                        # （规划中：peft_config 由 YAML peft 段解析后传入 setup/build_model；
                                        #  peft 非 TrainerConfig 一级字段）
        self.model, self.optimizer_init = build_model(
            cfg.model,
            peft_config=self.peft_config,
            distributed_setup=self.distributed_setup,
        )

        # ═══════════════════════════════════════════════════════
        # m2 tokenizer — build_dataloader() 内部通过 _build_tokenizer() 获取
        # ═══════════════════════════════════════════════════════

        # ═══════════════════════════════════════════════════════
        # m3 ds (Dataset) — build_dataloader() 内部通过 load_dataset() 获取
        # ═══════════════════════════════════════════════════════

        # ═══════════════════════════════════════════════════════
        # m4 sampler — build_dataloader() 内部逻辑决定
        # ═══════════════════════════════════════════════════════

        # ═══════════════════════════════════════════════════════
        # m5 dataloader — build_dataloader() 末尾构造
        # ═══════════════════════════════════════════════════════

        self.dataloader, self.tokenizer = build_dataloader(
            cfg.dataset,          # 规划中：dataset 待加入 TrainerConfig
            cfg.dataloader,       # 规划中：dataloader 待加入 TrainerConfig
            cfg.model,
            cfg.packed_sequence,  # 规划中：packed_sequence 待加入 TrainerConfig
            seed=cfg.training.seed,  # 规划中：seed 待加入 TrainingConfig
            local_batch_size=1,  # 规划中：step_scheduler 待加入 TrainerConfig
            ...
        )

        # ═══════════════════════════════════════════════════════
        # m6 optimizer
        # 来源: cfg.optimizer → AdamW.Config(lr=0.0001, weight_decay=0.1, ...)
        # 已是强类型 Config！直接 .build(model) 创建真正的 torch.optim.AdamW
        # 返回: list[torch.optim.Optimizer]
        # ═══════════════════════════════════════════════════════
        self.optimizer = cfg.optimizer.build(
            self.model,
            optimizer_init=self.optimizer_init,
            device_mesh=self.mesh.device_mesh,
        )

        # ═══════════════════════════════════════════════════════
        # m7 loss_fn
        # 来源: cfg.loss → CausalLMLoss.Config(ignore_index=-100)
        # .build() 内部: CausalLMLoss(config=...)
        # 返回: nn.Module
        # ═══════════════════════════════════════════════════════
        self.loss = cfg.loss.build()

        # ═══════════════════════════════════════════════════════
        # m8 lr_scheduler
        # 来源: cfg.lr_scheduler → CosineWithWarmup.Config(warmup_ratio=0.05, ...)
        # .build(optimizer, step_scheduler) 内部:
        #   ① 未设置字段从 step_scheduler 推断默认值
        #   ② 创建 OptimizerParamScheduler(optimizer, lr_warmup_steps=100, ...)
        # 返回: OptimizerParamScheduler
        # 注: self.step_scheduler 由 cfg.step_scheduler.build() 构造
        #    （规划中：step_scheduler 待加入 TrainerConfig，见 §4.1 ③.10）
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

**关键时序**：PEFT 注入在分片**之前**（§6.3 Step 4 / §8.3 ②），非 LoRA 冻结在 FSDP2 包裹 + 权重加载**之后**（§8.3 ⑨.5，对应 §4 时序树 ③.4.5.12 的 `_freeze_non_lora_params` 步骤）——因为冻结操作需要遍历已包裹、已物化的参数。

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

### 6.7 `build_model` 与 `OptimizerInit`（Recipe 编排入口）

> **实现状态**：本节为规划中的 Recipe 侧编排入口，尚未在代码中落地。
> 当前模型构建仍走现有 `ModelSpec` registry 流程。

`build_model()` 是 Recipe 内部编排入口（§4.1 ③.4 / §4.2 m1），与 `HyperAutoModel.from_pretrained` 的职责区分：

- `from_pretrained` 是 HF 入口，**返回单 model**（PreTrainedModel）。
- `build_model` **返回 `(model, optimizer_init)`**——内部调用 `from_pretrained`（自定义路径）或 `_build_model`（HF 原生路径）完成 meta→shard→load，并从 `distributed_setup` / ShardingPlan 导出 `OptimizerInit`，供 `Recipe.setup()` 调用 `cfg.optimizer.build(model, optimizer_init=...)` 时使用，避免 Recipe 重复推导 param 分组与 mesh 信息。

```python
# hyper_models/components/models/common/

def build_model(
    model_cfg,                        # ModelConfig（name + weights_path 等）
    peft_config=None,                 # 由 setup() 从 YAML peft 段解析后传入（规划中）
    distributed_setup=None,
    **kwargs,
) -> tuple["nn.Module", "OptimizerInit"]:
    """高层 build_model 入口。

    Returns:
        (model, optimizer_init)
        - model: 已分片、权重已加载的模型（meta→shard→load 完成）
        - optimizer_init: 见 OptimizerInit。
    """
    # ① 调用 HF 兼容入口构建模型（自定义路径走 from_pretrained，含 meta→shard→load）
    model = HyperAutoModelForCausalLM.from_pretrained(
        model_cfg.weights_path,
        distributed_setup=distributed_setup,
        peft_config=peft_config,
        **kwargs,
    )

    # ② 从 distributed_setup / ShardingPlan 导出 OptimizerInit（param 分组、mesh、is_peft）
    #    weight_decay 由 Recipe 侧从 cfg.optimizer 读取后经 Optimizer.Config.build 生效；
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

    Recipe.setup() 将其传给 Optimizer.Config.build(model, optimizer_init=...)，
    避免 Recipe 侧重复推导 param 分组与 mesh 信息。
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
        注：最终分组由 Optimizer.Config.build 内部以 self.weight_decay 重做，
        此处 param_groups 为预分组描述，供调用方/调试参考。
        """
        mesh_ctx = getattr(distributed_setup, "mesh_context", None) if distributed_setup else None
        device_mesh = mesh_ctx.device_mesh if mesh_ctx is not None else None
        is_peft = peft_config is not None

        # 简化的 decay/no_decay 分组（完整分组逻辑由 Optimizer.Config.build 内部完成）
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


def _is_no_decay(name: str) -> bool:
    """判定参数是否应归入 no_decay 组（常规规则：bias 与 1D norm 权重不衰减）。

    被 OptimizerInit.from_distributed_setup 与 AdamW.Config.build 复用。
    匹配 "bias"、层归一化权重（名字含 "norm"/"ln"/"layernorm"）等常见模式。
    """
    if name.endswith("bias"):
        return True
    if "norm" in name.lower() or "ln" in name.lower() or "layernorm" in name.lower():
        return True
    return False
```

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
    # 辅助工厂定义在 hyper_models/components/distributed/fsdp2.py 与 pipelining.py（见 §8.1/§8.2）
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

`_instantiate_fsdp2` 的 canonical 签名（06 文档引用 "01 §8.1 `_instantiate_fsdp2(*, config, mesh_context)`"，以此为准）：

```python
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

    # ⑨.5 PEFT 非 LoRA 参数冻结（在 FSDP2 + 权重加载之后；与 §6.4 / §4 ③.4.5.12 一致）
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
