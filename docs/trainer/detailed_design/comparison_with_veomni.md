# Hyper-Parallel 重构方案 vs VeOmni Trainer 对比分析

> 编写日期：2026-07-23
> 对比范围：Hyper-Parallel 重构方案（基于 01~06 详细设计） vs VeOmni VeOmni 现有 trainer 方案
> 评价视角：**算法人员**的易用性、易读性、易调试性

---

## 目录

1. [配置系统对比](#1-配置系统对比)
2. [模型加载与并行化](#2-模型加载与并行化)
3. [训练编排方式](#3-训练编排方式)
4. [Callback 系统](#4-callback-系统)
5. [训练循环](#5-训练循环)
6. [数据管道](#6-数据管道)
7. [Checkpoint 与断点续训](#7-checkpoint-与断点续训)
8. [多任务扩展](#8-多任务扩展)
9. [分布式信号处理](#9-分布式信号处理)
10. [总体评价与结论](#10-总体评价与结论)

---

## 1. 配置系统对比

### Hyper-Parallel 重构方案

```
YAML 文件 + CLI --field=value overrides
   → parse_training_args()（hyper_models/config/manager.py）
   → resolve_root() / resolve_component()（hyper_models/config/resolver.py：
       _target_ 经 import_target 即时解析 → 签名校验 → coerce_value typed 转换 → 立即构造）
   → TrainerConfig（强类型配置树，hyper_models/trainer/config.py，9 个一级字段）
   → typed .build()（Configurable 协议，hyper_models/config/configurable.py）
```

> 注：原设计的 `load_yaml_config()` → ConfigNode 弱类型 → RecipeConfig 桥接 → `.instantiate()` 方案**未实现、不会实现**；实际已落地为上述强类型方案（commit 78a79c0f），与 VeOmni 的 `_typed_config.py` 同属强类型路线。

**核心思想**：YAML 即 IoC 容器。所有组件通过 `_target_: fully.qualified.ClassName` 声明类型，框架不硬编码任何 `if-else` 分支。

```yaml
optimizer:
  _target_: torch.optim.AdamW
  lr: 2.0e-4
  weight_decay: 0.1
```

**对算法人员的影响**：
- ✅ 新增组件只需改 YAML，零 Python 代码改动
- ✅ 非开发用户也可独立配置训练
- ✅ `_target_` 字符串路径可以精确追踪到类定义（"torch.optim.AdamW" 直接定位）
- ✅ `_target_` 在配置解析期即被 import_target 解析 + 构造签名校验 + coerce_value typed 转换，拼写/类型错误在启动时暴露（非训练中途）
- ✅ 无 ConfigNode 中间黑盒：YAML 直接构造为强类型 dataclass（TrainerConfig / Configurable.Config），IDE 可跳转、可补全

### VeOmni

```
Python 入口 → parse_args() → VeOmniArguments（强类型 dataclass 树）
                               → 直接访问属性 args.train.optimizer.lr
```

**核心思想**：纯 Python 类型系统约束。

```python
@dataclass
class OptimizerConfig:
    type: str = "adamw"
    lr: float = 5e-5
    weight_decay: float = 0.01

args = parse_args("train.yaml")
# args.train.optimizer.lr 是 float，IDE 自动补全
```

**对算法人员的影响**：
- ✅ IDE 自动补全，编译期类型检查，开发期错误捕获更早
- ✅ 从 `args.train.optimizer.lr` 可直接 Ctrl+Click 跳到 `OptimizerConfig` 类定义
- ❌ 新增组件需要修改 `arguments_types.py` 中的 dataclass 定义 + `parser.py`
- ❌ YAML 配置需要与 Python 类型定义保持同步，否则解析失败

### 易用性对比

| 场景 | Hyper-Parallel | VeOmni |
|------|---------------|--------|
| 调参（改 lr） | 改 YAML 一行 | 改 YAML 或命令行参数 |
| 新增优化器类型 | YAML 改 `_target_` 即可 | 改 `arguments_types.py` 的 `OptimizerConfig` |
| 类型错误发现时机 | 配置解析期（resolve_component 签名校验 + coerce_value typed 转换，启动时暴露） | 编译期（IDE 报错） |
| 代码追踪路径 | `_target_` 字符串 → import_target 解析 → 强类型 Config 类定义（IDE 可跳转） | 属性访问 → `OptimizerConfig` 类定义 |
| 配置嵌套深度 | 点号路径 `cfg.get("step_scheduler.local_batch_size")` | 属性链 `args.train.step_scheduler.local_batch_size` |

**结论**：两者均为强类型方案，**类型安全性基本持平**——Hyper-Parallel 已落地强类型解析（TrainerConfig + coerce_value typed 校验，commit 78a79c0f），配置错误在启动期暴露；VeOmni 为编译期 IDE 检查，开发期反馈略早。Hyper-Parallel 的 `_target_` IoC 方案在**扩展性**（新增组件零代码改动）更好。原"ConfigNode 弱类型"的差距结论已失效。

---

## 2. 模型加载与并行化

### Hyper-Parallel 重构方案

```python
# 方式 A：一步到位（HF 兼容）
model = HyperAutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3.5-4B",
    distributed_setup=DistributedSetup(tp=4, cp=2),
)

# 内部流程（_build_model）：
# meta device 构建 → PEFT 注入 → 参数冻结 → ShardingPlanner.plan() →
# apply_sharding_plan()（含 PrecompiledBoundary 编译期通信规划）→
# FSDP2 包裹 → to_empty 物化 → 每 rank 独立加载权重
```

**核心创新**：
- **ShardingPlanner 6-phase 管线**：ParamClassifier → BoundaryGrouper → SemanticRoleInference → TemplateLookup → ChainPropagator → SpecialHandler
- **每模型从 ~1367 行降至 ~20 行**：`ARCH_OVERRIDES` 声明式规则替代过程式 `parallelize.py`
- **双模式**：validate 模式校验 placement 正确性；production 模式 `_local_params_context` 解包 DTensor 为零开销
- **PrecompiledBoundary**：编译期通信规划，运行时零 dispatch

**对算法人员的影响**：
- ✅ 新模型适配成本极低：注册 `MODEL_ARCH_MAPPING` + 20 行分片规则
- ✅ `from_pretrained` 入口与 HF 完全一致，心理模型迁移成本低
- ✅ validate 模式可以提前发现 placement 错误，不用等训练 crash
- ❌ 需要理解 `ShardingTemplate` / `ParamRole` / `Placement` 概念
- ❌ 6-phase 管线内部逻辑复杂，调试分片问题需要深入理解

### VeOmni

```python
# 两步走
self.model = build_foundation_model(
    config_path="...", weights_path="...", torch_dtype="bfloat16",
    init_device="meta",
)
self.model = build_parallelize_model(
    self.model, init_device="meta", weights_path="...",
    mixed_precision=..., enable_gradient_checkpointing=...,
    basic_modules=[...],                        # 需要手动指定
)
```

**核心思想**：单函数入口封装所有并行化逻辑。

**对算法人员的影响**：
- ✅ 模型加载和并行化职责清晰，两行代码完成
- ✅ `build_parallelize_model` 参数集中，所有并行化选项可见
- ❌ 需要手动指定 `basic_modules`（不拆分的模块列表），容易遗漏
- ❌ 不同模型需要不同的参数组合，需要阅读文档
- ❌ 并行化函数内部逻辑复杂（483 行），调试困难

### 易用性对比

| 场景 | Hyper-Parallel | VeOmni |
|------|---------------|--------|
| 新模型适配 | 注册 20 行分片规则 | 配置 `basic_modules` + 参数组合 |
| 并行化认知模型 | 声明式 DSL（模板 + 角色） | 函数式参数（单函数入口） |
| 调试分片错误 | validate 模式提前校验 | 运行时 crash 才暴露 |
| 运行时开销 | 编译期规划，零 dispatch | 运行时 FSDP 管理通信 |
| 模型结构理解需求 | 需要理解 transformer 边界 | 只需要知道 `basic_modules` |

**结论**：Hyper-Parallel 的 ShardingPlanner 在**新模型适配效率**和**分片正确性验证**上显著优于 VeOmni。但需要算法人员理解更多的概念（ShardingTemplate / ParamRole），学习曲线更陡。VeOmni 的 `build_parallelize_model` 单函数入口更直观，但灵活性不足。

---

## 3. 训练编排方式

### Hyper-Parallel 重构方案

```python
main() → recipe.setup(cfg) → recipe.run_train_validation_loop()
```

**Recipe.setup() 显式编排**：
```python
def setup(self, cfg: TrainerConfig):
    # ① 分布式初始化
    self.dist_env = initialize_distributed("nccl")
    # ③ RNG
    self.rng = StatefulRNG(seed=..., ranked=True)
    # ⑦ Loss
    self.loss_fn = self.cfg.loss.build()
    # ⑩ Checkpointer（规划中：checkpoint 字段待加入 TrainerConfig）
    self.checkpointer = self.cfg.checkpoint.build(...)
    # ⑪ Model
    self.model, self.optimizer_init = build_model(...)
    # ⑫ Optimizer
    self.optimizer = self.cfg.optimizer.build(...)
    # ⑬ DataLoader
    self.dataloader, self.tokenizer = build_dataloader(...)
    # ⑮ StepScheduler（规划中：step_scheduler 字段待加入 TrainerConfig）
    self.step_scheduler = self.cfg.step_scheduler.build(...)
    # ⑯ LR Scheduler
    self.lr_scheduler = self.cfg.lr_scheduler.build(...)
    # ⑰ 注册 checkpoint 追踪状态
    self.register_state("model", "model")
    self.register_state("optimizer", "optimizer")
    ...
    # ⑱ 断点续训
    self.load_checkpoint(...)
```

**核心设计**：
- 18 步组件构建按依赖顺序显式排列
- 每条语句的调用方式明确（typed `.build()`，Configurable 协议；原设计的 untyped `.instantiate()` 路径已取消）
- 每个组件的创建时机和依赖关系一目了然

**对算法人员的影响**：
- ✅ 打开 `setup()` 就能看到完整的组件构建流程，无需跳转多个文件
- ✅ 新增组件只需在 `setup()` 中加一行，然后注册到 `__state_tracked`
- ✅ 组件构建顺序由依赖关系天然决定，不会出现"DataLoader 还没创建就引用它"的错误
- ❌ 18 步顺序构建，代码较长（约 100 行）
- ❌ 子类 override `setup()` 时，需要理解哪些步骤必须保留

### VeOmni

```python
# BaseTrainer.__init__ 中串联
self._setup()                           # 分布式初始化
self._build_model()                     # 模型构建
self._freeze_model_module()             # 模型冻结
self._build_model_assets()              # 模型资产（tokenizer）
self._build_data_transform()            # 数据转换
self._build_dataset()                   # 数据集
self._build_collate_fn()                # Collate 函数
self._build_dataloader()                # DataLoader
self._build_parallelized_model()        # 并行化
self._build_optimizer()                 # 优化器
self._build_lr_scheduler()              # LR 调度器
self._build_training_context()          # 训练上下文
self._init_callbacks()                  # Callback 初始化
```

**核心设计**：
- 13 个 `_build_*` 方法按职责拆分
- 每个方法职责单一，可被子类独立 override
- Composition 模式：`TextTrainer` 持有 `BaseTrainer` 实例，选择性调用

**对算法人员的影响**：
- ✅ 每个 `_build_*` 方法职责单一，代码量小（平均 30-50 行）
- ✅ 子类只需 override 需要修改的方法，其他复用
- ❌ `__init__` 中 13 个方法串联，但调用链分散在多个类中（`TextTrainer.__init__` 中手动拼接）
- ❌ Composition 模式下 `base._build_*` 调用链理解成本高——需要知道哪些方法调了哪些没调
- ❌ `_build_parallelized_model` 内部逻辑复杂，但对外隐藏了细节

### 易用性对比

| 场景 | Hyper-Parallel | VeOmni |
|------|---------------|--------|
| 理解组件构建流程 | 打开 `setup()` 全可见 | 需要追踪 `TextTrainer.__init__` → `BaseTrainer._build_*` |
| 新增组件 | 在 `setup()` 加一行 | 新增 `_build_*` 方法 + 在 `__init__` 中调用 |
| 子类定制 | override `setup()` + 选择性保留 | override 特定 `_build_*` 方法 |
| 代码量（核心类） | ~1105 行（BaseRecipe） | ~815 行（BaseTrainer） |
| 理解成本 | 中等（线性阅读） | 高（多类跳转 + Composition） |

**结论**：Hyper-Parallel 的显式 `setup()` 在**可读性**和**透明度**上优于 VeOmni 的 `_build_*` 方法串联。算法人员打开 `setup()` 就能看到完整的组件构建流程，不需要跳转到多个类。VeOmni 的拆分在**职责单一**上更好，但 Composition 模式增加了理解成本。

---

## 4. Callback 系统

### Hyper-Parallel 重构方案（混合方案）

**设计原则**：核心训练流程显式编排在 Recipe 中；外围关注点通过 Callback 处理。

```python
# 核心训练：显式
metrics = self._run_train_optim_step(batches)

# 外围关注点：Callback
state = StepState(step=..., epoch=..., is_ckpt_step=..., ...)
self.callback_manager.on_step_end(state)
```

**3 个回调点**：
| 回调点 | 用途 |
|--------|------|
| `on_train_begin()` | 资源初始化 |
| `on_step_end(state)` | 所有步级操作（checkpoint/验证/日志/GC） |
| `on_train_end()` | 资源清理 |

**StepState**：Frozen dataclass，所有时序标记由 `StepScheduler` 统一计算透传。

```python
@dataclass(frozen=True)
class StepState:
    step: int
    epoch: int
    is_ckpt_step: bool          # 由 StepScheduler 计算
    is_val_step: bool           # 由 StepScheduler 计算
    is_log_step: bool           # 由 StepScheduler 计算
    loss: float
    grad_norm: float | None
    lr: float
    ...
```

**对算法人员的影响**：
- ✅ 核心训练流程（forward/backward/optimizer step）在 Recipe 中可见，不会被 callback 链淹没
- ✅ 时序标记集中管理，不用猜测"当前步该做什么"
- ✅ `StepState` 是 frozen dataclass，callback 只读不写，无副作用
- ✅ 想要自定义行为（如新的监控指标），只需注册新 callback
- ❌ 1 个 `on_step_end` 点承载所有外围操作，单个 callback 的性能问题会影响整个步

### VeOmni

**纯 Callback 事件驱动**：

```python
def on_step_end(self, loss=None, loss_dict=None, grad_norm=None):
    self.environ_meter_callback.on_step_end(...)
    self.tqdm_callback.on_step_end(...)
    self.wandb_callback.on_step_end(...)
    self.profile_callback.on_step_end(...)
    self.checkpointer_callback.on_step_end(...)     # 自己计算 is_ckpt_step
    self.hf_ckpt_callback.on_step_end(...)           # 自己计算 is_hf_save_step
    self.evaluate_callback.on_step_end(...)           # 自己计算 is_val_step
    self.moe_monitor_callback.on_step_end(...)
```

**6 个回调点**：`on_train_begin/end`、`on_epoch_begin/end`、`on_step_begin/end`

**设计特点**：
- 每个 callback 独立判断"现在该做什么"（`global_step % save_steps == 0`）
- 参数通过 `loss=`, `loss_dict=`, `grad_norm=` 等独立关键字传递
- 7 个 callback 在 `on_step_end` 中串联执行

**对算法人员的影响**：
- ❌ 核心训练流程被 callback 分散——`train_step` 中只有 `forward_backward` 循环，看不到 checkpoint/验证
- ❌ 每个 callback 独立计算时序判断，逻辑分散
- ❌ 7 个 callback 串联执行，性能开销不易察觉
- ❌ 调试时需要追踪 `on_step_begin` → `train_step` → `on_step_end` 的事件流
- ✅ 扩展性强——新增功能只需实现新 callback
- ✅ 参数解耦——每个 callback 只接收自己需要的参数

### 易用性对比

| 场景 | Hyper-Parallel（混合） | VeOmni（纯 Callback） |
|------|----------------------|---------------------|
| 理解训练流程 | 核心显式可见，外围 callback 收口 | 核心流程被 callback 分散 |
| 调试步级操作 | 打开 `run_train_validation_loop` 看 StepState | 追踪 7 个 callback 的 `on_step_end` |
| 自定义扩展 | 注册新 callback | 注册新 callback |
| 时序判断一致性 | StepScheduler 统一计算 | 每个 callback 独立计算 |
| 回调点数量 | 3 个 | 6 个 |
| 核心逻辑可见性 | **高** | **低** |

**结论**：Hyper-Parallel 的混合方案在**可读性**和**调试友好性**上显著优于 VeOmni 的纯 Callback 方案。核心训练流程显式在 Recipe 中，外围操作通过 `StepState` 透传的时序标记统一驱动，callback 不需要自己计算"现在该做什么"。

---

## 5. 训练循环

### Hyper-Parallel 重构方案

```python
def run_train_validation_loop(self):
    self.callback_manager.on_train_begin()
    try:
        for epoch in self.step_scheduler.epochs:
            self.step_scheduler.set_epoch(epoch)
            for batches in self.step_scheduler:
                # ── 核心：显式 ──
                train_metrics = self._run_train_optim_step(batches)
                # ── 外围：Callback ──
                state = StepState(step=..., is_ckpt_step=..., ...)
                self.callback_manager.on_step_end(state)
    finally:
        self.callback_manager.on_train_end()
```

**StepScheduler 统一控制训练节奏**：
```python
# 所有时序判断集中在一个地方
self.step_scheduler.is_ckpt_step   # 需要保存 checkpoint
self.step_scheduler.is_val_step    # 需要运行验证
self.step_scheduler.is_log_step    # 需要远程日志
self.step_scheduler.is_gc_step     # 需要垃圾回收
self.step_scheduler.sigterm_received  # 收到 SIGTERM
```

**对算法人员的影响**：
- ✅ 打开 `run_train_validation_loop` 就能看到完整的训练流程骨架
- ✅ `StepScheduler` 迭代器自动处理 grad_acc 分组，不用手动管理
- ✅ 所有时序标记由 `StepScheduler` 统一计算，不会出现"checkpoint 回调认为该保存了但验证回调还没跑完"的时序问题
- ❌ `StepScheduler` 是一个需要理解的抽象概念

### VeOmni

```python
def train(self):
    self.on_train_begin()
    for epoch in range(self.start_epoch, args.train.num_train_epochs):
        self.on_epoch_begin()
        self.data_iterator = VeOmniIter(self.train_dataloader, ...)
        for _ in range(self.start_step, args.train_steps):
            self.train_step(self.data_iterator)
        self.on_epoch_end()
    self.on_train_end()

def train_step(self, data_iterator):
    self.state.global_step += 1
    micro_batches = next(data_iterator)       # 手动从 iterator 取
    self.on_step_begin(micro_batches=...)
    for micro_step, micro_batch in enumerate(micro_batches):
        loss, loss_dict = self.forward_backward_step(micro_batch)
    grad_norm = veomni_clip_grad_norm(...)
    self.optimizer.step()
    self.lr_scheduler.step()
    self.optimizer.zero_grad()
    self.on_step_end(loss=..., grad_norm=...)
```

**VeOmniIter 包装 DataLoader**：
```python
class VeOmniIter:
    """支持 BackgroundPrefetcher 的迭代器包装。"""
    def __init__(self, dataloader, use_background_prefetcher=False):
        ...
```

**对算法人员的影响**：
- ✅ 训练循环简洁直观
- ❌ grad_acc 分组在 `train_step` 中手动从 `data_iterator` 取，StepScheduler 不负责分组
- ❌ checkpoint/验证/日志的触发判断分散在多个 callback 的 `on_step_end` 中
- ❌ 核心训练逻辑（forward/backward）和外围操作（checkpoint/日志）在 `train_step` 中混在一起

### 易用性对比

| 场景 | Hyper-Parallel | VeOmni |
|------|---------------|--------|
| 理解训练骨架 | 打开 `run_train_validation_loop` 一目了然 | 需要追踪 `train()` → `train_step()` → `on_step_end()` |
| 步进迭代 | `StepScheduler` 自动分组 | `data_iterator` 手动取 |
| 时序判断 | 集中管理 | 分散在多个 callback |
| 验证循环 | 在 `run_train_validation_loop` 中显式 | 通过 `EvaluateCallback` 的 `on_step_end` |
| 背景预取 | 无 | `BackgroundPrefetcher` 线程预取 |

**结论**：Hyper-Parallel 的 `StepScheduler` + 混合方案在**可读性**和**一致性**上优于 VeOmni。但 VeOmni 的 `BackgroundPrefetcher` 是 Hyper-Parallel 尚未覆盖的生产级特性。训练循环骨架上两者差异不大，但 Hyper-Parallel 的时序标记集中管理显著降低了认知负担。

---

## 6. 数据管道

> 经调研 Automodel（~30 个数据集类 + 完整 loader/collate/packing 体系）和 VeOmni（~15 个核心模块）后，此节为详细对比分析。

### 6.1 整体架构对比

| 维度 | Automodel | VeOmni | Hyper-Parallel 当前 |
|------|-----------|--------|-------------------|
| 数据集类型 | ~30 个（Chat、Megatron、HellaSwag、SQuAD、VLM、Audio、Diffusion、BAGEL 等） | ~10 个（Mapping、Iterable、Interleave、WeightedMultiSource、Energon 等） | 2 个（HF + Megatron） |
| 数据源混合 | 单源 | WeightedMultiSource（3 种耗尽策略，token 级别重加权） | 单源 |
| 对话格式 | ChatDataset + format_chat_template | Transform 注册表 + ChatTemplate 注册表 | 无 |
| 动态 Batching | 无 | ✅ DynBszBuffer（2 种模式，token 预算） | 无 |
| 数据变换层 | 内嵌在 dataset 中 | Transform 注册表（10+ 种） | 内嵌在 build_dataloader 中 |
| 后台预取 | 无 | BackgroundPrefetcher | 无 |
| SP Collator | 无 | 3 步管线（MainCollator） | 仅 CP sharding 契约 |
| 媒体工具 | VLM utils（图像/视频/LMDB） | multimodal/(image/video/audio)_utils | 无 |
| 配置类型化 | _typed_config.py 强类型解析 | VeOmniArguments 强类型 dataclass | 强类型 dataclass 解析（TrainerConfig + Configurable，commit 78a79c0f） |
| 状态管理 | StatefulDataLoader | StatefulDataLoader + 快照 | StatefulDataLoader |

### 6.2 数据集类型覆盖

**Automodel 的优势**：拥有最完整的数据集类型体系，覆盖 5 个模态（LLM、VLM、Multimodal、Audio、Diffusion），每个模态下又有多个子类型：

| 模态 | 数据集类型 | 用途 |
|------|-----------|------|
| LLM | ChatDataset | 对话格式 SFT，支持 mask_history、mask_reasoning |
| LLM | ColumnMappedTextInstructionDataset | 通用指令微调 |
| LLM | MegatronPretraining | 高性能预训练（.bin/.idx） |
| LLM | HellaSwag / SQuAD / GLUE MRPC | 评估 |
| LLM | RetrievalDataset | 检索训练 |
| LLM | AgentChat / XLAM | Agent SFT |
| LLM | DeltaLakeDataset | 流式 Delta Lake |
| VLM | RdrDataset / CordV2 / MedPix / LlavaOnevision / MetaDataset | VLM SFT |
| Multimodal | SftJSONLIterableDataset / T2IIterableDataset / UnifiedEdit | BAGEL 多模态 |
| Audio | Cv17Dataset | 语音 |
| Diffusion | TextToImageDataset / TextToVideoDataset | 文生图/视频 |

**VeOmni 的优势**：`WeightedMultiSourceDataset` 是大规模训练的关键能力，支持：
- 多源按权重采样（const / changing 调度）
- token 级别重加权（长样本获得更高概率）
- 3 种耗尽策略（`first_exhausted` / `all_exhausted` / `never_exhausted`）
- 完整 checkpoint 恢复（每源状态 + 随机状态）

**Hyper-Parallel 的缺失**：当前仅支持 HF `load_dataset()` 和 Megatron 两种类型，缺少：
- **P0**: ChatDataset（对话格式 SFT 训练无法启动）
- **P0**: WeightedMultiSourceDataset（多源混合训练无法启动）
- **P2**: 评估数据集（HellaSwag、SQuAD）
- **P3**: BAGEL 多模态、Diffusion

### 6.3 对话格式与 Chat Template

**VeOmni**：4 种 ChatTemplate（default / llama2 / chatml / Janus），通过 `CHAT_TEMPLATE_REGISTRY` 注册。

**Automodel**：`format_chat_template()` 支持：
- `mask_history`：只对 assistant 回复计算 loss（其余 -100）
- `mask_reasoning_content`：屏蔽 reasoning 标签内部 loss
- 多轮对话的 assistant mask 通过 O(n_turns) 次 `apply_chat_template()` 定位边界

**Hyper-Parallel**：当前完全缺失。default_collater 只做 padding，没有对话模板化和 loss mask 构建。需在 `hyper_models/components/datasets/llm/formatting_utils.py` 中补齐。

### 6.4 动态 Batching

**VeOmni**：`DynBszBuffer` + `TextBatchingStrategy` + `DynamicBatchSizeDataLoader`，两种运行模式：
- `runtime="main"`: 主进程基于 token 预算动态组 batch
- `runtime="worker"`: DataLoader worker 进程内组 batch（更精确的 checkpoint 恢复）

两种 token 计数模式：
- `"total"`: 按 `attention_mask.sum()` 计数（物理 token）
- `"effective"`: 按 `labels != IGNORE_INDEX` 计数（仅计算 loss 的 token）

**Automodel**：无动态 batching，固定 batch_size。

**Hyper-Parallel**：当前完全缺失。变长序列训练时填充浪费严重。需在 `hyper_models/components/datasets/dynamic_batching.py` 中补齐，核心设计见 [02_data_pipeline_addendum.md](02_data_pipeline_addendum.md#23-动态-batching基于-token-数)。

### 6.5 数据变换层

**VeOmni**：通过 `DATA_TRANSFORM_REGISTRY` 注册 10+ 种变换，将数据预处理与数据集解耦：

| 变换名 | 用途 |
|--------|------|
| `"plaintext"` | 纯文本分块 + tokenize |
| `"conversation"` | 对话模板化 + loss mask |
| `"dpo"` | DPO chosen/rejected 拼接 |
| `"classification"` | 分类标签映射 |
| `"qwen2_vl"` / `"qwen3_vl"` | VLM 图像+文本处理 |
| `"dit_online"` / `"dit_offline"` | DiT 数据 |

**Hyper-Parallel**：当前无变换层，tokenization 和预处理逻辑分散在 `build_dataloader` 内部和 dataset 的 `__getitem__` 中。需在 `hyper_models/components/datasets/transforms/registry.py` 中补齐。

### 6.6 生产级特性

| 特性 | VeOmni | Automodel | Hyper-Parallel 当前 |
|------|--------|-----------|-------------------|
| BackgroundPrefetcher | ✅ 后台线程预取 | ❌ | ❌ |
| 动态 Batching | ✅ 2 种模式 | ❌ | ❌ |
| 多源加权采样 | ✅ WeightedMultiSource | ✅（单源） | ❌ |
| 数据平衡（VLM 视觉 tokens） | ✅ Qwen3VL DataBalance | ❌ | ❌ |
| 流式 Reservoir shuffle | ❌ | ✅ ReservoirSampler | ❌ |
| LRU 缓存 Lazy Transform | ❌ | ✅ LazyMappedDataset | ❌ |
| 并行预 tokenize | ❌ | ✅ tokenize_dataset_parallel | ❌ |

### 6.7 易用性对比

| 场景 | VeOmni | Automodel | Hyper-Parallel 当前 | Hyper-Parallel 补齐后 |
|------|--------|-----------|-------------------|---------------------|
| 快速启动 SFT | 4 步分别调用，需理解 Transform 注册表 | 一次调用 build_dataloader | 一次调用 build_dataloader | 一次调用 build_dataloader |
| 多源数据混合 | ✅ 配置即用 | 手动拼接 | ❌ | ✅ WeightedMultiSource |
| 对话格式训练 | ✅ ChatTemplate 注册 | ✅ ChatDataset | ❌ | ✅ ChatDataset + formatting_utils |
| 变长序列效率 | ✅ 动态 batching | ❌ 固定 batch | ❌ 固定 batch | ✅ 动态 batching |
| 训练吞吐量 | ✅ BackgroundPrefetcher | ❌ | ❌ | ✅ BackgroundPrefetcher |
| 定制数据处理 | ✅ override Transform | 修改 dataset 类 | 修改 build_dataloader 内部 | ✅ TransformRegistry |
| 数据源切换 | 手动配置 dataloader_type | _target_ 自动分发 | _target_ 自动分发 | _target_ 自动分发 |
| 配置类型安全 | ✅ 强类型 dataclass | ✅ _typed_config.py | ✅ 强类型 dataclass 解析（coerce_value typed 校验，commit 78a79c0f） | ✅ 已落地 |
| 评估数据集 | ❌ | ✅ HellaSwag/SQuAD | ❌ | ✅（P2） |

### 6.8 结论

**Hyper-Parallel 当前数据管道严重不足，是重构方案中最大的薄弱环节。**

- 当前设计仅覆盖了 LLM 基础的 `build_dataloader` 7 步流程，缺少对话格式（ChatDataset）、多源混合（WeightedMultiSource）、动态 Batching、Chat Template 等**核心能力**
- 相比之下，Automodel 有 30 个数据集类 + 完整的 Chat Template 体系，VeOmni 有 10 个数据集类 + 动态 Batching + BackgroundPrefetcher + Transform 注册表
- 补齐方案已输出至 [02_data_pipeline_addendum.md](02_data_pipeline_addendum.md)，按 P0/P1/P2 优先级规划，**总计新增约 15.5 人天**

**核心建议**：以 Automodel 的数据集类体系为骨架（ChatDataset、MegatronPretraining、formatting_utils），以 VeOmni 的生产级特性为补充（动态 Batching、BackgroundPrefetcher、Transform 注册表、WeightedMultiSource），构建完整的数据管道。

> 补充设计已合并至 [02_data_pipeline.md](02_data_pipeline.md)（§5~§18），原 addendum 文件已删除。

---

## 7. Checkpoint 与断点续训

### Hyper-Parallel 重构方案

```python
# 自动追踪——BaseRecipe.__setattr__
class BaseRecipe:
    def __setattr__(self, key, value):
        # 自动检测 stateful 组件并注册到 __state_tracked
        if is_model(value) or has_load_restore_state(value) or ...:
            self.__state_tracked.add(key)

# 自动保存——save_checkpoint 遍历 __state_tracked
def save_checkpoint(self, ...):
    for key in sorted(self.__state_tracked):
        if is_model(getattr(self, key)): ...
        elif is_optimizer(getattr(self, key)): ...
        # 按 kind 自动分发，无需手动构造 state dict

# 自动恢复——load_checkpoint 遍历 __state_tracked
def load_checkpoint(self, restore_from):
    for key in sorted(self.__state_tracked):
        self._load_state_by_kind(name, kind, path)   # 按 kind 分发恢复
```

**核心设计**：
- `__setattr__` 重写：赋值时自动检测并注册 stateful 组件
- 保存/恢复时按 `__state_tracked` 自动分发，无需手动管理
- `Checkpointer` 统一管理 DCP 切分保存 + HF safetensors 导出
- `StateDictAdapter` 透明处理 HF key ↔ 模型内部 key 映射

**对算法人员的影响**：
- ✅ 永远不需要手动追踪"哪些组件需要保存"——框架自动完成
- ✅ 新增组件只要实现了 `state_dict()` 和 `load_state_dict()`，自动被追踪
- ✅ 断点续训只需 `restore_from: LATEST`，自动恢复所有组件状态
- ❌ `__setattr__` 重写是隐式行为，需要理解其工作原理才能正确使用

### VeOmni

```python
# 手动构造 state dict——在 CheckpointerCallback 中
def _save_checkpoint(self, state: TrainerState):
    ckpt_state = {
        "model": self.trainer.model,
        "optimizer": self.trainer.optimizer,
        "extra_state": {
            "global_step": state.global_step,
            "lr_scheduler": self.trainer.lr_scheduler.state_dict(),
            "train_dataloader": train_dataloader_state,  # 手动获取
            "environ_meter": self.trainer.environ_meter.state_dict(),  # 手动获取
            "torch_rng_state": torch.get_rng_state(),  # 手动获取
        },
    }
    self.trainer.checkpointer.save(save_checkpoint_path, ckpt_state, ...)

# 手动恢复——在 CheckpointerCallback 中
def _load_checkpoint(self):
    self.trainer.checkpointer.load(load_path, state, ...)
    self.trainer.state.global_step = state["extra_state"]["global_step"]
    self.trainer.lr_scheduler.load_state_dict(state["extra_state"]["lr_scheduler"])
    # 手动恢复每个组件
```

**核心设计**：
- 通过 `extra_state` dict 手动传递所有需要保存的状态
- 保存/恢复时每个组件需要手动构造和解析

**对算法人员的影响**：
- ❌ 新增组件需要手动在 `_save_checkpoint` 和 `_load_checkpoint` 中添加/获取
- ❌ 容易遗漏——新增一个 RNG 状态、dataloader 状态等，需要记得在三个地方修改
- ❌ `extra_state` 的 key 拼写错误在运行时才暴露
- ✅ 灵活——可以保存任意额外的状态（如 `environ_meter`）

### 易用性对比

| 场景 | Hyper-Parallel | VeOmni |
|------|---------------|--------|
| 新增状态追踪组件 | 自动注册（实现 `state_dict` 即可） | 手动在 3 个地方修改 |
| 遗漏风险 | 低（框架自动追踪） | 高（手动管理） |
| 灵活度 | 中等（按 kind 分发） | 高（任意 extra_state） |
| 实现复杂度 | 高（__setattr__ 重写） | 低（手动 dict） |
| 正确性保障 | 高（框架保证完整性） | 低（依赖开发者仔细） |

**结论**：Hyper-Parallel 的 `__state_tracked` 自动追踪机制在**易用性**和**正确性保障**上显著优于 VeOmni 的手动 `extra_state` 管理。算法人员永远不需要担心"我忘了保存某个组件的状态"。VeOmni 的 `extra_state` 更灵活，但灵活性的代价是必须手动维护完整性。

---

## 8. 多任务扩展

### Hyper-Parallel 重构方案

```
BaseRecipe
  ├─ FinetuneRecipe（LLM 标准训练）
  └─ FinetuneRecipeForVLM（VLM 多模态训练）
```

**扩展方式**：继承 `BaseRecipe`，覆盖 `setup()` / `run_train_validation_loop()` / `_forward_backward_step()` 等方法。

### VeOmni

```
BaseTrainer
  ├─ TextTrainer（Composition）
  ├─ VLMTrainer（Composition）
  ├─ TextDPOTrainer（Composition）
  ├─ DiTTrainer（Composition）
  └─ BaseRLTrainer（继承 BaseTrainer）
```

**扩展方式**：Composition 模式为主（5 个中有 4 个），使用 `BaseTrainer.__new__` + 手动拼接 `_build_*` 步骤。

### 易用性对比

| 场景 | Hyper-Parallel | VeOmni |
|------|---------------|--------|
| 变体数量 | 2 个（LLM + VLM） | 5 个（Text/VLM/DiT/DPO/RL） |
| 扩展模式 | 继承 | Composition（多数） |
| 样板代码 | 少（继承天然复用） | 多（手动拼接 _build_*） |
| 与基础设施耦合 | 低（共享 BaseRecipe） | 高（需要理解 BaseTrainer 的 13 个 _build_*） |

**结论**：VeOmni 在**变体数量**和**生产验证度**上领先（5 种变体已在实际业务中使用）。Hyper-Parallel 的继承模式在**扩展效率**上更好（更少的样板代码），但当前只覆盖了 LLM 和 VLM 两种任务。VeOmni 的 Composition 模式虽然样板代码多，但在**灵活性**上更好——可以自由选择调用哪些 `_build_*` 步骤。

---

## 9. 分布式信号处理

### Hyper-Parallel 重构方案

- `DistributedSignalHandler` + `StepScheduler.sigterm_received`：SIGTERM 时优雅退出并保存 checkpoint
- 集成了 `StepScheduler.__iter__` 中，每步检查

### VeOmni

- 未在核心训练循环中看到显式的 SIGTERM 处理

**结论**：Hyper-Parallel 对 SIGTERM 有更前瞻的处理，VeOmni 依赖外部机制。

---

## 10. 总体评价与结论

### 总评表

| 维度 | Hyper-Parallel 重构方案 | VeOmni | 胜出 |
|------|----------------------|--------|:----:|
| **配置系统易用性** | `_target_` IoC 灵活 + 强类型解析（typed 校验、IDE 友好） | 强类型 dataclass，IDE 友好 | 持平（Hyper-Parallel 扩展性更优） |
| **模型并行化适配** | ShardingPlanner 声明式，~20 行/模型 | `build_parallelize_model` 函数式 | **Hyper-Parallel** |
| **训练流程可读性** | 显式 `setup()` + 混合 Callback | 纯 Callback + 分散的 `_build_*` | **Hyper-Parallel** |
| **调试友好性** | StepState 透传时序标记，单步追踪 | 7 个 callback 串联，调用栈深 | **Hyper-Parallel** |
| **Checkpoint 完整性** | `__state_tracked` 自动追踪 | 手动 `extra_state` 管理 | **Hyper-Parallel** |
| **数据管道成熟度** | 简洁但缺少生产级特性 | 有 BackgroundPrefetcher/动态 batching | VeOmni |
| **多任务扩展** | 继承模式简洁，但变体少 | Composition 灵活，5 种变体验证 | VeOmni |
| **运行时性能** | PrecompiledBoundary 零 dispatch | FSDP 运行时管理 | **Hyper-Parallel** |
| **学习曲线** | 概念多（ShardingTemplate/ParamRole/Configurable） | 概念少（纯 Python 类型） | VeOmni |

### 结论

**Hyper-Parallel 重构方案在算法人员的易用性、易读性、易调试性上整体优于 VeOmni**，具体表现为：

1. **易读性胜出**：显式 `setup()` + 混合 Callback 方案使训练流程一目了然。算法人员打开 `setup()` 就能看到完整的组件构建流程，打开 `run_train_validation_loop` 就能看到完整的训练骨架。VeOmni 的纯 Callback 方案将核心训练流程分散在多个 callback 中，需要跳转多个文件才能理解完整流程。

2. **易调试性胜出**：`StepState` 将所有时序标记集中透传，算法人员不需要追踪多个 callback 的独立判断逻辑。`__state_tracked` 自动追踪确保没有组件状态被遗漏。VeOmni 的 7 个 callback 串联和手动 `extra_state` 管理增加了调试时的认知负担。

3. **易用性各有优劣**：
   - **配置**：两者均为强类型 dataclass 解析（IDE 补全、typed 校验）——Hyper-Parallel 已落地 TrainerConfig 强类型方案（commit 78a79c0f）；且 Hyper-Parallel 的 `_target_` IoC 在扩展性上更好（新增组件零代码改动）
   - **模型适配**：Hyper-Parallel 的 ShardingPlanner 显著优于 VeOmni 的 `build_parallelize_model`（~20 行 vs 千行级 `parallelize.py`）
   - **数据管道**：VeOmni 更成熟，有 `BackgroundPrefetcher` 和 `dynamic_batching` 等生产级特性
   - **多任务**：VeOmni 已验证 5 种变体，Hyper-Parallel 当前只有 2 种

### 最终建议

**如果目标是构建一个面向算法人员的新训练框架，应以 Hyper-Parallel 重构方案为骨架，借鉴 VeOmni 的成熟特性**：

1. **保留** Hyper-Parallel 的分布式并行组件（ShardingPlanner、PrecompiledBoundary、DTensor 双模式）——这是超越 VeOmni 的核心差异化能力
2. **保留** Hyper-Parallel 的显式 Recipe 编排 + 混合 Callback 方案——在可读性和可调试性上显著优于 VeOmni
3. **保留** `__state_tracked` 自动追踪——在 checkpoint 完整性上优于 VeOmni 的手动管理
4. **借鉴** VeOmni 的 `BackgroundPrefetcher` 和 `dynamic_batching`——补齐生产级数据管道
5. ~~**借鉴** VeOmni 的 `VeOmniArguments` 的强类型设计~~（已达成，本项关闭）——Hyper-Parallel 已落地强类型配置解析（`parse_training_args` → `resolve_root` → `TrainerConfig`，coerce_value typed 校验，commit 78a79c0f），原 ConfigNode 弱类型方案已取消
6. **补齐** Hyper-Parallel 的 Recipe 变体数量——参考 VeOmni 的 DPO/RL/DiT 实现

**一句话总结**：Hyper-Parallel 重构方案的**架构设计**（声明式并行化 + 显式 Recipe 编排 + 混合 Callback + 自动状态追踪）在理论上优于 VeOmni，但需要在**生产级特性**（BackgroundPrefetcher、动态 batching、多任务变体）上补齐，才能真正在算法人员的日常使用中全面超越 VeOmni。