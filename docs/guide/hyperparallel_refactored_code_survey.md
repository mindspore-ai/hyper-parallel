# HyperParallel 重构后代码调研

> 范围：`feat/trainer-dev-pr-1080` 当前代码。本文只描述重构后的训练架构、当前调用链和完成度，不讨论低精训练方案。
>
> 读者：需要接入新模型、训练组件或分布式能力的开发者。

---

## 1. 结论

当前分支正在把旧的“模型目录内过程式并行化 + `BaseTrainer` 单体训练器”迁移为三层结构：

```text
配置与组装层      hyper_models/config + hyper_models/trainer
训练编排层        hyper_models/recipes
可复用组件层      hyper_models/components
```

其中最成熟、最具方向性的部分是 `hyper_models/components/distributed/`：它用 `ShardingPlanner` 声明并推导并行策略，用 `PrecompiledBoundary` 固化边界通信，并以 Dual-mode DTensor 同时支持生产执行和布局校验。

新训练骨架已能通过 `examples/training_skeleton/` 跑通配置解析、模型构建、dummy data、训练循环、优化器和状态保存流程；但真实数据管道、完整 checkpoint、FSDP2、PP、PEFT 和 AutoModel 权重加载仍存在 stub 或待接线部分。因此它是“架构与接口已成形、端到端最小闭环可跑”的重构中间态，不应视为所有训练功能已生产可用。

---

## 2. 重构目标与旧代码关系

### 2.1 新旧职责划分

| 旧实现方向 | 新实现方向 | 当前落点 |
|---|---|---|
| `hyper_parallel/models/*/parallelize.py` 按模型硬编码 TP/CP/EP/FSDP | 声明式模块边界与参数 placement | `hyper_models/components/distributed/sharding_planner.py` |
| 运行时根据 tensor layout 判断并重分布 | build 期编译输入/输出通信计划 | `hyper_models/components/distributed/precompiled_boundary.py` |
| `BaseTrainer` 聚合模型、并行、循环、回调 | Recipe 显式编排核心训练步骤，组件负责单一能力 | `hyper_models/recipes/` + `hyper_models/components/` |
| 旧三层 YAML 配置 | 强类型 `TrainerConfig` + `_target_` 组件构建 | `hyper_models/config/` + `hyper_models/trainer/config.py` |
| 旧 DCP/训练状态入口 | `BaseRecipe.register_state()` 统一登记生命周期状态 | `hyper_models/recipes/base_recipe.py` |

旧 `hyper_parallel/core/dtensor/`、`core/shard/` 和 `core/fully_shard/` 并未被重写：它们仍是 DTensor dispatch、通信和 HSDP/FSDP 的底层实现。新代码的变化在于将用户侧和模型侧的“如何组织并行”迁移到 `hyper_models/components/distributed/`。

### 2.2 新目录地图

```text
hyper_models/
  config/             # YAML 解析、类型解析、CLI dotted override
  trainer/            # 顶层 TrainerConfig
  _transformers/      # HF-compatible AutoModel 构建与基础设施编排
  recipes/            # 任务训练流程，例如 FinetuneRecipe
  data/               # data build 接口；当前为 dummy-data stub
  components/
    distributed/      # ShardingPlanner、Dual-mode DTensor、FSDP2/PP 接口
    training/         # StepScheduler、callback、grad accumulation、RNG
    optim/            # Optimizer/LR scheduler Config 与构建
    loss/             # loss 和 token-normalization 逻辑
    checkpoint/       # checkpoint Config 与 Checkpointer 接口
    models/common/    # build_model 和 OptimizerInit 导出
```

`components/` 的原则是“可独立使用，不能反向依赖 Recipe 或 AutoModel”。例如分布式组件可以直接用于任意 `nn.Module`：先 `ShardingPlanner.plan()`，再 `apply_sharding_plan()`；训练骨架只是它的一个消费者。

---

## 3. 从 YAML 到训练的主调用链

当前推荐入口是 `examples/training_skeleton/main.py`。它读取 YAML，解析为 `TrainerConfig`，按 `recipe` 创建 `FinetuneRecipe` 并运行。

```text
train.yaml
  -> hyper_models.config.parse_training_args()
  -> TrainerConfig
  -> RECIPE_REGISTRY[config.recipe]
  -> FinetuneRecipe.setup(config)
  -> FinetuneRecipe.run_train_validation_loop()
```

### 3.1 配置层

`hyper_models/config/resolver.py` 将 YAML mapping 递归解析为强类型 dataclass 或带 `_target_` 的组件 Config。`hyper_models/config/manager.py` 在此基础上支持：

- 一个 YAML 训练配置文件；
- `--accelerator.tp_size=4` 形式的 CLI dotted override；
- unknown field、缺失字段和类型不匹配的早期报错。

顶层 `TrainerConfig` 位于 `hyper_models/trainer/config.py`，当前持有：模型、optimizer、lr scheduler、loss、训练参数、并行参数、checkpoint、step scheduler、数据配置、PEFT 等。组件的运行时对象通过 `cfg.xxx.build(runtime_dependencies)` 构建，而不是训练循环内读取任意字典。

### 3.2 Recipe setup

`FinetuneRecipe.setup()` 是当前训练编排的核心。其顺序可概括为：

```text
initialize_distributed
  -> create DistributedSetup / MeshContext
  -> build callback manager、loss、checkpoint
  -> build_model
  -> build optimizer
  -> build train/validation dataloader
  -> build StepScheduler、LR scheduler
  -> register checkpoint states
  -> load checkpoint
```

`build_model()` 进入 `HyperAutoModelForCausalLM.from_pretrained()`，然后由 `_transformers/infrastructure.py` 统一编排模型基础设施。

### 3.3 模型基础设施编排

`apply_model_infrastructure()` 已冻结的顺序是：

```text
PP split
  -> PEFT
  -> QAT / FP8
  -> freeze
  -> ShardingPlanner.plan
  -> apply_sharding_plan
  -> torch.compile
  -> FSDP2 wrap
  -> meta materialize + load base model
```

这条顺序的含义是：模型结构变换先完成，随后统一推导和应用并行布局，最后才由 compile/FSDP2 管理运行时。当前实现中，ShardingPlanner/Applier 已可用；PEFT、QAT/FP8、freeze、真实权重加载、FSDP2 和 PP 的部分实现仍是 stub 或接口占位。

---

## 4. 分布式核心：Dual-mode DTensor

### 4.1 为什么有双模式

同一份 `ShardingPlan` 支持两种执行模式：

| 模式 | 参数 | 前向计算 | 用途 |
|---|---|---|---|
| `validate_mode=True` | 保持 DTensor | DTensor dispatch 推导 placement，并校验声明 | 验证并行策略正确性 |
| production | build 期解包为本地 plain tensor | local compute + 预编译通信边界 | 实际训练，避免运行时 DTensor dispatch 开销 |

生产模式不是放弃布局信息。`ShardingPlan` 仍保存参数和 I/O 的 placement 契约；参数在 FSDP2 前被 `_local_params_context()` 解包为 local tensor，同时产生 `tp_grad_info`，供后续 HSDP/FSDP 处理 TP replicated parameter 的梯度同步。

### 4.2 规划与应用

```text
model + DeviceMesh
  -> ShardingPlanner.plan()
       1. 参数角色分类
       2. 模块边界分组
       3. 语义角色推断
       4. 模板 -> ModuleShardingSpec
       5. 相邻边界 I/O 契约传播
       6. 特殊参数处理声明
  -> ShardingPlan
  -> apply_sharding_plan()
       A. 参数转 DTensor / EP expert 预处理
       B. special handler
       C. 生产模式 local-param 解包或 validate 保留 DTensor
       D. 注入 PrecompiledBoundary 和 forward wrapper
       E. 恢复 tied weight 本 rank alias
```

每个 `ModuleShardingSpec` 的核心是四个 I/O placement：

```text
in_src  : tensor 到达边界时的 placement
in_dst  : 模块内部所需 placement
out_src : 模块自然产生的 placement
out_dst : 下游所需 placement
```

`PrecompiledBoundary` 在 build 期比较这些 placement，生成 `RedistOp` 列表。生产 forward 只顺序执行已生成的输入通信、原模块 local forward 和输出通信，不再逐次做 layout 推断。

### 4.3 复杂模块的扩展口

并非所有逻辑都可由普通 DTensor dispatch 推导，例如 CP attention 的 K/V all-gather、EP MoE all-to-all。`ModuleShardingSpec` 预留了两类机制：

- `inner_target` / `inner_wrapper`：定位子模块并替换其内部 forward，当前主要服务 CP attention。
- `use_local_map` / `local_compute_fn`：让整个模块在 local-region skeleton 中计算，框架负责 DTensor/local 的边界拼接，当前主要服务 MoE。

因此模型新增特殊并行逻辑时，应优先通过 ShardingPlan override 和上述扩展点表达，而不是回到模型目录内写整套过程式 `parallelize.py`。

---

## 5. 训练循环与状态管理

### 5.1 核心显式，外围回调

`FinetuneRecipe.run_train_validation_loop()` 保留前向、反向、梯度累积、clip、optimizer step、LR step 的显式顺序；日志、评估、checkpoint、进度条、WandB、GC 和 SIGTERM 则由 `CallbackManager` 在 step 结束时处理。

```text
for batches in StepScheduler:
  -> _run_train_optim_step(batches)
       -> 统计全局有效 label token 数
       -> 对每个 microbatch 运行 forward/backward
       -> 仅在此处按全局 token 数缩放梯度
       -> clip -> optimizer.step -> scheduler.step
  -> CallbackManager.on_step_end(StepState)
```

这个设计的关键约束是：loss 先按 token 求和，跨 DP/CP 聚合分子/分母后再计算均值；梯度只在一个位置除以全局 token 数，避免 gradient accumulation、DP 和 CP 叠加时重复归一化。

### 5.2 Checkpoint 状态追踪

`BaseRecipe` 不要求每个组件各自处理训练目录。Recipe 在 setup 中显式登记：

```python
register_state("model", "model")
register_state("optimizer", "optimizer")
register_state("lr_scheduler", "lr_scheduler")
register_state("rng", "rng")
register_state("dataloader", "dataloader")
register_state("step_scheduler", "train_state")
```

save/load 两侧遍历同一登记表，根据 kind 决定路径和序列化方式。这个机制的价值是新增一个有状态组件时，只需完成其 `state_dict/load_state_dict` 并登记一次，避免 save 路径和 load 路径各改一处而失配。

---

## 6. 当前完成度

| 子系统 | 当前状态 | 说明 |
|---|---|---|
| 强类型 YAML + CLI override | 已实现 | `hyper_models/config/` 已有 resolver 和单测 |
| Recipe 训练循环 | 已实现骨架 | 训练、grad accumulation、callbacks、token normalization 已有实现与测试 |
| ShardingPlanner + Dual-mode Applier | 已实现 | 当前重构的核心完成部分，含 TP/CP/EP、local region 和校验模式 |
| `PrecompiledBoundary` | 已实现 | build 期生成通信计划，生产期直接执行 |
| HF-compatible AutoModel 入口 | 已实现编排骨架 | 实际模型覆盖、权重加载和部分基础设施仍在演进 |
| Optimizer / LR scheduler | 基础实现 | 接口和普通实现已具备；部分高级策略仍待完善 |
| checkpoint | 基础实现/占位 | 状态登记完整；后端当前以轻量实现替代完整 DCP 能力 |
| dataloader | stub | 当前返回 dummy dict dataset |
| FSDP2 manager | stub | `parallelize()` 接口存在，完整实现待接入 |
| AutoPipeline / PP | stub | 接口存在；PP runtime 尚未闭环 |
| PEFT、QAT/FP8、freeze | stub | 已预留在 infrastructure 的固定时序中 |

`examples/training_skeleton/` 是当前最可靠的端到端入口：它明确以 tiny local GPT-2、dummy data 和 stub infrastructure 证明骨架时序，而非声称完成大模型训练交付。

---

## 7. 推荐阅读与开发路径

### 7.1 先读什么

1. [训练骨架代码详解](../trainer/code_guides/training_skeleton_code_walkthrough.md)：理解 Recipe setup、step loop 和状态追踪。
2. [Dual-mode DTensor 详细设计](../trainer/detailed_design/05_dual_mode_dtensor_parallel_strategy.md)：理解 ShardingPlan、生产/校验双模式与 local region。
3. [分布式组件教程](trainer/components_distributed_tutorial.md)：从独立 `nn.Module` 使用 planner/applier。
4. [重构计划](../trainer/detailed_design/hyper_parallel_refactor_plan.md)：查看旧架构迁移目标和未完成项。

### 7.2 新能力放在哪里

| 需求 | 首选位置 |
|---|---|
| 新 YAML 字段或 `_target_` 组件 | `hyper_models/trainer/config.py` 与 `hyper_models/config/` |
| 新训练节奏或 step 外围行为 | `components/training/`，必要时注册 callback |
| 新优化器、loss、scheduler | `components/optim/`、`components/loss/` |
| 新并行布局、特殊通信或模型局部计算 | `components/distributed/` 的 plan override / local region 扩展点 |
| HF 模型加载或模型基础设施时序 | `hyper_models/_transformers/` |
| 某任务的训练流程 | `hyper_models/recipes/<task>/` |

不要在新模型中复制旧 `hyper_parallel/models/*/parallelize.py` 的大段 TP/CP/EP/FSDP 过程式逻辑。新模型应优先补充架构识别、参数角色覆盖和必要的 `ModuleShardingSpec` override；只有 DTensor 无法自然表达的局部计算才实现 `local_compute_fn`。

---

## 8. 近期落地建议

按依赖顺序，当前架构更适合先补齐：

1. 真实 `MeshContext`、FSDP2 manager 和 checkpoint/DCP，建立稳定的单机/多机恢复闭环。
2. 真实 dataloader、tokenizer、CP 数据切分和权重加载，替换 skeleton 的 dummy 输入。
3. 用一个原生 HF Transformer 完成 `HyperAutoModel -> planner -> applier -> FSDP2 -> Recipe` 的完整训练验证。
4. 在该闭环稳定后，再引入模型特有的 MoE、MLA、低精、融合算子等能力。

这样能让组件化与双模式 DTensor 先成为可信基础，而不是在数据、权重、FSDP 和模型契约尚未闭环时叠加更多训练特性。
