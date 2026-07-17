# Hyper-Parallel 重构方案 v3

> 基于 6 份详细设计文档的综合重构计划。
> 设计文档：[detailed_design/](detailed_design/) — 01~06 共 6 份，覆盖 HF 兼容层、数据管道、训练循环、Checkpoint、双模式 DTensor、分布式基础设施。

---

## 1. 重构的目的与动机

### 1.1 核心问题

当前 Hyper-Parallel 的训练流程有三个结构性缺陷：

| 问题 | 现状 | 根因 |
|------|------|------|
| **无法对接 HF 生态** | 自研 `ModelSpec` 注册协议 + 三层配置，无法 `from_pretrained("Qwen/Qwen3.5-4B")` | 缺少 HF `AutoModel` 兼容层 |
| **并行化硬编码** | 每模型独立 `parallelize.py`（实际 qwen3_5: 1367 行、moe: 2554 行，if-elif 分支决定 TP/CP/FSDP 顺序）<br>注：05 文档 §1 所述"~400 行"为 AutoModel 参考代码量，非 Hyper-Parallel 实际代码量 | 无统一的分片策略推导引擎 |
| **运行时通信判断** | 每次 forward 重复解析 ShardingConfig、动态 resolve mesh、条件判断 collective 类型 | 无编译期通信规划 |

### 1.2 重构目标

1. **HF 原生兼容**：`HyperAutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-4B")` 开箱即用，自定义模型使用 `_target_` IoC 声明式配置
2. **声明式并行**：ShardingPlanner 自动推导 DTensor 分片策略，7 种 `ShardingTemplate` + 14 种 `ParamRole` 覆盖 90% 模型，每模型从 ~1367 行（qwen3_5 实际）/~2554 行（moe 实际）降至 ~20 行
3. **编译期通信规划**：`PrecompiledBoundary` 在 `apply_sharding_plan()` 时一次性生成全部通信计划，运行时零条件判断，零 DTensor dispatch 开销
4. **双模式可切换**：单一 `validate_placement` 开关控制校验模式（DTensor 传播验证）和生产模式（build 期 `_local_params_context` 一次性解包 + `tp_grad_info` 旁路 + FSDP 管梯度 + PrecompiledBoundary 通信注入）
5. **Checkpoint 跨配置重分片**：保存 DTensor local shard + placements + mesh_dim_names 作为 DCP 元数据（`full_state_dict=False`），load 时按新 mesh 重分片

---

### 1.3 核心架构决策：并行组件与训练流程解耦

一个关键设计决策：**`hyper_models/components/distributed/` 是独立可用的，不依赖 Hyper-Parallel 的训练流程**。

```
┌─────────────────────────────────────────────────────────┐
│  Layer 3: 训练流程 (recipes/, _transformers/)             │
│    FinetuneRecipe、build_dataloader、Checkpointer        │
│    可选——用户可以用任意训练框架替换                         │
├─────────────────────────────────────────────────────────┤
│  Layer 2: 模型层 (hyper_models/components/models/)                    │
│    LlamaForCausalLM、Qwen3_5ForCausalLM                 │
│    零 DTensor 代码——分片策略外部注入                       │
├─────────────────────────────────────────────────────────┤
│  Layer 1: 并行组件 (hyper_models/components/distributed/)  ★ 独立可用  │
│    ShardingPlanner、ShardingApplier、PrecompiledBoundary  │
│    FSDP2Manager、MeshContext、_local_params_context       │
│    零依赖：不 import recipes/、_transformers/、models/     │
└─────────────────────────────────────────────────────────┘
```

**两种使用方式**：

```python
# 方式 A：全量训练流程（from_pretrained 一步到位）
# DistributedSetup 是 frozen dataclass（06 §3.3），经 build() 类方法构造
setup = DistributedSetup.build(
    strategy="fsdp2",
    parallelism_sizes=ParallelismSizes(tp_size=4))
model = HyperAutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-4B",
    distributed_setup=setup)

# 方式 B：只用并行组件，自己管理一切
from hyper_models.components.distributed import ShardingPlanner, apply_sharding_plan
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-4B")
plan = ShardingPlanner().plan(model, mesh.device_mesh, tp_size=4)
apply_sharding_plan(model, plan, mesh.device_mesh)
# model 已分片，可用 PyTorch Lightning / HF Trainer / 手写循环
```

**`hyper_models/components/distributed/` 的依赖边界**：

| 可以依赖 | 不可以依赖 |
|---------|----------|
| `torch`、`torch.distributed`、`DTensor` | `recipes/` |
| 自身数据结构（`ShardingPlan`、`ModuleShardingSpec`、`NamedPlacement`） | `_transformers/`（from_pretrained 等） |
| `shared/`（平台抽象、工具函数） | `hyper_models/components/models/`（任何具体模型） |
| `hyper_models/components/config/node.py`（ConfigNode，用于 `_target_` 场景；`loader.py` 仅含 `load_yaml_config`） | `hyper_models/components/datasets/`、`hyper_models/components/training/` |

---

## 2. 训练流程构建的哲学与思路

### 2.1 `_target_` IoC：YAML 即容器

核心思想：**YAML 配置文件本身就是一个轻量级 IoC 容器**。所有组件（模型、优化器、调度器、数据集、DataLoader、Tokenizer、Loss、Checkpoint）通过 `_target_: fully.qualified.ClassName` 声明类型，框架不硬编码任何 `if-else` 分支。

```
YAML 文件 (train.yaml)
    │  load_yaml_config()
    ▼
ConfigNode 树（所有 _target_ 和 *_fn 已即时解析为 Python callable）
    │  RecipeConfig(cfg)  ← 类型化桥接
    ▼
RecipeConfig（typed 属性已就绪，类型校验完成）
    │  cfg.xxx.instantiate(**runtime_kwargs)  ← untyped 组件
    │  cfg.xxx.build(**runtime_deps)          ← typed 组件
    ▼
Python 对象（模型、优化器、DataLoader...）
```

**两类消费路径**：

| 路径 | 组件 | 调用方式 | 原理 |
|------|------|---------|------|
| **untyped `.instantiate()`** | model, dataset, dataloader, tokenizer, collate, peft | `cfg.xxx.instantiate(**runtime)` | ConfigNode 的 key-value 1:1 映射到构造函数参数 |
| **typed `.build()`** | optimizer, lr_scheduler, step_scheduler, loss_fn, checkpoint | `cfg.xxx.build(**runtime_deps)` | 需要复杂的预处理（参数分组、model.parts 遍历），YAML 值无法直接作为构造函数参数 |

详见 [01_hf_compatibility_layer.md §2-4](detailed_design/01_hf_compatibility_layer.md)。

### 2.2 `from_pretrained` 统一入口

所有模型通过 `HyperAutoModelForCausalLM.from_pretrained()` 加载，内部自动完成：

1. **YAML 解析** → ConfigNode（`_target_` 即时解析）
2. **类型化桥接** → RecipeConfig
3. **分布式拓扑构建** → `DistributedSetup.build()` → MeshContext
4. **基础设施创建** → `instantiate_infrastructure()` → ShardingPlanner + FSDP2Manager
5. **模型架构分发** → `get_is_hf_model()` 查 MODEL_ARCH_MAPPING → 自定义模型 / HF 原生
6. **核心编排** → `_build_model()`：
   - meta device 空壳构建（零显存）
   - PEFT/QAT 注入（在分片之前）
   - ShardingPlanner 自动推导 ShardingPlan
   - apply_sharding_plan（DTensor 分片 + PrecompiledBoundary 编译）
   - 每 rank 独立读取 safetensors 权重（零 NCCL）
   - FSDP2 包裹（DP 维度）

详见 [01_hf_compatibility_layer.md §4-8](detailed_design/01_hf_compatibility_layer.md)。

### 2.3 Recipe.setup() 组件编排

`FinetuneRecipe.setup(cfg)` 是全部训练组件的唯一组装入口。18 步构建按依赖顺序执行，每个组件通过 `.instantiate()` 或 `.build()` 创建，新增组件类型只需修改 YAML 的 `_target_`，无需改动任何 Python 代码。

详见 [03_training_loop.md §5](detailed_design/03_training_loop.md)。

### 2.4 数据管道：`build_dataloader()` 一步到位

数据管道的全部构建在 `build_dataloader()` 一次调用中完成——tokenizer、dataset、sampler、packed sequence、collate、dataloader，通过 `cfg_ds._target_` 自动分发到 HF datasets 或 Megatron 路径。

详见 [02_data_pipeline.md §2-3](detailed_design/02_data_pipeline.md)。

---

## 3. DTensor 使用的哲学与思路

### 3.1 核心原则：不侵入模型代码

DTensor 分片策略**完全外部注入**。HF 模型零改动——`ModuleShardingSpec` 以声明式配置的形式描述每个模块边界的 I/O 契约（`in_src`/`in_dst`/`out_src`/`out_dst`）和参数 placement。运行时直接按声明执行，不做推断。

详见 [05_dual_mode_dtensor_parallel_strategy.md §1-3](detailed_design/05_dual_mode_dtensor_parallel_strategy.md)。

### 3.2 ShardingPlanner：从 ~1367/~2554 行到 ~20 行

ShardingPlanner 通过 6-phase 推导管线自动生成 `ShardingPlan`：

```
Phase 1: ParameterClassifier.classify(model)
    ├─ 命名规则匹配（14 种 ParamRole，按参数名后缀识别）
    └─ 架构规则覆盖（ARCH_OVERRIDES / MODEL_FAMILIES）

Phase 2: BoundaryGrouper.group(model)
    └─ 每个 transformer layer → attention + mlp + norm 边界

Phase 3: SemanticRoleInference.infer()
    └─ 从参数角色推断模块类型（attention / mlp / norm / embed / lm_head）

Phase 4: TemplateLookup.lookup(boundary_type)
    └─ 查 ShardingTemplate → _build_spec_from_template() → ModuleShardingSpec

Phase 5: ChainPropagator.propagate(specs)
    └─ 链式传播：填充缺省 in_src + 检测模板错误 + 处理首/尾模块

Phase 6: SpecialHandler.apply()
    └─ 特殊参数自定义分片（gated_delta 等；fused_qkv 已在 Phase 4 TemplateLookup 中处理）
```

`ParamRole`（14 个枚举值）是命名规则与 `ShardingTemplate` 之间的桥梁——它把"参数叫什么"映射到"参数放在 mesh 的哪个维度"。14 个 Role 最终映射到仅 4 个 placement 字段：`colwise_placement`(Shard(0))、`rowwise_placement`(Shard(1))、`norm_placement`(Replicate)、`moe_expert_placement`。

详见 [05_dual_mode_dtensor_parallel_strategy.md §3.5-3.6](detailed_design/05_dual_mode_dtensor_parallel_strategy.md)。

### 3.3 双模式：validate 校验 + production 零开销

双模式共用 Phase A/B（参数分片 + 边界编译），仅在 Phase C 分叉。**关键变更（与附录 A.2 一致）**：production 模式下 `_local_params_context` 在 `apply_sharding_plan` 内 **build 期一次性解包**（`fully_shard` 之前调用，永久不恢复），参数此后由 FSDP 以 `LOCAL_PARAM` + `tp_grad_info` 管理；forward 期不再做参数 swap，仅执行预编译通信。

```
┌──────────────────────────────────────────────────────────┐
│              apply_sharding_plan(model, plan, mesh)        │
│                                                          │
│  Phase A: _shard_module_params()                           │
│    distribute_tensor() → 参数转换为 DTensor (meta 上)       │
│                                                          │
│  Phase B: PrecompiledBoundary(spec, mesh)                  │
│    比较 src vs dst placement → 生成 RedistOp 序列           │
│    所有通信统一走 DTensor.redistribute()（NCCL 自动选择）    │
│                                                          │
│  Phase C: 双模式分叉                                        │
│                                                          │
│    if validate_mode:                                      │
│      _wrap_validate_forward():                            │
│        DTensor 传播 → 记录实际 out_src                        │
│        assert actual_out_src == spec.out_src                │
│        assert actual_out_dst == spec.out_dst（仅终端模块）    │
│                                                          │
│    else (production mode):                                │
│      _local_params_context(model)  ← build 期一次性解包      │
│        DTensor._local_tensor → nn.Parameter（永久替换）     │
│        返回 tp_grad_records（FQN → TP placement）           │
│      build_tp_grad_info(plan, tp_mesh)                     │
│        从 ShardingPlan.modules[fqn].spec.params 读取        │
│        （production 下参数已无 DTensor 可读）                │
│      _wrap_production_forward():                          │
│        PrecompiledBoundary.pre_forward(x) → in_dst          │
│        original_forward(x_local)  ← 纯 local tensor 计算    │
│        PrecompiledBoundary.post_forward(y) → out_dst        │
│                                                          │
│  返回 (model, tp_grad_info) → 供 fully_shard 使用           │
└──────────────────────────────────────────────────────────┘
```

**production 模式的梯度链路**：参数解包为 plain local tensor 后，由 FSDP2 `fully_shard` 以 `LOCAL_PARAM` + `tp_grad_info` 管理——DP 维度走 HSDP layout-driven 梯度同步，TP 维度由 `tp_grad_info` 旁路触发 `all_reduce`（复用 `core/fully_shard/` 现有机制）。参数不再持有 DTensor，forward/backward 全程零 DTensor dispatch。

**validate 模式的关键设计决策**：中间模块的 `out_dst` 校验是冗余的——链式传播已保证 `A.out_dst == B.in_src`，只需校验终端模块的 `out_dst`（通过 `_is_terminal` 标记）。

详见 [05_dual_mode_dtensor_parallel_strategy.md §4-5](detailed_design/05_dual_mode_dtensor_parallel_strategy.md) 与附录 A.2。

### 3.4 编译期通信规划：PrecompiledBoundary

传统方式每次 forward 都要做：解析 ShardingConfig → 比较 src/dst placement → 判断 collective 类型 → 执行。`PrecompiledBoundary` 将这一切提升到编译期：

- **编译期**（`apply_sharding_plan` 时）：遍历所有 `ModuleShardingSpec`，为每个模块边界生成 `RedistOp` 序列
- **运行时**（每次 forward）：直接遍历预编译的 `RedistOp` 列表执行 `DTensor.redistribute()`
- `RedistOp.collective_type` 是调试/分析标签（`identity` / `all_gather` / `reduce_scatter` / `redistribute`），实际通信由 DTensor 内部自动选择最优 NCCL collective

### 3.5 与 FSDP2 的层叠关系

DTensor（TP/CP/SP）和 FSDP2（DP）是**正交的并行维度**，按固定顺序层叠：

```
Layer 3: FSDP2 (DP)      ← fsdp2_manager.parallelize() 最后应用
Layer 2: DTensor (TP/CP)  ← apply_sharding_plan() 先应用
Layer 1: nn.Module        ← 原始 HF 模型
```

详见 [06_distributed_infrastructure.md §4.2](detailed_design/06_distributed_infrastructure.md)。

---

---

## 4. AI 辅助开发模式

### 4.1 分工原则

6 份详细设计文档已覆盖全部模块的**接口签名、数据结构、调用时序和关键实现代码**。基于这份详尽的设计，AI Agent 可以独立完成大部分编码和测试工作。人工聚焦于三个高价值环节：

| 环节 | AI 负责 | 人工负责 |
|------|---------|---------|
| **编码实现** | 根据设计文档中的接口签名和代码骨架，批量生成完整实现 | 方案验收——确认 AI 产出的代码与设计意图一致 |
| **测试编写** | 根据验收标准批量生成单测和集成测试 | 用例调试——运行测试，分析失败原因（可借助 AI），修复边界 case |
| **文档更新** | 根据代码变更自动更新 API 文档和配置参考 | 最终兜底——确认所有验收标准通过，签署发布 |

### 4.2 工时折算

| 任务类型 | 纯人工 | AI 辅助 | 折算比 | 说明 |
|---------|:-----:|:-----:|:-----:|------|
| 机械性编码（目录迁移、import 替换、模板代码） | 1x | **0.15x** | ~7:1 | Agent 批量处理，人工 diff 验收 |
| 逻辑编码（根据设计文档实现函数/类） | 1x | **0.25x** | ~4:1 | Agent 生成，人工 review 逻辑正确性 |
| 测试编写（单测 + 集成测试 + roundtrip） | 1x | **0.2x** | ~5:1 | Agent 生成 pytest 用例，人工补边界 |
| 调试/集成（失败用例排查、跨模块联调） | 1x | **0.5x** | ~2:1 | AI 辅助分析，人工做最终判断 |
| 架构/设计（方案评审、设计文档） | 1x | **0.8x** | ~1.25:1 | 人主导，AI 辅助文档和图表 |

### 4.3 总工时对比

| 指标 | 纯人工 | AI 辅助 |
|------|:-----:|:-----:|
| 总人·日 | 240 | **~80** |
| 关键路径 | 120 人·日 | **~40 人·日** |
| 1 人全职 | ~12 月 | **~4 月** |
| 2 人并行 | ~6 月 | **~2.5 月** |
| 3 人并行 | ~4 月 | **~2 月** |

> 注：§4.3 关键路径 ~40 人·日（AI 辅助）是取整估算值；§6.2 精确计算为 42.5 人·日（M_A→M_B→M_C→M_E→M_F→M_H→M_L→M_M），因 M_K（高级特性 PP/AC/混合精度）与关键路径并行开发，不阻塞 MVP（M3 端到端训练）。上下浮动约 ±3 人·日，取决于并行开发重合度。


## 5. 需求分解（按 `main()` 调用时序编排）

> 工时单位：人·日（AI 辅助，1 人·日 ≈ 8h，含 AI 编码+人工验收+调试+集成测试）

需求按 `main()` 的调用顺序组织——从 YAML 加载到训练循环退出，每个模块对应调用树中的一个或多个节点。每个模块需求展开为子需求表，子需求标注对应的设计文档章节。

```
main() 调用时序                         对应模块
══════════════════════════════════════════════════════════
① load_yaml_config()          ──→  M_A  ConfigNode / RecipeConfig
② RecipeConfig(cfg)           ──→  (同上)
③ recipe = FinetuneRecipe()   ──→  M_H  训练循环
④ recipe.setup(cfg)           ──→  (以下全部)
  ④.1 initialize_distributed  ──→  M_B  分布式基础设施
  ④.2 DistributedSetup.build  ──→  (同上)
  ④.3 self.mesh               ──→  M_B  分布式基础设施 (MeshContext 赋值)
  ④.4 model = from_pretrained ──→  M_C  HF 兼容层
    ④.4.1 infrastructure       ──→  M_B  分布式基础设施 (FSDP2Manager)
    ④.4.2 MODEL_ARCH_MAPPING   ──→  M_C  HF 兼容层
    ④.4.3 _init_model          ──→  M_C  HF 兼容层
    ④.4.4 PEFT 注入            ──→  M_C  HF 兼容层
    ④.4.5 参数冻结             ──→  M_C  HF 兼容层
    ④.4.6 sharding_planner     ──→  M_D  双模式 DTensor
    ④.4.7 apply_sharding       ──→  M_D  双模式 DTensor
    ④.4.8 load_base_model      ──→  M_E  权重加载
    ④.4.9 fsdp2.parallelize    ──→  M_B  分布式基础设施
  ④.5 loss_fn                  ──→  M_G  Optimizer / Loss
  ④.6 checkpointer             ──→  M_F  Checkpoint
  ④.7 optimizer                ──→  M_G  Optimizer / Loss
  ④.8 build_dataloader         ──→  M_I  数据管道
  ④.9 step_scheduler           ──→  M_H  训练循环
  ④.10 lr_scheduler            ──→  M_G  Optimizer / Loss
  ④.11 load_checkpoint         ──→  M_F  Checkpoint
⑤ run_train_validation_loop    ──→  M_H  训练循环
  ⑤.1 _run_train_optim_step    ──→  M_H  训练循环
  ⑤.1.2 _forward_backward      ──→  M_H  训练循环 + M_D PrecompiledBoundary
══════════════════════════════════════════════════════════
                                外加: M_J  模型实现 (Llama/Qwen3.5/MoE)
                                      M_K  高级特性 (PP/AC/混合精度)
                                      M_L  CLI/监控
                                      M_M  测试/文档/迁移指南
```

> 注：调用树步骤编号存在跳跃（如 ⑤.1.1 缺失），因为本树是简化视图——合并/省略了若干中间步骤，并非笔误。编号归属见下方"编号统一约定"。

---

### 编号统一约定

**canonical 编号来源**：[01_hf_compatibility_layer.md §4.1](01_hf_compatibility_layer.md) 的 `main()`/`setup()` 时序图是全部文档中步骤编号（①②③④⑤…）的**唯一 canonical 来源**。本计划 §5 调用树、05/06 文档内部时序图各自使用简化编号；凡跨文档引用步骤编号，一律以 01 的时序图为准，其余编号通过下表对照换算。

本计划 §5 调用树编号 → 01 canonical 编号对照表：

| 本计划 §5 | 01 canonical | 步骤 |
|:---------|:------------|------|
| ①②③ | ①②③ | load_yaml_config / RecipeConfig / FinetuneRecipe |
| ④.1 | ④.1 | initialize_distributed |
| ④.2 | ④.3 | create_distributed_setup_from_config（DistributedSetup.build） |
| ④.3 | ④.4.1 | mesh = distributed_setup.mesh_context |
| ④.4 | ④.4 | model = cfg.model.instantiate → from_pretrained |
| ④.4.1 | ④.4.2 | infrastructure（ShardingPlanner / FSDP2Manager / AutoPipeline） |
| ④.4.2 | ④.4.4 | get_is_hf_model（MODEL_ARCH_MAPPING 查表） |
| ④.4.3 | ④.4.5.2 | _init_model（meta device 空壳构建） |
| ④.4.4 | ④.4.5.3 | PEFT 注入（_apply_peft） |
| ④.4.5 | ④.4.5.6 | 参数冻结（_apply_parameter_freezing） |
| ④.4.6 | ④.4.5.7 | sharding_planner.plan() |
| ④.4.7 | ④.4.5.8 | apply_sharding_plan() |
| ④.4.8 | ④.4.5.11 | load_base_model（01 已统一定稿：④.4.5.11 = load_base_model，前置 model.to_empty 物化为同一步内动作） |
| ④.4.9 | ④.4.5.10 | fsdp2_manager.parallelize() |
| ④.5 | ④.5 | loss_fn |
| ④.6 | ④.7 | checkpointer |
| ④.7 | ④.8 | optimizer |
| ④.8 | ④.9 | build_dataloader |
| ④.9 | ④.11 | step_scheduler |
| ④.10 | ④.12 | lr_scheduler |
| ④.11 | ④.13 | load_checkpoint |
| ⑤ | ⑤ | run_train_validation_loop |

> 06 文档 §2 时序图使用其内部简化编号（如 ④.2=DistributedSetup.build、④.4=FSDP2Manager、
> ④.5.1-④.5.3=plan/apply/parallelize），为简化视图，不与上表逐级对应；
> canonical 编号一律以 01 §4.1 为准，跨文档引用时按上表口径换算。

---

### M_A · ConfigNode / RecipeConfig 配置系统

> **调用位置**: ①② — `main()` 最前两步，所有后续模块的基础
> **设计文档**: [01 §2-3](01_hf_compatibility_layer.md)

| # | 子需求 | 说明 | 工时(日) | 设计文档 |
|---|--------|------|:------:|---------|
| M_A.1 | `ConfigNode.__init__` + `_wrap()` | dict→ConfigNode 递归包装，`_target_` 和 `*_fn` 即时解析（eager resolution） | 1 | 01 §2.3 |
| M_A.2 | `_resolve_target()` | `"torch.optim.AdamW"` → `<class AdamW>`，支持点分隔导入和 `file.py:attr` | 0.5 | 01 §2.4 |
| M_A.3 | `translate_value()` | YAML 字符串 → Python 原生类型（`ast.literal_eval`） | 0.5 | 01 §2.5 |
| M_A.4 | `ConfigNode.instantiate()` + `_instantiate_value()` | 递归实例化嵌套 `_target_` | 1 | 01 §2.9-2.10 |
| M_A.5 | `ConfigNode` 辅助方法 | `get()` / `get_as_string()` / `to_dict()` / `to_yaml_dict()` / `instantiate_path()` | 0.5 | 01 §2.7/2.11 |
| M_A.6 | 安全模型 | `ALLOWED_IMPORT_PREFIXES` + `_is_allowed_module()` + `ENABLE_USER_MODULES` | 0.5 | 01 §2.12 |
| M_A.7 | `RecipeConfig` typed 属性 | `_callable_and_kwargs()` + optimizer/scheduler/loss/checkpoint cached_property + `__getattr__` untyped 透传 | 1 | 01 §3 |
| | **小计** | | **5** | |

---

### M_B · 分布式基础设施

> **调用位置**: ④.1 / ④.2 / ④.4.1 / ④.4.9 — 拓扑构建 → FSDP2 包裹
> **设计文档**: [06_distributed_infrastructure.md](06_distributed_infrastructure.md)

| # | 子需求 | 说明 | 工时(日) | 设计文档 |
|---|--------|------|:------:|---------|
| M_B.1 | `ParallelismSizes` | 用户声明并行度 dataclass（tp/cp/pp/dp/ep） | 0.5 | 06 §3.1 |
| M_B.2 | `MeshContext.build()` | 从 sizes + strategy → 构建 `DeviceMesh`，自动推导 dp_size | 1 | 06 §3.2 |
| M_B.3 | `MeshContext` properties | `tp_size`/`cp_size`/`dp_size`/`tp_rank` 从 DeviceMesh 实时读取 | 0.5 | 06 §3.2 |
| M_B.4 | `DistributedSetup.build()` | 统一拓扑容器，整合 MeshContext + strategy_config + pipeline_config | 0.5 | 06 §3.3 |
| M_B.5 | `FSDP2Config` | `sequence_parallel`、`activation_checkpointing`、`mp_policy`、`offload_policy` 等 | 0.5 | 06 §4.1 |
| M_B.6 | `FSDP2Manager.parallelize()` | `fsdp2_strategy_parallelize()` — 在 DTensor 分片后包裹 DP 维度 | 1 | 06 §4.2 |
| M_B.7 | `_local_params_context` ✅ 已交付 | build 期一次性解包 DTensor→plain（`fully_shard` 前调用），永久替换——实现于 `hyper_models/components/distributed/sharding/apply.py` | 0.5 | 06 §5 |
| M_B.7a | `tp_grad_info` 机制 + `build_tp_grad_info` ✅ 已交付 | TP 维度梯度 all-reduce 旁路——复用现有 HSDP layout-driven 梯度同步，新增 `tp_grad_info` 记录 TP placement 信息——实现于 `hyper_models/components/distributed/tp_grad.py` | 1 | 06 §1.2, 05 §6.7 |
| M_B.8 | 单元测试 | Mesh 构建 + FSDP2 包裹 + 零拷贝验证 | 0.5 | — |
| | **小计** | （已交付 1.5，剩余 4.5） | **6** | |

---

### M_C · HF 兼容层

> **调用位置**: ④.4 — `from_pretrained` → `_build_model` 完整链路
> **设计文档**: [01 §5-7](01_hf_compatibility_layer.md)

| # | 子需求 | 说明 | 工时(日) | 设计文档 |
|---|--------|------|:------:|---------|
| M_C.1 | `MODEL_ARCH_MAPPING` | `OrderedDict[arch_name → (module_path, class_name)]` + `_resolve_custom_model_cls()` 懒加载 | 0.5 | 01 §5 |
| M_C.2 | `get_is_hf_model()` | 查表判定 → 自定义模型 / HF 原生降级 | 0.5 | 01 §5 |
| M_C.3 | `_BaseHyperAutoModelClass` | 多重继承 HF AutoModel + `from_pretrained`/`from_config`/`_build_model` 类方法 | 1.5 | 01 §6.1 |
| M_C.4 | `HyperAutoModel*` 类族 | `ForCausalLM`、`ForImageTextToText`、`ForSequenceClassification` | 0.5 | 01 §6.1 |
| M_C.5 | `from_pretrained` 完整实现 | 参数列表（distributed_setup/device_mesh/torch_dtype/attn_implementation/validate_placement/compile_config 等） | 1 | 01 §6.2 |
| M_C.6 | `_build_model()` 核心编排 | 11 步 canonical meta 链路：meta device → PEFT → freeze → ShardingPlanner.plan → apply_sharding_plan（含 `_local_params_context` build 期解包 + build_tp_grad_info，返回 tp_grad_info）→ `fully_shard`（meta, tp_grad_info）→ `to_empty` → `load_base_model` → PP hooks（原"CP hooks"已取消——D-01''：CP K/V all-gather 在 apply_sharding_plan 内编译期注入，01 §8.3 ⑩） | 1.5 | 01 §6.3, 05 §4.1 |
| M_C.7 | `_init_model()` | 自定义模型 / HF 原生路径分发 + meta device 空壳构建 | 1 | 01 §7 |
| M_C.8 | PEFT 注入（`_apply_peft`） | 在分片之前插入 LoRA 层 | 0.5 | 01 §6.4 |
| M_C.9 | 参数冻结（`_apply_parameter_freezing`） | `FreezeConfig` + 在分片之前应用 | 0.5 | 01 §6.5 |
| M_C.10 | 集成测试 | Llama 3.2 1B 单 GPU `from_pretrained` → 推理输出与 HF 一致 | 0.5 | — |
| | **小计** | | **8** | |

---

### M_D · 双模式 DTensor 核心

> **状态**: ✅ **已完成**（2026-07）。已交付符号：`ShardingPlanner` / `apply_sharding_plan`（`sharding_applier.py`）、`PrecompiledBoundary`（`precompiled_boundary.py`）、`build_tp_grad_info`（`tp_grad.py`）、`_local_params_context`（`sharding/apply.py`），全部位于 `hyper_models/components/distributed/`；`tests/components/distributed/` 300 个用例全绿。
> **调用位置**: ④.4.6-7（编译期规划 + 运行时应用）+ ⑤.1.2（训练时 PrecompiledBoundary 执行）
> **设计文档**: [05_dual_mode_dtensor_parallel_strategy.md](05_dual_mode_dtensor_parallel_strategy.md)

| # | 子需求 | 说明 | 工时(日) | 设计文档 |
|---|--------|------|:------:|---------|
| M_D.1 | `NamedPlacement` + `resolve_placements()` | `{TP: Shard(0)}` → `[Replicate(), Shard(0)]` | 0.5 | 05 §3.2.1 |
| M_D.2 | `ModuleShardingSpec` | `params` + `in_src`/`in_dst`/`out_src`/`out_dst` + `is_boundary` + `_is_terminal` | 1 | 05 §3.2 |
| M_D.3 | `ShardingPlan` | `modules: dict[str, ModuleShardingSpec]` + `special_handlers` + `mesh_dim_names` | 0.5 | 05 §3.1 |
| M_D.4 | `ShardingTemplate` + `TEMPLATES` | 7 种模板（attention/mlp/norm/embed/lm_head/moe_gate/moe_mlp）× SP/non-SP | 1 | 05 §3.5 |
| M_D.5 | `_build_spec_from_template()` | 14 种 ParamRole → 4 个 placement 字段映射 | 1 | 05 §3.6.3.1 |
| M_D.6 | `ParameterClassifier`（Phase 1） | 命名规则匹配 + `ARCH_OVERRIDES` 架构覆盖 | 1.5 | 05 §3.6.1 |
| M_D.7 | `BoundaryGrouper`（Phase 2） | transformer layer → attention + mlp + norm 边界 | 0.5 | 05 §3.6 |
| M_D.8 | `SemanticRoleInference`（Phase 3） | FQN 模式 + ParamRole 组合 → boundary_type | 0.5 | 05 §3.6.2 |
| M_D.9 | `TemplateLookup`（Phase 4） | 查 ShardingTemplate → `_build_spec_from_template()` | 0.5 | 05 §3.6.3.1 |
| M_D.10 | `ChainPropagator`（Phase 5） | 填充缺省 in_src + 检测模板错误 + 处理首/尾模块 | 1.5 | 05 §3.6.5 |
| M_D.11 | `SpecialHandler`（Phase 6） | fused_qkv 合并、gated_delta 等自定义分片 | 1 | 05 §3.6.6 |
| M_D.12 | `ShardingPlanner.plan()` 主入口 | 串联 6 个 Phase | 0.5 | 05 §3.6.6 |
| M_D.13 | `_shard_module_params()`（Phase A） | `distribute_tensor()` → DTensor，支持 meta tensor | 0.5 | 05 §4.2 |
| M_D.14 | `PrecompiledBoundary`（Phase B） | `_compile_input_plan()`/`_compile_output_plan()` + `RedistOp` | 2 | 05 §4.3 |
| M_D.15 | `_wrap_production_forward()`（Phase C 生产） | `boundary.pre_forward` → `forward` → `boundary.post_forward`（参数已由 `_local_params_context` 提前解包为 plain tensor） | 0.5 | 05 §4.4 |
| M_D.15a | 梯度等价性测试 | 校验模式（DTensor 全传播）vs 生产模式（`_local_params_context` + HSDP 梯度同步）输出梯度一致——验证 `tp_grad_info` 旁路正确性 | 1 | 05 §6.7 |
| M_D.16 | `_wrap_validate_forward()`（Phase C 校验） | DTensor 传播 + `out_src` 校验 + 终端模块 `out_dst` 校验 | 0.5 | 05 §5 |
| M_D.17 | `PlacementMismatchError` | 含 module_name、expected、actual、stage（简单异常类，随 M_D.16 一并交付，工时独立计列） | 0.5 | 05 §5 |
| | **小计** | （15 人日全部已交付） | **15** | |

---

### M_E · StateDictAdapter + 权重加载

> **调用位置**: ④.4.8 — `load_base_model()` — 每 rank 独立读磁盘，零 NCCL
> **设计文档**: [01 §10](01_hf_compatibility_layer.md) + [04 §5.3-5.4](04_checkpoint.md)

| # | 子需求 | 说明 | 工时(日) | 设计文档 |
|---|--------|------|:------:|---------|
| M_E.1 | `ConversionMapping` | 合并预定义映射 + 模型自定义映射 | 0.5 | — |
| M_E.2 | `Qwen3_5DenseStateDictAdapter` | `_fp32_params.A_log` ↔ `A_log` + MTP 重命名 | 1 | 01 §10.2 |
| M_E.3 | `Qwen3_5MoeStateDictAdapter` + VLM adapter | MoE expert 合并 + VLM 层级映射 | 1 | — |
| M_E.4 | `_load_hf_checkpoint_preserving_dtype()` | 所有 rank 并行读 safetensors → 全量 CPU state dict | 0.5 | 04 §5.4 |
| M_E.5 | `_load_full_state_dict_into_model()` | `set_model_state_dict(full_state_dict=True)`，不设 broadcast_from_rank0 | 0.5 | 04 §5.4 |
| M_E.6 | roundtrip 测试 | save→load 一致 + 跨 TP 配置加载 | 0.5 | — |
| | **小计** | | **4** | |

---

### M_F · Checkpoint

> **调用位置**: ④.6（创建 Checkpointer）+ ④.11（断点续训恢复）+ ⑤ `save_checkpoint()`（训练中保存）
> **设计文档**: [04_checkpoint.md](04_checkpoint.md)

| # | 子需求 | 说明 | 工时(日) | 设计文档 |
|---|--------|------|:------:|---------|
| M_F.1 | `CheckpointingConfig` | 含 `model_state_dict_keys`、`diffusers_compatible`、`original_model_root_dir` 等完整字段 | 0.5 | 04 §4 |
| M_F.2 | `Checkpointer.__init__()` | StorageWriter/Reader + Addons 注册 + 异步 stager | 0.5 | 04 §5.1 |
| M_F.3 | `Checkpointer.save_model()` | 5 阶段：ModelState→Adapter.to_hf→index mapping→dcp.save→consolidate | 1 | 04 §5.2 |
| M_F.4 | `Checkpointer.load_model()` | 3 路径：MoE tensor merging / Safetensors fast path / DCP resume + `key_mapping` | 1 | 04 §5.3 |
| M_F.5 | `load_base_model()` | 单一 `path` 参数签名 + 本地缓存 fallback（`_get_hf_safetensors_reference_path` 负责 `root_dir`+`model_name` 解析） | 0.5 | 04 §5.3 |
| M_F.6 | `ModelState` / `OptimizerState` | DCP 兼容 + tied weights + PEFT 处理 | 0.5 | 04 §6 |
| M_F.7 | `_extract_dtensor_metadata()` | 可观测性层（调试/审计/ShardingPlan diff） | 0.5 | 04 §7 |
| M_F.8 | 异步保存 + 故障恢复 | `dcp.async_save()` + `DistributedSignalHandler` + LATEST symlink + `load_checkpoint()` 6 状态恢复 | 1 | 04 §5.5/§8 |
| | **小计** | | **5.5** | |

---

### M_G · Optimizer / LR Scheduler / Loss

> **调用位置**: ④.5（loss_fn）+ ④.7（optimizer）+ ④.10（lr_scheduler）+ ⑤.1.2（calculate_loss）
> **设计文档**: [03 §9-10](03_training_loop.md)

| # | 子需求 | 说明 | 工时(日) | 设计文档 |
|---|--------|------|:------:|---------|
| M_G.1 | `OptimizerConfig` 子类体系 | `AdamWConfig` + `OptimizerFromFactoryConfig`（escape hatch） | 0.5 | 03 §9.3 |
| M_G.2 | `build_optimizer_config()` | factory + kwargs → OptimizerConfig 归一化入口 | 0.5 | 03 §9.4 |
| M_G.3 | `OptimizerConfig.build(model)` | 参数分组（decay/no_decay）+ model.parts 遍历 + 实例化 | 0.5 | 03 §9.5 |
| M_G.4 | `LRSchedulerConfig` + `OptimizerParamScheduler` | step-based 设计（与 AutoModel checkpoint 兼容） | 0.5 | 03 §9.6 |
| M_G.5 | `calculate_loss()` dispatcher | `FusedLinearCrossEntropy` 路径 + 标准 logit-based 路径 | 0.5 | 03 §10 |
| M_G.6 | `WarmupCosineScheduler` (ratio-based wrapper) | 可选的便利包装：`warmup_steps_ratio` → 绝对步数 | 0.5 | 03 §9.6 |
| | **小计** | | **3** | |

---

### M_H · 训练循环

> **调用位置**: ③/④/⑤ — `FinetuneRecipe()` → `setup()` → `run_train_validation_loop()` → `_run_train_optim_step()` → `_forward_backward_step()`
> **设计文档**: [03_training_loop.md](03_training_loop.md)

| # | 子需求 | 说明 | 工时(日) | 设计文档 |
|---|--------|------|:------:|---------|
| M_H.1 | `BaseRecipe.__setattr__` | 自动注册 stateful 组件到 `__state_tracked` | 0.5 | 03 §3 |
| M_H.2 | `BaseRecipe.save/load_checkpoint()` | 遍历 `__state_tracked` 分类保存/恢复 | 0.5 | 03 §3 |
| M_H.3 | `StepScheduler` | grad_acc 分组 + ckpt/val/log step 判断 + SIGTERM + state_dict/load_state_dict | 1 | 03 §4 |
| M_H.3a | `TrainingCallback` / `StepState` / `CallbackManager` | 混合 Callback 系统：StepState frozen dataclass、3 个回调点（on_train_begin/on_step_end/on_train_end）、CallbackManager 注册与调用 | 1 | 03 §4.2 |
| M_H.3b | 内置 Callback 实现 | `CheckpointCallback` / `EvaluateCallback` / `LoggingCallback` / `TqdmCallback` / `WandbCallback` / `GCCallback` / `SIGTERMHandler` | 1 | 03 §4.2 |
| M_H.3c | `build_callback_manager()` | 根据 cfg 自动注册内置 callback 的工厂函数 | 0.5 | 03 §4.2 |
| M_H.4 | `FinetuneRecipe.setup()` | 18 步组件构建（含 `build_callback_manager` 注册 Callback） | 1 | 03 §5 |
| M_H.5 | `run_train_validation_loop()` | 混合方案：核心训练显式 + 外围关注点通过 `callback_manager.on_step_end(StepState)` 驱动 | 0.5 | 03 §6 |
| M_H.6 | `_run_train_optim_step()` | 三阶段：统计 token → 梯度累积 → clip+step+scheduler | 0.5 | 03 §7 |
| M_H.7 | `_forward_backward_step()` | batch→GPU + CP 准备 + forward + calculate_loss + backward | 0.5 | 03 §8 |
| M_H.8 | 集成测试 | Qwen3.5-0.8B 8 GPU 1000 步 loss 正常下降 + 断点续训 | 0.5 | — |
| | **小计** | | **7** | |

---

### M_I · 数据管道

> **调用位置**: ④.8 — `build_dataloader()` 一步到位
> **设计文档**: [02_data_pipeline.md](02_data_pipeline.md)

| # | 子需求 | 说明 | 工时(日) | 设计文档 |
|---|--------|------|:------:|---------|
| M_I.1 | `build_dataloader()` 主流程 | 7 步：tokenizer → dataset → 分片 → packing → sampler → collate → dataloader | 1 | 02 §3 |
| M_I.2 | `_build_tokenizer()` | 4 路分发（无 key / null / 无 `_target_` / 有 `_target_`） | 0.5 | 02 §4 |
| M_I.3 | HF datasets 路径 | `datasets.load_dataset()` + `FirstRankPerNode` 下载 | 0.5 | 02 §3 Step2 |
| M_I.4 | Megatron 路径 | `MegatronPretraining` 封装 + `create_megatron_sampler()` | 0.5 | 02 §6 |
| M_I.5 | Packed Sequence（THD + NEAT） | bin-packing + `seq_lens`/`seq_lens_padded`/`qkv_format="thd"` 元数据 | 0.5 | 02 §7 |
| M_I.6 | `StatefulDistributedSampler` | DP rank 分片 + state_dict/load_state_dict 断点续训 | 0.5 | 02 §7.2 |
| M_I.7 | Collate（per-key pad + PP 链式包装） | `___PAD_TOKEN_IDS___` + `_make_pp_collate()` | 0.5 | 02 §5 |
| | **小计** | | **4** | |

---

### M_J · 模型实现

> **调用位置**: ④.4.3 — `_init_model()` → `_resolve_custom_model_cls()` 懒加载
> **设计文档**: [01 §12](01_hf_compatibility_layer.md)、[05 §10](05_dual_mode_dtensor_parallel_strategy.md)（新模型上线流程）

| # | 子需求 | 说明 | 工时(日) | 设计文档 |
|---|--------|------|:------:|---------|
| M_J.1 | `LlamaForCausalLM` | 继承 `HFCheckpointingMixin` + HF 同名类 + BackendConfig | 1 | 01 §12 |
| M_J.2 | 注册 Llama 到 `MODEL_ARCH_MAPPING` | 一行注册 | 0.5 | 01 §12 |
| M_J.3 | `Qwen3_5ForCausalLM` | GatedDeltaNet + MTP 特殊逻辑 | 1.5 | 05 §10 |
| M_J.4 | Qwen3.5 `ARCH_OVERRIDES` + `SpecialHandler` | GatedDeltaNet SPECIAL + `gated_delta_tp_shard` | 1 | 05 §10 |
| M_J.5 | `Qwen3_5MoeForConditionalGeneration` | MoE expert 合并 + EP 分片 | 1.5 | — |
| M_J.6 | `MoEFSDPSyncMixin` | FSDP2 下 expert 梯度同步 | 0.5 | — |
| M_J.7 | 端到端训练 + HF 一致性测试 | 3 模型 TP=2/4 100 步 + 输出容差 1e-5 | 2 | — |
| | **小计** | | **8** | |

---

### M_K · 高级特性

> **调用位置**: ④.4.1（PP）+ ④.4.9（AC/混合精度）
> **设计文档**: [01 §8.3](01_hf_compatibility_layer.md)

| # | 子需求 | 说明 | 工时(日) | 设计文档 |
|---|--------|------|:------:|---------|
| M_K.1 | `AutoPipeline` | 按 transformer layers 均分切分 + `PipelineStage` | 1.5 | 01 §8.3 |
| M_K.2 | `ScheduleGPipe`/`Schedule1F1B` | PP 调度策略 | 1 | — |
| M_K.3 | PP collate 链式包装 | `add_causal_masks_to_batch()` | 0.5 | 02 §5 |
| M_K.4 | Activation Checkpointing | `checkpoint_wrapper` 每层 + full/selective 开关 | 1 | — |
| M_K.5 | `MixedPrecisionPolicy` | `param_dtype`/`reduce_dtype`/`output_dtype` | 0.5 | — |
| M_K.6 | fp32 参数隔离 | `_keep_in_fp32_modules` + `cast_model_to_dtype` skip | 0.5 | — |
| M_K.7 | 端到端测试 | PP=2/4 loss 一致 + AC 显存降低 >30% | 1 | — |
| | **小计** | | **6** | |

---

### M_L · CLI / 监控 / 调试

> **调用位置**: main() 入口前（CLI 解析）+ ⑤ 训练循环中（日志/MFU）
> **设计文档**: —

| # | 子需求 | 说明 | 工时(日) |
|---|--------|------|:------:|
| M_L.1 | `cli/app.py` | `hyper-parallel config.yaml --train.max_steps 500` | 1 |
| M_L.2 | `RankFilter` + `MetricLogger` | rank 0 控制台 + JSONL 文件 | 0.5 |
| M_L.3 | `WandbLogger` + `MLflowLogger` | 远程日志 | 0.5 |
| M_L.4 | `MetricsSample` + `calculate_mfu()` | loss/grad_norm/lr/tps/mfu | 0.5 |
| M_L.5 | Recipe YAML 模板 | LLM + VLM + 多节点 3 套参考配置 | 0.5 |
| | **小计** | | **3** | |

---

### M_M · 测试 / 文档 / 迁移指南

> **调用位置**: 全部模块完成后
> **设计文档**: —

| # | 子需求 | 说明 | 工时(日) |
|---|--------|------|:------:|
| M_M.1 | DTensor 零拷贝验证 | `dt._local_tensor` 修改 ↔ `dt` 同步 | 0.5 |
| M_M.2 | 校验/生产模式等价性 | 同一 batch 两种模式输出一致 | 0.5 |
| M_M.2a | FSDP2 梯度同步集成测试 | TP + FSDP2 联合训练场景下梯度 all-reduce 正确性——验证 `tp_grad_info` 机制在 HSDP layout-driven 框架下正确运行 | 0.5 |
| M_M.3 | PrecompiledBoundary 组合测试 | 所有 `in_src→in_dst` 组合 | 0.5 |
| M_M.4 | HF 兼容性测试 | 3 个模型 from_pretrained 推理一致性 | 0.5 |
| M_M.5 | Checkpoint roundtrip | save→load + 跨 TP 配置 | 0.5 |
| M_M.6 | 端到端训练 + 断点续训 | Qwen3.5-0.8B 8 GPU 1000 步 | 1 |
| M_M.7 | 迁移指南 | 旧配置→新配置映射 + API 变化对照 | 0.5 |
| M_M.8 | 快速入门 + 自定义模型指南 | 5 步跑通 + 新模型添加 ShardingTemplate | 1 |
| M_M.9 | 配置参考手册 + 设计文档校对 | YAML 字段完整说明 + 6 份设计文档最终更新 | 0.5 |
| | **小计** | | **6** | |

---

### 总需求汇总

| 模块 | 调用位置 | AI 辅助人·日 | 状态（2026-07） |
|------|---------|:----------:|----------------|
| M_A  ConfigNode / RecipeConfig | ①② | 5 | 未开始 |
| M_B  分布式基础设施 | ④.1/④.2/④.4.9 | 6 | 部分完成（B.7/B.7a 已交付 1.5，剩余 4.5） |
| M_C  HF 兼容层 | ④.4 全部子步骤 | 8 | 未开始 |
| M_D  双模式 DTensor | ④.4.6-7 + ⑤.1.2 | 15 | ✅ 已完成（300 用例全绿）。未闭环子项：`dtensor_utils.py` 待创建（06）、`tp_grad_info` 消费端接线待落地（05 §6.7.1 / D-12）。`init_ep_token_dispatchers` 已由 ep_utils 落地闭环（2026-07-20：`_hf_native_ep_compute` + `_ep_all_to_all` + `MOE_ROUTER_ADAPTERS`，经 planner EP 注入意图挂接，05 §6.4.7/§6.4.8 D-09/D-10） |
| M_E  权重加载 | ④.4.8 | 4 | 未开始 |
| M_F  Checkpoint | ④.6/④.11/⑤ | 5.5 | 未开始 |
| M_G  Optimizer / Loss | ④.5/④.7/④.10/⑤.1.2 | 3 | 未开始 |
| M_H  训练循环 | ③④⑤ | 7 | 未开始 |
| M_I  数据管道 | ④.8 | 4 | 未开始 |
| M_J  模型实现 | ④.4.3 | 8 | 未开始 |
| M_K  高级特性 | ④.4.1/④.4.9 | 6 | 未开始 |
| M_L  CLI / 监控 | main 入口 + ⑤ | 3 | 未开始 |
| M_M  测试 / 文档 | 全部完成后 | 6.0 | 未开始 |
| **合计** | | **80.5** (~4 人·月) | 已交付 16.5，**剩余 64.0** |


## 6. 依赖关系与里程碑

> **记号约定（消歧）**：本章有两套独立命名空间，切勿混淆——
> - `M_A`–`M_M`（带字母下标）：**模块标识**，对应第五章各模块及其人·日估算（如 `M_A` ConfigNode=5、`M_D` 双模式 DTensor=15）。依赖图 §6.1、关键路径 §6.2 均用此记号。
> - `M1`–`M4`（纯数字）：**集成里程碑**，对应 §6.3 里程碑表的累计人·日节点（M1 基础设施就绪=16、M2 DTensor 核心=35、M3 端到端=60.5、M4 全特性发布=78.5）。§6.2 说明文字中的 M3/M4 指里程碑。
> 二者无一一对应；模块人·日之和约等于里程碑累计人·日。

### 6.1 依赖图

各模块的设计依据以详细设计文档中的修正版本为准：

- **Checkpoint (M_F)**：详见 [04_checkpoint.md §5.3](04_checkpoint.md) —— `load_base_model()` 单一 `path` 参数签名（`_get_hf_safetensors_reference_path` 负责 `root_dir`+`model_name` 解析）、本地缓存 fallback
- **梯度同步 (M_B.7a / M_D.15a)**：详见 [05_dual_mode_dtensor_parallel_strategy.md §6.7](05_dual_mode_dtensor_parallel_strategy.md) —— `tp_grad_info` + HSDP layout-driven 梯度同步机制
- **CP 数据管道 (M_I)**：详见 [02_data_pipeline.md §5.3/§7](02_data_pipeline.md) —— CP `seq_lens`/`seq_lens_padded` 对齐、CP+SP+THD 三层粒度说明

```
M_A (ConfigNode, 5) — 根
 ├─ M_B (分布式基础设施, 6) ← M_A   [B.7/B.7a 已交付 1.5，剩余 4.5]
 │   ├─ M_C (HF 兼容层, 8) ← M_A, M_B
 │   │   ├─ M_E (权重加载, 4) ← M_C
 │   │   │   └─ M_F (Checkpoint, 5.5) ← M_C, M_E
 │   │   │       └─ M_H (训练循环, 5) ← M_C, M_F, M_G, M_I
 │   │   │           └─ M_L (CLI/监控, 3) ← M_H ──┐
 │   │   └─ M_D (双模式 DTensor, 15) ← M_B        │   ✅ 已完成（2026-07，300 用例全绿）
 │   │       └─ M_J (模型实现, 8) ← M_C, M_D      ├── M_M (测试/文档, 6.0)
 │   │           └─ M_K (高级特性, 6) ← M_J, M_B  │  （并行，不阻塞 MVP 关键路径）
 │   ├─ M_G (Optimizer/Loss, 3) ← M_A ──────────── M_H
 │   └─ M_I (数据管道, 4) ← M_A ────────────────── M_H
 └─ (基础抽象: ParamRole, StateDictAdapter, HFCheckpointingMixin, BackendConfig)

 全部 M_A-M_L 完成 ── M_M (测试/文档, 6.0)
```

### 6.2 关键路径

```
M_A → M_B → M_C → M_E → M_F → M_H → M_L → M_M
 5 + 6 + 8 + 4 + 5.5 + 5 + 3 + 6.0 = 42.5 人·日（AI 辅助，单线程 ~2 人·月）| 纯人工参考 120 人·日
```

**剩余关键路径（2026-07 rebase）**：M_D（15）已完成，M_B.7/B.7a（1.5）已交付。剩余关键路径从 M_B 的 mesh/FSDP2 部分（06 §3-4，M_B.1-B.6/B.8）起算：

```
M_A → M_B 剩余 → M_C → M_E → M_F → M_H → M_L → M_M
 5 + 4.5 + 8 + 4 + 5.5 + 5 + 3 + 6.0 = 41.0 人·日（AI 辅助，单线程 ~2 人·月）
```

> 说明：M_K（高级特性 PP/AC/混合精度）与上述路径并行开发，其端到端测试在 M_K.7 内闭环，
> 不阻塞 MVP（M3 端到端训练）关键路径；M4 全特性发布时才合并 M_K。

### 6.3 里程碑

| 里程碑 | 累计人·日 | 交付物 | 验收标准 | 状态（2026-07） |
|--------|:------:|--------|---------|----------------|
| **M1: 基础设施就绪** | 16 | ConfigNode + from_pretrained + MeshContext 可用 | Llama 3.2 1B 单 GPU from_pretrained 推理与 HF 一致 | 进行中（M_B.7/B.7a 已交付 1.5/16） |
| **M2: DTensor 核心可用** | 35 | ShardingPlanner + ShardingApplier + PrecompiledBoundary 可用 | Llama TP=4 校验模式全部 pass + 生产模式 loss 正常下降 + 梯度等价性测试通过（同 batch production vs validation `param.grad` 容差内一致） | 核心件 M_D 已完成（15/35），待 M1 + M_E |
| **M3: 端到端训练可用** | 62.5 | Qwen3.5-0.8B 完整训练 + 断点续训 + HF ckpt 导出 | loss 与现有 Hyper-Parallel 一致 | 已交付 16.5/62.5 |
| **M4: 全特性发布** | 80.5 | 全特性 + 文档 + 测试 + 迁移指南 | 外部用户可独立完成第一个训练 | 已交付 16.5/80.5 |

各里程碑覆盖的模块：

| 里程碑 | 包含模块 | 累计人·日 |
|--------|---------|:------:|
| **M1: 基础设施就绪** | M_A (5) + M_B (6) + M_C 部分（`from_pretrained`/`_init_model`，5） | 16 |
| **M2: DTensor 核心可用** | M1 + M_D (15，✅ 已完成) + M_E (4) | 35 |
| **M3: 端到端训练可用** | M2 + M_C 剩余 (3) + M_F (5.5) + M_G (3) + M_H (7) + M_I (4) + M_J 主体 (5) | 62.5 |
| **M4: 全特性发布** | M3 + M_J 剩余 (3) + M_K (6) + M_L (3) + M_M (6) | 80.5 |

### 6.4 资源配置建议

| 配置 | 工期（AI 辅助） | 纯人工参考 | 并行策略 |
|------|:------------:|:--------:|---------|
| **1 人** | ~4 月 | ~12 月 | 单线程全栈推进 |
| **2 人** | ~2.5 月 | ~6 月 | 1 人 HF 兼容+训练+数据（M_C→M_E→M_F→M_G→M_I→M_H→M_L）；1 人 DTensor+分布式+模型（M_B→M_D→M_J→M_K）；M_A 为共享地基 |
| **3 人** | ~2 月 | ~4 月 | +1 人测试+文档+模型适配（M_M + M_J 适配） |

---

## 7. 风险与缓解

| 风险 | 概率 | 缓解 |
|------|:----:|------|
| `_local_params_context` 破坏模型内部引用 | 中 | 先用 Llama（结构简单）验证，再推到 Qwen3.5 |
| ShardingPlanner 对非标准架构覆盖不足 | 中 | 90% 自动推导 + 10% `ARCH_OVERRIDES` + SpecialHandler 兜底 |
| DCP 跨配置重分片正确性 | 低 | M4 验收 roundtrip 测试覆盖 TP=2→4（归 M_F Checkpoint） |
| Qwen3.5-MoE expert 合并逻辑复杂 | 中 | 参考 AutoModel 已验证的 `WeightConverter` |
| 旧模型迁移成本 | 高 | 先 Llama（最简单）→ Qwen3.5-Dense → MoE/VLM |

---

## 附录 A：与现有代码的兼容策略与迁移路径

### A.1 用户可见 API 兼容

1. `scripts/train_lm.py` 保留为兼容入口，内部转发到新 Recipe
2. 旧 `parallelize.py` → 提取为 `ARCH_OVERRIDES` 规则 + SpecialHandler
3. 旧三层配置与 `_target_` YAML 共存至少 2 个版本
4. `DATASET_REGISTRY` / `register_spec()` 包装为 `MODEL_ARCH_MAPPING` 别名

### A.2 底层 DTensor/Shard 体系迁移路径

重构目标目录 `hyper_models/components/distributed/` 全新，现有 `hyper_parallel/core/` 已有一套自研体系。
迁移策略为**封装复用**而非重写：

| 现有模块 | 策略 | 说明 |
|---------|------|------|
| `core/dtensor/` | **复用** | DTensor ops（实际 5,626 行；早期草稿所称 "25k+ 行" 为高估），作为 DTensor dispatch 实现保留 |
| `core/shard/` | **复用** | `core/shard/` 底层算子保留为 DTensor dispatch 后端（被 `DTensor.redistribute()` 和 `distribute_tensor()` 内部复用）；旧的高层 shard plan 编排 API 被 `ShardingPlanner`/`ShardingApplier` 替代。用户侧代码只能通过新的 `hyper_models/components/distributed/` 入口使用分片功能。 |
| `core/fully_shard/` (HSDP) | **复用并扩展** | `_build_layout_driven_group_info()`/`all_reduce_grad()`/`DTENSOR_UNIFIED` 模式——新设计复用此梯度同步机制，新增 `tp_grad_info` 旁路 |
| `core/context_parallel/` | **不再使用** | CP 通信由 `_wrap_cp_inner_attention` + `flex_cp_allgather`（all-gather K/V，05 D-01'' 定稿）编译期注入；`core/context_parallel/` 的 ring attention 实现不再使用 |
| `dmodule/` | **替换** | 运行时 redistribute → `PrecompiledBoundary` 编译期化 |
| `models/*/parallelize.py` | **替换** | 过程式硬编码 → `ARCH_OVERRIDES` + ShardingPlanner |

**关键**：`_local_params_context` 在 build 期一次性解包 DTensor→local（`fully_shard` 前），
之后参数由 FSDP/HSDP 以 `LOCAL_PARAM` + `tp_grad_info` 管理。
TP 梯度 all-reduce 复用现有 HSDP layout-driven 梯度同步。详见 05 文档 §6.7。

> **行数修正对人日估算的影响**：`core/dtensor/` 实际 5,626 行（非早期估计的 25k+）。该行数不参与人日估算——
> 迁移策略为封装复用而非重写，相关工时（M_D/M_B）按新组件的接口适配量估算，与存量代码行数无关，
> 故 §5/§6 的人日估算维持不变（且 M_D 已按原估算 15 人日内完成交付，实证估算成立）。

> **`_local_params_context` 的 canonical 位置**：实现在 `hyper_models/components/distributed/sharding/apply.py`（见 05 §4.4），
> `hyper_models/components/distributed/dtensor_utils.py` 为跨模块引用入口（re-export），非重复实现。

## 附录 B：关键术语对照

| 旧术语 | 新术语 | 说明 |
|--------|--------|------|
| `LLMTrainer` | `FinetuneRecipe` | 组合式训练编排 |
| `ModelSpec` | `MODEL_ARCH_MAPPING` + `HyperAutoModel` | HF 兼容模型注册 |
| `parallelize_fn` (~1367行 qwen3_5 / ~2554行 moe) | `ARCH_OVERRIDES` (~20行) + ShardingPlanner | 声明式替代过程式 |
| `_redistribute_inputs` (运行时判断) | `PrecompiledBoundary` (编译期) | 零运行时 dispatch |
| `scripts/train_lm.py` | `cli/app.py` | CLI 入口 |
| `dcp_save/dcp_load` | `Checkpointer.save_model/load_model` | 含 DTensor 元数据 |
| 旧三层配置 | `_target_` IoC YAML | 声明式替代结构化 |

## 附录 C：详细设计文档索引

| # | 文档 | 覆盖内容 |
|---|------|---------|
| 01 | [01_hf_compatibility_layer.md](detailed_design/01_hf_compatibility_layer.md) | ConfigNode → RecipeConfig → from_pretrained → _build_model → 各模块衔接 |
| 02 | [02_data_pipeline.md](detailed_design/02_data_pipeline.md) | build_dataloader：tokenizer → dataset → sampler → collate → dataloader |
| 03 | [03_training_loop.md](detailed_design/03_training_loop.md) | Recipe.setup() → run_train_validation_loop → optimizer step → forward/backward |
| 04 | [04_checkpoint.md](detailed_design/04_checkpoint.md) | Checkpointer：DCP save/load → 跨配置重分片 → 异步保存 → 故障恢复 |
| 05 | [05_dual_mode_dtensor_parallel_strategy.md](detailed_design/05_dual_mode_dtensor_parallel_strategy.md) | ShardingPlanner 6-phase 推导 + ShardingApplier 双模式 + PrecompiledBoundary |
| 06 | [06_distributed_infrastructure.md](detailed_design/06_distributed_infrastructure.md) | DistributedSetup + MeshContext + FSDP2Manager + _local_params_context |
