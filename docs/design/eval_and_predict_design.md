# HyperParallel Eval 与 Predict 设计方案

| 项目 | 内容 |
|---|---|
| 状态 | P0 已实现（[PR #1568](https://github.com/mindspore-ai/hyper-parallel/pull/1568)）；P1 及以后为设计契约 |
| 源码基线 | `hyper-parallel` master @ `02077404` |
| 关联 RFC | [#1504](https://github.com/mindspore-ai/hyper-parallel/issues/1504) |
| 适用范围 | Torch-only；独立 loss 评估与离线批量生成；Ascend NPU / Nvidia GPU |
| 约束来源 | [AGENTS.md](../../AGENTS.md)、[项目规则](../../.agent/rules/project-overview.md)、[分布式规则](../../.agent/rules/distributed.md)、[测试规则](../../.agent/rules/testing.md) |

---

## 1. 目标与功能范围

### 1.1 目标：四个环节构成一个可比较结论的闭环

方案增加 `run_mode: train | eval | predict` 入口，让训练产出的 checkpoint 能被独立评分或生成，不依赖训练态组件。

```text
        G1 独立 loss 评估              G2 独立批量生成
        teacher-forcing              prompt → 自回归 → 结果落盘
        产出一个确定的标量             产出一组可读文本
                │                            │
                └──────────────┬─────────────┘
                               ▼
                 G3 分布式正确性：在声明的拓扑下算得对
                               │
                               ▼
                 G4 可复现评测：结果可解释、可比较、可重判
```

| 目标 | 解决的问题 | 产物 |
|---|---|---|
| **G1** | 模型在固定数据上的目标函数表现如何 | 全局 token 加权 loss；有纯 NLL 时给 PPL |
| **G2** | 模型对给定输入实际生成了什么 | 按输入顺序排列的 JSONL |
| **G3** | 单机正确的逻辑在多卡下是否仍正确 | 每个开放拓扑的通信/归属规则 + 对应 ST |
| **G4** | 这次结果能否被解释和复算 | manifest：模型、数据、模板、参数、seed、环境 |

**为什么需要 G1**：训练日志里的 loss 嵌在 Trainer、训练数据计划和日志归约里，不能可靠地给任意 checkpoint 打分——按 batch 均值平均、混入 aux 目标、受训练设置影响，都让跨 checkpoint 比较失效。

**G1 与 G2 并列，不是流水线**：`run_mode: predict` 不要求先跑 eval。

### 1.2 功能总览与优先级

阶段编号：`P0` 评估 → `P1` 批量生成 → `P2` 并行拓扑 → `P3` 生成式指标 → `P4` VLM；`B0–B3` 是基准支线。

#### eval 功能

| 功能 | 优先级 | 说明 |
|---|---|---|
| 固定数据范围上计算全局 loss | **P0 ✅** | token 加权；跨 DP 归约后才算最终值 |
| 计算 perplexity | **P0 ✅** | 仅 `lm_nll` 统计项；`objective` 配 PPL 直接报错 |
| 加载指定 checkpoint | **P0 ✅** | `restore_from` 支持路径与 `LATEST` |
| model-only 恢复（不建 optimizer） | **P0 ✅** | 独立任务特有的恢复入口 |
| 选择数据切分 | **P0 ✅** | `valid` / `test` / `dataset`；不隐式用训练集 |
| 限制评估范围 | **P0 ✅** | `max_samples`（样本数）、`max_batches`（前向轮数） |
| 覆盖评估专用 dataset / dataloader | **P0 ✅** | `evaluation.dataset` / `evaluation.dataloader` |
| 多指标输出 | **P0 ✅** | Target 列表；内置 Loss / Perplexity |
| 单卡 / DP / dense FSDP+HSDP | **P0 ✅** | |
| **EP**（MoE 必需） | P2 | 规模上去后"能否放下"的前提 |
| **PP**（大模型必需） | P2 | 需 forward-only schedule |
| CP | P2 | 长上下文 |
| TP / SP | P2 | 统计域最易算错，放最后 |
| 生成式评估（Rouge/BLEU） | P3 | 需要生成循环 |
| VLM 评估 | P4 | |

#### predict 功能

| 功能 | 优先级 | 说明 |
|---|---|---|
| txt / jsonl / json 输入 | P1 | txt 每行一条；jsonl 含 `id` + `prompt` 或 `messages` |
| chat template 渲染 | P1 | 显式选择模板；支持 system/user/assistant/tool |
| 批量生成 | P1 | 左 padding 组 batch；`batch_size` 是每 lane 请求数 |
| 按长度分桶 | P1 | 受 `max_batch_tokens` 约束；输出按原序 |
| 结构化返回 + JSONL 落盘 | P1 | 含 `finish_reason`；单 writer、原子提交 |
| DP 分片 | P1 | 不要求输入数被 DP 整除 |
| 生成式指标 | P3 | |
| FSDP / TP / CP / EP 生成 | P2 | 逐项开放 |
| PP 生成 | P2 之后 | 需 token 回馈 + 跨级同步停止 |
| 多采样 n>1 | B0 | 需请求级随机流 |
| 断点续跑 | B0 | 需实验指纹校验 |

#### 基准扩展（支线，不阻塞主线）

| 阶段 | 内容 |
|---|---|
| B0 | n>1、请求级 RNG、续跑 manifest |
| B1 | AIME（纯函数判分） |
| B2 | LiveCodeBench（官方 harness，外部沙箱） |
| B3 | SWE-bench（外部 agent + 官方 harness） |

### 1.3 边界：不做什么

| 排除项 | 理由 |
|---|---|
| 训练中周期评估 | `EvaluateCallback` 保持现状。迭代器交错、RNG、FSDP 与梯度状态隔离是独立问题 |
| 在线服务（continuous batching、paged KV、MindIE） | 服务化与离线任务问题域不同 |
| beam search | 首期只做现有采样与贪心 |
| 训练图编译下的评估 | 编译属训练态，独立任务显式拒绝 |
| 基准执行环境（沙箱、docker、agent 工具链） | 依赖外部工具链版本与安全边界；核心库只保证输出可被消费 |

首期面向 `TextTrainer`。**不从训练支持情况推导推理支持情况**——训练支持的拓扑不等于 eval/predict 支持。

### 1.4 提前说明：目标 ≠ 已交付

| 层次 | 内容 |
|---|---|
| 目标 | G1–G4 全部 |
| P0（PR #1568 已交付） | 独立 loss/PPL、model-only 恢复、独立评估数据计划、dense DP/FSDP |
| P0 **未**包含 | `predict` 可运行（仅进 schema，调用即报错）、TP/CP/EP/PP、生成式指标 |
| G3/G4 的完整含义 | 贯穿所有阶段的约束，不代表 P0 支持全部拓扑或位级复现 |

---

## 2. 整体架构

### 2.1 分层

```text
┌──────────────────────────────────────────────────────────────┐
│ examples/training_demo/run_text.py     按 run_mode 分发        │
└──────────────────────────────┬───────────────────────────────┘
                               │
┌──────────────────────────────▼───────────────────────────────┐
│ trainer/  = 任务宿主                                          │
│   配置树 · 组件装配 · 权重恢复 · 模态适配                       │
│                                                              │
│   config/tasks.py       EvaluationConfig / PredictionConfig   │
│   config/trainer.py     TrainerConfig（聚合上述两者）           │
│   runtime/run_mode.py   build_steps_for / validate_*          │
│   runtime/eval_step.py   LossEvalStep（eval 的 trainer 侧适配器）│
│   text_trainer.py       run() / evaluate()                    │
└──────────┬────────────────────────────────┬──────────────────┘
           │ 单向依赖                        │ 单向依赖
┌──────────▼───────────┐          ┌─────────▼──────────────┐
│ evaluation/（叶子）   │          │ predict/（叶子，P1）    │
│  types / aggregation │          │  predictor / io        │
│  metrics / evaluator │          │                        │
└──────────┬───────────┘          └─────────┬──────────────┘
           └──────────────┬─────────────────┘
                  ┌───────▼────────┐
                  │ infer/          │  generate（唯一生成循环）
                  │ torch.distributed│
                  └────────────────┘
```

### 2.2 依赖规则

| # | 规则 | 为什么 |
|---|---|---|
| R1 | `evaluation/`、`predict/` 不得 import `trainer/` | 否则只想用 `LossMetric` 也要拉起整条训练栈 |
| R2 | 两个叶子包不得互相 import | 无共享逻辑需求，耦合只会让单测互相拖累 |
| R3 | `evaluation/` 只通过注入的 step 获取数据 | 一旦认识 batch 字段/CP 切分，就绑回 Trainer，R1 也破 |
| R4 | `predict/` 不构建模型、不解释训练配置 | 否则出现第二套权重加载逻辑 |
| R5 | 生成循环、采样、停止只存在于 `infer/` | 两条 decode 循环必然分叉 |
| R6 | 模型特性由 model-owned adapter 声明 | 核心循环不出现 `isinstance` 分支 |

`ModelAdapterSpec` 当前**没有** `supports_generation` 字段；新增 capability 属于本方案改动。

### 2.3 为什么配置和 step 在 `trainer/` 下

这是最容易被质疑的一点：`evaluation/`、`predict/` 是独立任务，为什么它们的配置和 eval step 在 `trainer/` 下？

#### 2.3.1 根因：整棵配置树都挂在 `trainer/` 下

根因是“配置树作为一个整体挂在 trainer 下”，eval/predict 的配置只是搭了顺风车，并非“evaluation 属于 trainer”。

`TrainerConfig` 名为 trainer，实际是**应用配置根**：

```text
TrainerConfig
├── optimizer / training / lr_scheduler     训练专属
├── dataset / dataloader / checkpoint       共享
├── evaluation                              与训练无关
└── prediction                              与训练无关
     ▲
     │  解析器 resolver.py 在这里
     │  共享原语 Target / DatasetConfig / DataLoaderConfig 也在这里
     └─ 任何配置区块都只能放在这里，否则形成循环
```

同理 `runtime/eval_step.py`：那个 step 需要 `trainer.get_batch / model / loss_fn`，按定义只能在 trainer 侧。

#### 2.3.2 配置层本已是一个独立分层

实测 12 个配置模块的内部依赖：

| 层次 | 模块 |
|---|---|
| 零内部依赖 | `target`、`training`、`stress_detect` |
| 依赖基础层 | `data ← target`、`optimization ← target`、`replacement ← target`、`parallelism ← replacement, target` |
| 聚合者 | `tasks ← data, target`；`trainer ← 全部`；`resolver ← 全部` |

也就是说 `trainer/config/` **已经是一个干净的分层**，问题只是它的命名和位置挂在 trainer 下。整体上移（`trainer/config/` → `hyper_parallel/config/`）不会产生环——`trainer/`、`models/` 都只是单向依赖它。

代价在引用面：**90 个文件**引用 `hyper_parallel.trainer.config`（`hyper_parallel/` 内 23 个，`tests`/`examples` 67 个）。

#### 2.3.3 目标结构

```text
hyper_parallel/
├── config/          ← 配置层独立：Target / Dataset / Evaluation / Prediction / 根配置
├── evaluation/      ← 叶子：评估逻辑
├── predict/         ← 叶子：生成逻辑
├── tasks/           ← 装配层：run_mode 分发、eval step
├── trainer/         ← 训练执行
└── infer/           ← 生成循环
```

#### 2.3.4 本次的处理

| 项 | 决定 | 理由 |
|---|---|---|
| 配置层上移 | **不在本 PR 做** | 需改 90 个文件，与“新增 eval/predict”是两个独立议题，应单独开 refactor PR |
| eval step 上移 | **不做** | `TextTrainer.evaluate()` 构造 step，step 又需要 trainer，搬出去会形成双向依赖，需先重新设计装配入口 |
| 文件改名 | **已完成**：`runtime/evaluation.py` → `runtime/eval_step.py` | 消除与顶层 `evaluation/` 的重名，零行为变化 |

**当前的心智模型**：把 `trainer/` 理解为**任务宿主**而不是“训练”——它拥有配置、装配、权重恢复，以及把任务接到模型/数据栈上的适配器；训练只是它的一个模式。包名是历史包袱，改名（`trainer/` → `tasks/` 或 `app/`）是比配置上移更大的破坏性变更。

| 关注点 | 归属 | 原因 |
|---|---|---|
| 评估循环、指标、归约 | `evaluation/` | 不碰 trainer，可脱机单测 |
| 评估配置 | `trainer/config/tasks.py` | 依赖 DatasetConfig/Target；TrainerConfig 聚合它 |
| eval step | `trainer/runtime/eval_step.py` | 需要 get_batch/model/loss_fn |
| 训练循环、optimizer、callbacks | `trainer/` | 只有训练模式需要 |

### 2.4 三种模式的装配计划

`build_steps_for(run_mode)` 返回该模式实际执行的步骤；跳过是**声明**而非副作用。

```text
train   : setup → model → loss → assets → transform → dataset → collate
          → dataloader → get_batch → train_iters
          → optimizer → lr_scheduler → training_context → graph_compiler → callbacks

eval    : setup → model → loss → assets → transform → dataset → collate
          → dataloader → eval_batch_plan → get_batch → model_restore

predict : setup → model → assets → model_restore
```

| 步骤 | train | eval | predict |
|---|---|---|---|
| 分布式初始化 / mesh / seed | ✅ | ✅ | ✅ |
| 模型构建 + 预训练权重 + 并行化 | ✅ | ✅ | ✅ |
| model-only restore | — | ✅ | ✅ |
| tokenizer / chat template | ✅ | ✅ | ✅ |
| dataset / dataloader / get_batch | ✅ | ✅ | — |
| loss_fn | ✅ | ✅ | — |
| optimizer / lr_scheduler | ✅ | — | — |
| callbacks / graph compiler | ✅ | — | — |

`optimizer` 在配置层可选；`run_mode: train` 缺它时在 `TrainerConfig.__post_init__` 直接报错——即使绕过校验器直接 `train()` 也不会静默跑错。

独立任务不启用 activation checkpoint / swap / 训练编译包装；但**前向混合精度与 `loss_parallel` 上下文保留**——它们是前向必需项，不是训练专属。

---

## 3. Evaluation 设计

### 3.1 设计方案

#### 3.1.1 核心机制：step 注入 + 可加统计量 + 一次归约

`evaluation/` 是叶子包，**没有任何取数据的代码**。它唯一的输入是一个可调用对象：

```python
class Evaluator:
    def __init__(
            self,
            step: Callable[[int], EvalStepResult],   # ← 唯一的数据来源
            metrics: List[Metric],
            context: ReductionContext,
            device: torch.device,
            *,
            compute_samples: bool = True,
    ): ...

    def run(self, max_batches=None) -> EvaluationReport:
        ...
        result = self._step(round_index)   # 每轮只调这一个函数
```

| Evaluator 知道 | Evaluator 不知道 |
|---|---|
| 轮次索引 → `EvalStepResult` | batch 里有哪些字段 |
| 统计量的分子/分母/kind | 怎么分词、怎么 pad |
| 何时归约、怎么归约 | CP 切分、TP 广播 |
| 指标如何从统计量派生 | 模型是什么、loss_fn 是什么 |

真正的取数在 Trainer 侧的 `LossEvalStep` 里：

```python
model_inputs, loss_inputs = self.trainer.get_batch(self._iterator)   # 取数（含 CP/TP）
token_counts = count_loss_token(dict(loss_inputs))                   # 数有效 token
outputs = self.trainer.model(**model_inputs, use_cache=False)        # 前向，无 KV cache
local_loss = self.trainer.loss_fn(model_output=outputs, labels=labels)
return EvalStepResult(statistics=..., sample_count=...)              # 交给 Evaluator
```

这样切分的三个好处：叶子包成立（R1）、可用假 step 脱机单测、模态中立（文本与将来的 VLM 共用同一个 Evaluator）。

#### 3.1.2 为什么是"可加统计量"

每个 step 只报告 **分子 / 分母**，不报告 batch 均值：

```text
loss = SUM(所有唯一贡献的 numerator) / SUM(所有唯一贡献的 denominator)
```

如果沿途按 batch mean 平均，变长序列 + packing + token mask 会让结果依赖于 batch 怎么切——同一个权重、同一份数据，换个 DP 就得到另一个数。

累计在设备上以 FP64 进行，归约时拆成精确的 FP32 `(hi, lo)` 对传输（**HCCL 不支持 FP64 集合通信**），收到后在 FP64 中还原。禁止每 batch `.item()`。

#### 3.1.3 `lm_nll` 与 PPL

`lm_nll` **不是 YAML 字段**，是 step 给统计项贴的标签：

```text
loss_fn 返回 {"foundation_loss": t, "mtp_loss": t}
      → statistic 名 = key 去掉 "_loss"     →  "foundation" / "mtp"
      → statistic_kinds 显式覆盖？有 → 用它
      → 否则 名字 == "foundation" ? lm_nll : objective
```

规则：

- `foundation` 默认就是 `lm_nll`，标准因果 LM 无需额外配置；
- MoE router aux、MTP、正则项是 `objective`，**永不进 PPL**；
- PPL 在归约之后派生：`ppl = exp(global_lm_nll)`，不是各 rank 的 PPL 再平均；
- `PerplexityMetric` 指向非 `lm_nll` 统计会报错。

配置 PPL 就是在 `metrics` 里放一个指向 `foundation` 的 `PerplexityMetric`（见 §3.5）。

#### 3.1.4 归一化声明

不同 loss 模块报告的数值口径不同，评估必须知道是哪一种：

| 声明 | 含义 | 转换 |
|---|---|---|
| `token_mean` | 本地有效 token 上的均值 | 分子 = 值 × 本地有效 token 数 |
| `sum` | 本地总和 | 分子 = 值 |
| `global_mean` | 已跨域归约 | **跳过**（再按 rank 计数会放大 group size 倍） |

未声明归一化的 loss 直接报错，不猜口径。`ModelOutputLoss` 声明 `token_mean`。

#### 3.1.5 重复评估与游标复位

batch sampler 的 `consumed_samples` 跑完一轮停在末尾，只丢弃 iterator 会让第二次评估读到 0 个样本。`LossEvalStep.reset()` 负责把游标归零并清空动态 loader 候选缓冲，`Evaluator.reset()` 在每轮开始时调用它。

### 3.2 控制流图

```text
  run_text.py eval.yaml
        │
        ├─ parse_training_args ──► TrainerConfig(run_mode="eval")
        ├─ validate_run_config ──► 纯配置检查（不初始化进程组）
        │
        └─ TextTrainer(config)
              ├─ _setup            分布式 / mesh / seed
              ├─ _build_model      权重 + 并行化
              ├─ _build_loss
              ├─ assets             tokenizer / chat template
              ├─ dataset(valid)     train split 被丢弃
              ├─ dataloader
              ├─ restore_model_weights   ← 覆盖权重（无 optimizer）
              ├─ _build_get_batch
              │
              └─ evaluate()
                    ├─ validate_run_capabilities   拓扑检查
                    ├─ build_metrics               实例化 + 查重
                    ├─ model.eval()                关 dropout
                    │
                    ├─ Evaluator.run()
                    │     ┌──────────────────────────────────────┐
                    │     │ loop:  step(i) ──► EvalStepResult   │
                    │     │          metric.update(statistics)   │
                    │     │ until: 数据耗尽 或 max_batches       │
                    │     ├──────────────────────────────────────┤
                    │     │ once:  metric.synchronize()  ← 唯一  │
                    │     │         一次 all_reduce               │
                    │     ├──────────────────────────────────────┤
                    │     │ read:  metric.compute()   ← 纯读取    │
                    │     └──────────────────────────────────────┘
                    │
                    ├─ EvaluationReport
                    └─ model.train() 恢复（若原本在训练态）
```

**一次评估只有 1 次全局归约**（`synchronize`）。`compute` 可重复调用（日志 + 落盘），不会二次累计。

### 3.3 功能清单与优先级

见 §1.2 的 eval 表。按依赖展开：

| 阶段 | 功能 | 前提 |
|---|---|---|
| P0 ✅ | loss/PPL、checkpoint 恢复、数据范围控制、多指标 | — |
| P2-1 | **EP** eval | 路由通信在 `no_grad` 下正常；损失统计域与单卡一致 |
| P2-2 | **PP** eval | forward-only schedule + 末级标量回传 |
| P2-3 | CP eval | 真实模型 CP mask/cache 验证 |
| P2-4 | TP/SP eval | 统计域规范化（vocab-parallel 已跨 TP 归约，SP 需再 reduce） |
| P3 | 生成式评估 | 依赖 P1 |
| P4 | VLM | processor / 视觉输入 / cache 契约 |

**P2 内部的顺序不是任意的**：模型大到一定程度后，EP 与 PP 是"能否放下"的前提（见 §4.3 的规模对照），而 checkpoint 通常按训练布局保存——沿用同布局评估可避免整体 reshard。TP/SP 放最后，因为统计域最易算错。

### 3.4 接口

标注 **P0** 的是已实现签名，可直接调用；**P1/P2** 是设计契约。

#### 配置（P0）

```python
# hyper_parallel/trainer/config/tasks.py
@dataclass
class EvaluationConfig:
    mode: Literal["loss", "generation"] = "loss"
    split: Literal["valid", "test", "dataset"] = "valid"
    batch_size: int = 1                  # 每个逻辑 DP lane 每次前向的样本数
    max_samples: Optional[int] = None    # 全局有序样本范围上限；None = 整个有限 split
    max_batches: Optional[int] = None    # 前向轮数上限
    dataset: Optional[DatasetConfig] = None
    dataloader: Optional[DataLoaderConfig] = None
    metrics: List[Target[Any]] = field(default_factory=list)

# hyper_parallel/trainer/config/trainer.py
@dataclass
class TrainerConfig:
    model: Target[Any]
    run_mode: Literal["train", "eval", "predict"] = "train"
    optimizer: Optional[OptimizerConfig] = None   # train 模式仍必填
    model_assets: Optional[ModelAssetsConfig] = None
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    ...
```

#### 运行模式（P0）

```python
# hyper_parallel/trainer/runtime/run_mode.py
def build_steps_for(run_mode: str, evaluation_mode: str | None = None) -> tuple[str, ...]:
    """返回该模式要执行的装配步骤，按执行顺序。"""

def validate_run_config(config: Any) -> None:
    """分布式初始化前的纯配置检查。"""

def validate_run_capabilities(config: Any, mesh_context: Any) -> None:
    """模型/mesh 解析后的拓扑检查。"""
```

#### 评估包（P0）

```python
# hyper_parallel/evaluation/types.py
KIND_LM_NLL = "lm_nll"
KIND_OBJECTIVE = "objective"

@dataclass
class LossStatistics:
    numerator: torch.Tensor     # 设备端标量：本贡献域的 loss 总和
    denominator: torch.Tensor   # 设备端非负标量：与分子口径一致的有效 token 数
    kind: str = KIND_OBJECTIVE

@dataclass
class EvalStepResult:
    statistics: Dict[str, LossStatistics] = field(default_factory=dict)
    sample_count: torch.Tensor | None = None   # 留在设备上，避免每 batch 同步

@dataclass
class ReductionContext:
    group: Optional[Any] = None
    world_size: int = 1
```

```python
# hyper_parallel/evaluation/evaluator.py
@dataclass
class EvaluationReport:
    metrics: Dict[str, float]
    num_samples: int
    num_tokens: float
    num_batches: int
    seconds: float
    status: str = "ok"

class Evaluator:
    def __init__(self, step, metrics, context, device, *, compute_samples=True): ...
    def reset(self) -> None: ...
    def run(self, max_batches: Optional[int] = None) -> EvaluationReport: ...
```

```python
# hyper_parallel/evaluation/metrics/base.py
class Metric(ABC):
    def reset(self) -> None: ...
    def update(self, statistics) -> None: ...
    def synchronize(self, context, device) -> None: ...
    def compute(self) -> Dict[str, float]: ...
    @property
    def num_tokens(self) -> float: ...

class LossMetric(Metric):
    def __init__(self, statistic: str = "foundation", name: str = "loss"): ...

class PerplexityMetric(LossMetric):
    def __init__(self, statistic: str = "foundation", name: str = "perplexity"): ...

def build_metrics(targets) -> List[Metric]:
    """实例化；拒绝重复输出键。"""
```

#### Trainer 侧 step（P0）

```python
# hyper_parallel/trainer/runtime/eval_step.py
class LossEvalStep:
    def __init__(
            self,
            trainer: "BaseTrainer",
            *,
            split_name: str = "valid",
            max_batches: Optional[int] = None,
            statistic_kinds: Optional[Mapping[str, str]] = None,
    ): ...
    def reset(self) -> None: ...
    def __call__(self, round_index: int) -> Optional[EvalStepResult]:
        """返回 None 表示数据耗尽或达到轮次上限。"""
```

#### checkpoint（P0）

```python
# hyper_parallel/components/checkpoint/model_restore.py
@dataclass
class RestoreOutcome:
    path: Optional[str] = None
    step: Optional[int] = None
    external_applied: bool = False

def restore_model_weights(model, config, mesh_context, *, strict=True) -> RestoreOutcome:
    """只恢复模型：不创建 optimizer，不读 scheduler / cursor / RNG。"""
```

#### Trainer 入口（P0）

```python
class TextTrainer:
    def __init__(self, config: TrainerConfig) -> None:   # 按 run_mode 裁剪装配
    def run(self) -> None:                               # 分发
    def evaluate(self) -> EvaluationReport: ...
```

### 3.5 使用示例

#### 3.5.1 配置文件

```yaml
run_mode: eval

model:
  _target_: examples.training_demo.qwen3_moe.cropped_qwen3_moe.build_cropped_qwen3_moe
  config_path: /models/Qwen3-30B-A3B
  num_hidden_layers: 4
  local_files_only: true
  torch_dtype: bfloat16

model_assets:                              # 顶层可选；缺省回退到 dataset.model_assets
  tokenizer:
    _target_: hyper_parallel.data.text.build_tokenizer.AutoTokenizer.from_pretrained
    pretrained_model_name_or_path: /models/Qwen3-30B-A3B
    local_files_only: true

checkpoint:
  restore_from: /checkpoints/global_step_1000   # 或 LATEST；eval/predict 只读不写
  save_ckpt: false

training:
  backend: hccl
  seed: 42
  init_device: meta

accelerator:
  tp_size: 1
  cp_size: 1
  ep_size: 1
  pp_size: 1

fsdp_config:
  dp_shard_size: 1        # 单卡或纯 DP 副本写 1；>1 走 FSDP

dataset:
  _target_: hyper_parallel.data.text.build_dataset.build_online_text_mapping_dataset
  data_path: /data/valid.jsonl
  data_config:
    labels_are_shifted: true
    split: "0, 100, 0"    # 全部进 valid
  data_transform:
    _target_: hyper_parallel.data.text.text_transform.build_text_transform
    data_type: plaintext
    text_keys: text
    max_seq_len: 2048

dataloader:
  _target_: hyper_parallel.data.batching.TokenBatchLoader
  min_buffered_samples: 16
  collate_fn:
    _target_: hyper_parallel.data.batching.build_online_text_collate_fn
  get_batch:
    _target_: hyper_parallel.data.batching.TextParallelBatch
    source_type: online
    attention_mode: dense
  sampler_type: single
  drop_last: true

evaluation:
  mode: loss
  split: valid
  batch_size: 2          # 每个逻辑 DP lane 每次前向的样本数
  max_samples: null      # null = 整个有限 split
  max_batches: null
  metrics:
    - _target_: hyper_parallel.evaluation.metrics.LossMetric
      statistic: foundation
      name: loss
    - _target_: hyper_parallel.evaluation.metrics.PerplexityMetric
      statistic: foundation      # 必须是 lm_nll；foundation 默认就是
      name: perplexity
```

#### 3.5.2 启动

**config_file 是位置参数**，不是 `--config_file`：

```bash
# 单卡
torchrun --nproc_per_node=1 examples/training_demo/run_text.py eval.yaml

# DP=8
torchrun --nproc_per_node=8 examples/training_demo/run_text.py eval.yaml

# 带 dotted override
torchrun --nproc_per_node=8 examples/training_demo/run_text.py eval.yaml \
    --evaluation.max_samples=1024 \
    --checkpoint.restore_from=/checkpoints/global_step_2000
```

#### 3.5.3 输出

```json
{
  "metrics": {"loss": 3.476061928272247, "perplexity": 32.332144731309725},
  "num_samples": 20,
  "num_tokens": 640.0,
  "num_batches": 20,
  "seconds": 0.93,
  "status": "ok"
}
```

| 字段 | 含义 | 跨 rank 归约 |
|---|---|---|
| `metrics` | 键由各 metric 的 `name` 决定 | 是 |
| `num_samples` | 真实样本数（剔除 dummy） | 是 |
| `num_tokens` | 全局有效 token 数 = loss 分母 | 是 |
| `num_batches` | 前向轮数 | **否**，本 rank |
| `seconds` | 墙钟时间 | **否**，本 rank |
| `status` | `ok` / `perplexity_overflow` | — |

#### 3.5.4 程序化调用

```python
from hyper_parallel.trainer.config.parser import load_training_config
from hyper_parallel.trainer.runtime.run_mode import validate_run_config
from hyper_parallel.trainer.text_trainer import TextTrainer

config = load_training_config("eval.yaml")     # 或 load_training_config("eval.yaml", ["--evaluation.batch_size=4"])
validate_run_config(config)
trainer = TextTrainer(config)
report = trainer.evaluate()                   # 返回 EvaluationReport
print(report.to_dict())
```

三个配置入口的分工：

| 函数 | 输入 | 场景 |
|---|---|---|
| `parse_training_args(argv)` | 命令行参数 | 命令行入口 |
| `load_training_config(path, overrides)` | 文件路径 + override | **程序里从文件加载（常规）** |
| `resolve_config(mapping)` | 已解析的 YAML dict | 测试、配置生成、不落盘 |

`resolve_config` **不接受文件路径**；传字符串会报 `YAML root must be a mapping`。

**`evaluate()` 与 `run()` 二选一**：`run()` 在 eval 模式下就是转调 `evaluate()`，两者连着写会评估两遍。

多卡时这段代码要在 `torchrun` 下运行（分布式初始化读 `RANK`/`WORLD_SIZE`）。

#### 3.5.5 CLI override

YAML 之后的 `--字段路径=值`，优先级高于 YAML。

| 形式 | 例子 | 结果 |
|---|---|---|
| 配置区字段 | `--evaluation.batch_size=4` | ✅ 生效 |
| 另一配置区字段 | `--checkpoint.restore_from=/ckpt/x` | ✅ 生效 |
| 嵌套 target 参数 | `--model.torch_dtype=float32` | ✅ 生效 |
| 整体替换列表 | `--evaluation.metrics=[]` | ✅ 生效 |
| **列表元素索引** | `--evaluation.metrics[0].statistic=x` | ❌ `unknown field 'metrics[0]'` |
| **修改 `_target_`** | `--model._target_=other.Builder` | ❌ 明确拒绝 |

典型用法——一份 YAML 扫多个 checkpoint：

```bash
for step in 1000 2000 3000; do
  torchrun --nproc_per_node=8 examples/training_demo/run_text.py eval.yaml \
      --checkpoint.restore_from=/ckpt/global_step_${step}
done
```

要调单个 metric 的 `statistic` / `name`，只能改 YAML。

### 3.6 已知限制与待办

| 项 | 现状 | 建议 |
|---|---|---|
| **PPL 配置错误发现太晚** | kind 校验发生在 `compute()`，即**整轮评估跑完之后**（实测：全部轮次执行完才报错）。大模型评估可能数小时，白跑一次 | 让 step 声明将产出的统计项及 kind，`Evaluator` 启动时校验 |
| `statistic_kinds` 未接入 YAML | 只有 `LossEvalStep` 构造参数可用；配置层无法声明 | 视需要增加 `evaluation.statistic_kinds` |
| **配置层挂在 `trainer/` 下** | `trainer/config/` 已是独立分层，仅位置与命名挂在 trainer 下；引用 90 个文件 | 单独 refactor PR：`trainer/config/` → `hyper_parallel/config/`，见 §2.3 |
| `trainer/` 包名与实际角色不符 | 实际是“任务宿主”，训练只是其中一种模式 | 改名成本高于配置上移，暂不做 |
| `restore_model_weights` 传入 `save_extra_state_per_rank` | 保存侧参数影响恢复路径构造，loader 实际自动探测布局 | 去掉该参数 |
| 独立任务对 `save_ckpt: true` 静默忽略 | 不写、不提示 | 启动时打 warning |
| `split: "0, 100, 0"` 需手工改 | 从训练 YAML 复制时 valid 可能只有 1% | 文档说明；后续可考虑提供校验 |

---

## 4. Predict 设计

> 本章全部为 P1 设计，尚未实现。

### 4.1 设计方案

#### 4.1.1 薄壳原则

Predictor 不写任何生成逻辑。循环、采样、KV cache、停止、左 pad 剥离**全部来自 `infer/`**；它只做：读输入 → 组 batch → 调 `generate` → 截取 completion → 落盘。

#### 4.1.2 参数与运行时分层

现有 `GenerationConfig` 把可序列化参数和运行时对象混在一起，P1 拆成两层：

```text
GenerationOptions（可序列化，进 YAML）
    max_new_tokens / temperature / top_k / top_p / do_sample
    eos_token_id / pad_token_id / repetition_penalty / use_cache
            │ Trainer 组装
            ▼
GenerationRuntime（不可序列化，代码注入）
    model_adapter / logits_layout / logits_group
    sampling_owner / sampling_group / forward_sync_group / device / rng
```

运行时字段**不由 YAML 透传**。特殊 token 缺省为 `auto`，解析顺序：任务显式配置 → 模型 generation 元数据 → tokenizer。

#### 4.1.3 多轮对话：支持"带历史"，不支持"交互循环"

| 形态 | 支持 | 说明 |
|---|---|---|
| 一次输入带多轮历史（`messages` 数组） | ✅ P1 | 渲染 chat template 后生成**下一条** assistant 回复 |
| 交互式多轮（生成 → 反馈 → 再生成） | ❌ | 属于 agent 层；Predictor 是批量离线任务 |
| 多轮间的 prefix cache 复用 | ❌ | 首版每轮全量重新编码与 prefill；模板重写/截断会破坏前缀关系 |

历史由调用方组织，Predictor 只负责"给定一段对话，生成最后一条 assistant"。

#### 4.1.4 输出提交

每个逻辑数据分片一个 writer，写独立临时目录 → 关闭 → 提交 shard manifest → 协调 rank 校验（预期分片全部成功、每个 `(input_id, sample_idx)` 恰好一次）→ 按 ordinal 合并 → 原子提交。

P1 要求共享文件系统；已有输出默认报错，`overwrite=true` 才替换；失败保留带 run ID 的临时分片。

### 4.2 控制流图

```text
  run_text.py predict.yaml
        │
        └─ TextTrainer(config)
              ├─ _setup
              ├─ _build_model
              ├─ model_assets          tokenizer + chat template
              ├─ restore_model_weights
              │
              └─ Predictor.run_file()
                    │
        ┌───────────▼────────────────────────────────────────┐
        │ 1. 读输入   .txt / .jsonl / .json                   │
        │ 2. 校验     缺 id 用稳定序号；重复 id 报错；空输入报错 │
        │ 3. 编码     prompt 直接编码；messages 走 chat template│
        │ 4. 组 batch 左 padding；可选长度分桶（max_batch_tokens）│
        │ 5. 生成     infer.generate（prefill + decode）        │
        │ 6. 截取     按每行真实 prompt_lengths，不按 padded 宽度 │
        │ 7. 排序     按输入 ordinal 恢复原序                    │
        │ 8. 提交     单 writer → 校验 → 原子提交                │
        └────────────────────────────────────────────────────┘
```

**pad token 等于 EOS 时也不能通过扫描 padding 推断停止**；`finish_reason` 由生成循环给出，不靠事后数 pad。

### 4.3 功能清单与优先级

| 阶段 | 功能 | 说明 |
|---|---|---|
| P1 | 文本 Predictor（txt/jsonl/json、prompt/messages） | 单卡与独立 DP |
| P1 | `GenerationOptions` 拆分、结构化返回 | 兼容旧 `generate()` 返回 Tensor |
| P1 | 输出提交与 manifest | 单 writer、原子提交、基础复现信息 |
| P2 | FSDP/HSDP 生成同步 | 全域停止、dummy 前向、缓存验证 |
| P2 | TP/CP/EP 生成 | 采样 owner、token 广播、输出去重 |
| P2 之后 | PP 生成 | token 回馈第一级 + 跨级同步停止 |
| B0 | n>1 多采样、请求级 RNG、断点续跑 | 按 `(input_id, sample_idx)` 恢复 |
| P3 | 生成式指标 | |
| B1–B3 | AIME / LCB / SWE 适配 | 判分与 harness 在外部 |

**为什么大模型需要这些并行**（决定 P2 的顺序）：

- 7B 约 14 GiB，单卡或 DP 可放；
- 70B 约 140 GiB，至少 FSDP；
- 数百 B MoE 约 0.5–1 TB，FSDP + EP；
- 1T 级约 2 TB，FSDP + EP + PP。

其中 **EP 是 MoE 的必需项**（专家权重占大头），**PP 是深层大模型的必需项**。但 predict 对这些的要求比 eval 高：decode 阶段每步都要通信，且 KV cache 本身可能超过权重。

### 4.4 接口

#### 配置（P0 已声明，P1 生效）

```python
# hyper_parallel/trainer/config/tasks.py
@dataclass
class PredictionConfig:
    input: Optional[str] = None
    output_file: Optional[str] = None
    batch_size: int = 1
    max_batch_tokens: Optional[int] = None
    bucket_by_length: bool = False
    overwrite: bool = False
    generation: Optional[Any] = None
```

#### 生成接口（P1 设计）

```python
# hyper_parallel/infer/config.py（新增）
@dataclass
class GenerationOptions:
    max_new_tokens: int = 256
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 1.0
    do_sample: bool = False
    eos_token_id: Optional[int | list[int]] = None   # None = auto
    pad_token_id: Optional[int] = None
    repetition_penalty: float = 1.0
    use_cache: bool = True

# hyper_parallel/infer/runtime.py（新增）
@dataclass
class GenerationRuntime:
    model_adapter: Any
    logits_layout: Any
    logits_group: Any
    sampling_owner: int | None
    sampling_group: Any
    forward_sync_group: Any
    device: torch.device
    rng: torch.Generator | None

@dataclass
class GenerationResult:
    sequences: torch.Tensor
    prompt_lengths: torch.Tensor
    generated_lengths: torch.Tensor
    finish_reasons: list[str]        # "eos" | "length" | "stop"
```

#### Predictor（P1 设计）

```python
# hyper_parallel/predict/predictor.py（新增）
class Predictor:
    def __init__(self, model, tokenizer, options: GenerationOptions,
                 runtime: GenerationRuntime, config: PredictionConfig): ...
    def run_file(self) -> PredictionSummary: ...

@dataclass
class PredictionSummary:
    num_requests: int
    num_samples: int
    output_file: str
    seconds: float
```

### 4.5 使用示例

#### 4.5.1 输入格式

```text
# txt：每行一个 prompt
今天天气怎么样？

# jsonl：prompt 形式
{"id": "q-1", "prompt": "今天天气怎么样？"}

# jsonl：messages 形式（单轮）
{"id": "q-2", "messages": [{"role": "user", "content": "今天天气怎么样？"}]}

# jsonl：messages 形式（多轮带历史，支持 system / tool）
{"id": "q-3", "messages": [
  {"role": "system", "content": "你是一个助手。"},
  {"role": "user", "content": "北京有多大？"},
  {"role": "assistant", "content": "约 1.64 万平方公里。"},
  {"role": "user", "content": "人口呢？"}
]}
```

`prompt` 与 `messages` 互斥。缺 `id` 用稳定输入序号；重复 `id` 执行前报错。

#### 4.5.2 配置文件

```yaml
run_mode: predict

model:
  _target_: examples.training_demo.qwen3_moe.cropped_qwen3_moe.build_cropped_qwen3_moe
  config_path: /models/Qwen3-30B-A3B
  num_hidden_layers: 4
  local_files_only: true

model_assets:
  tokenizer:
    _target_: hyper_parallel.data.text.build_tokenizer.AutoTokenizer.from_pretrained
    pretrained_model_name_or_path: /models/Qwen3-30B-A3B
    chat_template: tokenizer          # 或显式模板名
  # chat_template_kwargs: {enable_thinking: false}

checkpoint:
  restore_from: /checkpoints/global_step_1000   # predict 只读不写

training:
  backend: hccl
  seed: 42

prediction:
  input: /inputs/prompts.jsonl
  output_file: /outputs/predictions.jsonl
  batch_size: 4                # 每个逻辑 DP lane 的最大请求数
  max_batch_tokens: 8192       # padded prompt tokens 预算
  bucket_by_length: true
  overwrite: false
  generation:
    max_new_tokens: 256
    do_sample: false
    use_cache: true
```

#### 4.5.3 启动

```bash
torchrun --nproc_per_node=8 examples/training_demo/run_text.py predict.yaml
```

#### 4.5.4 输出

```json
{"id":"q-1","ordinal":0,"sample_idx":0,"generated_text":"今天晴，25 度。","prompt_tokens":9,"generated_tokens":8,"finish_reason":"eos"}
```

| 字段 | 说明 |
|---|---|
| `finish_reason` | `eos` / `length` / `stop` |
| `prompt_tokens` / `generated_tokens` | 来自生成循环的结构化返回 |
| `sample_idx` | n>1 时的候选序号（B0 起启用） |

### 4.6 已知限制与待办

| 项 | 说明 |
|---|---|
| P1 不支持 n>1 | 需请求级随机流（B0），否则改变 batch 组成即改变统计分布 |
| P1 不承诺断点续跑 | B0 增加，需匹配模型/数据/模板/参数/seed 等实验指纹 |
| 多轮无 prefix cache | 首版全量重编码 + prefill |
| `checkpoint.save_*` 静默忽略 | predict 不装配 callbacks，保存路径不存在 |
| PP 生成未设计 | token 回馈与跨级同步停止是独立机制，不复用 PP eval |

---

## 5. 测试方案

### 5.1 分层

| 层 | 位置 | 约束 |
|---|---|---|
| UT | `tests/ut/evaluation/`、`tests/ut/predict/` | 无分布式、CPU 可跑；用 stub 模型与假 step |
| ST | `tests/torch/evaluation/`、`tests/torch/predict/` | `torchrun_case` 多卡；launcher 不导入 torch/hyper_parallel，worker 执行真实计算 |

配置 / Trainer / 数据 / checkpoint 的测试放在各自既有目录。

### 5.2 UT 用例清单

| 测试面 | 用例 | 覆盖点 |
|---|---|---|
| **指标数值** | 不等长 batch 的 token 加权 | 手算对照，证明不是 batch 均值平均 |
| | `compute()` 可重复调用 | 不触发二次 all_reduce |
| | 未 `synchronize` 即 `compute` | 报错 |
| | 零分母 | 返回空而非除零 |
| | kind 中途改变 | 报错 |
| | PPL 指向 `objective` | 报错 |
| **Evaluator 循环** | `max_batches` 截断 | 轮次上限 |
| | step 返回 `None` | 正常结束 |
| | 空数据 | 报错而非零 loss |
| | `reset()` 复位 | 重复 run 结果一致 |
| **数值一致性** | 假模型 vs 单遍参考 | 不等长 batch、全 mask 样本、max_batches、未声明归一化、无梯度、数据耗尽 |
| | 2 进程 gloo | DP 归约 == 单进程；并验证"按 rank 均值平均"会得到不同结果 |
| **运行模式** | `build_steps_for` 三种模式 | eval/predict 不含 optimizer/callbacks |
| | 未知模式 | 报错 |
| | eval 缺 metrics / 缺 dataset / compile / activation checkpoint | 各自报错 |
| | predict 未实现 | `NotImplementedError` |
| | 拓扑门禁 | pp/tp/cp/ep > 1 均拒绝 |
| **配置解析** | 无 optimizer 的 eval 配置 | 可解析 |
| | `metrics` 列表解析 | 成 Target 并可按序 build |
| | 默认 `run_mode: train` 缺 optimizer | 报错 |
| | 非法字段值（batch_size=0 等） | 报错 |

**已验证结果**：`tests/ut/evaluation` 42 例通过；`tests/ut` 相关目录合计 1241 例通过、0 失败。

### 5.3 ST 用例与运行方式

| 用例 | 拓扑 | 断言 |
|---|---|---|
| 单卡 vs 独立参考 | 1 | loss / ppl / token 数 / 样本数 一致 |
| DP 全局指标一致 | 2（本地另验 4） | 各 rank 报告完全相同 |
| checkpoint 恢复 | 1 与 2 | 同一 DCP checkpoint 在两种拓扑下结果一致 |
| 缺 checkpoint | 1 | `FileNotFoundError`，不回退随机权重 |
| （P1）HF greedy 对齐 | 1 | 相同权重/模板/EOS 下 token 序列一致 |
| （P2）FSDP 生成同步 | 2+ | 不同 EOS 长度、空 lane、dummy 下组内采样一致 |

运行方式（仓库既有约定）：

```bash
# 单卡
pytest tests/torch/evaluation/test_eval_e2e.py::test_eval_matches_reference_single_card -q -s

# 双卡
pytest tests/torch/evaluation/test_eval_e2e.py::test_eval_dp_ranks_report_identical_global_metrics -q -s
```

launcher 通过 `torchrun_case(worker, case_name, num_proc=N)` 启动，worker 在临时目录里自造 tokenizer 与语料，不依赖外部资产。

### 5.4 验收标准

| 阶段 | 必须通过 |
|---|---|
| P0 ✅ | checkpoint 恢复前后一致；完整样本覆盖；不同 token 数加权一致；旧 train 配置回归；DP 各 rank 指标一致 |
| P1 | 不需要 dataset/optimizer；HF greedy 对齐；非整除输入完整有序；prompt/completion 边界；未覆盖拓扑默认拒绝 |
| P2 | 每个新拓扑的 loss 统计与单卡参考一致；分布式生成停止不挂起；dummy 行不进入分母 |
| P3 | reference 不泄漏进 prompt；分布式统计等于单进程参考 |

**验收必须用真实模型与真实后端**。toy/stub 只用于契约测试，不作为"某模型已支持"的证据。

---

## 附录：实现状态对照

| 内容 | 状态 |
|---|---|
| `run_mode` / 配置裁剪 / model-only restore / Evaluator / Loss+PPL | P0 已实现（PR #1568） |
| 单卡 / DP / dense FSDP eval | P0 已实现并验证 |
| 验证范围 | 随机初始化的 2 层小模型 + 合成语料；未做全尺寸模型验收 |
| predict 全部内容 | 设计 |
| EP / PP / CP / TP / SP | 设计 |
| 生成式指标 / VLM / 基准 | 设计 |
