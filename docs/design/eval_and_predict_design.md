# HyperParallel Eval 与 Predict 方案设计

| 项目 | 内容 |
|---|---|
| 状态 | 修订草案。P0 已由 [PR #1568](https://github.com/mindspore-ai/hyper-parallel/pull/1568) 实现；P1 及以后的接口为设计契约 |
| 修订日期 | 2026-11-19 |
| 源码基线 | `hyper-parallel` master @ `02077404` |
| 适用范围 | Torch-only 独立文本评估与离线批量生成；基准评测作为扩展 |
| 关联 RFC | [#1504](https://github.com/mindspore-ai/hyper-parallel/issues/1504) |
| 约束来源 | [AGENTS.md](../../AGENTS.md)、[项目规则](../../.agent/rules/project-overview.md)、[分布式规则](../../.agent/rules/distributed.md)、[测试规则](../../.agent/rules/testing.md) |

---

## 0. 阅读指南

本文按“做什么 → 怎么用 → 怎么设计 → 怎么验收”组织。不同读者从不同章节进入：

| 你想知道 | 看哪节 |
|---|---|
| 这个方案解决什么问题 | §1 目标与边界 |
| eval 和 predict 分别支持什么功能 | §2 功能范围 |
| 怎么配置、怎么跑一条评估或生成任务 | §3 使用方法 |
| 模块怎么分层、谁依赖谁 | §4 架构与依赖 |
| 换一个 run_mode 时到底构建了哪些组件 | §5 运行模式与组件装配 |
| 公共接口有哪些、签名是什么 | §6 接口设计 |
| 现有代码缺什么、要改哪里 | §7 源码基线与必须补齐的能力 |
| 权重怎么加载、找不到 checkpoint 会怎样 | §8 权重加载契约 |
| 评估用哪份数据、尾批怎么处理 | §9 独立评估数据语义 |
| loss 怎么加权、PPL 什么时候可用 | §10 Evaluator 与指标契约 |
| 生成任务的输入输出与提交语义 | §11 Predictor 与生成接口 |
| 多卡下谁算、谁通信、谁写结果 | §12 分布式执行契约 |
| 有哪些配置项、什么时候校验 | §13 配置 Schema 与校验 |
| 显存怎么估、长上下文什么限制 | §14 性能与显存 |
| 生成式评估与基准怎么接 | §15 生成式评估与基准扩展 |
| 先做什么、每阶段验收什么 | §16 分期与验收 |
| 要写哪些测试 | §17 测试计划 |
| 实现前必须确认什么 | §18 实现前检查点 |

---

## 1. 目标与边界

### 1.1 四个目标如何构成一个闭环

方案增加统一的 `run_mode: train | eval | predict` 入口，复用现有模型构建、数据处理、loss 与生成实现。四个目标不是并列的功能清单，而是从模型输出到可比较结论的一条依赖链：

```text
   ┌─────────────────────────────────────────────────────────────┐
   │                    离线模型评测闭环                          │
   │                                                             │
   │   G1 独立 loss 评估        G2 独立批量生成                   │
   │   teacher-forcing        自由生成 + 结果落盘                 │
   │   一个确定的标量          一组可读的文本                      │
   │        │                        │                            │
   │        └──────────┬─────────────┘                            │
   │                   ▼                                         │
   │         G3 分布式正确性：声明的拓扑下，以上两者都算得对        │
   │                   │                                         │
   │                   ▼                                         │
   │         G4 可复现评测：结果可解释、可比较、可重新判分          │
   └─────────────────────────────────────────────────────────────┘
```

| 目标 | 解决的问题 | 交付的产物 | 为什么需要它 |
|---|---|---|---|
| **G1 独立 loss 评估** | 模型在固定数据上的目标函数表现如何 | 全局 token 加权 loss；具备纯语言模型 NLL 时给出 perplexity | 训练 loss 嵌在 Trainer、训练数据计划和日志归约里，不能可靠地给任意 checkpoint 打分 |
| **G2 独立批量生成** | 模型对给定输入实际生成了什么 | 按输入顺序排列的 JSONL 结果 | 生成需要 prompt 解析、chat template、采样、停止与落盘，是独立的一条链，不是 loss 的另一种展示 |
| **G3 分布式正确性** | 单机正确的逻辑在分布式下是否仍正确 | 每个开放拓扑的通信/归属规则 + 对应 ST | 只看最终标量或文本，发现不了漏样本、重复计数、归属错乱、rank 挂起 |
| **G4 可复现评测** | 这次结果能否被解释和复算 | manifest：模型、数据、模板、参数、seed、环境、scorer 版本 | 换 checkpoint 或换 tokenizer 都会改变结果，没有 manifest 就无法归因 |

### 1.2 目标不是功能清单

G1–G4 描述的是**完整方案的目标和正确性约束**，不等于某个 PR 已交付的内容。三者的关系：

```text
   目标 (G1-G4)  ──── 是"最终要闭合什么"
        │
        ├── 阶段 (P0-P4, B0-B3) ──── 是"按什么顺序交付"
        │
        └── PR #1568 ──── 是"P0 这一格实际做完了什么"
```

| 层次 | 内容 |
|---|---|
| 目标 | G1–G4 全部 |
| P0（PR #1568 已交付） | 独立 loss/PPL 评估、model-only 权重恢复、独立评估数据计划、dense DP（含 FSDP） |
| P0 **未**包含 | `predict` 可运行（仅进入配置 schema，调用即报错）、TP/CP/EP/PP、生成式指标 |
| G3/G4 的完整含义 | 贯穿所有阶段的约束，不代表 P0 就支持全部拓扑或全部级别的位级复现 |

### 1.3 边界

边界按"能否形成可验证的离线评测闭环"划定。

**在范围内**：独立 loss/PPL 评估、离线批量生成、生成式评估、基准适配（AIME/LiveCodeBench/SWE-bench 的格式与判分）。

**不在范围内**，以及每一项的排除理由：

| 排除项 | 理由 |
|---|---|
| 训练中周期评估 | `EvaluateCallback` 保持现状。它的迭代器交错、RNG、FSDP 与梯度状态隔离是独立问题，不能承诺"只加一个触发点" |
| 在线服务（continuous batching、paged KV、MindIE 对接） | 服务化与离线任务的问题域不同 |
| beam search | 首期只支持现有采样与贪心 |
| 训练图编译模式下的评估 | 编译包装属训练态，独立任务显式拒绝 |
| 基准执行环境（沙箱、docker harness、agent 工具链） | 依赖外部工具链版本与安全边界，留在外部；核心库只保证输出格式可被消费 |

首期面向 `TextTrainer`。VLM、PP、TP/CP/EP 生成按能力矩阵分期接入；**不从训练支持情况推导推理支持情况**——训练支持的拓扑不等于 eval/predict 支持。

MindFormers 仅作为三态分发、评估循环与生成任务分工的概念参考；接口和实现以本仓库的 Torch 组件契约为准。

---

## 2. 功能范围

本节回答"eval 和 predict 各自到底能做什么"。能力矩阵的最终版本见 §12.3；这里给的是功能视角。

### 2.1 `run_mode: eval` 支持的功能

| 功能 | 状态 | 说明 |
|---|---|---|
| 在固定数据范围上计算全局 loss | P0 已交付 | token 加权，跨 DP 归约后才是最终值 |
| 计算 perplexity | P0 已交付 | 仅当统计项声明为 `lm_nll`；`objective` 项配 PPL 直接报错 |
| 加载指定 checkpoint 覆盖权重 | P0 已交付 | `checkpoint.restore_from` 支持具体路径与 `LATEST` |
| model-only 恢复（不建 optimizer） | P0 已交付 | 独立任务不构建 optimizer，因此需要独立恢复入口 |
| 选择数据切分 | P0 已交付 | `split: valid / test / dataset`；不隐式把训练集当验证集 |
| 限制评估范围 | P0 已交付 | `max_samples`（全局样本数）、`max_batches`（前向轮数） |
| 独立配置评估数据与 dataloader | P0 已交付 | `evaluation.dataset` / `evaluation.dataloader` 覆盖顶层配置 |
| 多指标输出 | P0 已交付 | `evaluation.metrics` 是 Target 列表，按配置顺序执行 |
| 单卡 / DP / dense FSDP+HSDP | P0 已交付 | FSDP 下参数分片，评估时按需 unshard |
| TP / SP / CP / EP / PP | 拒绝 | 见 §12.3 能力矩阵与各阶段条件 |
| VLM 评估 | 拒绝 | P4 |
| 生成式评估（Rouge/BLEU） | 拒绝 | `evaluation.mode: generation`，P3 |

### 2.2 `run_mode: predict` 支持的功能

| 功能 | 状态 | 说明 |
|---|---|---|
| 从 txt/jsonl/json 读输入 | P1 设计 | txt 每行一个 prompt；jsonl 每条含 `id` + `prompt` 或 `messages` |
| chat template 渲染 | P1 设计 | 显式选择模板，记录模板版本与参数；不按模型名在核心代码里分支 |
| 批量生成 | P1 设计 | 左 padding 组 batch；`prediction.batch_size` 是每个 DP lane 的请求数 |
| 按 token 长度分桶 | P1 设计 | 受 `max_batch_tokens` 约束；最终按输入 ordinal 恢复顺序 |
| 输出 JSONL（含 finish_reason） | P1 设计 | 单 writer per shard，校验后原子提交 |
| DP 分片并行 | P1 设计 | 不要求总输入数被 DP 整除 |
| FSDP / TP / CP / EP 生成 | 拒绝 | P2 逐项开放 |
| PP 生成 | 拒绝 | P4 之后另行设计 |
| 多采样 n>1 | 拒绝 | B0，需请求级随机流 |
| 断点续跑 | 拒绝 | B0 |

### 2.3 eval 与 predict 对照

两者并列，不是流水线：`run_mode: predict` 不要求先跑 eval。

| 维度 | eval | predict |
|---|---|---|
| 输入 | 带标签的数据（prompt + 参考回答） | 只有 prompt / messages |
| 前向方式 | teacher-forcing，一次前向覆盖所有位置 | 自回归，逐 token 生成 |
| 是否用 KV cache | 否（`use_cache=False`） | 是（`use_cache=True`） |
| 产物 | 一个确定的标量（loss/PPL） | 一组文本 |
| 方差 | 极低（同一权重同一数据完全可复现） | 有采样噪声 |
| 主要用途 | 判断 checkpoint 好坏、验证并行数值一致、数据管线排障 | 观察模型实际输出、为生成式指标和基准提供输入 |
| 是否构建 optimizer | 否 | 否 |
| 是否构建 dataset | 是（valid/test） | 否（输入是文件） |

### 2.4 什么情况下用哪个

```text
需要判断"这个 checkpoint 比上一个好吗"？
    ├─ 想看得分，不希望被采样噪声干扰 ────────► run_mode: eval
    └─ 想看模型实际怎么回答 ─────────────────► run_mode: predict

想验证并行改动没有改变数值？
    └─ 用 eval：DP2/DP4 的 loss 必须完全一致（文本无法承担这个判断）

想给模型做基准评分（AIME/LCB/SWE）？
    └─ 先 predict 生成，再由外部 harness 判分
```

---

## 3. 使用方法

### 3.1 前置条件

模型构建入口、数据和 tokenizer 的配置方式与训练一致。独立任务多了三项要求：

| 项 | 要求 |
|---|---|
| `run_mode` | 显式写 `eval` 或 `predict`；省略等于 `train` |
| `model_assets` | 顶层可选；不配置时回退到 `dataset.model_assets` |
| `optimizer` | 独立任务可省略；`run_mode: train` 仍必填 |

### 3.2 跑一次独立 loss 评估

**完整 YAML**（`eval.yaml`）：

```yaml
run_mode: eval

model:
  _target_: examples.training_demo.qwen3_moe.cropped_qwen3_moe.build_cropped_qwen3_moe
  config_path: /models/Qwen3-30B-A3B
  num_hidden_layers: 4
  local_files_only: true
  torch_dtype: bfloat16

model_assets:
  tokenizer:
    _target_: hyper_parallel.data.text.build_tokenizer.AutoTokenizer.from_pretrained
    pretrained_model_name_or_path: /models/Qwen3-30B-A3B
    local_files_only: true

checkpoint:
  restore_from: /checkpoints/global_step_1000   # 或 LATEST
  save_ckpt: false                              # 评估不写 checkpoint

training:
  backend: hccl
  seed: 42
  init_device: meta

accelerator:
  tp_size: 1
  cp_size: 1
  ep_size: 1
  pp_size: 1

# dp_shard_size>1 时模型走 FSDP；单卡或完整副本评估写 1
fsdp_config:
  dp_shard_size: 1

dataset:
  _target_: hyper_parallel.data.text.build_dataset.build_online_text_mapping_dataset
  data_path: /data/valid.jsonl
  data_config:
    labels_are_shifted: true
    split: "0, 100, 0"          # 全部样本进 valid
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
  mode: loss              # loss | generation
  split: valid            # valid | test | dataset
  batch_size: 2           # 每个逻辑 DP lane 每次前向的样本数
  max_samples: null       # null = 选定 split 全量
  max_batches: null       # null = 不限前向轮数
  metrics:
    - _target_: hyper_parallel.evaluation.metrics.LossMetric
      statistic: foundation
      name: loss
    - _target_: hyper_parallel.evaluation.metrics.PerplexityMetric
      statistic: foundation
      name: perplexity
```

**启动**：

```bash
# 单卡
torchrun --nproc_per_node=1 examples/training_demo/run_text.py --config_file eval.yaml

# DP=8
torchrun --nproc_per_node=8 examples/training_demo/run_text.py --config_file eval.yaml

# 不改 YAML，用 dotted override 临时缩小范围
torchrun --nproc_per_node=8 examples/training_demo/run_text.py \
    --config_file eval.yaml \
    --evaluation.max_samples=1024 \
    --checkpoint.restore_from=/checkpoints/global_step_2000
```

**输出**（日志中的 `Evaluation report`）：

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

| 字段 | 含义 |
|---|---|
| `metrics` | 按 `evaluation.metrics` 配置顺序产生；键名由各 metric 的 `name` 决定 |
| `num_samples` | 跨 rank 归约后的真实样本数（不含 dummy） |
| `num_tokens` | 全局有效 token 数，即 loss 的分母 |
| `num_batches` | 本 rank 执行的前向轮数 |
| `status` | `ok`；PPL 溢出时为 `perplexity_overflow`，此时只保留 loss |

### 3.3 跑一次批量生成（P1 设计）

**完整 YAML**（`predict.yaml`）：

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
    chat_template: tokenizer          # 或显式模板名字
  # chat_template_kwargs: {enable_thinking: false}

checkpoint:
  restore_from: /checkpoints/global_step_1000
  save_ckpt: false

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
  return_token_ids: false
  generation:
    max_new_tokens: 256
    do_sample: false
    use_cache: true
    # eos_token_id / pad_token_id 缺省为 auto，按模型元数据与 tokenizer 解析
```

**输入格式**（三选一）：

```text
# txt：每行一个 prompt
今天天气怎么样？

# jsonl：prompt 形式
{"id": "q-1", "prompt": "今天天气怎么样？"}

# jsonl：messages 形式（走 chat template）
{"id": "q-2", "messages": [{"role": "user", "content": "今天天气怎么样？"}]}
```

**启动**：

```bash
torchrun --nproc_per_node=8 examples/training_demo/run_text.py --config_file predict.yaml
```

**输出**（`outputs/predictions.jsonl`，按输入顺序）：

```json
{"id":"q-1","ordinal":0,"sample_idx":0,"generated_text":"今天晴，25 度。","prompt_tokens":9,"generated_tokens":8,"finish_reason":"eos"}
```

### 3.4 程序化调用

不经过 YAML 时，直接使用 Trainer 与评估包：

```python
from hyper_parallel.trainer.config.resolver import resolve_config
from hyper_parallel.trainer.runtime.run_mode import validate_run_config
from hyper_parallel.trainer.text_trainer import TextTrainer

# 1) 从 YAML 根映射解析出 TrainerConfig（与 parse_training_args 同一条路径）
config = resolve_config(raw_yaml_mapping)
validate_run_config(config)              # 纯配置检查，不初始化进程组

# 2) 构建并评估
trainer = TextTrainer(config)            # 已按 run_mode 裁剪装配
report = trainer.evaluate()              # 返回 EvaluationReport
print(report.to_dict())

# 3) 或者直接按 run_mode 分发
trainer.run()
```

只想要评估原语、不用 Trainer 时：

```python
import torch
from hyper_parallel.evaluation import (
    Evaluator, LossMetric, PerplexityMetric, LossStatistics,
    EvalStepResult, ReductionContext,
)

def step(round_index: int) -> EvalStepResult | None:
    """返回一轮的可加统计量；返回 None 表示数据耗尽。"""
    if round_index >= len(batches):
        return None
    mean_loss, n_tokens = batches[round_index]
    return EvalStepResult(
        statistics={
            "foundation": LossStatistics(
                numerator=torch.tensor(mean_loss * n_tokens, dtype=torch.float64),
                denominator=torch.tensor(float(n_tokens), dtype=torch.float64),
                kind="lm_nll",
            )
        },
        sample_count=torch.tensor(1.0, dtype=torch.float64),
    )

evaluator = Evaluator(
    step,
    metrics=[LossMetric(), PerplexityMetric()],
    context=ReductionContext(),          # 单进程：无 group
    device=torch.device("cpu"),
)
report = evaluator.run()
```

### 3.5 CLI override 规则

沿用仓库既有的 dotted override，优先级高于 YAML：

```bash
--evaluation.batch_size=4
--checkpoint.restore_from=/ckpt/global_step_2000
--evaluation.metrics[0].statistic=foundation
```

不改变 `_target_`（resolver 显式拒绝），只改已配置 target 的参数。

### 3.6 什么会失败，失败在哪一步

校验分三个时点，越早失败越省时间：

| 时点 | 例子 | 典型耗时 |
|---|---|---|
| 纯配置、分布式初始化前 | `run_mode: eval` 但没配 `evaluation.metrics` | 毫秒级 |
| 模型/mesh 解析后、任务开始前 | 拓扑不支持（TP=2）、checkpoint 不存在、无 `lm_nll` 却配了 PPL | 秒级 |
| 输入批构建后、前向前 | prompt 为空、超上下文长度 | 首个 batch |

实测中容易踩到的一条：`fsdp_config.dp_shard_size` 的默认值会让模型进入 FSDP 路径，而 `fully_shard` 只接受 `npu`/`cuda` 设备。在纯 CPU 环境或未显式声明 `fsdp_config` 时，会在模型构建阶段报
`AssertionError: hyper_parallel.fully_shard support device in [torch.npu, torch.cuda], but got 'cpu'`。
单卡或纯 DP 副本评估应显式写 `fsdp_config: {dp_shard_size: 1}`。

```text
parse_training_args ──► validate_run_config ──► TextTrainer(config) ──► run()
                              │                     │
                        纯配置错误              能力/权重错误
                        （未初始化进程组）        （已建模型，未前向）
```

---

## 4. 架构与依赖

### 4.1 分层图

```text
┌───────────────────────────────────────────────────────────────────┐
│ examples/training_demo/run_text.py      模式分发入口               │
└───────────────────────────────┬───────────────────────────────────┘
                                │
┌───────────────────────────────▼───────────────────────────────────┐
│ trainer/                        配置解析 · 组件装配 · 模态适配      │
│                                                                   │
│  config/tasks.py           EvaluationConfig / PredictionConfig     │
│  config/trainer.py         run_mode / model_assets / evaluation    │
│  runtime/run_mode.py       build_steps_for / validate_run_config   │
│  runtime/evaluation.py     LossEvalStep ─────────────┐            │
│  text_trainer.py           run() / evaluate()        │            │
│  base.py                   _build_dataloader(…)      │            │
└───────────────────────────────┬──────────────────────┼────────────┘
                                │                      │
        ┌───────────────────────▼──────────┐  ┌────────▼───────────────────┐
        │ evaluation/   (叶子)             │  │ predict/   (叶子，P1)      │
        │  types.py      统计量/归约上下文 │  │  predictor.py  请求编排    │
        │  aggregation.py 一次归约         │  │  io.py         读入/落盘    │
        │  metrics/      Loss/PPL          │  │  types.py      记录类型    │
        │  evaluator.py  有界循环          │  └────────┬───────────────────┘
        └───────────────────────┬──────────┘           │
                                │                      │
┌───────────────────────────────▼──────────────────────▼────────────┐
│ infer/    generate（唯一生成循环） · sampler · KVCache             │
│ torch.distributed / 仓库 collectives 薄封装                        │
└───────────────────────────────────────────────────────────────────┘

箭头 = 允许的 import 方向。没有一条箭头回到 trainer/。
```

### 4.2 依赖规则

| # | 规则 | 违反会怎样 | 为什么 |
|---|---|---|---|
| R1 | `evaluation/`、`predict/` 不得 import `trainer/` | `evaluation` 会被迫加载整条训练栈 | 只想用 `LossMetric` 也要拉起 models/data/optimizer |
| R2 | `evaluation/` 与 `predict/` 不得互相 import | 两个功能耦合，单测互相拖累 | 它们只共享 `infer/`，没有共享逻辑需求 |
| R3 | `evaluation/` 只通过注入的 step 获取数据 | 它会认识 batch 字段、CP 切分、TP 广播 | 一旦认识，就绑回 Trainer，R1 也破了 |
| R4 | `predict/` 不构建模型、不解释训练配置 | 出现第二套权重加载逻辑 | 一边支持新 checkpoint 格式，另一边会漏 |
| R5 | 生成循环、采样、停止状态更新只在 `infer/` | 两条 decode 循环分叉 | 修了 EOS 的 bug，另一边没修 |
| R6 | `benchmarks/` 只做格式转换与统计 | 核心库背上容器依赖与安全边界 | 沙箱/官方 harness 有独立生命周期 |
| R7 | 模型特性由 model-owned adapter 声明 | 核心循环出现 `isinstance` 分支 | 与仓库既有 adapter 原则冲突 |

**R7 的状态提醒**：`ModelAdapterSpec` 当前**没有** `supports_generation` 字段。新增 capability/provider 属于本方案改动，文档中的 `spec.supports_generation` 都是"待新增"，不是已有能力。

### 4.3 依赖的代码验证

`evaluation/` 的实际 import 闭包（可由 `grep` 核对）：

```text
types.py          → torch
aggregation.py    → torch, torch.distributed, evaluation.types
metrics/base.py   → evaluation.aggregation, evaluation.types
evaluator.py      → torch, evaluation.metrics.base, evaluation.types
```

没有任何一行出现 `hyper_parallel.trainer`。反之 `trainer/runtime/evaluation.py` 与 `text_trainer.py` 可以 import `evaluation/`——方向单向。

### 4.4 模块职责

| 模块位置 | 职责 | 明确不做 |
|---|---|---|
| `trainer/config/tasks.py` | `EvaluationConfig`、`PredictionConfig`（两个非训练任务的配置区） | 不做运行时校验（交给 run_mode） |
| `trainer/runtime/run_mode.py` | 模式校验、装配计划、拓扑能力检查 | 不构建任何组件 |
| `trainer/runtime/evaluation.py` | eval step：取批、token 计数、loss 适配 | 不做归约、不做指标计算 |
| `components/checkpoint/model_restore.py` | model-only 权重恢复 | 不恢复 optimizer/scheduler/RNG |
| `evaluation/types.py` | 统计量与归约上下文的数据类 | 不 import torch 之外的东西 |
| `evaluation/aggregation.py` | 一次全局归约 | 不感知模型或 batch |
| `evaluation/metrics/` | Loss、Perplexity | 不重复累积或通信 |
| `evaluation/evaluator.py` | 有界循环、调用一次 synchronize | 不直接调模型 forward |
| `infer/{config,runtime}.py`（P1） | 生成参数与运行时对象分层 | 不读任务 YAML 的运行时字段 |
| `predict/{predictor,io,types}.py`（P1） | 请求编排、编解码、结果提交 | 不写生成循环 |

---

## 5. 运行模式与组件装配

### 5.1 入口与分发

```text
examples/training_demo/run_text.py
    │
    ├─ parse_training_args()        YAML + --dotted.override
    ├─ validate_run_config(config)  纯配置检查（不初始化进程组）
    └─ TextTrainer(config).run()
              │
              ├─ run_mode == "train"   → train()
              ├─ run_mode == "eval"    → evaluate() → EvaluationReport
              └─ run_mode == "predict" → 当前明确报错（P1 之前）
```

既有的 `examples/training_demo/train_text.py` 继续直接调用 `train()`；它收到非 `train` 配置时应在训练装配前报错，避免按错误的构建计划执行训练。

### 5.2 构建计划

`build_steps_for(run_mode)` 返回该模式下要执行的装配步骤集合，由 `TextTrainer.__init__` 消费。这样"跳过 optimizer"是一个可测试的声明，而不是构造函数写短的副作用。

```text
train   : setup → model → loss → assets → transform → dataset → collate
          → dataloader → get_batch → train_iters
          → optimizer → lr_scheduler → training_context → graph_compiler → callbacks

eval    : setup → model → loss → assets → transform → dataset → collate
          → dataloader → eval_batch_plan → get_batch → model_restore
          （无 optimizer / lr_scheduler / training_context / graph_compiler / callbacks）

predict : setup → model → assets → model_restore
          （无 dataset / collate / dataloader / loss）
```

| 步骤 | train | eval | predict | 说明 |
|---|---|---|---|---|
| 配置校验、分布式初始化、mesh、seed | ✅ | ✅ | ✅ | 共用 |
| 构建模型、预训练权重、并行化 | ✅ | ✅ | ✅ | 权重来源只有这一条路径 |
| 独立任务 model-only restore | — | ✅ | ✅ | 不建 optimizer 时的恢复入口 |
| tokenizer / processor / chat template | ✅ | ✅ | ✅ | 文本任务都需要 |
| dataset / transform / collate / get_batch | ✅ | ✅ | — | predict 输入是文件 |
| loss_fn / bind_model / prepare_model_inputs | ✅ | ✅ | — | 不评估就不需要 loss |
| 前向上下文、dtype 策略 | ✅ | ✅ | ✅ | 前向必需项不是训练专属 |
| optimizer / attach_optimizer / lr_scheduler | ✅ | — | — | 无梯度路径 |
| train_iters / 梯度累积 / backward context | ✅ | — | — | 无训练循环 |
| callbacks / graph compiler | ✅ | — | — | 无周期事件 |

`optimizer` 在配置层改为可选；`TrainerConfig.__post_init__` 中的 optimizer 相关验证、`_validate_optimize_dtype` 只在存在 optimizer 时执行。`run_mode: train` 缺 `optimizer` 时在 `__post_init__` 直接报错——这样即使有人绕过 `validate_run_config` 直接调用 `train()`，也不会静默跑错。

### 5.3 三种模式的装配时序

**train**（既有路径不变）：

```text
_setup → _build_model → _build_loss → assets → transform → dataset → collate
      → dataloader → get_batch → train_iters → optimizer → lr_scheduler
      → training_context → graph_compiler → callbacks → train()
```

**eval**：

```text
_setup ──► _build_model ──► _build_loss ──► assets ──► transform ──► eval dataset
                                                          │
                              ┌───────────────────────────┘
                              ▼
                        restore_model_weights()     ← 独立任务在这里覆盖权重
                              │
                              ▼
                  dataloader(valid) ──► get_batch ──► evaluate()
                                                          │
                                          ┌───────────────┴────────────┐
                                          ▼                            ▼
                                    Evaluator.run()             model.train() 恢复
```

**predict**（P1）：

```text
_setup ──► _build_model ──► assets ──► restore_model_weights()
                                              │
                                              ▼
                                      Predictor.run_file()
                                              │
                                     predict/io → infer.generate → jsonl
```

### 5.4 独立任务不启用的训练态选项

| 选项 | 行为 |
|---|---|
| activation checkpoint | 不启用（没有 backward 要重算） |
| activation swap | 不启用 |
| 训练编译包装（`compile.enabled`） | 显式请求时首期报错 |
| 前向混合精度 / `loss_parallel` 上下文 | **保留**——它们是前向必需项，不是训练专属 |

模式构造**有效 model-build options**，记录其与输入配置的差异，不原地修改共享配置对象。

模型集成诊断使用适合独立任务的 build-only 路径；要求 optimizer 或训练步骤回调的诊断模式首期拒绝。异常清理按分布式启动器约定执行，不在可能已有 rank 失败的 `finally` 中无条件 barrier。

### 5.5 资产来源

```text
顶层 model_assets 有显式配置？──────────────── 是 ──► 用它
        │
        否
        ▼
dataset.model_assets 有显式配置？──────────── 是 ──► 用它
        │
        否
        ▼
      tokenizer = None（预 tokenized Indexed 数据合法）
```

| 规则 | 说明 |
|---|---|
| 顶层优先 | 顶层 `model_assets` 存在时覆盖 dataset 的 |
| 冲突报错 | 两处都有显式配置且不同 → 校验失败，不静默选择 |
| 空配置不算配置 | `ModelAssetsConfig()` 没有任何 target，视为未配置 |
| 比较方式 | Target 按规范化 `_target_` 路径与参数比较，不用对象身份 |
| predict 可无 dataset | 此时只从顶层 assets 构建 tokenizer |
| tokenizer 路径缺省 | 沿用 `model.pretrained_model_name_or_path` |

chat template、特殊 token、模型 revision 必须进入结果 manifest。

predict 不调用 `_build_data_transform`。从已有训练配置启动 predict 时，dataset 下的 assets 只作为兼容配置来源，不触发数据加载。

配置 resolver 需同步支持顶层 assets 及 evaluation 内嵌的 `DatasetConfig`/`DataLoaderConfig`，复用既有专用解析规则——不能只加 dataclass 字段而遗漏 Target 子树解析。

---

## 6. 接口设计

本节集中展示公共接口。**标注 P0 的是 PR #1568 已实现的签名**，可直接按此调用；**标注 P1 的是设计契约**，实现可能微调。

### 6.1 配置接口（P0 已实现）

```python
# hyper_parallel/trainer/config/tasks.py

@dataclass
class EvaluationConfig:
    mode: Literal["loss", "generation"] = "loss"
    split: Literal["valid", "test", "dataset"] = "valid"
    batch_size: int = 1                  # 每个逻辑 DP lane 每次前向的样本数
    max_samples: Optional[int] = None    # 全局有序样本范围上限
    max_batches: Optional[int] = None    # 前向轮数上限
    dataset: Optional[DatasetConfig] = None
    dataloader: Optional[DataLoaderConfig] = None
    metrics: List[Target[Any]] = field(default_factory=list)


@dataclass
class PredictionConfig:
    input: Optional[str] = None
    output_file: Optional[str] = None
    batch_size: int = 1
    max_batch_tokens: Optional[int] = None
    bucket_by_length: bool = False
    overwrite: bool = False
    generation: Optional[Any] = None


# hyper_parallel/trainer/config/trainer.py

@dataclass
class TrainerConfig:
    model: Target[Any]
    run_mode: Literal["train", "eval", "predict"] = "train"
    optimizer: Optional[OptimizerConfig] = None   # train 模式仍必填
    model_assets: Optional[ModelAssetsConfig] = None
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    prediction: PredictionConfig = field(default_factory=PredictionConfig)
    dataset: Optional[DatasetConfig] = None
    dataloader: Optional[DataLoaderConfig] = None
    # …其余既有字段不变
```

`EvaluationConfig.__post_init__` 只做字段级校验（如 `batch_size >= 1`）。"eval 必须有 metrics"属于**跨区、跨模式**的规则，放在 `validate_run_config` 里——否则默认构造的 `EvaluationConfig()` 会让所有训练配置报错。

### 6.2 运行模式接口（P0 已实现）

```python
# hyper_parallel/trainer/runtime/run_mode.py

class BuildStep:
    """装配步骤名。"""
    SETUP = "setup"
    MODEL = "model"
    LOSS = "loss"
    MODEL_ASSETS = "model_assets"
    DATA_TRANSFORM = "data_transform"
    DATASET = "dataset"
    COLLATE = "collate"
    DATALOADER = "dataloader"
    EVAL_BATCH_PLAN = "eval_batch_plan"
    GET_BATCH = "get_batch"
    MODEL_RESTORE = "model_restore"
    TRAIN_ITERS = "train_iters"
    OPTIMIZER = "optimizer"
    LR_SCHEDULER = "lr_scheduler"
    TRAINING_CONTEXT = "training_context"
    GRAPH_COMPILER = "graph_compiler"
    CALLBACKS = "callbacks"


def build_steps_for(run_mode: str, evaluation_mode: str | None = None) -> tuple[str, ...]:
    """返回该模式要执行的装配步骤，按执行顺序。"""


def validate_run_config(config: Any) -> None:
    """分布式初始化前的纯配置检查。

    Raises:
        ValueError: 配置与 run_mode 不匹配（如 eval 缺 metrics）。
        NotImplementedError: 模式已声明但未交付（如 predict）。
    """


def validate_run_capabilities(config: Any, mesh_context: Any) -> None:
    """模型/mesh 解析后的拓扑检查。

    Raises:
        NotImplementedError: 拓扑未在能力矩阵中开放。
    """
```

### 6.3 evaluation 包接口（P0 已实现）

```python
# hyper_parallel/evaluation/types.py

KIND_LM_NLL = "lm_nll"        # 真正的因果 LM 负对数似然，可派 PPL
KIND_OBJECTIVE = "objective"  # 其他目标项（aux / MTP / 正则），永不进 PPL


@dataclass
class LossStatistics:
    numerator: torch.Tensor     # 设备端标量：本贡献域的 loss 总和
    denominator: torch.Tensor   # 设备端非负标量：与分子口径一致的有效 token 数
    kind: str = KIND_OBJECTIVE


@dataclass
class EvalStepResult:
    statistics: Dict[str, LossStatistics] = field(default_factory=dict)
    sample_count: torch.Tensor | None = None   # 设备端标量，保留在设备上避免每 batch 同步


@dataclass
class ReductionContext:
    group: Optional[Any] = None   # 贡献域进程组；None 表示单进程
    world_size: int = 1

    @property
    def distributed(self) -> bool: ...
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

    def to_dict(self) -> Dict[str, Any]: ...


class Evaluator:
    def __init__(
            self,
            step: Callable[[int], EvalStepResult],   # 轮次索引 → 该轮统计量
            metrics: List[Metric],
            context: ReductionContext,
            device: torch.device,
            *,
            compute_samples: bool = True,
    ) -> None: ...

    def reset(self) -> None:
        """重置指标、轮次计数，并调用 step.reset()（若有）。"""

    def run(self, max_batches: Optional[int] = None) -> EvaluationReport:
        """执行至多 max_batches 轮，然后做一次全局归约。"""
```

```python
# hyper_parallel/evaluation/metrics/base.py

class Metric(ABC):
    def reset(self) -> None: ...
    def update(self, statistics: Mapping[str, LossStatistics]) -> None: ...
    def synchronize(self, context: ReductionContext, device: torch.device) -> None: ...
    def compute(self) -> Dict[str, float]: ...   # 纯读取，不触发集合通信
    @property
    def num_tokens(self) -> float: ...


class LossMetric(Metric):
    def __init__(self, statistic: str = "foundation", name: str = "loss") -> None: ...


class PerplexityMetric(LossMetric):
    def __init__(self, statistic: str = "foundation", name: str = "perplexity") -> None: ...


def build_metrics(targets: Iterable[Any]) -> List[Metric]:
    """实例化配置的 metric target，并拒绝重复的输出键。"""
```

### 6.4 Trainer 侧 step 接口（P0 已实现）

```python
# hyper_parallel/trainer/runtime/evaluation.py

class LossEvalStep:
    def __init__(
            self,
            trainer: "BaseTrainer",
            *,
            split_name: str = "valid",
            max_batches: Optional[int] = None,
            statistic_kinds: Optional[Mapping[str, str]] = None,
    ) -> None: ...

    def reset(self) -> None:
        """把 DataLoader 游标与候选缓冲复位到 split 起始位置。

        必需：batch sampler 的 consumed_samples 跑完一轮后停在末尾，
        只丢弃 iterator 会让第二次评估读到 0 个样本。
        """

    def __call__(self, round_index: int) -> Optional[EvalStepResult]:
        """取一个 batch、前向、算 loss，返回可加统计量。

        返回 None 表示数据耗尽或达到轮次上限。
        """
```

step 内部对 loss 归一化的要求：

```python
class SomeLoss(torch.nn.Module):
    # 声明本 loss 报告的数值是"本地 token 均值"、"本地和"还是"已跨域归约"
    loss_normalization = "token_mean"   # token_mean | sum | global_mean

    def forward(self, *, model_output, labels): ...
```

未声明的 loss 在评估时直接报错，而不是猜一个口径。

```python
# hyper_parallel/trainer/text_trainer.py

class TextTrainer:
    def __init__(self, config: TrainerConfig) -> None:
        """按 build_steps_for(config.run_mode) 裁剪装配。"""

    def run(self) -> None:
        """按 config.run_mode 分发到 train() 或 evaluate()。"""

    def evaluate(self) -> EvaluationReport:
        """执行一次有界 loss 评估。"""
```

### 6.5 checkpoint 接口（P0 已实现）

```python
# hyper_parallel/components/checkpoint/model_restore.py

@dataclass
class RestoreOutcome:
    path: Optional[str] = None      # 解析后的 checkpoint 目录
    step: Optional[int] = None      # 目录名中记录的 step
    external_applied: bool = False  # 是否恢复了模型外部 sidecar


def restore_model_weights(
        model: Any,
        config: Any,
        mesh_context: Any,
        *,
        strict: bool = True,
) -> RestoreOutcome:
    """把权重恢复到已构建、已并行化的模型上。

    只提供模型 state skeleton：不创建 optimizer，也不读取
    scheduler / dataloader cursor / 训练 RNG。

    Raises:
        FileNotFoundError: restore_from 指定的 checkpoint 不存在。
        RuntimeError: key 不匹配，或外部 sidecar 无法在无 optimizer 下恢复。
    """
```

### 6.6 生成接口（P1 设计）

```python
# hyper_parallel/infer/config.py（新增）

@dataclass
class GenerationOptions:
    """可序列化的采样参数；进入任务 YAML 的只有这一类。"""
    max_new_tokens: int = 256
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 1.0
    do_sample: bool = False
    eos_token_id: Optional[int | list[int]] = None   # None = auto 解析
    pad_token_id: Optional[int] = None
    repetition_penalty: float = 1.0
    use_cache: bool = True


# hyper_parallel/infer/runtime.py（新增）

@dataclass
class GenerationRuntime:
    """不可序列化的运行时对象，由 Trainer 注入，不来自 YAML。"""
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
    """结构化返回；generate() 默认仍返回 Tensor。"""
    sequences: torch.Tensor
    prompt_lengths: torch.Tensor
    generated_lengths: torch.Tensor
    finish_reasons: list[str]        # "eos" | "length" | "stop"
```

```python
# hyper_parallel/predict/predictor.py（新增）

class Predictor:
    def __init__(
            self,
            model: Any,
            tokenizer: Any,
            options: GenerationOptions,
            runtime: GenerationRuntime,
            config: PredictionConfig,
    ) -> None: ...

    def run_file(self) -> PredictionSummary:
        """读取 config.input，批量生成，按输入顺序写出 config.output_file。"""


@dataclass
class PredictionSummary:
    num_requests: int
    num_samples: int
    output_file: str
    seconds: float
```

Predictor 是薄壳：循环、采样、KV cache、停止、左 pad 剥离**全部来自 `infer/`**，它只做 IO 与批组织。

---

## 7. 源码基线与必须补齐的能力

| 当前实现 | 已有能力 | 本方案需要补齐的部分 |
|---|---|---|
| [TextTrainer](../../hyper_parallel/trainer/text_trainer.py)、[VLMTrainer](../../hyper_parallel/trainer/vlm_trainer.py) | 通过 `BaseTrainer.__new__` 创建基础对象，自行编排构建步骤 | 模式裁剪必须作用于实际的 Trainer 装配路径，不能只修改 `BaseTrainer.__init__` |
| [TrainerConfig](../../hyper_parallel/trainer/config/trainer.py) | Target/dataclass、YAML 和 dotted override | optimizer 当前必填，配置和模型构建校验也读取 optimizer；须按模式解除依赖 |
| [模型构建](../../hyper_parallel/trainer/base.py) | 原子构建、预训练权重、并行化、dtype | 复用构建入口，保留前向必需上下文，按模式处理训练专属选项 |
| [checkpoint callback](../../hyper_parallel/trainer/callbacks/checkpoint_callback.py) | `on_train_begin` 才执行 `restore_from` | 抽出独立 model-only 恢复入口，避免跳过 callbacks 后漏载训练权重 |
| [数据构建](../../hyper_parallel/data/text/build_dataset.py) | train/valid/test 三分构建 | 独立评估的数据范围不再由 train_iters 推导；当前 eval_iters=0 会得到零 valid/test 样本 |
| [batch runtime](../../hyper_parallel/data/batching/get_batch.py) | 从 iterator 读数据，执行 CP 切分、TP 广播，返回 model/loss inputs | 评估循环必须保留 iterator 和数据 owner 语义；非 owner 也须参与通信 |
| [ModelOutputLoss](../../hyper_parallel/components/losses/model_output.py) | `loss_fn(model_output=..., labels=...)` 返回 Tensor 或 loss 字典 | 单独统计有效 token，声明 loss 归一化与 NLL 语义，不假设返回 `(loss, n_tokens)` |
| [token 计数](../../hyper_parallel/trainer/runtime/loss_aggregation.py) | shift_labels、ignore index、loss_mask 对齐 | 在 Trainer step 内复用，不让 evaluation 反向依赖 Trainer |
| [generate](../../hyper_parallel/infer/generation.py) | prefill/decode、采样、缓存、每行 EOS、左 padding 去除 | 分布式采样/停止协议、结构化返回；保留单一生成循环 |
| [GenerationConfig](../../hyper_parallel/infer/utils.py) | 生成参数和运行时 group/cache/callback 混合存放 | 分离可序列化参数和运行时对象，消除配置重复定义 |
| [prefill mask](../../hyper_parallel/infer/utils.py) | 构建 `[B, 1, S, S]` additive mask | 长 prompt 的二次方显存必须纳入预算，长上下文需单独验收 |

本文优先修正上述契约，再扩展功能。已有 helper 可复用，不等于其训练期默认值适合独立任务。

---

## 8. 权重加载契约

### 8.1 加载顺序

```text
模型原子构建 + 预训练权重加载
        │
        ▼
并行化后的模型布局就绪（FSDP 分片、TP 切分已生效）
        │
        ▼
可选 checkpoint.restore_from 覆盖模型权重
        │
        ▼
模型 / loss / 生成能力校验
        │
        ▼
首次任务前向
```

`restore_from` 沿用现有路径或 `LATEST` 语义。所有 rank 必须解析为**同一个**已完成 checkpoint；找不到、未完成或缺关键权重时失败，**禁止回退到初始权重继续评估**——否则会为没人训练过的模型报出一组看似正常的分。

manifest 记录最终解析路径、checkpoint step 和模型配置标识。

创建 restore skeleton 前必须冻结 `state_dict` 结构：PEFT 和任何注册参数或替换有状态模块的 binding 均先完成。恢复后的 loss/生成适配只能改变不影响权重结构的调用行为；现有只替换 forward 的 chunk binding 可在此后应用。

### 8.2 model-only restore

```text
CheckpointerCallback._load_checkpoint()          restore_model_weights()
        │                                                │
        ├─ 路径解析（含 LATEST）                          ├─ 路径解析（含 LATEST）
        ├─ 模型 state skeleton                            ├─ 模型 state skeleton
        ├─ optimizer skeleton + 状态                      ├─ （不创建 optimizer）
        ├─ lr_scheduler / dataloader cursor / RNG         ├─ （不读取这些状态）
        └─ 训练恢复策略                                   └─ 独立任务固定 model-only
```

不能把 callback 包在一个带空 optimizer 的假 Trainer 中使用——那会让"model-only"变成声明而不是性质。

外部权重放行时保留 `before_checkpoint_load → dense load/install → after_checkpoint_load` 顺序和 dense/external 参数精确分工，剥离任何要求 optimizer 已存在的调用；不能只恢复 `state_dict` 中的 dense 部分。

`restore_optimizer / restore_train_state / restore_dataloader_state` 属于训练恢复选项，在独立任务 manifest 中记录为未应用。checkpoint 的保存选项不触发写入。

### 8.3 各权重类型的首期策略

| 权重类型 | 首期策略 |
|---|---|
| 标准完整模型 checkpoint | 支持；严格校验请求的模型 key、shape、dtype 与布局 |
| PEFT adapter-only checkpoint | 基座先就绪，按已有 adapter 映射恢复；仅允许声明的基座缺失，不全面放宽 strict |
| 模型外部参数 / host sidecar | 仅在 provider 支持无 optimizer 的完整恢复时放行；否则明确拒绝并列出缺失能力 |
| 跨拓扑恢复 | 沿用 DCP 的可重分片能力，但支持承诺以该模型、布局组合的 ST 为准 |

---

## 9. 独立评估数据语义

### 9.1 配置与范围

`evaluation.dataset`、`evaluation.dataloader` 可单独指定；缺省时回退到顶层配置。`evaluation.split` 决定读哪份数据：

| `split` | 数据来源 | 什么时候用 |
|---|---|---|
| `valid` | 三分返回的验证集 | 训练时留出的验证集 |
| `test` | 三分返回的测试集 | 最终报告 |
| `dataset` | 单个未切分 Dataset | 数据本身就是评估集 |

**不隐式把训练集当验证集。** 三分 builder 返回单 Dataset 时选 `valid` 会明确报错。

范围由两个独立上限描述：

```text
全局有序样本 (N 个)
   │
   ├── max_samples = M  →  只保留前 M 个（切分前截断）
   │
   └── 分配到 DP lane，每 lane 每次前向 batch_size 个
            │
            └── max_batches = R  →  所有 lane 共同执行 R 轮前向上限
```

| 字段 | 单位 | 语义 |
|---|---|---|
| `evaluation.batch_size` | 样本/次前向 | 每个逻辑 DP lane 的批次；独立于训练 micro/global batch 与梯度累积 |
| `max_samples` | 样本 | 切分前全局有序范围上限；`null` = 整个有限 split |
| `max_batches` | 轮次 | 所有 lane 共同执行的前向轮数；`null` = 不限 |

- `null` 表示不施加该项上限；`0` 是非法值。
- 同时设置时先截取 `max_samples`，再应用 `max_batches`。
- 改变 DP/batch_size 会改变 `max_batches` 对应的样本量；跨拓扑对比应固定 `max_samples` 或完整范围。
- 有限数据集允许全量评估；无确定结束边界的 iterable 必须提供上限，首期不支持通用流式评估。
- 独立任务**不消费** `training.eval_iters`；该字段保留训练语义，不自动转换单位。

### 9.2 Dataset builder 适配

增加显式评估构建请求，将 split、seed、有限数据范围传入现有 builder。**不得**调用 `_get_train_valid_test_num_samples(training_config)` 来规划独立评估——那要求配置里凭空出现 `train_iters`。

P0 使用有限 mapping/Indexed 数据和 fixed batching，拒绝 cyclic sampler、通用 iterable、动态 token batching/packing。Online mapping 按固定顺序遍历选定 split，首期要求一个 source record 转换为一个 ModelSample；像长 plaintext 展开多窗口这样的 transform 明确拒绝。以后支持一对多转换时，先建立有限 `(source_id, chunk_idx)` 索引，再做分片与步数规划。

`batch_size`、`max_samples` 和样本计数均按最终逻辑 ModelSample 计，原始 source record 数可另报。Indexed 数据复用底层索引、split 和样本构建算法，增加"一遍有限样本"模式：固定序列长度、尾窗口规则和 seed，不因请求样本数而重复多个 epoch。

仅构建选中的 split；若底层共享索引初始化不可避免，允许复用该初始化，但不创建 train dataloader 或推导训练迭代数。评估 shuffle 默认关闭，重复 evaluate 时使用全新 iterator/sampler 状态。

### 9.3 尾批、owner 与执行计划

现有 sampler 的 `drop_last=True` 会漏样本；直接设 `False` 可能让各 rank 前向次数不同。评估 batch plan 先确定全局样本 ID 与轮数，再分配各 lane 的槽位：

```text
全局样本 ID:  [0] [1] [2] [3] [4] [5] [6] [7] [8]   (9 个)
DP=2, batch_size=2, 每轮每 lane 2 个槽位

轮次 1   lane0: ID 0,1     lane1: ID 2,3
轮次 2   lane0: ID 4,5     lane1: ID 6,7
轮次 3   lane0: ID 8,dummy lane1: dummy,dummy      ← 尾轮补齐
                 │
                 └─ valid_sample_mask=False
                    分子 = 0，分母 = 0，样本数 = 0
```

规则：

1. 真实样本在评估范围内**恰好消费一次**，保留稳定 sample ID。
2. 共享前向 collective 的 rank 执行**同样的轮数**；尾轮不足槽位以 dummy 补齐。
3. dummy 使用合法非空输入，携带 `valid_sample_mask=False`；step 把其 target/loss mask 排除，分子、分母和样本数贡献为零。
4. 不只依赖现有 `padding_flag`；有效性标记必须随 batch transport 正确传递，并与 causal shift、CP/TP 切分后的标签对齐。
5. `get_batch` 仍接收 iterator，返回 `(model_inputs, loss_inputs)`。非数据 owner rank 可以没有 dataloader，但必须按公共计划参与广播和前向。
6. owner 在进入当轮模型通信前发布数据可用状态；不允许某 rank 因局部 `StopIteration` 提前结束整个循环。

空数据集在前向前明确报错。真实样本全部被 mask 时，保留样本计数并标记无有效 token；若整个任务的指标分母为零，则失败并说明原因。

---

## 10. Evaluator 与指标契约

### 10.1 控制流

```text
Trainer.evaluate()                    Evaluator              LossEvalStep            Metrics
      │                                  │                        │                    │
      ├─ validate_run_capabilities()     │                        │                    │
      ├─ build_metrics(targets)          │                        │                    │
      ├─ model.eval()                    │                        │                    │
      ├─ Evaluator(step, metrics, …)     │                        │                    │
      ├─ run(max_batches) ──────────────►│                        │                    │
      │                                  ├─ reset() ─────────────►│                    │
      │                                  │                        ├─ 游标复位           │
      │                                  │                        │                    │
      │                                  ├─ step(0) ──────────────►                    │
      │                                  │                        ├─ get_batch()       │
      │                                  │                        ├─ model(use_cache=False)
      │                                  │                        ├─ loss_fn()         │
      │                                  │◄── EvalStepResult ─────┤                    │
      │                                  ├─ metric.update() ──────────────────────────►│
      │                                  │   …循环到耗尽或 max_batches…                 │
      │                                  │                                             │
      │                                  ├─ metric.synchronize() ────────────────────►│
      │                                  │   （整个 run 只做一次 all_reduce）            │
      │◄── EvaluationReport ─────────────┤                                             │
      │                                  │                   metric.compute()（纯读取）│
      └─ model.train() 恢复（若原本在训练态）                                          │
```

### 10.2 Trainer step 的职责

Evaluator **不**直接调用模型的任意 forward。Trainer 注入的 step 负责：

1. 调用既有 `get_batch`，得到设备上的 `model_inputs` / `loss_inputs`。
2. 对齐有效样本标记，同步更新模型实际消费的 labels/shift_labels 与统计侧标签/mask，再调用 `count_loss_token(loss_inputs)`；CP 场景保留 `shift_labels` 语义。dummy 必须从 CE 本身和统计分母**同时**排除。
3. 应用 `loss_fn.prepare_model_inputs`（若存在），保留模型所需前向上下文。
4. 在 `torch.no_grad()` 下执行 `model(..., use_cache=False)`。
5. 按现有接口调用 `loss_fn(model_output=outputs, labels=...)`，规范化为命名统计项；及时释放 outputs/logits。

`bind_model` 属于构建步骤，不能每 batch 重复执行。loss-only 模型输出可没有 logits；仅输出 logits 的模型需要配置能够消费它的 loss_fn——默认 `ModelOutputLoss` 不自动补算 CE。

### 10.3 归一化声明

不同 loss 模块报告的数值口径不同，评估必须知道是哪一种才能正确加权：

| 声明 | 含义 | 转换 |
|---|---|---|
| `token_mean` | 本地有效 token 上的均值 | 分子 = 值 × 本地有效 token 数 |
| `sum` | 本地有效 token 上的总和 | 分子 = 值 |
| `global_mean` | 已跨贡献域归约 | **跳过**：再按 rank 计数会放大 group size 倍 |

未声明归一化的 loss 直接报错，不猜口径。

现有 chunk loss 路径提供 `loss_sum / valid_token_count`，可由对应 adapter 直接产生 NLL 统计；其返回的训练 objective 可能还含 aux loss，不能替代 NLL。chunk loss 的 TP/PP/loss_parallel 限制继续由既有能力校验执行。

local count 为零时直接贡献零分子，不执行 `NaN × 0`。causal shift、`IGNORE_INDEX` 和 `loss_mask` 必须与真实 CE 使用相同的有效 target 集合。

### 10.4 统计协议

每个 step 返回 `EvalStepResult`：

| 字段 | 语义 |
|---|---|
| `statistics: dict[str, LossStatistics]` | 命名的可加统计项；键是统计名（如 `foundation`） |
| `LossStatistics.numerator` | 设备端标量，当前贡献域的 loss 总和 |
| `LossStatistics.denominator` | 设备端非负标量，和分子口径一致的有效 token 数或声明权重 |
| `LossStatistics.kind` | `lm_nll` 或 `objective`；未声明 NLL 的项不能产生 PPL |
| `sample_count` | 排除 dummy 后的真实样本数，按数据 owner 计数；保留为设备标量 |

`sample_count` 全程留在设备上，只在整个 run 结束时归约一次——每 batch `.item()` 会让长评估多出成百上千次 host 同步。

### 10.5 聚合与 PPL

```text
       rank0                    rank1                    rank2
   Σnum₀ = 20.0            Σnum₁ = 120.0            Σnum₂ = 35.0
   Σden₀ = 10.0            Σden₁ = 30.0             Σden₂ = 12.0
        │                       │                        │
        └─────────────── all_reduce(SUM) ───────────────┘
                                │
                    Σnum = 175.0   Σden = 52.0
                                │
                    loss = 175.0 / 52.0 = 3.3654
                    ppl  = exp(3.3654) = 28.95    ← 仅当 kind == lm_nll
```

```text
loss = SUM(所有唯一贡献的 numerator) / SUM(所有唯一贡献的 denominator)
ppl  = exp(global_lm_nll)
```

**关键：PPL 在归约之后派生，不是各 rank 的 PPL 再平均。** 后者在有零贡献 rank 时会算出无意义的数。

`ReductionContext` 由 Trainer 根据 loss layout 构建，包含贡献域、归约组和副本结果传播规则。DP/CP 的不相交 token 贡献相加；TP/SP 已复制的统计只能计入一次。若某 loss 先在 TP/SP 形成了完整 NLL，则先规范化这一统计域，再跨 DP/CP 聚合，不能照搬训练 backward 的缩放系数。

P0 限 TP=CP=EP=PP=1，统计域为 DP；复合拓扑随 P2 的数值验收开放。**无 group 不自动解释为 WORLD**，单卡上下文直接跳过 collective。

累计分子至少 FP32，计数使用 int64；小规模参考计算使用 FP64。设备支持且长任务精度需要时可选 FP64 累计。最终结果落盘前才转 Python 标量。禁止每 batch `.item()` 或为日志额外执行全局 loss 归约。

**HCCL 不支持的 dtype 处理**：HCCL 的 `all_reduce` 不接受 FP64。评估在设备上以 FP64 累加，归约时拆成精确的 FP32 `(hi, lo)` 对传输，收到后在 FP64 中还原——累计精度与传输能力两头都保住：

```text
value(FP64) ──► hi = float32(value)
                lo = float32(value - float64(hi))
                 │
          all_reduce([hi, lo], SUM)
                 │
          float64(hi) + float64(lo)
```

`LossMetric` 输出指定统计项的全局均值；`PerplexityMetric` 消费同一份 `lm_nll` 结果，不重复累积或通信。MoE router aux、MTP、正则项等保留为 `objective`，不能计入 PPL。无法提供纯 NLL 时仍可输出 objective loss，但配置 PPL 必须报错。

### 10.6 Metric 生命周期

```text
reset() ──► update() × N ──► synchronize() ──► compute() × M
   │            │                 │                 │
 清空         记录引用         唯一一次 all_reduce   纯读取
                              （重复 compute 不再通信）
```

- Evaluator 统一调用一次 `synchronize`。
- `compute` 可重复调用（日志 + 落盘），不会二次累计。
- 每次 `run` 都 `reset`；多指标输出 key 冲突在运行前报错。
- `reset` 同时会调用 step 的 `reset()`（若存在）——见 §10.7。

### 10.7 重复评估与游标复位

```text
第一次 evaluate()
  sampler.consumed_samples: 0 ────────────► 16（split 末尾）
  step: 读到 20 个 batch

第二次 evaluate()（若不复位）
  sampler.consumed_samples: 16 ──► 16    ← 已经到底
  step: 读到 0 个 batch  → RuntimeError: consumed no real samples
```

`LossEvalStep.reset()` 负责把 `consumed_samples` 归零并清空动态 loader 的候选缓冲；`Evaluator.reset()` 在开始新一轮时调用它。这样"同一个 Task 连续跑两次评估"得到同一份结果，而不是第二次空转。

### 10.8 结果与状态

结果至少包含 `metrics`、`num_samples`、`num_tokens`、`num_batches`、`seconds`、`status`，以及 manifest 侧的实际权重来源和数据范围。

| 情况 | 处理 |
|---|---|
| 指标非有限（NaN） | 记录 `null` 和明确 `status`，不写非标准 JSON 字面量 |
| PPL 指数溢出 | 保留 NLL，标记 `perplexity_overflow`，从 `metrics` 中省略 PPL |
| 全局分母为零 | 失败并说明原因，不输出 0 |

Evaluator 在 run 前保存各模块 `training` 标志，切换 eval，并在 `finally` 恢复原标志。使用 `no_grad` 作为基线；`inference_mode` 仅在 wrapper/cache 兼容测试后作为优化。**该恢复不等同于完成训练中评估的状态隔离**。

---

## 11. Predictor 与生成接口

> 本节全部为 P1 设计。

### 11.1 参数与运行时对象分层

现有 `GenerationConfig` 把可序列化参数和运行时对象（process group、prefix cache、callable）混在一起。P1 拆成同一来源的两层：

```text
GenerationOptions（可序列化，进 YAML）
    max_new_tokens / temperature / top_k / top_p / do_sample
    eos_token_id / pad_token_id / repetition_penalty / use_cache
            │
            │ 由 Trainer 组装
            ▼
GenerationRuntime（不可序列化，由代码注入）
    model_adapter / logits_layout / logits_group
    sampling_owner / sampling_group / forward_sync_group
    device / rng
```

- 建议先在 infer 内抽出公共参数 dataclass，由既有 `GenerationConfig` 继承；参数验证也由 infer 维护，不复制到两个配置类。
- 特殊 token 使用显式 `auto` 值表示待解析，任务配置缺省为 auto；既有 `GenerationConfig` 的程序化默认值仍保持 EOS=2、pad=0，避免兼容性变化。
- 运行时字段**不由 YAML 的 Any 透传**。现有 `generate` 拒绝额外 kwargs；新增 runtime 与结构化返回必须作为显式兼容扩展实现并测试。
- 同时提供新旧两处配置且冲突时拒绝，不维持两套执行分支。

特殊 token 解析顺序：

```text
任务显式配置 ──► 模型 generation 元数据 ──► tokenizer
                                             │
                              都确定不了 → 明确报错
```

任务省略 `eos_token_id` 表示 auto；显式 `null` 表示禁用 EOS、仅按长度或自定义条件停止。兼容多 EOS ID 的任务需同时扩展停止与返回逻辑；未实现前不得仅接受 list 配置而仍做单值比较。

### 11.2 输入与批组织

```text
prediction.input
     │
     ├─ .txt   → 每行一个原始 prompt
     ├─ .jsonl → 每条含 id + (prompt | messages)，二者互斥
     └─ .json  → 同上，数组
     │
     ▼
解析与校验：缺 id 用稳定序号；重复 id 报错；空输入报错
     │
     ▼
编码：prompt 直接编码；messages 走显式 chat template
     │
     ▼
左 padding 组 batch（可选按长度分桶，受 max_batch_tokens 约束）
     │
     ▼
infer.generate（prefill + decode；采样/停止全在 infer 内）
     │
     ▼
按每行真实 prompt_lengths 截取 completion（不按 padded 宽度切）
     │
     ▼
按输入 ordinal 恢复顺序 → 写出
```

要点：

- 左 padding 的 collate 位于 predict IO；attention mask 与位置必须正确。**pad token 等于 EOS 时也不能通过扫描 padding 推断停止**。
- `prediction.batch_size` 是每个逻辑 DP lane 的最大请求数，**不要求**总输入数或 batch_size 被 DP 整除。
- `max_batch_tokens` 表示 padded prompt tokens，不代表完整 KV 显存预算。
- 默认不静默截断。`prompt_tokens + max_new_tokens` 超过已验证模型上下文时失败；截断策略若以后增加，须显式配置并记录。

### 11.3 生成返回与输出记录

既有 `generate()` 默认仍返回 Tensor，内容是去掉左 padding 后的 prompt + completion。可选结构化返回 `GenerationResult`（见 §6.6），默认行为不变。

Predictor 根据每行真实 `prompt_lengths` 截取生成部分；`generated_lengths` 与停止状态由生成循环提供，**不能靠去除 pad token 猜测**。

输出记录：

```json
{"id":"q-1","ordinal":0,"sample_idx":0,"generated_text":"答案是 123。","prompt_tokens":28,"generated_tokens":9,"finish_reason":"eos"}
```

| 字段 | 说明 |
|---|---|
| `finish_reason` | 至少区分 `eos`、`length`，以及生成循环实际触发的 `stop` |
| `prompt_tokens` / `generated_tokens` | 来自生成循环的结构化返回，不是事后数 pad |
| `sample_idx` | n>1 时的候选序号（B0 起启用） |

可选保存原始输入、prompt/generated token IDs；默认 `generated_text` 仅含生成内容。dummy 行不输出。事后文本截断单独记录后处理原因，不改写真实 `finish_reason`。

### 11.4 输出提交与重跑

```text
每个逻辑数据分片一个 writer
        │
        ▼
写本次任务独立临时目录  →  关闭文件  →  提交 shard manifest
        │
        ▼
协调 rank 校验：全部分片成功？每个 (input_id, sample_idx) 恰好一次？
        │
        ▼
按 ordinal/sample_idx 合并  →  原子提交最终文件
```

| 规则 | 说明 |
|---|---|
| 共享文件系统 | P1 要求；跨节点非共享盘在支持独立传输机制前拒绝 |
| 已有输出 | 默认报错；只有 `overwrite=true` 才替换 |
| 失败 | 保留带 run ID 的临时分片，不将其标为完成 |
| 断点续跑 | P1 不承诺；B0 增加，需匹配模型/权重、数据、模板、生成参数、`base_seed`、n 等不可变实验指纹 |

B0 的续跑还需注意：batch plan、DP 分片和拓扑作为各次尝试的执行记录，在已验证能力与逐请求随机流条件下允许变化——不能对整个 manifest 做字节级相等检查。逐请求随机流或完整恢复原批次 RNG 状态是采样续跑的前置条件。

---

## 12. 分布式执行契约

### 12.1 角色与通信域

```text
                 ┌──────────── 同一请求的参与者 ────────────┐
                 │                                          │
   data owner ───┤ 为一个逻辑输入分片读取记录                │
                 │                                          │
   TP/CP rank ───┤ 按声明布局算出正确的 next-token logits    │
                 │                                          │
 sampling owner ─┤ 采样出 next_tokens，再广播 token/状态/原因 │
                 │                                          │
 output writer ──┤ 每个数据分片一个 writer，副本不重复输出    │
                 └──────────────────────────────────────────┘

   forward sync group  = 所有因 FSDP/EP/其他前向 collective 相互依赖的 rank
                         （共同决定是否继续当前生成轮）
   metric contribution group = 只按真实样本/token 的唯一贡献归约
                         （与生成同步域分开定义）
```

| 角色/域 | 契约 |
|---|---|
| 数据 owner | 为一个逻辑输入分片读取记录；向同一请求的模型并行参与者提供一致输入 |
| sampling group | 共同处理同一组请求的 TP/CP rank；一个 owner 采样 token，再广播 token、结束状态和原因 |
| forward sync group | 所有因 FSDP/EP/其他前向 collective 相互依赖的 rank；共同决定是否继续当前生成轮 |
| metric contribution group | 按真实样本/token 的唯一贡献归约，与生成同步域分开定义 |
| output writer | 每个数据分片一个 writer；TP/CP 副本不重复输出 |

各 group 从已解析 mesh 与模型能力派生，**禁止以 `global_rank % N` 或缺省 WORLD 猜测**。已验证拓扑采用显式组映射；复合布局可保守采用整个 stage 同步组，覆盖全部前向通信依赖并接受额外空跑开销。本期不实现通用通信图分析器，未建立映射的组合拒绝启用。EP 不简单归为"和 TP 一样处理同一请求"。

### 12.2 采样与停止

同一请求的 TP/CP rank 先按声明布局获得正确的 next-token logits，sampling owner 生成 next_tokens，再广播给其他参与 rank。**固定 seed 只用于随机流复现，不替代广播。**

共享 forward sync group 的生成执行：

```text
轮 0     [prefill] [prefill] [prefill] [dummy]     ← 各 lane 共同轮数
轮 1     [decode]  [decode]  [decode]  [dummy]
轮 2     [decode]  [EOS→占位][decode]  [dummy]      ← 已完成行继续占位前向
轮 3     [停止检查：组内仍有人未完成？] ── 是 ──► 继续
                                         └─ 否 ──► 退出
```

1. 每个调度窗口先确定同步域内共同 batch 轮数；本地分桶产生更少批次、输入耗尽或某轮无真实请求时提供合法 dummy batch。所有参与 rank 按相同轮次进入 prefill；独立 DP 副本不强制共同轮数。
2. 每轮更新各请求的真实 finished 状态；采样组内状态一致。
3. 在一致位置对"是否仍有未完成请求"执行组内控制归约。
4. 仅当整个同步域结束，或共同达到生成轮数上限，才能退出。
5. 已完成 lane 继续以占位 token 执行必要前向，维护约定的 cache、position 和 batch shape；不重新 prefill 无关 prompt，不累计输出。**已完成行仍可能消耗模型计算和 KV 空间。**
6. 局部 EOS、stop callback、输入耗尽、取消**均不得**直接 `break` 跳过其他 rank 仍需参与的 collective。

首批 FSDP 生成验收使用缓存开启、共同 `max_new_tokens` 和可处理 padding/dummy 的模型适配器；缓存降级为全量重算等路径须另测。固定形状 collective 或 EP 实现可能还要求跨 lane 一致的张量形状，runtime 必须校验并补齐。

### 12.3 能力矩阵

下表为**计划交付范围**，不是当前源码的支持声明。每项必须由真实模型、设备后端和拓扑对应的 ST 放行。

#### 12.3.1 为什么这些并行是必需项

模型规模到一定量级后，`P0` 的"dense DP / 完整副本"就不再是一个可选项，而是**根本放不下**：

| 规模 | 权重（bf16） | 单卡 64 GiB 能否放下 | 必需的并行 |
|---|---|---|---|
| 7B | ~14 GiB | 可以 | 单卡或 DP |
| 70B | ~140 GiB | 不行 | 至少 FSDP |
| 数百 B（MoE） | ~0.5–1 TB | 不行 | FSDP + EP |
| 1T 级（MoE + 深层） | ~2 TB | 不行 | FSDP + EP + PP |

这不是性能优化问题，而是"能不能跑起来"的问题。因此：

- **EP 是必需项**：MoE 专家权重占大头，不切分到多卡时任何单卡都放不下。
- **PP 是必需项**：层数深、单层权重也大时，即使 FSDP 能分片参数，逐层 un-gather 的通信量与峰值显存仍可能超限；训练时用 PP 切层，推理沿用同一布局最省。
- **FSDP 是基线**：Dense 模型至少要参数分片才能放下。
- **TP/CP 是常见配套**：训练配置里通常与上述并存。

还有一个决定性理由：**checkpoint 是按训练布局存的**。大模型用纯 FSDP 评估意味着要把整个 checkpoint 重新分片成另一种布局；用与训练一致的 TP/EP/PP 布局评估，则不需要 reshard，既省时也避免转换引入的差异。

因此能力矩阵按"支撑多大模型"排序，而不是按实现难度排序。

#### 12.3.2 各维度真正缺什么

拒绝一个拓扑，不代表它需要的机制都一样。下表区分"只差数值验收"与"还缺机制"：

| 维度 | 缺什么 | 已有基础 |
|---|---|---|
| **EP** | 主要是验收：验证路由通信在 no_grad 前向下正常，以及专家切分不影响 loss 统计域 | 每个 token 的 loss 只算一次；router aux 是 `objective`，本就不进 PPL。训练侧 all-to-all dispatch/combine 在 no_grad 下同样可用 |
| **CP** | 验证真实模型的 CP mask、cache 与 shard metadata | token 按序列切分，各 CP rank 贡献**互不相交**，按 CP 组求和即可；训练侧已按 `dp_size × cp_size` 归约 |
| **TP / SP** | **统计域最容易算错**：vocab-parallel LM head 下 loss 已跨 TP 归约过，SP 时 token 计数还要在 TP 组内再 reduce；直接照搬训练 backward 的缩放系数会重复计数 | `mean_global_loss` 已有 `sequence_parallel` 分支处理这一情形，可复用其语义 |
| **PP** | **唯一需要新机制**：只有最后一级持有 LM head，需 forward-only 调度 + 末级标量回传 | `PipelineStage(has_backward=False)` 参数已存在（[stage.py](../../hyper_parallel/core/pipeline_parallel/stage.py)），但当前无生产调用方；三种 schedule 的 exec order 均含 BWD 段 |

PP loss eval 的具体做法：在最后 stage 计算统计，再向需要输出结果的 rank 传播**标量**——无需广播全量 logits。GPipe 的两段式 exec order（先全部 FWD、再全部 BWD）是三条 schedule 中最容易裁剪出 forward-only 变体的。

#### 12.3.3 矩阵

| 场景 | P0 loss eval | P1 predict | 放行条件 |
|---|---|---|---|
| 单卡文本模型 | ✅ 支持 | ✅ 支持 | loss/生成契约分别验证 |
| 完整权重的独立 DP 副本 | ✅ 支持 | ✅ 支持 | 无跨 lane 前向 collective；尾批完整且结果唯一 |
| Dense FSDP/HSDP，TP=CP=EP=PP=1 | ✅ 支持 | ❌ 拒绝 | 生成侧须在 P2 完成全域停止、dummy 前向和缓存验证 |
| **EP**（MoE 必需） | ❌ 拒绝 | ❌ 拒绝 | P2 最优先：先于其他维度放行。验收=路由通信在 no_grad 下正常、损失统计域与单卡一致 |
| **PP**（大模型必需） | ❌ 拒绝 | ❌ 拒绝 | P2：forward-only schedule + 末级标量回传；PP predict 另行设计 |
| CP | ❌ 拒绝 | ❌ 拒绝 | P2：验证真实模型 CP mask/cache/shard metadata；不能只凭 toy 示例放行 |
| TP / SP | ❌ 拒绝 | ❌ 拒绝 | P2：loss 统计域、logits layout、owner 采样和输出去重一起验收 |
| 复合并行（如 FSDP+EP+PP） | ❌ 拒绝 | ❌ 拒绝 | P2 按具体模型/布局逐项增加；复合布局的同步域须单独验收 |
| VLM | ❌ 拒绝 | ❌ 拒绝 | P4：增加 processor、视觉输入、模型前向和缓存契约 |

现有 [CP 示例](../../examples/generate/cp_generate_check.py) 和 [HF 对齐示例](../../examples/generate/hf_alignment_check.py) 用作测试模式参考，**不作为全部模型已支持的证据**。

**支持声明的含义**：矩阵中的"支持"针对**该行描述的完整布局**，不是单个维度。例如放行 EP 之后，"FSDP+EP" 是否支持取决于该复合布局自己的 ST 结果——复合布局的通信域不是各维度规则的简单叠加。

---

## 13. 配置 Schema 与校验

### 13.1 完整示例

见 §3.2（eval）与 §3.3（predict）。新增配置区的字段汇总：

| 区 | 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|---|
| 顶层 | `run_mode` | `train \| eval \| predict` | `train` | 模式分发 |
| 顶层 | `model_assets` | `ModelAssetsConfig` | `None` | tokenizer/chat template |
| `evaluation` | `mode` | `loss \| generation` | `loss` | 评估方式 |
| `evaluation` | `split` | `valid \| test \| dataset` | `valid` | 数据来源 |
| `evaluation` | `batch_size` | int ≥ 1 | `1` | 每 lane 每次前向样本数 |
| `evaluation` | `max_samples` | int \| null | `null` | 全局样本上限 |
| `evaluation` | `max_batches` | int \| null | `null` | 前向轮数上限 |
| `evaluation` | `dataset` / `dataloader` | 配置对象 | `null` | 覆盖顶层 |
| `evaluation` | `metrics` | `list[Target]` | `[]` | eval 模式必填至少一项 |
| `evaluation` | `generation` | `GenerationOptions` | `null` | P3 |
| `prediction` | `input` / `output_file` | str | `null` | P1 |
| `prediction` | `batch_size` | int ≥ 1 | `1` | 每 lane 请求数 |
| `prediction` | `max_batch_tokens` | int \| null | `null` | padded prompt 预算 |
| `prediction` | `bucket_by_length` | bool | `false` | 长度分桶 |
| `prediction` | `overwrite` | bool | `false` | 覆盖已有输出 |
| `prediction` | `generation` | `GenerationOptions` | `null` | 采样参数 |

### 13.2 校验时点

```text
   YAML + CLI override 全部应用
              │
              ▼
   ┌──────────────────────────────┐
   │ 时点 1：纯配置               │  模式、必填项、Target、字段单位、
   │ 分布式初始化前               │  指标冲突、依赖、compile、覆盖策略
   └──────────────┬───────────────┘
                  ▼
   ┌──────────────────────────────┐
   │ 时点 2：模型/mesh 解析后     │  capability、checkpoint、loss 归一化、
   │ 任务开始前                   │  NLL、assets、特殊 token、通信组、writer
   └──────────────┬───────────────┘
                  ▼
   ┌──────────────────────────────┐
   │ 时点 3：输入批构建后         │  非空 prompt、上下文长度、token budget、
   │ 前向前                       │  batch layout、reference 隔离、dummy mask
   └──────────────────────────────┘
```

| 时点 | 检查 |
|---|---|
| 1 | 模式、必填项、Target、字段单位、指标冲突、依赖、compile、输出覆盖策略 |
| 2 | capability、checkpoint、loss 归一化、NLL、assets、特殊 token、所有通信组与 writer、共享存储 |
| 3 | 非空 prompt、上下文长度、token budget、batch layout、参考答案隔离、dummy mask |

首期对未覆盖组合给出**具体错误**，例如"FSDP 生成尚未启用同步停止协议"，不能降级成未声明的运行方式。外部 metric 依赖缺失时，只有选用该 metric 才报错；不得静默跳过请求的指标。

`evaluation.mode = generation` 时要求 generation 和文本指标配置，不能同时启用 loss-only 指标。生成式评估由 Trainer 组合 infer 生成 callable 与独立 reference 数据，**不把包含答案的 teacher-forcing 序列直接作为 prompt**。

---

## 14. 性能与显存

### 14.1 首期默认策略

| 优化 | 行为 | 验证指标 |
|---|---|---|
| 去掉训练态组件 | 不创建 optimizer/main params、scheduler、backward 包装 | 启动时间、首个前向前显存 |
| 设备端指标累计 | batch 内不做 host loss 同步，结束后统一归约 | eval tokens/s、统计 collective 次数 |
| loss-only 前向 | `use_cache=False`，及时释放输出；支持时使用模型原有 chunk loss | 峰值 allocated/reserved memory |
| 生成长度分桶 | 有界窗口内分桶，按 token budget 组 batch，输出恢复原序 | padding 比例、prefill/decode 吞吐 |
| 有界 IO | 分块读输入、流式写分片，不把全量 token/logits 留内存 | host RSS、输出完整性 |

### 14.2 容量模型与长上下文

标准 MHA/GQA 的未分片 KV 粗估：

```text
KV bytes ≈ batch × total_sequence_length × layers × 2 × kv_heads × head_dim × dtype_bytes
```

模型使用 MLA、滑窗、不同层缓存布局或复制 KV heads 时，须由适配器给出对应估算。TP/CP 后的每卡 KV **不能一律除以并行度**；DP 仅分担请求，不降低单条请求的序列长度。

总预算还需包含参数常驻/临时 unshard、attention mask、prefill logits、通信 buffer 和 kernel workspace。

**dense prefill mask 本身是 O(B×S²)**：

| 序列长度 S | 单个 FP32 S×S 张量 |
|---|---|
| 8192 | 256 MiB |
| 32768 | 4 GiB |
| 65536 | 16 GiB |

实际构建还有临时分配。长生成与长 prompt 要分别测试。

当前 `max_new_tokens` 没有 64k 配置硬上限。64k 能力取决于模型上下文、attention/cache 路径和预算；长 prompt 支持必须先完成紧凑 mask 或融合 causal attention 适配及数值验证，**不通过"放开校验"宣称支持**。

### 14.3 后续优化及测量口径

FSDP decode 的逐层参数 all-gather 可能成为主要开销。P2 在显存允许时比较保留 unsharded 参数、reshard/prefetch 策略和不同推理布局，记录通信与显存权衡；不默认调用 `empty_cache`，也不以梯度 all-reduce 开关解决无梯度前向 OOM。

静态 batch 中已完成行仍参与 forward；按长度分桶先降低浪费，动态行压缩留到后续。多采样 n 不等于物理 batch size，应拆成有界 micro-batch。

性能报告固定模型/权重、设备、dtype、拓扑、输入/输出长度分布、seed 和 warmup。分别记录 prefill tokens/s、decode tokens/s、有效输出 tokens/s、峰值设备内存、启动耗时与输出 IO；**不得用少评样本或更短输出换取表面吞吐**。

---

## 15. 生成式评估与基准扩展

### 15.1 生成式评估

生成式 Evaluator 接收 prompt/reference batch、注入的生成 callable 和文本 metric。生成调用仍走 infer；**参考答案不得进入 prompt**。生成长度与截断行为进入评测元数据。

ROUGE 明确按样本聚合方式；corpus BLEU 汇总可加的匹配数、候选/参考长度后统一 compute，不能直接平均各 rank 的 BLEU。空输出保留为有效样本；mask/dummy 不进入分母。依赖按所选 metric 加载。

### 15.2 基准分层

```text
   benchmarks/（仓库内）                 外部项目
   ┌────────────────────────┐          ┌──────────────────────────┐
   │ 题集读取 → 标准请求     │          │ 官方 harness             │
   │ 输出提取（boxed/fence） │  jsonl   │ 沙箱执行（LCB）          │
   │ 纯函数判分与统计        │ ◄──────► │ docker harness（SWE）    │
   └────────────────────────┘          │ agent 工具链             │
                                        └──────────────────────────┘
```

| 基准 | 适配责任 | 判分契约 |
|---|---|---|
| AIME 25/26 | 题目 → messages；completion → 按版本固定的答案解析 | 整数 exact match、avg@k、maj@k、pass@k |
| LiveCodeBench | 固定 release/时间窗、官方 prompt/代码提取格式、输出转换 | 官方 harness 运行结果 → pass@k，保留失败与超时 |
| SWE-bench Verified | issue/repo 标识、外部生成或 agent backend adapter、patch 输出 | 官方 harness，报告 instance resolved rate |

可选 agent 在外部读取文件、执行命令、生成 patch；SWE-bench 核心接口接受 patch，不要求必须采用多轮 agent。工具协议、容器调度和上下文管理**不进入** infer/Predictor。

### 15.3 多采样与复现

适配层展开 `(input_id, sample_idx)`，保留生成循环每行一个候选的语义。不同样本允许得到相同文本，**不能去重后重采样**，否则改变统计分布。

复现分两级：

| 级别 | 固定什么 | 承诺什么 |
|---|---|---|
| P1 任务级复现 | seed、输入顺序、batch plan、拓扑、软件环境 | manifest 记录这些条件；不承诺改变 batch/DP 后逐样本输出相同 |
| B0 请求级随机流 | 按 `base_seed` + 稳定 input ID + `sample_idx` 派生种子；sampling owner 为每个候选维护独立 generator | 支持改变分桶、分片与续跑；即使随机流相同，batch shape、kernel、归约顺序或设备差异仍可能改变 logits——不承诺 token 序列在任意执行布局下位级一致 |

greedy 下 n>1 默认拒绝，避免把重复生成误当作多次随机采样。采样参数、n、模板、模型 revision、数据 hash、提取器与 harness 版本均进入 manifest。

### 15.4 判分口径

AIME 按指定版本解析最终答案，统一整数格式后检查 `0 <= answer <= 999`；**不对任意输出取模**。超范围、无法解析、缺失输出视为失败，保留原始 completion 与解析状态。该范围来自 [MAA 官方说明](https://maa.org/maa-invitational-competitions/)。

对每题 n 个采样、其中 c 个正确：

```text
avg@n  = 各题 c/n 的平均值
pass@k = 各题 [1 - C(n-c, k) / C(n, k)] 的平均值，要求 n >= k
maj@k  = 对固定 k 个样本投票，比较最高票答案与标准答案
```

`maj@k` 的平票规则固定并版本化；本方案默认平票判失败。解析失败保留在样本预算中，**不通过删除失败记录提高分数**。`avg@n` 不等于 `pass@n`，greedy accuracy 与 sampled pass@1 分开标注。

LiveCodeBench 固定题集 release、日期窗、prompt、代码提取器和 [官方 harness](https://github.com/LiveCodeBench/LiveCodeBench) commit；超时、编译错误和错误代码计失败。环境故障作为运行错误报告，不静默减少分母。

SWE-bench 提交 `instance_id/model_name_or_path/model_patch`，遵循 [官方评测格式](https://www.swebench.com/SWE-bench/guides/evaluation/)。resolved 采用官方实例判定，不以测试用例通过百分比替代。报告区分已解决、未解决与基础设施错误，记录模型和 agent 配置以避免把 agent 差异归因于模型。

### 15.5 停止、会话与执行隔离

首版自定义 stop string 可以生成后截断，但必须把"实际生成 token"与"展示/计分截断文本"分开记录；代码 fence 或第一个 boxed **不默认等于**正确停止位置。行级 stop 扩展在 infer 内合并 finished mask，并遵守分布式同步停止协议。

多轮会话首版全量重新编码与 prefill。prefix cache 复用只有在重新编码后的 token 前缀完全一致、缓存长度/位置/mask 兼容时才成立；chat template、工具消息重写与截断可能破坏前缀关系。

LCB/SWE 执行复用外部工具链，在隔离容器中限制 CPU、内存与时限；镜像和依赖准备完成后，测试执行阶段禁网。核心库不内嵌生成代码执行器。

---

## 16. 分期与验收

阶段编号的含义一览（P = 主线功能，B = 基准评测支线；编号顺序即交付顺序）：

| 阶段 | 一句话说明 | 交付物的使用者 |
|---|---|---|
| **P0** | 独立 loss/PPL 评估 | 想给 checkpoint 打分的人 |
| **P1** | 批量生成（Predictor） | 想看模型实际输出的人 |
| **P2** | 并行拓扑逐项放行 | **需要多卡才能放下的模型** |
| **P3** | 生成式指标（Rouge/BLEU） | 需要文本相似度指标的人 |
| **P4** | VLM 评估 | 多模态模型 |
| **B0–B3** | 基准评测（AIME / LCB / SWE） | 做基准报告的人；不阻塞主线 |

编号不是"版本号"，只表示"什么时候能做什么"。P2 之所以排在 P1 之后，是因为它同时覆盖生成与评估两条线的并行支持，依赖 P0/P1 已经跑通的基础链路。

```text
P0 ──► P1 ──► P2 ──► P3 ──► P4
 │      │      │        │       │
 │      │      │        │       └─ VLM
 │      │      │        └───────── 生成式评估、文本指标
 │      │      └────────────────── FSDP 生成同步；loss eval 的 EP → PP → CP → TP/SP
 │      └───────────────────────── 文本 Predictor、manifest、输出提交
 └──────────────────────────────── 独立 loss/PPL 评估（PR #1568）

B0 ──► B1 ──► B2 ──► B3       （基准，不阻塞 P0/P1）
 │      │      │      │
 │      │      │      └─ SWE-bench 格式与 adapter
 │      │      └──────── LiveCodeBench
 │      └─────────────── AIME
 └────────────────────── n>1、请求级 RNG、续跑 manifest
```

功能和对应正确性测试必须同阶段交付。P1 依赖 P0 的模式装配、assets 与 model-only restore；P2 按能力逐项放行，不等待一个包含所有拓扑的大 PR。

P2 内部的顺序不是任意的：模型大到一定程度后，EP 与 PP 是"能否放下"的前提（§12.3.1），而 checkpoint 通常按训练布局保存。先放行 EP/PP 可以让大规模模型用与训练一致的布局评估，避免整体 reshard。TP/SP 放在最后，是因为它的统计域最容易算错，需要最仔细的数值对照。

| 阶段 | 内容 | 必须通过的验收 |
|---|---|---|
| **P0** | 模式/配置/资产解耦、权重恢复、有限 eval 数据、Evaluator、Loss/PPL；单卡/DP/Dense FSDP loss eval | checkpoint 恢复前后一致；完整样本覆盖；不同 token 数加权一致；全量 indexed 语义；旧 train 配置回归 |
| **P1** | 文本 Predictor、prompt/messages、基础 manifest、GenerationOptions、结构化返回、单卡/独立 DP、输出提交 | 不需要 dataset/optimizer；HF greedy 对齐；非整除输入完整有序；prompt/completion 边界；默认拒绝未覆盖拓扑 |
| **P2** | FSDP/HSDP 生成同步。loss eval 的拓扑按**大模型可用性**排序：EP → PP → CP → TP/SP（见 §12.3.1）；复合布局按实际使用顺序逐项接入 | 不同 EOS 长度、空 lane、dummy、组内采样一致；真实模型数值/输出一致；超时保护验证不挂起；每个新拓扑的 loss 统计与单卡参考一致 |
| **P3** | 生成式评估、文本指标；已支持拓扑的统计扩展 | reference 不泄漏；分布式统计等于单进程参考；可选依赖与空输出语义 |
| **P4** | VLM（PP loss eval 已在 P2 交付） | VLM processor/视觉输入/cache 的模型验收 |
| **B0** | n>1、请求级 RNG、恢复指纹校验与续跑 manifest 扩展 | 固定环境下改变 batch 分桶/DP 请求分片，随机流稳定；重复/缺失候选检测 |
| **B1** | AIME，依赖 P1+B0 | n=16 有界 micro-batch 出分；超范围答案拒绝；avg/maj/pass 与手算一致 |
| **B2** | LCB，依赖 P1+B0 | 固定 release/时间窗，官方 harness 小集对齐，pass@1/pass@10 |
| **B3** | SWE 格式/外部 backend adapter | 固定子集官方 resolved rate；模型、agent、harness 元数据齐全 |

B0–B3 不阻塞 P0/P1；使用未开放拓扑或长上下文时，必须先通过对应 P2/容量验收。每个合入 PR 遵循仓库单 commit 约定，阶段可按依赖拆分多个 PR。

---

## 17. 测试计划

UT 位于 `tests/ut/evaluation/`、`tests/ut/predict/`，配置/Trainer/数据/checkpoint 测试放在各自既有目录；ST 位于 `tests/torch/evaluation/`、`tests/torch/predict/`。遵循[测试规则](../../.agent/rules/testing.md)，分布式 launcher 不导入 Torch 或 HyperParallel，worker 执行真实计算。

| 测试面 | 必测案例 |
|---|---|
| 构建与配置 | 默认 train；eval/predict 无 optimizer；predict 无 dataset；空缺省 assets 不冲突、等值 Target 不误判；真实冲突；dotted override；未支持组合早失败 |
| 权重 | 初始权重与训练 checkpoint 明显不同的模型；`restore_from`/`LATEST`；缺 key；adapter-only；外部状态拒绝；optimizer 未创建 |
| 数据覆盖 | N=0、N=1、N<DP、N 不整除 DP×batch；`max_samples`/`max_batches`；新 iterator 重复评估；全量 indexed 不重复 epoch；P0 一对多 transform/动态 packing 明确拒绝 |
| token 与 loss | 不同长度、ignore index、loss_mask、shift_labels、全 mask、dummy；local mean/sum/global_mean；辅助损失不进入 PPL |
| 指标生命周期 | 多 batch 手算；一次 synchronize；compute 重复调用；reset；零分母；非有限值；重复 metric key |
| 生成接口 | 左 padding、pad=EOS、不同 prompt 长度、EOS 在首 token、`max_new_tokens=0`、长度终止、结构化返回与旧 Tensor 返回一致 |
| 分布式生成 | 同请求跨 TP/CP token 一致；FSDP/EP 不同结束步；尾批空 lane；错误 group/layout 被拒绝；collective 超时保护 |
| 输出 | ID 唯一且完整、顺序恢复、单 writer、原子提交、已存在输出、失败分片、B0 续跑 manifest 校验 |
| 生成指标/基准 | prompt/reference 隔离；BLEU 充分统计量；n 与 k；重复样本保留；无效答案；harness 环境错误与模型失败分开 |
| 性能 | warmup 后固定工作量，eval/predict 吞吐、峰值内存、通信次数；长 prompt 与长 decode 分别测量 |

DP 数值对比必须使用**相同全局样本 ID 和有效 token 集合**，不能只比较两个恰好相同的最终 loss。TP/CP 既测 toy 契约也测真实模型；greedy 对齐先确认相同权重、tokenizer、模板、EOS 和 mask，再比较 token 序列。

---

## 18. 实现前检查点

以下事项由实现 PR 提供证据，不作为默认可用假设：

| 检查点 | 关闭条件 |
|---|---|
| Indexed 全量有限范围 | 明确逻辑 sample 定义、尾窗口策略、有效 token 覆盖；固定 seed 可重复 |
| 自定义 loss 的统计域与 NLL | 每种支持的 loss adapter 声明归一化与辅助项，单卡/分布式对照通过 |
| FSDP 无梯度生命周期 | 连续 eval/generate 无参数/通信 buffer 泄漏，dummy 前向和 reshard 行为验证 |
| 模型外部权重 | 无 optimizer restore 能完整恢复外部参数；不支持时在前向前拒绝 |
| CP/EP 缓存与通信 | 模型 provider 声明有效组合，真实模型 prefill/decode 与停止测试通过 |
| 长上下文 | 参数、KV、mask、logits 和临时内存预算一致，代表长度实测通过 |
| 数据与输出持久化 | 全局样本范围、单 writer、输出完整提交、manifest 可追溯 |

本文交付设计与验收契约。实现完成前，不将计划中的并行组合、64k 长度或基准分数标记为已支持。
