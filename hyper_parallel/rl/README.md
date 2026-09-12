<!-- markdownlint-disable MD033 -->
<h1 align="center">
  <img src="docs/assets/hyper-rl-logo.svg" width="120" height="80" alt="HyperParallel-RL logo"><br>
  HyperParallel-RL · Experimental
</h1>

<p align="center"><strong>A lightweight, extensible reinforcement learning framework.</strong></p>

<p align="center">Built on HyperParallel and vLLM, with explicit training orchestration and modular interfaces.</p>

<p align="center"><strong>极简易用 · 易于扩展 · 昇腾亲和</strong></p>

<p align="center"><strong>中文</strong> | <a href="README.en.md">English</a></p>

<p align="center"><a href="#why-hyper-rl">Why HyperParallel-RL</a> · <a href="#架构">架构</a> · <a href="#安装与环境">安装</a> · <a href="#快速开始">快速开始</a> · <a href="#扩展与定制">扩展与定制</a> · <a href="#支持范围">支持范围</a> · <a href="#文档">文档</a> · <a href="#citation">Citation</a>
</p>
<!-- markdownlint-enable MD033 -->

---

HyperParallel-RL 是面向 LLM、VLM、世界模型、具身智能和 Agentic AI 的强化学习框架，以**极简易用、易于扩展、昇腾亲和**为设计原则，以**规模化高效训练**为架构目标。

用户通过 Python 定义任务、交互、工具与奖励，基础策略优化与多轮 Agent 交互复用同一训练链路。HyperParallel 承载并行训练，vLLM 承载采样，HyperParallel-RL 通过显式编排、统一轨迹与策略发布连接两者。

> **实验版本**：当前验证范围为单节点 Ascend NPU、同步 GRPO、Qwen3 dense 及部分 MoE 路径。异步、多模态、世界模型与具身智能的端到端训练尚未提供。具体能力与验证状态见[支持范围](#支持范围)。

**开始使用**：[安装环境](#安装与环境) → [运行一个训练步](#快速开始)　｜　**开始定制**：[替换奖励函数](#替换奖励函数)

---

<!-- markdownlint-disable-next-line MD033 -->
<a id="why-hyper-rl"></a>

## 🌟 Why HyperParallel-RL

### 🪶 极简易用

减少部署依赖，保持训练流程易于理解和运行。

- **精简技术栈**：HyperParallel 负责训练与并行，vLLM 负责采样；同步训练直接编排，无需 Ray。
- **显式训练流程**：SyncTrainer 组织采样、学习、权重发布、评估与恢复，便于追踪执行和定位问题。
- **明确的运行起点**：提供[固定运行镜像](docs/hyper_rl_runtime_image.md)、模型配置与启动脚本，通过 Qwen3-4B [单步检查](#快速开始)验证训练链路。
- **人类可读，Agent 可追踪**：人类通过[功能导航](../../docs/rl-navigation.md)定位配置、实现与测试；开发 Agent 从 [AGENTS.md](../../AGENTS.md) 进入 [HyperParallel-RL 规则](../../.agent/rules/hyper-rl.md)，按任务读取同一套权威文档。

### 🧩 易于扩展

明确扩展边界，复用已有训练链路。

- **算法与编排解耦**：Advantage 与 policy loss 可注册扩展，兼容现有训练角色的算法无需复制 Trainer；采样与学习的执行机制由核心编排定制。
- **Python 定义任务、工具与奖励**：替换评分逻辑或增加工具交互，无需修改分布式训练与权重发布实现。见[奖励定制示例](#替换奖励函数)。
- **两种 Agentic 交互方式**：Environment 由框架驱动逐轮交互，AgentProgram 承载用户程序；内置 Codex / DeepSeek Harness 已接入，自定义程序仍需适配。见 [Agentic RL](docs/agentic_rl.md)。
- **模型与后端按边界适配**：新模型复用 HyperParallel 能力，补齐训推适配与权重映射；其他推理后端可在下游 fork 中扩展并独立验证。见[扩展接口](docs/architecture.md#扩展点)与[基础设施选型](docs/design.md#基础设施选型)。

### ⚙️ 昇腾亲和

结合昇腾的计算、内存与通信能力组织训练和采样。

- **FSDP 并行训练**：复用 HyperParallel 的状态分片能力及预取、通算重叠等优化机制，以已验证的 FSDP、TP、EP 组合支持 dense 与部分 MoE 路径。见 [FSDP 优化](../../docs/guide/fsdp.md#fsdp-性能优化)。
- **共卡与分离部署**：Qwen3 dense 支持训推共卡或独立设备部署，分别通过 NPU IPC、HCCL 发布权重；共卡按阶段释放与恢复采样侧资源。MoE 当前限于共卡，见[支持范围](#支持范围)。
- **流式权重同步**：分桶传输与确认后释放缓冲控制临时内存占用；显式 direct-reshard 按训推分片交集传输参数，同时保留 full-gather 路径。见 [vLLM Rollout](docs/vllm_rollout.md)。

底层优化基础还包括 HyperParallel 的[单边通信与多核 MoE 通算重叠](../../docs/guide/multicore_moe.md)，其 HyperParallel-RL 接入与端到端收益待验证。

### 🔗 训推一致性

保留生成依据，校验概率差异，明确策略发布边界。

- **原始样本贯通训练**：保留生成 token、logprobs、动作掩码与策略身份，避免重新分词改变样本；工具反馈与环境观察仅作为上下文，不参与 policy loss。见[轨迹合同](rl/dataset/contracts.py)。
- **同源语义与 Bit-Exact 校验**：Qwen3 dense 的 Trainer 与 Hyper-vLLM 共享参数语义和 TP 切分方案；可选校验在更新前逐位比较有效动作 token 的 FP32 raw logprobs，失败即阻止更新与发布。
- **已发布策略才可采样**：同步流程消费同一已发布版本的轨迹；新权重完成传输、worker 身份校验与缓存重置后才恢复采样，参数发布与概率一致性分别验证。

Bit-Exact 默认关闭，已验证范围为单节点 Ascend、BF16/eager、Qwen3 dense + Hyper-vLLM、训推 matched TP1/TP2；不等同于梯度或收敛保证。完整条件与结果见[训推一致性文档](docs/qwen3_training_inference_consistency.md)。

---

<!-- markdownlint-disable-next-line MD033 -->
<a id="架构"></a>

## 🏗️ 架构

![HyperParallel-RL 同步架构：任务扩展、vLLM 采样、统一轨迹、HyperParallel 训练与策略发布](docs/assets/hyper-rl-architecture.svg)

用户扩展定义任务与学习目标，HyperParallel-RL 核心组织训练闭环，基础设施承载推理与并行训练。实线箭头表示采样到策略发布的循环，虚线表示扩展入口。

组件与状态边界见[架构文档](docs/architecture.md)，配置、实现与测试入口见[功能导航](../../docs/rl-navigation.md)。

---

<!-- markdownlint-disable-next-line MD033 -->
<a id="安装与环境"></a>

## 📦 安装与环境

推荐使用**仓库源码 + 固定运行镜像**。当前运行环境为 Linux ARM64 与 Ascend NPU；以下基础示例使用四张空闲 NPU，Agentic 示例使用两张。宿主机需具备 Docker、兼容的 NPU driver，以及模型和数据存储空间。镜像所需空间与驱动要求见[运行镜像](docs/hyper_rl_runtime_image.md#宿主要求)。

### 1. 获取源码

```bash
git clone --branch rl https://gitcode.com/mindspore/hyper-parallel.git
cd hyper-parallel
```

后续命令均从仓库根目录执行。已有源码时，使用与运行镜像兼容的版本。

### 2. 准备运行镜像

```bash
docker pull swr.cn-east-3.myhuaweicloud.com/huawei-hyper-rl/hyper-rl:v0.22.1rc1-arm64

npu-smi info
```

镜像包含 CANN、Torch / torch-npu、Transformers 与 vLLM / vLLM-Ascend。启动脚本挂载源码、driver、模型、数据和结果目录，并设置导入路径，无需额外执行 `pip install`。

开发时直接修改本地源码，后续运行会挂载修改后的版本。镜像 digest、依赖校验与源码构建方式见[运行镜像文档](docs/hyper_rl_runtime_image.md)。

---

<!-- markdownlint-disable-next-line MD033 -->
<a id="快速开始"></a>

## 🚀 快速开始

完成[环境准备](#安装与环境)后，从仓库根目录运行以下任一示例。替换宿主机路径，使用完整 Qwen3-4B checkpoint 与 tokenizer，并通过 `npu-smi info` 选择空闲、健康的设备；共用设备时依次运行。

<!-- markdownlint-disable-next-line MD033 -->
<a id="qwen-example"></a>

### 示例一：Qwen 基础训练

**Qwen3-4B + GSM8K，四卡单步同步 GRPO。** 数据目录需包含 `train.parquet`、`test.parquet`，使用 `prompt`、`extra_info` 列；格式与转换见[数据准备](examples/agents/gsm8k/prepare_gsm8k_m3.py)及[读取规则](rl/dataset/data_source.py)。

```bash
export HYPER_QWEN3_TP_MODEL_ROOT=/absolute/path/to/Qwen3-4B
export HYPER_QWEN3_TP_DATA_ROOT=/absolute/path/to/gsm8k
export HYPER_QWEN3_TP_RESULT_ROOT=/absolute/path/to/results/qwen
export HYPER_QWEN3_TP_VISIBLE_DEVICES=0,1,2,3
export HYPER_QWEN3_TP_TRAINER_TP=2
export HYPER_QWEN3_TP_ROLLOUT_TP=2
export HYPER_QWEN3_TP_MAX_STEPS=1

./hyper_parallel/rl/examples/scripts/run_qwen3_tp_docker.sh colocated
```

**成功判据**：退出码为 0，结果目录日志中 `train/global_step=1`、`policy/version=1`，`train/total_loss` 为有限值。此检查关闭评估、保存与 Bit-Exact。

<!-- markdownlint-disable-next-line MD033 -->
<a id="agentic-example"></a>

### 示例二：Agentic 多轮交互

**Qwen3-4B + Search-R1，两卡两步检索问答。** 模型调用本地检索工具并根据观察继续回答，无需外部 Agent CLI。数据目录需包含 `train.parquet`（`prompt`、`answer`）和 `corpus.jsonl`，见[数据准备](examples/agents/search_R1/prepare_search_r1_data.py)。

```bash
export HYPER_VLLM_IMAGE=swr.cn-east-3.myhuaweicloud.com/huawei-hyper-rl/hyper-rl:v0.22.1rc1-arm64
export HYPER_VLLM_MODEL_ROOT=/absolute/path/to/Qwen3-4B
export HYPER_VLLM_DATA_ROOT=/absolute/path/to/hotpotqa
export HYPER_VLLM_RESULT_ROOT=/absolute/path/to/results/search-r1
export HYPER_VLLM_VISIBLE_DEVICES=0,1
export HYPER_VLLM_MODEL_IMPLEMENTATION=native
export HYPER_AGENTIC_TASK=search_r1

./hyper_parallel/rl/examples/scripts/run_qwen3_4b_agentic_docker.sh
```

**成功判据**：退出码为 0，结果目录 `train.log` 中 `train/global_step=2`、`policy/version=2`，并生成 `checkpoints/step_2/checkpoint_complete.json`。通过日志样本检查实际工具交互。

两个示例验证运行流程，不代表学习收益。完整训练使用[训练入口](examples/train_rl.py)，基于 [GSM8K](examples/configs/qwen3_4b_gsm8k_vllm_production.yaml) 或 [Search-R1](examples/agents/search_R1/configs/multi_turn.yaml) 配置调整训练预算、评估与保存，不沿用检查脚本的固定步数判据。

其他入口：[程序化 Agent](docs/agentic_rl.md) · [部署与采样](docs/vllm_rollout.md) · [Bit-Exact 校验](docs/qwen3_training_inference_consistency.md) · [MoE 模型](docs/moe_models.md)。

---

<!-- markdownlint-disable-next-line MD033 -->
<a id="扩展与定制"></a>

## 🧩 扩展与定制

HyperParallel-RL 提供任务、数据与算法层的扩展入口。单轮任务与多轮 Agent 交互共用训练链路；涉及训练机制的定制，可基于 fork 修改相应模块。

| 扩展范围 | 实现入口 |
| --- | --- |
| 奖励与任务环境 | [GSM8K 示例](examples/agents/gsm8k/agent.py)展示环境与奖励实现，通过 `agentic.module_path` 和 `agentic.environment` 选择任务 |
| 多轮交互或自定义循环 | 使用 [Environment](rl/agentic/envs/base.py)由框架驱动交互；[AgentProgram](rl/agentic/core/program_runner.py)承载程序化循环，内置路径及新增 harness 的接入边界见 [Agentic RL](docs/agentic_rl.md)；环境观察不参与 policy loss |
| 数据与训练样本 | [数据源](rl/dataset/data_source.py)负责输入，[轨迹与 batch 合同](rl/dataset/contracts.py)定义进入学习阶段的数据 |
| 学习算法与更新过程 | 从 [loss](rl/algorithm/loss.py)、[advantage](rl/algorithm/advantage.py)和 [SyncTrainer](rl/trainer.py)定位学习逻辑，借助[功能导航](../../docs/rl-navigation.md)找到相关测试 |

### 替换奖励函数

以下示例复用 GSM8K 多轮环境与计算器，仅替换终局评分。将代码保存到 `hyper_parallel/rl/examples/agents/custom_reward.py`：

```python
from examples.agents.gsm8k.agent import GSM8KMultiTurnEnvironment, build_gsm8k_environment
from rl.agentic.core.types import EpisodeContext
from rl.agentic.envs.environment import ENVIRONMENTS
from rl.dataset.contracts import PromptRecord


def exact_match(answer: str, prompt: PromptRecord) -> float:
    """Score the final answer against the dataset target."""
    return float(answer.strip() == str(prompt.ground_truth).strip())


@ENVIRONMENTS.register("custom_gsm8k")
def build_environment(context: EpisodeContext) -> GSM8KMultiTurnEnvironment:
    """Reuse multi-turn tool interaction with a custom terminal reward."""
    env = build_gsm8k_environment(context)
    if not isinstance(env, GSM8KMultiTurnEnvironment):
        raise ValueError("custom_gsm8k requires interaction_mode=multi_turn")
    env.reward_function = exact_match
    return env
```

在完整训练 YAML 中使用 [GSM8K 多轮配置](examples/agents/gsm8k/configs/multi_turn.yaml)的 `agentic` 设置，并将 `module_path` 改为 `examples.agents.custom_reward`、`environment` 改为 `custom_gsm8k`。该示例只替换多轮任务的终局评分，不修改 Trainer 或权重发布流程。

---

<!-- markdownlint-disable-next-line MD033 -->
<a id="支持范围"></a>

## 📋 支持范围

当前验证平台为**单节点 Ascend 910B3**，环境要求见[运行镜像](docs/hyper_rl_runtime_image.md)。✅ 已支持 · ◐ 部分验证 · 🧪 仅组件可用 · ○ 规划；状态仅适用于所列组合，不代表长期学习收益已验证。

### 训练与任务

| 能力 | 状态 | 范围 |
| :--- | :---: | :--- |
| **同步 GRPO** | ✅ | 采样、学习、发布、评估与恢复；advantage / loss 可扩展，无需 Ray |
| **单轮与多轮工具任务** | ✅ | Python 定义环境、工具与奖励，共用 token-first 轨迹；环境观察不参与 loss |
| **程序化 Agent** | ◐ | Codex / DeepSeek Harness 已接入，自定义程序需适配，见 [Agentic RL](docs/agentic_rl.md) |
| **采样与部署** | ✅ | Hyper/Native-vLLM；Qwen3 dense 支持共卡与分离部署，MoE 限共卡；DP/TP/EP 组合见 [vLLM Rollout](docs/vllm_rollout.md) |
| **权重同步与运行保障** | ✅ | 流式 full-gather / 显式 direct-reshard、IPC/HCCL、策略发布校验、checkpoint 恢复及 console / W&B 指标；见[架构合同](docs/architecture.md) |
| **Bit-Exact 校验** | ✅ | 可选更新前 logprob 校验，仅限 Qwen3 dense + Hyper-vLLM matched TP1/TP2，见[完整条件](docs/qwen3_training_inference_consistency.md) |
| **PPO / GAE / Critic** | 🧪 | 数学与角色组件已有，尚无端到端训练路径 |
| **异步与多节点训练** | ○ | Ray 采样/学习并发、策略滞后与恢复；扩展多节点及长耗时 Agent 异步 |
| **多模态训练与交互** | ○ | 视觉 RL、媒体与动作对齐、多模态 Agent |

### 代表模型

| 模型 | 状态 | 验证范围或目标 |
| :--- | :---: | :--- |
| **Qwen3 dense** | ✅ | 单节点 GRPO；Hyper/Native-vLLM TP1/TP2，Trainer 支持 TP1、pure TP2 与 FSDP-shard×TP2 |
| **Qwen3-30B-A3B** | ✅ | Native/Hyper 两步闭环与受控非零更新，四卡 TP2/EP4 权重发布验证 |
| **Moonlight-16B-A3B-Instruct** | ◐ | Hyper 两步非零学习通过；Native 发布通过，连续非零学习验收未通过 |
| **Qwen2.5-VL-7B-Instruct** | ○ | 图像数学问答与多模态 Agent |
| **DeepSeek-V3 完整模型** | ○ | 完整 checkpoint 的多节点同步基线与异步训练对照 |

MoE 的具体配置与验收见[模型文档](docs/moe_models.md)；规划依赖与量化标准见 [TODO](docs/TODO.md)，使用限制见[当前边界](#当前边界)。

---

<!-- markdownlint-disable-next-line MD033 -->
<a id="规划方向"></a>

## 🗺️ 规划方向

- **异步与规模化训练**：以 Ray 支撑采样与学习并发，先验证单节点异步，再以完整 DeepSeek-V3 验证多节点训练。
- **多模态训练**：以 Qwen2.5-VL-7B-Instruct 的图像数学问答打通视觉 RL，再验证多模态 Agent 交互。
- **Agent 学习与异步执行**：完善已有多轮环境与程序化接入，以工具和状态化代码任务验证学习效果，扩展长耗时任务的异步执行。

各方向通过完整 recipe、学习结果与资源成本对照验收；模型、依赖及量化标准统一见 [TODO](docs/TODO.md)。世界模型与具身智能属于长期定位，具体交付范围尚待定义。

---

<!-- markdownlint-disable-next-line MD033 -->
<a id="当前边界"></a>

## 📌 当前边界

- **运行范围**：当前为单节点 Ascend 同步 GRPO；模型与拓扑以[支持范围](#支持范围)为准。MoE 暂不支持分离部署与专家内部 TP；PPO / Critic 仅有组件，尚无端到端训练路径。
- **验证范围**：功能运行通过不等于长期学习收益已验证。Bit-Exact 仅覆盖指定 Qwen3 dense + Hyper-vLLM 配置的更新前 logprobs，不代表梯度、更新后参数或收敛保证，详见[一致性文档](docs/qwen3_training_inference_consistency.md)。
- **恢复与计数**：已有检查点恢复实现，但样本/token 消费计数尚未在训练步累加，不应据此统计实际训练量。见[状态与恢复边界](docs/architecture.md#状态所有权)。
- **部署要求**：vLLM 的 RLHF/refit 开发接口仅用于受信任、隔离的训练网络；环境与驱动要求见[运行镜像](docs/hyper_rl_runtime_image.md)。

---

<!-- markdownlint-disable-next-line MD033 -->
<a id="文档"></a>

## 📚 文档

### 运行与验证

| 文档 | 内容 |
| --- | --- |
| [运行镜像](docs/hyper_rl_runtime_image.md) | 镜像下载、校验、固定依赖和宿主要求 |
| [Agentic RL](docs/agentic_rl.md) | 内部环境、Codex / DeepSeek Harness 的配置、轨迹与接入边界 |
| [MoE 模型](docs/moe_models.md) | 模型、TP/EP 配置、组件归属与验证边界 |
| [Qwen3 训练-推理一致性](docs/qwen3_training_inference_consistency.md) | Bit-Exact 定义、recipe 和验收门禁 |

### 架构与演进

| 文档 | 内容 |
| --- | --- |
| [设计目标与原则](docs/design.md) | 基础设施选型、模块边界与扩展取舍 |
| [HyperParallel-RL 架构](docs/architecture.md) | 组件、数据合同、训练生命周期和边界 |
| [vLLM Rollout](docs/vllm_rollout.md) | 资源归属、采样准入、权重事务与失败语义 |
| [交付计划](docs/TODO.md) | 阶段任务、依赖关系与验收标准 |

### 开发与代码导航

| 文档 | 内容 |
| --- | --- |
| [功能导航](../../docs/rl-navigation.md) | 配置 → 入口 → 分支 → 数据/指标 → 测试 |
| [Module Map](../../.agent/rules/rl/module-map.md) | 子系统归属、代码位置与对应合同文档 |
| [公共模块修改说明](docs/public_module_changes.md) | RL 目录外修改的必要性和接口影响 |

---

<!-- markdownlint-disable-next-line MD033 -->
<a id="citation"></a>

## 📝 Citation

如使用 HyperParallel-RL，可引用以下软件条目，并在实验中注明所用 commit。

```bibtex
@software{hyper_rl_2026,
  title = {HyperParallel-RL: A Lightweight, Extensible Reinforcement Learning Framework},
  year  = {2026},
  url   = {https://gitcode.com/mindspore/hyper-parallel},
  note  = {HyperParallel-RL module in the HyperParallel repository}
}
```
