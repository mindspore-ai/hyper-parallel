# ⚡ Hyper-RL

**A lightweight framework for online reinforcement learning with LLMs and VLMs.**

Built on HyperParallel and vLLM, with explicit training orchestration and modular interfaces.

Hyper-RL 的长期目标是：**以精简、可扩展的核心，支持从基础强化学习到多轮 Agent、多模态与大规模异步训练，并尽可能实现训推一致。**

**极简易用、易于扩展、Agentic 原生**是实现这一目标的设计约束。框架聚焦 LLM/VLM 在线强化学习，由用户定义任务与交互程序。当前由 HyperParallel / HyperAutoModel 承载训练、vLLM 提供采样，SyncTrainer 显式编排同步训练流程；任务、工具与奖励通过 Python 定义。异步、多模态与更大规模训练按阶段建设，训推一致是我们追求的目标[Bit-Exact 校验](docs/qwen3_training_inference_consistency.md)，交付计划见 [TODO](docs/TODO.md)。

[💡 Why Hyper-RL](#why-hyper-rl) · [🏗️ 架构](#架构) · [🎯 支持范围](#支持范围) · [📦 安装与环境](#安装与环境) · [🚀 快速开始](#快速开始) · [🧩 扩展与定制](#扩展与定制) · [📚 文档](#文档)

> **当前运行范围：**单节点 Ascend NPU、同步 GRPO，支持 Qwen3 dense 及部分 MoE 路径；模型、部署与学习验证状态见[支持范围](#支持范围)。
>
> 开始使用：[安装环境](#安装与环境) → [运行一个训练步](#快速开始)；开始定制：[替换奖励函数](#替换奖励函数)。

## Why Hyper-RL

| 设计选择 | 能力与价值 |
| --- | --- |
| 🪶 **精简的基础设施** | HyperParallel / HyperAutoModel 负责训练，vLLM 负责采样；同步路径无需 Ray，一个 SyncTrainer 显式组织训练流程。 |
| 🤖 **Agentic 原生的任务接口** | 环境、工具与奖励通过 Python 定义。单轮与多轮任务共用[轨迹合同](rl/dataset/contracts.py)，区分策略动作与环境观察；程序化 Agent 组件复用该合同。 |
| 🔗 **可靠的训练与策略发布** | 原始 token、logprobs、动作掩码与策略身份贯穿训练。[新策略完成传输与校验后恢复采样](docs/vllm_rollout.md)，并在已验证组合中提供更新前 [Bit-Exact 校验](docs/qwen3_training_inference_consistency.md)。 |
| 🧩 **明确的扩展边界** | 任务与兼容现有角色的算法在对应模块扩展，无需复制训练流程；采样与学习关系、更新顺序等机制变化可直接修改编排。具体取舍见[设计原则](docs/design.md)。 |
| 📖 **可读、可追踪的实现** | 显式调用与状态归属便于追踪训练流程；[功能导航](../../docs/rl-navigation.md)连接配置、实现与测试。开发 Agent 从 [AGENTS.md](../../AGENTS.md) 进入 [Hyper-RL 规则](../../.agent/rules/hyper-rl.md)，按任务读取权威文档与代码。 |

## 架构

![Hyper-RL 同步架构：任务扩展、vLLM 采样、统一轨迹、HyperParallel 训练与策略发布](docs/assets/hyper-rl-architecture.svg)

任务调用 vLLM 生成样本，轨迹经 `Trajectory → ExperienceBatch` 进入学习阶段。SyncTrainer 编排训练与发布；V+1 完成权重传输、身份校验与缓存重置后，才用于下一轮采样。图中虚线标出任务与算法扩展位置；内置 Codex / DeepSeek Harness 已通过 ProgramAgentRunner 接入，任意自定义 runner 仍需适配。

Trainer 与 Hyper-vLLM 复用模型语义。可选 Bit-Exact 比较更新前的 FP32 raw selected-token logprobs，适用组合见[支持范围](#支持范围)，完整限制见[当前边界](#当前边界)。

组件合同见 [Hyper-RL 架构](docs/architecture.md)，配置与实现入口见 [功能导航](../../docs/rl-navigation.md)。

## 支持范围

**✅ 已支持** · **◐ 部分验证** · **🧪 组件可用** · **○ 规划**

已支持能力以对应模型与部署验证为准，不代表已证明长期学习收益；部分验证表示仅部分路径通过验收；组件可用尚未开放完整训练路径；规划能力尚未实现。

### ⚙️ 训练与运行时

| 能力 | 状态 | 支持说明 |
| :--- | :---: | :--- |
| **同步在线训练** | ✅ 已支持 | SyncTrainer 编排采样、奖励处理、策略更新与发布，并管理评估和检查点恢复；同步路径无需 Ray |
| **训练与采样部署** | ✅ 已支持 | Qwen3 dense 支持 colocated 共卡与 disjoint 分离部署，共用训练主循环；MoE 当前限定 colocated |
| **vLLM 采样** | ✅ 已支持 | Hyper-vLLM / Native-vLLM，支持 DP 请求路由与 TP；透传 Prefix Cache、Chunked Prefill 配置，具体组合与约束见 [vLLM Rollout](docs/vllm_rollout.md) |
|**异卡训练**| ○ 规划 | 计划完成异卡训推分离的训练流程闭环|
| **异步训练** | ○ 规划 | 计划通过 Ray 管理资源、任务与采样队列，实现采样和学习并发；需定义样本策略版本、允许的滞后及更新校正 |

### 🤖 任务、Agent 与奖励

| 能力 | 状态 | 支持说明 |
 | :--- | :---: | :--- |
 | **单轮任务** | ✅ 已支持 | 环境接收模型输出并计算奖励，适用于问答、推理等任务；提供 [GSM8K 示例](examples/agents/gsm8k/agent.py) |
 | **多轮工具交互** | ✅ 已支持 | Environment 管理观察、动作、奖励与终止；ToolEnvironment 组合协议解析、工具执行与终局评分；提供 [GSM8K 示例](examples/agents/gsm8k/configs/multi_turn.yaml)和[Search-R1 示例](examples/agents/search_R1/configs/multi_turn.yaml)|
 | **程序化 Agent** | ◐ 部分验证 | 内置 Codex / DeepSeek Harness 通过 ProgramAgentRunner 接入配置入口；任意自定义 runner 尚不能直接通过 YAML 接入，运行与验证范围见 [Agentic RL](docs/agentic_rl.md)；提供[Codex 示例](examples/agents/gsm8k/configs/codex_multi_turn.yaml)和 [DeepSeek 示例](examples/agents/gsm8k/configs/deepseek_multi_turn.yaml)   |
 | **统一轨迹标准** | ✅ 已支持 |internal、Codex 和 DeepSeek 最终均生成标准 Trajectory，再转换为 ExperienceBatch |
 | **自定义Env、奖励与工具** | ✅ 已支持 | 通过 Python 定义评分逻辑、注册工具与任务环境；工具执行支持超时和并发限制 |
 | **Token-first 轨迹** | ✅ 已支持 | 原始 token、logprobs、动作掩码、reward 与策略身份保持关联；环境观察不参与 policy loss，见[轨迹合同](rl/dataset/contracts.py) |
 | **历史对齐** | ◐ 部分验证 | 能够拼接多轮 token 历史并检测 prompt 重写；Harness 只允许有限的内容压缩，尚不支持 Harness 压缩后的完整重新对齐 |
 | **可中断与可恢复交互** | ○ 规划 | 支持长时间 episode 在超时、资源切换、训练阶段切换或主动调度时暂停，并从保存点继续执行，实现长程交互 |
 | **上下文压缩** | ○ 规划 | 当 episode 接近上下文上限时，将早期工具调用、环境观察和推理历史压缩为结构化摘要，并从压缩后的上下文继续交互 |
 | **多模态交互** | ○ 规划 | 将图像等输入及对应交互信息纳入统一任务接口，扩展样本对齐、模型适配与端到端验证 |
 | **长尾感知调度** | ○ 规划 |面向不同 episode 长度和工具延迟造成的长尾等待，引入 Partial Rollout、动态调度等方案，提高训练设备利用率和样本有效性。 |
 | **Multi-Agent** | ○ 规划 |支持多个 Agent 在同一个任务中按角色协作，包括任务分解、消息传递、工具共享、子任务执行和结果汇总 |
 | **MA信用分配** | ○ 规划 |在团队奖励之外，支持角色奖励、子任务奖励、turn-level 奖励和贡献度估计 |
### 🧮 算法与训练角色

| 能力 | 状态 | 支持说明 |
| :--- | :---: | :--- |
| **GRPO** | ✅ 已支持 | 打通训练闭环，包含可训练的 Actor、冻结的 Reference 和推理侧 Rollout，计算组相对优势估计、策略损失与 KL 项|
| **PPO** | 🧪 组件可用 | 已具备 Actor、Critic、Reference 和 Rollout 角色模块，以及 PPO 算法组件，尚未打通端到端训练闭环|
| **算法扩展** |  ○ 规划 | 计划完善 PPO 端到端训练支持，并在现有算法框架上扩展 GSPO|

### 🧩 模型与并行

| 能力 | 状态 | 支持说明 |
| :--- | :---: | :--- |
| **Qwen3 dense** | ✅ 已支持 | 单节点 GRPO；共卡和异卡Trainer FSDP×TP，配合 Hyper-vLLM / Native-vLLM DPxTP；训推一致的实现|
| **Qwen3-30B-A3B** | ✅ 已支持 | 单节点 GRPO; 共卡 Trainer FSDP×TPxEP，配合 Hyper-vLLM / Native-vLLM DPxTPxEP |
| **Moonlight-16B-A3B-Instruct** | ✅ 已支持 |单节点 GRPO; 共卡Trainer FSDP×TPxEP，配合 Hyper-vLLM / Native-vLLM DPxTPxEP   |
| **静态专家并行** | ✅ 已支持 | 两个 MoE 模型复用 HyperParallel TP-extend-EP 与 rollout 静态 EP；dense 与专家权重分别按 TP / EP 描述归属 |
|**异卡实验接入**| ○ 规划 | 补齐 Qwen3-30B-A3B 和 Moonlight-16B-A3B-Instruct 异卡 RL 实验闭环｜
|**训推一致**|○ 规划 | 实现 Qwen3-30B-A3B 的训推一致，并尽可能降低 Moonlight-16B-A3B-Instruct 的训推误差|
| **更大规模 MoE** | ○ 规划 | 计划基于现有模型、并行与流式权重同步能力扩展模型规模和拓扑；具体规模与性能以后续验收为准 |

MoE 的完整 checkpoint、并行组合、发布验收与学习验收分别记录在 [MoE 模型](docs/moe_models.md)。模型家族适配不代表该家族全部 checkpoint、并行配置或部署方式均已验证。

### 🛡️ 策略发布、校验与恢复

| 能力 | 状态 | 支持说明 |
| :--- | :---: | :--- |
| **训推权重倒换** | ✅ 已支持 | 采用流式传输，默认通过 full-gather 聚合后重新切分，结合 swap 模式完成训推权重同步；支持direct_reshard，实现训练侧与推理侧权重分片的直接映射与传输|
| **通信传输** | ✅ 已支持 | 支持同卡部署（colocated）下通过 NPU IPC 共享权重，以及分卡部署（disjoint）下通过 HCCL 传输权重|
| **数值一致性检测** | ✅ 已支持 | 可选 Bit-Exact：限定已验证的 Qwen3 dense + Hyper-vLLM matched DP/TP，满足逐token的logprob完全一致，见[一致性定义与门禁](docs/qwen3_training_inference_consistency.md) |
| **评估与可观测性** | ✅ 已支持 | 评估采样、任务奖励统计，以及训练、采样与策略发布指标；提供 console / W&B 日志后端 |
| **检查点与恢复** | ✅ 已支持 | Actor、optimizer、scheduler、RNG 和 dataloader state；完成标记校验及恢复后的策略发布，见[恢复合同](docs/architecture.md#checkpoint-与恢复) |
| **权重同步优化** | ○ 规划 | 计划利用昇腾单边通信能力优化训推权重同步，降低通信开销与同步延迟，提升权重传输效率 |

当前验证平台为单节点 Ascend 910B3。固定依赖、镜像 digest 与宿主要求见[运行镜像](docs/hyper_rl_runtime_image.md)；未覆盖能力见[当前边界](#当前边界)。

## 安装与环境

推荐使用**仓库源码 + 固定运行镜像**。当前运行环境为 Linux ARM64 与 Ascend NPU；以下快速开始使用四张空闲 NPU。宿主机需具备 Docker、兼容的 NPU driver，以及模型和数据存储空间。镜像所需空间与驱动要求见[运行镜像](docs/hyper_rl_runtime_image.md#宿主要求)。

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

## 快速开始

以下示例运行 **Qwen3-4B + GSM8K 的单步同步 GRPO**，Trainer 与 rollout 共用四张 NPU，验证采样、训练与策略发布链路。

### 1. 准备模型与数据

```bash
export HYPER_QWEN3_TP_MODEL_ROOT=/absolute/path/to/Qwen3-4B
export HYPER_QWEN3_TP_DATA_ROOT=/absolute/path/to/gsm8k
export HYPER_QWEN3_TP_RESULT_ROOT=/absolute/path/to/results
```

将占位路径替换为宿主机实际目录：

| 目录 | 所需内容 |
| :--- | :--- |
| 模型 | 完整 Qwen3-4B checkpoint、`config.json` 与 tokenizer 文件 |
| 数据 | `train.parquet` 和 `test.parquet`；当前 recipe 使用 `prompt`、`extra_info` 列 |
| 结果 | 可写目录，启动脚本自动创建并保存运行日志 |

`prompt` 为问题文本或消息列表，`extra_info` 可为答案字符串或包含 `answer` 的对象。已有 GSM8K Parquet 的采样与转换可参考[数据准备脚本](examples/agents/gsm8k/prepare_gsm8k_m3.py)；读取规则见[数据加载器](rl/dataset/data_source.py)。

### 2. 运行一个训练步

从 `npu-smi info` 中选择四张空闲且 `Health=OK` 的设备，替换下方设备编号：

```bash
export HYPER_QWEN3_TP_VISIBLE_DEVICES=0,1,2,3
export HYPER_QWEN3_TP_TRAINER_TP=2
export HYPER_QWEN3_TP_ROLLOUT_TP=2
export HYPER_QWEN3_TP_MAX_STEPS=1

./hyper_parallel/rl/examples/scripts/run_qwen3_tp_docker.sh colocated
```

此入口使用 Hyper-vLLM，执行小批量、短生成的单步运行检查，关闭评估、检查点保存与 Bit-Exact；其结果不代表训练收敛或非零学习验收。

### 3. 检查运行结果

脚本应以退出码 0 结束。日志保存在 `${HYPER_QWEN3_TP_RESULT_ROOT}` 下，文件名包含 deployment、模型实现、Trainer / rollout TP 和权重策略。

| 检查项 | 预期结果 |
| :--- | :--- |
| `train/global_step` | 推进到 1 |
| `policy/version` | 推进到 1，表示新策略已完成发布 |
| `train/total_loss` | 有限数值，无 NaN / Inf；零损失本身不证明学习有效 |

运行完整训练时，使用[训练入口](examples/train_rl.py)与 [GSM8K recipe](examples/configs/qwen3_4b_gsm8k_vllm_production.yaml)，配置路径、设备、评估和恢复选项。单步检查脚本会覆盖部分 YAML 设置，不作为完整训练入口。

其他运行方式：

| 目标 | 入口 |
| :--- | :--- |
| Native-vLLM | 设置 `HYPER_QWEN3_TP_IMPLEMENTATION=native` 后运行同一脚本 |
| Disjoint 部署 | 配置互不重叠的 Trainer / rollout 设备，参数要求见 [vLLM Rollout](docs/vllm_rollout.md) |
| Bit-Exact 校验 | 使用独立的[一致性运行入口与门禁](docs/qwen3_training_inference_consistency.md) |
| MoE 模型 | 按 [MoE 模型](docs/moe_models.md)选择模型、TP / EP 拓扑与验证路径 |

## 扩展与定制

Hyper-RL 提供任务、数据与算法层的扩展入口。单轮任务与多轮 Agent 交互共用训练链路；涉及训练机制的定制，可基于 fork 修改相应模块。

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

## 规划方向

基础能力按同步训练、使用与效果验证、Ray 异步、模态与规模扩展推进。Agentic 独立演进，按需复用这些基础能力；程序化 Agent 接入不作为通用异步或多模态训练的前置条件。

以少量[代表模型与完整 recipe](docs/TODO.md#代表模型与能力证明)验证基础 RL、MoE、多模态和大模型异步能力；Agentic 通过多轮工具与状态化任务验证扩展能力。规划先完善样本分组、概率差异、奖励来源与恢复语义，再以达到同等质量的时间和资源成本验证系统收益。模型目标、阶段依赖与验收标准统一维护在 [TODO](docs/TODO.md)，设计取舍见[设计目标与原则](docs/design.md)。

## 当前边界

- 当前样本/token 消费计数字段尚未在训练步累加；恢复实现与待补验收分别见[状态所有权](docs/architecture.md#状态所有权)和 [TODO](docs/TODO.md#m1可运行可定制的同步基础版本)。
- 未列入支持范围的组合不具备端到端支持；MoE disjoint 与专家内部 TP 尚未支持。
- PPO / GAE / Critic 端到端、Trainer CP/PP、多节点、异步 / off-policy rollout、动态专家重分配、动态扩缩容与透明 generation retry 尚未提供端到端支持。
- Bit-Exact 验证范围不包含 backward、gradient、optimizer state、更新后参数或训练收敛，也不覆盖 Native-vLLM 或 TP4/TP8。
- vLLM RLHF/refit development endpoints 只能运行在受信任、隔离的训练网络。

## 文档

| 文档 | 内容 |
| --- | --- |
| [交付计划](docs/TODO.md) | 阶段任务、依赖关系与验收标准 |
| [设计目标与原则](docs/design.md) | 基础设施选型、模块边界与扩展取舍 |
| [Agentic RL](docs/agentic_rl.md) | 内部环境、Codex / DeepSeek Harness 的配置、轨迹与接入边界 |
| [Hyper-RL 架构](docs/architecture.md) | 组件、数据合同、训练生命周期和边界 |
| [vLLM Rollout](docs/vllm_rollout.md) | 资源归属、采样准入、权重事务与失败语义 |
| [Qwen3 训练-推理一致性](docs/qwen3_training_inference_consistency.md) | Bit-Exact 定义、recipe 和验收门禁 |
| [MoE 模型](docs/moe_models.md) | 模型、TP/EP 配置、组件归属与验证边界 |
| [运行镜像](docs/hyper_rl_runtime_image.md) | 镜像下载、校验、固定依赖和宿主要求 |
| [公共模块修改说明](docs/public_module_changes.md) | RL 目录外修改的必要性和接口影响 |
| [功能导航](../../docs/rl-navigation.md) | 配置 → 入口 → 分支 → 数据/指标 → 测试 |
| [Module Map](../../.agent/rules/rl/module-map.md) | 子系统归属、代码位置与对应合同文档 |
