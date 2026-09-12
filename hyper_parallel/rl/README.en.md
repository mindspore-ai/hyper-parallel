<!-- markdownlint-disable MD033 -->
<h1 align="center">
  <img src="docs/assets/hyper-rl-logo.svg" width="120" height="80" alt="HyperParallel-RL logo"><br>
  HyperParallel-RL · Experimental
</h1>

<p align="center"><strong>A lightweight, extensible reinforcement learning framework.</strong></p>

<p align="center">Built on HyperParallel and vLLM, with explicit training orchestration and modular interfaces.</p>

<p align="center"><strong>Simple to use · Easy to extend · Ascend-friendly</strong></p>

<p align="center"><a href="README.md">中文</a> | <strong>English</strong></p>

<p align="center"><a href="#why-hyper-rl">Why HyperParallel-RL</a> · <a href="#architecture">Architecture</a> · <a href="#installation-and-environment">Installation</a> · <a href="#quick-start">Quick Start</a> · <a href="#extensions-and-customization">Customization</a> · <a href="#supported-capabilities">Capabilities</a> · <a href="#documentation">Documentation</a> · <a href="#citation">Citation</a>
</p>
<!-- markdownlint-enable MD033 -->

---

HyperParallel-RL is a reinforcement learning framework for LLMs, VLMs, world models, embodied intelligence, and agentic AI. Its design principles are **simple to use, easy to extend, and Ascend-friendly**, with **scalable, efficient training** as its architectural goal.

Users define tasks, interactions, tools, and rewards in Python. Basic policy optimization and multi-turn agent interactions share a training pipeline. HyperParallel handles parallel training, vLLM handles sampling, and HyperParallel-RL connects them through explicit orchestration, shared trajectories, and policy publication.

> **Experimental:** Current validation covers single-node Ascend NPUs, synchronous GRPO, Qwen3 dense, and selected MoE paths. End-to-end asynchronous, multimodal, world-model, and embodied-intelligence training is not yet available. See [supported capabilities](#supported-capabilities) for scope and validation status.

**Get started:** [Install](#installation-and-environment) → [Run one training step](#quick-start) · **Customize:** [Replace the reward function](#replace-the-reward-function)

---

<!-- markdownlint-disable-next-line MD033 -->
<a id="why-hyper-rl"></a>

## 🌟 Why HyperParallel-RL

### 🪶 Simple to use

Fewer deployment dependencies and a training flow that is easy to follow and run.

- **A minimal stack:** HyperParallel handles training and parallelism; vLLM handles sampling. Synchronous training uses direct orchestration without Ray.
- **An explicit training flow:** SyncTrainer organizes sampling, learning, weight publication, evaluation, and recovery, making execution easier to trace and debug.
- **A clear starting point:** A [pinned runtime image](docs/hyper_rl_runtime_image.md), model configurations, and launch scripts support a Qwen3-4B [single-step check](#quick-start) of the training pipeline.
- **Human-readable, agent-traceable:** Developers use the [feature navigation](../../docs/rl-navigation.md) to locate configurations, implementations, and tests. Coding agents enter through [AGENTS.md](../../AGENTS.md) and the [HyperParallel-RL rules](../../.agent/rules/hyper-rl.md), reading the same authoritative documentation as needed.

### 🧩 Easy to extend

Clear extension boundaries with a reusable training pipeline.

- **Algorithms decoupled from orchestration:** Advantage estimators and policy losses support registered extensions. Algorithms compatible with existing training roles reuse the Trainer; sampling and learning execution mechanisms can be customized in core orchestration.
- **Tasks, tools, and rewards in Python:** Replace scoring logic or add tool interactions without modifying distributed training or weight publication. See the [reward example](#replace-the-reward-function).
- **Two agentic interaction styles:** Environment supports framework-driven turns; AgentProgram hosts user programs. Built-in Codex / DeepSeek harnesses are integrated; custom programs still require adaptation. See [Agentic RL](docs/agentic_rl.md).
- **Model and backend adaptation at defined boundaries:** New models reuse HyperParallel capabilities with training/inference adapters and weight mapping. Other inference backends can be extended and independently validated in downstream forks. See [extension interfaces](docs/architecture.md#扩展点) and [infrastructure choices](docs/design.md#基础设施选型).

### ⚙️ Ascend-friendly

Organize training and sampling around Ascend computation, memory, and communication capabilities.

- **FSDP training:** Reuse HyperParallel state sharding and optimization mechanisms such as prefetching and communication–computation overlap, with validated FSDP, TP, and EP combinations for dense and selected MoE paths. See [FSDP optimization](../../docs/guide/fsdp.md#fsdp-性能优化).
- **Colocated and disjoint deployment:** Qwen3 dense supports shared or separate training/inference devices, publishing weights through NPU IPC or HCCL respectively. Colocated deployment releases and restores inference resources by phase. MoE currently requires colocation; see [supported capabilities](#supported-capabilities).
- **Streaming weight synchronization:** Bucketed transfer and buffer release after acknowledgment bound temporary memory use. Explicit direct-reshard transfers parameter intersections between training and inference shards; full-gather remains available. See [vLLM Rollout](docs/vllm_rollout.md).

HyperParallel also provides [one-sided communication and multicore MoE communication–computation overlap](../../docs/guide/multicore_moe.md) as a foundation for further optimization. HyperParallel-RL integration and end-to-end benefits remain to be validated.

### 🔗 Training–inference consistency

Preserve generation evidence, check probability differences, and define policy publication boundaries.

- **Original samples throughout training:** Preserve generated tokens, logprobs, action masks, and policy identity, avoiding sample changes from re-tokenization. Tool feedback and environment observations provide context and do not contribute to policy loss. See the [trajectory contract](rl/dataset/contracts.py).
- **Shared semantics and Bit-Exact checks:** For Qwen3 dense, the Trainer and Hyper-vLLM share parameter semantics and TP sharding plans. Optional pre-update checks compare FP32 raw logprobs bit by bit on valid action tokens; failure blocks the update and publication.
- **Sampling uses published policies:** The synchronous flow consumes trajectories from one published version. Sampling resumes only after weight transfer, worker identity checks, and cache reset complete; parameter publication and probability consistency are validated separately.

Bit-Exact is disabled by default and validated for single-node Ascend, BF16/eager, Qwen3 dense + Hyper-vLLM with matched TP1/TP2. It does not guarantee gradients or convergence. See [training–inference consistency](docs/qwen3_training_inference_consistency.md) for full conditions and results.

---

<!-- markdownlint-disable-next-line MD033 -->
<a id="architecture"></a>

## 🏗️ Architecture

![HyperParallel-RL synchronous architecture: task extensions, vLLM sampling, shared trajectories, HyperParallel training, and policy publication](docs/assets/hyper-rl-architecture.svg)

User extensions define tasks and learning objectives; the HyperParallel-RL core organizes the training loop; infrastructure handles inference and parallel training. Solid arrows show the sampling-to-publication loop, and dashed lines mark extensions. Diagram labels are currently in Chinese.

See [Architecture](docs/architecture.md) for component and state boundaries, and [feature navigation](../../docs/rl-navigation.md) for configurations, implementations, and tests.

---

<!-- markdownlint-disable-next-line MD033 -->
<a id="installation-and-environment"></a>

## 📦 Installation and environment

Use **repository source code with the pinned runtime image**. The current runtime requires Linux ARM64 and Ascend NPUs; the basic example below uses four available NPUs and the agentic example uses two. The host needs Docker, a compatible NPU driver, and storage for models and data. See the [runtime image](docs/hyper_rl_runtime_image.md#宿主要求) for storage and driver requirements.

### 1. Get the source

```bash
git clone --branch rl https://gitcode.com/mindspore/hyper-parallel.git
cd hyper-parallel
```

Run all subsequent commands from the repository root. If you already have a checkout, use a revision compatible with the runtime image.

### 2. Prepare the runtime image

```bash
docker pull swr.cn-east-3.myhuaweicloud.com/huawei-hyper-rl/hyper-rl:v0.22.1rc1-arm64

npu-smi info
```

The image includes CANN, Torch / torch-npu, Transformers, and vLLM / vLLM-Ascend. The launcher mounts source code, drivers, models, data, and output directories, and sets import paths; no additional `pip install` is required.

During development, edit the local source directly; subsequent runs mount the updated checkout. See the [runtime image documentation](docs/hyper_rl_runtime_image.md) for the image digest, dependency checks, and source build instructions.

---

<!-- markdownlint-disable-next-line MD033 -->
<a id="quick-start"></a>

## 🚀 Quick start

After [environment setup](#installation-and-environment), run either example from the repository root. Replace host paths, use a complete Qwen3-4B checkpoint and tokenizer, and select idle, healthy devices with `npu-smi info`. Run sequentially when sharing devices.

<!-- markdownlint-disable-next-line MD033 -->
<a id="qwen-example"></a>

### Example 1: Qwen basic training

**Qwen3-4B + GSM8K, one synchronous GRPO step on four NPUs.** The data directory needs `train.parquet` and `test.parquet` with `prompt` and `extra_info` columns. See [data preparation](examples/agents/gsm8k/prepare_gsm8k_m3.py) and [parsing rules](rl/dataset/data_source.py).

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

**Success:** Exit code 0, `train/global_step=1`, `policy/version=1`, and finite `train/total_loss` in the results directory logs. Evaluation, checkpoint saving, and Bit-Exact are disabled for this check.

<!-- markdownlint-disable-next-line MD033 -->
<a id="agentic-example"></a>

### Example 2: Agentic multi-turn interaction

**Qwen3-4B + Search-R1, two retrieval-QA steps on two NPUs.** The model uses a local retrieval tool and continues answering from observations; no external agent CLI is required. Prepare `train.parquet` with `prompt` and `answer` columns and `corpus.jsonl` using the [data preparation script](examples/agents/search_R1/prepare_search_r1_data.py).

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

**Success:** Exit code 0, `train/global_step=2` and `policy/version=2` in the results directory’s `train.log`, and a `checkpoints/step_2/checkpoint_complete.json` file. Inspect logged samples for actual tool interaction.

Both examples validate execution, not learning gains. For full training, use the [training entry point](examples/train_rl.py) with the [GSM8K](examples/configs/qwen3_4b_gsm8k_vllm_production.yaml) or [Search-R1](examples/agents/search_R1/configs/multi_turn.yaml) configuration, adjusting the training budget, evaluation, and saving instead of using the launchers’ fixed-step checks.

Other entry points: [Programmatic agents](docs/agentic_rl.md) · [Deployment and sampling](docs/vllm_rollout.md) · [Bit-Exact checks](docs/qwen3_training_inference_consistency.md) · [MoE models](docs/moe_models.md).

---

<!-- markdownlint-disable-next-line MD033 -->
<a id="extensions-and-customization"></a>

## 🧩 Extensions and customization

HyperParallel-RL exposes task, data, and algorithm extension points. Single-turn tasks and multi-turn agent interactions share the training pipeline. Training mechanisms can be customized by modifying the relevant modules in a fork.

| Extension | Implementation entry point |
| --- | --- |
| Rewards and task environments | The [GSM8K example](examples/agents/gsm8k/agent.py) defines environments and rewards; select a task through `agentic.module_path` and `agentic.environment` |
| Multi-turn interaction or custom loops | [Environment](rl/agentic/envs/base.py) provides framework-driven interaction; [AgentProgram](rl/agentic/core/program_runner.py) hosts programmatic loops. See [Agentic RL](docs/agentic_rl.md) for built-in paths and harness integration boundaries. Environment observations are excluded from policy loss |
| Data and training samples | The [data source](rl/dataset/data_source.py) handles inputs; [trajectory and batch contracts](rl/dataset/contracts.py) define data entering learning |
| Algorithms and updates | Locate learning logic in [loss](rl/algorithm/loss.py), [advantage](rl/algorithm/advantage.py), and [SyncTrainer](rl/trainer.py); use [feature navigation](../../docs/rl-navigation.md) to find related tests |

### Replace the reward function

This example reuses the GSM8K multi-turn environment and calculator, replacing only terminal scoring. Save it as `hyper_parallel/rl/examples/agents/custom_reward.py`:

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

In the full training YAML, use the `agentic` settings from the [GSM8K multi-turn configuration](examples/agents/gsm8k/configs/multi_turn.yaml), change `module_path` to `examples.agents.custom_reward`, and set `environment` to `custom_gsm8k`. This replaces only terminal scoring for the multi-turn task; it does not modify the Trainer or weight publication.

---

<!-- markdownlint-disable-next-line MD033 -->
<a id="supported-capabilities"></a>

## 📋 Supported capabilities

Currently validated on **single-node Ascend 910B3**; see the [runtime image](docs/hyper_rl_runtime_image.md) for environment requirements. ✅ Supported · ◐ Partially validated · 🧪 Components only · ○ Planned. Status applies only to the listed combinations and does not establish long-term learning gains.

### Training and tasks

| Capability | Status | Scope |
| :--- | :---: | :--- |
| **Synchronous GRPO** | ✅ | Sampling, learning, publication, evaluation, and recovery; extensible advantage/loss implementations, no Ray required |
| **Single-turn and multi-turn tool tasks** | ✅ | Python environments, tools, and rewards share token-first trajectories; environment observations are excluded from loss |
| **Programmatic agents** | ◐ | Codex / DeepSeek harnesses integrated; custom programs require adaptation. See [Agentic RL](docs/agentic_rl.md) |
| **Sampling and deployment** | ✅ | Hyper/Native-vLLM; Qwen3 dense supports colocated and disjoint deployment, MoE is colocated only. See [vLLM Rollout](docs/vllm_rollout.md) for DP/TP/EP combinations |
| **Weight synchronization and runtime support** | ✅ | Streaming full-gather / explicit direct-reshard, IPC/HCCL, publication checks, checkpoint recovery, and console / W&B metrics. See [architecture contracts](docs/architecture.md) |
| **Bit-Exact checks** | ✅ | Optional pre-update logprob checks, limited to Qwen3 dense + Hyper-vLLM matched TP1/TP2. See [full conditions](docs/qwen3_training_inference_consistency.md) |
| **PPO / GAE / Critic** | 🧪 | Mathematical and role components exist; no end-to-end training path yet |
| **Asynchronous and multi-node training** | ○ | Ray-based sampling/learning overlap, policy lag, and recovery; extensions to multiple nodes and long-running agents |
| **Multimodal training and interaction** | ○ | Visual RL, media/action alignment, and multimodal agents |

### Representative models

| Model | Status | Validation scope or target |
| :--- | :---: | :--- |
| **Qwen3 dense** | ✅ | Single-node GRPO; Hyper/Native-vLLM TP1/TP2, with Trainer TP1, pure TP2, or FSDP-shard×TP2 |
| **Qwen3-30B-A3B** | ✅ | Native/Hyper two-step loops and controlled nonzero updates; four-device TP2/EP4 weight publication validated |
| **Moonlight-16B-A3B-Instruct** | ◐ | Hyper passes two-step nonzero learning; Native publication passes, but sustained nonzero learning has not passed acceptance |
| **Qwen2.5-VL-7B-Instruct** | ○ | Image-based math questions and multimodal agents |
| **Full DeepSeek-V3 model** | ○ | Multi-node synchronous baseline and asynchronous comparison using the full checkpoint |

See [MoE Models](docs/moe_models.md) for detailed configurations and acceptance results, [TODO](docs/TODO.md) for planned dependencies and quantitative criteria, and [current limitations](#current-limitations) for usage boundaries.

---

<!-- markdownlint-disable-next-line MD033 -->
<a id="planned-directions"></a>

## 🗺️ Planned directions

- **Asynchronous training at scale:** Use Ray to overlap sampling and learning, validate single-node asynchronous execution first, then multi-node training with the full DeepSeek-V3 model.
- **Multimodal training:** Establish visual RL with Qwen2.5-VL-7B-Instruct on image-based math questions, then validate multimodal agent interactions.
- **Agent learning and asynchronous execution:** Refine existing multi-turn environments and programmatic integration, validate learning on tool and stateful code tasks, and extend asynchronous execution to long-running tasks.

Each direction requires complete recipes, learning results, and resource-cost comparisons. Models, dependencies, and quantitative criteria are maintained in [TODO](docs/TODO.md). World models and embodied intelligence remain long-term directions with concrete delivery scope yet to be defined.

---

<!-- markdownlint-disable-next-line MD033 -->
<a id="current-limitations"></a>

## 📌 Current limitations

- **Runtime scope:** Current execution is single-node Ascend synchronous GRPO, limited to the models and topologies in [supported capabilities](#supported-capabilities). MoE does not yet support disjoint deployment or intra-expert TP. PPO / Critic components do not yet provide end-to-end training.
- **Validation scope:** Functional success does not establish long-term learning gains. Bit-Exact covers pre-update logprobs only in specified Qwen3 dense + Hyper-vLLM configurations; it does not guarantee gradients, updated parameters, or convergence. See [consistency documentation](docs/qwen3_training_inference_consistency.md).
- **Recovery and counters:** Checkpoint recovery is implemented, but sample/token consumption counters are not yet incremented during training steps and must not be used to measure actual training volume. See [state and recovery boundaries](docs/architecture.md#状态所有权).
- **Deployment requirements:** vLLM RLHF/refit development endpoints are intended only for trusted, isolated training networks. See the [runtime image](docs/hyper_rl_runtime_image.md) for environment and driver requirements.

---

<!-- markdownlint-disable-next-line MD033 -->
<a id="documentation"></a>

## 📚 Documentation

Linked product documentation is currently primarily in Chinese. Both README editions describe the same capabilities and validation boundaries.

### Running and validation

| Document | Contents |
| --- | --- |
| [Runtime Image](docs/hyper_rl_runtime_image.md) | Image download, verification, pinned dependencies, and host requirements |
| [Agentic RL](docs/agentic_rl.md) | Internal environments, Codex / DeepSeek harness configuration, trajectories, and integration boundaries |
| [MoE Models](docs/moe_models.md) | Models, TP/EP configurations, component ownership, and validation boundaries |
| [Qwen3 Training–Inference Consistency](docs/qwen3_training_inference_consistency.md) | Bit-Exact definitions, recipes, and acceptance gates |

### Architecture and evolution

| Document | Contents |
| --- | --- |
| [Design Goals and Principles](docs/design.md) | Infrastructure choices, module boundaries, and extension tradeoffs |
| [HyperParallel-RL Architecture](docs/architecture.md) | Components, data contracts, training lifecycle, and boundaries |
| [vLLM Rollout](docs/vllm_rollout.md) | Resource ownership, sampling admission, weight transactions, and failure semantics |
| [Delivery Plan](docs/TODO.md) | Milestones, dependencies, and acceptance criteria |

### Development and code navigation

| Document | Contents |
| --- | --- |
| [Feature Navigation](../../docs/rl-navigation.md) | Configuration → entry point → branch → data/metrics → tests |
| [Module Map](../../.agent/rules/rl/module-map.md) | Subsystem ownership, code locations, and authoritative contracts |
| [Public Module Changes](docs/public_module_changes.md) | Rationale and interface impact of changes outside the RL directory |

---

<!-- markdownlint-disable-next-line MD033 -->
<a id="citation"></a>

## 📝 Citation

If you use HyperParallel-RL, you can cite the software below and record the commit used in your experiments.

```bibtex
@software{hyper_rl_2026,
  title = {HyperParallel-RL: A Lightweight, Extensible Reinforcement Learning Framework},
  year  = {2026},
  url   = {https://gitcode.com/mindspore/hyper-parallel},
  note  = {HyperParallel-RL module in the HyperParallel repository}
}
```
