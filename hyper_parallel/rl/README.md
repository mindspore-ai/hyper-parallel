# Hyper-RL

Hyper-Parallel 原生 LLM 强化学习运行时。当前主线是 Qwen3.5-0.8B、GSM8K、GRPO 与 colocated vLLM 的
单节点强同步训练闭环。

## 当前状态

| 能力 | 状态 |
|---|---|
| Qwen3.5-0.8B + GSM8K + GRPO | 双 Ascend NPU 端到端训练循环已验证 |
| Colocated rollout | FSDP DP2 + vLLM DP2/TP1 + NPU IPC 已验证 |
| 在线权重更新 | sleep、train、refit、wake 和 policy version 闭环已验证 |
| Correctness | Hyper/native 两步学习门禁与 20-step soak 已通过 |
| Production profile | 已实现；当前 batch-invariant canary 和完整 1,868-step 训练尚未运行 |
| Qwen3.5 精度对齐 | 本地 Hyper TP1/BF16/batch-one fresh-prefill 门禁已观察到 bit-exact；decode、TP2 和多请求尚未验收 |
| PPO/Critic | CPU 数学和能力契约已覆盖，尚未声明 Qwen NPU 端到端完成 |
| 多节点、colocated TP、异步训练 | 尚未实现 |

“端到端训练循环已验证”表示已经真实执行 rollout、reward、backward、optimizer step、vLLM refit 和版本推进；
不表示 checkpoint/resume、完整 GSM8K 训练或收敛已经完成。

## 运行环境

已验证环境：

- 2 张 Ascend 910B3 NPU；
- CANN 9.0.0；
- Docker 镜像 `hyper-parallel/unified-e2-dev:v0.22.1rc1`；
- Torch/torch-npu 2.10.0、Transformers 5.5.4；
- vLLM 0.22.1、vLLM-Ascend 0.22.1rc1；
- Qwen3.5-0.8B-Base；
- 包含 `prompt` 和 `extra_info` 列的 GSM8K parquet 数据。

通用安装环境和旧 Hyper 内置生成环境不是本次 vLLM 验收环境。Alignment patch 仅接受上述固定
vLLM-Ascend 源文件版本，不匹配时会在启动前失败。

## 基础运行

在仓库根目录设置统一 launcher 环境：

```bash
export HYPER_VLLM_IMAGE=hyper-parallel/unified-e2-dev:v0.22.1rc1
export HYPER_VLLM_MODEL_ROOT="$(pwd)/../models/Qwen3.5-0.8B-Base"
export HYPER_VLLM_DATA_ROOT="$(pwd)/../data/gsm8k"
export HYPER_VLLM_RESULT_ROOT="$(pwd)/.rollout-results/qwen35-grpo"
export ASCEND_RT_VISIBLE_DEVICES=0,1
export HCCL_IF_BASE_PORT=62000
export HCCL_NPU_SOCKET_PORT_RANGE=62000-62100
```

先运行最小 colocated 系统 smoke：

```bash
./hyper_parallel/rl/examples/scripts/run_qwen3_5_vllm_docker.sh \
  grpo-colocated-dp-smoke
```

再准备 M3 固定数据并运行两步学习门禁：

```bash
./hyper_parallel/rl/examples/scripts/run_qwen3_5_vllm_docker.sh grpo-m3-select
./hyper_parallel/rl/examples/scripts/run_qwen3_5_vllm_docker.sh grpo-m3-2step
```

其他可用 action 可通过不带参数运行该脚本查看。并行作业应使用不同的 `HYPER_VLLM_RESULT_ROOT`、
HCCL 端口范围和容器名。

## Qwen3.5 Alignment

Inference alignment 是默认关闭的实验性能力，用于让 Hyper Qwen3.5 与 canonical Hyper 模型在受控
fresh-prefill 边界内逐层对齐。启用后，launcher 会严格校验并应用仓库内的 vLLM-Ascend patch，选择：

- separate BF16 causal-conv/SiLU；
- Torch FP32 GDN gating；
- Transformers canonical Torch GDN recurrence；
- fusion-attention fresh prefill。

运行已验收的门禁：

```bash
export HYPER_VLLM_MODEL_IMPLEMENTATION=hyper
export HYPER_VLLM_ALIGNMENT=true

./hyper_parallel/rl/examples/scripts/run_qwen3_5_vllm_docker.sh \
  alignment-prefill-gate
```

`HYPER_VLLM_ALIGNMENT` 只接受 `true` 或 `false`，默认为 `false`。旧的
`HYPER_VLLM_NUMERICAL_PROFILE` 不再使用。Alignment 仅支持 Hyper Qwen3.5、BF16、非量化 checkpoint 和
eager execution，不支持 prefix caching、chunked prefill、speculative decoding、graph capture 或 standalone
`refit` action。Torch GDN recurrence 优先保证数值一致性，会明显降低吞吐。

本地 bit-exact 观察严格限定为 Ascend 910B3、TP1、batch one、单个完整且无历史 cache 的 fresh prefill；
24 层输出、final norm、BF16 full logits 和 FP32 full-vocabulary log-softmax 均 exact。该结果不能外推到
decode、cache reuse、TP2、多请求调度或完整 RL 训练。为其他 action 设置 alignment 只用于继续实验，不代表
这些路径已经通过 exact 验收。`VLLM_BATCH_INVARIANT` 是独立 inference 开关，门禁中保持关闭；Trainer
Linear/MM batch-invariant 尚未接入当前分支。

## 数据答案来源

配置中的 `data.answer_column` 一旦显式设置即为权威答案来源。未设置时，数据集依次尝试
`reward_model.ground_truth`，再自动推断 `extra_info`、`answer` 或 `solution` 列，避免自动 fallback 覆盖用户配置。

## 验收边界

| Action | Exit 0 证明 | 不证明 |
|---|---|---|
| `grpo-colocated-dp-smoke` | sleep/train/IPC/refit 系统闭环 | 非零梯度或收敛 |
| `grpo-m3-2step` | mixed reward、非零梯度、norm 探针变化和 version 1/2 | GSM8K 收敛 |
| `grpo-m3-soak` | 20-step 生命周期稳定 | 每步梯度都非零 |
| `alignment-prefill-gate` | TP1/BF16/batch-one fresh prefill 逐层和 full-logit bit-exact | decode、cache reuse、TP2 或多请求 exact |
| `production-benchmark` | standalone 并发数值与吞吐 A/B | colocated 完整训练可用 |
| `grpo-production-canary` | production-shaped 单步生命周期 | 数值对齐或完整训练 |
| `grpo-production` | 完整 profile 执行 | 当前尚无成功运行证据 |

## 代码结构

```text
hyper_parallel/rl/
├── rl/
│   ├── algorithm/       # GRPO/PPO recipes
│   ├── dataset/         # prompt data and batch construction
│   ├── roles/rollout/   # vLLM engine, model adapters, and weight refit
│   ├── agentic/         # environments and runners
│   └── trainer.py       # synchronous training lifecycle
├── examples/
├── tests/
└── docs/
```

所有 RL rollout 统一使用 `rollout.engine=vllm`。`rollout.vllm.model_implementation=hyper|native`
只选择 vLLM 进程内的模型实现，不代表两套 rollout backend。

## 文档

- [架构重构记录](docs/architecture_refactor_plan.md)
