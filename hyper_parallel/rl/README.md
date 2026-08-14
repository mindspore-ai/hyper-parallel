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
| 精度对齐 | standalone A/B 仍有 sampled-token log-prob 差异，后续单独处理 |
| PPO/Critic | CPU 数学和能力契约已覆盖，尚未声明 Qwen NPU 端到端完成 |
| 多节点、colocated TP、异步训练 | 尚未实现 |

“端到端训练循环已验证”表示已经真实执行 rollout、reward、backward、optimizer step、vLLM refit 和版本推进；
不表示 checkpoint/resume、完整 GSM8K 训练或收敛已经完成。

## 运行环境

已验证环境：

- 2 张 Ascend 910B3 NPU；
- CANN 9.0.0；
- Docker 镜像 `hyper-parallel/unified-e2-dev:v0.22.1rc1`；
- Torch/torch-npu 2.10.0；
- vLLM 0.22.1、vLLM-Ascend 0.22.1rc1；
- Qwen3.5-0.8B-Base；
- 包含 `prompt` 和 `extra_info` 列的 GSM8K parquet 数据。

通用安装文档和旧 Hyper 内置生成环境不是本次 vLLM 验收环境。固定版本、镜像 ID、模型/数据哈希和插件要求见
[Qwen3.5 vLLM 兼容与实验基线](docs/vllm_compatibility.md)。

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

详细命令、并行作业隔离、20-step soak 和 experimental production profile 见
[Qwen3.5 + GRPO 复现指南](docs/reproduce_grpo.md)。

## 验收边界

| Action | Exit 0 证明 | 不证明 |
|---|---|---|
| `grpo-colocated-dp-smoke` | sleep/train/IPC/refit 系统闭环 | 非零梯度或收敛 |
| `grpo-m3-2step` | mixed reward、非零梯度、norm 探针变化和 version 1/2 | GSM8K 收敛 |
| `grpo-m3-soak` | 20-step 生命周期稳定 | 每步梯度都非零 |
| `production-benchmark` | standalone 并发数值与吞吐 A/B | colocated 完整训练可用 |
| `grpo-production-canary` | production-shaped 单步生命周期 | 数值对齐或完整训练 |
| `grpo-production` | 完整 profile 执行 | 当前尚无成功运行证据 |

最新真实硬件结果、已知问题和未完成门禁见
[vLLM 在线 Rollout 故障记录与验收状态](docs/vllm_online_rollout_status.md)。

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

- [架构设计](docs/architecture.md)
- [代码结构](docs/code_structure.md)
- [Qwen3.5 + GRPO 复现指南](docs/reproduce_grpo.md)
- [Qwen3.5 vLLM 兼容与实验基线](docs/vllm_compatibility.md)
- [vLLM 在线 Rollout 故障记录与验收状态](docs/vllm_online_rollout_status.md)
