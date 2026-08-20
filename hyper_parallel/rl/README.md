# Hyper-RL

Hyper-RL 是 HyperParallel 原生的同步大模型强化学习运行时。Trainer 使用 Transformers、HyperModels 和
HyperParallel FSDP，rollout 使用多个 colocated vLLM TP1 replicas；当前端到端实现和验证的算法为 GRPO。

## 已实现功能

| 功能                           | 当前状态                                                                                             |
| ------------------------------ | ---------------------------------------------------------------------------------------------------- |
| Qwen3-4B、Qwen3.5-0.8B Trainer | Transformers 模型、HyperModels/HyperParallel FSDP2、BF16、optimizer 和 checkpoint                    |
| Native-vLLM rollout            | Qwen3、Qwen3.5 generation、raw logprobs、NPU IPC refit 和 cache reset                                |
| Hyper-vLLM rollout             | 基于 Transformers 模型定义的 Qwen3/Qwen3.5 adapter，保留 vLLM scheduler、sampling 和 KV cache        |
| 完整 GRPO 闭环                 | rollout、reward、advantage、reference、backward、optimizer、refit、cache reset 和 policy publication |
| 在线权重同步                   | colocated NPU IPC，worker-owned version/fingerprint，失败时不发布新版本                              |
| 训推一致性                     | Qwen3 Hyper-vLLM optimizer 前逐 token FP32 bit-exact 门禁                                            |
| 评估与监控                     | GSM8K evaluation、console/W&B、learning gate 和性能指标                                              |
| Checkpoint                     | Actor DCP、rank-local optimizer/RNG/dataloader state、完成 manifest 和 reload 验证                   |
| PPO/Critic                     | 数学和接口具备 CPU 测试；尚未声明 Qwen NPU 端到端支持                                                |

Qwen3/Qwen3.5 与 Native/Hyper-vLLM 四种 TP1 组合均已通过两步 GRPO learning smoke，覆盖 mixed reward、
非零梯度、Actor fingerprint change、NPU IPC refit、cache reset 和连续 policy version `1/2`。

## 运行架构

```text
FSDP Trainer
  → vLLM rollout replicas 生成 token 和 raw logprobs
  → reward、reference logprobs 和 GRPO advantages
  → Actor backward 与 optimizer update
  → pause rollout，NPU IPC refit 到全部 replicas
  → worker identity 校验与 hard cache reset
  → 发布下一 policy version
  → 下一训练步
```

详细的角色职责、数据合同和生命周期见 [Hyper-RL 架构](docs/architecture.md)。

## 已验证环境

| 组件                | 版本或范围                                              |
| ------------------- | ------------------------------------------------------- |
| 硬件                | 单节点 2 或 8 张 Ascend 910B3 NPU                       |
| CANN                | 9.0.0                                                   |
| Docker image        | `hyper-parallel/unified-e2-dev:v0.22.1rc1`            |
| Torch / torch-npu   | 2.10.0                                                  |
| Transformers        | 5.5.4                                                   |
| vLLM                | 0.22.1                                                  |
| vLLM-Ascend         | 0.22.1rc1                                               |
| Batch-invariant ops | 1.0.0                                                   |
| Flash Attention NPU | 0.2.0b1，Qwen3 consistency smoke 需要单独安装固定 wheel |
| 模型                | Qwen3-4B、Qwen3.5-0.8B-Base                             |
| 数据                | GSM8K parquet，包含`prompt` 和 `extra_info` 列      |

宿主机需要可用的 Ascend driver、Docker 和至少两张空闲 NPU。以下命令均在仓库根目录执行，并将结果写入已被 Git
忽略的 `.rollout-results/`。

## Smoke：完整 RL 训练闭环

以下 Qwen3.5 两步 smoke 会自动筛选固定 mixed-reward GSM8K 子集，然后真实执行两步 rollout、reward、backward、
optimizer、NPU IPC refit、cache reset 和 policy V1/V2 发布。

```bash
export HYPER_VLLM_IMAGE=hyper-parallel/unified-e2-dev:v0.22.1rc1
export HYPER_VLLM_MODEL_ROOT=/absolute/path/to/Qwen3.5-0.8B-Base
export HYPER_VLLM_DATA_ROOT=/absolute/path/to/gsm8k
export HYPER_VLLM_RESULT_ROOT="$(pwd)/.rollout-results/readme-qwen35-m3"
export HYPER_VLLM_VISIBLE_DEVICES=0,1
export HYPER_VLLM_MODEL_IMPLEMENTATION=native
export HYPER_VLLM_TIMEOUT_SECONDS=3600
export HCCL_IF_BASE_PORT=62000
export HCCL_NPU_SOCKET_PORT_RANGE=62000-62100

./hyper_parallel/rl/examples/scripts/run_qwen3_5_vllm_docker.sh grpo-m3-2step
```

将 `HYPER_VLLM_MODEL_IMPLEMENTATION` 改为 `hyper` 可验证 Hyper-vLLM adapter。成功运行必须 exit 0，并在
`${HYPER_VLLM_RESULT_ROOT}/${HYPER_VLLM_MODEL_IMPLEMENTATION}-grpo-m3-2step.log` 中看到：

```text
train/optimizer_steps > 0
train/gradient_norm > 0
reward/min < reward/max
policy/fingerprint_changed = 1
policy/version = 1, 2
```

## Smoke：Qwen3 训推一致性

该 smoke 使用两张 NPU、两个 colocated Hyper-vLLM TP1 replicas 和真实 GRPO update。在第一次 optimizer update 前，
Trainer 会对 rollout 生成的原始 token 重新计算 FP32 raw logprobs，并逐 token 比较 bit pattern。

准备从 `flash-attention-npu` commit `c7528a4bdb0a33f21e181cd25108c1e60c11d061` 构建的 wheel：

```text
flash_attn_npu-0.2.0b1-cp312-cp312-linux_aarch64.whl
SHA256: 9f58e114b77f72079111e2f86fa9750d3be39d1ec9324b309588a540a3e9e12b
```

然后执行：

```bash
export HYPER_QWEN3_IMAGE=hyper-parallel/unified-e2-dev:v0.22.1rc1
export HYPER_QWEN3_MODEL_ROOT=/absolute/path/to/Qwen3-4B
export HYPER_QWEN3_DATA_ROOT=/absolute/path/to/gsm8k
export HYPER_QWEN3_RESULT_ROOT="$(pwd)/.rollout-results/readme-qwen3-consistency"
export HYPER_QWEN3_VISIBLE_DEVICES=0,1
export HYPER_QWEN3_FA3_WHEEL=/absolute/path/to/flash_attn_npu-0.2.0b1-cp312-cp312-linux_aarch64.whl
export HCCL_IF_BASE_PORT=62200
export HCCL_NPU_SOCKET_PORT_RANGE=62200-62300

./hyper_parallel/rl/examples/scripts/run_qwen3_consistency_docker.sh
```

脚本会校验 wheel SHA256、准备固定 mixed-reward 数据，并执行一步真实训练。成功运行必须 exit 0，且
`${HYPER_QWEN3_RESULT_ROOT}/qwen3-consistency-smoke.log` 包含：

```text
training/pre_update_exact_valid = 1
training/pre_update_exact_tokens > 0
training/pre_update_mismatch_count = 0
training/pre_update_max_abs_diff = 0
training/pre_update_mean_abs_diff = 0
train/optimizer_steps > 0
train/gradient_norm > 0
policy/fingerprint_changed = 1
policy/version = 1
```

Profile 会严格校验 Qwen3、Torch NPU、Hyper-vLLM TP1 和固定依赖版本，缺少 FA3 或 batch-invariant kernel 时直接
失败，不会静默回退。完整合同、8-NPU 结果和性能分析见
[Qwen3 训练-推理一致性](docs/qwen3_training_inference_consistency.md)。

## 主要配置

| 配置                                                         | 用途                                     |
| ------------------------------------------------------------ | ---------------------------------------- |
| `examples/configs/local_qwen3_5_0_8b_gsm8k_vllm_m3.yaml`   | Qwen3.5 两步 learning smoke              |
| `examples/configs/local_qwen3_4b_gsm8k_vllm_m3.yaml`       | Qwen3 两步 learning 与 consistency smoke |
| `examples/configs/qwen3_5_0_8b_gsm8k_vllm_production.yaml` | Qwen3.5 长运行 recipe                    |
| `examples/configs/qwen3_4b_gsm8k_vllm_production.yaml`     | Qwen3 8-NPU consistency recipe           |

`rollout.engine` 当前统一为 `vllm`。`rollout.vllm.model_implementation=native|hyper` 只切换 vLLM 进程中的模型
实现，不代表两套 rollout backend。通用配置的 `consistency.profile` 默认为 `off`；Qwen3 production recipe 显式启用
已验证 profile。

## 当前边界

- 当前 Trainer 只验证纯 FSDP；Hyper-vLLM 固定 TP1。
- 不支持 Trainer TP、TP2 Hyper-vLLM、多节点 colocated 或异步 rollout。
- Native-vLLM 与 Hyper-vLLM 不承诺生成相同 token 或 logprobs。
- 当前 worker fingerprint 是 norm canary，不是 full-policy digest。
- Qwen3 bit-exact 结论不覆盖 graph、长时间 soak、完整故障注入、收敛或其他模型。
- vLLM RLHF/refit development endpoints 使用不安全序列化，只能部署在受信任、隔离的训练网络。

## 文档

- [Hyper-RL 架构](docs/architecture.md)
- [Qwen3 训练-推理一致性](docs/qwen3_training_inference_consistency.md)
