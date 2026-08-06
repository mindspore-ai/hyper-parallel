# Qwen3.5 + GRPO 复现指南

本文复现 Hyper-Parallel 原生训练与生成路径。vLLM 当前完成注册和权重同步契约，不属于这次在线训练结果。

## 环境

已验证环境：

- 2 × Ascend NPU；
- CANN 9.0.0；
- Docker 镜像 `slime-small-test-preserved:20260720`；
- Qwen3.5-0.8B 权重；
- GSM8K parquet 数据。

建议目录：

```text
/home/whr/
├── hyper-parallel/
├── models/Qwen3.5-0.8B/
└── gsm8k/
    ├── train-00000-of-00001.parquet
    └── test-00000-of-00001.parquet
```

launcher 会把仓库父目录挂载到容器的 `/workspace`。设备、镜像和端口可通过 `NPU_DEVICE_0`、`NPU_DEVICE_1`、`HYPER_RL_IMAGE`、`MASTER_PORT` 和 `HCCL_NPU_SOCKET_PORT_RANGE` 覆盖。

## 单元测试

在包含 Hyper-Parallel、PyTorch、pandas、pyarrow 和 pytest 的环境执行：

```bash
cd /home/whr/hyper-parallel
HYPER_PARALLEL_PLATFORM=torch \
  PYTHONPATH=hyper_parallel/rl \
  python -m pytest -q hyper_parallel/rl/tests
```

当前结果为 `51 passed`，覆盖 GRPO/PPO 数学、Critic 按需创建、ExperienceBuilder、Hyper/vLLM registry、策略版本、Agentic trajectory、数据集、reward、monitoring 和 AsyncTrainer 的 fail-fast 边界。

## 快速 smoke

```bash
cd /home/whr/hyper-parallel
HYPER_RL_CONFIG=/workspace/hyper-parallel/hyper_parallel/rl/examples/configs/local_qwen3_5_0_8b_gsm8k_smoke.yaml \
  ./hyper_parallel/rl/examples/scripts/run_qwen3_5_0_8b_gsm8k_docker.sh
```

快速配置运行 1 step，每个 prompt 生成 2 个 response、每条最多 32 tokens。最终代码验证结果：

```text
rollout/generated_tokens=128
rollout/generation_seconds=16.1354
system/max_memory_allocated_gb=7.59611
train/optimizer_steps=1
```

随机 batch 可能全部同奖励，此时 group advantage、policy loss 和 gradient 为零是正确的 GRPO 行为。

## 有效更新 smoke

```bash
cd /home/whr/hyper-parallel
./hyper_parallel/rl/examples/scripts/run_qwen3_5_0_8b_gsm8k_docker.sh
```

默认配置每个 prompt 生成 6 个 response、每条最多 300 tokens。已验证运行得到：

```text
reward/mean=0.0833333
train/policy_loss=0.0222063
train/gradient_norm=4.91406
train/old_policy_kl=0.000462391
train/optimizer_steps=2
```

不同驱动、算子和采样实现可能改变具体数值。成功判据应是 loss 有限、存在有效 reward 差异、梯度非零、optimizer step 完成且进程正常退出。

## 非 Docker 运行

```bash
cd /home/whr/hyper-parallel
export TORCHRUN_BIN=/path/to/torchrun
export HYPER_RL_CONFIG=/home/whr/hyper-parallel/hyper_parallel/rl/examples/configs/qwen3_5_0_8b_gsm8k.yaml
export ASCEND_RT_VISIBLE_DEVICES=0,1
./hyper_parallel/rl/examples/scripts/run_qwen3_5_0_8b_gsm8k.sh
```

非 Docker 运行时，需要把 YAML 中的模型和数据路径改为宿主机实际路径。

## 常见问题

- world size 与 `dp_shard` 不一致：默认要求 `NPROC_PER_NODE=2` 且配置为 `dp_shard: 2`。
- 模型或数据不存在：检查容器内 `/workspace/models` 和 `/workspace/gsm8k`。
- 快速 smoke 为零梯度：运行默认 6×300 配置验证有效更新。
- vLLM 在线训练报 refitter 错误：当前没有绑定具体部署拓扑，不能绕过策略权重同步检查。
- HCCL 端口冲突：为并行作业指定独立的 master port 和 socket port range。
