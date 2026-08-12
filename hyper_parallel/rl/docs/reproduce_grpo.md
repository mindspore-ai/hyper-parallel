# Qwen3.5 + GSM8K + GRPO 复现指南

本文描述本仓库统一 vLLM launcher 的环境配置和基础命令。当前已经打通双卡
Qwen3.5-0.8B + GSM8K + GRPO 训练循环；完整 1,868-step production 训练尚未执行。

## 环境要求

已验证软件栈：

| 组件 | 版本 |
|---|---|
| 硬件 | 2 × Ascend 910B3 |
| CANN | 9.0.0 |
| Torch / torch-npu | 2.10.0 / 2.10.0 |
| Transformers | 5.5.4 |
| vLLM / vLLM-Ascend | 0.22.1 / 0.22.1rc1 |
| Docker image | `hyper-parallel/unified-e2-dev:v0.22.1rc1` |
| Model | Qwen3.5-0.8B-Base |

Docker 必须支持 `--privileged`，宿主机需要提供 launcher 中挂载的 Ascend driver、`dcmi` 和 `npu-smi`。
通用 `pip install` 环境和旧 `slime-small-test-preserved` 镜像不属于当前 vLLM 验收环境。

建议目录：

```text
<workspace>/
├── hyper-rl/
├── models/Qwen3.5-0.8B-Base/
└── data/gsm8k/
    ├── train.parquet
    └── test.parquet
```

统一 launcher 分别将仓库、模型、数据和结果目录挂载为：

```text
<repo-root>                         -> /workspace/hyper-parallel
HYPER_VLLM_MODEL_ROOT              -> /models/Qwen3.5-0.8B-Base
HYPER_VLLM_DATA_ROOT               -> /data/gsm8k
HYPER_VLLM_RESULT_ROOT             -> /results
```

GSM8K 文件必须直接包含 `prompt` 和 `extra_info` 列。固定 artifact 哈希见
[Qwen3.5 vLLM 兼容与实验基线](vllm_compatibility.md)。

## Launcher 配置

在 `hyper-rl` 仓库根目录执行：

```bash
export HYPER_VLLM_IMAGE=hyper-parallel/unified-e2-dev:v0.22.1rc1
export HYPER_VLLM_MODEL_ROOT="$(pwd)/../models/Qwen3.5-0.8B-Base"
export HYPER_VLLM_DATA_ROOT="$(pwd)/../data/gsm8k"
export HYPER_VLLM_RESULT_ROOT="$(pwd)/.rollout-results/qwen35-grpo"
export HYPER_VLLM_MODEL_IMPLEMENTATION=hyper
export ASCEND_RT_VISIBLE_DEVICES=0,1
export HCCL_IF_BASE_PORT=62000
export HCCL_NPU_SOCKET_PORT_RANGE=62000-62100
```

| 变量 | 默认值或用途 |
|---|---|
| `HYPER_VLLM_IMAGE` | 固定 vLLM 开发镜像 |
| `HYPER_VLLM_MODEL_ROOT` | 宿主机模型目录 |
| `HYPER_VLLM_DATA_ROOT` | 宿主机 GSM8K 目录 |
| `HYPER_VLLM_RESULT_ROOT` | 日志、报告、M3 数据和 checkpoint 根目录 |
| `HYPER_VLLM_MODEL_IMPLEMENTATION` | `hyper` 或 `native` |
| `HYPER_VLLM_TIMEOUT_SECONDS` | 容器内部主动超时 |
| `HYPER_VLLM_DETACHED` | `true` 时后台运行并返回容器 ID |
| `HYPER_VLLM_CONTAINER_NAME` | 后台运行的唯一容器名 |
| `HYPER_VLLM_NETWORK_MODE` | Docker 网络模式，默认 `bridge` |
| `HYPER_VLLM_LOG_SUFFIX` | 并行实验日志后缀，只允许字母、数字、点、下划线和连字符 |
| `ASCEND_RT_VISIBLE_DEVICES` | 当前作业使用的物理 NPU |
| `HCCL_IF_BASE_PORT` | 当前作业 HCCL 基础端口 |
| `HCCL_NPU_SOCKET_PORT_RANGE` | 当前作业 HCCL socket 范围 |

`MASTER_PORT`、`NPU_DEVICE_0/1`、`HYPER_RL_IMAGE` 和 `HYPER_RL_CONFIG` 属于旧 Hyper 内置生成 launcher，
不会配置统一 vLLM launcher。

单个隔离作业可以使用 HCCL 默认端口；并行作业必须同时使用不同 NPU、不同 `HCCL_IF_BASE_PORT`、不重叠的
`HCCL_NPU_SOCKET_PORT_RANGE` 和不同结果目录。Docker bridge 和 NPU 可见性不能隔离物理 HCCL 监听端口。

## 基础流程

### 1. Adapter 与 refit 检查

```bash
ASCEND_RT_VISIBLE_DEVICES=0 \
  ./hyper_parallel/rl/examples/scripts/run_qwen3_5_vllm_docker.sh rollout-tp1

ASCEND_RT_VISIBLE_DEVICES=0 \
  ./hyper_parallel/rl/examples/scripts/run_qwen3_5_vllm_docker.sh refit
```

`rollout-tp1` 验证模型加载和生成，`refit` 验证权重变化、worker replicated norm 探针和 policy version 1。

### 2. Colocated 系统 smoke

```bash
./hyper_parallel/rl/examples/scripts/run_qwen3_5_vllm_docker.sh \
  grpo-colocated-dp-smoke
```

该 action 验证 FSDP DP2、vLLM DP2/TP1、sleep、训练、NPU IPC refit、wake 和清理。随机 reward 全同导致零梯度
是允许结果，因此它不作为学习信号门禁。

### 3. M3 两步学习门禁

先在同一个结果目录生成固定 mixed-reward 数据，再运行两步训练：

```bash
./hyper_parallel/rl/examples/scripts/run_qwen3_5_vllm_docker.sh grpo-m3-select
./hyper_parallel/rl/examples/scripts/run_qwen3_5_vllm_docker.sh grpo-m3-2step
```

`grpo-m3-2step` 要求 mixed reward、非零梯度、Actor/worker norm 探针变化和严格 policy version 1/2。该门禁已经在
Hyper 和 native 上通过，但不证明 GSM8K 收敛。

切换实现时必须使用新的结果目录，并重新执行 selector，不能复用另一实现生成的 M3 数据：

```bash
export HYPER_VLLM_MODEL_IMPLEMENTATION=native
export HYPER_VLLM_RESULT_ROOT="$(pwd)/.rollout-results/qwen35-grpo-native"
./hyper_parallel/rl/examples/scripts/run_qwen3_5_vllm_docker.sh grpo-m3-select
./hyper_parallel/rl/examples/scripts/run_qwen3_5_vllm_docker.sh grpo-m3-2step
```

### 4. 20-step 生命周期 soak

```bash
./hyper_parallel/rl/examples/scripts/run_qwen3_5_vllm_docker.sh grpo-m3-soak
```

soak 关闭每步学习信号要求，但保留真实 rollout、backward、optimizer、refit、norm 探针、版本和 sleep/wake，用于验证
生命周期稳定性。Hyper/native 均已有 20-step 成功证据。

### 5. Standalone production A/B

```bash
ASCEND_RT_VISIBLE_DEVICES=0 \
  ./hyper_parallel/rl/examples/scripts/run_qwen3_5_vllm_docker.sh production-benchmark
```

该 action 比较同一热 server 的 concurrency 1 与 12，并启用 vLLM-Ascend batch-invariant。当前 Hyper/native 均达到
6 倍以上吞吐；native token 12/12 exact，Hyper token 11/12 exact，但 sampled-token log-prob 门限仍失败，因此该命令
当前预期以非零状态保存诊断报告。

## Experimental Production Profile

`qwen3_5_0_8b_gsm8k_vllm_production.yaml` 使用 DP2、每 rank 2 prompts、每 prompt 6 responses、12 路请求并发、
`max_num_seqs=16`、2 GiB KV cache 和 batch-invariant。一个 drop-last epoch 为 1,868 update steps，处理
7,472/7,473 条训练数据。

当前 profile 已实现，但以下门禁尚未完成：

- 启用 batch-invariant 后的 colocated production canary；
- 完整 1,868-step 训练；
- sampled-token log-prob 精度对齐；
- graph、NZ、多节点和 colocated TP。

运行完整训练前必须先使用独立结果目录执行 canary：

```bash
RUN_ID=qwen35-gsm8k-prod-canary
export HYPER_VLLM_RESULT_ROOT="$(pwd)/.rollout-results/${RUN_ID}"

./hyper_parallel/rl/examples/scripts/run_qwen3_5_vllm_docker.sh \
  grpo-production-canary
```

canary 启动前会删除旧 marker；成功后写入绑定当前源码、production YAML、镜像 ID、模型/数据 artifact 和实现的
profile identity。`grpo-production` 会强制检查该 identity；只有 canary exit code、日志、refit 和清理均通过后，
才能在明确接受当前精度边界的前提下运行：

```bash
./hyper_parallel/rl/examples/scripts/run_qwen3_5_vllm_docker.sh grpo-production
```

native 必须使用独立结果目录，并执行自己的 canary/production 配对：

```bash
export HYPER_VLLM_MODEL_IMPLEMENTATION=native
export HYPER_VLLM_RESULT_ROOT="$(pwd)/.rollout-results/qwen35-gsm8k-prod-native"
./hyper_parallel/rl/examples/scripts/run_qwen3_5_vllm_docker.sh grpo-production-canary
./hyper_parallel/rl/examples/scripts/run_qwen3_5_vllm_docker.sh grpo-production
```

只有同一实现的 canary 和 production 会有意共享结果目录；不同实现或不同作业不能共享结果目录、容器名或 HCCL
范围。

## 后台运行

```bash
export HYPER_VLLM_DETACHED=true
export HYPER_VLLM_CONTAINER_NAME=qwen35-m3-hyper

container_id=$(
  ./hyper_parallel/rl/examples/scripts/run_qwen3_5_vllm_docker.sh grpo-m3-soak
)
exit_code=$(docker wait "${container_id}")
docker inspect --format '{{.State.Status}} {{.State.ExitCode}} {{.State.OOMKilled}}' \
  "${container_id}"
[[ "${exit_code}" == "0" ]] || exit "${exit_code}"
```

后台模式下必须同时检查容器 exit code 和结果目录中的持久化日志。容器内部 watchdog 仍由
`HYPER_VLLM_TIMEOUT_SECONDS` 控制。

## Focused Tests

在固定镜像或等价依赖环境中执行：

```bash
HYPER_PARALLEL_PLATFORM=torch \
PYTHONPATH=hyper_parallel/rl \
python -m pytest -q \
  hyper_parallel/rl/tests/test_architecture.py \
  hyper_parallel/rl/tests/test_colocated_vllm.py \
  hyper_parallel/rl/tests/test_dataset.py
```

完整 pytest 本轮没有重新执行，不应引用历史 passed 数量作为当前提交的完整回归结论。

## 常见问题

- 模型或数据目录不存在：检查 `HYPER_VLLM_MODEL_ROOT` 和 `HYPER_VLLM_DATA_ROOT`。
- M3 数据不存在：先在同一 `HYPER_VLLM_RESULT_ROOT` 下运行 `grpo-m3-select`。
- 共卡配置被拒绝：当前要求单节点、`dp_shard > 1`、CPU offload、forward 后 reshard 和 rollout TP1。
- CPU collective backend 错误：共卡 CPU offload 使用 `cpu:gloo,npu:hccl`，不要强制所有 tensor 走纯 HCCL。
- HCCL bind 失败：为每个并行作业设置独立基础端口和不重叠的 socket 范围。
- vLLM 内存不足：降低 cache budget 或模型规模；当前 strict loader 的 packed buffer 需要容纳完整策略。
- batch-invariant 已启用但 A/B 失败：该开关不等于当前 token/log-prob 精度已经验收，查看状态文档中的实际报告。

完整实验数据、故障分析和当前验收矩阵见
[vLLM 在线 Rollout 故障记录与验收状态](vllm_online_rollout_status.md)。
