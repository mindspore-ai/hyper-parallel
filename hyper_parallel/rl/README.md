# Hyper-RL

Hyper-RL 是 HyperParallel 面向大语言模型强化学习的同步训练运行时。它使用 HyperAutoModel 与 HyperParallel
FSDP/TP 训练 Actor，通过一个共享的 vLLM DP×TP deployment 生成样本，并在训练过程中发布最新策略权重。

本版本已基于最新 `upstream/master@b9fa61a9` 的 HyperAutoModel、Trainer 和 distributed checkpoint 架构完成合入，
不再依赖旧 trainer_dev 训练流程或 RL 专用 Trainer 副本。

当前发布范围聚焦单节点 Ascend NPU 上的 Qwen3 GRPO，支持 Trainer TP1/TP2、Hyper-vLLM/Native-vLLM TP1/TP2，
以及 Qwen3 + Hyper-vLLM matched TP1/TP2 在 colocated 和 disjoint deployment 下的训练-推理 bit-exact。

## 设计原则

**统一训练架构**：Trainer 直接加载 Transformers 模型定义，并使用 HyperAutoModel 的 FSDP/TP、optimizer、
checkpoint 和 gradient clipping，不维护 RL 专用 Trainer 副本。

**训推一体**：Trainer 与 Hyper-vLLM 都以 Transformers Qwen3 定义为模型语义来源，并复用 HyperParallel 的
ShardingPlanner、TP placement、tied-weight 和 source-layout contract。Hyper-vLLM 只在推理执行边界接入 vLLM paged
attention、KV cache 和 worker lifecycle，使模型结构与分布式切分保持一致，同时保留高性能推理能力。

**共享 Rollout**：Colocated 与 disjoint 都使用一个 coordinator、一个 endpoint 和 vLLM 管理的 DP×TP workers。
Hyper-RL 不实现第二个 Router，也不建立 Trainer rank 到 rollout worker 的固定映射。

**强策略一致性**：Policy version 单调递增。Generation、权重事务、cache reset 和 resume 都验证 worker-local
identity；失败时所有 Trainer ranks 同步退出，未完成策略不可见。

**一致性能力隔离**：普通训练允许 Trainer TP 与 rollout TP 不同。只有显式设置 `consistency.enabled=true` 时，
才启用 Qwen3 Ascend 数值 recipe，并要求 Hyper-vLLM 与 matched TP。

## 架构简介

```text
Transformers Qwen3
    -> HyperAutoModel FSDP / FSDP+TP Trainer
        -> one shared vLLM endpoint
            -> upstream DP router
                -> DP engine 0 -> TP workers
                -> DP engine 1 -> TP workers
                -> ...
```

- `colocated`：Trainer 与 rollout 共用 NPU，通过 sleep/wake 切换 residency，使用 NPU IPC 发布权重。
- `disjoint`：Trainer 与 rollout 使用不相交的 NPU，rollout 保持 resident，使用 HCCL 发布权重。
- TP1 权重同步自动使用 full-gather；Qwen3 TP2 支持 full-gather、direct-reshard 和 full-gather fallback。
- vLLM upstream 负责 DP request routing 与 frontend 数量；已删除 rank-local server 和额外 topology 配置。

详细设计见 [Hyper-RL 架构](docs/architecture.md) 和 [vLLM Rollout](docs/vllm_rollout.md)。

## 关键特性

- Trainer
  - [x] Qwen3 TP1、pure TP2 和 FSDP-shard×TP2
  - [x] BF16、global gradient norm、AdamW 和 DCP resume
  - [ ] Trainer CP、PP、EP 与多节点
- 训推一体
  - [x] Trainer/Hyper-vLLM 共享 Transformers Qwen3 模型语义
  - [x] 共享 HyperParallel TP planner、placement、tied-weight 和 layout contract
  - [x] vLLM paged attention/KV cache 作为推理专用执行叶子
- Rollout
  - [x] Hyper-vLLM 与 Native-vLLM TP1/TP2
  - [x] Colocated NPU IPC 与 disjoint HCCL
  - [x] Prefix Cache、Chunked Prefill、persistent async HTTP admission
  - [ ] 异步/off-policy rollout、动态扩缩容和透明 generation retry
- 在线权重发布
  - [x] Full-gather 与 TP-aware direct-reshard
  - [x] Transaction abort、fallback、worker identity 和 source-derived manifest
  - [ ] 多节点与 rollout EP
- 训练-推理一致性
  - [x] Qwen3 + Hyper-vLLM matched TP1/TP2
  - [x] Colocated `FSDP-shard2×TP2→DP2×TP2`
  - [x] Disjoint `FSDP-shard2×TP2→DP2×TP2`
  - [x] Optimizer update 前 FP32 raw selected-token logprob bit-exact
  - [ ] Native-vLLM、TP4/TP8 和多节点 bit-exact

## 代码规模

核心实现按 `hyper_parallel/rl/rl/**/*.py` 统计，不包含 README/docs、测试、examples、Docker 和配置文件：

| 模块 | Python 文件 | 物理行数 |
| --- | ---: | ---: |
| Weight sync | 7 | 5,374 |
| Trainer/config/evaluation | 5 | 2,415 |
| Rollout | 8 | 2,304 |
| Monitoring/utils | 9 | 1,039 |
| Consistency | 4 | 937 |
| Agentic | 7 | 885 |
| Algorithm | 4 | 785 |
| Policy/model | 5 | 763 |
| Dataset | 4 | 667 |
| **合计** | **53** | **15,169** |

去除空行和纯注释后约为 13,312 行，其中仍包含 docstring。vLLM plugin shim 和 `examples/train_rl.py` 是工程入口，
未计入上述核心实现。

## 已验证环境

| 组件 | 版本或范围 |
| --- | --- |
| 硬件 | 单节点 Ascend 910B3；colocated TP2 使用 4 NPU，disjoint 完整拓扑使用 8 NPU |
| Docker image | `swr.cn-east-3.myhuaweicloud.com/huawei-hyper-rl/hyper-rl:v0.22.1rc1-arm64` |
| CANN | 9.0.0 |
| Torch / torch-npu | 2.10.0 |
| Transformers | 5.5.4 |
| vLLM / vLLM-Ascend | 0.22.1 / 0.22.1rc1 |
| batch-invariant-ops | 1.0.0 |
| flash-attn-npu | 0.2.0b1 |
| 数据 | GSM8K parquet，包含 `prompt` 和 `extra_info` |

镜像下载与校验见 [运行镜像](docs/hyper_rl_runtime_image.md)。正式 launcher 默认直接使用该公开镜像，运行时不需要手工
tag、安装 wheel、配置 `PYTHONPATH` 或加载 CANN。

## 快速开始

以下是一条从公开镜像到正式运行的完整流程。所有命令都在包含 `hyper_parallel/rl` 的仓库根目录执行。

### 1. 下载镜像

```bash
docker pull \
  swr.cn-east-3.myhuaweicloud.com/huawei-hyper-rl/hyper-rl:v0.22.1rc1-arm64
```

两个 launcher 已默认使用该地址，不需要再执行 `docker tag`。如果机器已有自定义本地镜像，可分别通过
`HYPER_QWEN3_IMAGE` 或 `HYPER_QWEN3_TP_IMAGE` 覆盖。

### 2. 准备模型、数据和结果目录

```text
/absolute/path/to/Qwen3-4B/
    config.json
    tokenizer_config.json
    model*.safetensors

/absolute/path/to/gsm8k/
    train.parquet
    test.parquet
```

设置共享路径并执行启动前检查：

```bash
export HYPER_RL_MODEL_ROOT=/absolute/path/to/Qwen3-4B
export HYPER_RL_DATA_ROOT=/absolute/path/to/gsm8k
export HYPER_RL_RESULT_ROOT=/absolute/path/to/results

test -f "${HYPER_RL_MODEL_ROOT}/config.json"
test -f "${HYPER_RL_DATA_ROOT}/train.parquet"
test -f "${HYPER_RL_DATA_ROOT}/test.parquet"
mkdir -p "${HYPER_RL_RESULT_ROOT}"

npu-smi info
```

只选择 `Health=OK` 且没有其他运行进程的 NPU。Launcher 会自动挂载仓库源码、driver、模型、数据和结果目录。

### 3. 运行 Qwen3 普通 TP 训练与推理

四卡示例运行 Trainer `FSDP-shard2×TP2` 与 rollout `DP2×TP2`：

```bash
export HYPER_QWEN3_TP_MODEL_ROOT="${HYPER_RL_MODEL_ROOT}"
export HYPER_QWEN3_TP_DATA_ROOT="${HYPER_RL_DATA_ROOT}"
export HYPER_QWEN3_TP_RESULT_ROOT="${HYPER_RL_RESULT_ROOT}/normal-tp2"
export HYPER_QWEN3_TP_VISIBLE_DEVICES=0,1,2,3
export HYPER_QWEN3_TP_TRAINER_TP=2
export HYPER_QWEN3_TP_ROLLOUT_TP=2
export HYPER_QWEN3_TP_IMPLEMENTATION=hyper

./hyper_parallel/rl/examples/scripts/run_qwen3_tp_docker.sh colocated
```

设置 `HYPER_QWEN3_TP_IMPLEMENTATION=native` 可运行 Native-vLLM。普通模式允许独立设置 Trainer/rollout TP；
已验证 TP1→TP2 与 TP2→TP1。默认使用 direct-reshard 和 full-gather fallback，也可以显式选择：

```bash
export HYPER_QWEN3_TP_WEIGHT_SYNC_STRATEGY=full_gather
export HYPER_QWEN3_TP_WEIGHT_SYNC_FALLBACK=none
```

异卡使用同一 launcher 和配置结构，只需提供完整且不重叠的 Trainer/rollout 设备集合。以下示例运行
Trainer `FSDP-shard2×TP2` 到 rollout `DP2×TP2`：

```bash
export HYPER_QWEN3_TP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HYPER_QWEN3_TP_TRAINER_COUNT=4
export HYPER_QWEN3_TP_ROLLOUT_DP=2

./hyper_parallel/rl/examples/scripts/run_qwen3_tp_docker.sh disjoint
```

### 4. 运行 Qwen3 TP1/TP2 Bit-Exact

TP1 默认使用两张 NPU；TP2 使用四张 NPU，并设置 `HYPER_QWEN3_TP=2`：

```bash
export HYPER_QWEN3_MODEL_ROOT="${HYPER_RL_MODEL_ROOT}"
export HYPER_QWEN3_DATA_ROOT="${HYPER_RL_DATA_ROOT}"
export HYPER_QWEN3_RESULT_ROOT="${HYPER_RL_RESULT_ROOT}/consistency-tp2"
export HYPER_QWEN3_VISIBLE_DEVICES=0,1,2,3
export HYPER_QWEN3_TP=2

./hyper_parallel/rl/examples/scripts/run_qwen3_consistency_docker.sh colocated
```

验证 direct-reshard 时增加：

```bash
export HYPER_QWEN3_WEIGHT_SYNC_STRATEGY=direct_reshard
export HYPER_QWEN3_WEIGHT_SYNC_FALLBACK=full_gather
```

Disjoint TP2 使用相同 consistency recipe 和参数名，8 卡完整拓扑命令为：

```bash
export HYPER_QWEN3_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HYPER_QWEN3_TP=2
export HYPER_QWEN3_TRAINER_COUNT=4
export HYPER_QWEN3_ROLLOUT_DP=2

./hyper_parallel/rl/examples/scripts/run_qwen3_consistency_docker.sh disjoint
```

成功运行时日志保存在所选 result root。Bit-exact 每步必须出现非零 valid token，并满足 mismatch count、max absolute diff 和
mean absolute diff 均为 0；失败会返回非零退出码，不会静默发布新 policy version。

Bit-exact 的比较时点、数值 recipe 和验收指标见
[Qwen3 训练-推理一致性](docs/qwen3_training_inference_consistency.md)。

## 配置边界

- `rollout.vllm.deployment`：`colocated|disjoint`。
- `rollout.vllm.model_implementation`：`hyper|native`。
- `rollout.vllm.weight_sync.strategy`：`full_gather|direct_reshard`。
- `rollout.vllm.weight_sync.fallback_strategy`：`none|full_gather`。
- `consistency.enabled=false` 时两侧 TP 可以不同；设为 `true` 时只允许 Qwen3 Hyper-vLLM matched TP。
- Disjoint 必须提供与 rollout DP×TP 数量一致、且不与 Trainer 重叠的 `visible_devices`。
- 已删除并显式拒绝：`rollout.vllm.topology`、`request_concurrency`、`api_server_count` 及旧 topology 环境变量。

## 当前限制

- 当前只声明 Qwen3、单节点 Torch NPU 和同步 GRPO。
- PPO/GAE/Critic 具备数学与接口测试，但需要 Critic 的端到端配置仍会被拒绝。
- Bit-exact 不覆盖 Native-vLLM、backward、gradient、optimizer state、更新后参数或收敛表现。
- 不声明 TP4/TP8、多节点、graph、speculative decoding、长期 soak 或跨 workload 性能最优。
- vLLM RLHF/refit development endpoints 使用不安全序列化，只能运行在受信任、隔离的训练网络。

## 文档

- [Hyper-RL 架构](docs/architecture.md)：角色、数据合同、生命周期和 checkpoint。
- [vLLM Rollout](docs/vllm_rollout.md)：ownership、request admission、权重事务和失败语义。
- [Qwen3 训练-推理一致性](docs/qwen3_training_inference_consistency.md)：bit-exact 定义、配置和门禁。
- [运行镜像](docs/hyper_rl_runtime_image.md)：公开镜像下载、校验和宿主要求。
- [公共模块修改说明](docs/public_module_changes.md)：面向 CODEOWNER 的 RL 目录外修改、必要性和接口影响。
