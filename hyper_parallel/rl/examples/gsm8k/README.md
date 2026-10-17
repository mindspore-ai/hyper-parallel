# GSM8K 运行配置与脚本

## 选择入口

只改变部署方式、TP 大小或运行参数时，复用同一个入口并覆盖参数，不再复制完整 YAML。

| 脚本 | 用途与支持值 |
| --- | --- |
| [Agent 入口](scripts/run_qwen3_4b_agentic_docker.sh) | `HYPER_AGENTIC_TASK=gsm8k`（默认）、`gsm8k_multi`、`gsm8k_codex`、`gsm8k_deepseek`；两卡 Qwen3-4B |
| [TP 入口](scripts/run_qwen3_tp_docker.sh) | `colocated` / `disjoint`；Hyper / native、训练 TP 与推理 TP 可分别选择；不声明严格一致性 |
| [一致性入口](scripts/run_qwen3_consistency_docker.sh) | `colocated`（默认）/ `disjoint`；`HYPER_QWEN3_TP=1` / `2`，训推 TP 相同，开启严格一致性 |

三个脚本均支持在末尾追加 `--section.field=value`，可选值在配置字段及脚本参数旁注明。
不同入口使用各自的任务预算、设备配置和验收要求。
Agent 入口检查版本、checkpoint 和更新次数：internal 每步为 1 次，外部 harness 按实际调用行数允许多次更新。
例如选择 Codex 或 DeepSeek Harness，只改变 Agent 任务参数：

```bash
export HYPER_VLLM_MODEL_ROOT=/path/to/models/Qwen3-4B
export HYPER_GSM8K_DATA_ROOT=/path/to/data/gsm8k
export HYPER_VLLM_VISIBLE_DEVICES=0,1
export HYPER_AGENTIC_TASK=gsm8k_codex  # gsm8k | gsm8k_multi | gsm8k_codex | gsm8k_deepseek
bash hyper_parallel/rl/examples/gsm8k/scripts/run_qwen3_4b_agentic_docker.sh \
  --train.model_init_cpu_threads=8
```

DeepSeek 指 harness，不是模型；它使用 Qwen3-4B。选择 `gsm8k_deepseek` 时，使用独立的
`HYPER_DEEPSEEK_MODEL_ROOT`、`HYPER_DEEPSEEK_DATA_ROOT` 和 `HYPER_DEEPSEEK_VISIBLE_DEVICES`
指定同类资源。四卡 MoE Codex 的命令见下文。

## 保留的配置

| 配置 | 独立保留的原因 |
| --- | --- |
| [Qwen3-4B 基础 GRPO](configs/qwen3_4b_gsm8k_vllm_production.yaml) | 单轮、内部多轮、短程与一致性脚本共用 |
| [GSPO](configs/qwen3_4b_gsm8k_gspo.yaml) | 序列级损失与算法字段不同，不能只改 GRPO 的算法名 |
| [PPO](configs/qwen3_4b_gsm8k_ppo.yaml) | 独立 Critic 及其优化配置 |
| [模型奖励](configs/qwen3_4b_gsm8k_model_reward.yaml) | 独立 Reward Model 服务及评分配置 |
| [Codex](configs/codex_multi_turn.yaml) | Codex CLI 协议、提示与预算 |
| [DeepSeek Harness](configs/deepseek_multi_turn.yaml) | SDK 协议、提示与预算 |
| [Qwen3-30B-A3B 单轮](configs/qwen3_30b_a3b_gsm8k_vllm.yaml) | MoE 模型与 EP/EDP 拓扑 |
| [Qwen3-30B-A3B Codex](configs/qwen3_30b_a3b_codex_multi_turn.yaml) | 四卡 MoE Codex、多轮预算与可选 R3 |

单轮和 TP2 consistency 入口使用基础配置及脚本参数覆盖；短程入口保留短程预算。
内部多轮使用以下参数，无需独立 YAML。

### 内部多轮参数

在基础 GRPO 配置上覆盖以下字段即可；这是 internal runner，不调用 Codex：

```bash
torchrun --nproc_per_node=8 hyper_parallel/rl/train_rl.py \
  hyper_parallel/rl/examples/gsm8k/configs/qwen3_4b_gsm8k_vllm_production.yaml \
  --agentic.interaction_mode=multi_turn \
  --agentic.apply_chat_template=true \
  --agentic.max_turns=4 \
  --agentic.max_observation_tokens=256
```

该命令保留基础配置的八卡完整训练规模；两卡短程检查使用 Agent 入口的 `gsm8k_multi` 模式。
工具协议默认 `json_function_call`，超时 5 秒、并发 2、每轮最多调用 2 次。

## Qwen3-30B-A3B Codex 多轮与 R3

[四卡运行配置](configs/qwen3_30b_a3b_codex_multi_turn.yaml)使用 Qwen3-30B-A3B、Codex CLI 0.152.1 和 GRPO。
训练为 TP2、DP shard2、EP2、EDP shard2，native vLLM 为 TP4、DP1；保持 consistency off 和 EPLB off。
任务预算与 [Codex 配置](configs/codex_multi_turn.yaml)一致：每题 4 个候选、最多 2 次模型调用、
每调用最多 256 tokens、8192 上下文、两步训练。预算耗尽而未完成答案的候选记零奖励，保留采样动作。
该配方默认关闭评测和 checkpoint，用于两步功能检查，不代表完整训练或收敛配方。

### 环境和路径

使用已准备好的四卡 Ascend Docker 训练容器，运行时为 vLLM 0.23.0、vLLM-Ascend 0.23.0.post1；
可使用 [v0.23 Code Agent 镜像](../../docker/README.md#仓库型-code-agent-与-qwen35-家族)。
GSM8K 的默认镜像现已统一为该 v0.23 镜像；本四卡 MoE 配方仍使用下面的直接训练命令，
不使用固定两卡的 Qwen3-4B Agent 启动器。
容器需要以下挂载；只读取已有模型和数据，不自动下载：

| 内容 | 容器路径 |
| --- | --- |
| 当前仓库源码 | `/repo` |
| Qwen3-30B-A3B 权重与 tokenizer | `/models/Qwen3-30B-A3B` |
| GSM8K `train.parquet`、`test.parquet` | `/data/gsm8k` |
| 可写结果目录 | `/results` |

按 [Docker 宿主要求](../../docker/README.md#宿主要求)挂载兼容的 Ascend driver。
以下命令在容器内、仓库根目录执行；设备编号替换为已分配的四张卡：

```bash
cd /repo
source /usr/local/Ascend/ascend-toolkit/set_env.sh
git config --global --add safe.directory /repo
python -m pip install --no-deps --no-build-isolation -e .
bash hyper_parallel/rl/docker/install_runtime.sh
export PYTHONPATH=/repo/hyper_parallel/rl:/repo:${PYTHONPATH:-}
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_HOST_IP=127.0.0.1
export GLOO_SOCKET_IFNAME=lo
export HCCL_IF_BASE_PORT=62300
export HCCL_NPU_SOCKET_PORT_RANGE=62300-62399
python -c 'import hyper_parallel; print(hyper_parallel.__file__)'
codex --version
```

导入应指向 `/repo`，Codex 版本应为 0.152.1。示例中的 Git 配置只在该容器内生效。
训练 HCCL 端口为 62300–62399，配方中 vLLM 使用 62600–62699；同时运行其他任务时需分配不冲突的端口。

### 默认运行与显式开启 R3

不加覆盖参数时，R3 关闭，初始化线程数保持环境原值，vLLM sleep level 为 1：

```bash
torchrun --nnodes=1 --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=62299 \
  hyper_parallel/rl/train_rl.py \
  hyper_parallel/rl/examples/gsm8k/configs/qwen3_30b_a3b_codex_multi_turn.yaml
```

开启 R3，并将每个训练 rank 的模型初始化线程数限制为 8：

```bash
torchrun --nnodes=1 --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=62299 \
  hyper_parallel/rl/train_rl.py \
  hyper_parallel/rl/examples/gsm8k/configs/qwen3_30b_a3b_codex_multi_turn.yaml \
  --train.router_replay.mode=R3 \
  --train.model_init_cpu_threads=8
```

初始化结束后恢复原线程设置，训练进程仍为四个。
R3 不改变任务预算、奖励规则或 Codex 协议；本配方已设置所需的推理 TP4/DP1。
支持范围、路由采集与重放合同见 [MoE R3](../../docs/moe_code_agent.md#moe-r3)。
CPU 内存不足时可追加 `--rollout.vllm.sleep_level=2`，适用条件见
[推理生命周期](../../docs/vllm_rollout.md#policy-生命周期)。

## 其他现有脚本

GSM8K Docker 启动脚本可在命令末尾接收 `train_rl.py` 的 `--section.field=value` 覆盖参数，例如：

```bash
bash hyper_parallel/rl/examples/gsm8k/scripts/run_qwen3_4b_agentic_docker.sh \
  --train.model_init_cpu_threads=8 \
  --rollout.vllm.sleep_level=2
```

仍须按该脚本原要求配置镜像、模型、数据和设备环境变量；这里的 dense 示例不启用 R3。
TP 和 consistency 脚本需先给出原来的 `colocated` 或 `disjoint` 位置参数，再追加覆盖参数。
覆盖参数必须指向 YAML 中已有字段；不传参数时保持原行为。Search-R1 的宿主启动器也支持相同形式的训练参数，
SWE-Smith 与单轮 code 的直接训练命令则沿用 `train_rl.py` 接口。

验收入口及边界见 [功能与验证说明](../../docs/moe_code_agent.md#系统测试)。
