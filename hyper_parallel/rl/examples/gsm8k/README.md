# Qwen3-4B / GSM8K

## 默认八卡 50 步训练

| 配方 | 配置 | 启动参数 |
| --- | --- | --- |
| Hyper-RL 单轮 | [单轮50步](configs/qwen3_4b_gsm8k_50step.yaml) | `hyper` |
| Hyper-RL + Codex | [Codex 50步](configs/qwen3_4b_gsm8k_codex_50step.yaml) | `codex` |

准备八张空闲 Ascend NPU（`device_share=false`）、Docker、Git、Python3.9+及[运行镜像](../../docker/README.md)。
数据目录需含固定6400题的 `train.parquet` 和独立500题的 `test.parquet`，prompt 不超过1024 tokens。
每步128题×8候选、一次优化器更新；在0/10/20/30/40/50步验证，均采用verl严格GSM8K奖励，Hyper每5步保存checkpoint。

在仓库根目录运行；`RL_RESULTS` 必须是尚不存在的新目录，其父目录已存在：

```bash
export RL_MODEL=/absolute/path/to/Qwen3-4B
export RL_DATA=/absolute/path/to/prepared-gsm8k
export RL_RESULTS=/absolute/path/to/new-result
bash hyper_parallel/rl/examples/gsm8k/scripts/run_50step_docker.sh hyper
```

末尾参数可换成 `codex`。后台运行同一入口：

```bash
tmux new-session -d -s gsm8k50 \
  -e RL_MODEL="$RL_MODEL" -e RL_DATA="$RL_DATA" -e RL_RESULTS="$RL_RESULTS" \
  'bash hyper_parallel/rl/examples/gsm8k/scripts/run_50step_docker.sh codex'
```

默认使用 Hyper v0.23 Code Agent 镜像。
`RL_IMAGE` 可覆盖镜像，`RL_WAIT_TIMEOUT` 设置等待空闲时间（默认3600秒）。
启动器保存源码、配置和数据文件哈希，安装当前源码并应用已知版本的采样器修补，记录显存、利用率及外部占用。
日志写入 `RL_RESULTS`；失败保留证据，不自动重跑。数据题数、去重及训练/验证无交集须在启动前准备确认。

所有规则评分统一使用 `agent.py::compute_gsm8k_reward()`：在回复末尾300字符中提取最后一个 `#### 数字`，
去掉数字中的逗号后与标准答案字符串比较；不再接受无标记答案或将 `42.0` 归一化为 `42`。
Codex/DeepSeek及内部多轮复用同一规则；独立Reward Model配方仍使用模型评分。

## 核心设置

两份 Hyper 配方采用 GRPO、BF16计算/FP32主参数、lr=1e-6、KL=0.001、DP/FSDP8、训练TP1。
Actor训练CPU offload关闭、Reference卸载、逐微批同步；生成前卸载训练状态、训练前恢复。
Actor/old log-prob/Reference动态预算均为2048 tokens，超预算单条成批（仍须满足显存容量）；DP均衡关闭、概率模式为dense。
推理TP2×DP4，每副本256并发、显存比例0.8、批token上限8192，开启prefix cache、chunked prefill和图执行。
单次输出上限1024；单轮上下文2048，Codex上下文3096、最多两次调用，不强制使用工具。
Codex开启CPU准备重叠（CPU比例0.5），每rank128个episode、Gateway上限1024；仅Codex返回采样log-prob，均由Actor重算old log-prob。
字段约束见[训练架构](../../docs/architecture.md)和[Agentic说明](../../docs/agentic_rl.md)。

## 最终结果

| 运行 | 硬件 | 最终验证reward | 平均训练秒/步 |
| --- | --- | ---: | ---: |
| 历史verl | 8×64GiB | 91.4% | 180.24 |
| Hyper-RL单轮 | 8×32GiB | 94.4% | 122.47 |
| Hyper-RL + Codex | 8×32GiB | 93.0% | 220.38 |

均完成50步；时间不含验证及保存。历史verl早于采样修复，硬件/版本不同；Codex还改变了prompt和调用方式，
因此不是严格同配置性能对照，也不代表最新代码所有修复均已通过50步验收。当前32GiB verl配方仍缺完整验收。
详细结果、曲线、优化说明和数据审计/绘图脚本保存在本地 `hyper_parallel/rl/output/hyper-rl-results/gsm8k/`，不随主仓提交。
verl 基线配置及独立启动脚本也保存在该本地目录；主仓仅提供 Hyper-RL 运行入口。

## 其他功能

| 功能 | 配置或说明 |
| --- | --- |
| 基础 GRPO / 内部多轮 / TP 与一致性测试 | [基础配置](configs/qwen3_4b_gsm8k_base.yaml)，供功能测试及并行策略入口复用 |
| GSPO / PPO / 模型奖励 | [GSPO](configs/qwen3_4b_gsm8k_gspo.yaml)、[PPO](configs/qwen3_4b_gsm8k_ppo.yaml)、[Reward Model](configs/qwen3_4b_gsm8k_model_reward.yaml) |
| DeepSeek Harness | [配置](configs/deepseek_multi_turn.yaml)、[运行要求](../../docs/agentic_rl.md) |
| Qwen3-30B-A3B | [单轮](configs/qwen3_30b_a3b_gsm8k_vllm.yaml)、[Codex](configs/qwen3_30b_a3b_codex_multi_turn.yaml)、[MoE与R3](../../docs/moe_code_agent.md#启用-r3) |
| TP / 一致性检查 | [TP入口](scripts/run_qwen3_tp_docker.sh)、[一致性入口](scripts/run_qwen3_consistency_docker.sh)、[合同](../../docs/qwen3_training_inference_consistency.md) |

内部多轮runner复用[基础GRPO配方](configs/qwen3_4b_gsm8k_base.yaml)，设置
`agentic.interaction_mode=multi_turn`、`apply_chat_template=true`、`max_turns=4`、`max_observation_tokens=256`。
在已安装运行时的容器内通过 `train_rl.py <配置文件>` 运行；路径、设备和预算以所选YAML为准。
