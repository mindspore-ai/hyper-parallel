# GSM8K 训练示例

[English](README.md) | **简体中文**

## 训练配方

| 任务 | 配置 |
| --- | --- |
| Qwen3-4B GRPO，使用内部 runner | [GRPO](configs/qwen3_4b_gsm8k_base.yaml) |
| Qwen3-4B，使用 Codex | [Codex GRPO](configs/qwen3_4b_gsm8k_codex_50step.yaml) |
| GSPO，使用内部 runner | [GSPO](configs/qwen3_4b_gsm8k_gspo.yaml) |
| PPO，使用 Critic 与内部 runner | [PPO](configs/qwen3_4b_gsm8k_ppo.yaml) |
| 模型奖励，使用内部 runner | [Reward Model](configs/qwen3_4b_gsm8k_model_reward.yaml) |
| DeepSeek Harness | [DeepSeek Harness](configs/deepseek_multi_turn.yaml) |
| Qwen3-30B-A3B，使用内部 runner | [MoE GRPO](configs/qwen3_30b_a3b_gsm8k_vllm.yaml) |
| Qwen3-30B-A3B，使用 Codex | [MoE Codex](configs/qwen3_30b_a3b_codex_multi_turn.yaml) |

按算法和 agent runner 选择配方，再通过 CLI 覆盖部署相关路径。
Codex 配方保留已有的 50-step 文件名；实际运行步数由 `--train.max_steps` 选择。
路由重放见 [MoE 指南](../../docs/moe_code_agent.zh.md)，
外部 runner 见 [Agentic RL 指南](../../docs/agentic_rl.zh.md)。

## 运行训练

按照 [Docker 指南](../../docker/README.zh.md)准备运行环境。
设置当前环境的模型、数据集与输出路径。
以下命令在八张设备上运行内部 GRPO 配方。

~~~bash
export GSM8K_MODEL=/absolute/path/to/Qwen3-4B
export GSM8K_DATA=/absolute/path/to/gsm8k
export GSM8K_RESULTS=/absolute/path/to/new-result
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python -m torch.distributed.run --nproc_per_node=8 \
  hyper_parallel/rl/train_rl.py \
  hyper_parallel/rl/examples/gsm8k/configs/qwen3_4b_gsm8k_base.yaml \
  --model.weights_path="$GSM8K_MODEL" \
  --model.tokenizer_path="$GSM8K_MODEL" \
  --data.train_path="$GSM8K_DATA/train.parquet" \
  --data.test_path="$GSM8K_DATA/test.parquet" \
  --rollout.vllm.max_num_seqs=8 \
  --train.max_steps=50 \
  --train.checkpoint.output_dir="$GSM8K_RESULTS/checkpoints" \
  --logging.backends='[console]' \
  --logging.wandb.mode=disabled
~~~

短程两步检查使用 `--train.max_steps=2`。
按可用硬件调整所选配方的设备拓扑与生成限制。
使用 Codex 或 DeepSeek Harness 时，先配置对应 runner 及其依赖，再启动训练。

## 奖励与 Agent 交互

规则奖励由 `agent.py::compute_gsm8k_reward()` 实现。
它从生成文本中提取答案，并与数据集答案比较。
模型奖励配方改用其配置的奖励模型。

通过所选 YAML 配方配置多轮交互、聊天模板、
轮次上限及观察预算，也可使用 CLI 覆盖本次运行的这些设置。

张量并行运行见 [TP 启动器](scripts/run_qwen3_tp_docker.sh)，
rollout 配置见 [vLLM 指南](../../docs/vllm_rollout.zh.md)。
