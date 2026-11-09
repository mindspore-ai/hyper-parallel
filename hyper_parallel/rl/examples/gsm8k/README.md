# GSM8K training examples

**English** | [简体中文](README.zh.md)

## Training recipes

| Task | Configuration |
| --- | --- |
| Qwen3-4B GRPO with the internal runner | [GRPO](configs/qwen3_4b_gsm8k_base.yaml) |
| Qwen3-4B with Codex | [Codex GRPO](configs/qwen3_4b_gsm8k_codex_50step.yaml) |
| GSPO with the internal runner | [GSPO](configs/qwen3_4b_gsm8k_gspo.yaml) |
| PPO with a Critic and the internal runner | [PPO](configs/qwen3_4b_gsm8k_ppo.yaml) |
| Model-based rewards with the internal runner | [Reward Model](configs/qwen3_4b_gsm8k_model_reward.yaml) |
| DeepSeek Harness | [DeepSeek Harness](configs/deepseek_multi_turn.yaml) |
| Qwen3-30B-A3B with the internal runner | [MoE GRPO](configs/qwen3_30b_a3b_gsm8k_vllm.yaml) |
| Qwen3-30B-A3B with Codex | [MoE Codex](configs/qwen3_30b_a3b_codex_multi_turn.yaml) |

Select a recipe for the algorithm and agent runner, then override deployment-specific paths in the CLI.
The Codex recipe retains its existing 50-step filename; use `--train.max_steps` to choose the run length.
See the [MoE guide](../../docs/moe_code_agent.md) for router replay and the
[Agentic RL guide](../../docs/agentic_rl.md) for external runners.

## Run training

Prepare the runtime using the [Docker guide](../../docker/README.md).
Set the model, dataset, and output paths for your environment.
The following command runs the internal GRPO recipe with eight devices.

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

Use `--train.max_steps=2` for a short two-step run.
Adjust the selected recipe's device topology and generation limits to match the available hardware.
For Codex and DeepSeek Harness, configure the corresponding runner and its dependencies before launching.

## Rewards and agent interaction

The rule-based reward is implemented in `agent.py::compute_gsm8k_reward()`.
It extracts the answer from the generated text and compares it with the dataset answer.
The model-based reward recipe uses its configured reward model instead.

Use the selected YAML recipe to configure multi-turn interaction, chat templates,
turn limits, and observation budgets. CLI overrides can adjust these values for a run.

See the [TP launcher](scripts/run_qwen3_tp_docker.sh) for tensor-parallel runs and the
[vLLM guide](../../docs/vllm_rollout.md) for rollout settings.
