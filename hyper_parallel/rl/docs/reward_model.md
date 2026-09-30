# Colocated reward model contract

A top-level `reward_model` section makes HyperParallel-RL score completed trajectories
through a frozen vLLM service. Without this section, the task environment keeps its
rule-based reward. The task scorer owns reward semantics; the Trainer owns service
lifecycle and batch handoff.

## Supported scope

- Single node, Qwen3 dense, the internal `gsm8k_tools` environment, and colocated vLLM rollout.
- GRPO or GSPO. External Codex/DeepSeek runners, MoE, PPO, and disjoint rollout
  cannot use the current colocated reward service.
- Generative scoring calls `v1/chat/completions`. Discriminative scoring calls
  `classify` and requires a trained `SequenceClassification` checkpoint.
  The example Qwen3-4B generative model checks the scoring path; it is not a
  calibrated preference reward model.
- Configuration and CPU tests are present. Real NPU training, reward quality,
  and convergence still require separate acceptance.

## Configuration

Start from the [Qwen3-4B GSM8K recipe](../examples/gsm8k/configs/qwen3_4b_gsm8k_model_reward.yaml).
Its relevant fields are:

```yaml
algorithm:
  name: gspo
  loss_aggregation: seq-mean-token-mean
reward_model:
  scorer: examples.gsm8k.agent:score_gsm8k_environment_reward
  model_path: /models/Qwen3-4B
  scoring: generative
  deployment: colocated
  tensor_parallel_size: 1
  data_parallel_size: 8
  port: 8200
  server_hccl_if_base_port: 64200
  server_hccl_npu_socket_port_range: 64200-64299
```

RM TP times DP must equal the rollout device count. The RM needs its own port and
a valid non-overlapping HCCL port range. Training requires `cpu_offload=true`.
Adjust the model, data, device count, and output path for the target machine.
Use `hyper-parallel/hyper-rl:v0.22.1rc1-unified-arm64` and follow the
[runtime image guide](../docker/README.md) for source and Ascend driver mounts.

`reward_model.scorer` names an async `module:function` that accepts a prompt,
a completed trajectory, and the reward client, then returns a finite scalar or
`RewardResult`. The GSM8K generative scorer requires one JSON score. The
discriminative scorer uses the reward model's own chat template and classifier.

## Lifecycle and failures

`SyncTrainer` samples the original tokens and log probabilities, sleeps the rollout
service, starts or wakes the reward service, scores the entire batch, sleeps the
reward service, then updates the Actor and publishes a new policy. Evaluation
uses the same scoring path. Training and recovery check the service state.

Before scoring, the code validates trajectory identity, policy version, and
pending-reward state. A TP group scores each logical candidate once. Rewards,
components, and evidence are committed only after every result passes finite
value and dtype checks. Transient request errors can be retried as configured;
protocol errors, service failures, and incomplete scores abort the batch rather
than becoming zero rewards.

Implementation: [configuration](../rl/config.py),
[Trainer](../rl/trainer.py), [service client](../rl/reward_model/client.py),
[batch scoring](../rl/reward_model/scoring.py), and
[task scorer](../examples/gsm8k/agent.py). CPU contract tests live in
`tests/ut/rl/reward_model/test_model_reward.py`. Real NPU acceptance
needs a prepared model, GSM8K data, and available devices.
