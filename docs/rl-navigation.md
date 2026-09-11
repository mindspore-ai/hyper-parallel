# hyperparallel-RL 功能导航图

> 每个 hyperparallel-RL 功能或关键契约占一行：配置或入口 → 执行分支 → 数据/指标 → 代表测试；`—` 表示不适用。按标题定位，通常只需读取受影响的一节。
>
> 配置、入口、行为、指标或代表测试变化时，在同一个 diff 更新对应行；内部实现变化但链路仍然有效时无需修改。
>
> 路径相对 `hyper_parallel/rl/`，精确符号写作 `文件.py::Class.method` 或 `文件.py::function`。Catalog checker 校验文件、符号和链接；配置含义及测试覆盖仍需评审。

---

## 1. 算法与策略

| 功能 | 配置键 | 入口 | 核心分支 | 数据/指标 | 测试 |
| --- | --- | --- | --- | --- | --- |
| GRPO 算法 | `algorithm.name=grpo` | `rl/algorithm/loss.py::build_algorithm` | `rl/algorithm/loss.py::GRPOAlgorithm.build_targets`, `rl/algorithm/loss.py::GRPOAlgorithm.compute_actor_loss` | `ActorUpdateMetrics`; `train/total_loss`, `train/policy_loss`, `train/kl_loss`, `train/clip_fraction` | `rl_tests/ut/algorithm/test_algorithm_registry.py` |
| PPO 算法 | `algorithm.name=ppo` | `rl/algorithm/loss.py::build_algorithm` | `rl/algorithm/loss.py::PPOAlgorithm.build_targets`, `rl/algorithm/loss.py::PPOAlgorithm.compute_actor_loss`, `rl/algorithm/loss.py::PPOAlgorithm.compute_critic_loss` | `ActorUpdateMetrics`; `train/total_loss`, `train/policy_loss`, `train/kl_loss`, `train/clip_fraction` | `rl_tests/ut/algorithm/test_algorithm_registry.py` |
| 优势估计:GRPO | `algorithm.name=grpo` → advantage | `rl/algorithm/advantage.py` | `rl/algorithm/advantage.py::GroupRelativeAdvantageEstimator.estimate` | `ExperienceBatch.advantages`; `train/advantage_mean`, `train/advantage_std`, `train/advantage_min`, `train/advantage_max` | `rl_tests/ut/algorithm/test_algorithm_advantage.py` |
| 优势估计:GAE | `algorithm.name=ppo`（PPO 内部选择 GAE） | `rl/algorithm/loss.py::PPOAlgorithm.__init__` | `rl/algorithm/advantage.py::GAEAdvantageEstimator.estimate` | `ExperienceBatch.advantages`; `train/advantage_mean`, `train/advantage_std`, `train/advantage_min`, `train/advantage_max` | `rl_tests/ut/algorithm/test_algorithm_advantage.py` |
| 奖励:GSM8K | `agentic.module_path=examples.agents.gsm8k.agent`, `agentic.environment=gsm8k_tools` | `rl/agentic/envs/environment.py::load_agentic_module` | `examples/agents/gsm8k/agent.py::compute_gsm8k_reward` | `Trajectory.reward`; `reward/mean`, `reward/accuracy`, `reward/min`, `reward/max` | 待补示例奖励专项测试（现有算法 registry 测试不覆盖该函数） |

## 2. Rollout

| 功能 | 配置键 | 入口 | 核心分支 | 数据/指标 | 测试 |
| --- | --- | --- | --- | --- | --- |
| vLLM rollout 引擎 | `rollout.engine=vllm` | `rl/roles/rollout/registry.py::build_rollout_engine` | `rl/roles/rollout/vllm.py::VLLMGenerationEngine.generate` | `GenerationResult`; `rollout/sequence_count`, `rollout/generated_tokens`, `rollout/tokens_per_second` | `rl_tests/ut/rollout/test_vllm_runtime.py` |
| Qwen3 rollout 适配 | `rollout.vllm.model_implementation=hyper\|native` | `rl/config.py::_validate_model_implementation` | `rl/roles/rollout/vllm_qwen3.py::HyperQwen3ForCausalLM` | `GenerationResult.sequences`, `GenerationResult.rollout_log_probs` | `rl_tests/ut/trainer/test_config_runtime.py`, `rl_tests/ut/rollout/test_vllm_runtime.py` |
| MoE 模型与静态 EP | checkpoint `model_type`；`train.accelerator.ep`、`rollout.vllm.enable_expert_parallel` | `rl/config.py::build_model_registration` | `rl/roles/rollout/vllm_qwen3_moe.py`、`rl/roles/rollout/vllm_deepseek_v3.py`、`rl/roles/rollout/vllm_moe.py` | `train/gradient_norm`、`policy/version` | `rl_tests/ut/trainer/test_config_runtime.py`、`rl_tests/ut/rollout/test_vllm_moe.py` |
| rollout 拓扑 | `rollout.vllm.deployment` | `rl/config.py::_validate_vllm_basics` | `rl/roles/rollout/topology.py::resolve_vllm_rollout_topology` | `VLLMRolloutTopology` | `rl_tests/ut/rollout/test_rollout_topology.py` |
| vLLM 运行时生命周期 | `rollout.vllm.*`(port、dp/tp、max_num_seqs…) | `rl/roles/rollout/vllm.py::build_vllm_engine` | `rl/roles/rollout/vllm.py::VLLMGenerationEngine.generate` | `GenerationResult.worker_policy_version`, `GenerationResult.worker_policy_fingerprint` | `rl_tests/ut/rollout/test_vllm_runtime.py` |

> **已按设计移除：**`rollout.vllm.topology`、`rollout.vllm.request_concurrency`、`rollout.vllm.api_server_count`。`rl/config.py` 会拒绝这些配置，`rl_tests` 有对应测试。

## 3. 权重同步

| 功能 | 配置键 | 入口 | 核心分支 | 数据/指标 | 测试 |
| --- | --- | --- | --- | --- | --- |
| 权重同步策略 | `rollout.vllm.weight_sync.strategy` | `rl/config.py::_validate_vllm_weight_sync`, `rl/roles/rollout/vllm.py::build_vllm_engine` | `rl/roles/weight_sync/config.py::resolve_weight_sync_config`, `rl/roles/weight_sync/transfer.py::build_weight_transfer` | `weight_sync/configured_full_gather`、`weight_sync/fallback_count` | `rl_tests/ut/weight_sync/test_weight_sync_strategy.py` |
| 有界 full-gather | `rollout.vllm.weight_sync.strategy=full_gather`、`bucket_size_mb` | `rl/roles/weight_sync/config.py::resolve_weight_sync_config` | `rl/roles/weight_sync/streaming_full_gather.py` | `weight_sync/streaming_bucket_count` | `rl_tests/ut/weight_sync/test_streaming_full_gather.py` |
| 传输 IPC/HCCL | `deployment`(colocated→IPC,disjoint→HCCL) | `rl/roles/weight_sync/transfer.py` | `rl/roles/weight_sync/hccl.py`, `rl/roles/weight_sync/transfer.py` | `PolicySnapshot.payload` | `rl_tests/ut/weight_sync/test_weight_sync_transport.py` |
| 发布生命周期 | — | `rl/trainer.py::SyncTrainer._publish_policy` | `rl/roles/weight_sync/vllm_worker.py::_finish_custom_weight_update` | `PolicySnapshot.version`; `policy/version`, `policy/fingerprint_changed` | `rl_tests/ut/trainer/test_trainer_orchestration.py::test_trainer_publication_releases_training_state_before_rollout_wake`、`rl_tests/ut/weight_sync/test_weight_sync_transaction.py` |

## 4. 一致性(bit-exact)

| 功能 | 配置键 | 入口 | 核心分支 | 数据/指标 | 测试 |
| --- | --- | --- | --- | --- | --- |
| bit-exact 门 | `consistency.enabled=true` | `rl/config.py`, `rl/trainer.py` | `rl/consistency/gates.py::validate_pre_update_consistency` | `GenerationResult.rollout_log_probs`; `training/pre_update_exact_valid`, `training/pre_update_exact_tokens`, `training/pre_update_mismatch_count`, `training/pre_update_max_abs_diff`, `training/pre_update_mean_abs_diff` | `rl_tests/ut/consistency/test_consistency.py` |
| Qwen3-Ascend 数值配方 | `consistency.enabled=true` | `rl/consistency/qwen3_dense.py::configure_consistency_profile`, `rl/roles/rollout/vllm_plugin.py`(经 `HYPER_RL_CONSISTENCY_PROFILE` 装 rollout 侧) | `rl/consistency/qwen3_dense.py`, `rl/consistency/vllm_ascend.py` | `HYPER_RL_CONSISTENCY_PROFILE` | `rl_tests/ut/trainer/test_config_runtime.py`(trainer 侧), `rl_tests/ut/rollout/test_vllm_runtime.py`(rollout 侧) |

> 一致性冒烟使用 `run_qwen3_consistency_docker.sh`；`run_qwen3_tp_docker.sh` 明确关闭 consistency，不能用于声明 bit-exact。判定标准见 [`qwen3_training_inference_consistency.md`](../hyper_parallel/rl/docs/qwen3_training_inference_consistency.md#修改门禁)。

## 5. Trainer / dataset / env

| 功能 | 配置键 | 入口 | 核心分支 | 数据/指标 | 测试 |
| --- | --- | --- | --- | --- | --- |
| 同步训练主循环 | `train.max_steps` | `rl/trainer.py::SyncTrainer.train` | `rl/trainer.py::SyncTrainer._train_step` | `ExperienceBatch`, `ActorUpdateMetrics`; `train/global_step`, `train/optimizer_steps` | `rl_tests/ut/trainer/test_trainer_orchestration.py` |
| 数据源(Parquet) | `data.train_path` | `rl/trainer.py::SyncTrainer._build_tokenizer_and_data` | `rl/dataset/data_source.py::PromptDataset.__getitem__`, `rl/dataset/data_source.py::build_prompt_records` | prompt sample → `PromptRecord` | `rl_tests/ut/data/test_data_source.py` |
| Batch builder | — | `rl/dataset/batch_builder.py::build_experience_batch` | `rl/dataset/batch_builder.py` | `ExperienceBatch` | `rl_tests/ut/data/test_experience_preparer.py`, `rl_tests/ut/algorithm/test_algorithm_registry.py` |
| Agentic 环境 | `agentic.module_path`, `agentic.environment=gsm8k_tools`, `agentic.max_turns` | `rl/agentic/envs/environment.py::load_agentic_module` | `examples/agents/gsm8k/agent.py::build_gsm8k_environment`, `rl/agentic/core/session.py` | `EpisodeContext`, `Trajectory.reward`; `reward/mean`, `reward/accuracy` | `rl_tests/ut/agentic/agentic_ut.py`（通用交互合同） |
| Codex harness | `agentic.runner=codex`, `agentic.codex.*` | `rl/trainer.py::SyncTrainer._build_rollout_runtime` | `rl/agentic/codex/`, `rl/agentic/core/program_runner.py`, `rl/roles/rollout/worker.py::CodexRolloutManager` | `reward/*`, `policy/version` | `rl_tests/ut/agentic/agentic_ut.py` |
| DeepSeek Harness | `agentic.runner=deepseek`, `agentic.deepseek.*` | `rl/trainer.py::SyncTrainer._build_rollout_runtime` | `rl/agentic/deepseek/`, `rl/agentic/core/program_runner.py`, `rl/roles/rollout/worker.py::DeepSeekRolloutManager` | `reward/*`, `policy/version` | `rl_tests/ut/agentic/agentic_ut.py` |
| 评估 Evaluation | `evaluation.enabled` | `rl/trainer.py::SyncTrainer.train` | `rl/evaluation.py::Evaluator.run` | `reward/mean`, `reward/accuracy` | `rl_tests/ut/trainer/test_trainer_orchestration.py` |
| 配置校验 | 顶层 YAML 键 | `rl/config.py::validate_config` | `rl/config.py::build_runtime_config` | `TrainerConfig` | `rl_tests/ut/trainer/test_config_runtime.py` |

## 6. Policy：Actor / Reference / Critic

| 功能 | 配置键 | 入口 | 核心分支 | 数据/指标 | 测试 |
| --- | --- | --- | --- | --- | --- |
| Actor logprobs 重算 | `consistency.enabled` 或诊断采集；`train.micro_batch_size` | `rl/trainer.py::SyncTrainer._prepare_experience` | `rl/roles/policy/actor.py::Actor.compute_log_probs`, `rl/roles/policy/actor.py::Actor.sequence_log_probs` | next-token 对齐的 FP32 logprobs，用于诊断与更新前一致性检查；`training/pre_update_exact_valid`（开启一致性时） | `rl_tests/ut/policy/test_policy_compute.py`、`rl_tests/ut/consistency/test_consistency.py`（CPU/mock）；旧 NPU 专项验收脚本已移除，替代覆盖待核对 |
| Actor 前反向与参数更新 | `train.micro_batch_size`, `train.response_mini_batch_size`, `train.policy_update_epochs`, `train.optimizer.*` | `rl/trainer.py::SyncTrainer._train_step` | `rl/roles/policy/actor.py::Actor.update`, `rl/roles/policy/actor.py::Actor.forward_backward`, `rl/roles/policy/actor.py::Actor._optimizer_step` | `ExperienceBatch.old_log_probs`, `advantages`, `loss_action_mask`; `ActorUpdateMetrics`; `train/policy_loss`, `train/gradient_norm`, `train/learning_rate`, `train/optimizer_steps` | `rl_tests/ut/policy/test_policy_update.py`、`rl_tests/ut/trainer/test_trainer_orchestration.py::test_trainer_step_orchestrates_required_role_outputs_in_order`（CPU/mock）；旧 NPU 专项验收脚本已移除，替代覆盖待核对 |
| Reference 冻结与推理 | `algorithm.name` 的角色/数据需求；当前 GRPO/PPO 均要求 Reference，`algorithm.kl_coef=0` 不取消该角色 | `rl/trainer.py::SyncTrainer._build_models_and_optimizers`, `rl/trainer.py::SyncTrainer._prepare_experience` | `rl/roles/policy/actor.py::Actor.__init__`（无 optimizer，冻结参数）, `rl/roles/policy/actor.py::Actor.compute_log_probs` | `ExperienceBatch.reference_log_probs`; `train/kl_loss` | `rl_tests/ut/policy/test_actor_roles.py`、`rl_tests/ut/trainer/test_trainer_orchestration.py::test_trainer_builds_independent_trainable_and_reference_roles`；Reference 数值专项验收待补 |
| Critic 值计算与更新（仅组件） | `algorithm.name=ppo` 声明需求，但当前端到端配置拒绝 Critic 算法 | `rl/trainer.py::SyncTrainer._prepare_experience`, `rl/trainer.py::SyncTrainer._train_step`（预留编排） | `rl/roles/policy/critic.py::Critic.compute_values`, `rl/roles/policy/critic.py::Critic.sequence_values`, `rl/roles/policy/critic.py::Critic.update`, `rl/roles/policy/critic.py::Critic.forward_backward` | `ExperienceBatch.values`, `returns`; `CriticUpdateMetrics`; `critic/value_loss`, `critic/optimizer_steps`（接入后） | `rl_tests/ut/policy/test_policy_update.py::test_critic_values_and_update_follow_policy_contract`（CPU/mock）；端到端验收待补 |

> Actor 重算的 logprobs 用于诊断与一致性校验，训练样本中的 `old_log_probs` 仍以 rollout 返回值为来源。NPU 验收入口需要相应硬件与运行环境，独立于默认单元测试执行。Critic 的编排测试注入 mock，不能作为实际 Critic 实现或 PPO 端到端支持的证明。
