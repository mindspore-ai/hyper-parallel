# HyperParallel-RL 功能导航图

每行记录配置或入口 → 执行分支 → 数据/指标 → 代表测试。本文对应当前仓库实现；配置与合同变化时更新相关行，
内部实现调整但合同与入口未变时无需扩写。架构边界见 [RL 架构](rl-architecture.md)。

路径约定：`rl/`、`examples/` 相对 `hyper_parallel/rl/`；`tests/`、`hyper_parallel/` 相对仓库根目录。
`文件.py::Class.method` 或 `文件.py::function` 指明实现符号。测试列是代表覆盖，不是所有场景均已通过的声明。
UT 包含 CPU 计算与 mock；真实模型、通信和数值效果需执行对应 ST。
目录校验器只校验 AGENTS 清单，本文的路径、符号和语义需另外核查。

## 1. 配置、模型与支持范围

| 功能 | 配置或入口 | 实现分支 | 数据或指标 | 代表测试 |
| --- | --- | --- | --- | --- |
| 运行镜像 | 默认统一镜像；启动脚本及 `RL_ST_*_IMAGE` 可覆盖 | [镜像下载与校验](../hyper_parallel/rl/docker/README.md) | 同一镜像包含 RL 基础依赖、Codex CLI 和 DeepSeek Harness | — |
| 配置校验与运行配置 | YAML 顶层字段 | `rl/config.py::validate_config`、`rl/config.py::build_runtime_config` | 主项目 `TrainerConfig` | `tests/ut/rl/trainer/test_config_runtime.py` |
| 模型身份与 Qwen3 加载 | `model.registry_name`、`model.name`、`model.weights_path` | `rl/roles/model_setup.py::resolve_model`、`hyper_parallel/rl/rl/roles/qwen3_builder.py::Qwen3AutoModel` | `ModelRegistration`；checkpoint 身份仅接受 Qwen3 dense | `tests/ut/rl/trainer/test_qwen3_master.py`、`tests/ut/rl/trainer/test_config_runtime.py::test_removed_model_families_fail_before_runtime_construction` |
| 训练拓扑边界 | `train.accelerator.dp_shard`、`tp`、`dp_replicate`、`cp`、`pp`、`ep` | `rl/config.py::_trainer_topology`、`rl/config.py::_validate_trainer_ep` | TP1/TP2；正整数 FSDP 分片数；其余维度固定为 1 | `tests/ut/rl/trainer/test_config_runtime.py::test_training_topology_is_checked_for_all_engines`、`tests/ut/rl/trainer/test_config_runtime.py::test_dense_runtime_rejects_expert_parallelism` |

## 2. 算法、目标与策略角色

| 功能 | 配置或入口 | 实现分支 | 数据或指标 | 代表测试 |
| --- | --- | --- | --- | --- |
| GRPO | `algorithm.name=grpo` | `rl/algorithm/loss.py::GRPOAlgorithm.build_targets`、`rl/algorithm/loss.py::GRPOAlgorithm.compute_actor_loss` | 分组优势；`train/policy_loss`、`train/kl_loss` | `tests/ut/rl/algorithm/test_algorithm_registry.py`、`tests/ut/rl/algorithm/test_algorithm_loss.py` |
| PPO | `algorithm.name=ppo`、`algorithm.gamma`、`algorithm.gae_lambda` | `rl/algorithm/loss.py::PPOAlgorithm.build_targets`、`rl/algorithm/loss.py::PPOAlgorithm.compute_actor_loss`、`rl/algorithm/loss.py::PPOAlgorithm.compute_critic_loss` | advantages、returns；`critic/value_loss` | `tests/ut/rl/algorithm/test_algorithm_loss.py`、`tests/ut/rl/trainer/test_ppo_targets.py` |
| 分组优势与 GAE | 由算法选择 estimator | `rl/algorithm/advantage.py::GroupRelativeAdvantageEstimator.estimate`、`rl/algorithm/advantage.py::GAEAdvantageEstimator.estimate` | `TargetOutput` | `tests/ut/rl/algorithm/test_algorithm_advantage.py` |
| PPO bootstrap 与优势归一化 | rollout 终止状态、Critic 最后一个 token 的值 | `rl/dataset/batch_builder.py::get_bootstrap_values`、`rl/dataset/batch_builder.py::ExperiencePreparer.prepare` | 冻结的旧 values、bootstrap、returns、loss mask | `tests/ut/rl/trainer/test_ppo_targets.py::TestPPOTargets.test_gae_bootstrap_and_observation_gap`、`tests/ut/rl/data/test_experience_preparer.py` |
| Actor 重算 | 日志诊断或 `consistency.enabled=true` | `rl/trainer.py::SyncTrainer._prepare_experience`、`rl/roles/policy/actor.py::Actor.compute_log_probs` | Actor logprobs 用于诊断；训练 old_log_probs 保持 rollout 来源 | `tests/ut/rl/policy/test_policy_compute.py`、`tests/ut/rl/consistency/test_consistency.py` |
| Actor 更新 | `train.micro_batch_size`、`train.response_mini_batch_size`、`train.policy_update_epochs` | `rl/roles/policy/actor.py::Actor.update`、`rl/roles/policy/actor.py::Actor.forward_backward` | `ActorUpdateMetrics`；`train/gradient_norm`、`train/optimizer_steps` | `tests/ut/rl/policy/test_policy_update.py` |
| Reference | 内置 GRPO/PPO 的角色需求；`algorithm.kl_coef=0` 不移除 Reference | `rl/trainer.py::SyncTrainer._build_models_and_optimizers`、`rl/roles/policy/actor.py::Actor.__init__` | 冻结模型；`ExperienceBatch.reference_log_probs` | `tests/ut/rl/policy/test_actor_roles.py`、`tests/ut/rl/trainer/test_trainer_orchestration.py` |
| PPO Critic | `train.critic.weights_path`、`optimizer`、`micro_batch_size`、`response_mini_batch_size`、`update_epochs` | `rl/roles/policy/critic.py::build_value_model`、`rl/roles/policy/critic.py::Critic.compute_values`、`rl/roles/policy/critic.py::Critic.update` | 价值头已接入；`CriticUpdateMetrics`、`critic/value_loss`、`critic/optimizer_steps` | `tests/ut/rl/trainer/test_ppo_value_model.py`、`tests/ut/rl/policy/test_policy_update.py` |

## 3. Rollout 与 Agentic

| 功能 | 配置或入口 | 实现分支 | 数据或指标 | 代表测试 |
| --- | --- | --- | --- | --- |
| vLLM 生成 | `rollout.engine=vllm` | `rl/roles/rollout/registry.py::build_rollout_engine`、`rl/roles/rollout/vllm.py::VLLMGenerationEngine.generate` | `GenerationResult`；`rollout/generated_tokens`、`rollout/tokens_per_second` | `tests/ut/rl/rollout/test_vllm_runtime.py` |
| Native / Hyper Qwen3 | `rollout.vllm.model_implementation` 为 `native` 或 `hyper` | `rl/roles/model_setup.py::resolve_vllm_model`、`rl/roles/rollout/vllm_plugin.py::register_hyper_models`、`rl/roles/rollout/consistency_models/qwen3/model.py::HyperQwen3ForCausalLM` | 两种模型实现共享生成及权重更新控制合同 | `tests/ut/rl/rollout/test_qwen3_adapter.py`、`tests/ut/rl/rollout/test_vllm_plugin.py` |
| 服务拓扑与设备 | `rollout.vllm.deployment`、`data_parallel_size`、`tensor_parallel_size`、`visible_devices` | `rl/config.py::_validate_vllm_basics`、`rl/roles/rollout/topology.py::resolve_vllm_rollout_topology` | 共享服务；colocated/disjoint 的设备与训练规模约束 | `tests/ut/rl/rollout/test_rollout_topology.py`、`tests/ut/rl/trainer/test_config_runtime.py` |
| 运行容量 | `rollout.vllm.max_model_len`、`max_num_seqs`、`max_num_batched_tokens` | `rl/config.py::resolve_vllm_automatic_limits`、`rl/config.py::_validate_vllm_limits` | 自动解析容量并校验约束 | `tests/ut/rl/trainer/test_config_runtime.py` |
| GSM8K 环境与奖励 | `agentic.module_path=examples.gsm8k.agent`、`agentic.environment=gsm8k_tools` | `rl/agentic/envs/environment.py::load_agentic_module`、`examples/gsm8k/agent.py::build_gsm8k_environment`、`examples/gsm8k/agent.py::compute_gsm8k_reward` | `Trajectory.reward`；`reward/mean`、`reward/accuracy` | `tests/ut/rl/agentic/agentic_ut.py` 验证通用合同；真实 GSM8K 奖励/学习证据见 `tests/torch/rl/test_rl_st.py` |
| 内部交互与工具 | `agentic.runner=internal`、`agentic.max_turns` | `rl/agentic/core/session.py::AgentSession`、`rl/agentic/tools/executor.py::ToolExecutor` | 带版本的 EpisodeContext、Trajectory 与工具结果 | `tests/ut/rl/agentic/agentic_ut.py`、`tests/ut/rl/rollout/test_worker.py` |
| Codex 程序 | `agentic.runner=codex`、`agentic.codex.*` | `rl/roles/rollout/worker.py::CodexRolloutManager`、`rl/agentic/codex/`、`rl/agentic/core/program_runner.py` | 程序生成轨迹、token/logprob 与策略版本 | `tests/ut/rl/agentic/agentic_ut.py`；NPU case `codex-agent` |
| DeepSeek 程序 | `agentic.runner=deepseek`、`agentic.deepseek.*` | `rl/roles/rollout/worker.py::DeepSeekRolloutManager`、`rl/agentic/ds_harness/`、`rl/agentic/core/program_runner.py` | Agent Harness 接入，模型仍按 Qwen3 dense 边界校验 | `tests/ut/rl/agentic/agentic_ut.py`；NPU case `deepseek-agent` |

## 4. 权重同步与发布

| 功能 | 配置或入口 | 实现分支 | 数据或指标 | 代表测试 |
| --- | --- | --- | --- | --- |
| 策略选择 | `rollout.vllm.weight_sync.strategy`、`bucket_size_mb` | `rl/roles/weight_sync/config.py::resolve_weight_sync_config`、`rl/roles/weight_sync/transfer.py::build_weight_transfer` | 默认 full_gather；`weight_sync/configured_full_gather`、`weight_sync/configured_direct_reshard` | `tests/ut/rl/weight_sync/test_weight_sync_strategy.py` |
| 完整参数 full-gather | `strategy=full_gather` | `rl/roles/weight_sync/transfer.py::FullGatherStrategy`、`rl/roles/weight_sync/packed_weight.py::build_packed_weight_buckets` | 整参数逐桶物化；桶大小不是大参数的绝对内存上限 | `tests/ut/rl/weight_sync/test_packed_weight.py`、`tests/ut/rl/weight_sync/test_weight_sync_transport.py` |
| Direct reshard | `strategy=direct_reshard` | `rl/roles/weight_sync/transfer.py::DirectReshardStrategy`、`rl/roles/weight_sync/layout.py` | 源/目标布局、交集与传输计划 | `tests/ut/rl/weight_sync/test_direct_reshard.py`、`tests/ut/rl/weight_sync/test_weight_sync_transport.py` |
| 设备传输 | `deployment=colocated` 或 `disjoint` | `rl/roles/weight_sync/ipc.py`、`rl/roles/weight_sync/hccl.py` | colocated→IPC；disjoint→HCCL | `tests/ut/rl/weight_sync/test_weight_sync_transport.py`、`tests/ut/rl/weight_sync/test_weight_sync_worker.py` |
| 版本提交与失败 | `PolicySnapshot` | `rl/trainer.py::SyncTrainer._publish_policy`、`rl/roles/weight_sync/transfer.py::WeightPublisher.publish`、`rl/roles/weight_sync/sync.py::ActorRolloutWeightSync.prepare_for_rollout` | 核对提交版本；`policy/version`；异常向上传播，无自动 fallback | `tests/ut/rl/weight_sync/test_weight_sync_transaction.py`、`tests/ut/rl/trainer/test_trainer_orchestration.py::test_trainer_publication_releases_training_state_before_rollout_wake` |

## 5. 一致性门禁

| 功能 | 配置或入口 | 实现分支 | 数据或指标 | 代表测试 |
| --- | --- | --- | --- | --- |
| 更新前一致性 | `consistency.enabled=true` | `rl/trainer.py::SyncTrainer._prepare_experience`、`rl/consistency/gates.py::validate_pre_update_consistency` | 策略版本、有效 action token；`training/pre_update_exact_valid`、`training/pre_update_exact_tokens` | `tests/ut/rl/consistency/test_gates.py`、`tests/ut/rl/consistency/test_consistency.py` |
| Qwen3 Ascend 配方 | `consistency.enabled`、`rollout.vllm.batch_invariant` | `rl/consistency/qwen3_dense.py::configure_consistency_profile`、`rl/roles/rollout/vllm_plugin.py::register_hyper_models` | `HYPER_RL_CONSISTENCY_PROFILE` 与指定模型/运行环境 | `tests/ut/rl/trainer/test_config_runtime.py`、`tests/ut/rl/rollout/test_vllm_runtime.py` |
| 更新后诊断 | 一致性配方开启时 | `rl/consistency/gates.py::measure_post_update_old_policy_mismatch` | 更新后参数与旧 rollout 的差异；不要求学习更新后仍 bit-exact | `tests/ut/rl/consistency/test_gates.py` |

## 6. 数据、训练、保存与观测

| 功能 | 配置或入口 | 实现分支 | 数据或指标 | 代表测试 |
| --- | --- | --- | --- | --- |
| 同步主循环 | `train.max_steps` | `rl/trainer.py::SyncTrainer.train`、`rl/trainer.py::SyncTrainer._train_step` | `RLTrainerState`；`train/global_step` | `tests/ut/rl/trainer/test_trainer_orchestration.py` |
| 退出与初始化失败清理 | Trainer 初始化失败或训练退出 | `rl/process_cleanup.py::cleanup_processes`、`rl/process_cleanup.py::destroy_process_group` | 服务关闭、生命周期状态复位、进程组与缓存释放 | `tests/ut/rl/trainer/test_process_cleanup.py`、`tests/ut/rl/trainer/test_trainer_orchestration.py` |
| Parquet prompt 数据 | `data.train_path`、`data.test_path` | `rl/dataset/data_source.py::PromptDataset`、`rl/dataset/data_source.py::build_prompt_records` | `PromptRecord` 与批次 | `tests/ut/rl/data/test_data_source.py` |
| 经验与目标 | rollout 结果、Reference、可选 Critic | `rl/dataset/batch_builder.py::build_experience_batch`、`rl/dataset/batch_builder.py::ExperiencePreparer.prepare` | `ExperienceBatch` 的 mask、old_log_probs、advantages、returns | `tests/ut/rl/data/test_contracts.py`、`tests/ut/rl/data/test_experience_preparer.py` |
| 保存与恢复 | `train.checkpoint.save_steps`、`save_final`、`load_path` | `rl/checkpoint.py::RLCheckpointManager`、`rl/trainer.py::SyncTrainer.train` | 角色模型/优化器/调度器、数据进度、global_step、RNG；恢复后重新发布策略 | `tests/ut/rl/trainer/test_checkpoint.py`、`tests/ut/rl/trainer/test_ppo_targets.py::TestPPOCheckpoint.test_both_roles_survive_resume` |
| 评估 | `evaluation.enabled`；保存边界和最终保存调度 | `rl/evaluation.py::Evaluator.run`、`rl/trainer.py::SyncTrainer._complete_step` | `validation/` 指标及样本表 | `tests/ut/rl/utils/test_evaluation.py`、`tests/ut/rl/trainer/test_trainer_orchestration.py` |
| 指标与记录 | `logging.backends`、`logging.log_steps`、`logging.wandb.*` | `rl/utils/monitoring/metrics.py::build_training_metrics`、`rl/utils/monitoring/tracker.py` | `train/`、`critic/`、`reward/`、`rollout/`、`policy/`、`weight_sync/` | `tests/ut/rl/utils/test_monitoring_metrics.py`、`tests/ut/rl/utils/test_monitoring_tracker.py` |
| 学习证据门 | `train.learning_gate.enabled` | `rl/utils/monitoring/metrics.py::enforce_learning_gate` | 奖励差异、有限且非零的梯度等检查；不保证算法收敛 | `tests/ut/rl/utils/test_monitoring_metrics.py` |

## 7. ST 配方与覆盖边界

`tests/common/rl_st_cases.py` 中的 `CASES` 和 `prepare_config` 是 UT/ST 共用的配置来源。
`tests/ut/rl/trainer/test_config_runtime.py::test_all_st_recipes_pass_production_validation` 只证明配方通过配置校验，
不证明训练已执行。`tests/torch/rl/test_rl_st.py::test_rl_system` 才是实际 NPU 入口，
`tests/torch/rl/st_runtime.py` 负责运行和恢复阶段，`tests/torch/rl/st_evidence.py` 检查结果。

| 配方组 | case 名称 | 关注内容 |
| --- | --- | --- |
| Dense GRPO | `dense-tp2-direct` | 训练、Actor 发布、拓扑与传输 |
| 一致性 | `dense-tp2-consistency-full` | 指定配方下的更新前一致性与版本 |
| 程序 Agent | `codex-agent`、`deepseek-agent` | 外部程序轨迹与训练衔接 |
| 保存恢复 | `checkpoint-resume` | 分阶段保存、恢复与继续训练 |
| PPO | `ppo-tp1-full` | Actor/Critic 更新 |

真实 ST 使用 `level1`、`allcards` 标记，执行前需具备模型、数据、镜像和设备；默认 PR 门禁不能代替全部配方验收。
运行条件见 [ST 说明](../hyper_parallel/rl/docs/hyper-rl-st.md)，PPO 验证边界见
[PPO 文档](../hyper_parallel/rl/docs/ppo.md)，bit-exact 的条件见
[训练推理一致性](../hyper_parallel/rl/docs/qwen3_training_inference_consistency.md)。
当前不包含 MoE/EP、自动策略 fallback，也不包含旧独立 streaming-full-gather 模块。
