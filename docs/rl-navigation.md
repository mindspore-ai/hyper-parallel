# HyperParallel 功能导航图

> **可追溯性门禁。** 对每个面向用户的功能,本图用一行回答:*用哪个配置键开启 →
> 从哪个模块进入 → 核心逻辑在哪个文件 → 产出哪个指标 → 由哪个测试守护。*
> Agent 应能不还原隐藏控制流就走完这几列;人应能一遍读完。
>
> **维护规则:** 修改已记录的配置、行为、入口或代表测试时,在同一个 diff 更新对应行。
> 内部实现变化但导航仍然有效时,在报告中说明核对结果,无需为了门禁修改文档。
> 本表保留稳定入口和代表测试,不枚举所有内部函数与指标;不另建一份手工清单。
>
> 列含义:`配置键` = 配置路径或 CLI 开关;`入口` = 第一个消费它的模块;
> `核心分支` = 核心逻辑文件;`指标` = 产出的 metric key;`测试` = 守护它的测试。
> 按功能域分组。`配置键` 中的 `>` 表示 YAML 内的嵌套路径。
> RL 路径相对 `hyper_parallel/rl/`。精确符号使用 `文件.py::Class.method`
> 或 `文件.py::function`;检查器静态校验这些引用,无需导入训练依赖。
> 文件内移动符号无需修改行号;跨文件移动或改名会使检查失败。
> 配置含义、指标语义和测试覆盖关系仍需人工评审,检查通过不代表语义正确。

---

## 1. 并行能力(核心库)

| 功能 | 配置键 | 入口 | 核心分支 | 指标 | 测试 |
| --- | --- | --- | --- | --- | --- |
| DTensor redistribute | `redistribute()` | `core/dtensor/` | `core/dtensor/` | —(进程内) | `tests/ut/` DTensor 用例 |
| Tensor parallel | `parallelize_module()` | `core/tensor_parallel/` | `core/tensor_parallel/` | — | `tests/torch/` TP 用例 |
| FSDP shard | `fully_shard()` | `core/fully_shard/` | `core/fully_shard/` | — | `tests/torch/` FSDP 用例 |
| HSDP | `hsdp_*.py` | `core/fully_shard/` | `core/fully_shard/` | — | `tests/torch/` HSDP 用例 |
| Pipeline stage | `PipelineStage` | `core/pipeline_parallel/` | `core/pipeline_parallel/` | — | `tests/torch/` PP 用例 |
| Activation checkpoint | `checkpoint_wrapper()` | `core/activation_checkpoint/` | `core/activation_checkpoint/` | — | `tests/torch/` activation 用例 |
| Distributed checkpoint | `save/load` | `core/distributed_checkpoint/` | `core/distributed_checkpoint/` | — | `tests/` DCP 用例 |

> 这部分刻意保持模块级粗粒度,因为逐算子/逐策略的细节属于 `core/` 及其自身测试。
> 下面的 **RL** 部分才是本图重点守护的对象 —— 那里最容易丢失
> 功能→分支→指标→测试 的追溯链。

---

## 2. Hyper-RL —— 算法与策略

| 功能 | 配置键 | 入口 | 核心分支 | 指标 | 测试 |
| --- | --- | --- | --- | --- | --- |
| GRPO 算法 | `algorithm.name=grpo` | `rl/algorithm/loss.py` | `rl/algorithm/advantage.py`(优势), `rl/algorithm/loss.py`(损失) | `train/total_loss`, `train/policy_loss`, `train/kl_loss`, `train/clip_fraction` | `rl_tests/test_algorithm_registry.py` |
| PPO 算法 | `algorithm.name=ppo` | `rl/algorithm/loss.py` | `rl/algorithm/loss.py` | 同 GRPO | `rl_tests/test_algorithm_registry.py` |
| 优势估计:GRPO | `algorithm.name=grpo` → advantage | `rl/algorithm/advantage.py` | `rl/algorithm/advantage.py::GroupRelativeAdvantageEstimator.estimate` | `train/advantage` 分布 | `rl_tests/test_algorithm_registry.py` |
| 优势估计:GAE | `algorithm.name=ppo`（PPO 内部选择 GAE） | `rl/algorithm/loss.py::PPOAlgorithm.__init__` | `rl/algorithm/advantage.py::GAEAdvantageEstimator.estimate` | `train/advantage` 分布 | `rl_tests/test_algorithm_registry.py` |
| 奖励:GSM8K | `agentic.module_path=examples.agents.gsm8k.agent`, `agentic.environment=gsm8k_tools` | `rl/agentic/envs/environment.py::load_agentic_module` | `examples/agents/gsm8k/agent.py::compute_gsm8k_reward` | `reward/mean`, `reward/accuracy`, `reward/min`, `reward/max` | 待补示例奖励专项测试（现有算法 registry 测试不覆盖该函数） |

## 3. Hyper-RL —— rollout

| 功能 | 配置键 | 入口 | 核心分支 | 指标 | 测试 |
| --- | --- | --- | --- | --- | --- |
| vLLM rollout 引擎 | `rollout.engine=vllm` | `rl/roles/rollout/registry.py` | `rl/roles/rollout/vllm.py`, `rl/roles/rollout/vllm_qwen3.py` | 生成期(rollout record) | `rl_tests/test_vllm_runtime.py` |
| Qwen3 rollout 适配 | `rollout.vllm.model_implementation=hyper\|native` | `rl/roles/rollout/vllm_qwen3.py` | `rl/roles/rollout/vllm_qwen3.py` | — | `rl_tests/test_qwen3_launcher.py`, `test_qwen3_tp_launcher.py` |
| rollout 拓扑 | `rollout.vllm.deployment` | `rl/config.py::_validate_vllm_basics` | `rl/roles/rollout/topology.py` | — | `rl_tests/test_rollout_topology.py`, `test_disjoint_topology.py` |
| vLLM 运行时生命周期 | `rollout.vllm.*`(port、dp/tp、max_num_seqs…) | `rl/config.py` | `rl/roles/rollout/worker.py` | — | `rl_tests/test_vllm_runtime.py` |

> **已按设计移除**(不要重新引入;`config.py` 会直接拒绝):
> `rollout.vllm.topology`、`rollout.vllm.request_concurrency`、
> `rollout.vllm.api_server_count`。由 `rl/config.py` 校验及 `rl_tests` 中的拒绝用例守护。

## 4. Hyper-RL —— 权重同步

| 功能 | 配置键 | 入口 | 核心分支 | 指标 | 测试 |
| --- | --- | --- | --- | --- | --- |
| 权重同步策略 | `rollout.vllm.weight_sync.strategy` | `rl/config.py::_validate_vllm_weight_sync`, `rl/roles/rollout/vllm.py`(`build_vllm_engine` 处传入) | `rl/roles/weight_sync/transfer.py::build_weight_transfer`, `rl/roles/rollout/vllm.py::build_vllm_engine` | —(可观察 `configured_strategy` / `last_strategy` / `fallback_count`) | `rl_tests/test_direct_reshard.py`(`test_build_weight_transfer_skips_reshard_for_pure_dp`, `test_direct_failure_aborts_transaction_then_uses_full_gather`, `test_successful_direct_publication_updates_strategy_counters`) |
| 传输 IPC/HCCL | `deployment`(colocated→IPC,disjoint→HCCL) | `rl/roles/weight_sync/transfer.py` | `rl/roles/weight_sync/hccl.py`, `rl/roles/weight_sync/transfer.py` | — | `rl_tests/test_hccl_weight_sync.py` |
| 发布生命周期 | policy version / fingerprint | `rl/roles/weight_sync/vllm_worker.py` | `rl/roles/weight_sync/` | worker 本地 identity | `rl_tests/test_checkpoint_manager.py` |

> `rl/roles/weight_sync/` 是后果最重的子系统 —— 6 个模块 + `__init__.py` / 约 5.4k 行。
> 布局契约位于 `rl/roles/weight_sync/layout.py`。

## 5. Hyper-RL —— 一致性(bit-exact)

| 功能 | 配置键 | 入口 | 核心分支 | 指标 | 测试 |
| --- | --- | --- | --- | --- | --- |
| bit-exact 门 | `consistency.enabled=true` | `rl/config.py`(经 `rl/consistency/qwen3_dense.py` `configure_consistency_profile` 校验), `rl/trainer.py`(`pre-update` 阶段调用) | `rl/consistency/qwen3_dense.py`(配方/安装), `rl/consistency/gates.py::validate_pre_update_consistency`, `rl/trainer.py`(调用时机) | `training/pre_update_exact_valid`, `training/pre_update_exact_tokens`, `training/pre_update_mismatch_count`, `training/pre_update_max_abs_diff`, `training/pre_update_mean_abs_diff`;通过 = `mismatch_count`/`max_abs_diff`/`mean_abs_diff` 全为 `0`(即 `0/0/0`) | `rl_tests/test_trainer_orchestration.py`(`test_pre_update_consistency_accepts_identical_worker_owned_policy`, `test_pre_update_consistency_rejects_equal_values_with_different_bits`, `test_post_update_negative_control_counts_exact_bit_changes`) |
| Qwen3-Ascend 数值配方 | `consistency.enabled=true` | `rl/consistency/qwen3_dense.py::configure_consistency_profile`, `rl/roles/rollout/vllm_plugin.py`(经 `HYPER_RL_CONSISTENCY_PROFILE` 装 rollout 侧) | `rl/consistency/qwen3_dense.py`, `rl/consistency/vllm_ascend.py` | — | `rl_tests/test_config.py`(trainer 侧), `rl_tests/test_vllm_runtime.py`(rollout 侧) |

> 跑一致性冒烟必须用专用启动脚本(`run_qwen3_consistency_docker.sh`);普通 TP
> 启动脚本(`run_qwen3_tp_docker.sh`)显式关闭了 consistency,**不能**用它来声称
> bit-exact。完整策略矩阵见 `.agent/skills/hyper-rl-dev/references/npu-smoke.md`。

## 6. Hyper-RL —— trainer / dataset / env

| 功能 | 配置键 | 入口 | 核心分支 | 指标 | 测试 |
| --- | --- | --- | --- | --- | --- |
| 同步训练主循环 | `train.max_steps` | `rl/trainer.py` | `rl/trainer.py` | `train/global_step`, `train/optimizer_steps` | `rl_tests/test_trainer_orchestration.py` |
| 数据源(Parquet) | `data.train_path` | `rl/dataset/data_source.py` | `rl/dataset/data_source.py` | — | `rl_tests/test_data_source.py` |
| Batch builder | — | `rl/dataset/batch_builder.py` | `rl/dataset/batch_builder.py` | — | `rl_tests/test_experience_preparer.py`, `test_algorithm_registry.py` |
| Agentic 环境 | `agentic.module_path`, `agentic.environment=gsm8k_tools`, `agentic.max_turns` | `rl/agentic/envs/environment.py::load_agentic_module` | `examples/agents/gsm8k/agent.py::build_gsm8k_environment`, `rl/agentic/core/session.py` | 每轮 reward | `rl_tests/test_agentic_runner.py::test_agent_runner_preserves_two_turn_eos_mask_and_logprobs`（通用 runner 契约） |
| 评估 Evaluation | `evaluation.enabled` | `rl/evaluation.py` | `rl/evaluation.py` | `reward/*` | `rl_tests/test_trainer_orchestration.py` |
| 配置校验 | 顶层 YAML 键 | `rl/config.py` | `rl/config.py` | — | `rl_tests/test_config.py` |

---

## 附录 —— 顶层配置 schema

允许的顶层键(来自 `rl/config.py` 的 `_EXPECTED_TOP_LEVEL`):

```text
model · data · rollout · agentic · algorithm · evaluation · train · logging · consistency
```

`rollout.vllm.weight_sync.strategy` 接受 `direct_reshard` 与 `full_gather`
(配合 `fallback_strategy ∈ {none, full_gather}`);默认即 **`full_gather`**,
`bucket_size_mb` 默认 128。当 `rollout.vllm.tensor_parallel_size == 1` 时实际
策略恒为 `full_gather`;`direct_reshard` 目前仅支持 Qwen3。

---

## 提交前自检清单

1. 这个功能在本图里有行吗?没有就**加一行** —— 若是新配置键,放到对应的功能域下。
2. 如果你**移动了逻辑**(换了模块),更新该行的 `入口` / `核心分支`。
3. 如果功能的关键指标发生变化,更新 `指标` 列。
4. 如果代表测试改名或迁移,更新 `测试` 列;新增内部测试无需逐项登记。
5. 改动是否涉及**已按设计移除**的行?复查 `config.py` 是否仍然拒绝它,并在 diff 中说明。
6. 新功能添加稳定入口和代表测试;新子系统同时更新架构索引。
7. 运行 `python3 .agent/scripts/check_agents_catalog.py`;审查 diff 中的导航关系是否仍成立。
