# Hyper-RL Module Map

Start with [the RL rule](../hyper-rl.md). This file locates ownership;
[architecture](../../../docs/rl-architecture.md) explains boundaries and
[feature navigation](../../../docs/rl-navigation.md) maps configuration to implementation and tests.
Paths below are relative to `hyper_parallel/rl/` unless prefixed with `hyper_parallel/` or `tests/`.

| Area | Code or entry | Contract |
| --- | --- | --- |
| Configuration and CLI | `rl/config.py`, `rl/cli.py`, `train_rl.py` | [Runtime architecture](../../../hyper_parallel/rl/docs/architecture.md) |
| Synchronous roles and orchestration | `rl/trainer.py`, `rl/roles/model_setup.py` | [Training step](../../../hyper_parallel/rl/docs/architecture.md#一个训练步) |
| Asynchronous orchestration | `rl/async_training/`, `examples/async_training/launch.py` | [Rayless GRPO](../../../hyper_parallel/rl/docs/architecture.md#rayless-asynchronous-grpo) |
| Data and episode identity | `rl/dataset/`, `rl/agentic/core/types.py` | [Data contracts](../../../hyper_parallel/rl/docs/architecture.md#边界合同) |
| Algorithms and policy roles | `rl/algorithm/`, `rl/roles/policy/actor.py`, `rl/roles/policy/critic.py` | [Feature navigation](../../../docs/rl-navigation.md) |
| Token batching and DP balancing | `rl/roles/policy/dynamic_batch.py`, `rl/roles/policy/dp_balance.py`; selector in `hyper_parallel/data/batching/packing.py` | [Training contract](../../../hyper_parallel/rl/docs/architecture.md); RL owns optimizer membership, row restoration and DP call counts |
| Token log-probability projection | `rl/roles/policy/token_log_probs.py`, `rl/roles/policy/token_log_probs_adapter.py` | [Runtime architecture](../../../hyper_parallel/rl/docs/architecture.md) |
| Phase offload | `rl/roles/phase_offload.py` | [Residency contract](../../../hyper_parallel/rl/docs/architecture.md#master-模型接入); shared HSDP owns storage bookkeeping |
| MoE router replay | `rl/router_replay.py`, `rl/roles/policy/router_adapter.py`, `rl/roles/rollout/router_capture.py` | [R3 contract](../../../hyper_parallel/rl/docs/vllm_rollout.md#moe-r3-路由重放) |
| Reward Model service | `rl/reward_model/`; task scorers in `examples/` | [RM contract](../../../hyper_parallel/rl/docs/reward_model.md) |
| Teacher distillation | `rl/distillation/`, `rl/algorithm/distillation.py` | [OPD/MOPD](../../../hyper_parallel/rl/examples/distillation/README.md) |
| Qwen3 construction | `hyper_parallel/models/qwen3/adapter/`, `rl/roles/qwen3_builder.py` | [Model boundary](../../../hyper_parallel/rl/docs/architecture.md#master-模型接入); main-project callers use the shared AutoModel |
| vLLM and topology | `rl/roles/rollout/vllm.py`, `rl/roles/rollout/topology.py`, `rl/roles/rollout/worker.py`, `rl/roles/rollout/vllm_plugin.py` | [Rollout contract](../../../hyper_parallel/rl/docs/vllm_rollout.md) |
| Hyper Qwen3 inference adapters | `rl/roles/rollout/consistency_models/qwen3/` | [Consistency](../../../hyper_parallel/rl/docs/qwen3_training_inference_consistency.md) |
| Agent environments and tools | `rl/agentic/core/trajectory_runner.py`, `rl/agentic/core/session.py`, `rl/agentic/envs/environment.py`, `rl/agentic/tools/` | [Agentic RL](../../../hyper_parallel/rl/docs/agentic_rl.md) |
| External programs and CPU preparation | `rl/agentic/core/program_runner.py`, `rl/agentic/core/prepared_batch.py`, `rl/agentic/codex/`, `rl/agentic/ds_harness/`, `rl/agentic/mcp_server.py` | [Agentic RL](../../../hyper_parallel/rl/docs/agentic_rl.md) |
| Shared isolated Codex execution | `rl/agentic/codex/container_execution.py`, `rl/agentic/envs/docker_workspace.py`, `rl/agentic/core/gateway_transport.py` | [Isolation and failure contracts](../../../hyper_parallel/rl/docs/agentic_rl.md); stdlib-only host imports |
| Tool evidence | `rl/tool_protocol.py`, `rl/tool_xml.py`, `rl/tool_diagnostics.py` | [Failure attribution](../../../hyper_parallel/rl/docs/agentic_rl.md#工具失败归因与收尾) |
| GSM8K | `examples/gsm8k/agent.py`, `examples/gsm8k/configs/`, `examples/gsm8k/scripts/` | [Recipes and result scope](../../../hyper_parallel/rl/examples/gsm8k/README.md); strict rule reward, eight-card guard |
| Search-R1 | `examples/search_r1/agent.py`, `examples/search_r1/container_runtime.py`, `examples/search_r1/workspace.py`, `examples/search_r1/reward.py` | [HotpotQA file search](../../../hyper_parallel/rl/docs/search_r1.md); task-specific host handoff and citation checks |
| Single-turn code | `examples/code/` | [Private tests and SandboxFusion](../../../hyper_parallel/rl/examples/code/README.md) |
| Repository Code Agent | `examples/swe_smith/` | [Recipes, images and launch](../../../hyper_parallel/rl/examples/swe_smith/README.md), [implementation and acceptance](../../../hyper_parallel/rl/docs/swe_smith.md) |
| Weight publication and transport | `rl/weight_sync/` | [full_gather/direct_reshard and IPC/HCCL](../../../hyper_parallel/rl/docs/vllm_rollout.md) |
| Consistency gates | `rl/consistency/` | [Qwen3 consistency](../../../hyper_parallel/rl/docs/qwen3_training_inference_consistency.md) |
| Persistence and evaluation | `rl/checkpoint.py`, `rl/evaluation.py` | [Checkpoint and resume](../../../hyper_parallel/rl/docs/architecture.md#checkpoint-与恢复) |
| Metrics, failures and cleanup | `rl/utils/monitoring/`, `rl/utils/failure_reporting.py`, `rl/utils/process_cleanup.py` | [Runtime architecture](../../../hyper_parallel/rl/docs/architecture.md) |
| Runtime installation and patches | `docker/`, `pyproject.toml`; package manifests in root `setup.py` and `MANIFEST.in` | [Runtime image](../../../hyper_parallel/rl/docker/README.md) |
| Tests | `tests/ut/rl/`, `hyper_parallel/rl/tests/st/`, `tests/common/rl_st_cases.py` | [ST procedure](../../../hyper_parallel/rl/README.md#系统测试); CPU contracts and NPU acceptance are separate |

Shared model and distributed changes follow their own main-project rules.
Historical results, external baselines and analysis helpers remain in ignored `output/`; tracked launchers and tests must not depend on them.
