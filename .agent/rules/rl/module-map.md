# Hyper-RL Module Map

Use this index to locate ownership; all code paths below are relative to the repository root.
Interface semantics live in product docs. Feature-to-config/code/metric/test traces live in
[feature navigation](../../../docs/rl-navigation.md); project composition lives in
[architecture](../../../docs/rl-architecture.md). Start work with [the RL rule](../hyper-rl.md).

| Area | Code or entry | Detail |
| --- | --- | --- |
| Design and boundaries | `hyper_parallel/rl/rl/config.py`, `hyper_parallel/rl/rl/roles/model.py` | [Design](../../../hyper_parallel/rl/docs/design.md), [feature inventory](../../../hyper_parallel/rl/docs/current_feature_inventory.md) |
| Roadmap | — | [Milestones and acceptance](../../../hyper_parallel/rl/docs/TODO.md) |
| Config and CLI | `hyper_parallel/rl/rl/config.py`, `hyper_parallel/rl/examples/train_rl.py` | [Runtime architecture](../../../hyper_parallel/rl/docs/architecture.md), [feature navigation](../../../docs/rl-navigation.md) |
| Synchronous orchestration | `hyper_parallel/rl/rl/trainer.py` | [Runtime architecture](../../../hyper_parallel/rl/docs/architecture.md) |
| Persistence and evaluation | `hyper_parallel/rl/rl/checkpoint.py`, `hyper_parallel/rl/rl/evaluation.py` | [PPO and resume](../../../hyper_parallel/rl/docs/ppo.md), [runtime architecture](../../../hyper_parallel/rl/docs/architecture.md) |
| Prompt and experience data | `hyper_parallel/rl/rl/dataset/data_source.py`, `hyper_parallel/rl/rl/dataset/contracts.py`, `hyper_parallel/rl/rl/dataset/batch_builder.py` | [Runtime architecture](../../../hyper_parallel/rl/docs/architecture.md) |
| Algorithms and rewards | `hyper_parallel/rl/rl/algorithm/`, `hyper_parallel/rl/rl/registry.py` | [Feature navigation](../../../docs/rl-navigation.md), [PPO](../../../hyper_parallel/rl/docs/ppo.md) |
| Actor, Reference, Critic | `hyper_parallel/rl/rl/roles/policy/actor.py`, `hyper_parallel/rl/rl/roles/policy/critic.py` | [PPO](../../../hyper_parallel/rl/docs/ppo.md), [runtime architecture](../../../hyper_parallel/rl/docs/architecture.md) |
| Training model and optimizer factories | `hyper_parallel/rl/rl/roles/model.py`, `hyper_parallel/rl/rl/distributed.py` | [Qwen3 master adaptation](../../../hyper_parallel/rl/docs/qwen3_master_adaptation.md) |
| Qwen3 loading and parallel adaptation | `hyper_parallel/models/qwen3/runtime.py`, `hyper_parallel/models/qwen3/adapter/` | [Qwen3 master adaptation](../../../hyper_parallel/rl/docs/qwen3_master_adaptation.md); main-project rules also apply |
| Rollout model adapter | `hyper_parallel/rl/rl/roles/rollout/models/qwen3/`, `hyper_parallel/rl/rl/roles/rollout/vllm_plugin.py` | [vLLM rollout](../../../hyper_parallel/rl/docs/vllm_rollout.md) |
| Generation and topology | `hyper_parallel/rl/rl/roles/rollout/vllm.py`, `hyper_parallel/rl/rl/roles/rollout/topology.py`, `hyper_parallel/rl/rl/roles/rollout/worker.py` | [vLLM rollout](../../../hyper_parallel/rl/docs/vllm_rollout.md) |
| Agentic contracts and environment | `hyper_parallel/rl/rl/agentic/core/`, `hyper_parallel/rl/rl/agentic/envs/environment.py`, `hyper_parallel/rl/rl/agentic/tools/` | [Agentic RL](../../../hyper_parallel/rl/docs/agentic_rl.md) |
| External Agent programs and MCP | `hyper_parallel/rl/rl/agentic/codex/`, `hyper_parallel/rl/rl/agentic/deepseek/`, `hyper_parallel/rl/rl/agentic/core/program_runner.py`, `hyper_parallel/rl/rl/agentic/mcp_server.py` | [Agentic RL](../../../hyper_parallel/rl/docs/agentic_rl.md); DeepSeek here names the harness, not a supported model family |
| GSM8K examples | `hyper_parallel/rl/examples/agents/gsm8k/`, `hyper_parallel/rl/examples/configs/` | [RL README](../../../hyper_parallel/rl/README.md), [Agentic RL](../../../hyper_parallel/rl/docs/agentic_rl.md) |
| Publication lifecycle | `hyper_parallel/rl/rl/roles/weight_sync/config.py`, `hyper_parallel/rl/rl/roles/weight_sync/sync.py`, `hyper_parallel/rl/rl/roles/weight_sync/transfer.py` | [vLLM rollout](../../../hyper_parallel/rl/docs/vllm_rollout.md) |
| Layout, packing and transport | `hyper_parallel/rl/rl/roles/weight_sync/layout.py`, `hyper_parallel/rl/rl/roles/weight_sync/model_adapter.py`, `hyper_parallel/rl/rl/roles/weight_sync/packed_weight.py`, `hyper_parallel/rl/rl/roles/weight_sync/ipc.py`, `hyper_parallel/rl/rl/roles/weight_sync/hccl.py` | [vLLM rollout](../../../hyper_parallel/rl/docs/vllm_rollout.md) |
| vLLM update endpoints | `hyper_parallel/rl/rl/roles/weight_sync/vllm_client.py`, `hyper_parallel/rl/rl/roles/weight_sync/vllm_worker.py` | [vLLM rollout](../../../hyper_parallel/rl/docs/vllm_rollout.md) |
| Consistency | `hyper_parallel/rl/rl/consistency/` | [Qwen3 consistency](../../../hyper_parallel/rl/docs/qwen3_training_inference_consistency.md) |
| Metrics and logging | `hyper_parallel/rl/rl/utils/monitoring/` | [Feature navigation](../../../docs/rl-navigation.md), [runtime architecture](../../../hyper_parallel/rl/docs/architecture.md) |
| Runtime installation | `hyper_parallel/rl/docker/`, `hyper_parallel/rl/examples/scripts/`, `hyper_parallel/rl/pyproject.toml` | [Runtime image](../../../hyper_parallel/rl/docs/hyper_rl_runtime_image.md) |
| UT | `tests/ut/rl/` | [UT guide](../../../hyper_parallel/rl/docs/hyper_rl_ut.md) |
| ST and shared recipes | `tests/torch/rl/`, `tests/common/rl_st_cases.py` | [ST guide](../../../hyper_parallel/rl/docs/hyper-rl-st.md) |

The current RL implementation registers Qwen3 dense and GRPO/PPO. Missing MoE adapters are not alternate ownership
locations. For a shared-project change, review its actual diff and the applicable module rules instead of relying on a
historical public-module change list. The [Qwen3 adaptation document](../../../hyper_parallel/rl/docs/qwen3_master_adaptation.md)
explains the model-specific integration; packaging changes belong to the root `setup.py` and `MANIFEST.in`.
