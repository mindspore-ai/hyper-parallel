# Hyper-RL Module Map

Use this index to locate ownership; all code paths below are relative to the repository root.
Interface semantics live in product docs. Feature-to-config/code/metric/test traces live in
[feature navigation](../../../docs/rl-navigation.md); project composition lives in
[architecture](../../../docs/rl-architecture.md). Start work with [the RL rule](../hyper-rl.md).

| Area | Code or entry | Detail |
| --- | --- | --- |
| Design and boundaries | `hyper_parallel/rl/rl/config.py`, `hyper_parallel/rl/rl/roles/model_setup.py` | [Design](../../../hyper_parallel/rl/docs/design.md), [feature inventory](../../../hyper_parallel/rl/docs/current_feature_inventory.md) |
| Roadmap | — | [Milestones and acceptance](../../../hyper_parallel/rl/docs/TODO.md) |
| Config and CLI | `hyper_parallel/rl/rl/config.py`, `hyper_parallel/rl/train_rl.py` | [Runtime architecture](../../../hyper_parallel/rl/docs/architecture.md), [feature navigation](../../../docs/rl-navigation.md) |
| Synchronous orchestration | `hyper_parallel/rl/rl/trainer.py` | [Runtime architecture](../../../hyper_parallel/rl/docs/architecture.md) |
| Persistence and evaluation | `hyper_parallel/rl/rl/checkpoint.py`, `hyper_parallel/rl/rl/evaluation.py` | [PPO and resume](../../../hyper_parallel/rl/docs/ppo.md), [runtime architecture](../../../hyper_parallel/rl/docs/architecture.md) |
| Prompt and experience data | `hyper_parallel/rl/rl/dataset/data_source.py`, `hyper_parallel/rl/rl/dataset/contracts.py`, `hyper_parallel/rl/rl/dataset/batch_builder.py`, `hyper_parallel/rl/rl/dataset/episodes.py` | [Runtime architecture](../../../hyper_parallel/rl/docs/architecture.md) |
| Algorithms and rewards | `hyper_parallel/rl/rl/algorithm/`, `hyper_parallel/rl/rl/registry.py` | [Feature navigation](../../../docs/rl-navigation.md), [PPO](../../../hyper_parallel/rl/docs/ppo.md) |
| Actor, Reference, Critic | `hyper_parallel/rl/rl/roles/policy/actor.py`, `hyper_parallel/rl/rl/roles/policy/critic.py` | [PPO](../../../hyper_parallel/rl/docs/ppo.md), [runtime architecture](../../../hyper_parallel/rl/docs/architecture.md) |
| Training model and optimizer factories | `hyper_parallel/rl/rl/roles/model_setup.py` | [Qwen3 master adaptation](../../../hyper_parallel/rl/docs/qwen3_master_adaptation.md) |
| Process and service cleanup | `hyper_parallel/rl/rl/process_cleanup.py` | [Runtime architecture](../../../hyper_parallel/rl/docs/architecture.md) |
| Qwen3 adapters and RL construction compatibility | `hyper_parallel/models/qwen3/adapter/`, `hyper_parallel/rl/rl/roles/qwen3_builder.py` | [Qwen3 master adaptation](../../../hyper_parallel/rl/docs/qwen3_master_adaptation.md); main-project callers use the shared AutoModel builder |
| Training-inference consistency model adapters | `hyper_parallel/rl/rl/roles/rollout/consistency_models/qwen3/`, `hyper_parallel/rl/rl/roles/rollout/vllm_plugin.py` | [vLLM rollout](../../../hyper_parallel/rl/docs/vllm_rollout.md) |
| Generation and topology | `hyper_parallel/rl/rl/roles/rollout/vllm.py`, `hyper_parallel/rl/rl/roles/rollout/topology.py`, `hyper_parallel/rl/rl/roles/rollout/worker.py` | [vLLM rollout](../../../hyper_parallel/rl/docs/vllm_rollout.md) |
| Agentic contracts and environment | `hyper_parallel/rl/rl/agentic/core/`, `hyper_parallel/rl/rl/agentic/envs/environment.py`, `hyper_parallel/rl/rl/agentic/tools/` | [Agentic RL](../../../hyper_parallel/rl/docs/agentic_rl.md) |
| External Agent programs and MCP | `hyper_parallel/rl/rl/agentic/codex/`, `hyper_parallel/rl/rl/agentic/ds_harness/`, `hyper_parallel/rl/rl/agentic/core/program_runner.py`, `hyper_parallel/rl/rl/agentic/mcp_server.py` | [Agentic RL](../../../hyper_parallel/rl/docs/agentic_rl.md); DeepSeek here names the harness, not a supported model family |
| Tool evidence and trainability | `hyper_parallel/rl/rl/tool_protocol.py`, `hyper_parallel/rl/rl/roles/rollout/vllm_plugin.py` | [Agentic RL](../../../hyper_parallel/rl/docs/agentic_rl.md#工具失败归因与收尾) |
| GSM8K examples | `hyper_parallel/rl/examples/gsm8k/`, `hyper_parallel/rl/examples/gsm8k/configs/` | [RL README](../../../hyper_parallel/rl/README.md), [Agentic RL](../../../hyper_parallel/rl/docs/agentic_rl.md) |
| Single-turn code tasks | `hyper_parallel/rl/examples/code/`, `hyper_parallel/rl/examples/code/configs/` | [Code example](../../../hyper_parallel/rl/examples/code/README.md); private stdio tests, remote SandboxFusion judging, and reviewed data preparation |
| Publication lifecycle | `hyper_parallel/rl/rl/roles/weight_sync/config.py`, `hyper_parallel/rl/rl/roles/weight_sync/sync.py`, `hyper_parallel/rl/rl/roles/weight_sync/transfer.py` | [vLLM rollout](../../../hyper_parallel/rl/docs/vllm_rollout.md) |
| Layout, packing and transport | `hyper_parallel/rl/rl/roles/weight_sync/layout.py`, `hyper_parallel/rl/rl/roles/weight_sync/model_adapter.py`, `hyper_parallel/rl/rl/roles/weight_sync/packed_weight.py`, `hyper_parallel/rl/rl/roles/weight_sync/ipc.py`, `hyper_parallel/rl/rl/roles/weight_sync/hccl.py` | [vLLM rollout](../../../hyper_parallel/rl/docs/vllm_rollout.md) |
| vLLM update endpoints | `hyper_parallel/rl/rl/roles/weight_sync/vllm_client.py`, `hyper_parallel/rl/rl/roles/weight_sync/vllm_worker.py` | [vLLM rollout](../../../hyper_parallel/rl/docs/vllm_rollout.md) |
| Consistency | `hyper_parallel/rl/rl/consistency/` | [Qwen3 consistency](../../../hyper_parallel/rl/docs/qwen3_training_inference_consistency.md) |
| Metrics and logging | `hyper_parallel/rl/rl/utils/monitoring/` | [Feature navigation](../../../docs/rl-navigation.md), [runtime architecture](../../../hyper_parallel/rl/docs/architecture.md) |
| Runtime installation | `hyper_parallel/rl/docker/README.md`, `hyper_parallel/rl/docker/install_runtime.sh`, `hyper_parallel/rl/docker/patches/`, `hyper_parallel/rl/pyproject.toml` | [Runtime image](../../../hyper_parallel/rl/docker/README.md) |
| UT | `tests/ut/rl/` | [Feature contracts and validation](../../../hyper_parallel/rl/docs/moe_code_agent.md#单元测试) |
| ST and shared recipes | `hyper_parallel/rl/tests/st/test_rl_st.py`, `hyper_parallel/rl/tests/st/test_feature_st.py`, `tests/common/rl_st_cases.py` | [Standalone ST](../../../hyper_parallel/rl/README.md#系统测试); temporarily outside the main-project PR gate |

The runtime supports Qwen3 dense GRPO/PPO and Qwen3-MoE GRPO under the
[documented rollout limits](../../../hyper_parallel/rl/docs/vllm_rollout.md).
MoE construction uses the shared `hyper_parallel/models/qwen3_moe/` recipe; RL weight conversion remains under
`hyper_parallel/rl/rl/roles/weight_sync/`. Shared-project changes follow their own module rules.
Packaging changes belong to the root `setup.py` and `MANIFEST.in`.
