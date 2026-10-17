# Hyper-RL Module Map

Use this index to locate ownership; all code paths below are relative to the repository root.
Interface semantics live in product docs. Feature-to-config/code/metric/test traces live in
[feature navigation](../../../docs/rl-navigation.md); project composition lives in
[architecture](../../../docs/rl-architecture.md). Start work with [the RL rule](../hyper-rl.md).

| Area | Code or entry | Detail |
| --- | --- | --- |
| Design and boundaries | `hyper_parallel/rl/rl/config.py`, `hyper_parallel/rl/rl/roles/model_setup.py` | [Design](../../../hyper_parallel/rl/docs/design.md), [supported scope](../../../hyper_parallel/rl/README.md#支持范围) |
| Roadmap | — | [Development directions](../../../hyper_parallel/rl/README.md#规划方向) |
| Config and CLI | `hyper_parallel/rl/rl/config.py`, `hyper_parallel/rl/train_rl.py` | [Runtime architecture](../../../hyper_parallel/rl/docs/architecture.md), [feature navigation](../../../docs/rl-navigation.md) |
| Synchronous orchestration | `hyper_parallel/rl/rl/trainer.py` | [Runtime architecture](../../../hyper_parallel/rl/docs/architecture.md) |
| Persistence and evaluation | `hyper_parallel/rl/rl/checkpoint.py`, `hyper_parallel/rl/rl/evaluation.py` | [Checkpoint and resume](../../../hyper_parallel/rl/docs/architecture.md#checkpoint-与恢复) |
| Prompt and experience data | `hyper_parallel/rl/rl/dataset/data_source.py`, `hyper_parallel/rl/rl/dataset/contracts.py`, `hyper_parallel/rl/rl/dataset/batch_builder.py`, `hyper_parallel/rl/rl/dataset/episodes.py` | [Runtime architecture](../../../hyper_parallel/rl/docs/architecture.md) |
| Colocated reward service | `hyper_parallel/rl/rl/reward_model/`; GSM8K rule and model scorers in `hyper_parallel/rl/examples/gsm8k/agent.py` | [Reward model contract](../../../hyper_parallel/rl/docs/reward_model.md); NPU acceptance pending |
| Algorithms | `hyper_parallel/rl/rl/algorithm/`, `hyper_parallel/rl/rl/registry.py` | [Feature navigation](../../../docs/rl-navigation.md), [training step](../../../hyper_parallel/rl/docs/architecture.md#一个训练步) |
| Actor, Reference, Critic | `hyper_parallel/rl/rl/roles/policy/actor.py`, `hyper_parallel/rl/rl/roles/policy/critic.py` | [Runtime architecture](../../../hyper_parallel/rl/docs/architecture.md) |
| MoE router replay | `hyper_parallel/rl/rl/router_replay.py`, `hyper_parallel/rl/rl/roles/policy/router_adapter.py`, `hyper_parallel/rl/rl/roles/rollout/router_capture.py` | [Replay contract](../../../hyper_parallel/rl/docs/vllm_rollout.md#moe-r3-路由重放), [usage and verification](../../../hyper_parallel/rl/docs/moe_code_agent.md#moe-r3); RL-owned capture/state/instance adaptation, no shared model changes |
| Training model and optimizer factories | `hyper_parallel/rl/rl/roles/model_setup.py` | [Model construction](../../../hyper_parallel/rl/docs/architecture.md#master-模型接入) |
| Process and service cleanup | `hyper_parallel/rl/rl/utils/process_cleanup.py` | [Runtime architecture](../../../hyper_parallel/rl/docs/architecture.md) |
| Qwen3 adapters and RL construction compatibility | `hyper_parallel/models/qwen3/adapter/`, `hyper_parallel/rl/rl/roles/qwen3_builder.py` | [Model construction](../../../hyper_parallel/rl/docs/architecture.md#master-模型接入); main-project callers use the shared AutoModel builder |
| Training-inference consistency model adapters | `hyper_parallel/rl/rl/roles/rollout/consistency_models/qwen3/`, `hyper_parallel/rl/rl/roles/rollout/vllm_plugin.py` | [vLLM rollout](../../../hyper_parallel/rl/docs/vllm_rollout.md) |
| Generation and topology | `hyper_parallel/rl/rl/roles/rollout/vllm.py`, `hyper_parallel/rl/rl/roles/rollout/topology.py`, `hyper_parallel/rl/rl/roles/rollout/worker.py` | [vLLM rollout](../../../hyper_parallel/rl/docs/vllm_rollout.md) |
| Agentic contracts and environment | `hyper_parallel/rl/rl/agentic/core/trajectory_runner.py`, `hyper_parallel/rl/rl/agentic/core/`, `hyper_parallel/rl/rl/agentic/envs/environment.py`, `hyper_parallel/rl/rl/agentic/tools/` | [Agentic RL](../../../hyper_parallel/rl/docs/agentic_rl.md) |
| External Agent programs and MCP | `hyper_parallel/rl/rl/agentic/codex/`, `hyper_parallel/rl/rl/agentic/ds_harness/`, `hyper_parallel/rl/rl/agentic/core/program_runner.py`, `hyper_parallel/rl/rl/agentic/mcp_server.py` | [Agentic RL](../../../hyper_parallel/rl/docs/agentic_rl.md); DeepSeek here names the harness, not a supported model family |
| Tool evidence and trainability | `hyper_parallel/rl/rl/tool_protocol.py`, `hyper_parallel/rl/rl/roles/rollout/vllm_plugin.py` | [Agentic RL](../../../hyper_parallel/rl/docs/agentic_rl.md#工具失败归因与收尾) |
| GSM8K examples | `hyper_parallel/rl/examples/gsm8k/`, `hyper_parallel/rl/examples/gsm8k/configs/` | [Shared launchers and recipes](../../../hyper_parallel/rl/examples/gsm8k/README.md), [Agentic RL](../../../hyper_parallel/rl/docs/agentic_rl.md) |
| Search-R1 HotpotQA | `hyper_parallel/rl/examples/search_r1/`, `hyper_parallel/rl/examples/search_r1/configs/qwen3_4b_search_r1.yaml` | [Search-R1 procedure](../../../hyper_parallel/rl/docs/search_r1.md); isolated article search, citation audit, and training acceptance |
| Single-turn code tasks | `hyper_parallel/rl/examples/code/`, `hyper_parallel/rl/examples/code/configs/`, `hyper_parallel/rl/examples/code/scripts/` | [Code example](../../../hyper_parallel/rl/examples/code/README.md); private stdio tests, remote SandboxFusion judging, reviewed data preparation, and Docker training launcher |
| Repository Code Agent | `hyper_parallel/rl/examples/swe_smith/`, `hyper_parallel/rl/rl/agentic/envs/docker_workspace.py`, `hyper_parallel/rl/examples/swe_smith/container_execution.py` | [Codex repository contract and recipes](../../../hyper_parallel/rl/examples/swe_smith/README.md), [validation procedure and scope](../../../hyper_parallel/rl/examples/swe_smith/README.md#图执行与验证范围); program_callable entry, independent SWE-Smith grading and per-call GRPO; CPU contracts in `tests/ut/rl/agentic/swe_smith/` |
| SWE-Smith implementation and limits | `hyper_parallel/rl/examples/swe_smith/agent.py`, `hyper_parallel/rl/examples/swe_smith/prepare_images.py`, `hyper_parallel/rl/examples/swe_smith/infer.py` | [Development scope and acceptance](../../../hyper_parallel/rl/docs/swe_smith.md); local ARM64 and remote amd64 short eager RL, remote reference-budget inference; cross-training-DP groups and long-budget learning remain unverified |
| SWE-Smith local/remote launch | `hyper_parallel/rl/examples/swe_smith/launch.py`, `hyper_parallel/rl/examples/swe_smith/run_config.py` | [Unified host procedure](../../../hyper_parallel/rl/examples/swe_smith/README.md#本地与远端统一运行入口); explicit local socket or SSH sandbox, prepared data/image fingerprints, short inference and checkpoint resume; CPU contracts in `tests/ut/rl/agentic/swe_smith/` |
| SWE-Smith published ARM64 images | `hyper_parallel/rl/examples/swe_smith/configs/images_arm64.json`, `hyper_parallel/rl/examples/swe_smith/configs/selected_ids.txt` | [Local nine-instance downloads](../../../hyper_parallel/rl/examples/swe_smith/README.md#本地-arm64-九例镜像); SWR image tags, fixed image IDs and registry digests, six training and three validation instances |
| Qwen3.5-family text RL | `hyper_parallel/models/qwen3_5/adapter/selected_log_probs.py`, `hyper_parallel/rl/rl/roles/model_setup.py`, `hyper_parallel/rl/rl/weight_sync/` | [Model boundaries](../../../hyper_parallel/rl/examples/swe_smith/README.md#模型配方); Qwen3.5-9B and Qwen3.8-27B share public text loading and complete native text publication |
| Publication lifecycle | `hyper_parallel/rl/rl/weight_sync/config.py`, `hyper_parallel/rl/rl/weight_sync/sync.py`, `hyper_parallel/rl/rl/weight_sync/transfer.py` | [vLLM rollout](../../../hyper_parallel/rl/docs/vllm_rollout.md) |
| Layout, direct planning, packing and transport | `hyper_parallel/rl/rl/weight_sync/layout.py`, `hyper_parallel/rl/rl/weight_sync/model_adapter.py`, `hyper_parallel/rl/rl/weight_sync/packed_weight.py`, `hyper_parallel/rl/rl/weight_sync/ipc.py`, `hyper_parallel/rl/rl/weight_sync/hccl.py` | [vLLM rollout](../../../hyper_parallel/rl/docs/vllm_rollout.md) |
| vLLM update endpoints | `hyper_parallel/rl/rl/weight_sync/vllm_client.py`, `hyper_parallel/rl/rl/weight_sync/vllm_worker.py` | [vLLM rollout](../../../hyper_parallel/rl/docs/vllm_rollout.md) |
| Consistency | `hyper_parallel/rl/rl/consistency/` | [Qwen3 consistency](../../../hyper_parallel/rl/docs/qwen3_training_inference_consistency.md) |
| Metrics and logging | `hyper_parallel/rl/rl/utils/monitoring/` | [Feature navigation](../../../docs/rl-navigation.md), [runtime architecture](../../../hyper_parallel/rl/docs/architecture.md) |
| Runtime installation | `hyper_parallel/rl/docker/README.md`, `hyper_parallel/rl/docker/install_runtime.sh`, `hyper_parallel/rl/docker/patches/`, `hyper_parallel/rl/pyproject.toml` | [Runtime image](../../../hyper_parallel/rl/docker/README.md) |
| UT | `tests/ut/rl/` | [Feature contracts and validation](../../../hyper_parallel/rl/docs/moe_code_agent.md#单元测试) |
| ST and shared recipes | `hyper_parallel/rl/tests/st/test_rl_st.py`, `hyper_parallel/rl/tests/st/test_feature_st.py`, `tests/common/rl_st_cases.py` | [Standalone ST](../../../hyper_parallel/rl/README.md#系统测试); temporarily outside the main-project PR gate |

The runtime supports Qwen3 dense GRPO/PPO/GSPO and Qwen3-MoE GRPO under the
[documented rollout limits](../../../hyper_parallel/rl/docs/vllm_rollout.md).
MoE construction uses the shared `hyper_parallel/models/qwen3_moe/` recipe; RL weight conversion remains under
`hyper_parallel/rl/rl/weight_sync/`. Shared-project changes follow their own module rules.
Packaging changes belong to the root `setup.py` and `MANIFEST.in`.
