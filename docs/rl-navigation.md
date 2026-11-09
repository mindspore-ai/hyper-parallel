# Hyper-RL Feature Navigation

**English** | [简体中文](rl-navigation.zh.md)

Paths are relative to `hyper_parallel/rl/`; `tests/ut/rl/` is relative to the repository root.

| Feature | Configuration / entry point | Implementation | Validation |
| --- | --- | --- | --- |
| Construction and support matrix | `train_rl.py`, this run's YAML | `rl/config/loader.py`, `rl/config/`, `rl/trainer/build.py` | `tests/ut/rl/trainer/test_config_runtime.py` |
| Explicit replacement rules | `train.plan_overrides`, role overrides | `rl/config/resolve.py`, `rl/workers/adapters/qwen3.py` | `test_qwen3_master.py`, `test_moe_config.py` |
| Synchronous training | Default execution mode | `rl/trainer/sync.py` | `test_trainer_orchestration.py` |
| Disjoint asynchronous training | `async_training` | `rl/trainer/fully_async.py`, `rl/async_runtime/producer.py` | `tests/ut/rl/fully_async/` |
| Colocated asynchronous training | `colocate_async` | `rl/trainer/colocate_async.py`, `rl/async_runtime/colocated.py` | `tests/ut/rl/colocate_async/` |
| GRPO/PPO/GSPO | `algorithm.name` | `rl/algorithm/`, `rl/workers/actor.py`, `critic.py` | `tests/ut/rl/algorithm/`, `policy/` |
| Probabilities, targets, and bootstrap | `old_log_prob_source`, algorithm requirements | `rl/workers/targets.py`, `rl/data/experience.py` | Target ordering, probability identity, and PPO tests |
| Shared optimizer and residency | optimizer, accelerator | `rl/workers/engine/` | Optimizer precision, phase offload, and checkpoint tests |
| Dynamic batching / DP balance | Three token-budget types, `balance_dp_tokens` | `rl/workers/input/` | Batch budget, gradient, and row-order tests |
| Agent and actual trajectories | `agentic`, examples | `rl/agentic/`, `rl/data/` | `tests/ut/rl/agentic/`, `data/` |
| Native-vLLM | `model_implementation: native` | `rl/rollout/vllm.py`, `vllm_plugin.py` | Rollout runtime, plugin, and sampler tests |
| full_gather IPC/HCCL | weight_sync, deployment | `rl/weight_sync/` | Transaction, transport, worker, and MoE mapping tests |
| RM | `reward_model` | `rl/reward_model/runtime.py` | `tests/ut/rl/reward_model/` |
| Teacher / OPD / MOPD | `distillation` | `rl/distillation/`, `rl/algorithm/distillation.py` | `tests/ut/rl/distillation/` |
| MoE R3 | `train.router_replay.mode` | `rl/workers/adapters/router_*`, `rl/rollout/router_capture.py` | Router replay and MoE tests |
| Save and resume | `train.checkpoint` | `rl/trainer/checkpoint.py`, `rl/async_runtime/protocol.py` | Schema, precision, topology, version, RNG, and buffer recovery tests |

Device-test entry points reside in `hyper_parallel/rl/tests/st/`; allocate devices following the [system test instructions](../hyper_parallel/rl/README.md#system-tests) before running them.
See the [runtime architecture](../hyper_parallel/rl/docs/architecture.md) and [native-vLLM contract](../hyper_parallel/rl/docs/vllm_rollout.md).
