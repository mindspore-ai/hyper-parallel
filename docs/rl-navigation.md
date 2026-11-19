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
| Qwen3 共享架构 | `consistency: {schema_version: 1, mode: architecture}` | `rl/unified_execution/`；共用公开 builder、`train.plan_overrides` 和 TP/SP | `tests/ut/rl/unified_execution/test_contract.py`、`examples/consistency/scripts/run_qwen3_moe_docker.sh` |
| Qwen3-MoE 共享架构 | `examples/consistency/configs/qwen3_moe_4l_rl.yaml` | `rl/unified_execution/models/` 描述模型，复用公共 builder、HP TP/SP/EP 和 GroupedExperts；训练动态 EP，推理 MC2/AllGather 有效行 GMM 支持图捕获 | `tests/ut/rl/unified_execution/test_contract.py`、`tests/ut/rl/unified_execution/test_weights.py`、`examples/consistency/scripts/run_qwen3_moe_docker.sh` |
| Prefill 分段图 | `compilation_config.mode: 3`、`cudagraph_mode: FULL_AND_PIECEWISE` | `rl/unified_execution/runtimes/vllm/entrypoints.py`；HP forward 进入 vLLM 分段编译 | `examples/consistency/scripts/run_qwen3_moe_docker.sh` |
| Level 3 batch invariant | `consistency.mode: batch_invariant`；provider=`torch_npu_level3`，fill=false | `rl/unified_execution/core/batch_invariant.py`；Trainer 与推理 worker 独立初始化和核验 | `tests/ut/rl/unified_execution/test_batch_invariant.py`、`tests/ut/rl/rollout/test_unified_runtime.py`、两步 RL 示例 |
| 独立 PIK | `PikPlan` / `prepare_pik` / `pik_linear`；不自动替换模型 | `rl/unified_execution/ops/pik/`；Level 3 BMM、Triton 固定树、HCCL 搬运 | `tests/ut/rl/unified_execution/test_pik.py`、`tests/st/test_pik.py`；详见 Unified Execution README |
| Dense Qwen3 PIK | `tests/ut/rl/unified_execution/fixtures/qwen3_dense_pik.yaml`；两端共用 plan_overrides | `unified_execution/models/bindings.py`、`execution_plan.py`；HP 参数切分与 PIK 固定规约/SP 输出 | `tests/ut/rl/unified_execution/test_pik.py`、`examples/consistency/scripts/run_qwen3_moe_docker.sh`；详见 Unified Execution README |
| Qwen3-MoE 普通行 PIK | `examples/consistency/configs/qwen3_moe_4l_rl.yaml` | 复用 `models/bindings.py` / `execution_plan.py`；仅 attention o_proj，专家内部无 PIK/ETP | `tests/ut/rl/unified_execution/test_pik.py`、`tests/st/test_pik.py`、两步 RL 示例；详见 Unified Execution README |
| 替换组合的梯度验收 | 同前向原生 VJP、局部参考与真实更新；CP 另有逐参数对照，PP/ETP 未接入 | `models/bindings.py` 保留非 SP 的原生 dY SUM；独立 PIK 合同不变 | `tests/st/test_pik.py`、`tests/st/test_unified_attention.py`、`tests/st/test_unified_expert_parallel.py` |
| 共享 V4 attention | `consistency.attention_backend: flash_attention_npu_v4`；`native` 保留阶段 1A 路径 | `rl/unified_execution/ops/attention/flash_attention_npu_v4.py` 数值调用；`rl/unified_execution/runtimes/hyper_parallel/attention.py` mask/梯度接入；`rl/unified_execution/runtimes/vllm/attention/ascend_v4.py` paged KV/图元数据 | `tests/ut/rl/unified_execution/test_attention.py`、`tests/ut/rl/unified_execution/test_attention_backend.py`、`tests/st/test_unified_attention.py` |

Device-test entry points reside in `hyper_parallel/rl/tests/st/`; allocate devices following the [system test instructions](../hyper_parallel/rl/README.md#system-tests) before running them.
See the [runtime architecture](../hyper_parallel/rl/docs/architecture.md) and [native-vLLM contract](../hyper_parallel/rl/docs/vllm_rollout.md).

Unified execution, supported topologies and reproducible experiments are documented in the
[unified execution guide](../hyper_parallel/rl/docs/unified_execution_guide.md) and
[MoE implementation](../hyper_parallel/rl/docs/unified_execution_moe.md).

Training CP consistency is described in [Unified Execution](../hyper_parallel/rl/docs/unified_execution_guide.md#训练-attention-与-cp).
Its explicit recipe is [qwen3_moe_4l_cp_rl.yaml](../hyper_parallel/rl/examples/consistency/configs/qwen3_moe_4l_cp_rl.yaml),
and its CPU/device acceptance lives in
[CP contracts](../tests/ut/rl/unified_execution/test_context_parallel.py) and
[CP system tests](../hyper_parallel/rl/tests/st/test_context_parallel.py).

Inference PCP is configured independently through `rollout.vllm.context_parallel_size`;
see the [PCP contract](../hyper_parallel/rl/docs/unified_execution_guide.md#推理-attention-与-pcp),
[explicit recipe](../hyper_parallel/rl/examples/consistency/configs/qwen3_moe_4l_cp_rl.yaml),
[CPU contracts](../tests/ut/rl/unified_execution/test_prefill_context_parallel.py) and
[device tests](../hyper_parallel/rl/tests/st/test_prefill_context_parallel.py).

HP EP groups are derived from the full DP×PCP×TP domain using `rollout.vllm.expert_parallel_size`;
see the [cross-PCP EP recipe](../hyper_parallel/rl/examples/consistency/configs/qwen3_moe_4l_cp_rl.yaml)
and [domain and acceptance design](../hyper_parallel/rl/docs/unified_execution_moe.md#专家域与权重所有权).
The same PCP device launcher checks expert groups and every owned weight element over multiple publications.
For the implemented role-specific EP paths, see the
[MoE numerical and transport contract](../hyper_parallel/rl/docs/unified_execution_moe.md).

## Unified MoE EP transports

- Recipe: `rl.unified_execution.runtimes.hyper_parallel.moe.unified_ep_compute_fn`, with MC2 fast-path `mc2_capacity` and
  `inference_backend` options in `train.plan_overrides`.
- Training: [training_ep.py](../hyper_parallel/rl/rl/unified_execution/ops/moe/training_ep.py) reuses HP dispatch
  and autograd; original FP32 router weights enter the ordered slot merge.
- Inference: [mc2_ep.py](../hyper_parallel/rl/rl/unified_execution/ops/moe/mc2_ep.py) and
  [allgather_ep.py](../hyper_parallel/rl/rl/unified_execution/ops/moe/allgather_ep.py) provide graph paths; [expert_parallel.py](../hyper_parallel/rl/rl/unified_execution/runtimes/vllm/parallel.py)
  checks capability and scheduler/graph bounds, and owns a separate MC2 communicator with HP rank order.
  [moe.py](../hyper_parallel/rl/rl/unified_execution/runtimes/vllm/moe.py) selects the
  per-batch transport from native host metadata; legal larger batches use AllGather.
- Merge: [combine.py](../hyper_parallel/rl/rl/unified_execution/ops/moe/combine.py) preserves owner selection
  and FP32 slot order; [CPU checks](../tests/ut/rl/unified_execution/test_moe_combine.py) cover order and gradients.
- Checks: [CPU role contract](../tests/ut/rl/unified_execution/test_unified_ep.py),
  [mc2_capacity and compiled dispatch](../tests/ut/rl/unified_execution/test_expert_execution.py),
  [device transport acceptance](../hyper_parallel/rl/tests/st/test_unified_expert_parallel.py), and
  [two-step publication loop](../hyper_parallel/rl/tests/st/test_prefill_context_parallel.py).
- Boundaries: [MoE execution contract](../hyper_parallel/rl/docs/unified_execution_moe.md#数值契约).
