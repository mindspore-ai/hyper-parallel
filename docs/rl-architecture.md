# HyperParallel-RL Architecture and Main-Project Boundaries

**English** | [简体中文](rl-architecture.zh.md)

The main project owns models, parallelism, optimizers, and DCP; RL owns algorithms, trajectories, rewards, and training scheduling.
This refactor retains the separate `hyper-parallel-rl` distribution and top-level `rl` import without changing main-project implementations.

`train_rl.main → trainer/build.py` is the assembly entry point.
Synchronous, disjoint asynchronous, and colocated asynchronous loops reside in `trainer/`, each with explicit component dependencies rather than inheriting a generic role runtime.
Configuration is split into schema, resolve, limits, and validate; model replacement rules are declared directly in this run's RL YAML.
Data contracts reside in `data/`, algorithms and roles in `algorithm/` and `workers/`, generation and weight publication in `rollout/`, and CPU asynchronous protocols in `async_runtime/`.

Only native-vLLM and streaming full_gather remain, using IPC/HCCL according to deployment.
The old consistency machinery, Hyper inference models, and direct_reshard have been removed. New checkpoints explicitly use schema 2 and are incompatible with historical snapshots.
GRPO/PPO/GSPO, MoE, RM, Teacher, Agent, and both asynchronous modes retain their existing support matrices; directory migration does not expand support.

See the [runtime architecture](../hyper_parallel/rl/docs/architecture.md) for detailed ordering and state contracts,
[feature navigation](rl-navigation.md) for configuration and validation entry points, and [RL rules](../.agent/rules/hyper-rl.md) for development boundaries.
Record CPU contract validation, short device flows, and learning outcomes separately.

## Unified Training and Inference

The optional [unified execution runtime](../hyper_parallel/rl/rl/unified_execution/README.md)
reuses the HP model builder and complete `train.plan_overrides` for Qwen3 Dense and MoE.
HP owns TP/SP and expert sharding; vLLM owns scheduling, paged KV cache and graph execution.
Shared V4 attention, Level 3, [row-parallel PIK and ordered FP32 expert merging](../hyper_parallel/components/parallelism_invariant/) define the numerical path.
Training MoE reuses HP dynamic EP and its autograd. Rollout uses native vLLM-Ascend routing and expert GEMMs. Each expert slot returns to its source for weighted accumulation in router order. The model registers each layer by its stable path for graph replay. Training CP reuses HP sequence sharding and differentiable K/V gather, with RL-owned target alignment and TP×CP request sharing. Inference PCP reuses native Ascend MRV2 scheduling and KV caches, derives HP EP groups from the full DP×PCP×TP domain according to the EP degree, and publishes to all DP×PCP replicas. This retains native scheduler ownership, expert weight layout and publication protocol. PCP1 retains its prior execution path; DCP, PP and expert-internal PIK remain separate work.
