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
