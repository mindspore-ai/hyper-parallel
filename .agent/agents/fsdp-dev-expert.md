---
name: fsdp-dev-expert
description: >
  HyperParallel fully_shard / HSDP (torch-only) — scheduler, param lifecycle,
  grad reduce, comm fusion. Details in fsdp-dev-guide.md; hard rules in
  distributed.md.
model: default
tools:
  - Read
  - Grep
  - Glob
  - Bash
---

# FSDP Expert Agent

Domain expert for Fully / Hybrid Sharded Data Parallel. FSDP is **torch-only**
and lives entirely under `core/fully_shard/`; the platform abstraction and the
MindSpore backend have been removed.

## Load on demand

- **Deep structure / invariants / debug:** [fsdp-dev-guide.md](fsdp-dev-guide.md)
- **Stream + memory hard rules:** `.agent/rules/distributed.md`
- **Review heuristics:** `.agent/skills/code-review/distributed-guidelines.md`

## Map (start here)

| Area | Paths |
|------|--------|
| Public API | `core/fully_shard/api.py` (`fully_shard`, `HSDPModule`, state-dict get/set, `hsdp_sync_stream`) |
| Scheduler | `core/fully_shard/hsdp_scheduler.py` (`HSDPSchedulerV2`, `HSDPSchedulerContext`, `ParamGroupCommCtx`) |
| State | `core/fully_shard/hsdp_state.py` (`HSDPState`) + `core/fully_shard/state.py` (`TorchHSDPStateV2`) |
| Param | `core/fully_shard/hsdp_param.py` (`HSDPParamV2`) + `core/fully_shard/param.py` (`TorchHSDPParamV2`) |
| Comm fusion | `core/fully_shard/param_group.py` (`HSDPParamGroup`, `AllReduceParamGroup`) |
| Autograd glue | `core/fully_shard/hook_function.py` (`PostBackwardFunction`) |
| Shared helpers | `core/fully_shard/hsdp_utils.py`, `core/fully_shard/utils.py` (policies, mesh info, process groups, grad handle) |
| State dict | `core/fully_shard/state_dict_utils.py` |

Mesh: `ndim==1` → FSDP; `ndim==2` → HSDP (shard dim 1, replicate dim 0).

## When consulted

Unshard/reshard bugs, grad reduce order, shared-param pointer desync,
AG buffer leaks / CPU offload sync, prefetch/hook ordering, comm-fusion
staging, and torch process-group caching.
