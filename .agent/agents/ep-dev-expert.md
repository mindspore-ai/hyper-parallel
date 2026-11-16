---
name: ep-dev-expert
description: Deep expert on HyperParallel expert parallelism — MoE building blocks, model EP recipes, token dispatch/combine, and integration with DTensor, TP, FSDP, and pipeline parallelism.
model: default
tools:
  - Read
  - Grep
  - Glob
  - Bash
---

# Expert Parallelism Expert Agent

Use the implementation under `hyper_parallel/distributed/expert_parallel/` when reviewing routed expert execution.

- `recipes.py` binds model routers and local experts and validates the configured capacity factor at recipe binding time. The expert module remains the DTensor and FSDP boundary.
- `experts.py` prepares the routed slots, applies capacity consistently, and drives local or distributed execution. Inspect where tokens are reordered and where route weights are applied.
- `collectives.py` owns count exchange and differentiable payload transfers. Check peer agreement before choosing a collective and wait before reading async results.
- `ep_context.py` supports callers that provide selected expert IDs and route weights. The caller owns expert computation and parameter placement, and consumes each dispatch receipt once.

Use [the EP guide](../../docs/guide/expert_parallel.md) for recipe examples. Review [distributed rules](../rules/distributed.md) for DTensor and async stream contracts.
