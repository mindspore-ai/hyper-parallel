---
name: dynamic-static-unification
description: Rules for dynamic/static graph unified code (one codebase serving both eager and graph capture)
paths:
  - hyper_parallel/compile/**
  - hyper_parallel/components/**
---

# Dynamic-Static Unification


## Scope And Principles

- Prefer a single code path serving both eager and graph mode (FX trace / graph capture). Dual-path branches (`if compiling:` / `if is_graph_mode:`) are acceptable only at a sanctioned seam — see Code Style.
- These rules bind **all** development, not only graph-mode work. Code written for eager today must default to the unified interfaces — functional ops, `differentiable_*` / funcol collectives (see Traceable Collectives), tensorized per-batch scalars — so that graph capture is a property of the entry point, not a rewrite of model code. Eager-only code that a tracer cannot consume is tech debt for the next graph-mode consumer.
- Dynamic-static unification is generally better achieved through **architectural boundaries** (out-of-graph seam, stateless metadata, shape staticization on the data side) than by patching eager code until it "also runs under the graph".
- Resolve graph-mode-hostile code in a fixed order: (1) one unified code path; (2) a sanctioned dual-path seam that keeps the static graph working (see Code Style); (3) only when no correct graph-mode implementation exists at all, gate by an explicit policy and fall back to eager (see Capability Gating).

## Code Style

- In-place writes that aot recompute would discard must be rewritten as functional equivalents; the change lands on the **same code**, which remains correct under eager.
- When a single code path is genuinely impossible, keep the static graph working via a **dual-path seam**: dispatch once through `_is_tracing_or_compiling()` (`hyper_parallel/core/utils/communication.py`) inside a shared helper, implement both branches with documented-equivalent semantics, and assert eager/graph parity with tests. The `differentiable_*` collectives wrappers are the canonical example. Do **not** scatter `torch.compiler.is_compiling()` checks through model code — that is path sprawl, not a seam.
- Data-dependent Python control flow is forbidden inside the graph (e.g., branching on `nonzero` / device tensor values); value-based dispatch, when required, happens at the out-of-graph seam.
- Python scalars that vary per batch (thresholds, ratios) must be tensorized before entering the graph, or lifted to compile-time constants with retrace conditions declared explicitly.

## Traceable Collectives

- Eager collectives (`torch.distributed.all_reduce` / `all_gather_into_tensor` / `reduce_scatter_tensor` / `all_to_all_single`) are imperative and in-place: they return `None` and mutate their input, so a tracer sees nothing and the collective is **silently dropped** from the captured graph.
- Replace them with the functional equivalents in `torch.distributed._functional_collectives` (funcol). They lower to registered `torch.ops._c10d_functional.*` ops with meta kernels, never mutate their input, and return the reduced tensor; eager execution stays correct because the result is an `AsyncCollectiveTensor` that waits on first access.

Rewrite pattern:

```python
import torch.distributed._functional_collectives as funcol

# Before: in-place eager collective — invisible to the tracer, dropped from the graph
reduced = tensor.contiguous()
torch.distributed.all_reduce(reduced, group=group)
return reduced

# After: one code path for eager and graph — functional, returns the reduced tensor
return funcol.all_reduce(tensor.contiguous(), "sum", group)
```

- Prefer the shared wrappers in `hyper_parallel/core/utils/communication.py`: they centralize the eager (`torch.distributed.nn.functional`) vs funcol seam behind `_is_tracing_or_compiling()` so call sites stay single-path.

| Wrapper (`core/utils/communication.py`) | Eager collective replaced | Funcol used under trace |
|------------------------------------------|---------------------------|-------------------------|
| `differentiable_all_reduce(data, op, group)` | `dist.all_reduce` (in-place) | `funcol.all_reduce(data, "sum", group)` |
| `differentiable_all_gather_concat(data, group, _, concat_dim)` | `dist_func.all_gather` (list) | `funcol.all_gather_single(data, concat_dim, group)` |
| `differentiable_reduce_scatter(data, dev_num, axis, op, group)` | `dist_func.reduce_scatter` | `funcol.reduce_scatter_single(data, "sum", axis, group)` |
| all-to-all (EP dispatch, `distributed/expert_parallel/collectives.py`; async variant `_AsyncA2ALazyBwd`) | `dist_func.all_to_all_single` | `funcol.all_to_all_single(x, out_splits, in_splits, group)` |

- Semantics to preserve when rewriting:
  - funcol takes the reduce op as a **string** (`"sum"` / `"product"` / `"max"` / `"min"` / `"avg"`); `_reduce_op_to_str` in `communication.py` maps `dist.ReduceOp` to these names.
  - funcol **returns a new tensor** — never expect the input argument to be updated; always use the return value (`t = funcol.all_reduce(t, "sum", group)`).
  - The eager result is an `AsyncCollectiveTensor`; materialize explicitly with `wait_async_tensor()` / `funcol.wait_tensor()` only when the wait must happen at a defined point (see also Streams rules in `distributed.md`).
- If PyTorch offers no funcol equivalent for a collective needed inside traced code, that is a capability-gating decision (warn once, fall back to eager as a whole) — not a reason to re-add eager collectives behind an `is_compiling()` branch.

## Capability Gating And Fallback

- Gating is the **last resort**: it applies only when no correct graph-mode implementation exists — after single-path and sanctioned dual-path seam options have both been ruled out (see Scope And Principles). A dual-path seam that keeps the graph running is preferred over an eager fallback.
- Gate graph-unfriendly features by **capability**: a compile policy declares the support matrix (arch / parallel-strategy dimensions); unsupported combinations **warn loudly once and fall back to eager as a whole** — no silent degradation.
- Flag semantics stay identical across modes: one recipe serves all supported backend combinations.
- The replay contract must be documented: pytree structure, dtype, device, and static shapes of subsequent batches must match the first capture; contract changes require an explicit re-capture — never retrace automatically.

## Testing And Acceptance

- UTs for dynamic-static unification changes must cover: eager baseline numerics, graph-mode numerics, and parity assertions between the two (same inputs, same weights).
- Changes touching fallback strategy require gating-behavior tests (warn triggered, fallback path correct).
- Test authoring follows `.agent/rules/testing.md` and `.agent/rules/unit-test.md`; assertion style follows `.agent/rules/test-assertion-style.md`.
