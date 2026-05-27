---
name: distributed-op-dev
description: Implementation constraints for HyperParallel distributed operators — YAML registration, preprocess, infer_layout, get_expand_impl.
paths:
  - hyper_parallel/core/shard/ops/**
---

# Distributed Operator Implementation Constraints

Applies to all code in `hyper_parallel/core/shard/ops/`.

---

## Class Selection (evaluate in order)

Before creating a new class, check the following in order:

1. **YAML-only registration** — If the operator is purely element-wise (output shape = broadcast of inputs, any dimension can be independently sharded), add an entry to an existing YAML file (`element_wise_ops.yaml`, `torch_element_wise.yaml`, etc.) pointing to an existing class. No new Python file is needed.

   | Class | Use when |
   |-------|---------|
   | `ElementWiseDistributedOp` | Element-wise, output layout tracks broadcastable inputs |
   | `ElementWiseWithPartialDistributedOp` | Same, but output may carry Partial state |
   | `TupleElementWiseDistributedOp` | Inputs/outputs are both tuples, element-wise semantics |
   | `ReshapeDistributedOp` | Changes shape without rearranging data (reshape, view, flatten) |

2. **Inherit from an existing class** — If the operator is semantically similar but needs small customizations (e.g., additional validation, `_MS_PRIMITIVE_OP_NAMES` routing), subclass the closest existing base class and override only what differs.

3. **New class with three-phase dispatch** — Only when no existing class fits. New classes must follow the three-phase dispatch model: `preprocess → infer_layout(cache_values) → get_expand_impl`.

**Must NOT** create a new class directly inheriting from `DistributedOp` when an existing specialized base class covers the operator's layout semantics.

---

## YAML Registration

**Must:**
- Register the Primitive class name (PascalCase) for MindSpore side; register the torch function name for PyTorch side.
- Place the entry in the appropriate `yaml/*.yaml` file under `core/shard/ops/yaml/`.

**Must NOT:**
- Set `infer_layout_suffix` — suffix logic (WithShape, WithTupleExpand, etc.) must be handled inside `preprocess`.
- Register `mint.*` interface names — `mint` calls resolve to Primitives; register the Primitive name only.

---

## `_normalize_*_args` Function

**Purpose:** Unified entry point for all platform call sites (torch / mint / Primitive). Resolves interface differences so that `preprocess` always receives a consistent argument shape regardless of which platform invoked the operator.

Every operator must define a module-level `_normalize_*_args` function to unify cross-platform interface differences before `preprocess` processes arguments.

**Must:**
- Return `(args_tuple, kwargs_dict)`.
- Default: return all parameters as positional args, `kwargs = {}`.
- Keyword-only parameters (declared after `*` in the torch interface) must stay in `kwargs`. MindSpore `mint.*` functional_overload interfaces follow the same rule as PyTorch.
- MindSpore mint interfaces with `kwonlyargs` declared in `mindspore/ops/api_def/{op_name}.yaml` behave like functional_overload: those parameters must stay in `kwargs`. Interfaces without `kwonlyargs` are all-positional. (Note: `ops/op_def/yaml/` defines Primitive structure — args/returns/dtypes; `ops/api_def/` declares the Python-level `kwonlyargs`.)

**Must NOT:**
- Put keyword-only parameters in `args` — they cannot be passed positionally.
- Put non-keyword-only parameters in `kwargs` for MindSpore Primitives that do not declare `kwonlyargs`.

---

## `preprocess`

**Purpose:** The only place that touches raw `args`/`kwargs`. Calls `_normalize_*_args`, unwraps DTensors to local tensors via `.to_local()`, and packs the minimal information needed for layout inference into `cache_values`. Everything downstream (`infer_layout`, `get_expand_impl`) works exclusively from the outputs of this function.

**Signature:**
```python
def preprocess(self, args: tuple, kwargs: dict) -> tuple:
```

**Must:**
- Be implemented by every new operator that uses the three-phase dispatch (i.e., does not inherit from an existing specialized base class). Returning `None` falls back to the legacy dispatch path, which is forbidden for such operators.
- Call `_normalize_*_args(*args, **kwargs)` as the first step.
- Call `.to_local()` on every DTensor input; pass raw local tensors in `local_args` / `local_kwargs`.
- Build `cache_values` containing only information that affects layout inference or must be validated: Layout objects first, then scalar parameters (int, bool, tuple, etc.). Absent optional tensors use `None` as a placeholder.
- Route `local_args` / `local_kwargs` by `self.op_name`:
  - `self.op_name in _MS_PRIMITIVE_OP_NAMES` and the op has **no** `kwonlyargs` in `mindspore/ops/api_def/{op_name}.yaml` → all positional, `local_kwargs = {}`.
  - `self.op_name in _MS_PRIMITIVE_OP_NAMES` and the op **has** `kwonlyargs` → those listed params go in `local_kwargs`.
  - Otherwise (PyTorch function) → keyword-only parameters (declared after `*`) in `local_kwargs`.
- Declare `_MS_PRIMITIVE_OP_NAMES` as a class-level `frozenset`; derive its members from the YAML registration file (MindSpore-side entries only).

**Must NOT:**
- Contain any validation or error-raising logic — all validation belongs in `infer_layout`.
- Store shape information in `cache_values` unless shape affects layout (e.g., broadcasting rules).
- Call `infer_layout` or perform operator compute.

---

## `infer_layout`

**Purpose:** Pure layout mathematics — takes `cache_values` built by `preprocess` and derives the output layout(s). All validation (sharding constraints, mesh consistency, type checks) happens here; no compute, no access to the original `args`/`kwargs`.

**Signature:**
```python
def infer_layout(self, cache_values: list) -> Tuple[tuple, None]:
```

**Must:**
- Call `self._check_partial_inputs(input_layouts)` as the first statement. Unlike the legacy base class, the new dispatch flow does not call this automatically.
- Call `self._validate_input_layouts(...)` immediately after. Every operator must define `_validate_input_layouts` as a `@staticmethod`; it performs all layout compatibility checks (sharding constraints, mesh consistency, etc.) and raises `ValueError` for violations.
- Validate all other inputs early (fail-fast): type checks, range checks, mesh shape consistency, sharding constraints — before any layout derivation.
- Use `layout.alias_tensor_map` to inspect per-dimension sharding. `"None"` means Replicate. StridedShard produces a tuple mapping; check `isinstance(mapping, (tuple, list))` before comparing.
- Format all error messages as exactly: `f"For {self.op_name}, <what> should be <expected>, but got <actual>."` — three-part, comma-separated. Never hardcode the operator name or use `self.__class__.__name__`.
- Use the `Rules:` section in the docstring to enumerate what is and is not allowed.
- Return `((output_layout_1, ...), None)`. Always a two-element tuple; second element is always `None`.
- `deepcopy` output layout when it is derived directly from an input layout to prevent aliasing.

**Must NOT:**
- Use `layout.tensor_map` for shard checks — it does not support StridedShard tuple values.
- Perform any operator compute or collective communication.
- Read information that was not included in `cache_values` (no access to `args` / `kwargs` here).

---

## `get_expand_impl`

**Purpose:** Returns a replacement callable when the local operator call must differ from the default (e.g. adjusted parameters per rank, fused collectives). If the local call is identical to the single-device call, return `None` and the dispatch layer calls `func` directly. All per-rank constants are computed here (outside the closure) so they are not recomputed on every forward pass.

**Signature:**
```python
def get_expand_impl(  # pylint: disable=W0237
        self,
        func: Optional[Callable],
        infer_result: tuple,
        cache_values: list,
) -> Optional[Callable]:
```
`# pylint: disable=W0237` is required because the new parameter names (`cache_values`) differ from the legacy base class signature (`layouts`, `extra_args`).

**Must:**
- Use guard clauses (early `return None`) for all cases where no special handling is needed. Place guard clauses before any closure definition.
- Complete all computation that does not depend on the closure's runtime arguments (e.g., scaling factors derived from `infer_result` or `cache_values`) outside the closure body.
- Name all inner closure functions with a leading underscore (e.g. `_expand_impl`, `_row_shard_impl`). Unnamed lambdas are only acceptable for trivial one-expression cases.
- Define the closure and immediately `return` it — no logic between the closure definition and the return statement.
- Capture all pre-computed values via closure variables.
- Use `platform.*` APIs (via the module-level `platform = get_platform()`) for any collective or tensor operations. Do not call `torch.*` or `mindspore.*` directly.

**Must NOT:**
- Put computation that can be done outside the closure inside the closure — it would execute on every forward pass.
- Pass `infer_result[1]` (extra_info) to `op_impl` — the dispatch layer calls `op_impl(*local_args, **local_kwargs)` directly; extra_info is only available inside `get_expand_impl`.
- Raise errors here — all validation must be done in `infer_layout`.
- Import or call `torch` / `mindspore` directly — use the platform abstraction.

---

## `wrap_output`

Do not override unless the operator requires non-standard output packing (e.g., selective DTensor wrapping for tuple outputs). The base class default handles single-tensor and uniform tuple outputs correctly.
