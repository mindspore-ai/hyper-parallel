---
name: distributed-op-testing
description: Testing constraints for HyperParallel distributed operators. Covers UT and ST file structure, naming, forbidden patterns, and assertion format.
paths:
  - tests/ut/core/shard/ops/**
  - tests/mindspore/st/shard/ops/**
  - tests/torch/shard/ops/**
---

# Distributed Operator Testing Constraints

Applies to all distributed operator tests under `tests/`. Two test layers are required per operator:

| Layer | Directory | Purpose |
|-------|-----------|---------|
| UT | `tests/ut/core/shard/ops/` | CPU-only logic verification; covers all error paths |
| ST (MindSpore) | `tests/mindspore/st/shard/ops/` | Real multi-card distributed execution; success paths only |
| ST (PyTorch) | `tests/torch/shard/ops/` | Real multi-card distributed execution; success paths only |

---

## UT Constraints

**File naming:** `test_<source_file_name>.py` — strict 1-to-1 with `hyper_parallel/core/shard/ops/<source_file_name>.py`. One UT file per source file; do not split.

**Framework:** write UT case bodies as `unittest.TestCase`. Do **not** write pytest-native test functions/fixtures as the primary structure for these UT files. Still **run** them with pytest. (Runner vs authoring — see `.agent/rules/testing.md`.)

**setUp / tearDown must:**

- Clear `EXISTING_COMM_GROUPS`, `_DEVICE_MESH_MAP`, and `_LAYOUT_CACHE` in both methods. `_build_layout` uses a content-hash key; a layout mutated by one test will poison subsequent tests sharing the same mesh shape.
- Carry `-> None` return type annotations.

**Must NOT:**

- Call `get_platform()` or use any platform instance — platform is mocked via `@patch` in UT.
- Add `@arg_mark` to UT tests — that decorator is for ST only.

**infer_layout calls:** pass a single `cache_values` list — `op.infer_layout(cache_values)`, not `op.infer_layout(layouts, extra_args)`.

**get_expand_impl calls:** third argument is `cache_values` — `op.get_expand_impl(func, (output_layouts, None), cache_values)`.

**get_expand_impl verification:**

- If the operator does NOT override `get_expand_impl` (returns `None`): verify once per test class with a comment explaining why. Use a direct assertion without variable assignment to avoid lint warnings.
- If the operator overrides `get_expand_impl` (returns callable): verify callable in every test method.

**Error assertion format:** `self.assertRaisesRegex(ValueError, "lowercase substring")` — match a substring of the three-part message, not the full string.

**Required scenario coverage:**

- Data parallel, model parallel, hybrid parallel, all replicated, negative dim index, partial input error.
- All operator-specific error paths (sharded forbidden dim, layout inconsistency, invalid args).
- Preprocess routing (if `_MS_PRIMITIVE_OP_NAMES` is defined): verify keyword-only params land in `local_kwargs` for PyTorch ops, and in `local_args` (or correct `local_kwargs` for Primitives with `kwonlyargs`) for MindSpore Primitives.

---

## ST Common Constraints

**Error cases belong in UT, never ST.** Do not use `pytest.raises`, `assertRaises`, or `try/except` as a pass condition in ST.

**Every ST test function must include a numerical comparison** against a standalone (single-device) reference. A shape-only assertion (`result.shape == (...)`) is not a valid comparison unless the scenario makes numerical comparison impossible (e.g., random dropout); in that case, add a comment explaining why.

**device_mesh: each dimension must be ≤ 2.** `(2,)`, `(2,2)`, and `(2,2,2)` are allowed; `(4,)`, `(8,)`, `(2,4)` are not.

---

## ST — New Framework (OpShardCase)

ST tests for shard ops use the declarative `OpShardCase` framework under `tests/shard_ops/framework/`.

### Directory Layout

```text
tests/torch/shard/ops/
├── test_shard_ops_suite.py      # pytest entry for all Torch cases
├── cases/                       # one file per operator
│   ├── __init__.py              # auto-discovers case_*.py
│   ├── case_sort.py
│   └── case_cat.py
└── framework/                   # Torch-specific backend
    ├── __init__.py
    └── backend_torch.py

tests/mindspore/st/shard/ops/    # same layout for MindSpore
├── test_shard_ops_suite.py
├── cases/
│   ├── __init__.py
│   ├── case_sort.py
│   └── case_cat.py
└── framework/
    ├── __init__.py
    └── backend_mindspore.py
```

### Case Definition

Each `cases/case_{op}.py` defines test functions and registers cases:

```python
import torch  # or mindspore as ms
from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from tests.shard_ops.framework import (
    CompareSpec, InputSpec, OpShardCase, register,
)

def _sort_dp(x):
    return torch.sort(x, dim=-1)

register(OpShardCase(
    name="sort_ops_2d_dp",
    fn=_sort_dp,
    inputs=[InputSpec(shape=(8, 16), init="randn", seed=42)],
    placements=[(Shard(0), Replicate())],
    compare=CompareSpec.equal(),
    tags=("cpu_level0", "npu_level0"),
))
```

| Field | Purpose |
|-------|---------|
| `name` | Unique case identifier; convention: `{op}_ops_{scenario}` (e.g. `cat_ops_dp_dim1`) |
| `fn` | The operator under test. Receives `(*inputs, *derived_inputs, *extra_inputs, **kwargs)` |
| `inputs` | `InputSpec(shape, init, seed, dtype, data)` — declarative tensor specs (primary tensors only) |
| `placements` | One placement tuple per input. **Tuple length == mesh ndim** (see Placement Convention below) |
| `compare` | `CompareSpec.equal()` or `CompareSpec.allclose(rtol, atol)` |
| `tags` | `("cpu_level0", "npu_level0")` for Torch; `("npu_level0",)` for MindSpore |
| `mesh_shape` | No default — declare **explicitly** on every case. `(2,)` for 1D, `(2,2)` for 2D, `(2,2,2)` for 3D |
| `mesh_dim_names` | e.g. `("dp","tp")`. Match mesh_shape cardinality. For ordinary ops these are **cosmetic** (placements act by axis index; the bucketer canonicalizes them by ndim, so naming is documentation-only). For MC2 ops (`needs_mesh=True`) they are **live** — the fn resolves its comm group via `get_group(group_dim)`, so keep real names and match the `group_dim` in `kwargs` |
| `extra_inputs` | Non-tensor args (scalars, dims, seq-lens) passed through unchanged — not distributed |
| `compare_outputs` | For tuple outputs: indices to compare, e.g. `(0,)` to compare only output[0] |
| `derived_inputs` | `DerivedSpec` list — inputs computed once on full tensors then sliced (see Derived Inputs below) |

**Placement Convention (critical).** A placement tuple has **one entry per mesh axis** (length == `len(mesh_shape)`), **not** per tensor dimension. Entry `i` describes how the tensor is laid out on mesh axis `i`; `Shard(d)` shards **tensor dimension `d`** on that axis, `Replicate()` replicates on it.

```python
# 4-D tensor on a 2-D mesh (dp, tp): length-2 tuple (one per mesh axis)
mesh_shape=(2, 2)
placements=[(Shard(0), Shard(1))]   # dp axis shards tensor dim 0; tp axis shards tensor dim 1

# 2-D tensor on a 1-D mesh (tp): length-1 tuple
mesh_shape=(2,)
placements=[(Shard(1),)]            # tp axis shards tensor dim 1 (the k dim)
```

A tuple longer than the mesh ndim "works" only when the surplus entries are `Replicate()` (a no-op); a misplaced `Shard` silently degrades to replicate or raises `IndexError`. Always match the mesh ndim exactly.

**Derived Inputs.** When the op consumes a tensor *derived* from the primary inputs via a device op needing global information (e.g. attention `sm_max/sm_sum` from `FlashAttentionScore` over the full K dim), declare it via `derived_inputs=[DerivedSpec(fn, placement), ...]`. The framework computes `fn(*full_primary_tensors)` **once on the full tensors** (before distribution), then slices the result per `placement` for the parallel path — it is never recomputed on shards. Derived tensors are appended after the primary tensors: `fn(*primary, *derived, *extra, **kwargs)`. Do **not** fake derived stats with random `data=` (wrong values) and do **not** recompute them inside `fn` (the device op would see inconsistently-sharded inputs). If the derived value is computable from local shards instead (e.g. an MC2 op whose reference is a plain `matmul`), branch inside `fn` on `isinstance(x, DTensor)` rather than using `derived_inputs`.

**InputSpec fields:**

| Field | Default | Description |
|-------|---------|-------------|
| `shape` | required | Tensor dimensions |
| `init` | `"randn"` | `"randn"` / `"uniform"` / `"ones"` / `"zeros"` / `"arange"` |
| `seed` | `None` | Deterministic seed for random init |
| `dtype` | `"float32"` | Platform-neutral dtype name |
| `data` | `None` | `np.ndarray` for exact values; overrides `init` |

**Suite entry** (`test_shard_ops_suite.py`) filters cases by `tag_include`:

```python
_GROUPS_CPU_LEVEL0 = build_suite_groups(
    cases_pkg=CASES_PKG,
    tag_include={"cpu_level0"}, fail_fast=True,
)
```

### Local Test (Two Modes)

**Mode 1 — CLI:** Run specific cases by file or name:

```bash
# Torch (CPU default)
python -m tests.shard_ops.framework --case sort_ops_2d_dp --num-proc 4
python -m tests.shard_ops.framework --device-type npu --case sort_ops_2d_dp --num-proc 4

# MindSpore — set HYPER_PARALLEL_PLATFORM=mindspore (it overrides --framework)
HYPER_PARALLEL_PLATFORM=mindspore \
  python -m tests.shard_ops.framework --framework mindspore --device-type npu \
  --case argsort_ops_dp --num-proc 4
```

- `--num-proc` **must equal `math.prod(mesh_shape)`** (2 for `(2,)`, 4 for `(2,2)`, 8 for `(2,2,2)`); a mismatch makes ranks ≠ mesh size and HCCL hangs.
- `HYPER_PARALLEL_PLATFORM` takes precedence over `--framework`; for MindSpore CLI runs set it to `mindspore` or the Torch cases load instead.

**Mode 2 — Suite entry with env filter:** Run via pytest, filter cases within groups:

```bash
HYPER_PARALLEL_SHARD_CASE_FILTER="sort_ops_*" \
  pytest tests/torch/shard/ops/test_shard_ops_suite.py::test_shard_ops_cpu_level0 -vs
```

### Gate Routing

Cases are not assigned a fixed level field. The `tags` tuple carries platform-level routing:

```python
tags=("cpu_level0", "npu_level0")   # runs level0 on both
tags=("cpu_level0", "npu_level1")   # level0 on CPU, level1 on NPU
tags=("npu_level0",)                # NPU only
```

Suite entry functions use `tag_include` to select cases. No `level` field on `OpShardCase`.

---

## Assertion Format

Canonical style: `.agent/rules/test-assertion-style.md` (f-strings, both values, ≤120 cols).

Op-specific extras above (e.g. `assertRaisesRegex` substring matching) still apply.