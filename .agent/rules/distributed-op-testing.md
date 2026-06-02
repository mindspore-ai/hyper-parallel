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

**Framework:** `unittest.TestCase` only. Do not use pytest.

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

**level_mark assignment:** All Ascend ST test cases default to `level1`. Do not use `level0` unless the user explicitly requests it.

---

## MindSpore ST Constraints

**File naming:**
- Runner: `test_parallel_op_{op_name}.py`
- Impl: `_test_parallel_op_{op_name}.py` (same directory, `_` prefix)

**Runner must:**
- Compute `IMPL_FILE` using `str(Path(__file__).resolve().parent / "_test_parallel_op_{op_name}.py")`. No hardcoded paths.
- Use `parallel_run([MindSporeCase(...)])` — not the legacy `msrun_case`.
- Pass only `worker_num`, `local_worker_num`, `glog_v` to `MindSporeCase`. Do not pass `master_port`.
- Have each `parallel_run` group sum to exactly 8 cards (`sum(worker_num) == 8`). At most one non-full group per file (when total cases cannot fill 8 cards).
- List all test function names and brief descriptions in each runner function's `Description` docstring.

**Impl must:**
- Define `setup_module()` that calls `ms.set_device("Ascend")` and `D.init()`.
- Invoke the operator under test via `mint.*` interface when a mint interface exists. Do not call the MindSpore Primitive class directly — users access operators through mint, and ST must validate that path.

---

## PyTorch ST Constraints

**File naming:** same as MindSpore — `test_parallel_op_{op_name}.py` (runner) and `_test_parallel_op_{op_name}.py` (impl).

**Runner must:**
- Compute `IMPL_FILE` using `str(Path(__file__).resolve().parent / "_test_parallel_op_{op_name}.py")`.
- Use `parallel_run([TorchCase(...)])` — not legacy `torchrun_case`.
- Pass only `num_proc` to `TorchCase`. Do not pass `master_port` (auto-assigned).
- Each `parallel_run` group: `sum(num_proc) ≤ 8`.
- If the operator interface is a native torch interface (not `torch_npu`), add a `_gloo` variant function in the runner with the same name suffixed `_gloo`. Its `@arg_mark` must use `plat_marks=["cpu_linux"]` and `level_mark="level0"`. The gloo variant calls `parallel_run` with the same cases.

**Impl must:**
- Import device utilities from `tests.torch.utils`: `from tests.torch.utils import _DEVICE_TYPE, init_backend, to_device`.
- Use `init_backend()` instead of `init_dist()` to initialize the distributed backend.
- Use `to_device(tensor)` to move tensors to the target device. Do not call `.npu()`, `.cuda()`, or `.to("npu")` directly.
- Verify both layout correctness and numerical equivalence against a standalone reference.

---

## Assertion Format

- Use f-strings for all assertion messages.
- Print both compared values in the message.
- Keep each line ≤ 120 characters; use parentheses to wrap multi-line messages.

```python
# Correct
assert np.allclose(a, b, 1e-3, 1e-3), (
    f"Data parallel mismatch: standalone={a}, parallel={b}"
)

# Wrong — no values printed
assert np.allclose(a, b, 1e-3, 1e-3), "Test failed"
```
