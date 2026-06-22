---
name: dist-op-dev
description: Distributed operator development. Reads a confirmed plan file, implements operator code and tests, and runs until all executable tests pass. Goal mode — no step-by-step confirmation needed.
---

# Distributed Operator Development

## Trigger

The user provides a confirmed plan file or specifies the latest plan:

```text
/dist-op-dev implement according to .agent/skills/dist-op-analysis/plans/Sort_dist_op_plan.md
```

If the plan contains any `[待确认]` items, resolve them with the user before proceeding.

Verification reports (UT/ST results, debug records) are saved to `.agent/skills/dist-op-dev/reports/{OpName}_report.md`. This directory is excluded by `.gitignore`. **Write the report in Chinese.**

---

## Execution Flow

Goal mode — execute the following steps autonomously without step-by-step confirmation.

### Step 1: Read the Plan

Extract from the plan file:

- Operator name, platform scope (PyTorch / MindSpore / Both)
- Base class selection (Path A / B / C, see Step 2)
- File paths and interface skeletons
- UT test case list
- MindSpore ST test case list (if applicable)
- PyTorch ST test case list (if applicable)

### Step 2: Implement the Operator

**Reference implementations (new dispatch flow — use these as templates):**

| File | What it demonstrates |
|------|---------------------|
| `parallel_sort.py` | `preprocess` + `infer_layout(cache_values)` + `_MS_PRIMITIVE_OP_NAMES` — clean dual-platform structure |
| `parallel_rotary_position_embedding.py` | `preprocess` + `@staticmethod _validate_input_layouts` + `infer_layout(cache_values)` — most complete new-flow example |
| `parallel_matmul.py` | `preprocess` + `get_expand_impl` — reference when expansion logic is needed |

> **Warning:** Most other `parallel_*.py` files use the **legacy dispatch** (`infer_layout(layouts, extra_args)`, no `preprocess`). Do **not** copy their dispatch signatures or structural patterns. Use the files listed above as templates.

Execute the path specified in the plan's base class selection:

**Path A: YAML-only registration** (no new Python file)

Add an entry to the appropriate YAML file (e.g. `element_wise_ops.yaml`, `torch_element_wise.yaml`) pointing to an existing class (`ElementWiseDistributedOp`, etc.). No new Python file.

**Path B: Inherit an existing base class**

Create `hyper_parallel/core/shard/ops/parallel_{op_name}.py`, inherit the base class specified in the plan, override only the methods that need customization, and add the YAML entry.

**Path C: New class + three-phase dispatch**

Create `hyper_parallel/core/shard/ops/parallel_{op_name}.py`, implementing in this order:

1. `_normalize_{op_name}_args(...)` — module-level; unifies interface differences; returns `(args_tuple, kwargs_dict)`
2. `{OpName}DistributedOp({BaseClass})` class:
   - `_MS_PRIMITIVE_OP_NAMES = frozenset({...})` (dual-platform: MindSpore-side YAML registration names)
   - `preprocess(self, args, kwargs)`
   - `@staticmethod _validate_input_layouts(...)`
   - `infer_layout(self, cache_values)`
   - `get_expand_impl(self, func, infer_result, cache_values)` (if needed)

Create `hyper_parallel/core/shard/ops/yaml/{op_name}_ops.yaml` for registration.

Implementation must comply with `.agent/rules/distributed-op-dev.md`.

### Step 3: Write UT

File: `tests/ut/core/shard/ops/test_parallel_{op_name}.py`

Follow all UT constraints in `.agent/rules/distributed-op-testing.md`.

### Step 4: Write MindSpore ST (if applicable)

Create `tests/mindspore/st/shard/ops/cases/case_{op_name}.py`. Use the declarative `OpShardCase` framework:

```python
import mindspore as ms
from tests.shard_ops.framework import CompareSpec, InputSpec, OpShardCase, register
from hyper_parallel.core.dtensor.placement_types import Replicate, Shard

def _op_dp(x):
    return ms.mint.argsort(x, dim=-1)

register(OpShardCase(
    name="argsort_ops_dp",
    fn=_op_dp,
    inputs=[InputSpec(shape=(8, 16), init="randn", seed=42)],
    placements=[(Shard(0), Replicate())],
    compare=CompareSpec.equal(),
    tags=("npu_level0",),
))
```

- Use `ms.mint.xxx()` — never call MindSpore Primitives directly
- Tags: `("npu_level0",)` (4-card / `(2,2)`) or `("npu_level1",)` (2-card / 8-card) for MindSpore-only cases
- Cases are auto-discovered via `cases/__init__.py`; no manual import needed
- **Placement length == mesh ndim**, and if the op needs derived stats use `derived_inputs` — see `.agent/rules/distributed-op-testing.md`

Follow all ST constraints in `.agent/rules/distributed-op-testing.md`.

### Step 5: Write PyTorch ST (if applicable)

Create `tests/torch/shard/ops/cases/case_{op_name}.py` — same declarative pattern as MindSpore, but with `torch.xxx()` APIs:

```python
import torch
from tests.shard_ops.framework import CompareSpec, InputSpec, OpShardCase, register
from hyper_parallel.core.dtensor.placement_types import Replicate, Shard

def _op_dp(x):
    return torch.sort(x, dim=-1)

register(OpShardCase(
    name="sort_ops_2d_dp",
    fn=_op_dp,
    inputs=[InputSpec(shape=(8, 16), init="randn", seed=42)],
    placements=[(Shard(0), Replicate())],
    compare=CompareSpec.equal(),
    tags=("cpu_level0", "npu_level0"),
))
```

- Tags: `("cpu_level0", "npu_level0")` for Torch cases that run on both backends
- Use standard `torch.*` APIs (not `torch_npu`)
- Follow all ST constraints in `.agent/rules/distributed-op-testing.md`

### Step 6: Run Tests and Fix Until All Pass

**Environment detection**

```bash
npu-smi info >/dev/null 2>&1 && echo "ascend" || echo "no-ascend"
```

- Ascend environment: run all tests (UT + Torch gloo ST + Torch Ascend ST + MindSpore ST)
- Non-Ascend environment: run UT and Torch gloo ST only; skip Ascend ST

**Run commands**

```bash
# UT (always run)
pytest -vs tests/ut/core/shard/ops/test_parallel_{op_name}.py

# Torch CPU (gloo) — level0 and level1
pytest -vs tests/torch/shard/ops/test_shard_ops_suite.py::test_shard_ops_cpu_level0
pytest -vs tests/torch/shard/ops/test_shard_ops_suite.py::test_shard_ops_cpu_level1

# Torch Ascend (hccl) — level0 and level1
pytest -vs tests/torch/shard/ops/test_shard_ops_suite.py::test_shard_ops_ascend_level0
pytest -vs tests/torch/shard/ops/test_shard_ops_suite.py::test_shard_ops_ascend_level1

# MindSpore (Ascend only) — level0 and level1
pytest -vs tests/mindspore/st/shard/ops/test_shard_ops_suite.py::test_shard_ops_ascend_level0
pytest -vs tests/mindspore/st/shard/ops/test_shard_ops_suite.py::test_shard_ops_ascend_level1

# Fast iteration on your op only — env filter (suite) or CLI (single case)
HYPER_PARALLEL_SHARD_CASE_FILTER="{op}_ops_*" \
  pytest tests/torch/shard/ops/test_shard_ops_suite.py::test_shard_ops_cpu_level0 -vs
python -m tests.shard_ops.framework --case {op}_ops_dp --num-proc 4              # torch
HYPER_PARALLEL_PLATFORM=mindspore \
  python -m tests.shard_ops.framework --framework mindspore --device-type npu \
  --case {op}_ops_dp --num-proc 4                                                # mindspore
```

`--num-proc` must equal `math.prod(mesh_shape)`. For MindSpore CLI runs export `HYPER_PARALLEL_PLATFORM=mindspore`.

**Failure handling**

- Identify the root cause — do not skip or use suppress/skip as a workaround
- Search existing operators for similar patterns and follow the established approach
- Re-run after each fix until all executable tests pass
- Append each run result (including failure output and fix steps) to `reports/{OpName}_report.md`

---

## On Completion

Once all executable tests pass, report to the user:

- List of files created/modified
- Test results (UT, gloo ST, Ascend ST — status of each)
- If any Ascend ST was skipped, remind the user to run them manually in an Ascend environment

Prompt the user to run `/commit` to commit the code.

---

## Out of Scope

- git commit / push (handled by the `autogit` skill)
- CI gate monitoring (handled by the `gate-doctor` skill)
