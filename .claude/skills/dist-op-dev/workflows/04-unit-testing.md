# Workflow 4: Unit Testing (UT)

## Priority Rule

If there is any conflict between:

- previous context code
- other test files
- examples generated earlier
- this document

You **MUST follow the rules defined in this document.**

Do **NOT reuse testing code patterns** from previous context unless they match this specification.

Please follow the testing guidelines defined in: `rules/distributed-op-testing.md`.

## Goal

Verify the correctness of distributed operator class `infer_layout` and `get_expand_impl` logic, ensure all supported scenarios from the analysis report are covered, and unsupported scenarios raise explicit errors.

## Input

- **Analysis Report**: `.claude/skills/dist-op-dev/analysis-results/{OpName}-analysis.md` (from Step 1)
- **Python Implementation Class**: Distributed operator class from Workflow 2

## Output

- **pytest Test File**: `tests/ut/core/shard/ops/test_parallel_*.py`

**Note**: UT test cases are platform-agnostic, verifying basic functionality of distributed operator classes. **If there is no new Python distributed operator class implementation (Scenario 0 fully using base class), no need to write UT test cases.**

---

## Test Architecture

All tests **must follow the architecture template below.**

Do **NOT copy layout construction or operator invocation patterns** from previous context.

Always start from this template.
```python
import os
import unittest
from unittest.mock import patch
os.environ["HYPER_PARALLEL_PLATFORM"] = "mindspore"

from hyper_parallel.core.dtensor.dtensor import _build_layout
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from hyper_parallel.core.shard.ops.parallel_xxx import XxxDistributedOp
from hyper_parallel.core.dtensor.device_mesh import init_device_mesh, _DEVICE_MESH_MAP
from hyper_parallel.platform.platform import EXISTING_COMM_GROUPS

op = XxxDistributedOp("Xxx")

class TestParallelXxx(unittest.TestCase):
    def setUp(self):
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()

    def tearDown(self):
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()

    def _make_2x2x2_mesh(self, mock_platform):
        mock_platform.get_rank.return_value = 0
        mock_platform.get_world_size.return_value = 8
        return init_device_mesh(
            device_type="npu", mesh_shape=(2, 2, 2),
            mesh_dim_names=("dp", "cp", "mp"), init_backend=False
        )
```

---

## Step 2: Construct Tensor Layouts

All tensor layouts **must follow the layout construction rule.**

```python
@patch("hyper_parallel.core.dtensor.device_mesh.platform")
def test_operator_data_parallel_success(self, mock_platform):
    """Test with data parallel sharding."""
    self._setup_mock_platform(mock_platform, world_size=8)
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 4),
        mesh_dim_names=("dp", "mp"),
        init_backend=False  # Important: no actual backend for UT
    )
    x_placements = (Shard(0), Shard(1))
    x_layout = _build_layout(mesh, x_placements, 2)
```
Rules:
  **Use `@patch` decorator to mock the platform for unit tests:**
  Always define placements explicitly
  Always use _build_layout
  Layout dimension must match tensor rank


---

## Step 3: Implement Test Scenarios

Each test file must include the following scenario coverage.

### Supported Scenarios (Required — from analysis report)

Each supported sharding scenario listed in the analysis report must have a corresponding UT test:

| Test Scenario | Description | Verify |
|---------------|-------------|--------|
| **Data Parallel (DP)** | Tensor sharded along batch dimension | `infer_layout` output tensor_map |
| **Model Parallel (MP)** | Tensor sharded along model dimension | `infer_layout` output tensor_map |
| **Hybrid Parallel** | Tensor sharded along multiple dimensions | `infer_layout` output tensor_map |
| **Context Parallel (CP)** | If applicable, tensor sharded along context dimension | `infer_layout` output tensor_map |
| **Partial Status** | If output is partial, verify `set_partial_by_dev_axis` | Partial flag and axis |
| **get_expand_impl** | If analysis report requires it, verify callable or None | Return value |

### Unsupported Scenarios (Required — from analysis report)

Each unsupported scenario listed in the analysis report must have a corresponding UT test that verifies the explicit error:

| Test Scenario | Expected Error | Verify |
|---------------|---------------|--------|
| **Invalid input layout** | `ValueError` with clear message | `assertRaises` |
| **Mesh shape mismatch** | `ValueError` with clear message | `assertRaises` |
| **Contracting dim conflict** | `ValueError` with clear message | `assertRaises` |
| **Other unsupported cases** | `ValueError` with clear message | `assertRaises` |

### Edge Cases (Required)

| Test Scenario | Description |
|---------------|-------------|
| **Scalar/None Input** | One input layout is None (scalar operand) |
| **Single input operator** | Only one input layout provided |
| **Negative dimension index** | If operator uses negative dim indexing |

### Broadcast Scenarios (Required for WithShape operators)

| Test Scenario | Description |
|---------------|-------------|
| **Scalar Broadcast** | One input is scalar (layout is None), `extra_args` includes `input_shapes` |
| **Shape Broadcast** | Different rank shapes, `extra_args` includes `input_shapes` |

If the operator may produce a **partial output layout**, the unit test MUST validate the partial state including the following steps:
1. Construct input layouts
2. Call `infer_layout`
3. Verify that the output layout is partial
4. Verify the `partial` attribute

Example:
```python
output_layout = op.infer_layout((x_layout, y_layout), extra_args)
assert output_layout.is_partial()
assert output_layout.partial == [None, 'sum'] # NOTE: This is only an example. Replace it with the correct partial result for the current operator.
```
If the operator does not produce partial state:
```python
output_layout = op.infer_layout((x_layout, y_layout), extra_args)
assert not output_layout.is_partial()
```
**Partial State Rule:**
`Layout.partial` is a list aligned with `alias_name`.
Example:
alias_name = ("dp", "mp")
If the operator requires SUM aggregation on `mp`:partial = [None, "sum"]
Rules:
- length equals number of device axes
- non-communication axes must be None
- only communication axis can have reduction type

---

## Step 5: Expand Implementation Validation (If Applicable)

**IMPORTANT: Every infer_layout test MUST be followed by get_expand_impl validation.**

### When impl should be None

For operators that do NOT need expand implementation (most operators), use **direct assertion without variable assignment**:

```python
# Correct - direct assertion avoids linter warning
assert op.get_expand_impl(None, output_layout, (x_layout,), extra_args) is None, (
    f"get_expand_impl should return None, "
    f"got{op.get_expand_impl(None, output_layout, (x_layout,), extra_args)}"
)

# Wrong - assigning None triggers linter warning
impl = op.get_expand_impl(None, output_layout, (x_layout,), extra_args)
assert impl is None
```

**Operators with None impl:**
- Simple element-wise operators (Add, Mul, ReLU, etc.)
- Reshape-like operators (ExpandDims, Squeeze, Transpose, etc.)
- Operators without partial state handling

### When impl should be callable

For operators that need expand implementation, verify callable:

```python
# Correct - variable assignment is fine for callable
impl = op.get_expand_impl(None, output_layout, (x_layout,), extra_args)
assert callable(impl), (
    f"get_expand_impl should return callable, got {type(impl)}"
)
```

**Operators with callable impl:**
- Gather with sharded index dimension (returns partial state)
- MatMul with partial output (needs reduce-scatter)
- FlashAttentionScore (needs parameter adjustment)
- Operators requiring collective communication for output

Rules:
- Must validate get_expand_impl after every infer_layout
- Use direct assertion for None return value
- Cannot skip validation after infer_layout

---

## UT Coverage Checklist

- [ ] All supported scenarios from analysis report covered
- [ ] All unsupported scenarios from analysis report raise explicit `ValueError`
- [ ] `infer_layout` output tensor_map verified for each scenario
- [ ] `get_expand_impl` return value verified (callable or None) per analysis report
- [ ] Partial status setting verified (if applicable)
- [ ] Edge cases covered (None input, single input, negative index)
- [ ] Broadcast scenarios covered (for WithShape operators)
- [ ] `setUp`/`tearDown` clear `EXISTING_COMM_GROUPS` and `_DEVICE_MESH_MAP`
- [ ] Each test has docstring with Feature/Description/Expectation format
- [ ] **NO `@arg_mark` decorator** — UT tests are plain pytest/unittest, never use `@arg_mark`
- [ ] **NO distributed launch** — UT tests run locally with mocked platform, never use `parallel_run`

- [ ] Expand implementation validation (if `get_expand_impl` exists)

