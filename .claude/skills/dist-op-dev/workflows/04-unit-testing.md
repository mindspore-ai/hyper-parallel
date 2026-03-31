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

Verify the correctness of distributed operator class `infer_layout` logic, ensure coverage of various layout combinations and edge cases.

## Input

- **YAML Config**: Operator registration config from Workflow 3
- **Python Implementation**: Distributed operator class from Workflow 2

## Output

- **pytest Test File**: `tests/ut/core/shard/ops/test_parallel_*.py`


**Note**: **UT test cases are platform-agnostic**, verifying basic functionality of distributed operator class. **If there is no new Python distributed operator class implementation or modification, no need to write UT test cases** (e.g., Scenario 0 fully using base class).

---

## Step 1: Initialize the Test Architecture

All tests **must follow the architecture template below.**

Do **NOT copy layout construction or operator invocation patterns** from previous context.

Always start from this template.
```python
import os
import unittest
from unittest.mock import patch
import numpy as np
os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

from hyper_parallel.core.dtensor.dtensor import _build_layout
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from hyper_parallel.core.shard.ops.parallel_xxx import XxxDistributedOp
from hyper_parallel.platform import get_platform
from hyper_parallel.core.dtensor.device_mesh import (
    init_device_mesh,
    _DEVICE_MESH_MAP
)
from hyper_parallel.platform.platform import EXISTING_COMM_GROUPS

op = XxxDistributedOp("op_name")

class Test<OperatorName>(unittest.TestCase):
    """Unit tests for <OperatorName>DistributedOp."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Clear global state
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()
        self.platform = get_platform()
    
    def tearDown(self):
        """Clean up after each test method."""
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()
    
    def _setup_mock_platform(self, mock_platform, platform_type=None, world_size=8):
        """Configure common mock-platform attributes used across tests."""
        if platform_type is not None:
            mock_platform.platform_type = platform_type
        mock_platform.get_rank.return_value = 0
        mock_platform.get_world_size.return_value = world_size
        mock_platform.tensor_to_numpy.side_effect = (
            lambda t: t.numpy() if hasattr(t, "numpy") else np.array(t)
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

### Basic Functionality (Required)

| Test Scenario | Description | Code Example |
|---------------|-------------|--------------|
| **Same Layout** | Two inputs have same tensor_map | `x_placements = (Shard(0), Shard(1)), x_layout = _build_layout(mesh, x_placements, 2)` |
| **Different Layout** | Inputs sharded on different dimensions | `x_placements = (Shard(1), Shard(0)), x_layout = _build_layout(mesh, x_placements, 2)` |
| **Mixed Layout** | Some dimensions sharded, some replicated | `x_placements = (Shard(0), Replicate()), x_layout = _build_layout(mesh, x_placements, 2)` |
| **All Replicated** | Inputs not sharded on any dimension | `x_placements = (Replicate(), Replicate()), x_layout = _build_layout(mesh, x_placements, 2)` |

### Edge Cases (Required)

| Test Scenario | Description |
|---------------|-------------|
| **Scalar Input** | One input is scalar (layout is None) |
| **Negative Index** | Negative dimension index (e.g., -1 means last dimension) |
| **None Input** | Some operators may have None inputs |

### Broadcast Scenarios (Required for WithShape operators)

| Test Scenario | Description | Code Example |
|---------------|-------------|--------------|
| **Scalar Broadcast** | One input is scalar (layout is None) | `input_shapes = [(4, 256, 128), ()]` |
| **Shape Broadcast** | Different rank shapes | `input_shapes = [(16, 256, 128), (1, 256, 128)]` |
| **Rank Alignment** | Right-aligned tensor_map | (2, -1, -1) vs (-1, -1, -1) |

**Important Note**: Broadcast scenario tests **must** pass `input_shapes` parameter:

Example:
```python
extra_args = {"input_shapes": input_shapes}
output_layout = op.infer_layout((x_layout, y_layout), extra_args)
```

---

## Step 4: Partial State Validation (If Applicable)

| Test Scenario | Description |
|---------------|-------------|
| **Partial Input** | Input with partial state |
| **Partial Output** | Verify output partial state is set correctly |

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

- [ ] Basic Functionality

  - [ ] Same layout (DP/MP/Hybrid)
  - [ ] Different layout (different dimension sharding)
  - [ ] Mixed layout (sharded + replicated)
  - [ ] All replicated (no sharding)

- [ ] Edge Cases

  - [ ] Scalar input
  - [ ] Negative index
  - [ ] None input (if applicable)

- [ ] Broadcast Scenarios (only for WithShape operators)

  - [ ] Scalar broadcast (Scalar + Tensor)
  - [ ] Shape broadcast (different rank shapes)
  - [ ] Rank alignment (right-aligned tensor_map)

- [ ] Partial State (if applicable)

  - [ ] Input partial state
  - [ ] Output partial state setting

- [ ] Expand implementation validation (if `get_expand_impl` exists)

