---
name: distributed-op-testing
description: Testing conventions for HyperParallel distributed operators. Covers UT/ST test patterns, assertion styles, and cross-platform testing for MindSpore and PyTorch.
paths:
  - tests/ut/core/shard/ops/test_parallel_*.py
  - tests/mindspore/st/shard/ops/*.py
  - tests/torch/shard/ops/parallel_op_*.py
  - tests/torch/shard/ops/test_parallel_op_*.py
---

# HyperParallel Distributed Operator Testing Guide

Testing conventions for distributed operator development in HyperParallel. Covers Unit Tests (UT), Integration Tests (ST), and cross-platform testing patterns.

> **Note**: For f-string formatting, line length limits, and other assertion style rules, See [test-assertion-style.md](test-assertion-style.md) for complete guidelines.

## Unit Test (UT) Conventions

### File Location

- **Path**: `tests/ut/core/shard/ops/test_parallel_{operator_name}.py`
- **Platform-agnostic**: Tests run without actual distributed communication or GPU/NPU

### Test Framework

**IMPORTANT: Use Python's built-in `unittest` framework, NOT pytest.**

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

### Assertion Style

**Always use direct property access and include error messages:**

```python
# Correct
assert output_layout.tensor_map == expected_map, (
    f"Data Parallel test failed. Expected {expected_map}, "
    f"got {output_layout.tensor_map}"
)

# Wrong - no error message
assert output_layout.tensor_map == expected_map

# Wrong - using dict access
assert output_layout.to_dict()["tensor_map"] == expected_map
```

### Mock Platform Pattern

**Use `@patch` decorator to mock the platform for unit tests:**

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

    placements = (Shard(0), Replicate())
    x_layout = _build_layout(mesh, placements, 2)

    output_layout = self.op.infer_layout((x_layout,), extra_args)

    # Verify tensor_map
    assert output_layout.tensor_map == (1, -1), f"..."

    # Verify partial state (if applicable)
    assert output_layout.is_partial(), f"..."
    assert output_layout.partial == ["sum", None], f"..."
```

### Helper Methods for Mesh Creation

**Use helper methods instead of pytest fixtures to avoid shadowing warnings:**

```python
def _make_2x2x2_mesh(self, mock_platform):
    """Set up mock and return a standard 2x2x2 (dp, sp, mp) mesh."""
    self._setup_mock_platform(mock_platform, world_size=8)
    return init_device_mesh(device_type="npu", mesh_shape=(2, 2, 2), mesh_dim_names=("dp", "sp", "mp"))

def _make_4x2_mesh(self, mock_platform):
    """Set up mock and return a standard 4x2 (dp, mp) mesh."""
    self._setup_mock_platform(mock_platform, world_size=8)
    return init_device_mesh(device_type="npu", mesh_shape=(4, 2), mesh_dim_names=("dp", "mp"))

def _make_8_mesh(self, mock_platform, mesh_dim_name="dp"):
    """Set up mock and return a standard 8-element mesh."""
    self._setup_mock_platform(mock_platform, world_size=8)
    return init_device_mesh(device_type="npu", mesh_shape=(8,), mesh_dim_names=(mesh_dim_name,))
```

### Required Test Scenarios

| Scenario | Description | Example Test Name |
|----------|-------------|-------------------|
| **Data Parallel** | Batch dimension sharded | `test_*_data_parallel_success` |
| **Model Parallel** | Feature dimension sharded | `test_*_model_parallel_success` |
| **Hybrid Parallel** | Multiple dimensions sharded | `test_*_hybrid_parallel_success` |
| **All Replicated** | No sharding | `test_*_all_replicated` |
| **Negative Dim** | Negative dimension index | `test_*_negative_dim` |
| **Partial State** | Input with partial status | `test_*_partial_input` |
| **Multi-axis** | Tuple of axes | `test_*_multi_axis_tuple` |
| **3D Tensor** | Higher dimensional tensors | `test_*_3d_tensor` |
| **Error Cases** | Invalid inputs, sharded dim | `test_*_sharded_dim_failure` |
| **get_expand_impl** | Verify expand implementation | `test_*_get_expand_impl` |

### Error Testing Pattern

Use `self.assertRaisesRegex` to assert exception type with matching error message:

```python
def test_activation_with_axis_sharded_dim_failure(self, op):
    """Test error when computing on sharded dimension."""
    with self.assertRaisesRegex(ValueError, "requires the reduction axis to be un-sharded"):
        op.infer_layout((x_layout,), (0,))

def test_activation_with_axis_partial_input(self, op):
    """Test error for partial input when not allowed."""
    x_layout.set_partial_by_dev_axis("dp", "sum")
    with self.assertRaisesRegex(ValueError, "has Partial status which is not allowed"):
        op.infer_layout((x_layout,), (1,))
```

### get_expand_impl Verification

**IMPORTANT: Only verify `get_expand_impl` once per test class if the operator does NOT override it.**

If the operator uses the default implementation (returns `None`), verify it in **only one test case** with a comment explaining why other tests don't need this verification.
If the operator has a custom `get_expand_impl` that returns a callable, verify it in **every test case**:

The `get_expand_impl` method returns either:

- `None`: No expand implementation needed (most operators)
- `callable`: A function that handles distributed execution (e.g., Gather, MatMul with partial state)

**When impl should be None:**

For operators that do NOT need expand implementation, use **direct assertion without variable assignment**:

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

**When impl should be callable:**

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

---

## Integration Test (ST) Common Conventions

### Level Mark Guidelines

**ST tests use `level_mark` to control test execution priority:**

| Level | Description | When to Use |
|-------|-------------|-------------|
| **level0** | CI gate tests | Key operators **AND** key scenarios (Data Parallel, Model Parallel) |
| **level1** | Extended tests | Non-key operators OR non-key scenarios |

**Rules for level assignment:**

1. **Use `level0` for: Key operators AND key scenarios (both conditions must be met)**
   - Key operators: MatMul, Softmax, LayerNorm, etc.
   - Key scenarios: Data Parallel, Model Parallel, Hybrid Parallel
   - Example: LayerNorm + Data Parallel → level0 ✅

2. **Use `level1` for: Non-key operators OR non-key scenarios**
   - Non-key operators: any operator not in key operator list
   - Non-key scenarios: All Replicated, Negative Dim, 3D Tensor
   - Example: LayerNorm + All Replicated → level1 (non-key scenario)
   - Example: CustomOp + Data Parallel → level1 (non-key operator)

### Error Cases in UT Only

**IMPORTANT: Error/exception test cases should ONLY be in Unit Tests (UT), NOT in Integration Tests (ST).**

| Test Type | Purpose | Error Cases? |
|-----------|---------|--------------|
| **UT** | Logic verification, error handling | ✅ Yes - all error cases |
| **ST** | End-to-end distributed execution | ❌ No - only success cases |

**Reasons:**

- UT runs without distributed environment, faster and more reliable for error detection
- ST requires 8-card environment, expensive resource for error cases
- Error handling is logic-level, not distributed-level

**Examples:**

```python
# ❌ Wrong: Error case in ST
# File: tests/mindspore/st/shard/activation_with_axis_shard_in_python.py
def test_activation_with_axis_error():  # DO NOT add error cases in ST
    with pytest.raises(ValueError):
        ...
```

### Combining Multiple Operators

**Combine operators with same distributed implementation in one test function:**

```python
def test_activation_with_axis_data_parallel_1():
    """Test both Softmax and Swiglu in data parallel."""
    # Setup mesh and placements...

    # Test Softmax
    x_softmax = Tensor(np.random.randn(d, m, k).astype(np.float32))
    # ... softmax test code ...

    # Test Swiglu
    x_swiglu = Tensor(np.random.randn(d, m, k).astype(np.float32))
    # ... swiglu test code ...
```

### Required Test Scenarios

| Scenario | Level | Description |
|----------|-------|-------------|
| **Data Parallel** | level0 (key op) / level1 (non-key op) | Batch dimension sharded |
| **Model Parallel** | level0 (key op) / level1 (non-key op) | Feature dimension sharded |
| **Hybrid Parallel** | level0 (key op) / level1 (non-key op) | Multiple dimensions sharded |
| **All Replicated** | level1 | No sharding |
| **Negative Dim** | level1 | Negative dimension index |

---

## Integration Test (ST) - MindSpore Specifics

### File Structure

| File | Location | Purpose |
|------|----------|---------|
| **Runner File** | `tests/mindspore/st/shard/ops/test_parallel_op_*.py` | Test runner with `@arg_mark` decorators |
| **Implementation File** | `tests/mindspore/st/shard/ops/_test_parallel_op_*.py` | Actual test implementation |

### Runner File Pattern

```python
from pathlib import Path

from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import parallel_run, MindSporeCase

IMPL_FILE = str(Path(__file__).resolve().parent / "_test_parallel_op_activation_with_axis.py")

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_activation_with_axis_group1():
    """
    Feature: parallel run case in _test_parallel_op_activation_with_axis
    Description:
        1. test_activation_with_axis_data_parallel_1 — Softmax + Swiglu data parallel
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(IMPL_FILE, "test_activation_with_axis_data_parallel_1", worker_num=8, local_worker_num=8, glog_v=2),
    ])
```

### Implementation File Pattern

```python
import numpy as np
import mindspore as ms
import mindspore.communication.management as D
from mindspore import nn, Tensor, ops
from hyper_parallel import init_device_mesh, shard_module
from hyper_parallel.core.dtensor import distribute_tensor
from hyper_parallel.core.placement_types import Shard, Replicate
from hyper_parallel.core.shard.sharding_plan import ShardingPlan

def setup_module():
    ms.set_device("Ascend")
    D.init()

base_mesh_shape = (2, 2, 2)
base_alias_name = ("dp", "cp", "tp")

class SoftmaxNet(nn.Cell):
    def __init__(self, axis=-1, device_mesh=None, relu_strategy=None):
        super().__init__()
        self.softmax = ops.Softmax(axis)
        self.relu = ms.nn.ReLU()
        if relu_strategy is not None and device_mesh is not None:
            sharding_plan = ShardingPlan(input_plan={"input": relu_strategy})
            shard_module(self.relu, device_mesh=device_mesh, sharding_plan=sharding_plan)

    def construct(self, x):
        out = self.softmax(x)
        out = self.relu(out)
        out = out + 1
        return out

def test_activation_with_axis_data_parallel_1():
    """
    Feature: ActivationWithAxis (Softmax and Swiglu) in python shard.
    Description: Test with data parallel.
    Expectation: Run success.
    """
    ms.set_seed(1)
    np.random.seed(1)

    mesh = init_device_mesh(device_type="npu", mesh_shape=base_mesh_shape, mesh_dim_names=base_alias_name)
    x_placements = (Shard(0), Replicate(), Replicate())
    relu_placements = (Shard(0), Replicate(), Replicate())

    # Test Softmax
    d, m, k = 8, 16, 32
    x_softmax = Tensor(np.random.randn(d, m, k).astype(np.float32))
    standalone_softmax_net = SoftmaxNet(axis=-1)
    standalone_softmax_output = standalone_softmax_net(x_softmax)
    x_softmax_local = distribute_tensor(x_softmax, mesh, x_placements)
    parallel_softmax_net = SoftmaxNet(axis=-1, device_mesh=mesh, relu_strategy=relu_placements)
    parallel_softmax_output = parallel_softmax_net(x_softmax_local)
    parallel_softmax_output = parallel_softmax_output.full_tensor()
    assert np.allclose(standalone_softmax_output.asnumpy(), parallel_softmax_output.asnumpy(), 1e-3, 1e-3), \
        f"Softmax data parallel test failed"

    # Test Swiglu (can be combined in same function)
    # ...
```

---

## Integration Test (ST) - PyTorch Specifics

### File Structure

| File | Location | Purpose |
|------|----------|---------|
| **Runner File** | `tests/torch/shard/ops/test_parallel_op_*.py` | Test runner with `@arg_mark` decorators |
| **Implementation File** | `tests/torch/shard/ops/_test_parallel_op_*.py` | Actual test implementation |

### Runner File Pattern

```python
from pathlib import Path

from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import parallel_run, TorchCase

IMPL_FILE = str(Path(__file__).resolve().parent / "_test_parallel_op_max.py")

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_max_group1():
    """
    Feature: parallel run case in _test_parallel_op_max
    Description:
        1. test_max_element_wise — max element-wise
        2. test_max_dim_reduce_sharded — max dim reduce sharded
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(IMPL_FILE, "test_max_element_wise", num_proc=4),
        TorchCase(IMPL_FILE, "test_max_dim_reduce_sharded", num_proc=4),
    ])

@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_parallel_op_max_group1_gloo():
    """
    Feature: parallel run case in _test_parallel_op_max
    Description:
        1. test_max_element_wise — max element-wise
        2. test_max_dim_reduce_sharded — max dim reduce sharded
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(IMPL_FILE, "test_max_element_wise", num_proc=4),
        TorchCase(IMPL_FILE, "test_max_dim_reduce_sharded", num_proc=4),
    ])
```

### Implementation File Pattern

```python
import numpy as np
import torch
from hyper_parallel import init_device_mesh
from hyper_parallel.core.dtensor import _build_layout, distribute_tensor
from hyper_parallel.core.placement_types import Shard, Replicate
from tests.torch.utils import init_dist
from tests.torch.shard.utils import local_to_global

np.random.seed(42)
standalone_input_2d_np = np.random.randn(8, 16).astype(np.float32)

def test_distributed_softmax_data_parallel():
    """
    Feature: dtensor + torch.nn.functional.softmax with data parallel
    Description:
        - Input: (8, 16) sharded on dim 0 (batch dimension).
        - Softmax on dim 1 (feature dimension, unsharded).
    Expectation: Success with correct layout and numerical equivalence.
    """
    init_dist()

    standalone_input = torch.from_numpy(standalone_input_2d_np).npu()
    standalone_output = torch.nn.functional.softmax(standalone_input, dim=1)

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    x_placements = (Shard(0), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_output = torch.nn.functional.softmax(dist_input, dim=1)

    # Verify layout
    expected_layout = _build_layout(mesh, (Shard(0), Replicate()), 2)
    assert dist_output.layout == expected_layout, \
        f"Softmax data parallel layout mismatch: expected {expected_layout}, got {dist_output.layout}"

    # Verify numerical equivalence
    gathered_output = local_to_global(dist_output)
    assert torch.allclose(standalone_output, gathered_output, rtol=1e-5, atol=1e-5), \
        f"Softmax data parallel output mismatch"
```

### PyTorch API Usage

```python
# Use torch.nn.functional.softmax or torch.softmax
dist_output = torch.nn.functional.softmax(dist_input, dim=1)
dist_output = torch.softmax(dist_input, dim=-1)

# Use .npu() for NPU device
standalone_input = torch.from_numpy(input_np).npu()

# Use local_to_global to gather distributed output
gathered_output = local_to_global(dist_output)
```

---

## Test Coverage Checklist

### For Each Distributed Operator

- [ ] **UT Tests** (All error cases go here)
  - [ ] Data Parallel success case
  - [ ] Model Parallel success case
  - [ ] Hybrid Parallel success case
  - [ ] All Replicated case
  - [ ] Negative dimension index
  - [ ] Multi-axis tuple (if applicable)
  - [ ] 3D tensor cases
  - [ ] **Sharded dimension error case** ⚠️ UT only
  - [ ] **Input consistency error case** ⚠️ UT only
  - [ ] **Invalid extra args error case** ⚠️ UT only
  - [ ] **Partial state error case** ⚠️ UT only
  - [ ] `get_expand_impl` verification

- [ ] **ST Tests (MindSpore)** (Success cases only, no error cases)
  - [ ] Shell file with `@arg_mark` decorators
  - [ ] Implementation file with setup_module
  - [ ] Data Parallel test (level0 for key operators)
  - [ ] Model Parallel test (level0 for key operators)
  - [ ] Hybrid Parallel test (level0 for key operators)
  - [ ] All Replicated test (level1)
  - [ ] Negative dimension test (level1)
  - [ ] Multiple operators combined (if applicable)

- [ ] **ST Tests (PyTorch)** (Success cases only, no error cases)
  - [ ] Implementation file with `init_dist`
  - [ ] Layout verification
  - [ ] Numerical equivalence verification

---

## Common Mistakes to Avoid

| Mistake | Correct Approach |
|---------|------------------|
| `output_layout.to_dict()["tensor_map"]` | `output_layout.tensor_map` |
| No error message in assert | Always include descriptive error message |
| Separate test functions for similar operators | Combine in one function with fixture or inline |
| Missing layout verification in ST | Always verify both layout and numerical equivalence |
| **Error cases in ST** | **Error cases ONLY in UT** |
| Using `level0` for non-key operator OR non-key scenario | `level0` requires **key operator AND key scenario** |
| Non-key scenarios with `level0` | All Replicated, Negative Dim, 3D Tensor → `level1` |
| Non-key operator with `level0` | Non-key operators always use `level1` |
