# Workflow 5: Integration Testing (ST)

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

Verify end-to-end distributed execution correctness in 8-card distributed environment, compare standalone vs distributed output results.

## Input

- **YAML Config**: Operator registration config from Workflow 3
- **Python Implementation**: Distributed operator class from Workflow 2
- **Operator Semantics**: Interface info from analysis report

## Output

- **ST Shell File**: `tests/mindspore/st/shard/ops/test_*_shard_in_python.py` or `tests/torch/shard/ops/test_ops_*.py`
- **ST Implementation File**: `tests/mindspore/st/shard/ops/*_shard_in_python.py` or `tests/torch/shard/ops/*.py`

---

## Step 1: Configure Distributed Test Environment

ST tests must run in an **8-card distributed environment.**

**MindSpore and Pytorch Launch Command**:
| Platform | Distributed Launch Command | Card Count Config |
|----------|---------------------------|-------------------|
| **MindSpore** | `MindsporeCase` | 8 cards |
| **PyTorch** | `TorchCase` | 8 cards |

If the local environment cannot run 8-card tests, still **create or update the ST files**

---

## Step 2: Define Test Dimensions

ST tests must cover:

| Test Dimension | Check Item | Description |
|----------------|------------|-------------|
| **Functional Verification** | 8-card vs standalone | Compare Standalone and Parallel output |
| **Parallel Strategy** | DP/MP/Hybrid Parallel | Correctness of different parallel strategies |
| **Performance Verification** | Cache mechanism | Whether Layout inference cache is effective |
| **Decorator Standards** | @arg_mark marker | Compliant with CI scan requirements |

---

## Step 3: Define Test Scenarios

| Test Scenario | Operator Examples | Required |
|---------------|-------------------|----------|
| **Data Parallel (DP)** | Any operator | ✅ Required |
| **Model Parallel (MP)** | Large model operators | ✅ Required |
| **Hybrid Parallel** | Linear layers | ✅ Required |
| **Broadcast Scenario** | Greater, Add, etc. | ⚙️ Required for WithShape operators |

---

## Step 4: Implement ST Test Case

Each test case should follow the structure below:
1. Initialize distributed environment
2. Configure device mesh
3. Configure tensor layout
4. Execute distributed operator
5. Compare standalone and distributed outputs

### Mindspore Example

**Test case Example**
```python
import mindspore as ms
import mindspore.communication.management as D
from mindspore import Tensor, ops
from hyper_parallel import init_device_mesh, shard_module
from hyper_parallel.core.dtensor import distribute_tensor
from hyper_parallel.core.placement_types import Shard, Replicate
from hyper_parallel.core.shard.sharding_plan import ShardingPlan

# Initialize device mesh
mesh = init_device_mesh(
    device_type="npu",
    mesh_shape=(2, 4),
    mesh_dim_names=("dp", "mp")
)

def compare_results(standalone_output, parallel_output, rtol=1e-5, atol=1e-8):
    """
    Compare standalone and parallel outputs.

    Args:
        standalone_output: Output from standalone execution.
        parallel_output: Output from distributed execution.
        rtol: Relative tolerance.
        atol: Absolute tolerance.

    Raises:
        AssertionError: If outputs do not match.
    """
    if isinstance(standalone_output, Tensor):
        standalone_output = standalone_output.asnumpy()
    if isinstance(parallel_output, Tensor):
        parallel_output = parallel_output.asnumpy()

    import numpy as np
    assert np.allclose(standalone_output, parallel_output, rtol=rtol, atol=atol), \
        f"Standalone output:\n{standalone_output}\nParallel output:\n{parallel_output}"
```

### Pytorch Example:

**Test case Example**
```python
import torch
from tests.torch.utils import init_dist
from tests.torch.shard.utils import local_to_global, global_to_local

def test_add_data_parallel():
    """Test add operator with data parallel."""
    # Initialize distributed environment
    init_dist()

    # Configure sharding
    from hyper_parallel.core.dtensor import _build_layout
    from hyper_parallel import init_device_mesh
    from hyper_parallel.core.placement_types import Shard, Replicate
    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))
    x_placements = (Shard(0), Replicate())
    x_layout = _build_layout(mesh, x_placements, 2)

    # Execute distributed operator
    dist_input = global_to_local(standalone_input, x_layout)
    dist_output = torch.repeat_interleave(dist_input, repeats, dim=dim)

    # Compare with standalone result
    # Layout correctness check
    assert dist_output.layout == x_layout, "output layout mismatch input"
    # output correctness check
    gathered_output = local_to_global(dist_output)
    assert torch.allclose(
        standalone_output, gathered_output, atol=1e-5
    ), "output mismatch between standalone and distributed"
```

---

## Step 5: Implement ST Shell File

Every ST test must use the **@arg_mark decorator**.

### Mindspore Example
```python
from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import parallel_run, MindSporeCase

IMPLEMENTATION_FILE = "greater_shard_in_python.py"

@arg_mark(
    plat_marks=["platform_gpu"],   # Platform marker
    level_mark="level1",                  # Test level: level0/level1
    card_mark="allcards",                 # Card count marker
    essential_mark="essential"            # Essential marker
)
def test_greater_data_parallel():
    """
    Feature: Greater operator.
    Description: Test greater with data parallel.
    Expectation: Run success and match standalone result.
    """
    parallel_run([
        MindSporeCase(IMPLEMENTATION_FILE, "test_greater_data_parallel", 11500, 8, 8, 2),
    ])
```

### Pytorch Example
```python
from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import parallel_run, TorchCase

PARALLEL_OP_MAX = "parallel_op_repeat_interleave.py"

@arg_mark(
    plat_marks=["platform_gpu"],   # Platform marker
    level_mark="level1",                  # Test level: level0/level1
    card_mark="allcards",                 # Card count marker
    essential_mark="essential"            # Essential marker
)
def test_distributed_repeat_interleave_layout_inference():
    """
    Feature: test parallel op repeat_interleave.
    Description: test parallel op repeat_interleave.
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(PARALLEL_OP_MAX, "test_distributed_repeat_interleave_layout_inference", 10500, 8),
    ])
    
```

---

## Reference Test Files

| Reference Type | File Location | Description |
|----------------|---------------|-------------|
| **ST Shell File** | `tests/mindspore/st/shard/test_one_hot_shard_in_python.py` | Test entry, config parameters |
| **ST Implementation File** | `tests/mindspore/st/shard/one_hot_ext_shard_in_python.py` | Actual test logic |
| **PyTorch ST** | `tests/torch/shard/ops/parallel_op_repeat_interleave.py` | PyTorch platform ST examples |
| **PyTorch ST Shell File** | `tests/torch/shard/ops/test_parallel_op_repeat_interleave.py` | PyTorch Test entry, config parameters |

---

## ST Coverage Checklist

Before submitting, ensure the following test scenarios are covered:

- [ ] **Functional Verification**
  - [ ] 8-card environment execution successful
  - [ ] Standalone vs distributed output consistent
  - [ ] Edge cases handled correctly

- [ ] **Parallel Strategy**
  - [ ] Data Parallel (DP) test passed
  - [ ] Model Parallel (MP) test passed
  - [ ] Hybrid Parallel test passed (if applicable)

- [ ] **Broadcast Scenario** (operators supporting WithShape)
  - [ ] Scalar broadcast test passed
  - [ ] Shape broadcast test passed

- [ ] **Decorator Standards**
  - [ ] Using @arg_mark decorator
  - [ ] plat_marks set correctly
  - [ ] level_mark set correctly
  - [ ] card_mark set correctly
  - [ ] essential_mark set if needed

---

## Common Issues

**Q1: ST test failed with "device not found"?**

A: Check environment configuration:
- Confirm device visibility: `export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`
- Confirm sufficient device count (at least 8 cards)

**Q2: Standalone vs distributed output inconsistent?**

A: Troubleshooting steps:
1. Check if `infer_layout` inference logic is correct
2. Check if communication operators (AllReduce/AllGather) are missing
3. Check if Partial state handling is correct
4. Check if inputs are using correct Layout

**Q3: Broadcast scenario test failed?**

A: Check configuration:
1. Confirm `infer_layout_suffix: WithShape` is configured in YAML
2. Confirm `input_shapes` parameter is passed in ST test
3. Confirm `input_shapes` is handled correctly in `infer_layout`

**Q4: @arg_mark marker error causing CI failure?**

A: Check markers:
- `plat_marks` must be valid platform name (e.g., `platform_gpu`)
- `level_mark` must be `level0` or `level1`
- `card_mark` must be `allcards` or `onecard`

---

## Success Criteria

- [ ] Created ST Shell file (test_ops_*.py)
- [ ] Created ST Implementation file (*_shard_in_python.py)
- [ ] Used @arg_mark decorator to mark test cases
- [ ] Implemented compare_results comparison function
- [ ] 8-card environment test passed
- [ ] Standalone vs distributed output consistent (rtol=1e-5, atol=1e-8)
- [ ] Covered required test scenarios (DP/MP/Broadcast)

