# Workflow 5: Integration Testing (ST)

## Goal

Verify end-to-end distributed execution correctness in 8-card distributed environment, compare standalone vs distributed output results.

## Input

- **YAML Config**: Operator registration config from Workflow 3
- **Python Implementation**: Distributed operator class from Workflow 2
- **Operator Semantics**: Interface info from analysis report

## Output

- **ST Shell File**: `tests/mindspore/st/shard/test_ops_*.py` or `tests/torch/shard/ops/test_ops_*.py`
- **ST Implementation File**: `tests/mindspore/st/shard/*_shard_in_python.py` or `tests/torch/shard/ops/*.py`

---

## Test Environment

| Platform | Distributed Launch Command | Card Count Config |
|----------|---------------------------|-------------------|
| **MindSpore** | `msrun --worker_num=8` | 8 cards |
| **PyTorch** | `torchrun --nproc_per_node=8` | 8 cards |

---

## Test Dimensions

| Test Dimension | Check Item | Description |
|----------------|------------|-------------|
| **Functional Verification** | 8-card vs standalone | Compare Standalone and Parallel output |
| **Parallel Strategy** | DP/MP/Hybrid Parallel | Correctness of different parallel strategies |
| **Performance Verification** | Cache mechanism | Whether Layout inference cache is effective |
| **Decorator Standards** | @arg_mark marker | Compliant with CI scan requirements |

---

## Test Scenarios

| Test Scenario | Operator Examples | Required |
|---------------|-------------------|----------|
| **Data Parallel (DP)** | Any operator | ✅ Required |
| **Model Parallel (MP)** | Large model operators | ✅ Required |
| **Hybrid Parallel** | Linear layers | ✅ Required |
| **Broadcast Scenario** | Greater, Add, etc. | ⚙️ Required for WithShape operators |

---

## MindSpore ST Implementation

### Shell File Location

`tests/mindspore/st/shard/test_ops_greater.py`

### run_case Function Template

```python
import os
import shutil
from tests.common.mark_utils import arg_mark

def run_case(file_name, case_name, master_port):
    """Run test case."""
    file_base = os.path.splitext(file_name)[0]
    dir_to_remove = f"./{file_base}/{case_name}"
    if os.path.exists(dir_to_remove):
        shutil.rmtree(dir_to_remove)

    # MindSpore uses msrun to launch distributed
    cmd = (
        f"export GLOG_v=2 && "
        f"msrun --worker_num=8 --local_worker_num=8 "
        f"--master_addr=127.0.0.1 --master_port={master_port} "
        f"--join=True --log_dir=./{dir_to_remove}/msrun_log "
        f"pytest -s -v {file_name}::{case_name}"
    )
    ret = os.system(cmd)
    assert ret == 0, f"Test case {case_name} failed with return code {ret}"
```

### Test Case Template (with @arg_mark decorator)

```python
from hyper_parallel import DTensor, Layout, shard
from hyper_parallel.core.device_mesh import init_device_mesh
import mindspore as ms
from mindspore import Tensor

# Initialize device mesh
device_mesh = init_device_mesh((2, 2, 2), ("dp", "cp", "mp"))
layout = Layout(device_mesh.mesh_shape, device_mesh.alias_name, device_mesh.rank_list)

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
    np.testing.assert_allclose(standalone_output, parallel_output, rtol=rtol, atol=atol)
```

### Real Test Case Example

```python
@arg_mark(
    plat_marks=["platform_gpu"],  # Platform marker
    level_mark="level0",                  # Test level: level0/level1
    card_mark="allcards",                 # Card count marker
    essential_mark="essential"            # Essential marker
)
def test_greater_data_parallel():
    """
    Feature: Greater operator.
    Description: Test greater with data parallel.
    Expectation: Run success and match standalone result.
    """
    file_name = "greater_shard_in_python.py"
    case_name = "test_greater_data_parallel"
    master_port = 11400
    run_case(file_name, case_name, master_port)

    def test_greater_data_parallel():
        # Test logic
        pass
```

---

## PyTorch ST Implementation

### Shell File Location

`tests/torch/shard/ops/test_ops_elementwise.py`

### run_case Function Template

```python
import os
import shutil
from tests.common.mark_utils import arg_mark

def run_case(file_name, case_name, master_port):
    """Run test case."""
    file_base = os.path.splitext(file_name)[0]
    dir_to_remove = f"./{file_base}/{case_name}"
    if os.path.exists(dir_to_remove):
        shutil.rmtree(dir_to_remove)

    # PyTorch uses torchrun to launch distributed
    cmd = (
        f"torchrun --nproc_per_node=8 --master_addr=127.0.0.1 "
        f"--master_port={master_port} "
        f"pytest -s -v {file_name}::{case_name}"
    )
    ret = os.system(cmd)
    assert ret == 0, f"Test case {case_name} failed with return code {ret}"
```

### Test Case Example

```python
import torch
import torch.distributed as dist

def test_add_data_parallel():
    """Test add operator with data parallel."""
    # Initialize distributed environment
    dist.init_process_group("nccl")
    rank = dist.get_rank()

    # Create test data
    x = torch.randn(4, 4).to(f"cuda:{rank}")
    y = torch.randn(4, 4).to(f"cuda:{rank}")

    # Configure sharding
    from hyper_parallel import Layout, DTensor
    layout = Layout((2, 2, 2), ("dp", "cp", "mp"), list(range(8)))
    x_dtensor = DTensor.from_local(x, layout("dp", "None", "None"))
    y_dtensor = DTensor.from_local(y, layout("dp", "None", "None"))

    # Execute distributed operator
    result = torch.add(x_dtensor, y_dtensor)

    # Compare with standalone result
    if rank == 0:
        x_standalone = x_dtensor.to_local().clone()
        y_standalone = y_dtensor.to_local().clone()
        expected = torch.add(x_standalone, y_standalone)
        assert torch.allclose(result.to_local(), expected)
```

---

## Reference Test Files

| Reference Type | File Location | Description |
|----------------|---------------|-------------|
| **ST Shell File** | `tests/mindspore/st/shard/test_ops_xxx.py` | Test entry, config parameters |
| **ST Implementation File** | `tests/mindspore/st/shard/xxx_shard_in_python.py` | Actual test logic |
| **PyTorch ST** | `tests/torch/shard/ops/parallel_op_*.py` | PyTorch platform ST examples |

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
- `card_mark` must be `allcards` or specific card count

---

## Success Criteria

- [ ] Created ST Shell file (test_ops_*.py)
- [ ] Created ST Implementation file (*_shard_in_python.py)
- [ ] Used @arg_mark decorator to mark test cases
- [ ] Implemented compare_results comparison function
- [ ] 8-card environment test passed
- [ ] Standalone vs distributed output consistent (rtol=1e-5, atol=1e-8)
- [ ] Covered required test scenarios (DP/MP/Broadcast)

---

## Next Step

After ST tests pass, proceed to **[Workflow 6: Git Commit and PR Creation](./06-git-commit.md)**

**Input:** All modified code, operator name
**Goal:** Create feature branch, complete lint check, commit, push, and create PR if needed
