# Workflow 4: Unit Testing (UT)

## Goal

Verify the correctness of distributed operator class `infer_layout` logic, ensure coverage of various layout combinations and edge cases.

## Input

- **Python Implementation Class**: Distributed operator class from Workflow 2

## Output

- **pytest Test File**: `tests/mindspore/ut/parallel_ops_infer/test_parallel_*.py`

**Note**: **UT test cases are platform-agnostic**, verifying basic functionality of distributed operator class. **If there is no new Python distributed operator class implementation or modification, no need to write UT test cases** (e.g., Scenario 0 fully using base class).

---

## Test Architecture Configuration

```python
import pytest
from hyper_parallel import Layout

# If operator fully uses base class
from hyper_parallel.core.shard.ops.parallel_elementwise import ElementWiseDistributedOp

base_mesh_shape = (2, 2, 2)
base_alias_name = ("dp", "cp", "mp")
base_rank_list = list(range(8))
```

---

## Test Scenario Coverage

### Basic Functionality (Required)

| Test Scenario | Description | Code Example |
|---------------|-------------|--------------|
| **Same Layout** | Two inputs have same tensor_map | `layout("dp", "cp", "mp")` |
| **Different Layout** | Inputs sharded on different dimensions | `x_layout = layout("dp", "None", "mp")` |
| **Mixed Layout** | Some dimensions sharded, some replicated | `layout("dp", "None", "None")` |

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

```python
extra_args = {"input_shapes": input_shapes}
output_layout = op.infer_layout((x_layout, y_layout), extra_args)
```

### Tuple Alias (Recommended)

| Test Scenario | Description | Code Example |
|---------------|-------------|--------------|
| **Composite Sharding** | One dimension sharded across multiple device axes | `layout(("dp", "cp"), "None")` |

### Partial State (If Applicable)

| Test Scenario | Description |
|---------------|-------------|
| **Partial Input** | Input with partial state |
| **Partial Output** | Verify output partial state is set correctly |

---

## UT Coverage Checklist

- [ ] Basic Functionality

  - [ ] Same layout (DP/MP/Hybrid)
  - [ ] Different layout (different dimension sharding)
  - [ ] Mixed layout (sharded + replicated)

- [ ] Edge Cases

  - [ ] Scalar input
  - [ ] Negative index
  - [ ] None input (if applicable)

- [ ] Broadcast Scenarios (only for WithShape operators)

  - [ ] Scalar broadcast (Scalar + Tensor)
  - [ ] Shape broadcast (different rank shapes)
  - [ ] Rank alignment (right-aligned tensor_map)

- [ ] Tuple Alias Composite Patterns

  - [ ] `(("dp", "mp"), "None")` and similar scenarios

- [ ] Partial State (if applicable)

  - [ ] Input partial state
  - [ ] Output partial state setting

---

## Next Step

After UT tests pass, proceed to **[Workflow 5: Integration Testing (ST)](./05-integration-testing.md)**

**Input:** YAML config, operator semantics
**Goal:** Verify end-to-end distributed execution correctness in 8-card environment
