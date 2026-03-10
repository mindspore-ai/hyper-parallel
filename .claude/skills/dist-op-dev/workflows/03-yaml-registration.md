# Workflow 3: YAML Registration

## Goal

Register operator in YAML config file, configure `infer_layout_suffix`, so the framework can correctly dispatch to distributed operator implementation.

## Input

- **Analysis Report**: Analysis report output from Step 1
- **Python Implementation Class Info**: Class name, file name, base class, broadcast support

## Output

- **YAML Config Entry**: Operator registration item in `hyper_parallel/core/shard/ops/yaml/*.yaml`

---

## infer_layout_suffix Configuration

| Suffix Value | Corresponding Method | extra_args Content | Usage Scenario |
|--------------|---------------------|-------------------|----------------|
| **(no suffix)** | `_with_layout_infer` | `[scalar/non-tensor args]` | Simple operators (no broadcast) |
| **`WithShape`** | `_with_layout_infer_with_shape` | `[..., input_shapes]` | **Operators supporting broadcast** ⭐ |
| `Reshape` | `_with_layout_infer_reshape` | `[target_shape, input_shape]` | Shape transformation |
| `Slice` | `_with_layout_infer_slice` | `[begin, end, global_shape]` | Slice operations |

> **Important Note**: If the operator supports broadcasting (scalar broadcast, shape broadcast), **must** configure `infer_layout_suffix: WithShape`. Otherwise ST test broadcast scenarios will fail with error, unable to get `input_shapes` info.

---

## Configuration Scenarios

### Scenario A: Basic Registration (No Broadcast Support)

**Configuration Example**:
```yaml
ReLU:
  dist_op_name: _relu_dist_op
  distributed_op_class: ElementWiseDistributedOp
  distributed_op_file: parallel_elementwise
```

### Scenario B: Support Broadcast (WithShape) ⭐ Most Common

**Configuration Example**:
```yaml
Greater:
  dist_op_name: _greater_dist_op
  distributed_op_class: ElementWiseDistributedOp
  distributed_op_file: parallel_elementwise
  infer_layout_suffix: WithShape
```

### Scenario C: Shape Transformation (Reshape)

**Configuration Example**:
```yaml
Reshape:
  dist_op_name: _reshape_dist_op
  distributed_op_class: ReshapeDistributedOp
  distributed_op_file: parallel_reshape
  infer_layout_suffix: Reshape
```

### Scenario D: Slice Operation (Slice)

**Configuration Example**:
```yaml
Slice:
  dist_op_name: _slice_dist_op
  distributed_op_class: SliceDistributedOp
  distributed_op_file: parallel_slice
  infer_layout_suffix: Slice
```

### Scenario E: Fully Use Base Class (Recommended)

**Configuration Example**:
```yaml
__default__:
  distributed_op_class: ElementWiseDistributedOp
  distributed_op_file: parallel_elementwise

Mish:
  dist_op_name: _mish_dist_op

ReLU:
  dist_op_name: _relu_dist_op
```

---

## Platform Differences

### MindSpore YAML Files

| YAML File | Applicable Operator Type | Operator Name Style |
|-----------|-------------------------|---------------------|
| `element_wise_ops.yaml` | Element-wise operations (no broadcast) | PascalCase |
| `element_wise_ops_with_shape.yaml` | Element-wise operations supporting broadcast | PascalCase |
| `reduce_ops.yaml` | Reduction operations | PascalCase |
| `reshape_ops.yaml` | Shape transformations | PascalCase |
| `matmul_ops.yaml` | Matrix multiplication | PascalCase |
| `gather_ops.yaml` | Index operations | PascalCase |

### PyTorch YAML Files

| YAML File | Applicable Operator Type | Operator Name Style |
|-----------|-------------------------|---------------------|
| `torch_element_wise.yaml` | Element-wise operations (no broadcast) | snake_case |
| `torch_element_wise_with_shape.yaml` | Element-wise operations supporting broadcast | snake_case |
| `torch_reduce.yaml` | Reduction operations | snake_case |
| `torch_mm.yaml` | Matrix multiplication | snake_case |
| `torch_reshape.yaml` | Shape transformations | snake_case |
| `torch_squeeze.yaml` | Squeeze dimension | snake_case |
| `torch_unsqueeze.yaml` | Unsqueeze dimension | snake_case |

---

## Success Criteria

- [ ] Selected correct YAML file (by operator category)
- [ ] Configured all required fields (operator name, class name, file name)
- [ ] Configured correct `infer_layout_suffix` (WithShape/no suffix/Reshape, etc.)
- [ ] Operators supporting broadcast configured with `WithShape` suffix
- [ ] PyTorch operators use snake_case, MindSpore operators use PascalCase
- [ ] Cross-platform reuse operators point to same implementation class (if applicable)

---

## Next Step

After YAML registration is complete, proceed to **[Workflow 4: Unit Testing (UT)](./04-unit-testing.md)**

**Input:** Python implementation class
**Goal:** Verify `infer_layout` logic correctness, cover various layout combinations
