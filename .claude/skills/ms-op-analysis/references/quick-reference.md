# Quick Reference

This document provides common commands and interface quick reference for MindSpore operator distributed implementation analysis.

---

## Common Commands Quick Reference

### File Search

```bash
# Find YAML file
ls mindspore/ops/op_def/yaml/{op_name}_op.yaml

# Find Info class file
find . -name "*{op_name}_info.cc"

# Find base class definition
find . -name "operator_info.h"
find . -name "arithmetic_info.h"

# List all Info class files
ls mindspore/ccsrc/frontend/parallel/ops_info/*.cc
```

### Content Search

```bash
# Search for specific function
grep -r "InferTensorInfo" mindspore/ccsrc/frontend/parallel/ops_info/{op}.cc

# Search for specific class
grep -r "class {Op}Info" mindspore/ccsrc/frontend/parallel/ops_info/

# Search for operator registration
grep -r "REGISTER({Op}Info)" mindspore/ccsrc/
```

### Path Validation

```bash
# Validate YAML directory
ls mindspore/ops/op_def/yaml/ | head -5

# Validate distributed operator directory
ls mindspore/ccsrc/frontend/parallel/ops_info/ | head -5

# Validate generated directory
ls mindspore/python/mindspore/ops/auto_generate/
```

---

## Info Class Core Methods Quick Reference

### Layout Flow Methods

| Method | Description | Return Value |
|--------|-------------|--------------|
| `InferTensorInfo()` | Infer input/output `TensorInfo` | `Status` |
| `InferInputTensorInfo()` | Infer input `TensorInfo` | `Status` |
| `InferOutputTensorInfo()` | Infer output `TensorInfo` (recommended to override) | `Status` |

### Strategy Flow Methods

| Method | Description | Return Value |
|--------|-------------|--------------|
| `InferTensorMap()` | Infer output `tensor_map` | `Status` |
| `InferDevMatrixShape()` | Infer device matrix `shape` | `Status` |
| `GenerateStrategies()` | Generate all valid parallel strategies | `std::vector<StrategyPtr>` |

### Graph Replacement Methods

| Method | Description | Return Value |
|--------|-------------|--------------|
| `ReplaceNodeInputOrAttrs()` | Replace input tensors or modify node attributes | `Status` |
| `replace_graph()` | Replace original node with communication operator subgraph | `Status` |

---

## TensorInfo Related Types

### TensorInfo Structure

```cpp
class TensorInfo {
 public:
  TensorLayout tensor_layout_;  // Contains shape, tensor_map, dev_matrix_shape

  // Getter methods
  TensorLayout GetTensorLayout() const;
  Shape GetShape() const;              // Tensor's physical shape
  TensorMap GetTensorMap() const;      // Sharding strategy (-1 means not sharded)
  DevMatrixShape GetDevMatrixShape() const;  // Device matrix shape
};
```

### TensorMap Type

```cpp
// TensorMap: std::vector<int32_t>, stores sharding strategy for each dimension
// -1: Not sharded
// 0, 1, 2, ...: Corresponds to some dimension of device matrix

TensorMap map;
// Example 1: [0, 1, -1] - First two dimensions sharded, third not sharded
// Example 2: [-1, -1, -1] - Not sharded (replicate)
// Example 3: [0, -1, 1] - First and third dimensions sharded
```

---

## hyper-parallel Interface Mapping

### infer_layout Function

```python
def infer_layout(self, inputs_dict, device_mesh):
    """Infer input/output layout

    Args:
        inputs_dict: Input tensor Layout dict, e.g., {'x': Layout(...), 'w': Layout(...)}
        device_mesh: Device mesh, e.g., DeviceShape([2, 2])

    Returns:
        Dict: Output tensor Layout dict, e.g., {'output': Layout(...)}
    """
    # 1. Get input Layout
    x_layout = inputs_dict['x']
    w_layout = inputs_dict['w']

    # 2. Input validation (reference MindSpore input validation logic)

    # 3. Infer output tensor_map (reference MindSpore output inference logic)

    # 4. Construct output Layout
    out_layout = Layout(...)

    # 5. Return output Layout dict
    return {self.name: out_layout}
```

### get_expand_impl Function

```python
def get_expand_impl(self, tensor_dict):
    """Expand implementation, return new computation graph

    Args:
        tensor_dict: Input tensor dict

    Returns:
        Dict: Computation graph dict
    """
    # 1. Get input DTensor
    x = tensor_dict['x']
    w = tensor_dict['w']

    # 2. Call original operator (automatically sharded in distributed setting)
    output = ops.{OperatorName}(x, w, **self.attrs)

    # 3. Add communication operators (reference MindSpore replace_graph logic)
    if self.need_allreduce:
        output = platform.all_reduce(output, op=SUM)

    # 4. Return computation graph
    return {self.name: output}
```

---

## Communication Operators Quick Reference

### all_reduce

```python
# AllReduce: Reduce operation on data across all devices
result = platform.all_reduce(
    tensor,           # Input DTensor
    op=SUM,           # Reduce operation: SUM, MAX, MIN, PROD
    group=None        # Communication group (default: world group)
)
```

### all_gather

```python
# AllGather: Gather shards from all devices along some axis
result = platform.all_gather(
    tensor,           # Input DTensor
    axis=0,           # Axis to gather on
    group=None        # Communication group
)
```

### reduce_scatter

```python
# ReduceScatter: First reduce, then scatter along axis to different devices
result = platform.reduce_scatter(
    tensor,           # Input DTensor
    op=SUM,           # Reduce operation
    axis=0,           # Axis to scatter on
    group=None        # Communication group
)
```

---

## Naming Rules Quick Reference

### MindSpore Naming

| Type | Naming Rule | Examples |
|------|-------------|----------|
| **YAML File Name** | snake_case + `_op.yaml` | `matmul_op.yaml`, `add_op.yaml` |
| **Operator Primitive Name** | snake_case | `matmul`, `add` |
| **Python Class Name** | PascalCase | `MatMul`, `Add` |
| **Info Class Name** | PascalCase + `Info` suffix | `MatMulInfo`, `AddInfo` |
| **Info Class File** | snake_case + `_info.cc` | `matmul_info.cc`, `add_info.cc` |

### Conversion Examples

```
YAML name: matmul_op.yaml
    ↓ Convert
Operator name: matmul
    ↓ Convert
Primitive class name: MatMul
    ↓ Convert
Info class name: MatMulInfo
    ↓ Convert
Info class file: matmul_info.cc
```

---

## Reference Documents

- **Detailed Mechanism Explanation**: `references/operator-info-mechanism.md`
- **Complete Analysis Cases**: `references/typical-cases.md` (MatMul/Add/FlashAttentionScore)
- **File Path Summary**: `references/reference-docs.md`
