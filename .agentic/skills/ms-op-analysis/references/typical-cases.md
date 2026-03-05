# Typical Operator Complete Analysis Cases

This document demonstrates MindSpore operator distributed implementation analysis patterns through three complete analysis cases.

---

## Case 1: MatMul (Matrix Multiplication)

### Operator Basic Information

- **Operator Name**: MatMul
- **Platform Type**: MindSpore
- **YAML Definition**: `matmul_op.yaml`
- **Info Class**: `MatMulInfo`

### Info Class Analysis

**File Location:** `mindspore/ccsrc/frontend/parallel/ops_info/matmul_info.cc`

**Base Class Inheritance:**
```cpp
MatMulInfo → MatMulInfoBase → OperatorInfo
```

### Distributed Implementation Analysis

#### 1. Input Validation Logic

**Reference Source:**
- **MindSpore Reference**: `InferMatrixDim()`
- **File Location**: `mindspore/ccsrc/frontend/parallel/ops_info/matmul_info.cc`

**Validation Rules:**
- Inputs `x` and `w` corresponding dimensions must be compatible (matrix multiplication rules)
- Consider `transpose_a` and `transpose_b` attributes

#### 2. Layout Inference Logic

**Reference Source:**
- **MindSpore Reference**: `InferOutputTensorInfo()`
- **File Location**: `mindspore/ccsrc/frontend/parallel/ops_info/matmul_info.cc`

**Inference Rules (Simplified):**
```
output_map[i] = x_map[i]  (if x is sharded on dimension i)
output_map[j] = w_map[j]  (if w is sharded on dimension j)

Constraints:
x's sharding on dimension k must match w's sharding on dimension k
(because these two dimensions need matrix multiplication inner product summation)
```

#### 3. Graph Replacement Logic

**Reference Source:**
- **MindSpore Reference**: `replace_graph()`
- **File Location**: `mindspore/ccsrc/frontend/parallel/ops_info/matmul_info.cc`

**Communication Operators:**
- **Scenario**: When both inputs are sharded on common dimension
- **Operator**: AllReduce(op=SUM)
- **Reason**: Sharded products need to be summed across all devices

### hyper-parallel Implementation Mapping

**Python Implementation:**

```python
class MatMulDistributedOp(DistributedOpBase):
    def infer_layout(self, inputs_dict, device_mesh):
        """Infer input/output layout"""
        x_layout = inputs_dict['x']
        w_layout = inputs_dict['w']

        # Input validation: Check shape compatibility
        if x_layout.tensor_shape[1] != w_layout.tensor_shape[0]:
            raise ValueError("Shape not match for MatMul")

        # Infer output tensor_map
        x_map = x_layout.tensor_map
        w_map = w_layout.tensor_map

        # Output first dimension comes from x's first dimension
        out_map = [x_map[0]]
        # Output second dimension comes from w's second dimension
        out_map.append(w_map[1])

        # Construct output Layout
        out_layout = Layout(...)

        return {self.name: out_layout}

    def get_expand_impl(self, tensor_dict):
        """Expand implementation, return new computation graph"""
        x = tensor_dict['x']
        w = tensor_dict['w']

        # Call original MatMul operator
        output = ops.MatMul(x, w, **self.attrs)

        # Check if AllReduce is needed
        if self.need_allreduce:
            output = platform.all_reduce(output, op=SUM)

        return {self.name: output}
```

---

## Case 2: Add (Element-wise Addition)

### Operator Basic Information

- **Operator Name**: Add
- **Platform Type**: MindSpore
- **YAML Definition**: `add_op.yaml`
- **Info Class**: `AddInfo`

### Info Class Analysis

**File Location:** `mindspore/ccsrc/frontend/parallel/ops_info/add_info.cc`

**Base Class Inheritance:**
```cpp
AddInfo → ArithmeticInfo → OperatorInfo
```

**ArithmeticInfo (Binary Operator Base Class):**

```cpp
class ArithmeticInfo : public OperatorInfo {
 public:
  // Binary operator common implementation
  Status InferTensorInfo() override;

 protected:
  // Check if two inputs have same Layout
  Status CheckInputsTensorInfo();
};
```

### Distributed Implementation Analysis

#### 1. Input Validation Logic

**Reference Source:**
- **MindSpore Reference**: `ArithmeticInfo::CheckInputsTensorInfo()`
- **File Location**: `mindspore/ccsrc/frontend/parallel/ops_info/`

**Validation Rules:**
- Two inputs must have the same Layout
- Output Layout same as input Layout
- Support broadcasting (automatic expansion when shape dimensions are insufficient)

#### 2. Layout Inference Logic

**Reference Source:**
- **MindSpore Reference**: `ArithmeticInfo::InferTensorInfo()`
- **File Location**: `mindspore/ccsrc/frontend/parallel/ops_info/`

**Inference Rules:**
```
output_tensor_map = input_tensor_map
output_shape = broadcast(x_shape, y_shape)
```

#### 3. Graph Replacement Logic

Add operator usually doesn't need graph replacement because:
- Element-wise operations don't need additional communication
- Both inputs operate on same devices

### hyper-parallel Implementation Mapping

**Python Implementation:**

```python
class ElementWiseBinaryOp(DistributedOpBase):
    """Element-wise binary operation base class"""

    def infer_layout(self, inputs_dict, device_mesh):
        """Infer input/output layout"""
        x_layout = inputs_dict['x']
        y_layout = inputs_dict['y']

        # Input validation: Layout must match
        if x_layout.tensor_map != y_layout.tensor_map:
            raise ValueError("Input layouts must match")

        # Output Layout same as input Layout
        out_layout = Layout(...)

        return {self.name: out_layout}

    def get_expand_impl(self, tensor_dict):
        """Expand implementation, directly call original operator (no additional communication)"""
        x = tensor_dict['x']
        y = tensor_dict['y']

        # Directly call original operator
        output = ops.Add(x, y, **self.attrs)

        return {self.name: output}
```

---

## Analysis Pattern Summary

### 1. Analyze YAML Definition

- Identify input/output parameters
- Identify attribute parameters
- Understand operator's basic semantics

### 2. Locate Info Class

```bash
# Find Info class file
find mindspore/ccsrc/frontend/parallel/ops_info/ -name "*{op_name}_info.cc"
```

### 3. Analyze Info Class Structure

- Inheritance relationship (what is base class)
- Which methods are overridden
- Which distributed logic is implemented

### 4. Extract Reference Logic

| Step | Goal | Reference Method |
|------|------|------------------|
| 1 | Input validation logic | `CheckInputsTensorInfo()` / Input validation part |
| 2 | Layout inference logic | `InferTensorInfo()` / `InferOutputTensorInfo()` / `InferTensorMap()` |
| 3 | Graph replacement logic | `replace_graph()` / `ReplaceNodeInputOrAttrs()` |

### 5. Map to hyper-parallel

```python
# infer_layout implementation
def infer_layout(self, inputs_dict, device_mesh):
    # 1. Input validation (reference MindSpore input validation)
    # 2. Layout inference (reference MindSpore output inference)
    # 3. Return output Layout dict
    return {self.name: out_layout}

# get_expand_impl implementation
def get_expand_impl(self, tensor_dict):
    # 1. Call original operator (automatically sharded in distributed setting)
    # 2. Add communication operators (reference MindSpore replace_graph)
    # 3. Return new computation graph
    return {self.name: output}
```
