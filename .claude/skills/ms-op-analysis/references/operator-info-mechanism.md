# Operator Info Mechanism Details

This document provides in-depth explanation of MindSpore distributed operator implementation's core mechanism - Info class and its Layout/Strategy flows.

---

## Background Overview

MindSpore's distributed operator implementation uses the Info class pattern, where each operator has a corresponding Info class responsible for:

1. **Layout Inference**: Infer input/output distributed sharding information
2. **Strategy Inference**: Infer valid parallel strategies
3. **Graph Replacement**: Replace original node with communication operator subgraph (if needed)

**Info Class Location:**
```
mindspore/ccsrc/frontend/parallel/ops_info/
├── operator_info.h           # Base class definition
├── matmul_info.cc
├── add_info.cc
└── ...
```

---

## Info Class Hierarchy Structure

### Base Class: OperatorInfo

All operator Info classes inherit from `OperatorInfo`:

```cpp
class OperatorInfo {
 public:
  // Distributed inference core methods
  virtual Status Infer(const OperatorParams &inputs, const OperatorParams &outputs) = 0;
  virtual Status InferTensorInfo() = 0;          // Layout flow
  virtual Status InferTensorMap() = 0;           // Strategy flow
  virtual Status InferDevMatrixShape() = 0;
  virtual Status InferMirrorOps() = 0;

  // Graph replacement methods
  virtual Status ReplaceNodeInputOrAttrs() override;  // Input replacement
  virtual Status replace_graph(const CNodePtr &cnode) override;  // Subgraph replacement

  // Strategy generation
  virtual std::vector<StrategyPtr> GenerateStrategies(int32_t stage_id) = 0;
};
```

### Intermediate Base Classes

Some operators have more specific base classes:

#### ArithmeticInfo (Binary Operator Base Class)

```cpp
class ArithmeticInfo : public OperatorInfo {
 protected:
  // Binary operator common logic
  virtual Status InferTensorInfo() override;
  virtual Status InferTensorMap() override;
};

// Operators inheriting this base class
class AddInfo : public ArithmeticInfo {};
class SubInfo : public ArithmeticInfo {};
class MulInfo : public ArithmeticInfo {};
```

#### MatMulInfoBase (Matrix Multiplication Base Class)

```cpp
class MatMulInfoBase : public OperatorInfo {
 protected:
  // Matrix multiplication common logic
  virtual Status InferMatrixDim() = 0;  // Infer matrix dimensions
  virtual Status InferBiasTensorInfo() = 0;  // Infer bias TensorInfo

  // Operators inheriting this base class
  class MatMulInfo : public MatMulInfoBase {};
  class BatchMatMulInfo : public MatMulInfoBase {};
};
```

### Info Class Hierarchy Diagram

```
OperatorInfo (Base class)
├── ArithmeticInfo (Binary operators)
│   ├── AddInfo
│   ├── SubInfo
│   ├── MulInfo
│   └── DivInfo
├── MatMulInfoBase (Matrix multiplication)
│   ├── MatMulInfo
│   └── BatchMatMulInfo
├── ReduceInfo (Reduction operators)
│   └── ReduceSumInfo
└── ... Other base classes
```

---

## Two Inference Flows

MindSpore distributed operators have **two inference flows**:

### Layout Flow (Recommended)

**Characteristics:**
- Uses `TensorInfo` and `TensorLayout` as abstraction
- More flexible inference, supports complex sharding patterns
- Applicable to all operators

**Core Methods:**

| Method | Description | hyper-parallel Implementation |
|--------|-------------|------------------------------|
| `InferTensorInfo()` | Infer input/output `TensorInfo` | Core implementation of `infer_layout` |
| `InferOutputTensorInfo()` | Output inference in Layout flow | Output inference in `infer_layout` |
| `InferInputTensorInfo()` | Input inference in Layout flow | Input validation in `infer_layout` |

**TensorInfo Structure:**

```cpp
class TensorInfo {
public:
  TensorLayout tensor_layout_;  // Contains shape, tensor_map, dev_matrix_shape

  TensorLayout GetTensorLayout() const;
  Shape GetShape() const;              // Tensor's physical shape
  TensorMap GetTensorMap() const;      // Sharding strategy (-1 means not sharded)
  DevMatrixShape GetDevMatrixShape() const;  // Device matrix shape
};
```

### Strategy Flow (Traditional)

**Characteristics:**
- Directly infers `tensor_map` and `dev_matrix_shape`
- Simpler logic, but limited flexibility
- Used more in legacy operators

**Core Methods:**

| Method | Description | hyper-parallel Implementation |
|--------|-------------|------------------------------|
| `InferTensorMap()` | Infer output `tensor_map` | tensor_map return value in `infer_layout` |
| `InferDevMatrixShape()` | Infer device matrix `shape` | device_mesh usage in `infer_layout` |
| `GenerateStrategies()` | Generate all valid parallel strategies | Strategy enumeration in `infer_layout` |

---

## Comparison of Two Flows

| Dimension | Layout Flow | Strategy Flow |
|-----------|-------------|---------------|
| **Abstraction Level** | Higher (TensorInfo/TensorLayout) | Lower (direct tensor_map manipulation) |
| **Flexibility** | Higher, supports complex sharding patterns | Lower |
| **Applicable Scenarios** | All operators, especially complex ones | Simple operators, legacy operators |
| **Recommendation Level** | ✅ Recommended | ⚠️ Traditional approach |
| **hyper-parallel Mapping** | Complete `infer_layout` implementation | tensor_map part in `infer_layout` |

---

## Graph Replacement Mechanism

### Why Graph Replacement is Needed

When operators need additional communication operations in distributed environment, need to **replace original node with communication operator subgraph**.

**Common Scenarios:**
- After Reduce operation need to aggregate results (e.g., ReduceSum with certain sharding)
- Need to broadcast local results to all devices
- Need communication on specific dimensions (e.g., AllReduce, AllGather)

### Graph Replacement Methods

| Method | Description | Call Timing |
|--------|-------------|-------------|
| `ReplaceNodeInputOrAttrs()` | Replace input tensors or modify node attributes | Compilation optimization phase |
| `replace_graph()` | Replace original node with communication operator subgraph | Compilation optimization phase |

### hyper-parallel Implementation Mapping

MindSpore's `replace_graph` corresponds to hyper-parallel's `get_expand_impl`:

| MindSpore | hyper-parallel |
|-----------|----------------|
| `replace_graph(const CNodePtr &cnode)` | `get_expand_impl(self, tensor_dict)` |
| Construct `AllReduce` / `AllGather` subgraph | Use `platform.all_reduce()` / `all_gather()` |
| Return new node | Return new computation graph |

---

## Input Validation Mechanism

### Why Input Validation is Needed

Distributed operators must validate input Layout compatibility, otherwise may cause:

- Shape mismatch
- Sharding strategy conflict
- Unnecessary or incorrect communication operations

### Validation Logic Location

Validation logic is typically implemented in the following methods:

| Method | Layout Flow | Strategy Flow |
|--------|-------------|---------------|
| Input Validation | Beginning of `InferOutputTensorInfo()` | Beginning of `InferTensorMap()` |
| Output Inference | Latter part of `InferOutputTensorInfo()` | Latter part of `InferTensorMap()` |

---

## Common Operator Pattern Summary

### Element-wise Binary Operators (Add, Sub, Mul, Div)

**Characteristics:**
- Two inputs must have the same Layout
- Output Layout same as input Layout
- Usually don't need graph replacement

**Inference Logic:**
```
input1_layout == input2_layout
output_layout = input1_layout
```

### Matrix Multiplication Operators (MatMul, BatchMatMul)

**Characteristics:**
- Support different sharding strategies (on different matrix dimensions)
- Output Layout inferred from inputs
- May need AllReduce (if summation needed)

**Inference Logic:**
```
output_map[i] = input_map1[i]  (if input_map1[i] != input_map2[k])
output_map[j] = input_map2[j]  (other cases)
```

### Reduction Operators (ReduceSum, ReduceMean)

**Characteristics:**
- Reduce on specified dimension
- Output tensor_map on reduced dimension is -1 (not sharded)
- May need AllReduce / ReduceScatter

**Inference Logic:**
```
output_map[reduce_dim] = -1
output_map[other_dim] = input_map[other_dim]
```

---

## Reference Documents

| Document | Description | Location |
|----------|-------------|----------|
| `operator_info.h` | Base class definition | `mindspore/ccsrc/frontend/parallel/operator_info.h` |
| `arithmetic_info.h` | Binary operator base class | `mindspore/ccsrc/frontend/parallel/ops_info/` |
| `matmul_info.cc` | MatMul complete implementation | `mindspore/ccsrc/frontend/parallel/ops_info/matmul_info.cc` |
| `add_info.cc` | Add complete implementation | `mindspore/ccsrc/frontend/parallel/ops_info/add_info.cc` |
