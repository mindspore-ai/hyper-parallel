# PyTorch and MindSpore Operator Mapping Rules

This document details the mapping rules between PyTorch operators and MindSpore operators to help quickly find corresponding operators.

---

## Naming Style Differences

### Operator Names

| Platform | Naming Style | Example |
|------|----------|------|
| **PyTorch** | lowercase_underscore | `torch.add`, `torch.matmul`, `torch.batch_matmul` |
| **MindSpore** | PascalCase | `Add`, `MatMul`, `BatchMatMul` |

### Conversion Rules

```
PyTorch: torch.{op_name}
    ↓ Convert
MindSpore: {OpName} (PascalCase)

Examples:
torch.add → Add
torch.matmul → MatMul
torch.batch_matmul → BatchMatMul
torch.nn.functional.linear → Linear
```

---

## Parameter Name Mapping Rules

### Common Parameter Name Differences

| PyTorch Parameter | MindSpore Parameter | Description |
|-------------|---------------|------|
| `dim` | `axis` | Dimension parameter |
| `keepdim` | `keep_dims` | Keep dimensions |
| `input` | `input` / `x` | Input tensor |
| `other` | `other` / `y` | Second input |
| `out` | - | MindSpore usually doesn't support out parameter |
| `dtype` | `dtype` | Data type |
| `device` | - | MindSpore uses context to set device |

### Parameter Type Differences

| Parameter Type | PyTorch | MindSpore |
|----------|---------|-----------|
| **Dimension Parameter** | `int` or `tuple[int, ...]` | `int` or `tuple[int, ...]` |
| **Boolean Parameter** | `bool` | `bool` |
| **Shape Parameter** | `tuple[int, ...]` | `tuple[int, ...]` |

---

## Operator Mapping Reference Table

### Element-wise Binary Operators

| PyTorch | MindSpore | Parameter Difference | Distributed Implementation Reuse |
|---------|-----------|----------|----------------|
| `torch.add` | `Add` | None | ✅ Complete reuse |
| `torch.sub` | `Sub` | None | ✅ Complete reuse |
| `torch.mul` | `Mul` | None | ✅ Complete reuse |
| `torch.div` | `Div` | `rounding_mode` parameter | ✅ Complete reuse |
| `torch.pow` | `Pow` | None | ✅ Complete reuse |
| `torch.neg` | `Neg` | None | ✅ Complete reuse |
| `torch.abs` | `Abs` | None | ✅ Complete reuse |

### Comparison Operators

| PyTorch | MindSpore | Parameter Difference | Distributed Implementation Reuse |
|---------|-----------|----------|----------------|
| `torch.eq` | `Equal` | None | ✅ Complete reuse |
| `torch.ne` | `NotEqual` | None | ✅ Complete reuse |
| `torch.gt` | `Greater` | None | ✅ Complete reuse |
| `torch.ge` | `GreaterEqual` | None | ✅ Complete reuse |
| `torch.lt` | `Less` | None | ✅ Complete reuse |
| `torch.le` | `LessEqual` | None | ✅ Complete reuse |

### Matrix Operations

| PyTorch | MindSpore | Parameter Difference | Distributed Implementation Reuse |
|---------|-----------|----------|----------------|
| `torch.matmul` | `MatMul` | None | ✅ Complete reuse |
| `torch.mm` | `MatMul` | PyTorch only supports 2D | ✅ Complete reuse |
| `torch.bmm` | `BatchMatMul` | PyTorch only supports 3D | ✅ Complete reuse |
| `torch.nn.functional.linear` | `Linear` | Different parameter names | ✅ Complete reuse |

### Reduction Operators

| PyTorch | MindSpore | Parameter Difference | Distributed Implementation Reuse |
|---------|-----------|----------|----------------|
| `torch.sum` | `ReduceSum` | `dim` → `axis` | ✅ Complete reuse |
| `torch.mean` | `ReduceMean` | `dim` → `axis` | ✅ Complete reuse |
| `torch.max` | `ReduceMax` | `dim` → `axis` | ✅ Complete reuse |
| `torch.min` | `ReduceMin` | `dim` → `axis` | ✅ Complete reuse |
| `torch.prod` | `ReduceProd` | `dim` → `axis` | ✅ Complete reuse |

### Shape Operations

| PyTorch | MindSpore | Parameter Difference | Distributed Implementation Reuse |
|---------|-----------|----------|----------------|
| `torch.reshape` | `Reshape` | None | ✅ Complete reuse |
| `torch.view` | `Reshape` | None | ✅ Complete reuse |
| `torch.transpose` | `Transpose` | Different parameter format | ✅ Complete reuse |
| `torch.permute` | `Transpose` | Different parameter format | ✅ Complete reuse |
| `torch.cat` | `Concat` | `dim` → `axis` | ✅ Complete reuse |
| `torch.split` | `Split` | Different parameter format | ✅ Complete reuse |
| `torch.chunk` | `Split` | Different parameter format | ✅ Complete reuse |

### Activation Functions

| PyTorch | MindSpore | Parameter Difference | Distributed Implementation Reuse |
|---------|-----------|----------|----------------|
| `torch.relu` | `ReLU` | None | ✅ Complete reuse |
| `torch.sigmoid` | `Sigmoid` | None | ✅ Complete reuse |
| `torch.tanh` | `Tanh` | None | ✅ Complete reuse |
| `torch.softmax` | `Softmax` | `dim` → `axis` | ✅ Complete reuse |
| `torch.gelu` | `GeLU` | None | ✅ Complete reuse |

### Index Operations

| PyTorch | MindSpore | Parameter Difference | Distributed Implementation Reuse |
|---------|-----------|----------|----------------|
| `torch.gather` | `Gather` | Different parameter format | ✅ Complete reuse |
| `torch.index_select` | `IndexSelect` | `dim` → `axis` | ✅ Complete reuse |
| `torch.scatter` | `Scatter` | Different parameter format | ✅ Complete reuse |

---

## Special Case Handling

### Operators Without Direct Mapping

When a PyTorch operator has no directly corresponding MindSpore operator:

1. **Find semantically similar operators**
2. **Analyze whether it can be implemented through operator combination**
3. **If completely unmappable, mark as "requires custom implementation"**

### Composite Mapping Examples

| PyTorch | MindSpore Combination | Description |
|---------|---------------|------|
| `torch.nn.functional.linear(x, w, b)` | `MatMul(x, w.T) + b` | Linear layer = matrix multiplication + bias |
| `torch.layer_norm` | `LayerNorm` | Direct mapping |

### Parameter Adaptation Example

```python
# PyTorch
torch.sum(input, dim=1, keepdim=True)

# MindSpore
mindspore.ops.ReduceSum()(input, axis=1, keep_dims=True)
```

---

## Mapping Decision Process

```
1. Identify PyTorch operator semantics
    ↓
2. Find corresponding MindSpore operator
    ↓
3. Compare parameter differences
    ↓
4. Determine mapping strategy
    ├── Completely consistent → Direct reuse
    ├── Parameter differences → Add adaptation code
    └── No corresponding operator → Custom implementation
```

---

## Reference Documentation

- **PyTorch Official Documentation**: https://pytorch.org/docs/stable/
- **MindSpore Official Documentation**: https://www.mindspore.cn/docs/en/master/index.html
