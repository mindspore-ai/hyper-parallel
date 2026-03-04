# Quick Reference

This document provides common commands and interface quick reference for PyTorch operator analysis.

---

## PyTorch Interface Query

### Python Help System

```python
import torch

# View help documentation
help(torch.add)
help(torch.matmul)

# View function signature
import inspect
print(inspect.signature(torch.add))

# View source code location
print(inspect.getfile(torch.add))
```

### Common Interface Signatures

```python
# Element-wise binary operators
torch.add(input, other, *, alpha=1, out=None) → Tensor
torch.sub(input, other, *, alpha=1, out=None) → Tensor
torch.mul(input, other, *, out=None) → Tensor
torch.div(input, other, *, rounding_mode=None, out=None) → Tensor

# Matrix multiplication
torch.matmul(input, other, *, out=None) → Tensor
torch.mm(input, mat2, *, out=None) → Tensor
torch.bmm(input, mat2, *, out=None) → Tensor

# Reduction operators
torch.sum(input, dim=None, keepdim=False, *, dtype=None, out=None) → Tensor
torch.mean(input, dim=None, keepdim=False, *, dtype=None, out=None) → Tensor
torch.max(input, dim=None, keepdim=False, *, out=None) → Tensor

# Shape operations
torch.reshape(input, shape) → Tensor
torch.transpose(input, dim0, dim1) → Tensor
torch.cat(tensors, dim=0, *, out=None) → Tensor
```

---

## Parameter Name Mapping Quick Reference

| PyTorch | MindSpore | Description |
|---------|-----------|------|
| `dim` | `axis` | Dimension parameter |
| `keepdim` | `keep_dims` | Keep dimensions |
| `input` | `x` / `input` | Input tensor |
| `other` | `y` / `other` | Second input |

---

## Operator Mapping Quick Reference

### Element-wise Operators

| PyTorch | MindSpore |
|---------|-----------|
| `torch.add` | `Add` |
| `torch.sub` | `Sub` |
| `torch.mul` | `Mul` |
| `torch.div` | `Div` |
| `torch.pow` | `Pow` |

### Matrix Operations

| PyTorch | MindSpore |
|---------|-----------|
| `torch.matmul` | `MatMul` |
| `torch.mm` | `MatMul` |
| `torch.bmm` | `BatchMatMul` |
| `torch.nn.functional.linear` | `Linear` |

### Reduction Operators

| PyTorch | MindSpore |
|---------|-----------|
| `torch.sum` | `ReduceSum` |
| `torch.mean` | `ReduceMean` |
| `torch.max` | `ReduceMax` |
| `torch.min` | `ReduceMin` |

### Shape Operations

| PyTorch | MindSpore |
|---------|-----------|
| `torch.reshape` | `Reshape` |
| `torch.transpose` | `Transpose` |
| `torch.cat` | `Concat` |
| `torch.split` | `Split` |

---

## Analysis Report Naming Rules

| Platform | Naming Rule | Example |
|------|----------|------|
| **PyTorch** | `torch.{op_name}-analysis.md` | `torch.matmul-analysis.md` |
| **MindSpore** | `{OpName}-analysis.md` | `MatMul-analysis.md` |

---

## Reference Documentation

- **Mapping Rules Details**: `./mapping-rules.md`
- **MindSpore Analysis Reference**: `../ms-op-analysis/references/`
