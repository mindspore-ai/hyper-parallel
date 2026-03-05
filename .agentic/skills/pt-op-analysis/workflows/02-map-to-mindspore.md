# Workflow 2: Map to MindSpore Operator

## Objective

Find MindSpore operators that are semantically equivalent to PyTorch operators and establish mapping relationships.

## Input

- **PyTorch Operator Semantics**: Obtained from Workflow 1
- **Parameter List**: PyTorch operator parameter information

## Output

- **MindSpore Operator Name**: Corresponding MindSpore primitive name
- **Mapping Relationship**: Parameter name mapping, default value differences
- **Difference Description**: Differences between the two platforms

---

## Mapping Methods

### Method 1: Direct Mapping

For operators with completely consistent semantics, establish mapping relationships directly.

**Examples:**
| PyTorch | MindSpore | Description |
|---------|-----------|------|
| `torch.add` | `Add` | Element-wise addition, consistent semantics |
| `torch.matmul` | `MatMul` | Matrix multiplication, consistent semantics |
| `torch.sum` | `ReduceSum` | Reduction sum, consistent semantics |

### Method 2: Composite Mapping

For composite operations in PyTorch, mapping to combinations of multiple MindSpore operators may be needed.

**Examples:**
| PyTorch | MindSpore Combination | Description |
|---------|---------------|------|
| `torch.nn.functional.linear` | `MatMul` + `Add` | Linear layer = matrix multiplication + bias addition |
| `torch.softmax` | `Softmax` | Direct mapping |

### Method 3: No Direct Mapping

For PyTorch-specific operators, there may be no directly corresponding MindSpore operator.

**Handling Approach:**
1. Find semantically similar MindSpore operators
2. Analyze whether it can be implemented through operator combination
3. If completely unmappable, mark as "requires custom implementation"

---

## Parameter Mapping Rules

### Naming Style Differences

| PyTorch Style | MindSpore Style | Example |
|-------------|---------------|------|
| lowercase_underscore | PascalCase | `input` → `input` (parameter names unchanged) |
| keyword arguments | positional arguments | `dim=None` → `axis=None` |

### Common Parameter Name Mapping

| PyTorch Parameter | MindSpore Parameter | Description |
|-------------|---------------|------|
| `dim` | `axis` | Dimension parameter |
| `keepdim` | `keep_dims` | Keep dimensions |
| `input` | `input` | Input tensor |
| `other` | `other` / `y` | Second input |
| `out` | - | MindSpore usually doesn't support out parameter |

### Default Value Differences

| Operator | PyTorch Default | MindSpore Default | Description |
|------|---------------|-----------------|------|
| `sum` | `dim=None` | `axis=()` | Default behavior may differ |
| `matmul` | - | - | No default value difference |

---

## Common Mapping Reference Table

### Element-wise Operators

| PyTorch | MindSpore | Parameter Difference |
|---------|-----------|----------|
| `torch.add` | `Add` | None |
| `torch.sub` | `Sub` | None |
| `torch.mul` | `Mul` | None |
| `torch.div` | `Div` | `rounding_mode` → `rounding_mode` |
| `torch.pow` | `Pow` | None |
| `torch.neg` | `Neg` | None |

### Matrix Operations

| PyTorch | MindSpore | Parameter Difference |
|---------|-----------|----------|
| `torch.matmul` | `MatMul` | None |
| `torch.mm` | `MatMul` | PyTorch only supports 2D |
| `torch.bmm` | `BatchMatMul` | PyTorch only supports 3D |
| `torch.nn.functional.linear` | `Linear` | Different parameter names |

### Reduction Operators

| PyTorch | MindSpore | Parameter Difference |
|---------|-----------|----------|
| `torch.sum` | `ReduceSum` | `dim` → `axis` |
| `torch.mean` | `ReduceMean` | `dim` → `axis` |
| `torch.max` | `ReduceMax` | `dim` → `axis` |
| `torch.min` | `ReduceMin` | `dim` → `axis` |

### Shape Operations

| PyTorch | MindSpore | Parameter Difference |
|---------|-----------|----------|
| `torch.reshape` | `Reshape` | None |
| `torch.transpose` | `Transpose` | Different parameter format |
| `torch.cat` | `Concat` | `dim` → `axis` |
| `torch.split` | `Split` | Different parameter format |

---

## Difference Analysis Key Points

### 1. Semantic Differences

Check whether operator semantics are completely consistent across both platforms:
- **Completely Consistent**: Can directly reuse distributed implementation
- **Partial Differences**: Need adaptation code
- **Completely Different**: Need custom implementation

### 2. Parameter Differences

Analyze parameter differences:
- **Different Parameter Names**: Need parameter mapping
- **Different Default Values**: Need to explicitly specify parameters
- **Different Parameter Types**: Need type conversion

### 3. Behavior Differences

Analyze behavior differences in boundary cases:
- **Empty Input**: Whether handling is consistent
- **Boundary Values**: e.g., handling of `dim=-1`
- **Error Handling**: Differences in error messages

---

## Success Criteria

- [ ] Found corresponding MindSpore operator
- [ ] Established parameter mapping relationship
- [ ] Identified parameter differences (names, default values, types)
- [ ] Analyzed semantic differences
- [ ] Documented behavior differences in boundary cases

---

## Next Step

After mapping is complete, proceed to **[Workflow 3: Call MindSpore Analysis SKILL](./03-call-mindspore-analysis.md)**

**Input**: MindSpore operator name, MindSpore source code path
**Objective**: Reuse `ms-op-analysis` to analyze distributed implementation
