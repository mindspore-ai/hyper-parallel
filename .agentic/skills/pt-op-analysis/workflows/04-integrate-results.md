# Workflow 4: Integrate Analysis Results

## Objective

Integrate PyTorch interface information and MindSpore distributed implementation analysis results to form a complete analysis report.

## Input

- **PyTorch Interface Information**: Obtained from Workflow 1
- **MindSpore Mapping Relationship**: Obtained from Workflow 2
- **MindSpore Distributed Implementation**: Obtained from Workflow 3

## Output

- **Complete Analysis Results**: Integrated report containing information from both platforms

---

## Integration Structure

### 1. Operator Basic Information

```markdown
## Operator Basic Information

| Attribute | PyTorch | MindSpore |
|------|---------|-----------|
| **Operator Name** | torch.matmul | MatMul |
| **Platform Type** | PyTorch | MindSpore |
| **Operator Category** | Matrix Multiplication | Matrix Multiplication |
```

### 2. Interface Definition Comparison

```markdown
## Interface Definition Comparison

### PyTorch Interface
```python
torch.matmul(input, other, *, out=None) → Tensor
```

### MindSpore Interface
```python
mindspore.ops.MatMul()(x, w) → Tensor
```

### Parameter Mapping
| PyTorch Parameter | MindSpore Parameter | Description |
|-------------|---------------|------|
| input | x | First input matrix |
| other | w | Second input matrix |
| out | - | MindSpore doesn't support |
```

### 3. Distributed Implementation Plan

```markdown
## Distributed Implementation Plan

### MindSpore Reference Sources
- **Info Class**: `MatMulInfo`
- **File Location**: `mindspore/ccsrc/frontend/parallel/ops_info/matmul_info.cc`
- **Base Class**: `MatMulInfoBase`

### Layout Inference Logic
(Reuse MindSpore analysis results)

### Graph Replacement Logic
(Reuse MindSpore analysis results)
```

### 4. HyperParallel Implementation Reference

```markdown
## HyperParallel Implementation Reference

### Base Class Selection
- **Recommended Base Class**: `MatMulDistributedOp`
- **Reason**: PyTorch and MindSpore MatMul semantics are consistent, can reuse same implementation

### YAML Registration
```yaml
# MindSpore Registration
MatMul:
  dist_op_name: _matmul_dist_op
  distributed_op_class: MatMulDistributedOp
  distributed_op_file: parallel_matmul

# PyTorch Registration
matmul:
  dist_op_name: _torch_matmul_dist_op
  distributed_op_class: MatMulDistributedOp
  distributed_op_file: parallel_matmul
```
```

---

## Difference Handling

### Parameter Difference Handling

| Difference Type | Handling Method | Example |
|----------|----------|------|
| **Different Parameter Names** | Add mapping in implementation | `dim` → `axis` |
| **Different Default Values** | Explicitly specify parameters | `keepdim=False` → `keep_dims=False` |
| **Missing Parameters** | Add default value handling | PyTorch `out` parameter ignored |

### Behavior Difference Handling

| Difference Type | Handling Method | Example |
|----------|----------|------|
| **Different Boundary Behavior** | Unify handling in implementation | Handling of `dim=-1` |
| **Different Error Handling** | Unify error messages | Throw same exception type |

---

## Reuse Strategy

### Complete Reuse

When PyTorch and MindSpore operator semantics are completely consistent:
- **Distributed Implementation**: Directly reuse MindSpore analysis results
- **YAML Registration**: Both platforms point to same implementation class
- **Test Cases**: Write separately, but test logic is the same

### Partial Reuse

When there are parameter or behavior differences:
- **Distributed Implementation**: Reuse core logic, add adaptation code
- **YAML Registration**: Both platforms point to same implementation class
- **Test Cases**: Write separately, covering respective differences

### Complete Custom

When no corresponding MindSpore operator can be found:
- **Distributed Implementation**: Custom based on local reference and operator semantics
- **YAML Registration**: Register separately
- **Test Cases**: Completely custom

---

## Success Criteria

- [ ] Integrated PyTorch interface information
- [ ] Integrated MindSpore mapping relationship
- [ ] Integrated MindSpore distributed implementation analysis
- [ ] Annotated differences between the two platforms
- [ ] Determined reuse strategy (complete reuse/partial reuse/complete custom)

---

## Next Step

After integration is complete, proceed to **[Workflow 5: Analysis Result Output](./05-analysis-output.md)**

**Input**: Complete analysis results
**Objective**: **Generate standardized analysis report document** (🔴 **This step is mandatory and cannot be skipped**)
