# Workflow 5: Analysis Result Output

## 🔴 Critical Warning

> **This step is mandatory and cannot be skipped!**
>
> After analysis is complete, you **must** generate a standardized report document, otherwise subsequent development processes will not be able to execute correctly due to missing reference sources.

---

## Objective

Generate a standardized analysis report document based on all results from previous workflows, following a unified template format.

## Input

- **PyTorch Interface Information**: Obtained from Workflow 1
- **MindSpore Mapping Relationship**: Obtained from Workflow 2
- **MindSpore Distributed Implementation**: Obtained from Workflow 3
- **Integration Results**: Obtained from Workflow 4

## Output

- **Standardized Analysis Report**: `.agentic/skills/dist-op-dev/analysis-results/torch.{OpName}-analysis.md`

---

## Template Usage Process

### Step 1: Read Template File

Template file location:
```
.agentic/skills/dist-op-dev/templates/operator-analysis-template.md
```

### Step 2: Fill in Analysis Results

Follow the template's section structure and fill in the following content:

#### 2.1 Operator Basic Information

```markdown
## 1. Operator Basic Information

| Attribute | PyTorch | MindSpore |
|------|---------|-----------|
| **Operator Name** | torch.matmul | MatMul |
| **Platform Type** | PyTorch | MindSpore |
| **Operator Category** | Matrix Multiplication | Matrix Multiplication |
| **Semantic Description** | Matrix multiplication, supports high dimensions | Matrix multiplication, supports high dimensions |
```

#### 2.2 Interface Definition

```markdown
## 2. Interface Definition

### PyTorch Interface
```python
torch.matmul(input, other, *, out=None) → Tensor
```

### MindSpore Interface
```python
mindspore.ops.MatMul()(x, w) → Tensor
```

### Parameter Mapping Table
| PyTorch Parameter | MindSpore Parameter | Type | Required | Description |
|-------------|---------------|------|------|------|
| input | x | Tensor | ✅ | First input matrix |
| other | w | Tensor | ✅ | Second input matrix |
| out | - | Tensor | ❌ | MindSpore doesn't support |
```

#### 2.3 Distributed Implementation Plan

```markdown
## 3. Distributed Implementation Plan

### MindSpore Reference Sources
- **Info Class**: `MatMulInfo`
- **File Location**: `mindspore/ccsrc/frontend/parallel/ops_info/matmul_info.cc`
- **Base Class**: `MatMulInfoBase`

### Layout Inference Logic
(Reuse MindSpore analysis results)

### Graph Replacement Logic
(Reuse MindSpore analysis results)
```

#### 2.4 HyperParallel Implementation Reference

```markdown
## 4. HyperParallel Implementation Reference

### Base Class Selection
- **Recommended Base Class**: `MatMulDistributedOp`
- **Reason**: PyTorch and MindSpore MatMul semantics are consistent, can reuse same implementation

### YAML Registration Configuration
```yaml
# PyTorch Registration
matmul:
  dist_op_name: _torch_matmul_dist_op
  distributed_op_class: MatMulDistributedOp
  distributed_op_file: parallel_matmul
```
```

### Step 3: Save Analysis Document

Document save location:
```
.agentic/skills/dist-op-dev/analysis-results/torch.{OpName}-analysis.md
```

**Naming Rules:**
- PyTorch Operators: Use complete interface name, e.g., `torch.matmul-analysis.md`, `torch.add-analysis.md`

---

## Document Quality Checklist

After completing the analysis document, check each item:

### Structural Completeness Check

- [ ] Document contains all the following sections:
  - [ ] Operator Basic Information (PyTorch + MindSpore)
  - [ ] Interface Definition Comparison
  - [ ] Parameter Mapping Table
  - [ ] Distributed Implementation Plan
  - [ ] HyperParallel Implementation Reference

### Content Quality Check

- [ ] **PyTorch Interface Complete**: Contains function signature and parameter description
- [ ] **MindSpore Mapping Accurate**: Mapping relationship is correct
- [ ] **Parameter Differences Annotated**: Clearly annotate parameter differences
- [ ] **Distributed Logic Clear**: Layout inference logic has detailed explanation
- [ ] **Reference Sources Accurate**: MindSpore function names and file paths are correct

### Reuse Strategy Check

- [ ] Determined reuse strategy (complete reuse/partial reuse/complete custom)
- [ ] If complete reuse, annotated both platforms point to same implementation class
- [ ] If partial reuse, annotated adaptation code location
- [ ] If complete custom, provided implementation suggestions

---

## Success Criteria

- [ ] Generated standardized analysis report
- [ ] Document save location is correct: `.agentic/skills/dist-op-dev/analysis-results/torch.{OpName}-analysis.md`
- [ ] Completed all checklist items in quality check
- [ ] All reference sources are accurately annotated
- [ ] Reuse strategy is clearly defined

---

## Workflow Complete

✅ **This SKILL analysis process is complete!**

Next, the analysis report will be used for:
1. **Plan Confirmation**: Show key content of analysis results to user
2. **Development Reference**: Guide subsequent Python implementation, YAML registration, UT/ST development
3. **Experience Accumulation**: Save as team knowledge base

The caller (dist-op-dev) will guide the user through the complete 6-step development process based on this report.
