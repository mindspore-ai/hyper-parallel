# Workflow 5: Analysis Result Output

## 🔴 Critical Warning

> **This step is required and cannot be skipped!**
>
> After analysis is complete, you **must** generate a standardized report document, otherwise subsequent development process will not execute correctly due to missing reference sources.

---

## Goal

Generate standardized analysis report document based on all results from previous workflows, following unified template format.

## Input

- **Operator Basic Info**: Name, platform type, YAML definition
- **Distributed Implementation Details**: Info class analysis results, key logic code
- **Reference Source Annotations**: MindSpore function name, file location, key code snippets
- **Fallback Plan**: If applicable, inference plan based on local reference

## Output

- **Standardized Analysis Report**: `.agentic/skills/dist-op-dev/analysis-results/{OpName}-analysis.md`

---

## Template Usage Flow

### Step 1: Read Template File

Template file location:
```
.agentic/skills/dist-op-dev/templates/operator-analysis-template.md
```

This template defines standardized report structure and required content.

### Step 2: Fill Analysis Results

Fill in the following content according to template's section structure:

#### 2.1 Operator Interface Definition

From **Workflow 2: Primitive Query**

- Input parameters: name, type, constraints
- Output info: type, shape changes
- Attribute parameters: name, type, default value

#### 2.2 Distributed Implementation Plan (Core)

From **Workflow 3 & 4: Distributed Operator Query and Practical Query**

Key parts include:

**A. Input Constraints and Layout Validation**
- Reference source: MindSpore Info class input validation logic
- Description: How to validate input Layout compatibility

**B. Layout Inference**
- Reference source: `InferOutputTensorInfo` in Layout flow or `InferTensorMap` in Strategy flow
- Description: How to infer output tensor_map based on inputs

**C. Input Replacement/Subgraph Replacement**
- Reference source: `ReplaceNodeInputOrAttrs` or `replace_graph`
- Description: Whether need to construct communication operator subgraph

**D. MindSpore Reference Annotation** (Important!)
Must clearly annotate:
- MindSpore reference: Which specific function
- File location: `mindspore/ccsrc/frontend/parallel/ops_info/{op}_info.cc`
- Key code logic: Related code snippets

#### 2.3 HyperParallel Local Reference

From **Workflow 4.4: Fallback Handling**

- Base class selection: Which base class to inherit
- Similar operators: Which existing operators are similar
- Implementation strategy: Recommended implementation approach

### Step 3: Save Analysis Document

Document save location:
```
.agentic/skills/dist-op-dev/analysis-results/{OpName}-analysis.md
```

**Naming Rules:**
- MindSpore operators: Use primitive class name (PascalCase), e.g., `MatMul-analysis.md`
- PyTorch operators: Use full interface name, e.g., `torch.matmul-analysis.md`

---

## Reference Source Annotation Standards

### Required Reference Sources to Annotate

| Implementation Content | MindSpore Reference | File Location | Necessity |
|------------------------|---------------------|---------------|-----------|
| Input Constraints | Input validation logic (Strategy/Layout flow) | `{op}_info.cc` | ✅ Required |
| Layout Inference | Output inference logic | `{op}_info.cc` | ✅ Required |
| Input Replacement | ReplaceNodeInputOrAttrs | `{op}_info.cc` | ⚠️ If applicable |
| Subgraph Replacement | replace_graph | `{op}_info.cc` | ⚠️ If applicable |

### Annotation Format Example

```markdown
## Input Constraints and Layout Validation

**Reference Source:**
- **MindSpore Reference**: Input validation logic (Layout flow)
- **File Location**: `mindspore/ccsrc/frontend/parallel/ops_info/matmul_info.cc`
- **Key Code Logic**:
  ```cpp
  Status MatMulInfo::InferOutputTensorInfo() {
    // Check input Layout compatibility
    if (inputs_tensor_info_[0].tensor_layout().tensor_shape().dim(i) !=
        inputs_tensor_info_[1].tensor_layout().tensor_shape().dim(j)) {
      MS_LOG(ERROR) << "Shape not match";
      return Status::FAILED;
    }
    // ... infer output
    return Status::OK();
  }
  ```

**Constraint Description:**
- Inputs x and w must have the same shape on corresponding dimensions
- Support different data type conversions
- ...
```

---

## Document Quality Checklist

After completing analysis document, must check item by item:

### Structure Completeness Check

- [ ] Document contains all the following sections:
  - [ ] Operator basic information
  - [ ] Interface semantics (input/output/parameters)
  - [ ] Distributed implementation plan
    - [ ] Input constraints and Layout validation
    - [ ] Layout inference
    - [ ] Input replacement/subgraph replacement (if applicable)
  - [ ] HyperParallel implementation reference
  - [ ] Reference source summary

### Content Quality Check

- [ ] **Input parameters complete**: Includes parameter name, type, constraints
- [ ] **Output info clear**: Describes output type and shape changes
- [ ] **Distributed logic clear**: Layout inference logic has detailed description
- [ ] **Reference sources accurate**:
  - [ ] MindSpore function name correct
  - [ ] File path accurate
  - [ ] Code snippets consistent with actual implementation
- [ ] **Local reference reasonable**: Base class selection, similar operator recommendation have basis

### Fallback Plan Check (if applicable)

- [ ] Clearly marked "⚠️ Based on **local inference**"
- [ ] Explained inference basis (similar operators/operator semantics)
- [ ] Provided base class selection rationale
- [ ] Given reference implementation file path

### Format Standard Check

- [ ] Markdown format correct
- [ ] Code snippets use correct syntax highlighting
- [ ] File paths use backtick markers
- [ ] Important notes use appropriate prefixes (⚠️ ✅ etc.)

---

## Success Criteria

- [ ] Generated standardized analysis report
- [ ] Document save location correct: `.agentic/skills/dist-op-dev/analysis-results/{OpName}-analysis.md`
- [ ] Completed all checklist items in quality checklist
- [ ] All reference sources accurately annotated
- [ ] If applicable, fallback plan clearly marked

---

## Workflow Complete

✅ **This SKILL analysis workflow is complete!**

Next, the analysis report will be used for:
1. **Plan Confirmation**: Show key content of analysis results to user
2. **Development Reference**: Guide subsequent Python implementation, YAML registration, UT/ST development
3. **Experience Accumulation**: Save as team knowledge base

The caller (dist-op-dev) will guide user to complete the full 6-step development process based on this report.
