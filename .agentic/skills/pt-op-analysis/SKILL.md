---
name: pt-op-analysis
description: 【Internal Tool SKILL】Analyze PyTorch operator interfaces and map to MindSpore operators, providing reference for HyperParallel distributed operator development. This SKILL is automatically called by dist-op-dev, no need for users to use directly.
---

# 【Internal Tool】PyTorch Operator Analysis Workflow

> ⚠️ **Important Note**: This SKILL is an **internal tool SKILL**, automatically called by `dist-op-dev`. When developing HyperParallel distributed operators, please call the main SKILL `/skill dist-op-dev` uniformly, which will automatically handle operator analysis.
>
> This SKILL's documentation is for understanding the analysis flow reference, **not recommended to call directly**.

## This SKILL's Responsibility

When the main SKILL needs to analyze PyTorch operators, this SKILL is responsible for:

1. **Identify PyTorch operator interface**: Analyze PyTorch operator semantics and parameters
2. **Map to MindSpore operator**: Find semantically equivalent MindSpore operator
3. **Call MindSpore analysis**: Reuse `ms-op-analysis` distributed implementation analysis

## When Called

- When user calls `/skill dist-op-dev` and declares to interface with **PyTorch operator**

---

## Prerequisites

- **MindSpore code path**: Need MindSpore source code path to analyze distributed implementation
- **PyTorch operator name**: PyTorch operator name provided by user (e.g., `torch.add`, `torch.matmul`)

---

## Execution Flow Overview

PyTorch operator analysis follows a **5-step process**, from identifying interface to generating analysis report:

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  1.Identify       │ ──▶ │  2.Map to        │ ──▶ │  3.Call MindSpore│
│    PyTorch        │     │    MindSpore     │     │    Analysis      │
│    Interface      │     │    Operator      │     │    SKILL         │
└──────────────────┘     └──────────────────┘     └──────────────────┘
                                                              │
                                                               ▼
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  5.Analysis       │ ◀── │  4.Integrate     │ ◀── │                  │
│    Output         │     │    Results       │     │                  │
│  🔴Required output│     │    PyTorch+MS    │     │                  │
└──────────────────┘     └──────────────────┘     └──────────────────┘
```

### Workflow Execution Checklist

When using this SKILL to analyze PyTorch operators, create a TODOLIST, execute the following workflows in order:

- [ ] **[Step 1](workflows/01-identify-interface.md)**: Identify PyTorch Operator Interface
  - Goal: Analyze PyTorch operator semantics, parameters, return values
  - Input: PyTorch operator name
  - Output: Operator semantics, parameter list, return value type

- [ ] **[Step 2](workflows/02-map-to-mindspore.md)**: Map to MindSpore Operator
  - Goal: Find semantically equivalent MindSpore operator
  - Input: PyTorch operator semantics
  - Output: MindSpore operator name, mapping relationship, difference description

- [ ] **[Step 3](workflows/03-call-mindspore-analysis.md)**: Call MindSpore Analysis SKILL
  - Goal: Reuse `ms-op-analysis` to analyze distributed implementation
  - Input: MindSpore operator name, MindSpore source code path
  - Output: MindSpore distributed implementation analysis results

- [ ] **[Step 4](workflows/04-integrate-results.md)**: Integrate Analysis Results
  - Goal: Integrate PyTorch interface info and MindSpore distributed implementation
  - Input: PyTorch interface info, MindSpore analysis results
  - Output: Complete analysis results (including both platforms' mapping relationship)

- [ ] **[🔴Step 5](workflows/05-analysis-output.md)**: Analysis Result Output (Required, cannot skip)
  - Goal: Generate standardized analysis report document
  - Input: Complete analysis results
  - Output: Standardized analysis report `.agentic/skills/dist-op-dev/analysis-results/{OpName}-analysis.md`

> **🔴 Special Warning**: Step 5 (Analysis Result Output) is **required and cannot be skipped**. If report is not generated, subsequent development process will not execute correctly due to missing reference sources!

---

## Key Decision Points

| Decision Point | Criteria | Options | Impact |
|----------------|----------|---------|--------|
| **Mapping Method** | Semantic equivalence level | Direct mapping/Composite mapping/No mapping | Determines analysis complexity |
| **Parameter Difference** | Parameter name and default value | Fully consistent/Partial difference/Completely different | Determines adapter code volume |
| **Distributed Implementation Reuse** | MindSpore has implementation | Direct reuse/Partial reference/Fully custom | Determines development effort |

---

## PyTorch and MindSpore Operator Mapping Rules

### Naming Style Differences

| Platform | Naming Style | Examples |
|----------|--------------|----------|
| **PyTorch** | snake_case | `torch.add`, `torch.matmul`, `torch.nn.functional.linear` |
| **MindSpore** | PascalCase | `Add`, `MatMul`, `Linear` |

### Common Mapping Reference

| PyTorch Operator | MindSpore Operator | Description |
|------------------|-------------------|-------------|
| `torch.add` | `Add` | Element-wise addition |
| `torch.matmul` | `MatMul` | Matrix multiplication |
| `torch.nn.functional.linear` | `Linear` | Linear layer |
| `torch.sum` | `ReduceSum` | Reduction sum |
| `torch.mean` | `ReduceMean` | Reduction mean |
| `torch.cat` | `Concat` | Concatenation |
| `torch.split` | `Split` | Split |
| `torch.gather` | `Gather` | Index selection |

> **Detailed mapping rules**: See [references/mapping-rules.md](references/mapping-rules.md)

---

## Quick Reference

### Output Document Structure

Generated analysis report `.agentic/skills/dist-op-dev/analysis-results/{OpName}-analysis.md` contains:

1. **PyTorch Interface Definition**: Parameters, return values, semantics
2. **MindSpore Mapping Relationship**: Corresponding MindSpore operator, difference description
3. **Distributed Implementation Plan**: Reuse MindSpore analysis results
4. **HyperParallel Implementation Reference**: Base class selection, implementation strategy

### Common Commands

```python
# PyTorch help documentation
import torch
help(torch.add)
help(torch.matmul)

# View function signature
import inspect
print(inspect.signature(torch.add))
```

### Typical Mapping Examples

| PyTorch Interface | MindSpore Primitive | Distributed Implementation Reuse |
|-------------------|---------------------|----------------------------------|
| `torch.add(input, other)` | `Add()` | ✅ Full reuse |
| `torch.matmul(input, other)` | `MatMul()` | ✅ Full reuse |
| `torch.nn.functional.linear(input, weight, bias)` | `Linear()` | ✅ Full reuse |

---

## Reference Document Paths

- **Workflow detailed steps**: `workflows/` directory
  - Identify interface: `workflows/01-identify-interface.md`
  - Map operator: `workflows/02-map-to-mindspore.md`
  - Call analysis: `workflows/03-call-mindspore-analysis.md`
  - Integrate results: `workflows/04-integrate-results.md`
  - Analysis output: `workflows/05-analysis-output.md` (🔴Required)

- **Knowledge reference documents**: `references/` directory
  - Mapping rules: `references/mapping-rules.md`
  - Quick reference: `references/quick-reference.md`

---

## Execution Constraints

1. **Must map to MindSpore**: PyTorch operator analysis must find corresponding MindSpore operator to reuse distributed implementation analysis
2. **Execute in order**: Core flow (1-5) must execute in numbered order
3. **Call ms-op-analysis**: Step 3 must call `ms-op-analysis` SKILL to analyze distributed implementation
4. **🔴Required output**: After analysis completes **must** execute Workflow 5 "Analysis Result Output"
