---
name: ms-op-analysis
description: 【Internal Tool SKILL】Analyze MindSpore operator primitive definitions and distributed implementations, providing reference for HyperParallel distributed operator development. This SKILL is automatically called by dist-op-dev, no need for users to use directly.
---

# 【Internal Tool】MindSpore Operator Analysis Workflow

> ⚠️ **Important Note**: This SKILL is an **internal tool SKILL**, automatically called by `dist-op-dev`. When developing HyperParallel distributed operators, please call the main SKILL `/skill dist-op-dev` uniformly, which will automatically handle operator analysis.
>
> This SKILL's documentation is for understanding the analysis flow reference, **not recommended to call directly**.

## This SKILL's Responsibility

When the main SKILL needs to analyze MindSpore operators, this SKILL is responsible for:

1. **Query operator primitive definition**: Analyze YAML files, extract input/output/parameter info
2. **Query distributed implementation**: Analyze Info class, extract Layout flow and subgraph replacement logic
3. **Output standardized results**: Provide implementation reference for `infer_layout` and `get_expand_impl`

## When Called

- When user calls `/skill dist-op-dev` and declares to interface with **MindSpore operator**
- When `pt-op-analysis` needs to analyze the mapped MindSpore distributed implementation

---

## Prerequisites

MindSpore code is in a separate repository, need to provide **absolute path** to MindSpore source code for analysis.

**How to get path:**
1. **User provides actively**: User provides path in prompt (e.g., `/d/workspace/mindspore`)
2. **Ask user proactively**: If not provided in prompt, must ask user for MindSpore source code path

**Path requirements:**
- Must contain `mindspore/ops/op_def/yaml/` directory (operator YAML definitions)
- Must contain `mindspore/ccsrc/frontend/parallel/` directory (distributed implementations)
- Recommend using latest version of MindSpore source code

---

## Execution Flow Overview

MindSpore operator analysis follows a **5-step process**, from getting path to generating analysis report:

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  1.Get MindSpore  │ ──▶ │  2.Primitive     │ ──▶ │  3.Distributed   │
│    Source Path    │     │    Query         │     │    Op Query      │
└──────────────────┘     └──────────────────┘     └──────────────────┘
                                                              │
                                                               ▼
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  5.Analysis       │ ◀── │  4.Practical     │ ◀── │                  │
│    Output         │     │    Query Steps   │     │                  │
│  🔴Required output│     │                  │     │                  │
└──────────────────┘     └──────────────────┘     └──────────────────┘
```

### Workflow Execution Checklist

When using this SKILL to analyze MindSpore operators, create a TODOLIST, execute the following workflows in order:

- [ ] **[Step 1](workflows/01-get-mindspore-path.md)**: Get MindSpore Source Code Path
  - Goal: Get absolute path to MindSpore source code, validate path validity
  - Input: User prompt
  - Output: MindSpore source code path (validated)
  - Execution: Get from prompt or ask user; validate required directories exist

- [ ] **[Step 2](workflows/02-primitive-query.md)**: Operator Primitive Query
  - Goal: Read YAML definition, extract operator's input, output, parameter info
  - Input: MindSpore source code path, operator name
  - Output: YAML primitive definition (input parameters, output info, attribute parameters), naming mapping
  - Execution: Locate YAML file; extract interface info; understand naming rules (YAML name → class name → Info class name)

- [ ] **[Step 3](workflows/03-distributed-op-query.md)**: Distributed Operator Query
  - Goal: Locate and analyze MindSpore operator's distributed implementation class (Info class)
  - Input: MindSpore source code path, operator name, naming mapping
  - Output: Info class file path, base class inheritance, registration info, implementation summary
  - Execution: Locate Info class file; analyze class definition; identify key methods (InferTensorInfo/replace_graph, etc.)

- [ ] **[Step 4](workflows/04-practical-query.md)**: Practical Query Steps
  - Goal: Given specific operator name, execute complete analysis query and handle fallback cases
  - Input: Operator name (e.g., `MatMul`), MindSpore source code path
  - Output: Analysis results (YAML definition, Info class implementation, distributed logic summary), fallback plan (if applicable)
  - Execution: Complete 5-step query; handle fallback when distributed implementation not found

- [ ] **[🔴Step 5](workflows/05-analysis-output.md)**: Analysis Result Output (Required, cannot skip)
  - Goal: Generate standardized analysis report document
  - Input: Complete analysis results, reference code annotations, fallback plan (if applicable)
  - Output: Standardized analysis report `.agentic/skills/dist-op-dev/analysis-results/{OpName}-analysis.md`
  - Execution: Read template; fill analysis results; save document; **quality checklist item-by-item confirmation**

> **🔴 Special Warning**: Step 5 (Analysis Result Output) is **required and cannot be skipped**. If report is not generated, subsequent development process will not execute correctly due to missing reference sources!

---

## Key Decision Points

| Decision Point | Criteria | Options | Impact |
|----------------|----------|---------|--------|
| **Inference Flow Type** | Info class implementation | Layout flow/Strategy flow/Hybrid flow | Determines reference method and hyper-parallel mapping |
| **Base Class Selection** | Inheritance relationship | OperatorInfo/Intermediate base class (ArithmeticInfo/MatMulInfoBase, etc.) | Affects inheritance logic analysis |
| **Communication Need** | Sharding dimension | Need AllReduce/AllGather/ReduceScatter/Not needed | Affects get_expand_impl implementation |
| **Fallback or Not** | Distributed implementation | Info class complete/Info class empty/Not exist | Determines output local inference plan |

---

## Core Concept Explanation

### Info Class Two Inference Flows

MindSpore distributed operators have **two inference flows**:

| Flow Type | Core Methods | Characteristics | Applicable Scenarios |
|-----------|--------------|-----------------|---------------------|
| **Layout Flow** | `InferTensorInfo()`, `InferInputTensorInfo()`, `InferOutputTensorInfo()` | Uses `TensorInfo` and `TensorLayout` abstraction, more flexible | All operators (recommended) |
| **Strategy Flow** | `InferTensorMap()`, `InferDevMatrixShape()` | Directly infers `tensor_map` and `dev_matrix_shape` | Simple operators, legacy operators |

**hyper-parallel mapping rules:**
- Layout flow → Complete `infer_layout` implementation
- Strategy flow → `tensor_map` part in `infer_layout`

### Communication Operator Types

| Communication Operator | Purpose | MindSpore Correspondence | hyper-parallel Correspondence |
|------------------------|---------|-------------------------|------------------------------|
| **AllReduce** | Reduction (Sum/Max/Min/Prod) | `ReplaceNode` constructs AllReduce | `platform.all_reduce()` |
| **AllGather** | Collect shards | `ReplaceNode` constructs AllGather | `platform.all_gather()` |
| **ReduceScatter** | Reduction then scatter | `ReplaceNode` constructs ReduceScatter | `platform.reduce_scatter()` |

---

## Quick Reference

### Output Document Structure

Generated analysis report `.agentic/skills/dist-op-dev/analysis-results/{OpName}-analysis.md` contains:

1. **Operator Interface Definition**: Input, output, parameter info
2. **Distributed Implementation Plan** (Core):
   - Input constraint and Layout validation → Reference MindSpore input validation logic
   - Layout inference → Reference MindSpore output inference logic
   - Input replacement/subgraph replacement → Reference MindSpore graph replacement logic
3. **HyperParallel Local Reference**: Base class selection, similar operators, implementation strategy
4. **Reference Source Annotation**: MindSpore function name, file location, key code

### Common Commands

```bash
# Find YAML file
ls mindspore/ops/op_def/yaml/{op_name}_op.yaml

# Find Info class file
find . -name "*{op_name}_info.cc"

# Validate path
ls mindspore/ops/op_def/yaml/ | head -5
ls mindspore/ccsrc/frontend/parallel/ops_info/ | head -5
```

### Typical Operator Mapping

| Operator | Info Class | Base Class | Inference Flow |
|----------|------------|------------|----------------|
| MatMul | MatMulInfo | OperatorInfo | Layout |
| Add | AddInfo | ArithmeticInfo | Strategy → Layout |
| BatchMatMul | BatchMatMulInfo | MatMulInfoBase | Layout |
| ReduceSum | ReduceSumInfo | OperatorInfo | Layout |

---

## Reference Document Paths

- **Workflow detailed steps**: `workflows/` directory
  - Get path: `workflows/01-get-mindspore-path.md`
  - Primitive query: `workflows/02-primitive-query.md`
  - Distributed query: `workflows/03-distributed-op-query.md`
  - Practical query: `workflows/04-practical-query.md`
  - Analysis output: `workflows/05-analysis-output.md` (🔴Required)

- **Knowledge reference documents**: `references/` directory
  - Layout/Strategy mechanism details: `references/operator-info-mechanism.md`
  - Typical cases: `references/typical-cases.md` (MatMul/Add/FlashAttentionScore)
  - File path summary: `references/reference-docs.md`
  - Quick reference: `references/quick-reference.md`

---

## Execution Constraints

1. **First step must complete**: Getting MindSpore source code path is the foundation for all subsequent analysis, no path means no analysis possible
2. **Execute in order**: Core flow (1-5) must execute in numbered order, subsequent depends on previous knowledge
3. **Input requirements**: When providing operator name, must specify naming format (prefer YAML snake_case naming, e.g., `matmul` not `MatMul`)
4. **Incomplete marking**: If distributed implementation not found, must execute fallback handling flow (step 4.4), cannot terminate analysis directly
5. **🔴Required output**: After analysis completes **must** execute Workflow 5 "Analysis Result Output", generate `.agentic/skills/dist-op-dev/analysis-results/{OpName}-analysis.md` document
6. **Checklist**: After completion, must confirm item-by-item against the checklist in Workflow 5
