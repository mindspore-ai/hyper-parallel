# Workflow 1: Operator Analysis

## Goal

Obtain the operator's interface definition, distributed implementation plan, and HyperParallel local implementation reference to provide basis for subsequent implementation.

## Input

- **Operator Name**: Operator name (MindSpore: `MatMul`, PyTorch: `matmul` or `torch.matmul`)
- **Platform Type**: User-provided when available, otherwise auto-detected

## Output

- **Analysis Report File**: `{OpName}-analysis.md` (🔴required)
- **Report Location**: `.claude/skills/dist-op-dev/analysis-results/`
- **Not committed to Git**: Clearly marked at the beginning of the document, saved in local skill directory

---

## Step 1: Auto-detect Operator Type

Determine platform based on operator name format:

| Format | Platform | Examples |
|--------|----------|----------|
| PascalCase | MindSpore | `MatMul`, `BatchMatMul`, `Add` |
| snake_case | PyTorch | `matmul`, `batch_matmul`, `add` |

---

## Step 2: Call Corresponding Analysis Tool

**MindSpore Operator**: Automatically call `ms-op-analysis` SKILL to query YAML primitive definition, analyze Info class distributed implementation, extract Layout flow and subgraph replacement logic.

**PyTorch Operator**: Automatically call `pt-op-analysis` SKILL to query PyTorch interface semantics, map to MindSpore operator, call `ms-op-analysis` to analyze distributed implementation.

---

## Step 3: Get MindSpore Source Code Path (if needed)

If the prompt does not provide MindSpore path, ask the user:

```text
Please provide the absolute path to MindSpore source code, for example:
- Linux/Mac: /home/user/workspace/mindspore
- Windows: D:\workspace\mindspore

Path requirements:
- Contains `mindspore/ops/op_def/yaml/*_op.yaml` and `mindspore/ops/op_def/yaml/doc/*_doc.yaml` (operator definitions)
- Contains `mindspore/ccsrc/frontend/parallel/ops_info/*_info.cc` and `mindspore/ccsrc/frontend/parallel/ops_info/*_info.h` (distributed implementations)
```

---

## Step 4: Generate Analysis Report (🔴Required Step)

**Read Template**: `templates/operator-analysis-template.md`

**Fill Content**:

1. Operator basic information (name, platform, category, naming mapping)
2. Interface definition analysis (input parameters, output information, attribute parameters, constraints)
3. **Distributed Implementation Plan (Core)**
   Perform distributed logic analysis based on MindSpore operator implementation.

   The analysis should include the following aspects:

   - **Input constraint and layout validation logic analysis**
     - Identify how the operator validates distributed strategy or layout compatibility.
     - Analyze constraints between input tensor layouts (e.g. same tensor_map, shardable axes).
     - Identify conditions that may raise errors in distributed scenarios.

   - **Layout inference logic analysis**
     - Determine how the output layout is derived from input layouts.
     - Extract tensor_map propagation rules.
     - Identify whether layout is inherited, transformed, or newly constructed.
     - Identify cases involving broadcast, dimension mapping, or axis transformation.
     - If the operator introduces reduction or aggregation across a sharded dimension, determine whether the output layout needs a `partial` attribute (e.g., `partial = ['sum']`) and set it accordingly.

   - **Graph replacement or communication expansion logic analysis**
     - Identify whether the operator introduces distributed communication.
     - Typical patterns include:
       - AllReduce
       - ReduceScatter
       - AllGather
       - local computation + aggregation
     - Analyze the semantic purpose of the communication.
     - Only recognize the communication pattern for analysis. Do **not** generate or call any communication operators in the code.
     - If graph replacement exists, explain the transformation strategy.

   - **Source code discovery strategy**
     - Do not rely on fixed function names.
     - Instead scan the operator implementation and identify functions related to:
       - input validation
       - layout inference
       - communication inference
       - graph replacement
     - Typical candidate functions may include:
       - `CheckStrategy`
       - `CheckInputLayout`
       - `InferTensorMap`
       - `InferOutputTensorInfo`
       - `InferForwardCommunication`
       - `InferBias`
       - `replace_graph`
       - `ReplaceNodeInputOrAttrs`

4. HyperParallel local implementation reference
   - Recommended base class
   - Similar operator reference
   - Implementation strategy planning
5. Implementation checklist

**Save Path**: `.claude/skills/dist-op-dev/analysis-results/{OpName}-analysis.md`

---

## Success Criteria

- [ ] Analyzed operator's primitive definition (input/output/parameters)
- [ ] Analyzed distributed Info class implementation (Layout flow/subgraph replacement)
- [ ] Marked MindSpore reference source (function name, file location, key code)
- [ ] Recommended HyperParallel base class and implementation method
- [ ] **[Key] Generated analysis report file** `.claude/skills/dist-op-dev/analysis-results/{OpName}-analysis.md`
- [ ] User confirmed: interface definition, reference source, base class selection, implementation plan

