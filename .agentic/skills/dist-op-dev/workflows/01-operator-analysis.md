# Workflow 1: Operator Analysis

## Goal

Obtain the operator's interface definition, distributed implementation plan, and HyperParallel local implementation reference to provide basis for subsequent implementation.

## Input

- **Operator Name**: Operator name (MindSpore: `MatMul`, PyTorch: `torch.matmul`)
- **Platform Type**: Auto-detected (PascalCase=MindSpore, snake_case=PyTorch)

## Output

- **Analysis Report File**: `{OpName}-analysis.md` (🔴required)
- **Report Location**: `.agentic/skills/dist-op-dev/analysis-results/`
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
- Contains mindspore/ops/op_def/yaml/ directory (operator definitions)
- Contains mindspore/ccsrc/frontend/parallel/ops_info/ directory (distributed implementations)
```

---

## Step 4: Generate Analysis Report (🔴Required Step)

**Read Template**: `templates/operator-analysis-template.md`

**Fill Content**:

1. Operator basic information (name, platform, category, naming mapping)
2. Interface definition analysis (input parameters, output information, attribute parameters, constraints)
3. **Distributed Implementation Plan (Core)**
   - Input constraint and Layout validation (CheckInputLayout) source code logic analysis
   - Layout inference plan (InferOutputTensorInfo) source code logic analysis
   - Input replacement/subgraph replacement plan (ReplaceNodeInputOrAttrs / replace_graph) source code logic analysis
   - **Must mark MindSpore reference source**: function name, file location, key code logic
4. HyperParallel local implementation reference
   - Recommended base class
   - Similar operator reference
   - Implementation strategy planning
5. Implementation checklist

**Save Path**: `.agentic/skills/dist-op-dev/analysis-results/{OpName}-analysis.md`

---

## Success Criteria

- [ ] Analyzed operator's primitive definition (input/output/parameters)
- [ ] Analyzed distributed Info class implementation (Layout flow/subgraph replacement)
- [ ] Marked MindSpore reference source (function name, file location, key code)
- [ ] Recommended HyperParallel base class and implementation method
- [ ] **[Key] Generated analysis report file** `.agentic/skills/dist-op-dev/analysis-results/{OpName}-analysis.md`
- [ ] User confirmed: interface definition, reference source, base class selection, implementation plan

---

## Next Step

After analysis is complete, proceed to **[Workflow 2: Python Implementation](./02-python-implementation.md)**

**Input:** `.agentic/skills/dist-op-dev/analysis-results/{OpName}-analysis.md`

**Goal:** Create distributed operator implementation class, implement `infer_layout` and `get_expand_impl`
