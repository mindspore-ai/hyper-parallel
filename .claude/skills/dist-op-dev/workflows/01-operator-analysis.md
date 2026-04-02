# Workflow 1: Operator Analysis

## Goal

Obtain the operator's interface definition, distributed implementation plan, and HyperParallel local implementation reference to provide basis for subsequent implementation.

## Input

- **MindSpore mint Interface**: e.g., `mint.matmul`, `mint.nn.functional.relu`
- **PyTorch Interface**: e.g., `torch.matmul`, `torch.nn.functional.linear`
- **MindSpore Source Code Path**: Absolute path to MindSpore source (e.g., `/root/workspace/mindspore`)
- **PyTorch Source Code Path**: Absolute path to PyTorch source (e.g., `/root/workspace/pytorch`)

## Output

- **Analysis Report File**: `{OpName}-analysis.md` (🔴required)
- **Report Location**: `.claude/skills/dist-op-dev/analysis-results/`
- **Not committed to Git**: Clearly marked at the beginning of the document, saved in local skill directory

---

## Step 1: Generate MindSpore Primitive Info (🔴required)

Execute the following command to generate MindSpore operator primitive information:

```bash
python {ms_path}/mindspore/python/mindspore/ops_generate/gen_ops.py
```

This generates operator primitive classes from YAML definitions. The generated files are located at:
- `mindspore/python/mindspore/ops/auto_generate/gen_ops_prim.py` — Python primitive interfaces (✅ use this, NOT `array_ops.py` stubs)
- `mindspore/ops/op_def/auto_generate/` — C++ Primitive definitions

**Critical**: The auto-generated `gen_ops_prim.py` contains the real Primitive class definitions. Do NOT use hand-written stubs in `ops/operations/array_ops.py` — those are fallback wrappers, not the actual Primitives.

---

## Step 2: Call dist-op-analysis SKILL

**Critical**: NEED to run this step after step 1.

Call the `dist-op-analysis` SKILL with the following information:
- MindSpore mint interface name (e.g., `mint.matmul`)
- PyTorch operator name (e.g., `torch.matmul`)
- MindSpore source code path
- PyTorch source code path

The agent will:
1. Locate interface definitions from the provided source paths
2. Trace mint → the underlying function (e.g., `tril_ext`) → the auto-generated Primitive class from `gen_ops_prim.py`
3. Trace ATen op strategies for PyTorch from `torch/distributed/tensor/_ops/`
4. Provide layout derivation logic and sharding strategies
5. Recommend HyperParallel implementation approach (suffix, base class, get_expand_impl needs)

**Strategy selection rule**: When MindSpore and PyTorch distributed strategies differ, **always prefer the more feature-rich strategy** (the one that supports more sharding dimensions). For example, if MindSpore supports sharding batch + last 2 dims while PyTorch only supports replication, choose MindSpore's approach and note PyTorch's limitation as a risk point.

---

## Step 3: Synthesize Analysis Results

Combine the dist-op-analysis SKILL output with the generated MindSpore primitive info to produce:
- Complete interface specifications for both MindSpore and PyTorch
- Distributed sharding strategies from both frameworks
- HyperParallel implementation recommendations (base class, suffix, get_expand_impl)

---

## Step 4: Generate Analysis Report (🔴Required Step)

**Read Template**: `templates/operator-analysis-template.md`

**Fill Content**:

1. Operator basic information (MindSpore mint interface name, PyTorch interface name, category, naming mapping)
2. **MindSpore analysis**: interface definition, Primitive mapping, distributed strategy reference
3. **PyTorch analysis**: interface definition, ATen op strategy reference from `torch/distributed/tensor/_ops/`
4. **Distributed Implementation Plan (Core)**
   - Supported sharding scenarios (DP, MP, Hybrid, CP, etc.)
   - Unsupported scenarios with explicit error messages
   - Layout inference logic for each supported scenario
   - `get_expand_impl` requirements and formulas
5. HyperParallel local implementation reference
   - Recommended base class and inheritance strategy
   - Recommended `infer_layout_suffix`
   - Similar operator reference
   - Implementation strategy planning
6. Implementation checklist

**Save Path**: `.claude/skills/dist-op-dev/analysis-results/{OpName}-analysis.md`

---

## Success Criteria

- [ ] Executed `gen_ops.py` to generate MindSpore primitive info (MUST run before using dist-op-analysis SKILL)
- [ ] Called dist-op-analysis SKILL with source code paths
- [ ] Primitive mapping traced to auto-generated `gen_ops_prim.py` class (NOT hand-written stubs)
- [ ] Analyzed MindSpore mint interface definition and Primitive mapping
- [ ] Analyzed PyTorch interface definition and distributed strategy
- [ ] When MindSpore and PyTorch strategies differ, selected the more feature-rich strategy
- [ ] Listed supported distributed scenarios with layout inference logic
- [ ] Listed unsupported scenarios with explicit error messages
- [ ] Recommended HyperParallel base class, suffix, and implementation method
- [ ] **[Key] Generated analysis report file** `.claude/skills/dist-op-dev/analysis-results/{OpName}-analysis.md`
- [ ] User confirmed: interface definition, supported/unsupported scenarios, base class selection, implementation plan

