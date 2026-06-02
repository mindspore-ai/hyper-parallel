---
name: dist-op-analysis
description: Distributed operator analysis. Analyzes operator interfaces provided by the user and outputs a standardized implementation plan. Requires human confirmation before development begins.
---

# Distributed Operator Analysis

## Full Distributed Operator Development Workflow

This skill is the entry point of the full workflow. The complete chain is:

```
1. /dist-op-analysis   Analyze interfaces → output plan → STOP, wait for human confirmation
         ↓  (human confirms plan)
2. /dist-op-dev        Read plan → implement → write tests → run tests → fix until all pass
                       Verification report saved to dist-op-dev/reports/
         ↓
3. /commit  (autogit)  Lint check → commit → push to fork branch
         ↓
4. /gate-doctor        Monitor CI gate → retest flakes → fix failures → gate turns green
```

Steps 3 and 4 are standard for any PR and are not specific to distributed operators.

---

## Trigger

The user provides any combination of:
- PyTorch interface (e.g. `torch.nn.functional.linear`)
- MindSpore mint interface (e.g. `mint.sort`)
- MindSpore Primitive name (e.g. `SortExt`)

## Output Language

**Write the plan in Chinese.** All section headings, descriptions, constraint rules, test case names and their descriptions in the plan file must be in Chinese. Code identifiers, file paths, and YAML snippets remain in their original form.

## Step 1: Confirm Platform Scope

Determine the target platform(s) from the provided interfaces and **state them explicitly in the plan**:

| User provides | Platform | Confirmation needed |
|--------------|----------|-------------------|
| torch interface only | PyTorch only | Confirm MindSpore is out of scope? |
| mint interface only / Primitive only | MindSpore only | Confirm PyTorch is out of scope? |
| torch + mint / torch + Primitive | Both platforms | No extra confirmation needed |

**If only a single-platform interface is provided, the plan must explicitly state the platform scope and ask the user to confirm.**

## Step 2: MindSpore Interface Analysis

### 2.1 mint Interface → Primitive Mapping

Check `mindspore/mindspore/ops/api_def/{op_name}.yaml`:

- **`kwonlyargs` field present** → mint interface supports keyword arguments (functional_overload mode), behaves like PyTorch. Parameters listed in `kwonlyargs` go into `kwargs` in `_normalize_*_args`.
- **`kwonlyargs` field absent** → mint interface is all-positional (Primitive mode); `kwargs` is empty in `_normalize_*_args`.

If `api_def/{op_name}.yaml` does not exist, trace the import path in `mindspore/python/mindspore/mint/__init__.py` to locate the underlying Primitive name (PascalCase), and treat as all-positional.

**The plan must state**: the mint interface → Primitive name mapping and whether `kwonlyargs` exists (with the specific parameter names).

### 2.2 Primitive Interface Details

Read the Primitive definition from:
- Structure: `mindspore/mindspore/ops/op_def/yaml/{primitive_snake_case}_op.yaml` (args/returns/dtypes)
- Python kwonlyargs: `mindspore/mindspore/ops/api_def/{op_name}.yaml` (if `kwonlyargs` field exists)
- Docs: `mindspore/mindspore/ops/op_def/yaml/doc/{primitive_snake_case}_doc.yaml`

Extract: parameter names, dtypes, shape constraints, hardware support (Atlas A2/A3/950, etc.), known limitations.

### 2.3 PyTorch Interface Details

Refer to the official PyTorch documentation for the interface signature, parameter semantics, and constraints.

## Step 3: Distributed Layout Derivation Analysis

Based on the operator semantics, analyze the following and write them into the plan:

1. **Which dimensions are naturally independent** (can be sharded): e.g. batch, head, sequence
2. **Which dimensions cannot be sharded**: e.g. reduce dim, rotation dim
3. **Broadcast and shard constraints for auxiliary inputs** (e.g. cos/sin)
4. **Output layout derivation rule**: deepcopy(input), new layout, or Partial
5. **Whether `get_expand_impl` is needed**: required when distributed result differs from single-device result
6. **`cache_values` composition**: Layout objects first, then scalars; only include information that affects layout inference
7. **Allowed shard scenarios**: list valid scenarios in a table (mesh, tensor_map per input); invalid scenarios are described in the constraint rules section, not in this table

## Step 4: Implementation Design

Design file paths and content skeletons following `templates/plan-template.md`:

### 4.0 Base Class Selection (must be stated in the plan)

Evaluate in order and explicitly state the choice and rationale in the plan:

1. **YAML-only registration** (no new Python file): operator has purely element-wise semantics (output shape = broadcast of inputs, any dimension can be independently sharded) — add an entry to an existing YAML file, reusing `ElementWiseDistributedOp` or `ElementWiseWithPartialDistributedOp`.
2. **Inherit an existing base class**: operator semantics are close to an existing base class but need minor customization (e.g. adding `_MS_PRIMITIVE_OP_NAMES` routing, extra validation) — subclass the nearest base class and override only what differs.
   - `ElementWiseDistributedOp` / `ElementWiseWithPartialDistributedOp` — element-wise
   - `TupleElementWiseDistributedOp` — element-wise with tuple inputs/outputs
   - `ReshapeDistributedOp` — changes shape without rearranging data
3. **New class + three-phase dispatch**: when no existing base class fits — create a new class implementing `preprocess → infer_layout(cache_values) → get_expand_impl`.

**Must NOT** create a new class directly from `DistributedOp` when an existing specialized base class covers the operator's layout semantics.

### 4.1 File and Interface Design

Key requirements:

- **`_validate_input_layouts`**: only required for new classes using three-phase dispatch; defined as `@staticmethod`, called by `infer_layout` after `_check_partial_inputs`. Performs layout validity checks only (sharding constraints, mesh consistency, etc.) — no derivation.
- **MindSpore ST**: the Impl file must call the operator via the `mint.{op_name}` interface. Direct use of the Primitive class is forbidden. Users access operators through mint, so ST must validate that path.
- **PyTorch ST**: if the operator interface is a native torch interface (not `torch_npu`), the Runner must include a `_gloo` variant function.

## Step 5: Output the Plan

Save the plan to:
```
hyper-parallel/.agent/skills/dist-op-analysis/plans/{OpName}_dist_op_plan.md
```

This directory is excluded by `.gitignore` and will not be committed.

Fill in the complete plan following `templates/plan-template.md`, then **stop and wait for user confirmation**. Do not begin any code development.

If there is any ambiguity (Primitive name cannot be determined, kwonlyargs need verification), mark the item as `[待确认]` in the plan — do not guess.
