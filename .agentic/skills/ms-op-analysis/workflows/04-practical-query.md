# Workflow 4: Practical Query Steps and Fallback Handling

## Goal

Given a specific operator name, execute complete analysis query flow, and handle fallback when distributed implementation is not found.

## Input

- **Operator Name**: User-provided operator name (e.g., `MatMul`, `Add`, `FlashAttentionScore`)
- **MindSpore Source Code Path**: Validated valid path

## Output

- **Analysis Results**: YAML definition, Info class implementation, distributed logic summary
- **Reference Code**: Reference sources for `infer_layout` and `get_expand_impl`
- **Fallback Plan**: If distributed implementation not found, provide alternative plan based on local reference

---

## Complete Query Flow

### Step 1: Validate Operator Name Format

First confirm the naming style of operator name to determine correct query method:

| Naming Style | Examples | Query Method |
|--------------|----------|--------------|
| PascalCase | `MatMul`, `BatchNorm` | Direct mapping, convert to snake_case to query YAML |
| snake_case | `matmul`, `batch_norm` | Use directly as YAML file name to query |
| snake_case primitive name | `matmul`, `add_n` | Query YAML file directly |

**Conversion Rules:**
- PascalCase → snake_case: `MatMul` → `matmul`, `BatchNorm` → `batch_norm`
- Info class name: snake_case + `Info`: `matmul` → `MatMulInfo`

### Step 2: Query YAML Definition

```bash
# Find YAML file
cd {mindspore_path}
ls mindspore/ops/op_def/yaml/{op_name}_op.yaml

# Find documentation YAML
ls mindspore/ops/op_def/yaml/doc/{op_name}_op.yaml
```

**If YAML file not found:**
1. Check if name conversion is correct (PascalCase → snake_case)
2. Try different naming variants (e.g., `matmul` vs `matmul_op`)
3. If still not found, proceed to **Fallback Handling 4.4**

### Step 3: Analyze YAML Content

Read YAML file, extract key information:

```yaml
prim: matmul
inputs:
  - name: x
    dtype: tensor
  - name: w
    dtype: tensor
outputs:
  - name: output
    dtype: tensor
attrs:
  - name: transpose_a
    dtype: bool
    param_type: required
  - name: transpose_b
    dtype: bool
    param_type: optional
```

**Information to Record:**
- Input parameters: name, type, constraints
- Output info: type, shape changes
- Attribute parameters: name, type, default value, required or not

### Step 4: Query Info Class Implementation

```bash
# Find Info class file
find {mindspore_path}/mindspore/ccsrc/frontend/parallel/ops_info/ -name "*{op_name}_info.cc"
```

**Content to Analyze:**

1. **Inheritance Relationship**: What is the base class
2. **Key Methods**: Which distributed inference methods are implemented
   - `InferTensorMap()` → Strategy flow
   - `InferTensorInfo()` → Layout flow
   - `replace_graph()` → Subgraph replacement
3. **Input Validation**: How to validate input Layout compatibility
4. **Output Inference**: How to infer output Layout/Strategy

**If Info class file not found:**
- Proceed to **Fallback Handling 4.4**

### Step 5: Extract Distributed Implementation Reference

Based on Info class implementation type, extract corresponding reference code:

#### Type A: Only Layout Flow Implemented

```cpp
// Reference source: InferTensorInfo or InferOutputTensorInfo
Status MatMulInfo::InferOutputTensorInfo() {
  // Output inference logic
  // This logic corresponds to hyper-parallel's infer_layout implementation
  return Status::OK();
}
```

**hyper-parallel Implementation Reference:**
- Input validation logic: Check input Layout compatibility
- Output inference logic: Calculate output tensor_map

#### Type B: Only Strategy Flow Implemented

```cpp
// Reference source: InferTensorMap
Status MatMulInfo::InferTensorMap() {
  // Infer output TensorMap
  // This logic corresponds to hyper-parallel's tensor_map return value
  return Status::OK();
}
```

**hyper-parallel Implementation Reference:**
- Strategy enumeration: All valid sharding strategies
- tensor_map inference: Infer output distribution based on inputs

#### Type C: Graph Replacement Implemented

```cpp
// Reference source: replace_graph or ReplaceNodeInputOrAttrs
Status MatMulInfo::replace_graph(const CNodePtr &cnode) {
  // Construct AllReduce / AllGather / ReduceScatter communication operators
  // This logic corresponds to hyper-parallel's get_expand_impl
  return Status::OK();
}
```

**hyper-parallel Implementation Reference:**
- Communication operator construction: Usage of AllReduce/AllGather/ReduceScatter
- Subgraph replacement: How to replace original node with communication operators

---

## Fallback Handling Flow

### 4.4.1 Case Classification

| Case | Description | Handling Method |
|------|-------------|-----------------|
| YAML file not found | Operator name incorrect or doesn't exist | Ask user to confirm correct operator name |
| Info class not found | Operator doesn't have distributed support | Provide alternative plan based on similar operators |
| Info class implementation empty | Operator distributed implementation incomplete | Infer based on base class and common patterns |

### 4.4.2 Alternative Plan Based on Similar Operators

If target operator's distributed implementation not found, find semantically related operators:

```bash
# Find similar operators' Info classes
# Example: Add not found, search for Mul, Sub and other binary operators
ls {mindspore_path}/mindspore/ccsrc/frontend/parallel/ops_info/*add_info.cc
ls {mindspore_path}/mindspore/ccsrc/frontend/parallel/ops_info/*mul_info.cc
```

**Similar Operator Reference Table:**

| Target Operator | Similar Operators | Reference Value |
|-----------------|-------------------|-----------------|
| `Sub` | `Add`, `Mul` | Binary operator pattern |
| `Div` | `Mul` | Binary operator pattern |
| `Pow` | `Mul` | Binary operator pattern |
| `MatMul` | `BatchMatMul` | Matrix operation pattern |
| `Concat` | `Split` | Tensor concatenation/split pattern |

### 4.4.3 Inference Plan Based on Local Reference

If no reference in MindSpore, based on hyper-parallel's existing operator implementations:

#### Base Class Selection Guide

| Operator Type | Recommended Base Class | Reference Implementation |
|---------------|------------------------|-------------------------|
| Element-wise binary operator | `ElementWiseBinaryOp` | Add, Mul |
| Matrix multiplication | `MatMulDistributedOp` | MatMul |
| Reduction operator | `ReduceOp` | ReduceSum |
| Concatenation operator | `ConcatOp` | Concat |
| Split operator | `SplitOp` | Split |

#### Implementation Strategy Inference

1. **Element-wise Operators**:
   - Inputs must have the same Layout
   - Output Layout same as input Layout
   - Communication operator: AllReduce (if reduction needed)

2. **Matrix Multiplication Operators**:
   - Support different sharding on matrix dimensions
   - Output Layout inference: `out_map = in1_map op in2_map` (op depends on sharding dimension)
   - Communication operator: AllReduce (if sum needed), AllGather (if broadcast needed)

3. **Reduction Operators**:
   - Reduced on specified dimension
   - Output tensor_map on that dimension is -1 (not sharded)
   - Communication operator: ReduceScatter (if sharding needed) + AllReduce

### 4.4.4 Generate Fallback Analysis Report

Even if MindSpore's distributed implementation is not found, still need to generate analysis report, mark source as "local inference":

```markdown
## Distributed Implementation Plan

### ⚠️ Note

MindSpore distributed implementation for this operator not found in source code. The following plan is based on local reference and operator semantics inference.

### Base Class Selection
Recommended base class: `ElementWiseBinaryOp`
Reference implementation: `hyper_parallel/ops/add.py`

### Layout Inference Logic
- **Input Validation**: Two inputs must have the same Layout
- **Output Inference**: Output Layout same as input Layout

### Communication Operators (if needed)
- Case: When reduction operation is needed
- Operator: AllReduce(op=SUM)
- Reference: Implementation in `hyper_parallel/ops/add.py`
```

---

## Success Criteria

- [ ] Executed complete query flow (steps 1-5)
- [ ] Extracted key logic of distributed implementation
- [ ] If distributed implementation not found, executed fallback handling (4.4.1-4.4.4)
- [ ] Generated complete analysis results (including reference code annotations)
- [ ] Recorded whether it's direct reference or fallback inference

---

## Next Step

After practical query is complete, proceed to **[Workflow 5: Analysis Result Output](./05-analysis-output.md)**

**Input:** Complete analysis results, reference code annotations, fallback plan (if applicable)
**Goal:** **Generate standardized analysis report document** (🔴 **This step is required and cannot be skipped**)
