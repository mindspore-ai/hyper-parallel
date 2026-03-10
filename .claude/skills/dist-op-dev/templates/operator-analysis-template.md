# HyperParallel Distributed Operator Implementation Plan - {operator_name}

<!-- 
FILLING GUIDE:
This template contains placeholders in the format {placeholder_name}. 
Replace each placeholder with the actual value during the analysis phase.

Common placeholders:
- {operator_name}: Actual operator name (e.g., MatMul, Greater)
- {platform}: Platform type - "MindSpore" or "PyTorch"
- {category}: Operator category - ElementWise / MatMul / Reduce / Reshape / Gather
- {semantic_description}: Brief description of operator functionality
- {yaml_name}: YAML primitive definition file name
- {primitive_class}: Generated Python class name
- {info_class}: Distributed implementation class name (MindSpore Info class)
- {generation_time}: Timestamp when this document was generated

For detailed placeholder descriptions, refer to the analysis workflow documentation.
-->

> **Document Purpose**: This file is an intermediate product of the operator analysis phase, summarizing the operator's interface definition, distributed implementation plan, and reference sources for confirmation before implementation and reference in subsequent processes.
> **Document Status**: ⚠️ Local experience file, not committed via Git
> **Generation Time**: {generation_time}

---

## 1. Operator Basic Information

| Attribute | Value | Description |
|------|-----|------|
| **Operator Name** | {operator_name} | |
| **Platform Type** | {platform} | MindSpore / PyTorch |
| **Operator Category** | {category} | ElementWise / MatMul / Reduce / Reshape / Gather, etc. |
| **Semantic Description** | {semantic_description} | Operator functionality overview |

### Naming Mapping

| Level | Name | Description |
|------|------|------|
| YAML Definition | {yaml_name} | Primitive definition file name |
| Primitive Class Name | {primitive_class} | Generated Python class name |
| Info Class Name | {info_class} | Distributed implementation class name |
| PyTorch Interface | {pytorch_name} | If applicable |

---

## 2. Interface Definition

### Input Parameters

| Parameter Name | Type | Required | Default | Constraints | Description |
|--------|------|------|--------|------|------|
{input_params_table}

### Output Information

| Attribute | Value | Description |
|------|-----|------|
| **Output Count** | {output_count} | |
| **Output Type** | {output_type} | Tensor/Scalar/Tuple, etc. |
| **Output Shape** | {output_shape} | Shape change relative to input |

### Attribute Parameters

| Attribute Name | Type | Default | prim_init | Constraints | Description |
|--------|------|--------|-----------|------|------|
{attr_params_table}

### Constraint Conditions

**Input Constraints**：{input_constraints}

**Attribute Constraints**：{attr_constraints}

---

## 3. Distributed Implementation Plan

### Input Constraints and Layout Validation

- **Reference Source**: `{check_input_layout_func}` @ `{check_input_layout_file}`
- **Validation Rules**:
  - {validation_rule1}
  - {validation_rule2}
- **Convert to HyperParallel**: Perform the same validation at the beginning of `infer_layout()`

### Layout Inference Plan (🔴 Output Code Logic Summary)

| Scenario | Input Layout | Output Layout | Inference Logic |
|------|------------|------------|----------|
{layout_inference_table}

- **Reference Source**: `{infer_output_func}` @ `{infer_output_file}`
- **Partial Status**: {need_partial}, Condition: {partial_condition}, Type: {partial_type}

### Input Replacement/Subgraph Replacement (🔴 Output Code Logic Summary)

- **ReplaceNodeInputOrAttrs**: `{replace_input_func}` @ `{replace_input_file}`
- **replace_graph**: `{replace_graph_func}` @ `{replace_graph_file}`
- **Replacement Rules**: {replacement_rules}
- **Communication Operators**: {communication_ops}

---

## 4. HyperParallel Reference Implementation

### Recommended Base Class

- **Base Class**: `{recommended_base_class}`
- **Reason**: {base_class_reason}
- **File**: `{base_class_file}`

### Similar Operator Reference

- **Reference Operator**: `{reference_op_1}`
- **Reference File**: `{reference_file_1}`
- **Reusable Logic**:
  - `infer_layout`: {reusable_infer_1}
  - `get_expand_impl`: {reusable_expand_1}

### Implementation Strategy

**Implementation Method**: `{implementation_approach}` (Scenario 0/1/2)

#### infer_layout Implementation

```python
def infer_layout(self, layouts, extra_args):
    # 1. Input validation (refer to {check_input_layout_func})
    {input_validation_code}

    # 2. Layout inference (refer to {infer_output_func})
    {layout_inference_code}

    # 3. Partial handling (if needed)
    {partial_handling_code}

    return {return_statement}
```

#### get_expand_impl Implementation (if needed)

```python
def get_expand_impl(self, func, output_layout, layouts, extra_args):
    # Trigger condition judgment (refer to {replace_graph_func})
    {trigger_condition_code}

    # Return expanded implementation or None
    return {expand_return}
```

### YAML Registration Configuration

```yaml
{operator_name}:
  dist_op_name: {dist_op_name}
  distributed_op_class: {distributed_class}
  distributed_op_file: {distributed_file}
  infer_layout_suffix: {suffix}
```

**Selection Reason**: {suffix_reason}

---

## 5. Checklist

### MindSpore Analysis Coverage

- [ ] `{check_input_layout_func}` validation logic
- [ ] `{infer_output_func}` inference rules
- [ ] Partial status setting conditions and type
- [ ] `{replace_input_func}` replacement scenarios and logic summary
- [ ] `{replace_graph_func}` graph construction pattern logic summary
- [ ] Communication operator insertion timing

### HyperParallel Preparation

- [ ] Base class `{recommended_base_class}` is appropriate
- [ ] Have read `{reference_file_1}` implementation code
- [ ] Implementation method `{implementation_approach}` determined
- [ ] YAML `suffix` set to `{suffix}`
- [ ] UT test scenarios (DP/MP/broadcast) planned
- [ ] ST 8-card environment test cases prepared

### Risk Points

| Risk Point | Impact | Mitigation Measure |
|--------|------|----------|
{risk_points_table}

---

## 6. Reference Materials

### MindSpore Source Code

| File | Purpose | Key Functions |
|------|------|----------|
| `{yaml_file_ms}` | YAML primitive definition | - |
| `{info_header}` | Info class declaration | Class definition |
| `{info_impl}` | Info class implementation | `{check_input_layout_func}`, `{infer_output_func}`, `{replace_graph_func}` |

### HyperParallel Reference

| File | Reference Operator | Reference Content |
|------|----------|----------|
| `{local_ref_file_1}` | `{reference_op_1}` | `infer_layout`, `get_expand_impl` |

---

## 7. Verification Record

| Verification Item | Expected | Actual | Status |
|--------|------|------|------|
| UT Test | Pass | {ut_result} | {ut_status} |
| ST Test | Pass | {st_result} | {st_status} |
| Pylint | Pass | {pylint_result} | {pylint_status} |
| Lizard | Pass | {lizard_result} | {lizard_status} |
| CI Gate | Pass | {ci_result} | {ci_status} |

**Notes**: {verification_notes}

---

*This document is automatically generated by the operator analysis process and is for local reference only. Do not commit to the Git repository.*
