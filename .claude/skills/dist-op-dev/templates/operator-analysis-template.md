# HyperParallel Distributed Operator Implementation Plan - {operator_name}

<!--
FILLING GUIDE:
This template contains placeholders in the format {placeholder_name}.
Replace each placeholder with the actual value during the analysis phase.

Common placeholders:
- {operator_name}: Actual operator name (e.g., MatMul, matmul)
- {ms_mint_interface}: MindSpore mint interface (e.g., mint.matmul)
- {torch_interface}: PyTorch interface (e.g., torch.matmul)
- {category}: Operator category - ElementWise / MatMul / Reduce / Reshape / Gather
- {semantic_description}: Brief description of operator functionality
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
| **MindSpore mint Interface** | {ms_mint_interface} | e.g., `mint.matmul` |
| **PyTorch Interface** | {torch_interface} | e.g., `torch.matmul` |
| **Operator Category** | {category} | ElementWise / MatMul / Reduce / Reshape / Gather, etc. |
| **Semantic Description** | {semantic_description} | Operator functionality overview |

### Naming Mapping

| Level | MindSpore | PyTorch |
|------|------|------|
| Interface Name | {ms_mint_interface} | {torch_interface} |
| Primitive/ATen Name | {ms_primitive_name} | {torch_aten_name} |
| YAML Entry Name | {yaml_name_ms} | {yaml_name_torch} |

---

## 2. MindSpore Interface Analysis

### Interface Definition

- **Source Location**: `{ms_interface_file}`
- **Full Signature**: `{ms_full_signature}`

### Input Parameters

| Parameter Name | Type | Required | Default | Constraints | Description |
|--------|------|------|--------|------|------|
{ms_input_params_table}

### Output Information

| Attribute | Value | Description |
|------|-----|------|
| **Output Count** | {ms_output_count} | |
| **Output Type** | {ms_output_type} | Tensor/Scalar/Tuple, etc. |
| **Output Shape** | {ms_output_shape} | Shape change relative to input |

### Primitive Mapping

- **Primitive Name**: `{ms_primitive_name}`
- **YAML Definition**: `{ms_yaml_file}`
- **Generated Primitive Class**: `{ms_generated_class}`

---

## 3. PyTorch Interface Analysis

### Interface Definition

- **Source Location**: `{torch_interface_file}`
- **Full Signature**: `{torch_full_signature}`

### Input Parameters

| Parameter Name | Type | Required | Default | Constraints | Description |
|--------|------|------|--------|------|------|
{torch_input_params_table}

### Output Information

| Attribute | Value | Description |
|------|-----|------|
| **Output Count** | {torch_output_count} | |
| **Output Type** | {torch_output_type} | Tensor/Scalar/Tuple, etc. |
| **Output Shape** | {torch_output_shape} | Shape change relative to input |

### Distributed Strategy Reference

- **DTensor Strategy File**: `{torch_strategy_file}` (e.g., `torch/distributed/tensor/_ops/_matrix_ops.py`)
- **Registration Pattern**: `{torch_registration}` (e.g., `@register_op_strategy(aten.matmul.default)`)
- **Strategy Summary**: {torch_strategy_summary}

---

## 4. Distributed Implementation Plan

### Supported Sharding Scenarios

| Scenario | Input Layout | Output Layout | Sharding Strategy | Notes |
|------|------------|------------|----------|------|
{supported_scenarios_table}

### Unsupported Scenarios (with explicit errors)

| Scenario | Reason | Error Message |
|------|------|------|
{unsupported_scenarios_table}

### Layout Inference Logic (🔴 Output Code Logic Summary)

| Scenario | Input Layouts | Output Layout | Inference Logic |
|------|------------|------------|----------|
{layout_inference_table}

- **Partial Status**: {need_partial}, Condition: {partial_condition}, Type: {partial_type}

### get_expand_impl Analysis

- **Need get_expand_impl**: {need_expand_impl}
- **Trigger Condition**: {expand_trigger_condition}
- **Implementation Logic**: {expand_impl_logic}
- **Scaling Formula**: {scaling_formula}

---

## 5. HyperParallel Reference Implementation

### Recommended Base Class

- **Base Class**: `{recommended_base_class}`
- **Reason**: {base_class_reason}
- **File**: `{base_class_file}`

### Recommended infer_layout_suffix

- **Suffix**: `{recommended_suffix}`
- **Reason**: {suffix_reason}

### Similar Operator Reference

- **Reference Operator**: `{reference_op_1}`
- **Reference File**: `{reference_file_1}`
- **Reusable Logic**:
  - `infer_layout`: {reusable_infer_1}
  - `get_expand_impl`: {reusable_expand_1}

### Implementation Strategy

**Implementation Method**: `{implementation_approach}` (Scenario 0/1/2: Scenario 0=fully use base class, Scenario 1=extend base class, Scenario 2=fully custom)

#### infer_layout Implementation

```python
def infer_layout(self, layouts, extra_args):
    # 1. Input validation
    {input_validation_code}

    # 2. Layout inference
    {layout_inference_code}

    # 3. Partial handling (if needed, important)
    {partial_handling_code}

    return {return_statement}
```

#### get_expand_impl Implementation (if needed)

```python
def get_expand_impl(self, func, output_layout, layouts, extra_args):
    # Trigger condition judgment
    {trigger_condition_code}

    # Return expanded implementation or None
    return {expand_return}
```

### YAML Registration Configuration

```yaml
# MindSpore YAML entry
{ms_operator_name}:
  dist_op_name: {ms_dist_op_name}
  distributed_op_class: {distributed_class}
  distributed_op_file: {distributed_file}
  infer_layout_suffix: {suffix}

# PyTorch YAML entry (if different)
{torch_operator_name}:
  dist_op_name: {torch_dist_op_name}
  distributed_op_class: {distributed_class}
  distributed_op_file: {distributed_file}
  infer_layout_suffix: {suffix}
```

---

## 6. Checklist

### Analysis Coverage

- [ ] MindSpore mint interface definition analyzed
- [ ] MindSpore Primitive mapping identified
- [ ] PyTorch interface definition analyzed
- [ ] PyTorch DTensor strategy reference found
- [ ] Supported sharding scenarios listed with layout inference logic
- [ ] Unsupported scenarios listed with explicit error messages
- [ ] `get_expand_impl` requirements determined
- [ ] Partial status handling determined

### HyperParallel Preparation

- [ ] Base class `{recommended_base_class}` is appropriate
- [ ] `infer_layout_suffix` set to `{recommended_suffix}`
- [ ] Have read `{reference_file_1}` implementation code
- [ ] Implementation method `{implementation_approach}` determined
- [ ] UT test scenarios (supported + unsupported) planned
- [ ] ST 8-card environment test cases prepared

### Risk Points

| Risk Point | Impact | Mitigation Measure |
|--------|------|----------|
{risk_points_table}

---

## 7. Reference Materials

### MindSpore Source Code

| File | Purpose | Key Functions |
|------|------|----------|
| `{ms_interface_file}` | mint interface definition | - |
| `{ms_yaml_file}` | YAML primitive definition | - |
| `{ms_generated_file}` | Generated primitive class | - |

### PyTorch Source Code

| File | Purpose | Key Functions |
|------|------|----------|
| `{torch_interface_file}` | PyTorch interface definition | - |
| `{torch_strategy_file}` | DTensor distributed strategy | - |

### HyperParallel Reference

| File | Reference Operator | Reference Content |
|------|----------|----------|
| `{local_ref_file_1}` | `{reference_op_1}` | `infer_layout`, `get_expand_impl` |

---

## 8. Verification Record

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
