# Workflow 2: Python Code Implementation

## Goal

Create distributed operator implementation class, implement `infer_layout` and `get_expand_impl` methods.

## Input

- **Analysis Report**: `.agentic/skills/dist-op-dev/analysis-results/{OpName}-analysis.md` (from Step 1 output)

## Output

- **Implementation File**: `hyper_parallel/core/shard/ops/parallel_*.py`
- **Implementation Class**: Inherited from `DistributedOp` or its subclass

---

## Implementation Scenarios

### Scenario 0: Fully Use Base Class (Recommended)

**Applicable Conditions**: Operator semantics completely match the base class, no custom logic needed.

**Implementation Method**: Do NOT define a new class in Python file, directly specify the base class in YAML registration file, the framework will automatically create a base class instance.

**Note**: No need to write any Python code, only YAML registration is needed, and **no need to write UT test cases**.

---

### Scenario 1: Extend Base Class

**Applicable Conditions**: Main logic matches base class, but has special parameters or additional validation/preprocessing logic.

**Implementation Method**: Inherit base class and override `infer_layout`, complete validation before calling `super().infer_layout`.

---

### Scenario 2: Fully Custom

**Applicable Conditions**: Operator semantics do not match any existing base class, need fully custom layout inference logic.

**Implementation Method**: Inherit `DistributedOp` and override `infer_layout` and `get_expand_impl`, handle all edge cases.

---

## Core Method Implementation Guide

### `infer_layout(layouts, extra_args)` Implementation Points

**Implementation Reference Priority**:
1. **First Priority**: Layout inference logic output by `ms-op-analysis` SKILL
2. **Second Priority**: Local reference operator (e.g., `parallel_matmul.py`) `infer_layout` implementation
3. **Third Priority**: Base class default implementation (only applicable to simple scenarios)

**Core Implementation Logic**:
```python
def infer_layout(self, layouts, extra_args):
    # 1. Input validation (refer to MindSpore input validation logic)
    self._validate_inputs(layouts, extra_args)

    # 2. Infer output tensor_map (core logic, refer to MindSpore output inference logic)
    output_tensor_map = self._compute_output_tensor_layout(layouts)

    # 3. Construct output Layout
    output_layout = Layout(...)
    out_layout = output_layout(*output_tensor_map)
    return out_layout
```

### `get_expand_impl(func, output_layout, layouts, extra_args)` Implementation Points

**Implementation Reference Priority**:
1. **First Priority**: `replace_graph` and `ReplaceNodeInputOrAttrs` implementation output by `ms-op-analysis` SKILL
2. **Second Priority**: Local reference operator `get_expand_impl` implementation
3. **Third Priority**: Return `None` (use original operator, no special handling)

---

## Success Criteria

- [ ] Selected correct implementation method (Scenario 0/1/2)
- [ ] Implemented `infer_layout` method (Scenario 1/2) or confirmed using base class (Scenario 0)
- [ ] Implemented `get_expand_impl` (if needed) or returned None
- [ ] Handled all edge cases (None input, abnormal parameters, broadcast, etc.)
- [ ] Code complies with Pylint/Lizard standards
- [ ] Referenced MindSpore reference source from analysis report

---

## Next Step

After Python implementation is complete, proceed to **[Workflow 3: YAML Registration](./03-yaml-registration.md)**

**Input:** Analysis report, Python implementation class info
**Goal:** Register operator in YAML config file, configure infer_layout_suffix
