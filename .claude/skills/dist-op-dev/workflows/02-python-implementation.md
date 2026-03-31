# Workflow 2: Python Code Implementation

## Goal

Create distributed operator implementation class, implement `infer_layout` and `get_expand_impl` methods.

## Input

- **Analysis Report**: `.claude/skills/dist-op-dev/analysis-results/{OpName}-analysis.md` (from Step 1 output)

## Output

- **Implementation File**: `hyper_parallel/core/shard/ops/parallel_*.py`
- **Implementation Class**: Inherited from `DistributedOp` or its subclass

---

## Step 1: Determine Implementation Scenario

Before implementing the operator, determine which scenario applies.

**Scenario 0: Fully Use Base Class (Recommended)**
**Applicable Conditions:** Operator semantics completely match the base class, no custom logic needed.
**Implementation Method**: Do NOT define a new class in Python file, directly specify the base class in YAML registration file, the framework will automatically create a base class instance.
**Note**: No need to write any Python code, only YAML registration is needed, and **no need to write UT test cases**.

---

**Scenario 1: Extend Base Class**
**Applicable Conditions**: Main logic matches base class, but has special parameters or additional validation/preprocessing logic.
**Implementation Method**: Inherit base class and override `infer_layout`, complete validation before calling `super().infer_layout`.

---

**Scenario 2: Fully Custom**
**Applicable Conditions**: Operator semantics do not match any existing base class, need fully custom layout inference logic.
**Implementation Method**: Inherit `DistributedOp` and override `infer_layout` and `get_expand_impl`, handle all edge cases.

---

## Step 2: Implement `infer_layout(layouts, extra_args)`

**Purpose:**
`infer_layout` is responsible for: Inferring the output layout based on input layouts.
This function must perform **layout reasoning only**.

**Implementation Reference Priority**:
1. **First Priority**: Analysis report from Step 1 — supported/unsupported scenarios, layout inference logic
2. **Second Priority**: Local reference operator (e.g., `parallel_matmul.py`) `infer_layout` implementation
3. **Third Priority**: Base class default implementation (only applicable to simple scenarios)

**Core Implementation Logic**:
1. Validate Inputs: Follow MindSpore input validation logic.
2. Parse Input Layouts:Extract layout information from the input list.
3. Derive Tensor Map: Compute the tensor map for the output layout.
4. Construct Output Layout: Create the output Layout object.
5. Set Partial State (If Required): If the operator introduces distributed communication (e.g., reduction), the partial state **must be set**.

```python
def infer_layout(self, layouts, extra_args):
    # 1. Input validation (check mesh shape match, layout compatibility)
    self._validate_inputs(layouts, extra_args)

    # 2. Infer output tensor_map (core logic from analysis report)
    output_tensor_map = self._compute_output_tensor_layout(layouts)

    # 3. Construct output Layout
    output_layout = Layout(...)
    out_layout = output_layout(*output_tensor_map)

    # 4. Set partial status if needed
    if need_partial:
        out_layout.set_partial_by_dev_axis(axis, 'sum')

    return out_layout
```

    # 5. You must set partial state if communication is required(important)
    if need_reduce:
        output_layout.set_partial_by_dev_axis(axis_name, "sum")

    return output_layout
```
---

## Step 3: Implement `get_expand_impl(func, output_layout, layouts, extra_args)`

**Purpose:**
`get_expand_impl` describe the **distributed expansion semantics** of the operator.
Reference file: `.claude/skills/dist-op-dev/analysis-results/{OpName}-analysis.md`

**Implementation Reference Priority**:
1. **First Priority**: Analysis report from Step 1 — get_expand_impl analysis section
2. **Second Priority**: Local reference operator `get_expand_impl` implementation
3. **Third Priority**: Return `None` (use original operator, no special handling)

---

## Success Criteria

- [ ] Selected correct implementation method (Scenario 0/1/2)
- [ ] Implemented `infer_layout` method (Scenario 1/2) or confirmed using base class (Scenario 0)
- [ ] Implemented `get_expand_impl` (if needed per analysis report) or returned None
- [ ] Handled all supported scenarios listed in analysis report
- [ ] Unsupported scenarios raise explicit `ValueError` with clear error messages matching analysis report
- [ ] Code complies with Pylint/Lizard standards
- [ ] Referenced analysis report for layout inference logic and sharding strategies

