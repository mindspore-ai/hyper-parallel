# Code Standards Requirements

This document details the code standard requirements that need to be followed during HyperParallel distributed operator development.

---

## Lint Check Tools

autogit automatically runs the following checks before committing:

| Tool | Check Content | Threshold |
|------|---------------|-----------|
| **pylint** | Python code style | max-line-length=120 |
| **lizard** | Cyclomatic complexity | CCN≤19, NLOC≤100 |
| **codespell** | Spelling check | - |
| **markdownlint** | Markdown format | Standard rules |

---

## Pylint Requirements

### Must Pass Check Items

| Error Code | Description | Example |
|------------|-------------|---------|
| **E0602** | Using undefined variable | `Undefined variable 'layout'` |
| **E1101** | Instance has no such member | `Module 'layout' has no 'to_dict' member` |
| **F0010** | Import error | `Unable to import 'hyper_parallel'` |
| **E1128** | Assigning to function call result | `Assignment to function call` |

### Ignorable Warnings

| Warning Code | Reason | Suggestion |
|--------------|--------|------------|
| **C0114** | Missing module docstring | Can ignore, but recommend adding |
| **R0913** | Too many function arguments (>7) | Can ignore if necessary |
| **C0301** | Line too long (>100 chars) | Can ignore, but recommend keeping readable |
| **R0903** | Too few public methods in class | Can ignore for utility classes |

### Pylint Configuration Example

```ini
# .pylintrc
[MESSAGES CONTROL]
disable=
    C0114,  # Missing module docstring
    R0913,  # Too many arguments
    C0301,  # Line too long

[DESIGN]
max-args=8
max-locals=15
max-returns=6
max-branches=12
max-statements=50
```

### Local Check Commands

```bash
# Check single file
pylint hyper_parallel/core/shard/ops/parallel_elementwise.py

# Check multiple files
pylint hyper_parallel/core/shard/ops/parallel_*.py tests/mindspore/ut/parallel_ops_infer/*.py

# Ignore specific warnings
pylint --disable=C0114,R0913 hyper_parallel/core/shard/ops/parallel_elementwise.py
```

---

## Lizard Requirements

### Core Metrics

| Metric | Requirement | Recommended |
|--------|-------------|-------------|
| **Function Length** | < 100 lines | < 80 lines |
| **Cyclomatic Complexity (CCN)** | < 19 | < 10 |
| **Parameter Count** | < 8 | < 6 |

### Methods to Reduce Complexity

#### Method 1: Split Long Functions

```python
# Before (function too long: 120 lines)
def infer_layout(self, layouts, extra_args):
    # 50+ lines of validation
    # 50+ lines of computation
    # 20+ lines of post-processing
    pass

# After (refactored)
def infer_layout(self, layouts, extra_args):
    """Infer output layout based on input layouts."""
    self._validate_inputs(layouts, extra_args)
    self._check_broadcast_compatibility(layouts, extra_args)
    return self._compute_output_layout(layouts)

def _validate_inputs(self, layouts, extra_args):
    """Validate input layouts and extra arguments."""
    # validation logic
    pass

def _check_broadcast_compatibility(self, layouts, extra_args):
    """Check if broadcast is compatible."""
    # broadcast check logic
    pass

def _compute_output_layout(self, layouts):
    """Compute output layout based on input layouts."""
    # computation logic
    pass
```

#### Method 2: Reduce CCN (Reduce Nesting)

```python
# Before (high CCN: 25)
def infer_layout(self, layouts, extra_args):
    if layouts:
        if layouts[0]:
            if condition1:
                if condition2:
                    if condition3:
                        # nested logic
    return output

# After (low CCN: 8)
def infer_layout(self, layouts, extra_args):
    """Infer output layout based on input layouts."""
    if not layouts or not layouts[0]:
        raise ValueError("Invalid layouts")

    if condition1:
        return self._handle_condition1(layouts)

    if condition2 and condition3:
        return self._handle_condition2(layouts)

    return self._default_layout(layouts)
```

### Local Check Commands

```bash
# Check single file
lizard hyper_parallel/core/shard/ops/parallel_elementwise.py

# Check multiple files
lizard hyper_parallel/core/shard/ops/parallel_*.py

# Set custom thresholds
lizard -L 100 -C 19 -a 8 hyper_parallel/core/shard/ops/parallel_*.py
```

---

## Type Annotation Requirements

### Type Annotation Usage

**Public functions must have type annotations:**

```python
# ❌ Wrong: Missing type annotations
def infer_layout(self, layouts, extra_args):
    pass

# ✅ Correct: Complete type annotations
from typing import Tuple, Optional, Any

def infer_layout(
    self,
    layouts: Tuple[Optional['Layout'], ...],
    extra_args: Tuple[Any, ...]
) -> 'Layout':
    """Infer output layout based on input layouts."""
    pass
```

**Optional types use Optional:**

```python
# Scalar input may be None
def infer_layout(
    self,
    layouts: Tuple[Optional['Layout'], ...],
    extra_args: Tuple[Any, ...]
) -> Optional['Layout']:
    """Infer output layout based on input layouts."""
    if not layouts:
        return None
    pass
```

---

## Docstring Requirements

### Docstring Format

**Use NumPy style or Google style:**

```python
def infer_layout(
    self,
    layouts: Tuple[Optional['Layout'], ...],
    extra_args: Tuple[Any, ...]
) -> 'Layout':
    """
    Infer output layout based on input layouts.

    Args:
        layouts: Layouts of input tensors.
        extra_args: Additional arguments.

    Returns:
        Output layout.

    Raises:
        ValueError: If layouts are invalid or incompatible.
    """
    pass
```

### Elements That Must Have Docstrings

| Element | Requirement |
|---------|-------------|
| **Public classes** | ✅ Must have docstring |
| **Public methods** | ✅ Must have docstring |
| **Private methods** | ⚙️ Recommend adding |
| **Complex functions** (CCN > 5) | ✅ Must have docstring |

---

## Test Requirements

### Unit Testing (UT)

| Requirement | Standard |
|-------------|----------|
| **Test file naming** | `test_parallel_<category>_ops.py` |
| **Test function naming** | `test_<scenario>_<operator>()` |
| **Test decorator** | Use `@pytest.mark.parametrize` for parameterization |
| **Docstring** | Each test function must have docstring |
| **Coverage** | Core logic coverage ≥ 80% |

### Integration Testing (ST)

| Requirement | Standard |
|-------------|----------|
| **Test file naming** | `test_ops_<operator>.py` or `*_shard_in_python.py` |
| **Test decorator** | Use `@arg_mark` decorator |
| **Test environment** | 8-card distributed environment |
| **Output comparison** | Must compare standalone vs distributed output (rtol=1e-5, atol=1e-8) |

### Test Commands

```bash
# UT tests
pytest tests/mindspore/ut/parallel_ops_infer/ -v

# Single UT file
pytest tests/mindspore/ut/parallel_ops_infer/test_parallel_elementwise_ops.py -v

# Single test case
pytest tests/mindspore/ut/parallel_ops_infer/test_parallel_elementwise_ops.py::test_broadcast_scalar_operand -v

# Debug mode
pytest --pdb tests/mindspore/ut/parallel_ops_infer/test_parallel_elementwise_ops.py::test_broadcast_scalar_operand

# Generate coverage report
pytest --cov=hyper_parallel tests/mindspore/ut/parallel_ops_infer/ --cov-report=html
```

---

## Git Commit Standards

autogit uses **Conventional Commits** standard:

### Format

```
<type>(<scope>): <subject>

<body>
```

### Type Categories

| Type | Description |
|------|-------------|
| **feat** | New feature |
| **fix** | Bug fix |
| **refactor** | Refactoring |
| **docs** | Documentation |
| **test** | Testing |
| **chore** | Build/tooling |

### Example

```
feat(shard): add Greater operator distributed support

- Add GreaterDistributedOp class with broadcast support
- Register Greater in element_wise_ops_with_shape.yaml
- Add UT cases covering DP/MP and broadcast scenarios
- Add ST cases for 8-card verification
```

---

## Success Criteria

| Check Item | Success Criteria |
|------------|------------------|
| **Pylint** | No errors, warning count < 10 |
| **Lizard** | Function length < 100, CCN < 19 |
| **UT Test** | All passed |
| **ST Test** | All passed (8-card environment) |
| **Coverage** | ≥ 80% (if applicable) |

---

## References

- **Git Commit and PR Standards**: `../workflows/06-git-commit.md`
- **Quick Reference**: `./quick-reference.md`
