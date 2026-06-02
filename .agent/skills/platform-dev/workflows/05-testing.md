# Workflow 5: Testing

## Goal

Add unit tests (UT) and system/distributed tests (ST) covering the platform changes for both backends.

## Steps

### 5.1 Unit Tests (UT)

**Location:**
- PyTorch (unit): `tests/ut/` (e.g. `tests/ut/core/`, `tests/ut/platform/torch/`)
- MindSpore: `tests/mindspore/ut/`

**Conventions:**
- Use pytest with `@arg_mark` markers from `tests/common/mark_utils.py`
- No distributed setup needed for UT
- Test both normal and edge cases
- Use platform abstraction in tests where possible

**Template:**
```python
# Copyright 2024 Huawei Technologies Co., Ltd
# (Apache 2.0 license header)

import pytest
from tests.common.mark_utils import arg_mark

@arg_mark(...)
def test_new_feature_basic():
    """Test basic functionality of the new feature."""
    from hyper_parallel.platform import get_platform
    platform = get_platform()
    # Test implementation
    ...

@arg_mark(...)
def test_new_feature_edge_case():
    """Test edge cases."""
    ...
```

### 5.2 Distributed Tests (ST)

**Location:**
- PyTorch: `tests/torch/st/`
- MindSpore: `tests/mindspore/st/`

**Conventions:**
- Use `torchrun_case()` for PyTorch distributed tests (8-card)
- Use `msrun_case()` for MindSpore distributed tests (8-card)
- Compare distributed output with single-card reference

**Template (PyTorch):**
```python
from tests.common.mark_utils import arg_mark
from tests.common.torch_utils import torchrun_case

@arg_mark(...)
def test_feature_distributed():
    """Test feature in distributed setting."""
    torchrun_case(
        nprocs=8,
        test_fn=_test_feature_impl,
    )

def _test_feature_impl():
    from hyper_parallel.platform import get_platform
    platform = get_platform()
    # Distributed test implementation
    ...
```

### 5.3 Test Coverage Checklist

| Scenario | UT | ST |
|----------|----|----|
| Basic functionality | Required | If applicable |
| Error handling (invalid inputs) | Required | - |
| Async operations (handle.wait) | If applicable | Required |
| Cross-stream synchronization | - | Required |
| Memory lifecycle (resize_(0)) | If applicable | If applicable |
| Multiple process groups | - | If applicable |
| Edge: single rank | If applicable | - |
| Edge: empty tensor | If applicable | - |

### 5.4 Run Tests

```bash
# Run UT
/test

# Or manually
pytest tests/ut/path/to/test.py -v
pytest tests/mindspore/ut/path/to/test.py -v
```

## Output

- Test files in `tests/torch/` and/or `tests/mindspore/`
- All tests passing

## Next Step

Proceed to **[Workflow 6: Git Commit](./06-commit.md)**
