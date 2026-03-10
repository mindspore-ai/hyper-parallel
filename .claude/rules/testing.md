---
description: Testing conventions and patterns
paths:
  - tests/**
  - "*_test.py"
  - "test_*.py"
---

## Framework

pytest with custom markers defined in `tests/common/mark_utils.py`.

## Markers

```python
@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",        # level0 (CI gate) / level1 (extended)
    card_mark="allcards",       # allcards / 1card / 2cards
    essential_mark="essential"
)
```

## Test Types

- **Unit tests** (`tests/torch/ut/`, `tests/mindspore/ut/`): Run without distributed setup, no GPU required for logic tests
- **System tests** (`tests/mindspore/st/`): Distributed, require `msrun` launcher
- **Torch distributed tests**: Use `torchrun` via `tests/torch/utils.torchrun_case()`

## Patterns

- Mock distributed environments for unit tests — don't require actual multi-GPU setup
- Use `torchrun_case(nproc_per_node=N, file_name=..., case_name=...)` for PyTorch distributed tests
- Use `msrun_case(glog_v=3, file_name=..., case_name=..., master_port=...)` for MindSpore distributed tests
- Gracefully skip tests when required hardware is unavailable
- Each test function should be self-contained — avoid shared mutable state between tests
