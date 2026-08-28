---
name: platform
description: Cross-platform development rules for PyTorch and MindSpore backends
paths:
  - hyper_parallel/platform/**
---

## Platform Abstraction

- Every feature must work behind the abstraction layer — platform-specific logic goes in `platform/torch/` and `platform/mindspore/` respectively
- **Backend lazy import:** In `platform/torch/**` and `platform/mindspore/**`, import `torch` / `mindspore` (and related modules) **inside methods** when needed, with `# pylint: disable=C0415`. Elsewhere in the repo, prefer module-level imports per `code-style.md`.
- Use `from hyper_parallel.platform import get_platform` to auto-detect backend
- `HYPER_PARALLEL_PLATFORM` env var can force `"torch"` or `"mindspore"`
- All collective ops (`all_reduce`, `all_gather`, `reduce_scatter`) go through `platform.*`
- `DTensorBase` and `Tensor` are platform-specific tensor types

## Cross-Platform Checklist

- When modifying `platform/torch/`, check if `platform/mindspore/` needs a corresponding change
- When modifying `platform/mindspore/`, check if `platform/torch/` needs a corresponding change
- Abstract Platform base class is in `platform/platform.py` — new platform APIs must be added there first
- Test on both backends when possible; at minimum, ensure the other backend's imports don't break
