---
description: Testing conventions and patterns — ST launcher must stay free of torch/mindspore imports
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

- **Unit tests** (`tests/ut/`): Run without distributed setup, no GPU required for logic tests
- **MindSpore ST** (`tests/mindspore/st/`): Distributed via `msrun`, or one-card in-process ST
- **Torch distributed ST** (`tests/torch/**` launchers): Spawn workers via `torchrun`

## ST launcher import rule (mandatory)

**Goal:** pytest collection / parent launcher process must **not** pay `torch` /
`torch_npu` / `mindspore` / `hyper_parallel` import cost just to spawn workers.
Workers still import frameworks normally.

### Do

- Spawn from launchers with:
  - `from tests.common.distributed_launcher import torchrun_case, msrun_case`
  - and/or `from tests.common.parallel_case import parallel_run, TorchCase, MindSporeCase`
- Keep `tests/torch/__init__.py` and `tests/mindspore/__init__.py` limited to
  setting `HYPER_PARALLEL_PLATFORM` — **never** import `hyper_parallel` there.
- Put worker bodies in `_test_*.py` (or other non-`test_` modules). Launchers only
  pass file + case name into `torchrun_case` / `msrun_case` / `TorchCase` /
  `MindSporeCase`.
- For **in-process** ST that must use the framework: use a **thin** `test_*.py`
  (marks + `importlib` call only) and move heavy code to `_…_impl.py` so
  collecting the thin module does not load the framework.
- Name shared helpers **without** the `test_` prefix (e.g. `fully_shard_common.py`),
  so pytest does not collect them during suite discovery.

### Do not

- Import `tests.torch.utils` or `tests.mindspore.st.utils` from a launcher —
  those modules import `torch` / `mindspore`. They are **worker-only**.
- Import `hyper_parallel`, `torch`, or `mindspore` at module top of a
  `test_*.py` that only orchestrates `parallel_run` / `torchrun_case` /
  `msrun_case`.
- Mix launcher cases and heavy in-process assertions in the same `test_*.py`
  without splitting (launcher stays light; in-process goes to another module
  or `_…_impl.py`).
- Import case packages / backends at suite-planning time when AST metadata
  scan is enough (see `tests/shard_ops/framework` + `load_case_plan_from_package`).

### Quick check

After editing a launcher, importing the module in a clean interpreter must not
load `torch`, `mindspore`, or `hyper_parallel` into `sys.modules`.

## Patterns

- Mock distributed environments for unit tests — don't require actual multi-GPU setup
- Use `torchrun_case(file_name=..., case_name=..., master_port=..., num_proc=N)` /
  `msrun_case(...)` from `tests.common.distributed_launcher` (signatures as in that module)
- Prefer `parallel_run([TorchCase(...), ...])` / `MindSporeCase` when several
  cases share an 8-card device budget
- Gracefully skip tests when required hardware is unavailable
- Each test function should be self-contained — avoid shared mutable state between tests
