---
name: platform-dev
description: HyperParallel platform abstraction layer development. Use when adding new platform APIs, implementing cross-platform features (FSDP/HSDP/Pipeline/Activation Checkpoint), creating DTensorBase extensions, or modifying collective operations. Covers both PyTorch and MindSpore backends.
---

# HyperParallel Platform Development Skill

> Guides development of cross-platform features in the `platform/` abstraction layer — adding new Platform APIs, implementing backend-specific features (FSDP, HSDP, Pipeline Parallelism, Activation Checkpoint), extending DTensorBase, and managing collective operations across PyTorch and MindSpore backends.

## When to Use This Skill

- Adding a **new method** to the Platform abstraction (`platform/platform.py`)
- Implementing a **new feature** in `platform/torch/` or `platform/mindspore/`
- Modifying **FSDP, HSDP, Pipeline Parallelism, or Activation Checkpoint** platform code
- Extending **DTensorBase** (torch or mindspore)
- Adding or modifying **collective operations** (all_gather, all_reduce, reduce_scatter, etc.)
- Implementing **stream synchronization** or **memory lifecycle** patterns
- Working on **process group management** or **device/RNG management**

## Architecture Overview

```text
platform/
├── platform.py                    # Platform base class (~100+ abstract methods)
├── torch/                         # PyTorch backend
│   ├── platform.py                # TorchPlatform(Platform)
│   ├── dtensor.py                 # DTensorBase (torch.Tensor subclass)
│   ├── function_override.py       # DTensor backward hooks
│   ├── init_weights.py            # init_on_device context manager
│   ├── group_utils.py             # Process group creation
│   ├── clip_grad.py               # Distributed gradient clipping
│   ├── activation_checkpoint/     # SAC + Activation Swap
│   ├── fully_shard/               # FSDP (state, param, scheduler, hooks)
│   ├── hsdp/                      # HSDP (state, param, scheduler, grad hooks)
│   └── pipeline_parallel/         # Pipeline stages + micro-batch
└── mindspore/                     # MindSpore backend
    ├── platform.py                # MindSporePlatform(Platform)
    ├── dtensor.py                 # DTensorBase (ms.Tensor subclass)
    ├── init_weights.py            # init_on_device context manager
    ├── parameter_init.py          # Parameter initialization with slice_index
    ├── platform_graph.py          # Graph construction utilities
    ├── custom_pass/               # Custom graph passes
    ├── fully_shard/               # FSDP (state, param, scheduler, hooks)
    ├── hsdp/                      # HSDP (state, param, scheduler, grad hooks)
    └── pipeline_parallel/         # Pipeline stages + micro-batch
```

## How to Use

Call this skill with your task description:

```bash
# Add a new Platform API
/platform-dev Add a new `scatter()` collective operation to the Platform abstraction

# Implement a feature for one backend
/platform-dev Implement activation swap support for MindSpore backend

# Modify FSDP behavior
/platform-dev Fix the unshard scheduling in torch FSDP to support prefetch

# Extend DTensorBase
/platform-dev Add a new property to DTensorBase for tracking communication state
```

---

## Execution Flow

```text
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  1. Scope        │ ──▶ │  2. Base Class   │ ──▶ │  3. Backend      │
│     Analysis     │     │     API Design   │     │     Implementation│
│  Identify what   │     │  platform.py     │     │  torch/ + ms/    │
│  needs to change │     │  abstract method │     │  concrete impl   │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                           │
            ┌───────────────────────────────────────────────┘
            ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  4. Cross-Platform│ ──▶ │  5. Testing     │ ──▶ │  6. Git Commit   │
│     Verification │     │  UT + ST        │     │  & PR Creation   │
│  Parity check    │     │  Both backends  │     │  Call autogit    │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

### Workflow Execution Checklist

- [ ] **[Step 1](workflows/01-scope-analysis.md)**: Scope Analysis
  - Goal: Identify affected files, understand existing patterns, determine change scope
  - Output: List of files to modify, change strategy

- [ ] **[Step 2](workflows/02-base-class-api.md)**: Base Class API Design
  - Goal: Define/modify abstract methods in `platform/platform.py`
  - Output: Updated Platform base class with new/modified abstract methods

- [ ] **[Step 3](workflows/03-backend-implementation.md)**: Backend Implementation
  - Goal: Implement concrete methods in `platform/torch/` and `platform/mindspore/`
  - Output: Working implementation for both backends (or one with NotImplementedError for the other)

- [ ] **[Step 4](workflows/04-cross-platform-verification.md)**: Cross-Platform Verification
  - Goal: Ensure feature parity, consistent semantics, no abstraction violations
  - Output: Verification report

- [ ] **[Step 5](workflows/05-testing.md)**: Testing
  - Goal: Add UT and ST tests covering both backends
  - Output: Test files in `tests/torch/` and/or `tests/mindspore/`

- [ ] **[Step 6](workflows/06-commit.md)**: Git Commit & PR
  - Goal: Commit, push, and optionally create PR via autogit
  - Output: Feature branch with clean commit

---

## Key Decision Points

| Decision | Criteria | Options | Impact |
|----------|----------|---------|--------|
| **Change Scope** | New API vs modifying existing | New abstract method / Modify existing / Internal only | Files affected, backward compat |
| **Backend Priority** | Which backend first | Torch first / MindSpore first / Both together | Development order |
| **Feature Parity** | Both backends needed? | Full parity / One backend + NotImplementedError | Test coverage |
| **Stream Sync** | Async operations involved? | Sync / Async with handle / Event-based | Correctness risk |
| **Memory Pattern** | Buffer management needed? | resize_(0) / Reuse / Allocate new | Memory efficiency |

---

## Quick Reference

See [references/quick-reference.md](references/quick-reference.md) for:
- File location guide
- Platform API categories
- Cross-platform type mapping
- Common patterns and anti-patterns

See [references/architecture.md](references/architecture.md) for:
- Platform abstraction design
- DTensorBase dispatch mechanism
- FSDP/HSDP state lifecycle
- Stream synchronization patterns
- Memory management patterns

---

## Hard Rules

1. **Never import torch/mindspore directly** in platform-agnostic code — use `get_platform()`
2. **New Platform APIs must be added to base class first** (`platform/platform.py`)
3. **Both backends must be considered** — implement or raise `NotImplementedError`
4. **Cross-platform type differences** — torch uses `torch.device` vs mindspore uses `str`; torch uses `ProcessGroup` vs mindspore uses `str` group names
5. **Lazy imports** — use `# pylint: disable=C0415` inside methods
6. **handle.wait()** before reading async collective output
7. **event.record(src) → event.wait(dst)** for cross-stream dependencies
8. **resize_(0)** to free device memory, never access freed storage

---

## Related Skills

| Skill | When to Use |
|-------|-------------|
| **code-review** | After implementation, review for distributed correctness |
| **autogit** | Commit, push, create PR |
| **dist-op-dev** | When implementing distributed operator support (not platform layer) |
