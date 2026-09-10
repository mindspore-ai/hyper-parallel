---
name: project-overview
description: Global project context — hard rules that prevent common distributed bugs (canonical shortlist; link here instead of restating)
---

# Project Overview

## Project Identity

HyperParallel — distributed parallel library for Ascend NPU + Nvidia GPU, PyTorch + MindSpore backends.

## Hard Rules (violating these causes bugs)

Canonical shortlist for always-on context. Full patterns:
`.agent/rules/distributed.md`. Do **not** duplicate long explanations in agents/skills.

- Platform-agnostic code uses `get_platform()` during the staged Platform retirement
- Torch-only DFunction imports Torch directly; do not reintroduce Platform or MindSpore dispatch there
- `layout.is_partial()` is a **method**, not a property — must call with parentheses
- `handle.wait()` must be called before accessing async collective output
- `reduce_partial()` must be called before `redistribute()` when layout has partial state
- Storage freed via `resize_(0)` must not be accessed afterward
- Cross-stream access requires event sync: `event.record(stream_A)` → `event.wait(stream_B)`

## Architecture Invariants

- Framework-neutral features use the platform abstraction layer (`platform/`).
- `hyper_parallel/core/multicore/` is Torch-only: direct Torch imports, no Platform dispatch or
  MindSpore implementation. Its SHMEM component is private to Multicore. This exception does not
  extend to other `core/` modules except the explicitly Torch-only DFunction.
- `hyper_parallel/core/shard/dfunction.py` inherits directly from `torch.autograd.Function` and has
  no MindSpore implementation.
- DTensor = local shard + DeviceMesh + Placements
- Distributed ops registered via YAML (`core/shard/ops/yaml/`)
