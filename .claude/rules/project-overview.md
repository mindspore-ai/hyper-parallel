---
description: Global project context — hard rules that prevent common distributed bugs
---

## Project Identity

HyperParallel — distributed parallel library for Ascend NPU + Nvidia GPU, PyTorch + MindSpore backends.

## Hard Rules (violating these causes bugs)

- Never import `torch` or `mindspore` directly in platform-agnostic code — use `get_platform()`
- `layout.is_partial()` is a **method**, not a property — must call with parentheses
- `handle.wait()` must be called before accessing async collective output
- `reduce_partial()` must be called before `redistribution()` when layout has partial state
- Storage freed via `resize_(0)` must not be accessed afterward
- Cross-stream access requires event sync: `event.record(stream_A)` → `event.wait(stream_B)`

## Architecture Invariants

- All features behind platform abstraction layer (`platform/`)
- DTensor = local shard + DeviceMesh + Placements
- Distributed ops registered via YAML (`core/shard/ops/yaml/`)
