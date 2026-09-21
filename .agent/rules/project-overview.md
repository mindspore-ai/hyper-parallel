---
name: project-overview
description: Global project context — hard rules that prevent common distributed bugs (canonical shortlist; link here instead of restating)
---

# Project Overview

## Project Identity

HyperParallel — distributed parallel library for Ascend NPU + Nvidia GPU, on PyTorch.

## Hard Rules (violating these causes bugs)

Canonical shortlist for always-on context. Full patterns:
`.agent/rules/distributed.md`. Do **not** duplicate long explanations in agents/skills.

- The `platform/` abstraction layer and the MindSpore backend are gone — use native Torch APIs
- Pipeline uses native Torch tensor, autograd, and distributed APIs
- DFunction imports Torch directly; never reintroduce Platform or MindSpore dispatch
- `layout.is_partial()` is a **method**, not a property — must call with parentheses
- `handle.wait()` must be called before accessing async collective output
- `reduce_partial()` must be called before `redistribute()` when layout has partial state
- Storage freed via `resize_(0)` must not be accessed afterward
- Cross-stream access requires event sync: `event.record(stream_A)` → `event.wait(stream_B)`

## Architecture Invariants

- Every module is Torch-only: direct `torch.distributed` / `torch.Tensor` / autograd APIs.
- `hyper_parallel/core/multicore/` keeps direct Torch imports; its SHMEM component is private to
  Multicore.
- `hyper_parallel/core/shard/dfunction.py` inherits directly from `torch.autograd.Function`.
- DTensor = local shard + DeviceMesh + Placements
- Distributed ops registered via YAML (`core/shard/ops/yaml/`)
