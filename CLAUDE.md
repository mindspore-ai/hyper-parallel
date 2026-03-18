# HyperParallel — CLAUDE.md

## Project Overview

HyperParallel is an **easy-to-use, high-performance distributed parallel acceleration library** for distributed model training, inference and reinforcement learning. It provides unified abstractions for:

- **Data Parallelism (DP)** — replicate model across devices, aggregate gradients
- **Fully Sharded Data Parallelism (FSDP)** — shard parameters, gradients, and optimizer states across data-parallel ranks
- **Tensor Parallelism (TP)** — shard model weights/activations across devices
- **Expert Parallelism (EP)** — distribute MoE expert layers across devices
- **Context Parallelism (CP)** — shard sequence/context dimension across devices
- **Pipeline Parallelism (PP)** — split model stages across devices
- **Activation Checkpoint (Recomputation)** — trade compute for memory by recomputing activations
- **Activation Swap** — swap activations to CPU memory during forward and prefetch during backward
- **Parameter Offload** — offload parameters to CPU/NVMe to reduce device memory usage
- **Optimizer Offload** — offload optimizer states to CPU/NVMe to reduce device memory usage
- **Hybrid strategies** — combine the above freely

Primary target hardware: **Ascend NPU and Nvidia GPU**. Primary framework: **PyTorch and MindSpore**.

---

## Key Modules

| Module | Location | Purpose |
|--------|----------|---------|
| **Platform** | `platform/` (`platform.py`, `torch/`, `mindspore/`) | Abstraction layer — use `get_platform()`, never import torch/mindspore directly |
| **DTensor** | `core/dtensor/` (`dtensor.py`, `device_mesh.py`, `layout.py`, `placement_types.py`, `tensor_redistribution.py`, `redistribute_infer.py`, `random.py`, `init_weights.py`, `parameter_init.py`) | Distributed tensor (local shard + DeviceMesh + Placements); DeviceMesh defines multi-dim device topology; Layout maps tensor-to-mesh; redistribution cached by `compact_str + rank_id` |
| **Shard** | `core/shard/` (`api.py`, `custom_shard.py`, `_op_dispatch.py`, `ops/`) | `shard_module()` / `custom_shard()` entry points; YAML registry (`ops/yaml/`) + Python impl (`ops/parallel_*.py`) |
| **FSDP** | `core/fully_shard/`, `platform/*/fully_shard/` | Parameter sharding/unsharding lifecycle |
| **HSDP** | `core/hsdp/`, `platform/*/hsdp/` | Hybrid shard data parallel (legacy + new scheduler) |
| **Pipeline** | `core/pipeline_parallel/`, `platform/*/pipeline_parallel/` | Stage scheduling, micro-batch, cross-stage comm |
| **Activation** | `core/activation_checkpoint/`, `platform/torch/activation_checkpoint/` | Selective activation checkpoint (SAC) + activation swap (offload to CPU, prefetch on backward) |
| **Checkpoint** | `core/checkpoint/` | Distributed checkpoint save/load (planner, storage, reshard) |
| **Collectives** | `collectives/cc.py` | Process group management |
| **Tests** | `tests/torch/`, `tests/mindspore/` | `ut/` (unit), `st/` (system/distributed) |

---

## Coding Conventions

> Full details in `.claude/rules/code-style.md` (auto-applied globally).

- **License**: Apache 2.0 header on all `.py` files (lines 1-16)
- **Style**: PEP 8, ~120-char lines; C++ Google style (120 chars)
- **Naming**: Classes `PascalCase`, functions/vars `snake_case`, private `_leading_underscore`
- **Docstrings**: Google-style (`Args:`, `Returns:`, `Raises:`, `Example:`)
- **Type hints**: Required on all public function signatures
- **Errors**: `ValueError` with descriptive messages; validate at boundaries
- **Lazy imports**: `# pylint: disable=C0415` inside methods

---

## Testing

> Full details in `.claude/rules/testing.md` (auto-applied for `tests/**`).

- **Framework**: pytest with custom markers (`@arg_mark` in `tests/common/mark_utils.py`)
- **Unit tests**: `tests/torch/ut/`, `tests/mindspore/ut/` — no distributed setup needed
- **Distributed tests**: `torchrun_case()` for PyTorch, `msrun_case()` for MindSpore (8-card)

---

## Key Implementation Notes

> Full rules for DTensor, memory lifecycle, and stream synchronization are in `.claude/rules/distributed.md` (auto-applied when editing `core/`, `collectives/`, `fully_shard/`, `hsdp/`). The code-review skill's `distributed-guidelines.md` and `review-checklist.md` contain complete patterns with code examples.

**Quick reference — three highest-risk areas:**

1. **DTensor** — `is_partial()` is a **method** not property (must call with parentheses); call `reduce_partial` before `redistribute()`; ops registered via YAML in `core/shard/ops/yaml/`; use `SkipDTensorDispatch` in grad hooks for raw local tensor ops
2. **Memory lifecycle** — `resize_(0)` to free device memory after use; null `param.grad` after consumption; `_clear_recv_buffer()` + `clear_cache()` per micro-batch; `wait_offload/wait_load` for activation swap; prefer buffer reuse (`resize_`) over reallocation
3. **Stream synchronization** — `handle.wait()` before reading async collective output; events for cross-stream deps (`event.record(src)` → `event.wait(dst)`); `non_blocking=True` needs stream sync before read; `grad_sync_stream` only in legacy HSDP path

---

## Git Workflow

- **Commits**: Conventional Commits (`feat:`, `fix:`, `docs:`), ~80 chars subject,
  imperative voice, reasoning in body
- **Squash**: Squash WIP commits before opening PR

---

## Claude Code Configuration (`.claude/`)

### Skills

| Skill | Description | Usage |
|-------|-------------|-------|
| **autogit** | GitCode fork workflow automation (commit, PR, status, squash) | `/skill autogit` |
| **code-review** | Code review for distributed correctness, stream sync, memory safety, cross-platform consistency | `/skill code-review` |
| **dist-op-dev** | Distributed operator development workflow (analysis → impl → test → commit) | `/skill dist-op-dev` |
| **platform-dev** | Platform abstraction layer development (new APIs, FSDP/HSDP/Pipeline, DTensorBase, collectives) | `/skill platform-dev` |
| **ms-op-analysis** | [Internal] MindSpore operator analysis, called by `dist-op-dev` | — |
| **pt-op-analysis** | [Internal] PyTorch operator analysis, called by `dist-op-dev` | — |

### Commands

| Command | Description |
| ------- | ----------- |
| `/commit` | Delegates to `autogit commit` — stage, lint-check (no pylint), commit, push |
| `/test` | Delegates to `autogit test` — test stage: pylint + lints + pytest |
| `/create-pr` | Delegates to `autogit pr` — create PR to upstream/master |
| `/code-review` | Delegates to `code-review` skill |

### Agents

| Agent | Model | Tools | Role |
| ----- | ----- | ----- | ---- |
| **planner** | default | Read, Grep, Glob, Bash | Read-only implementation planning before multi-file changes |
| **code-verifier** | haiku | Read, Grep, Glob, Bash | Automated lint + test verification (delegates to `autogit test`) |
| **code-reviewer** | sonnet | Read, Grep, Glob, Bash | Post-change code review (distributed-first) |
| **dtensor-dev-expert** | opus | Read, Grep, Glob, Bash | DTensor, Layout, redistribution, op dispatch |
| **fsdp-dev-expert** | opus | Read, Grep, Glob, Bash | FSDP/HSDP, parameter sharding, gradient reduction |
| **pipeline-dev-expert** | opus | Read, Grep, Glob, Bash | Pipeline parallelism, micro-batch, activation swap |

### Rules (auto-applied by path)

| Rule | Scope |
| ---- | ----- |
| **project-overview** | Global — project identity, architecture invariants, top safety concerns |
| **code-style** | Global — conventions, naming, patterns |
| **distributed** | `core/**`, `collectives/**`, `**/fully_shard/**`, `**/hsdp/**` — stream sync, memory, DTensor |
| **platform** | `platform/**` — cross-platform abstraction |
| **multi-platform-features** | `core/**`, `platform/**` — multi-backend and list/collection API consistency |
| **testing** | `tests/**` — test patterns, markers, distributed test helpers |
