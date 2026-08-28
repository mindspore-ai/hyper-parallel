---
name: platform-dev
description: >
  Develop HyperParallel platform abstraction: new Platform APIs, torch/mindspore
  backends, FSDP/HSDP/Pipeline/Activation Checkpoint, DTensorBase, collectives.
  Use when changing platform/ or cross-backend features. Not for shard op
  YAML/impl (use dist-op-dev) or git/CI (autogit / gate-doctor).
---

# Platform Development

Guides cross-platform work under `platform/`. **This file is the index** —
load the matching workflow / reference before that step.

## When to use

- New method on `platform/platform.py`, or impl in `platform/torch|mindspore/`
- FSDP / HSDP / Pipeline / Activation Checkpoint platform code
- DTensorBase extensions; collectives; stream sync / memory lifecycle patterns

## Execution (autonomous checklist)

1. [workflows/01-scope-analysis.md](workflows/01-scope-analysis.md)
2. [workflows/02-base-class-api.md](workflows/02-base-class-api.md)
3. [workflows/03-backend-implementation.md](workflows/03-backend-implementation.md)
4. [workflows/04-cross-platform-verification.md](workflows/04-cross-platform-verification.md)
5. [workflows/05-testing.md](workflows/05-testing.md)
6. [workflows/06-commit.md](workflows/06-commit.md) → **autogit**

Tree, API map, patterns: [references/architecture.md](references/architecture.md),
[references/quick-reference.md](references/quick-reference.md),
[references/decisions.md](references/decisions.md).

## Hard rules

1. No bare `import torch` / `mindspore` in platform-agnostic code — `get_platform()`
2. New Platform APIs land in base class first (`platform/platform.py`)
3. Consider both backends — implement or `NotImplementedError`
4. Type gaps: torch `device`/`ProcessGroup` vs mindspore `str` groups — check signatures
5. Lazy framework imports inside methods under `platform/torch|mindspore/` + `# pylint: disable=C0415`
6. Stream/memory: `handle.wait()`, `event.record→wait`, `resize_(0)` — see `.agent/rules/distributed.md`

## Out of scope

- Distributed op YAML / `parallel_*.py` → **dist-op-dev**
- Commit / PR / CI → **autogit** / **gate-doctor**
- Full distributed audit after change → **code-review**
