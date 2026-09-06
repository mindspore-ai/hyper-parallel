# HyperParallel — AGENTS.md

## Project Overview

HyperParallel is an **easy-to-use, high-performance distributed parallel acceleration library** for distributed model training, inference and reinforcement learning. It provides unified abstractions for DP, FSDP/HSDP, TP, EP, CP, PP, activation checkpoint/swap, and parameter/optimizer offload. Hybrid strategies combine freely.

Primary target hardware: **Ascend NPU and Nvidia GPU**. Primary framework: **PyTorch and MindSpore**.

---

## Scoped Instructions

- For anything under `hyper_parallel/rl/**`, read `.agent/rules/hyper-rl-workflow.md` completely (auto-applied via `paths:`) and apply it together with this repository-level file.

---

## Dev Commands

```bash
# Editable install (required so edits to this repo are what Python imports)
cd /path/to/hyper-parallel && pip install -e .

# Verify import path points at this checkout (not a stale worktree / site-packages copy)
python -c "import hyper_parallel, os; print(hyper_parallel.__file__)"

# Unit tests (no multi-card)
pytest -vs tests/ut

# Lint / commit / PR (GitCode fork workflow)
python3 .agent/skills/autogit/scripts/autogit.py check
python3 .agent/skills/autogit/scripts/autogit.py commit -m "feat: ..."
python3 .agent/skills/autogit/scripts/autogit.py pr

# AGENTS.md Skills/Agents table vs disk; confirms docs/index links + no doc drift.
# Non-zero exit is BLOCKING. Run it whenever the diff touches *.md (also in:
# autogit check / autogit commit).
python3 .agent/scripts/check_agents_catalog.py
```

Distributed ST helpers: `torchrun_case()` / `msrun_case()` via `tests.common.distributed_launcher`, or `parallel_case` (see `.agent/rules/testing.md`).

---

## Env Gotchas / Do Not

- **Editable must track this tree.** If `pip show hyper_parallel` shows another path (e.g. a deleted `.worktrees/...`), re-run `pip install -e .` from repo root. Do not rely on `PYTHONPATH` alone.
- **Never** `import torch` / `import mindspore` in platform-agnostic `core/` code — use `get_platform()`.
- **Never** invent Jenkins build numbers or force-push shared branches in agent workflows.
- Hard distributed rules (canonical): `.agent/rules/project-overview.md` + `.agent/rules/distributed.md` — do not restate long-form elsewhere; link instead.

---

## Key Modules

> Canonical architecture: [`docs/rl-architecture.md`](docs/rl-architecture.md) — the
> single source of truth for module map + platform abstraction + RL deployment
> picture. Feature→flag→branch→metric→test traceability:
> [`docs/rl-navigation.md`](docs/rl-navigation.md). This table is a **pointer**, not a
> copy; update the architecture doc, not this table, when modules change.

| Module | Location | Purpose |
|--------|----------|---------|
| **Platform** | `platform/` (`platform.py`, `torch/`, `mindspore/`) | Abstraction — `get_platform()`, never import backends in core |
| **DTensor** | `core/dtensor/` | Local shard + DeviceMesh + Placements; redistribution cache |
| **Shard** | `core/shard/` | `shard_module()` / YAML ops + `parallel_*.py` |
| **Tensor parallel** | `core/tensor_parallel/` | `parallelize_module()`, `ParallelStyle`, mesh context |
| **FSDP / HSDP** | `core/fully_shard/`, `platform/*/fully_shard/` | Param shard/unshard; HSDP under same trees (`hsdp_*.py`) |
| **Pipeline** | `core/pipeline_parallel/`, `platform/*/pipeline_parallel/` | Stage schedule, micro-batch, P2P |
| **Activation** | `core/activation_checkpoint/`, `platform/torch/activation_checkpoint/` | SAC + activation swap |
| **Checkpoint** | `core/distributed_checkpoint/` | Distributed save/load |
| **Collectives** | `collectives/cc.py` | Process groups |
| **RL** | `hyper_parallel/rl/` | Sync LLM RL runtime (Qwen3+GRPO baseline) — see `rl/` note below |
| **Tests** | `tests/ut/`, `tests/torch/`, `tests/mindspore/` | UT + distributed ST |

For anything under `hyper_parallel/rl/`, the source layout and interface
contracts come from `.agent/skills/hyper-rl-dev/references/module-map.md`
(loaded with the `hyper-rl-dev` skill); RL architecture is in
[`hyper_parallel/rl/docs`](hyper_parallel/rl/docs/architecture.md).

---

## Coding Conventions

> Full details: `.agent/rules/code-style.md` (global hard constraint).

- Apache 2.0 header on `.py` (lines 1–16); PEP 8 / ~120 cols; Google-style docstrings; type hints on public APIs
- Imports at module top except platform backends: lazy `torch`/`mindspore` inside methods + `# pylint: disable=C0415`
- Load `code-style.md` before generate / edit / commit / review; auto-fix before proceeding

---

## Testing

> Full details: `.agent/rules/testing.md`. How-to for new UT: skill `add-unit-test`.

- **Runner:** pytest + `@arg_mark` (`tests/common/mark_utils.py`)
- **Authoring:** prefer `unittest.TestCase` where existing UT does (pytest still runs them)
- UT: `tests/ut/` — no distributed setup
- ST: `torchrun_case` / `msrun_case` / `parallel_case` (multi-card) — **launchers must not import** `torch` / `mindspore` / `hyper_parallel` (see testing.md § ST launcher import rule)

---

## Key Implementation Notes

> Canonical detail: `.agent/rules/distributed.md`. Review examples: `.agent/skills/code-review/distributed-guidelines.md` (not a second SoT).

Highest-risk reminders (see rules for full text):

1. **DTensor** — `is_partial()` is a method; `reduce_partial` before `redistribute`; YAML op registry; `SkipDTensorDispatch` in grad hooks
2. **Memory** — `resize_(0)` after free; null consumed `grad`; clear PP recv buffers per micro-batch; swap wait_load/offload
3. **Streams** — `handle.wait()` before reading async output; event sync across streams; sync after `non_blocking=True`

---

## Git Workflow

- Conventional Commits (`feat:` / `fix:` / `docs:`), **~80-char** subject, imperative (see `.agent/rules/code-style.md` § Commit Convention; code line width is ~120)
- Squash WIP before opening PR; use **autogit** for GitCode fork + upstream
- Optional git hook: copy `.agent/hooks/commit-msg` → `.git/hooks/commit-msg` (rejects AI attribution trailers; `autogit` also checks)

---

## Hooks (harness-specific)

Configured in `.agent/settings.json` (Claude Code–style `PostToolUse` matchers). Scripts under `.agent/hooks/`:

| Hook | When | Purpose |
|------|------|---------|
| `enforce-code-style.sh` | After Write/Edit | Style guard on edited files |
| `check-op-yaml.sh` | After Write/Edit | Op YAML ↔ impl pairing checks |
| `commit-msg` | git `commit-msg` (manual install) | Block AI attribution trailers |

**Target harness today:** Claude Code–compatible agent settings consuming `.agent/settings.json`. Other IDEs may not run these hooks — treat **autogit check / pre-commit / CI** as the cross-tool fallback. Do not assume hooks fire outside that harness.

---

## AI Agent Configuration (`.agent/`)

**Layering:** `AGENTS.md` = ambient law · `rules/` = path-scoped constraints · `skills/` = invocable workflows (SoT for procedures) · `agents/` = thin personas / quick advisors (must not duplicate skill formulas).

### Skills

| Skill | Description | Usage |
|-------|-------------|-------|
| **autogit** | GitCode fork: commit, PR, status, squash, lint/test gates | `/commit`, `/create-pr`, `/test`, … |
| **code-review** | Full distributed review (stream/memory/DTensor/cross-platform) | `/code-review` |
| **dist-op-analysis** | Operator analysis → plan (human confirm) | called before dist-op-dev |
| **dist-op-dev** | Implement + test from confirmed plan | `/dist-op-dev` |
| **platform-dev** | Platform APIs, FSDP/HSDP/PP, DTensorBase, collectives | `/skill platform-dev` |
| **hyper-rl-dev** | Implement Hyper-RL from approved design → CPU gate + NPU smoke | `/skill hyper-rl-dev` (design-first: rule `hyper-rl-workflow`) |
| **gate-doctor** | GitCode PR gate diagnose → autofix to green | 门禁 / autofix / `/retest` |
| **parallel-strategy-analyzer** | DP/FSDP/TP/PP/EP/CP strategy + cost estimate | `/parallel-strategy-analyzer` |
| **readability-first** | Readability + agent-traceability gate (simplicity, one-fact-one-place, nav-map sync) | invoke before any change / review |
| **add-unit-test** | How-to for `tests/ut` (procedures) | when adding UT / coverage |

### Commands

| Command | Description |
| ------- | ----------- |
| `/commit` | `autogit commit` |
| `/test` | `autogit test` (pytest `tests/ut`) |
| `/create-pr` | `autogit pr` → upstream |
| `/code-review` | `code-review` skill (+ mandatory `code-style.md`) |
| `/gen-commit-msg` | Message only, no commit |

### Agents

| Agent | Role |
| ----- | ---- |
| **planner** | Read-only multi-file implementation plan |
| **code-verifier** | 5-phase verify: style/lint, tests, cross-platform, report |
| **simple-code-reviewer** | Fast checklist only — not full `/code-review` |
| **code-reviewer** | Thin proxy → `skills/code-review` (full review) |
| **dtensor-dev-expert** | DTensor / layout / redistribute / op dispatch |
| **tensor-dev-expert** | `parallelize_module` / `ParallelStyle` / mesh |
| **fsdp-dev-expert** | FSDP/HSDP shard, grad reduce, memory lifecycle |
| **pipeline-dev-expert** | PP schedule, micro-batch, activation swap hooks |
| **ep-dev-expert** | Expert parallel / MoE routing |
| **activation-dev** | Activation recompute/swap + LlamaFactory ordering (details in `activation-dev-guide.md`) |
| **llamafactory-hp** | LlamaFactory integration surface (activation details → `activation-dev`) |
| **parallel-strategy-analyzer** | Thin proxy → `skills/parallel-strategy-analyzer` |

### Rules (auto-applied by path)

| Rule | Scope |
| ---- | ----- |
| **project-overview** | Global — identity + hard-rule shortlist |
| **code-style** | Global |
| **readability** | Global — human-readable first, agent-traceable minimum gate (rules → skill `readability-first`) |
| **distributed** | `core/**`, `collectives/**`, `**/fully_shard/**` |
| **platform** | `platform/**` |
| **multi-platform-features** | `core/**`, `platform/**` — multi-backend / list APIs |
| **testing** | `tests/**` |
| **unit-test** | `tests/ut/**` — hard constraints; how-to → skill `add-unit-test` |
| **hyper-rl-workflow** | `hyper_parallel/rl/**` — RL context + design-first constraints; process → skill `hyper-rl-dev` |
| **distributed-op-dev** / **distributed-op-testing** / **test-assertion-style** | Op impl & tests (scoped) |
