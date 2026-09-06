---
description: Design-first working rule for Hyper-RL — context + constraints for hyper_parallel/rl/**
paths:
  - hyper_parallel/rl/**
---

# Hyper-RL Working Method

## RL Context (loaded on every RL session)

**What it is:** a general-purpose LLM RL framework, composed with HyperParallel.
Trainer uses Transformers + HyperAutoModel — currently with
HyperParallel FSDP/TP, and potentially any HyperAutoModel parallelism (FSDP/HSDP/PP/DP)
as generalization proceeds. Rollout uses one shared **vLLM / vLLM-Ascend** deployment
(Hyper-vLLM or Native-vLLM). The **currently verified** path is **single-node,
synchronous, on Torch/Ascend NPU, with Qwen3 + GRPO** — that is the demo baseline,
not the limit of the framework. **Generalization targets** — aligned with where
the industry and research community are going: **agentic RL**, where environments
are harnesses like [DeepSeek's harness](https://github.com/deepseek-ai/DeepSeek-Harness)
that let the model act, observe and learn from outcomes; large-scale distributed and
continual learning; more general RL paradigms and model/reward families; and more
efficient, higher-throughput paths. Today these are future scope: any work toward
them must not regress the verified path.

### Source Layout

> Canonical directory tree + file-level responsibilities are in
> [`module-map.md` § Subsystem → File Map](../skills/hyper-rl-dev/references/module-map.md#subsystem--file-map),
> which is the single source of truth. Do **not** restate the tree here — link
> it. This rule states constraints; the layout lives there. The source-root
> consequence is in § Two Facts below and is not repeated in module-map.

The `Trainer` is the **only** top-level orchestrator. `Algorithm` only declares
math and data needs. The Reference Actor is a separate frozen model, not the
current actor in evaluation mode. File-level responsibilities and interface
contracts: `.agent/skills/hyper-rl-dev/references/module-map.md` (loaded with the skill).

### Two Facts You Must Know

- **`rl` is a source root, not a sub-package.** There is deliberately **no**
  `hyper_parallel/rl/__init__.py`. All RL code imports the top-level package as
  `rl.*` (`from rl.config import ...`), resolved because pytest's prepend mode
  inserts `hyper_parallel/rl/` into `sys.path` when that dir has no `__init__.py`.
  **Consequence:** always run pytest from the repo root; never `cd` into `rl/` first.
- **`model_implementation=hyper|native` selects the vLLM worker model, not two
  rollout backends.** Trainer, request handling, ownership, policy lifecycle and
  weight-sync controller are shared regardless of `model_implementation`.

### Docs

[`architecture.md`](../../hyper_parallel/rl/docs/architecture.md) (roles, data contracts, lifecycle, checkpoint) ·
[`vllm_rollout.md`](../../hyper_parallel/rl/docs/vllm_rollout.md) (ownership, admission, weight transaction, failure semantics) ·
[`qwen3_training_inference_consistency.md`](../../hyper_parallel/rl/docs/qwen3_training_inference_consistency.md) (bit-exact definition, config, gate) ·
[`hyper_rl_runtime_image.md`](../../hyper_parallel/rl/docs/hyper_rl_runtime_image.md) (image, checksum, host) ·
[`public_module_changes.md`](../../hyper_parallel/rl/docs/public_module_changes.md) (CODEOWNER-facing changes outside `rl/`).

## Design First (applies to every code change)

Before writing any code — including bug fixes and refactors, not only new features:

1. **Propose a design and a test method, then wait for human approval before writing code.**
   - **Design**: approach, interfaces, edge cases, and the files that will be touched.
   - **Test method**: how the change will be verified — which CPU tests, which NPU smoke run, and the concrete pass criteria.
2. Do not implement until the user confirms. After approval, implement autonomously via the `hyper-rl-dev` skill — no per-line re-confirmation.

## Scope Boundaries

- Applies to `hyper_parallel/rl/` (relative to the repo root), including its tests, examples, and docs.
- **HyperParallel source may be modified for an RL feature.** Each such change must state clearly (i) what
  is modified, (ii) its impact on surroundings — the boundary with the verified Qwen3 + GRPO path, other
  consumers, and whether the touched API is **public** (listed in `hyper_parallel/__init__.py` `__all__`) or
  **internal** (`_`-prefixed module). Prefer a public API where one exists; upstream-style changes are fine
  when clearly scoped.
- **Known internal coupling**: `rl/` already imports internal HyperParallel modules —
  [`config.py`](../../hyper_parallel/rl/rl/config.py) pulls `HyperAutoModelForCausalLM` from `auto_models._transformers`,
  and [`rl_tests/test_master_qwen3_contracts.py`](../../hyper_parallel/rl/rl_tests/test_master_qwen3_contracts.py) imports
  `auto_models._transformers.{checkpoint_loader,infrastructure}` to lock upstream contracts. Changes to those internal
  modules must be assessed against HyperParallel itself and its own tests, not only `rl/`.
- Keep the verified Qwen3 + GRPO path intact; changes must not regress it.
- **Removed / rejected by design**: a second Router, rank-local server, per-rank port,
  `rollout.vllm.topology`, `request_concurrency`, `api_server_count`. These are architectural
  deletions (see `architecture.md`); do not reintroduce them.
- **Not yet in scope (current demo is single-node sync)**: multi-node, async/off-policy
  rollout, dynamic scaling, transparent generation retry. These are future generalization
  targets, not hard rejections — but any work toward them must not regress the verified
  Qwen3 + GRPO path.

## Test Gates (minimum, before commit)

- **CPU gate**: `python -m pytest -q hyper_parallel/rl/rl_tests`, from the repo root (there is **no** `hyper_parallel/rl/tests/` — that was a stale leftover and has been removed). Scoped run to the changed file first, then full suite.
- **NPU smoke**: run the launcher matching the change scope — `run_qwen3_tp_docker.sh {colocated|disjoint}` (ordinary TP, consistency **disabled**) or `run_qwen3_consistency_docker.sh {colocated|disjoint}` (bit-exact) when the change touches consistency/weight-sync. Exact commands, report fields, and the no-free-NPU skip rule: `.agent/skills/hyper-rl-dev/references/npu-smoke.md`.

## One PR, One Commit

- **One PR = one commit on the source branch.** After a PR (merge request) is opened, it should end up as exactly one commit — no interim "fixup" / "WIP" commits. Squash them before the PR is ready for review (see `autogit squash #N`).
- **Do not mix unrelated work into an open PR.** If the working tree mixes your changes with someone else's in-progress work, commit only your files and leave the rest untouched instead of sweeping everything into one commit.
- This is the canonical rule for Hyper-RL work — state it in context; do not restate the wording elsewhere.

## Bit-Exact Gate (consistency.enabled=true only)

Per comparison step: **valid token count > 0** and `mismatch_count == max_abs_diff == mean_abs_diff == 0` (rendered `0/0/0`). Non-zero exit → no version publish. The ordinary-TP launcher explicitly disables consistency and cannot be used to claim bit-exact.

> Full commands, strategy matrix, contracts, and failure semantics live in the `hyper-rl-dev` skill references (`npu-smoke.md`, `module-map.md`, `weight-sync.md`, `design-notes.md`). This rule states the constraints; it does not restate the procedures.
