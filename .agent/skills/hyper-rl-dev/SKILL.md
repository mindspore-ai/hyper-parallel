---
name: hyper-rl-dev
description: >
  Implement Hyper-RL changes from a confirmed design; run the RL CPU gate and
  NPU smoke, then commit. Goal mode — no per-step confirmation after approval.
  Triggers: hyper-rl, rl 改动, /hyper-rl-dev.
---

# Hyper-RL Development

Implements a Hyper-RL change after the human has approved the design + test
method (see `.agent/rules/hyper-rl-workflow.md` — **Design First**). This skill
turns an approved design into code and gates. It is **not** the place to decide
the design; if no approved design exists, produce one and wait for confirmation.

Reports (Chinese): `.agent/skills/hyper-rl-dev/reports/{Change}_report.md`
(gitignored if `.agent/skills/hyper-rl-dev/reports/.gitignore` has `*`).

## Trigger

An approved design, e.g.:

```text
/hyper-rl-dev implement per .agent/rules/hyper-rl-workflow.md (design approved 2026-08-29)
```

## Design input

Before implementation, describe the approach, affected files and interfaces,
edge cases, and exact verification criteria. Follow the approval rule above;
once approved, use the execution flow below without per-step confirmation.
This skill is the only procedural checklist; references contain detailed
contracts and commands, not another workflow.

## Preconditions (fail fast, do not silently proceed)

- Read `.agent/rules/hyper-rl-workflow.md` + `.agent/rules/code-style.md` before touching any file.
- Confirm the design cites the **exact files** and the **concrete pass
  criteria** for both the CPU gate and the NPU smoke. If the design says only
  "run tests" without a pass/fail threshold, go back one step — do not guess.

## Execution flow (autonomous, in this order)

1. **Scope** — diff the design against
   [references/module-map.md](references/module-map.md) to confirm which RL
   subsystems (config / trainer / dataset+agentic / algorithm+policy / rollout /
   weight_sync / consistency) the change touches. If it also needs a
   HyperParallel source change, keep it but record it per
   `.agent/rules/hyper-rl-workflow.md` § Scope Boundaries (public/internal
   boundary, impact on surrounding consumers) — do not silently frame it as
   out of scope.
2. **Implement** — respect the layouts and contracts in
   [references/module-map.md](references/module-map.md) and the design/interface
   notes in [references/design-notes.md](references/design-notes.md). No new
   second Router / rank-local server / `rollout.vllm.topology` — those are
   rejected by design.
3. **CPU gate** — run the affected `rl_tests`, then the full suite:
   `python -m pytest -q hyper_parallel/rl/rl_tests`. All must pass from the
   repo root. Report per-suite status in the report file.
4. **NPU smoke** — run the launcher matching the change (see
   [references/npu-smoke.md](references/npu-smoke.md)). Record hardware, world
   size, dtype, parallel dimensions, model/checkpoint, exact command and the
   pass/fail outcome. If the environment has no free/healthy NPU, **do not
   claim the smoke passed** — mark it skipped in the report and remind the user.
5. **Complete** — write `.agent/skills/hyper-rl-dev/reports/{Change}_report.md`
   (Chinese): files touched, per-suite CPU status, NPU smoke status/verdict,
   bit-exact `mismatch/max-abs/mean-abs` values if relevant, and a checklist
   against the design. Then prompt `/commit`; `/gate-doctor` only if CI is
   involved.

## Out of scope

- git commit / push / PR → **autogit**
- GitCode gate diagnosis → **gate-doctor**
- Design decision / approval → the human, per `hyper-rl-workflow.md`

## Decision notes

- `consistency.enabled` controls the bit-exact recipe — default off, matched
  TP only when on. Ordinary TP and consistency use **different** launchers;
  never run consistency smoke to prove gradient/convergence.
- Weight-sync strategy matrix in
  [references/weight-sync.md](references/weight-sync.md) — pick the smallest
  strategy that satisfies the change; default to
  `full_gather` unless the design says otherwise.
