---
name: perf-playbook
description: >
  Performance-tuning playbook for HyperParallel training: which optimization
  手段 fits which bottleneck, the fused-operator and CANN/PTA environment
  references, the feature-to-scenario map, and the tuning-summary report
  template. Use for 性能调优 / 性能爬坡 / 调优手段选择 / 调优总结 / perf
  tuning playbook, after a profiling breakdown has named the bottleneck.
  Reference-and-template skill: it chooses and records the手段, the
  `profiling` and `memory-analysis` skills produce the breakdown it reads.
---

# Perf Playbook

Given a named bottleneck, pick the tuning手段 that addresses it, apply it,
and record the climb. This skill is **references + a report template**, not a
collector: the `profiling` skill breaks performance down into compute /
communication / free / optimizer, the `memory-analysis` skill breaks memory
down, and this playbook maps those findings to手段 and keeps the summary.

**This file is the index.** The per-topic references load on demand.

## When to use

1. A profiling or memory breakdown has named the bottleneck (not before —
   tuning without a breakdown is guessing).
2. Look up the手段 that addresses that bottleneck class in
   `references/feature-scenarios.md`.
3. Apply one手段 at a time; re-measure same-node back-to-back; keep what wins.
4. Record each step in the tuning summary (`templates/tuning-summary.md`).

## References (load the one that matches the bottleneck)

| Bottleneck / need | Read |
|---|---|
| Which手段 for which scenario (fix-router, fusion, recompute, swap, overlap, parallel re-layout) | [references/feature-scenarios.md](references/feature-scenarios.md) |
| Fused-operator catalog (what exists, constraints, where it helps) | [references/fused-ops.md](references/fused-ops.md) |
| CANN / PTA environment variables that affect performance | [references/cann-pta-envs.md](references/cann-pta-envs.md) |

## Report template

`templates/tuning-summary.md` — the climb record: each手段 tried, the
bottleneck it targeted, the before/after breakdown, kept or reverted, and the
final best-config breakdown with a perf-climb chart. One row per experiment,
newest last.

## Hard rules

- **Breakdown first.** No手段 is applied before a profiling / memory
  breakdown names the bottleneck it targets; "it felt slow" is not a target.
- **One手段 at a time, re-measured same-node back-to-back.** Cross-time or
  cross-pool A/B is invalid (shared discipline with `autoresearch`).
- **Reproducible evidence.** Every before/after number carries its command,
  commit and environment; a手段 with no measured delta is not "kept".
- **Mechanism sanity-check.** Before claiming a speedup, confirm its
  magnitude is physically possible from the targeted bottleneck's share of
  step time; a number that cannot come from the mechanism is a measurement
  error, not a win.
- This skill records and chooses手段; it does not itself collect profiling
  or snapshot data — call `profiling` / `memory-analysis` for that.
