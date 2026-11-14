---
name: perf-playbook
description: >
  Performance-tuning playbook for HyperParallel training: which optimization
  technique fits which bottleneck, the fused-operator and CANN/PTA environment
  references, operator-level climb methodology, utilization conventions and
  acceptance targets (MFU/HFU/throughput), optimizer-phase overhead
  accounting, the feature-to-scenario map, and the tuning-summary report
  template. Use to select tuning techniques, set MFU targets, or summarize
  performance optimization after profiling has identified the bottleneck.
  This reference-and-template skill selects and records techniques; the
  `profiling` and `memory-analysis` skills produce the breakdown it reads.
---

# Perf Playbook

Given an identified bottleneck, select a tuning technique that addresses it, apply it,
and record the climb. This skill is **references + a report template**, not a
collector: the `profiling` skill breaks performance down into compute /
communication / free / optimizer, the `memory-analysis` skill breaks memory
down, and this playbook maps those findings to techniques and records the summary.

**This file is the index.** The per-topic references load on demand.

## When to use

1. A profiling or memory breakdown has named the bottleneck (not before —
   tuning without a breakdown is guessing).
2. Look up the technique that addresses that bottleneck class in
   `references/feature-scenarios.md`.
3. Apply one technique at a time; re-measure on the same node in consecutive runs; retain verified improvements.
4. Record each step in the tuning summary (`templates/tuning-summary.md`).

## References (load the one that matches the bottleneck)

| Bottleneck / need | Read |
|---|---|
| Techniques by scenario (fix-router, fusion, recomputation, offload, overlap, parallel layout) | [references/feature-scenarios.md](references/feature-scenarios.md) |
| Fused-operator catalog (what exists, constraints, where it helps) | [references/fused-ops.md](references/fused-ops.md) |
| Official Ascend fusion API snapshots, version limits and training applicability | [references/ascend-fusion/index.md](references/ascend-fusion/index.md) |
| CANN / PTA environment variables that affect performance | [references/cann-pta-envs.md](references/cann-pta-envs.md) |
| Utilization conventions, acceptance targets (MFU/HFU/throughput/step time), target-setting and gap decomposition | [references/mfu-targets.md](references/mfu-targets.md) |
| Operator-level climb: localize, time it under several clocks, decide whether a fused path is really unlocked | [references/fusion-methodology.md](references/fusion-methodology.md) |
| Optimizer-phase overhead: Newton–Schulz FLOPs, accounting the executed work that MFU does not count, what to do about it | [references/optimizer-overhead.md](references/optimizer-overhead.md) |
| Performance-driven training RSI, architecture FLOPs audits, scoped workload tuning, and tiered gates | [references/training-rsi.md](references/training-rsi.md) |
| Runtime failures, material performance defects, layout initialization, and original-workload regression | [references/failure-repair.md](references/failure-repair.md) |

## Scripts

`scripts/fetch_ascend_docs.py` — retrieve official API text with URL/version/hash provenance,
or `--query "rmsnorm rsqrt"` to search the local fusion catalog. Rejects challenge/error pages;
an official API entry does not prove current-runtime compatibility or training backward support.
The bounded experiment workflow is [mfu-riser](../mfu-riser/SKILL.md), which calls this knowledge
base after collecting actual operator evidence.

`scripts/extra_flops.py` — aggregate the executed-but-not-model work
(Newton–Schulz, router fix-ups, selective recompute, profiled shares) into the
`hfu_extra_tflops_per_step` value, with a per-term derivation block for the
tuning summary. Every term must declare its source; see
[references/optimizer-overhead.md](references/optimizer-overhead.md) §3.

## Evidence provenance

Each reference identifies its **evidence provenance** so readers can assess its applicability.

| Label | Meaning | How to use it |
|---|---|---|
| Prior knowledge | General engineering practice or a published algorithm, without measurements in this repository | Use as guidance. If measurements contradict it, prioritize the measurements and update the reference. |
| Measured case | Findings from a recorded optimization experiment in this repository, with a ledger location | Cite its accounting convention and measured scale. Revalidate before extrapolating to another configuration. |
| Mixed | Prior knowledge calibrated by measured cases | Apply the distinction marked within the reference. |

**Updating prior knowledge from measurements is part of the workflow.** Record the
discrepancy in the gap list, revise the reference, and relabel it as mixed or a measured
case. An absolute target borrowed from another configuration, such as "MFU should exceed
40%," is not an acceptance criterion. See [references/mfu-targets.md](references/mfu-targets.md), section 3.

## Report template

`templates/tuning-summary.md` — the optimization record: each technique tried, the
bottleneck it targeted, the before/after breakdown, kept or reverted, and the
final best-config breakdown with a perf-climb chart. One row per experiment,
newest last.

## Hard rules

- **Breakdown first.** No technique is applied before a profiling / memory
  breakdown names the bottleneck it targets; "it felt slow" is not a target.
- **One technique at a time, re-measured on the same node in consecutive runs.** Cross-time or
  cross-pool A/B is invalid (shared discipline with `autoresearch`).
- **Reproducible evidence.** Every before/after number carries its command,
  commit and environment; a technique with no measured improvement is not "kept".
- **Mechanism sanity-check.** Before claiming a speedup, confirm its
  magnitude is physically possible from the targeted bottleneck's share of
  step time; a number that cannot come from the mechanism is a measurement
  error, not a win.
- This skill records and selects techniques; it does not itself collect profiling
  or snapshot data — call `profiling` / `memory-analysis` for that.
