---
name: precision-acceptance
description: Confirm task-specific precision criteria; otherwise use the default window, loss error bars and secondary checks
paths:
  - "docs/design/**"
  - "hyper_parallel/tools/precision_align/**"
  - ".agent/skills/precision/**"
---

# Precision Acceptance

The acceptance bar a model bring-up must clear before its precision is
declared validated. This file is the **source of truth for numerical acceptance criteria**; the
`precision` skill runs the comparison and the `feature-doc` skill's precision
sub-report records the criteria selected for the task. Do not invent a looser
bar in response to a failed comparison.

## Confirm the task's criteria before acceptance

Before model precision acceptance starts, ask the user for the required window
(number of compared steps), comparison metrics, and allowed absolute/relative
errors, including whether multiple conditions use AND or OR. Reuse explicit
instructions already supplied for this task; do not ask for the same decision
again. If the answer is incomplete, clarify the missing fields before launching.

Use the user's specified criteria. Only when the user has not specified a
custom criterion use the defaults below: normally 1,000 steps and mean absolute
loss error <= 0.02 **OR** mean relative loss error <= 2%, with the existing
model-scale window adjustment. When adopting defaults, record that choice and
its source before execution. A pending request is not permission to change a
criterion. Do not silently shorten the window or relax a threshold after failure.

Record the user's instruction or default-rule reference, planned window, actual
paired steps, metric definitions, aggregation, thresholds, AND/OR logic, and
per-check results in the report. Distinguish a user-defined acceptance result
from compliance with a different built-in tool gate. If a tool cannot express
the selected criteria, report that limitation and use a documented comparison
of the retained evidence; do not relabel the tool's result. Unspecified secondary
checks retain the defaults below. Missing evidence remains unverified.

## Baseline (state it explicitly, always)

A precision result is meaningless without the baseline it was compared to.
The baseline is a reference run of the **same model, same config, same
parallel layout, same data order**, differing only in the one axis under
test (e.g. the reference implementation vs. the migrated one, or device vs.
a trusted reference). Record the baseline's commit, config and environment
alongside the candidate's; a cross-config or cross-run comparison is not a
baseline.

## Default primary criterion (1000 steps)

When no custom criteria were specified, train both candidate and baseline for **1000 steps** (reduced from the
historical 5000-step / 12-hour bar; 1000 steps is the standing acceptance
length) under identical inputs, and compare the per-step training loss:

- **mean absolute error ≤ 0.02**, **OR**
- **mean relative error ≤ 2%**

Either one passing accepts the primary criterion. Report both numbers, the
step count actually run, and the command / commit / environment.

For a model bring-up run under the `hf-model-integration` stage machine,
the window is sized by model scale instead (that skill's S5 table, set by
operator decision): 1000 steps below ~10B parameters, fewer above, down to
10 at the trillion scale. Compute the same two error bars over the window
actually run, state its length next to them, and claim only what that
window supports — a 50-step result is a 50-step result, not a pass of the
1000-step criterion.

> **Scope of these two numbers.** They bound the **aggregate error of the
> loss trajectory over the whole run**. They are not a per-tensor, per-API or
> per-operator tolerance, and must not be reused as one: a 0.02 band applied
> to a single tensor's relative error is wide enough to hide the first
> diverging operator while still flagging its downstream propagation, which
> inverts the "first divergence, not last symptom" discipline. A tool that
> wants a per-tensor band has to justify that band on its own terms; judging
> an individual tensor against a reference is bitwise unless a reason is
> stated. (This happened: a dump-comparison helper took its per-API
> thresholds from this section and consequently reported symptoms instead of
> causes - see the `precision` skill's `references/align-flow.md`.)

## Default secondary checks (all required unless the user specifies otherwise)

1. **grad_norm** — the gradient-norm trajectory tracks the baseline within
   the same error band as the loss; a diverging grad_norm with a matching
   loss is a red flag, not a pass.
2. **First step** — step-0/step-1 loss matches the baseline tightly (the
   first step is before optimizer drift accumulates, so it isolates
   forward/backward correctness); a first-step mismatch means a real
   numerical difference, not noise.
3. **Determinism** — two candidate runs with the same seed and config
   produce the same loss trajectory (within the backend's documented
   bitwise/ulp band); non-determinism invalidates the error-bar comparison.
4. **Resume from checkpoint** — a run resumed from a mid-training checkpoint
   continues the same trajectory as the uninterrupted run; a resume
   discontinuity is an acceptance failure even when the 1000-step error
   passes.

## Evidence discipline

- Every number carries its **command, commit hash and environment**; a bare
  number is not evidence (shared with `autoresearch`, `gate-doctor`,
  `feature-doc`).
- **Positive evidence only.** "Ran without error" is not a precision pass;
  state the measured error against the stated baseline.
- **A/B must be same-node, back-to-back** when the comparison is a
  performance-adjacent precision run; cross-time or cross-pool comparison is
  invalid.
- An unmeasured check stays `[not measured]` with the reason; never fill a plausible
  number.

## Known-noise boundary

bf16 tie differences are physical, not bugs: when the candidate and baseline
diverge only on tie positions whose reference-score gap is below one bf16
ulp, that is within acceptance. State the tie count and worst gap rather
than claiming bitwise identity when ties exist.
