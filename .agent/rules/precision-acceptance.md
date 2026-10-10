---
name: precision-acceptance
description: Precision acceptance spec for model bring-up — 1000-step error bars plus grad_norm, first-step, determinism and resume checks
paths:
  - "docs/design/**"
  - "hyper_parallel/tools/precision_align/**"
  - ".agent/skills/precision/**"
---

# Precision Acceptance

The acceptance bar a model bring-up must clear before its precision is
declared validated. This file is the **source of truth for the 口径**; the
`precision` skill runs the comparison and the `feature-doc` skill's precision
sub-report cites these thresholds. Do not invent a looser bar per feature.

## Baseline (state it explicitly, always)

A precision result is meaningless without the baseline it was compared to.
The baseline is a reference run of the **same model, same config, same
parallel layout, same data order**, differing only in the one axis under
test (e.g. the reference implementation vs. the migrated one, or device vs.
a trusted reference). Record the baseline's commit, config and environment
alongside the candidate's; a cross-config or cross-run comparison is not a
baseline.

## Primary criterion (1000 steps)

Train both candidate and baseline for **1000 steps** (reduced from the
historical 5000-step / 12-hour bar; 1000 steps is the standing acceptance
length) under identical inputs, and compare the per-step training loss:

- **mean absolute error ≤ 0.02**, **OR**
- **mean relative error ≤ 2%**

Either one passing accepts the primary criterion. Report both numbers, the
step count actually run, and the command / commit / environment.

## Secondary checks (all required)

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
- An unmeasured check stays `[待测]` with the reason; never fill a plausible
  number.

## Known-noise boundary

bf16 tie differences are physical, not bugs: when the candidate and baseline
diverge only on tie positions whose reference-score gap is below one bf16
ulp, that is within acceptance. State the tie count and worst gap rather
than claiming bitwise identity when ties exist.
