---
name: debug
description: >
  Single-node debugging SOP for HyperParallel: scope of responsibility,
  required inputs, the debug flow, fix constraints, verify constraints, stop
  conditions, and the output format. Use for debug / 单机调试 / bug 定位 /
  报错定位 / 单卡复现, when a failure can be reproduced on one card or one
  node. Cluster-wide / multi-rank failures go to the cluster-debug skill.
---

# Debug

A disciplined single-node debug loop: reproduce small, find the first cause,
fix at the root, verify in the smallest scope, and know when to stop. For
multi-rank / cluster failures (HCCL timeout, partial-rank exit, collective
mismatch) use `cluster-debug` instead.

**This file is the index + the SOP.** Deeper routing loads from references.

## Scope of responsibility

- **In scope**: a failure reproducible on **one card / one node** — crash,
  wrong result, numerical issue, perf regression localizable without
  multi-rank collectives.
- **Out of scope**: cluster-wide / multi-rank failures → `cluster-debug`;
  precision acceptance / dump compare → `precision`; perf breakdown →
  `profiling` / `memory-analysis`. Hand off rather than guess across lines.

## Required inputs (refuse to guess without them)

1. **Reproduction**: the exact command + commit + environment that triggers
   it; if not yet reduced, the first job is to reduce to the smallest repro.
2. **Observed vs expected**: the actual error / wrong value, and what was
   expected (with the baseline if it is a correctness/precision issue).
3. **Scope**: single card? single node? which config axis.

Without a reproduction, the first step is to obtain one, not to speculate.

## Debug flow

1. **Reduce**: shrink to the smallest reproducing case (small shapes, single
   card, few steps). A smaller repro localizes faster and de-risks the fix.
2. **First cause, not last symptom**: find the earliest point where behavior
   diverges from expected (first NaN, first wrong op, first failing
   assertion). Downstream failures are propagation.
3. **Falsifiable hypothesis**: form a hypothesis that an experiment can
   disprove (toggle the suspect feature same-config, add a placebo arm);
   do not read logs and guess. A control only rules out the explanation it
   targets.
4. **Fix at the root**: fix the cause, not the symptom. Do not silence the
   signal (no broad except, no assert-softening, no skip/xfail of the real
   failure) to make it pass.
5. **Verify small, then widen**: confirm the fix on the small repro, then on
   the original case; confirm the first-cause point no longer diverges and
   downstream recovers.

## Fix constraints

- **Root cause only.** No symptom patches, no silencing. If the real fix is
  large or risky, say so and stop for a decision rather than papering over.
- **Positive verification.** "No longer errors" is not enough; show the fixed
  behavior produces the expected value / state. A fallback path and a fixed
  path can both be error-free.
- **Smallest blast radius.** Prefer the change that touches the least; a
  device-only quirk gets a device-guarded fix, not a global behavior change.
- **Reproducible evidence** for the fix: command + commit + environment
  (shared with `autoresearch`, `precision`, `gate-doctor`).

## Verify constraints

- Verify on the **same repro** that showed the bug, then the original case.
- For a flaky bug, re-run enough to show the fix holds, not a single pass.
- Record what was verified and where, so the next reader can re-run it.

## Stop conditions

- **Done**: first-cause no longer diverges, fix verified on repro + original,
  evidence recorded.
- **Hand off**: the failure is out of scope (multi-rank → cluster-debug;
  precision口径 → precision) — stop and route.
- **Escalate**: the root fix is large / architectural, or the repro cannot be
  obtained — stop and surface the finding for a decision, do not force a
  symptom patch.

## Output format

Report: the reproduction, the first cause (file / op / line), the root-cause
explanation, the fix (what changed and why it is root not symptom), the
verification (command + result on repro and original), and any residual /
handoff. See `references/common-bugs.md` for recurring single-node bug
classes and their first-cause signatures.
