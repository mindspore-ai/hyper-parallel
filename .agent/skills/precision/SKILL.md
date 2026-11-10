---
name: precision
description: >
  Compare numerical results and localize model bring-up failures on Ascend.
  Collect MindStudio msprobe dumps, compare candidate and reference runs,
  and find the first diverging module, operator, or source location. Use for
  numerical parity checks, NaN/Inf diagnosis, loss spikes, non-convergence,
  or dump comparison. Produce an evidence-backed report using the
  precision-acceptance rule.
---

# Precision

Collect numerical evidence, compare it with an explicit baseline, and
localize the first divergence. The acceptance criteria are defined in
[`precision-acceptance`](../../rules/precision-acceptance.md): the loss
trajectory, gradient norm, first step, determinism, and checkpoint resume.
This skill supplies the evidence evaluated by that rule.

**This file is the index.** Load the comparison and triage references when
their details are needed.

> **Validation scope.** The public MindStudio **msprobe** interfaces and
> fields were checked on CANN 9.0.0 / torch_npu 2.9.0.post2 / Ascend 910.
> Criteria and instrumentation specific to an internal `precision_align`
> implementation remain **pending validation** until that implementation is
> available.

## Subcommands

```bash
python .agent/skills/precision/scripts/<tool>.py <options>
```

| Intent | Tool | Purpose |
|---|---|---|
| Launch a capture | `collect_dump.py --run-dir <d> --launch "<cmd>" --task statistics --step 0-1` | Write the msprobe configuration, export `PRECISION_DUMP_*`, launch the workload, and verify a dump for every requested step. |
| Instrument the training process | `precision_hook.py` | Call `start()`, `capture()`, and `step()` inside training; msprobe requires in-process instrumentation. |
| Compare two captures | `diff_dump.py --candidate <d1> --baseline <d2> [--step N]` | Run `msprobe compare` and report the first diverging API or module. |
| Triage numerical failures | `references/triage-nan-spike-diverge.md` | Diagnose NaN/Inf, loss spikes, and non-convergence. |

**Capture requires in-process instrumentation.** In the validated msprobe
version, `MSPROBE_LOG_LEVEL` is the only `MSPROBE_*` environment entry.
There is no configuration-path environment variable. The training process
must construct `PrecisionDebugger`; exporting a path alone produces no
dump, and an empty comparison can misleadingly report no difference.

`PrecisionDebugger` is a process-level singleton. Launch candidate and
baseline separately. See [Comparison workflow](references/align-flow.md).

## References

| When | Read |
|---|---|
| Capture, compare, and find the first divergence | [Comparison workflow](references/align-flow.md) |
| Diagnose NaN/Inf, loss spikes, or non-convergence | [Numerical triage](references/triage-nan-spike-diverge.md) |
| Apply the declared acceptance criteria | [Precision acceptance](../../rules/precision-acceptance.md) |

## Report

Use `templates/precision-report.md` to record the compared runs, earliest
divergence, measured errors, baseline, and proposed fix. Every measurement
must include the command, source commit, and environment.

## Hard rules

- **Declare the baseline.** Compare the same model, configuration, parallel
  layout, and data order, changing only the axis under test. Record both
  runs' commits, configurations, and environments.
- **Make evidence reproducible.** Preserve commands, commits, environments,
  and dump paths.
- **Find the first divergence.** A downstream NaN can be a propagated
  symptom. Default to bitwise comparison when localizing the first changed
  operation; a broad tolerance can hide the initiating difference.
- **Use the acceptance rule.** Its 0.02 absolute / 2% relative limits apply
  to the aggregate loss trajectory over the declared window. They are not
  per-tensor tolerances.
- **Verify capture before comparison.** Missing dumps do not pass.
  `collect_dump.py` checks each requested step.
- **Do not gate on msprobe's `Result` column.** A reference capture reported
  `pass` on a row with a four-digit relative error.
- **Report BF16 ties explicitly.** Differences confined to tied selections
  whose reference-score gap is below one BF16 ULP use the acceptance rule's
  known-noise boundary. Record the tie count and worst gap.
- Mark unmeasured results **not measured** and unavailable internal
  criteria **pending validation**. Do not invent results.
