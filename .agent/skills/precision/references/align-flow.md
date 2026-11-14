# Numerical Comparison Workflow

Use public MindStudio **msprobe** to instrument the training process,
capture candidate and baseline in separate launches, and locate the first
API or module exceeding the declared comparison criterion.

The interfaces, fields, and column names below were checked with msprobe
installed alongside CANN 9.0.0 / torch_npu 2.9.0.post2 on Ascend 910.

## 0. Instrument the training process

The validated msprobe version has no environment-variable entry for a
configuration path. It reads `MSPROBE_LOG_LEVEL`, but instrumentation starts
only when the training process constructs
`PrecisionDebugger(config_path=...)`. Exporting a path and launching an
uninstrumented workload creates no dump. An empty comparison can then
misreport no divergence.

The capture has two components:

| Component | File | Responsibility |
|---|---|---|
| Launcher | `scripts/collect_dump.py` | Write configuration, export `PRECISION_DUMP_*`, launch the workload, and check every requested step. |
| Training hook | `scripts/precision_hook.py` | Read the exported variables, construct the debugger, and wrap a step with `capture()`. |

```python
import sys
sys.path.insert(0, ".agent/skills/precision/scripts")
import precision_hook

precision_hook.start()
for batch in loader:
    with precision_hook.capture():
        train_step(batch)
    precision_hook.step()
```

**Use one capture per process.** `PrecisionDebugger` is a process-level
singleton. Constructing a second instance silently retains the first
configuration, leaving the second destination empty.
`precision_hook.start()` rejects repeated initialization. Launch candidate
and baseline independently.

Declare a baseline with the same model, configuration, parallel layout,
and data order, differing only in the tested axis. See
[Precision acceptance](../../../rules/precision-acceptance.md).

## 1. Configure the dump

The installed `config.json` schema has top-level `task`, `dump_path`,
`rank`, `step`, and `level`. Place `scope`, `list`, `data_mode`, and
`summary_mode` inside the block named after the task. A top-level `scope`
is ignored.

```json
{
  "task": "statistics",
  "dump_path": "<dir>", "rank": [], "step": [0, 1], "level": "L1",
  "statistics": {"scope": [], "list": [], "data_mode": ["all"],
                 "summary_mode": "statistics"}
}
```

- **`step` is a list of step IDs.** `[0, 2]` captures steps 0 and 2 and
  skips step 1. msprobe expands a string such as `"0-2"` to 0, 1, and 2.
  This skill writes an explicit integer list for per-step validation.
  An empty list means all steps.
- Start with L1 API-level capture. Use L0 with a selected scope for module
  or source-level localization.
- `statistics` records per-tensor summary statistics. `tensor` records
  complete tensors and is the task whose comparison includes `Cosine`.
- Brackets, commas, and other special characters in dump paths can trigger
  `FileCheckException`. The launcher rejects these paths early.

```bash
python .agent/skills/precision/scripts/collect_dump.py \
    --run-dir <d> --launch "<training command>" --task statistics --level L1 --step 0-1
```

The launcher verifies a step directory containing `dump.json` for every
configured step. A zero workload exit code with missing captures is a
failure, with a diagnostic about absent in-process instrumentation.

## 2. Compare captures

```bash
python .agent/skills/precision/scripts/diff_dump.py \
    --candidate <cand_dump> --baseline <base_dump> [--step N]
```

The underlying command is
`msprobe compare -tp <target> -gp <golden> -o <out>`. It does not accept
`-f pytorch`, `-np`, or `-bp`. The default output is CSV.
Both input paths must identify **step directories containing `dump.json`**,
not dump roots. Select the newest
`compare_result_proc<pid>_<timestamp>.csv` by modification time rather than
lexical file-name order.

### Comparison-table pitfalls

1. **Columns depend on the task.** Statistics tables contain `Max diff`,
   `Min diff`, `Mean diff`, `L2norm diff`, and `*RelativeErr`. Tensor
   tables contain `Cosine`, `EucDist`, `MaxAbsErr`, `MaxRelativeErr`,
   and `One Thousandth Err Ratio`. Statistics tables have no `Cosine`.
2. **Relative-error units differ.** A statistics cell can be a percentage
   such as `1.1049723756906076%`; a tensor cell can be the raw ratio
   `0.013333333656`. Passing the former directly to `float()` fails.
   Silently skipping those rows can incorrectly report no divergence.
3. **`Result` is not an acceptance criterion.** The reference capture
   included `pass` on a row with `MaxRelativeErr=9368`. Preserve it for
   context but judge the recorded numerical metrics.

Skip `unsupported` entries for non-tensor arguments such as `bias=None`.
Do not skip NaN: nonfinite metrics are failures under every tolerance.

### Bitwise comparison by default

A nonzero difference is reported by default. Explicit `--max-rel` or
`--cosine-min` options enable a tolerance band. Document its justification,
because a band can conceal the first changed operation.

In a controlled test that multiplied one layer's weights by 1.01, the
changed weights had `MaxRelativeErr=0.0133` and `Cosine=0.999998`.
Both passed a 0.02 / 0.99 band while downstream output failed. That band
identified propagation rather than the initiating difference.

The acceptance rule's 0.02 / 2% limits govern the **aggregate loss
trajectory**, normally over 1,000 steps. They must not be reused as
per-tensor tolerances. For BF16 selection ties, follow the rule's
below-one-ULP boundary and report the tie count and worst score gap.

## 3. Report

Use `templates/precision-report.md` to record the earliest divergence,
measured errors, commands, commits, environments, and the proposed fix.

## Evidence requirements

- Report the earliest anomalous operation; explain downstream propagation
  separately.
- Use reproducible commands and same-node, back-to-back comparisons.
- Before interpreting a negative finding, verify nonempty dumps and complete
  step coverage. `collect_dump.py` checks this automatically.
