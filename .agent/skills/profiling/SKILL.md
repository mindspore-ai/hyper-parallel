---
name: profiling
description: >
  Collect and analyze Ascend training profiles with torch_npu.profiler or
  MindStudio msprof. Produce a reproducible breakdown of compute, exposed
  communication, device idle time, optimizer execution, accelerator core
  categories, and link bandwidth. Use for performance profiling, bottleneck
  localization, or a performance breakdown report. The perf-playbook skill
  uses the resulting evidence to select optimization techniques.
---

# Profiling

Collect a trace and parse its output files to identify where training time is
spent. The breakdown is **rule-based and reproducible**: the same trace must
produce the same result. Use `perf-playbook` to turn the evidence into an
optimization experiment.

**This file is the index.** Load the output-file guide and breakdown rules
when their details are needed.

> **Validation scope.** File names, column headers, and accounting identities
> were checked on **CANN 9.0.0 + torch_npu 2.9.0.post2 / Ascend 910**.
> Classification rules are version **v2**. FlashAttention subcategories,
> multi-group communication-axis attribution, and module/layer aggregation
> remain **pending validation**; see `references/breakdown-dimensions.md`.

## Subcommands

```bash
python .agent/skills/profiling/scripts/<tool>.py <options>
```

| Intent | Tool | Purpose |
|---|---|---|
| Collect a trace | `collect_prof.py --run-dir <d> --launch "<cmd>" [--warmup 3 --active 3]` | Launch the workload and verify complete output through `analyse.done` and `step_trace_time.csv`. |
| Instrument a standalone training script | `prof_hook.profile_steps()` | Create an in-process profiler context using the `PROF_*` variables exported by `collect_prof`. |
| Generate JSON and HTML | `breakdown.py --prof-dir <d> --out report.html [--json rec.json]` | Parse, classify, and validate the trace. Exit codes are 0 for all checks passing, 2 for a failed check, and 1 for missing output. |

`--prof-dir` accepts an `ASCEND_PROFILER_OUTPUT` directory, its `*_ascend_pt`
parent, or a directory containing multiple rank outputs. Multi-rank outputs
are parsed separately for each rank.

For a standalone training script:

```python
import sys
sys.path.insert(0, ".agent/skills/profiling/scripts")
from prof_hook import profile_steps

with profile_steps() as prof:
    for batch in loader:
        train_step(batch)
        prof.step()  # Advance the warmup/active schedule once per step.
```

## Capture inside HyperParallel

Prefer the Trainer's native capture hook in this repository:
`--profiling.enabled=true --profiling.start_step=N --profiling.end_step=M
--profiling.trace_dir=DIR`. `ProfilingCallback` advances the profiler once
per optimizer step on one rank. `collect_prof.py --mode inrun` requires the
training script to call `prof_hook.profile_steps()` itself; use it for
standalone scripts rather than the stock Trainer entry point.

The optimizer-interval count must be an integer multiple of the captured
steps. A composite Muon/AdamW optimizer, for example, produces two intervals
per step. The `Stage` identity uses a per-step absolute tolerance floor and
a stage-scaled tolerance. The ratio between `communication.json` and
`step_trace_time.csv` communication totals is reported rather than gated:
overlap and multi-rank aggregation can make them differ. An 8-rank FSDP run
produced a ratio of 1.77 with both files parsed correctly.

## References

| When | Read |
|---|---|
| Identify profiler files, column meanings, accounting identities, and parsing pitfalls | [Output files](references/output-files.md) |
| Apply time-accounting, accelerator-core, communication, and consistency rules | [Breakdown dimensions](references/breakdown-dimensions.md) |

## Report

`templates/perf-breakdown.html` contains six sections: step-time breakdown,
compute categories, AI CPU fallback hotspots, communication and link
bandwidth, consistency checks, and bottleneck findings. `breakdown.py`
populates them from the trace and records capture provenance, including
CANN/torch_npu versions, profiler schedule, and parallel groups.

## Hard rules

- **Localize before selecting an optimization.** This skill identifies the
  bottleneck; `perf-playbook` selects an intervention from validated evidence.
- **Use versioned, reproducible rules.** State the rule version in the report.
  Identical traces must produce identical breakdowns.
- **Collect valid steady-state evidence.** Skip warmup, capture at least two
  steady-state steps, and use `Level1` or higher so that
  `step_trace_time.csv` is available. `breakdown.py` checks these conditions
  and exits with code 2 on failure.
- **Do not use failed consistency checks as performance evidence.** The
  checks cover the `Stage` identity, total versus exposed communication,
  schedule, and optimizer intervals. A failed accounting check requires
  parser or input investigation before drawing a bottleneck conclusion.
- **Validate tolerance changes at two operating points.** The observed
  `Stage` drift includes a per-step constant of tens of microseconds. A
  relative-only tolerance rejected a roughly 0.51-second step, while an
  overly tight absolute tolerance rejected a roughly 2.6-second step.
  Validate changes across both fast and slow step-time regimes.
- **Record reproducible provenance.** Include the command, source commit,
  and environment of the profiled run.
- **Optimizer time is a subset of `Computing`.** The additive breakdown has
  only three terms: compute, exposed communication, and device idle time.
- **Separate HCCL launch kernels from AI CPU fallbacks.** HCCL
  `*AicpuKernel` entries run on AI CPU but belong to communication.
- Mark unsupported attribution as **pending validation** and missing input
  as **not parsed**. Never substitute an unmeasured value.
