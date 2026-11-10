# Breakdown Dimensions and Classification Rules

Rule version **v2** uses fields captured on CANN 9.0.0 + torch_npu 2.9.0.
The `op_summary` file pattern and `Task Type` column assumed by v1 are absent
from that output format. See [Output files](output-files.md) for field
origins and measured examples.

## Step-time accounting

`step_trace_time.csv` supplies the additive decomposition:

```text
Stage = Computing + Communication(Not Overlapped) + Free
```

| Dimension | Source | Interpretation | Additive term |
|---|---|---|---|
| Compute | `step_trace_time.csv:Computing` | Profiler-reported device compute time | Yes |
| Exposed communication | `step_trace_time.csv:Communication(Not Overlapped)` | Communication not hidden by computation | Yes |
| Device idle time | `step_trace_time.csv:Free` | Profiler-reported device idle interval | Yes |
| Optimizer | `operator_details.csv`, `Optimizer.step#*`, `Device Total` | Optimizer intervals aggregated per step | No; a subset of compute |

The optimizer is displayed separately to show its share of training time.
Adding it to the three disjoint terms would double-count work and could
produce a total above 100%.

Each share uses `Stage` as the denominator. Profiling adds overhead; use
the diagnostic fractions to localize work and an independent unprofiled
benchmark for end-to-end performance.

## Compute classification by accelerator core

`op_statistic.csv:Core Type` and `kernel_details.csv:Accelerator Core`
identify the execution resource directly.

| Recorded value | Category |
|---|---|
| `AI_CORE` | Cube |
| `AI_VECTOR_CORE` | Vector |
| `MIX_AIV` | Mixed |
| `AI_CPU` | AI CPU |
| `COMMUNICATION` | Communication kernel |

Preserve unknown values as recorded. Do not merge them into a neighboring
category. FlashAttention kernels can appear as `AI_CORE` or `MIX_AIV`.
Their second-level attribution requires the actual registered `OP Type`
and remains pending validation for each implementation.

### AI CPU communication kernels and operator fallbacks

HCCL collectives include AI CPU launch kernels such as
`allreduceAicpuKernel`. These contribute to the AI CPU category but must be
excluded from the operator-fallback hotspot table. In one two-device
capture they accounted for 2,841 of 16,497 microseconds, approximately 17%
of AI CPU time.

Fallback hotspots include actual operator execution on AI CPU, such as an
integer ArgSort or a `scatter_reduce` fallback.

## Communication dimensions

| Metric | Source | Interpretation |
|---|---|---|
| Per-step communication totals | `communication.json`, `Total Op Info` | Aggregate entry; omit it when enumerating individual operations. |
| Waiting fraction | `Wait Time(ms) / Elapse Time(ms)` | A high value points to synchronization or rank imbalance rather than bandwidth alone. |
| Link bandwidth | Per-link `Communication Bandwidth Info` | `MB/ms` equals `GB/s`; equivalent SDMA/SIO records must not be added together. |
| Overlap fraction | `Overlapped / Communication` in the step trace | A low overlap fraction is consequential when exposed communication also occupies a large step-time share. |

### Communication groups and parallel axes

`profiler_metadata.json:parallel_group_info` records `group_name` and
`global_ranks`.

- With one communication group, attribution is direct.
- With multiple groups, the group name must be joined to the `@<hash>`
  suffix in `communication.json`. The captured metadata does not expose
  that join key. Report the hashes and declared groups and mark automatic
  axis attribution as pending validation.

## Aggregation views

| View | Scope | Source |
|---|---|---|
| Operator type | Top-N cumulative durations by `OP Type` | `op_statistic.csv` |
| Kernel | Individual kernels, `aic_*`/`aiv_*` pipeline ratios, and `cube_utilization(%)` | `kernel_details.csv` |
| Framework interval | `aten::*`, autograd, and `Optimizer.step#*` | `operator_details.csv` |
| Module/layer | Requires capture with `with_modules=True` | Pending validation; the reference capture used `with_modules: false`. |

Framework intervals are nested and must not be summed indiscriminately.
Synchronization events can also distort attributed device durations. In a
reference capture, `wait_event` accumulated 24,882 microseconds of `Device
Self`, and a `c10d::allreduce_` interval inherited 5,370 microseconds even
though per-step communication was only 1,272 microseconds. Use the step
trace or `communication.json` for communication timing.

## Consistency checks

`breakdown.py` reports these checks and exits with code 2 when a required
check fails:

| Check | Basis |
|---|---|
| `Stage = Computing + Communication(Not Overlapped) + Free` | Profiler time-accounting identity, with an absolute floor and stage-scaled tolerance. |
| Total communication is at least exposed communication | Internal step-trace bound. |
| Startup was excluded | `profiler_info` reports a positive `schedule.warmup`, `skip_first`, or `wait`; the Trainer can delay capture without a separate profiler warmup phase. |
| At least two steady-state steps | `schedule.active >= 2` and at least two actual rows in `step_trace_time.csv`; a configured window alone does not establish capture coverage. |
| Optimizer intervals are an integer multiple of captured steps | Composite optimizers can emit several intervals per step. |

The cross-file ratio of `communication.json` elapsed time to step-trace
communication is reported separately. Exact equality held in the original
two-device capture, but is not a general invariant because overlap and
aggregation differ across workloads.

## Reproducibility

- Record the rule version. Recompute reports when accounting rules change.
- The same trace must produce the same breakdown.
- Highlight true AI CPU fallbacks, exposed communication, and device idle
  intervals, then pass the evidence to `perf-playbook`.
- Preserve capture provenance, including CANN/torch_npu versions, schedule,
  and parallel groups.
