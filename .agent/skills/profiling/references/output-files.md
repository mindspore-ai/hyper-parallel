# Profiler Output Files and Column Semantics

This reference covers public MindStudio **msprof** and
`torch_npu.profiler` output. File names and column headers were checked on
**CANN 9.0.0 + torch_npu 2.9.0.post2 / Ascend 910**. Differences in other
versions require validation.

## Directory layout

```text
<out>/<host>_<pid>_<timestamp>_ascend_pt/
|-- ASCEND_PROFILER_OUTPUT/   # Parsed files consumed by breakdown.py
|-- FRAMEWORK/               # Torch op_mark, op_range, and python_tracer data
|-- PROF_<n>_<timestamp>_<id>/ # Raw CANN host/device data
|-- logs/
|-- profiler_info.json       # profiler_info_<rank>.json for distributed runs
|-- profiler_metadata.json
```

`breakdown.py --prof-dir` accepts the parsed output directory, its
`*_ascend_pt` parent, or a parent containing multiple rank outputs. Each
rank is parsed independently.

## Captured files in ASCEND_PROFILER_OUTPUT

| File | Contents | Use |
|---|---|---|
| `step_trace_time.csv` | Per-step time accounting | Additive step-time breakdown |
| `op_statistic.csv` | Device time aggregated by operator type | Compute categories and AI CPU fallback hotspots |
| `kernel_details.csv` | Per-kernel details and `aic_*`/`aiv_*` pipeline metrics | Kernel analysis and `cube_utilization` |
| `operator_details.csv` | Torch intervals, including `aten::*`, autograd, and `Optimizer.step#*` | Optimizer timing and framework-level analysis |
| `communication.json` | Collective and point-to-point records by step | Communication timing and link bandwidth |
| `communication_matrix.json` | Communication-pair matrix | Communication topology |
| `api_statistic.csv` | Host ACL API statistics | Host dispatch bottlenecks |
| `trace_view.json` | Chrome tracing timeline | Visual inspection of overlap and idle intervals |
| `ascend_pytorch_profiler*.db` / `analysis.db` | Database representation of the capture | An alternative to CSV for large traces |
| `analyse.done` | Empty analysis-completion marker | Output-completeness check |

The captured version does not produce `*_op_summary*.csv`. A glob for that
older file pattern will not locate the operator data.

## step_trace_time.csv

```text
Device_id,Step,Computing,Communication(Not Overlapped),Overlapped,Communication,
Free,Stage,Bubble,Communication(Not Overlapped and Exclude Receive),Preparing
```

The parser checks the following time-accounting identity:

```text
Stage = Computing + Communication(Not Overlapped) + Free
```

| Column | Meaning | Additive role |
|---|---|---|
| `Computing` | Device computation, including the optimizer | Component |
| `Communication(Not Overlapped)` | Communication not hidden by computation | Component |
| `Free` | Idle time or dispatch gaps | Component |
| `Stage` | Sum of the three components | Total |
| `Communication` | Total communication time | Diagnostic only |
| `Overlapped` | Communication hidden by computation | Diagnostic only |
| `Bubble` | Pipeline bubble; zero without PP | Diagnostic only |
| `Preparing` | Host preparation before the step, such as fetching data | Diagnostic only |

The communication overlap fraction is `Overlapped / Communication`. Check
it together with the exposed-communication share of `Stage`.

## Optimizer timing

Use `Optimizer.step#<optimizer>.step` intervals in `operator_details.csv`,
such as `Optimizer.step#AdamW.step`. `Device Total Duration(us)` reports
the optimizer's attributed device duration. `Device Self` can be zero
because the optimizer is a parent interval. A simple optimizer emits one
interval per step; a composite optimizer can emit several.

Do not reconstruct optimizer time from operator-name matching. AdamW's
foreach path emits `ForeachMulScalar`, `ForeachLerpScalar`,
`ForeachAddcmulScalar`, `ForeachSqrt`, `ForeachDivScalarList`,
`ForeachAddScalar`, `Addcdiv`, and `Pows`. `Pows` for bias correction differs
from a model's `Pow` by only one character. A reference capture showed
about 2% difference between the two counting approaches, and name-based
accounting is not portable.

`Optimizer.zero_grad#*` has zero device duration when `set_to_none=True`
only changes host-side references. Optimizer time is a subset of
`Computing`, not an additional term in the step-time sum.

## Compute categories

`op_statistic.csv:Core Type` and `kernel_details.csv:Accelerator Core`
share the following recorded values:

| Value | Category |
|---|---|
| `AI_CORE` | Cube |
| `AI_VECTOR_CORE` | Vector |
| `MIX_AIV` | Mixed |
| `AI_CPU` | AI CPU, including HCCL launch kernels |
| `COMMUNICATION` | Communication kernels, such as `hcom_allReduce_` |

The `Task Type` column used by older documentation is absent.

Exclude HCCL `*AicpuKernel` launch kernels from the fallback hotspot table.
They belong to communication even though they execute on AI CPU. In a
two-device capture, these kernels accounted for about 17% of AI CPU time.
The parser retains that time in the AI CPU category while separating it
from true operator fallbacks, such as integer ArgSort.

## communication.json

The structure is `{"step<N>": {"collective": {...}, "p2p": {...}}}`.
Inner keys use `<operation>@<communication-group-hash>`, with an additional
aggregate entry named **`Total Op Info`**.

- Read `Total Op Info` for per-step totals. Exclude it when enumerating
  individual communication operations to avoid double-counting.
- `Communication Time Info` includes `Elapse Time(ms)`, `Transit Time(ms)`,
  `Wait Time(ms)`, `Synchronization Time(ms)`, `Idle Time(ms)`, and ratios.
  A high waiting fraction suggests synchronization or rank imbalance.
- `Communication Bandwidth Info` records `Transit Time(ms)` and
  `Transit Size(MB)` by RDMA, HCCS, PCIE, SDMA, or SIO link. **MB/ms equals
  GB/s**; do not divide by 1,000 again. The two-device reference all-reduce
  measured about 162 GB/s over SIO/SDMA. Those two records described the
  same path and must not be added together.
- The original capture had identical communication totals in this file
  and the step trace. Other captures can differ because of aggregation and
  overlap. The parser reports the cross-file ratio rather than enforcing
  equality for every workload.

## Capture provenance

- `profiler_info<_rank>.json` records `cann_version`, `torch_npu_version`,
  `rank_id`, `config.common_config.schedule` with `wait`, `warmup`,
  `active`, and `repeat`, and `config.experimental_config._profiler_level`.
  A single-device capture uses `profiler_info.json` without `rank_id`;
  a distributed capture uses `profiler_info_<rank>.json` with `rank_id`.
- `profiler_metadata.json` records `ENV_VARIABLES` and
  `parallel_group_info`, including `group_name`, `group_rank`, and
  `global_ranks`. One communication group can be attributed directly.
  With multiple groups, the join between group names and `@hash` requires
  additional evidence; the captured metadata does not expose it.

The schedule supports machine checks that warmup was skipped and at least
two steady-state steps were captured.

## Capture and parsing pitfalls

- Capture multiple steady-state steps after warmup. A single sample is
  insufficient for the configured consistency checks.
- Use profiling for relative attribution and an unprofiled run for
  end-to-end performance.
- `kernel_details.csv:Start Time(us)` can contain a trailing tab.
  `Input Shapes` can contain CSV-escaped quoted strings such as
  `"""1,2048,128"""`. Strip numeric cells before conversion.
