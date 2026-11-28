# Hierarchical performance diagnosis

**Evidence provenance:** Engineering procedure informed by training optimization
and published Roofline and profiler definitions. Hardware-counter support and
achievable ceilings require validation on the actual device and runtime.

## Start with end-to-end execution

Use complete training steps to locate exposed costs before tuning individual
hardware pipelines. Re-profile the current candidate after a material improvement;
an earlier trace may no longer identify its bottleneck.

| Scope | Evidence to obtain | Candidate interventions |
|---|---|---|
| Framework execution | Host launch gaps, synchronization, materialization, optimizer phases, recomputation, allocation retries | Remove redundant execution, change activation/parameter lifetime, batch updates, replace a slow module with an equivalent implementation |
| Pipeline parallelism | Same-microbatch stage boundaries, stage imbalance, P2P dependencies, fill/drain bubbles | Repartition stages, tune microbatches, change scheduling, overlap activation transfers |
| Parallel communication | Process-group topology, payload sizes, slow ranks, collective dependencies and exposed intervals | Relayout groups, aggregate messages, prefetch, overlap communication with independent computation |
| Module and existing kernel reuse | Hot call sites, actual shapes/layouts, forward/backward/recompute paths, supported native API variants | Reuse compatible HyperParallel, CANN/framework, or other published high-performance implementations; fuse an equivalent module |

Using an existing fast kernel is part of this first optimization level. It does
not require custom kernel development or an instruction-level investigation first.
Search the complete API family, including paged and backward variants. Preserve
model semantics and verify that the selected path actually executes.

For overlapped execution, rank candidates by removable **critical-path time**, not
the sum of kernel or collective durations. A locally faster collective may produce
little end-to-end benefit when prefetch already hides it. Conversely, a high-bandwidth
collective may still require scheduling changes when it blocks computation.

## Escalate when this workload reaches a plateau

Enter hardware-efficiency analysis when a fresh trace and recent valid experiments
show that the major framework, pipeline, module-reuse, and overlap opportunities
are addressed or constrained, while significant time remains in device execution
or data transfer. Record:

- the current workload and best valid MFU, target gap, and measured noise;
- the remaining exposed regions and their maximum plausible end-to-end benefit;
- recently tested interventions, their results, and unresolved coarse-level issues;
- the specific uncertainty that counters or a targeted microbenchmark can resolve.

The transition is **model- and workload-dependent**, not a universal MFU threshold.
One unsuccessful optimization or reaching 10%, 20%, or 25% MFU does not establish
a plateau. An isolated custom-kernel experiment remains reasonable earlier when
it is the shortest evidenced route to remove a dominant bottleneck; it should not
displace larger, readily available module or scheduling improvements.

## Measure hardware efficiency with matched denominators

Use the installed profiler's supported counters and versioned metric definitions.
Capture only representative ranks, operators, and steady-state steps initially.
If event groups cannot be collected together, use matched separate captures and
record replay variability. Instrumented runs are diagnostics; formal MFU comes
from a separate unprofiled run.

| Metric | Definition | Required scope and interpretation |
|---|---|---|
| Achieved CUBE throughput | Executed CUBE floating-point operations / elapsed seconds / 10^12 | Declare kernel, region, or step interval; counter or shape-derived estimate; dtype, dense/sparse mode, device/core coverage, and FMA counting |
| CUBE compute efficiency | Achieved CUBE TFLOPS / matching theoretical CUBE TFLOPS | Use the same precision, hardware scope, and clock/peak convention. Report an empirical shape-matched ceiling separately |
| HBM bandwidth efficiency | Measured HBM bytes / elapsed seconds / matching peak HBM bytes per second | Define read/write accounting and direction. Cache-resident tensor sizes are not measured HBM traffic |
| Interconnect bandwidth efficiency | Measured physical-link bytes / transfer interval / matching link capacity | Distinguish intra-node, intra-supernode, and inter-supernode links, direction, active links, and aggregation scope |
| Collective effective bandwidth | Declared collective data volume / full collective elapsed time | Includes algorithm and rank-wait effects; it is not automatically physical-link bandwidth |

A pipeline busy-cycle ratio, operator-category time share, or an activity label
such as `ArithmeticUtilization` is not automatically FLOPs divided by peak FLOPs.
Check the field's definition before converting it to TFLOPS. Do not substitute a
busy ratio when arithmetic counts are unavailable. Label unsupported metrics as
**not measured** and analytical counts as **estimated**.

Calibrate a new arithmetic counter path using fixed-shape operations with known
FLOPs. Check distinct matrix and vector workloads when both engines contribute.
Count each physical core once and verify coverage of the declared devices. Keep
profiler aggregate rows out of any sum over physical cores. For a step-scoped
capture, drain device work at both ends and verify counter inclusion with known
work before, inside, and after the collection window. Preserve the raw counts,
the calibration error, and any unobserved engine or operation class.

For complete-update HFU, retain receipts that show each rank has finished
device work at both capture boundaries. For multi-node runs, bind each global rank
to its local physical device before aggregating the counts. Normalize against a
separate unprofiled run with the same recipe, update window, and workload
geometry. If a full timeline export exceeds the analysis budget, use the
profiler's native arithmetic records once the event definitions are calibrated
for that runtime. Retain the raw counters, core coverage, and timing guards.
If the exporter times out after capture, [re-parse the native capture
and verify device coverage](../../profiling/SKILL.md#recover-a-partial-cann-arithmetic-export)
before using the recovered registers.

Do not divide CUBE operations by a Vector-unit peak, mix dense and sparse peaks,
or normalize a subset of ranks against the entire cluster. Report kernel-active
efficiency separately from step-wide utilization: a fast GEMM can coexist with
low MFU because of recomputation, optimizer work, or idle intervals. Do not inflate
MFU with those extra operations; they belong in HFU under its coverage contract.

For communication, show transfer time, rank waiting, and exposed time separately.
Compare like-sized messages on the same topology with an isolated communication
benchmark. Algorithm bandwidth and bus bandwidth have different byte conventions;
do not apply a collective correction factor without matching its definition.
In particular, do not compare aggregate cross-supernode traffic with a single
intra-supernode link's peak. Low achieved bandwidth can indicate latency or late
senders rather than an inefficient transport kernel.

## Turn counters into a testable intervention

For device-memory analysis, use arithmetic intensity
`I = executed arithmetic FLOPs / measured HBM bytes`. The Roofline bound is
`P <= min(P_compute_peak, B_HBM_peak * I)`, with consistent FLOPs/s and bytes/s.
Use measured sustainable ceilings alongside theoretical peaks when available.
The bound classifies headroom; it does not prove a root cause or predict an
end-to-end gain by itself. Analyze interconnect communication separately.

| Evidence | Investigate next | Possible intervention |
|---|---|---|
| Low CUBE efficiency, sufficient arithmetic intensity, no dominant transfer stall | Tile coverage, small/irregular GEMMs, tail handling, parallel work distribution, instruction dependencies | Better existing kernel, grouped/batched GEMM, tiling or scheduling changes |
| Low CUBE throughput near the memory Roofline | Repeated HBM traffic, intermediate tensors, cache reuse | Fusion, layout changes, reuse, reduced materialization |
| Both compute and memory throughput low | Host dispatch, synchronization, short kernels, insufficient parallel work | Fix execution granularity or scheduling before assuming bandwidth saturation |
| Low link efficiency on large transfers | Actual route, transfer fragmentation, concurrent groups, protocol and topology limits | Tune collective implementation, message grouping, or group layout |
| High link efficiency but substantial exposed communication | Dependency ordering and insufficient overlap opportunity | Prefetch, asynchronous overlap, topology-aware parallel layout |
| High kernel efficiency but low end-to-end MFU | Recompute, optimizer arithmetic, imbalance, idle or unprofiled phases | Return to framework and training-schedule optimization |

Record the predicted change in counters **and** exposed time, applicable shapes,
correctness checks, and a falsifying observation. Reuse a verified high-performance
implementation first when available. Custom or fused kernel development follows
when evidence supports the expected end-to-end benefit. Check real forward and
backward semantics, engagement, memory, and an independent unprofiled benchmark.

### Calibrate the device and the training shape

Measure an attained reference on the installed accelerator and software stack.
Use synchronized, repeated windows after warmup. Record the device, runtime,
precision, layout, operation shape, elapsed time, and repeated measurements.
Measure a large dense BF16 GEMM for a throughput reference, then benchmark the
actual forward and backward shapes, including transpose and stride. A neighboring
tile-aligned shape can reveal a dispatch or tiling discontinuity. A candidate
must include the cost of padding, copies, slicing, workspace, and any extra
reductions. Keep the model dimensions and trainable parameters fixed.

Use a streaming copy with distinct buffers and a working set beyond device cache
to estimate declared read-plus-write throughput. Use physical HBM counters on
the actual kernel when attributing a measured memory bottleneck. Declare whether
each byte figure comes from a counter, a copy workload, or an analytical bound.
For comparable device intervals, use `max(executed_flops / reference_compute_rate,
physical_hbm_bytes / reference_hbm_rate)` as an optimistic planning estimate.
Measured reference rates describe attained performance under stated conditions.
Keep the nominal MFU denominator and formal acceptance window fixed.

Weight each target shape by its occurrence count and exposed device time. When
considering an unusually slow GEMM, compare the isolated shape with its duration
in the training trace before attributing the gap to kernel efficiency. A faster
standalone shape only advances when its integration has credible step-time gain.

### Separate small-message latency from large-message throughput

Bucket collective enqueues by operation, dtype, declared per-rank payload,
process group and topology. Verify the byte convention and join enqueues to
device intervals before timing each class. Measure matched isolated payloads
when selecting a latency- or throughput-oriented change. Fit
`time = latency + declared_bytes / effective_bandwidth` within a consistent
algorithm regime and report variability and residuals. Split the fit on an
algorithm or route change. Estimated effective bandwidth is scoped to that
collective and its payload definition.

In the training trace, account separately for kernel duration, overlap, and time
exclusive of compute and other communication. A local exclusive interval can
guide coalescing priorities. Verify cross-rank readiness and the resulting
critical path, because waiting and later buffer releases can offset the benefit.
Include grouping, metadata, packing and peak-memory costs in the candidate.

## Agent, Skill, and Harness handoff

The profiling Skill supplies evidence and coverage limits. The Agent chooses the
diagnostic level, explains the transition, and maps evidence to a candidate using
this playbook. Record that rationale in the existing proposal `observation`,
`mechanism`, `expected_effect`, and `evidence` fields. The Harness executes the
declared diagnostic, correctness, benchmark, and cleanup commands; it does not
infer a plateau from MFU alone. Current tools do not automatically collect or
normalize every hardware counter described here.

For a fine-grained investigation, attach a table containing the raw metric name,
unit, formula, theoretical peak source, empirical ceiling, time interval, ranks,
shapes, runtime version, and raw artifact path. Preserve failed and inconclusive
experiments. Return to the coarse trace whenever the bottleneck changes.

## Technical references

- [Berkeley Lab Roofline model](https://amcr.lbl.gov/departments/computer-science-department/ppan/roofline-performance-model/): relates arithmetic intensity, compute throughput, and memory bandwidth.
- [Ascend CANN profiling options](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/910/devaids/Profiling/atlasprofiling_16_0011.html): metric names and supported collection modes are version-specific; verify the installed version.
- [NVIDIA collective bandwidth definitions](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md): distinguishes algorithm and bus bandwidth. This is an accounting reference, not evidence of HCCL performance or identical topology behavior.
