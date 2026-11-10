# Operator optimization methodology

> **Evidence provenance:** Mixed. General engineering practice and two measured
> fused-operator optimization runs, with ledgers under `.agent/skills/autoresearch/runs/`.
> See [evidence provenance](../SKILL.md#evidence-provenance).

`fused-ops.md` describes available fused operators and their constraints. This page
explains how to locate a fusion opportunity, measure its benefit, and establish that
it is ready for use. Apply it after localizing the bottleneck to operators or
postprocessing. For parallel-layout, memory, or communication bottlenecks, start with
`feature-scenarios.md`.

## 1. Localize from the training step to the operator

1. **Obtain a step breakdown first** with the `profiling` skill. Operator optimization
   affects end-to-end performance only to the extent that the compute phase contributes
   to the critical path.
2. **Aggregate by operator type.** Inspect cumulative duration, invocation count, and
   mean duration. Many short operators suggest dispatch overhead; a few long operators
   suggest kernel efficiency. These require different interventions.
3. **Check fallback first.** Search runtime logs for `npu_cpu_fallback`. Prioritize
   `AI CPU` entries in the operator summary. Fallback can be tens or hundreds of times
   slower, outweighing gains from fusion or parameter tuning.
4. **Build an isolated probe.** Export the hotspot's actual shapes and dtypes from the
   run instead of inferring them from model configuration. Create a microbenchmark with
   N warmup iterations and M timed iterations. Its lower noise makes it a useful basis
   for subsequent comparisons.
5. **Estimate the performance ceiling before editing.** Establish a lower time bound
   from a GEMM microbenchmark or bandwidth limit. The gap to that bound indicates
   whether further optimization is worthwhile.

## 2. Measure at multiple boundaries

One change can lead to different conclusions under different timing boundaries.
Record at least measurements 1 and 5 for each optimization.

| Measurement | What it measures | Limitation |
|---|---|---|
| 1. Isolated microbenchmark | Steady-state device time for one operator | Real workloads cover more shapes; a single test may miss a performance cliff. |
| 2. Synchronized wall clock | Host elapsed time with explicit device synchronization | Without synchronization, this measures dispatch rather than device execution. |
| 3. Sum of kernel durations | Profiler-reported kernel durations | Excludes inter-kernel gaps. Kernels can improve while step time remains unchanged. |
| 4. Host and dispatch time | Framework and Host launch overhead | A dispatch bottleneck can eliminate the end-to-end benefit of faster kernels. |
| 5. End-to-end step time | Time to complete the training step | Small gains require a noise threshold and repeated measurements. |
| 6. Feature-toggle comparison | Fused execution enabled and disabled in one process | An ineffective switch makes both runs look identical; see section 4. |

Three measurement requirements apply.

- **Sweep shapes.** Fused kernels can have cliffs at length limits, head-count
  divisibility boundaries, or alignment boundaries. Measure at least short, medium,
  and long shapes separately. Do not extrapolate a single-shape speedup.
- **Exclude first-call work.** First-time AOT compilation can take tens of minutes and
  cause a timeout that resembles an operator failure. One observed new-shape run
  required `HCCL_EXEC_TIMEOUT=7200`. Record warmup separately and measure steady state.
- **Use a noise threshold and automatic repeats for small gains.** Re-measure results
  within the threshold and compare the median using `autoresearch`'s
  `noise_confirm_reruns`. An observed single-digit-millisecond improvement initially
  fell inside a 2% noise band and would have been discarded without confirmation runs.

## 3. Acceptance criteria for a fused path

All six criteria must be satisfied before declaring the path ready. If any fails,
qualify the conclusion rather than reporting the optimization as fully enabled.

| Criterion | Evidence | Conclusion when unmet |
|---|---|---|
| Explicit eligibility checks | Observable per-call fallback for unsupported head counts, dtypes, shapes, or lengths | Report conditional support and its coverage. |
| Positive engagement marker | `engaged=1` or equivalent INFO output from the measured process | Invalidate the measurement. |
| Declared numerical equivalence | Elementwise equality or a justified tolerance | Report numerical acceptance as pending and exclude the result from accepted performance claims. |
| Shape coverage or explicit limits | Measurements for at least three shape ranges | Restrict the conclusion to measured shapes. |
| End-to-end improvement above noise | Change in measurement 5 exceeds the noise threshold, using repeat medians if needed | Report a local-only improvement. |
| Working, tested fallback | A disabled-feature run recorded in the ledger | Do not merge until the fallback is usable. |

A local-only improvement is a valid result. A tenfold microbenchmark speedup with no
end-to-end change often indicates Host dispatch overhead or a small original share of
step time. Return to localization in section 1 instead of continuing to optimize the
same kernel.

## 4. Observed failure modes

- **Silent fallback.** Both fused execution and fallback can produce no warnings.
  An engagement marker is therefore an acceptance requirement.
- **Reporting dispatch time as device time.** Unsynchronized wall-clock timing does
  not establish device-execution performance.
- **Uncaught asynchronous failures.** A non-OOM exception handler covers synchronous
  errors only. `ACL stream synchronize failed` can surface later at synchronization,
  outside the original `try` scope, invalidating the apparent measurement.
- **Single-shape overfitting.** Extrapolating one shape's speedup across sequence lengths
  can cause regressions in deployment.
- **Speculative patches.** Two recorded kernel changes had no effect when made without
  a falsifiable hypothesis. State the hypothesis first, then design a comparison that
  isolates one explanation.
- **Unverified negative results.** A change that never ran and a change that ran without
  benefit can look identical in a summary. Verify the switch before declaring failure.
- **Misattributed phase time.** Synchronization can charge adjacent work to the current
  phase. This caused a tens-of-milliseconds attribution error in one run. Cross-check
  suspicious phase timings with an isolated comparison.

## 5. Common hotspots and measured alternatives

| Symptom | Mechanism | Alternative |
|---|---|---|
| Integer sorting or `argsort` is unexpectedly slow | int32/int64 ArgSort lacks an AI Core path and falls back to AI CPU | For indices well below 2²⁴, sort exact FP32 keys and convert back. |
| Ascending first-k selection is slower by two orders of magnitude | The backend's `sort` kernel is much slower than `topk` | Use `topk(largest=False, sorted=True)`; when k is below the input width, this also avoids full sorting. |
| Very slow advanced-indexing gather | Fancy indexing runs far below memory bandwidth | Use an equivalent `index_select` operation. |
| `scatter_reduce` falls back to AI CPU | The path has no native implementation | When scatter indices are distinct, use `scatter_` followed by `view(...).amax(-1)`. |
| Slow postprocessing | int64 vector arithmetic and dtype-conversion chains add overhead | Use int32 positions and indices where valid, widening only at Torch indexing arguments. |
| Long stalls at Host readback | Data-dependent checks such as `if bool(x.any())` synchronize the pipeline | Use branch-free `where` expressions and unconditional bypass computation. |

**Do not silently reduce score precision.** In the measured case, adding four mantissa
bits reduced out-of-tolerance selection changes by roughly one order of magnitude and
`worst_gap` by about 1.5 orders. Meeting a 1 BF16-ULP tolerance required FP32's 23-bit
mantissa. Reproduce the reference comparison's arithmetic exactly. A BF16 operator
interface does not establish its internal accumulation precision.

## 6. Related references

- Bottleneck classification and technique selection: [feature-scenarios.md](feature-scenarios.md).
- Fused-operator constraints and registration: [fused-ops.md](fused-ops.md).
- Utilization conventions and acceptance targets: [mfu-targets.md](mfu-targets.md).
- Optimizer executed-work accounting: [optimizer-overhead.md](optimizer-overhead.md).
- Automated experiments with benchmarks, gates, and keep/discard decisions: the
  `autoresearch` skill. Its `runs/` ledgers record the two experiments, including
  each hypothesis and measurement.

**Scope of this reference.** It retains generalizable findings from the `run2` ledger.
It excludes feature-specific details such as block-boundary cleanup and fixed
`sparse_count` invocation policies, findings already covered elsewhere such as
`node_guard` admission checks and shared-storage compilation, and observations without
independent confirmation. Consult the ledger for the complete history.
