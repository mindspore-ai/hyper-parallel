# Planning a testable training optimization

Inputs include the current best trial, parallel-strategy analysis (PSA) prior, profiling
state, runtime environment and operator shapes, perf-playbook references, failure history,
and remaining budget. Retain only a few candidates with explicit mechanisms per iteration
and select one for execution.

## Decision process

1. Verify the MFU audit and runtime evidence, then inspect compute, exposed communication,
   and device idle (`Free`) time on the slowest ranks. Use complete steps including the
   optimizer. Keep overlapping intervals distinct when sizing the critical path.
2. Follow compute hotspots to actual operators and source call sites. Distinguish
   inefficient large GEMMs, dense sequences of small operators, memory transfers, Host
   dispatch, and CPU fallback. An AI_CPU HCCL launch kernel is communication, not operator
   fallback.
3. Look for a comparable implementation in HyperParallel, then check the installed
   framework and CANN operator family, including paged and backward entry points.
   Use public NVIDIA implementations of the same model as design references when useful.
   Verify dtype, layout, real shapes, mask/epsilon/scale semantics, device, CANN/PTA
   versions, and training support in the target runtime.
4. Form a falsifiable hypothesis, list alternative explanations, and select an inexpensive
   diagnostic. Capture missing shapes instead of substituting model hidden size. Locate
   a source call site for every proposed fusion chain.
5. Estimate time saved on the critical path, then assess implementation cost and uncertainty.
   Kernel durations summed across streams cannot be converted directly to step-time shares.
   If the critical-path share is unknown, label the expected gain as unknown and schedule
   a probe. For a large MFU gap, consider equivalent module/kernel replacements, fusion,
   collective scheduling, parallel layout, and activation lifetime changes. A proposed
   optimization should remove enough exposed time to justify its cost.
6. Compare numerics and microbenchmarks across actual shapes, layouts, dtypes, tail blocks,
   and empty experts before measuring complete training. Check whether a path is already
   fused to avoid substituting the same underlying kernel.

When evidence contains a runtime failure or material performance defect, include its
original workload and closure criteria in the proposal. Follow the
[repair procedure](../../perf-playbook/references/failure-repair.md) before treating
a fallback or a smaller input as a valid replacement. Keep asynchronous failure
localization and diagnostic synchronization separate from performance measurement.

For Muon workloads, inspect the HyperParallel path already in use before proposing an
optimizer change. Profile Gram and Newton–Schulz GEMMs, batch/group execution, momentum
updates, and state transfer. Equivalent replacements preserve the declared update rule
and are checked with the actual shapes and distributed state layout.

A useful ranking heuristic is expected end-to-end time saving × evidence confidence /
implementation and validation cost. Do not report uncalibrated success probabilities or
use a documentation microbenchmark as the predicted gain for the current workload.

```text
Speedup bound for one region = 1 / (1 - fraction + fraction / local_speedup)
Required throughput multiplier = target_mfu / baseline_mfu
  (useful model work, devices, and accounting convention remain fixed)
```

## Required proposal fields

Provide `observation` with rank, step, operator, and source; `mechanism`;
`expected_effect` with a range and justification, or an explanation of missing evidence;
`falsifier`; `compatibility` with verified and pending constraints; `actions`;
`evidence` file paths; and local `knowledge` file paths.

Choose `operator_scope` from the [candidate contract](contracts.md#planjson).
Use the measured path and expected end-to-end effect to choose an experiment.

`actions` may contain dependent changes. Explain dependencies and conflicts.
Measure independent changes separately, then perform ablations on successful combinations
within the budget. Without ablations, report only the combined benefit and do not assign
individual contributions. The playbook's one-technique default means one testable
hypothesis or necessary combination here, consistent with authorized combination search.

## Engagement and learning

- Check positive engagement markers, invocation counts, shape coverage, and trace kernels
  for each supported shape, rather than checking only the switch value.
- Compare forward, backward, recomputation, and optimizer phases to detect shifted costs.
- Keep repeatable slow steps in the measured window and correlate them with the trace.
- Report local gains when end-to-end performance does not improve. Kernel speedup alone
  does not justify `keep`.
- Classify negative results as non-engagement, incompatibility, numerical failure,
  bottleneck migration, noise, or a genuine lack of benefit.
- Update the current explanation, its evidence, and remaining unknowns each iteration
  instead of accumulating configuration switches.

When adding a perf-playbook case, record the triggering symptom, runtime versions and
shapes, source hashes, intervention, numerical and engagement evidence, complete-step
gain, applicability limits, and the condition for retrying a deferred candidate.
