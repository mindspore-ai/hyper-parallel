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
Target step budget = baseline_step_time * baseline_mfu / target_mfu
Required savings = baseline_step_time - target_step_budget
```

Inventory the removable intervals on the slow ranks from a separate profile.
Use the union of exposed intervals when several changes affect the same path.
If the entire eligible region is smaller than the required savings, broaden the
candidate scope. Align timing boundaries and revisit the trace after a material
gain. Apply this calculation within a fixed workload series.

`status` returns `target_step_budget_seconds` and `required_step_savings_seconds`
from the current best unprofiled window. Both fields are null before a valid trial.
This focuses subsequent planning at any utilization level, with the run's selected
MFU convention and device peak. At fixed work, reaching 35% from 30% requires
reducing the complete step time by about 14.3%.

With hardware counters, `status` also projects the covered HFU required at the
target MFU, assuming executed FLOPs per useful FLOP stay constant. Use HFU and MFU
from the same sampled steps for this ratio. Compare the projection with observed
device efficiency under that counter coverage. A large gap makes it useful to
measure and reduce extra execution, such as activation replay or excess attention
arithmetic, alongside efforts to remove idle time. Interpret this as a planning
estimate under a fixed ratio. Candidate acceptance still uses measured results.

Use the size of the deficit to choose the next experiment.

| Trace and workload evidence | Next experiment |
| --- | --- |
| Exposed communication and idle alone cannot cover the deficit on the critical path | Inspect the compute and memory path at the slow ranks. Check for avoidable mask or intermediate materialization, backend dispatch, and rematerialized work. |
| Recompute is material and the measured peak has headroom | Test selective activation retention. Rank by unprofiled seconds saved per extra GiB and verify the combined peak. |
| Large covered kernels already approach a shape-matched compute ceiling | Measure avoidable work and memory traffic, then bound the gain of a kernel rewrite before committing to it. |
| The task permits workload changes and update-level fixed costs are material | Estimate accumulation scaling in a linked capacity series. Check effective throughput, memory, and numerical equivalence within each selected workload. |

Use diagnostic traces to rank regions. A single-rank or perturbed capture provides
a bound only for its observed scope. After a material gain, capture the new critical
path and recompute the remaining deficit before ranking another expensive candidate.

When global batch is tunable, hold the microbatch shape, topology and model
fixed. Use measured one- and two-microbatch step times to fit an intercept
and a marginal microbatch cost. Use this estimate to rank the next capacity
probe. Verify the prediction with a real run. The intercept groups fixed
costs and does not isolate the optimizer. Check persistent gradient memory,
complete-step MFU, effective throughput and matched token normalization.
Give every global batch its own baseline and acceptance series.

Use the [accumulation planning command](../scripts/plan_accumulation.py) with
completed trial directories from one and two microbatch configurations:

```bash
python .agent/skills/mfu-riser/scripts/plan_accumulation.py \
  /path/to/one/trial /path/to/two/trial /path/to/four/trial \
  --probe 6 --target-mfu-percent 35 --max-relative-time-error 0.01
```

Keep sequence length, per-replica microbatch size, model source, topology, effective
token and FLOP work, and unprofiled measurement window fixed. Pass the accumulation
and global batch settings through the declared workload fields so the model source
manifest can match. Treat a two-point forecast as an experiment hypothesis. Add an
independent count, such as four, to check prediction error before a larger probe.
Set the error budget from comparable A/A timings before examining the replay.
After a failed check, inspect which costs changed with accumulation before fitting again.

The modeled MFU limit extrapolates this fit with unchanged per-microbatch work.
Use it to judge whether batch scaling alone could close the target gap. The tool
does not estimate peak memory. Before each probe, check resident gradients,
activation lifetimes, allocation peaks, and the workload's approved batch range.
Use same-batch numerical controls and independent, complete unprofiled validation
for the configuration selected for acceptance.

## Required proposal fields

Choose the diagnostic level using
[hierarchical performance diagnosis](../../perf-playbook/references/hierarchical-performance-diagnosis.md).
Resolve major framework, pipeline, existing-module/kernel, and overlap opportunities
before a broad hardware-efficiency investigation. Record why the current workload
has reached a plateau before escalating to CUBE, HBM, and link counters; low gains
from a few already-hidden collectives are not sufficient evidence. A targeted
counter capture can still be used earlier when it directly resolves the leading
bottleneck hypothesis. This is Agent planning guidance, not an automatic MFU gate.

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
