# Optimized Replacement Precision and Performance Experiments

Use these experiments for every module replacement, fused implementation, CP
algorithm, or EP algorithm described as optimized or high performance. Correctness
and performance are independent gates.

## Establish Two References

Distinguish:

- the **correctness reference**, which is the authoritative released module or an
  independent literal/eager oracle and may be slow;
- the **performance baseline**, which is the previous or non-optimized production
  implementation running on the same accelerator, topology, dtype, workload, and
  Trainer path.

They may be the same implementation, but do not benchmark a CPU oracle against an
accelerator candidate or compare different global batches and call the ratio a
replacement speedup. If no fair performance baseline exists, retain the precision
verdict and report `PERFORMANCE_INCONCLUSIVE`.

## Required Experiment Layers

### 0. Executable architecture gate

Disable the candidate replacement and construct the registered production model.
Run a small forward/backward, verify finite outputs and gradients, and snapshot
the source parameter FQNs, aliases, special initialization, and state-dict keys.
This is a hard prerequisite: an optimized candidate is not allowed to supply
missing model mathematics. After replacement, prove that the expected FQNs and
parameter identities are preserved.

### 1. Module precision parity

Use identical dimensions, weights, deterministic inputs, masks, positions, state,
and dtype. Compare public output, algorithm-relevant intermediates, input
gradients, every trainable parameter gradient, and finite status. Run FP32 first
and production dtype on the target accelerator second. A faster implementation
with unexplained precision failure is rejected.

### 2. Target-accelerator module microbenchmark

Benchmark the correctness reference or fair performance baseline against the
optimized candidate with representative shapes. Include forward and
forward-plus-backward when training is in scope. Keep dtype, layout, compile mode,
autograd, random state, and synchronization boundaries identical. Warm each
implementation until compile, allocator, and first-collective effects are outside
the measured interval. Record:

- p50/p90 latency and throughput over multiple steady-state measurements;
- peak allocated/reserved memory and temporary communication/workspace memory;
- kernel/backend selected and whether any fallback executed;
- for distributed modules, collective bytes, launch/wait points, and profiler or
  trace evidence for intended communication/computation overlap;
- measurement count, dispersion, device type, software versions, and shapes.

Run enough repetitions to expose noise and report the raw samples or structured
summary. Do not remove synchronization from only one side or include compilation
in only one measurement.

### 3. Same-topology end-to-end Trainer A/B

Construct a reference recipe/override and an optimized recipe/override through the
same production builder. Use one common full-state checkpoint at the same recorded
initial or warm-start step, identical
global inputs and hashes, model/optimizer/scheduler/RNG state, topology, dtype,
gradient accumulation, clipping, recomputation, and measured step interval. Only
the target implementation selection may differ.

First apply the ordinary loss, gradient-norm, parameter-probe, finite, and
checkpoint-layout acceptance rules. Then compare steady-state step p50/p90,
tokens/s, samples/s, peak memory, and the target module's profiler evidence. A
module microbenchmark win with an end-to-end regression must be reported, not
hidden by the microbenchmark.

Use separate evidence directories because the standard validation matrix changes
topology fields but does not switch arbitrary replacement providers. Preserve the
exact launch commands and resolved recipes. Do not hand-edit one run after it has
started.

### 4. CP and EP experiments

For CP, compare synchronous/reference and optimized communication implementations
at the same CP degree, TP degree, global sequence, batch, masks, and attention
kernel. Separately report CP1-to-CPn scaling efficiency. Record communicated
tensors and bytes, async launch/wait placement, overlap, padding/imbalance, and
peak communication buffers. CP1 versus CPn is not an implementation A/B.

For EP, compare reference and optimized dispatch/compute/combine at the same EP
degree, expert layout, learned routing, tokens, capacity, and grouped-kernel
setting. Separately report EP1-to-EPn scaling, token counts per local expert,
AllToAll bytes/time, load imbalance, empty experts, and local expert/optimizer
memory. Forced or sequential routing remains diagnostic-only.

If the optimized implementation changes mathematical scope, such as an activation
unsupported by grouped GEMM, it is not a valid candidate. Report the limitation
instead of benchmarking a different formula.

## Predeclare Acceptance

Put experiment IDs and thresholds in `analysis/optimization_plan.yaml` before
running. Thresholds may include minimum throughput or latency ratio, maximum p90
regression, maximum memory ratio, and required overlap/fallback conditions. Derive
them from the user/project performance target; there is no universal speedup
number. If no target exists, run a baseline-only pilot, estimate its run-to-run
envelope, freeze the plan, and require the formal candidate improvement to exceed
that envelope without an unexplained p90 or memory regression. Never derive the
threshold after inspecting the formal candidate result.

Use these independent verdicts:

- `PRECISION_PASS` or `PRECISION_FAIL` from parity and end-to-end numerical gates;
- `PERFORMANCE_PASS` when all declared performance thresholds pass without hidden
  fallback;
- `PERFORMANCE_FAIL` when a stable fair comparison misses a threshold;
- `PERFORMANCE_INCONCLUSIVE` when shapes, devices, baseline, sample stability, or
  profiler evidence are insufficient.

The combined optimized-target verdict is PASS only for `PRECISION_PASS` plus
`PERFORMANCE_PASS`. A model integration may still have a core numerical PASS while
an optional optimization is disabled or performance-inconclusive, but it must not
be advertised as an accepted high-performance path.

## Evidence Contract

Write `analysis/optimization_results.json` with one entry per experiment:

- inventory target and experiment ID;
- reference and candidate module/recipe/override;
- source revisions, resolved configs, topology, dtype, shapes, and input/state
  identity;
- precision metrics and tolerances;
- raw or linked steady-state samples plus p50/p90, throughput, memory, and ratios;
- backend/kernel/fallback and communication-overlap evidence;
- scope (`module`, `cropped_end_to_end`, or `full_pretrained_end_to_end`);
- independent precision/performance verdicts and reason;
- exclusions, instability, and required follow-up.

The current generic matrix report does not execute or ingest arbitrary replacement
A/B experiments. Keep its machine-generated status unchanged and render adjacent
English `optimization-summary.md` and Chinese `optimization-summary.zh-CN.md`
appendices from the inventory, plan, and results. Link all three optimization
artifacts and both appendices in the final handoff.

A cropped end-to-end result applies only to the crop. Representative full-sized
module shapes can support a module-level kernel conclusion, but they do not prove
full-model throughput or communication scaling.
