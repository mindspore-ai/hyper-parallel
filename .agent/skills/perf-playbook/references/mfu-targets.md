# Utilization conventions and acceptance targets (MFU / HFU / throughput)

> **Evidence provenance:** Mixed. Published conventions such as PaLM MFU and
> target-setting methods, with a utilization audit from a MoE pretraining run in
> this repository. See [evidence provenance](../SKILL.md#evidence-provenance).

How to measure utilization, set a defensible target, and decompose the
gap when the target is missed. Load this when a
tuning task arrives as a requirement — "reach MFU ≥ X", "throughput ≥ Y
tokens/s/die", "step time ≤ Z" — rather than as an open-ended climb.

## 1. Utilization measurement

Check whether the selected Trainer revision exposes an MFU meter. Where
`training.mfu` is available, enable per-step reporting in the training config:

```yaml
training:
  mfu:
    enabled: true
    peak_tflops_per_device: <dense peak at the training precision>
```

This publishes `performance/model_tflops_per_device`,
`performance/tokens_per_s_per_device`, `performance/mfu_percent`, and —
with full recompute and/or a declared extra term — `performance/hfu_percent`,
on the progress line and in the logs.

For a checkout without this interface, calculate utilization in the
experiment's benchmark wrapper. Record the declared FLOPs per update,
the aggregate effective tokens, participating die count, the timed update
range, elapsed time, and per-die peak. Emit the result through the metric
field configured in the experiment harness. Report the HFU coverage from
the independent [arithmetic counter capture](../../profiling/SKILL.md).

**The trainer's built-in estimate is PaLM-style.** Per-token model FLOPs =
6·N_mm + 12·L·H·d_head·S, where N_mm counts active matmul parameters only
(embedding lookup excluded; routed experts scaled by top_k/num_experts) and
the attention term is the full S×S form without a causal discount. State the
convention next to every number you report: the same run reads very
differently under 6N-active, causal-halved, or extended conventions, and
mixing them silently is how utilization claims go wrong.

For architecture-aware MFU, independently derive required forward and backward
FLOPs by component and observed batch geometry. Use valid query-key pairs per
causal segment and count only visible keys for windowed, sparse, or compressed
attention. Count active experts, recurrent blocks, visual grids, and supervised
output rows in their respective components. Keep the same useful work for two
equivalent execution paths. Recompute the reference and candidate from the same
component definition when adopting a new accounting basis. The Trainer's
full-square estimate remains available for historical comparisons. [PaLM, Appendix B](https://arxiv.org/pdf/2204.02311)

For conditional memory or embedding modules, derive the arithmetic of the active
projections, gates, convolutions and backward path. Do not multiply the capacity of
a lookup table by six: gathering entries is data movement, not a dense matrix multiply.
Separate Host arithmetic from accelerator arithmetic. If model work runs on Host,
report its contribution separately and state whether an end-to-end useful-work
estimate includes it; do not present that estimate as measured accelerator execution.

**HFU counts executed work.** With full activation recompute the forward
replay is added automatically. Other executed-but-not-model work — optimizer
Newton–Schulz iterations (Muon), router fix-ups, selective recompute — is
declared explicitly:

```yaml
    hfu_extra_tflops_per_step: <per-device TFLOPs per step, derived, not guessed>
```

Record the derivation of the extra term in the tuning summary. Prefer device
arithmetic counters when they cover the declared training phases and ranks.
Include recomputation and optimizer work in HFU and label the measured or estimated
coverage. Record communication, memory transfers, and Host dispatch as time or
bandwidth. Report MFU, HFU, and effective tokens/s/die for the same workload,
along with optimization wall time and measured allocation cost. Divide the
global token count in the measurement window by its elapsed seconds and the
number of participating dies. For supervised workloads, report input
tokens/s/die separately from supervised tokens/s/die.
When the parallel degree changes, verify how the optimizer assigns matrix
work to ranks before carrying its FLOP estimate to the new topology.

Match counter coverage to the reported source, workload, ranks and training phases.
Do not extrapolate one rank's count across heterogeneous expert or pipeline ranks
without evidence of equal work. Report uncovered arithmetic explicitly; Host sparse
optimizer work does not belong in accelerator HFU.

When device count is tunable, use the [parallel strategy analyzer](../../parallel-strategy-analyzer/SKILL.md)
and observed memory peaks to select feasible placements. Include every resident
expert and optimizer state in the memory estimate. Screen lower device counts at
fixed global batch, microbatch shape, model, and optimizer. Adjust gradient
accumulation for the effective data-parallel degree and record actual updates,
tokens per die, and communication. If operator shapes are underfilled, evaluate
microbatch size or sequence length in a separate capacity series. Report the
participating dies and the allocation scope with each chosen configuration.

Check the tensors produced by the collator when increasing the microbatch size.
A packed loader can merge samples into one physical row. Verify that the
attention kernel receives the intended batch dimension, position resets, and
document boundaries. With equal-length independent samples, a row-preserving
collator can grow the batch dimension while keeping causal visibility and labels
identical. Keep the boundary assertion on the implicit causal path.

Before a fixed-row capacity trial, exercise the actual tokenizer, transform,
and collator on representative records at the target length. Include short
source records and final fragments in the check. A tail-only filter can keep
a short record as its first window. Confirm that every tested row satisfies
the intended physical shape and causal boundaries. Record the data identity
when rebuilding the sample set. Separate batch-construction failures from
accelerator allocation failures using the first failing call site.

Report useful model FLOPs per effective token for each sequence-length point.
Additional attention work per token can raise MFU while reducing effective
tokens/s/die. Reserve fixed-workload speedup claims for matched model work.
Evaluate a longer context against its own throughput, memory, and numerical
requirements. If the measured context sweep approaches the supported length
with little remaining MFU gain, return to profiling a fixed shape before
investing in more capacity trials.

## 2. Acceptance targets

When the installed Trainer supports target fields, configure them as follows.
The standalone experiment harness can apply the same declared thresholds to
benchmark metrics emitted by an external wrapper.

```yaml
    target_mfu_percent: 16        # judged >= per logged step
    target_hfu_percent: 22        # requires full recompute or the extra term
    target_tokens_per_s_per_device: 5200
    target_step_time_s: 2.1       # judged <=
```

Each logged step is judged; training end prints one verdict line per target
(mean, met-step count, MET/MISSED). Impossible configurations — a
utilization target without a peak, an HFU target that could never be
measured, a non-positive threshold — are rejected at train begin.

## 3. Setting the target (never from thin air)

Absolute MFU depends heavily on the model and configuration: a sparse-MoE
run with a matmul-heavy optimizer can sit near 12% while healthy, a dense
model can sit far higher. Two defensible procedures:

1. **Baseline × factor.** Measure the baseline first (same nodes, same
   config, ≥30 logged steps, first step discarded), then set the target as
   baseline × an agreed factor. Fast, fair, and self-calibrating; the
   factor, not the absolute, carries the ambition.
2. **Roofline subtraction.** Start from the device's measured GEMM ceiling
   (microbenchmark, not the datasheet), subtract the known structural
   overheads one by one — optimizer phase share, recompute share,
   attention-term share, uncovered communication — and set the target
   slightly below the remainder. Each subtracted share must come from a
   measured breakdown (the `profiling` skill's optimizer/compute/comm/free
   split), so the target doubles as a decomposition of where the headroom is.

Register the target before optimization starts; moving it afterwards voids
the exercise.

## 4. When the target is missed: decompose, don't flail

Rank regions by their measured contribution to the critical path. The
following is a diagnostic checklist, not a fixed optimization order:

- Optimizer phase: inspect update kernels, state transfers, and
  synchronization. Muon Newton–Schulz work can be significant, but its
  priority depends on the measured share for the current configuration.
- Recomputation: measure replay time and its overlap with other work.
  The HFU−MFU difference describes additional FLOPs, not a wall-clock share.
- Attention: compare measured attention time with the architecture's
  useful-work term, especially at long sequence lengths.
- Exposed communication and device idle: use the step breakdown and
  cross-rank evidence to identify waits on the critical path.
- Input pipeline and logging: correlate Host events with device stalls,
  including persistent delays and step-to-step variation.

Avoid adding overlapping phase or operator durations as independent
step-time shares.

Address the largest measured share with the matching technique from
[feature-scenarios.md](feature-scenarios.md); one change at a time,
re-measure same-node back-to-back, judge against the registered target, and
record every attempt — void attempts included — in the tuning summary.

## 5. Measurement discipline (numbers that don't follow it don't count)

MFU, HFU, and kernel-level CUBE efficiency answer different questions. For
hardware-efficiency analysis after a workload-dependent plateau, use the
[counter and bandwidth conventions](hierarchical-performance-diagnosis.md#measure-hardware-efficiency-with-matched-denominators).
Kernel busy ratios and theoretical tensor traffic are not measured FLOPs or HBM
bandwidth. Keep these diagnostic metrics separate from end-to-end acceptance.

- Same nodes, same card count, same step count; discard the first step;
  report means over logged steps.
- Keep the config dump of every measured run; the re-run command must be
  reconstructable from the dump alone, never from memory.
- Judge each change immediately (fail fast); a regression parked "to check
  later" burns machine time twice.
- Use bitwise comparisons when the candidate claims bitwise equivalence.
  For an equivalence-preserving implementation with different floating-point
  rounding, declare and justify tolerances before testing. Fused or
  different-precision kernels require local output/gradient parity and a
  full-model check under the declared criterion, as specified in the
  [training RSI protocol](training-rsi.md#separate-correctness-from-performance).
  Similar loss alone does not establish numerical equivalence.
