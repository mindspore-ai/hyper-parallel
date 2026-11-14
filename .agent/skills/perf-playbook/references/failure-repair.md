# Repairing failures during training RSI

**Evidence provenance:** Mixed. Distributed initialization and asynchronous-device
failure investigations, expressed as a model-independent procedure. A hypothesis
remains unverified until its predicted behavior is observed on the current workload.

Use this procedure when a correctness failure or a material performance defect
interrupts S3 or S4. A shorter sequence, smaller batch, disabled module, or different
layout can help isolate a cause. Success on that diagnostic workload does not close
the original defect.

## Preserve the failure and define closure

Record the failing source and workload digests, actual runtime versions, topology,
first failing rank and update, last completed update, raw logs, and relevant traces.
Separate the first causal error from subsequent collective timeouts. Retain failures
and invalid measurements in the experiment ledger; do not use partial-step MFU as an
accepted result.

| State | Evidence needed before advancing |
|---|---|
| Observed | A reproducible symptom and its exact workload; unknown causes stay unknown |
| Localized | A source call site or execution dependency supported by a discriminating probe |
| Repaired locally | A regression test fails before the repair and passes afterward, with relevant output and gradient checks |
| Replayed | The original failing workload completes beyond its failure point and through the planned window, on all ranks |
| Accepted | The declared numerical gate passes and a separate uninstrumented run confirms the performance result |

For a performance defect, define the observable mechanism before changing code:
allocation retries, exposed waits, fallback invocation counts, or another measured
critical-path cost. Verify that the repair removes that cost rather than moving it
to another phase. Preserve any remaining unknowns in the result.

The Agent selects the probes and repair. The relevant diagnostic Skill supplies the
method. The Harness executes the configured regression and acceptance commands and
records their outcome. Add task-specific checks to the adapter's `validator_files`;
this document is a procedure, not an automatically enforced root-cause detector.

## Localize asynchronous device errors

An accelerator error may surface at a later synchronization, allocation, or unrelated
operator. A Python traceback alone does not identify the faulty kernel.

1. Align the earliest device/runtime error with per-rank logs. Capture the operator
   name, shape, dtype, layout, source caller, stream, and producer dependencies.
2. Add a narrowly scoped diagnostic around the suspect operation. Record before
   waiting for prior work, immediately before dispatch, and after its completion.
   Distinguish a failure in preceding work from one in the suspect operation.
3. Inspect producer/consumer events and storage lifetime. A stream fence can mask a
   missing dependency or prevent a premature free. A diagnostic run that succeeds
   only with extra synchronization is evidence for investigation, not a repair.
4. Reproduce on real shapes, including empty or ragged partitions and non-contiguous
   inputs where applicable. Validate outputs and all affected gradients.
5. Remove diagnostic synchronization and heavy tracing. Replay the original workload
   and verify the intended implementation still runs before measuring MFU.

An `Index` error is not evidence of OOM without an allocation failure. Similarly,
an AI CPU collective launch is not by itself a compute fallback. Use
[cluster-debug](../../cluster-debug/SKILL.md),
[debug](../../debug/SKILL.md), and [profiling](../../profiling/SKILL.md) for collection.

## Establish logical initialization identity across layouts

The same seed does not guarantee the same logical parameters after changing FSDP,
HSDP, TP, or EP. Local tensor shapes and random-number consumption may change.
Adding the global rank to a seed can also make replicated copies disagree.

- Compare replicas exactly where their logical shard coordinates coincide. Check
  shard offsets, global shapes, dtype, tied aliases, and uneven or empty partitions.
- For layout comparisons, restore a common logical checkpoint with resharding, or
  use an initializer whose values are proven to depend on logical tensor indices.
  A canonical full tensor followed by slicing is suitable for small parameters;
  it is not a scalable default for the full model.
- Verify the actual initialized tensors before any optimizer update. Save parameter
  coverage, digests, and checkpoint provenance. If unchanged large expert shards are
  replayed separately, verify their identity and document the remaining checkpoint
  coverage. A matching seed or successful checkpoint-load call is insufficient.
- For a repaired initializer, establish a new reference trajectory. Do not compare
  it against a baseline whose replicas were already inconsistent and classify the
  difference as harmless floating-point noise.

## Count optimizer updates and microbatches separately

Derive gradient accumulation from the resolved training schedule. In an ordinary
data-parallel schedule, accumulation is global batch divided by micro-batch size
and the number of independent data replicas. Tensor, context, and pipeline ranks
must not be counted as independent samples.

For example, 256 samples per update, 128 data replicas, and micro-batch size 1
produce two microbatches per update. A ten-update gate must compare twenty ordered
input records per rank. Check masks and positions as well as token IDs. A matching
but truncated record set on both sides must fail coverage validation.

Use `audit_rank_steps.py --microbatches-per-update` when its per-update JSONL adapter
is applicable. Other adapters must enforce equivalent all-rank coverage. Record
collector corrections separately from immutable training commands and original
scheduler outcomes; do not silently rewrite an old FAIL as a job-level PASS.

## Verify repeatability through native dispatch

When identical-state runs diverge, repeat the suspected native operation with the
same inputs and upstream gradient before spending another full-model run. Compare
its output and every affected gradient, including shared-key and parameter
reductions. A repeatable forward does not establish a repeatable backward.

Trace the effective determinism setting through the framework, extension binding,
workspace query, tiling plan, and device kernel. A binding can temporarily override
the setting while preparing an execution plan; restoring it before launch does not
prove that the plan selected deterministic reductions. Record the loaded extension
and operator-package digests as well as the requested framework settings. Check the
pinned implementation: documentation for another version is not runtime evidence.

Use a bounded paired probe to change only the suspected dispatch setting, then
replay the full-model comparison with its original numerical limits. Small native
gradient variation can be amplified by later updates, but its presence alone does
not establish the cause of a multi-step scalar mismatch. Keep local repeatability,
reference parity, and whole-model acceptance as separate results. Do not widen a
gate simply because repeated runs of the current implementation also fail it.

If an authorized performance mode uses nondeterministic reductions, declare that
mode and validate its output and gradient accuracy against the reference. A gate
that exercises only a different deterministic kernel does not validate the measured
path. This applies even when both phases use the same Git commit and global flags.

## Resume optimization with the repaired baseline

Close the original regression before declaring the affected path accepted. Then
continue the task's authorized sequence-length, batch, and parallel-layout search in
[linked workload series](../../mfu-riser/references/contracts.md#linked-series).
Short diagnostic workloads do not replace a requested long-context target.

Retain negative experiments with the tested shapes, runtime, mechanism, and retry
condition. Revisit them when a relevant dependency changes, rather than repeating
the same experiment. A kernel improvement within whole-model measurement noise
remains a local result; a later combination needs its own end-to-end acceptance.
