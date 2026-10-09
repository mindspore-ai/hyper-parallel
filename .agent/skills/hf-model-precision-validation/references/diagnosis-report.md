# Diagnosis and Report Contract

## Initial Loss Is Wrong

Check checkpoint identity and coverage first, followed by tokenizer/vocabulary,
sample order, label shift, padding/causal/packed mask, mask boolean convention,
weight transforms, tied weights, QKV layout, and unexpected initialization.

Compare the same batch at intermediate module boundaries. An expected loss value
from another dataset or tokenizer is not evidence.

If only the reported scalar loss has coarse BF16-sized jumps, inspect FSDP
`output_dtype`. When the model intentionally computes FP32 loss alongside BF16
activations, remove the recipe's output cast and require resolved
`output_dtype=None`.

## Loss Aligns but Norm Does Not

Inspect backward semantics, loss scaling/normalization, gradient accumulation
division, TP/CP/EP reductions, tied/shared gradients, DTensor placement, q/k norm
head-axis placement, and clipping/reporting order. Compare FP32 main gradients,
not only low-precision model gradients.

## First Step Aligns, Later Steps Diverge

Use bounded diagnostic experiments:

1. Set LR to zero to separate forward/backward differences from updates.
2. Compare pre-update FP32 main gradients and parameter-group membership.
3. Temporarily use a simple Torch Adam route if needed to bound Muon or another
   nonlinear optimizer; label this as a diagnostic-only comparison.
4. Compare main-parameter updates and copy-back/alias behavior.
5. Inspect router logits and top-k boundary changes.
6. If necessary, add experiment-only global input/output/gradient statistics per
   module, using communication to make rank-comparable summaries.

Remove debug instrumentation from product code after localization.

## Shared-State or Recomputation-Only Failure

List producer/consumer layer roles and the exact state keys read and written in
forward and backward replay. Check for a global mutable latest-value slot,
whole-layer replay that republishes state, consumer reads from the wrong source
layer, or detached producer activations. Reduce to the shortest producer-consumer
sequence, then compare gradients through the shared edge. Select the model's safe
checkpoint boundary; do not disable all recomputation without analysis.

## Validate-Only Failure

Treat it as a placement-contract defect until disproven. Production local tensors
may compute plausible values despite wrong metadata. First add a strict topology
that exposes attention internals, including q/k norm, RoPE cos/sin, and masks.

Do not turn a native model module into a `region_dispatch: false` black box to
make full-model Validate run. Doing so replaces operator propagation with a
declarative local boundary and cannot establish that the module's operators are
supported. Report the exact module/operator and the affected full-model Validate
scope as unsupported. Production and native module parity remain valid evidence,
but neither may be relabeled as Validate operator coverage.

`region_dispatch: false` remains available for an explicit communication/custom
compute injection (`local_compute_fn` or `inner_wrapper`). Its boundary contract
may be compared across Production and Validate, but its internal operators remain
outside dispatch coverage.

For a custom local region, compare Production and Validate on identical inputs.
Confirm its declared `out_src` matches the mathematical output and that tuple/list
structure is identical in both modes.

Audit operator support by the exact public callable name intercepted by DTensor,
not by a semantically similar implementation. For example, an implementation for
an accelerator-specific `npu_rms_norm` does not register PyTorch `rms_norm`, and a
5-D `conv3d` implementation does not cover `conv1d` or `conv2d`. Check the YAML
entry, constructor signature, dimensional assumptions, and live registry key.

## Matrix Failure Triage

Classify failures before changing tolerances:

1. Large first-step differences across FSDP/TP geometries with identical input
   hashes usually indicate different initial state. Compare model fingerprints and
   rerun from one common full-state DCP at the recorded initial or warm-start step.
2. Rank-dependent `KeyError`, malformed gathered payloads, or collective mismatch
   in an intermediate EP case usually means diagnostic control flow depended on
   locally absent lazy optimizer state. Gather the global state-name union and
   keep collective order identical.
3. Matching loss/model/main-gradient values with differing optimizer moments can
   be a topology-local optimizer ownership effect. Keep same-topology numeric
   comparison strict; use layout, finite, and DCP checks across topology unless
   the optimizer mathematics proves value invariance.
4. Parameter failures where absolute error is tiny near zero or full-value
   relative L2 is scale-appropriate require an explicit acceptance policy. For a
   summary-only tensor, do not claim elementwise `max_abs` or tensor
   `relative_l2`; use an explicitly named aggregate metric such as L2-norm
   relative drift. Do not silently switch from `all` to `any` or widen limits.
5. A short matrix can pass every configured case and still be incomplete. Compare
   disjoint category counts and exact topology names against the predeclared
   coverage ledger; PASS is not evidence for an omitted capability.

Always localize the first causal failure in launch order. A later rendezvous,
HCCL bind, timeout, or collective error is secondary when an earlier case left
child ranks or communication resources alive. Preserve the first failing case's
resolved config and stderr, verify process cleanup and port isolation, then rerun
only that case before interpreting subsequent matrix failures.

## OOM

Separate logits/loss memory, activations, optimizer/main-parameter state,
checkpoint-load peaks, and communication buffers. Use the HyperParallel activation
checkpoint interface when recomputation is authorized. Preserve requested model
shape, sequence length, topology, and precision unless the user explicitly changes
them, then rerun at least two steps.

If TP or fewer FSDP shards increases memory, do not assume optimizer state followed
the intended axis. Produce an FQN census of logical/local parameter bytes and
optimizer moments, including replicated/excluded modules and expert-to-EP
ownership. Attribute the delta to actual tensors before changing topology.

## Precision Passes but Performance Regresses

First verify that reference and candidate used identical topology, global workload,
dtype, compile mode, synchronization, recomputation, and measured steps. Separate
compile/allocator/first-collective/checkpoint I/O from steady state. Confirm the
optimized kernel actually ran and no optional-backend fallback silently selected
an eager path.

For CP, inspect collective bytes, async launch and first wait, stream/event
dependencies, padding, and whether projection/attention compute overlaps
communication. For EP, inspect token imbalance, empty experts, permutation and
AllToAll time, grouped-kernel eligibility, local expert count, and shared-expert
work. Attribute peak memory to parameters, optimizer state, activations,
communication buffers, or workspace. A stable miss of a predeclared threshold is
`PERFORMANCE_FAIL`; missing a fair baseline or stable evidence is
`PERFORMANCE_INCONCLUSIVE`, not PASS.

## Distributed Optimizer Checkpoint Validation

Optimizer restore occurs after model parallelization, so a DTensor parameter's
`.shape` is logical/global while `to_local().shape` is rank-local. For a
representative regular parameter and an uneven small parameter, collect rank-wise
evidence for the parameter, FP32 main parameter, `main_grad`, and each optimizer
moment:

- logical global shape;
- local shape and shard offset;
- device-mesh topology and placements;
- dtype and finite status;
- DCP metadata before save and after restore.

A small logical vector may validly have local `[1]` on some ranks and `[0]` on
others. Every rank must still report the same logical global shape, and optimizer
moments must use the same mesh and placements as their owning parameter. Validate
both same-topology continuation and at least one requested cross-topology restore.

Use the configured `model_init_dtype` during model construction. After checkpoint
restore, verify the live dtype instead of invoking an additional model-wide
conversion. Use HyperParallel's activation checkpoint/swap wrappers through their
public configuration and assert that model and optimizer FQNs remain stable in the
saved state.

## Report Layout

Store each run under an isolated timestamped directory containing:

```text
manifest.resolved.yaml
environment.json
integration_state.json
check/
checkpoint/
cases/
module_parity/
analysis/
  operator_support.json
  parallel_ownership.json
  issues.json
  optimized_capability_inventory.json
  optimization_plan.yaml
  optimization_results.json
comparison.json
summary.md
summary.zh-CN.md
optimization-summary.md
optimization-summary.zh-CN.md
```

Each launched case owns its stdout/stderr, structured `metrics.jsonl`, parameter
probes, memory observations, and checkpoint evidence below `cases/<case>/`.

`summary.md` is the detailed English report and `summary.zh-CN.md` is its detailed
Chinese counterpart. Both must state `PASS`, `FAIL`, or `BLOCKED` on line one;
report disjoint coverage counts; list every case, category, topology, threshold,
and compared step count; show maximum loss, pre-clip norm, and post-clip norm
differences; prove input/LR/checkpoint identity; summarize parameter probes and
optimizer-state comparison policy; distinguish formal deterministic evidence from
non-replayable Online smoke; include module-parity and exact operator-support
results; compare performance/memory after warm-up; and document exclusions plus
diagnostic-only changes.

Before rendering, require evidence dependency closure: a split-run handoff must
make its structure findings, module parity, and checkpoint coverage available in
the report output, and every referenced artifact must parse with a compatible
status. Distinguish three states explicitly:

- measured numeric evidence: report its actual maximum/relative error;
- exact or categorical evidence: report exact equality/status, not a fabricated
  numeric `n/a`;
- genuinely non-applicable evidence, such as checkpoint layout for a non-resume
  case: report `not applicable` with the reason.

A required-but-missing artifact is `BLOCKED`, never `n/a`. A source-only workflow
without downloaded pretrained weights reports checkpoint coverage as `NOT_LOADED`
while keeping DCP resume-layout acceptance separate.

For every claimed optimized module, CP path, and EP path, the optimization
appendices summarize its reuse decision, correctness reference, same-topology A/B
identity, module and end-to-end precision result, p50/p90/throughput/memory ratios,
fallback/overlap evidence, scope, and independent precision/performance verdicts.
Ordinary matrix performance remains descriptive; only a predeclared paired
experiment can support a high-performance acceptance claim.

The generic report command currently does not consume arbitrary optimization A/B
artifacts. Preserve its generated summaries and write the two optimization
appendices separately rather than hand-editing the matrix verdict.

Persist the exact callable audit, actual per-subtree placement evidence, and
resolved/open issue ledger as machine-readable artifacts under `analysis/` when
those audits are part of the handoff. Report generation must not infer ViT TP
from a TP launch or infer a missing callable from a related registered operator.
