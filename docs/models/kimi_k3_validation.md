# Kimi K3 validation

## Integrated candidate scope

The sections below document the original eager CP1 baseline and its contracts.
The current parallel candidate adds packed TND, gated MLA CP, KDA state/Ulysses
adapters, EP, the new encoder transport and model-owned visual recomputation.
Use the parallel recipes and record fresh evidence on the assembled source tree;
availability of a shared component is not numerical acceptance of a composition.
Structural-depth runs preserve 93 decoder layers and 27 visual layers with reduced
widths. Report this distinction from production capacity explicitly.

For integration debugging, an explicitly reduced 8-layer decoder can preserve
six KDA layers, two MLA layers, dense and MoE/QB branches, and the initial AttnRes
history. The original block size is 12 decoder layers: at least 13 layers are
needed to exercise the next history append without changing that size. A two-layer
visual tower exercises the encoder/projector bridge. Record the reduced schedule
with each run; this does not replace structural-depth qualification.

Packed eager convolution retains low-precision saved input and accumulates all
tap contributions to dX before one cast. This removes per-tap input-gradient
rounding. State CP still partitions convolution gradients across rank boundaries;
that separate halo merge and parameter-gradient reduction require their own
qualification, including fused joint-SiLU backward.

Same-topology continuation compares complete persisted state and aligned data.
Cross-topology replay verifies the same initial parameters/optimizer and global
data interval but does not claim a rank-local RNG or cursor remapping. Compare
full gradients and actual updates, including absolute/RMS scales and cosine;
do not use a universal percentage threshold on tiny parameters. Keep failed
strict module criteria visible even when an independent FP64 reference identifies
floating-point reduction error as their cause.

## Mainline normalization and shallow qualification

The Trainer accumulates token-weighted loss numerators and token counts while
streaming microbatches. It normalizes FP32 gradients once before clipping.
Checks must compare each microbatch's recorded count with the consumed prefix,
then compare the final denominator with all actual supervised targets. Counting
an entire accumulation window in advance is no longer part of this protocol.
Shifted multimodal labels still require their explicit counting contract.

A four-layer debugging crop can cover three KDA layers, one MLA layer, dense and
latent MoE branches. An explicitly reduced AttnRes block size of two exercises
history boundaries in this crop. Such a run does not retain the published block
schedule. Record the source revision, seed, sequence length and block size.

Fixed-input convolution replay separates forward errors, interior dX and the
halo boundary. Compare each core VJP with an independent FP32 recurrence as
well as CP1, so expected BF16 partitioning error is not mislabeled as a missing
state gradient. An owner-computes-boundary prototype must improve the complete
Trainer before it replaces the shared convolution implementation.

For numerical qualification, declare budgets before an independent seed/length
run. Track whole-gradient RMS/relative L2, scalar loss/norm drift, and error
relative to actual parameter updates. Inspect small-gradient tensors with their
absolute scales. Same-topology checkpoint continuation and mathematically
unchanged QB/scheduling/replay options require exact state and input agreement;
cross-topology BF16 runs do not require bitwise equality. Passing these shallow
checks does not establish production-scale convergence or memory capacity.

## Original baseline qualification

The model-owned validation provider checks the complete configured text layer
schedule, dense/MoE branch selection, residual modules, and QB state. QB-disabled
models retain the router correction bias and must not contain a QuantileBalancer.
The accepted baseline topology is TP=CP=EP=PP=1 with DP/FSDP ownership handled by
the shared framework.

## Portable architecture check

Use the final production class or
`adapter.validation.model_builder.build_validation_model` with a local, explicitly
sized configuration. Run forward/backward with all replacements disabled. Cover
both KDA and MLA, dense and routed/shared expert branches, AttnRes boundaries, raw
labels, aligned chunk targets, right padding, and QB enabled/disabled. Confirm
finite gradients, complete parameter names and state, and lazy optional imports.

A shallow smoke validates these chosen branches only. Full structural validation
must retain all released layer roles and document every dimension change.
Materialization through meta/to_empty, actual FSDP precision/layout and layer
recomputation require separate checks on the final assembled branch.

## Independent module parity

Edit the existing [manifest](../../examples/kimi_k3/module_validation.yaml) to
point to local candidate configuration, local released source and an output
directory. The provider reads only the two reviewed pure-Torch source symbols;
it validates the reference file hash before execution and never imports the
inference backend.

```bash
python -m hyper_parallel.tools.model_integration parity \
  --manifest examples/kimi_k3/module_validation.yaml --device cpu --dtype float32
```

The module cases cover SiTU and block AttnRes outputs and trainable/input
gradients. They do not constitute an independent oracle for KDA, MLA, latent MoE
or the complete QB step lifecycle. Comparing the reorganized native model with
its prior implementation checks preservation of behavior, not independent model
correctness.

## Distributed and training evidence

Use FP32 initialization/main parameters/main gradients/reduction and BF16 FSDP
compute without automatic forward-input casts. Record actual parameter layouts,
target counts, input identity, LR and optimizer/QB step boundaries. Evaluate
same-weight gradients separately from training trajectories after weights and
routes have diverged. Include absolute error and error relative to actual updates;
a small denominator or a fixed percentage alone is not a correctness verdict.

The integration exposes layer/branch recomputation, packed inputs, CP/EP,
checkpoint continuation and periodic evaluation. Qualify each selected combination
on the final assembled implementation. No static PASS report is included in this
example.

## Composite model and real data entry

`adapter.validation.model_builder.build_multimodal_validation_model` constructs
the complete locally configured image/video/text class. The composite validation
provider checks the visual tower depth, projector width, language layer schedule,
QB state and modality parameter patterns. `validate_multimodal_targets` observes
raw, aligned and chunked target modes, including padding and visual target
exclusion, without changing the inputs.

The existing image recipe enables `model_integration.mode: runtime`. Validate its
actual JSONL dataset and Omni batch entry at `micro_batch_size=1`; direct multi-row
collator/model checks do not establish multi-row transport support. Include local
raw video containers, pre-sampled frames, mixed image/text records, text-only and
zero-target batches, loader state restoration, and raw versus pre-shifted target
counts. Video provenance retains relative presentation timestamps and can report
unknown duration when explicit sampling avoids duration inference.

The standard multimodal provider now declares independent MoonViT block,
complete-tower and projector module cases. The reviewed local released source is
hash-pinned; its complex RoPE executes on a differentiable CPU oracle. The tower
case includes an image, a three-frame group, position interpolation and temporal
pooling. Each case selects the actual builder-owned candidate modules, sets the
requested device/dtype, and clears gradients before comparing shared components.
This is component parity, not full image/text or cross-topology qualification.

CPU preservation checks compare visual outputs, parameters, all present gradients
and absent-gradient identities with the prior native implementation. They also
check complete visual planner ownership and the public construction entry. These
checks are distinct from an independent MoonViT/projector oracle and from device
acceptance of the newly assembled branch. Full FSDP layout/precision,
checkpoint continuation and language-layer recomputation require their own final
recipe evidence. The parallel candidate exposes visual recomputation, encoder CP
and multimodal packing; the presence of these switches does not qualify their full
cross-product. Real-corpus convergence and production capacity remain unverified.

## Validation lifecycle checks

For periodic evaluation, use independent train and validation datasets with
`eval_iters > 0`. Compare evaluation enabled versus disabled at the same topology,
including `reshard_after_forward: false`. Cover unequal valid-target counts,
raw/aligned/chunked loss modes, train-eval-train, duplicate step/epoch triggers,
and restoration after validation exceptions. Compare full gradients and final
optimizer, scheduler, QB, data and RNG state, not just validation losses.

A validation window must preserve the stateful loader cursor and each module's
previous training mode. Training input hashes must exclude validation batches.
A CP callback that rejects every `model.eval()` call is incompatible with this
entry; cache generation and attention-weight outputs can remain unsupported.
Qualify multimodal CP/ECP separately from text, including text-only and dummy
visual work. A short train-eval-train smoke is not a ten-update numerical result.

Continuation must also cover datasets that actually consume NumPy randomness.
Plain global NumPy state is serialized into checkpoint metadata; independent
Generator instances and worker augmentations remain dataset-owned. Older saves
without this field cannot establish exact NumPy continuation. Check zero-target
QB windows separately from optimizer skips: a successful BF16 run does not prove
an AMP scaler or custom optimizer's skipped-step protocol.

## Memory and fused-operator qualification

Compare `layer`, `attention`, `ffn` and `attn_res` replay independently against a
fixed default-history baseline, then add swap. Check `blocks` both with and without
recomputation: a changed gradient accumulation order is not evidence that replay
itself is wrong. Record allocated and reserved memory separately. Exclude tensor
snapshot/profile steps from timing and memory summaries, and use isolated runs
for throughput claims.

Validate optional SiTU and AttnRes libraries through the public functional APIs
with recorded package versions and binary identities. Compare outputs and every
input/parameter gradient against eager and an independent higher-precision
reference over representative scales. Operator execution, module parity and
full-Trainer replacement are separate levels of evidence. Keep eager as the
baseline until the selected fused recipe has completed numerical qualification.

## Worker augmentation and skipped updates

For map-style datasets with random worker augmentation, set
`dataloader.checkpoint_worker_rng: true` and keep worker count, prefetch settings
and DP topology fixed across restoration. Compare the next ten consumed batches,
including actual augmented values. Dynamic packing must retain full augmented
buffer entries; index-only replay cannot reconstruct stochastic transforms.
Dataset-owned generators still require the dataset state protocol.

`training.skip_nonfinite_gradients: true` adds an opt-in pre-update decision using
the pre-clipping gradient norm. All training ranks agree before any optimizer
updates its parameters. A rejected transaction clears gradients and pending QB
observations, preserving main parameters, optimizer moments, router bias, QB
update count and learning-rate schedulers. Model-owned router finalizers must
provide `discard_router_step`; unsupported finalizers fail explicitly.

`global_step` continues to count consumed global batches. `optimizer_step` counts
applied optimizer updates and is saved with the checkpoint. Older checkpoints
without this counter use their global step, consistent with the previous
no-skip protocol. Step-based evaluation skips rejected transactions and uses
applied updates for its cadence when this policy is enabled. Checkpoint and
training iteration cadence still follow consumed batches, allowing a save at a
skipped boundary without replaying the discarded data on restoration.

Qualify an injected nonfinite gradient on only one rank, a save immediately after
the discarded batch, and ten subsequent successful updates after restoration.
Do not interpret arbitrary optimizer return values as skip signals. This policy
does not integrate an AMP GradScaler or custom optimizer that independently
skips after the pre-update decision. Runtime numerical diagnostics may still
reject nonfinite tensors before this policy; use build-only integration mode
when deliberately exercising the skip protocol.
