# Kimi K3 validation

## Integrated candidate scope

The sections below document the original eager CP1 baseline and its contracts.
The current parallel candidate adds packed TND, gated MLA CP, KDA state/Ulysses
adapters, EP, the new encoder transport and model-owned visual recomputation.
Use the parallel recipes and record fresh evidence on the assembled source tree;
availability of a shared component is not numerical acceptance of a composition.
Structural-depth runs preserve 93 decoder layers and 27 visual layers with reduced
widths. Report this distinction from production capacity explicitly.

Same-topology continuation compares complete persisted state and aligned data.
Cross-topology replay verifies the same initial parameters/optimizer and global
data interval but does not claim a rank-local RNG or cursor remapping. Compare
full gradients and actual updates, including absolute/RMS scales and cosine;
do not use a universal percentage threshold on tiny parameters. Keep failed
strict module criteria visible even when an independent FP64 reference identifies
floating-point reduction error as their cause.

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

Layer recomputation, uneven valid-token accumulation, checkpoint continuation,
packed inputs and CP/EP need separate future integrations and evidence for their
final assembled implementations. No static PASS report is included in this example.

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

CPU preservation checks compare visual outputs, parameters, all present gradients
and absent-gradient identities with the prior native implementation. They also
check complete visual planner ownership and the public construction entry. These
checks are distinct from an independent MoonViT/projector oracle and from device
acceptance of the newly assembled branch. Full FSDP layout/precision,
checkpoint continuation and language-layer recomputation require their own final
recipe evidence; visual recomputation, encoder CP and multimodal packing are not
exposed by this candidate. Real-corpus convergence and production capacity remain
unverified.
