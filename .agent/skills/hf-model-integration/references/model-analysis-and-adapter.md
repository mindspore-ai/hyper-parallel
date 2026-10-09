# Model Analysis and Adapter Selection

## Analyze the Authoritative Model

Use the actual released implementation and the installed Transformers source, and
record both revisions and import paths. If they differ, identify which one is the
training oracle. Do not infer contracts from another release or a similar family.

Capture at least:

- `model_type`, architecture class, config revision, and task facade;
- module FQNs/types and trainable parameter names, shapes, ties, and aliases;
- forward arguments, return type, cache contract, and generation-only branches;
- Q/K/V projection layout, Q and KV head counts, head dimension, q/k norm axes,
  RoPE inputs, and attention mask convention;
- dense/MLP or MoE router, top-k, expert layout, dispatch, and combine semantics;
- loss ownership and whether labels are already causally shifted;
- cross-layer producer/consumer state and safe recomputation boundaries;
- modality preprocessing, fusion, routing, and conditional execution;
- training-only auxiliary losses and inference-only cache/decode branches;
- checkpoint index, tensor names, sharding, and any quantization metadata.

For a resource-constrained crop, start from the complete authoritative config and
record every mutation. Classify each numeric field before changing it:

- parameter dimensions, such as hidden/intermediate widths and low-rank ranks,
  may be scaled when every dependent shape is scaled consistently;
- semantic cardinalities, such as grouping counts, routing top-k, compression
  ratios, and producer/consumer role indices, remain at their released values
  unless an independent semantic-equivalence argument proves otherwise;
- topology divisors, such as attention/KV head counts and local expert counts,
  must satisfy every declared TP/CP/EP degree after scaling;
- table/bucket sizes must preserve the model's hashing, padding, and lookup
  invariants rather than merely fitting memory.

The default acceptance crop preserves depth and scales only compatible parameter
specifications: keep every decoder and modality-tower layer in its released order,
keep every native role index, and reduce compatible widths, intermediate sizes,
attention heads/head dimensions, low-rank dimensions, expert counts, and
architecture-owned tables or buckets consistently. Write a crop-invariant ledger
containing the released value, cropped value, classification, dependent fields,
and the matrix topologies against which it was checked. This is required when late
or repeated layers carry distinct roles such as KV production, Full, Reindex,
Reuse, Engram, multimodal fusion, or auxiliary prediction.

Do not use a shallow layer crop merely because it fits the available device. It
cannot establish full structural coverage when removed layers introduce a new
role, a different producer/consumer transition, or a modality stage. A layer crop
is allowed only when the user explicitly requests a smoke test and the report
lists every omitted role; it must not be reported as all-structure coverage.
Never renumber a late role into an earlier layer and treat that as preservation of
the released topology. Never truncate a loaded weight row range and call it an
equivalent model.

Before implementation acceptance, instantiate the crop once without replacements
and once with the production replacement plan. For every requested topology,
evaluate all divisibility and local-cardinality constraints from the ledger. A
crop that runs at TP1 but cannot realize a declared TP degree is an invalid crop,
not a topology-specific runtime surprise.

## Select the Minimum Adapter Surface

`hyper_parallel/models/adapter_spec.py` exposes optional providers. Use only
those required by the model:

- generic planner support is sufficient: no model-specific provider;
- special semantic roles: sharding rules;
- optimized module structure: replacements and explicit weight transforms;
- custom sequence communication: context-parallel provider;
- custom MoE communication: expert-parallel provider;
- non-decoder execution branches: FSDP wrap/exclusion/execution-order providers;
- non-generic loss contract: model loss provider.

If the exact architecture is not available through HF, lazy-register the custom
model class separately from its `ModelAdapterSpec`. Reusing HF config, tokenizer,
or common layers does not make an incomplete HF architecture the correct model.

Registration makes providers available; the recipe selects module replacements
and CP/EP behavior through `plan_overrides`. Sharding rules and model loss may be
selected automatically. Put weight conversion associated with replacement in the
replacement's `make_transforms()` result rather than an unused placeholder.

## Final-Model Boundary Coverage

Plan against the final tree after module replacement. A replacement can introduce
mHC, recurrent, lookup, compressor, indexer, or other modules that match no
built-in template even though their source placeholders did. TP1 execution does
not make their contracts optional: parameter roles and activation boundaries also
drive Validate mode, future TP sizes, FSDP ownership, and checkpoint layout.

Use YAML `plan_overrides` when a depth-independent FQN glob can express a uniform
contract. HyperParallel globs use `fnmatch`, not regular expressions, and `*` may
span dots. A glob can create missing boundaries only when it declares at least one
concrete `params`, `in_src`, `in_dst`, `out_src`, or `out_dst` mapping. Write the
complete contract required by every matched module. Later globs may merge
topology-conditional behavior such as `local_compute_fn`, CP wrapping, or
sequence-parallel activation placement.

Treat these cases differently:

- a full-contract glob matches final modules: insert and validate one boundary per
  module;
- a partial glob matches existing boundaries: merge only its written fields;
- a partial glob matches a real module with no boundary: fail with the matched
  FQNs, missing-contract reason, and a contract skeleton;
- a glob matches neither a boundary nor a model module: report the likely typo.

Use an adapter-side factory only when the contract cannot be expressed from
stable final-model FQNs and declarative topology conditions. Do not hide a static
per-layer sharding table in an experiment runner.

## When the Framework Contract Is Missing

Do not solve a missing extension point with a branch in Trainer, model builders,
the planner, data batching, diagnostics, or generic components. In particular,
generic code must not test a model family/model type, import a family adapter,
match family-specific FQNs, or infer a capability from a family-only config field.

First restate the requirement without the current model's names. Classify it as a
construction lifecycle, module replacement, checkpoint transform, runtime input,
parameter role, parallel boundary, FSDP unit/order, loss, recomputation, or
validation capability. Then:

1. Prefer an existing `ModelAdapterSpec`, replacement, runtime-input, plan,
   materialization, or validation provider.
2. If none fits, design the smallest model-independent provider, protocol, or
   declarative field. The framework may invoke the contract, but it must not know
   which family implements it.
3. Keep the absent-provider path identical to the previous behavior and free of
   steady-state training overhead beyond an established no-op dispatch.
4. Implement the model-specific callable, mapping, or policy under
   `hyper_parallel/models/<family>/adapter/` and register it lazily.
5. Add a generic contract test for both provider-present and provider-absent
   behavior, plus a model adapter test proving the real use case. When practical,
   exercise the contract with a second synthetic or existing adapter so its shape
   is not accidentally tied to the motivating model.

If the task does not authorize a framework API change, provide the proposed
contract, ownership, call site, default behavior, and tests as the blocker. Do not
land a temporary family check with a promise to generalize it later.

## Registration

Create `hyper_parallel/models/<family>/adapter/registration.py` with lazy
provider callables. Importing `hyper_parallel.models` must not build a model,
initialize distributed state, import an unavailable optional backend, or access
the network.

The registry discovers family directories. Modify the central registry only for
a real alias where multiple model identifiers intentionally share one adapter.

## Replacement Contract

A replacement must:

- use the repository's module-replacement factory contract;
- match both the intended FQN pattern and source module type;
- preserve supported forward arguments and return structure;
- preserve training/eval state, parameters, buffers, aliases, and state-dict
  identity unless an explicit transform describes the change;
- fail clearly for unsupported Transformers versions or module layouts;
- compose `hyper_parallel/components/modules` instead of cloning kernels into a
  model adapter.

Choose replacement ownership by semantic responsibility, not by the convenience
of adapting a call signature:

- A reusable high-performance component that already owns the source structural
  contract must itself implement the standard `module/module_fqn/context`
  replacement-factory protocol. The recipe references that component directly,
  while the declarative `module_type` remains the source-type gate.
- Use an adapter-side replacement factory only for real model-family semantics:
  deriving model-specific configuration, translating forward/state/weight/layout
  contracts, or enforcing a family-only constraint that cannot be expressed by
  `module_type` plus the component's stable structural contract.
- Do not add an adapter wrapper that merely accepts the three executor arguments,
  repeats the YAML source-type check, and returns a generic component. Do not
  populate `ModelAdapterSpec.replacements` only to expose such wrappers. This
  indirection obscures the actual high-performance implementation without adding
  a model adaptation contract.

Do not replace both a parent and its descendant in one ambiguous plan. Treat an
unmatched required replacement as an error rather than silently running a mixed
implementation.
