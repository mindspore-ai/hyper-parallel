# Examples, Tests, and Validation Handoff

## Examples and Data Modes

Provide the examples that the requested scope needs:

- cropped Offline: complete HF config plus authorized layer crop, `from_config`;
- cropped Online: one-step functional smoke;
- full Offline: complete local checkpoint, `from_pretrained`;
- full Online: one-step functional smoke.

Formal cross-run precision conclusions require deterministic, replayable inputs.
Offline Indexed Dataset is the default. Online may be used only when it proves
deterministic sample order and emits identical per-step input hashes across every
candidate; otherwise it is a functional check of tokenization, packing, masks,
modalities, and Trainer plumbing.

Use the current data lifecycle in examples: `TokenBatchLoader` plus
`TextParallelBatch` for encoded text, and `OmniPackingLoader` plus
`OmniParallelBatch` for multimodal packing. Model-specific sample/batch encoding
belongs to `dataset.data_transform`; model-only forward metadata belongs to the
get-batch `runtime_input_adapter`. Treat any recipe that still names a removed
`DataBatchAdapter`, `batch_adapter`, legacy VLM collator, or model-specific
get-batch class as a failed preflight rather than documenting it as compatible.

For small cropped smoke cases, a launcher may explicitly generate deterministic
local data. A full Offline launcher must require existing `.bin`/`.idx` files and
must not silently download or synthesize replacement data. Use
`local_files_only=true` for training launches and report missing model, tokenizer,
or dataset assets clearly.

Audit label ownership. Packed Offline preparation with `--pack-to-seq-len N`
produces documents of `N+1` tokens so the reader can construct `tokens[:-1]` and
`tokens[1:]`. Ensure causal shift occurs exactly once and that
`labels_are_shifted` matches the emitted sample contract.

Resolve every committed launcher with its default device count before handoff.
Check that its world size realizes the declared TP/CP/PP/FSDP/EP mesh and that
`global_batch_size` is divisible by the resolved distributed micro-batch size.
Do not validate a smaller-rank override and infer that a larger launcher default
is valid.

For multimodal recipes, enumerate the final vision blocks, projector/aligner, and
fusion modules and compare those FQNs with every plan glob. Assert matched counts,
not only that a glob matched at least one module. Run at least one real-media
forward/backward with the launcher's default world size. Enable
`model_integration.mode=runtime` and require structured field-ownership,
modality-gradient, parameter-probe, and rank-count evidence; a success marker or
finite scalar loss alone is only a functional smoke.

## Focused Tests

Cover the applicable contracts:

- every newly added framework extension with provider-present and provider-absent
  generic fixtures; the absent path must retain prior behavior and normal-training
  performance characteristics;
- the motivating model through its registered adapter, without a framework-side
  family import, model-type branch, or family-specific FQN match;
- registry lookup and side-effect-free lazy import;
- replacement matching and required-pattern failure;
- forward/state-dict compatibility and weight-transform round trip;
- changed-module forward/backward against the authoritative HF or native
  implementation on identical weights and fixed tensors;
- mask convention, RoPE, cache, packed sequence, and q/k norm axes;
- CP wrapper versus non-CP attention mathematics;
- EP compute versus non-EP grouped-expert mathematics;
- reference versus optimized replacement output/intermediate/input-gradient and
  parameter-gradient parity in FP32 and production dtype;
- target-accelerator module microbenchmarks and same-topology end-to-end paired
  runs for every claimed high-performance replacement, CP, or EP implementation;
- explicit detection of optional-kernel fallback, plus synchronized warm-up,
  steady-state latency/throughput, peak memory, and communication-overlap evidence;
- planner placements in Production and Validate construction;
- complete final-model boundary coverage, including full-contract glob insertion
  and an actionable failure for a partial glob targeting an unplanned module;
- crop-invariant and topology-feasibility checks across every declared matrix
  degree, including a nontrivial TP divisor;
- launcher-default world-size, mesh, and global-batch compatibility;
- real-media runtime evidence proving vision/projector gradients and exact
  final-tree plan coverage for multimodal recipes;
- optimizer wrapper, FP32 main parameters, main gradients, and update/copy-back;
- DCP round trip for model, optimizer, scheduler, main parameters, train state,
  RNG, and dataloader state where supported;
- uneven FSDP parameters on both nonempty and empty ranks: parameter, main
  parameter, main gradient, and optimizer moment must agree on logical shape,
  mesh, and placements before save and after restore.

For module parity, compare public output plus algorithm-relevant intermediates.
Examples are attention compressed/index keys and selected candidates; MoE router
logits, score transform, top-k and combine weights; Engram hashes, lookup rows and
fusion output. Compare input and trainable-parameter gradients as well as forward
values. Run FP32 first to isolate semantics, then the production dtype on the
target accelerator, and record unsupported native kernels separately from a
mathematical mismatch.

Follow `.agent/rules/testing.md`. Distributed ST launchers must remain free of
top-level Torch, MindSpore, and HyperParallel imports.

## Handoff Manifest

Write the framework-generated schema-v1 `integration_handoff.yaml`. Its stable
fields are `schema_version`, `family`, `manifest`, `structure_findings`,
`module_parity`, `checkpoint_coverage`, and `status: PASS`. Before handoff, resolve
every referenced path relative to the handoff, parse each artifact, and require
its status to agree with the handoff. Copy/import that evidence into a split matrix
output before report generation; advancing the workflow state without the
referenced artifacts is not an evidence handoff. The resolved manifest and
evidence files carry revisions, assets, construction mode, data identity,
parallel topology, precision policy, and module/checkpoint results; do not copy
those facts into a second hand-maintained handoff schema. Never commit a static
PASS handoff as a user configuration example: it is evidence from one concrete
parity run and must be regenerated.

Keep `analysis/optimized_capability_inventory.json` and
`analysis/optimization_plan.yaml` beside the handoff. They are Agent-produced
design evidence rather than extra handoff-schema fields. Every optimized module,
CP path, and EP path required by the requested scope must link to an independent
precision oracle and a paired performance experiment. A functional or parity PASS
without the paired performance result cannot support a high-performance claim.
