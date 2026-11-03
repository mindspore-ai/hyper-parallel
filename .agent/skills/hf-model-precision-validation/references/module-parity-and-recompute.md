# Native Module Parity and Recomputation

## Choose the Oracle

Use the implementation that defines the released model, not automatically the
installed HF class. The oracle may be:

- the installed HF module when its source and checkpoint contracts match;
- the model repository's native inference/training module when HF lacks the
  released structure;
- a small literal reference written directly from a published formula only when
  neither implementation can run, with that limitation stated explicitly.

Record source revision, file, class/function, precision, and any inference-only
branches disabled for training. Do not compare two wrappers that both call the
same candidate implementation.

## Harness Contract

For each changed mathematical module:

1. Construct reference and candidate with the same dimensions and dtype.
2. Copy or map identical weights, and prove no missing or unexpected trainable
   leaf remains.
3. Feed cloned deterministic inputs, masks, positions, modality metadata, and
   state. Prevent aliasing from making two paths share an output tensor.
4. Compare public output and algorithm-relevant intermediates.
5. Backpropagate the same scalar objective and compare input plus every trainable
   parameter gradient.
6. Report maximum absolute error, relative L2 error, tolerance, and finite status.

Run CPU or accelerator FP32 first to detect semantic errors. Then run the production
dtype on the target accelerator. An unavailable native optimized kernel is an
operator-coverage limitation, not automatic evidence that the candidate math is
wrong; compare with an equivalent eager reference and record the missing kernel.

End-to-end self-consistency is not an architecture oracle: a baseline and every
candidate can share the same incorrect crop or implementation. Before accepting
module parity, reconcile semantic configuration fields and state/FQN structure
with the authoritative source independently of numerical runs. Module parity then
tests changed mathematics; the topology matrix tests distribution. Neither gate
substitutes for the other.

Before copying weights, classify configuration-conditional leaves. If the
authoritative module omits a gate, norm, cache tensor, or quantization parameter in
the selected high-precision/training configuration, the candidate must omit it or
declare a justified training-only initialization; do not manufacture a same-shaped
leaf to make state-dict loading convenient. Separate inference-only decode/KV
cache state from cross-layer tensors consumed by full-sequence training.

## Required Model Categories

Select all categories changed by the integration:

- embeddings or hashed lookup: token/hash indices, table rows, gate/fusion output,
  padding/dead-token behavior, input and table gradients;
- dense/MoE: router logits, exact score transform, top-k indices, normalized
  weights, auxiliary loss, expert outputs, combine output, and logit gradients;
- attention: projections and norms, RoPE, compressed/index keys, candidates,
  selected indices, masks, output, auxiliary loss, and backward;
- multimodal: patch embedding, vision blocks, projector/aligner, image insertion,
  routing, text-only bypass, and visual-parameter gradients.

For a score such as `sqrt(softplus(logits))`, verify the composed score and its
logit gradient explicitly. Also probe the target backend for the constituent
forward/backward operators. A mathematically equivalent composition is valid; a
dedicated fused operator is not a requirement.

## Shared Cross-Layer State

Build a minimal sequence that covers every producer and consumer role, such as
Full, Reindex, Reuse, and candidate-source layers. Assert which source-layer key
each consumer reads and that state from one forward cannot leak into another.
Compare both outputs and gradients through producer-to-consumer edges.

Whole-layer activation replay can be invalid when forward mutates or republishes
shared state. Determine safe targets from model semantics and express exceptions
with explicit `full.exclude_region` or `selective` module paths. Non-module replay
invariants, such as pinning a callable's discrete Top-K result, belong in the model
adapter's post-install activation-checkpoint hook; that hook must not choose or
expand targets. Report the actual wrapped FQNs instead of implying that an adapter
chooses safe regions automatically.

## Recomputation Matrix

Test at least these explicit selections:

- an empty `recompute_layer` set, which disables activation checkpointing;
- one intermediate set such as `[0-1]`;
- the maximum intended explicit layer set;
- repeated maximum-safe run to check stability;
- Production and strict Validate for one representative recomputation topology.

Use the same checkpoint, data hashes, global batch, routing seeds, and optimizer
state. Require aligned loss and global norm for at least ten consecutive paired
optimizer steps in formal end-to-end self-consistency, not just the first forward.
Read back the resolved layer set and record exact wrapper
count/FQNs, step time, peak memory, and whether
shared-state producers were retained or replayed.

Stable FQNs matter to checkpoint comparison. If activation wrappers change visible
module names, the case fails checkpoint compatibility. Use only wrapper
configurations that preserve the model's public state-dict identity.
