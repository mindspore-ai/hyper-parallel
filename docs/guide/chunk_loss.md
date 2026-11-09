# Chunk Loss Usage and Qwen3-MoE Integration

**English** | [简体中文](chunk_loss.zh.md)

Chunk Loss reduces peak device memory during output projection and cross-entropy for language models with large vocabularies. It processes the current rank's
local sequence dimension in chunks for LM Head computation, FP32 cross-entropy, and first-order gradients, without ever materializing the complete
`[batch, local_sequence, vocabulary]` logits.

The implementation has three layers:

- `components/losses/chunked_cross_entropy.py`: Model-independent chunked linear cross-entropy.
- `ChunkedCausalLMLoss` and Trainer wiring: Label handoff, local normalization, and model binding.
- `models/qwen3_moe/adapter/runtime/chunked_loss.py`: Qwen3-MoE-specific integration before the full LM Head.

## 1. Why Integration Must Happen Inside Model Forward

A normal causal-LM forward first executes:

```text
final hidden [B, S_local, H]
    -> lm_head
full logits [B, S_local, V]
    -> FP32 cross entropy
loss
```

Adding a loss only in `BaseTrainer.postforward()` is too late: the model has already materialized full logits and incurred the peak memory
usage. Deleting logits afterward cannot eliminate that peak. With Chunk Loss enabled, the Trainer first converts loss-only
labels into the model input protocol. The model-family adapter then obtains the final hidden states inside the root model's forward,
invokes the chunked loss directly, and skips the original full `lm_head(hidden_states)`.

The Qwen3-MoE adapter returns `ChunkedCausalLMOutput(logits=None)`. `ChunkedCausalLMLoss` checks
`logits is None` again, raising immediately if the model incorrectly returns to the full-logits path instead of silently degrading.

## 2. Inputs, Outputs, and Memory Complexity

The general interface is:

```python
from hyper_parallel.components.losses import chunked_cross_entropy

loss_sum = chunked_cross_entropy(
    hidden_states,
    targets,
    lm_head.weight,
    chunk_size=1024,
    ignore_index=-100,
)
```

| Parameter | Shape or type | Meaning |
| --- | --- | --- |
| `hidden_states` | `[B, S_local, H]`, floating | Final hidden states on the current rank |
| `targets` | `[B, S_local]`, `torch.long` | Target tokens already aligned with hidden states |
| `lm_head.weight` | `[V, H]`, floating | Full LM Head weights available for computation in the current root forward |
| `chunk_size` | positive `int` | Maximum local sequence length `C` processed per chunk |
| `ignore_index` | `int` | Target value excluded from cross-entropy |

The interface returns a graph-connected FP32 scalar `loss_sum`. It does not normalize internally by DP, CP, micro-batch count, or token count.

The main output temporary in the ordinary path grows with `B * S_local * V`. Chunk Loss bounds this term to
`B * C * V`, while retaining the `dHidden [B, S_local, H]` required by upstream training and the accumulated
`dWeight [V, H]`:

| Item | Ordinary full logits | Chunk Loss |
| --- | --- | --- |
| Logits extent per computation | `[B, S_local, V]` | `[B, C, V]` |
| Main cross-entropy temporaries | `O(B * S_local * V)` | `O(B * C * V)` |
| Retained before outer backward | Full logits graph and required activations | `dHidden` and accumulated `dWeight` |
| Parameter/FSDP gradients | Required | Still required |

It addresses the sequence-length-dependent temporary peak at the large-vocabulary output stage. It does not eliminate model parameters, optimizer states,
FSDP all-gathered weights, or final parameter gradients.

## 3. Complete Forward and Backward Flow

For each local sequence chunk:

1. Take `hidden_chunk [B, C, H]` and the corresponding `target_chunk [B, C]`.
2. Compute `hidden_chunk @ weight.T`, generating only `[B, C, V]` logits.
3. Convert logits to FP32 and compute cross-entropy with `reduction="sum"`.
4. Use `torch.func.grad_and_value` to obtain the chunk loss, `dHidden_chunk`, and `dWeight_chunk` together.
5. Write `dHidden_chunk` into its interval of the full local `dHidden`, and add `dWeight_chunk` to one
   `[V, H]` accumulator;
6. Release the chunk logits and chunk-level gradients before proceeding to the next chunk.

The custom autograd Function's outer backward does not regenerate logits. It multiplies the saved `dHidden` and
`dWeight` by the upstream scalar, returning them to the decoder graph and LM Head parameter respectively. This supports the Trainer's token
weighting and gradient accumulation without retaining a full logits graph.

The implementation supports first-order gradients and explicitly rejects second-order gradients. The initial version targets full-parameter training, always allocating
`dHidden` and `dWeight` accumulators without branches for a frozen decoder or frozen LM Head. Accumulator dtypes
follow the input and weight dtypes; cross-entropy numerical computation uses FP32.

## 4. Trainer Configuration

Configure the loss explicitly at the `TrainerConfig` root:

```yaml
loss_fn:
  _target_: hyper_parallel.components.losses.ChunkedCausalLMLoss
  chunk_size: 1024
  ignore_index: -100
```

`BaseTrainer._build_loss()` constructs this module after model replacement, sharding, FSDP, loading, and dtype conversion, then invokes
its `bind_model(model, distributed_setup)`. The model adapter registry resolves the Qwen3-MoE provider
automatically; users do not need to patch forward manually in recipes.

For each micro-batch, the Trainer:

1. Receives separate `model_inputs` and `loss_inputs` from the get-batch runtime.
2. Calls `ChunkedCausalLMLoss.prepare_model_inputs()` to read pregenerated `shift_labels` and the loss mask from loss inputs.
3. Removes ordinary `labels` and `shift_labels` from model inputs to prevent Qwen's own eager loss computation.
4. Adds explicit `chunk_loss_*` arguments and calls the model.
5. Receives `loss_sum`, `valid_token_count`, optional MoE auxiliary loss, and `logits=None` from the Qwen adapter.
6. Computes the current rank's valid-token mean in the loss module.
7. Uses the existing `mean_global_loss()` for global token weighting across DP/CP and micro-batches.
8. Returns precomputed gradients to ordinary autograd and FSDP during backward.

Ordinary `ModelOutputLoss` and custom losses implementing only `forward(model_output=..., labels=...)` are unaffected.
`bind_model` and `prepare_model_inputs` are optional lifecycle interfaces used only by model-integrated losses.

The generic `chunked_cross_entropy` primitive allows custom `ignore_index` values. The Trainer's global token counting currently
uses the data layer's shared value `-100`, so `ChunkedCausalLMLoss` rejects other values during binding to keep local loss
and DP/CP token weighting statistics consistent.

## 5. Preshifted Label Contract

The initial version accepts only `shift_labels`. This tensor already represents the next token predicted at each hidden-state position:

```text
hidden[:, 0:S, :] <-> shift_labels[:, 0:S]
```

The implementation retains all local hidden positions, then uses `loss_mask` to replace invalid targets with `ignore_index`. The data path must
perform the next-token shift before CP sharding and then shard inputs and targets together. This preserves identical semantics for CP=1 and CP>1
and avoids losing targets at rank boundaries. Supplying only ordinary `labels` raises an error, preventing a silent
second shift for CP=1 or omitted boundary tokens for CP>1.

## 6. Qwen3-MoE-Specific Behavior

The Qwen adapter activates only when inputs contain `chunk_loss_targets`. Ordinary inference and training without Chunk Loss still call
the original Qwen forward.

The Chunk Loss path:

1. Calls `model.model(...)` to obtain `last_hidden_state`.
2. Validates alignment between final hidden states and preshifted targets, and applies the loss mask.
3. Calls `chunked_cross_entropy` directly with `model.lm_head.weight`.
4. If `output_router_logits=true`, reuses Transformers' load-balancing loss and retains the original
   `router_aux_loss_coef`;
5. Returns `ChunkedCausalLMOutput` with explicit `logits=None`.

The training path requires `use_cache=false`, `return_dict=true`, and `logits_to_keep=0`. The LM Head must currently be a
bias-free `torch.nn.Linear`. Adapter binding is idempotent, and calls without Chunk Loss arguments fall back to the original forward.

## 7. FSDP, CP, EP, and Current Limitations

The initial version supports ordinary DP/FSDP, CP, and EP combinations with TP=1, PP=1, and loss parallel disabled. These unsupported boundaries fail fast before the first
forward:

- TP>1: Vocabulary or hidden-dimension sharding requires distributed softmax/cross-entropy; a local shard cannot be treated as full weights.
- PP>1: Final-stage loss inputs, token normalization, and stage output protocols are not yet integrated.
- Loss parallel: This would duplicate the logits/softmax layout definition of the current full-vocabulary Chunk Loss.

The FSDP root forward covers the Qwen decoder and LM Head. Chunk Loss uses the unsharded
`lm_head.weight` before that forward returns. Its `dWeight` is still passed as the Parameter's gradient to the existing FSDP
reduce-scatter/reshard path, without copying parameters or bypassing FSDP gradient management.

LM Heads with bias, LoRA/DoRA/QLoRA, or custom projection semantics are not yet supported. Graph capture under
`torch.compile(fullgraph=true)` is also not guaranteed.

## 8. Tuning

Smaller `chunk_size` values reduce peak chunk-level logits memory but increase LM Head calls, cross-entropy calls, and Python scheduling overhead. Larger values
are generally faster but require more peak memory. Start at 1024, then tune using vocabulary size, local sequence length, and measurements on the target device.
