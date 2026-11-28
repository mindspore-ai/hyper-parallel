# Encoder parallel execution options

DeepSeek V4.1 can pack image attention and distribute spatial projection within
the existing encoder parallel domain. Extend the `accelerator.encoder_parallel`
section of an encoder-enabled VLM recipe:

```yaml
accelerator:
  encoder_parallel:
    enabled: true
    encoder_cp_size: 2
    packed_attention: true
    overlap: false
    balance_cost: attention
    projector_mode: sharded_dispatch
```

Keep the recipe's encoder adapter and vision attention replacement. These options
do not replace that wiring or change the decoder CP/TP layout. The encoder parent
group is derived from decoder CP × TP; `encoder_cp_size` splits it into Encoder-DP
buckets and Encoder-CP groups. It must divide the parent group size and the
declared attention head counts.

## Attention and work assignment

| Option | Default | Behavior |
| --- | --- | --- |
| `overlap` | `false` | For packed DeepSeek attention, split Q/K/V projection and overlap input Ulysses communication in both forward and backward. Requires `packed_attention: true`. |
| `balance_cost` | `attention` | Keep squared-patch-count LPT by default. `encoder_flops` also includes the model's QKV/output projections and MLP in image assignment. |
| `packed_attention` | `false` | Concatenate images within a bucket, shard the combined patch sequence, and pad only its tail. NPU attention uses one TND call per ViT block with independent image boundaries. |

Packing retains each image's spatial RoPE and bidirectional attention mask. It
does not allow attention between images. Patch embedding, MLP and normalization
already operate on packed local rows; the new mode removes the per-image
attention and communication loop. Each bucket still runs every ViT block, and
checkpointing still replays those blocks.

With `overlap: false`, DeepSeek V4.1 exchanges its equal-shaped Q/K/V together
in one Ulysses all-to-all; output communication remains separate. This reduces
collective launches, not transferred bytes, and also applies to per-image attention.

With `overlap: true`, the projection's existing weights are evaluated in V, K, Q
order. V communication can overlap K projection/RoPE, and K communication can
overlap Q projection/RoPE. Backward submits dQ/dK/dV communication together;
after dQ arrives, Q projection backward can overlap the remaining transfers,
followed by K and V backward. MLA and ViT share the paired launch/wait implementation
in [the common Ulysses scheduler](../context_parallel/ulysses_overlap.py).
Output communication keeps its existing schedule. Each invocation, including
checkpoint replay, owns separate communication state. Parameter names and FSDP
projection boundaries are retained. The paired path supports complete first-order
training backward; partial `autograd.grad` and higher-order gradients are outside
its supported contract.

Splitting one GEMM into three can change BF16 rounding and adds launch costs.
Overlap is optional because its benefit depends on local patch count, group size,
and checkpointing. Enable it after comparing the intended training shape.
`packed_attention` remains independent: false retains per-image sharding/calls;
true shards the concatenated patch sequence and batches isolated images in TND.
At Encoder-CP size one, no communication or extra QKV packing is performed, and
the original combined projection is retained even when overlap is enabled.
Images retain squared-patch-count LPT assignment unless `balance_cost: encoder_flops`
is explicitly selected. For a DeepSeek V4.1 ViT block, the optional estimate is
`4 * D * T^2 + (8 * D^2 + 6 * D * F) * T`, where `T` is image patch count, `D` is
hidden width, and `F` is SwiGLU intermediate width. Costs sum over the vision
blocks and divide by the common Encoder-CP degree. The coefficients come from
matrix FLOP counts, not user-supplied timing weights. The bridge provides these
estimates; the generic runtime keeps the same LPT scheduler.

This option changes image-to-bucket assignment, not image boundaries or model
math. It does not model communication, layout copies, patch embedding, or the
aligner, and is not a calibrated runtime predictor. There is no load-balancing
opportunity at Encoder-DP size one. Similar image sizes often produce the same
assignment; heterogeneous sizes can reduce peak patch volume, but throughput
improvement must be measured. Changing batch shapes can change BF16 rounding.
Both cost policies leave the attention overlap setting independent.

## Spatial projection and decoder routing

| `projector_mode` | ViT feature transfer | Aligner computation |
| --- | --- | --- |
| `leader` (default) | Gather complete image features within Encoder-CP. | Rank zero projects real images; other ranks use graph-connected dummy calls to preserve FSDP call ordering. |
| `sharded_dispatch` | Keep local patches and send only required remote patches to their spatial-block owners with all-to-all-v. | Same unique output-row ownership, without gathering the complete feature bank. |

For V4.1's 3 × 3 spatial merge, each aligner output row consumes nine patches.
The adapter assembles these patches in the original channel-major order,
including spatial padding, and calls the unchanged projector weights and
activation through the module's forward boundary. Empty owners retain one dummy
call so FSDP and backward collectives remain aligned.

`sharded_dispatch` routes projected rows directly to the decoder token intervals that
consume them. Image start/end/newline markers remain model-owned. When decoder TP
without sequence parallelism requires replicated embeddings, the relevant
projected rows are delivered to each TP replica; the aligner projection itself
is not repeated for those replicas.

The model adapter caches up to 64 CPU spatial-index tables keyed by image grid
and merge ratio. It does not cache activations or batch-dependent rank routes.
Packing, index placement and output routing still require memory operations.

[The projector dispatcher](projector_dispatch.py) builds spatial-block ownership
within Encoder-CP and routes projected rows within the decoder CP-times-TP parent.
Both exchanges reuse the common differentiable variable-split all-to-all helper;
this does not by itself overlap the two dependent transfers.

## Compatibility

The default attention partition and leader projection remain unchanged. Sharded
projection requires the corresponding model bridge methods; unsupported adapters
fail during runtime construction. The existing training restrictions, including supported
micro-batch, loss layout and context-preserving vision checkpointing, still apply.

CPU/GPU packed attention uses the existing per-image attention path for numerical
reference. The single-call TND optimization described above is the NPU path.
