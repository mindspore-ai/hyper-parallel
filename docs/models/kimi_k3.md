# Kimi K3 native training integration

The model package implements the K3 text training architecture under the released
`KimiLinearForCausalLM` and `kimi_linear` identities. Configuration loading can
extract `text_config` from a local composite K3 configuration. The model is
executable without replacement rules or optional fused-kernel packages.

## Architecture

- KDA and gated NoPE MLA follow the configured one-based layer schedule.
- SiTU uses the configured gate and up saturation scales with FP32 intermediates.
- Latent MoE routes from the full hidden state, dispatches to latent experts,
  combines expert values before normalization/up projection, and adds shared experts.
- Quantile Balancing updates selection-only correction bias once per optimizer
  step. Histogram state and correction bias retain their declared integer/FP32 dtypes.
- Block attention residuals maintain per-forward, differentiable history and the
  configured layer block boundaries. History can use stacked tensors or the optional blocks representation.

The original parameter names are preserved. Initialization explicitly covers KDA
decay values, router correction/statistics, RMSNorm scales, and AttnRes projections.
The historical inference implementation is not imported or executed by training.

## Construction and ownership

`adapter/construction.py` loads local configuration and delegates to
`HyperAutoModelForCausalLM.from_config`. Registration, loss binding, FSDP units,
parameter roles, layer recomputation and validation remain model-owned providers.
The existing `kimi_linear` adapter directory forwards to the K3 text provider,
so cold discovery and explicit imports share one specification without central
registry special cases.

The original launcher defaults to DP/FSDP with TP=CP=EP=PP=1. The explicit
parallel recipe composes FSDP, CP and optional EP while retaining TP=PP=1.
Native K3 normalization, padding and routing semantics remain in K3 adapters;
communication and parameter ownership use shared framework components.

## Attention, experts and packed data

KDA can use eager reference operations or the optional shared FLA backend. The
CP adapter selects Ulysses, state CP or a hybrid schedule explicitly. Packed
inputs use global document boundaries and the shared TND metadata; a rank split
does not create a new document. State CP retains differentiable recurrent-state
exchange and convolution halos. Nonpacked fused state CP requires each local
sequence length to be a nonempty multiple of 64. The model selects short convolution
independently with `causal_conv_backend: eager`, `triton`, or `ascendc`, for dense
and packed inputs. Kernel replacement does not remove the separate rounding of
halo input-gradient contributions. The underlying local fused call does not provide a
standalone final-state-gradient API; the shared CP schedule owns that derivative.

Gated NoPE MLA uses the shared MLA module and CP runtime. The optional forward and
backward overlap schedules remain explicit. A BS validity mask is handled at the
K3 boundary; it is not passed to the NPU attention operator as a pairwise mask.
Grouped experts, standard EP and planned dispatch/replay are selectable. Keep the
same expert replacement at EP1 and EPn so parameters and optimizer ownership can
be compared. Quantile Balancing updates once after the optimizer step and excludes
padding, while recomputation must not collect the same routing observations twice.

Direct HF labels are unshifted. Indexed or pre-shifted targets are already aligned
and must not be shifted again. Packed document ends, visual targets and padding
are excluded before the complete-step token denominator is computed. Position IDs
reset at document boundaries. Cache-based decoding is outside this training entry.

SiTU and block attention residuals call public functional APIs. Their default
backend remains eager; `ascendc` requires the corresponding installed operator
package and explicit model validation. Importing the portable model does not
require FLA or the optional fused operator packages.

## Memory and validation scope

The model-owned provider selects decoder replay with `recompute_scope` in the
local model configuration (`text_config` for the composite model):

| Scope | Replayed inside each selected decoder | Original graph retained |
| --- | --- | --- |
| `layer` | Attention, FFN, both AttnRes calls and norms | None of these branches |
| `ffn` | FFN, both AttnRes calls and norms | Attention |
| `attention` | Attention, both AttnRes calls and norms | FFN |
| `attn_res` | Both AttnRes calls, norms and layer glue | Attention and FFN |

Select these regions with the framework checkpoint configuration:

```yaml
activation_checkpoint:
  mode: full
  selection:
    source: model_adapter_safe_regions
    layer_count: 93
```

Here `full` describes the enclosing checkpoint region; model-owned branch
exclusions provide the finer granularity. This is separate from generic
operator-level selective checkpointing. Branch exclusions currently require eager
checkpointing. With `vision_config.recompute: true`, visual blocks precede decoder
regions: selecting all 27 visual and 93 decoder blocks requires `layer_count: 120`.
The final output AttnRes is outside the decoder regions and is not replayed by this
selection.

AttnRes history is explicit, differentiable, per-forward state passed into each
checkpoint; it is not a mutable global cache. `attn_res_history_layout: stacked`
is the default. The optional `blocks` representation defers stacking until each
consumer and can change BF16 gradient accumulation order. It requires a separate
numerical comparison even when forward losses agree. Keeping blocks without
recomputation does not remove every consumer's saved stack.

Shared activation swap retains input storages reused by other layers, including
masks, vision position data and aliases. Set `swap_inputs: true` under
`activation_checkpoint` only with an explicitly validated combination. Checkpointing
does not free a history value that later consumers or autograd still require.
FSDP `reshard_after_forward: false` keeps gathered parameters until backward,
trading memory for fewer parameter all-gathers; it does not retain an extra forward
autograd graph. Communication fusion remains a separate FSDP option.

Structural-depth experiments retain all 93 decoder roles and use reduced widths
and expert sizes. They are not released-width capacity or convergence results.
Cross-topology checkpoint replay with a mapped global data interval is distinct
from same-topology restoration of optimizer, scheduler, data cursor and RNG.
Assess absolute/RMS error, full-gradient direction and actual parameter updates
alongside loss; a relative ratio for a tiny parameter is not a standalone verdict.
Published inference-weight compatibility and real-corpus stability still require
separate qualification. The available validation lifecycle is described below.

See [the LLM example](../../examples/kimi_k3/README.llm.md) and
[validation contracts](kimi_k3_validation.md). Architecture references are the
[released source](https://huggingface.co/moonshotai/Kimi-K3) and
[K3 technical report](https://arxiv.org/abs/2607.24653).

## Periodic validation and continuation

Set `training.eval_steps` or `training.eval_epochs` to enable periodic validation,
and set a positive `training.eval_iters`. One evaluation iteration uses the same
microbatch count as one global training batch. Supply an independent validation
split with enough batches for the complete window; the default recipes leave
periodic validation disabled.

The shared Trainer evaluates without autograd or cache generation. It weights
objectives by actual valid targets across the complete window and DP/CP owners,
rather than averaging unequal microbatch means. It restores module modes, the
stateful validation loader cursor, and Python/NumPy/CPU/local-device RNG. It leaves
optimizer, scheduler, QB observations and the training iterator untouched. Gathered
FSDP parameters are resharded after validation because there is no backward hook
for the keep-forward policy. Repeated evaluations revisit the saved validation
window.

This entry supports TP=PP=1 and stateless full-sequence KDA CP. Concurrent host
background prefetch is rejected until its RNG lifecycle can be isolated. Worker
augmentation and custom stateful transforms need their own continuation checks.
NumPy's global RNG is now saved in checkpoint metadata using weights-only-loadable
values. Older checkpoints remain readable, but a missing NumPy state cannot
reconstruct historical random data. Independent `numpy.random.Generator` objects
must be persisted by their owning dataset.

Same-topology training and validation checks do not establish skipped-update or
AMP GradScaler semantics: custom optimizer skip signals must be coordinated with
QB, scheduler and step counters before that combination is qualified.

## Image and video composition

`KimiK3ForConditionalGeneration` uses the composite `kimi_k3` identity. It owns
MoonViT, the patch-merger projector, expanded-media embedding scatter and the
native language model. Vision remains model-owned, following the existing native
model-family organization. Parameter layouts retain the released checkpoint
names; this does not establish published-weight loading compatibility.

The image entry builds local image/video/text JSONL through the standard Omni
dataset, K3 transform/collator, `OmniParallelBatch` and `VLMTrainer`. Pre-rendered
token prompts and processor-supported conversation records have distinct contracts.
Raw video sampling retains frame provenance; this does not implement an official
video conversation template.

The encoder bridge uses shared CPU patch transport and `EncoderLocalInputs`.
Patches are partitioned before device transfer. Real and dummy slots visit the
same visual FSDP units; the optional `project_many` interface batches their
projector call to preserve one FSDP forward/backward lifecycle. Whole-step target
counts are prepared on CPU, while device payloads remain streamed per microbatch.

The collator supports independent rows and document-isolated packed inputs with
explicit media spans. Decoder CP and encoder CP have separate ownership. The
parallel recipe lists the exact MoonViT block replacements so descendants are not
replaced accidentally. Enabling `vision_config.recompute` adds visual blocks to
the model's recomputation regions; select a matching region count in the Trainer.

Image, pre-sampled video, raw video, empty visual slots, CP/ECP, EP, recomputation
and checkpoint restoration each require their own composition evidence. Fused
vision operators, swap combinations, official video message rendering, large
capacity and real-data convergence are not implied by a successful small run.
See [the image/video example](../../examples/kimi_k3/README.image.md).
