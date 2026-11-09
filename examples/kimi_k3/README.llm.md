# Native Kimi K3 LLM pretraining

This example builds the text architecture from a local configuration and starts
random-initialized pretraining through the standard `TextTrainer`. It does not
load the released inference checkpoint. The released text identity
`KimiLinearForCausalLM` / `kimi_linear` is preserved.

## Inputs

Provide a local text `config.json` with `model_type: kimi_linear`, or a composite
K3 configuration containing `text_config`. Model dimensions and the one-based
KDA/MLA layer schedule are used as written; the launcher never crops a model.
The launcher reads explicit `vocab_size` and integer `eos_token_id` fields.
Inference quantization metadata requires the explicit
`--model.strip_inference_quantization=true` option for floating-point scratch
construction. This option does not load, convert, or dequantize weights.

The indexed data prefix must have nonempty `.bin` and `.idx` files. Each physical
row is one continuous causal attention sequence. The indexed reader can
concatenate source documents with EOS tokens inside that row; attention and
positions do not reset at EOS. The baseline recipe does not reset at EOS. For document-isolated packing, use
`train_llm_parallel_pretrain.yaml` with `reset_position_ids: true` and the packed
runtime adapter; explicit document boundaries drive the shared TND path.
The indexed parallel recipe also enables `eod_mask_loss` so an EOS input does
not supervise the first token of the next document.
Targets are already shifted by the indexed reader; the chunked loss must not
shift them again. If a pad ID is configured on the runtime adapter, padded input
and target positions must be masked consistently; ignored targets alone do not
exclude a real token from QB statistics.

## Run

Install this checkout in the intended environment before launching:

```bash
pip install -e .
bash examples/kimi_k3/run_llm_pretrain.sh /local/config.json /local/text_document
```

The default launch is four ranks, DP4/FSDP4, one microbatch per rank and a global
batch of four. TP, CP, EP and PP are one. `NPROC_PER_NODE` changes the FSDP group
and global batch together; the launcher requires `CP_SIZE=1`. Sequence/loss
parallelism, encoder parallelism and compilation are disabled. Paths must exist
locally; the launcher enables offline mode and does not download assets.

The recipe uses FP32 initialization, FP32 main parameters/gradients and reduction,
BF16 FSDP compute, and no automatic forward-input cast. The baseline defaults to
`eager` backends, stacked AttnRes history and disabled recomputation. The
model declares complete decoder layers as the supported recomputation regions.
Optional FLA, CANN fused-op, vision and model CP packages are unnecessary for the
baseline model import or direct CPU forward/backward.

The standard MoE step callback must finalize QB after each optimizer step; it
pools statistics over token-owning data ranks and updates correction bias once.
Backward recomputation must not accumulate the same observations again.

## Checkpoints

Enable the standard checkpoint callback for periodic model, optimizer and
rank-local training-state saves:

```bash
bash examples/kimi_k3/run_llm_pretrain.sh /local/config.json /local/text_document \
  --checkpoint.save_ckpt=true --checkpoint.save_steps=10 --checkpoint.save_epochs=0 \
  --checkpoint.save_optimizer=true --checkpoint.save_train_state=true \
  --checkpoint.save_extra_state_per_rank=true \
  --checkpoint.checkpoint_dir=/local/k3-checkpoints
```

Restore the latest checkpoint from that directory, retaining the same topology
and dataset configuration:

```bash
bash examples/kimi_k3/run_llm_pretrain.sh /local/config.json /local/text_document \
  --training.train_iters=20 --checkpoint.checkpoint_dir=/local/k3-checkpoints \
  --checkpoint.restore_from=LATEST --checkpoint.restore_optimizer=true \
  --checkpoint.restore_train_state=true --checkpoint.restore_dataloader_state=true
```

`save_ckpt` controls subsequent writes independently of restore. These are the
standard framework entry points; production-scale continuation and topology
changes require separate acceptance.

## Validation scope

Use [the validation guide](../../docs/models/kimi_k3_validation.md) for module
parity and architecture checks. A small architecture is suitable for a functional
smoke; it is not evidence of released-width capacity or long-run convergence.
The default recipe does not claim packed data, new CP/EP combinations, full
multimodal training, published MXFP4 weight loading, or production checkpoint
continuation. These require their own integrations and acceptance evidence.

## Explicit parallel recipe

[train_llm_parallel_pretrain.yaml](train_llm_parallel_pretrain.yaml) uses the
standard entry with four ranks, FSDP, CP2 and EP2. Configure local model and data
paths, tokenizer IDs, and model kernel backends before running it:

```bash
python -m torch.distributed.run --standalone --nproc_per_node=4 \
  --module scripts.train_lm examples/kimi_k3/train_llm_parallel_pretrain.yaml
```

For the NPU recipe, set model `kda_backend: triton`, `mla_backend: npu_fusion`,
`moe_backend: grouped` and retain `causal_conv_backend: eager` for packed inputs.
SiTU and AttnRes can remain eager independently. The exact MLA layer list in the
recipe follows the published 93-layer schedule; change it with any custom layer
schedule. Set CP/EP and FSDP group sizes together when changing the topology.
The optional kernels require a compatible installed CANN/FLA/Triton toolchain.
See [architecture and validation scope](../../docs/models/kimi_k3.md).

The local KDA adapter also accepts `kda_backend: ascendc`. Its native kernel owns
Q/K normalization; the Triton adapter retains the model's reciprocal rounding.
Both state CP and Ulysses select convolution independently through
`causal_conv_backend: eager`, `triton`, or `ascendc`; packed inputs use the same
explicit selection. AscendC requires the optional `fla_npu` package; Triton requires
the compatible FLA package. Qualify the complete training recipe when changing
either backend. These selections never silently fall back.

The mixed K3 CP wrapper uses `mla_overlap` for the shared forward and backward
schedule. The former `mla_backward_overlap` option is removed. Keep overlap off
for partial `autograd.grad` or higher-order differentiation. The standard K3 EP
path explicitly retains generic permutation/combine for a stable baseline;
new shared fused defaults do not implicitly change this model's recipe.

### Random-worker continuation and rejected updates

Use `dataloader.checkpoint_worker_rng: true` to checkpoint map-style worker CPU
randomness. Dynamic packing then saves full augmented buffer samples. Keep the
worker and DP topology unchanged; private dataset generators require their own
state protocol. See the [data guide](../../hyper_parallel/data/README.md).

The optional `training.skip_nonfinite_gradients: true` policy rejects nonfinite
pre-clipping norms across all ranks before updating parameters. QB observations
are discarded and optimizer/LR state is preserved. Checkpoints distinguish
consumed batches (`global_step`) from applied updates (`optimizer_step`). This is
a BF16 nonfinite-gradient policy, not a custom optimizer or AMP scaler protocol.
See the [validation contract](../../docs/models/kimi_k3_validation.md).

## Optional QB and expert memory settings

The local model JSON accepts these settings. Defaults keep int64 statistics,
separate gate/up weights, and no expert-local recomputation:

```json
{
  "quantile_histogram_dtype": "int32",
  "quantile_max_tokens_per_rank": 262144,
  "quantile_token_chunk_size": 4096,
  "quantile_expert_chunk_size": 128,
  "quantile_reduce_bucket_size": 8,
  "moe_backend": "grouped",
  "expert_fc1_fused": true,
  "expert_recompute": "fc1_activation",
  "ep_async_counts": true,
  "ep_replay_policy": "full"
}
```

Set the QB row bound from the **sum of local sequence rows across every
microbatch in an optimizer step**, including padding. A too-small bound raises
before the histogram overflows. Global reduction promotes to int64 if the
configured bound times the unique-token group sizes exceeds int32. The callback
batches compatible histograms, derives token counts from a histogram row, and
commits bias on device. Empty, skipped, evaluation and recomputed observations
retain their existing rules. Completed-step checkpoints keep their prior schema;
int32 midstep rank-local checkpoints additionally preserve the observed row bound.

QB reuses the router's selected expert indices when computing its histogram
cutoff. Masking those entries and taking the remaining maximum preserves the
original top-k order and tie decisions. Token and expert chunk sizes bound
histogram temporaries independently; smaller chunks add launches, so measure
both time and memory. Standalone `QuantileBalancer.accumulate` calls without
selected indices retain the original cutoff calculation.

`expert_fc1_fused` and `expert_recompute` apply to the packed experts installed by
`replace_kimi_k3_ep_moe`. Combined FC1 stores `[expert, gate_then_up, input]` as
`experts.w13`, so it executes one gate/up grouped GEMM without concatenating
weights in each forward. The model-owned HF converter imports and exports the
original numbered w1/w2/w3 names. An existing **training-state** checkpoint with
separate w1/w3 is not automatically a w13 optimizer checkpoint; keep the storage
layout fixed for DCP resume or explicitly migrate both parameters and optimizer
state. BF16 input-gradient accumulation differs between one and two FC1 GEMMs.

`expert_recompute` supports `none`, `activation` and `fc1_activation` for grouped
experts. These scopes exclude the router, QB, shared branch and token collectives.
For planned EP, `replay_routed: true` on the replacement rule retains the route
plan and replays the routed branch. With `ep_replay_policy: gemm_outputs`, Torch
selective checkpointing saves matrix outputs and replays dispatch, permutation
and elementwise work instead of repeating forward GEMMs. This retains more
activations than full replay; it does not overlap dispatch replay with GEMM
backward or eliminate the combine gradient's expert-output dependency.

`ep_async_counts` applies to planned EP. It starts pinned-host count transfer on
a side stream before local sorting and waits on its event at the c10d split-list
consumer. It reduces the exposed dependency window; ragged c10d still needs
host split sizes. EP1 derives counts from tensor shapes and does not copy them
to the host. Select these options independently when qualifying a workload.

With `ep_backend: planned`, `overlap_shared_expert: true` on the replacement rule
runs the shared expert on a separate device compute stream. The producer orders
that stream before launch and the final addition waits for it. The shared branch
stays outside routed replay, including `gemm_outputs`; CPU execution remains
synchronous. This schedules independent shared work alongside routed counts and
payload work. It does not supply a device-resident dispatcher or tile-level
communication/GEMM overlap in backward. Qualify the combined FSDP/CP/EP recipe
and measure overlap on the target shapes before claiming a throughput benefit.

## K3 per-head Muon

Replace the optimizer section in the chosen recipe, retaining the same model
configuration path and the ordinary Trainer checkpoint/scheduler setup:

```yaml
optimizer:
  _target_: hyper_parallel.components.optim.Muon
  fp32_main_params: true
  extra_adamw_name_keywords:
    - gate.weight
    - res_proj
    - q_conv1d
    - k_conv1d
    - v_conv1d
  muon_config:
    lr: 1.0e-5
    weight_decay: 0.1
    momentum: 0.95
    nesterov: true
    ns_steps: 5
    ns_variant: asym5
    matched_adamw_rms: 0.2
    reshape_fn:
      _target_: hyper_parallel.models.kimi_k3.adapter.optim.headwise_muon.build_reshape_fn_from_config
      config_path: /path/to/k3-training-config.json
      vision_per_head: false
  adamw_config:
    adamw_lr: 1.0e-5
    adamw_weight_decay: 0.1
    adamw_betas: [0.9, 0.95]
    adamw_eps: 1.0e-8
```

The learning rate is an integration example, not the published training recipe.
KDA Q/K/V and MLA expanded Q/K/V momentum matrices are orthogonalized per head.
MLA's interleaved K/V storage is split into disjoint views. Low-rank compressed
KV, output and gate projections retain ordinary matrix Muon. Packed experts
remain independent matrices; combined FC1 also separates gate/up logical
matrices. Norms, small router/residual projections and embeddings/LM heads use
the existing AdamW fallback. `vision_per_head: true` additionally splits the
MoonViT Q/K/V heads. This does not implement the report's P2P Muon transport.
