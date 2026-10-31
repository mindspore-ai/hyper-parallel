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
