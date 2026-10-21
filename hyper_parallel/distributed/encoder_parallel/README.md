# Experimental visual encoder parallelism

This selectively ports the model-neutral runtime from
[upstream PR #1132](https://github.com/mindspore-ai/hyper-parallel/pull/1132).
It does not include DeepSeek-V4.1, Engram, or the upstream data/trainer rewrite.
The existing text and Omni default configurations remain disabled.

## Execution

1. Derive an encoder parent from decoder `CP × TP`, preserving rank order.
2. Reshape it into `Encoder-DP × Encoder-CP`. Assign whole images to DP
   buckets by descending squared patch count (LPT load balancing).
3. Partition each image's patch sequence across Encoder-CP ranks. Visual
   attention exchanges sequence/head axes with Ulysses and masks padding.
   Different images never attend to one another.
4. Gather each image's hidden states, run the original spatial merger on
   its producer, and route image features to decoder owners with
   differentiable collectives. Reassemble features in original image order.
5. Dummy vision/projector work and zero-valued autograd anchors keep
   participation consistent for FSDP ranks with no local image work.

Only explicitly enabled visual subtrees are excluded from decoder TP
boundary inference. Their parameters remain declared as replicated in TP
and can still be managed by FSDP. Decoder layers retain their normal plan.

## Qwen3.6 adapter boundary

The Qwen3.5-MoE conditional-generation adapter uses native patch embedding,
HF vision position helpers, vision blocks, and the spatial merger. Only
the visual attention implementation and image feature provider change.
The text-only adapter does not register this capability.

Initial supported inputs/modes:

- Images with `T=1`, positive grids divisible by `spatial_merge_size`.
- One unpacked sample per microbatch, including multiple images or no image.
- Decoder `CP=1`, `sequence_parallel=false`, `loss_parallel=false`, `PP=1`.
- Encoder-CP divides decoder TP and all visual Q/K/V head counts.
- FSDP may wrap visual child modules and the outer model, but must not
  wrap `model.visual` or `model` as a separate unit: the bridge executes
  visual children before the native backbone forward.
- Vision activation checkpointing, compilation, placement validation,
  videos, and multi-sample packing are rejected explicitly.
- Requires the HF Qwen3.5-MoE vision implementation exposing
  `transformers.vision_utils` (the CPU oracle uses Transformers 5.13).

This is not a claim of full-model NPU/FSDP training qualification. In
particular, decoder CP/SP image insertion requires a separate adapter.
Do not enable this configuration for an existing long run without a
full-model baseline comparison.

## Configuration and tests

See `examples/qwen3_6_moe/train_online_vlm_encoder_parallel.yaml` for an
independent eight-rank example. It does not replace `train_online_vlm.yaml`.

```yaml
accelerator:
  tp_size: 2
  cp_size: 1
  sequence_parallel: false
  encoder_parallel:
    enabled: true
    encoder_cp_size: 1  # Encoder-DP=2; use 2 for Encoder-CP=2 instead
```

The example also declares the visual attention replacement. Enabling the
runtime without that replacement is an error, not an implicit fallback.

```bash
python -m pytest -q tests/ut/distributed/encoder_parallel
python -m pytest -q tests/torch/encoder_parallel/test_qwen_encoder_distributed.py
```

The second test uses two CPU/Gloo processes and the actual tiny HF vision
model; it compares image features and all parameter gradients for
Encoder-DP=2 and Encoder-CP=2, including an empty producer bucket. It does
not initialize an NPU or load the full language model.
