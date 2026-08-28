# Sidecar-planned Hugging Face multimodal distributed data example

This example uses 64 image-caption records downloaded from
`diffusers/pokemon-gpt4-captions`. With eight ranks, the default configuration
builds a ``(dp=2, mp=4)`` mesh and exercises the complete local sidecar path:

```text
rank 0 and rank 4 are the two data owners
  -> all ranks load the same lightweight SampleMeta sidecar
  -> metadata group {0, 4} All-Gathers costs and plans the whole optimizer step
  -> owner DataLoader workers fetch each planned packed unit and its JPEG payload
  -> collate_fn decodes only the JPEGs assigned to each data owner
  -> MP groups {0, 1, 2, 3} and {4, 5, 6, 7} distribute final microbatches
  -> all ranks consume the step and commit the same plan ID
```

The local manifest is map-style: each JSON row contains the caption, JPEG path,
dimensions, JPEG byte count, and checksum. Its lightweight fields are converted
to the explicit ``metadata=`` sequence before dataset iteration. No JPEG is read
until the whole-step plan is ready, and data owners then read their planned
sample IDs directly from shared storage. This sidecar path does not perform raw
sample A2A between data owners. Each manifest row stands in for one offline
packed local batch whose image is materialized at runtime, so the configuration
uses the default of one dataset unit per microbatch and does not set
``raw_sample_size``.

With four microbatches of one packed unit per data rank, one optimizer step
uses eight unique records. Every MP peer receives the same four records as the
data owner in its group:

```bash
torchrun --nproc_per_node=8 \
  examples/torch/distributed_data/online_huggingface_multimodal.py \
  --manifest /tmp/hp-hf-mm-validation.WXFrqm/pokemon_gpt4_64/manifest.json \
  --backend hccl \
  --model-parallel-size 4 \
  --num-workers 2 \
  --double-buffer
```

For a CPU smoke test, use `--backend gloo`. Use
``--model-parallel-size 1`` to recover the original pure-DP layout. The script
checks that all unique samples are present exactly once across DP ranks,
DataLoader workers performed the reads, every rank agreed on the plans, the
greedy balance reassigned samples across data ranks, and all ranks within an MP
group received identical microbatches.
