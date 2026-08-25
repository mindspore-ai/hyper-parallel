# Online Hugging Face multimodal data example

This example uses 64 image-caption records downloaded from
`diffusers/pokemon-gpt4-captions`. It demonstrates the default online metadata
path:

```text
PyTorch DataLoader workers read local JPEG bytes and captions
  -> metadata_fn derives text/vision/I/O costs
  -> metadata All-Gather plans one global microbatch
  -> packed-byte tensor A2A moves raw samples to planned DP ranks
  -> collate_fn decodes and validates only the received JPEGs
```

The local manifest is map-style: each JSON row contains the caption, JPEG path,
dimensions, and checksum. JPEG bytes are not stored in metadata and are not read
a second time after planning.

On the local 8-card machine, the downloaded 64-record dataset runs as one step
with four microbatches of two raw samples per rank:

```bash
torchrun --nproc_per_node=8 \
  examples/torch/distributed_data/online_huggingface_multimodal.py \
  --manifest /tmp/hp-hf-mm-validation.WXFrqm/pokemon_gpt4_64/manifest.json \
  --backend hccl \
  --num-workers 2 \
  --double-buffer
```

For a CPU smoke test, use `--backend gloo`. The script checks that all samples
are present exactly once, DataLoader workers performed the reads, every rank
agreed on the plans, and the greedy balance caused actual cross-owner A2A
movement.
