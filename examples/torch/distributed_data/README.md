# Distributed local-batch loading

The primary interface wraps an existing rank-local DataLoader. The user's
DataLoader continues to own sampling, workers, preprocessing, packing, image
loading, and collation. Only one data-owner rank in each model-parallel group
is iterated by HyperParallel; other ranks may pass `None`.

```python
def metadata_fn(local_batch, local_batch_id):
    return LocalBatchMeta(
        local_batch_id=local_batch_id,
        text_tokens=int(local_batch["attention_mask"].sum()),
    )


loader = build_distributed_dataset(
    local_data_loader if is_data_owner else None,
    mesh,
    DistributedDatasetConfig(micro_batch_num=4, dp_dim_names=("dp",)),
    metadata_fn=metadata_fn,
    communication_device=device,
)
```

Each data owner reads four consecutive DataLoader results for this step. The
framework gathers their metadata, balances the eight opaque local batches for
DP=2, exchanges reassigned batches between the two owners, and distributes each
result to the corresponding model-parallel ranks. It never applies another
sampler or DP stride to the user's DataLoader. Omitting `metadata_fn` selects
uniform-cost planning.

## Sidecar-planned Hugging Face example

This example uses image-caption records downloaded from
`diffusers/pokemon-gpt4-captions`. With eight ranks, the default configuration
builds a `(dp=2, mp=4)` mesh and exercises sidecar-first local-batch loading:

```text
rank 0 and rank 4 are the two data owners
  -> each owner reads lightweight LocalBatchMeta entries only
  -> metadata group {0, 4} plans DP × micro_batch_num local batches
  -> each target owner fetches its planned local_batch_id from shared storage
  -> the existing single-card callback reads and decodes the JPEG
  -> MP groups {0, 1, 2, 3} and {4, 5, 6, 7} receive identical local batches
  -> all ranks consume the step and commit the same plan ID
```

Each manifest row stands in for one complete local batch in this small example.
The distributed layer does not know that it contains an image and caption. It
only sees `LocalBatchMeta` and an opaque payload returned by
`fetch_local_batch()`. A production source can use its existing DataLoader,
packing, multimodal processing, and collation pipeline inside that callback.

No JPEG is read until the whole-step plan is ready. Because every data owner can
access the shared files by `local_batch_id`, this path does not perform payload
A2A between DP owners.

```bash
torchrun --nproc_per_node=8 \
  examples/torch/distributed_data/online_huggingface_multimodal.py \
  --manifest /tmp/hp-hf-mm-validation.WXFrqm/pokemon_gpt4_64/manifest.json \
  --backend hccl \
  --model-parallel-size 4 \
  --double-buffer
```

For a CPU smoke test, use `--backend gloo`. Use
`--model-parallel-size 1` for a pure-DP layout. With DP=2 and
`micro_batch_num=4`, one optimizer step plans eight local batches: four for each
data rank. The script validates DP coverage, plan agreement, MP replication,
target-rank direct reads, and committed local-batch offsets.

The existing global-map `OnlineLocalBatchSource` remains available when every
data owner can index the same complete local-batch source. In that mode,
HyperParallel owns the DP stride. Use the direct DataLoader interface above
when the user's sampler has already sharded data by DP rank.
