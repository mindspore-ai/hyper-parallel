# Distributed local-batch loading

The public interface wraps an existing rank-local DataLoader. The user's
DataLoader continues to own sampling, workers, preprocessing, packing, image
loading, and collation. Only one data-owner rank in each model-parallel group
is iterated by HyperParallel; other ranks may pass `None`.

```python
def metadata_fn(local_batch, local_batch_id):
    return LocalBatchMeta(
        local_batch_id=local_batch_id,
        text_tokens=int(local_batch["attention_mask"].sum()),
    )


loader = build_distributed_dataloader(
    local_data_loader if is_data_owner else None,
    mesh,
    DistributedDatasetConfig(micro_batch_num=4, dp_dim_names=("dp",)),
    metadata_fn=metadata_fn,
    communication_device=device,
)
```

For `DP=2` and `micro_batch_num=4`, each data owner reads four consecutive
DataLoader results. HyperParallel gathers metadata for those eight opaque
`local_batch` objects, balances them across the two owners, exchanges reassigned
batches, and delivers each result to the corresponding model-parallel ranks.
It never calls `dataset.__getitem__` itself and never applies another sampler,
DP stride, packing function, or collate function. Omitting `metadata_fn`
selects uniform-cost planning.

## Hugging Face DataLoader example

This example uses image-caption records downloaded from
`diffusers/pokemon-gpt4-captions`. With eight ranks, the default configuration
builds a `(dp=2, mp=4)` mesh and wraps the existing PyTorch DataLoaders:

```text
rank 0 and rank 4 are the two data owners
  -> each owner iterates its own DP-sharded DataLoader
  -> Dataset reads JPEG bytes and DataLoader workers run decode_and_collate
  -> metadata_fn observes each complete collated local_batch
  -> metadata group {0, 4} plans DP × micro_batch_num local batches
  -> owners exchange reassigned opaque local_batch payloads
  -> MP groups {0, 1, 2, 3} and {4, 5, 6, 7} receive identical local batches
  -> all ranks consume the step and commit the same plan ID
```

Each DataLoader iteration contains one record only to keep the example easy to
inspect. A production DataLoader may dynamically pack many raw samples into one
`local_batch`, or may yield an offline-packed sample that is already a complete
batch. HyperParallel treats both cases identically and never splits that output.

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
`--model-parallel-size 1` for a pure-DP layout. With DP=2 and
`micro_batch_num=4`, one optimizer step plans eight local batches: four for each
data rank. The script validates DP coverage, plan agreement, MP replication,
DP payload redistribution, and committed local-batch offsets.

The distributed-data implementation uses `torch` and `torch.distributed`
directly. This example requires no framework-selection environment variable.

This wrapper derives metadata after each DataLoader result is ready. It does not
accept a sidecar because sidecar metadata alone cannot make an arbitrary
DataLoader fetch planner-selected IDs. That optimization requires a separate
plan-aware sampler/DataLoader contract and is outside this API.
