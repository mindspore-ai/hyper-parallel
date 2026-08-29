# Distributed dynamic packing

This package implements a synchronous PyTorch data path inspired by the
Source Loader, Planner, and Data Constructor role separation in MegaScale-Data:

```text
mapping Dataset
  -> Source Loader: materialize raw samples and derive lightweight metadata
  -> Planner: assign individual samples to DP ranks and sequence bins
  -> CPU all-to-all: route only selected raw payloads
  -> Data Constructor: pack each sequence bin, then collate the local batch
  -> CPU broadcast: share the constructed batch with model-parallel peers
```

Dynamic sequence packing is an extension built on that role separation. The
paper describes sample-level scheduling and constructor-side microbatch
assembly, but it does not prescribe the online token-budget algorithm used
here.

## Batch semantics

- `seq_len` is the hard token capacity of one normal packed sequence.
- `local_batch_size` is the number of packed sequences one DP constructor
  returns from each iterator step.
- One synchronized iterator step therefore constructs
  `local_batch_size * dp_size` packed sequences across DP.
- Optimizer global batch size and gradient accumulation remain Trainer
  concerns. There is no `raw_sample_size` or `micro_batch_num` input.

The metadata callback and packer must use the same token accounting, including
special tokens and reserved multimodal placeholders. Oversized samples fail by
default; `oversized_policy="single"` explicitly allows an overflow sample to
occupy a bin alone.

## Public API

```python
from hyper_parallel.distributed_data import (
    DistributedDatasetConfig,
    SampleMetadata,
    build_distributed_dataloader,
)

loader = build_distributed_dataloader(
    dataset,
    mesh,
    DistributedDatasetConfig(seq_len=32768, local_batch_size=2),
    metadata_fn=lambda sample: SampleMetadata(pack_tokens=sample["length"]),
    pack_fn=pack_one_sequence,
    collate_fn=collate_packed_sequences,
)
```

Every Source Loader rank must provide a replica of the same logical
mapping-style Dataset. The Source Loaders apply their own global stride, so do
not pre-shard the Dataset again per Source rank; the builder collectively
checks that all Source replicas have the same length. Non-Source ranks may
pass the same object or `None`. By default, the Data Constructor ranks are also
the Sources; `source_loader_ranks` can assign additional readers.

Compose multiple datasets with a mapping-style mixer such as `ConcatDataset`
or Megatron `BlendableDataset` before calling the builder. The distributed
layer does not impose a sample field schema.

Online image-placeholder workflows fit naturally: `Dataset.__getitem__`
returns the raw payload or image locator, metadata describes its planned
token/cost footprint, and `pack_fn` performs target-constructor image loading
and replacement before collation.

If one offline sample is already a complete local batch, configure
`local_batch_size=1`, report its logical `pack_tokens`, and use identity-style
packing/collation. HyperParallel will treat the sample as one indivisible
balancing unit.

## Current boundaries

- Mapping-style Dataset and fixed topology.
- Synchronous planning with no in-flight background prefetch.
- CPU Gloo data plane and correctness-first framed pickle payloads.
- `drop_last=True`; every DP rank receives the same number of non-empty bins.
- Checkpoints are per rank and include transformed read-ahead payloads. Sample
  keys and plans replay exactly when the mapping Dataset is deterministic for a
  given index and epoch; arbitrary Dataset/worker RNG state is not captured.
  Every training rank must save and restore its own loader state.
- Sidecar ahead-of-fetch planning, elastic resharding, iterable-source exact
  replay, and structured zero-copy tensor transport are separate extensions.
