# Distributed dynamic packing

This package implements a synchronous PyTorch data path inspired by the
loading, planning, and construction separation in MegaScale-Data. The loading
role is intentionally named Dataset Reader here: multiple readers consume
deterministic partitions of one logical mapping Dataset; they do not represent
independent data sources.

```text
online metadata_fn
  -> Dataset Reader materializes samples and derives metadata
  -> Planner
  -> CPU/Gloo or device NCCL/HCCL A2A routes selected payloads

ahead-of-fetch sidecar metadata
  -> metadata-only Dataset Reader
  -> Planner
  -> target Data Constructor reads planned Dataset indices (no payload A2A)

both modes
  -> pack each sequence bin, then collate the local batch
  -> CPU broadcast to model-parallel peers
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

The metadata callback defines the Planner's token accounting, including special
tokens and reserved multimodal placeholders. A custom packer must use the same
accounting. Oversized samples fail by default; `oversized_policy="single"`
explicitly allows an overflow sample to occupy a bin alone.

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
)
```

PyTorch DataLoader execution options can be passed through
`dataloader_kwargs`:

```python
loader = build_distributed_dataloader(
    dataset,
    mesh,
    config,
    metadata_fn=metadata_fn,
    dataloader_kwargs={
        "num_workers": 8,
        "pin_memory": True,
        "prefetch_factor": 4,
        "persistent_workers": True,
        "timeout": 60,
        "worker_init_fn": worker_init_fn,
        "multiprocessing_context": "spawn",
        "pin_memory_device": "",
        "in_order": True,
    },
)
```

Values in `dataloader_kwargs` override the legacy `num_workers`,
`pin_memory`, `prefetch_factor`, and `persistent_workers` fields in
`DistributedDatasetConfig`. The supported keys are those four fields plus
`timeout`, `worker_init_fn`, `multiprocessing_context`, `pin_memory_device`,
and `in_order`; deterministic routing requires `in_order=True`. Accelerator
jobs using NPU or CUDA should generally select the `spawn` multiprocessing
context. `timeout` controls how long DataLoader waits for workers and is not a
distributed collective timeout. A custom `worker_init_fn` must be consistent
across ranks and picklable when using `spawn`; with persistent workers, PyTorch
invokes it only when each worker process is created.

Dataset and distributed batching semantics remain internal. Do not pass
`dataset`, `batch_size`, `shuffle`, `sampler`, `batch_sampler`, `collate_fn`,
`drop_last`, or `generator` through `dataloader_kwargs`.

When sidecar entries are aligned one-to-one with mapping-Dataset indices, pass
`metadata` instead of `metadata_fn`. Dataset Reader ranks need the sidecar; Data
Constructor ranks need the Dataset. With the default topology they provide both:

```python
loader = build_distributed_dataloader(
    dataset,
    mesh,
    config,
    metadata=sidecar_metadata,
    pack_fn=pack_one_sequence,
    collate_fn=collate_packed_sequences,
)
```

The Planner runs before any `dataset[index]` call. Each constructor then uses
its local DataLoader workers to fetch only assigned indices. For multiple
datasets, compose the datasets and sidecars in the same global index order.

When callbacks are omitted, the lossless defaults return the planned structure
without guessing the user sample schema:

```text
local batch tuple
  -> packing-bin tuple
       -> raw samples in deterministic planned order
```

Pass either callback only when model-specific construction is needed:

```python
loader = build_distributed_dataloader(
    dataset,
    mesh,
    config,
    metadata_fn=metadata_fn,
    pack_fn=pack_one_sequence,
    collate_fn=collate_packed_sequences,
)
```

Custom callback implementations are a user consistency contract and must be
the same on every rank. The build preflight does detect default-versus-custom
mode mismatches, but cannot reliably fingerprint arbitrary Python closures.

Every Dataset Reader rank must provide a replica of the same logical online
Dataset or sidecar metadata. The Dataset Readers apply their own global stride,
so do not pre-shard either input again per reader rank. In sidecar mode,
constructor Dataset length must match the sidecar length. Non-owning ranks may
pass the same object or `None`. By default, Data Constructor ranks are also the
Dataset Readers; `dataset_reader_ranks` can separate metadata scanning from
direct reads.

Compose multiple datasets with a mapping-style mixer such as `ConcatDataset`
or Megatron `BlendableDataset` before calling the builder. The distributed
layer does not impose a sample field schema.

Online image-placeholder workflows fit naturally: `Dataset.__getitem__`
returns the raw payload or image locator, metadata describes its planned
token/cost footprint, and `pack_fn` performs target-constructor image loading
and replacement before collation.

Online payload A2A defaults to CPU/Gloo. Pass a rank-local
`communication_device` to use the WORLD accelerator backend, or set
`payload_backend` explicitly:

```python
loader = build_distributed_dataloader(
    dataset,
    mesh,
    config,
    metadata_fn=metadata_fn,
    communication_device=torch.device("npu", local_rank),  # HCCL
)
```

Packed Python payloads still incur pickle plus Host-to-Device and Device-to-Host
copies, so device A2A should be benchmarked for the target payload size. Sidecar
direct reads bypass this transport entirely.

If one offline sample is already a complete local batch, configure
`local_batch_size=1`, report its logical `pack_tokens`, and use identity-style
packing/collation. HyperParallel will treat the sample as one indivisible
balancing unit.

## Current boundaries

- Mapping-style Dataset and fixed topology.
- Synchronous planning with no in-flight background prefetch.
- Gloo control plane and correctness-first framed pickle payloads; online A2A
  may use Gloo, NCCL, or HCCL.
- `drop_last=True`; every DP rank receives the same number of non-empty bins.
- Checkpoints are per rank and include transformed read-ahead payloads. Sample
  keys and plans replay exactly when the mapping Dataset is deterministic for a
  given index and epoch; arbitrary Dataset/worker RNG state is not captured.
  Every training rank must save and restore its own loader state.
- Elastic resharding, iterable-source exact replay, and structured zero-copy
  tensor transport are separate extensions.
