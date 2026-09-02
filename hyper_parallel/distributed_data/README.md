# Distributed dynamic packing

This package implements a synchronous PyTorch data path inspired by the
loading, planning, and construction separation in MegaScale-Data. The loading
role is intentionally named Dataset Reader here: multiple readers consume
deterministic partitions of one logical Dataset stream; they do not represent
independent data sources.

```text
online metadata_fn
  -> Dataset Reader materializes samples and derives metadata
  -> Step Sample Selection freezes the current stream prefix
  -> Balanced Placement
  -> CPU/Gloo or device NCCL/HCCL A2A routes selected payloads

ahead-of-fetch sidecar metadata
  -> metadata-only Dataset Reader
  -> Step Sample Selection reproduces the same stream membership
  -> Balanced Placement
  -> shared index space: target Constructor reads planned indices (no payload A2A)
  -> pre-sharded inputs: owning Reader reads planned local indices, then payload A2A

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

The metadata callback defines packing token accounting, including special
tokens and reserved multimodal placeholders. A custom packer must use the same
accounting. Oversized samples fail by default; `oversized_policy="single"`
explicitly allows an overflow sample to occupy a bin alone.

Step Sample Selection uses deterministic streaming packing to form exactly
`local_batch_size * dp_size` reference bins from the canonical epoch stream.
It freezes that union of sample IDs before workload balancing. Balanced
Placement may repack and reorder only those samples; it cannot skip a costly
sample, borrow a future sample, or move data across step boundaries. If the
sample-level packing heuristic reaches a dead end, it falls back to the
known-feasible reference grouping so conservation is exact.

`buffer_size_multiplier` controls physical read-ahead only. It can reduce
control-plane planning rounds, but changing it does not change step membership.

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
    DistributedDatasetConfig(
        seq_len=32768,
        local_batch_size=2,
        double_buffer=True,
    ),
    metadata_fn=lambda sample: SampleMetadata(pack_tokens=sample["length"]),
)
```

Online mode also accepts an iterable Dataset. Set
`dataset_already_sharded=True` when each Reader's Dataset already owns its
rank-local partition, as in an existing IterableDataset pipeline. HyperParallel
then consumes that local stream with `next()` and assigns Reader-local sample
positions only for planning and routing; it does not apply a second stride.
With the default `False`, HyperParallel strides one shared mapping or iterable
stream across Dataset Readers. Iterable Datasets own their shuffle order, so
use `shuffle=False` in this configuration.

With `double_buffer=True`, the first iterator call constructs its batch before
returning. After each successful delivery, a background thread prepares exactly
one subsequent distributed batch while the trainer consumes the current batch.
The background path uses the dedicated data control, payload, and model-delivery
process groups created by this package; it does not issue collectives on the
trainer's process groups. The default is `False`.

With this option enabled, `metadata_fn`, `pack_fn`, and `collate_fn` run on the
prefetch thread and may overlap model execution. They must not mutate shared
trainer state. Accelerator allocations inside callbacks should use explicit
devices instead of relying on a thread-local current device. Double buffering
also retains the current trainer batch and one constructed next batch at once,
so Host-memory usage increases accordingly.

Checkpointing drains and discards speculative construction without committing
its selected Reader samples, copies the completed-step state, and then restarts
one-step prefetch. Restoring the checkpoint therefore reconstructs the same next
step rather than treating a prefetched batch as consumed.

Trainer-side H2D is a separate slot because CP-specific mask preparation and
the accelerator copy stream are model-runtime concerns. `DeviceBatchPrefetcher`
implements the reusable stream/Event part:

```python
from hyper_parallel.distributed_data import DeviceBatchPrefetcher

device_prefetcher = DeviceBatchPrefetcher(
    accelerator.device,
    prepare_fn=prepare_cp_und_mask,  # optional CPU-only transformation
)

# The first batch has no previous compute to hide its copy behind.
device_prefetcher.prefetch(next(loader))
for step in range(train_steps):
    batch = device_prefetcher.wait()  # Event wait before model/broadcast reads
    loss = forward(batch)
    backward(loss)
    if step + 1 < train_steps:
        # Host double buffering normally makes next(loader) immediately ready.
        device_prefetcher.prefetch(next(loader))
```

The prefetcher lazily creates one accelerator copy stream, calls
`batch.to(device, non_blocking=True)` by default, records a per-batch ready
Event, and associates device tensors with the consuming stream after the wait.
Actual asynchronous H2D requires pinned Host tensors. A custom `move_fn` may be
used for a batch type without a compatible `to` method.

Every rank must continue to call `next(loader)` in the same order because the
distributed loader performs collective planning and model-parallel Host
delivery. In that normal path, each rank moves its delivered local batch and a
separate CP rank-0 device broadcast is unnecessary. A legacy rank-0-only
DataLoader may instead use `DeviceBatchPrefetcher` only on the source rank and
call `wait()` before its existing device broadcast.

Calling `next(loader)` for H2D prefetch advances the loader's completed-step
boundary. Do not checkpoint between that call and training the returned device
batch. On a checkpoint step, save the completed loader state before pulling the
next Host batch; delaying H2D for those infrequent steps preserves exact replay.

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

For rank-local sidecars, provide the local mapping Dataset and local metadata
on every Dataset Reader and enable the same sharding switch:

```python
loader = build_distributed_dataloader(
    local_dataset,
    mesh,
    DistributedDatasetConfig(
        seq_len=32768,
        local_batch_size=1,
        dataset_already_sharded=True,
    ),
    metadata=local_sidecar_metadata,
    pack_fn=pack_one_sequence,
    collate_fn=collate_packed_sequences,
)
```

The Reader scans metadata only, then reads just the selected local indices and
routes those payloads to their target constructors. This path needs payload
A2A because another constructor cannot index the Reader's private shard.

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

By default, every Dataset Reader rank provides a replica of the same logical
online Dataset or sidecar metadata and HyperParallel applies the Reader stride.
With `dataset_already_sharded=True`, each Reader instead provides its local
online stream or an aligned local Dataset/sidecar pair. Non-owning ranks may
pass the same object or `None`. By default, Data Constructor ranks are also the
Dataset Readers; `dataset_reader_ranks` can separate metadata scanning from
construction.

Compose multiple datasets with a mapping-style mixer such as `ConcatDataset`
or Megatron `BlendableDataset` before calling the builder. The distributed
layer does not impose a sample field schema.

Online image workflows fit naturally: the Dataset iterator or `__getitem__`
returns one raw sample, metadata describes its token/cost footprint, and the
constructor packs it after redistribution.

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
copies, so device A2A should be benchmarked for the target payload size. Shared
sidecar direct reads bypass this transport; pre-sharded sidecars use it.

If one offline sample is already a complete local batch, configure
`local_batch_size=1`, report its logical `pack_tokens`, and use identity-style
packing/collation. HyperParallel will treat the sample as one indivisible
balancing unit.

## Current boundaries

- Mapping or iterable online Dataset; sidecar reads require mapping access.
- At most one in-flight background batch when `double_buffer=True`; the first
  batch and non-double-buffer mode wait synchronously.
- Gloo control plane and correctness-first framed pickle payloads; online A2A
  may use Gloo, NCCL, or HCCL.
- `drop_last=True`; every DP rank receives the same number of non-empty bins.
- Checkpoints are per rank and include transformed read-ahead payloads. Sample
  keys and plans replay exactly when the Dataset stream is deterministic for a
  given epoch; arbitrary Dataset/worker RNG state is not captured.
  Every training rank must save and restore its own loader state.
- Elastic resharding and structured zero-copy tensor transport are separate
  extensions.
