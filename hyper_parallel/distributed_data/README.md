# Distributed sample balancing

This PyTorch data path separates Dataset Readers, planning, and Data
Constructors, inspired by MegaScale-Data. Step membership belongs to the
upstream producer, not to a metadata read-ahead window.

```text
native BatchSampler selects this step's Dataset-index occurrences
  -> metadata: look up only selected indices
               -> Planner -> target Constructor reads dataset[index], no payload A2A
  -> metadata_fn: read selected samples, extract metadata
                  -> Planner -> payload A2A when redistributed
  -> preserve complete Dataset outputs -> collate_fn

external_step_reader emits a complete local step, metadata, reference_bins
  -> freeze the union of Reader samples -> Planner
  -> payload A2A when redistributed -> pack_fn -> collate_fn

both paths
  -> broadcast the local batch to model-parallel peers (schema object + direct tensor leaves)
  -> commit consumed progress -> Trainer
```

Model-parallel batch broadcast uses one small `broadcast_object_list` call for
the Python structure and non-tensor values. Tensor leaves in dictionaries,
lists, and tuples use `dist.broadcast` directly, avoiding pickle copies. CPU
loads use the existing Gloo model group; when an accelerator
`communication_device` is supplied, a matching HCCL/NCCL model tensor group is
created and tensor leaves stay on device. Unsupported custom containers retain
the regular object-broadcast behavior.

There is no metadata-only streaming mode. Passing only `metadata` (including
Dataset-inferred metadata) without `batch_sampler` fails at build time.
Passing only `metadata_fn` without a sampler or external step source also fails.
The loader never expands a candidate window to infer how many samples a step
should contain; there is no refill/`attempt` loop.

## Batch semantics

- `seq_len` is the hard token capacity of one normal sequence bin.
- `local_batch_size` is the number of sequences returned per DP rank and yield.
  Native BatchSampler mode treats each whole Dataset output as one such unit.
- One synchronized iterator step covers `local_batch_size * dp_size` bins.
  Gradient accumulation and optimizer global batch size remain Trainer concerns.
- Native BatchSampler fixes index occurrences and uses singleton reference bins;
  `external_step_reader` fixes samples and supplies the original packing bins.
  Balancing may only rearrange that frozen sample set. It cannot skip expensive
  samples, borrow future samples, or change step membership.
- In external mode, the packer must agree with metadata token accounting,
  including special tokens and reserved image placeholders. If sample-level
  placement fails, the Planner falls back to the known-feasible reference bins.
- `buffer_size_multiplier` remains accepted for compatibility but does not affect
  these paths. `max_buffered_samples` is forwarded to external readers; native
  BatchSampler always emits its complete batch.

## Public API

```python
from hyper_parallel.distributed_data import (
    DistributedDatasetConfig,
    SampleMetadata,
    build_distributed_dataloader,
)
```

### Native HP BatchSampler: preserve each forward/backward round

Pass `batch_sampler=` to keep the native HP sampling policy. This is required
for ahead-of-fetch metadata. For online iterable pipelines, use
`external_step_reader` instead.

```python
from hyper_parallel.data.parallel import build_dataset_batch_sampler

# Build the rank-local sampler on every rank, using training DP coordinates.
# Only the Data Constructor in each DP replica advances its sampler.
batch_sampler = build_dataset_batch_sampler(
    total_samples=len(dataset),
    micro_batch_size=2,
    global_batch_size=32,
    dp_rank=dp_rank,
    dp_world_size=dp_size,
    sampler_type="cyclic",
    seed=1234,
)
loader = build_distributed_dataloader(
    dataset,
    mesh,
    DistributedDatasetConfig(seq_len=32768, local_batch_size=2),
    batch_sampler=batch_sampler,
    metadata_fn=lambda sample: SampleMetadata(pack_tokens=len(sample["tokens"])),
    collate_fn=native_collate_fn,
)
```

```text
each DP Constructor advances its native BatchSampler once
  -> freeze that round's native Dataset-index occurrences
  -> online: read those Dataset outputs, then derive metadata
     metadata: look up metadata for exactly those Dataset indices
  -> balance only this frozen round (one complete Dataset output per bin)
  -> online: payload A2A / shared metadata: target reads, no payload A2A
  -> original collate_fn receives complete Dataset outputs
  -> broadcast the local batch to model-parallel peers
  -> validate the batch, commit the native sampler cursor, and return the data
```

For DP=2 and `local_batch_size=2`, if native samplers yield `[0, 1]` and
`[2, 3]`, only these four occurrences can be balanced in that round. No samples
are borrowed from another gradient-accumulation round. Index mappings that
repeat a Dataset index remain repeated occurrences; routing keys distinguish
them by `global_sample_position` rather than deduplicating the index.
In this mode that position is a stable DP-major occurrence slot within the
native round, not a physical document index or the sampler's internal shuffle
permutation position.

In the HP text Trainer, retain the original Indexed Dataset and collator and
enable this path with:

```yaml
dataset:
  data_config:
    packing_stage: dataset
    load_balance: native_batch_sampler
    seq_length: 32768
    distributed_dataloader:
      double_buffer: true
```

The Trainer builds its normal `build_dataset_batch_sampler` first, including
`single`/`cyclic`, `data_sharding`, and the supported index rearrangement map.
It passes that sampler to the collective loader instead of applying another
Reader stride. Worker settings and the original `dataloader.collate_fn` are
retained. The initial text adapter supplies token lengths, not a calibrated TND
cost model; fixed-length GPT outputs therefore do not automatically gain compute
balance from this opt-in. The lower-level API accepts user-provided costs through
`metadata_fn` or `metadata`.
For CUDA/NPU meshes the Trainer passes the foreground rank-local device to the
existing device payload transport; CPU meshes use Gloo.

Current boundaries of the native-sampler path:

- `dataset[index]` stays **whole**. Native GPT cross-document slicing, shifted
  labels, loss masks, and positions are not rebuilt. Internal independent TND
  fragments are **not yet extracted or balanced**. An offline-packed record also
  stays whole.
- Supply an HP-compatible native sampler on every rank. Only Constructor ranks
  need the mapping Dataset, and only those ranks advance their samplers. Readers
  must currently coincide with Constructors. `shuffle=False` and
  `dataset_already_sharded=False` are required in `DistributedDatasetConfig`:
  the native sampler already owns both decisions. `pack_fn` must be omitted and
  `drop_last=True` is required. Iterable sources require `external_step_reader`.
- Supplied metadata must describe the **logical native Dataset outputs** after
  blend/shuffle/sample-index mapping, not raw document IDs. Raw `.idx` lengths
  alone do not describe GPT-internal EOD/TND boundaries. This version does not
  automatically generate native-GPT metadata; the Trainer opt-in uses online
  metadata. Shared-metadata reads also require deterministic, rank-independent
  Dataset outputs.
- Save and restore through `loader.state_dict()` / `load_state_dict()`, not
  through the original sampler: its live cursor may include one prefetched
  round. Native cursor snapshots count Dataset-output occurrences, never tokens
  or fragments, and can restore at a synchronized forward/backward boundary with
  unchanged DP topology, global batch size, and sampler policy.
- Native-round membership and pre-collation Dataset fields are preserved, not
  bitwise training results. Worker RNG replay, batch-dependent transforms,
  dropout, reduction order, a concrete TND runtime adapter, and full NPU training
  parity are outside this sampler integration. The Trainer path rejects PP and
  a second `DynamicBatchDataLoader` selection stage.

Native-sampler regression coverage:

```bash
HYPER_PARALLEL_PLATFORM=torch python -m pytest -q \
  tests/ut/distributed_data/test_batch_sampler.py \
  tests/ut/data/test_indexed_source_dataset.py \
  tests/torch/distributed_data/test_batch_sampler_gloo.py \
  tests/torch/distributed_data/test_indexed_text_gloo.py
```

### Native HP image-text Dataset (VLMTrainer)

VLMTrainer can use the same native-sampler path. Add `load_balance` to the
existing VLM Dataset configuration; keep the original transform and collator:

```yaml
dataset:
  _target_: hyper_parallel.data.vlm.build_vlm_dataset
  data_path: /data/conversations.json
  data_config:
    source_type: online
    load_balance: native_batch_sampler
    distributed_dataloader:
      double_buffer: true
  data_transform:
    _target_: hyper_parallel.data.vlm.build_vlm_data_transform
    max_seq_len: 32768
dataloader:
  _target_: hyper_parallel.data.batching.FixedBatchDataLoader
  dataloader_type: single
  drop_last: true
  num_workers: 2
  prefetch_factor: 2
  collate_fn:
    _target_: hyper_parallel.data.vlm.build_vlm_collator
    packing: false
training:
  micro_batch_size: 2
  global_batch_size: 32
```

This is a data-section example to merge into an existing VLMTrainer model/run
configuration. The distributed sequence capacity is inferred from the built
transform's `max_seq_len`; `data_config.seq_length` is unnecessary. If both are
supplied, they must agree. With DP=8 this example selects 16 global samples per
forward/backward round and accumulates two rounds per optimizer update.

```text
native HP BatchSampler selects each round's Dataset-index occurrences
  -> original VLM Dataset + processor read/decode/encode complete conversations
  -> vlm_sample_metadata extracts padded width and image patch counts
  -> gather metadata and balance this frozen round across DP ranks
  -> A2A complete image-text sample dictionaries
  -> original VLMCollator: stack text [local_batch_size, max_seq_len]
                          concatenate image patches and image grids
  -> VLMGetBatch -> model forward/backward -> gradient accumulation
```

- A conversation may contain multiple images and turns, but remains **one
  indivisible sample**. Input IDs, prompt-masked labels, attention masks,
  modality markers, pixels and grids move together. No label shifting, new
  padding policy, or cross-conversation packing is introduced.
- The default `hyper_parallel.data.vlm.metadata.vlm_sample_metadata` estimator
  uses `sum(T * H * W)` over the processor's actual `image_grid_thw` as encoder
  workload, and padded text width as LLM workload and `pack_tokens`. Raw vision
  patches are not the same as merged LLM image placeholders. This is a simple
  workload proxy, **not a calibrated cost model**. Custom estimates can be
  supplied via `metadata_fn` when calling `build_distributed_dataloader` directly.
- The Trainer integration uses **online metadata**, extracted after the native
  Dataset transform. It cannot balance CPU image decoding/processing already
  performed by the Readers. Native `_TransformDataset` still performs its
  initial trainable-label filtering; no precomputed metadata is automatically generated.
  Manually supplied metadata through the lower-level API must match the
  filtered/transformed Dataset index space, not the original JSON row numbers.
- The HP VLM runtime currently requires **TP=CP=PP=1**; this integration keeps
  that boundary and supports DP. It does not add VLM packing, video/audio
  adapters, or a new model-parallel batch-broadcast implementation.
- Omit `load_balance` to retain the native DataLoader. Enabled loaders own
  checkpoint cursors and epochs; VLMTrainer preserves a restored first-epoch
  cursor and waits for outstanding prefetch before distributed teardown.
  CUDA/NPU payload communication uses the existing device transport; the
  tests below cover CPU/Gloo, not full accelerator-model accuracy or speed.

VLM regression coverage uses real local images and the original HP Dataset,
transform, collator and get-batch path with a deterministic test processor
(no remote assets). It checks sample/field conservation, DP2 payload A2A,
worker loading, double buffering, resume, and native-vs-balanced gradients for
an image-conditioned toy loss using HP token normalization:

```bash
HYPER_PARALLEL_PLATFORM=torch python -m pytest -q \
  tests/ut/data/vlm/test_metadata.py \
  tests/ut/trainer/test_vlm_trainer.py \
  tests/torch/distributed_data/test_vlm_gloo.py
```

### Migration from metadata streaming

The Trainer shortcut `packing_stage: distributed_dataloader` depended on
streaming metadata selection and now raises a migration error. For native HP
training, retain `packing_stage: dataset` and set
`load_balance: native_batch_sampler`. This preserves whole GPT Dataset outputs;
it does **not** rebalance document fragments inside a packed GPT sequence.

The lower-level Indexed source Dataset and text packing helpers remain available
for custom producers. To retain dynamic packing across raw source samples, that
producer must define complete steps and provide an `external_step_reader`.
HP no longer infers step boundaries by scanning source metadata.

With `double_buffer=True`, the first iterator call constructs its batch before
returning. After each batch is returned, a background thread prepares exactly
one subsequent distributed batch while the trainer consumes the current batch.
The prefetch worker is created once and reused for subsequent steps.
`_prepare_next_batch()` runs on that thread using the dedicated data control and
payload process groups. `_broadcast_batch()` runs on the caller thread using the
dedicated model-parallel batch group, then `_consume_batch()` handles the
end-of-stream sentinel, commits Reader progress, and returns the training data.
These operations do not
issue collectives on the trainer's process groups. The default is `False`.

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
broadcast. In that normal path, each rank moves its received local batch and a
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
    batch_sampler=batch_sampler,
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

### Ahead-of-fetch metadata

Supply metadata aligned one-to-one with the mapping Dataset's logical index
space. Each Constructor has the same Dataset and metadata, plus its DP-local
native sampler:

```python
loader = build_distributed_dataloader(
    dataset,
    mesh,
    config,
    batch_sampler=batch_sampler,
    metadata=precomputed_metadata,
    collate_fn=native_collate_fn,
)
```

The sampler determines exactly which indices belong to this step. Readers look
up metadata only for those occurrences, the Planner balances them, and each
Constructor reads only its assigned payloads. No payload A2A is needed; metadata
gather and plan broadcast still occur. Do not pass `pack_fn`: each Dataset
output remains one indivisible sample. The default collator returns a tuple of
these samples.

All Constructors must be able to read the same global Dataset indices.
Reader-private shards with metadata routing are no longer supported by this
builder. Compose multiple datasets and their metadata in the same global index
order, then let the native BatchSampler own DP slicing and shuffling.

### External online steps

An external Reader may use an IterableDataset or any existing local pipeline;
random indexed access is unnecessary:

```python
loader = build_distributed_dataloader(
    None,
    mesh,
    config,
    external_step_reader=reader,  # only Dataset Reader ranks supply an instance
    pack_fn=pack_one_sequence,
    collate_fn=collate_packed_sequences,
    communication_device=torch.device("npu", local_rank),
)
```

The Reader exposes `fill`, `metadata`, `reference_bins`, `selected_payloads`,
`commit`, `exhausted`, `batch_position`, `state_dict`, `load_state_dict`, and
`set_epoch`. Each fill produces **one complete local step**, not a read-ahead
candidate window. HP does not apply a second stride or manage this external
producer's DataLoader workers; configure those on the producer itself.
`metadata` and `metadata_fn` must be omitted because the Reader owns extraction.

The default packer and collator preserve the structure as a tuple of bins,
each containing the raw samples. Custom callbacks must be consistent across
ranks. Double buffering still overlaps planning, payload routing, and
construction with training.

Online payload A2A defaults to CPU/Gloo. A rank-local `communication_device`
uses the WORLD accelerator backend by default (HCCL for NPU or NCCL for CUDA);
`payload_backend` can override it. Python payloads still incur serialization
and Host/device copies, so benchmark the target workload. Shared metadata mode
bypasses payload transport entirely.

## Current boundaries

- Native BatchSampler requires a mapping Dataset; metadata also requires shared indices.
- Iterable/streaming online data requires an external complete-step Reader.
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

## Buffered node-local balancing

For automatic Host/H2D double buffering around an existing rank-local loader,
v1 cost estimation, Gloo node-local LPT balancing and rank-zero DP logs, see
[the reference configuration](NODE_LOCAL_BALANCING.md). This is a separate
opt-in builder; the native pipeline and its checkpoint path above are unchanged.
