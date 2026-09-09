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

### Native HP BatchSampler: preserve each forward/backward round

Pass `batch_sampler=` to keep the native HP sampling policy. This is separate
from the stream-based dynamic packing path above; omitting it retains the old
behavior.

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
     sidecar: look up metadata for exactly those Dataset indices
  -> balance only this frozen round (one complete Dataset output per bin)
  -> online: payload A2A / shared sidecar: target reads, no payload A2A
  -> original collate_fn receives complete Dataset outputs
  -> deliver the local batch to model-parallel peers
  -> commit the native sampler cursor after delivery
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
  `drop_last=True` is required. Iterable sources keep the existing stream path.
- A supplied sidecar must describe the **logical native Dataset outputs** after
  blend/shuffle/sample-index mapping, not raw document IDs. Raw `.idx` lengths
  alone do not describe GPT-internal EOD/TND boundaries. This version does not
  automatically generate a native-GPT sidecar; the Trainer opt-in uses online
  metadata. Shared-sidecar reads also require deterministic, rank-independent
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
  initial trainable-label filtering; no sidecar is automatically generated.
  A manually supplied sidecar through the lower-level API must match the
  filtered/transformed Dataset index space, not the original JSON row numbers.
- The HP VLM runtime currently requires **TP=CP=PP=1**; this integration keeps
  that boundary and supports DP. It does not add VLM packing, video/audio
  adapters, or a new model-parallel batch-delivery implementation.
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

### HyperParallel unpacked Indexed text data

The HP Indexed provider keeps its existing `GPTDataset` behavior by
default. To balance the source sequences inside each packed row, use an
unpacked `.bin/.idx` corpus and select constructor-side packing:

```yaml
dataset:
  _target_: hyper_parallel.data.text.build_indexed_text_dataset
  data_path: /data/corpus_text_document
  data_config:
    seq_length: 32768
    split: "98, 1, 1"
    mock_data: false
    is_dataset_from_mr: false
    simple_blend: "no"
    data_lazy_load: true
    distributed_walk: false
    packing_stage: distributed_dataloader
    create_attention_mask_in_dataloader: true
    distributed_dataloader:
      buffer_size_multiplier: 2.0
      double_buffer: true
```

This changes the path to:

```text
.idx sequence length -> Step Sample Selection -> balanced plan
                     -> target Constructor reads .bin by planned index
                     -> fixed [local_batch_size, seq_len] text batch
```

`seq_len` and `local_batch_size` are derived from `data_config.seq_length` and
`training.micro_batch_size`. The configured DataLoader worker count,
`pin_memory`, `persistent_workers`, and `prefetch_factor` are reused. Other
worker execution options (`timeout`, `worker_init_fn`, `multiprocessing_context`,
`pin_memory_device`, and `in_order`) are forwarded through `dataloader_kwargs`.
For accelerator jobs, set `dataloader.multiprocessing_context: spawn`.
Other `DistributedDatasetConfig` tuning fields may be placed below
`data_config.distributed_dataloader`.
The provider uses shared indices, so `dataset_already_sharded` cannot be
overridden here. Every Constructor needs access to the same corpus files.
The Trainer selects the built-in text packing and collation callbacks;
`dataloader.collate_fn` is optional in this mode.

The `.idx` lengths form an implicit sidecar, so the shared-index path plans
before payload reads and does not use payload A2A. Every packed row records
source boundaries in `cu_seq_lens`; padding labels use `-100`. The HP Trainer also
switches `ParallelBatch` to the `indexed_source` contract so those boundaries
isolate attention between source samples. Position IDs restart at each source
boundary, independent of its new packing offset. This path requires
`create_attention_mask_in_dataloader: true`; compressed attention also requires
an `attention_runtime_adapter` that consumes those boundaries. Implicit full-row
causal attention cannot be used for independently packed documents.

Each source record is shifted independently (`input_ids=text[:-1]`,
`labels=text[1:]` by default), and source boundaries stay explicit even when
the final EOD appears only in the labels. This preserves independent-document
semantics; it is not token-for-token equivalent to GPTDataset concatenating
documents across fixed-length sample boundaries. `labels_are_shifted` must
remain true. With `add_extra_token_to_sequence=false`, the final label of each
source record is ignored instead.

Multiple prefixes retain weighted blending and lazy metadata lookup. A source
epoch is sized by the longest source relative to its weight, with shorter
sources repeated. Training continues across dynamic epochs until `train_iters`
is reached; checkpoint restore keeps the consumed source cursor. Before
tearing down communication groups, TextTrainer waits for double-buffer prefetch
to finish. Standalone users can call `loader.wait_for_prefetch()` for the same
synchronization without consuming the prepared batch.

This first version requires unpacked `.bin/.idx`, `is_dataset_from_mr=false`,
`simple_blend=no`, `drop_last=true`, no pipeline parallelism, and source sequences no longer
than `seq_length`. Data produced with `--pack-to-seq-len` remains a pre-packed
record and must keep the default `packing_stage: dataset` path.

The HP Trainer path also needs `torchdata` (tested with 0.11.0) for its shared
batching imports. After installing the project and its training dependencies,
the CPU regression commands are:

```bash
HYPER_PARALLEL_PLATFORM=torch python -m pytest -q \
  tests/ut/data/test_indexed_source_dataset.py \
  tests/ut/distributed_data
HYPER_PARALLEL_PLATFORM=torch python -m pytest -q \
  tests/torch/distributed_data/test_indexed_text_gloo.py
```

The integration test creates real `.bin/.idx` files and verifies DP2/TP2
delivery, index-only planning, no payload A2A, double buffering, and loss/gradient
parity against independently evaluated documents with unequal valid-token
counts. The unit tests cover weighted blending, persistent spawned workers,
checkpoint replay, and the unchanged default GPT Dataset path.

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
