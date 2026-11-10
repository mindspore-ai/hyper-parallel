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

external_step_source emits a complete local step of raw-sample bins
  -> HP derives metadata and freezes the source step -> Planner
  -> node-local payload A2A when redistributed -> pack_fn -> collate_fn
  -> buffered H2D -> Trainer (pure DP, no loader checkpoint)

native BatchSampler
  -> broadcast the local batch to model-parallel peers (locally declared layout + tensor-only collectives)
  -> commit consumed progress -> Trainer
```

Model-parallel batch broadcast uses only tensor collectives. Each peer declares
an identical `model_batch_spec` locally: container structure, tensor dtypes,
fixed dimensions and static values never travel over the network. A small int64
header carries EOF/error status and dynamic dimensions; tensor leaves use
`dist.broadcast`. Pure DP needs no declaration. Built-in text collation provides
its declaration automatically; custom model-parallel batches must supply one
(see [Tensor-only model broadcasts](#tensor-only-model-broadcasts)).

The native Trainer builds one `DataLoaderParallelContext` and passes it to both
Dataset construction and `TextParallelBatch` / `OmniParallelBatch`. With
`dataloader.load_balance` enabled, its `tp_source_rank=None` means every model
rank advances the distributed loader, which owns the MP batch broadcast.
`get_batch` then performs CP slicing, dtype/device normalization and model/loss
input construction without a second TP broadcast. Without balancing, the
source remains TP rank zero at each CP coordinate and the existing TP
broadcast runs as before. There is no loader marker or iterator-wrapper probe.
The MP group still spans all non-DP mesh dimensions; CP slicing preserves global
`cu_seq_lens`. Native balanced integration continues to require PP=1.
Custom batch runtimes should accept the Trainer's `dataloader_context` and use
its source/transport policy. Direct API users pairing a distributed loader with
a built-in batch runtime must pass a context created with `tp_source_rank=None`.

Configuration, callback results, sample conservation, packing capacities, and
checkpoint compatibility remain validated at their boundaries. Fixed-group
snapshots and internally generated tensor descriptors are not revalidated at
every stage. Payload A2A uses one pickle format without an application-level
checksum; duplicate received keys and missing/unexpected Constructor payloads
still fail. All ranks in a job must run the same code version. Layout validation errors are signaled to model peers in the batch header; the
accelerator path also forwards Constructor preparation failures this way. Other runtime failures still require coordinated
shutdown by the training driver; this is not distributed fault recovery.

Checkpoint compatibility is checked from the saved state fields, without a
separate format version. Legacy `version` fields are ignored during restore.

`DistributedDatasetConfig.metadata_mode` selects the metadata source explicitly.
The default `False` is online mode: pass `metadata_fn` and let HP derive
metadata from the samples selected by the native `batch_sampler` (or by an
`external_step_source`). Set it to `True` for offline metadata: explicitly pass
`metadata`, an indexed object implementing `__len__` and `__getitem__`. Online
mode calls `metadata_fn(sample)` after reading the sample; offline mode reads
`metadata[index]` before fetching its payload. Offline mode requires a native
`batch_sampler`, and the metadata and Dataset indices must have the same length
and ordering. Missing metadata arguments fail at build time.
Passing `metadata` in online mode or `metadata_fn` in offline mode fails at
build time. There is no metadata-only streaming mode.
The loader never expands a candidate window to infer how many samples a step
should contain; there is no refill/`attempt` loop.

`DistributedDatasetConfig.broadcast_plan` defaults to `True`: one Planner
computes the plan and broadcasts it to the data-plane ranks. Set it to `False`
to let every data-plane rank compute the complete plan from the same gathered
metadata, skipping the plan broadcast. This applies to native sampler and
external-step loading. Keep configuration, cost models, and assignment policies
identical and deterministic across ranks. Metadata gathering, payload exchange,
and model-parallel batch broadcast still follow the normal loading path.

## Batch semantics

- `seq_len` is the hard token capacity of one normal sequence bin.
- `local_batch_size` is the number of sequences returned per DP rank and yield.
  Native BatchSampler mode treats each whole Dataset output as one such unit.
- One synchronized iterator step covers `local_batch_size * dp_size` bins.
  `balance_window_size` defaults to one. Native loading can jointly plan several
  accumulation rounds, bounded by the sampler's optimizer and epoch boundaries.
- Native BatchSampler fixes index occurrences and uses singleton reference bins;
  `external_step_source` fixes samples and lets HP preserve the original packing
  bins.
  Balancing may only rearrange that frozen sample set. It cannot skip expensive
  samples or change window membership. External sources retain one-step membership.
- In source mode, the packer must agree with metadata token accounting,
  including special tokens and reserved image placeholders. If sample-level
  placement fails, the Planner falls back to the known-feasible reference bins.
- Native BatchSampler always emits its complete batch. An external source emits one
  complete local step; HP owns metadata extraction, payload caching, and final
  packing/collation. Checkpointing is supported only by the sampler path.

The Planner consumes the step's metadata and original grouping directly:
`planner.plan(samples, reference_bins=reference_bins, step=step)`.
`samples` contains `SampleMetadata` entries with a bound `key`; `reference_bins`
contains their `SampleKey` values grouped in original data-rank/bin order. This
does not select or read additional samples. Original bins remain available for cost
comparison and fallback.

`plan()` also accepts keyword-only `target_slot_count=None`. It defaults to the
physical DP size and must be a positive integer multiple of that size. Each slot
represents one `(micro_batch, DP rank)` destination holding `local_batch_size`
non-empty bins. `plan_window()` passes `DP size * actual window rounds`, then
splits the resulting slots into one plan per micro-batch. The Planner's
`data_parallel_size` always remains the physical DP size: objectives are summed
over rounds of that size, and no Planner copy or temporary DP-size mutation is
needed. Custom algorithms retain their existing `assign(data_parallel_size=...)`
interface, where this argument denotes the target-slot count.

Metadata callbacks and precomputed metadata may leave `key=None`. Readers bind
a key to a copy for each sampling occurrence, so repeated indices and reused
metadata objects remain independent. Stream positions are stored only in
`metadata.key.global_sample_position`.
The caller is responsible for a complete, valid step and reference grouping;
the Planner does not repeat request validation. Placement capacity enforcement
and the final sample-conservation audit (when `validate=True`) remain unchanged.

## Public API

External applications pass their selected steps as `external_step_source` to
`build_distributed_dataloader`, alongside `DistributedDatasetConfig` and the
`metadata_fn`, `pack_fn` and `collate_fn` callbacks. This path
automatically evaluates node-local balancing, double buffers and stages H2D,
and yields ready device microbatches. Sample exchange occurs only when the
algorithm's objective improves by more than `min_balance_gain`.
Set `balance_group_size` to partition a large pure-DP mesh into independent
balancing groups, such as ten 1024-rank groups on a 10240-rank job. Native
BatchSampler loading does not use this option and retains its global data plane.
See [external-step integration](NODE_LOCAL_BALANCING.md) for the complete
contract. `cost_model` and `balancing_algorithm` are independently replaceable.
Native BatchSampler loading retains its selection/checkpoint mechanisms and
uses these same policy arguments; the default FLOPs model requires `model_config`.
Setting `DistributedDatasetConfig.repack_sequences=True` selects built-in text
repacking and defaults to normalized causal Attention square costs, including new PAD.

```python
from hyper_parallel.distributed_data import (
    DistributedDatasetConfig,
    SampleMetadata,
    build_distributed_dataloader,
)
```

### Native HP BatchSampler: preserve each forward/backward round

Pass `batch_sampler=` to keep the native HP sampling policy. This is required
for ahead-of-fetch metadata. For online iterable pipelines, use the
`external_step_source` API instead.

The loader validates the sampler interface, DP parameters, and cursor alignment
locally. It does not hash sampler settings or index mappings for cross-rank or
checkpoint comparisons. Restore delegates to the native sampler's
`load_state_dict`; keep its sampling policy unchanged to replay the same samples.

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
    DistributedDatasetConfig(seq_len=32768, local_batch_size=2, metadata_mode=False),
    batch_sampler=batch_sampler,
    metadata_fn=lambda sample: SampleMetadata(pack_tokens=len(sample["input_ids"])),
    collate_fn=native_collate_fn,
    cost_model=my_cost_model,
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
  -> commit Host preparation and retain the CPU batch in its window
  -> stage device batches within device_prefetch_depth and broadcast to model-parallel peers
  -> deliver the batch and publish any completed optimizer-step checkpoint
```

For DP=2 and `local_batch_size=2`, if native samplers yield `[0, 1]` and
`[2, 3]`, the default window balances only these four occurrences. Larger windows
collect consecutive rounds within the same optimizer step. Index mappings that
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
    seq_length: 32768
dataloader:
  load_balance: {}
```

`dataloader.load_balance` is an optional mapping. Its presence enables balancing,
even when empty (`load_balance: {}`); a bare YAML `load_balance:` also resolves to
an empty mapping. Omit the field to use the native DataLoader. Boolean values such
as `true` and `false` are rejected. Configuration serialization preserves the
absence of this field, so saving and reloading a disabled run does not enable it.

Put `DistributedDatasetConfig` tuning fields in this block, including
`balance_window_size`, `device_prefetch_depth`, `min_balance_gain`, `log_balance`,
`repack_sequences`, `pad_token_id`, `metadata_mode`, `broadcast_plan`,
`communication_backend`, `dataset_reader_ranks`, `planner_rank`,
`oversized_policy`, `packing_budgets`, and `balance_group_size`. Their defaults
and validation are the same as the Python configuration. In particular,
`repack_sequences` and `log_balance` default to `false`, `metadata_mode` defaults
to `false`, `balance_window_size` and `device_prefetch_depth` default to `1`, and
`min_balance_gain` defaults to `0.0`.

The Trainer derives `seq_len` from `dataset.data_config.seq_length` or the
transform's `max_seq_len`, `local_batch_size` from `training.micro_batch_size`,
and the DP topology and seed from the runtime. These fields, `drop_last` and
`dataset_already_sharded` cannot be overridden inside `load_balance`. Native
worker settings such as `num_workers` and `pin_memory` remain directly under
`dataloader`. Policy factories (`cost_model`, `balancing_algorithm`) and
`model_batch_spec` also remain directly under `dataloader`.

Move the former `dataset.data_config.load_balance` switch and all balancing
options into `dataloader.load_balance`; flatten the former
`data_config.distributed_dataloader` options into the same block. Obsolete
balancing locations are rejected. Dataset parsing, labels and model-input
options remain in `dataset.data_config`.

### Balance several gradient-accumulation rounds

Set `dataloader.load_balance.balance_window_size` to the
maximum number of consecutive forward/backward rounds to plan together. For
DP8, micro-batch size one, accumulation ten, and a three-round window:

```yaml
training:
  global_batch_size: 80
  micro_batch_size: 1
dataloader:
  load_balance:
    repack_sequences: true
    balance_window_size: 3
    device_prefetch_depth: 1
    min_balance_gain: 0.0
```

Set `dataloader.load_balance.min_balance_gain` to the minimum relative improvement
required to accept a balanced window. It defaults to `0.0`, must be finite and
in `[0, 1)`, and requires a strictly greater gain.

Each optimizer step uses windows of `3 + 3 + 3 + 1` rounds. A three-round
window jointly balances 24 original records across eight ranks and three
micro-batches. With `repack_sequences: true`, the independent sequences inside
those records can be repacked across that entire window. Without repacking,
whole records are redistributed. Each `next(loader)` still returns one normal
micro-batch; optimizer frequency and global sample membership stay unchanged.

For direct construction, pass `balance_window_size=3` in
`DistributedDatasetConfig`. Accumulation is derived from
`batch_sampler.global_batch_size / batch_sampler.global_micro_batch_size`.
The positive integer defaults to one. Values larger than accumulation are
clipped at the next optimizer boundary. Epoch tails contain only the available
complete distributed micro-batches; windows never borrow from the next epoch.
External-step loaders require `balance_window_size=1`.
In Trainer configuration, this option belongs under `dataloader.load_balance`.

Planning treats each `(micro-batch, DP rank)` as a destination. Custom assignment
policies therefore receive `DP size * window rounds` virtual ranks, ordered by
micro-batch then physical DP rank. The acceptance gate sums the per-round
objectives; the default makespan objective estimates the sum of each round's
slowest-rank cost. Samples are transferred, collated, and copied to the device
one micro-batch at a time.

For LPT policies with one pack per rank and multiple rounds, the planner also
schedules intact packs by decreasing backbone cost. It compares the unscheduled
candidate, scheduled candidate, and scheduled original layout using the sum of
per-round objectives, preserving better candidates for variance or custom LPT
objectives. Every candidate is validated before scheduling, and the final gain
must still strictly exceed `min_balance_gain` relative to the original order.
This can improve a window even when document-level repacking fails. It does not
change attention boundaries inside a pack. Single-round windows, multiple local
packs per rank, and non-LPT custom policies retain their existing behavior.

Native loading double-buffers Host balancing windows independently of device
prefetch. A persistent Host producer handles reads, splitting, metadata,
window-level planning, per-micro-batch A2A, packing/collation and final
pinned-memory preparation. The window
contains CPU batches; it does not enqueue H2D for the entire window.

On NPU/CUDA the main thread launches H2D on a copy stream and model broadcasts
on the caller's training stream, with no device worker thread or dedicated
broadcast stream. Constructor ranks stage H2D for up to
`device_prefetch_depth` future micro-batches, excluding the current training batch.
Model peers receive a batch only when `next(loader)` delivers that micro-batch;
shape/EOF headers and payload broadcasts are never part of Device lookahead.
Set `dataloader.load_balance.device_prefetch_depth` in Trainer YAML, or the same field
on `DistributedDatasetConfig` when building directly. It defaults to `1` (current
plus next); `0` stages on demand while retaining Host window buffering, and `2`
retains current plus two future batches. Values must be non-negative integers
(booleans are rejected), identical across ranks. Depth is independent of
`balance_window_size` and may exceed it. CPU loading ignores this option;
external-step loaders accept only the default value and keep their step buffering.
Data-plane and model-group collectives each retain a
single owning thread and separate streams/groups. Non-Constructors hold Host
placeholders and receive each real batch, including EOF, through the model-group
broadcast at delivery. Future Constructor inputs retain only their copy event;
the training stream waits for the current input's copy event, then orders its
model broadcast before computation. A marker after the broadcast is retained
only for explicit drains, including partially submitted broadcasts on errors;
training does not wait on that marker. Tensor storage is recorded on the consumer stream.
CPU-only loading retains window preparation and broadcast on one producer.

Advancing the iterator retires the previous input: callers must finish submitting
its uses on the consuming stream and release its references before advancing.
Each input allocation spends one of `device_prefetch_depth + 1` slots. Advancing
returns a slot with its consumption event. Optional lookahead queries the event
and skips staging if it is incomplete; only demanded input or an explicit drain
may synchronize it before allocating a replacement. A stream wait alone would still allow
CPU-side allocations ahead of completion. Deeper buffering can absorb preparation
jitter but consumes more input memory; start with the default depth of `1`.
Text and VLM Trainers consume accumulation batches one at a time and emit
per-micro-batch metric callbacks. Inputs retained by callers, side-stream work
not joined to the consuming stream, activations and communication temporaries
are outside this input buffering bound. With depth `0`, the next delivery waits
for retirement and staging; there is no Device lookahead to hide that latency.

Balancing reports are disabled by default for native and external-step loading.
Enable them explicitly in the Trainer configuration:

```yaml
dataloader:
  load_balance:
    log_balance: true
```

Direct API users can set `DistributedDatasetConfig(..., log_balance=True)`.
Reports use INFO level on `hyper_parallel.distributed_data.balance_logging`;
the Trainer enables INFO by default. Setting the logger to INFO without enabling
`log_balance` does not emit balancing reports.

Each native `[dp-balance]` report describes one balancing window, including its epoch,
one-based optimizer iteration within that epoch and micro-batch range, original/candidate/accepted
objectives, gain threshold and acceptance decision. Per-round layouts show each
DP rank's packs with token counts, sample counts and estimated LLM costs. These
costs use the configured model's units, not measured milliseconds.

Only the configured planner rank writes these reports (normally global rank 0),
including when every reader independently plans with `broadcast_plan=False`.
Reports are frozen during planning and printed when the main thread starts
consuming that window, so speculative future windows do not appear early.
An optimizer step with several windows emits each report as its window becomes
current; a short epoch tail reports only its available rounds. Logs add no collectives, device reads or
synchronization, and are not included in checkpoints. Setting `log_balance=False`
or a WARNING-or-higher level on this logger disables native statistics
formatting as well as output.

Prefetch size follows `balance_window_size`, clipped at each optimizer boundary.
The loader computes this round count once per window and passes it explicitly
to the Reader before constructing any micro-batches. The Reader reads that many
source rounds, or the remaining complete rounds at epoch end; it does not infer
another window size from its speculative sampler cursor. Planning reuses these
buffered samples throughout the window.
Accumulation ten and a three-round window prepare `3 + 3 + 3 + 1` batches in
separate windows. At most the remaining current-window inputs and one future
window are buffered, rather than two complete optimizer steps. The producer
does not start a third window until the second becomes current. Thus the loader
buffers at most twice the configured window's micro-batch count on Host.
Device lookahead is separately bounded by `device_prefetch_depth`. Neither
bound is a byte limit on variable-size payloads, temporary exchange/staging
buffers or inputs retained by training. The Reader has no separate lookahead
thread; the Host producer owns all speculative reads.
Accumulation one follows the same background pipeline. A blocked or failed
future window does not prevent delivery of current inputs; its error is raised
when that window is requested. `close()` drains production.

Epoch exhaustion stops iteration while retaining the Host thread, its data-plane
stream and the H2D copy stream for reuse. `set_epoch()` drains any
pending preparation before resetting source and delivery state, then reuses
the same Host thread and streams. Explicit `close()` or context-manager exit ends
the loader lifetime; a closed loader cannot be restarted with `set_epoch()`.
Text and VLM Trainers close their collective loader before distributed teardown.

The Trainer retains its existing per-micro-batch loss normalization. If valid
token counts differ between rounds, cross-round rearrangement can change token
weights; optimizer sample conservation does not imply identical losses or gradients.
Window-level token normalization is not enabled by this data-loading option.

Only a window ending at a complete optimizer boundary carries a checkpoint
snapshot. The public cursor advances after the final micro-batch of that window
has been returned, completing the step. Intermediate windows do not advance it. During an
incomplete step it retains the preceding optimizer boundary, so restoring that
cursor replays the whole incomplete step. A short epoch tail does not advance it.
The prefetched future window is excluded, even if the live sampler has advanced.
Checkpoints store no remaining-round count, pending window metadata, plans or raw
sample tensors. Restore reads and plans again from the saved sampler boundary;
the Dataset, split callback and planning policies must remain deterministic.
Save model, optimizer, RNG and loader state together after the optimizer update:
the loader cannot observe that update, and partially accumulated gradients are
not saved. Legacy complete-step checkpoints remain readable; partial-step
checkpoints and initial sampler cursors inside an optimizer step are rejected.

Native `FixedBatchDataLoader` and balanced loading can restore each other's
checkpoints when toggling `dataloader.load_balance`. Keep the dataset, sampling
policy and seed unchanged, and save at a complete optimizer-step boundary.
GBS may change at restoration: the absolute sample cursor is retained and the
next optimizer step uses the target GBS. The shared cursor also supports DP-size
changes for `single` sampling and `cyclic`
sampling with `data_sharding=False`. The target configuration must still satisfy
`GBS % (micro_batch_size * DP) == 0`. Per-rank shuffled buckets
(`data_sharding=True`) cannot be repartitioned this way. Restoration validates
the saved source DP before recording the target DP in subsequent checkpoints.
Dataset tails follow the target sampler's native `drop_last` policy starting
at the restored cursor, so every DP rank still receives complete micro-batches.

For example, after consuming 30 samples with GBS 10, restoring with GBS 8 consumes
the next eight occurrences and saves cursor 38. An optional
`optimizer_step_offset` in `_index_sampler_state` records the shifted boundary:
it is 6 in this example, so `(38 - 6) % 8 == 0`. The next save at cursor 46 is also
a complete step. Both loaders use this same sampler field; zero offsets are
omitted, and legacy checkpoints default to zero. Epoch changes reset the offset.
Validation checks the **source** GBS and offset, not divisibility by the new GBS.
Changing GBS or DP rebuilds the fixed loader's iterator from its delivered cursor
instead of replaying worker snapshots with the old batching geometry. With
unchanged geometry, native worker snapshots remain available, including their
internal micro-batch positions before the delivered boundary.

The loader does not recompute the Trainer's historical `global_step` or scheduler
state. Configure the remaining training budget and learning-rate policy for the
new GBS; source-sample continuity does not imply identical optimization results.
Both loaders use the existing top-level `_index_sampler_state` field for
`consumed_samples`, `epoch`, `global_batch_size` and `dp_world_size`. Balanced
loading exports only these existing native loader fields:

```python
{
    "_index_sampler_state": {
        "consumed_samples": 4,
        "epoch": 2,
        "global_batch_size": 4,
        "dp_world_size": 1,
    },
    "_iterator_finished": False,
    "dataset_state": {"dataset_size": 24, "seed": 1234},
}
```

`dataset_state` contains payload-reader compatibility checks, or `None` on ranks
without a payload reader. It is separate from the user's Dataset state in an
ordinary torchdata checkpoint; restoring an ordinary checkpoint does not
interpret the user's fields as these checks. New balanced checkpoints contain no
Reader trees or separate `global_rank`, `epoch`, `step`, or `stopped` fields.
The loader derives its micro-batch position from the committed sample count,
rebuilds Reader ownership from the current mesh and configuration, and restores
all Reader epochs from the sampler. Model peers without a Reader derive the
same cursor from the initial position and completed rounds.

Single-process fixed loading returns its original state unchanged. With workers, the top-level
cursor includes deliveries since the last snapshot; torchdata's nested snapshot
is retained unchanged for same-mode restoration. Balanced loading does not
fabricate worker snapshots. A fixed loader receiving only the shared cursor
rebuilds its iterator from that cursor.

The Trainer uses the loader's optional `state_dict_for_load()` method when
building the checkpoint load template. This no-argument method returns a fresh
mapping with `None` placeholders for sampler and iterator fields, plus legacy
Reader aliases needed only during loading. Missing progress cannot silently become
the new loader's initial zero cursor. New checkpoints are still saved through
`state_dict()` and contain no loading placeholders or duplicated sampler state.
Custom embedded-DCP callers should use `state_dict_for_load()` for the loader
subtree of their load template, then pass the loaded mapping to `load_state_dict()`.
DCP remains independent of the loader's state format and Reader implementation.

Legacy single-process fixed checkpoints and balanced Reader checkpoints also
retain a usable sampler cursor, including nonzero initial positions. Old worker
snapshots without a portable cursor are rejected for cross-mode restoration.
Legacy nested Reader sampler states remain readable and are saved in the new
top-level form after restoration; an existing top-level cursor takes precedence.
Legacy model peers without Reader state can resume in the same mode; until an
epoch reset gives them a known initial cursor, they retain their legacy progress
fields instead of exporting an inferred absolute cursor. Switching modes preserves source-sample
progress, not arbitrary Dataset/worker RNG state or identical per-step losses.
Same-mode fixed loading continues to use torchdata's own restoration rules.

### Window-aware greedy placement and bounded refinement

Ordinary LPT can fragment remaining capacity and fall back to the original
layout even when that layout is feasible. The default assignment policy,
`FeasibleLPTBalancingAlgorithm`, automatically attempts bounded feasibility repair
after greedy failure and refines successful `makespan` assignments. In multi-round
windows with one pack per rank, placements minimize the sum of scheduled round
maxima, with remaining capacity as the first tie-breaker. For offline-packed text, enable balancing and document
repacking in the Trainer with:

```yaml
dataloader:
  load_balance:
    repack_sequences: true
```

No separate repair option is needed. The builder and planner use the same
default policy. An explicitly supplied or configured `balancing_algorithm`
is preserved; callers can select `LPTBalancingAlgorithm()` for ordinary greedy
assignment without repair.

Refinement requires one pack per rank, at least two non-oversized packs, and no
additional `packing_budgets`. It uses suffix subset sums to keep two-pack
completion feasible and non-empty, including underfilled packs. There is no pack-count cutoff: a window
shares a budget of at most `max_repair_pairs` checks (default 64), including failed
or memory-limited attempts. Candidates are generated on demand from each round's
highest-cost pack and up to twelve heavy, light or nearby-cost partners, ranked
by their potential window gain. After accepting a move, repair recomputes the
bottlenecks; a pair can be checked again only if one of its packs has changed.
The complete pair set is never enumerated.

For `makespan`, each repair must lower the sum of round maxima after intact-pack
scheduling. Other objectives retain the original full-pack failure repair and round order. The
planner still validates sample conservation and capacity, compares the scheduled
reference, and applies `min_balance_gain` to the complete window. Each pair's
estimated bitset table is capped at 256 MiB to accommodate 1M-token packs with
many documents; this is not a process-memory limit. Unsupported domains retain
ordinary LPT. The deterministic check budget bounds repair attempts, not elapsed
CPU time; larger inputs still cost more to read, sort and score. The heuristic
does not guarantee an optimal layout.

Feasibility repair also applies to a single round without gradient accumulation.
Intact-pack round scheduling requires multiple rounds and also applies to
ordinary LPT. Both preserve the selected sample occurrences and use the
existing capacity checks and benefit gate. Checkpoint replay rebuilds plans from
the saved optimizer boundary, so exact replay requires unchanged planning policies.

The Trainer builds its normal `build_dataset_batch_sampler` first, including
`single`/`cyclic`, `data_sharding`, and the supported index rearrangement map.
It passes that sampler to the collective loader instead of applying another
Reader stride. Worker settings and the original `dataloader.collate_fn` are
retained. The text adapter supplies physical token lengths as `P`, with `D=0`;
it does not extract internal TND boundaries. Fixed-length GPT outputs therefore
do not automatically gain compute balance from this route. The Trainer supplies
its effective model configuration to the default FLOPs model when text repacking
is disabled. Architectures
outside that model's latent-attention scope require an explicit `cost_model`;
metadata's old proxy `cost` is not a fallback.
The Python builder accepts `cost_model` and `balancing_algorithm` directly.
AutoModels YAML can instead declare optional factories under `dataloader`:

```yaml
dataloader:
  load_balance: {}
  cost_model:
    _target_: my_training.cost.SampleCost  # user-provided factory
  balancing_algorithm:
    _target_: hyper_parallel.distributed_data.FeasibleLPTBalancingAlgorithm
    objective: makespan
    max_repair_pairs: 64
```

These factories are constructed only when `dataloader.load_balance` is present;
they may accept the runtime `model_config`. Explicit Python policy arguments
take precedence over configured factories. Omitting both selects default costs
and the default feasible-LPT policy.
The Trainer passes the foreground rank-local device for final H2D. NPU meshes
default to HCCL data transport; CPU/CUDA meshes default to Gloo data transport.
Explicit `communication_backend` settings take precedence. Accelerator model
peers use a separate HCCL/NCCL tensor-broadcast group when needed.

Current boundaries of the native-sampler path:

- `dataset[index]` stays **whole** by default. Online native loading can opt in
  to independent-sequence repacking with `split_fn` and `pack_fn`. The selected
  record set stays fixed; the original sequence grouping is the fallback when
  a feasible, sufficiently better placement cannot be found.
  The Reader retains source keys grouped by micro-batch and original pack;
  splitting replaces each pack's key with its document keys in source order.
  These groups travel with metadata, without copying payloads. The Planner
  combines them in micro-batch, logical DP-rank and local-pack order to obtain
  `reference_bins`, rather than sorting metadata to reconstruct the source layout.
- Supply an HP-compatible native sampler on every rank. Only Constructor ranks
  need the mapping Dataset, and only those ranks advance their samplers. Readers
  must currently coincide with Constructors. The native sampler owns ordering
  and DP slicing, so `dataset_already_sharded=False` is required in
  `DistributedDatasetConfig`, and `drop_last=True` is required. Without
  `split_fn`, `pack_fn` must be omitted. Pure-DP iterable sources use
  `external_step_source`.
- Supplied metadata must describe the **logical native Dataset outputs** after
  blend/shuffle/sample-index mapping, not raw document IDs. Raw `.idx` lengths
  alone do not describe GPT-internal EOD/TND boundaries. This version does not
  automatically generate native-GPT metadata. The Trainer defaults to online
  metadata and accepts `dataset.metadata` when metadata mode is enabled.
  Shared-metadata reads also require deterministic, rank-independent
  Dataset outputs.
- Save and restore through `loader.state_dict()` / `load_state_dict()`, not
  through the original sampler: its live cursor may include a prefetched window.
  Native cursor snapshots count Dataset-output occurrences, never
  tokens or fragments, and can restore at a complete optimizer-step boundary with
  an unchanged sampler policy. GBS may change; DP size may change for sequential
  or globally shuffled sampling as described above; checkpoint-file redistribution
  and arbitrary worker RNG migration are outside this cursor contract.
- Whole-record loading preserves pre-collation Dataset fields. Repacking must
  preserve each independent sequence's tokens, shifted labels, attention and
  position semantics. Neither mode promises bitwise training results. Worker
  RNG replay, batch-dependent transforms, dropout, reduction order and full NPU
  training parity are outside this sampler integration. The Trainer path rejects PP and
  a second `DynamicBatchDataLoader` selection stage.

### Repack offline-packed text within a native window

Set `repack_sequences: true` inside the `dataloader.load_balance` block.
Direct API users configure the same behavior on `DistributedDatasetConfig`:

```python
data_config = DistributedDatasetConfig(
    seq_len=262144,
    local_batch_size=1,
    repack_sequences=True,
    pad_token_id=151645,
    log_balance=True,
)
loader = build_distributed_dataloader(
    dataset, mesh, data_config,
    batch_sampler=batch_sampler,
    metadata_fn=metadata_fn,
    device=device,
)
```

`repack_sequences` and `log_balance` default to `False`; `pad_token_id` defaults
to `None` and must otherwise be a non-negative integer. The PAD setting only
affects built-in repacking. `log_balance=True` also requires INFO-level logging.
`metadata_fn` receives each split sequence and must report its actual token count.
Each Dataset output must provide one-dimensional `input_ids`, pre-shifted
`labels`, and explicit leading-zero `cu_seq_lens` covering the complete record.
The boundaries must identify independent attention sequences. No boundary is
inferred from supervision masks or guessed from token values.

The builder installs `split_packed_text_sample`, `pack_indexed_text_samples`
and, unless explicitly overridden, `collate_indexed_text_sequences`. The Trainer
forwards its balancing configuration to this same entry. Metadata and workload costs are evaluated
per independent sequence. Set `dataloader.load_balance.pad_token_id` in Trainer YAML to the Dataset's dedicated
PAD ID to remove its trailing PAD tokens before planning. Trimming requires both
that token ID and `labels=-100`; ignored prompt tokens and internal occurrences
are preserved. Without this setting source tokens are preserved and new PAD uses
ID zero. A record containing only PAD is rejected.

The default text repacking cost is `CausalAttentionCostModel`: for document lengths
`L_i` and newly appended padding length `P`, each pack costs `sum(L_i**2) + P**2`.
All linear terms and the common causal Attention coefficient are omitted; units
are token squared, not FLOPs or milliseconds. Greedy placement, bounded pair
repair, complete-pack scheduling, acceptance, and logs use this same metric.
The window objective remains the sum of each round's maximum rank cost, and
`relative_gain = (original - candidate) / original`. Each rank sums the costs of
its packs. Explicit Python or configured cost models override this default;
custom models can expose `padding_cost(padding_tokens)` in their own cost units.

The original records remain the sampler's unit of
selection and checkpoint progress. A runtime `SampleKey.sequence_index`
distinguishes sequences within each record occurrence, including repeated
Dataset indices. All sequences in a window are delivered before its sampler
cursor is committed; public checkpoint progress advances only at complete
optimizer-step boundaries.

Use `TextParallelBatch(source_type="packed_text", reset_position_ids=True)`
with `create_attention_mask_in_dataloader: true`. The `packed_text` contract
reads `input_ids`, pre-shifted `labels`, and explicit `cu_seq_lens`, independently
of whether load balancing is enabled. Compressed attention also
requires the model's runtime adapter to pass the new boundaries to attention.
Repacking currently requires online metadata (`metadata_mode: false`) and a
native `batch_sampler`; external-step sources cannot enable it. Supplying
`split_fn` with precomputed metadata is rejected. Custom `split_fn`/`pack_fn`
callbacks remain supported with `repack_sequences=False`; combining either with
`repack_sequences=True` is rejected to avoid mixing packing contracts. An explicit
`collate_fn` or `cost_model` still overrides its respective default. The config
PAD ID is not passed to custom callbacks; bind their own parameters explicitly.
Custom-callback mode retains `DefaultCostModel(model_config)` unless a cost model
is supplied.

The [Qwen3 repacking recipe](../../examples/training_demo/qwen3/train_balanced.yaml)
uses HP TextTrainer and the existing Qwen3 adapter. Its
[packed Dataset](../../examples/training_demo/qwen3/packed_dataset.py) reads a
Hugging Face Dataset saved with `save_to_disk`, containing `input_ids`,
pre-shifted `labels`, and `cu_seqlens`. It exposes these boundaries as
`cu_seq_lens` to the shared repacker. Dense document attention consumes the
rebuilt boundaries and resets positions; no model-specific unpadding is used.

The example initializes Qwen3 from a local model configuration, with FP32 main
parameters and BF16 compute. Use a matching vocabulary and adjust sequence length,
batch size, and model dimensions to available memory. The configured PAD ID must
be a dedicated padding token in the source dataset; do not use an ID also present
in real document text.

```bash
torchrun --standalone --nproc_per_node=2 --module examples.training_demo.train_text \
  examples/training_demo/qwen3/train_balanced.yaml \
  --model.config.pretrained_model_name_or_path=/path/to/Qwen3-config \
  --dataset.data_path=/path/to/packed_arrow_dataset
```

The recipe supplies a `dataloader.load_balance` block with `repack_sequences: true`,
`balance_window_size: 3`, `device_prefetch_depth: 1` (the default), and
`pad_token_id: 151645`. It uses online metadata because repacking must inspect
source boundaries. For precomputed metadata without repacking, use the
[ahead-of-fetch contract](#ahead-of-fetch-metadata) below.

With DP8 and micro batch 1, each round still selects eight original records and
produces eight packed rows. Each row is padded to `seq_len`, with ignored labels
and a separate attention segment for padding. The supplied PAD ID is also used
for this new segment. Repacking rebuilds `cu_seq_lens` from the real documents
and appends the physical row length when padding is present: document lengths
4 and 5 in a length-12 row yield `[0, 4, 9, 12]`. Attention still computes the
PAD sequence; no Attention-side unpadding is required. Native logs show real
document tokens in `seq_len`, appended tokens in `padding`, and both square
terms in `cost`. Fully occupied original rows can
leave little room for a better feasible partition; the default greedy algorithm
then retains the original grouping. Repacking does not imply a guaranteed speedup.

Native-sampler regression coverage:

```bash
  python -m pytest -q \
  tests/ut/distributed_data/test_batch_sampler.py \
  tests/ut/distributed_data/test_sequence_repacking.py \
  tests/torch/distributed_data/test_batch_sampler_gloo.py \
  tests/torch/distributed_data/test_dynamic_packing_gloo.py
```

### Native HP image-text Dataset (VLMTrainer)

VLMTrainer can use the same native-sampler path. Add `load_balance: {}` to the
existing `dataloader` configuration; keep the original Dataset, transform and collator:

```yaml
dataset:
  _target_: hyper_parallel.data.omni.build_dataset.build_online_omni_mapping_dataset
  data_path: /data/conversations.json
  data_config:
    source_type: online
  data_transform:
    _target_: hyper_parallel.data.omni.omni_transform.AutoProcessorTransform
    max_seq_len: 32768
dataloader:
  load_balance: {}
  _target_: hyper_parallel.data.batching.FixedBatchDataLoader
  sampler_type: single
  drop_last: true
  num_workers: 2
  prefetch_factor: 2
  collate_fn:
    _target_: hyper_parallel.data.batching.build_collate_fn.build_omni_collate_fn
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
- `hyper_parallel.data.omni.metadata.vlm_sample_metadata`
  describes padded width as `pack_tokens`/`P`, sets `D=0`, and exposes
  `image_patches=sum(T * H * W)` from the processor's actual `image_grid_thw`.
  Raw patches are not merged LLM image placeholders. Legacy proxy costs remain
  in the metadata for compatibility; the planner uses only the resolved cost
  model's estimates. Supply custom estimates through `cost_model`, separately
  from the metadata callback. The YAML above must be paired with a compatible
  default-model configuration or a user cost-model factory as shown earlier.
- The Trainer integration uses **online metadata**, extracted after the native
  Dataset transform. It cannot balance CPU image decoding/processing already
  performed by the Readers. Native `_TransformDataset` still performs its
  initial trainable-label filtering; no precomputed metadata is automatically generated.
  Manually supplied metadata through the lower-level API must match the
  filtered/transformed Dataset index space, not the original JSON row numbers.
- The shared Text/Omni batch path supports DP, TP and CP, with **PP=1** in
  native balanced loading. Model-specific parallelism restrictions still apply.
  This integration does not add VLM packing or video/audio adapters. The
  DP2/CP2/TP2 Gloo test checks batch distribution and CP slicing, not full
  multimodal model execution.
- Omit `dataloader.load_balance` to retain the native DataLoader. Enabled loaders own
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
python -m pytest -q \
  tests/ut/data/omni/test_metadata.py \
  tests/ut/data/vlm/test_collator.py \
  tests/torch/distributed_data/test_batch_sampler_gloo.py
```

### Native sampling and external step producers

For native HP training, add a `dataloader.load_balance` mapping. The existing Dataset and
sampler select the records in each balancing window. Whole records stay intact
by default; `repack_sequences: true` enables document-level repacking using
explicit sequence boundaries.

Custom producers can use the text packing helpers with raw source samples.
The producer must define complete steps and provide an `external_step_source`.
The source only iterates raw local steps; HP derives metadata, caches payloads,
and applies packing/collation. HP does not infer step boundaries by scanning
source metadata. Source recovery belongs to the caller's training components.

Both entry points use the same cost-model and assignment pipeline (default: LPT
with bounded repair) and automatic H2D on the selected accelerator. Native loading
buffers Host windows with independent Device lookahead; external sources buffer
complete steps. The direct distributed-data API always enables balancing. CPU execution keeps
batches on the host. The original step membership and gain gate are unchanged.
Gloo defaults to CPU batches; pass an explicit NPU/CUDA `device` to enable H2D
while retaining Gloo for data-plane communication.
Native sampler loading retains checkpoint/resume and metadata direct reads;
metadata mode still skips payload A2A.

`next(loader)` returns inputs with readiness ordered on the consumer stream.
For native loading, the Host thread pins final collated tensors; the main thread
launches non-blocking H2D on a copy stream. When that micro-batch is requested,
the caller's current training stream waits for its copy event and broadcasts
it to model peers before computing on it. Future copy events are not added to
the current training stream. The delivered batch retains a completion marker
for explicit drains; no broadcast-to-training stream handoff is needed.
The copy prefetcher retains pinned sources until their copy events complete;
device result wrappers hold only device inputs, their event and the target device.
External-step results separately retain the original CPU metering inputs and
publish them in `last_host_batch` at delivery. Tensor leaves in standard containers move recursively;
custom batch objects with a `to` method remain supported for pure DP.

The native Host thread owns metadata exchange, planning, payload A2A and CPU
construction, using a dedicated data-plane stream when required. It starts the
next Host window when the caller takes the first batch from the current window.
The main thread owns model-peer communication and training launches on the same
caller stream, using separate communication groups. No second native prefetch
thread or dedicated model-input broadcast stream is created. A backend may
still use internal communication streams. Broadcasts now follow earlier work
on the training stream, so overlap with the previous round's computation is
not guaranteed; H2D lookahead remains independent.

```python
batch = next(loader)  # prepares the next Host window and attempts device lookahead
loss = forward(batch)
backward(loss)
```

Native `prefetch()` attempts to fill Constructor H2D lookahead. CPU loading,
non-Constructors and depth-zero loading request only a Host window. Optional
staging skips unfinished Host preparation or a busy input slot instead of waiting
for either. It performs no model-peer collectives, shape-header reads or readiness
all-reduces. Peers may therefore make different numbers of optional H2D requests;
all peers must still call `next(loader)` in the same order. Data-plane window
collectives continue in the Host producer, and explicit drains can wait for peers.

For example, micro-batch 0 computes while Constructor copies micro-batch 1.
On the next delivery, peers exchange micro-batch 1's dynamic dimensions/EOF,
broadcast its tensors, and then compute micro-batch 1. This leaves MP broadcast
on the current batch's critical path, while avoiding future header synchronization
before current computation can be submitted. A speculative H2D/Host failure is
queued in order and forwarded to peers only when its batch is consumed.

When no copied input is queued, `next(loader)` waits for required preparation.
The first input and depth-zero mode have no H2D lookahead. Actual overlap depends
on Host readiness, input slots, CPU scheduling and device copy/compute resources;
a small broadcast cost is a workload assumption, not a guaranteed latency bound.

### Tensor-only model broadcasts

For custom MP batches, pass `model_batch_spec` to `build_distributed_dataloader`,
or attach it as `collate_fn.model_batch_spec`. The explicit argument takes
precedence. The default is `None`; `collate_indexed_text_sequences` and the default
`repack_sequences=True` collator automatically declare `input_ids` and `labels`
as int64 `[local_batch_size, seq_len]`, and `cu_seq_lens` as int32 `[None]`.

The Trainer also declares standard GPT `input_ids`/`labels` batches (including
configured LTR masks) and `TextPackingCollator` outputs automatically. For
instruction, multimodal or custom fields, set `dataloader.model_batch_spec` to
the layout in Python, or configure a factory in YAML:
`model_batch_spec: {_target_: my_project.data.build_model_batch_spec}`.
The factory returns the same nested `TensorFieldSpec` declaration on every rank.
This setting takes precedence over the collator declaration. Pure DP needs no
layout; arbitrary varying Python field sets are not supported by tensor-only MP.

```python
from hyper_parallel.distributed_data import TensorFieldSpec

model_batch_spec = {
    "input_ids": TensorFieldSpec((1, 262144), torch.int64),
    "cu_seq_lens": TensorFieldSpec((None,), torch.int32),
    "task": "text",  # identical static value on every rank
}
# build_distributed_dataloader(..., collate_fn=collate, model_batch_spec=model_batch_spec)
```

`TensorFieldSpec(shape, dtype, device="device")` has two required arguments.
`shape` is a tuple of non-negative integers or `None` for dynamic dimensions;
`()` represents a scalar. `dtype` is a `torch.dtype` and must match exactly.
`device` is `"device"` (the rank's training device) or `"cpu"` (Host output).
Declarations support nested dicts with string keys, lists, tuples, and static
`None`/bool/int/float/string leaves. Source fields must match the declared layout.
Every rank must use the same field order, shapes, dtypes and static values;
the layout itself is not exchanged. Varying non-tensor metadata must be converted
to tensors in the collator. Arbitrary Python objects are not supported in MP
batches. External-step sources reject this option because they support pure DP.

Gloo data transport may use a separate HCCL/NCCL group for accelerator tensors.
Fields declared `device="cpu"` return on Host even when their transport uses an
accelerator backend. Metadata/planner and payload A2A serialization are separate
from this tensor-only model-broadcast protocol and remain unchanged.

Checkpoint through the native sampler loader after the optimizer update for all
micro-batches in the step. Its snapshot excludes speculative progress; requesting
it does not wait for or consume pending batches. Call
`wait_for_prefetch()` to complete requested H2D lookahead and wait for copy,
last-delivery and Host preparation before process-group teardown. This drain
does not broadcast future batches. A context manager also releases the
persistent worker on exit.
On early close, Constructors complete the requested lookahead before stopping
their Host workers, so different optional H2D progress cannot strand a peer in
the next data-plane collective. Close discards buffered input errors; delivery
and explicit `wait_for_prefetch()` remain responsible for reporting them.
Local-step loaders retain CPU metering inputs in `last_host_batch` and do not
support checkpoint/resume.

PyTorch DataLoader execution options can be passed through
`dataloader_kwargs`:

```python
loader = build_distributed_dataloader(
    dataset,
    mesh,
    DistributedDatasetConfig(seq_len=32768, local_batch_size=2, metadata_mode=False),
    batch_sampler=batch_sampler,
    metadata_fn=metadata_fn,
    cost_model=my_cost_model,
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

DataLoader execution options belong only in `dataloader_kwargs`, not in
`DistributedDatasetConfig`. Move any existing `num_workers`, `pin_memory`,
`prefetch_factor`, and `persistent_workers` constructor arguments into this mapping.
Their defaults remain `0`, `False`, `None`, and `False`, respectively; a `None`
prefetch factor leaves the underlying PyTorch default in effect. Trainer YAML
continues to use `dataloader.num_workers`, `dataloader.pin_memory`,
`dataloader.prefetch_factor`, and `dataloader.persistent_workers`; the Trainer
passes these through `dataloader_kwargs`. The supported keys are those four options plus
`timeout`, `worker_init_fn`, `multiprocessing_context`, `pin_memory_device`,
and `in_order`; deterministic routing requires `in_order=True`. Accelerator
jobs using NPU or CUDA should generally select the `spawn` multiprocessing
context. `timeout` controls how long DataLoader waits for workers and is not a
distributed collective timeout. A custom `worker_init_fn` must be consistent
across ranks and picklable when using `spawn`; with persistent workers, PyTorch
invokes it only when each worker process is created.

DataLoader topology, configuration and cost/assignment policies are not
fingerprinted across ranks or in checkpoints. Callers must keep topology, data
semantics and policies consistent across ranks and compatible with the saved
state. Worker count, prefetch settings, and other execution options may change
when resuming; Dataset outputs must remain deterministic. Global rank ownership
is still checked, readers validate their saved state, and the native sampler
validates its restored cursor and batch sizes. Legacy loader fingerprint and
plan-ID fields are ignored; component checkpoint formats must still be
compatible. Plans expose their step, sample assignments and costs
directly; no plan digest or `last_plan_id` is generated.

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
    DistributedDatasetConfig(seq_len=32768, local_batch_size=2, metadata_mode=True),
    batch_sampler=batch_sampler,
    metadata=precomputed_metadata,
    collate_fn=native_collate_fn,
    cost_model=my_cost_model,
)
```

The sampler determines exactly which indices belong to this step. Readers look
up metadata only for those occurrences, the Planner balances them, and each
Constructor reads only its assigned payloads. No payload A2A is needed; metadata
gather and plan broadcast still occur. Do not pass `pack_fn`: each Dataset
output remains one indivisible sample. The default collator returns a tuple of
these samples.

For HP Trainer, enable metadata mode in the loader balancing configuration:

```yaml
dataloader:
  load_balance:
    repack_sequences: false
    balance_window_size: 3
    metadata_mode: true
```

The Dataset returned by its builder must expose `metadata`: an indexable
`Sequence[SampleMetadata]` with the same length and logical index order as that
Dataset. Train, validation and test splits each supply their own aligned table.
The Trainer passes this table to the distributed loader without reading payloads
to generate it. A memory-mapped or lazy index adapter can implement this sequence;
a raw `.idx` path is not accepted as metadata. Missing metadata or a table length
mismatch fails at construction; the provider must ensure index alignment.
Online mode does not access this attribute. Explicit
`metadata_fn` remains incompatible with metadata mode, and online split/repack
is not supported in this mode.

All Constructors must be able to read the same global Dataset indices.
Reader-private shards with metadata routing are no longer supported by this
builder. Compose multiple datasets and their metadata in the same global index
order, then let the native BatchSampler own DP slicing and shuffling.

### External online steps

The source API separates the external producer from HP's distributed data
semantics. The source only iterates complete local steps; an optional epoch
hook resets source selection:

```python
class MyStepSource:
    def __iter__(self): ...  # yields local_batch_size raw-sample bins
    def set_epoch(self, epoch): ...
```

Pass it as `external_step_source`. HP calls `metadata_fn` for each raw sample,
keeps payloads while preparing the step, performs distributed placement, and
then invokes `pack_fn` and `collate_fn` on the final local batch:

```python
loader = build_distributed_dataloader(
    None,
    mesh,
    config,
    external_step_source=source,  # every rank supplies its own pure-DP raw steps
    metadata_fn=metadata_for_sample,
    pack_fn=pack_one_sequence,
    collate_fn=collate_packed_sequences,
    device=torch.device("npu", local_rank),
    cost_model=my_cost_model,
)
```

The source must yield exactly `local_batch_size` non-empty raw-sample bins per
step. This node-local route automatically buffers preparation/H2D and applies
the shared gain gate. It does not provide checkpoint/resume. HP owns metadata,
payload and placement state; source epoch handling stays in the source.

The caller's training components own checkpointing and restore the source to
its next unconsumed step before building a new loader. This layer maintains
only transient metadata, payloads, and placement for preparation and delivery;
it requires no source `state_dict`, `load_state_dict`, or Reader lifecycle
methods. `set_epoch` is optional. Configure DataLoader workers on the external
producer itself.

The default packer and collator preserve the structure as a tuple of bins,
each containing the raw samples. Custom callbacks must be consistent across
ranks.

Online payload A2A defaults to CPU/Gloo. A rank-local `device`
uses the WORLD accelerator backend by default (HCCL for NPU or NCCL for CUDA);
Python payloads still incur serialization and Host/device copies. The enabled
node-local path always exchanges raw samples with Gloo and uses the training
device only for final H2D. Shared metadata mode bypasses payload transport entirely.

## Current boundaries

- Native BatchSampler requires a mapping Dataset; metadata also requires shared indices.
- Iterable/streaming online data uses an external complete-step source on every pure-DP rank.
- Node-local balancing retains at most one prefetched step, including
  final H2D. The first step waits for preparation; the original path is synchronous.
- Gloo control plane and pickle payloads; online A2A
  may use Gloo, NCCL, or HCCL.
- `drop_last=True`; every DP rank receives the same number of non-empty bins.
- Native sampler checkpoints are per rank and save the last fully delivered
  optimizer-step cursor, excluding prefetched payloads and incomplete steps.
  Sample keys and plans replay when the Dataset stream and planning policies are
  deterministic for a given epoch; arbitrary Dataset/worker RNG state is not captured.
  Every training rank must save and restore its own loader state.
- Elastic resharding and structured zero-copy tensor transport are separate
  extensions.

## Buffered node-local balancing

For automatic Host/H2D buffering around an existing rank-local loader, v1 cost
estimation, node-local LPT balancing and rank-zero DP logs, see
[the reference configuration](NODE_LOCAL_BALANCING.md). Supply `model_config`
to use the default cost, or provide `cost_model`. A custom
`balancing_algorithm` owns assignment and its scalar objective; Hyper applies
the shared `min_balance_gain` acceptance threshold before sample exchange.
Gloo uses a speculative producer thread. HCCL keeps collective launches on
the training thread, with independent data groups and a data stream. Calling
`prefetch_plan()` after forward submission and `prefetch()` after backward
submission overlaps CPU planning, payload exchange, collation and H2D with
pending computation. Call these hooks consistently on all ranks before host
synchronization; ordinary iteration remains supported without the hooks.
