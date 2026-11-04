# DCP Distributed Checkpoint Guide

**English** | [简体中文](distributed_checkpoint.zh.md)

HyperParallel provides Distributed Checkpoint (DCP): each rank saves only its own shards, accompanied by global metadata describing which rank stored each region. Loading recomputes the fragments required by the target sharding strategy, so **changing parallel strategies requires no offline conversion**. DCP also supports asynchronous persistence, reading replicated tensors once and broadcasting them, plan caching, direct loading of Hugging Face weights and torch-sharded safetensors directories, and offline conversion to and from Hugging Face safetensors.

## Core Concepts

| Concept | Description |
|------|------|
| **Planner** | Decides what to write/read, translating state_dict into `WriteItem` / `ReadItem` lists with global coordinates |
| **Storage** | Decides how to write/read, implementing plans in a concrete file format, safetensors by default |
| **Metadata** | Maps `fqn` to global shape and chunk list; the sole basis for reading across sharding strategies. The coordinator writes `.metadata` last. |
| **Chunk** | A shard's `offsets` and `sizes` in the global tensor: an N-dimensional half-open box |
| **Resharding** | Intersect the requested region with stored regions; read the resulting intersections |

## Module Overview

DCP resides in `hyper_parallel/core/distributed_checkpoint/`:

| File | Description |
|------|------|
| `api.py` | Public entry points: `save` / `async_save` / `load` |
| `planner.py` | Planner abstractions and `SavePlan` / `LoadPlan` / `WriteItem` / `ReadItem` structures |
| `standard_planner.py` | Default `StandardSavePlanner` / `StandardLoadPlanner` implementations |
| `storage.py` | `StorageWriter` / `StorageReader` backend abstractions |
| `filesystem_storage.py` | Safetensors filesystem implementations: `FileSystemWriter` / `FileSystemReader` |
| `metadata.py` | Metadata structures including `Metadata`, `ChunkStorageMetadata`, `ChunkInfo`, and `BroadcastInfo` |
| `async_persist.py` | Asynchronous saving: staging with `DataCopier` and subprocess persistence |
| `ragged.py` | Geometry adaptation for uneven `RaggedShard` partitioning |
| `offline_transform.py` | Offline Hugging Face safetensors ⇄ DCP conversion |
| `utils.py` | Deduplication, identical-shard group inference, interval intersections, and phase timing |
| `broadcast.py` | Replicated-shard broadcast: group creation/destruction, sending, and batching |
| `checkpoint_io.py` | Reading/writing individual checkpoint files: safetensors / pickle |

HF format adapters and custom loading extensions are in `hyper_parallel/components/checkpoint/`, built on the DCP infrastructure above:

| File | Description |
|------|------|
| `hf_storage.py` | Readers for complete HF weights and torch-sharded safetensors directories |
| `quantized_hf_storage.py` | Generic quantized reader that reads regions and invokes Transformers dequantization |
| `remap_planner.py` | `RemapLoadPlanner`, which generates read plans from `RemapBlock` and local shards |
| `huggingface_load_planner.py` | `HFLoadPlanner`, which compiles model weight-conversion rules into region reads or full-conversion tasks |
| `huggingface_checkpointer.py` | Orchestrates HF weight loading, selects the loader, and generates loading reports |

The dependency direction is `components/checkpoint` → `core/distributed_checkpoint`; core does not import these components.

---

## Quick Start

```python
from hyper_parallel.core.distributed_checkpoint import save, load

# Save: each rank writes its shards; rank 0 writes .metadata.
save(state_dict, checkpoint_id="/ckpt/step_1000")

# Load: update state_dict in place and return None.
load(state_dict, checkpoint_id="/ckpt/step_1000")
```

> **Note:** `load` **modifies the supplied `state_dict` in place** rather than returning a new dictionary. Preconstruct tensors with the target shapes: DCP fills only **existing keys**. Otherwise data may silently remain unloaded. Optimizer state especially needs a warmup step so `optimizer.state` is nonempty.

---

## API Details

### `save`

```python
save(
    state_dict: dict[str, Any],
    *,
    checkpoint_id: Optional[Union[Path, str]] = None,
    storage_writer: Optional[StorageWriter] = None,
    planner: Optional[SavePlanner] = None,
    no_dist: bool = False,
    use_collectives: bool = True,
) -> Metadata
```

| Parameter | Description |
|------|------|
| `checkpoint_id` | Checkpoint directory; required without `storage_writer`, and used to construct `FileSystemWriter` |
| `storage_writer` | Custom storage backend; initialized with `checkpoint_id` when both are supplied |
| `planner` | Custom planner; defaults to `StandardSavePlanner()` |
| `no_dist` | `True` selects single-process saving and forces `use_collectives` off |
| `use_collectives` | `True` (default): exchange plans across ranks and write global `.metadata`; `False`: ranks do not communicate and each writes `{rank}.metadata` |

Returns global `Metadata`. A final `barrier` ensures every rank has persisted its data before return.

### `async_save`

```python
async_save(
    state_dict, *, checkpoint_id=None, storage_writer=None, planner=None,
    no_dist=False, use_collectives=True, use_gloo=False, callback=None,
) -> AsyncSaveResponse
```

**Staging is synchronous; persistence is asynchronous.** The training process first copies tensors into host memory, then a background subprocess handles planning, communication, and disk writes. Therefore:

- Once `async_save` returns, training may modify the original `state_dict` again.
- **`.metadata` is not yet guaranteed on disk**; wait for `AsyncSaveResponse` to confirm persistence:

```python
resp = async_save(state_dict, checkpoint_id="/ckpt/step_1000", callback=lambda: print("landed"))
# ... Training continues ...
metadata = resp.get_result(timeout=600)   # Equivalent to resp.persist_completion.result(timeout=600).
```

The subprocess and join thread are **deliberately non-daemon**: script exit waits for in-flight checkpoints to persist instead of discarding them.

### `load`

```python
load(
    state_dict, *, checkpoint_id=None, storage_reader=None, planner=None,
    no_dist=False, use_collectives=True,
    broadcast_replicated_tensors=True, broadcast_groups=None,
    broadcast_batch_bytes=6 * 1024 * 1024,
) -> None
```

| Parameter | Description |
|------|------|
| `use_collectives` | **Defaults to `True`**, like `save`: exchange plans across ranks and infer replica groups. `False` makes each rank read independently without coordination. |
| `broadcast_replicated_tensors` | **Defaults to `True`**: among ranks loading identical contents, one reads and broadcasts to the others, reducing requested reads in proportion to replica count. Reader assignment balances allocated bytes rather than always choosing the lowest rank. |
| `broadcast_groups` | Precreated broadcast groups `{rank_tuple: group}`; created on demand if omitted. **Requires `broadcast_replicated_tensors=True`**, otherwise raises `ValueError`. |
| `broadcast_batch_bytes` | Shards smaller than this threshold are **batched into one broadcast**; larger or equal shards are sent separately. Defaults to 6 MiB; 0 disables batching. Applies only with broadcasting. **All ranks must pass the same value**: each computes batch boundaries independently, so mismatches deadlock. |

> `use_collectives=True` (default) requires **every rank in the default process group to call `load`**. Plan exchange uses that group; absent ranks leave the others waiting forever in gather. Ranks with no data to load must participate with an empty state_dict. Explicitly set `use_collectives=False` to load on only a subset of ranks. `save` has the same requirement.
>
> Broadcasting works only with `use_collectives=True`: replica groups are inferred from identical read requests in all-gathered load plans. Without that all-gather, identical readers cannot be identified.

If `.metadata` is missing, metadata loading falls back to `{rank}.metadata`, as produced by saving with `use_collectives=False`, and disables `use_collectives` accordingly.

---

## Persisted Files

```text
/ckpt/step_1000/
├── .metadata               # use_collectives=True: global metadata written by the coordinator
├── 0.metadata              # use_collectives=False: rank-local metadata written by each rank
├── _rank0_.safetensors     # Tensor shards
├── _rank0_.bytes           # Non-tensor data: scalars, pickled objects, etc.
├── _rank1_.safetensors
└── ...
```

`finalize_checkpoint` writes `.metadata` last. It indicates that the directory is complete; an interrupted partial write has no such file.

---

## Three Coordination Modes for Asynchronous Saving

The subprocess cannot reuse training-side HCCL/NCCL groups directly because their C++ global state is invalid after `fork`. Three coordination modes are available:

| Combination | Behavior | Use case |
|------|------|----------|
| `use_collectives=True, use_gloo=True` | Recreate an independent **CPU Gloo group** in subprocesses to exchange plans and write results | Available ports and minimal coordination overhead |
| `use_collectives=True, use_gloo=False` | Exchange through **storage**: each rank writes `LOCAL_PLAN_{rank}.pkl` / `STORAGE_DATA_{rank}.pkl` with completion markers; multiple threads poll and read in parallel | Default when another communication group is inconvenient |
| `use_collectives=False` | No cross-rank coordination; each writes `{rank}.metadata` | Local debugging or consistency managed by the caller |

`use_gloo=True` requires `MASTER_ADDR` / `MASTER_PORT`, otherwise raising `AssertionError`. Storage-based coordination requires `checkpoint_id` as a mailbox directory.

```python
# Gloo coordination.
async_save(sd, checkpoint_id=path, use_collectives=True, use_gloo=True)
# Storage-based coordination.
async_save(sd, checkpoint_id=path, use_collectives=True, use_gloo=False)
# No coordination.
async_save(sd, checkpoint_id=path, use_collectives=False)
```

---

## Reshard During Loading (Resume with a Different Parallel Strategy)

**No additional interface is needed.** Saving records each shard's global coordinates. Loading computes the requested target-layout region, intersects it with metadata chunks, and reads only overlaps.

```python
# Save with TP=4.
save(state_dict_tp4, checkpoint_id="/ckpt/step_1000")

# Load directly with TP=2; preconstruct state_dict using the new sharding.
load(state_dict_tp2, checkpoint_id="/ckpt/step_1000")
```

Covered transitions include `TP4 → TP2`, `TP4 → DP2×TP2`, `DP2×TP2 → TP2`, and resharding between `fully_shard` layouts.

---

## Read Replicated Tensors Once and Broadcast

With replicated dimensions, such as DP replicas or TP `Replicate()` parameters, multiple devices hold identical data. Broadcasting lets one rank read each piece and send it to its group, **reducing duplicated read requests from N× to 1×**:

```python
import torch.distributed as dist

load(state_dict, checkpoint_id=path)   # Both options are enabled by default.

# Precreate groups for repeated loads to avoid group creation in the loading path.
groups = {ranks: dist.new_group(ranks=list(ranks)) for ranks in my_group_ranks if rank in ranks}
load(state_dict, checkpoint_id=path, use_collectives=True,
     broadcast_replicated_tensors=True, broadcast_groups=groups)
```

`StandardLoadPlanner.build_global_plan` decides **who reads each piece**. Every rank runs the same algorithm on the same all-gathered results, requiring no extra alignment communication: assign pieces in descending byte size, each to the group rank with the fewest assigned bytes. The assignment is stored in `ReadItem.source`, which `FileSystemReader` follows.

**Groups are defined per shard, not per tensor.** A shard is the local buffer returned by `to_local()`, uniquely identified by `dest_index` (`fqn` + `offset` + `index`). Ranks holding the same `dest_index` form a group. Each broadcast sends the full buffer: `dist.broadcast` requires contiguous memory, whereas narrowed subregions are often strided. Different shards of one tensor broadcast independently; if ranks 0/2 hold shard 1 and ranks 1/3 hold shard 2, these become independent 0→2 and 1→3 broadcasts. `BYTE_IO`, which stores pickled Python objects rather than tensor storage, never participates.

**Reads and broadcasts are pipelined.** `execute_read` traverses shards in total `(fqn, offset, index)` order. A rank immediately starts an asynchronous broadcast after reading its assigned shard, then reads the next without waiting. Each broadcast therefore overlaps the next read, with up to eight broadcasts in flight. Every rank uses identical `all_plans` and ordering, so enqueue order agrees automatically. Collectives within a group execute serially in enqueue order on the stream, ensuring correctness and avoiding deadlock. Shards with no waiting peers are read last, so they do not delay broadcasts and can fill the tail while broadcasts remain in flight.

**Small shards are batched.** Broadcast startup costs about 145 µs, nearly independent of data size: both 64 KiB and 1 MiB have that cost, and transfers need several MiB to reach 33 GiB/s. Below the measured crossover of about 4.6 MiB, startup dominates transfer time. Checkpoints contain many small norms, biases, and scalars. Shards below `broadcast_batch_bytes` are copied into contiguous staging, broadcast together, then scattered back to their buffers; larger shards remain separate.

Only shards with **identical group, source rank, and dtype** can share a batch: a broadcast has one group, one source, and one contiguous buffer. Identical traversal order gives all ranks identical batch boundaries without extra communication, just as for individual broadcasts.

Measurements on four devices with 512 × 64 KiB: individual broadcasts took 81.5 ms, one batched broadcast 1.1 ms. A fused `torch._foreach_copy_` scatter-back kernel took 1.4 ms, versus 11.8 ms for individual copies.

> The cost is device memory: batches remain live until broadcast completion, consuming up to `broadcast_batch_bytes` × eight in-flight operations, about 48 MiB by default. `torch.distributed._coalescing_manager` showed no benefit on torch_npu (81.0 versus 81.5 ms), so staging is used.

A background thread also **reads one or two shards ahead**, on the Torch backend only. Reading touches checkpoint files, not planners or devices, allowing overlap with main-thread H2D copies and broadcast submission. Batches return in submission order, preserving send order. Benefits come from moving `safe_open` overhead off the main thread and triggering kernel readahead when constructing views, discussed below. When a device alternates among more than `_TORCH_FILES_KEPT` (8) files, common in resharded loading, measured speedups are 14–17%; with good file locality and no handle-cache eviction, performance is essentially unchanged.

> **Read order intentionally matches broadcast order; do not separate them.** An experiment read in file order while the main thread consumed global order, using a completion table instead of FIFO to reduce cross-file switching. A 30B model became **38% slower**, from 15.3 s to 21.1 s, across two interleaved rounds of three variants with under 0.2 s variation. First, checkpoint files already concatenate keys lexicographically, exactly matching `(fqn, offset, index)` order: 9218/9218 entries with no gaps. Reads within each file are **already sequential**. Second, file-ordered page touches and global-ordered main-thread `copy_` accesses interleave on the same file, disrupting kernel sequential-access detection and readahead. Merely replacing FIFO with a completion table, without changing order, costs **7.6%** from 18,867 dictionary insertions and lookups.
>
> Consequently, cross-file switching has no inexpensive fix in this architecture. File-ordered reads require file-ordered `copy_`, whose order is also broadcast order. Changing it serializes readers, degrading total time from `max(readers)` to `sum(readers)`.

### Where Read Time Goes: mmap and the Page Cache

The Torch checkpoint reader uses **mmap, not `read()`**. Internally, `safe_open` uses `MmapOptions::map_copy_read_only`; the entire read path makes no read system calls. Data enters the process through **page faults** on mapped memory, so disk-read timing is determined by the first access to those bytes.

In `_fetch_torch_tensor_file`, `tensor_file.get_slice(key)[tensor_slices]` returns **a zero-copy view directly over mmap**. Its `data_ptr()` lies inside the file mapping shown by `/proc/self/maps`; repeated requests return the same address. Fetching therefore costs almost nothing. **Large tensor bytes are read later by `copy_` in `_apply_fetched`**, as measured for a 155.6 MiB embedding after clearing page cache with `posix_fadvise(DONTNEED)`:

| Statement | Time | Disk reads |
|---|---|---|
| `get_slice(key)` | 0.03 ms | 0 |
| `[tensor_slices]` | 0.74 ms | 0 |
| `target_tensor.copy_(tensor)` | 127.60 ms | 147.7 MiB (179 major faults) |
| Repeat the same `copy_` | 25.29 ms | 0 |

The final two rows differ fivefold because of the **page cache**. The kernel retains file pages in available memory, shown as reclaimable `buff/cache` by `free`; subsequent accesses use memory without changing disk counters.

> Zero disk reads for `[tensor_slices]` do not imply that view construction causes no page fault. The embedding begins the file, and header parsing in `safe_open` had already brought its first page into cache through readahead. That access caused a minor fault, leaving disk counters unchanged. View construction touches a page, but that page need not require disk I/O.

**The table applies only to large tensors.** Constructing a view touches one mapped page and triggers **kernel readahead**: `read_ahead_kb=4096` on the measured host brings in up to 4 MiB at once. For tensors smaller than that window, view construction reads everything and `copy_` needs no disk sectors. For much larger tensors, only the beginning is prefetched; `copy_` faults in the remainder. For 500 tensors from the middle of a file, totaling 381 MiB with MoE expert weights around 0.76 MiB each and avoiding header readahead:

| Path | `get_tensor(k)` / `get_slice(k)` | `[slice]` / `[spec]` | `copy_` |
|---|---|---|---|
| `get_tensor(k)[slice]` | 170.54 ms, **387.5 MiB** | 3.41 ms, 0 | 0 |
| `get_slice(k)[spec]` | 0.27 ms, 0 | 181.22 ms, **387.5 MiB** | ~0 |

Both paths read identical bytes at different statements. `get_tensor(k)` constructs the view immediately and touches the mapping; `get_slice(k)` retrieves metadata only and constructs the view at `[spec]`. Applying `[slice]` to an existing Torch tensor only changes sizes, strides, and storage_offset and **never faults in data**.

**The background thread really does prefetch, through the kernel.** It copies no bytes; constructing a view touches the mapping, and kernel readahead does the rest. When most checkpoint tensors fit within the readahead window, especially for MoE, this brings entire batches into page cache before main-thread `copy_`. In a 2.0 GiB case with 2486 tensors, view construction read 1903 MiB and `copy_` needed another 155 MiB. A single tensor far larger than the window, such as an embedding, still requires page faults during `copy_`.

> This depends on a sufficiently large readahead window and tightly packed tensors in the file. Deterministic prefetch would require explicit `madvise(MADV_WILLNEED)` in the reader thread or actual byte copies, with the latter consuming another host-memory buffer.

**`get_slice(k)[spec]` and `get_tensor(k)[spec]` measured equivalently** with safetensors 0.8.0: both return mmap views at identical addresses, take fractions of a millisecond with zero disk reads to construct the view, and have the same subsequent `copy_` time and read volume. The difference is that `get_slice` permits `get_shape()` without touching data, useful for key and shape validation.

**The slicing dimension matters.** A dimension-0 prefix slice, as in row parallelism, reads only the requested portion: half an embedding reads 74.98 MiB versus 148.88 MiB for the full tensor. A dimension-1 column slice produces a strided view whose `copy_` traverses every page, so **the entire tensor is read**, 148.88 MiB, just as without slicing. Column sharding saves no I/O during resharded loading.

This also determines broadcast savings. **DP replicas on one host read the same bytes from the same file.** Without broadcasting, the second rank hits page cache entirely instead of rereading disk. Eight-device measurements showed identical **56.9 GiB** in `/proc/diskstats` with broadcasting on or off. Broadcasting therefore saves no disk I/O for a single-host layout, with end-to-end differences around one second; before read/broadcast pipelining it could even hurt. Broadcasting matters when page cache cannot absorb duplication: **cross-node DP, shared storage/NFS, or checkpoints larger than memory**.

> Measure bytes with `/proc/diskstats`, not `majflt × 4 KiB`: one blocking fault can pull an entire readahead window into cache. The 1.866 GiB measurement above reported only 951 major faults. Anchor device matching to `^nvme[0-9]+n[0-9]+$` to avoid counting both disks and partitions and doubling totals. `posix_fadvise(DONTNEED)` also skips pages still mapped in page tables. Drop `safe_open` handles before measuring cold reads; one process per configuration is safest.

---

## Deduplication and Plan Caching

`StandardSavePlanner` enables two optimizations by default, both requiring `use_collectives=True`:

```python
StandardSavePlanner(
    enable_plan_caching=True,    # Cache the final plan and metadata.
    remove_redundancy=True,      # Deduplicate across ranks.
    save_to_minimum_rank=False,  # Ownership policy after deduplication.
)
```

- **Deduplication**: Write one copy of shards held by multiple ranks. By default, assign ownership to the plan with the **fewest planned bytes**, balancing load instead of concentrating replicas on rank 0. `save_to_minimum_rank=True` always selects the lowest rank.
- **Plan caching**: Class-level caching uses the state_dict key set. With unchanged model structure, subsequent saves skip the entire `build_local_plan` / `all_gather` / `build_global_plan` phase, the only communication that grows linearly with world size. For asynchronous saves, subprocess-generated caches return to the parent with results for reuse.

**Two automatic fallbacks** require no intervention but should be understood:

- `use_collectives=False` disables deduplication and plan caching because no global view exists.
- DTensors with `ragged_shard` in state_dict disable plan caching because geometry may change between steps.

---

## FSDP / HSDP Integration

DCP naturally composes with `fully_shard`: sharded parameters are already DTensors carrying complete layout information.

```python
from hyper_parallel import fully_shard, init_device_mesh
from hyper_parallel.core.distributed_checkpoint import save

mesh = init_device_mesh("npu", (dp_size,), mesh_dim_names=("dp",))
model = fully_shard(model, mesh=mesh)

save(model.state_dict(), checkpoint_id="/ckpt/step_1000")
```

---

## Load Safetensors Weights Directly

Two types of safetensors directories not written by DCP can be passed directly to `load`, without offline conversion first. Their directory rules conflict, so each has its own reader:

| | `HuggingFaceStorageReader` | `TorchShardedSafetensorsReader` |
|------|------|------|
| Input | Hugging Face weights produced by `save_pretrained`: each tensor is complete and appears in exactly one file | Unmerged shards written by torch `HuggingFaceStorageWriter(save_distributed=True)`: one file per rank, tensors spanning multiple files, no index |
| Files read | Files listed in `model.safetensors.index.json`; without an index, only `model.safetensors`, ignoring other safetensors files in the directory | All `*.safetensors` directly in the directory, without recursion |
| Tensor positions | If file headers contain `DCP_SHARDING_INFO`, all offsets must be 0 | Every file must contain `DCP_SHARDING_INFO`; shards sharing a tensor name are assembled by offsets and must cover the whole tensor without overlap |
| Wrong directory type | Nonzero offsets for any tensor raise `ValueError`; absence of both an index and `model.safetensors` raises `FileNotFoundError`. Both errors suggest `TorchShardedSafetensorsReader` | An index in the directory, or a file missing `DCP_SHARDING_INFO`, raises `ValueError` suggesting `HuggingFaceStorageReader` |

```python
from hyper_parallel.components.checkpoint import (
    HuggingFaceStorageReader,
    TorchShardedSafetensorsReader,
)
from hyper_parallel.core.distributed_checkpoint import StandardLoadPlanner, load

load(model.state_dict(), storage_reader=HuggingFaceStorageReader("/models/Qwen3-8B"))

# With tie_word_embeddings, HF stores tied weights only once; allow the missing keys.
load(model.state_dict(), storage_reader=HuggingFaceStorageReader("/models/Qwen3-8B"),
     planner=StandardLoadPlanner(allow_partial_load=True))

# A directory of torch-written shards that have not yet been merged.
load(model.state_dict(), storage_reader=TorchShardedSafetensorsReader("/ckpt/step_1000/sharded"))
```

When torchtitan saves in HF format, it first writes shards under the checkpoint's `sharded/` directory, then merges complete weights with an index. These directories correspond to the two readers above.

These readers change only the source of metadata. `FileSystemReader` deserializes `.metadata`; the safetensors readers read each file's JSON header for tensor dtypes and shapes, constructing the same `Metadata` under each tensor's stored name. Building metadata reads only headers, not tensor data. Subsequent planning, resharding, read-once replica broadcasting, and per-file reading remain unchanged, so:

- Targets can be DTensors with arbitrary sharding, and each rank reads only its own shard;
- Target dtype can differ from the file, such as bf16 files and fp32 parameters; conversion occurs during `copy_`.

**This layer only reads; it does not transform weights:**

| Limitation | Description |
|------|------|
| Names and shapes must match | state_dict keys must exactly match tensor names in the files, with identical global shapes. Renaming, merging q/k/v into `linear_qkv`, and stacking expert weights are outside this layer |
| Missing keys | Raise `Missing key in checkpoint state_dict` by default; pass `StandardLoadPlanner(allow_partial_load=True)` to allow them |
| Quantized weights | Read exactly as stored: FP8 weights remain FP8 values and scales are separate tensors, without dequantization |
| Read-only | Saving in HF format is not yet supported |
| Process group required | Like other `load` calls, requires `init_process_group`, even for a single process |

`load_metadata` raises an error instead of reading incorrect data when the same tensor shard appears in two files, shard dtypes differ, shards fail to cover the complete tensor due to missing files or overlap, a file containing `DCP_SHARDING_INFO` omits a tensor's offsets, or a dtype has no corresponding torch type.

---

## Online Dequantization During DCP Loading

Loading an HF directory through `HyperAutoModelForCausalLM.from_pretrained()` or
`HyperAutoModelForImageTextToText.from_pretrained()` uses DCP by default.
Ordinary BF16/FP32 directories continue to use `HuggingFaceStorageReader`. If the directory's `config.json` declares a supported
`quantization_config`, the shared `QuantizedHuggingFaceStorageReader` is selected automatically.

Currently supported under `quant_method: fp8` are two-dimensional E4M3 weights, packed FP4 weights, and their scales.
This capability is reused by storage format and is model-independent; unsupported quantization methods raise errors.
For example, DeepSeek V4.1 FP8/FP4 weights and UE8M0 scales can load directly without first generating an offline BF16 copy.

```text
Quantized file headers → logical BF16 Metadata → existing conversion and sharding planning
→ Read required weight/scale regions → Transformers Fp8Dequantize
→ Crop to requested regions → existing DCP copies and final completeness checks
```

FP4 has twice as many logical columns as packed storage columns. Planner offsets and lengths use logical coordinates; the Reader handles
packed-coordinate conversion, block alignment, and blockwise dequantization. Weight naming, expert merging, and high-performance module conversions retain their existing rules.
The model is first constructed from config on meta and sharded, without having Transformers read weights in advance.

Dequantization uses the installed Transformers `Fp8Dequantize` implementation, whose version must support the format.
Explicitly passing `quantization_config` at the DCP construction entry to request quantized-module initialization or requantization is currently rejected;
online decoding follows the checkpoint's own configuration. Decoding produces BF16, followed by dtype handling according to target parameters and
`model_init_dtype`. The official operator preserves FP4 negative zero, so zero sign bits may differ from those produced by
offline tools that normalize negative zero.

The cropped-layer DeepSeek example accepts either the official quantized directory or an existing BF16 directory directly:

```bash
bash examples/training_demo/deepseek_v41/run_deepseek_v41_vlm_online_load_hf.sh /data/DeepSeek-V4.1-Flash
```

---

## Offline Format Conversion (Hugging Face ⇄ DCP)

`offline_transform` provides single-process offline conversion for initializing training from HF weights and exporting training outputs to HF:

```python
from hyper_parallel.core.distributed_checkpoint.offline_transform import (
    convert_full_checkpoint_to_dcp,        # Full weight file / HF directory → DCP
    full_state_dict_to_dcp_format,         # Full in-memory state_dict → DCP
    dcp_to_full_state_dict,                # DCP shards → merged full state_dict
    parse_checkpoint_from_huggingface,     # Read HF safetensors: single file or shards + index
    save_state_dict_as_huggingface_format, # Write HF-style safetensors, optionally sharded
)

# HF → DCP
convert_full_checkpoint_to_dcp("/hf/Qwen3-30B", "/ckpt/dcp_init", src_platform="huggingface")

# DCP → HF
full_sd = dcp_to_full_state_dict("/ckpt/step_1000")
save_state_dict_as_huggingface_format("/export/hf", full_sd, max_shard_size="5GB")
```

`src_platform` accepts `"huggingface"` for HF directories or `"torch"` for full PyTorch checkpoints in pickle or `.safetensors` format.

---

## Custom Extensions

- **Storage backend**: Subclass `StorageWriter` / `StorageReader`, implement `execute_write`, `execute_read`, `load_metadata`, and related methods, then pass through `storage_writer=` / `storage_reader=`.
- **Planner**: Subclass `SavePlanner` / `LoadPlanner`, or derive from `StandardSavePlanner` / `StandardLoadPlanner` and override behavior.
- **Asynchronous staging of custom objects**: `async_save` dispatches copies by type through `DataCopier`. Unregistered types use generic deep copy with a warning. Register custom types explicitly:

```python
from hyper_parallel.core.distributed_checkpoint.async_persist import DataCopier

@DataCopier.register(MyState)
def _copy_my_state(obj):
    return MyState(...)   # Return a copy in host memory.
```

> `DTensor` subclasses `torch.Tensor`. Dispatch resolves exact types first, then the most-derived registered base class. For custom inheritance hierarchies, verify that the intended handler is selected.

---

## Debugging and Performance

**Enable per-phase timing logs** as the first step in diagnosing save bottlenecks:

```bash
export HP_LOG_CONFIG=DCP:INFO
# Example output:
# [INFO] [HP-DCP]: [rank=0] >>> func build_local_plan cost 0.0003 seconds
# [INFO] [HP-DCP]: [rank=0] >>> func execute_write cost 0.0061 seconds
```

Key phases include `save`, `_save_impl`, `build_local_plan`, `all_gather_object`, `build_global_plan`, `execute_write`, `execute_read`, and `build_staged_state_dict`, timed separately on every rank. Cache hits log `Hit final plan and metadata cache.`.

**Performance recommendations:**

1. Use `async_save` to overlap persistence with training computation; the training process pays only staging costs.
2. Keep `enable_plan_caching=True` so planning communication is paid only for the first save.
3. Keep `broadcast_replicated_tensors` enabled, as by default, when replicated dimensions exist, trading communication for I/O.
4. Prefer resharding during load when changing parallel strategies; use `offline_transform` only when an offline merge and reshard is necessary.

---

## Important Notes

1. **`load` fills existing keys only**: Preconstruct target tensors with correct shapes. Warm up optimizer state so `optimizer.state` is nonempty before loading.
2. **Identical FQNs are deduplicated across ranks**: This is appropriate for identical weights and optimizer state, but **rank-specific data such as RNG state** cannot be embedded under the same key or only one copy survives. Save it separately per rank or add a rank suffix to keys.
3. **Returning from `async_save` does not mean persistence is complete**: Staging is complete and state_dict can be modified, but wait for `get_result()` to confirm disk completion.
4. **`save` and `load` both default `use_collectives` to `True`**. Set `False` explicitly when independent rank-local operation is intended.
5. **`no_dist=True` forces `use_collectives` off**.

---

## Broadcast Loading with Pipeline Parallelism (PP)

PP breaks the assumption that all ranks have identical state_dict keys: each stage holds only its assigned layers, with `embed` on the first
stage and `head` on the last. Stages share almost no keys. The workflow handles this through three properties:

- `build_local_plan` is **entirely local**, and `all_gather_object` already permits different contents on different ranks.
- `build_global_plan` groups replicas by `dest_index` (`fqn` + `offset` + `index`), so stage 0 parameters
  naturally form groups within stage 0 and stage 1 parameters form groups within stage 1; **the groups are disjoint**.
- The ordering invariant is required only **within each group**. Members of a group belong to the same stage and traverse the same subsequence,
  so **stages need no mutual alignment**; one stage may run far ahead of another.

### Communication Groups: Create, Use Once, Destroy

Replica groups within a stage, such as its four devices or two devices in one TP column, usually do not correspond to existing communication groups
and must be created. **`dist.new_group(ranks=...)`** creates **exactly the supplied ranks**,
then returns the group to the caller.

This path **does not use `create_group`** from `core/dtensor/_utils.py`, which does more than needed: it expands a rank-list **template** into a partition of
WORLD and caches every resulting group process-wide. Both consequences are unsuitable for PP. Caching retains one-use
communicators and their device memory until the job ends, while template expansion **rejects** some PP-generated
rank lists, such as tied parameters across nonadjacent stages (`[0,1,6,7]` → `ValueError: Template must have
consistent intra-group step`). `new_group` avoids both problems.

Within `load()`, `broadcast_groups_for_load(...)` scopes groups and destroys them after reading in rank-tuple order. Every rank computes the same order,
and only group members initiate destruction after reaching the end of reading together. Caller-provided groups passed through
`broadcast_groups=` **are not owned** by this load and remain untouched.

**Group creation is a global collective operation.** Its entry condition is
`broadcast_replicated_tensors and use_collectives and world_size > 1`, three values shared by all ranks,
not whether the current rank has a shard to broadcast. Even a stage with fully sharded parameters and no replica shards must
participate, or another stage will wait indefinitely. Such ranks receive the backend's nonmember marker from `new_group`,
`-100` on Torch, which is neither stored in the group dictionary nor destroyed.

---

## Related System Tests

```bash
# Validate without NPUs using CPU and Gloo.
export HYPER_PARALLEL_TEST_DEVICE_TYPE=cpu
python -m torch.distributed.run --nproc-per-node=4 -m pytest -s \
    tests/torch/distributed_checkpoint/dcp_async_save.py::test_dcp_async_save_twice_reuses_the_plan_cache
```

| Test file | Coverage |
|------|------|
| `tests/torch/distributed_checkpoint/dcp_save_and_load.py` | Mixed saving/loading of DTensors, ordinary Tensors, and scalars; loading across sharding layouts |
| `tests/torch/distributed_checkpoint/dcp_async_save.py` | Three asynchronous coordination modes and plan-cache reuse |
| `tests/torch/distributed_checkpoint/dcp_broadcast_load.py` | Broadcast loading of replicated tensors with precreated or lazily created groups |
| `tests/torch/distributed_checkpoint/dcp_pipeline_load.py` | Read/broadcast pipelining: exceeding the in-flight limit, interleaved groups, nontensor state, and repeated loading |
| `tests/torch/distributed_checkpoint/dcp_pp_stage_load.py` | Broadcast loading with PP on 8 devices (`pp=2 × dp=2 × tp=2`): different stage parameters, a stage without replicated shards, interleaved shard order across stages, and tied parameters across stages |
| `tests/torch/distributed_checkpoint/dcp_resharding_api.py` | Resharded reads with multiple mesh combinations |
| `tests/torch/distributed_checkpoint/_test_dcp_tp_dp.py` | `fully_shard`, optimizer state, and flatten_state_dict |
| `tests/torch/distributed_checkpoint/dcp_plan_cache_minimal_api.py` | Plan-cache hits and isolation of model/optimizer caches |

Worker files omit the `test_` prefix or begin with `_`. Corresponding `test_*.py` wrappers launch multiprocess execution through `parallel_run`; the files above can also be run directly with `torch.distributed.run`.
