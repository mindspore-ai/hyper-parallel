# HyperParallel Data

**English** | [简体中文](README.zh.md)

`hyper_parallel.data` converts raw records or Indexed Datasets into model inputs and loss inputs for one
forward/backward pass. Modules are organized by data responsibility rather than model name:

```text
source
  -> transform
  -> sample selection / packing / collate
  -> DP / CP / TP data parallelism
  -> runtime inputs
  -> model
```

## 1. Directory Structure and Responsibilities

```text
data/
├── indexed/    # .idx/.bin, Indexed split, sample index, blend
├── online/     # Raw Mapping/Iterable sources, file/HF loading, and source blending
├── text/       # LLM tokenizers, chat templates, transforms, and Dataset entry points
├── omni/       # Image/video/audio processor transform lifecycle and Omni Dataset entry points
├── batching/   # Candidate pools, packing, collators, DataLoaders, and get-batch
├── parallel/   # Mapping DP samplers, DataLoader ownership, CP sharding, and TP broadcast
└── tools/      # Offline Indexed preparation and inspection tools
```

| Module | Input | Output | Outside its responsibility |
| --- | --- | --- | --- |
| `online` | Files, directories, globs, Hub Datasets | RawSample source | Tokenizers and model semantics |
| `text` | RawSample + tokenizer/template | Text ModelSample | Multimodal processors |
| `omni` | RawSample + processor | Omni planning/ModelSample | Source I/O and DP sampling |
| `batching` | ModelSample | Collated batch | Raw file loading |
| `parallel` | Dataset index or collated batch | Rank-local batch | Model transforms |
| `indexed` | `.idx/.bin` prefix | Fixed training samples | Online transforms |

`online` provides sources only, without LLM, VLM, or model-specific semantics. Model-specific processors/transforms
belong in model adapters, for example DeepSeek-V4.1:

```text
hyper_parallel/models/deepseek_v41/adapter/
├── processor.py
└── transform_fn.py
```

For DeepSeek-V4.1 Online VLM sample contracts, image expansion, label masks, and batch fields, see
[`docs/guide/data/deepseek_v41_vlm_online_data_guide.md`](../../docs/guide/data/deepseek_v41_vlm_online_data_guide.md).

## 2. Support Status

| Path | Status | Main entry point |
| --- | --- | --- |
| Indexed Text | Supported | `build_indexed_text_dataset` |
| Online Mapping Text | Supported | `build_online_text_mapping_dataset` |
| Online Iterable Text | Supported | `build_online_iterable_dataset` |
| Online Mapping Omni | Supported | `build_online_omni_mapping_dataset` |
| Online Iterable Omni | No top-level Dataset entry point yet | Reuse the `online` Iterable source interface |
| Offline Omni metadata | Transform lifecycle reserved | Dataset/provider not yet implemented |
| Omni TP/CP get-batch | Implemented; multi-device training validation pending | `OmniParallelBatch` supports only PP=1 |

Omni currently integrates image/VLM fields; directory structure and transform contracts can extend to video and audio.

## 3. Two Source Types

### 3.1 Mapping

Mapping sources have stable lengths and integer indices:

```text
source[index]
  -> RawSample
  -> lazy transform
  -> ModelSample
```

Local JSONL uses native byte-offset indexing, avoiding Arrow schema coercion for heterogeneous JSONL.
JSON, Parquet, CSV, Arrow, and Hub Datasets use Hugging Face `load_dataset(..., streaming=False)`.

Mapping processing order:

```text
Determine train/valid/test for each source
  -> construct a global blend index within each named split
  -> source-local deterministic shuffle/repeat
  -> shared BatchSampler shards indices across DP
  -> lazy transform
```

A single path defaults entirely to train:

```yaml
dataset:
  _target_: hyper_parallel.data.text.build_dataset.build_online_text_mapping_dataset
  data_path: /data/train/*.parquet
  data_config: {}
```

Ratio splits use Hugging Face split expressions and apply only to Mapping sources:

```yaml
dataset:
  _target_: hyper_parallel.data.text.build_dataset.build_online_text_mapping_dataset
  data_path: /data/all/*.parquet
  data_config:
    split: "98, 1, 1"  # train / valid / test; must total 100.
```

For independently prepared splits, use a path mapping; `valid` and `test` are optional:

```yaml
dataset:
  _target_: hyper_parallel.data.text.build_dataset.build_online_text_mapping_dataset
  data_path:
    train: /data/train/*.parquet
    valid: /data/valid/*.parquet
    test: /data/test/*.parquet
  data_config: {}
```

Hub Datasets and local paths use the same `data_path` field:

```yaml
dataset:
  _target_: hyper_parallel.data.text.build_dataset.build_online_text_mapping_dataset
  data_path: Salesforce/wikitext
  data_config:
    config_name: wikitext-2-raw-v1
    split: "98, 1, 1"
    cache_dir: null
```

Multiple Mapping sources use deterministic weighted scheduling:

```yaml
data_config:
  sources:
    - data_path: /data/domain_a/*.parquet
      weight: 0.7
    - data_path: /data/domain_b/*.parquet
      weight: 0.3
```

The top-level blend index stores `(source_id, source_logical_index)`. Each source's logical index then maps to
physical samples after deterministic shuffle/repeat, matching Indexed/Megatron's global-order-before-DP-sharding approach.

### 3.2 Iterable

Iterable sources use Hugging Face `load_dataset(..., streaming=True)` and have no randomly accessible global index:

```text
each source
  -> DP / DataLoader-worker shard
  -> local buffer shuffle
  -> local weighted blend
  -> lazy transform
```

Iterable therefore does not support runtime ratio splits. Input paths or native Hub splits must define train/valid/test.
The stream owns epoch, shuffle-buffer, and resume semantics; Mapping's global shuffle index does not apply.

Multiple Iterable sources default to `local_weighted`; Hugging Face-style
`global_interleave` is also available. These combine sources rather than define train/valid/test splits.

## 4. Text Path

Text transforms convert RawSample into `input_ids/labels`:

```text
Online source
  -> PlaintextTransform / TextConversationTransform
  -> input_ids [sample_length]
  -> labels    [sample_length]
  -> FixedBatchDataLoader or TokenBatchLoader
  -> TextPackingCollator
  -> input_ids/labels [1, packed_length] + cu_seq_lens
  -> TextParallelBatch
```

`TextParallelBatch` handles both Online Text and Indexed Text:

```text
The DataLoader owner reads a batch
  -> parse sequence boundaries
  -> CP sequence shard
  -> TP broadcast
  -> position_ids / loss_mask
  -> AttentionRuntime
  -> (model_inputs, loss_inputs)
```

`FixedBatchDataLoader` fixes the number of samples per batch; `TokenBatchLoader` selects a variable number of complete samples
from a candidate pool according to a token budget. Neither silently truncates an oversized sample.

## 5. Omni Transform Lifecycle

`OmniDataTransform` uses Energon-style hook composition. Implementations must satisfy one of three modes:

| Mode | Implemented hooks | Before candidate pooling | After selection |
| --- | --- | --- | --- |
| Eager Online | `encode_sample` | Full encoding | Return unchanged |
| Deferred Online | `preencode_sample` + `postencode_sample` | Planning metadata only | Full encoding |
| Offline metadata | `postencode_sample` | Retain existing metadata | Complete encoding |

Implementing both `encode_sample` and `preencode_sample/postencode_sample` is forbidden. Implementing only
`preencode_sample` without `postencode_sample` is also invalid.

`encode_batch` is the final batch adapter: after packing/collation, it converts generic fields into the model's forward contract.
DeepSeek-V4.1, for example, converts image patch lengths to offsets and generates image-to-batch mappings here.

Default AutoProcessor behavior:

```python
processor = AutoProcessor.from_pretrained(
    pretrained_model_name_or_path,
    trust_remote_code=True,
)
```

If the processor lacks `chat_template`, reuse `processor.tokenizer.chat_template`.
`AutoProcessorTransform` handles `apply_chat_template`; model adapters implement model-specific labels, media metadata, and batch
contracts.

Before transformation, relative image/video/audio paths are resolved against the RawSample source file's directory. Remote URLs, data URLs,
and absolute paths are unchanged.

## 6. Omni packing

Current Online Mapping Omni flow:

```text
build_online_mapping_source
  -> _OmniMappingDataset
  -> transform strategy prepares candidate
  -> OmniPackingLoader
       -> PackingCandidateBuffer
       -> PackingSelector.select_samples_to_pack
       -> dataset.encode_selected_sample
       -> SamplePacker.pack_selected_samples
       -> OmniCollator
       -> dataset.encode_batch
  -> OmniParallelBatch
  -> model
```

The default `FirstFitPackingSelector` uses `packing_length`, falling back to `input_ids` length when absent.
`SamplePacker` concatenates tokens, labels, and modality tensors, generates `cu_seq_lens`, and converts
`*_token_starts` from sample-local to packed-sequence coordinates.

The candidate pool emits a micro-batch only when both conditions hold:

```text
len(buffer) >= min_buffered_samples
and
sum(sample_cost) >= token_budget
```

Default:

```text
token_budget = micro_batch_size * data_transform.max_seq_len
```

Override the selection budget independently in DataLoader configuration:

```yaml
dataloader:
  _target_: hyper_parallel.data.batching.OmniPackingLoader
  token_budget: 128
  min_buffered_samples: 1
```

`token_budget` controls selection only, not truncation. If the first sample alone exceeds the budget, first-fit emits it as a standalone batch.
The transform's `max_seq_len` still validates the model's maximum allowed sample length.

## 7. Omni Distributed Constraints

`OmniParallelBatch` reads identical DP samples on TP0 at each CP coordinate. `CPBatchSharder` shards only token fields,
then `TPBatchBroadcaster` sends CP-local tokens and complete image fields to TP peers at that CP coordinate.
`pixel_values`, image grids, and global `image_token_starts` are not CP-sharded. Current requirements:

```text
PP=1; TP/CP paths require distributed process groups.
```

`get_batch.encoder_dp` defaults to `false`: every TP×CP rank receives complete images and computes ViT independently,
inserting only visual embeddings corresponding to local CP tokens into the text sequence. `true` is reserved for whole-image bucketing
and all-to-all routing of visual embeddings; it is unimplemented and raises immediately when configured.

DeepSeek-V4.1 attaches `DeepseekV41Runtime` through YAML `OmniParallelBatch.runtime_input_adapter`
for packed attention and image insertion coordinates. Every TP×CP rank repeats full ViT computation during forward,
then replaces embeddings at the intersection of global image spans and local CP token intervals. Gradient synchronization for ViT parameters replicated
across TP/CP still requires multi-device training validation.

EP/FSDP can be enabled through model parallel plans, but every rank must call distributed submodules in identical order and count.

DeepSeek-V4.1 currently invokes the vision tower once per image:

```text
for each image:
    vision(...)   # Triggers collectives when vision is FSDP-sharded.
    aligner(...)
```

With FSDP-sharded vision, each rank's micro-batch must therefore contain the same number of images. Different image sizes change only
computation volume; different image counts change the number of vision/FSDP collectives and may deadlock with
subsequent EP collectives.

The current DeepSeek-V4.1 smoke configuration uses one sample and one image per rank-local micro-batch:

```yaml
data_transform:
  _target_: hyper_parallel.models.deepseek_v41.adapter.data.transform_fn.build_deepseek_v41_omni_transform
  max_seq_len: 4096

dataloader:
  _target_: hyper_parallel.data.batching.OmniPackingLoader
  token_budget: 128       # Emit the first complete sample immediately; do not truncate to 128.
  min_buffered_samples: 1
```

This is the current baseline, not a general multi-image packing solution. The target multi-image FSDP approach follows PanGu:

```text
Real image count on this rank
  -> align global image count before the first vision call
  -> fill missing slots with minimal valid dummy images
  -> real/dummy mask
  -> run the same number of vision/aligner calls on every rank
  -> write only real images back to language-token spans
```

Dummy inputs must retain zero-gradient dependencies so FSDP forward and backward call counts both match. This image-call padding
is not implemented in HyperParallel; do not re-enable DeepSeek FSDP packing with variable image counts per rank until it is.

## 8. Indexed Text

The Indexed path uses two files sharing one prefix:

```text
<prefix>.idx  # dtype, length, pointer, document index
<prefix>.bin  # Contiguous token payload
```

Two mid-level Dataset types:

| Data form | Configuration | Dataset | Runtime behavior |
| --- | --- | --- | --- |
| Variable-length documents | `is_dataset_from_mr: false` | `GPTDataset` | Dynamic samples from document/sample/shuffle indices |
| Offline fixed-length records | `is_dataset_from_mr: true` | `GPTFromMRDataset` | Read fixed records directly |

Indexed processing order:

```text
build low-level source
  -> split train/valid/test
  -> build split-local mid-level Dataset
  -> blend sources
  -> sample shuffle index
  -> DP BatchSampler
```

`split: "98, 1, 1"` partitions low-level elements consecutively. Separate
`train_data_path/valid_data_path/test_data_path` may replace the shared split.

Standard blending uses weighted scheduling; `simple_blend: inter/intra` applies only to offline fixed records.
See `docs/guide/data/offline_preparation_guide.md` for offline preparation commands and format details.

## 9. Samplers, Epochs, and Resume

Mapping Datasets use `build_dataset_batch_sampler`:

- `single`: Consume the Dataset's existing global index sequentially, reusing the order across epochs.
- `cyclic`: Build a deterministic shuffle with `seed + epoch`.
- `drop_last`: Drop the tail that cannot fill a complete distributed micro-batch.
- `data_rearrange_map`: Add a mapping from logical indices to physical Dataset indices.

Dynamic DataLoader checkpoints save:

```text
source DataLoader cursor
+ BatchSampler consumed_samples / epoch
+ Unconsumed candidate buffer
```

Mapping buffers store indices by default, allowing transform replay; Iterable buffers store complete samples. Online
DataLoader resume currently requires unchanged `global_batch_size` and DP world size.

## 10. Configuration Examples

### Online Mapping Text

```yaml
dataset:
  _target_: hyper_parallel.data.text.build_dataset.build_online_text_mapping_dataset
  data_path: /data/train.jsonl
  data_transform:
    _target_: hyper_parallel.data.text.text_transform.build_text_transform
    data_type: plaintext
    text_keys: text
    max_seq_len: 4096
  data_config: {}

dataloader:
  _target_: hyper_parallel.data.batching.TokenBatchLoader
  min_buffered_samples: 200
  collate_fn:
    _target_: hyper_parallel.data.batching.build_online_text_collate_fn
  get_batch:
    _target_: hyper_parallel.data.batching.TextParallelBatch
    source_type: online
```

### Online Mapping DeepSeek-V4.1 Omni baseline

```yaml
dataset:
  model_assets:
    _target_: hyper_parallel.models.deepseek_v41.adapter.data.processor.build_deepseek_v41_processor
    config_path: /models/DeepSeek-V4.1-Flash
  data_transform:
    _target_: hyper_parallel.models.deepseek_v41.adapter.data.transform_fn.build_deepseek_v41_omni_transform
    max_seq_len: 4096
  _target_: hyper_parallel.data.omni.build_dataset.build_online_omni_mapping_dataset
  data_path: /data/train.jsonl
  data_config: {}

dataloader:
  _target_: hyper_parallel.data.batching.OmniPackingLoader
  token_budget: 128
  min_buffered_samples: 1
  drop_last: true
  num_workers: 0
  collate_fn:
    _target_: hyper_parallel.data.batching.build_omni_collate_fn
  get_batch:
    _target_: hyper_parallel.data.batching.OmniParallelBatch
```

## 11. Debugging

```yaml
debug:
  check_dataset: debug  # debug / info / warn; null follows the global level.
```

Alternatively, enable before Trainer initialization:

```python
from hyper_parallel.data.dataset_logging import enable_dataset_logging

enable_dataset_logging("debug")
enable_dataset_logging("debug", ranks=(1, 3))
enable_dataset_logging("debug", ranks=None)
```

Logs cover sources, splits/blending, Datasets, Samplers, DataLoaders, and parallel batch shapes, without recording sample text.
