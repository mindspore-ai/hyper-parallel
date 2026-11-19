# HyperParallel Data

**English** | [简体中文](README.zh.md)

Usage references: [YAML configuration](../../docs/guide/trainer/yaml_config.md),
[Omni Online integration](../../docs/guide/data/omni_online_data_guide.md),
[text/image SFT](../../docs/guide/data/qwen3_8_sft_data_guide.md), and
[Offline preparation options](../../docs/guide/data/offline_preparation_guide.md).
Section 10 covers training configuration, sample length, packing, and padding.

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

Text training selects `dataloader.get_batch.source_type: offline` for prepared
Indexed datasets and `online` for raw sources. The Indexed storage APIs retain their names.

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
  data_config:
    max_seq_len: 4096
```

Ratio splits use Hugging Face split expressions and apply only to Mapping sources:

```yaml
dataset:
  _target_: hyper_parallel.data.text.build_dataset.build_online_text_mapping_dataset
  data_path: /data/all/*.parquet
  data_config:
    max_seq_len: 4096
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
  data_config:
    max_seq_len: 4096
```

Hub Datasets and local paths use the same `data_path` field:

```yaml
dataset:
  _target_: hyper_parallel.data.text.build_dataset.build_online_text_mapping_dataset
  data_path: Salesforce/wikitext
  data_config:
    max_seq_len: 4096
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
  -> TokenBatchLoader
  -> TextPackingCollator
  -> input_ids/labels [B, padded_length] + cu_seq_lens
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

Online `TokenBatchLoader` selects complete samples into B rows, each limited to S tokens.
The configured `_target_` selects the loader directly: `TokenBatchLoader` for text packing,
`OmniPackingLoader` for Omni packing. The Trainer does not replace loader types.
Offline Indexed data with `build_indexed_collate_fn` continues to use fixed-sample batching.

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

Local JSONL, JSON, Parquet, and Arrow Mapping records retain their source file's absolute path. Transforms resolve
recognized relative media paths against that file's directory, including after multi-file composition, filtering, and
splitting. Remote URLs, data URLs, and absolute media paths remain unchanged. Supply absolute media paths when invoking
a transform directly or using a source without provenance metadata.

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

Set S once with `dataset.data_config.max_seq_len`. Both Online text and Omni
pack B independent sequences, each containing at most S tokens, then pad them into `[B,T]`.
`pad_to_length: true` (the default) pads each sequence to S; false uses the longest actual
sequence rounded up to TP×CP alignment. S must be divisible by TP×CP.

The candidate pool emits a micro-batch when both conditions hold:

```text
len(buffer) >= min_buffered_samples
and
sum(sample_cost) >= micro_batch_size * max_seq_len
```

```yaml
dataset:
  data_config:
    max_seq_len: 128
  data_transform:

dataloader:
  _target_: hyper_parallel.data.batching.OmniPackingLoader
```

Individual samples longer than S are skipped and counted. Samples that only overflow the
current packed sequence remain buffered for subsequent sequences. Source repetition retains
the buffer. At finite EOF, `drop_last` controls only a tail containing fewer than B packed sequences.

## 7. Omni Distributed Constraints

`OmniParallelBatch` reads identical DP samples on TP0 at each CP coordinate. `CPBatchSharder` shards only token fields,
then `TPBatchBroadcaster` sends CP-local tokens and complete image fields to TP peers at that CP coordinate.
`pixel_values`, image grids, and global `image_token_starts` are not CP-sharded. Current requirements:

```text
PP=1; TP/CP paths require distributed process groups.
```

Every TP×CP rank receives complete images and computes ViT independently, inserting only visual embeddings
corresponding to local CP tokens into the text sequence. Whole-image bucketing and all-to-all routing for encoder DP
remain unimplemented; the current `OmniParallelBatch` YAML does not accept `encoder_dp`.

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

The DeepSeek-V4.1 smoke configuration sets one shared packing and fixed-padding limit:

```yaml
dataset:
  data_config:
    max_seq_len: 4096
  data_transform:
    _target_: hyper_parallel.models.deepseek_v41.adapter.data.transform_fn.build_deepseek_v41_omni_transform

dataloader:
  _target_: hyper_parallel.data.batching.OmniPackingLoader
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
| Variable-length documents | `low_level_dataset._target_: ...GPTDataset` | `GPTDataset` | Dynamic samples from document/sample/shuffle indices |
| Offline fixed-length records | `low_level_dataset._target_: ...GPTFromMRDataset` | `GPTFromMRDataset` | Read fixed records directly |

The full target module is `hyper_parallel.data.indexed.indexed_pretrain_dataset`.

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
- `drop_last`: Online packing drops a finite-source tail only after it produces fewer than B packed rows;
  both single and cyclic retain raw tail records for packing. Offline fixed batching drops incomplete distributed batches.
- `data_rearrange_map`: Add a mapping from logical indices to physical Dataset indices.

Dynamic DataLoader checkpoints save:

```text
source DataLoader cursor
+ BatchSampler consumed_samples / epoch
+ Unconsumed candidate buffer
```

Mapping buffers store indices by default, allowing transform replay; Iterable buffers store complete samples.
Online dynamic loaders reject DP world-size changes on restore. Keep batch settings, parallelism, and sources unchanged
when verifying exact batch-by-batch replay.

Currently, HF Iterable buffer shuffle does not checkpoint the full HF shuffle buffer; restoring may lose records that
have not been yielded. Saving the candidate pool does not fix this. `shuffle: false` supports an unshuffled restore
scenario; shuffled restore remains an open issue. With `datasets 4.0.0`, cyclic/repeat restore across source passes also
has an epoch-cursor compatibility issue.

## 10. Configuration Examples

These data-related fragments must be combined with model and optimizer configuration. Complete recipes include
[Online text](../../hyper_parallel/models/qwen3_8/recipes/train_online.yaml) and
[Offline text](../../examples/training_demo/qwen3_moe/train_parallel_offline.yaml).

### Required Settings

Minimal templates contain required settings only. Online and Offline both require `dataset.data_config.max_seq_len`.

| YAML setting | Online raw data | Offline Indexed data |
| --- | --- | --- |
| `dataset._target_` | Text Mapping/Iterable or Omni Mapping builder | `build_indexed_text_dataset` |
| `dataset.data_path` | Files, directories, globs, path lists, or Hub ID; mutually exclusive with `data_config.sources` | Shared `.bin/.idx` prefix without extensions |
| Length limit | Required `dataset.data_config.max_seq_len` | `dataset.data_config.max_seq_len` |
| Training limit | Required pair `training.epochs` and `training.max_steps` | One of `training.train_iters` or `training.train_samples` |
| `dataset.data_config` | Requires `max_seq_len` | Requires `max_seq_len`, `low_level_dataset`, and `simple_blend` |
| Text runtime | `TextParallelBatch` with `source_type: online` | `TextParallelBatch` with `source_type: offline` |
| Omni runtime | `OmniParallelBatch` is intrinsically Online | No top-level Offline Omni Dataset entry point |

Online `max_steps` counts **optimizer steps per training epoch**; total steps are `epochs * max_steps`.
A training epoch ends at this step limit, independently of source EOF. `dataset.data_config.repeat` defaults to `true`:
EOF restarts the source while retaining candidates. `single` reuses source order; `cyclic` advances shuffle on source
passes. The Trainer's Online plan requires repeat, rejecting `data_config.repeat: false` and `dataloader.repeat: false`.
Do not combine this plan with `train_iters/train_samples` or `dataloader.load_balance`.

### YAML Sections and Source Order

`dataset.model_assets` loads the tokenizer/processor; `dataset.data_transform` encodes records into model samples.
`dataset.data_config` controls sources, splits, and reading options. `dataloader` controls sampling and candidate pools,
`collate_fn` pads the selected rows, and `get_batch` handles CP/TP transport and model inputs.
The model's maximum context setting does not automatically supply the Dataset's S.

### DataLoader Selection and Batching Mechanics

Let B=`training.micro_batch_size` and S=`dataset.data_config.max_seq_len`.
**The Dataset creates model samples; the loader selects samples; the packer concatenates them; the collator pads/stacks them.**
Mapping Datasets receive indices from a DP sampler. Iterable Datasets own streaming reads and sharding.
Each native loader yield is one forward/backward micro-batch. An optimizer step consumes
`global_batch_size / (B * DP)` micro-batches: global size 8, B=2 and DP=2 give two.
`get_batch` performs CP sharding and TP transport after collation; loaders return the unsharded `[B,T]`.

| Loader/entry | How one micro-batch is produced | Typical use |
| --- | --- | --- |
| `FixedBatchDataLoader` | Pass B Dataset outputs directly to the collator | Offline Indexed/SFT, already encoded fixed Omni rows |
| `TokenBatchLoader` | Buffer encoded text, select and concatenate B independent rows of at most S tokens, then pad | Online Mapping/Iterable text and text SFT |
| `OmniPackingLoader` | Select using length metadata, encode selected samples, pack B rows, pad, then encode the batch | Online multimodal packing, especially deferred media encoding |
| `DynamicBatchDataLoader` | Compatibility alias of `TokenBatchLoader` | Existing callers; prefer `TokenBatchLoader` for new configurations |
| `dataloader.load_balance` | Construct a collective loader that schedules costs across DP within native sampler windows | Balancing fixed/prepacked data; separate from Online candidate-buffer packing |

#### FixedBatchDataLoader: Fixed Record Count

```text
Dataset produces rows A[S], B[S], C[S], D[S]
    -> sampler chooses A, B
    -> collator([A, B]) -> batch [2,S]
Next batch chooses C, D; the loader does not concatenate A and B into one row.
```

B counts Dataset outputs. If an output already contains multiple rows, the collator determines the resulting row count.
Indexed sample construction may concatenate documents inside the Dataset; this is separate from Fixed loader batching.
Use `build_indexed_collate_fn` for ordinary stacking and `build_sft_collate_fn` for SFT records with document boundaries.
Kimi's eager transform produces one length-S row before `build_omni_collate_fn` stacks rows into `[B,S]`, another valid Fixed use.
Pairing Fixed with the text packing collator does not introduce a candidate buffer: without `cu_seq_lens`, that collator
concatenates the supplied samples as one row. Use `TokenBatchLoader` to guarantee B independent packed rows.

Merge this fragment into the complete Offline Indexed Dataset example later in this section:

```yaml
training:
  train_iters: 20
  micro_batch_size: 2
  global_batch_size: 8

dataset:
  data_config:
    max_seq_len: 4096

dataloader:
  _target_: hyper_parallel.data.batching.FixedBatchDataLoader
  collate_fn:
    _target_: hyper_parallel.data.batching.build_indexed_collate_fn
  get_batch:
    _target_: hyper_parallel.data.batching.TextParallelBatch
    source_type: offline
```

Each DP rank receives two length-4096 rows per forward pass. The native sampler drops incomplete tail batches by default.

#### TokenBatchLoader: Select Complete Samples into B Independent Rows

```text
Source samples -> candidate buffer using input_ids lengths (skip samples longer than S)
               -> candidate count >= min_buffered_samples AND total length >= B*S
               -> scan candidates: select those fitting this row; retain others and continue scanning
               -> concatenate one row with cu_seq_lens document boundaries
               -> repeat selection from the remaining buffer until B rows are formed
               -> collator chooses one common padding length -> [B,T]
```

The thresholds trigger selection; they are neither a buffer capacity limit nor a guarantee of exactly S tokens per row.
A later small candidate can fit even after a larger candidate was skipped, so selection is not strict contiguous chunking.
Skipped candidates remain available to later rows/batches. B*S is the readiness threshold; **the per-row packing limit is S**.

For S=4096, B=2 and `min_buffered_samples: 1`, candidates `[1800,2000,500,3000,4096]` total 11396 tokens,
exceeding the 8192-token readiness threshold:

```text
packing: row 1 [1800][2000] = 3800; retain 500, 3000, 4096
         row 2 [500 ][3000] = 3500; retain 4096 in the buffer
padding: row 1 [1800][2000][PAD 296] = 4096
         row 2 [500 ][3000][PAD 596] = 4096
                      -> input_ids/labels [2,4096]
                      -> CP=2: each CP rank [2,2048]
```

`pad_to_length: true` defaults to S per row; false uses the longest packed row rounded up to a TP×CP multiple.
Here the longest row is 3800: TP=4 and CP=2 give dynamic shape `[2,3800]`, then `[2,1900]` after CP sharding.
Padding uses input token 0 and label -100. `cu_seq_lens` records both real document boundaries and separate padding tails.
TP does not further divide each input row's length as CP does; TP×CP supplies the padding alignment constraint here.

Keep the Online Text Dataset/tokenizer/transform and use:

```yaml
training:
  epochs: 1
  max_steps: 10
  micro_batch_size: 2
  global_batch_size: 8

dataset:
  data_config:
    max_seq_len: 4096

dataloader:
  _target_: hyper_parallel.data.batching.TokenBatchLoader
  min_buffered_samples: 1  # Small demonstration pool; omitted default is 200
  collate_fn:
    _target_: hyper_parallel.data.batching.build_online_text_collate_fn
    pad_to_length: true   # Default; false aligns this batch's longest row to TP/CP
  get_batch:
    _target_: hyper_parallel.data.batching.TextParallelBatch
    source_type: online
```

#### OmniPackingLoader: Select before Media Encoding and Packing

Omni shares the B-row loop, readiness thresholds, padding, EOF handling and checkpoints with Token batching.
The work surrounding selection differs:

```text
Raw media/text -> Dataset eager encode or preencode (packing_length planning metadata)
               -> PackingSelector chooses candidates with total cost <= S
               -> encode_selected_sample runs postencode for selected candidates
               -> SamplePacker concatenates tokens/labels/media and offsets media token starts
               -> repeat B rows -> OmniCollator pads/combines media -> Dataset.encode_batch
```

The default `FirstFitPackingSelector` uses `packing_length`, falling back to `input_ids` length; when encoded tokens
are already available, cost is at least their actual length. Deferred lengths must include expanded media tokens;
an encoded row longer than S raises an error. Deferring expensive media encoding until selection reduces work/storage
for unselected candidates. Eager transforms already encode before buffering.
For two media samples costing 1800 and 2000 tokens including image expansion, one packed row contains 3800 tokens;
media tensors are combined and the second sample's media token starts are offset by the first sample's token length.

Keep the Online Omni Dataset and model-specific transform; replace the preceding dataloader with:

```yaml
dataloader:
  _target_: hyper_parallel.data.batching.OmniPackingLoader
  collate_fn:
    _target_: hyper_parallel.data.batching.build_omni_collate_fn
  get_batch:
    _target_: hyper_parallel.data.batching.OmniParallelBatch
```

The generic loader supports B=2; the current DeepSeek-V4.1 compact runtime still requires B=1.

#### EOF, Repetition, Tail Dropping and Resume

| Situation | Behavior |
| --- | --- |
| Dynamic source EOF with `repeat=true` | Keep the buffer, advance the source pass, restart reading and accumulate with retained candidates |
| Finite dynamic iteration with `repeat=false` | At EOF bypass both readiness thresholds and drain candidates; internal `drop_last=true` discards only the final batch with fewer than B rows |
| B rows exist but a row contains fewer than S real tokens | Keep and pad the batch; an underfilled row is not an incomplete batch |
| Fixed source has an incomplete batch | Native sampler `drop_last` applies; outer repetition cannot recover records already dropped |

The dynamic source sampler/loader uses `drop_last=false` so raw tail records reach the buffer.
Online Trainer repeats by default and runs `epochs * max_steps` optimizer steps without clearing candidates at training
epoch boundaries. Finite `repeat=false` draining is available to standalone Python callers; Online Trainer rejects it.
`drop_last`, worker settings and checkpoint storage strategy use internal defaults and are not YAML settings;
`min_buffered_samples` and `repeat` remain dynamic loader options.

For example, a retained `[500]` followed by source restart and `[1800,2000,...]` becomes `[500,1800,2000,...]`
in the same buffer. Reading continues until the thresholds hold; EOF does not itself force this group to be emitted.
Checkpoints retain source cursor/pass and unconsumed candidates: Mapping defaults to replay indices, Iterable to full samples.
Restore consumes already-ready candidates before advancing the source. Reproducing the data sequence requires the same DP size
and deterministic transforms.

`RepeatingDataLoader` wraps ordinary/Fixed loaders and repeats already formed batches; it does not pack candidates.
Dynamic loaders repeat internally. `HyperIter`/background prefetch forward iteration without deciding packing lengths.

#### load_balance: Cross-DP Scheduling

Omitting `load_balance` uses `_target_` directly; `load_balance: {}` constructs a `DistributedDataLoader`.
This path retains native sampler steps/windows and estimates costs, assigns samples across DP, and transports data to reduce
rank workload differences. Default `repack_sequences=false` preserves Dataset samples; enabling it splits along document
boundaries and repacks, potentially changing each rank's sample grouping. It does not use Token/Omni candidate buffers.

For four equal-length samples costing `[9,8,2,1]`, an initial assignment `[9,8]` versus `[2,1]` has costs 17/3.
When constraints permit and estimates are accurate, assigning `[9,1]` versus `[8,2]` gives 10/10.
This illustrates scheduling; perfect balance is not guaranteed for every window.

Add this block to the Offline Fixed example while retaining its collator/get_batch:

```yaml
dataloader:
  _target_: hyper_parallel.data.batching.FixedBatchDataLoader
  load_balance: {}  # Enable cost scheduling; retain existing collate_fn/get_batch
```

This requires a Mapping Dataset and device mesh; PP must currently be 1. Do not combine it with Token/Omni dynamic loaders
or Online `epochs/max_steps`. For repacking prepacked text, see the
[Qwen balanced recipe](../../examples/training_demo/qwen3/train_balanced.yaml).

The direct Python API also provides `LocalBalancingDataLoader`: it receives externally selected whole-step sample
groups, exchanges/regroups them within local DP groups, and leaves selection/termination to the source iterator.
It currently supports pure DP without checkpoint/resume; it is not a regular Trainer YAML loader choice.
See [local balancing integration](../distributed_data/NODE_LOCAL_BALANCING.md).

Implementation: [loaders and batching loop](batching/build_dataloader.py), [candidate selection](batching/dynamic_batch.py),
[field packing](batching/packing.py), [padding/collators](batching/build_collate_fn.py),
[repetition wrapper](batching/build_dataloader.py).

### Defaults (Omitted from Minimal Templates)

Omitted settings use these defaults. Configure them explicitly only to change behavior.

Worker execution uses internal defaults: `num_workers=0`, `pin_memory=false`, and `prefetch_factor=null`.
Internal `drop_last=true` drops incomplete final row batches; packing retains raw source tails.
Candidate checkpoints automatically use indices for Mapping sources and full samples for Iterable sources.
These five options cannot be set through YAML or CLI; `min_buffered_samples` and `repeat` remain configurable.

| Setting | Default | Meaning |
| --- | --- | --- |
| `training.micro_batch_size` | `1` | Packed rows B per DP rank per forward |
| `training.global_batch_size` | `8` | Global rows per optimizer step; divisible by B×DP |
| `training.seed` | `null` (Trainer initializes it to `42`) | Common source and sampler seed |
| `dataloader.sampler_type` | `single` | Reuse source order; `cyclic` shuffles across source passes |
| `dataloader.min_buffered_samples` | `200` | Candidate-count threshold, alongside total cost at least B×S |
| `dataloader.repeat` | `null`: Online Trainer repeats; standalone Python iteration is finite | Inherit the caller's repetition policy |
| `dataset.data_config.repeat` | `true` (Online Trainer) | Restart at EOF while retaining candidates |
| `dataloader.collate_fn.pad_to_length` | `true` | Pad to S; `false` selects dynamic padding |
| `dataset.data_config.split` | Online: all train when omitted; Offline: `"1, 0, 0"` | Train/valid/test membership; independent split paths are configured separately |
| `dataset.data_config.labels_are_shifted` | `true` (Text runtime) | Labels already align with next-token targets |
| `dataset.data_transform.text_keys` | `null`: plaintext uses `text`, sharegpt uses `conversations` | Override explicitly for a messages column |
| `dataset.data_transform.drop_ragged_tail` | `false` | Retain subsequent partial plaintext windows |
| `dataset.data_config.shuffle` | `true` (Iterable) | HF buffer shuffle; Mapping order uses sampler/source indices |
| `dataset.data_config.shuffle_buffer_size` | `10000` (Iterable) | Shuffle buffer capacity when enabled |
| `dataset.data_config.split_by_data_parallel` | `true` (Iterable) | Shard the source across DP ranks |
| `dataset.data_config.blend_strategy` | `local_weighted` (Iterable) | Multi-source combination; also supports `global_interleave` |
| `dataset.data_config.stopping_strategy` | `first_exhausted` (Iterable) | End a pass when any source is exhausted |
| `dataset.data_transform.sample_alignment_size` | `1` (DeepSeek-V4.1) | Align each model sample before packing |

### Trainer-Injected Arguments (Do Not Set in YAML)

The runtime supplies or overrides these arguments, even if Python signatures allow them.
Provide them manually only when calling the corresponding Python APIs outside the Trainer.

| Argument location | Source |
| --- | --- |
| `dataloader.batch_size` | `training.micro_batch_size` |
| `dataloader.dp_world_size` | Current mesh DP size |
| `dataloader.max_seq_len`, `dataloader.collate_fn.max_seq_len` | `dataset.data_config.max_seq_len` |
| `dataloader.seed`, `dataset.data_config.random_seed` | Training seed policy; configure `training.seed` |
| `dataloader.dataset/batch_sampler` | Trainer-built Dataset and sampler |
| `collate_fn.mesh_context`, `get_batch.mesh_context/device/dataloader_context` | Distributed runtime |
| `dataset.transform/tokenizer/training_config/mesh_context/dataloader_context/sampler_type` | Built components and Trainer settings |
| `dataset.data_transform.tokenizer/chat_template/processor/training_mode` | model_assets and `dataset.training_mode`, according to Text/Omni mode |

### Unsupported and Mutually Exclusive Settings

| Setting | Constraint |
| --- | --- |
| `dataloader.token_budget` | Removed; set the packing limit through transform `max_seq_len` |
| `OmniParallelBatch.encoder_dp` | Not accepted by the current interface |
| Online `repeat: false`, `dataloader.load_balance` | Unsupported with the current `epochs/max_steps` plan |
| `training.epochs/max_steps` and `train_iters/train_samples` | Mutually exclusive; Online uses the former, Offline the latter |
| `data_path` and `data_config.sources` | Select one; each source needs a path and positive weight |
| Iterable ratio `split` | Unsupported; Mapping ratio splits also conflict with pre-split path mappings |

### YAML Sample Length, Packing, and Padding

Let B=`training.micro_batch_size`, S=`dataset.data_config.max_seq_len`, and A=`TP×CP`:

Lengths count encoded `input_ids` tokens, including prompts and expanded media tokens.
`labels=-100` masks loss without removing the corresponding input tokens from the packing budget.

```text
raw record -> transform emits 0/1/many model samples -> candidate pool
           -> select complete samples totaling at most S per row -> select B rows
           -> collator pads every row to T -> [B,T] -> CP shards -> [B,T/CP]
```

| Setting | Meaning and stage |
| --- | --- |
| `dataset.data_config.max_seq_len: S` | Model-sample length constraint and packing limit per row; the Trainer injects it into the loader/collator |
| `training.micro_batch_size: B` | Packed rows per DP rank per forward pass, rather than raw record count |
| `training.global_batch_size` | Global rows per optimizer step; must be divisible by `B * DP`, giving the gradient accumulation count |
| `dataloader.min_buffered_samples` | Candidate-count threshold; selection also requires total candidate cost at least `B*S`; does not change S or B |
| `dataloader.collate_fn.pad_to_length: true` | Pad each row to S, producing `[B,S]` |
| `dataloader.collate_fn.pad_to_length: false` | Pad to the longest selected row rounded up to A: `T=ceil(max(row_length)/A)*A` |

S must be positive and divisible by A. **Both padding modes pack with limit S**: dynamic padding only reduces the final
tensor length after selection. DP does not determine alignment. `pad_to_length` accepts only a boolean;
the fixed row length comes from S.
The current loader does not accept a separate `token_budget` YAML option.

Character count differs from token count. `plaintext` tokenizes, appends EOS when available, and uses S+1-token windows
to build `input_ids=window[:-1]` and `labels=window[1:]`. One long record can therefore produce multiple samples of
at most S tokens. `drop_ragged_tail: true` drops subsequent incomplete windows; the default is `false`.
Text conversations are limited by their chat template and then shifted; for example, the tokenizer template keeps
the last at most S encoded tokens, yielding at most S-1 input tokens. Generic `AutoProcessorTransform` rejects media
samples longer than S; the current DeepSeek-V4.1 transform logs a warning and skips them. Neither truncates image spans.
Multimodal lengths include expanded image tokens.

Packing scans candidates in order and selects complete samples that fit the current row. Others remain for later rows
or batches; a row may be shorter than S. The loader skips and counts model samples still longer than S, without
truncating them during padding. `cu_seq_lens` retains document boundaries and separate padding segments.
Online collation pads inputs with 0, labels with -100, and loss masks with 0.

For S=4096 and B=2, suppose the pool contains only `[1800,2000,500,3000]`, as when draining a finite source at EOF.
The selected rows have lengths `[3800,3500]`:

| Padding (CP=2, TP=4, A=8) | Padded row lengths | Each DP rank before CP sharding | Each CP rank |
| --- | --- | --- | --- |
| `pad_to_length: true` | Both 4096; add 296 and 596 tokens | `[2,4096]` | `[2,2048]` |
| `pad_to_length: false` | Both 3800; add 0 and 300 tokens | `[2,3800]` | `[2,1900]` |

A longest row of 3797 also pads to 3800 with A=8. Model transforms can additionally align individual samples before
packing; DeepSeek-V4.1 `sample_alignment_size` controls that stage, defaults to 1, and differs from batch padding.
The compact DeepSeek-V4.1 VLM runtime currently requires B=1; the generic loader shape contract does not remove
model-specific batch-size constraints.

### Online Mapping Text

```yaml
training:
  epochs: 2
  max_steps: 100

dataset:
  _target_: hyper_parallel.data.text.build_dataset.build_online_text_mapping_dataset
  data_path: /data/train.jsonl
  model_assets:
    tokenizer:
      _target_: hyper_parallel.data.text.build_tokenizer.AutoTokenizer.from_pretrained
      pretrained_model_name_or_path: /models/tokenizer
  data_transform:
    _target_: hyper_parallel.data.text.text_transform.build_text_transform
    data_type: plaintext
  data_config:
    max_seq_len: 4096

dataloader:
  _target_: hyper_parallel.data.batching.TokenBatchLoader
  collate_fn:
    _target_: hyper_parallel.data.batching.build_online_text_collate_fn
  get_batch:
    _target_: hyper_parallel.data.batching.TextParallelBatch
    source_type: online
```

This example reads `{"text":"..."}` JSONL records. For SFT, use `data_type: sharegpt`,
`text_keys: messages` (or `conversations`), and `dataset.model_assets.chat_template: tokenizer`.
`alpaca` requires `column_mapping`; see the [SFT guide](../../docs/guide/data/qwen3_8_sft_data_guide.md).

### Online Iterable Text

Keep the preceding tokenizer, transform, loader, and `source_type: online`; replace the builder and path:

```yaml
dataset:
  _target_: hyper_parallel.data.text.build_dataset.build_online_iterable_dataset
  data_path: /data/train-*.jsonl
  data_config:
    max_seq_len: 4096
```

The minimal template uses the defaults table. For an unshuffled restore scenario, explicitly override
`data_config.shuffle: false` and review the limitations in Section 9. `blend_strategy` combines sources;
`stopping_strategy` ends a pass, and repeat decides whether to restart. Disabling `split_by_data_parallel` requires
external DP sharding to avoid duplicate records. Worker parallelism also depends on the file shard count.
Index replay is selected internally for Mapping sources; Iterable sources store complete samples by default.
Replaying an earlier index-based checkpoint requires source support for `get_item/output_index_for_resume`.

### Online Mapping DeepSeek-V4.1 Omni baseline

```yaml
training:
  epochs: 1
  max_steps: 100

dataset:
  training_mode: sft
  model_assets:
    _target_: hyper_parallel.models.deepseek_v41.adapter.data.processor.build_deepseek_v41_processor
    config_path: /models/DeepSeek-V4.1-Flash
  data_transform:
    _target_: hyper_parallel.models.deepseek_v41.adapter.data.transform_fn.build_deepseek_v41_omni_transform
  _target_: hyper_parallel.data.omni.build_dataset.build_online_omni_mapping_dataset
  data_path: /data/train.jsonl
  data_config:
    max_seq_len: 4096

dataloader:
  _target_: hyper_parallel.data.batching.OmniPackingLoader
  collate_fn:
    _target_: hyper_parallel.data.batching.build_omni_collate_fn
  get_batch:
    _target_: hyper_parallel.data.batching.OmniParallelBatch
    runtime_input_adapter:
      _target_: hyper_parallel.models.deepseek_v41.adapter.data.runtime.DeepseekV41Runtime
```

VLMTrainer requires explicit `dataset.training_mode: sft` or `pretrain`. Processors and transforms must match the model;
see the [DeepSeek-V4.1 guide](../../docs/guide/data/deepseek_v41_vlm_online_data_guide.md) for runtime adapters and constraints.

### Offline Indexed Text

```yaml
training:
  train_iters: 100

dataset:
  _target_: hyper_parallel.data.text.build_dataset.build_indexed_text_dataset
  data_path: /data/train_text_document
  model_assets:
    tokenizer:
      _target_: hyper_parallel.data.text.build_tokenizer.AutoTokenizer.from_pretrained
      pretrained_model_name_or_path: /models/tokenizer
      tokenizer_type: pretokenized
      vocab_size: 151936
      eod_token_id: 151645
  data_config:
    max_seq_len: 4096
    simple_blend: "no"
    low_level_dataset:
      _target_: hyper_parallel.data.indexed.indexed_pretrain_dataset.GPTDataset

dataloader:
  _target_: hyper_parallel.data.batching.FixedBatchDataLoader
  collate_fn:
    _target_: hyper_parallel.data.batching.build_indexed_collate_fn
  get_batch:
    _target_: hyper_parallel.data.batching.TextParallelBatch
    source_type: offline
```

`pretokenized` supplies metadata without tokenizing again. Both `vocab_size` and `eod_token_id` must match the original
corpus tokenizer; these example values are model-specific. Optionally set `pad_token_id` to an existing dedicated PAD ID.
`GPTDataset` builds length-S samples from documents at runtime; use `GPTFromMRDataset` for records already cut to S+1
tokens. Both provide shifted labels and ignore padding with -100. Offline length uses `max_seq_len`, independently of
Online candidate-pool packing/padding options. Replacing `train_iters` with `train_samples` gives
`train_samples // global_batch_size` optimizer steps.

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
