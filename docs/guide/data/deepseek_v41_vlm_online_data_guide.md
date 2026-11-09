# DeepSeek-V4.1 Online VLM Data Transformation Guide

**English** | [简体中文](deepseek_v41_vlm_online_data_guide.zh.md)

This guide explains how raw JSONL samples become model inputs for DeepSeek-V4.1 Online VLM training, including the responsibilities,
inputs, outputs, and constraints of the key interfaces. The main implementations are:

- `hyper_parallel/data/omni/omni_transform.py`
- `hyper_parallel/models/deepseek_v41/adapter/data/processor.py`
- `hyper_parallel/models/deepseek_v41/adapter/data/encoding.py`
- `hyper_parallel/models/deepseek_v41/adapter/data/image_processor.py`
- `hyper_parallel/models/deepseek_v41/adapter/data/transform_fn.py`

## 1. Supported Scope

The current pipeline targets Online JSONL image question-answering/SFT data with OpenAI-style `messages`. It supports text messages and
`image_url` image content blocks, but does not automatically normalize arbitrary VeOmni, ShareGPT, or Megatron-Energon samples into
this format. The “Energon-style” description of `OmniDataTransform` refers to the transform hook lifecycle, not input format compatibility.

The DeepSeek-V4.1 adapter currently handles images, not video or audio. Although the generic `prepare_messages()` can resolve some
`video` and `audio` paths, the model adapter has no corresponding encoding implementation.

## 2. JSONL Input Contract

Each line must be a JSON object with nonempty `messages`. The last message in an SFT sample must be from the assistant:

```json
{
  "id": "train_36",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "image_url",
          "image_url": {
            "url": "images/train/train_36.jpg"
          }
        },
        {
          "type": "text",
          "text": "Which base paper will be coated in-house?"
        }
      ]
    },
    {
      "role": "assistant",
      "content": "091"
    }
  ],
  "source": {
    "dataset": "example/docvqa",
    "split": "train",
    "row_index": 28
  }
}
```

`id` and `source` can be retained as tracking metadata, but `encode_sample()` does not read them. Only `messages` participates in encoding.

The `image_url.url` form shown above is recommended. Image values can be:

- Relative or absolute local paths;
- `http://` or `https://` URLs;
- base64 `data:` URL.

Do not manually write `<｜deepseek_image｜>` in text. Provide each image as a separate content block; the adapter inserts the model
placeholder and checks that the placeholder count matches the image count.

### 2.1 Data Source Routing FAQ

#### Interface: How Are Empty Omni Samples Filtered?

**FAQ: Which interface should callers use?**

DeepSeek-V4.1 Online VLM uses:

```python
build_online_omni_mapping_dataset(
    data_config=data_config,
    data_path=data_path,
    transform=transform,
    training_config=training_config,
)
```

This entry point validates `transform`, sets the data random seed from `training_config.seed`, and passes the filter to the generic Mapping
source builder. The core call relationship is:

```python
sample_filter = transform.is_valid_sample
source_dataset = build_online_mapping_source(
    data_path=data_path,
    data_config=dataset_config,
    sample_filter=sample_filter,
)
```

The complete source interface of `build_online_mapping_source()` is:

```python
build_online_mapping_source(
    *,
    data_config,
    data_path=None,
    sample_filter=None,
    train_valid_test_num_samples=None,
)
```

Here, `sample_filter` receives one raw Mapping record before transformation and returns whether it belongs to the logical Dataset;
`train_valid_test_num_samples` can override the natural lengths of the three splits, but the current top-level Omni entry does not pass it.

**FAQ: What is the contract of `is_valid_sample()`?**

The default Omni implementation is:

```python
@staticmethod
def is_valid_sample(sample: Mapping[str, Any]) -> bool:
    messages = sample.get("messages")
    if not isinstance(messages, list):
        raise ValueError("Omni sample messages must be a list")
    return bool(messages)
```

Therefore:

- `messages=[]` returns `False`, so the record is automatically dropped;
- Missing `messages` or a non-list value violates the input contract and raises an error;
- The filter must be stable and deterministic, independent of random numbers, DP rank, DataLoader worker, and access count;
- The filter only determines raw-record validity; it must not perform tokenization, image decoding, random augmentation, or packing.

This contract ensures that all processes build consistent lengths, splits, and integer indices for the same data.

**FAQ: What does the interface return?**

Without splits, it returns a finite Mapping Dataset supporting `len()` and integer `dataset[index]` access. With a ratio split or
pre-split paths, it returns:

```python
(train_dataset, valid_dataset, test_dataset)
```

An unconfigured or ultimately empty split is `None`. Only afterward is the result wrapped with the lazy Omni transform; building the source index
does not execute `encode_sample()`.

#### General: How Is the Data Source Routed?

**FAQ: Which local file formats does Online Mapping currently support?**

It supports `.jsonl`, `.json`, `.parquet`, `.csv`, and `.arrow`:

| Extension | Hugging Face loader name | Native `_JsonlMappingSource` available |
| --- | --- | ---: |
| `.jsonl` | `json` | Yes |
| `.json` | `json` | No |
| `.parquet` | `parquet` | No |
| `.csv` | `csv` | No |
| `.arrow` | `arrow` | No |

If the native JSONL conditions are not met, the code tries the generic Hugging Face loader; this does not mean that arbitrary file types are supported.

The overall routing rules are:

```text
Local JSONL (without splits, with ratio splits, or with pre-split paths)
  -> Native byte-offset _JsonlMappingSource

JSON/Parquet/CSV/Arrow (without splits, with ratio splits, or with pre-split paths)
  -> Hugging Face load_dataset(streaming=False)

Hub Dataset ID
  -> Hugging Face load_dataset(streaming=False)

Other local file formats
  -> Error
```

#### Scenario 1: Local JSONL

**FAQ: Which configurations use native `_JsonlMappingSource`?**

Local `.jsonl` uses the native reader, including unsplit data, ratio splits, and pre-split train/valid/test paths. The simplest unsplit
configuration is:

```yaml
dataset:
  _target_: hyper_parallel.data.omni.build_dataset.build_online_omni_mapping_dataset
  data_path: /data/docvqa/train.jsonl
  data_config: {}
```

At startup, the native reader scans files and records the byte offsets of valid records, then reads them on demand by integer index. It does not require Arrow
to infer a uniform schema for nested fields, making it suitable for heterogeneous `messages` where user `content` is a list of multimodal blocks
and assistant `content` is a string.

`data_path` can also be a directory containing only JSONL, a glob, or an ordered list of paths. For example, a glob:

```yaml
data_path: /data/docvqa/train-*.jsonl
data_config: {}
```

Directory scanning is nonrecursive and reads only immediate `*.jsonl` files. Path lists are resolved in configuration order, and every match must be a local
`.jsonl` file.

**FAQ: Does filtering empty samples leave gaps in the indices?**

No. The Omni builder passes `transform.is_valid_sample` to `_JsonlMappingSource`, and the native reader calls it for each record before storing its byte
offset. Records with `messages=[]` never enter the index; retained records form a new contiguous index starting at 0.

**FAQ: How are relative image paths resolved?**

Native `_JsonlMappingSource` attaches the internal `__online_source_path__` field to records. `prepare_messages()` uses it to resolve
relative media paths against the JSONL file's directory. For example:

```text
/data/docvqa/train.jsonl
/data/docvqa/images/train/train_36.jpg
```

The sample's `images/train/train_36.jpg` resolves to:

```text
/data/docvqa/images/train/train_36.jpg
```

#### Scenario 2: Ratio Splits or Pre-split Local Data

**FAQ: Does configuring a ratio split still use `_JsonlMappingSource`?**

Yes. Local `.jsonl` still uses the native reader when `data_config.split` is configured:

```yaml
data_path: /data/docvqa/all.jsonl
data_config:
  split: "98, 1, 1"
```

The native reader scans and filters all JSONL once to build contiguous valid byte-offset indices, then deterministically partitions them by valid sample count into
train, valid, and test. The three splits share one offset table and store only their index ranges, without rescanning files:

```text
JSONL -> is_valid_sample -> valid offsets -> ratio split views
```

Empty Omni samples therefore consume neither Mapping indices nor split-ratio quotas.

**FAQ: How are ratio-split boundaries calculated, and are samples shuffled?**

Ratio splitting partitions the filtered valid indices into contiguous ranges. Boundaries use the same rounding as Hugging Face percentage slicing, namely
closest rounding:

```python
train_end = round(train_percentage * valid_count / 100)
valid_end = round((train_percentage + valid_percentage) * valid_count / 100)
```

For example, with 2 empty samples among 10 raw records:

```text
Original IDs: 0, 1(empty), 2, 3, 4, 5(empty), 6, 7, 8, 9
Valid IDs:    0, 2, 3, 4, 6, 7, 8, 9
split:   50, 25, 25

train:   0, 2, 3, 4
valid:   6, 7
test:    8, 9
```

Split membership itself is not randomly shuffled, so adding empty-sample filtering does not otherwise change existing percentage-slicing semantics.
Source-local shuffling and BatchSampler shuffling after split construction are handled in later stages.

**FAQ: Do the three ratio splits copy the index, and is it saved to disk?**

No copies are made. The current minimal implementation stores the filtered `(JSONL file path, byte offset)` table in the Dataset object. Each split
holds only a `[begin, end)` view of the same table, so files are scanned once and the filter runs once per raw record.

The index persists only for the lifetime of the current Dataset; there is no sidecar index shared across runs yet. Each independently launched training
process that calls the builder still performs its own scan. DataLoader workers reuse the index and read records by offset using their own file handles.
Reusing indices across ranks or runs requires a separate shared index cache with a data fingerprint and filter version.

**FAQ: How do I configure train, valid, and test that are already stored separately?**

Use a Mapping of pre-split paths. For example, with JSONL:

```yaml
data_path:
  train: /data/docvqa/train.jsonl
  valid: /data/docvqa/valid.jsonl
  test: /data/docvqa/test.jsonl
data_config: {}
```

An individual split can also be a directory, glob, or ordered list of paths:

```yaml
data_path:
  train:
    - /data/docvqa/train-000.jsonl
    - /data/docvqa/train-001.jsonl
  valid: /data/docvqa/valid-*.jsonl
data_config: {}
```

Local JSONL preserves the configured split membership, then filters and builds native indices separately within train, valid, and test. It does not
replenish valid/test samples removed by filtering with samples from train.

Other formats, such as Parquet, use the same pre-split Mapping:

```yaml
data_path:
  train: /data/docvqa/train.parquet
  valid: /data/docvqa/valid.parquet
  test: /data/docvqa/test.parquet
data_config: {}
```

Split names must be `train`, `valid`, or `test`, and all splits must use the same format. Entirely local JSONL uses native
`_JsonlMappingSource`; JSON, Parquet, CSV, and Arrow use the Hugging Face loader.

**FAQ: How does the Hugging Face route filter empty samples, and what are its VLM limitations?**

For pre-split paths, `load_dataset()` first constructs each split Dataset, then calls the following on each:

```python
dataset.filter(transform.is_valid_sample)
```

For unsplit data with a configured ratio, the Hugging Face route filters the entire Dataset first, then deterministically partitions valid samples into contiguous ranges,
preserving the same filter-before-ratio semantics as native JSONL.

If the same field has different types across records or nested locations, Arrow schema inference can still fail before filtering. Local
heterogeneous Omni JSONL should use the native `.jsonl` route; other formats must have a stable schema beforehand.

The Hugging Face route does not automatically attach `__online_source_path__`. Images and other media must use absolute paths, or the dataset must explicitly
provide the correct `__online_source_path__`; application metadata such as `source.dataset` does not participate in path resolution.

#### Scenario 3: Hub Dataset ID

**FAQ: How is a Hub Dataset ID configured, and which reader does it use?**

A nonlocal `organization/dataset` uses the Hugging Face loader:

```yaml
data_path: example-org/docvqa-sft
data_config: {}
```

If the Hub Dataset has multiple configurations, select one with `data_config.config_name`. Hub data must still satisfy this guide's
OpenAI-style nonempty `messages` contract; being loadable by Hugging Face does not imply that the DeepSeek-V4.1 transform can encode it.

#### Scenario 4: Unsupported Local Formats

**FAQ: Which local formats immediately raise an error?**

For example, the following `.txt` is not automatically read as line-by-line plain text:

```yaml
data_path: /data/docvqa/train.txt
data_config: {}
```

`.txt`, `.yaml`, `.xlsx`, `.tar`, image directories, and similar inputs are not in the current Mapping source's local-format table and report an unsupported-format error.
Convert them to JSONL/JSON/Parquet/CSV/Arrow first, or extend the corresponding source reader.

### 2.2 Multi-source Split, Blend, Shuffle, and DP Sharding FAQ

#### Topic 1: How Does Each Source Determine Train, Valid, and Test?

**FAQ: What structures can a source return?**

Each loaded source returns one of two structures:

```text
Dataset
```

This means the source has no separate splits and contributes its entire Dataset only to train; alternatively:

```text
(train_dataset, valid_dataset, test_dataset)
```

This means the source determined its three splits using a ratio split or pre-split paths. Missing splits are `None`, for example:

```text
(source_train, None, source_test)
```

**FAQ: How can multiple sources combine ratio splits and pre-split paths?**

Use `data_config.sources`; each leaf source independently determines its route, filter, and splits:

```yaml
data_path: null
data_config:
  random_seed: 42
  sources:
    - data_path: /data/source_a/all-*.jsonl
      weight: 0.7
      split: "98, 1, 1"
    - data_path:
        train: /data/source_b/train.jsonl
        valid: /data/source_b/valid.jsonl
      weight: 0.3
```

Source A filters first, then produces `(A_train, A_valid, A_test)` using the ratio; source B preserves pre-split membership and filters within
each split. When `sources` is configured, a top-level `data_path` must not also be provided.

**FAQ: How are split and unsplit sources combined?**

Assume three sources:

| Source | Weight | Loaded result | Dataset lengths |
| --- | ---: | --- | --- |
| A | 0.6 | `(A_train, A_valid, A_test)` | `100 / 20 / 10` |
| B | 0.3 | `B_dataset` | `50` |
| C | 0.1 | `(C_train, None, C_test)` | `40 / 0 / 5` |

Unsplit B contributes only to train, and C has no valid split. Collecting datasets by split name therefore gives:

```text
train = [A_train, B_dataset, C_train]
valid = [A_valid]
test  = [A_test, C_test]
```

#### Topic 2: How Are Matching Splits Globally Blended?

**FAQ: Are train, valid, and test blended together?**

No. Each split independently constructs a DP-global `_OnlineBlendedMappingDataset`:

| Split | Participating datasets | Original weights | Default target size |
| --- | --- | --- | ---: |
| train | `A_train, B_dataset, C_train` | `0.6, 0.3, 0.1` | `100 + 50 + 40 = 190` |
| valid | `A_valid` | `0.6`, normalized to `1.0` | `20` |
| test | `A_test, C_test` | `0.6, 0.1`, normalized to `6/7, 1/7` | `10 + 5 = 15` |

Blending uses a deterministic largest-deficit schedule that records the following for every global logical position:

```text
(source_id, source_logical_index)
```

Global blending therefore means that all DP ranks see the same logical source schedule; it does not merge train, valid, and test into
a single Dataset.

**FAQ: What happens when fixed target sample counts are configured?**

If the lower-level `build_online_mapping_source()` receives the following `train_valid_test_num_samples`:

```python
(1000, 100, 50)
```

The three blended Dataset lengths are fixed at `1000 / 100 / 50`, instead of using the sums of natural lengths. When a target exceeds the available data,
source-local indices repeat across multiple epochs before deterministic shuffling. If a split has no available Dataset or its target is 0,
it returns `None`. The current `build_online_omni_mapping_dataset()` does not pass this parameter to the lower layer, so the top-level Omni
entry defaults to the sum of the natural lengths for each split.

#### Topic 3: What Is the Actual Order of Shuffling and DP Sharding?

**FAQ: After blending, does only train undergo global shuffling?**

This is not unconditional. Mapping source construction currently generates source-local deterministic
shuffle/repeat for every constructed split; additional global shuffling at the DataLoader stage depends on the shared BatchSampler configuration:

```text
Each source determines train/valid/test
  -> Build a DP-global blend index within each matching split
  -> Map each source's logical indices to physical samples after deterministic shuffle/repeat
  -> The shared BatchSampler generates rank-local DP indices according to configuration
  -> lazy transform / packing / collate
```

The three BatchSampler behaviors are:

| Configuration | Shuffle and DP sharding order |
| --- | --- |
| `sampler_type: single` | Partition global logical indices in order, then assign each DP rank contiguous micro-batches; no additional global shuffle |
| `sampler_type: cyclic`, `data_sharding: false` | Shuffle the entire global index range, then shard by DP-rank stride |
| `sampler_type: cyclic`, `data_sharding: true` | Assign each DP rank a contiguous bucket first, then shuffle within each rank-local bucket |

The current DeepSeek-V4.1 Online VLM baseline uses `sampler_type: single`, so the BatchSampler adds no extra
global shuffling. For a strict “global shuffle -> DP shard” order during training, configure:

```yaml
dataloader:
  sampler_type: cyclic
  data_sharding: false
```

All DP ranks then use the same `seed + epoch` global permutation and take nonoverlapping rank-local indices.

## 3. Processing Flow

Source construction and model transformation are separate stages:

```text
data_path / data_config.sources
  -> Each leaf source selects native JSONL or the Hugging Face loader
  -> is_valid_sample on raw samples
  -> Ratio split or preserve pre-split membership
  -> Global blend within matching splits
  -> source-local shuffle/repeat + BatchSampler + DP shard
  -> RawSample
```

Lazy model encoding begins only after retrieving a RawSample:

```text
JSONL RawSample
  -> prepare_messages
  -> chat_template(messages)
  -> image_processor(prompt, images)
  -> _get_assistant_start
  -> _build_labels
  -> _build_image_fields
  -> packing / collate
  -> encode_batch
  -> model batch
```

The Online Mapping transform is lazy: constructing a Dataset does not immediately call `encode_sample()`; encoding runs only when the DataLoader actually
indexes a sample. Training also constructs and shards the model first, so transform debug logs usually appear only after model initialization,
when the first batch begins loading.

## 4. Key Interfaces

| Interface | Input | Output | Main responsibility |
| --- | --- | --- | --- |
| `build_online_omni_mapping_dataset()` | `data_path`, `data_config`, Omni transform | Mapping Dataset or split tuple | Inject the random seed and `transform.is_valid_sample`, then wrap the lazy transform |
| `build_online_mapping_source()` | Source configuration, optional `sample_filter`, optional target lengths | Finite integer-indexed source or split tuple | Route readers, filter, split, and construct split-local blends |
| `OmniDataTransform.is_valid_sample()` | Raw Mapping record | `bool` | Drop empty `messages` and reject incorrect `messages` types |
| `_JsonlMappingSource` (internal) | Local JSONL files and filter | Byte-offset Mapping source | Preserve heterogeneous JSON and provide contiguous indices, split views, and source-file context |
| `build_deepseek_v41_processor()` | Model directory | `DeepseekV41Processor` | Load tokenizer and vision configuration; bind template and image-processing functions |
| `OmniDataTransform.prepare_messages()` | RawSample | Deep-copied messages | Validate `messages` and resolve relative media paths |
| `encode_messages()` / `chat_template` | Messages | Prompt, optional media records | Apply the DeepSeek-V4.1 conversation protocol and insert image placeholders |
| `prepare_vl_inputs()` / `image_processor` | Prompt and image records | Token IDs, token types, `ImageInput` | Read images, generate patches, and expand image placeholders |
| `DeepseekV41OmniTransform.encode_sample()` | RawSample | Model sample | Orchestrate complete single-sample encoding and check length |
| `_get_assistant_start()` | Messages, `ImageInput` | Token index | Locate the supervision start for the final assistant turn |
| `_build_labels()` | Input IDs, token types, start index | Labels | Mask context and all visual tokens |
| `_build_image_fields()` | List of `ImageInput` | Packing-ready image fields | Flatten patches and generate image-grid and start-position metadata |
| `encode_batch()` | Collated/packed batch | Model batch | Generate patch offsets and the image-to-batch mapping |

### 4.1 `build_deepseek_v41_processor()`

The processor reads the following vision parameters from `inference/config.json` in the model directory:

- ViT patch size;
- Minimum pixel count and maximum aspect ratio;
- aligner downsample ratio;
- Maximum visual tokens per image;
- Image placeholder token ID.

The processor also exposes two model-native entry points:

```python
processor.chat_template
processor.image_processor
```

The former binds `encode_messages()` and the latter binds `prepare_vl_inputs()`.

### 4.2 `prepare_messages()`

This interface performs only shallow input preparation:

1. Require `sample["messages"]` to be a nonempty list;
2. Deep-copy messages to avoid mutating the source record;
3. Resolve media paths in top-level content blocks relative to the JSONL directory.

It is not a general multimodal schema converter and does not convert `conversations + images`, Energon dataclasses, or arbitrary custom fields
into `messages`.

### 4.3 The First `chat_template()` Call

The first call from `encode_sample()` passes the complete messages to obtain the full prompt and image records in appearance order. For a ChartQA
sample, for example:

```text
<｜begin▁of▁sentence｜><｜User｜><｜deepseek_image｜>

How many hours were watched on Twitch in May 2021?<｜Assistant｜></think>91.9<｜end▁of▁sentence｜>
```

It also returns:

```python
media = {
    "images": [
        {
            "type": "image",
            "url": "/data/chartqa/images/train/example.jpg",
        }
    ]
}
```

The default is `thinking_mode="chat"`, so the assistant header is followed by `</think>`. Specific reasoning handling is also jointly controlled by
`thinking_mode` and `drop_thinking`.

### 4.4 `image_processor()`

At the template stage, each image occupies one `<｜deepseek_image｜>` token. Image processing reads and resizes the image, generates
ViT patches, and expands this placeholder into:

```text
IMAGE_START
(IMAGE * n_llm_w + IMAGE_NEW_LINE) * n_llm_h
IMAGE_END
```

The number of LLM tokens for one image is:

```text
n_llm_h * (n_llm_w + 1) + 2
```

All visual positions use `image_token_id` in `input_ids`; `token_types` distinguishes their semantics:

| `token_types` value | Meaning |
| ---: | --- |
| `-1` | `TEXT`: ordinary text or text-side special tokens |
| `0` | `IMAGE_START` |
| `1` | `IMAGE` |
| `2` | `IMAGE_NEW_LINE` |
| `3` | `IMAGE_END` |

`ImageInput.start` records the start of the image span in expanded `input_ids`.

### 4.5 The Second `chat_template()` Call and `_get_assistant_start()`

The second template call does not regenerate the full input. It renders only:

```python
messages[:-1]
```

Because the sample must end with an assistant message, removing it leaves the assistant header awaiting answer generation:

```text
<｜begin▁of▁sentence｜><｜User｜><｜deepseek_image｜>

How many hours were watched on Twitch in May 2021?<｜Assistant｜></think>
```

The prefix token count determines where the answer begins. Images in the full `input_ids` have already expanded from single placeholders into multiple visual tokens,
so the supervision start requires this correction:

```text
assistant_start
  = len(tokenizer.encode(prefix_prompt))
  + sum(image_input.types.numel() - 1 for prefix images)
```

Subtracting one accounts for the original image placeholder already included in the prefix length. Larger images have more visual tokens, moving the answer's
actual starting index further back.

This calculation depends on two ordering invariants:

- `media["images"]` follows the order of image placeholders in the prompt;
- `image_inputs` follows the same image-placeholder order.

### 4.6 `_build_labels()`

Labels first copy the complete `input_ids`:

```python
labels = input_ids.clone()
```

Two masks are then applied:

```python
labels[:assistant_start] = IGNORE_INDEX
labels[token_types != TEXT] = IGNORE_INDEX
```

The first masks system, user, historical assistant messages, and the current assistant header before the final assistant answer. The second masks
all visual positions. The resulting supervision region is:

```text
position >= assistant_start and token_types == TEXT
```

For the ChartQA sample above, supervision is:

| Content | Contributes to loss |
| --- | ---: |
| Visual image tokens | No |
| User question | No |
| `<｜Assistant｜></think>` | No |
| `91.9` | Yes |
| `<｜end▁of▁sentence｜>` | Yes |

Here, `TEXT` means nonvisual tokens. The EOS after the answer and text-side special tokens associated with thinking therefore also contribute to loss
according to whether they occur after `assistant_start`.

## 5. Single-sample Output Fields

Samples without images return only the first three token fields; samples with images also return the image fields:

| Field | Typical shape | Meaning |
| --- | --- | --- |
| `input_ids` | `[seq_len]` | Complete token sequence after image expansion |
| `labels` | `[seq_len]` | Labels retaining supervision only for the final assistant text |
| `token_types` | `[seq_len]` | Text and the four image-position types |
| `pixel_values` | `[total_patches, 3, patch_h, patch_w]` | ViT patches for all images |
| `image_patch_lengths` | `[num_images]` | Patch count for each image |
| `image_vit_grid_hw` | `[num_images, 2]` | ViT grid height and width for each image |
| `image_llm_grid_hw` | `[num_images, 2]` | LLM grid height and width for each image |
| `image_token_starts` | `[num_images]` | Start of each image span in the sample token sequence |

If expanded `input_ids` exceeds `max_seq_len`, the transform raises an error instead of silently truncating the sample.

## 6. Packing, Collation, and `encode_batch()`

`SamplePacker` concatenates `input_ids`, `labels`, and `token_types` along the token dimension and image data along the patch dimension, adding
the packed-sequence offset to each sample-local `image_token_starts`. `cu_seq_lens` stores sample boundaries to isolate
packed attention.

The collator can fix the global token length after packing to 1024 with
`dataloader.collate_fn.pad_to_length: 1024`; the default `null` only aligns to TP × CP. The target must be a multiple of TP × CP; packed lengths exceeding it raise an error.
Padding labels are `-100`, and padding is appended to `cu_seq_lens` as a separate final segment. Image patches, grids, and starting positions remain unchanged.
`max_seq_len` controls per-sample filtering, while `token_budget` controls packing selection; all three are independent. For configuration and CP slicing examples, see
[Fix the Input Length After Packing](omni_online_data_guide.md#fix-the-input-length-after-packing).

`encode_batch()` then converts the generic image metadata into the fields required by DeepSeek-V4.1 forward:

```python
image_patch_offsets = [0, cumulative_patch_count...]
image_batch_indices = [image_0_batch, image_1_batch, ...]
```

It also reshapes image-grid fields into two-dimensional matrices and `image_token_starts` into a one-dimensional vector. If the batch has no
`pixel_values`, the interface returns the batch unchanged.

## 7. Debug Logging

When checking encoding logic, log the following boundaries instead of full tensor contents:

- Absolute media paths after `prepare_messages()`;
- The full prompt and image-record count;
- `input_ids.numel()`, expanded image length, and `assistant_start`;
- `(labels != IGNORE_INDEX).sum()`;
- Per-image patch counts, ViT grids, and LLM grids.

The Online transform executes lazily, so its logs do not appear during Dataset construction. The current training configuration sets the root log level to
`INFO`; ordinary `logger.debug()` calls require DEBUG to be explicitly enabled on the corresponding logger. `debug.check_dataset: debug` configures
`hyper_parallel.data.*` dataset loggers and does not automatically raise the verbosity of `hyper_parallel.models.*` loggers.

In multi-device runs, avoid printing complete messages and prompts on every rank. If needed, restrict output to rank 0 or sample only a few
records; otherwise log volume grows quickly with both data volume and rank count.

## 8. Common Errors

### The Last Message Is Not from the Assistant

```text
DeepSeek-V4.1 SFT sample must end with an assistant message
```

The current label policy supervises only the final assistant turn, so training samples must end with an assistant message.

### Image Placeholder Count Does Not Match Image Count

```text
Found N image tokens but got M images
```

Check that each image content block is parsed correctly, and do not manually insert image special tokens into ordinary text.

### Image Path Not Found

Verify that the sample was read through a local JSONL Online source, or use absolute paths when calling manually. The application-level `source` field does not change the
base directory used for path resolution.

### Sample Exceeds the Maximum Length

Image token counts vary with image grids. Length is checked after image-placeholder expansion, so the raw question text length alone does not
represent the final sequence length.

### DEBUG Logs Are Missing

First check the logger's effective level and namespace:

```python
logger.isEnabledFor(logging.DEBUG)
logger.getEffectiveLevel()
```

For a logger under `hyper_parallel.models.*`, setting only `debug.check_dataset: debug` does not enable it.

## 9. Encoder Parallel Input Memory

With `accelerator.encoder_parallel.enabled`, `OmniParallelBatch` validates image metadata on CPU,
then extracts each rank's patch shard according to its Encoder-DP bucket and Encoder-CP segment. Complete `pixel_values`
remain on the DataLoader's CPU and are excluded from ordinary batch device broadcasts.

TP0 still reads data at each decoder CP coordinate. Because ranks in a TP group may belong to different Encoder-DP buckets
or Encoder-CP shards, TP0 transfers each destination's local patches separately, using at most 4 MiB of device staging.
The model receives `encoder_parallel_plan` and `encoder_parallel_inputs`; the latter contains only local packed patches.
Padding and dummy patches for empty buckets are packed on CPU. Vision RoPE also computes only local positions.

This input path takes effect automatically without extra configuration. It reduces additional device memory from transferring complete images; vision-feature gathering, aligner,
and attention workspaces still follow the existing execution plan. Direct calls to the model's native image interface continue to accept complete images.
