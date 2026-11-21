# Omni Online Data Integration Guide

**English** | [简体中文](omni_online_data_guide.zh.md)

For YAML settings and worked examples of sample length, packing, and padding, see
[Dataset usage](../../../hyper_parallel/data/README.md#10-configuration-examples).

Using the repository's Qwen2-VL example, this guide explains how to feed image-dialogue samples into `VLMTrainer`. The generic Omni path handles reading, filtering,
encoding, packing, and TP/CP batch transport. Transforms or runtime adapters handle model-specific processors, supervision labels,
and additional forward fields. For DeepSeek-V4.1-specific processing, see the
[DeepSeek-V4.1 Online VLM Data Transformation Guide](deepseek_v41_vlm_online_data_guide.md).

## 1. Minimal Sample

`AutoProcessorTransform` accepts nonempty `messages` and requires the last message to be from the assistant. The Qwen2-VL example uses this
processor-compatible format; use absolute image paths or paths relative to the local source file:

```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "image", "url": "/data/images/example.jpg"},
        {"type": "text", "text": "Describe this image briefly."}
      ]
    },
    {
      "role": "assistant",
      "content": [{"type": "text", "text": "A city skyline."}]
    }
  ]
}
```

Place multiple objects in a JSON array (`.json`) or write one object per line in `.jsonl`. The example script reads
`synthetic.json` by default; set `DATA_PATH` to use your own file. During source indexing,
`is_valid_sample()` drops `messages=[]`; missing `messages` or a non-list value raises an error. See the DeepSeek-V4.1 guide above for native JSONL indexing, relative media paths,
splits, and routing for other formats. Local JSONL/JSON/Parquet/Arrow Mapping records retain their source file's
absolute path, which determines the base directory for relative media paths, including multi-file sources.
Supply absolute media paths when invoking transforms directly or using records without source provenance.

## 2. YAML Integration Points

Minimal templates list required settings only. See the separate
[Dataset defaults table](../../../hyper_parallel/data/README.md#defaults-omitted-from-minimal-templates),
Trainer-injected arguments, and unsupported settings. Add optional overrides only when changing behavior.

Configure these two sections following [`examples/training_demo/train_vlm.yaml`](../../../examples/training_demo/train_vlm.yaml):

```yaml
training:
  epochs: 1
  max_steps: 100

dataset:
  training_mode: sft
  model_assets:
    _target_: hyper_parallel.data.omni.build_auto_processor
    pretrained_model_name_or_path: /path/to/Qwen2-VL-model
  data_transform:
    _target_: hyper_parallel.data.omni.AutoProcessorTransform
  _target_: hyper_parallel.data.omni.build_online_omni_mapping_dataset
  data_path: /path/to/train.json
  data_config:
    max_seq_len: 512

dataloader:
  _target_: hyper_parallel.data.batching.OmniPackingLoader
  collate_fn:
    _target_: hyper_parallel.data.batching.build_omni_collate_fn
  get_batch:
    _target_: hyper_parallel.data.batching.OmniParallelBatch
```

`model_assets` creates the processor, which the configuration resolver injects into `data_transform.processor`. The Dataset builder receives the
transform. `OmniPackingLoader` selects and packs candidate samples, `build_omni_collate_fn` assembles batches,
and `OmniParallelBatch` prepares distributed model inputs. The example uses this class directly without additional hooks.

VLMTrainer requires explicit `dataset.training_mode`; the generic AutoProcessor path uses `sft`.
Online `max_steps` counts optimizer steps per training epoch; this example runs 100 steps in total.
Source EOF repeats by default, independently of training epoch boundaries. `train_iters/train_samples` apply to Offline.

You can also inspect the Dataset interface independently of the Trainer:

```python
from types import SimpleNamespace

from hyper_parallel.data.omni import (
    AutoProcessorTransform,
    build_auto_processor,
    build_online_omni_mapping_dataset,
)

processor = build_auto_processor(
    pretrained_model_name_or_path="/path/to/Qwen2-VL-model",
    local_files_only=True,
)
transform = AutoProcessorTransform(max_seq_len=512, processor=processor)
dataset = build_online_omni_mapping_dataset(
    data_path="/path/to/train.json",
    data_config={},
    transform=transform,
    training_config=SimpleNamespace(seed=42),
)
encoded_sample = dataset[0]  # Without splits, the builder returns a Mapping Dataset.
print(encoded_sample["input_ids"].shape, encoded_sample["labels"].shape)
```

With ratio splits or presplit paths configured, the Dataset builder returns `(train, valid, test)`, using
`None` for absent splits. `transform.is_valid_sample` is automatically passed to the source builder as the raw-record filter; do not pass it again manually.

`AutoProcessorTransform` rejects samples exceeding `max_seq_len`; it **does not truncate or pad media sequences**.
`max_seq_len` also limits each packed sequence; B packed sequences form one micro-batch.
For long samples, explicitly implement truncation and synchronized image patch/grid handling in the model-specific transform, or clean the data beforehand.

### Fix the Input Length After Packing

Set `dataset.data_config.max_seq_len` once as the per-sequence packing limit S.
The loader independently packs B=`micro_batch_size` sequences, then the collator pads them:

```yaml
dataset:
  data_config:
    max_seq_len: 1024
  data_transform:

dataloader:
  _target_: hyper_parallel.data.batching.OmniPackingLoader
  collate_fn:
    _target_: hyper_parallel.data.batching.build_omni_collate_fn
```

- `pad_to_length: true` (the default) produces `[B,S]`; false pads to the longest
  actual packed sequence rounded up to TP×CP alignment, producing `[B,T]` with T≤S.
  S must be divisible by runtime TP×CP.
- Samples longer than S are skipped. A sample that does not fit the current packed sequence
  remains buffered for a later sequence. Repeating the source retains the buffer.
- Padding runs before CP slicing and TP broadcasting. B=2, S=1024, CP=2 gives `[2,1024]`
  globally and `[2,512]` on each CP rank. No DP communication is needed to determine S.
- Token padding is 0, labels are -100, and masks are 0. Each packed sequence's padding
  becomes a separate `cu_seq_lens` segment; image patches, grids, and local starts are retained.
- `pad_to_length` accepts only a boolean; packing and fixed padding share S.

Text Online's `build_online_text_collate_fn` uses the same settings.

## 3. From Raw Samples to Model Inputs

```text
JSON/JSONL record
  -> build_online_omni_mapping_dataset: filter empty messages and construct the Mapping source
  -> OmniDataTransform.prepare_messages: copy messages and process recognized media paths
  -> AutoProcessorTransform.encode_sample: encode with the processor and generate labels
  -> OmniPackingLoader: pack B sequences up to max_seq_len; SamplePacker concatenates matching fields
  -> OmniCollator: merge token and variable-length image fields
  -> OmniParallelBatch: CP-shard token fields, broadcast over TP, and generate loss_mask
  -> VLMTrainer -> model forward/loss/backward
```

`AutoProcessorTransform.encode_sample()` encodes each record twice:

1. Encode the full conversation with `add_generation_prompt=False` to obtain `input_ids`, `attention_mask`, image fields, and other model inputs.
2. Remove the final assistant response and use `add_generation_prompt=True` to obtain the prompt length, including the assistant start marker.

It first checks whether prompt tokens are a prefix of the complete conversation. If only token merging at a template boundary occurs, it verifies
the decoded text and common token prefix, conservatively masking boundary tokens. Template rewrites whose boundaries cannot be established raise an error.
The complete input tokens are not reconstructed by concatenation. It then copies `input_ids` into `labels`: the prompt region and
multimodal positions where `mm_token_type_ids != 0` are set to `-100`; final assistant text retains its token IDs. For example:

```text
input_ids:  [system / user / image / assistant prefix | answer / end]
labels:     [-100  / -100 / -100  / -100             | answer / end]
```

The processor itself does not automatically generate training `labels`. Here, labels have the same length as `input_ids` and have not been manually shifted left;
the example's Qwen2-VL causal LM applies the shift during loss computation. The transform returns 1-D token fields for one sample and complete
modality fields such as `pixel_values` and `image_grid_thw`. `OmniParallelBatch` requires existing `labels` and derives
`loss_mask` from `labels >= 0` when absent. Do not leave every token as a valid label, or user prompts and image placeholders
will also be supervised. Samples with no remaining supervised tokens raise an error.

`OmniParallelBatch` CP-shards only token fields. Image and other modality fields are not CP-sharded;
they remain complete at each TP×CP coordinate for each rank's vision branch. The image bucket
and all-to-all encoder-DP path is not implemented; the current YAML rejects `encoder_dp`.
PP/`pp_shared_data` is also unsupported. Models requiring special image-token
mappings or extra forward arguments should provide their own `OmniDataTransform` and, when needed,
`runtime_input_adapter`. Keep model-specific field rules out of the generic collator.

## 4. Integrate Another Omni Model

Keep the Dataset, packing, and parallel batch entry points unchanged, replacing only model-specific parts:

1. Configure a processor that understands the model's message format, or supply a custom `model_assets` builder.
2. Subclass `OmniDataTransform` and implement `encode_sample()`, returning at least equal-length `input_ids` and `labels`,
   with non-loss tokens set to `-100`. Image/video fields must match model forward argument names and shapes.
   To select samples from lightweight metadata before expensive media encoding, use the paired
   `preencode_sample()` / `postencode_sample()` methods instead; implement `encode_batch()` for batch-level processing.
3. Update YAML `dataset.model_assets` and `dataset.data_transform` to the new targets. If model-specific positions,
   masks, or image mappings must be constructed after TP/CP transport, configure an adapter in `dataloader.get_batch.runtime_input_adapter`.
   Ordinary processor fields do not need this hook.

`build_online_omni_mapping_dataset()` verifies that the transform is an `OmniDataTransform`; a class implementing only
`__call__(record)` is not a direct replacement. See the model-specific implementation in
[`DeepseekV41OmniTransform`](../../../hyper_parallel/models/deepseek_v41/adapter/data/transform_fn.py)
for reference.

## 5. Run the Example and Check Results

[`examples/training_demo/run_vlm.sh`](../../../examples/training_demo/run_vlm.sh) launches eight processes by default
and forwards `MODEL_PATH`, `DATA_PATH`, and command-line overrides to YAML. To use local models and samples:

```bash
cd /path/to/hyper-parallel
MODEL_PATH=/path/to/Qwen2-VL-model \
DATA_PATH=/path/to/train.json \
RUN_NAME=vlm_smoke \
bash examples/training_demo/run_vlm.sh --training.epochs=1 --training.max_steps=1
```

Logs are written to `output/training_demo/vlm/run_vlm_smoke.log`. First check that the data stage no longer reports
`Omni batch must contain labels`, then verify actual step/loss output. The first check proves only label integration, not successful backward computation
for the vision model. The supplied tiny Qwen2-VL example passed label and batch checks in the local Ascend environment,
but both single-device and eight-device trials encountered a native `SIGSEGV` during backward; the cause remains unconfirmed.
