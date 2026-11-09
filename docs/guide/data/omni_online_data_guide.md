# Omni Online Data Integration Guide

**English** | [简体中文](omni_online_data_guide.zh.md)

Using the repository's Qwen2-VL example, this guide explains how to feed image-dialogue samples into `VLMTrainer`. The generic Omni path handles reading, filtering,
encoding, packing, and TP/CP batch transport. Transforms or runtime adapters handle model-specific processors, supervision labels,
and additional forward fields. For DeepSeek-V4.1-specific processing, see the
[DeepSeek-V4.1 Online VLM Data Transformation Guide](deepseek_v41_vlm_online_data_guide.md).

## 1. Minimal Sample

`AutoProcessorTransform` accepts nonempty `messages` and requires the last message to be from the assistant. The Qwen2-VL example uses this
processor-compatible format; start with absolute image paths:

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
splits, and routing for other formats. Do not assume every JSON image path is resolved relative to its data file.

## 2. YAML Integration Points

Configure these two sections following [`examples/training_demo/train_vlm.yaml`](../../../examples/training_demo/train_vlm.yaml):

```yaml
dataset:
  model_assets:
    _target_: hyper_parallel.data.omni.build_auto_processor
    pretrained_model_name_or_path: /path/to/Qwen2-VL-model
    local_files_only: true
  data_transform:
    _target_: hyper_parallel.data.omni.AutoProcessorTransform
    max_seq_len: 512
  _target_: hyper_parallel.data.omni.build_online_omni_mapping_dataset
  data_path: /path/to/train.json
  data_config: {}

dataloader:
  _target_: hyper_parallel.data.batching.OmniPackingLoader
  collate_fn:
    _target_: hyper_parallel.data.batching.build_omni_collate_fn
  get_batch:
    _target_: hyper_parallel.data.batching.OmniParallelBatch
    encoder_dp: false
  sampler_type: single
  token_budget: 128
  min_buffered_samples: 1
  num_workers: 0
```

`model_assets` creates the processor, which the configuration resolver injects into `data_transform.processor`. The Dataset builder receives the
transform. `OmniPackingLoader` selects and packs candidate samples, `build_omni_collate_fn` assembles batches,
and `OmniParallelBatch` prepares distributed model inputs. The example uses this class directly without additional hooks.

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
`token_budget` controls the target number of tokens selected for one packing operation, not a per-sample truncation limit.
For long samples, explicitly implement truncation and synchronized image patch/grid handling in the model-specific transform, or clean the data beforehand.

### Fix the Input Length After Packing

Set `pad_to_length` on the collator target to pad global token sequences across different DP branches and micro-batches
to the same length:

```yaml
dataloader:
  _target_: hyper_parallel.data.batching.OmniPackingLoader
  collate_fn:
    _target_: hyper_parallel.data.batching.build_omni_collate_fn
    pad_to_length: 1024
  token_budget: 1024
  min_buffered_samples: 1
```

- `pad_to_length` defaults to `null`, aligning only to TP × CP. A positive integer specifies an exact global target length; booleans are rejected.
  The target must be divisible by runtime TP × CP, or collator construction raises `ValueError`.
- `token_budget` still controls only sample selection, independently of fixed padding. `pad_to_length` is neither inferred from `max_seq_len`,
  `micro_batch_size`, or `token_budget`, nor used to overwrite them.
- Padding occurs after packing, before CP slicing and TP broadcasting. For a target of 1024 and CP=2, the global input is
  `[1, 1024]`, and each CP rank receives `[1, 512]`. DP ranks need no communication to determine the target length.
- If the packed length exceeds the target, `ValueError` is raised without truncation or automatic target growth. A single sample exceeding
  `token_budget` may still be selected; ensure both valid sample lengths and the batching budget stay within the fixed target.
- Labels are padded with `-100` and masks with 0. Real sample boundaries are preserved, with a new padding segment appended. For example,
  padding `cu_seq_lens=[0, 300, 700]` to 1024 produces `[0, 300, 700, 1024]`.
  No empty segment is added if the target is already met. Fixed padding does not affect image patches, grids, or starting positions.

Text Online's `build_online_text_collate_fn` supports the same `pad_to_length` parameter. Use
`--dataloader.collate_fn.pad_to_length=1024` on the command line, or `=null` to restore alignment-only mode. Fixed lengths increase computation and memory
for short sequences, and do not equalize image-patch counts or expert-routing loads across DP ranks.

## 3. From Raw Samples to Model Inputs

```text
JSON/JSONL record
  -> build_online_omni_mapping_dataset: filter empty messages and construct the Mapping source
  -> OmniDataTransform.prepare_messages: copy messages and process recognized media paths
  -> AutoProcessorTransform.encode_sample: encode with the processor and generate labels
  -> OmniPackingLoader: select samples by token_budget; SamplePacker concatenates matching fields
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

In the default path without encoder DP, `OmniParallelBatch` CP-shards only token fields. Image and other modality fields are not CP-sharded;
they remain complete at each TP×CP coordinate for each rank's vision branch. The image bucket
and all-to-all path for `encoder_dp: true` is not implemented; PP/`pp_shared_data` is also unsupported. Models requiring special image-token
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
bash examples/training_demo/run_vlm.sh --training.train_iters=1
```

Logs are written to `output/training_demo/vlm/run_vlm_smoke.log`. First check that the data stage no longer reports
`Omni batch must contain labels`, then verify actual step/loss output. The first check proves only label integration, not successful backward computation
for the vision model. The supplied tiny Qwen2-VL example passed label and batch checks in the local Ascend environment,
but both single-device and eight-device trials encountered a native `SIGSEGV` during backward; the cause remains unconfirmed.
