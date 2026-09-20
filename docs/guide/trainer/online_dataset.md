# Online Dataset Format Conversion

Online text datasets are loaded as raw records and transformed lazily when a
sample is requested. The conversion layer supports ordinary plaintext records,
Instruction/Alpaca records, and ShareGPT conversations without changing the
Trainer entry points.

## Runtime Flow

```mermaid
flowchart TD
    A[JSON/JSONL/CSV/TXT/Parquet/Arrow or HF record] --> B[Online Dataset loader]
    B --> C{Configured transform}
    C -->|plaintext + text_template| D[Render instruction fields]
    C -->|conversation| E[Map role/content fields]
    D --> F[Tokenizer]
    E --> G[Chat template]
    F --> H[Chunk into ModelSamples]
    G --> H
    H --> I[Filter samples without trainable labels]
    I --> J[Online collator and packing]
    J --> K[ParallelBatch and Trainer]
```

The source dataset remains un-tokenized. Tokenization, role normalization,
chunking, and label validation happen at the transform boundary and are
performed lazily.

## Supported Sources

Local files are loaded through Hugging Face Datasets. The Online source accepts
JSON and JSONL (including gzip-compressed files), CSV, TXT, Parquet, and Arrow
files. Files can be passed directly or discovered recursively from a directory;
all files in one configured source must resolve to the same loader format.
Multiple paths can be configured as a comma-separated string or path list.

For example, a directory of Parquet shards can be used directly:

```yaml
dataset:
  data_path: /data/corpus/train
  data_config:
    dataset_type: iterable
```

Hugging Face Hub dataset IDs continue to use `hf_dataset_name` and the native
`datasets.load_dataset` builder. Text conversion fields (`text_keys`,
`text_template`, or conversation role/content keys) are independent of the
container format, provided the loaded records expose the configured columns.

HDF5, SQL, WebDataset, and Lance are not accepted as generic local-file
extensions by this loader. They require dedicated source configuration or a
Hugging Face dataset builder that exposes records through `load_dataset`.

## Instruction / Alpaca

Use `text_template` to render source fields into plaintext:

```yaml
dataset:
  model_assets:
    tokenizer:
      _target_: hyper_parallel.data.text.build_tokenizer.AutoTokenizer.from_pretrained
      pretrained_model_name_or_path: /path/to/model
  data_transform:
    _target_: hyper_parallel.data.text.build_data_transform.build_llm_data_transform
    data_type: plaintext
    text_template: "Instruction: {instruction}\nInput: {input}\nOutput: {output}"
    max_seq_len: 4096
  _target_: hyper_parallel.data.text.build_dataset.build_online_text_dataset
  data_path: /path/to/instruction.jsonl
  data_config:
    dataset_type: mapping
```

The template uses Python `str.format_map` semantics. Missing fields and
non-string rendered values fail at the transform boundary with a descriptive
error. When a template is configured, the loader does not apply the default
`text`-field pre-filter before rendering.

## ShareGPT

ShareGPT records can be converted by configuring their source field names:

```yaml
dataset:
  model_assets:
    chat_template: tokenizer
    tokenizer:
      _target_: hyper_parallel.data.text.build_tokenizer.AutoTokenizer.from_pretrained
      pretrained_model_name_or_path: /path/to/model
  data_transform:
    _target_: hyper_parallel.data.text.build_data_transform.build_llm_data_transform
    data_type: conversation
    text_keys: conversations
    role_key: from
    content_key: value
    max_seq_len: 4096
```

The transform emits the standard `role`/`content` message contract. Built-in
aliases include `human -> user`, `gpt`/`bot`/`model -> assistant`, and
`function`/`observation -> tool`. A `role_map` can override or extend these
aliases. Message metadata is preserved.

## Output Contract

Both transforms return a list of model samples containing `input_ids` and
`labels`. A single source record may produce multiple samples when it exceeds
`max_seq_len`. The online wrapper filters samples whose labels contain no
trainable token, and the online collator packs the resulting samples while
preserving document boundaries.

Runnable smoke-test configurations are provided in:

- `examples/training_demo/train_online_instruction.yaml`
- `examples/training_demo/train_online_sharegpt.yaml`

The matching demo records are under `examples/training_demo/data/`.
