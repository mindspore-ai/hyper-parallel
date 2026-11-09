# Offline Dataset Preparation Guide

**English** | [简体中文](offline_preparation_guide.zh.md)

This guide explains how to convert local JSON/JSONL files or Hugging Face datasets into training-ready Megatron Indexed Datasets.

Two command-line entry points are available:

- `offline_preparation`: Process local JSON/JSONL files, directories, or glob patterns.
- `huggingface_offline`: Download and process Hugging Face datasets, or process local data directly.

Each `json_key` produces a pair of `.bin` and `.idx` files.

## 1. Prerequisites

Run commands from the repository root, with HyperParallel, Transformers, and Datasets installed in the current Python environment.

Supported input extensions:

- `.json`
- `.jsonl`
- `.json.gz`
- `.jsonl.gz`

Each input line must be a JSON object, for example:

```json
{"text": "HyperParallel makes distributed training easier."}
{"text": "This is the second document."}
```

For multiple fields, every line must contain all fields specified by `--json-keys`:

```json
{"prompt": "Question", "answer": "Answer"}
```

## 2. Quick Start

### 2.1 Process a Local File

```bash
python -m hyper_parallel.data.tools.offline_preparation \
    --dataset-name-or-path ./data/train.jsonl \
    --output-prefix ./offline_datasets/train \
    --tokenizer-name-or-path Qwen/Qwen3-30B-A3B
```

### 2.2 Process a Local Directory

Supported files are processed together in sorted path order. Directory scanning is not recursive.

```bash
python -m hyper_parallel.data.tools.offline_preparation \
    --dataset-name-or-path ./data/train \
    --output-prefix ./offline_datasets/train \
    --tokenizer-name-or-path Qwen/Qwen3-30B-A3B \
    --workers 8 \
    --partitions 2
```

### 2.3 Match Files with a Glob

Quote glob patterns to prevent the shell from expanding them early:

```bash
python -m hyper_parallel.data.tools.offline_preparation \
    --dataset-name-or-path "./data/train-*.jsonl" \
    --output-prefix ./offline_datasets/train \
    --tokenizer-name-or-path Qwen/Qwen3-30B-A3B
```

### 2.4 Download and Process a Hugging Face Dataset

```bash
python -m hyper_parallel.data.tools.huggingface_offline \
    --dataset Salesforce/wikitext \
    --dataset-subset wikitext-103-raw-v1 \
    --dataset-split train \
    --output-prefix ./offline_datasets/wikitext/train \
    --tokenizer Qwen/Qwen3-30B-A3B \
    --workers 8 \
    --partitions 2
```

`--dataset` also accepts an existing local JSON/JSONL file or directory. Existing local paths are processed directly without accessing
Hugging Face Hub.

## 3. Common Configuration

### 3.1 Select Data Fields

The default field is `text`:

```bash
--json-keys text
```

Process multiple fields together:

```bash
--json-keys prompt answer
```

Each field produces its own `.bin` / `.idx` pair.

### 3.2 Configure EOD

By default, the tool appends the tokenizer's EOS token to every nonempty document, falling back to SEP when EOS is unavailable.

This is enabled by default without an argument. To disable it:

```bash
--append-eod false
```

To enable it explicitly:

```bash
--append-eod true
```

Combining `--pack-to-seq-len` with `--append-eod false` emits a warning. Fixed-length samples are still generated,
but original document boundaries do not contain EOD tokens.

### 3.3 Configure the Tokenizer

Use a Hugging Face model name:

```bash
--tokenizer-name-or-path Qwen/Qwen3-30B-A3B
```

The local entry point uses `--tokenizer-name-or-path`; the Hugging Face entry point uses `--tokenizer`. Both accept local tokenizer
directories. Only loading through Hugging Face `AutoTokenizer` is currently supported; custom tokenizer classes are not accepted. The tool
emits a corresponding warning at runtime.

Fast tokenizers are the default. To use a slow tokenizer:

```bash
--tokenizer-use-fast false
```

If the tokenizer repository requires custom code execution:

```bash
--trust-remote-code
```

Enable this only for trusted repositories.

Add special tokens:

```bash
--add-special-tokens '<|user|>' '<|assistant|>'
```

Specify a chat template:

```bash
--chat-template "{{ messages }}"
```

### 3.4 Configure Parallel Processing

```bash
--workers 8 --partitions 2
```

- `workers` is the total tokenizer worker count.
- `partitions` is the data partition count.
- `workers` must be divisible by `partitions`.
- Each worker loads a tokenizer; reduce `workers` when memory is insufficient.

To assign data consecutively in input order:

```bash
--keep-sequential-samples
```

Otherwise, records are assigned to partitions round-robin.

### 3.5 Generate Fixed-Length Samples

For a training sequence length of 4096:

```bash
--pack-to-seq-len 4096
```

Each output document then contains `4097` tokens, from which inputs and labels of length `4096` are constructed. Any final remainder shorter than
`4097` tokens in each partition is discarded without padding.

### 3.6 Sentence Splitting

Enable NLTK Punkt sentence splitting:

```bash
--split-sentences
```

Select the splitting language:

```bash
--split-sentences --lang english
```

Preserve newlines while splitting:

```bash
--split-sentences --keep-newlines
```

### 3.7 Select Worker Count Automatically

```bash
--find-optimal-num-workers \
--workers-to-check 16 32 64 \
--max-documents 100000
```

Candidate worker counts must be positive and divisible by `partitions`. The tool benchmarks candidates and reports throughput and a recommended configuration.

## 4. Local Entry-Point Arguments

Command:

```bash
python -m hyper_parallel.data.tools.offline_preparation [arguments]
```

| Argument | Required | Default | Description |
|---|---:|---:|---|
| `--dataset-name-or-path` | Yes | None | Local file, directory, or glob pattern |
| `--output-prefix` | Yes | None | Output path prefix without `.bin` or `.idx` suffix |
| `--json-keys` | No | `text` | One or more fields to process |
| `--tokenizer-name-or-path` | Yes | None | Hugging Face tokenizer name or local directory |
| `--chat-template` | No | `None` | Override the tokenizer's chat template |
| `--add-special-tokens` | No | `None` | Add one or more special tokens |
| `--tokenizer-use-fast` | No | `true` | Use a fast tokenizer: `true` or `false` |
| `--trust-remote-code` | No | Disabled | Allow custom code from the tokenizer repository |
| `--split-sentences` | No | Disabled | Split sentences with NLTK Punkt |
| `--keep-newlines` | No | Disabled | Preserve newlines during sentence splitting; requires `--split-sentences` |
| `--lang` | No | `english` | Punkt sentence-splitting language |
| `--append-eod` | No | `true` | Append EOD to nonempty documents: `true` or `false` |
| `--pack-to-seq-len` | No | `None` | Training sequence length; output documents have this length plus one; discard incomplete final samples |
| `--keep-sequential-samples` | No | Disabled | Assign samples consecutively in input order |
| `--keep-partition-files` | No | Disabled | Retain temporary JSON partition files |
| `--workers` | No | `8` | Total tokenizer worker count |
| `--partitions` | No | `1` | Data partition count; must be positive |
| `--find-optimal-num-workers` | No | Disabled | Benchmark worker candidates and report the fastest configuration |
| `--workers-to-check` | No | `16 32 64` | Candidate worker counts for automatic benchmarking |
| `--max-documents` | No | `100000` | Maximum documents per partition during worker benchmarking |
| `--log-interval` | No | `1000` | Documents processed between progress reports |

## 5. Hugging Face Entry-Point Arguments

Command:

```bash
python -m hyper_parallel.data.tools.huggingface_offline [arguments]
```

### 5.1 Dataset and Download Arguments

| Argument | Required | Default | Description |
|---|---:|---:|---|
| `--dataset` | Yes | None | Hugging Face dataset ID or existing local file/directory |
| `--dataset-subset` | No | `None` | Dataset configuration/subset name |
| `--dataset-split` | No | `train` | Split to process |
| `--revision` | No | `None` | Dataset revision, tag, or commit SHA; the Hub default is used if omitted |
| `--cache-dir` | No | `None` | Hugging Face Datasets cache directory |
| `--data-dir` | No | `None` | Data directory within the dataset repository |
| `--data-files` | No | `None` | One or more source files |
| `--num-proc` | No | `None` | Processes for Hugging Face data preparation; must be positive |
| `--download-dir` | No | Automatically generated | Downloaded JSONL directory, defaulting to `./download_datasets/<dataset>/` |

`revision` requires no prior local configuration. Supply it only when pinning a data version, for example:

```bash
--revision main
```

### 5.2 Preprocessing Arguments

| Argument | Required | Default | Description |
|---|---:|---:|---|
| `--json-keys` | No | `text` | One or more fields to process |
| `--tokenizer` | Yes | None | Hugging Face tokenizer name or local directory |
| `--tokenizer-use-fast` | No | `true` | Use a fast tokenizer: `true` or `false` |
| `--trust-remote-code` | No | Disabled | Allow custom code from the dataset or tokenizer repository |
| `--chat-template` | No | `None` | Override the tokenizer's chat template |
| `--add-special-tokens` | No | `None` | Add one or more special tokens |
| `--split-sentences` | No | Disabled | Split sentences with NLTK Punkt |
| `--keep-newlines` | No | Disabled | Preserve newlines during sentence splitting; requires `--split-sentences` |
| `--lang` | No | `english` | Punkt sentence-splitting language |
| `--append-eod` | No | `true` | Append EOD to nonempty documents: `true` or `false` |
| `--pack-to-seq-len` | No | `None` | Training sequence length; output documents have this length plus one; discard incomplete final samples |
| `--workers` | No | `8` | Total tokenizer worker count |
| `--partitions` | No | `1` | Data partition count; must be positive |
| `--keep-sequential-samples` | No | Disabled | Assign samples consecutively in input order |
| `--keep-partition-files` | No | Disabled | Retain temporary JSON partition files |
| `--find-optimal-num-workers` | No | Disabled | Benchmark worker candidates and report the fastest configuration |
| `--workers-to-check` | No | `16 32 64` | Candidate worker counts for automatic benchmarking |
| `--max-documents` | No | `100000` | Maximum documents per partition during worker benchmarking |
| `--log-interval` | No | `1000` | Documents processed between progress reports |

## 6. Python Interface

Configure and run preparation in Python with `OfflinePreparationConfig`:

```python
from hyper_parallel.data.tools.offline_config import OfflinePreparationConfig
from hyper_parallel.data.tools.offline_preparation import prepare_offline_dataset

config = OfflinePreparationConfig(
    dataset_name_or_path="./data/train.jsonl",
    output_prefix="./offline_datasets/train",
    tokenizer_name_or_path="Qwen/Qwen3-30B-A3B",
    json_keys=["text"],
    append_eod=True,
    workers=8,
    partitions=2,
    keep_sequential_samples=True,
)

prepare_offline_dataset(config.to_offline_args())
```

Configuration field names match command-line meanings, with these mappings:

- `dataset_subset_name` corresponds to `--dataset-subset`.
- `tokenizer_name_or_path` corresponds to the local `--tokenizer-name-or-path` or Hugging Face `--tokenizer`.
- `data_files` accepts a string, list of strings, or dictionary grouped by split.
- In Python configuration, `workers` defaults to `8` and `keep_sequential_samples` to `True`; other common defaults match the CLI.

`prepare_offline_dataset()` processes existing local data only. To download from the Hugging Face Hub, use the
`huggingface_offline` command-line entry point.

## 7. Output Files

Without sentence splitting, output filenames are:

```text
<output-prefix>_<json-key>_document.bin
<output-prefix>_<json-key>_document.idx
```

With `--split-sentences`, output filenames are:

```text
<output-prefix>_<json-key>_sentence.bin
<output-prefix>_<json-key>_sentence.idx
```

For example:

```bash
--output-prefix ./offline_datasets/train \
--json-keys text
```

Produces:

```text
./offline_datasets/train_text_document.bin
./offline_datasets/train_text_document.idx
```

With multiple partitions, each partition's `.bin/.idx` files are also retained. `--keep-partition-files` controls temporary
JSON partition files, not the final `.bin/.idx` files.

## 8. Complete Examples

### 8.1 Local Data with Multiple Fields

```bash
python -m hyper_parallel.data.tools.offline_preparation \
    --dataset-name-or-path "./data/train-*.jsonl" \
    --output-prefix ./offline_datasets/sft/train \
    --json-keys prompt answer \
    --tokenizer-name-or-path Qwen/Qwen3-30B-A3B \
    --append-eod true \
    --workers 8 \
    --partitions 2 \
    --keep-sequential-samples \
    --log-interval 1000
```

### 8.2 Fixed-Length Processing of a Hugging Face Dataset

```bash
python -m hyper_parallel.data.tools.huggingface_offline \
    --dataset Salesforce/wikitext \
    --dataset-subset wikitext-103-raw-v1 \
    --dataset-split train \
    --download-dir ./download_datasets/wikitext \
    --output-prefix ./offline_datasets/wikitext/train \
    --json-keys text \
    --tokenizer gpt2 \
    --append-eod true \
    --pack-to-seq-len 4096 \
    --workers 8 \
    --partitions 1
```

## 9. Frequently Asked Questions

### 9.1 `workers must be divisible by partitions`

Set `workers` to an integer multiple of `partitions`, for example `--workers 8 --partitions 2`.

### 9.2 EOD Token Not Found

`append_eod` is enabled by default, requiring `eos_token_id` or `sep_token_id`. If EOD is unnecessary:

```bash
--append-eod false
```

### 9.3 Tokenizer Workers Use Too Much Memory

Reduce `--workers`; every worker loads its own tokenizer.

### 9.4 Hugging Face Dataset Contains Custom Code

After confirming that the dataset source is trusted, add:

```bash
--trust-remote-code
```

### 9.5 View Command-Line Help

```bash
python -m hyper_parallel.data.tools.offline_preparation --help
python -m hyper_parallel.data.tools.huggingface_offline --help
```
