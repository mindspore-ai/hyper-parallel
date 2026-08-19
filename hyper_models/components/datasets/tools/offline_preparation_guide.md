# Offline Dataset Preparation Guide

本文档介绍如何将本地 JSON/JSONL 文件或 Hugging Face 数据集转换为训练可用的 Megatron Indexed Dataset。

工具提供两个命令行入口：

- `offline_preparation`：处理本地 JSON/JSONL 文件、目录或 glob 表达式。
- `huggingface_offline`：下载 Hugging Face 数据集后进行处理，也可以直接处理本地数据。

处理完成后，每个 `json_key` 会生成一组 `.bin` 和 `.idx` 文件。

## 1. 使用前准备

在仓库根目录执行命令，并确保当前 Python 环境已经安装 HyperParallel、Transformers 和 Datasets。

输入文件支持以下扩展名：

- `.json`
- `.jsonl`
- `.json.gz`
- `.jsonl.gz`

输入文件中的每一行必须是一个 JSON 对象。例如：

```json
{"text": "HyperParallel makes distributed training easier."}
{"text": "This is the second document."}
```

如果使用多个字段，每行应包含所有通过 `--json-keys` 指定的字段：

```json
{"prompt": "Question", "answer": "Answer"}
```

## 2. 快速开始

### 2.1 处理本地文件

```bash
python -m hyper_models.components.datasets.tools.offline_preparation \
    --dataset-name-or-path ./data/train.jsonl \
    --output-prefix ./offline_datasets/train \
    --tokenizer-name-or-path Qwen/Qwen3-30B-A3B
```

### 2.2 处理本地目录

目录中的受支持文件会按路径排序后统一处理。目录扫描不递归。

```bash
python -m hyper_models.components.datasets.tools.offline_preparation \
    --dataset-name-or-path ./data/train \
    --output-prefix ./offline_datasets/train \
    --tokenizer-name-or-path Qwen/Qwen3-30B-A3B \
    --workers 8 \
    --partitions 2
```

### 2.3 使用 glob 匹配文件

建议使用引号包裹 glob，避免 shell 提前展开：

```bash
python -m hyper_models.components.datasets.tools.offline_preparation \
    --dataset-name-or-path "./data/train-*.jsonl" \
    --output-prefix ./offline_datasets/train \
    --tokenizer-name-or-path Qwen/Qwen3-30B-A3B
```

### 2.4 下载并处理 Hugging Face 数据集

```bash
python -m hyper_models.components.datasets.tools.huggingface_offline \
    --dataset Salesforce/wikitext \
    --dataset-subset wikitext-103-raw-v1 \
    --dataset-split train \
    --output-prefix ./offline_datasets/wikitext/train \
    --tokenizer Qwen/Qwen3-30B-A3B \
    --workers 8 \
    --partitions 2
```

`--dataset` 也接受已存在的本地 JSON/JSONL 文件或目录。本地路径存在时，工具直接处理本地数据，不访问
Hugging Face Hub。

## 3. 常用配置

### 3.1 指定数据字段

默认处理 `text` 字段：

```bash
--json-keys text
```

一次处理多个字段：

```bash
--json-keys prompt answer
```

每个字段会单独生成一组 `.bin` 和 `.idx` 文件。

### 3.2 配置 EOD

工具默认在每个非空文档末尾追加 tokenizer 的 EOS token；tokenizer 没有 EOS 时尝试使用 SEP token。

默认开启，无需传参。需要关闭时使用：

```bash
--append-eod false
```

需要显式开启时使用：

```bash
--append-eod true
```

如果启用了 `--pack-to-seq-len`，同时将 `--append-eod` 设置为 `false`，工具会输出 warning。此时仍会生成
定长样本，但原始文档边界不会包含 EOD token。

### 3.3 配置 tokenizer

使用 Hugging Face 模型名称：

```bash
--tokenizer-name-or-path Qwen/Qwen3-30B-A3B
```

本地入口使用 `--tokenizer-name-or-path`，Hugging Face 入口使用 `--tokenizer`。两者都支持本地 tokenizer
目录。当前仅支持通过 Hugging Face `AutoTokenizer` 加载 tokenizer，暂不支持传入自定义 tokenizer 类；工具运行时
会输出相应 warning。

默认使用 fast tokenizer。需要使用 slow tokenizer 时：

```bash
--tokenizer-use-fast false
```

如果 tokenizer 仓库需要执行自定义代码：

```bash
--trust-remote-code
```

仅对可信仓库开启该选项。

添加额外 special token：

```bash
--add-special-tokens '<|user|>' '<|assistant|>'
```

指定 chat template：

```bash
--chat-template "{{ messages }}"
```

### 3.4 配置并行处理

```bash
--workers 8 --partitions 2
```

- `workers` 是 tokenizer worker 总数。
- `partitions` 是数据分区数。
- `workers` 必须能被 `partitions` 整除。
- 每个 worker 都会加载 tokenizer；内存不足时应降低 `workers`。

希望按输入顺序连续分配数据时使用：

```bash
--keep-sequential-samples
```

未设置时，数据按 round-robin 方式分配到各 partition。

### 3.5 生成定长样本

下面的配置适用于训练序列长度为 4096 的场景：

```bash
--pack-to-seq-len 4096
```

启用后，每个输出 document 包含 `4097` 个 token，用于构造长度为 `4096` 的 input 和 label。每个 partition
最后不足 `4097` 个 token 的残余部分会被直接丢弃，不进行 padding。

### 3.6 分句处理

启用 NLTK Punkt 分句：

```bash
--split-sentences
```

指定分句语言：

```bash
--split-sentences --lang english
```

分句时保留换行：

```bash
--split-sentences --keep-newlines
```

### 3.7 自动选择 worker 数量

```bash
--find-optimal-num-workers \
--workers-to-check 16 32 64 \
--max-documents 100000
```

候选 worker 数都必须大于零，并且能被 `partitions` 整除。工具会测试候选配置并输出处理速度和推荐配置。

## 4. 本地入口参数

入口命令：

```bash
python -m hyper_models.components.datasets.tools.offline_preparation [参数]
```

| 参数 | 是否必填 | 默认值 | 配置说明 |
|---|---:|---:|---|
| `--dataset-name-or-path` | 是 | 无 | 本地文件、目录或 glob 表达式 |
| `--output-prefix` | 是 | 无 | 输出路径前缀，不要添加 `.bin` 或 `.idx` 后缀 |
| `--json-keys` | 否 | `text` | 要处理的字段，可配置一个或多个字段 |
| `--tokenizer-name-or-path` | 是 | 无 | Hugging Face tokenizer 名称或本地目录 |
| `--chat-template` | 否 | `None` | 覆盖 tokenizer 的 chat template |
| `--add-special-tokens` | 否 | `None` | 添加一个或多个 special token |
| `--tokenizer-use-fast` | 否 | `true` | 是否使用 fast tokenizer，使用 `true` 或 `false` |
| `--trust-remote-code` | 否 | 关闭 | 允许 tokenizer 仓库执行自定义代码 |
| `--split-sentences` | 否 | 关闭 | 使用 NLTK Punkt 分句 |
| `--keep-newlines` | 否 | 关闭 | 分句时保留换行，需要配合 `--split-sentences` |
| `--lang` | 否 | `english` | Punkt 分句语言 |
| `--append-eod` | 否 | `true` | 是否在非空文档末尾追加 EOD，使用 `true` 或 `false` |
| `--pack-to-seq-len` | 否 | `None` | 训练序列长度；输出长度为该值加一的 document，最后不足一条的残余样本会被丢弃 |
| `--keep-sequential-samples` | 否 | 关闭 | 按输入顺序连续分配样本 |
| `--keep-partition-files` | 否 | 关闭 | 保留临时 JSON partition 文件 |
| `--workers` | 否 | `8` | tokenizer worker 总数 |
| `--partitions` | 否 | `1` | 数据分区数，必须大于零 |
| `--find-optimal-num-workers` | 否 | 关闭 | 测试候选 worker 数并报告最快配置 |
| `--workers-to-check` | 否 | `16 32 64` | 自动测试的 worker 候选值 |
| `--max-documents` | 否 | `100000` | worker 测试时每个 partition 最多处理的文档数 |
| `--log-interval` | 否 | `1000` | 每处理多少个文档输出一次进度 |

## 5. Hugging Face 入口参数

入口命令：

```bash
python -m hyper_models.components.datasets.tools.huggingface_offline [参数]
```

### 5.1 数据集和下载参数

| 参数 | 是否必填 | 默认值 | 配置说明 |
|---|---:|---:|---|
| `--dataset` | 是 | 无 | Hugging Face dataset ID，或已存在的本地文件/目录 |
| `--dataset-subset` | 否 | `None` | 数据集 configuration/subset 名称 |
| `--dataset-split` | 否 | `train` | 要处理的数据 split |
| `--revision` | 否 | `None` | 数据集 revision、tag 或 commit SHA；不指定时由 Hub 使用默认 revision |
| `--cache-dir` | 否 | `None` | Hugging Face Datasets 缓存目录 |
| `--data-dir` | 否 | `None` | 数据集仓库内部的数据目录 |
| `--data-files` | 否 | `None` | 指定一个或多个源数据文件 |
| `--num-proc` | 否 | `None` | Hugging Face 数据准备阶段使用的进程数，必须大于零 |
| `--download-dir` | 否 | 自动生成 | 下载后的 JSONL 保存目录，默认位于 `./download_datasets/<dataset>/` |

`revision` 不需要预先在本地配置。只有需要固定数据版本时才传入，例如：

```bash
--revision main
```

### 5.2 预处理参数

| 参数 | 是否必填 | 默认值 | 配置说明 |
|---|---:|---:|---|
| `--json-keys` | 否 | `text` | 要处理的字段，可配置一个或多个字段 |
| `--tokenizer` | 是 | 无 | Hugging Face tokenizer 名称或本地目录 |
| `--tokenizer-use-fast` | 否 | `true` | 是否使用 fast tokenizer，使用 `true` 或 `false` |
| `--trust-remote-code` | 否 | 关闭 | 允许数据集或 tokenizer 仓库执行自定义代码 |
| `--chat-template` | 否 | `None` | 覆盖 tokenizer 的 chat template |
| `--add-special-tokens` | 否 | `None` | 添加一个或多个 special token |
| `--split-sentences` | 否 | 关闭 | 使用 NLTK Punkt 分句 |
| `--keep-newlines` | 否 | 关闭 | 分句时保留换行，需要配合 `--split-sentences` |
| `--lang` | 否 | `english` | Punkt 分句语言 |
| `--append-eod` | 否 | `true` | 是否在非空文档末尾追加 EOD，使用 `true` 或 `false` |
| `--pack-to-seq-len` | 否 | `None` | 训练序列长度；输出长度为该值加一的 document，最后不足一条的残余样本会被丢弃 |
| `--workers` | 否 | `8` | tokenizer worker 总数 |
| `--partitions` | 否 | `1` | 数据分区数，必须大于零 |
| `--keep-sequential-samples` | 否 | 关闭 | 按输入顺序连续分配样本 |
| `--keep-partition-files` | 否 | 关闭 | 保留临时 JSON partition 文件 |
| `--find-optimal-num-workers` | 否 | 关闭 | 测试候选 worker 数并报告最快配置 |
| `--workers-to-check` | 否 | `16 32 64` | 自动测试的 worker 候选值 |
| `--max-documents` | 否 | `100000` | worker 测试时每个 partition 最多处理的文档数 |
| `--log-interval` | 否 | `1000` | 每处理多少个文档输出一次进度 |

## 6. Python 接口

也可以通过 `OfflinePreparationConfig` 在 Python 中配置并执行：

```python
from hyper_models.components.datasets.tools.offline_config import OfflinePreparationConfig
from hyper_models.components.datasets.tools.offline_preparation import prepare_offline_dataset

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

配置字段名称与命令行含义一致，其中：

- `dataset_subset_name` 对应 `--dataset-subset`。
- `tokenizer_name_or_path` 对应本地入口的 `--tokenizer-name-or-path` 或 Hugging Face 入口的 `--tokenizer`。
- `data_files` 可以配置字符串、字符串列表或按 split 分组的字典。
- Python 配置中的 `workers` 默认为 `8`，`keep_sequential_samples` 默认为 `True`；其他常用默认值与命令行一致。

`prepare_offline_dataset()` 只处理已经存在的本地数据。需要从 Hugging Face Hub 下载时，请使用
`huggingface_offline` 命令行入口。

## 7. 输出文件

未启用分句时，输出文件名称为：

```text
<output-prefix>_<json-key>_document.bin
<output-prefix>_<json-key>_document.idx
```

启用 `--split-sentences` 后，输出文件名称为：

```text
<output-prefix>_<json-key>_sentence.bin
<output-prefix>_<json-key>_sentence.idx
```

例如：

```bash
--output-prefix ./offline_datasets/train \
--json-keys text
```

会生成：

```text
./offline_datasets/train_text_document.bin
./offline_datasets/train_text_document.idx
```

使用多个 partition 时，还会保留各 partition 对应的 `.bin/.idx` 文件。`--keep-partition-files` 控制的是临时
JSON partition 文件，不影响最终 `.bin/.idx` 文件。

## 8. 完整示例

### 8.1 本地多字段数据

```bash
python -m hyper_models.components.datasets.tools.offline_preparation \
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

### 8.2 Hugging Face 数据集定长处理

```bash
python -m hyper_models.components.datasets.tools.huggingface_offline \
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

## 9. 常见问题

### 9.1 `workers must be divisible by partitions`

将 `workers` 调整为 `partitions` 的整数倍。例如使用 `--workers 8 --partitions 2`。

### 9.2 找不到 EOD token

默认开启 `append_eod`，因此 tokenizer 必须提供 `eos_token_id` 或 `sep_token_id`。如果数据不需要 EOD，可使用：

```bash
--append-eod false
```

### 9.3 tokenizer worker 占用内存过高

降低 `--workers`。每个 worker 都会加载一份 tokenizer。

### 9.4 Hugging Face 数据集包含自定义代码

确认数据集来源可信后，添加：

```bash
--trust-remote-code
```

### 9.5 查看命令行帮助

```bash
python -m hyper_models.components.datasets.tools.offline_preparation --help
python -m hyper_models.components.datasets.tools.huggingface_offline --help
```
