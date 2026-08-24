# Online Dataset 使用指南

Online Dataset 在训练过程中直接读取 Hugging Face Hub 或本地通用数据文件，并在样本被访问时完成
tokenize 和格式转换。它不需要预先生成 `.bin`、`.idx` 文件，适合快速验证数据、直接使用 Hugging Face
Dataset，以及无法提前完整落盘的大规模流式数据。

当前 Online Dataset 仅支持 PyTorch，并依赖 Hugging Face `datasets`。

## 1. Online 数据处理流程

```text
Hugging Face Hub / 本地文件
  -> mapping Dataset 或 iterable Dataset
  -> 读取 RawSample
  -> plaintext / conversation transform
  -> ModelSample
```

Online 数据源读取与 tokenizer、chat template 解耦：读取阶段只产生字段映射形式的 `RawSample`，transform
阶段再将其转换为只包含 `input_ids` 和 `labels` 的 `ModelSample`。

Online Dataset 提供两种读取模式：

| 模式 | Hugging Face 参数 | 特点 | 适用场景 |
|---|---|---|---|
| `mapping` | `streaming=False` | 有限长度，支持稳定的整数索引 | 有限数据集、需要随机访问 |
| `iterable` | `streaming=True` | 按需流式读取，无稳定长度和随机索引 | 大规模或无法完整落盘的数据 |

## 2. 环境准备

安装 Online Dataset 所需的可选依赖：

```bash
pip install datasets
```

使用 Hugging Face Hub 数据集时，需要保证运行环境能够访问 Hub，或者已经准备好相应缓存。私有数据集还需要
提前完成 Hugging Face 身份认证。

## 3. 使用 Hugging Face Hub 数据集

### 3.1 Mapping Dataset

下面的配置以 mapping 方式加载 WikiText，并在访问样本时 tokenize `text` 字段：

```yaml
dataset:
  model_assets:
    chat_template: null
    tokenizer:
      _target_: hyper_parallel.auto_models.components.datasets.llm.build_tokenizer.AutoTokenizer.from_pretrained
      pretrained_model_name_or_path: ./model
      use_fast: true
      trust_remote_code: true
  data_transform:
    _target_: hyper_parallel.auto_models.components.datasets.llm.build_data_transform.build_llm_data_transform
    data_type: plaintext
    text_keys: text
    max_seq_len: 2048
  _target_: hyper_parallel.auto_models.components.datasets.llm.build_online_text_dataset
  data_path: Salesforce/wikitext
  data_config:
    dataset_type: mapping
    hf_dataset_name: Salesforce/wikitext
    hf_config_name: wikitext-2-raw-v1
    namespace: train
    cache_dir: null
    show_progress: true
```

当配置了 `hf_dataset_name` 时，Online Dataset 调用 Hugging Face `load_dataset`：

```python
load_dataset(
    hf_dataset_name,
    name=hf_config_name,
    split=namespace,
    streaming=False,
    cache_dir=cache_dir,
)
```

`data_path` 是公共 Dataset 构建接口中的字段。当 `hf_dataset_name` 和 `data_path` 同时配置时，优先加载
`hf_dataset_name`，不会将 `data_path` 解析为本地路径。

### 3.2 Iterable Dataset

将 `dataset_type` 设置为 `iterable` 即可启用流式读取：

```yaml
dataset:
  model_assets:
    chat_template: null
    tokenizer:
      _target_: hyper_parallel.auto_models.components.datasets.llm.build_tokenizer.AutoTokenizer.from_pretrained
      pretrained_model_name_or_path: ./model
      use_fast: true
      trust_remote_code: true
  data_transform:
    _target_: hyper_parallel.auto_models.components.datasets.llm.build_data_transform.build_llm_data_transform
    data_type: plaintext
    text_keys: text
    max_seq_len: 2048
  _target_: hyper_parallel.auto_models.components.datasets.llm.build_online_text_dataset
  data_path: null
  data_config:
    dataset_type: iterable
    hf_dataset_name: Salesforce/wikitext
    hf_config_name: wikitext-2-raw-v1
    namespace: train
    cache_dir: null
    shuffle: true
    shuffle_buffer_size: 10000
    split_by_data_parallel: true
```

Iterable 模式调用 `load_dataset(..., streaming=True)`，不会先构建完整的本地 mapping Dataset。启用
`shuffle` 后，数据通过有限大小的 buffer 近似随机化，而不是对完整数据集进行全局随机排列。

Shuffle 随机种子来自 `training.seed`，默认值为 `42`。`shuffle_buffer_size` 必须大于零；buffer 越大，
随机程度通常越好，但会占用更多 CPU 内存。

## 4. 使用本地数据

不配置 `hf_dataset_name` 时，通过 `data_path` 加载本地文件：

```yaml
dataset:
  model_assets:
    chat_template: null
    tokenizer:
      _target_: hyper_parallel.auto_models.components.datasets.llm.build_tokenizer.AutoTokenizer.from_pretrained
      pretrained_model_name_or_path: ./model
      use_fast: true
      trust_remote_code: true
  data_transform:
    _target_: hyper_parallel.auto_models.components.datasets.llm.build_data_transform.build_llm_data_transform
    data_type: plaintext
    text_keys: text
    max_seq_len: 2048
  _target_: hyper_parallel.auto_models.components.datasets.llm.build_online_text_dataset
  data_path: ./data/train
  data_config:
    dataset_type: mapping
    hf_dataset_name: null
    namespace: train
    cache_dir: null
    show_progress: true
```

`data_path` 支持以下形式：

- 单个文件或目录路径。
- 使用逗号分隔的多个路径。
- 保留调用者顺序的路径列表。

支持的本地文件格式如下：

- JSON：`.json`、`.jsonl`
- Parquet：`.parquet`
- CSV：`.csv`
- Arrow：`.arrow`

本地数据需要满足以下约束：

- 一个 Dataset 只能使用一种文件格式，不能混合 JSON 和 Parquet 等格式。
- 目录扫描不递归，目录内文件按文件名排序。
- 显式传入路径列表时保留调用者指定的顺序。
- 路径不存在、目录中没有支持的文件或者文件格式混合时，构建阶段会直接报错。
- 当前不识别 `.json.gz`、`.jsonl.gz` 等压缩扩展名。

本地文件同样支持 `mapping` 和 `iterable`。框架根据文件扩展名选择 Hugging Face loader，并将解析后的文件列表
传给 `load_dataset`。

## 5. 数据转换

### 5.1 Plaintext

```yaml
model_assets:
  chat_template: null
  tokenizer:
    _target_: hyper_parallel.auto_models.components.datasets.llm.build_tokenizer.AutoTokenizer.from_pretrained
    pretrained_model_name_or_path: ./model
data_transform:
  _target_: hyper_parallel.auto_models.components.datasets.llm.build_data_transform.build_llm_data_transform
  data_type: plaintext
  text_keys: [text, content]
  max_seq_len: 2048
```

`text_keys` 可以是一个字段名，也可以是候选字段列表。配置列表时，框架使用记录中第一个存在的字段。

Plaintext transform 按以下步骤处理数据：

1. 从 `text_keys` 指定的字段读取文本。
2. 使用 tokenizer 的 `encode(..., add_special_tokens=False)` 生成 token ID。
3. 如果 tokenizer 定义了 `eos_token_id`，在文本末尾追加 EOS。
4. 生成 `input_ids`，并复制 `input_ids` 作为 `labels`。

当前一个源记录必须转换成恰好一个训练样本。由于多样本展开和 Online packing 尚未接入，plaintext 记录在追加
EOS 后不应超过 `max_seq_len`，否则会被 transform 分成多个片段并触发校验错误。建议在上游切分过长记录，或将
`max_seq_len` 设置为能够容纳单条记录的长度。

### 5.2 Conversation

```yaml
model_assets:
  chat_template:
    _target_: hyper_parallel.auto_models.components.datasets.llm.chat_template.ChatmlTemplate
  tokenizer:
    _target_: hyper_parallel.auto_models.components.datasets.llm.build_tokenizer.AutoTokenizer.from_pretrained
    pretrained_model_name_or_path: ./model
data_transform:
  _target_: hyper_parallel.auto_models.components.datasets.llm.build_data_transform.build_llm_data_transform
  data_type: conversation
  text_keys: messages
  max_seq_len: 2048
```

Conversation transform 从 `text_keys` 指定的字段读取消息列表，将其交给 chat template 编码，并把结果转换为
模型字段。Conversation 数据必须同时配置 tokenizer 和 chat template。

Online Dataset 会跳过 causal shift 后没有任何可训练 label 的无效样本。当前每条源记录仍必须产生一个最终可训练
样本，不支持将一条 conversation 稳定展开为多个样本。

## 6. Online 组批配置

Online Dataset 支持固定样本数组批和动态 token 组批。两种模式使用相同的
`build_online_text_collate_fn`；差别只在于 Collator 执行前如何选择样本：

| 模式 | DataLoader target | 每个 FB step 的样本数 | 适用数据源 |
|---|---|---:|---|
| Fixed Online | `FixedBatchDataLoader` | 固定 N 条 | Mapping 或 Iterable |
| Dynamic Online | `DynamicBatchDataLoader` | token budget 内动态 K 条 | 基础版本使用 Iterable |

`dataloader_type: single | cyclic` 只表示 Mapping Dataset 的 sampler 策略，不表示 Fixed/Dynamic 组批模式。
组批模式由 DataLoader 的 `_target_` 唯一确定。

### 6.1 Fixed Online

Fixed Online 每个 FB step 固定读取 `training.micro_batch_size=N` 条变长样本，然后将它们 packing 成一条
sequence：

```yaml
training:
  global_batch_size: 8
  micro_batch_size: 2

dataset:
  model_assets:
    chat_template: chatml
    tokenizer:
      _target_: hyper_parallel.auto_models.components.datasets.llm.build_tokenizer.AutoTokenizer.from_pretrained
      pretrained_model_name_or_path: /path/to/model

  data_transform:
    _target_: hyper_parallel.auto_models.components.datasets.llm.build_data_transform.build_llm_data_transform
    data_type: conversation
    text_keys: messages
    max_seq_len: 4096

  _target_: hyper_parallel.auto_models.components.datasets.llm.build_online_text_dataset
  data_path: null
  data_config:
    dataset_type: mapping
    hf_dataset_name: organization/dataset
    namespace: train

dataloader:
  # Fixed Online：固定选择 micro_batch_size=N 条样本。
  _target_: hyper_parallel.auto_models.components.datasets.FixedBatchDataLoader

  # 在固定 N 条样本被选中后执行 text packing。
  collate_fn:
    _target_: hyper_parallel.auto_models.components.datasets.build_online_text_collate_fn

  get_batch:
    _target_: hyper_parallel.auto_models.components.datasets.ParallelBatch
    source_type: online

  dataloader_type: cyclic
  data_rearrange_map: null
  data_sharding: false
  drop_last: true
  use_background_prefetcher: false
  num_workers: 0
  pin_memory: false
  prefetch_factor: null
```

对应流程为：

```text
Online Dataset
  -> 固定 N 条 ModelSample
  -> TextPackingCollator
  -> input_ids [1, packed_S] / labels [1, packed_S] / cu_seq_lens [N + 1]
```

### 6.2 Dynamic Online

Dynamic Online 连续读取单条样本，先由 `TextTokenBatcher` 根据派生的 token budget 动态选择 K 条，再调用与
Fixed Online 相同的 Collator。当前基础版本使用 Iterable Dataset，让数据源负责流式读取和 DP 分片：

```yaml
training:
  global_batch_size: 8
  micro_batch_size: 2

dataset:
  model_assets:
    chat_template: chatml
    tokenizer:
      _target_: hyper_parallel.auto_models.components.datasets.llm.build_tokenizer.AutoTokenizer.from_pretrained
      pretrained_model_name_or_path: /path/to/model

  data_transform:
    _target_: hyper_parallel.auto_models.components.datasets.llm.build_data_transform.build_llm_data_transform
    data_type: conversation
    text_keys: messages
    max_seq_len: 4096

  _target_: hyper_parallel.auto_models.components.datasets.llm.build_online_text_dataset
  data_path: null
  data_config:
    dataset_type: iterable
    hf_dataset_name: organization/dataset
    namespace: train
    shuffle: true
    shuffle_buffer_size: 10000
    split_by_data_parallel: true

dataloader:
  # Dynamic Online：TextTokenBatcher 在每个 FB step 动态选择 K 条样本。
  _target_: hyper_parallel.auto_models.components.datasets.DynamicBatchDataLoader

  # 组 batch 前至少缓存的候选样本数；不是最终 batch size。
  min_buffered_samples: 200

  # K 条样本选定后复用 Fixed Online 的 text packing。
  collate_fn:
    _target_: hyper_parallel.auto_models.components.datasets.build_online_text_collate_fn

  get_batch:
    _target_: hyper_parallel.auto_models.components.datasets.ParallelBatch
    source_type: online

  # Iterable Dataset 不创建 Mapping sampler，此字段保留统一配置结构。
  dataloader_type: single
  data_rearrange_map: null
  data_sharding: false
  drop_last: true
  use_background_prefetcher: false
  num_workers: 0
  pin_memory: false
  prefetch_factor: null
```

对应流程为：

```text
Online Iterable Dataset
  -> 连续读取单条 ModelSample
  -> TextTokenBatcher 按派生的 token budget 选择 K 条
  -> TextPackingCollator
  -> input_ids [1, packed_S] / labels [1, packed_S] / cu_seq_lens [K + 1]
```

token budget 不单独配置，由 `training.micro_batch_size * dataset.data_transform.max_seq_len` 计算。以上配置得到
`2 * 4096 = 8192`。它是 soft limit：普通样本组合后的 token 总量不超过该值；如果单条样本自身已经超过该值，
则该样本单独组成一个 FB batch。`min_buffered_samples` 是参与动态选择的最小候选样本数，不是最终 batch 的固定样本数。

Fixed Online 和 Dynamic Online 向后续 batch runtime 提供相同字段，因此后续处理不需要再次区分这两种组批方式。

## 7. 分布式处理

### 7.1 Mapping 缓存同步

Mapping 模式需要下载或构建 Hugging Face cache。分布式运行时，其处理顺序为：

```text
缓存构建 rank 加载数据并完成 cache
  -> Online Dataset barrier
  -> 其他数据拥有 rank 重新打开共享 cache
```

Online mapping 使用专用的长耗时同步屏障，避免其他 rank 在缓存尚未写完时读取不完整数据。各进程需要能够访问
`cache_dir` 指向的缓存；多节点环境应根据共享存储情况选择合适路径。

### 7.2 Iterable DP 分片

Iterable 模式可以在数据源侧按 DP rank 分片：

```yaml
data_config:
  dataset_type: iterable
  split_by_data_parallel: true
```

当 `split_by_data_parallel` 为 `true` 且 DP world size 大于 1 时，框架调用 Hugging Face
`split_dataset_by_node`：

```python
split_dataset_by_node(
    dataset,
    rank=dp_rank,
    world_size=dp_world_size,
)
```

这样每个 DP rank 从源端消费自己的数据分片。将其关闭后，各 DP rank 会分别遍历完整的上游数据流。

### 7.3 Dataset 构建归属

分布式场景下，Online Dataset 只在 TP rank 0、CP rank 0 对应的数据拥有进程上构建。Online 路径会忽略 Offline
Dataset 使用的 `data_index_cache` 语义，避免把 indexed Dataset 的索引缓存策略应用到 Hugging Face Dataset。

## 8. 参数说明

| 参数 | 适用模式 | 默认值 | 说明 |
|---|---|---:|---|
| `dataloader.get_batch.source_type` | 全部 | 无 | Online batch 必须设置为 `online` |
| `dataset_type` | 全部 | `mapping` | 可选 `mapping` 或 `iterable` |
| `hf_dataset_name` | 全部 | `null` | Hugging Face Dataset ID；配置后优先于 `data_path` |
| `hf_config_name` | Hub | `null` | Hugging Face Dataset configuration/subset 名称 |
| `namespace` | 全部 | `train` | 传给 Hugging Face `load_dataset` 的 split |
| `cache_dir` | 全部 | `null` | Hugging Face Dataset 缓存目录 |
| `show_progress` | 全部 | `true` | 是否显示 Hugging Face 进度条 |
| `shuffle` | iterable | `true` | 是否启用流式 buffer shuffle |
| `shuffle_buffer_size` | iterable | `10000` | Shuffle buffer 大小，必须大于零 |
| `split_by_data_parallel` | iterable | `true` | 是否按 DP rank 在数据源侧分片 |

Iterable shuffle 的 seed 由 `training.seed` 注入 Online Dataset；不要在 `data_config` 中单独配置 seed。

## 9. 常见问题

### 9.1 缺少 `datasets` 依赖

错误信息：

```text
Online LLM Dataset requires the optional 'datasets' package
```

安装 Hugging Face Datasets：

```bash
pip install datasets
```

### 9.2 找不到文本字段

如果出现 `Sample does not contain field`，检查 `data_transform.text_keys` 是否与原始记录字段一致。

### 9.3 一个源记录生成了多个样本

如果出现以下错误：

```text
An LLM transform must currently produce exactly one model sample per source record
```

说明单条 plaintext 记录在 tokenize 后超过 `max_seq_len`，或者自定义 transform 返回了多个样本。当前应在上游
切分记录、增大 `max_seq_len`，或者让自定义 transform 对每条记录只返回一个样本。

### 9.4 Streaming shuffle buffer 报错

`shuffle_buffer_size` 必须是正整数。Buffer 越大，随机程度通常越好，但会占用更多 CPU 内存。

### 9.5 分布式 Streaming 出现重复数据

确认 iterable 配置使用：

```yaml
split_by_data_parallel: true
```

关闭该选项后，每个 DP rank 都会消费完整数据流。

### 9.6 Hub 无法访问

网络不可用时，可以使用已经存在的 Hugging Face cache，或将数据下载为本地 JSON、Parquet、CSV、Arrow 文件，
然后清空 `hf_dataset_name` 并通过 `data_path` 加载。
