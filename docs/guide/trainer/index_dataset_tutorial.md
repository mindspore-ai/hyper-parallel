# Indexed Dataset 使用教程

本文说明如何准备并训练 indexed `.bin/.idx` 数据，覆盖两种数据类型：

- Non-packed 文档：原始文档独立存储，训练时动态组成固定长度样本。
- Packed/pre-cut 记录：离线阶段已经组成固定 `seq_length + 1` 记录，训练时直接读取。

离线转换工具的内部设计见
[`offline_preparation.md`](../../hyper_models/components/datasets/tools/offline_preparation.md)。

## 1. 选择数据模式

| 输入数据 | `.idx` 中的 sequence 长度 | Dataset 类型 | `is_dataset_from_mr` |
|---|---|---|---|
| Non-packed 文档 | 长度不一致 | `GPTDataset` | `false` |
| Packed/pre-cut 记录 | 全部为 `seq_length + 1` | `GPTFromMRDataset` | `true` |

使用限制：

- `is_dataset_from_mr: false`：输入只能是未预先 packing、未 padding 的变长文档，由 `GPTDataset` 在运行时构建
  document/sample/shuffle index 并组成训练样本。
- `is_dataset_from_mr: true`：输入必须已经离线 packing，每条记录严格包含 `seq_length + 1` 个 token；不足一条记录的
  尾部必须丢弃，数据中不能包含 PAD。tokenizer 如果定义了 PAD，其 ID 必须与 EOD/EOS 不同。

两种 Indexed 模式都使用固定样本数组批。Dataset 已经返回固定长度或预先 packing 的样本，DataLoader 只负责按
`training.micro_batch_size` 选择 N 条样本，`default_collate` 再将各个 `[S]` 字段堆叠为 `[N, S]`：

```yaml
training:
  global_batch_size: 8
  micro_batch_size: 1

dataset:
  data_config:
    # loss_mask、position_ids 和 attention_mask 由 get_batch 统一重建。
    create_ltor_fields_in_dataloader: false

dataloader:
  # Fixed Indexed：每个 FB step 固定选择 micro_batch_size=N 条样本。
  _target_: hyper_models.components.datasets.FixedBatchDataLoader

  # Indexed Dataset 已经完成 fixed-length/pre-packed 处理，这里只做字段堆叠。
  collate_fn:
    _target_: hyper_models.components.datasets.build_indexed_collate_fn

  # single/cyclic 只控制 Mapping Dataset 的采样顺序，不表示动态组批。
  dataloader_type: single
  data_rearrange_map: null
  data_sharding: false
  drop_last: true
  use_background_prefetcher: false
  num_workers: 0
  pin_memory: false
  prefetch_factor: null
```

完整的数据流为：

```text
GPTDataset / GPTFromMRDataset
  -> N 个固定长度样本（tokens、labels）
  -> build_indexed_collate_fn（PyTorch default_collate）
  -> tokens [N, S] / labels [N, S]
  -> get_batch 将 tokens 规范为 input_ids，并构建运行时字段
```

`create_ltor_fields_in_dataloader` 默认为 `false`。只有兼容不使用统一 `get_batch` 的旧流程时才显式设为 `true`；
此时 Dataset 会额外生成并返回 `loss_mask`、`position_ids` 和 `attention_mask`。

Indexed 样本不经过 Online 使用的 text transform。其长度和 packing 已由 `GPTDataset` 或离线数据阶段确定，
Indexed Dataset 直接返回 `tokens` 和 `labels`；字段规范化统一在 `get_batch` 完成。

这里的 Fixed 表示每个 FB step 的样本数 N 固定。它不会在 Collator 中再次 packing：离线 packing 由数据转换工具
完成，普通 `GPTDataset` 的样本组成则由 sample index 完成。

## 2. `.bin/.idx` 保存什么

一个 Dataset prefix 对应两个文件：

```text
<prefix>.bin  # 连续 token payload
<prefix>.idx  # dtype、sequence length、字节偏移和 document 边界
```

读取第 `i` 条 sequence 时：

```text
.idx.sequence_pointers[i] → 定位 .bin 字节偏移
.idx.sequence_lengths[i]  → 读取 token 数量
.bin                      → 返回 token ID 数组
```

EOD 与普通 token 一样存储在 `.bin` 中；`.idx` 不保存 EOD ID 或 EOD 位置。EOD ID 来自制作数据时使用的
tokenizer，因此训练必须加载同一个 tokenizer，或准确提供相同的 tokenizer 元数据。

## 3. Tokenizer 配置

### 3.1 有真实 Hugging Face tokenizer

目录包含 `tokenizer.json` 等文件时使用：

```yaml
tokenizer:
  _target_: hyper_models.components.datasets.llm.build_tokenizer.AutoTokenizer.from_pretrained
  pretrained_model_name_or_path: /path/to/tokenizer
  tokenizer_type: hf
  use_fast: true
  local_files_only: true
```

`vocab_size` 和 `eod` 自动取自 tokenizer。`use_fast: true` 使用 Rust `tokenizers` 后端；离线转换和训练应使用
同一 tokenizer 路径与同一后端。

### 3.2 只有 token 数据和已知元数据

缺少真实词表、不能解码文本时使用：

```yaml
tokenizer:
  _target_: hyper_models.components.datasets.llm.build_tokenizer.AutoTokenizer.from_pretrained
  pretrained_model_name_or_path: /path/to/stable/tokenizer-identity
  tokenizer_type: pretokenized
  vocab_size: 32000
  eod_token_id: 2
```

此模式返回的 `_DatasetTokenizer` 只是 metadata-only 占位类，不会加载 tokenizer 文件，也不提供文本编码、解码或
完整的特殊 token 行为。`pretrained_model_name_or_path` 只参与 Dataset cache identity；`vocab_size` 和
`eod_token_id` 必须与离线转换时的约定一致，不能根据经验随意填写。

该占位类只适合读取已经完成 tokenization 的数据和调试 Dataset 构建流程。正式接入一种预分词数据时，应提供与
数据制作阶段对应的 tokenizer 类，并在配置中通过 `_target_` 指向它，例如：

```yaml
tokenizer:
  _target_: package.tokenizers.PretokenizedTokenizer
  tokenizer_path: /path/to/tokenizer
```

对应类至少需要提供正确的 `vocab_size`、`eod`/`eos_token_id` 和稳定的 cache identity；`pad_token_id` 可选，但
如果提供，必须与 EOD/EOS 不同。如果流程包含原始文本转换或结果解码，还必须实现相同的 encode/decode 规则。

## 4. Non-packed 文档数据示例

### 4.1 准备数据

先确保本地 tokenizer 文件存在：

```text
/path/to/tokenizer/tokenizer.json
```

以 Hugging Face 文本数据集为例，转换命令为：

```bash
python -m hyper_models.components.datasets.tools.huggingface_offline \
  --dataset organization/dataset \
  --dataset-subset subset_name \
  --dataset-split train \
  --download-dir /path/to/raw_data \
  --json-keys text \
  --output-prefix /path/to/indexed/train \
  --tokenizer /path/to/tokenizer \
  --workers 8 \
  --append-eod true
```

输出 prefix 为：

```text
/path/to/indexed/train_text_document
```

该命令包含三个关键约定：

- 未传 `--pack-to-seq-len`：不在离线阶段 packing 或补齐定长记录，每个非空原始文档独立保存，因此 sequence
  长度可以不同。
- 传入 `--pack-to-seq-len N`：连续 packing 为长度 `N + 1` 的完整记录，输入结束时不足一条记录的尾部直接丢弃，
  不写入 PAD。
- 传入 `--append-eod true`（默认值）：在每个非空原始文档末尾追加 tokenizer 定义的 EOD/EOS token，用于在连续
  token 流中保留文档边界。

超过 tokenizer `model_max_length` 的转换 warning 不会截断数据；训练侧稍后按 `seq_length` 重新组成样本。

### 4.2 训练配置

核心配置为：

```yaml
dataset:
  model_assets:
    tokenizer:
      _target_: hyper_models.components.datasets.llm.build_tokenizer.AutoTokenizer.from_pretrained
      pretrained_model_name_or_path: /path/to/tokenizer
      tokenizer_type: hf
      use_fast: true
      local_files_only: true

  _target_: hyper_models.components.datasets.llm.build_indexed_text_dataset
  data_path: /path/to/indexed/train_text_document
  data_config:
    seq_length: 2048
    split: "1, 0, 0"
    is_dataset_from_mr: false

dataloader:
  get_batch:
    _target_: hyper_models.components.datasets.ParallelBatch
    source_type: indexed
```

Non-packed 数据必须使用 `is_dataset_from_mr: false`，由 `GPTDataset` 根据 sample index 动态组成定长训练样本；如果
错误配置为 `true`，变长文档会在固定长度校验或 Collator 组 batch 时失败。

`GPTDataset` 会生成或加载以下 cache：

```text
cache/GPTDataset_indices/
├── *-document_index.npy
├── *-sample_index.npy
└── *-shuffle_index.npy
```

`document_index` 保存训练使用的文档顺序，`sample_index` 保存每个 `seq_length + 1` 样本在文档流中的起止文档和
offset，`shuffle_index` 保存样本的最终读取顺序。读取第 `i` 条训练样本的完整过程为：

```text
i
→ shuffle_index[i]
→ sample_index 找到起止文档和 offset
→ document_index 找到真实文档编号
→ .idx 找到各文档的长度和 .bin 字节偏移
→ .bin 读取并拼接 seq_length + 1 个 token
→ text[:-1] 作为 input_ids
→ text[1:] 作为 labels
```

一条样本可以跨越多个原始文档，最终生成长度为 `seq_length` 的 `input_ids` 和 `labels`。

## 5. Packed/pre-cut 定长数据示例

### 5.1 数据结构

数据目录可以包含多个 prefix，例如：

```text
shards/source_000001.bin
shards/source_000001.idx
```

每个 prefix 的结构为：

```text
sequences=<record count>
documents=<document boundary count>
min_length=max_length=2049
dtype=int32
```

每条记录已经在离线阶段组成固定长度：

```text
2049 token
├── input_ids = text[:-1]  # 2048
└── labels    = text[1:]   # 2048
```

原始文档边界由 `.bin` 内部的 EOD token 表达。一条记录可以包含多个 EOD，也可能只是长文档中间片段而没有 EOD。

### 5.2 读取过程

`GPTFromMRDataset` 不生成 document/sample/shuffle cache，而是把每条 indexed sequence 直接作为一条训练记录：

```text
i
→ GPTFromMRDataset[i]
→ .idx.sequence_pointers[i] 找到 .bin 字节偏移
→ .idx.sequence_lengths[i] 得到 seq_length + 1
→ .bin 直接读取完整的 seq_length + 1 个 token
→ text[:-1] 作为 input_ids
→ text[1:] 作为 labels
→ 根据配置生成 attention_mask、loss_mask 和 position_ids
```

因此预切记录之间不会再次拼接。Dataset 本身不维护 shuffle cache；样本顺序由外层 BatchSampler 或多数据源 blend
决定。若使用多数据源 blend，外层会先把逻辑样本编号映射为数据源编号和该数据源内的记录编号，再执行上述读取过程。

### 5.3 Padding 处理

`GPTFromMRDataset` 不支持带 PAD 的预切记录。离线转换只写入完整的 `seq_length + 1` packed block，并丢弃最终
不足一条记录的尾部；读取时如果发现配置的 PAD 或内部 `-1` 哨兵会直接报错。

普通 `GPTDataset` 的输入同样不能离线 padding。它在运行时组成样本；训练和测试默认丢弃不完整尾部，验证集如果配置
保留残缺样本，则只在内存中使用 `_PAD_TOKEN_ID = -1` 补齐、生成 loss mask，并在送入模型前替换为合法 token ID。

### 5.4 训练配置

核心字段为：

```yaml
dataset:
  model_assets:
    tokenizer:
      _target_: hyper_models.components.datasets.llm.build_tokenizer.AutoTokenizer.from_pretrained
      pretrained_model_name_or_path: /path/to/tokenizer-identity
      tokenizer_type: pretokenized
      vocab_size: 32000
      eod_token_id: 2

  _target_: hyper_models.components.datasets.llm.build_indexed_text_dataset
  data_path: /path/to/shards
  data_config:
    seq_length: 2048
    split: "1, 0, 0"
    is_dataset_from_mr: true
    skip_data_check: false

dataloader:
  get_batch:
    _target_: hyper_models.components.datasets.ParallelBatch
    source_type: indexed
```

`is_dataset_from_mr: true` 在当前实现中表示使用 `GPTFromMRDataset` 直接读取固定的 `seq_length + 1` 记录。
它不会生成 document/sample/shuffle 三类动态 packing cache，也不会跨两个预切记录再次拼接。

Packed/pre-cut 数据必须使用 `is_dataset_from_mr: true`；如果错误配置为 `false`，`GPTDataset` 会再次建立 sample
index，并可能跨两个预切记录重新拼接。

模型词表必须覆盖数据中的所有 token ID，最大位置长度必须不小于 `seq_length`。仅替换 Dataset 配置而继续使用
不匹配的小词表模型会导致 embedding 越界。

## 6. EOD、attention 和 loss

EOD ID 必须与制作数据时的 tokenizer 一致。三个选项决定是否利用 packed 记录中的 EOD 边界：

```yaml
data_config:
  reset_position_ids: false
  reset_attention_mask: false
  eod_mask_loss: false
```

- `reset_position_ids`：EOD 后重新从位置 0 开始。
- `reset_attention_mask`：EOD 后的 token 不再关注前一个文档。
- `eod_mask_loss`：屏蔽配置定义的 EOD 边界 loss。

全部为 `false` 时，模型把 packed 块视为连续 token 流；读取和 shape 仍然正确，但不同原始文档之间不隔离。
