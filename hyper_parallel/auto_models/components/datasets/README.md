# Dataset 设计

## 1. 目标与边界

`hyper_parallel.auto_models.components.datasets` 负责把原始数据或 Indexed Dataset 转换为模型一次
forward/backward 所需的 batch。该模块的边界是：

```text
原始数据 / .idx + .bin
          ↓
Dataset 构建、分割与混合
          ↓
RawSample → ModelSample
          ↓
Sampler / DataLoader / Collator
          ↓
CP 切分 + TP 广播 + attention 元数据
          ↓
(model_inputs, loss_inputs)
```

本模块不负责模型构建、优化器、训练调度或 checkpoint 存储。LLM 的 Online 和 Indexed
路径已实现；Omni 目前只保留接口，还没有可用的 Online/Offline Dataset 实现。

建议按以下顺序阅读：

```text
整体流水线与数据契约（第 2～3 节）
    → Online / Indexed 两条数据路径（第 4～6 节）
    → packing、DataLoader 和分布式 batch（第 7～10 节）
    → 配置示例、目录和调试参考（第 11～14 节）
```

## 2. 总体构建流程

`TextTrainer` 按固定顺序构建数据流水线：

```text
TextTrainer
    │
    ├─ _build_model_assets()
    │    ├─ tokenizer
    │    └─ chat_template
    │
    ├─ _build_data_transform()
    │    ├─ PlaintextTransform
    │    ├─ TextConversationTransform
    │    └─ IdentityDataTransform
    │
    ├─ BaseTrainer._build_dataset()
    │    ├─ build_online_text_dataset
    │    └─ build_indexed_text_dataset
    │
    ├─ _build_collate_fn()
    │    ├─ build_online_text_collate_fn
    │    └─ build_indexed_collate_fn
    │
    ├─ BaseTrainer._build_dataloader()
    │    ├─ FixedBatchDataLoader
    │    └─ DynamicBatchDataLoader
    │
    └─ _build_get_batch()
         └─ ParallelBatch
```

`dataset._target_` 可返回单个 Dataset，也可返回 `(train, valid, test)`。`BaseTrainer` 统一保存三个
split，再为每个非空 split 构建 DataLoader。

## 3. 数据契约

### 3.1 RawSample

RawSample 是数据源刚读出的记录，字段由数据源决定，例如：

```python
{"text": "A short document."}
```

```python
{"conversation": [{"role": "user", "content": "Hello"}]}
```

Online Dataset 输出 RawSample，然后在 `transform_dataset.py` 中延迟执行 tokenizer 或 chat template。
Indexed Dataset 已经保存 token ID，不经过 Online transform。

### 3.2 ModelSample

Online 路径的 ModelSample 使用：

```text
input_ids: [sample_seq_len]
labels:    [sample_seq_len]
```

Indexed 路径的 ModelSample 使用：

```text
tokens: [seq_length]
labels: [seq_length]
```

`PlaintextTransform` 和 `TextConversationTransform` 可返回单个 ModelSample 或有序的 ModelSample 列表。
固定 batch 路径要求一条 RawSample 最终只产生一条 ModelSample；一对多结果需要使用动态 batch 路径。

### 3.3 Collated batch

Online collator 把 K 条变长 ModelSample 拼成：

```text
input_ids:  [1, packed_length]
labels:     [1, packed_length]
cu_seq_lens: [0, len(sample_0), len(sample_0)+len(sample_1), ...]
```

Indexed collator 使用 PyTorch `default_collate` 堆叠固定长度样本：

```text
tokens: [micro_batch_size, seq_length]
labels: [micro_batch_size, seq_length]
```

### 3.4 ParallelBatch 输出

`ParallelBatch` 先把 Online/Indexed 字段统一为 `input_ids` 和 `labels`，然后输出：

```text
model_inputs = {
    input_ids,
    labels,
    position_ids,
    attention_mask,
    swa_mask,
    packed_seq_params,
}

loss_inputs = {
    labels,
    loss_mask,
}
```

## 4. LLM Online Dataset

Online 路径在训练进程内读取原始数据并执行 transform：

```text
本地 JSON/JSONL/Parquet/CSV/Arrow 或 Hugging Face Dataset
                         ↓
              mapping / iterable source
                         ↓
                  RawSample
                         ↓
       PlaintextTransform / TextConversationTransform
                         ↓
             变长 input_ids + labels
                         ↓
          FixedBatchDataLoader / DynamicBatchDataLoader
                         ↓
              TextPackingCollator
                         ↓
       packed tokens + labels + cu_seq_lens
```

### 4.1 Mapping 与 Iterable

`online` 表示训练读取数据时才执行 tokenizer 和 label 构造；`mapping/iterable` 表示原始记录的
访问方式。Online 不等于实时或无限数据流。

| 类型 | Hugging Face 参数 | 访问方式 | 排序与 DP 分片 |
| --- | --- | --- | --- |
| `mapping` | `streaming=False` | `dataset[index]` | BatchSampler 分配下标 |
| `iterable` | **`streaming=True`** | `for sample in dataset` | 数据源 shuffle 后按 DP rank 切分 |

Online Mapping 是有限、可按下标访问的数据集，读取时再执行 transform：

```text
dataset[100]
    → Hugging Face Dataset[100]
    → RawSample
    → transform/tokenizer
    → ModelSample(input_ids, labels)
```

这里的 Mapping-style 表示 Dataset 支持 `dataset[index]`；单条 RawSample 通常也是一个
`Mapping[str, Any]`，例如 `{"text": "hello"}`，两者含义不同。

Online Iterable 的关键是 Hugging Face `load_dataset(..., streaming=True)`。开启后返回
`IterableDataset`，数据在迭代时按需读取，而不是构造成支持随机下标访问的 Dataset，因此通常不支持
`dataset[100]` 或 `len(dataset)`：

```text
for RawSample in Hugging Face IterableDataset
    → transform/tokenizer
    → yield ModelSample
```

Iterable 描述的是访问方式，不代表数据一定无限。由于它没有可随机排列的全局下标，shuffle 使用
有限 buffer，DP 分片和迭代状态恢复也由数据源负责。

简单来说：Mapping 是“给我第 100 条”，Iterable 是“给我下一条”。

### 4.2 固定样本数与动态 token batch

Mapping/Iterable 与 Fixed/Dynamic 是两组独立概念：前者决定如何读取源数据，后者决定如何选择一个
batch 的 ModelSample。四种组合都支持：

| 组合 | 行为 |
| --- | --- |
| Mapping + Fixed | 按下标选择固定 K 条 |
| Mapping + Dynamic | 按下标读取候选样本，再按 token budget 选择可变 K 条 |
| Iterable + Fixed | 从数据流顺序取得固定 K 条 |
| Iterable + Dynamic | 从数据流填充 buffer，再按 token budget 选择可变 K 条 |

`FixedBatchDataLoader` 每次选取固定数量的 ModelSample，再由 Online collator 把它们拼接。

`DynamicBatchDataLoader` 使用：

```text
token_budget = micro_batch_size * data_transform.max_seq_len
```

`TextTokenBatcher` 先缓存候选样本，每次尽量选取总 token 数不超过 budget 的 K 条样本。单条超长
样本会独立成 batch。动态 DataLoader 的 checkpoint 同时保存源 DataLoader 游标与未消费 token buffer。

### 4.3 Online 对齐填充

Online collator 会把 packed length 对齐到统一序列分片大小：

```text
alignment = cp_size * tp_size    # 开启 TP sequence parallel
alignment = cp_size              # 未开启 TP sequence parallel
```

`input_ids` 尾部填 0，`labels` 尾部填 `IGNORE_INDEX`。这段对齐 token 作为一条合成 sequence
记入 `cu_seq_lens`，但不参与 loss。

## 5. LLM Indexed Dataset

Indexed 路径使用共同前缀的两个文件：

```text
<prefix>.idx  # dtype、sequence_lengths、sequence_pointers、document_indices
<prefix>.bin  # 连续 token payload
```

`sequence_pointers[i]` 是第 i 条 sequence 在 `.bin` 中的字节偏移，不是 JSON 或 `{text: ...}`。
`sequence_lengths[i]` 是 token 数。

```text
sequence_id = 1
    ↓
.idx -> pointer=4098, length=2049, dtype=uint16
    ↓
.bin -> frombuffer(offset=4098, count=2049)
    ↓
NumPy token ID sequence
```

一条 low-level sequence 始终从同一个 `<prefix>.bin` 读取。多数据源 blend 只选择某个源中的样本，
不会把两个 prefix 的 token 拼成同一条 mid-level sample。

### 5.1 离线预处理的两种输出

是否传入 `--pack-to-seq-len` 决定 indexed sequence 的语义，也决定运行期 Dataset 类型：

```text
原始 JSONL
    │
    ├─ 不传 --pack-to-seq-len
    │      ↓
    │   每个原始文档作为一条变长 indexed sequence
    │      ↓
    │   GPTDataset
    │      ↓
    │   训练初始化时动态跨文档 packing
    │
    └─ --pack-to-seq-len N
           ↓
        离线连续拼接 token
           ↓
        每 N + 1 个 token 写成一条 indexed sequence
           ↓
        GPTFromMRDataset
           ↓
        直接读取，不再动态 packing
```

`--append-eod true` 在每个非空原始文档末尾追加 tokenizer 的 EOS/EOD token。离线 packing 后，
这个 token 是恢复原始文档边界的依据。

当前 `--partitions P` 为每个 preprocessing partition 建立独立 token buffer。每个 partition 只写入完整
`N + 1` 块，末尾不足一块的 token 会丢弃；后续 merge 只合并已写入的 sequence，不会跨
partition 继续 packing。`--keep-sequential-samples` 只决定原始记录如何分配给 partition，不会传递残余
token buffer。

### 5.2 Low-level：IndexedDataReader

Low-level 只解析 `.idx` 并读取 `.bin`，不生成训练字段：

```text
IndexedDataReader[i]
    → sequence_pointers[i]
    → sequence_lengths[i]
    → .bin 连续 token ID
```

`reader.get(sequence_id, offset, length)` 可以只读一条 sequence 内的指定 token 区间。mmap 模式还支持
连续 sequence slice；非 mmap 模式不支持 slice。

### 5.3 Mid-level：GPTDataset

`GPTDataset` 面向变长文档 sequence，在构建 split 时生成三组 cache：

| 索引 | 格式 | 作用 |
| --- | --- | --- |
| `document_index` | `[document_id, ...]` | 定义经过 epoch 重复和 shuffle 后的文档顺序 |
| `sample_index` | `[(document_position, offset), ...]` | 定义每条固定长度样本的起点；相邻两项定义读取范围 |
| `shuffle_index` | `[sample_id, ...]` | 把逻辑 Dataset 索引映射到 sample |

具体读取：

```text
GPTDataset[i]
    → shuffled_sample_id = shuffle_index[i]
    → sample_index[shuffled_sample_id : shuffled_sample_id + 2]
    → 得到起止 document_position 和 offset
    → document_index[document_position] 得到真实 sequence_id
    → IndexedDataReader.get() 读取一个或多个文档片段
    → 拼成 seq_length + add_extra_token_to_sequence 个 token
    → text[:-1] 作为 tokens，text[1:] 作为 labels
```

`sample_index` 保存的第一列是 `document_index` 中的位置，不是真实文档 ID，也不是文档区间的
token 求和。例如：

```text
document_index = [7, 2, 10, 4]
sample_index   = [(0, 100), (2, 50)]

起点 (0, 100) -> document_index[0] = sequence 7，从 offset 100 开始
终点 (2, 50)  -> document_index[2] = sequence 10，到 offset 50 结束

实际读取：sequence 7 的尾部 + sequence 2 的全部 + sequence 10 的头部
```

这里的 `7 → 2 → 10` 是文档访问顺序，不是 `7 + 2 + 10`。

### 5.4 Mid-level：GPTFromMRDataset

`GPTFromMRDataset` 面向已经预切的固定长度 sequence，不构建 document/sample/shuffle cache：

```text
GPTFromMRDataset[i]
    → sequence_id = split_indices[i]
    → IndexedDataReader[sequence_id]
    → 直接读取完整的 seq_length + add_extra_token_to_sequence 个 token
    → text[:-1] 作为 tokens，text[1:] 作为 labels
```

该路径的约束是：

- `is_dataset_from_mr: true`。
- 每条 indexed sequence 长度必须等于 `seq_length + add_extra_token_to_sequence`。
- 当前预切记录不允许包含 PAD token。
- split 单位是已经 packing 的 record，不是原始文档。
- 样本顺序由 split、blend 和外层 BatchSampler 决定。

### 5.5 GPTDataset 与 GPTFromMRDataset 选择

| 数据格式 | `is_dataset_from_mr` | Dataset | 是否构建 document/sample/shuffle cache |
| --- | --- | --- | --- |
| 变长原始文档 sequence | `false` | `GPTDataset` | 是 |
| 固定 `seq_length + extra` 的预切 record | `true` | `GPTFromMRDataset` | 否 |
| Mock 数据 | `mock_data: true` | `MockGPTDataset` | 否 |

不能把离线 packing 数据交给 `GPTDataset` 再次组样，也不能把变长文档交给
`GPTFromMRDataset` 直读。

## 6. Indexed split 与多数据源 blend

Indexed Dataset 分为三层：

```text
Low-level
IndexedDataReader(prefix)
    → sequence token IDs
          │
          ▼
Mid-level
GPTDataset / GPTFromMRDataset
    → tokens + labels
          │
          ▼
Blend-level
BlendedDataset / SimpleBlendedDataset
    → 选择 source Dataset 和 source-local sample
```

### 6.1 Split

`split: "98, 1, 1"` 先归一化为 train/valid/test 比例，再按 low-level element 数量计算连续范围：

```text
num_elements = 1000
split_matrix = [(0.00, 0.98), (0.98, 0.99), (0.99, 1.00)]

train_indices = [0, ..., 979]
valid_indices = [980, ..., 989]
test_indices  = [990, ..., 999]
```

`GPTDataset` 的 element 是变长文档 sequence；`GPTFromMRDataset` 的 element 是预切 record。
`train_data_path`/`valid_data_path`/`test_data_path` 可以让三个 split 使用完全独立的数据源，此时不再使用共享
`split`。

### 6.2 单数据源

```text
prefix
  → IndexedDataReader
  → (train_indices, valid_indices, test_indices)
  → (train_mid_dataset, valid_mid_dataset, test_mid_dataset)
```

`data_lazy_load: false` 会在构建阶段打开 low-level reader 并立即构建 mid-level Dataset。
`data_lazy_load: true` 则先创建 `LazyDatasetProxy`，首次调用 `len()`、`__getitem__()` 或访问
`unique_identifiers` 时才构建真实 Dataset，并在 proxy 内复用。

### 6.3 多数据源

假设：

```text
prefixes = [books, code]
train mid-level Datasets = [books_train, code_train]
```

Standard blend 使用两列索引：

```text
dataset_index        = [0, 1, 0, 0]
dataset_sample_index = [0, 0, 1, 2]

BlendedDataset[0] -> datasets[0][0] -> books_train[0]
BlendedDataset[1] -> datasets[1][0] -> code_train[0]
BlendedDataset[2] -> datasets[0][1] -> books_train[1]
BlendedDataset[3] -> datasets[0][2] -> books_train[2]
```

`books_train` 和 `code_train` 是两个不同 prefix 对应的 train mid-level Dataset，不是文件名或 token 内容。

| `simple_blend` | 顶层 Dataset | 映射格式 | 顺序 |
| --- | --- | --- | --- |
| `no` | `BlendedDataset` | `dataset_index` + `dataset_sample_index` | 按归一化权重生成确定性调度 |
| `inter` | `SimpleBlendedDataset` | `_locations = [(dataset_id, sample_id), ...]` | 各数据源轮流取一条 |
| `intra` | `SimpleBlendedDataset` | `_locations = [(dataset_id, sample_id), ...]` | 先取完一个数据源，再取下一个 |

`inter` 和 `intra` 只支持 `is_dataset_from_mr: true`。三种 blend 最终都返回被选 source 的
`tokens`、`labels` 和可选 LTR 字段，并额外标记 `dataset_id`。

## 7. 不同阶段的 packing 与组样语义

| 名称 | 样本选择 | packing 输出 | 边界规则 |
| --- | --- | --- | --- |
| 离线 packing | `offline_preparation.py` 连续读取原始文档 token | `N + 1` 定长 indexed record | 可跨原始文档，当前不跨 preprocessing partition |
| GPT 动态组样 | `GPTDataset` 从变长 indexed document 组成固定长度样本 | `seq_length + extra` 的逻辑样本 | 可跨 `document_index` 中的文档，不跨 data source/split |
| Online fixed batch packing | `FixedBatchDataLoader` 每批选择固定 K 条 ModelSample | 一条 packed micro-batch + `cu_seq_lens` | 保留每条 ModelSample 的 attention 边界 |
| Online dynamic batch packing | `DynamicBatchDataLoader` 按 token budget 选择可变 K 条 ModelSample | 一条 packed micro-batch + `cu_seq_lens` | 保留每条 ModelSample 的 attention 边界 |

两种 Online 模式只在样本选择策略上不同；选出的 ModelSample 最终都由 `TextPackingCollator` 拼接。
Fixed 模式固定样本数，packed token 数可变；Dynamic 模式限制 token budget，样本数可变。

“离线预处理”不必然表示“离线 packing”。不传 `--pack-to-seq-len` 时，tokenize 和 `.idx/.bin`
写入仍然是离线执行，只是固定长度组样延迟到 `GPTDataset` 构建阶段。

## 8. 统一 DataLoader 采样与恢复

Mapping Dataset 使用 `build_dataset_batch_sampler`：

- `single`：从 `consumed_samples` 后顺序消费一个 epoch。
- `cyclic`：按 seed 和 epoch 生成可重现顺序，可选 data sharding。
- `data_rearrange_map`：在返回 Dataset index 前做额外逻辑到物理索引映射。
- `drop_last`：控制是否丢弃不完整的分布式 batch。

Iterable Dataset 不构建 BatchSampler，由上游 stream 管理顺序和游标。

DataLoader checkpoint 的目标是同时恢复：

```text
source cursor
+ BatchSampler consumed_samples / epoch
+ DynamicBatchDataLoader 未消费 buffer
```

## 9. 分布式读取与 batch 分发

### 9.1 Dataset/DataLoader 所有权

`DataLoaderParallelContext` 从 Trainer mesh 提取 DP/TP/CP 信息。每个 CP coordinate 上的 TP rank 0 是
DataLoader 数据源 rank：

```text
(dp_rank, cp_rank, tp_rank=0) -> 构建/迭代 Dataset 和 DataLoader
(dp_rank, cp_rank, tp_rank>0) -> 不直接读取 source batch
```

共享存储场景下，global rank 0 先生成 Indexed/Hugging Face cache，其他 DataLoader rank 在 barrier 后重新打开。
`data_index_cache` 可使需要消费现有索引 cache 的 rank 也构建 Dataset 对象。

### 9.2 一个 batch 的分发顺序

```text
TP rank 0 读取完整 collated batch
              ↓
OnlineBoundaryResolver / IndexedBoundaryResolver
生成全局 cu_seq_lens
              ↓
CPBatchSharder 按 sequence 维连续切分 input_ids/labels
              ↓
TPBatchBroadcaster 在 TP group 内广播 CP-local tensor
同时广播全局 cu_seq_lens
              ↓
每个 TP rank 本地构建 position_ids/loss_mask/attention 元数据
```

Online 直接使用 collator 输出的 `cu_seq_lens`。Indexed 从 EOD token 和每个 DataLoader row 的结尾恢复
sequence 边界。CP 只切分 token tensor，`cu_seq_lens` 保持全局坐标，供 position ID、dense mask 或
compressed attention adapter 使用。

## 10. Attention 与 LTR 字段

`ParallelBatch` 统一构建：

- `position_ids`：默认为全局位置在当前 CP slice 上的区间；`reset_position_ids` 可按 sequence 边界归零。
- `loss_mask`：`labels >= 0` 的位置参与 loss；`eod_mask_loss` 可额外屏蔽 EOD。
- dense attention：根据 `cu_seq_lens`、causal 语义和 sliding window 生成 mask。
- compressed attention：通过 `AttentionRuntimeAdapter` 生成 `packed_seq_params`。

Indexed Dataset 仍保留 `create_ltor_fields_in_dataloader` 兼容配置，但统一训练路径由
`ParallelBatch` 在 CP/TP 分发后构建实际使用的 LTR 字段。

## 11. 配置组合

### 11.1 Online Mapping + Dynamic batch

```yaml
dataset:
  _target_: hyper_parallel.auto_models.components.datasets.llm.build_online_text_dataset
  data_path: /data/train.jsonl
  data_transform:
    _target_: hyper_parallel.auto_models.components.datasets.llm.build_data_transform.build_llm_data_transform
    data_type: plaintext
    text_keys: text
    max_seq_len: 4096
  data_config:
    dataset_type: mapping

dataloader:
  _target_: hyper_parallel.auto_models.components.datasets.DynamicBatchDataLoader
  min_buffered_samples: 200
  collate_fn:
    _target_: hyper_parallel.auto_models.components.datasets.build_online_text_collate_fn
  get_batch:
    _target_: hyper_parallel.auto_models.components.datasets.ParallelBatch
    source_type: online
```

将 `dataset_type` 改为 `iterable` 可切换到 `streaming=True` 的数据源；将 DataLoader target 改为
`FixedBatchDataLoader` 可切换为固定 K 条样本的 batch。两项配置相互独立。

### 11.2 Indexed 变长文档

```yaml
dataset:
  _target_: hyper_parallel.auto_models.components.datasets.llm.build_indexed_text_dataset
  data_path: /data/corpus_text_document
  data_config:
    seq_length: 4096
    split: "98, 1, 1"
    mock_data: false
    is_dataset_from_mr: false
    simple_blend: "no"
    data_lazy_load: true
    create_ltor_fields_in_dataloader: false

dataloader:
  _target_: hyper_parallel.auto_models.components.datasets.FixedBatchDataLoader
  collate_fn:
    _target_: hyper_parallel.auto_models.components.datasets.build_indexed_collate_fn
  get_batch:
    _target_: hyper_parallel.auto_models.components.datasets.ParallelBatch
    source_type: indexed
```

### 11.3 Indexed 离线预切记录

先预处理：

```bash
python -m hyper_parallel.auto_models.components.datasets.tools.offline_preparation \
    --dataset-name-or-path /data/train.jsonl \
    --output-prefix /data/train \
    --json-keys text \
    --tokenizer-name-or-path /models/tokenizer \
    --append-eod true \
    --pack-to-seq-len 4096
```

再配置：

```yaml
dataset:
  _target_: hyper_parallel.auto_models.components.datasets.llm.build_indexed_text_dataset
  data_path: /data/train_text_document
  data_config:
    seq_length: 4096
    split: "98, 1, 1"
    mock_data: false
    is_dataset_from_mr: true
    simple_blend: "no"
    skip_data_check: false
```

## 12. 目录与职责

```text
datasets/
├── __init__.py                         # 公共 DataLoader/Collator/ParallelBatch 导出
├── dataset_logging.py                  # Dataset 专用日志
├── batching/
│   ├── build_collate_fn.py             # Indexed stack 与 Online packing
│   ├── build_dataloader.py             # Fixed/Dynamic DataLoader 与 token batcher
│   ├── get_batch.py                    # DataLoader batch 到模型/loss 输入
│   ├── sequence_boundaries.py          # Online/Indexed sequence 边界
│   └── attention_runtime.py            # dense mask 与 compressed adapter 接口
├── parallel/
│   ├── dataloader_parallel.py          # Dataset/DataLoader rank 所有权与 cache barrier
│   ├── batch_sampler.py                # DP mapping-style sampler
│   └── batch_parallel.py               # CP 切分与 TP 广播
├── llm/
│   ├── build_tokenizer.py              # tokenizer adapter
│   ├── chat_template.py                # conversation template
│   ├── build_data_transform.py         # plaintext/conversation transform
│   ├── transform_dataset.py            # 延迟 transform wrapper
│   ├── build_dataset.py                # Online/Indexed 公共构建入口
│   ├── online_dataset.py               # mapping/iterable 选择
│   ├── online_mapping_dataset.py       # 有限随机访问源
│   ├── online_iterable_dataset.py      # streaming 源
│   ├── online_utils.py                 # 文件解析和 Hugging Face 加载
│   ├── indexed_data_config.py          # 路径发现与 GPT config
│   ├── indexed_dataset.py              # Indexed provider 和 Dataset 类型选择
│   ├── indexed_split_builder.py        # low/mid/blend 与 train/valid/test
│   ├── indexed_data_reader.py          # .idx/.bin 读取
│   ├── indexed_pretrain_dataset.py     # GPT/Mock/MR mid-level Dataset
│   ├── indexed_blended_dataset.py      # 标准权重 blend
│   ├── indexed_simple_blended_dataset.py # MR inter/intra blend
│   ├── indexed_lazy_dataset.py         # 延迟构建 proxy
│   ├── indexed_helpers.py              # C++ sample/blend index 桥接
│   └── csrc/indexed_helpers.cpp        # sample/blend index 实现
├── omni/                               # Omni 占位接口，尚未实现
└── tools/
    ├── offline_preparation.py          # 本地 JSON/JSONL 到 Indexed Dataset
    ├── huggingface_offline.py          # Hugging Face 下载后转换
    ├── indexed_dataset.py              # Indexed writer
    ├── read_indexed_dataset.py         # 读取检查工具
    └── offline_preparation_guide.md    # 离线预处理用户指南
```

`hyper_parallel/auto_models/components/data/` 是待废弃旧目录。新实现和配置 target 必须使用
`hyper_parallel.auto_models.components.datasets` 路径。

## 13. 组件职责边界

| 组件 | 负责 | 不负责 |
| --- | --- | --- |
| source Dataset | 文件/Hugging Face/Indexed IO | 模型并行通信 |
| data transform | RawSample 到 ModelSample | DP 采样和 batch packing |
| split builder | Indexed split、low/mid/blend 组装 | DataLoader batch 分发 |
| BatchSampler | mapping-style 顺序、DP 分配、消费进度 | Iterable source 分片 |
| DataLoader | 样本选择、worker、prefetch、恢复 | attention mask |
| Collator | 堆叠或 Online packing、生成 Online 边界 | CP/TP 通信 |
| ParallelBatch | 字段归一化、CP 切分、TP 广播、LTR/attention | 读取原始文件 |
| Omni | 保留多模态接口 | 当前不承诺可用 Dataset 实现 |

## 14. 调试与日志

Dataset logger 默认跟随全局日志级别，Trainer 配置可以单独开启：

```yaml
debug:
  check_dataset:   # debug、info、warn；null 表示跟随全局级别
```

也可在 Trainer 初始化前调用：

```python
from hyper_parallel.auto_models.components.datasets import enable_dataset_logging

enable_dataset_logging("debug")                    # rank 0
enable_dataset_logging("debug", ranks=(1, 3))      # 指定 rank
enable_dataset_logging("debug", ranks=None)        # 所有 rank
```

日志覆盖 Dataset 类型选择、split/blend、cache、DataLoader、Sampler 和 parallel batch 形状，不记录
样本文本内容，也不在 `__getitem__` 热路径持续输出。
