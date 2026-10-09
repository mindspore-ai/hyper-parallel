# HyperParallel Data

`hyper_parallel.data` 将原始记录或 Indexed Dataset 转换为一次
forward/backward 所需的模型输入和 loss 输入。模块按数据职责分层，不按模型名称组织：

```text
source
  -> transform
  -> sample selection / packing / collate
  -> DP / CP / TP data parallelism
  -> runtime inputs
  -> model
```

## 1. 目录与职责

```text
data/
├── indexed/    # .idx/.bin、Indexed split、sample index、blend
├── online/     # 原始 Mapping/Iterable source、文件/HF 加载、source blend
├── text/       # LLM tokenizer、chat template、transform、Dataset 入口
├── omni/       # image/video/audio processor transform 生命周期、Omni Dataset 入口
├── batching/   # 候选池、packing、collator、DataLoader、get-batch
├── parallel/   # Mapping DP sampler、DataLoader 所有权、CP shard、TP broadcast
└── tools/      # Indexed 离线制作和检查工具
```

| 模块 | 输入 | 输出 | 不负责 |
| --- | --- | --- | --- |
| `online` | 文件、目录、glob、Hub Dataset | RawSample source | tokenizer、模型语义 |
| `text` | RawSample + tokenizer/template | Text ModelSample | 多模态 processor |
| `omni` | RawSample + processor | Omni planning/ModelSample | source IO、DP 采样 |
| `batching` | ModelSample | collated batch | 原始文件加载 |
| `parallel` | Dataset index 或 collated batch | rank-local batch | 模型 transform |
| `indexed` | `.idx/.bin` prefix | 固定训练样本 | Online transform |

`online` 只提供 source，不承载 LLM、VLM 或具体模型语义。模型特有的 processor/transform
放在模型 adapter 中，例如 DeepSeek-V4.1：

```text
hyper_parallel/models/deepseek_v41/adapter/
├── processor.py
└── transform_fn.py
```

DeepSeek-V4.1 Online VLM 的样本契约、图片展开、label mask 和 batch 字段说明见
[`docs/guide/data/deepseek_v41_vlm_online_data_guide.md`](../../docs/guide/data/deepseek_v41_vlm_online_data_guide.md)。

## 2. 支持状态

| 路径 | 状态 | 主要入口 |
| --- | --- | --- |
| Indexed Text | 已支持 | `build_indexed_text_dataset` |
| Online Mapping Text | 已支持 | `build_online_text_mapping_dataset` |
| Online Iterable Text | 已支持 | `build_online_iterable_dataset` |
| Online Mapping Omni | 已支持 | `build_online_omni_mapping_dataset` |
| Online Iterable Omni | 尚未提供顶层 Dataset 入口 | 可复用 `online` Iterable source seam |
| Offline Omni metadata | transform 生命周期已预留 | Dataset/provider 尚未落地 |
| Omni TP/CP get-batch | 已实现，待多卡训练验证 | `OmniParallelBatch` 仅支持 PP=1 |

当前 Omni 已接入 image/VLM 字段；目录和 transform contract 可继续扩展 video、audio。

## 3. 两类 source

### 3.1 Mapping

Mapping source 有稳定长度和整数索引：

```text
source[index]
  -> RawSample
  -> lazy transform
  -> ModelSample
```

本地 JSONL 使用原生 byte-offset 索引，避免 heterogeneous JSONL 被 Arrow 强制合并 schema。
JSON、Parquet、CSV、Arrow 和 Hub Dataset 使用 Hugging Face `load_dataset(..., streaming=False)`。

Mapping 的顺序为：

```text
每个 source 确定 train/valid/test
  -> 同名 split 内构造全局 blend index
  -> source-local deterministic shuffle/repeat
  -> 公共 BatchSampler 做 DP index shard
  -> lazy transform
```

单路径默认全部作为 train：

```yaml
dataset:
  _target_: hyper_parallel.data.text.build_dataset.build_online_text_mapping_dataset
  data_path: /data/train/*.parquet
  data_config: {}
```

比例 split 使用 Hugging Face split expression，只适用于 Mapping：

```yaml
dataset:
  _target_: hyper_parallel.data.text.build_dataset.build_online_text_mapping_dataset
  data_path: /data/all/*.parquet
  data_config:
    split: "98, 1, 1"  # train / valid / test，合计必须为 100
```

已制作好的独立 split 使用路径 Mapping；`valid`、`test` 可省略：

```yaml
dataset:
  _target_: hyper_parallel.data.text.build_dataset.build_online_text_mapping_dataset
  data_path:
    train: /data/train/*.parquet
    valid: /data/valid/*.parquet
    test: /data/test/*.parquet
  data_config: {}
```

Hub Dataset 与本地路径使用同一个 `data_path`：

```yaml
dataset:
  _target_: hyper_parallel.data.text.build_dataset.build_online_text_mapping_dataset
  data_path: Salesforce/wikitext
  data_config:
    config_name: wikitext-2-raw-v1
    split: "98, 1, 1"
    cache_dir: null
```

多源 Mapping 使用确定性权重调度：

```yaml
data_config:
  sources:
    - data_path: /data/domain_a/*.parquet
      weight: 0.7
    - data_path: /data/domain_b/*.parquet
      weight: 0.3
```

顶层 blend index 保存 `(source_id, source_logical_index)`；每个 source 的逻辑索引再映射到
确定性 shuffle/repeat 后的物理样本。这与 Indexed/Megatron 的“先构造全局顺序，再做 DP shard”一致。

### 3.2 Iterable

Iterable source 使用 Hugging Face `load_dataset(..., streaming=True)`，没有可随机访问的全局 index：

```text
each source
  -> DP / DataLoader-worker shard
  -> local buffer shuffle
  -> local weighted blend
  -> lazy transform
```

因此 Iterable 不支持运行期比例 split。应由输入路径或 Hub 原生 split 决定 train/valid/test。
其 epoch、shuffle buffer 和恢复语义由 stream 自身维护；不能套用 Mapping 的全局 shuffle index。

多源 Iterable 默认使用 `local_weighted`；也可选择 Hugging Face 风格的
`global_interleave`。二者是 source 组合策略，不是 train/valid/test split。

## 4. Text 路径

Text transform 负责 RawSample 到 `input_ids/labels`：

```text
Online source
  -> PlaintextTransform / TextConversationTransform
  -> input_ids [sample_length]
  -> labels    [sample_length]
  -> FixedBatchDataLoader 或 TokenBatchLoader
  -> TextPackingCollator
  -> input_ids/labels [1, packed_length] + cu_seq_lens
  -> TextParallelBatch
```

`TextParallelBatch` 统一处理 Online Text 与 Indexed Text：

```text
DataLoader owner 读取 batch
  -> 解析 sequence boundaries
  -> CP sequence shard
  -> TP broadcast
  -> position_ids / loss_mask
  -> AttentionRuntime
  -> (model_inputs, loss_inputs)
```

`FixedBatchDataLoader` 固定每批样本数；`TokenBatchLoader` 按 token budget 从候选池选择
可变数量的完整样本。二者都不会把单条超长样本静默截断。

## 5. Omni transform 生命周期

`OmniDataTransform` 采用 Energon 风格的 hook 组合。实现类必须满足以下三种模式之一：

| 模式 | 实现 hook | 候选池之前 | 选中之后 |
| --- | --- | --- | --- |
| Eager Online | `encode_sample` | 完整 encode | 原样返回 |
| Deferred Online | `preencode_sample` + `postencode_sample` | 只生成 planning metadata | 完整 encode |
| Offline metadata | `postencode_sample` | 保留已有 metadata | 完成 encode |

禁止同时实现 `encode_sample` 和 `preencode_sample/postencode_sample`。只有
`preencode_sample` 而没有 `postencode_sample` 也属于无效组合。

`encode_batch` 是最终 batch adapter：它在 packing/collate 之后把通用字段转换为模型 forward contract。
例如 DeepSeek-V4.1 在这里将图片 patch length 转换为 offset，并生成 image-to-batch 映射。

AutoProcessor 默认模式：

```python
processor = AutoProcessor.from_pretrained(
    pretrained_model_name_or_path,
    trust_remote_code=True,
)
```

如果 processor 没有 `chat_template`，会复用 `processor.tokenizer.chat_template`。
`AutoProcessorTransform` 只负责 `apply_chat_template`；模型特有的 label、media metadata 和 batch
contract 由模型 adapter 实现。

相对图片、视频和音频路径在 transform 前根据 RawSample 的 source 文件目录解析；远端 URL、data URL
和绝对路径保持不变。

## 6. Omni packing

当前 Online Mapping Omni 流程：

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

默认 `FirstFitPackingSelector` 使用 `packing_length`；字段不存在时使用 `input_ids` 长度。
`SamplePacker` 拼接 token、label、modality tensor，生成 `cu_seq_lens`，并把
`*_token_starts` 从 sample-local 坐标转换为 packed-sequence 坐标。

候选池同时满足以下条件才生成 micro-batch：

```text
len(buffer) >= min_buffered_samples
and
sum(sample_cost) >= token_budget
```

默认：

```text
token_budget = micro_batch_size * data_transform.max_seq_len
```

可在 DataLoader 配置中独立覆盖候选选择预算：

```yaml
dataloader:
  _target_: hyper_parallel.data.batching.OmniPackingLoader
  token_budget: 128
  min_buffered_samples: 1
```

`token_budget` 只控制候选选择，不截断样本。若第一条样本本身超过 budget，first-fit 会让它单独成 batch。
transform 的 `max_seq_len` 仍负责验证模型允许的最大样本长度。

## 7. Omni 分布式约束

`OmniParallelBatch` 在每个 CP 坐标的 TP0 读取相同 DP 样本，通过 `CPBatchSharder` 仅切分 token 字段，
再通过 `TPBatchBroadcaster` 将 CP-local token 和完整图像字段送到同一 CP 坐标的 TP peers。
`pixel_values`、图像网格和全局 `image_token_starts` 不按 CP 切分。当前要求：

```text
PP=1；TP/CP 路径需要分布式进程组
```

`get_batch.encoder_dp` 默认为 `false`：每个 TP×CP rank 接收完整图片并独立计算 ViT，
只将对应本地 CP token 的视觉 embedding 写入文本序列。`true` 留作按完整图片分桶、
经 all-to-all 路由视觉 embedding 的路径；该路径尚未实现，配置为 `true` 会立即报错。

DeepSeek-V4.1 在 YAML 中为 `OmniParallelBatch.runtime_input_adapter` 挂载 `DeepseekV41Runtime`，
负责 packed attention 和图像插入坐标；各 TP×CP rank 在模型前向中重复执行完整 ViT，
再按全局图像 span 与本地 CP token 区间的交集替换 embedding。TP/CP 下 ViT 重复参数的
梯度同步仍需多卡训练验证。

EP/FSDP 可由模型并行计划启用，但模型必须保证每个 rank 以相同顺序和次数调用分布式子模块。

DeepSeek-V4.1 当前逐图片调用 vision tower：

```text
for each image:
    vision(...)   # vision 被 FSDP shard 时会触发 collective
    aligner(...)
```

因此，在 vision 使用 FSDP 时，各 rank 的一个 micro-batch 必须包含相同数量的图片。图片尺寸不同只造成
计算量差异；图片数量不同会造成 vision/FSDP collective 调用次数不同，并可能与后续 EP collective
形成死锁。

当前 DeepSeek-V4.1 smoke 配置使用一条样本、一张图片组成一个 rank-local micro-batch：

```yaml
data_transform:
  _target_: hyper_parallel.models.deepseek_v41.adapter.data.transform_fn.build_deepseek_v41_omni_transform
  max_seq_len: 4096

dataloader:
  _target_: hyper_parallel.data.batching.OmniPackingLoader
  token_budget: 128       # 第一条完整样本立即出 batch，不截断到 128
  min_buffered_samples: 1
```

这只是当前 baseline，不是通用多图 packing 方案。多图 FSDP 的目标方案参考 PanGu：

```text
本 rank real image count
  -> 在 vision 第一次调用前对齐 global image count
  -> 不足部分补最小合法 dummy image
  -> real/dummy mask
  -> 每个 rank 执行相同次数的 vision/aligner
  -> 仅 real image 写回语言 token span
```

dummy 必须保留零梯度依赖，使 forward 和 backward 的 FSDP 调用次数都一致。该 image-call padding
尚未在 HyperParallel 实现；在实现前，不要恢复每 rank 可变图片数的 DeepSeek FSDP packing。

## 8. Indexed Text

Indexed 路径使用同一 prefix 的两个文件：

```text
<prefix>.idx  # dtype、length、pointer、document index
<prefix>.bin  # 连续 token payload
```

两种 mid-level Dataset：

| 数据形式 | 配置 | Dataset | 运行期行为 |
| --- | --- | --- | --- |
| 变长文档 | `is_dataset_from_mr: false` | `GPTDataset` | document/sample/shuffle index 动态组样 |
| 离线预切定长 record | `is_dataset_from_mr: true` | `GPTFromMRDataset` | 直接读取固定 record |

Indexed 的逻辑顺序为：

```text
build low-level source
  -> split train/valid/test
  -> build split-local mid-level Dataset
  -> blend sources
  -> sample shuffle index
  -> DP BatchSampler
```

`split: "98, 1, 1"` 按 low-level element 连续划分。独立的
`train_data_path/valid_data_path/test_data_path` 可替代共享 split。

标准 blend 使用权重调度；`simple_blend: inter/intra` 只适用于离线预切 record。
离线制作命令和格式说明见 `docs/guide/data/offline_preparation_guide.md`。

## 9. Sampler、epoch 与恢复

Mapping Dataset 使用 `build_dataset_batch_sampler`：

- `single`：顺序消费 Dataset 已构造好的全局索引；跨 epoch 复用相同顺序。
- `cyclic`：使用 `seed + epoch` 构造确定性 shuffle。
- `drop_last`：丢弃不能组成完整 distributed micro-batch 的尾部。
- `data_rearrange_map`：在逻辑 index 与物理 Dataset index 之间增加映射。

动态 DataLoader checkpoint 保存：

```text
source DataLoader cursor
+ BatchSampler consumed_samples / epoch
+ 未消费 candidate buffer
```

Mapping 默认按 index 保存 buffer，可重放 transform；Iterable 默认保存完整 sample。当前 Online
DataLoader resume 要求 `global_batch_size` 和 DP world size 不变。

## 10. 配置示例

### Online Mapping Text

```yaml
dataset:
  _target_: hyper_parallel.data.text.build_dataset.build_online_text_mapping_dataset
  data_path: /data/train.jsonl
  data_transform:
    _target_: hyper_parallel.data.text.text_transform.build_text_transform
    data_type: plaintext
    text_keys: text
    max_seq_len: 4096
  data_config: {}

dataloader:
  _target_: hyper_parallel.data.batching.TokenBatchLoader
  min_buffered_samples: 200
  collate_fn:
    _target_: hyper_parallel.data.batching.build_online_text_collate_fn
  get_batch:
    _target_: hyper_parallel.data.batching.TextParallelBatch
    source_type: online
```

### Online Mapping DeepSeek-V4.1 Omni baseline

```yaml
dataset:
  model_assets:
    _target_: hyper_parallel.models.deepseek_v41.adapter.data.processor.build_deepseek_v41_processor
    config_path: /models/DeepSeek-V4.1-Flash
  data_transform:
    _target_: hyper_parallel.models.deepseek_v41.adapter.data.transform_fn.build_deepseek_v41_omni_transform
    max_seq_len: 4096
  _target_: hyper_parallel.data.omni.build_dataset.build_online_omni_mapping_dataset
  data_path: /data/train.jsonl
  data_config: {}

dataloader:
  _target_: hyper_parallel.data.batching.OmniPackingLoader
  token_budget: 128
  min_buffered_samples: 1
  drop_last: true
  num_workers: 0
  collate_fn:
    _target_: hyper_parallel.data.batching.build_omni_collate_fn
  get_batch:
    _target_: hyper_parallel.data.batching.OmniParallelBatch
```

## 11. 调试

```yaml
debug:
  check_dataset: debug  # debug / info / warn；null 表示跟随全局级别
```

也可在 Trainer 初始化前启用：

```python
from hyper_parallel.data.dataset_logging import enable_dataset_logging

enable_dataset_logging("debug")
enable_dataset_logging("debug", ranks=(1, 3))
enable_dataset_logging("debug", ranks=None)
```

日志覆盖 source、split/blend、Dataset、Sampler、DataLoader 和 parallel batch 形状，不记录样本文本内容。
