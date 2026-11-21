# HyperParallel Data

[English](README.md) | **简体中文**

使用入口：[YAML 配置结构](../../docs/guide/trainer/yaml_config.zh.md)、
[Omni Online 接入](../../docs/guide/data/omni_online_data_guide.zh.md)、
[文本/图文 SFT](../../docs/guide/data/qwen3_8_sft_data_guide.md)、
[Offline 制作参数](../../docs/guide/data/offline_preparation_guide.zh.md)。
本文第 10 节给出 Online/Offline 的训练配置片段与长度、packing、padding 参数说明。

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
[`docs/guide/data/deepseek_v41_vlm_online_data_guide.md`](../../docs/guide/data/deepseek_v41_vlm_online_data_guide.zh.md)。

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

文本训练通过 `dataloader.get_batch.source_type: offline` 选择预处理后的 Indexed 数据，
通过 `online` 选择原始数据。底层 Indexed 存储 API 保留原名。

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
  data_config:
    max_seq_len: 4096
```

比例 split 使用 Hugging Face split expression，只适用于 Mapping：

```yaml
dataset:
  _target_: hyper_parallel.data.text.build_dataset.build_online_text_mapping_dataset
  data_path: /data/all/*.parquet
  data_config:
    max_seq_len: 4096
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
  data_config:
    max_seq_len: 4096
```

Hub Dataset 与本地路径使用同一个 `data_path`：

```yaml
dataset:
  _target_: hyper_parallel.data.text.build_dataset.build_online_text_mapping_dataset
  data_path: Salesforce/wikitext
  data_config:
    max_seq_len: 4096
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
  -> TokenBatchLoader
  -> TextPackingCollator
  -> input_ids/labels [B, padded_length] + cu_seq_lens
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

Online 使用 `TokenBatchLoader` 按每行 S 的上限选取完整样本，组成 B 行。
`_target_` 直接决定 loader：文本 packing 使用 `TokenBatchLoader`，Omni packing 使用 `OmniPackingLoader`，
Trainer 不再自动替换类型。
Offline Indexed 配合 `build_indexed_collate_fn` 时仍使用固定样本组批。

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

本地 JSONL、JSON、Parquet、Arrow Mapping 记录保留来源文件的绝对路径，transform 用该文件所在目录
解析可识别的相对媒体路径。多文件组合、过滤和 split 后仍各自使用原文件目录；远端 URL、data URL 和
绝对媒体路径保持不变。直接调用 transform 或使用没有来源路径元数据的数据时，应自行提供绝对路径。

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

只需用 `dataset.data_config.max_seq_len` 设置 S。Online 文本和 Omni 分别 packing B 条序列，
每条最多 S 个 token，再统一 padding 得到 `[B,T]`。`pad_to_length: true`（默认）每条补到 S；
false 补到当前最长序列，并向上按 TP×CP 对齐。S 必须能被 TP×CP 整除。

候选池同时满足以下条件才生成 micro-batch：

```text
len(buffer) >= min_buffered_samples
and
sum(sample_cost) >= micro_batch_size * max_seq_len
```

```yaml
dataset:
  data_config:
    max_seq_len: 128
  data_transform:

dataloader:
  _target_: hyper_parallel.data.batching.OmniPackingLoader
```

单条样本超过 S 时跳过并计数；样本自身不超限但当前 packed 序列放不下时，留在 buffer 参与后续序列。
重复读取不清空 buffer；有限源只有不足 B 条 packed 序列的尾批受 `drop_last` 控制。

## 7. Omni 分布式约束

`OmniParallelBatch` 在每个 CP 坐标的 TP0 读取相同 DP 样本，通过 `CPBatchSharder` 仅切分 token 字段，
再通过 `TPBatchBroadcaster` 将 CP-local token 和完整图像字段送到同一 CP 坐标的 TP peers。
`pixel_values`、图像网格和全局 `image_token_starts` 不按 CP 切分。当前要求：

```text
PP=1；TP/CP 路径需要分布式进程组
```

每个 TP×CP rank 接收完整图片并独立计算 ViT，只将对应本地 CP token 的视觉 embedding 写入文本序列。
按完整图片分桶、经 all-to-all 路由视觉 embedding 的 encoder-DP 路径尚未实现；
当前 `OmniParallelBatch` YAML 不接受 `encoder_dp` 参数。

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

DeepSeek-V4.1 smoke 配置统一使用一个 packing 与固定 padding 长度上限：

```yaml
dataset:
  data_config:
    max_seq_len: 4096
  data_transform:
    _target_: hyper_parallel.models.deepseek_v41.adapter.data.transform_fn.build_deepseek_v41_omni_transform

dataloader:
  _target_: hyper_parallel.data.batching.OmniPackingLoader
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
| 变长文档 | `low_level_dataset._target_: ...GPTDataset` | `GPTDataset` | document/sample/shuffle index 动态组样 |
| 离线预切定长 record | `low_level_dataset._target_: ...GPTFromMRDataset` | `GPTFromMRDataset` | 直接读取固定 record |

完整 target 模块为 `hyper_parallel.data.offline.dataset`。

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
- `drop_last`：Online packing 在组出 packed 行之后丢弃不足 B 行的有限源尾批；
  `single/cyclic` 均保留可参与 packing 的原始记录尾部。Offline 固定组批按 distributed micro-batch 丢尾。
- `data_rearrange_map`：在逻辑 index 与物理 Dataset index 之间增加映射。

动态 DataLoader checkpoint 保存：

```text
source DataLoader cursor
+ BatchSampler consumed_samples / epoch
+ 未消费 candidate buffer
```

Mapping 默认按 index 保存 buffer，可重放 transform；Iterable 默认保存完整 sample。
Online 动态 loader 拒绝改变 DP world size 的恢复。逐批对齐验证应保持 batch、并行配置及源数据不变。

当前 HF Iterable 开启 buffer shuffle 时，checkpoint 未完整保存 HF shuffle buffer，恢复可能丢失尚未输出的
记录；保存候选池不能消除这一问题。`shuffle: false` 可用于无 shuffle 的恢复场景，不能代表 shuffle 恢复
问题已经修复。`datasets 4.0.0` 的 cyclic/repeat 跨源轮次恢复还存在 epoch 游标兼容问题。

## 10. 配置示例

以下是与模型、优化器配置组合使用的数据片段；完整 recipe 可参考
[Online 文本](../../hyper_parallel/models/qwen3_8/recipes/train_online.yaml) 和
[Offline 文本](../../examples/training_demo/qwen3_moe/train_parallel_offline.yaml)。

### 必选项

下方最小模板只写必选项；Online 与 Offline 都必须配置 `dataset.data_config.max_seq_len`。

| YAML 配置 | Online 原始数据 | Offline Indexed 数据 |
| --- | --- | --- |
| `dataset._target_` | Text Mapping/Iterable 或 Omni Mapping builder | `build_indexed_text_dataset` |
| `dataset.data_path` | 文件、目录、glob、路径列表或 Hub ID；与 `data_config.sources` 二选一 | `.bin/.idx` 的共同 prefix，不带扩展名 |
| 长度上限 | 必须配置 `dataset.data_config.max_seq_len` | `dataset.data_config.max_seq_len` |
| 训练终止条件 | 必须一起设置 `training.epochs`、`training.max_steps` | `training.train_iters`、`training.train_samples` 二选一 |
| `dataset.data_config` | 必须设置 `max_seq_len` | 必须设置 `max_seq_len`、`low_level_dataset`、`simple_blend` |
| Text runtime | `TextParallelBatch` + `source_type: online` | `TextParallelBatch` + `source_type: offline` |
| Omni runtime | `OmniParallelBatch` 自带 Online 语义 | 当前没有顶层 Offline Omni Dataset 入口 |

Online 的 `max_steps` 是**每个训练 epoch 的 optimizer step 数**，总步数为 `epochs * max_steps`；
epoch 到达这个步数即切换，不表示已经遍历完原始文件。`dataset.data_config.repeat` 默认 `true`，
源耗尽时重读，保留候选池；源轮次由 EOF 推进。`single` 重用顺序，`cyclic` 随源轮次更新 shuffle。
当前 Trainer 的 Online 计划要求 repeat，不能设置 `data_config.repeat: false` 或 `dataloader.repeat: false`，
也不能与 `train_iters/train_samples` 或 `dataloader.load_balance` 混用。

### YAML 各节点与数据顺序

`dataset.model_assets` 加载 tokenizer/processor，`dataset.data_transform` 决定如何把记录编码为模型样本；
`dataset.data_config` 管理源、split 和读取选项。`dataloader` 管理采样及候选池，`collate_fn` 管理组批后的
padding，`get_batch` 管理 CP/TP 传输和模型输入。模型配置里的最大上下文长度不会自动替代 Dataset 的 S。

### DataLoader 的组批原理与选型

设 B=`training.micro_batch_size`，S=`dataset.data_config.max_seq_len`。
**Dataset 决定一条记录如何变成模型样本，loader 决定选哪些样本，packer 决定怎样拼接，collator 决定怎样 padding/stack。**
Mapping Dataset 先由 sampler 分配本 DP rank 的索引；Iterable Dataset 自行完成流式读取和分片。
原生 loader 每次 yield 对应一个 forward/backward micro-batch；每个 optimizer step 消费
`global_batch_size / (B * DP)` 个 micro-batch，例如全局 8、B=2、DP=2 时为 2 个。
组批后的 CP 切分、TP 传输由 `get_batch` 完成，loader 返回的是切分前的 `[B,T]`。

| loader/入口 | 如何产生一个 micro-batch | 典型场景 |
| --- | --- | --- |
| `FixedBatchDataLoader` | sampler 取 B 条 Dataset 输出，直接交给 collator | Offline Indexed、离线 SFT、已经编码好的固定行 Omni |
| `TokenBatchLoader` | 编码后的文本进入 buffer，按每行不超过 S 选样、拼接，重复 B 次，再 padding | Online Mapping/Iterable 文本、文本 SFT |
| `OmniPackingLoader` | 按长度元数据选样，再执行选中样本的编码、packing、padding 和 batch 编码 | Online 图文/音视频 packing，特别是延迟媒体编码 |
| `DynamicBatchDataLoader` | `TokenBatchLoader` 的兼容别名 | 旧 Python/YAML 调用；新配置直接用 `TokenBatchLoader` |
| `dataloader.load_balance` | 构建 collective loader，在原生 sampler 的样本窗口内做跨 DP 成本调度 | 已有固定/预打包数据的负载均衡；不是第四种 Online buffer loader |

#### FixedBatchDataLoader：固定记录数，不维护 packing buffer

```text
Dataset 已生成的行 A[S]、B[S]、C[S]、D[S]
    -> sampler 本次选择 A、B
    -> collator([A, B]) -> batch [2,S]
下一批选择 C、D；loader 不把 A、B 的内容重新拼成一行。
```

因此 B 表示 Dataset 输出条数；若一条 Dataset 输出本身含多行，最终行数还由 collator 决定。
Indexed 的样本构造可能已经拼接多个文档，这发生在 Dataset 内部，不能理解成 Fixed loader 在做 packing。
`build_indexed_collate_fn` 使用普通 stack；含文档边界的 SFT 数据使用 `build_sft_collate_fn`。
Kimi 的 eager transform 先生成长度 S 的单行，再由 `build_omni_collate_fn` 堆成 `[B,S]`，也是合法的 Fixed 用法。
Fixed loader 直接搭配文本 packing collator，并不会获得候选池；无 `cu_seq_lens` 的多条文本会被该 collator
当成一行拼接，不能据此保证 B 条独立 packed 行。需要该保证时使用 `TokenBatchLoader`。

下面是合并到现有 Offline Indexed 配置的片段，沿用第 10 节后面的完整 Dataset 示例：

```yaml
training:
  train_iters: 20
  micro_batch_size: 2
  global_batch_size: 8

dataset:
  data_config:
    max_seq_len: 4096

dataloader:
  _target_: hyper_parallel.data.batching.FixedBatchDataLoader
  collate_fn:
    _target_: hyper_parallel.data.batching.build_indexed_collate_fn
  get_batch:
    _target_: hyper_parallel.data.batching.TextParallelBatch
    source_type: offline
```

每个 DP rank 每次 forward 得到两条长 4096 的行；原生 sampler 默认丢弃不足完整尾批的记录。

#### TokenBatchLoader：从候选池选完整样本，分别拼出 B 行

```text
源样本 -> 按 input_ids 的 token 长度加入候选 buffer（超过 S 的样本跳过）
       -> 候选数 >= min_buffered_samples 且候选总长度 >= B*S
       -> 扫描 buffer，能放入当前行就选择，放不下就保留，继续扫描后面的候选
       -> 拼接为一行，保留每个样本的 cu_seq_lens 边界
       -> 再从剩余 buffer 选下一行，直到选出 B 行
       -> collator 为这 B 行选择同一个 padding 长度 -> [B,T]
```

两个门槛控制何时开始选样，不是 buffer 容量上限，也不保证每行都能恰好填满 S。
被跳过的候选可能在下一行或下一批选中；后面的小样本仍可能填进当前行，因而不是严格连续切段。
B*S 是候选总量的门槛，**每行的 packing 上限始终是 S**。

例如 S=4096、B=2、`min_buffered_samples: 1`，候选长度为 `[1800,2000,500,3000,4096]`。
候选总长 11396 已超过 8192，触发一次组批：

```text
packing：第 1 行 [1800][2000] = 3800；500、3000、4096 都放不下，保留
         第 2 行 [500 ][3000] = 3500；4096 仍保留在 buffer
padding：第 1 行 [1800][2000][PAD 296] = 4096
         第 2 行 [500 ][3000][PAD 596] = 4096
                      -> input_ids/labels [2,4096]
                      -> CP=2 时每个 CP rank [2,2048]
```

`pad_to_length: true` 默认把每行补到 S；`false` 补到最长 packed 行，并向上对齐到 TP×CP 的倍数。
上述最长行 3800，若 TP=4、CP=2，动态 padding 得到 `[2,3800]`，CP 后 `[2,1900]`。
padding 的 input token 为 0，label 为 -100；`cu_seq_lens` 同时记录真实样本和独立 padding 段的边界。
TP 不会像 CP 一样把每行长度再除以 TP，TP×CP 只用于这里的长度对齐约束。

沿用 Online Text 的 Dataset/tokenizer/transform，配置片段如下：

```yaml
training:
  epochs: 1
  max_steps: 10
  micro_batch_size: 2
  global_batch_size: 8

dataset:
  data_config:
    max_seq_len: 4096

dataloader:
  _target_: hyper_parallel.data.batching.TokenBatchLoader
  min_buffered_samples: 1  # 演示小候选池；省略时默认 200
  collate_fn:
    _target_: hyper_parallel.data.batching.build_online_text_collate_fn
    pad_to_length: true   # 默认 true；false 使用本批最长行并做 TP/CP 对齐
  get_batch:
    _target_: hyper_parallel.data.batching.TextParallelBatch
    source_type: online
```

#### OmniPackingLoader：先选样，再执行媒体编码和 packing

它与 Token loader 共用 B 行循环、候选门槛、padding、EOF 和 checkpoint 机制；差别在选样前后的工作：

```text
原始图文 -> Dataset eager encode 或 preencode（产出 packing_length 等规划信息）
         -> buffer 按 PackingSelector 的成本选择总量 <= S 的候选
         -> encode_selected_sample：执行被选中样本的 postencode
         -> SamplePacker：拼接 token/labels/媒体字段，并平移媒体 token 起点
         -> 重复 B 行 -> OmniCollator padding/拼接媒体 -> Dataset.encode_batch
```

默认 `FirstFitPackingSelector` 使用 `packing_length`，缺省时使用 `input_ids` 长度；已有编码 token 时成本
不会低于实际长度。延迟编码的 `packing_length` 必须包含展开的媒体 token，编码后行长超过 S 会报错。
选择后才做重媒体编码，可减少未被选中候选的编码和缓存开销；eager transform 则已在入池前完成编码。
例如两条图文样本的总成本分别是 1800、2000（均包含图片 token），可以拼成一行 3800；
图片张量随样本合并，第二条样本的媒体 token 起点加上前一条的 token 长度。

沿用 Online Omni Dataset 和模型对应的 transform，将上例的 dataloader 改为：

```yaml
dataloader:
  _target_: hyper_parallel.data.batching.OmniPackingLoader
  collate_fn:
    _target_: hyper_parallel.data.batching.build_omni_collate_fn
  get_batch:
    _target_: hyper_parallel.data.batching.OmniParallelBatch
```

B=2 是通用 loader 的能力；当前 DeepSeek-V4.1 compact runtime 仍限制 B=1，选择模型时需一并考虑。

#### EOF、repeat、drop_last 与断点恢复

| 情况 | 实际处理 |
| --- | --- |
| 动态 loader 源文件 EOF，`repeat=true` | 不清空 buffer；推进源轮次、重新读取，再与预留候选一起累积到组批门槛 |
| 动态 loader 有限遍历，`repeat=false` | EOF 后绕过两个候选门槛，排空剩余候选；内部 `drop_last=true` 只丢弃最终不足 B 行的 batch |
| 某行真实长度小于 S，但有 B 行 | 正常保留并 padding，不因行没填满 S 而 drop |
| Fixed loader 的源尾部不足完整 batch | 由原生 sampler 的 `drop_last` 处理；外层重复不会把已经丢弃的尾记录拿回来补批 |

动态 loader 底层 source sampler/loader 的 `drop_last` 为 false，避免在样本进入 buffer 前丢弃原始尾记录。
Online Trainer 默认重复，使用 `epochs * max_steps` 控制总训练步数，训练 epoch 边界不清空候选池；
`repeat=false` 的有限排空仅适用于独立 Python 调用，当前 Online Trainer 不接受该设置。
`drop_last`、worker 和 checkpoint 存储策略使用内部默认值，不通过 YAML 设置；
`min_buffered_samples`、`repeat` 保留为动态 loader 参数。

例如上次留下 `[500]`，源 EOF 后重新读到 `[1800,2000,...]`，同一个 buffer 变为
`[500,1800,2000,...]`；它还需要继续积累到候选门槛，不会因为遇到 EOF 就立即吐出这一组。
checkpoint 保存 source cursor、源轮次以及尚未消费的 buffer；Mapping 默认保存重放索引，Iterable 默认保存完整样本。
恢复时先使用已经满足门槛的缓存，再继续读取源；保持相同 DP 大小和确定性的 transform 才能重建同一数据序列。

`RepeatingDataLoader` 是普通/Fixed loader 外层的重读包装，重复的是已经组好的 batch，不负责候选 packing；
动态 loader 在自身内部重读，因此无需这个包装。`HyperIter`/后台预取只负责迭代传递，不决定 packing 行长。

#### load_balance：跨 DP 调度，与本地 buffer packing 分开

省略 `load_balance` 时直接使用 `_target_`；配置 `load_balance: {}` 会构建 `DistributedDataLoader`。
该路径保留原生 sampler 的训练步和样本窗口，用成本估计、跨 DP 分配及数据传输来减少各 rank 的计算差异。
默认 `repack_sequences=false` 保留 Dataset 样本；显式开启时按文档边界拆分、重新 packing，
因此可能改变每个 rank 的样本组合。它不调用 Token/Omni loader 的候选 buffer。

例如一批四个等长但媒体/注意力成本不同的样本为 `[9,8,2,1]`：原分配可能为 `[9,8]` 和 `[2,1]`，
在约束允许且估计成本准确时，可分配成 `[9,1]` 和 `[8,2]`，两边成本从 17/3 接近到 10/10。
这是成本调度示意，不保证每个窗口都能完全均衡。

在上面的 Offline Fixed 配置中加入：

```yaml
dataloader:
  _target_: hyper_parallel.data.batching.FixedBatchDataLoader
  load_balance: {}  # 开启成本调度，沿用已有 collate_fn/get_batch
```

该片段须与原 collate/get_batch 合并；需要 Mapping Dataset、device mesh，且当前 PP 必须为 1。
不要与 Token/Omni 动态 loader 或 Online `epochs/max_steps` 组合。
已有预打包文本的重 packing 示例见 [Qwen balanced 配置](../../examples/training_demo/qwen3/train_balanced.yaml)。

直接使用 Python API 时还有 `LocalBalancingDataLoader`：接收外部已经选好的整步样本组，
在本地 DP 分组内交换、重组，源迭代器继续负责选样和结束条件；当前仅支持纯 DP，不支持断点恢复。
它不是 Trainer YAML 的普通 loader 选项，接入示例见 [本地均衡说明](../distributed_data/NODE_LOCAL_BALANCING.md)。

实现入口：[loader 与组批循环](batching/build_dataloader.py)、[候选选择](batching/dynamic_batch.py)、
[字段 packing](batching/packing.py)、[padding/collator](batching/build_collate_fn.py)、
[重复读取包装](batching/build_dataloader.py)。

### 默认值（不写入最小模板）

以下参数省略时使用默认值；只有需要改变行为时才显式覆盖。

worker 执行选项使用内部默认值：`num_workers=0`、`pin_memory=false`、`prefetch_factor=null`。
内部 `drop_last=true` 丢弃不足完整行数的最终尾批；packing 保留原始 source 尾部样本。
候选池 checkpoint 自动为 Mapping 保存索引、为 Iterable 保存完整样本。
这五项不接受 YAML/CLI 配置，`min_buffered_samples` 和 `repeat` 仍可设置。

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `training.micro_batch_size` | `1` | 每个 DP rank 每次 forward 的 packed 行数 B |
| `training.global_batch_size` | `8` | 每个 optimizer step 的全局行数，须能被 B×DP 整除 |
| `training.seed` | `null`（Trainer 初始化为 `42`） | 统一注入 source 与 sampler 的随机种子 |
| `dataloader.sampler_type` | `single` | 重用源顺序；可选 `cyclic` 随源轮次 shuffle |
| `dataloader.min_buffered_samples` | `200` | 候选数门槛，还需满足候选总成本至少为 B×S |
| `dataloader.repeat` | `null`：Trainer Online 启用重复；独立 Python 调用为有限遍历 | 继承调用方的重复策略 |
| `dataset.data_config.repeat` | `true`（Online Trainer） | 源耗尽后重读，保留候选池 |
| `dataloader.collate_fn.pad_to_length` | `true` | padding 到 S；`false` 为动态 padding |
| `dataset.data_config.split` | Online：未配置则全部 train；Offline：`"1, 0, 0"` | train/valid/test 划分；独立 split 路径另行配置 |
| `dataset.data_config.labels_are_shifted` | `true`（Text runtime） | 输入 labels 已按预测目标位移 |
| `dataset.data_transform.text_keys` | `null`：plaintext 读 `text`，sharegpt 读 `conversations` | 选择输入列，messages 列需显式覆盖 |
| `dataset.data_transform.drop_ragged_tail` | `false` | 保留 plaintext 后续不足完整窗口的尾段 |
| `dataset.data_config.shuffle` | `true`（Iterable） | HF buffer shuffle；Mapping 顺序由 sampler/source index 决定 |
| `dataset.data_config.shuffle_buffer_size` | `10000`（Iterable） | shuffle 缓冲容量，仅 shuffle 开启时使用 |
| `dataset.data_config.split_by_data_parallel` | `true`（Iterable） | 源负责 DP 分片 |
| `dataset.data_config.blend_strategy` | `local_weighted`（Iterable） | 多源组合策略，可选 `global_interleave` |
| `dataset.data_config.stopping_strategy` | `first_exhausted`（Iterable） | 任一源耗尽结束本次源遍历 |
| `dataset.data_transform.sample_alignment_size` | `1`（DeepSeek-V4.1） | 单样本进入 packing 前的对齐粒度 |

### Trainer 注入项（不要在 YAML 中设置）

这些参数由运行时生成。即使 Python 签名允许传入，Trainer 也会提供或覆盖其值；
独立调用 Python API 时，才按接口要求手动传入。

| 参数位置 | 值的来源 |
| --- | --- |
| `dataloader.batch_size` | `training.micro_batch_size` |
| `dataloader.dp_world_size` | 当前 mesh 的 DP 大小 |
| `dataloader.max_seq_len`、`dataloader.collate_fn.max_seq_len` | `dataset.data_config.max_seq_len` |
| `dataloader.seed`、`dataset.data_config.random_seed` | 由训练 seed 策略生成，用户统一配置 `training.seed` |
| `dataloader.dataset/batch_sampler` | Trainer 构建的 Dataset 和 sampler |
| `collate_fn.mesh_context`、`get_batch.mesh_context/device/dataloader_context` | 当前分布式运行环境 |
| `dataset.transform/tokenizer/training_config/mesh_context/dataloader_context/sampler_type` | 已构建组件与 Trainer 配置 |
| `dataset.data_transform.tokenizer/chat_template/processor/training_mode` | model_assets 与 `dataset.training_mode`，按 Text/Omni 路径注入 |

### 不支持与互斥项

| 配置 | 约束 |
| --- | --- |
| `dataloader.token_budget` | 已无此参数；packing 上限只通过 transform 的 `max_seq_len` 设置 |
| `OmniParallelBatch.encoder_dp` | 当前接口不接受 |
| Online 的 `repeat: false`、`dataloader.load_balance` | 不支持当前 `epochs/max_steps` 训练计划 |
| `training.epochs/max_steps` 与 `train_iters/train_samples` | 不能混用；前者用于 Online，后者用于 Offline |
| `data_path` 与 `data_config.sources` | 二选一；多源时每项必须有路径及正数权重 |
| Iterable 的比例 `split` | 不支持；Mapping 比例 split 也不能与预分割路径 Mapping 混用 |

### YAML 如何控制单条长度、packing 与 padding

设 B=`training.micro_batch_size`，S=`dataset.data_config.max_seq_len`，A=`TP×CP`：

这里的长度以编码后的 `input_ids` token 数计算，包括 prompt 和展开的媒体 token；
`labels=-100` 只屏蔽 loss，不会让对应输入 token 从 packing 长度中消失。

```text
原始记录 -> transform 输出 0/1/多条模型样本 -> 候选池
         -> 每行选择若干完整样本，总长不超过 S -> 重复选择 B 行
         -> collator 统一 padding 到 T -> [B,T] -> CP 切分 -> [B,T/CP]
```

| 参数 | 含义及作用阶段 |
| --- | --- |
| `dataset.data_config.max_seq_len: S` | 模型样本长度约束，同时作为每行 packing 上限；Trainer 自动传给 loader/collator，无需重复设置 |
| `training.micro_batch_size: B` | 每个 DP rank、每次 forward 的 packed 行数，不是原始记录条数 |
| `training.global_batch_size` | 每个 optimizer step 的全局行数；须能被 `B * DP` 整除，商为梯度累积次数 |
| `dataloader.min_buffered_samples` | 候选样本数门槛；与候选总成本至少为 `B*S` 一起触发选样，不改变 S 或 B |
| `dataloader.collate_fn.pad_to_length: true` | 每行补到 S，得到 `[B,S]` |
| `dataloader.collate_fn.pad_to_length: false` | 补到本 batch 最长 packed 行，再向上按 A 对齐：`T=ceil(max(row_length)/A)*A` |

S 必须是正整数且能被 A 整除。固定和动态 padding 的 **packing 上限始终都是 S**，动态 padding 只缩短
选样后的最终张量；DP 不参与长度对齐。`pad_to_length` 只接受 bool，固定行长统一来自 S。
当前 loader 不接受独立的 `token_budget` YAML 参数。

原始文本的字符数不是 token 长度。`plaintext` 先 tokenize、追加可用的 EOS，再用 S+1 token 窗口构造
`input_ids=window[:-1]`、`labels=window[1:]`，因此一条长记录可生成多个长度不超过 S 的样本；
`drop_ragged_tail: true` 可丢弃后续不足完整窗口的尾段，默认 `false`。文本对话由 chat template 限长后
再 shift labels；例如 tokenizer template 保留最后至多 S 个编码 token，再得到至多 S-1 个输入 token。
通用 `AutoProcessorTransform` 对超过 S 的媒体样本报错；当前 DeepSeek-V4.1 transform 警告并跳过，
均不裁断图像 span。多模态长度包含展开的图片 token。

packing 按候选顺序选择可放入当前行的完整样本，放不下的留到下一行/批次；某行可以小于 S。
loader 遇到仍超过 S 的模型样本会跳过并计数，不会靠 padding 截断。`cu_seq_lens` 保留文档边界，
padding 段也有独立边界。Online collator 的输入 padding 为 0，label 为 -100、loss mask 为 0。

例如 S=4096、B=2，候选池只有 `[1800,2000,500,3000]` 四条样本（如有限源 EOF 排空时），
选出两行 `[3800,3500]`：

| padding 设置（CP=2、TP=4，A=8） | padding 后长度 | 每个 DP rank 在 CP 切分前的形状 | 每个 CP rank 形状 |
| --- | --- | --- | --- |
| `pad_to_length: true` | 两行都是 4096，分别补 296、596 token | `[2,4096]` | `[2,2048]` |
| `pad_to_length: false` | 两行都是 3800，分别补 0、300 token | `[2,3800]` | `[2,1900]` |

另如最长行是 3797，动态 padding 在 A=8 时补到 3800。样本自身也可能先由模型 transform 对齐；
DeepSeek-V4.1 的 `sample_alignment_size` 控制这一层，默认 1，与最终 batch padding 分开。
当前 DeepSeek-V4.1 compact VLM runtime 仍要求 B=1；通用 loader 的形状契约不取消模型自身的 batch 限制。

### Online Mapping Text

```yaml
training:
  epochs: 2
  max_steps: 100

dataset:
  _target_: hyper_parallel.data.text.build_dataset.build_online_text_mapping_dataset
  data_path: /data/train.jsonl
  model_assets:
    tokenizer:
      _target_: hyper_parallel.data.text.build_tokenizer.AutoTokenizer.from_pretrained
      pretrained_model_name_or_path: /models/tokenizer
  data_transform:
    _target_: hyper_parallel.data.text.text_transform.build_text_transform
    data_type: plaintext
  data_config:
    max_seq_len: 4096

dataloader:
  _target_: hyper_parallel.data.batching.TokenBatchLoader
  collate_fn:
    _target_: hyper_parallel.data.batching.build_online_text_collate_fn
  get_batch:
    _target_: hyper_parallel.data.batching.TextParallelBatch
    source_type: online
```

该例读取 `{"text":"..."}` 格式的 JSONL。SFT 使用 `data_type: sharegpt`、
`text_keys: messages`（或 `conversations`），并配置 `dataset.model_assets.chat_template: tokenizer`；
`alpaca` 需设置 `column_mapping`，完整字段示例见 [SFT 指南](../../docs/guide/data/qwen3_8_sft_data_guide.md)。

### Online Iterable Text

沿用上例的 tokenizer、transform、loader 和 `source_type: online`，只替换 Dataset builder 与路径：

```yaml
dataset:
  _target_: hyper_parallel.data.text.build_dataset.build_online_iterable_dataset
  data_path: /data/train-*.jsonl
  data_config:
    max_seq_len: 4096
```

最小模板使用默认值表中的顺序策略；如需无 shuffle 的恢复场景，额外覆盖 `data_config.shuffle: false`，
并参阅第 9 节的恢复限制。
`blend_strategy` 决定多源组合方式，`stopping_strategy` 决定何时结束一次源遍历，
`repeat` 决定结束后是否重读。关闭 `split_by_data_parallel` 后需自行
保证 DP 数据分片，避免各 rank 读到重复记录。Iterable 的 worker 可用并行度还取决于输入文件分片数。
Mapping 源内部采用索引回放，Iterable 源默认保存完整样本。
回放此前按索引保存的 checkpoint 时，源需要实现 `get_item/output_index_for_resume`。

### Online Mapping DeepSeek-V4.1 Omni baseline

```yaml
training:
  epochs: 1
  max_steps: 100

dataset:
  training_mode: sft
  model_assets:
    _target_: hyper_parallel.models.deepseek_v41.adapter.data.processor.build_deepseek_v41_processor
    config_path: /models/DeepSeek-V4.1-Flash
  data_transform:
    _target_: hyper_parallel.models.deepseek_v41.adapter.data.transform_fn.build_deepseek_v41_omni_transform
  _target_: hyper_parallel.data.omni.build_dataset.build_online_omni_mapping_dataset
  data_path: /data/train.jsonl
  data_config:
    max_seq_len: 4096

dataloader:
  _target_: hyper_parallel.data.batching.OmniPackingLoader
  collate_fn:
    _target_: hyper_parallel.data.batching.build_omni_collate_fn
  get_batch:
    _target_: hyper_parallel.data.batching.OmniParallelBatch
    runtime_input_adapter:
      _target_: hyper_parallel.models.deepseek_v41.adapter.data.runtime.DeepseekV41Runtime
```

VLMTrainer 必须显式配置 `dataset.training_mode: sft` 或 `pretrain`。processor 和 transform 必须匹配模型，
额外 runtime adapter 及并行限制见 [DeepSeek-V4.1 指南](../../docs/guide/data/deepseek_v41_vlm_online_data_guide.zh.md)。

### Offline Indexed Text

```yaml
training:
  train_iters: 100

dataset:
  _target_: hyper_parallel.data.text.build_dataset.build_indexed_text_dataset
  data_path: /data/train_text_document
  model_assets:
    tokenizer:
      _target_: hyper_parallel.data.text.build_tokenizer.AutoTokenizer.from_pretrained
      pretrained_model_name_or_path: /models/tokenizer
      tokenizer_type: pretokenized
      vocab_size: 151936
      eod_token_id: 151645
  data_config:
    max_seq_len: 4096
    simple_blend: "no"
    low_level_dataset:
      _target_: hyper_parallel.data.offline.dataset.GPTDataset

dataloader:
  _target_: hyper_parallel.data.batching.FixedBatchDataLoader
  collate_fn:
    _target_: hyper_parallel.data.batching.build_indexed_collate_fn
  get_batch:
    _target_: hyper_parallel.data.batching.TextParallelBatch
    source_type: offline
```

`pretokenized` 不重新 tokenize；必须同时提供原语料对应的 `vocab_size/eod_token_id`，上面的值仅作示例，
不可照搬到其他 tokenizer。`pad_token_id` 可选，已有专用 PAD 时填其真实 ID。
`GPTDataset` 在运行时从文档组出长度 S 的样本；已按 S+1 token 预切 record 时改用 `GPTFromMRDataset`。
两者提供已 shift 的 labels，padding label 为 -100。Offline 长度由 `max_seq_len` 控制，
不使用 Online 的候选池 packing/padding 选项。`train_samples` 替代 `train_iters` 时，
总 optimizer steps 为 `train_samples // global_batch_size`。

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
