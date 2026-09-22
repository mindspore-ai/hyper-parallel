# DeepSeek-V4.1 Online VLM 数据转换指南

本文说明 DeepSeek-V4.1 Online VLM 训练中，原始 JSONL 样本如何转换为模型输入，以及各关键接口的职责、
输入输出和约束。对应实现主要位于：

- `hyper_parallel/data/omni/omni_transform.py`
- `hyper_parallel/models/deepseek_v41/adapter/data/processor.py`
- `hyper_parallel/models/deepseek_v41/adapter/data/encoding.py`
- `hyper_parallel/models/deepseek_v41/adapter/data/image_processor.py`
- `hyper_parallel/models/deepseek_v41/adapter/data/transform_fn.py`

## 1. 支持范围

当前链路面向采用 OpenAI 风格 `messages` 的 Online JSONL 图片问答/SFT 数据。它支持文本消息以及
`image_url` 图片内容块，但不负责把任意 VeOmni、ShareGPT 或 Megatron-Energon Sample 自动归一化为
该格式。`OmniDataTransform` 的“Energon 风格”指 transform hook 生命周期，而不是输入数据格式兼容。

DeepSeek-V4.1 adapter 当前处理图片，不处理视频或音频。通用 `prepare_messages()` 虽然可以解析部分
`video`、`audio` 路径，但模型 adapter 没有对应的编码实现。

## 2. JSONL 输入契约

每行必须是一个 JSON object，并包含非空 `messages`。SFT 样本的最后一条消息必须是 assistant：

```json
{
  "id": "train_36",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "image_url",
          "image_url": {
            "url": "images/train/train_36.jpg"
          }
        },
        {
          "type": "text",
          "text": "Which base paper will be coated in-house?"
        }
      ]
    },
    {
      "role": "assistant",
      "content": "091"
    }
  ],
  "source": {
    "dataset": "example/docvqa",
    "split": "train",
    "row_index": 28
  }
}
```

`id` 和 `source` 是可保留的追踪元数据，但 `encode_sample()` 不读取它们。参与编码的是 `messages`。

推荐使用上例中的 `image_url.url` 形式。图片值可以是：

- 相对或绝对本地路径；
- `http://` 或 `https://` URL；
- base64 `data:` URL。

不要在文本中手写 `<｜deepseek_image｜>`。图片必须作为独立 content block 提供，adapter 会负责插入模型
占位符并校验占位符数量与图片数量一致。

### 2.1 数据源路由 FAQ

#### 接口：空 Omni 样本如何进入过滤流程？

**FAQ：上层应该调用哪个接口？**

DeepSeek-V4.1 Online VLM 使用：

```python
build_online_omni_mapping_dataset(
    data_config=data_config,
    data_path=data_path,
    transform=transform,
    training_config=training_config,
)
```

该入口负责校验 `transform`、从 `training_config.seed` 设置数据随机种子，并把过滤器传给通用 Mapping
source builder。核心调用关系是：

```python
sample_filter = transform.is_valid_sample
source_dataset = build_online_mapping_source(
    data_path=data_path,
    data_config=dataset_config,
    sample_filter=sample_filter,
)
```

`build_online_mapping_source()` 的完整数据源接口是：

```python
build_online_mapping_source(
    *,
    data_config,
    data_path=None,
    sample_filter=None,
    train_valid_test_num_samples=None,
)
```

其中 `sample_filter` 接收一条尚未 transform 的原始 Mapping record，返回该记录是否属于逻辑 Dataset；
`train_valid_test_num_samples` 可覆盖三个 split 的自然长度，但当前 Omni 顶层入口没有传入该参数。

**FAQ：`is_valid_sample()` 的契约是什么？**

Omni 默认实现是：

```python
@staticmethod
def is_valid_sample(sample: Mapping[str, Any]) -> bool:
    messages = sample.get("messages")
    if not isinstance(messages, list):
        raise ValueError("Omni sample messages must be a list")
    return bool(messages)
```

因此：

- `messages=[]` 返回 `False`，记录被自动丢弃；
- `messages` 缺失或不是 list 表示输入契约错误，直接报错；
- filter 必须稳定、确定，不得依赖随机数、DP rank、DataLoader worker 或访问次数；
- filter 只判断原始记录是否有效，不执行 tokenization、图片解码、随机增强或 packing。

只有满足该契约，所有进程才能为相同数据建立一致的长度、split 和整数 index。

**FAQ：接口最终返回什么？**

没有 split 时返回一个有限、支持 `len()` 和整数 `dataset[index]` 的 Mapping Dataset；配置 ratio split 或
预分割路径时返回：

```python
(train_dataset, valid_dataset, test_dataset)
```

未配置或最终为空的 split 使用 `None`。返回结果随后才会被 lazy Omni transform 包装，构建 source index
时不会执行 `encode_sample()`。

#### 通用：如何判断数据源路由？

**FAQ：Online Mapping 当前支持哪些本地文件格式？**

支持 `.jsonl`、`.json`、`.parquet`、`.csv` 和 `.arrow`：

| 扩展名 | Hugging Face loader 名称 | 可走原生 `_JsonlMappingSource` |
| --- | --- | ---: |
| `.jsonl` | `json` | 是 |
| `.json` | `json` | 否 |
| `.parquet` | `parquet` | 否 |
| `.csv` | `csv` | 否 |
| `.arrow` | `arrow` | 否 |

“原生 JSONL 条件不满足”只表示代码会尝试通用 Hugging Face loader，不表示任意文件类型都受支持。

整体路由规则是：

```text
本地 JSONL（无 split、ratio split 或预分割路径）
  -> 原生 byte-offset _JsonlMappingSource

JSON/Parquet/CSV/Arrow（无 split、ratio split 或预分割路径）
  -> Hugging Face load_dataset(streaming=False)

Hub Dataset ID
  -> Hugging Face load_dataset(streaming=False)

其他本地文件格式
  -> 报错
```

#### 场景一：本地 JSONL

**FAQ：什么配置会走原生 `_JsonlMappingSource`？**

本地 `.jsonl` 走原生 reader，包括无 split、ratio split 和 train/valid/test 预分割路径。最简单的无 split
配置是：

```yaml
dataset:
  _target_: hyper_parallel.data.omni.build_dataset.build_online_omni_mapping_dataset
  data_path: /data/docvqa/train.jsonl
  data_config: {}
```

原生 reader 启动时扫描文件并记录有效记录的 byte offset，后续通过整数索引按需读取。它不要求 Arrow
为嵌套字段推断统一 schema，适合 user `content` 是多模态 block 列表、assistant `content` 是字符串的
异构 `messages`。

`data_path` 也可以使用只包含 JSONL 的目录、glob 或有序路径列表。例如 glob：

```yaml
data_path: /data/docvqa/train-*.jsonl
data_config: {}
```

目录扫描不递归，只读取目录直属的 `*.jsonl`。路径列表按配置顺序解析，并要求每个匹配结果都是本地
`.jsonl` 文件。

**FAQ：空样本过滤后会不会造成索引空洞？**

不会。Omni builder 把 `transform.is_valid_sample` 传给 `_JsonlMappingSource`，原生 reader 在记录 byte
offset 前逐条调用它。`messages=[]` 的记录不会进入索引，保留的记录会重新形成从 0 开始的连续索引。

**FAQ：相对图片路径如何解析？**

原生 `_JsonlMappingSource` 会给记录附加内部字段 `__online_source_path__`。`prepare_messages()` 使用它将
相对媒体路径解析到 JSONL 文件所在目录。例如：

```text
/data/docvqa/train.jsonl
/data/docvqa/images/train/train_36.jpg
```

样本中的 `images/train/train_36.jpg` 会解析为：

```text
/data/docvqa/images/train/train_36.jpg
```

#### 场景二：ratio split 或预分割的本地数据

**FAQ：配置 ratio split 后还会走 `_JsonlMappingSource` 吗？**

会。本地 `.jsonl` 配置 `data_config.split` 后仍走原生 reader：

```yaml
data_path: /data/docvqa/all.jsonl
data_config:
  split: "98, 1, 1"
```

原生 reader 只扫描和过滤一次全部 JSONL，得到连续的有效 byte-offset 索引，再按有效样本数确定性切分
train、valid、test。三个 split 共享同一份 offset 表，只保存各自的索引区间，不会重复扫描文件：

```text
JSONL -> is_valid_sample -> valid offsets -> ratio split views
```

因此空 Omni 样本既不占 Mapping index，也不占 split 比例名额。

**FAQ：ratio split 如何计算边界，是否会随机打乱？**

ratio split 在过滤后的有效索引上做连续区间切分。边界使用与 Hugging Face percentage slicing 一致的
closest rounding：

```python
train_end = round(train_percentage * valid_count / 100)
valid_end = round((train_percentage + valid_percentage) * valid_count / 100)
```

例如 10 条原始记录中有 2 条空样本：

```text
原始 id： 0, 1(empty), 2, 3, 4, 5(empty), 6, 7, 8, 9
有效 id： 0, 2, 3, 4, 6, 7, 8, 9
split：   50, 25, 25

train：   0, 2, 3, 4
valid：   6, 7
test：    8, 9
```

split membership 本身不使用随机 shuffle，避免在补齐空样本过滤时额外改变既有 percentage slicing 语义。
split 构建完成后的 source-local shuffle 和 BatchSampler shuffle 由后续阶段负责。

**FAQ：三个 ratio split 是否复制索引，索引会保存到磁盘吗？**

不会复制。当前最小实现把过滤后的 `(JSONL 文件路径, byte offset)` 表保存在 Dataset 对象中，三个 split
仅持有同一张表上的 `[begin, end)` view，因此文件只扫描一次、filter 每条原始记录只执行一次。

该索引只在当前 Dataset 生命周期内持久存在，尚未生成跨运行共享的 sidecar index。每个独立启动并调用
builder 的训练进程仍会各自扫描一次；DataLoader worker 不会重新建索引，只会使用自己的文件句柄按 offset
读取记录。需要跨 rank/跨运行复用时，应另行增加带数据 fingerprint 和 filter 版本的共享索引缓存。

**FAQ：已经分别保存 train、valid、test 时如何配置？**

使用预分割路径 Mapping。例如 JSONL：

```yaml
data_path:
  train: /data/docvqa/train.jsonl
  valid: /data/docvqa/valid.jsonl
  test: /data/docvqa/test.jsonl
data_config: {}
```

单个 split 也可以是目录、glob 或有序路径列表：

```yaml
data_path:
  train:
    - /data/docvqa/train-000.jsonl
    - /data/docvqa/train-001.jsonl
  valid: /data/docvqa/valid-*.jsonl
data_config: {}
```

本地 JSONL 会保留用户给定的 split 归属，然后分别在 train、valid、test 内过滤并建立原生索引，不会把
过滤后缺少的 valid/test 样本从 train 中补齐。

Parquet 等其他格式使用相同的预分割 Mapping：

```yaml
data_path:
  train: /data/docvqa/train.parquet
  valid: /data/docvqa/valid.parquet
  test: /data/docvqa/test.parquet
data_config: {}
```

split 名称只能是 `train`、`valid`、`test`，并且各 split 必须使用同一种格式。纯本地 JSONL 走原生
`_JsonlMappingSource`；JSON、Parquet、CSV 和 Arrow 走 Hugging Face loader。

**FAQ：Hugging Face 路径如何过滤空样本，有哪些 VLM 限制？**

预分割路径先通过 `load_dataset()` 构造各 split Dataset，再分别调用：

```python
dataset.filter(transform.is_valid_sample)
```

未分割数据配置 ratio 时，Hugging Face 路径先过滤完整 Dataset，再对有效样本执行确定性的连续区间切分，
与原生 JSONL 保持相同的 filter-before-ratio 语义。

对于同一字段在不同记录或嵌套位置具有不同类型的数据，Arrow schema 推断仍可能在 filter 前失败。本地
异构 Omni JSONL 应使用原生 `.jsonl` 路径；其他格式应预先保证 schema 稳定。

Hugging Face 路径也不会自动附加 `__online_source_path__`。图片等媒体应使用绝对路径，或者由数据集显式
提供正确的 `__online_source_path__`；`source.dataset` 等业务元数据不参与路径解析。

#### 场景三：Hub Dataset ID

**FAQ：Hub Dataset ID 如何配置，走哪个 reader？**

非本地的 `组织名/数据集名` 走 Hugging Face loader：

```yaml
data_path: example-org/docvqa-sft
data_config: {}
```

如果 Hub Dataset 有多个配置，可以通过 `data_config.config_name` 指定。Hub 数据仍须满足本指南的
OpenAI 风格非空 `messages` 契约；能够被 Hugging Face 加载不代表能够被 DeepSeek-V4.1 transform 编码。

#### 场景四：不支持的本地格式

**FAQ：哪些本地格式会直接报错？**

例如以下 `.txt` 不会自动按纯文本逐行读取：

```yaml
data_path: /data/docvqa/train.txt
data_config: {}
```

`.txt`、`.yaml`、`.xlsx`、`.tar`、图片目录等不在当前 Mapping source 的本地格式表中，会报告格式不支持。
需要先转换为 JSONL/JSON/Parquet/CSV/Arrow，或扩展对应的数据源 reader。

### 2.2 多 source 的 split、blend、shuffle 与 DP shard FAQ

#### 主题一：每个 source 如何确定 train、valid、test？

**FAQ：一个 source 可能返回什么结构？**

每个 source 加载后返回以下两种结构之一：

```text
Dataset
```

表示 source 没有独立 split，整个 Dataset 只贡献给 train；或者：

```text
(train_dataset, valid_dataset, test_dataset)
```

表示 source 已通过 ratio split 或预分割路径确定三个 split。缺失的 split 使用 `None`，例如：

```text
(source_train, None, source_test)
```

**FAQ：多个 source 如何同时配置 ratio split 和预分割路径？**

使用 `data_config.sources`，每个 leaf source 独立确定自己的路由、filter 和 split：

```yaml
data_path: null
data_config:
  random_seed: 42
  sources:
    - data_path: /data/source_a/all-*.jsonl
      weight: 0.7
      split: "98, 1, 1"
    - data_path:
        train: /data/source_b/train.jsonl
        valid: /data/source_b/valid.jsonl
      weight: 0.3
```

source A 先过滤再按 ratio 生成 `(A_train, A_valid, A_test)`；source B 保留预分割归属，并在各 split 内
过滤。配置 `sources` 时不能再同时传顶层 `data_path`。

**FAQ：有的 source 已分割、有的没有分割时如何组合？**

假设有三个 source：

| Source | 权重 | 加载结果 | 各 Dataset 长度 |
| --- | ---: | --- | --- |
| A | 0.6 | `(A_train, A_valid, A_test)` | `100 / 20 / 10` |
| B | 0.3 | `B_dataset` | `50` |
| C | 0.1 | `(C_train, None, C_test)` | `40 / 0 / 5` |

未分割的 B 只进入 train；C 不提供 valid。因此按同名 split 收集后的结果是：

```text
train = [A_train, B_dataset, C_train]
valid = [A_valid]
test  = [A_test, C_test]
```

#### 主题二：同名 split 如何做 global blend？

**FAQ：train、valid、test 会混在一起 blend 吗？**

不会。每个 split 独立构造一个 DP-global `_OnlineBlendedMappingDataset`：

| Split | 参与的 Dataset | 原始权重 | 默认 target size |
| --- | --- | --- | ---: |
| train | `A_train, B_dataset, C_train` | `0.6, 0.3, 0.1` | `100 + 50 + 40 = 190` |
| valid | `A_valid` | `0.6`，归一化后为 `1.0` | `20` |
| test | `A_test, C_test` | `0.6, 0.1`，归一化后为 `6/7, 1/7` | `10 + 5 = 15` |

blend 使用确定性的 largest-deficit 调度，为每个全局逻辑位置记录：

```text
(source_id, source_logical_index)
```

因此 global blend 表示“所有 DP rank 看到同一份逻辑 source 调度表”，不是把 train、valid、test 合并成
一个 Dataset。

**FAQ：配置固定目标样本数后如何处理？**

如果底层 `build_online_mapping_source()` 收到以下 `train_valid_test_num_samples`：

```python
(1000, 100, 50)
```

三个 blended Dataset 的长度分别固定为 `1000 / 100 / 50`，不再使用自然长度之和。目标数大于实际数据量
时，source-local 索引会重复多个 epoch 后再做确定性 shuffle；某个 split 没有可用 Dataset 或目标数为 0
时，该 split 返回 `None`。当前 `build_online_omni_mapping_dataset()` 没有向底层传递该参数，因此 Omni
顶层入口默认使用各 split 的自然长度之和。

#### 主题三：shuffle 和 DP shard 的真实顺序是什么？

**FAQ：blend 后是否只有 train 做 global shuffle？**

不是无条件如此。当前 Mapping source 构建阶段会为每个已构造的 split 生成 source-local deterministic
shuffle/repeat；是否在 DataLoader 阶段额外执行 global shuffle，取决于公共 BatchSampler 配置：

```text
每个 source 确定 train/valid/test
  -> 同名 split 内构造 DP-global blend index
  -> 每个 source 的 logical index 映射到 deterministic shuffle/repeat 后的物理样本
  -> 公共 BatchSampler 按配置生成 rank-local DP indices
  -> lazy transform / packing / collate
```

BatchSampler 的三种行为是：

| 配置 | shuffle 与 DP shard 顺序 |
| --- | --- |
| `sampler_type: single` | 按 global logical index 顺序分块，再给每个 DP rank 连续的 micro-batch；不额外 global shuffle |
| `sampler_type: cyclic`、`data_sharding: false` | 先 shuffle 完整 global index 区域，再按 DP rank stride shard |
| `sampler_type: cyclic`、`data_sharding: true` | 先给每个 DP rank 一个连续 bucket，再在 rank-local bucket 内 shuffle |

当前 DeepSeek-V4.1 Online VLM baseline 使用 `sampler_type: single`，因此没有 BatchSampler 层的额外
global shuffle。如果训练阶段需要严格的“global shuffle -> DP shard”，配置为：

```yaml
dataloader:
  sampler_type: cyclic
  data_sharding: false
```

这样所有 DP rank 使用同一个 `seed + epoch` global permutation，再各自取得不重叠的 rank-local indices。

## 3. 处理流程

数据源构建和模型 transform 是两个阶段：

```text
data_path / data_config.sources
  -> 每个 leaf source 选择原生 JSONL 或 Hugging Face loader
  -> 原始样本 is_valid_sample
  -> ratio split 或 preserve pre-split
  -> 同名 split 内 global blend
  -> source-local shuffle/repeat + BatchSampler + DP shard
  -> RawSample
```

取得一条 RawSample 后，才进入惰性的模型编码：

```text
JSONL RawSample
  -> prepare_messages
  -> chat_template(messages)
  -> image_processor(prompt, images)
  -> _get_assistant_start
  -> _build_labels
  -> _build_image_fields
  -> packing / collate
  -> encode_batch
  -> model batch
```

Online Mapping transform 是惰性的：构造 Dataset 时不会立即调用 `encode_sample()`；只有 DataLoader 实际
索引样本时才会执行编码。训练流程还会先构建并切分模型，因此 transform 调试日志通常在模型初始化完成、
开始读取第一个 batch 后才出现。

## 4. 关键接口

| 接口 | 输入 | 输出 | 主要职责 |
| --- | --- | --- | --- |
| `build_online_omni_mapping_dataset()` | `data_path`、`data_config`、Omni transform | Mapping Dataset 或 split tuple | 注入随机种子和 `transform.is_valid_sample`，再包装 lazy transform |
| `build_online_mapping_source()` | source 配置、可选 `sample_filter`、可选目标长度 | 有限整数索引 source 或 split tuple | 路由 reader、过滤、split，并构造 split-local blend |
| `OmniDataTransform.is_valid_sample()` | 原始 Mapping record | `bool` | 丢弃空 `messages`，拒绝错误的 `messages` 类型 |
| `_JsonlMappingSource`（内部） | 本地 JSONL 文件和 filter | byte-offset Mapping source | 保留异构 JSON，提供连续索引、split view 和源文件上下文 |
| `build_deepseek_v41_processor()` | 模型目录 | `DeepseekV41Processor` | 加载 tokenizer 和视觉配置，绑定模板与图片处理函数 |
| `OmniDataTransform.prepare_messages()` | RawSample | 深拷贝后的 messages | 校验 `messages`，解析相对媒体路径 |
| `encode_messages()` / `chat_template` | messages | prompt，可选 media records | 应用 DeepSeek-V4.1 对话协议，插入图片占位符 |
| `prepare_vl_inputs()` / `image_processor` | prompt、图片 records | token IDs、token types、`ImageInput` | 读取图片、生成 patch，并展开图片占位符 |
| `DeepseekV41OmniTransform.encode_sample()` | RawSample | model sample | 组织完整单样本编码流程并执行长度检查 |
| `_get_assistant_start()` | messages、`ImageInput` | token 下标 | 找到最后一轮 assistant 的监督起点 |
| `_build_labels()` | input IDs、token types、起点 | labels | 屏蔽上下文和所有视觉 token |
| `_build_image_fields()` | `ImageInput` 列表 | packing-ready 图片字段 | 展平 patch 并生成图片网格、起点元数据 |
| `encode_batch()` | collated/packed batch | 模型 batch | 生成 patch offsets 和 image-to-batch 映射 |

### 4.1 `build_deepseek_v41_processor()`

processor 从模型目录的 `inference/config.json` 读取以下视觉参数：

- ViT patch size；
- 最小像素数和最大宽高比；
- aligner downsample ratio；
- 单图最大视觉 token 数；
- 图片占位 token ID。

processor 同时暴露两个模型原生入口：

```python
processor.chat_template
processor.image_processor
```

前者绑定 `encode_messages()`，后者绑定 `prepare_vl_inputs()`。

### 4.2 `prepare_messages()`

该接口只做浅层输入准备：

1. 要求 `sample["messages"]` 是非空 list；
2. 深拷贝 messages，避免修改 source record；
3. 基于 JSONL 文件目录解析顶层 content block 中的媒体路径。

它不是通用多模态 schema 转换器，不会把 `conversations + images`、Energon dataclass 或任意自定义字段
转换为 `messages`。

### 4.3 第一次 `chat_template()`

`encode_sample()` 第一次传入完整 messages，获得完整 prompt 和按出现顺序排列的图片 records。以 ChartQA
样本为例：

```text
<｜begin▁of▁sentence｜><｜User｜><｜deepseek_image｜>

How many hours were watched on Twitch in May 2021?<｜Assistant｜></think>91.9<｜end▁of▁sentence｜>
```

同时返回：

```python
media = {
    "images": [
        {
            "type": "image",
            "url": "/data/chartqa/images/train/example.jpg",
        }
    ]
}
```

默认 `thinking_mode="chat"`，因此 assistant header 后面是 `</think>`。具体 reasoning 处理还受
`thinking_mode` 和 `drop_thinking` 共同控制。

### 4.4 `image_processor()`

模板阶段的一张图片只占一个 `<｜deepseek_image｜>` token。图片处理阶段会读取图片、调整尺寸、生成
ViT patches，并把该占位符展开为：

```text
IMAGE_START
(IMAGE * n_llm_w + IMAGE_NEW_LINE) * n_llm_h
IMAGE_END
```

单张图片的 LLM token 数是：

```text
n_llm_h * (n_llm_w + 1) + 2
```

所有视觉位置在 `input_ids` 中都使用 `image_token_id`，通过 `token_types` 区分其语义：

| `token_types` 值 | 含义 |
| ---: | --- |
| `-1` | `TEXT`，普通文本或文本侧特殊 token |
| `0` | `IMAGE_START` |
| `1` | `IMAGE` |
| `2` | `IMAGE_NEW_LINE` |
| `3` | `IMAGE_END` |

`ImageInput.start` 记录图片 span 在展开后 `input_ids` 中的起始位置。

### 4.5 第二次 `chat_template()` 与 `_get_assistant_start()`

第二次模板调用不是重复生成完整输入，而是只渲染：

```python
messages[:-1]
```

由于样本最后一条消息必须是 assistant，去掉它后，模板会保留等待生成答案的 assistant header：

```text
<｜begin▁of▁sentence｜><｜User｜><｜deepseek_image｜>

How many hours were watched on Twitch in May 2021?<｜Assistant｜></think>
```

该前缀 token 数确定答案开始位置。图片在完整 `input_ids` 中已经从一个占位符展开为多个视觉 token，
所以监督起点还需要修正：

```text
assistant_start
  = len(tokenizer.encode(prefix_prompt))
  + sum(image_input.types.numel() - 1 for prefix images)
```

减一是因为原来的单个图片占位 token 已经包含在前缀长度中。图片越大，视觉 token 越多，最终答案的
实际下标越靠后。

这个计算依赖两个顺序不变量：

- `media["images"]` 的顺序与 prompt 中图片占位符顺序一致；
- `image_inputs` 的顺序与图片占位符顺序一致。

### 4.6 `_build_labels()`

labels 首先复制完整 `input_ids`：

```python
labels = input_ids.clone()
```

然后应用两层 mask：

```python
labels[:assistant_start] = IGNORE_INDEX
labels[token_types != TEXT] = IGNORE_INDEX
```

第一层屏蔽最终 assistant 回答之前的 system、user、历史 assistant 和当前 assistant header。第二层屏蔽
所有视觉位置。最终监督范围是：

```text
position >= assistant_start and token_types == TEXT
```

对于前述 ChartQA 样本，监督关系是：

| 内容 | 是否参与 loss |
| --- | ---: |
| 图片视觉 token | 否 |
| 用户问题 | 否 |
| `<｜Assistant｜></think>` | 否 |
| `91.9` | 是 |
| `<｜end▁of▁sentence｜>` | 是 |

这里的 `TEXT` 实际表示“非视觉 token”。因此答案后的 EOS，以及 thinking 相关的文本侧特殊 token，也会按
它们是否位于 `assistant_start` 之后决定是否参与 loss。

## 5. 单样本输出字段

无图片样本只返回前三个 token 字段；有图片样本还会返回图片字段：

| 字段 | 典型形状 | 含义 |
| --- | --- | --- |
| `input_ids` | `[seq_len]` | 图片已展开后的完整 token 序列 |
| `labels` | `[seq_len]` | 只保留最终 assistant 文本监督的 labels |
| `token_types` | `[seq_len]` | 文本和四种图片位置类型 |
| `pixel_values` | `[total_patches, 3, patch_h, patch_w]` | 所有图片的 ViT patches |
| `image_patch_lengths` | `[num_images]` | 每张图片的 patch 数 |
| `image_vit_grid_hw` | `[num_images, 2]` | 每张图片的 ViT 网格高宽 |
| `image_llm_grid_hw` | `[num_images, 2]` | 每张图片的 LLM 网格高宽 |
| `image_token_starts` | `[num_images]` | 图片 span 在样本 token 序列中的起点 |

如果展开后的 `input_ids` 长度超过 `max_seq_len`，transform 会直接报错，不会静默截断样本。

## 6. Packing、Collate 与 `encode_batch()`

`SamplePacker` 会沿 token 维拼接 `input_ids`、`labels` 和 `token_types`，沿 patch 维拼接图片数据，并把
每个样本局部的 `image_token_starts` 加上 packed sequence offset。`cu_seq_lens` 保存样本边界，用于隔离
packed attention。

`encode_batch()` 再把通用图片元数据转换成 DeepSeek-V4.1 forward 所需字段：

```python
image_patch_offsets = [0, cumulative_patch_count...]
image_batch_indices = [image_0_batch, image_1_batch, ...]
```

同时将图片网格字段整理为二维矩阵，将 `image_token_starts` 整理为一维向量。若 batch 中没有
`pixel_values`，该接口原样返回 batch。

## 7. 调试日志

建议在确认编码逻辑时记录以下边界，而不是打印 tensor 全量内容：

- `prepare_messages()` 后的媒体绝对路径；
- 完整 prompt 和图片 record 数量；
- `input_ids.numel()`、图片展开长度和 `assistant_start`；
- `(labels != IGNORE_INDEX).sum()`；
- 单图 patch 数、ViT 网格和 LLM 网格。

Online transform 是惰性执行的，所以日志不会在 Dataset 构造时出现。当前训练配置的根日志级别为
`INFO`；普通 `logger.debug()` 需要对应 logger 显式启用 DEBUG。`debug.check_dataset: debug` 配置的是
`hyper_parallel.data.*` dataset logger，不会自动提升 `hyper_parallel.models.*` logger 的级别。

多卡运行时还应避免每个 rank 打印完整 messages 和 prompt。必要时只允许 rank 0 输出，或只采样少量
record，否则日志量会随数据量和 rank 数快速放大。

## 8. 常见错误

### 最后一条不是 assistant

```text
DeepSeek-V4.1 SFT sample must end with an assistant message
```

当前 label 策略只监督最后一轮 assistant，因此训练样本必须以 assistant 结束。

### 图片占位符数与图片数不一致

```text
Found N image tokens but got M images
```

检查每个图片 content block 是否被正常解析，不要在普通文本中手写图片特殊 token。

### 图片路径找不到

确认样本由本地 JSONL Online source 读取，或者手工调用时使用绝对路径。业务 `source` 字段不会改变路径
解析基准。

### 样本超过最大长度

图片 token 数会随图片网格变化。长度检查发生在图片占位符展开之后，因此只检查原始问题文本长度不能
代表最终序列长度。

### DEBUG 日志没有输出

先检查 logger 的有效级别和命名空间：

```python
logger.isEnabledFor(logging.DEBUG)
logger.getEffectiveLevel()
```

如果 logger 位于 `hyper_parallel.models.*`，仅设置 `debug.check_dataset: debug` 不会启用它。
