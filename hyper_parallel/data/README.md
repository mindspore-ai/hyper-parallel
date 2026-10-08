# HyperParallel 数据模块

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
├── online/     # 公共 Mapping/Iterable 加载、索引访问与 source blend
├── nv_meta/    # 已准备的 .nv-meta SQLite/tar 索引读取、解码与构建
├── text/       # LLM tokenizer、chat template、transform、Dataset 入口
├── omni/       # image/video/audio processor transform 生命周期、Omni Dataset 入口
├── batching/   # 候选池、packing、collator、DataLoader、get-batch
├── parallel/   # Mapping DP sampler、DataLoader 所有权、CP shard、TP broadcast
└── tools/      # Indexed 离线制作和检查工具
```

| 模块 | 输入 | 输出 | 不负责 |
| --- | --- | --- | --- |
| `online` | 文件、目录、glob、Hub Dataset、索引记录 | RawSample source | tokenizer、模型语义 |
| `nv_meta` | 已准备的 SQLite 元数据与 tar | 原始字节记录，或经 adapter 解码的字段 | tokenizer、模型语义 |
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
| `.nv-meta` Text/Omni | 支持原始记录和预处理张量 | 原有 builder + `data_config.format: nv_meta`，内置常用解码 |
| Omni TP/CP get-batch | 已实现，待多卡训练验证 | `OmniParallelBatch` 仅支持 PP=1 |

当前 Omni 已接入 image/VLM 字段；目录和 transform contract 可继续扩展 video、audio。

## 2.1 在现有入口选择 nv-meta 格式

Text/Omni 继续使用已有 Dataset builder、`data_path` 和 `data_config`。
只有显式设置 `data_config.format: nv_meta` 才进入 nv-meta 构建路径；省略 `format` 时，
沿用原来的文件/Hub 加载、过滤、混合与 transform 路径，不探测目录并自动切换格式。
现有训练示例不需要迁移。

这里新增的是一种数据存储格式的接入，不改变 Text/Omni 的模型处理层级。
`nv_meta` 提供索引读取和记录解码，`text`/`omni` 继续负责分词、模型处理及批次编码。
训练入口的支持范围如下；表中的函数均沿用已有入口名：

| 训练场景 | Dataset 的 `_target_` | 访问方式 |
| --- | --- | --- |
| 文本原始记录或预先分词的文本 | `hyper_parallel.data.text.build_dataset.build_online_text_mapping_dataset` | 单源或多源 Mapping |
| 文本逐条迭代 | `hyper_parallel.data.text.build_dataset.build_online_iterable_dataset` | 单源 Iterable |
| 原始图文或完整预处理多模态张量 | `hyper_parallel.data.omni.build_online_omni_mapping_dataset` | 单源或多源 Mapping |

本次不新增 Omni Iterable 训练入口，也不把 nv-meta 接到已有 `.idx/.bin` 的
`build_indexed_text_dataset` 入口。提前分词的 nv-meta 仍通过 Text 入口读取。

```yaml
dataset:
  _target_: hyper_parallel.data.omni.build_online_omni_mapping_dataset
  data_path: /datasets/webdataset
  data_config:
    format: nv_meta
  data_transform:
    _target_: hyper_parallel.data.omni.AutoProcessorTransform
    max_seq_len: 4096
```

这是合并到已有训练配置的 Dataset 片段，不是完整训练文件；保留模型的 `model_assets`、
DataLoader、优化器及并行配置。已有模型专用 transform 时继续使用它。
默认读取每个样本的 JSON 对象，并解析 messages 中指向本样本图片 part 的引用，无需配置 adapter。
其他常用布局只增加 `data_config.record_part`：`txt` 返回 `text`，`tokens.npy` 返回 `input_ids`，
`npz`/`pt` 返回保存的字段字典。它指定 tar 内的 part 名称，不是整个数据集的文件格式。
Text 使用原有 `hyper_parallel.data.text.build_dataset.build_online_text_mapping_dataset`；
流式 Text 使用 `build_online_iterable_dataset`；Omni 使用 `build_online_omni_mapping_dataset`。
访问方式由所选 builder 决定，不重复配置 mode。

| 配置位置 | 职责 |
| --- | --- |
| `dataset.data_path` | 单源路径，与 `data_config.sources` 互斥 |
| `dataset.data_config.format` | `nv_meta` 显式选择 prepared metadata；旧配置省略该项 |
| `dataset.data_config` | split、shuffle、reader、混合以及已有 HP ownership/cache 和 batch 选项 |
| `dataset.data_config.record_part` | 内置解码的记录 part，默认 `json` |
| `dataset.data_config.sample_adapter` | nv-meta 特殊样本布局的可选覆盖 |
| `dataset.data_transform` | Text/Omni 原有模型处理过程 |

不再配置 `source` 对象，也不提供 provider 注册或能力协议。nv-meta 不接受
`data_config.path`、`data_config.data_path` 等路径别名；Mapping 多源的每个条目各有一个
`data_path`，表示不同数据集。Text/Omni 入口沿用既有函数，从 Trainer 的 mesh 派生
`DataLoaderParallelContext`；Text Iterable 也可复用调用者已提供的 context。
`nv_meta.build_dataset.build_nv_meta_dataset` 只接收这份加载上下文，训练计划单独传递。
它在确定归属后才打开 reader，失败时关闭已打开的 reader。

内置 adapter 根据 `record_part` 创建，不扫描数据猜格式。`data_config.sample_adapter` 使用 HP
已有的嵌套 `_target_` 构建与序列化机制，替换默认解码；提供它后，外层 `record_part` 不参与解码选择。
adapter 不隐式接收 Trainer 参数；tokenizer 和模型编码仍由 `data_transform` 负责。

```text
现有 Text/Omni Dataset builder
  -> format 为 nv_meta：build_nv_meta_dataset -> nv-meta reader + 索引访问视图
  -> 未配置 format：原有文件/Hub 加载路径
  -> canonical RawSample -> 原有 Text/Omni 包装 -> packing / DataLoader
```

`nv_meta/build_dataset.py` 集中处理该格式的校验、构建归属及 reader 组装；
`online/source_views.py` 负责索引访问、适配、过滤、恢复及 metadata 负载分配。
Text/Omni 只调用具体的 nv-meta 构建函数，不处理 SQLite/tar；nv-meta 不依赖模型 transform。
Omni 保留 Mapping 包装类和 preencode/postencode/batch 生命周期。适配器输出可直接消费的
媒体对象或正确解析的媒体引用，原始媒体和预处理张量均通过现有入口接入。

## 2.2 已建立索引的 `.nv-meta` 数据集

`.nv-meta` 是准备阶段生成的 SQLite 样本索引。reader 按 `sample_parts` 的 byte offset 从 tar
读取所需字段，不扫描或整体解压 shard。`dataset.yaml`、`split.yaml`（或 `split.json`）和
`.info.json`（或 `.info.yaml`）只作为元数据读取，不执行其中的 `__module__`、`__class__`
或用户代码。安装可选格式依赖：`pip install PyYAML braceexpand`；远程 tar 另需 fsspec
及对应协议后端。支持 Energon 的 brace shard 选择器，例如 `shard-{00000..00999}.tar`，
以及 `val`/`valid`/`validation` split 别名；不存在的 split 不会退回读取全部数据。

常见目录布局如下。必须提前准备好索引；仅有 tar 文件或一个空的 `.nv-meta` 目录不能训练。
本 PR 读取已有索引，不提供生产数据的索引制作命令。

```text
/datasets/webdataset/
├── .nv-meta/
│   ├── dataset.yaml
│   ├── split.yaml       # 也可为 split.json
│   ├── .info.json       # 也可为 .info.yaml
│   └── index.sqlite     # 包含样本及各 part 的实际内容偏移
└── shard-00000.tar      # 未整体压缩的 tar
```

`dataset.data_path` 指向数据根目录或其中的 `.nv-meta` 目录，不填写 tar 的通配符。
SQLite 至少需要 `samples`、`sample_parts` 以及可解析的 shard 路径；每个 part 必须有
`content_byte_offset/content_byte_size` 或受支持的 `byte_offset/byte_size`。
不能仅凭格式版本号判断兼容性，也不支持用 gzip、bzip2、xz、zstd 整体压缩的 tar 做这些偏移读取。

例如 tar 成员 `sample_000001.tokens.npy` 的样本 key 是 `sample_000001`，part 名是 `tokens.npy`；
配置写 `record_part: tokens.npy`。`record_part` 不填写完整成员名，也不是自动搜索文件的规则。
同理，`sample_000001.json` 与 `sample_000001.jpg` 是同一个样本的两个 part。

nv-meta 支持 Mapping 和 Iterable。Mapping 保留全局索引，由 HP 原有 BatchSampler
分配 DP 样本；Iterable 按 DP × worker 直接步进到本 worker 的索引，不逐条遍历其他 worker 的位置。
有限 Iterable 只丢弃不足一个完整 DP 轮次的尾部，各 rank 再由 worker 分摊；worker 数不改变
每个 rank 应有的样本数。`repeat: true` 则循环使用索引；仍需保证各 rank 能产出有效训练批次。
reader 保持 prepared 索引顺序，shuffle、重复和访问模式只由公共访问视图处理，不建立全量
Python shuffle 索引。单源 Iterable 的 replay key
直接标识物理记录，恢复候选池时不按当前 epoch 或 worker 数重新推导样本。
游标 checkpoint 则要求 prepared 数据、配置、DP world size 和 worker 数量保持不变；
`persistent_workers` 在下次迭代开始时读取共享 epoch。不要在一个迭代器仍运行时切换 epoch。

Text/Omni Mapping 入口从 Trainer 的 mesh 派生数据加载归属，训练 YAML 无需配置并行上下文。
Iterable worker 只保存 DP rank/size，不持有父进程的通信组或同步回调；worker 编号在 worker 内获取。

常规训练只需选择格式并配置路径；样本布局不同才指定 `record_part`，拆分和访问策略按需选择：

| Dataset 配置项 | 默认值 | 说明 |
| --- | --- | --- |
| `data_path` | 无 | 数据根目录或 `.nv-meta` 目录，与 `data_config.sources` 互斥 |
| `data_config.record_part` | `json` | 内置解码的 part 名称，见第 2.3 节 |
| `data_config.split` | `train` | 选择已有的 `train`、`valid`、`test`；`val`/`validation` 等价于 `valid` |
| `data_config.exclude` | 无 | 排除完整 shard 或 `shard路径/样本key`；与元数据内的排除项合并 |
| `data_config.shuffle` | `false` | 控制单源索引打乱；Mapping 扩展目标长度及多源混合的行为见下文 |
| `data_config.repeat` | `false` | 单源 Iterable 循环读取 |
| `data_config.sources` | 无 | Mapping 多源列表，每项含 `data_path`；`weight` 默认 `1` |

以下为按需读取、缓存与大规模调度选项，通常保持默认值即可；nv-meta 只接受
`split`、`required_parts`，不再提供重复的 `split_name`、`parts` 配置名：

| Dataset 高级配置项 | 默认值 | 说明 |
| --- | --- | --- |
| `data_config.required_parts` | adapter 声明或所有 parts | part 名或列表，如 `txt`、`[json, img1.jpg]`；省略或 `null` 使用 adapter 声明 |
| `data_config.metadata_cache_size` | `4096` | 每进程、每个 reader 的 metadata LRU 条目上限；`0` 关闭该缓存 |
| `data_config.max_open_shards` | `64` | 每进程、每个 reader 的 tar/fsspec 文件句柄上限 |
| `data_config.cache_dir` | 用户缓存目录下 `hyper_parallel/nv_meta` | 只读 mmap 索引缓存；可配置节点本地 SSD |
| `data_config.cache_timeout` | `600` | 等待同一索引缓存写锁的秒数 |
| `data_config.read_buffer_size` | `8388608` | 同一样本相邻 parts 合并读取的窗口字节上限 |
| `data_config.read_balance` | 关闭 | 可选 metadata 读计划 |
| `data_config.balance_group_size` | `null` | 仅 Iterable：每个全局 worker 在均衡窗口中的样本数量；总窗口为该值 × DP 数 × worker 数 |
| `data_config.max_balance_group_size` | `262144` | Iterable 自动规划窗口上限 |
| `data_config.output_index_for_resume` | `false` | 单源 Iterable 的 output-index replay |
| `data_config.filter_samples` | `false` | 默认访问时校验并报错；开启后丢弃未通过校验的记录 |

`split` 选择一个已准备好的数据划分，不支持文件/Hub 路径的 `"98, 1, 1"` 比例配置。
当前 nv-meta builder 每次返回一个选定划分的数据集，不自动返回 train/valid/test 三元组；
现有 Trainer 会把单个返回值接到训练槽位。不要把 `split: valid` 理解为同时启用验证集。
单独检查验证数据可直接创建 `NvMetaDataset(..., split="valid")`。

随机种子复用 `training.seed`，缺省为 `42`。单源 Mapping 在目标样本数不超过源长度且
`shuffle: false` 时按索引顺序读取；目标数量超过源长度时会启用确定性的重复排列。
多源 Mapping 始终按种子打散全局配额及源内顺序，不承诺 `shuffle: false` 时逐源连续拼接。
这些计划中的“样本数”是原始记录数；一条文本记录经切分可产生多个训练样本，packing 后的批次数也会变化。

读取默认值只在 `NvMetaDataset` 定义，builder 仅转交显式配置。上述数值是有界的初始值，
不是针对所有存储和集群测得的最优值。普通训练无需填写整张表：节点缓存位置不同才调整
`cache_dir`；文件句柄受限或频繁切换 shard 时调整 `max_open_shards`；metadata 重复访问与
内存占用冲突时调整 `metadata_cache_size`；远程小请求较多时测量后调整 `read_buffer_size`。
`cache_timeout` 仅控制等待其他进程构建同一索引缓存的锁超时，不是网络读取超时。
混合多个源时，每个 reader 各有自己的缓存和句柄额度，需要同时考虑源数和 worker 数。

默认不额外打印读取参数。启用 HP 已有的数据调试日志后，每个 reader 构建完成时记录一次
实际生效的路径、split、parts、缓存目录和上述数值，包括未显式填写的默认值；不会逐样本打印。
DEBUG 日志默认只选择 global rank 0。训练 YAML 使用已有配置：

```yaml
debug:
  check_dataset: debug
```

首次构建将选定 split 的 SQLite 索引流式编译为只读缓存；之后的 rank/worker 只打开
manifest 并按需 mmap，同节点由操作系统共享文件页，不通过 pickle 复制全量索引。
媒体成本也保存在缓存中，读取成本不再为每个 worker 扫描整张 SQLite 表。
缓存指纹覆盖 metadata、SQLite 路径/大小/修改时间、split 和 part 选择；文件锁串行化同一
指纹的构建，完整文件写好后原子发布。prepared 数据和 tar 应在训练期间保持不变，
SQLite WAL 必须先完成 checkpoint；该缓存不是对每次 payload 读取重新校验内容的机制。

adapter 可以声明 `required_parts = ("json", "jpg")`；显式 reader 配置可覆盖这一声明。
reader 只返回选中的 parts，未声明的图片、视频、音频不会单独读取或解码；合并相邻读取可能包含少量间隙字节。
默认 JSON adapter 不限制 parts，会读取当前样本的全部 parts；不会自动根据 JSON 引用再按需读取媒体。
只需查看指定样本的文字和图片时，可直接按索引访问并明确选择 parts：

```python
from hyper_parallel.data.nv_meta import NvMetaDataset

dataset = NvMetaDataset("/data/prepared", required_parts=["txt", "img1.jpg"])
try:
    locations = dataset.index_metadata(123)  # 只查询该样本的位置，不读取媒体内容。
    sample = dataset[123]  # 只读取第 123 个样本选中的 parts。
finally:
    dataset.close()
```

原始记录直接返回 Mapping，包含 `parts`、`sample_key`、`__key__`、source info 和可选媒体 metadata；
`NvMetaDataset.index_metadata()` 可只读取位置与代价，返回的 `NvMetaSampleIndex` 使用统一的
`NvMetaPartLocation` 描述 byte range。
具备 `media_metadata` 表及所需字段的 SQLite 索引可提供媒体成本；缺少真实 part offset 时明确报错，
不推测 tar 布局。`index_metadata(123)` 中的 `123` 是所选 split/exclude 之后的逻辑编号，
不必等于 SQLite 原始 `sample_index`；返回值会同时给出这两个编号。

默认在样本访问时调用 transform 的 `is_valid_sample`，返回 `False` 时报错，不在构建时扫描 payload。
显式开启 `filter_samples` 后，Mapping 内容过滤需要读取候选记录建立有效索引；这不能描述为
零 payload 读取。过滤索引使用紧凑整数数组，过滤后的长训练重复索引使用有界/惰性
计划。未打乱的单源 Mapping 使用 range；大规模单源重复和 Iterable 默认计划不建立全长 Python
整数列表。适配与模型处理保持按需执行，构建阶段不额外增加一层逐样本格式分派。
`filter_samples` 只跳过判断函数返回 `False` 的记录，不吞掉解码异常、缺字段异常或文件读取错误。

nv-meta 多源配置仅支持 Mapping，位于 `data_config.sources`。子源只支持 `data_path`、
`weight`、`split`、`exclude`：路径必填，权重默认 `1` 且必须为正的有限数；`split` 和 `exclude`
省略时继承外层。解码、缓存和访问选项统一放在外层 `data_config`，不在子源中重新配置。
错误会指出具体 `sources` 条目，并列出支持的写法；不支持嵌套 `sources`。

```yaml
data_config:
  format: nv_meta
  record_part: txt
  sources:
    - data_path: /datasets/books
      weight: 3
    - data_path: /datasets/code
      weight: 1
      split: train
```

混合时省略 `read_balance` 或设为 `none`；需要 metadata 读成本均衡时使用单个 `dataset.data_path`。
Text Iterable 使用单个 `dataset.data_path`；多源 Iterable 配置在打开 reader 前报错。
nv-meta Mapping blend 先按权重确定精确整数配额，再惰性计算全局与 source-local 索引；
调度状态内存为 O(K)，K 为来源数，不随训练目标样本数增长。这是 nv-meta 的混合实现，
原有文件/Hub Online 混合算法、默认顺序及 checkpoint 格式保持不变。
选择按索引保存时，单源 Iterable 使用物理样本索引恢复候选池，不在候选池 checkpoint 中保存媒体 payload。
按索引恢复要求 adapter/transform 确定且处理配置不变；含随机增强时使用 `dataloader.save_by_idx: false`
保存完整候选样本。Mapping 恢复还要求使用相同的数据、种子、混合权重与采样配置。

训练时使用已有的 `dataloader.save_by_idx` 选择候选池保存方式：Mapping 默认 `true`，Iterable 默认
`false`。对于确定性文本 Iterable，可在原有 `dataloader` 下设置 `save_by_idx: true`；DataLoader
会自动启用输出索引，不需要再配置 `data_config.output_index_for_resume`。后者主要用于直接迭代
数据源：开启后输出 `(样本, 物理索引)`，`get_item(物理索引)` 可重读记录。
保存完整候选样本只避免恢复时重新生成这些候选，不额外保证所有随机增强的端到端确定性。

metadata 读计划可以按 `pixels`、`frames`、`duration`、`bytes` 或 `media_bytes` 调度。它适用于
单源且 `filter_samples: false` 的训练路径；访问时的校验不会改变索引分配。开启过滤时不支持
该计划。Mapping 仅在相应 global micro-batch 内重排，要求 `sampler_type: single`、
无 `data_rearrange_map`，并检查 DP/micro-batch 配置与 sampler 一致。
Mapping 和 Iterable 都按窗口惰性生成 metadata 计划，缓存当前窗口。窗口内使用最小堆维护
可分配的 rank/worker，保留按负载、slot 编号选择的顺序，避免为每条样本扫描所有 worker。
显式 `balance_group_size` 也受窗口上限约束；拓扑所需窗口超限时明确报错。
读成本均衡默认关闭。各 worker 会独立计算同一窗口，超大 worker 拓扑下存在重复规划开销；
启用前应测量规划耗时与消除数据倾斜的收益，metadata 成本不是模型计算耗时的精确预测。

需要时使用下面一种明确写法，合并到单源配置；不必同时填写任何别名：

```yaml
dataset:
  data_config:
    read_balance:
      metric: bytes
      strategy: greedy
dataloader:
  sampler_type: single
  data_rearrange_map: null
```

`bytes` 是所选 parts 的字节数；`sample` 是等成本计数。`media_bytes`、`pixels`、`frames`、
`duration` 需要媒体元数据。没有媒体元数据的 part 贡献为零；存在媒体记录但缺少 `media_bytes`
时使用该 part 的字节数，其余缺失指标按零计算。不能据此推断真实解码成本。
实现也接受 `read_balance: bytes`、`balance_by/balance_policy`、`by/policy` 和 `lpt` 别名；
`lpt` 与 `greedy` 使用同一算法。省略、`false` 或字符串 `none` 可关闭；空字典会启用默认
`sample/greedy`，`true` 不是受支持的开启方式。YAML 布尔值写 `true/false`，不要写成引号包裹的字符串。

payload tar 可以使用 fsspec URL，metadata 目录仍需本地可读。reader 合并同一样本中
相邻的已选 part 范围，受读取窗口上限约束；它不是整 shard 下载或跨样本预取。
合并读取可能包含少量间隙字节，单个超大 part 仍需完整加载，`read_buffer_size` 不是样本内存上限。
文件句柄和 metadata LRU 属于各进程，worker 重开 mmap/句柄，不继承可变文件游标。
远程存储仍需结合请求延迟、节点缓存和 worker 数测量实际吞吐。

直接处理解码后的样本时也可调用 `build_nv_meta_dataset(data_path=..., access_mode="mapping")`，
其中公共 `access_mode` 默认 Mapping，也支持 Iterable。这个便捷函数只构造适配后的原始来源，
默认解码 JSON，其他布局同样用 `data_config.record_part` 指定；直接读取原始字节使用 `NvMetaDataset`。
独立调用不传 `dataloader_context` 时本地构建；需要分布式归属时，传入已派生的加载上下文。
它不绑定模型 transform；训练配置使用上面的 Text/Omni 入口。packing、worker、prefetch、
`persistent_workers`、`pin_memory` 等仍属于已有 DataLoader 配置。

nv-meta 训练在每个 optimizer step 开始前，由 `SynchronizedBatchReader` 预读一个
完整梯度累积步，并用一次小型 collective 确认所有 rank 就绪。任一 rank 提前结束时，
所有 rank 在最短完整 step 处结束，舍弃无法组成完整 step 的尾部，避免一部分 rank
继续进入模型 collective。额外 Host 缓冲最多为一个梯度累积步的 collated batches；
设备传输沿用原有 runtime：Text 逐 micro-batch 传输，VLM 仍在 step 前准备全部 micro-batch
的设备输入。因此该协议解决结束一致性，不代表降低了 VLM 峰值显存。保存 checkpoint 必须在完整 step 边界。
其他格式的训练保留原流程。直接自定义训练循环时，也需要使用这个完整 step 协议。
nv-meta 的有限 epoch 耗尽后继续下一 epoch，直到完成配置的 optimizer steps；
无法提供任何完整 step 的数据会报错，避免空转。评估吞吐时需把 CPU 准备和就绪协调计入端到端耗时，
不能只比较 step callback 计时。

## 2.3 nv-meta 训练样本与配置

`online` 表示运行时的数据处理路径，不表示联网。本地 tar 同样可以在线 tokenize；
提前写好 token/张量的 tar 也使用同一个 reader。`.nv-meta` 的 prepare 步骤只建立索引，
不代表已经完成模型预处理。以下片段用于替换现有训练配置的对应字段，不改变模型、并行、
优化器及 model assets 配置。模型/分词器资产也应提前准备到本地，才能完全离线运行。

| 已保存内容 | `data_config.record_part` | 模型处理 |
| --- | --- | --- |
| `sample.json` 中的 `text`、`messages` 或训练字段 | 默认 `json`，可省略 | 对应 Text/Omni transform |
| `sample.txt` 中的 UTF-8 文本 | `txt` | 原有 `PlaintextTransform` |
| `sample.tokens.npy` 中的一维整数 token 文档 | `tokens.npy` | `PretokenizedTextTransform` |
| NPZ/PT 中的 `input_ids` 与对齐的 `labels` | `npz` / `pt` | `PretokenizedTextTransform` |
| NPZ/PT 中的完整多模态单样本张量 | `npz` / `pt` | Omni builder 配置 `preprocessed: true` |

上述约定共用一个内置 decoder；`record_part` 是准确的 part 名称，例如 `record.json`，
不会搜索任意 JSON 或猜测字段。支持的后缀为 `json`、`txt`/`text`、`npy`、`npz`、`pt`/`pth`；
其中 TXT/TEXT 返回 `text`，NPY 返回 `input_ids`，JSON/NPZ/PT/PTH 返回字段字典。
CSV、Parquet、图片或音视频不能直接作为内置 decoder 的 `record_part`；特殊布局需要自定义 adapter。
TXT/NPY/NPZ/PT 自动限制 reader 只读选中的 part。
JSON 仍可能引用可变媒体字段，因此默认保留当前样本的全部 parts。这里只解析 HP 样本约定，
不会执行 `.nv-meta/dataset.yaml` 声明的 Energon sample class、field-map 表达式或自定义代码。

解码配置只涉及下面这些键；表中的 `sample_adapter.*` 选项属于 `NvMetaSampleAdapter`，
其他自定义 adapter 使用自身的参数：

| 配置键（相对于 `dataset.data_config`） | 默认值 | 用法与边界 |
| --- | --- | --- |
| `record_part` | `json` | 默认 adapter 的快捷配置；填写准确的 part 名，不支持 `null` |
| `sample_adapter` | `null` | 省略或 `null` 使用默认 adapter；提供对象或嵌套 `_target_` 后，由它负责解码，忽略外层 `record_part` |
| `sample_adapter.record_part` | `json` | 支持上面的 part 后缀；`null` 关闭主记录解码，此时必须配置非空 `field_map` |
| `sample_adapter.field_map` | 空字典 | 输出字段名到 part 名的映射，如 `text: caption.txt`；映射结果覆盖主记录中的同名字段，不解析字段路径表达式 |
| `sample_adapter.image_mode` | `pil` | 可选 `pil` 或 `bytes`，用于样本中引用的图片；不负责视频/音频或模型专用预处理 |

`null` 不等于省略：只配置外层 `record_part: null` 会因默认 adapter 缺少记录和字段映射而报错。
需要完全按字段映射解码时，将 `record_part: null` 放在 `sample_adapter` 内，并配置 `field_map`。
reader 的 `required_parts` 仍独立控制读取范围，见第 2.2 节。

特殊字段重命名、多个 part 合并或模型专用媒体解码，才配置 `dataset.data_config.sample_adapter`。
例如将 `caption.txt` 作为文本字段，以下配置放在现有 `dataset` 下：

```yaml
data_config:
  format: nv_meta
  sample_adapter:
    _target_: hyper_parallel.data.nv_meta.NvMetaSampleAdapter
    record_part: null  # 不读主记录，字段全部由下面的映射提供
    field_map:
      text: caption.txt
```

adapter 在构建时创建一次；多源混合共用这份配置，不在各个 source 下重复配置。

**原始文本**：每个样本包含 UTF-8 `txt` part，tokenizer 沿用 TextTrainer 的 model assets。

```yaml
dataset:
  _target_: hyper_parallel.data.text.build_dataset.build_online_text_mapping_dataset
  data_path: /datasets/text
  data_config:
    format: nv_meta
    record_part: txt
  data_transform:
    _target_: hyper_parallel.data.text.text_transform.PlaintextTransform
    max_seq_len: 4096

dataloader:
  _target_: hyper_parallel.data.batching.TokenBatchLoader
  collate_fn:
    _target_: hyper_parallel.data.batching.build_online_text_collate_fn
  get_batch:
    _target_: hyper_parallel.data.batching.TextParallelBatch
    source_type: online
```

需要将长文档切分并组 batch 时，沿用 `TokenBatchLoader` 和 `build_online_text_collate_fn`。
固定样本 DataLoader 要求 transform 每次恰好输出一个样本。Iterable 只需换成原有
`build_online_iterable_dataset`；流式重复由 `data_config.repeat` 控制。
`max_seq_len: 4096` 是示例值，不是 nv-meta 的默认值；这里的 transform 要求显式提供正整数。
应沿用模型和注意力实现支持的长度。候选池默认 token 预算为
`training.micro_batch_size × max_seq_len`，独立覆盖位置是 `dataloader.token_budget`。
已有 Text 的 `labels_are_shifted`、注意力 mask 等设置仍然保留，不因换存储格式而改变。

**原始图文**：第 2.1 节的配置使用 `AutoProcessorTransform`，搭配原有 `OmniPackingLoader`、
`build_omni_collate_fn` 和模型 processor。对应一个样本的 `json` part 示例为：

```json
{"messages": [
  {"role": "user", "content": [
    {"type": "image", "image": "part:jpg"},
    {"type": "text", "text": "描述这张图片。"}
  ]},
  {"role": "assistant", "content": [{"type": "text", "text": "一只猫。"}]}
]}
```

processor 对文本进行分词，并按模型约定处理图片。通用 `AutoProcessorTransform` 要求对话以
assistant 回复结束，仅监督最后一条 assistant 回复，并屏蔽提示词及 processor 标识的多模态 token。
模型有自己的标签、图片位置或批次字段约定时，继续使用该模型的 transform。

tar 中同一样本还须包含 `jpg` part。`part:img1.jpg`、`part:img2.jpg` 可引用多张图。
adapter 只解码实际引用的图片；`required_parts` 可限制 reader 的 payload I/O。
内置 adapter 输出 processor 可直接接收的 PIL 图片对象；DeepSeek-V4.1 在 `data_config` 下配置
`sample_adapter: {_target_: hyper_parallel.data.nv_meta.NvMetaSampleAdapter, image_mode: bytes}`，
输出其原生 `data: bytes` 图片块，保留原模型 transform，不经 base64
转换或临时文件。样本内重复引用同一 part 只解码一次。外部相对媒体路径以数据根目录为基准。

**预先 tokenize 的文本**：保留 Text builder，选择 token part 和预处理 transform：

```yaml
data_path: /datasets/tokenized-text
data_config:
  format: nv_meta
  record_part: tokens.npy
data_transform:
  _target_: hyper_parallel.data.text.text_transform.PretokenizedTextTransform
  max_seq_len: 4096
```

仅有 `input_ids` 时，对完整 token 文档构造 `输入 = tokens[:-1]`、`标签 = tokens[1:]`，
再按长度切分；EOS 应已写入文档。
已保存 labels 时不再次移位或切分，要求符合 HP Text 的 next-token 对齐、loss mask 约定和长度上限。
此时可用 `data_config.record_part: npz` 或 `data_config.record_part: pt` 读取字段字典。可选二值 `loss_mask`
会转换为 labels 中的 `-100`，因此经过 HP packing 后仍生效；全无监督的记录不产生训练样本。
不要保存 `attention_mask`、`position_ids` 到此 Text 输入：HP 根据 packing 边界和 batch
配置重建它们，transform 会明确拒绝这些字段，避免悄悄丢失自定义语义。
不会调用 tokenizer；CPU long tensor 在无需掩码或 dtype 转换时直接复用。

**完整预处理的多模态样本**：在现有 Omni builder 配置中选择张量 part 并开启 `preprocessed`，
model assets 和模型的 `data_transform` 保持原配置：

```yaml
preprocessed: true
data_path: /datasets/preprocessed-omni
data_config:
  format: nv_meta
  record_part: pt
```

对应 DataLoader 保留 `OmniPackingLoader`，并显式设置 `min_buffered_samples: 1`，
减少在候选池中滞留已解码媒体张量的数量。该设置不会修改已有 DataLoader 默认值；
`max_seq_len` 限制 token 数，不限制图片像素、视频帧数或单样本媒体字节数。

每个 `pt` part 保存一个 CPU tensor 字典，包括 `input_ids`、`labels` 及模型所需媒体/位置字段；
也可以使用 `npz`。要求它等价于该模型在线 `encode_sample` 的输出，保留该模型自己的 label
对齐规则，不混用 Text 和 Omni 的移位约定。builder 跳过在线样本编码、保留 `encode_batch`，
所以 DeepSeek-V4.1 等模型的 packing 后字段转换仍执行。transform 校验通用 token 字段的
dtype、单样本 shape 及序列上限；媒体字段能否被模型消费仍取决于该模型的契约。
token/mask 字段要求一维，位置字段允许模型需要的前置轴，
但最后一维必须等于 token 长度；通用 packing/CP 不接受预存的 N×N attention mask。
可变长图文样本不会被自动截断。NPY/NPZ 禁用 pickle；PT 使用
`weights_only=True` 和 CPU 加载，可以保留 BF16 等张量 dtype。

只离线计算 packing 长度等元数据时，继续使用原有 `preencode_sample` / `postencode_sample`
契约，**不要**开启 `preprocessed`。完整预处理必须包含实际模型输入，不能只有长度信息。

图片解码由内置 adapter 提供。视频/音频可使用模型 processor 已支持的外部文件引用，或其支持的
已解码 NPY 数组；tar 中压缩视频/音频需要匹配该 processor 的业务 adapter。完整离线媒体特征
可随 PT/NPZ 字段直接接入。nv-meta 接入不会新增模型本身尚未支持的模态或并行方式。
CPU 回归覆盖真实 SQLite/tar、Text/Omni builder、packing、loss/backward 和恢复契约。
模型本身的模态支持及 Omni `PP=1` 限制仍然适用；大型模型的显存、加速器吞吐和多节点稳定性
需使用目标模型、数据分布和集群配置验收，不能从 CPU 测试推导。

### nv-meta 常见配置问题

| 现象 | 对照检查 |
| --- | --- |
| 缺少元数据或内容偏移 | 确认索引已准备完成，路径指向数据根目录；不能只放 tar 或伪造偏移 |
| 找不到某个 part | 对照 tar 内同一样本的准确 part 名；`required_parts` 必须包含解码器和消息实际引用的字段 |
| 配了 JSON 但样本是纯文本 | 使用 `record_part: txt`；JSON 主记录必须是对象，不是裸字符串或数组 |
| `record_part: null` 报错 | 默认快捷配置不支持它；完全按字段解码时，在 `sample_adapter` 内配置 `record_part: null` 和非空 `field_map` |
| 多源配置报错 | 只用于 Mapping；移除外层 `data_path`；子源只填路径、权重、划分和排除项 |
| 开启均衡后 sampler 报错 | Mapping 使用 `single`，不设置索引重排；不能同时使用多源混合或内容过滤 |
| 不能形成完整训练步 | 检查有效样本数量、packing、微批大小、DP 数及梯度累积；无效记录不是靠增大缓存解决 |
| checkpoint 恢复不匹配 | 使用相同数据、配置和并行拓扑；游标恢复还要求 worker 数不变，在完整优化器步边界保存 |

## 3. 两类 source

本节介绍仓库原有的文件/Hub 加载行为。nv-meta 虽复用 Mapping/Iterable 接口，但底层始终是有限
SQLite 索引，不调用 Hugging Face 流式加载；其 split、多源和恢复限制以第 2 节为准。

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
`AutoProcessorTransform` 调用 `apply_chat_template` 生成输入，并构造仅监督最后一条 assistant
回复的 labels；模型特有的媒体元数据和批次字段约定由模型 adapter 实现。

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
