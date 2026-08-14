# Dataset 架构

## 一、固定构建顺序

LLM 与 Omni Trainer 始终按照以下顺序构建数据流水线：

```python
# LLM 与 Omni 私有阶段
self._build_tokenizer()
self._build_model_assets()
self._build_data_transform()

# LLM 与 Omni 公共阶段
self.base._build_dataset()
self.base._build_collate_fn()
self.base._build_dataloader()
self.base._build_get_batch()
```

前三个阶段依赖具体模态，由 LLM 和 Omni 分别实现。后四个阶段共享构建入口，其中 Dataset、Collator、
DataLoader 属于静态数据流水线，GetBatch 通过公共执行骨架组装不同的 LLM/Omni BatchProcessor。

`_build_model_assets()` 保持为 Trainer 实现；datasets 目录只提供它实际使用的
`llm/chat_template.py` 和 `omni/processors/`，不再增加无实际逻辑的转发函数。

`hyper_models/components/data/` 是待废弃的旧目录。新数据体系不得依赖、修改或向该目录增加实现。

## 二、目录结构

```text
hyper_models/components/datasets/
│
├── README.md
├── __init__.py
├── contracts.py
├── collator.py
├── build_dataset.py
├── build_collate_fn.py
├── build_dataloader.py
├── batch.py
├── batch_adapter.py
│
├── parallel/
│   ├── batch_context.py
│   ├── dataset_context.py
│   ├── batch_sampler.py
│   ├── batch_transport.py
│   ├── cp_sharder.py
│   └── pipeline_router.py
│
├── llm/
│   ├── __init__.py
│   ├── build_tokenizer.py
│   ├── build_data_transform.py
│   ├── chat_template.py
│   ├── dataset.py
│   ├── online_dataset.py
│   ├── online_mapping_dataset.py
│   ├── online_iterable_dataset.py
│   ├── online_utils.py
│   ├── transform_dataset.py
│   ├── indexed_data_config.py
│   ├── build_indexed_dataset.py
│   ├── indexed_split_builder.py
│   ├── indexed_blended_dataset.py
│   ├── indexed_simple_blended_dataset.py
│   ├── indexed_lazy_dataset.py
│   ├── indexed_data_reader.py
│   ├── indexed_sample_index.py
│   ├── indexed_pretrain_dataset.py
│   ├── collator.py
│   └── get_batch.py
│
└── omni/
    ├── __init__.py
    ├── build_tokenizer.py
    ├── build_data_transform.py
    ├── dataset.py
    ├── online_dataset.py
    ├── offline_dataset.py
    ├── collator.py
    ├── get_batch.py
    └── processors/
        ├── __init__.py
        ├── image.py
        ├── video.py
        └── audio.py
```

## 三、公共阶段

### 3.1 `build_dataset.py`

公共 `_build_dataset()` 调用入口。它接收 Trainer 已经构建好的 `data_transform`，再调用配置中的
LLM 或 Omni Dataset 实现。

```text
BaseTrainer._build_dataset()
    │
    ▼
datasets/build_dataset.py::build_dataset()
    │
    ├── llm/dataset.py
    └── omni/dataset.py
```

公共层不解析 `.idx/.bin`、JSONL 或多模态文件。

### 3.2 `build_collate_fn.py`

公共 `_build_collate_fn()` 调用入口，同时提供 micro-batch 数量计算与 `MakeMicroBatchCollator`。

```text
BaseTrainer._build_collate_fn()
    │
    ▼
datasets/build_collate_fn.py::build_collate_fn()
    │
    ├── llm/collator.py
    └── omni/collator.py
```

### 3.3 `build_dataloader.py`

公共 `_build_dataloader()` 调用入口。LLM 与 Omni 使用同一个 DataLoader、Sampler 和分布式采样策略。

### 3.4 Runtime GetBatch

```text
RuntimeBatchAdapter
├── DistributedBatchTransport       # 公共：TP0 读取、TP 广播、设备搬运
├── ContextParallelBatchSharder     # Dataset：只按 cp_rank/cp_size 连续切分
├── LLMBatchProcessor               # LLM：tokens/input_ids、labels、loss_mask
├── OmniBatchProcessor              # Omni：文本和图像/视频/音频字段
└── PipelineBatchRouter             # PP 字段级路由接口，当前未实现
```

`LLMBatchProcessor` 与 `OmniBatchProcessor` 是平级实现，不互相继承。当前支持 TP 传输和文本 CP 切分；
当 `pp_shared_data=True` 且启用 PP 时，必须提供真正的 `PipelineBatchRouter`，否则明确抛出
`NotImplementedError`。不使用全 batch PP 广播作为临时实现。

Dataset CP 不依赖模型 Attention、K/V 通信或 `components/distributed/cp_utils.py`。它只负责连续切分
`input_ids/labels/loss_mask/position_ids` 等 Dataset 字段；`attention_mask` 保持完整，由模型运行时决定如何使用。
全局序列长度必须能被 `cp_size` 整除，Dataset CP 不猜测模型侧的 padding 和 mask 规则。

## 四、LLM 私有阶段

```text
LLMTrainer
│
├── _build_tokenizer()
│   └── llm/build_tokenizer.py
│
├── _build_model_assets()
│   └── llm/chat_template.py
│
└── _build_data_transform()
    └── llm/build_data_transform.py
        ├── PlaintextTransform
        ├── ConversationTransform
        └── PretokenizedTransform
```

LLM Dataset 实现：

```text
llm/dataset.py
├── online  → llm/online_dataset.py        # JSON/JSONL → RawSample
├── offline → llm/build_indexed_dataset.py # .idx/.bin → RawSample
└── common  → llm/transform_dataset.py     # RawSample → data_transform → ModelSample
```

Online 已实现 VeOmni 风格的 mapping/iterable 主线：支持本地 JSON/JSONL/Parquet/CSV/Arrow 和显式的
Hugging Face Dataset，iterable 模式支持 shuffle buffer、DP rank 切分，并由 Hugging Face Dataset 与
StatefulDataLoader 保存流式断点状态。Online 和 Offline 共用 `transform_dataset.py`：Online 使用
`PlaintextTransform`/`ConversationTransform` 完成 tokenizer 或 chat template 编码；Offline 使用
`PretokenizedTransform` 归一化 `.idx/.bin` 已生成的 tokens、labels、mask 和 position IDs。读取层不包含
tokenizer、模板或字段加工逻辑。一条源记录展开多个模型样本的 packing 能力保留接口，当前不实现。

Online 数据依赖作为可选安装项提供：`pip install 'hyper-parallel[data]'`。

`build_indexed_dataset.py` 只保留 indexed pretrain Dataset 的总流程；`indexed_data_config.py` 负责 indexed
路径解析和构建参数归一化；`indexed_split_builder.py` 构建 train/validation/test，并处理 shared blend 与
per-split blend；`indexed_blended_dataset.py` 和 `indexed_simple_blended_dataset.py` 分别实现标准权重混合与
MR inter/intra 混合；`indexed_lazy_dataset.py` 延迟到首次访问再构建真实 Dataset；
`indexed_data_reader.py` 读取 `.idx/.bin`；`indexed_sample_index.py` 构建并缓存
document/sample/shuffle 三类采样索引；`indexed_pretrain_dataset.py` 承载 GPT、Mock 和预切分记录三种运行期
Dataset。

## 五、Omni 私有阶段

```text
OmniTrainer
│
├── _build_tokenizer()
│   └── omni/build_tokenizer.py
│
├── _build_model_assets()
│   └── omni/processors/
│
└── _build_data_transform()
    └── omni/build_data_transform.py
```

Omni Dataset 实现：

```text
omni/dataset.py
├── online  → omni/online_dataset.py
└── offline → omni/offline_dataset.py
```

Omni 可以复用 LLM 的文本 tokenizer、chat template 和在线记录读取能力，但 LLM 不得依赖 Omni。

## 六、运行阶段

```text
DataLoader
    │
    ▼
Dataset.__getitem__(index)
    │
    ├── 读取 RawSample
    └── data_transform(RawSample)
            │
            ▼
        ModelSample
            │
            ▼
LLMCollator / OmniCollator
            │
            ▼
MakeMicroBatchCollator
            │
            ▼
list[micro_batch dict]
            │
            ▼
Trainer.train_step()
```

## 七、职责约束

| 文件 | 负责 | 不负责 |
| --- | --- | --- |
| `build_dataset.py` | 公共 Dataset 构建阶段 | 解析具体数据格式 |
| `build_collate_fn.py` | 公共 Collator 构建阶段 | LLM/Omni 字段策略 |
| `build_dataloader.py` | 公共 DataLoader 构建阶段 | tokenizer 和 transform |
| `parallel/dataset_context.py` | Dataset 分布式构建策略和生命周期 | Batch 运行期通信 |
| `parallel/batch_sampler.py` | DP 样本切分和可恢复 batch 采样 | Dataset 文件读取 |
| `llm/build_data_transform.py` | LLM RawSample 转换 | 文件读取和采样 |
| `llm/indexed_data_config.py` | indexed 路径解析和参数归一化 | Dataset 采样 |
| `llm/build_indexed_dataset.py` | indexed GPT 数据构建 | 重新 tokenize |
| `llm/indexed_pretrain_dataset.py` | GPT、Mock、MR 运行期采样 | 路径解析和 split 构建 |
| `omni/build_data_transform.py` | 文本和多模态转换 | DataLoader 构建 |
| `llm/omni collator.py` | 模态字段合并 | 数据读取 |

所有新数据代码和配置 target 必须使用 `hyper_models.components.datasets` 路径。

## 八、调试日志

Dataset 关键构建日志统一使用 `DEBUG` 级别，默认不输出。通过 Trainer 配置开启时，默认只由 rank 0 输出：

```yaml
debug:
  check_dataset: true
  check_nan_inf: false
```

需要指定输出 rank 时，可在 Trainer 初始化前使用接口：

```python
from hyper_models.components.datasets import enable_dataset_debug_logging

enable_dataset_debug_logging()          # rank 0
enable_dataset_debug_logging(ranks=(1, 3))
enable_dataset_debug_logging(ranks=None)  # all ranks
```

该 logger 会覆盖 Dataset、DataLoader、Sampler、Indexed cache 和 Online source 等子模块。日志只记录构建决策、
类型、数量与缓存状态，不记录样本内容，也不在 `__getitem__` 热路径输出。
