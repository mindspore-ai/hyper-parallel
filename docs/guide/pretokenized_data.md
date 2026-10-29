# 预处理数据读取：owner 设计与 JT 数据兼容

本实现对应《LLM 预训练与 SFT 数据系统》第四章的“已准备好的记录 → 组 batch → get_batch”部分。
适用于预训练和 SFT；共同前提是记录中的 labels 已经逐文档对齐为下一词目标。

## 本次实现范围

| 设计要求 | 本次实现 |
| --- | --- |
| 目标只对齐一次 | `PreTokenizedTransform` 要求 `labels_alignment: next_token`，不分词、不 shift、不追加 EOS、不截断 |
| 记录内部边界 | 提供边界则校验并保留；缺少边界按单文档处理，统一输出 `cu_seq_lens` |
| 严格校验已发布数据 | 默认遇到无有效目标的记录报错；JT 可显式选择 `empty_target_policy: drop` |
| 一行拼接组批 | `build_online_text_collate_fn()`，输出 `[1,S]`，保留记录内部边界 |
| Padding 处理 | 沿用尾部补 0、labels 补 -100，并给 PAD 尾段登记独立边界 |
| CP/TP 传输 | token/label 按 CP 切分；复用原有的全局边界保留和 TP 广播流程，不新增通信元数据 |
| 运行时字段 | `source_type: pretokenized` 按边界重置段内位置；labels 决定监督范围 |
| Attention 隔离 | dense 路径使用内部边界；compressed 路径必须提供消费边界的模型 runtime adapter |
| JT 预处理字段 | 可配置 `boundaries_key: cu_seqlens`，支持二值 `loss_mask`，不修改源数组 |

这是预处理读取功能的实现草案，不是 Issue 2097 整套数据系统已交付的声明。
离线模板编码、多字段 `sft_indexed` 写入器/读取器、manifest 发布与校验、来源追溯、断点数据恢复，
仍由相应数据源/制作模块负责；本次没有另建一套存储格式或改变现有采样、重排、loss 归一化策略。
已有 Indexed 预训练窗口读取流程也不改动。未来多字段读取器返回下述记录后即可复用本接口。

## 记录约定

```python
record = {
    "input_ids": [11, 12, 13, 14, 21, 22],
    "labels": [-100, 13, 14, 2, 22, 2],
    "boundaries": [0, 4, 6],
}
```

- 所有字段为 CPU 上的一维数组/列表；input_ids 和 labels 等长且非空。
- labels 必须已对齐。形状正确不能证明目标语义正确，数据制作方仍需校验 tokenizer、模板和目标约定。
- token 必须为非负 int32 范围内的整数；labels 还允许 -100。可设置 `vocab_size` 做额外上界检查。
- boundaries 是累计位置，必须从 0 严格递增至 token 总长度；不是每段长度。
- 有边界字段时校验并保留，缺少时按单文档补成 `[0, length]`；非法边界报错，不回退。
  不根据 token 内容猜测文档数量；多个文档的记录应提供正确边界，字段名不同用 `boundaries_key` 映射。
- 可选的二值 loss_mask 折叠到 labels：mask=0 的位置改为 -100。源数组不被修改。
- 输入记录必须未补 PAD。不要通过 label=-100 或 token 等于某个 ID 推断真实长度。
  预先补过 PAD 的外部数据须由源适配器利用明确的真实长度还原，不能猜测。
- 当前接口不自动把只有 token 的预训练文档转换成训练窗口；应使用现有 GPT 组样路径，或先生成 labels。

## Python 接入与 JT 兼容

```python
from datasets import load_from_disk
from hyper_parallel.data.online.mapping import MappingTransformDataset
from hyper_parallel.data.text import PreTokenizedTransform
from hyper_parallel.data.batching import build_online_text_collate_fn

source = load_from_disk("/path/to/sft_data_256k_demo")
transform = PreTokenizedTransform(
    boundaries_key="cu_seqlens",  # JT 已有数据的字段名
    empty_target_policy="drop",   # 显式保留原 JT 无监督记录过滤行为
)
# 在固定条数采样之前过滤，不能在一个已组好的 GBS 内悄悄减少样本。
source = source.filter(transform.is_valid_sample)
dataset = MappingTransformDataset(source, transform)
collate_fn = build_online_text_collate_fn(mesh_context=None)
```

公共入口名为 `PreTokenizedTransform`。旧 JT YAML 中的 `PreTokenizedSFTTransform` 改用此名，
并明确写出 `boundaries_key: cu_seqlens` 和需要的过滤策略。
没有在公共 master 增加 JT 专属类名或引入 `models/jt_deepseek_v3`。

JT 在原路径中使用 `source_type: online` 时，仍可保留自己的位置配置及 runtime adapter。
使用新的严格路径时改为 `source_type: pretokenized`：启用段内位置重置，禁止二次 shift 和 EOD loss 过滤。
对标准 packed 文档这是 owner 指定语义；切换训练配置前应确认它与原 JT recipe 的语义一致。

## YAML 接入片段

以下是现有训练配置中的数据部分，不是独立模型训练配方。`data_path` 可指向已有 JSONL/Arrow/Parquet
记录文件；HF `save_to_disk` 目录也可按上面的 Python 示例用现有 `MappingTransformDataset` 包装。

```yaml
dataset:
  _target_: hyper_parallel.data.text.build_dataset.build_online_text_mapping_dataset
  data_path: /path/to/prepared_records.jsonl
  model_assets:
    tokenizer: null
    chat_template: null
  data_transform:
    _target_: hyper_parallel.data.text.PreTokenizedTransform
    boundaries_key: boundaries  # JT 数据改为 cu_seqlens
    labels_alignment: next_token
    empty_target_policy: error  # JT 需旧过滤语义时显式设 drop
  data_config:
    labels_are_shifted: true
    eod_mask_loss: false

dataloader:
  _target_: hyper_parallel.data.batching.FixedBatchDataLoader
  collate_fn:
    _target_: hyper_parallel.data.batching.build_online_text_collate_fn
  get_batch:
    _target_: hyper_parallel.data.batching.TextParallelBatch
    source_type: pretokenized
    attention_mode: dense
```

组批沿用一行拼接方式，输出 `[1,sum(S)]`，必要时在尾部补齐以满足并行对齐要求。
本 PR 不提供 `pack` 布局开关或按行补齐组批，不修改 GBS、MBS、梯度累积或 loss 归一化。
模型侧 MTP 的 shift 仍须尊重文档边界；本功能不能替代模型侧 MTP 正确性修复。

长序列不要使用 dense mask。改用 `attention_mode: compressed`，并设置模型拥有的
`runtime_input_adapter`，让它消费传入的全局 `cu_seq_lens`。
仅配置 adapter 不能自动证明其 attention 语义正确，模型接入方仍需验证。

## Owner 例子对应输出

原记录 A+B 长 8，内部边界 `[0,4,8]`；记录 C 长 4，边界 `[0,4]`。对齐到 16：

```text
input_ids:     [A | B | C | PAD]       shape=[1,16]
labels:        [A目标 | B目标 | C目标 | -100,-100,-100,-100]
cu_seq_lens:   [0,4,8,12,16]
position_ids:  [0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3]
```

真实 token 只关注本段的当前与之前位置，不能关注其他文档或 PAD。PAD query 保留独立尾段的因果
可见关系以避免全遮罩行，其 labels 始终为 -100，不参与 loss。PAD 的位置编号沿用普通区间内
从 0 递增的规则，不再单独清零。labels=-100 只表示不参与监督，不用于推断文档边界或真实长度。

## 验证范围

单测覆盖 owner 示例、JT 字段与 mask、只读源数组、不合法输入、普通样本拼接回归、默认尾部补 0、多文档跨 CP 切片、
TP 源/接收端元信息一致、旧无边界批次回归，以及拼接与独立文档计算的 CPU attention loss/梯度一致性。
并行通信单测使用模拟 collective，不等于真实 NPU 多卡验收。整网 JT/MTP 精度和吞吐需要后续在
JT overlay 接入后验证；本分支没有修改模型、路由、优化器或发起训练任务。
