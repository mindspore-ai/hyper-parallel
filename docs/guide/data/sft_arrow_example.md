# 预处理 Arrow 数据读取与组批示例

适用于已经分词、已准备好下一词 labels 的预训练/SFT 数据，记录内可以包含多个文档。
直接读取记录，不再分词、移动 labels 或重新装箱；多个 Arrow 文件各自构建 Dataset、切分后再混合。

## 数据示例

假设数据目录如下，每个文件包含一条记录。以下展示的是 Arrow 记录的字段内容，不是将文本改名为 `.arrow`：

```text
prepared-arrow-data/
  a.arrow
  b.arrow
```

`a.arrow` 中的记录：

```json
{
  "input_ids": [11, 2, 13, 21, 22, 0, 0, 0],
  "labels": [2, 13, -100, 22, -100, -100, -100, -100],
  "cu_seq_lens": [0, 3, 5, 8],
  "loss_mask": [0, 1, 1, 1, 1, 0, 0, 0]
}
```

`b.arrow` 中的记录：

```json
{
  "input_ids": [31, 32, 41, 0, 0, 0, 0, 0],
  "labels": [32, -100, 42, -100, -100, -100, -100, -100],
  "cu_seq_lens": [0, 2, 3, 8, -1],
  "loss_mask": [1, 1, 1, 0, 0, 0, 0, 0]
}
```

- `input_ids`：已分词的输入 token ID。
- `labels`：每个输入位置已经对齐的预测目标，长度与 input_ids 相同；`-100` 不参与 loss。
- `cu_seq_lens`：记录内部的累计文档边界，例如 `[0, 3, 5]` 表示长度为 3 和 2 的两个文档。
  尾部 `-1` 是边界补齐值，读取时清除；省略边界时将整条记录视为一个文档。
- `loss_mask`：可选的二值监督掩码，0 对应的位置在 CPU 读取时转成 `labels=-100`。
  不提供该列时，直接使用 labels 中已有的忽略标记；边界不能代替 loss_mask。

文件须为 HF Arrow IPC stream 格式。传入 HF 保存目录时，按其中各个物理 Arrow 文件分别处理，
不恢复 `state.json` 的逻辑顺序或 DatasetDict 的 split；应传入本次需要混合的源文件或目录。

两条记录在预处理时都已补齐到 8 个 token，尾部 labels 为 -100，padding 使用独立边界区间。

## 如何配置 YAML

下面给出数据接入 YAML 示例。将相关字段合并到实际训练 YAML，同时保留模型、优化器和并行配置；
此示例本身不是完整的训练启动配置。

本例使用下列配置选择读取方式、字段映射和组批入口：

```yaml
training:
  seed: 42
  train_iters: 1
  global_batch_size: 2
  micro_batch_size: 2

debug:
  check_nan_inf: false
  check_dataset: debug

dataset:
  _target_: hyper_parallel.data.text.build_dataset.build_indexed_text_dataset
  # 替换为实际 Arrow 源文件或目录。
  data_path: ./prepared-arrow-data
  model_assets:
    tokenizer:
      _target_: hyper_parallel.data.text.build_tokenizer.AutoTokenizer.from_pretrained
      pretrained_model_name_or_path: unused
      tokenizer_type: pretokenized
      vocab_size: 128
      eod_token_id: 2
  data_config:
    seq_length: 8  # 必填：每条完整记录的 token 数，不是内部子文档长度
    low_level_dataset:
      _target_: hyper_parallel.data.indexed.SFTFromMRDataset
      input_ids_key: input_ids
      labels_key: labels
      boundaries_key: cu_seq_lens
      loss_mask_key: loss_mask
    simple_blend: intra
    split: "1,0,0"
    labels_are_shifted: true
    eod_mask_loss: false

dataloader:
  _target_: hyper_parallel.data.batching.FixedBatchDataLoader
  num_workers: 0
  drop_last: true
  collate_fn:
    _target_: hyper_parallel.data.batching.build_sft_collate_fn
  get_batch:
    _target_: hyper_parallel.data.batching.TextParallelBatch
    source_type: indexed
    # 仅限本示例的小序列；长序列应配置模型支持的 compressed attention。
    attention_mode: dense
    reset_position_ids: true
```

使用时注意：

- 将 `data_path` 改为实际源文件或目录。每个 `.arrow` 文件独立切分，本例 `split: "1,0,0"` 表示全部用于训练，
  `simple_blend: intra` 按数据源顺序混合。
- 数据字段名不同时修改 `*_key`，例如数据中叫 `cu_seqlens`，则设置 `boundaries_key: cu_seqlens`。
- 上述 YAML 使用 `tokenizer_type: pretokenized` 提供 tokenizer 元数据，不重新分词；
  `vocab_size: 128`、`eod_token_id: 2` 仅适用于示例，需替换为实际模型的词表大小和 EOD ID。
- `seq_length` 必填，没有默认值；默认 `skip_data_check: false`，读取每条记录时检查长度，偏长或偏短都报错。
  SFT labels 已对齐，检查长度直接等于 `seq_length`，不额外加一个 token。
- labels 已对齐，必须保持 `labels_are_shifted: true`；监督以 labels 为准，设置 `eod_mask_loss: false`。
- 全局 batch 和并行配置需满足实际训练要求；本例无并行切分、MBS=2，仅用于说明组批结果。
- `check_dataset: debug` 开启数据模块的调试日志，用于查看路径发现、Dataset 构建及并行取批形状。
  它不保证打印所有 tensor 内容，也不替代数据正确性校验。

## 组批结果

上述两条记录都满足 `seq_length: 8`，组批后保留两行，得到 `[B,S]=[2,8]`：

```text
input_ids:    [[11, 2, 13, 21, 22, 0, 0, 0], [31, 32, 41, 0, 0, 0, 0, 0]]
labels:       [[-100, 13, -100, 22, -100, -100, -100, -100], [32, -100, 42, -100, -100, -100, -100, -100]]
boundaries:   [[0, 3, 5, 8], [0, 2, 3, 8]]
position_ids: [[0, 1, 2, 0, 1, 0, 1, 2], [0, 1, 0, 0, 1, 2, 3, 4]]
loss_mask:    [[0, 1, 0, 1, 0, 0, 0, 0], [1, 0, 1, 0, 0, 0, 0, 0]]
有效监督位置数: 4
```

本例 collate 不改变行长度。下游累计边界转换为 `[0, 3, 5, 8, 10, 11, 16]`，
但 tokens 和 labels 保持 `[2,8]`。正确消费这些边界的 attention 会隔离不同文档以及 padding 与真实文档。

## 长度校验与 32K、256K 数据

`seq_length` 必须显式填写。默认 `skip_data_check: false` 时，SFT Dataset 在取出每条记录、组批之前，
要求 tokens 和 labels 长度一致且等于 `seq_length`；偏短、偏长都会报错，不自动截断、补齐或重新装箱。
Arrow 长度检查按记录读取时执行，不在初始化时扫描全量文件；开启 lazy load 时也执行相同检查。

- 固定 32K 的记录配置 `seq_length: 32768`。
- 固定 256K 的记录配置 `seq_length: 262144`。
- 一条 256K 记录内部的子文档可以长短不一，只要求整条记录的总长度为 256K。
- 32K 与 256K 记录混用，默认会因长度不匹配报错；如需统一长度，应先预处理数据及其 labels、边界。

与 GPTFromMRDataset 沿用同一个 `skip_data_check` 开关：显式设置为 true 可跳过长度检查，
但 `seq_length` 仍必填，字段、标签和边界的合法性检查仍保留。此时 collate 可将变长行补至批内最长长度，
输出 S 不再保证等于 `seq_length`，不建议以此规避数据与配置不一致的问题。

collate 仍保留 TP×CP 对齐补齐逻辑。为保持最终 S 等于配置长度，`seq_length` 应满足所用并行策略的对齐要求；
32K、256K 在常用的 2 的幂次并行规模下可满足该要求。

示例的 `attention_mode: dense` 仅用于小序列，不要直接用于 256K。
真实长序列训练应配置 `compressed` 及能够消费文档边界、支持 `[B,S]` 的模型 runtime adapter。
数据读取支持某长度，不代表模型位置编码、attention 实现和显存容量已经满足该长度的训练。
