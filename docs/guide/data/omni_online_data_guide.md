# Omni Online 数据接入指南

本文以仓库中的 Qwen2-VL 示例说明如何把图片对话样本接入 `VLMTrainer`。通用 Omni 链路负责读取、过滤、
编码、packing 和 TP/CP batch 传输；模型特有的 processor、监督标签和额外 forward 字段由 transform 或
runtime adapter 负责。DeepSeek-V4.1 的模型专用处理见
[DeepSeek-V4.1 Online VLM 数据转换指南](deepseek_v41_vlm_online_data_guide.md)。

## 1. 最小可用样本

`AutoProcessorTransform` 接收非空 `messages`，并要求最后一条是 assistant。Qwen2-VL 示例使用如下
processor 兼容格式；图片路径请先使用绝对路径：

```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "image", "url": "/data/images/example.jpg"},
        {"type": "text", "text": "Describe this image briefly."}
      ]
    },
    {
      "role": "assistant",
      "content": [{"type": "text", "text": "A city skyline."}]
    }
  ]
}
```

可以将多个这样的对象放进 JSON 数组（`.json`），也可以每行一个对象写成 `.jsonl`。示例脚本默认读取
`synthetic.json`；要使用自己的文件，可通过 `DATA_PATH` 指定。`messages=[]` 会在 source 建索引时被
`is_valid_sample()` 丢弃；缺失 `messages` 或其类型不是 list 则会报错。JSONL 的原生索引、相对媒体路径、
split 与其他文件格式的路由细节见上面的 DeepSeek-V4.1 指南；不要假定任意 JSON 图片路径都按文件目录解析。

## 2. YAML 接入点

参照 [`examples/training_demo/train_vlm.yaml`](../../../examples/training_demo/train_vlm.yaml) 配置以下两段：

```yaml
dataset:
  model_assets:
    _target_: hyper_parallel.data.omni.build_auto_processor
    pretrained_model_name_or_path: /path/to/Qwen2-VL-model
    local_files_only: true
  data_transform:
    _target_: hyper_parallel.data.omni.AutoProcessorTransform
    max_seq_len: 512
  _target_: hyper_parallel.data.omni.build_online_omni_mapping_dataset
  data_path: /path/to/train.json
  data_config: {}

dataloader:
  _target_: hyper_parallel.data.batching.OmniPackingLoader
  collate_fn:
    _target_: hyper_parallel.data.batching.build_omni_collate_fn
  get_batch:
    _target_: hyper_parallel.data.batching.OmniParallelBatch
    encoder_dp: false
  sampler_type: single
  token_budget: 128
  min_buffered_samples: 1
  num_workers: 0
```

`model_assets` 创建 processor，配置解析器将其注入 `data_transform.processor`；Dataset builder 接收
transform。`OmniPackingLoader` 完成候选样本选择和 packing，`build_omni_collate_fn` 组装 batch，
`OmniParallelBatch` 为模型 forward 准备分布式输入。示例直接使用该类，无需额外 hook。

不经过 Trainer 时，也可以单独检查 Dataset 接口：

```python
from types import SimpleNamespace

from hyper_parallel.data.omni import (
    AutoProcessorTransform,
    build_auto_processor,
    build_online_omni_mapping_dataset,
)

processor = build_auto_processor(
    pretrained_model_name_or_path="/path/to/Qwen2-VL-model",
    local_files_only=True,
)
transform = AutoProcessorTransform(max_seq_len=512, processor=processor)
dataset = build_online_omni_mapping_dataset(
    data_path="/path/to/train.json",
    data_config={},
    transform=transform,
    training_config=SimpleNamespace(seed=42),
)
encoded_sample = dataset[0]  # 无 split 时返回 Mapping Dataset
print(encoded_sample["input_ids"].shape, encoded_sample["labels"].shape)
```

配置 ratio split 或预分割路径后，Dataset builder 返回 `(train, valid, test)`，其中不存在的 split 为
`None`。`transform.is_valid_sample` 会自动作为原始记录过滤器传入 source builder；无需再手动传一次。

`max_seq_len` 目前仅保存在 `AutoProcessorTransform` 中，**不会自动截断或填充样本**；
`token_budget` 决定一次 packing 尽量选择多少 token，也不是硬性的单样本截断上限。
接入较长样本时，应在模型专用 transform 中明确实现截断及图像 patch/grid 同步处理，或提前清洗数据。

## 3. 从原始样本到模型输入

```text
JSON/JSONL record
  -> build_online_omni_mapping_dataset: 过滤空 messages，建立 Mapping source
  -> OmniDataTransform.prepare_messages: 复制消息，处理可识别的媒体路径
  -> AutoProcessorTransform.encode_sample: processor 编码并生成 labels
  -> OmniPackingLoader: 按 token_budget 选择样本，SamplePacker 拼接同名字段
  -> OmniCollator: 合并 token 与变长图像字段
  -> OmniParallelBatch: CP 切 token 字段、TP 广播、生成 loss_mask
  -> VLMTrainer -> 模型 forward/loss/backward
```

`AutoProcessorTransform.encode_sample()` 对同一条记录编码两次：

1. 完整对话，`add_generation_prompt=False`，得到 `input_ids`、`attention_mask`、图片字段等模型输入；
2. 去掉最后的 assistant 回复，`add_generation_prompt=True`，得到包含 assistant 起始标记的 prompt 长度。

它检查 prompt token 是否为完整对话的前缀，然后复制 `input_ids` 得到 `labels`：prompt 区间以及
`mm_token_type_ids != 0` 的多模态位置设为 `-100`，最后的 assistant 文本保留 token ID。例如：

```text
input_ids:  [system / user / image / assistant prefix | answer / end]
labels:     [-100  / -100 / -100  / -100             | answer / end]
```

processor 本身不会自动生成训练 `labels`。此处标签与 `input_ids` 等长、尚未手工左移；示例使用的
Qwen2-VL Causal LM 在计算 loss 时处理位移。transform 返回单条样本的一维 token 字段，以及完整的
`pixel_values`、`image_grid_thw` 等模态字段；`OmniParallelBatch` 要求 batch 中已有 `labels`，并在缺少
`loss_mask` 时由 `labels >= 0` 推导。不要让所有 token 都保留为有效标签，否则用户提示和图像占位符
也会参与监督。

默认非 encoder-DP 路径中，`OmniParallelBatch` 只对 token 字段做 CP 分片；图像等模态字段不按 CP
切分，而是在 TP×CP 坐标上保持完整，供各 rank 的视觉分支使用。`encoder_dp: true` 的图像 bucket
与 all-to-all 路径尚未实现；PP/`pp_shared_data` 当前也不支持。模型如果还需要特殊的 image-token
映射或额外 forward 参数，应提供模型自己的 `OmniDataTransform`，并按需配置
`runtime_input_adapter`，不要把模型专用字段规则写进通用 collator。

## 4. 接入另一种 Omni 模型

保持 Dataset、packing 和并行 batch 入口不变，只替换模型专用部分：

1. 配置能处理该模型消息格式的 processor，或提供自己的 `model_assets` builder。
2. 继承 `OmniDataTransform` 并实现 `encode_sample()`；至少返回同长度的 `input_ids`、`labels`，
   其中不参与 loss 的 token 为 `-100`。图片、视频等字段应与模型 forward 的参数名称、形状一致。
   若需要先按轻量元数据选择样本，再做昂贵的媒体编码，可改用成对的
   `preencode_sample()` / `postencode_sample()`；如需 batch 级整理，可实现 `encode_batch()`。
3. 将 YAML 的 `dataset.model_assets`、`dataset.data_transform` 改为新目标。若 TP/CP 传输后还需构造
   模型专用位置、mask 或图片映射，在 `dataloader.get_batch.runtime_input_adapter` 配置模型 adapter；
   普通 processor 字段无需该 hook。

`build_online_omni_mapping_dataset()` 会检查 transform 是 `OmniDataTransform`，因此单独实现一个
`__call__(record)` 的类不能直接替代它。可参考
[`DeepseekV41OmniTransform`](../../../hyper_parallel/models/deepseek_v41/adapter/data/transform_fn.py)
的模型专用实现。

## 5. 运行示例与检查

脚本 [`examples/training_demo/run_vlm.sh`](../../../examples/training_demo/run_vlm.sh) 默认启动 8 个进程，
并将 `MODEL_PATH`、`DATA_PATH` 和命令行覆盖项传入 YAML。使用自己的本地模型和样本：

```bash
cd /path/to/hyper-parallel
MODEL_PATH=/path/to/Qwen2-VL-model \
DATA_PATH=/path/to/train.json \
RUN_NAME=vlm_smoke \
bash examples/training_demo/run_vlm.sh --training.train_iters=1
```

日志写入 `output/training_demo/vlm/run_vlm_smoke.log`。先检查数据阶段不再出现
`Omni batch must contain labels`，再检查是否实际输出 step/loss；前者只证明标签接入，不能证明视觉
模型的反向计算已跑通。当前提供的 tiny Qwen2-VL 示例在本地 Ascend 环境中已通过标签与 batch 检查，
但单卡、8 卡试跑均在 backward 阶段发生原生 `SIGSEGV`，其原因尚未确认。
