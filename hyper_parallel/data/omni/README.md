# Omni 图文数据管线

本分支基于 master 的共享 Online 数据源、Omni transform 和 batching 接口，增加
Qwen 图文训练所需的图片绑定与监督边界处理。通用数据读取、packing、恢复与广播
继续使用 master 的实现。

## 数据与配置

完整训练示例见 [Qwen VLM 配置](../../../examples/qwen3_6_moe/train_online_vlm.yaml)
和 [P2P KV all-gather 配置](../../../examples/qwen3_6_moe/train_online_vlm_p2p_kv_allgather.yaml)。
配置中的模型、processor 和数据路径需替换为本地资源。使用
`build_online_omni_mapping_dataset`、`build_auto_processor` 和
`hyper_parallel.data.batching.build_omni_collate_fn`。

本地 JSON、JSONL 和 Parquet 样本保留来源路径，图片相对路径以源文件目录为基准。
Parquet 使用 Hugging Face datasets 读取、过滤和划分，需安装 datasets 与 pyarrow；
图片解码留到 transform 访问样本时执行。空数据源遵循共享 builder 的 None 返回约定。

## 外置图片

样本可以包含 messages 和独立 images 列表：

```python
{
    "messages": [
        {"role": "user", "content": "比较 <image> 和 <image>。"},
        {"role": "assistant", "content": "第一张图片……"},
    ],
    "images": [
        {"bytes": png_bytes, "path": None},
        {"bytes": None, "path": "images/second.png"},
    ],
}
```

字符串或 typed text 中的 `<image>`，以及没有显式来源的 image block，按对话顺序
消耗 images。显式带来源的 image block 不消耗该列表。bytes 优先于 path，解码为
完整 RGB PIL 图片；缺图、多图或损坏的字节会报错。

## 监督与并行

AutoProcessorTransform 仅监督最终 assistant 响应。针对 generation prompt 与完整
响应在边界处发生 token 合并的模板，回退到可验证的公共前缀边界；无法确认监督
范围或没有有效标签的样本会报错。该策略不改变文本路径的多轮监督。

Qwen 的 `QwenVLMParallelBatch` 提供模型专用 CP 输入与视觉特征插入契约；视觉编码器
并行通过模型 adapter 和 master encoder runtime 接入。不能仅凭保留完整 pixel_values
就认定任意模型支持 CP。多样本 packing、视频以及其他并行组合必须满足各 adapter
的显式校验要求。

## 验证

`tests/ut/data/omni/test_build_dataset.py` 覆盖真实 Parquet、来源路径、过滤顺序、图片
字节及 worker pickle；`test_omni_transform.py` 与 `test_omni_data.py` 覆盖图片绑定及
processor 监督边界。分布式 Qwen 特征和梯度测试位于 `tests/torch/encoder_parallel/`。
CPU 单测不能替代完整 NPU 训练验证。
