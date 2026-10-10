# 通用训练 loss 与吞吐日志

沿用现有 TextTrainer / VLMTrainer 入口，设置日志间隔：

```yaml
training:
  logging_steps: 1
```

现有 LoggingCallback 在 rank 0 输出；进度条和远端日志回调读取同一份 `step_train_metrics` / `step_env_metrics`。诊断 loss 与吞吐统计不按模型名称分支，也不加入反传目标。

## 字段和口径

| 字段 | 含义 |
| --- | --- |
| `training/total_loss`、已有分项目标 | 原 loss 模块返回的反传目标及聚合结果 |
| `training/<loss_metrics key>` | 模型实际提供的诊断标量，例如 LM、MTP、MoE、视觉、音频或对比学习 loss |
| `training/aux_loss`、`training/indexer_loss` | output 存在相应标量时兼容读取；不反推、不伪造目标 |
| `training/lr`、`training/grad_norm` | 原学习率和梯度范数 |
| `performance/step_time` | 秒；等待设备完成后，取全 world 最慢 rank 的步时 |
| `performance/tokens_per_second` | 有效监督文本 token/s；Text/VLM 入口使用训练侧 causal 对齐后的 labels 与 loss_mask 交集 |
| `performance/input_tokens_per_second` | `input_ids` 的输入 token/s，包含 padding；不是图像 patch/s 或音频帧/s |
| `performance/samples_per_second` | 全局逻辑样本/s；packed 情况按 batch 行计数，不是内部文档数量 |
| `performance/throughput_tflops_per_device` | 估算模型 FLOPs / 最慢步时 / world_size / 1e12 |
| `data/step_tokens`、`data/step_input_tokens`、`data/step_samples` | 本步对应计数 |
| `data/consumed_tokens`、`data/consumed_samples` | 原累计计数 |

TextTrainer 的计时包含步内微批读取；VLMTrainer 保留原来预取后开始计时的边界。比较两种 Trainer 的端到端速度时，应另外统一数据加载的计时边界。

## 诊断 loss 接口

模型在原 output 上附加可选的标量字典即可，支持普通对象、dict 和 HF ModelOutput：

```python
output.loss_metrics = {
    "language_loss": language_loss.detach(),
    "vision_loss": vision_loss.detach(),
    "contrastive_loss": contrastive_loss.detach(),
}
```

Trainer 在 backward 前 detach/clone，在设备上累计，步尾读取。诊断按微步及 DP/CP rank 做算术均值，与训练目标的 token 加权可能不同，不能再次相加重构 `total_loss`。值必须是 TP 上已复制的标量；CP 局部贡献须由模型先合并为所需语义，不能直接冒充完整序列均值。key 必须是非空、不含 `/` 的名称，在一个优化器步及 DP/CP rank 间一致，不得覆盖 `total_loss` 等训练字段。没有执行的目标就省略，不打印伪造的 0。

共享 `calculate_mtp_loss(..., loss_metrics=metrics)` 可收集各深度乘训练系数前的 CE；调用者把 metrics 放入 output 后才会记录。`aux_loss` 不自动改名为 `load_balancing_loss`，因为模型的系数和归一化方式可能不同。

## 默认 Transformer FLOPs

不设置 `flops_estimator` 时，按结构配置解析常见 HF / Megatron 字段，支持 dense / MoE、MHA / GQA / MLA、GeLU / SwiGLU、混合 dense/expert 层、共享专家与显式 latent projection，不依赖 `model_type`。MoE 计算激活的 top-k 专家，而非全部存储专家。词表优先采用实际 `padded_vocab_size`。MTP 只统计能识别到的实际执行模块，不把 checkpoint 中存在但未执行的层算进去。

计算由两部分组成：token 线性项（Q/K/V/O、MLP、输出 head 等）和 attention 二次项。普通序列的二次项为 `batch_size × sequence_length²`；真实 packed attention 使用 `Σ L_i²`，不能把整条 packed 长度平方。TextParallelBatch 向统计侧保留全局 `cu_seq_lens`，不把额外统计字段传给模型。投影/MLP 仍按实际输入 tensor 长度计数，不因标签 mask 减少工作量。

乘加按 2 FLOPs；普通训练按前向、输入梯度、权重梯度合计 3 倍；causal attention 使用参考实现的半平方估计。CP 还原完整序列工作量，统计侧除去 CP 副本后在 DP+CP 汇总；TP 不重复算模型副本，最后除全 world 卡数。

该值沿用 MF/Megatron 的主要模型算子估算口径，省略路由、归一化等未建模小算子、优化器、通信、重计算及内核额外 padding；不是硬件实测 FLOPs 或 MFU。

## 多模态、冻结组件和自定义结构

默认估算器不把多模态的语言分支当成整模 FLOPs。复合模型或存在冻结参数时，配置完整的 `CompositeFlopsEstimator`，或提供自己的 callable。以下示例适用于已经形成 `[B, P, patch_dim]` patch token 的视觉输入：

```yaml
flops_estimator:
  _target_: hyper_parallel.trainer.runtime.flops.CompositeFlopsEstimator
  components:
    # patch embedding：patch_dim=768 -> vision hidden=1024
    - input_key: pixel_values
      linear_dimensions: [768, 1024]
      forward_backward_factor: 1
    # 冻结的视觉 encoder，使用 no_grad 执行
    - input_key: pixel_values
      config_path: vision_config
      forward_backward_factor: 1
    # 训练中的多模态 projector
    - input_key: pixel_values
      linear_dimensions: [1024, 4096, 4096]
    # input_ids 的长度必须等于 decoder 实际执行长度，含图像占位 token
    - input_key: input_ids
      config_path: text_config
      causal: true
      include_logits: true
      cp_partitioned: true
```

这些维度是示例，需要改成实际结构；`config_path` 支持点号子路径。每项只能选择 Transformer 配置或线性层维度链之一。应声明全部实际执行分支，包括 patch embedding、编码器、projector、decoder 和其他 heads。音频的 Transformer encoder 与线性投影可以按同一方式声明；卷积、cross-attention、动态分辨率 merge 等不能直接套成普通 Transformer，需自定义完整估算器。

每个组件的 `forward_backward_factor`：完全训练为 3；冻结且 `no_grad` 为 1；权重冻结但仍需输入梯度为 2（参数 GEMM 按 2 倍，core attention 的两个操作数均为激活，仍按 3 倍）。`input_key` 必须对应处理后的二维/三维 token tensor，不能把原始 `[B,C,H,W]` 图像或波形长度当作 token 数。对于实际无 padding 执行的变长分支，可配置 `lengths_key` 指向一维全局序列长度 tensor：使用其和与平方和；它必须已经描述完整 CP 序列。处理 padded tensor 时应使用实际执行的 padded 长度。

`cp_partitioned: true` 表示该输入的序列维已按 CP 切分，需要还原；编码器输入如为 CP 复制则保留默认 false。仅当缺失输入确实表示该分支未执行时，设置 `optional_input: true`，缺失时贡献为 0；否则缺失任何必需输入都省略整模 TFLOP/s。

更复杂结构通过以下契约接入，无需改 Trainer：

```python
class MyFlopsEstimator:
    def __init__(self, model_config, model=None):
        # 在这里缓存结构和执行信息，不保存 activation。
        ...

    def __call__(self, batch, *, cp_size=1):
        # 返回完整逻辑 CP 微批的模型 FLOPs，TP/CP 副本必须一致。
        # 返回有限非负标量或设备 scalar tensor；信息不全时返回 None。
        # 不在微步调用 .item()，不执行额外模型 forward。
        ...
```

```yaml
flops_estimator:
  _target_: my_package.metrics.MyFlopsEstimator
```

未知/不支持的 attention 结构、缺少必需输入、无法确认的冻结执行方式会省略 TFLOP/s 并提示一次，其他日志继续工作。配置的组件清单是否完整由接入者保证，框架不会凭配置推断未声明分支。负数、NaN/Inf 等非法返回值在步尾报错；DP/CP rank 的可用状态先一致归约，避免部分 rank 跳过 collective。PP 分段模型本次不输出整模 FLOPs。

## 参考与验证

参考固定源码，便于复核公式变化：

- [Megatron-LM training.py，53204264](https://github.com/NVIDIA/Megatron-LM/blob/53204264a03aee9bd4a97cf68896312417662cff/megatron/training/training.py)：常规与 packed 工作量分离、dense/MoE 和 MTP 估算。
- [MindFormers models/utils.py，d3f5dc94](https://gitcode.com/mindspore/mindformers/blob/d3f5dc943f6bf50cfa238db9c320ba4ff75ac52e/mindformers/models/utils.py)：通用 Transformer FLOPs。
- [MF PyNative loss callback](https://gitcode.com/mindspore/mindformers/blob/d3f5dc943f6bf50cfa238db9c320ba4ff75ac52e/mindformers/pynative/callback/loss_callback.py)：诊断 loss 与日志口径。

仓内复现：

```bash
python -m pytest -q tests/ut/trainer tests/torch/trainer/test_training_metrics_distributed.py
```

覆盖手算公式、结构别名、packed、组合组件/冻结因子、真实 batching/VLMTrainer 入口、非法指标、输出间隔、反传不变，以及 4 进程 Gloo 的 DP/CP/TP 计数。这些是功能与数值验证，不代替 NPU/GPU 的实际吞吐基准。
