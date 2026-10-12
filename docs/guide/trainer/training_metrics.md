# LLM 训练 loss 与吞吐日志

## 使用方法

沿用 TextTrainer 的 YAML 和训练命令，只设置输出间隔：

```yaml
training:
  logging_steps: 1
```

每个优化器步结束后，rank 0 打印一行；设为 N 时只打印第 N、2N…步的本步值，不做 N 步滑动平均，<=0 关闭终端逐步输出。进度条和现有监控回调仍读取同一份 `step_train_metrics` / `step_env_metrics`。

普通支持的 LLM 不需要额外吞吐 YAML。模型需要补充诊断时，保留原 `output.loss`，在输出对象上附加标量字典：

```python
output.loss_metrics = {
    "lm_loss": lm_loss.detach(),
    "load_balancing_loss": router_loss.detach(),
}
```

dense 模型只填实际存在的项。支持普通对象、dict 和 HF ModelOutput 动态属性；也兼容读取输出已有的 `aux_loss` / `indexer_loss`。不会把 `aux_loss` 强行改名为 `load_balancing_loss`，两者系数和归一化可能不同。

共享 `calculate_mtp_loss(..., loss_metrics=metrics)` 可收集各深度乘训练系数前的 `mtp_1_loss`、`mtp_2_loss` 等；调用者把 `metrics` 放入输出后才会打印。通用 Trainer 不直接访问某个模型的 router、MTP 或专家模块来读取损失。

接口契约：key 为非空且不含 `/` 的名称，在同一步的微批及 DP/CP rank 间保持一致，不覆盖原训练字段；值为 TP 上已复制的标量。CP 局部贡献须先在模型侧还原成定义所需的完整序列值。Trainer detach/clone 后累计，步尾做微批、DP/CP 算术均值，独立于训练侧的 token 加权；诊断不能再次相加重建 total_loss。缺失项省略，不能伪造 0。

## 每步打印哪些字段

| 输出字段 | 含义与出现条件 |
| --- | --- |
| `step`、`total_steps`、`epoch` | 已完成的优化器步、Trainer 算出的总步数、当前 epoch；总步数不可得时省略 |
| `training/total_loss` | 原 loss 模块聚合出的训练目标，保持既有归一化和权重 |
| `training/<原 loss_dict 分项>` | 训练模块已有的命名目标；不改名或重复相加 |
| `training/<loss_metrics key>` | 模型实际提供的诊断 loss，例如 `lm_loss`、`mtp_1_loss`、`load_balancing_loss` |
| `training/aux_loss`、`training/indexer_loss` | 输出对象存在同名标量时兼容读取；不猜测未提供的值 |
| `training/lr` | 步末 scheduler/optimizer 参数组中最大的学习率，保持原口径；调度器已执行更新 |
| `training/grad_norm` | 原 Trainer 传入的梯度范数；未开启裁剪且没有相关诊断时，既有 Trainer 返回 0，此处不额外扫描参数 |
| `performance/step_time` | 秒；设备同步后，取整个 world 中最慢 rank 的步时 |
| `performance/tokens_per_second` | 全局有效监督 token/s，TextTrainer 使用训练侧已对齐的 labels/loss_mask 计数 |
| `performance/input_tokens_per_second` | 全局实际输入 tensor token/s，包含 padding，不因标签 mask 减少 |
| `performance/samples_per_second` | 全局 batch 行数/s；packed 行内文档不另算样本 |
| `performance/throughput_tflops_per_device` | 可估算结构的模型 FLOPs / 最慢步时 / world_size / 1e12 |
| `data/step_tokens`、`data/step_input_tokens` | 本优化器步的有效监督与输入 token 数，含全部梯度累积微步 |
| `data/step_samples` | 本步实际全局 batch size，已排除 CP/TP 重复副本 |
| `data/consumed_tokens`、`data/consumed_samples` | callback 当前累计计数；已有 state_dict 接口尚未自动接入 checkpoint，不宣称重启后自动延续 |
| `memory/device_max_allocated_gb`、`memory/device_max_reserved_gb` | 仅加速器输出；原有 DP+CP 组内最大峰值，实际单位 GiB，非单步增量，也非全 world 峰值 |

## 吞吐如何计算

统计一个完整优化器步内的全部微批。设最慢 rank 耗时为 `t` 秒，全局有效监督 token 数为 `U`，实际输入 token 数为 `T`，全局 batch 行数为 `B`，设备数为 `W`：

```text
有效监督 token/s = U / t
输入 token/s     = T / t
样本/s          = B / t
TFLOP/s/device  = F_global / (t × W × 10^12)
```

tokens 在 DP+CP 组内求和；样本数求和后除 CP；TP 副本不参与重复求和。每个 CP rank 的 FLOPs 先还原完整序列、再除 CP，随后 DP+CP 求和，最后除 world_size。计时在步首设备同步后开始，步末同步后结束；TextTrainer 包含步内取数、前向、反向和优化器更新，不含随后的日志写出、checkpoint。已有 BaseTrainer 预取边界不变。

模型公式只按结构字段解析，不按模型名字选择：dense / MoE，MHA / GQA / MLA，GeLU / SwiGLU，混合 dense/expert 层、共享专家、显式 MoE latent projection。HF 和 Megatron 常用字段别名兼容；词表优先用 `padded_vocab_size`，MoE 按激活 top-k 而不是全部存储专家计算。

将前向工作拆成 token 线性系数 `P` 和 attention 二次系数 `A`：

```text
F_micro = 3 × (P × T_micro + A × Q_micro)
普通 causal attention: Q_micro = batch_size × sequence_length²
独立 packed attention: Q_micro = Σ(document_length²)
```

`P` 包括 Q/K/V/O、dense 或激活专家 MLP、共享专家、输出 head，以及实际 MTP 层的投影；乘加计 2 FLOPs。`A` 使用 MF/Megatron 的 causal 半平方近似。系数 3 代表前向、输入梯度和权重梯度。普通模型仅有 checkpoint MTP 配置而没有执行模块时，不增加 MTP；支持的 MTP 模块按实际层数和 MLP 类型计数。

分段长度必须描述实际 forward：构造了分文档 dense mask 时，TextParallelBatch 仅向统计侧保留 `cu_seq_lens`；使用 compressed attention 时，读取模型 forward 的 `actual_seq_len`（batch=1、全局累计结束位置）。只有 Arrow/online 原始边界但没有实际分段 forward 时按完整 causal 序列估算。CP 使用全局边界；投影项仍按全部实际输入（含 padding）计算，不能用 loss_mask token 数代替。

这是模型主要算子的估算吞吐，省略路由、激活/归一化等未建模小算子、优化器、通信、重计算和内核额外 padding，不是硬件实测 FLOPs、MFU 或加速比。未知结构、sliding/sparse/linear/cross attention、部分参数冻结、PP 分段模型或缺少必需元数据时省略 TFLOPs，并提示一次；loss、token/s 和样本/s 继续输出。非法负数/NaN/Inf FLOPs 在步尾拒绝，先归约可用状态避免 rank 间 collective 不一致。

## 边界和验证

本次只处理 LLM；不修改 VLMTrainer，不增加多模态组件配置，不新增 loss scale/跳步状态或参数范数扫描。Megatron 的这些可选诊断需要 optimizer/AMP 状态来源，不能靠日志层推断。本次保留可供 LLM 扩展命名诊断的统一接口。

```bash
python -m pytest -q tests/ut/trainer tests/torch/trainer/test_training_metrics_distributed.py
```

仓内覆盖手算结构公式、packed/CP 长度、真实 TextParallelBatch 接线、输出频率、loss/梯度不变以及 4 进程 Gloo 的 DP/CP/TP 去重。这是功能验证，不替代 NPU/GPU 训练吞吐基准。

参考 2026-10-11 核验的官方分支源码：

- [MF master PyNative LossCallback](https://gitcode.com/mindspore/mindformers/blob/d3f5dc943f6bf50cfa238db9c320ba4ff75ac52e/mindformers/pynative/callback/loss_callback.py) 与 [FLOPs 公式](https://gitcode.com/mindspore/mindformers/blob/d3f5dc943f6bf50cfa238db9c320ba4ff75ac52e/mindformers/models/utils.py)，`d3f5dc943f6bf50cfa238db9c320ba4ff75ac52e`。
- [Megatron-LM main training.py](https://github.com/NVIDIA/Megatron-LM/blob/a8a5977f7611e9a8aac205eabd2f9b43f786bc8f/megatron/training/training.py)，`a8a5977f7611e9a8aac205eabd2f9b43f786bc8f`。
