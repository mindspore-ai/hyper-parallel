# Chunk Loss 使用与 Qwen3-MoE 接入指南

Chunk Loss 用于降低大词表语言模型在输出投影和交叉熵阶段的峰值显存。它沿当前 rank 的
本地序列维分块执行 LM Head、FP32 交叉熵和一阶梯度计算，任何时刻都不会生成完整的
`[batch, local_sequence, vocabulary]` logits。

当前实现包含三层：

- `components/losses/chunked_cross_entropy.py`：与模型无关的分块线性交叉熵；
- `ChunkedCausalLMLoss` 和 Trainer 接线：负责标签交接、局部归一化与模型绑定；
- `models/qwen3_moe/adapter/chunk_loss.py`：Qwen3-MoE 在完整 LM Head 之前的专用适配。

## 1. 为什么必须在模型 forward 内接入

普通 causal-LM 前向先执行：

```text
final hidden [B, S_local, H]
    -> lm_head
full logits [B, S_local, V]
    -> FP32 cross entropy
loss
```

如果只在 `BaseTrainer.postforward()` 中新增一个 loss，模型已经生成了完整 logits，峰值显存已经
发生，事后删除 logits 不能消除该峰值。因此，启用 Chunk Loss 时，Trainer 会先把 loss-only
标签转换为模型输入协议；模型族 adapter 再在 root model 的 forward 内取得 final hidden，直接
调用分块 loss，并跳过原来的完整 `lm_head(hidden_states)`。

Qwen3-MoE adapter 返回 `ChunkedCausalLMOutput(logits=None)`。`ChunkedCausalLMLoss` 会再次检查
`logits is None`，若模型错误地走回完整 logits 路径会立即报错，而不是静默退化。

## 2. 输入、输出与显存复杂度

通用接口如下：

```python
from hyper_parallel.components.losses import chunked_cross_entropy

loss_sum = chunked_cross_entropy(
    hidden_states,
    targets,
    lm_head.weight,
    chunk_size=1024,
    ignore_index=-100,
)
```

| 参数 | 形状或类型 | 含义 |
| --- | --- | --- |
| `hidden_states` | `[B, S_local, H]`, floating | 当前 rank 的 final hidden |
| `targets` | `[B, S_local]`, `torch.long` | 已和 hidden 对齐的目标 token |
| `lm_head.weight` | `[V, H]`, floating | 当前 root forward 中可计算的完整 LM Head 权重 |
| `chunk_size` | positive `int` | 单次处理的最大本地序列长度 `C` |
| `ignore_index` | `int` | 不参与交叉熵的目标值 |

接口返回图连接的 FP32 标量 `loss_sum`，不在内部做 DP、CP、micro-batch 或 token 数归一化。

普通路径的主要输出临时量随 `B * S_local * V` 增长。Chunk Loss 把这一项限制为
`B * C * V`，但仍保留训练必须交还上游的 `dHidden [B, S_local, H]` 和累计的
`dWeight [V, H]`：

| 项目 | 普通完整 logits | Chunk Loss |
| --- | --- | --- |
| 单次 logits 范围 | `[B, S_local, V]` | `[B, C, V]` |
| 交叉熵主要临时量 | `O(B * S_local * V)` | `O(B * C * V)` |
| outer backward 前保留 | 完整 logits 图及必要激活 | `dHidden`、累计 `dWeight` |
| 参数/FSDP 梯度 | 必须存在 | 仍必须存在 |

因此它解决的是大词表输出阶段随序列长度增长的临时峰值，不会把模型参数、优化器状态、
FSDP all-gather 权重或最终参数梯度变成零。

## 3. 完整前反向流程

每个本地序列块执行以下过程：

1. 取 `hidden_chunk [B, C, H]` 和对应 `target_chunk [B, C]`；
2. 计算 `hidden_chunk @ weight.T`，只生成 `[B, C, V]` logits；
3. logits 转 FP32，执行 `reduction="sum"` 的交叉熵；
4. 用 `torch.func.grad_and_value` 同时得到该块的 loss、`dHidden_chunk` 和 `dWeight_chunk`；
5. 把 `dHidden_chunk` 写入完整本地 `dHidden` 的对应区间，并把 `dWeight_chunk` 加到一个
   `[V, H]` accumulator；
6. 释放该块 logits 和块级梯度后再进入下一块。

自定义 autograd Function 的 outer backward 不会重新生成 logits。它把保存的 `dHidden`、
`dWeight` 乘以上游标量，分别交回 decoder graph 与 LM Head 参数。这样支持 Trainer 的 token
加权和梯度累积，同时避免保存完整 logits graph。

该实现保证一阶梯度并显式拒绝二阶梯度。首版面向全参数训练，固定分配 `dHidden` 和
`dWeight` 两个 accumulator，不为冻结 decoder 或冻结 LM Head 增加分支。accumulator dtype
跟随输入和权重 dtype，交叉熵数值计算使用 FP32。

## 4. Trainer 配置

在 `TrainerConfig` 根节点显式配置 loss：

```yaml
loss_fn:
  _target_: hyper_parallel.components.losses.ChunkedCausalLMLoss
  chunk_size: 1024
  ignore_index: -100
```

`BaseTrainer._build_loss()` 在模型完成替换、分片、FSDP、加载和 dtype 转换后构建该模块，并调用
它的 `bind_model(model, distributed_setup)`。Qwen3-MoE 的 provider 通过模型 adapter registry
自动解析，用户不需要在 recipe 中手写 forward patch。

每个 micro-batch 的 Trainer 流程是：

1. batch adapter 返回独立的 `model_inputs` 与 `loss_inputs`；
2. `ChunkedCausalLMLoss.prepare_model_inputs()` 从 loss 输入读取预先生成的 `shift_labels` 和 loss mask；
3. 从模型输入移除普通 `labels`、`shift_labels`，避免 Qwen 自己计算 eager loss；
4. 增加显式的 `chunk_loss_*` 参数并调用模型；
5. Qwen adapter 返回 `loss_sum`、`valid_token_count`、可选 MoE aux loss，且 `logits=None`；
6. loss 模块先形成当前 rank 的有效 token mean；
7. 现有 `mean_global_loss()` 再完成 DP/CP 与 micro-batch 的全局 token 加权；
8. backward 把预计算梯度交回普通 autograd 和 FSDP。

普通 `ModelOutputLoss` 以及只实现 `forward(model_output=..., labels=...)` 的自定义 loss 不受影响。
`bind_model` 和 `prepare_model_inputs` 是 model-integrated loss 才使用的可选生命周期接口。

通用 `chunked_cross_entropy` primitive 允许自定义 `ignore_index`。Trainer 的全局 token 计数目前
遵循数据层统一值 `-100`，所以 `ChunkedCausalLMLoss` 会在 bind 阶段拒绝其他值，防止局部 loss
和 DP/CP token 权重统计不同步。

## 5. 预先 shifted 的 labels 契约

首版只接受 `shift_labels`。该 tensor 已经表示每个 hidden 位置要预测的下一个 token：

```text
hidden[:, 0:S, :] <-> shift_labels[:, 0:S]
```

实现保留全部本地 hidden 位置，再用 `loss_mask` 把无效目标覆盖为 `ignore_index`。数据路径必须
在 CP 切分前完成 next-token shift，再把 input 和 target 一起切到各 rank；这样 CP=1 与 CP>1
使用同一语义，也不会丢失 rank 边界目标。只提供普通 `labels` 会直接报错，避免 CP=1 静默
二次 shift 或 CP>1 静默漏算边界 token。

## 6. Qwen3-MoE 专用行为

Qwen adapter 只在输入包含 `chunk_loss_targets` 时启用。普通推理或未配置 Chunk Loss 的训练仍调用
原始 Qwen forward。

Chunk Loss 路径按以下顺序执行：

1. 调用 `model.model(...)` 得到 `last_hidden_state`；
2. 校验 final hidden 与预先 shifted targets 对齐，并应用 loss mask；
3. 直接以 `model.lm_head.weight` 调用 `chunked_cross_entropy`；
4. 若 `output_router_logits=true`，复用 Transformers 的 load-balancing loss，并保留原
   `router_aux_loss_coef`；
5. 返回 `ChunkedCausalLMOutput`，明确令 `logits=None`。

训练路径要求 `use_cache=false`、`return_dict=true` 和 `logits_to_keep=0`。LM Head 当前必须是无
bias 的 `torch.nn.Linear`。adapter 的绑定是幂等的，未收到 Chunk Loss 参数时会回退原 forward。

## 7. FSDP、CP、EP 与当前限制

第一版支持 TP=1、PP=1、loss parallel 关闭时的普通 DP/FSDP、CP 和 EP 组合。以下边界会在首个
forward 之前 fail fast：

- TP>1：词表或 hidden 维分片需要分布式 softmax/交叉熵，不能直接把本地 shard 当完整权重；
- PP>1：最后 stage 的 loss 输入、token 归一化与 stage 输出协议尚未接入；
- loss parallel：会和当前完整词表 Chunk Loss 重复定义 logits/softmax 布局。

FSDP root forward 覆盖 Qwen decoder 和 LM Head；Chunk Loss 在该 forward 返回前使用 unshard 后
的 `lm_head.weight`。它产生的 `dWeight` 仍作为该 Parameter 的梯度交给既有 FSDP
reduce-scatter/reshard，不复制参数，也不绕开 FSDP 的梯度管理。

当前还不支持带 bias、LoRA/DoRA/QLoRA 或自定义投影语义的 LM Head，也不承诺
`torch.compile(fullgraph=true)` 下的图捕获。

## 8. 调优

较小的 `chunk_size` 降低块级 logits 峰值，但增加 LM Head、交叉熵和 Python 调度次数；较大的值
通常更快但峰值更高。建议从 1024 开始，再按词表大小、本地序列长度和设备实测选择。
