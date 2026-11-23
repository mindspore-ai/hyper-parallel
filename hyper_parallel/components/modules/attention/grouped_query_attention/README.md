# GQA

`GroupedQueryAttention` 提供分组查询注意力，支持从 `config` 直接构造或转换兼容的来源模块。

- Q 与 KV 使用各自的头数，默认融合 QKV 投影，也支持独立投影表示。
- 支持 Ulysses、KV AllGather、混合 CP，以及已就绪输入的通算重叠。
- 支持 packed 文档和非 packed 的 padding mask；带输出 gate 的变体见 [gated GQA](../gated_grouped_query_attention/README.md)。
- 显式选择 `RingGroupedQueryAttention` 可使用 Ring / Ulysses+Ring；它与默认 GQA family 分开，不更改已有配方的 CP 算法。

## Ring CP

在现有 Qwen3-MoE 配方的 attention replacement 中，将 `_target_` 改为 `RingGroupedQueryAttention`，并按设备数设置 CP：

```yaml
accelerator:
  cp_size: 4

plan_overrides:
  - match: "*.self_attn"
    replace:
      kind: module
      source_type: transformers.models.qwen3_moe.modeling_qwen3_moe.Qwen3MoeAttention
      implementation:
        _target_: hyper_parallel.components.modules.RingGroupedQueryAttention
        ulysses_degree: 1
        overlap: false
        load_balance: none
```

`ulysses_degree` 必须整除 `cp_size`；设为 1 选择纯 Ring，大于 1 时先做 Ulysses 交换再使用 Ring 子组。Ring 接受 dense 输入或带全局累计边界的 batch-one packed 输入，不支持显式 mask、dropout、KV cache 或 sliding window。基础算法、反向和输入布局要求见 [Ring CP 实现](../../../../distributed/context_parallel/ring_attention.py)。

## YAML 配置

下面是 TP1、CP4、U2×AG2 的配置片段，合入已有训练配方并保留其他 replacement：

```yaml
accelerator:
  tp_size: 1
  cp_size: 4
  sequence_parallel: false

plan_overrides:
  - match: "model.layers.*.self_attn"
    replace:
      kind: module
      implementation:
        _target_: hyper_parallel.components.modules.GroupedQueryAttention
        ulysses_degree: 2
        overlap: true
```

`cp_size: 1` 选择 local 成员；大于 1 时选择 CP 成员。DP/FSDP 等拓扑仍需与总设备数匹配。
完整示例见 [Qwen3-MoE packed 配方](../../../../models/qwen3_moe/recipes/train.yaml) 和
[dense 离线配方](../../../../../examples/training_demo/qwen3_moe/train_parallel_full_offline.yaml)。

令 `P=cp_size`、`U=ulysses_degree`，KV AllGather 子组大小为 `A=P/U`：

| ulysses_degree（P>1） | 并行方式 | CP4 示例 |
| --- | --- | --- |
| 省略、null 或 P | 纯 Ulysses | U4 |
| 1 | Q 保持本地，KV AllGather | AG4 |
| 其他整除 P 的正整数 U | Ulysses + KV AllGather | U2×AG2 |

`overlap` 默认 false；true 提前提交通信，与独立计算重叠，不改变投影参数表示。
普通 YAML 无需设置 `fuse_qkv`，默认融合 QKV。
现有 Qwen3-MoE 配方显式设置 U=1，因此只增大 cp_size 会选择 KV AG。

## 混合 CP 与通算掩盖

Ulysses 在 U 张卡之间用 AllToAll 将序列分片转换为头分片；KV AllGather 再沿 A 维补齐 KV 序列，
Q 仍按 query 区间分工。例如 U2×AG2 中，每个 rank 负责一半头、一半 query 序列，并取得对应头的完整 KV。
这样可以扩展 CP 规模，而无需把 KV 复制到 Q 的头数；头数整除约束作用于 TP×U，不是 TP×CP。

`overlap: true` 让已完成投影、Norm/RoPE 的输入立即通信，直到 FA 消费前才等待：

- Ulysses 的 Q/K/V 分别就绪后提交；后续输入的准备工作可覆盖已提交的通信。
  默认融合 QKV 已一次完成三路投影，可利用的主要窗口是剩余 Norm/RoPE 等处理；独立投影还可利用后续投影计算。
- 混合 CP 中，K/V 各自完成 Ulysses 交换后便提交其 AllGather，不先等待 Q，缩短串行等待。

GQA 的反向使用已有可微通信接口；这里主要优化输入前向调度，不包含 MLA 的投影反向配对调度。

## 文件职责

| 文件 | 作用 |
| --- | --- |
| [grouped_query_attention.py](grouped_query_attention.py) | 参数构造与转换、投影/Norm/RoPE、FA 和共享 forward |
| [grouped_query_attention_family.py](grouped_query_attention_family.py) | 选择 local/CP 成员，声明布局并校验配置 |
| [cp_grouped_query_attention.py](cp_grouped_query_attention.py) | CP 输入通信、文档与 mask 映射、输出恢复 |
| [ring_grouped_query_attention.py](ring_grouped_query_attention.py) | Ring / Ulysses+Ring attention 阶段和 packed 长度转换 |
| [ring_grouped_query_attention_family.py](ring_grouped_query_attention_family.py) | 独立 Ring family 入口，保持默认 GQA family 行为 |

## 执行流程

```text
输入准备 → 输入交换 → FA → 恢复本地 token → 输出投影
```

输入准备使用共享的投影、Norm 和 RoPE 数学方法。CP 成员负责通信和调度，local 成员不产生 CP 通信。
开启 overlap 后，准备阶段便可提交已就绪的数据；流程中的阶段不表示所有计算和通信串行执行。
TP/SP 由框架原有的布局重分布处理。

## 支持范围

- 输入及输出 hidden 均为本地 `[B,S,D]`。Q、KV 头数都须能被 `TP×U` 整除；当前不隐式复制 KV 头。
- packed 需要模型/数据入口传入全局文档累计边界和正确的位置表，不能同时使用显式 mask。
  TND 长度描述文档隔离；RoPE 仍使用模型提供的 cos/sin，在 FA 前执行。
- 非 packed 支持全局 `[B,Sglobal]` padding mask（True/1 为有效 token），以及 Q/K 可见性 mask。
- CP 是训练路径，拒绝非空 KV cache。overlap 不保证每项前反向通信都能完全隐藏。
