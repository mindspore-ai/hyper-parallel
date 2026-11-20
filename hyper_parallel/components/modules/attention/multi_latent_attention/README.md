# MLA

`MultiLatentAttention` 提供 DeepSeek 风格的多头潜变量注意力，支持从 `config` 直接构造或转换兼容的来源模块。

- 支持 expanded QKV 与 latent KV 两种表示；latent KV 在通信后展开所需头。Q 保持展开表示，仅在 Ulysses 子组内交换。
- 支持 Ulysses、KV AllGather、混合 CP，以及按因果文档需要选择性分发 KV。
- 支持前向和已实现路径的反向通算重叠。NoPE＋输出 gate 变体见 [gated MLA](../gated_multi_latent_attention/README.md)。

## YAML 配置

下面是 TP1、CP4、latent KV 纯 Ulysses 的配置片段，合入已有 MLA 模型训练配方：

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
        _target_: hyper_parallel.components.modules.MultiLatentAttention
        strategy: latent_kv_head
        overlap: true
```

`cp_size: 1` 使用 local 成员；大于 1 时选择 CP 成员。其余 DP/FSDP 拓扑与模型配置沿用完整配方。

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `strategy` | `expanded_ulysses` | expanded_ulysses 使用展开 QKV；latent_kv_head 通信压缩 KV，再展开所需头 |
| `ulysses_degree` | null | 省略/null 为纯 Ulysses；1 为 KV AG；其他整除 CP 大小的值为 Ulysses×AG |
| `overlap` | false | 开启该拓扑已支持的前向和反向调度 |
| `kv_exchange` | `all_gather` | AG/混合路径可改为 all_to_all_v，按因果文档需要选择性发送 KV |

令 `P=cp_size`、`U=ulysses_degree`，AG 子组大小为 `P/U`。
例如 CP4 下，省略 U 为 U4，U=1 为 AG4，U=2 为 U2×AG2。
strategy 选择通信表示，U 选择并行拓扑；虽然默认策略名含 ulysses，也可通过 U 选择 AG/混合。

## Latent CP 与前反向掩盖

`expanded_ulysses` 先展开 Q/K/V 再通信；`latent_kv_head` 传输压缩 KV 和共享 K 位置通道，
接收后只展开本 rank 负责的头，再交给 FA。Q 仍使用展开表示，在 Ulysses 子组内交换。
latent 可减少 KV 通信量，实际节省取决于 latent 宽度、头数和 CP 拓扑；FA 仍计算展开后的 Q/K/V。
混合 CP 同时按头和 query 区间分工，AG 维补齐各 query 区间所需的 KV。

`overlap: true` 同时启用该拓扑支持的前向和反向调度。纯 Ulysses、TP1 的主要窗口如下
（A2A 为 AllToAll，AG 为 AllGather，RS 为 ReduceScatter）：

| 路径 | 通信与计算怎样重叠 |
| --- | --- |
| expanded 前向 | 按 V → K → Q 提交 A2A；V 通信期间组装 K，K 通信期间计算 Q 投影与 RoPE，FA 前等待输入就绪 |
| latent 前向 | KV latent AG 与 Q 投影/RoPE 重叠；随后 Q A2A 与 KV 上投影、组装重叠 |
| expanded 反向 | 提前提交 dQ/dK/dV A2A；dQ 返回后计算 Q 分支梯度，覆盖尚未完成的 dK/dV 通信 |
| latent 反向 | dQ A2A 与 KV 分支反向重叠；KV latent 梯度的 SUM RS 与 KV 投影权重梯度计算重叠 |
| 输出反向 | 先算输出投影的输入梯度并提交 dO A2A，再计算输出投影权重梯度，FA 反向前等待 dO |

反向使用配对的通信提交/等待节点及投影梯度调度，FA 继续使用原有反向。
带 gate 时还能利用独立 gate 计算，具体窗口见 [gated MLA](../gated_multi_latent_attention/README.md#执行流程)。

## 文件职责

| 文件 | 作用 |
| --- | --- |
| [multi_latent_attention.py](multi_latent_attention.py) | MLA 参数构造与转换、投影/RoPE、FA 和共享 forward |
| [multi_latent_attention_family.py](multi_latent_attention_family.py) | 选择 local/CP 成员，声明布局并校验配置 |
| [cp_multi_latent_attention.py](cp_multi_latent_attention.py) | CP 拓扑、文档分区、输入调度和输出恢复 |
| [cp_mla_backward_overlap.py](cp_mla_backward_overlap.py) | 投影与通信的配对反向，以及输出通信重叠 |

## 执行流程

```text
输入准备 → 输入交换 → FA → 恢复本地 token → 输出投影
```

输入准备计算 down 投影与 Norm，得到 Q/KV latent 和共享 K。交换阶段按策略安排上投影与通信：
expanded 先展开再发送，latent KV 先通信再展开所需头。
数学方法和 forward 由算法类共享，CP 成员只负责通信与调度。
开启 overlap 时可提前提交已就绪的数据，不要求各阶段串行执行。

## 支持范围

- 普通 MLA 的 TP 仅支持纯 Ulysses，头数须能被 `TP×U` 整除；AG/混合目前要求 TP1。
- packed CP 当前要求 B1，非 packed 支持 B>1；CP 要求 dropout=0，不支持 KV cache 或隐式 sliding window。
- 纯 Ulysses 支持全局布尔 Q/K 可见性 mask；AG/混合要求 causal 且无显式 mask。
  packed 使用全局累计文档边界，不能与显式 mask 同时提供。默认因果路径不创建全局 S×S mask。
- RoPE 使用模型提供的本地 token 位置表，TND 文档边界不替代位置编码。MoME、parameter-sink 来源不受支持。
- all_to_all_v 仅用于存在 query 分区的 AG/混合路径，不能配置在纯 Ulysses 上。

| overlap 路径 | 已支持的调度 |
| --- | --- |
| 纯 Ulysses、TP1 | 上述输入前向、配对输入反向及输出反向调度 |
| AG/混合、TP1 | 输入前向重叠；混合 U>1 可复用输出反向调度，输入反向仍用原生 collective |
| 纯 Ulysses、TP>1 | 已支持的前向重叠，不使用 TP1 专用配对反向 |

这些是实现能力范围；实际收益取决于形状与设备，不保证所有通信完全隐藏。
