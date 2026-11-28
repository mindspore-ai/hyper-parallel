# Gated MLA

`GatedMultiLatentAttention` 实现 Kimi 风格的 NoPE＋独立 sigmoid 输出 gate，支持 `config` 原生构造和兼容来源转换。
它复用 [MLA](../multi_latent_attention/README.md) 的低秩投影、expanded/latent KV 表示与 CP 拓扑，并调度 gate 参与通算重叠。
NoPE 与 gate 是该算法的固定语义，不能用它直接替换需要 RoPE 的普通 DeepSeek attention。

## YAML 配置

以下片段用于兼容的 NoPE＋gate 模型，选择 TP1、CP4、latent KV 纯 Ulysses：

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
        _target_: hyper_parallel.components.modules.GatedMultiLatentAttention
        strategy: latent_kv_head
        overlap: true
```

strategy、ulysses_degree、overlap 和 kv_exchange 的选择及默认值见 [MLA 配置](../multi_latent_attention/README.md#yaml-配置)。
CP1 使用 local 成员，仍执行 gate；overlap 开关只改变调度，不决定是否有 gate。

K3 的 Tensor 返回协议使用同一个公共入口，在 implementation 中设置 `source_contract: tensor`；
默认 tuple 协议仍返回 `(output, attention_weights)`。tensor 协议接受 K3 的 mask、position_ids、cache
参数顺序，NoPE 忽略位置编码，CP 仍拒绝实际 cache。旧的 `models.kimi_k3.adapter.mla` 入口已移除。
需要保留独立 down 参数时再设置 `fuse_latent_qkv: false`，不必新增 K3 专用 CP wrapper。

## 文件职责

| 文件 | 作用 |
| --- | --- |
| [gated_multi_latent_attention.py](gated_multi_latent_attention.py) | NoPE、独立 gate 投影、sigmoid 与输出乘法 |
| [gated_multi_latent_attention_family.py](gated_multi_latent_attention_family.py) | local/CP 选型与 TP1 能力校验，复用 MLA 布局 |
| [cp_gated_multi_latent_attention.py](cp_gated_multi_latent_attention.py) | 将 gate 安排到输入或输出通信窗口 |
| [cp_gated_mla_backward_overlap.py](cp_gated_mla_backward_overlap.py) | gated 输出联合反向与输入侧 gate 梯度调度 |

## 执行流程

```text
输入准备 → 输入交换 → FA → 恢复本地 O → 乘 gate → 输出投影
G = sigmoid(g_proj(X))
Y = o_proj(flatten(O) × G)
```

NoPE 跳过位置旋转，保留共享 K 通道和 checkpoint 宽度。
开启 overlap 后，expanded 纯 Ulysses 将 gate 计算放入输入通信窗口；
latent/混合的 U>1 路径利用 O 返回通信窗口计算 gate。

## 支持范围

- 当前要求 TP1；packed CP 要求 B1，非 packed 支持 B>1，CP 要求 dropout=0。
- 文档边界、mask、KV 交换及 cache 限制沿用 [MLA 支持范围](../multi_latent_attention/README.md#支持范围)。
- 纯 Ulysses 支持已实现的配对输入/输出反向；AG/混合不具备同样的完整输入反向调度。
- overlap=false 或 U=1 时不启用上述 gate 通信窗口；实际收益取决于计算和通信的相对耗时。
