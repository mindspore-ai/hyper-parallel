# Gated GQA

`GatedGroupedQueryAttention` 实现 Qwen3.5 风格的逐 query 通道 sigmoid 输出 gate，保留 Q/K Norm 和 RoPE。
gate 与 QKV 使用融合投影，CP 复用 [GQA](../grouped_query_attention/README.md) 的通信流程。
支持 `config` 原生构造和兼容来源转换，仅替换 full-attention 层，Gated DeltaNet 继续使用原实现。

## YAML 配置

以下片段为 Qwen3.5 full attention 选择 TP1、CP4、KV AllGather；保留模型原配方的其他 replacement：

```yaml
accelerator:
  tp_size: 1
  cp_size: 4
  sequence_parallel: false

plan_overrides:
  - match: "*.self_attn"
    replace:
      kind: module
      source_type: transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5Attention
      exact_type: true
      implementation:
        _target_: hyper_parallel.components.modules.GatedGroupedQueryAttention
        ulysses_degree: 1
        overlap: true
```

`ulysses_degree` 和 `overlap` 沿用 [GQA 配置](../grouped_query_attention/README.md#yaml-配置)：
省略 U 为纯 Ulysses，U=1 为 KV AG，其余整除值为混合；overlap 默认 false。
CP1 选择 local 成员。gate 始终存在，不需要额外开关。
Qwen3.5 MoE 应使用其对应的 source_type；完整示例见
[稠密模型配方](../../../../models/qwen3_8/recipes/train_online_p2p_kv_allgather.yaml) 和
[MoE 配方](../../../../models/qwen3_6_moe/recipes/train_online_p2p_kv_allgather.yaml)。

## 文件职责

| 文件 | 作用 |
| --- | --- |
| [gated_grouped_query_attention.py](gated_grouped_query_attention.py) | 带 gate 的参数布局、checkpoint 转换与输出数学 |
| [gated_grouped_query_attention_family.py](gated_grouped_query_attention_family.py) | local/CP 选型，复用 GQA 布局和能力校验 |
| [cp_gated_grouped_query_attention.py](cp_gated_grouped_query_attention.py) | 组合 GQA 的 CP 流程与 gated 数学实现 |

## 执行流程

```text
QKV/gate 投影、QK Norm/RoPE → 输入交换 → FA → 恢复本地 O → 乘 sigmoid(gate) → 输出投影
```

gate 保持本地 token 顺序，在 O 恢复后逐通道相乘。CP 的 overlap 沿用 GQA 输入调度；
gate 来自融合投影，不使用 gated MLA 的独立 gate 投影调度。

## 支持范围

- 当前要求 Q/K 与 V 的 head width 相等；Q/KV 头数均须能被 `TP×U` 整除。
- packed、padding mask、位置表和 CP cache 限制沿用 [GQA 支持范围](../grouped_query_attention/README.md#支持范围)。
- 此模块处理 Qwen3.5 full attention；完整混合模型中的 GDN 和其他层需要各自的配方支持。
