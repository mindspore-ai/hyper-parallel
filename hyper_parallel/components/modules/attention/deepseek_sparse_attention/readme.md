# DeepSeek Sparse Attention

`DeepseekSparseAttention` 提供 DeepSeek-V3.2 的稀疏注意力，使用 Indexer 为每个 query
选择参与计算的 key。训练配方通过替换完整的注意力模块接入，支持单卡和 KV AllGather 上下文并行。

## YAML 配置

下面是 TP1、CP2 的配置片段，合入已有模型训练配方；完整示例见
[V3.2 并行配方](../../../../models/deepseek_v32/recipes/train_parallel_offline.yaml)。

```yaml
accelerator:
  tp_size: 1
  cp_size: 2
  sequence_parallel: false

plan_overrides:
  - match: "*.self_attn"
    replace:
      kind: module
      implementation:
        _target_: hyper_parallel.components.modules.DeepseekSparseAttention
        kv_cp_size: 2
        kv_cp_strategy: allgather
        cp_comm_overlap: true
        dense_warmup: null
```

启用 MTP 跨 step 共享时，在支持 MTP 的模型配方中单独设置模型 config：

```yaml
model:
  config_overrides:
    index_share_for_mtp_iteration: true
```

该字段属于模型 config，不属于 attention replacement 的 `implementation` 参数。每个 MTP DSA
attention 的 config 必须将 `layer_idx` 标记为 MTP 层（`layer_idx >= num_hidden_layers`），MTP 模型
adapter 需通过 `MultiTokenPrediction` 创建每步 attention 路径一致的 decoder；否则配置会在构建时或
forward 时失败，不会静默退回共享。通用组件当前不负责把被 DeepSeek-V3.2 adapter 忽略的 MTP checkpoint
加载回来，也不提供 serving 引擎的 request cache 生命周期。

这些构造选项写在 `replace.implementation` 下：

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `kv_cp_size` | `1` | 上下文并行大小，与 `accelerator.cp_size` 保持一致；1 表示不开启 CP。 |
| `kv_cp_strategy` | `allgather` | KV 通信策略，当前仅支持 `allgather`。 |
| `cp_comm_overlap` | `false` | 是否启用异步 KV 收集。 |
| `dense_warmup` | `null` | `null` 跟随模型的 `dsa_dense_warm_up`；`true`/`false` 固定 dense warm-up 或 sparse 阶段。 |

TP 度和总 CP 度仍写在 `accelerator` 中，改 `accelerator.cp_size` 时须同步改 `kv_cp_size`。
`*.self_attn` 匹配完整的注意力模块，框架自动选择本地或上下文并行实现；
通用规则见 [Attention YAML 配置通用规则](../readme.md#yaml-配置通用规则)。

## 参数切分：sparse_attention_sharding

[`sparse_attention_sharding`](deepseek_sparse_attention_family.py) 声明参数和输入输出的并行布局。
下面是 TP > 1 时的参数切分方式，线性层权重的形状为 `[输出维, 输入维]`：

| 参数 | TP 布局 | 切分方式 |
| --- | --- | --- |
| `q_b_proj`、`kv_b_proj` | `Shard(0)` | 按输出维切分，每张卡负责部分主注意力头；若有 bias 则同样切分 |
| `o_proj.weight` | `Shard(1)` | 按输入维切分，与本卡负责的注意力头对应 |
| `o_proj.bias`（若有） | `Replicate()` | 在 TP 组内复制 |
| `linear_qkv`、`q_a_layernorm`、`kv_a_layernorm` 的全部参数 | `Replicate()` | 低秩投影和归一化保持完整 |
| `indexer.wq_b`、`indexer.wk`、`indexer.k_norm`、`indexer.weights_proj` 的全部参数 | `Replicate()` | Indexer 在 TP 组内复制，保留完整的 Indexer 头 |

CP 按序列维切分输入和输出，不切分参数。例如 TP2 × CP2 时，每张卡负责一半主注意力头和
一半 query token，Indexer 仍保留完整的头数。TP1 时所有参数都不做 TP 切分。

表中描述的是计算布局；启用 FSDP 后，参数的存储分片仍由 FSDP 管理。

## V3.2 Indexer Share

V3.2 local/CP member 支持 `dsa_indexer_types`、`dsa_index_topk_freq`、
`dsa_index_skip_topk_offset` 和 `dsa_index_share_loss`。Shared 层复用前方 Full 层的选择，
不含自身 Indexer 参数。MTP decoder 通过 `MultiTokenPrediction` 接入时，训练路径支持
`index_share_for_mtp_iteration: true`：首步发布 Top-K 与 KV，后续 step 复用首步结果并跳过
Indexer/KV 投影；配置写在 `model.config_overrides`。该实现不包含推理引擎的 request cache、
speculative rollback/compaction 接入。CP consumer 复用 producer 已收集的 bank，但要求等长非 packed
输入和 adapter 提供的逐步位置；CP packed/TND 会 fail-fast。边界、配置和状态语义见
[MTP IndexShare 设计](../../../../../docs/design/dsa_indexer_share.md#11-mtp-indexshare-kvshare-设计)。
CP 下 Full 收集三路 key bank，Shared 只收集本层 main KV/RoPE key；
具体配置、边界和验证见 [Indexer Share 设计](../../../../../docs/design/dsa_indexer_share.md)。
