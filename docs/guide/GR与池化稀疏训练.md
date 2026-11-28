# GR 与池化稀疏训练

本页介绍 Qwen gated residual（GR）、QSA 和 GLM KPool 的基础组件。
数学模块与并行执行分开：本次提供可训练的算子和 indexer，不改变 Trainer、CP、EP 或优化器。
GR 可替换 HF 子模块；QSA/KPool 的完整模型、packed loss 和并行训练配方仍需单独接入。

## 能力与实现

| 部件 | PyTorch 参考 | 昇腾融合实现 | 返回值与梯度 |
| --- | --- | --- | --- |
| GR | 分组 offset RMSNorm、低秩 read、inject | `text_gated_residual` 及其官方 autograd 包装 | mixed、可选 injection weights；输入与参数均可导 |
| 池化路由 | 按 query 分块计算评分、选块 | `pool_key_indexer` | 原始 token 索引；离散选路不可导 |
| QSA 主 attention | 按 query 分块的 GQA | `generic_block_sparse_attention` / `generic_block_sparse_attention_grad` | output、detached LSE；Q/K/V 可导 |
| GLM 主 attention | absorbed MLA，共享 KV 的梯度在 FP32 累计 | `sparse_flash_mla` / `sparse_flash_mla_grad` 的 original-token sparse 模式 | output、detached LSE；Q 与共享 KV 可导 |
| 块级 teacher | 从相同路由和 LSE 重建选中 token 的概率，再聚合到块 | 当前为分块 PyTorch 实现 | detached teacher，不反传主模型 |
| indexer KL | 所选完整块上的 KL sum | `sparse_lightning_indexer_kl_loss_grad` | 标量 loss；index Q、pooled K、head weights 可导 |

模型模块在 `components/modules/gated_residual.py` 和 `components/modules/attention/pooled_indexer.py`。
算子契约及后端集中在 `components/ops/impl/` 下的同名文件，公共入口来自 `components.ops`。
`block_sparse_attention` 族包含独立 Q/K/V 的 `standard` 和共享 KV 的 `mla` 两个变体。

## 配置

```yaml
ops:
  gated_residual: cann
  pooled_indexer: cann
  block_sparse_attention: cann
  pooled_indexer_kl: cann
```

上述配置只控制已经调用公共算子接口的模块，不会自动改写整套 HF 模型。
省略配置时这些族选择 `native`；显式选择 `cann` 后，缺少依赖、符号或不支持的 shape 会报错，
不会在执行失败后静默回退。CANN 运行时、ops-transformer Python 包装、host tiling 和 device binary
必须配套。验证基于未经修改的 ops-transformer master；仅有旧 CANN 中的同名接口不代表能力一致。

## GR 接入

`GatedResidual(config=...)` 可从配置构造；`GatedResidual(module=hf_module)` 复用已加载参数对象，
保留 `hc_norm.weight`、`input_mix_weight_down/up.weight` 和 `block_inject_weight.weight` 的键名。
返回结构为 `mixed` 或 `(mixed, original_residual, injection_weights)`；decoder 负责子层和残差写回。
它不是 mHC Sinkhorn，也不是 attention 的 output gate。

例如 HF `Qwen4ExpForCausalLM` 的替换配置：

```yaml
plan_overrides:
  - match: "model.layers.*.attn_hyper_connection"
    replace:
      kind: module
      implementation:
        _target_: hyper_parallel.components.modules.GatedResidual
  - match: "model.layers.*.mlp_hyper_connection"
    replace:
      kind: module
      implementation:
        _target_: hyper_parallel.components.modules.GatedResidual
  - match: "model.hyper_connection_mixer"
    replace:
      kind: module
      implementation:
        _target_: hyper_parallel.components.modules.GatedResidual
```

GR 的融合训练支持 FP16/BF16、C=2..10、H 为 256 的倍数且不超过4096、R 为64的倍数且不超过640，
还需满足 workspace 约束。参数 scale 是 `1 + weight`，归一化在每条 H 维 residual stream 内进行。
融合版与 eager 的舍入顺序不同，不保证逐位一致。近零梯度经过 AdamW 归一化后，更新相对误差
可能比梯度误差明显；应结合 FP32 参考、实际 loss 和主参数更新评价，不将短程训练等同长期收敛。

## 池化路由与训练流程

```text
backbone hidden / Q latent（detach）
  → indexer 投影、Norm、池化 → index Q / pooled K / head weights
  → PoolKeyIndexer 评分与离散 TopK → 完整块 IDs + 当前 query 的可见尾块
                                         │
主 attention 的 Q/K/V 或 Q/shared KV ──────┤
  → 稀疏 attention → output + detached LSE ┘
  → 从相同的主 Q/K、路由、scale、LSE 聚合块级 teacher（detach）
  → pooled_indexer_kl → indexer 投影/Norm/池化参数梯度
```

`KPoolIndexer` 使用 token K 的 LayerNorm、gate+APE 的逐通道池化和 learned head weights；
当前要求 NoPE 与 always-select-tail。`QSAIndexer` 使用块平均、offset RMSNorm，以及块首位置的
partial RoPE。二者可从 config 构造或复用对应 HF indexer 的参数。

`forward` 返回不可导的原始 token 索引；训练用 `project` 和 `pool` 保留 indexer 参数梯度。
官方 PoolKeyIndexer 自带 `1/sqrt(D)` 评分缩放。其 tail 表示文档最终残余块，因此适配层另外生成
**每个 query 当前未满块的可见 token**。未超过预算的前缀选择全部可见完整块，无需评分。
池化 key 只用于 indexer 打分，不能替换主 attention 的原始 token KV。

主 attention 输入为 `[B,S,H,D]`，完整块 IDs 为 `[B,S,K]`，无效槽为 -1。
`block_sparse_attention(q,k,v,...)` 用于 QSA；`block_sparse_mla(q,kv,...)` 用于 GLM，
其中 `kv` 是 `[B,S,1,512]`。GLM 将 K 上投影吸收到 Q、V 上投影放到 attention 后，
不展开各头 K/V；softmax scale 仍按原 QK 维度定义，而非改成 `1/sqrt(512)`。
无 sink 通过官方 kernel 必需的 `-inf` sink 表达，其概率质量为零。

```python
from hyper_parallel.components.ops import block_sparse_mla, pooled_indexer_kl
from hyper_parallel.components.modules.attention.pooled_indexer import sparse_block_teacher

# q_absorbed / kv_latent 是主 attention 输入；block_ids 来自同一次 indexer 选路。
out, lse = block_sparse_mla(q_absorbed, kv_latent, block_ids, block_size=16, scale=256**-0.5)
teacher = sparse_block_teacher(q_absorbed, kv_latent, lse, block_ids, pool_size=16, scale=256**-0.5)
kl_sum = pooled_indexer_kl(index_q, pooled_k, head_weights, block_ids, teacher, pool_size=16)
loss = lm_loss + kl_coefficient * kl_sum / valid_query_count
```

`pooled_indexer_kl` 返回 sum，调用方负责系数、有效 query 数和分布式归一化。
无完整块的 query 对该目标贡献为零；强制尾块参与主 attention，但不加入本目标的候选归一化。
teacher 是**所选完整块上的归一化分布**，不是全量 dense teacher，不能替代独立定义的 dense warmup。
这是一种显式的稀疏训练目标，不声称已经复现 GLM/Qwen 的完整预训练配方。

## 融合限制与验证范围

- PoolKeyIndexer：D128，FP16/BF16，4/8/16/32/64 个 index heads，pool=1..128。
  token 预算需能被 pool 整除；不超过2048，或为3072/4096/5120/6144/7168/8192。
  原生与融合 TopK 对并列分数可能选择不同块，比较应使用集合并区分固定路由与独立路由。
- QSA 主 attention：D≤256、Q heads 能被 KV heads 整除，GQA group≤256；pool≤128，
  group>128 时 pool≤64。QSAG 前后向使用不同的 pattern 布局，适配层统一其语义。
- GLM 主 attention：共享一个 D512 KV head，Q heads≤128，FP16/BF16；只使用官方 original-token 模式。
  Q、KV 在 kernel 入口连续化；token IDs 的展开不是 KV 张量展开。
- KL：保持实际1..8192块容量，不能填到512等档位来改变推断的预算。QSA 的4个 index heads
  在 KL 入口补零到8个，保持打分不变。FP16 使用 loss scaling，并同时检查很小的梯度。
- indexer 模块支持无 padding 的 BSH 和累计长度表示的 packed 文档，文档不能跨 batch 行。
  主 attention/teacher/KL 当前按独立的 BSH 序列调用；不能把多个文档简单拼成一条序列后调用。
  整条 packed 训练、padding、TP/CP/EP/FSDP 归一化、编译、高阶导数和长程收敛尚不在本次验收范围。
- native 参考按 query 分块限制临时矩阵；训练 autograd 仍可能保存多个分块，不是长序列节省显存方案。

共享用例覆盖 native 公式、输出/梯度、teacher 隔离与真实后端。真实模型证据按场景记录在 PR：
GR 为裁剪 HF 两层训练；QSA 为原 attention 宽度的单层饱和预算训练；GLM 为保留原 attention
头维度但裁剪 hidden/down 宽度的共享 D512 单层训练，均不能推断为完整模型已验收。

64头共享 KV 的融合梯度使用局部绝对容差：BF16 为0.025（原0.02），FP16为0.004（原0.003）；
relative L2、rtol、输出、其他梯度与 native/oracle 检查均不放宽。
这是针对多头累加后少数抵消坐标的舍入误差；原始超限数、FP32定位及训练对照在 PR 中保留。

参考：[官方 GR](https://atomgit.com/cann/ops-transformer/tree/master/mhc/text_gated_residual)、
[PoolKeyIndexer](https://atomgit.com/cann/ops-transformer/tree/master/attention/pool_key_indexer)、
[SparseFlashMla](https://atomgit.com/cann/ops-transformer/tree/master/attention/sparse_flash_mla)、
[MindSpeed-MM QSA](https://github.com/Ascend/MindSpeed-MM/tree/master/mindspeed_mm/fsdp/ops/qsa)。
