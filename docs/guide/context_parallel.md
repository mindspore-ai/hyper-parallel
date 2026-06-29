# CP 上下文并行使用指南

HyperParallel 的 Context Parallel（CP）是面向注意力计算边界的 `ParallelStyle`。它通过 forward hooks 把每个 CP rank 上的本地序列分片包装为 DTensor，并在进入/离开 core attention 或 DSA 算子边界时完成 Q/K/V 的重排与通信。

CP 不负责切分数据集、token ids 或整层 Transformer 输入。启用 CP 后，调用模型前仍需要让每个 CP rank 只拿到自己的序列切片，并保证 position ids、RoPE 起始位置和 attention mask 与全局序列窗口对齐。

## 适用边界

| 接口 | 真实作用 |
|------|----------|
| `ContextParallel` | 给普通 core attention 注册同步 CP hooks。默认处理 forward 参数中的 Q/K/V。 |
| `AsyncContextParallel` | 在普通 core attention 上注册异步 CP hooks；传入 Q/K/V producer 模块时，可把投影计算与 CP 通信重叠。 |
| `DSAIndexerContextParallel` | 给 DeepSeek Sparse Attention（DSA）的 indexer 边界改写 query/key/weights 的 DTensor placement。 |
| `DSASparseAttentionContextParallel` | 给 DSA sparse attention 边界改写 query/key/value/topk/rope 的 placement。 |
| `DSAIndexerLossContextParallel` | 给 DSA indexer-loss 边界改写输入 placement，并处理部分 loss/grad 输出。 |
| `AsyncDSA*ContextParallel` | DSA 边界的异步 all-gather 版本；提供 handoff 模块时提前发起 key/value 侧 gather，否则回退到同步 gather。 |

## `ContextParallel` 模式

`ContextParallel` 由 `ulysses_degree` 决定执行模式。CP size 等于传入 `device_mesh.mesh.numel()`。

| 模式 | 参数 | 通信与 placement | 约束 |
|------|------|------------------|------|
| Pure Ulysses | `ulysses_degree=None`，也是默认值 | Q/K/V 在进入 attention 前从 `Shard(seq_dim)` all-to-all 到 `Shard(head_dim)`，输出再从 head shard 转回 sequence shard。 | `num_heads % cp_size == 0` |
| Pure Colossal AI | `ulysses_degree=1` | Q 保持 `Shard(seq_dim)`，K/V 沿序列维 all-gather 到 `Replicate()`。 | 无 head 数整除 CP 的约束 |
| Hybrid | `1 < ulysses_degree < cp_size` | 先在 Ulysses 子组上对 Q/K/V 做 seq->head all-to-all，再在 Colossal 子组上 gather K/V。 | `cp_size % ulysses_degree == 0`，且 `num_heads % ulysses_degree == 0` |

常用参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `seq_dim` | `1` | 序列维下标。BSHD 使用 `1`，BNSD 使用 `2`。 |
| `head_dim` | `2` | head 维下标。BSHD 使用 `2`，BNSD 使用 `1`。 |
| `qkv_indices` | `(0, 1, 2)` | attention forward 中 Q/K/V 的位置参数下标。 |
| `qkv_kwarg_names` | `()` | Q/K/V 以关键字参数传入时的参数名。 |
| `use_local_output` | `False` | 为 `True` 时 public output 总是 local tensor；为 `False` 时 plain tensor 输入返回 local tensor，CP DTensor 输入保持 CP DTensor，TP 等非 CP DTensor 输入会在输出处去掉 CP 维并保留原来的非 CP layout。 |
| `load_balance` | `False` | 仅支持 `ulysses_degree=1`。使用 Head-Tail Q 交换做 Colossal CP 负载均衡。 |

Hybrid CP 可以直接传 1-D CP mesh；如果传 2-D mesh，mesh 维度需要命名，且第一维按 Colossal 子组、第二维按 Ulysses 子组使用，通常写作 `mesh_dim_names=("co", "ds")`。

`load_balance=True` 会把本地 Q 分成两半并与对端 rank 做 P2P 交换，内部会调用两次 attention。此时 attention 内部如果要构造 mask，应使用 K 的序列长度作为全局长度来源；Q 在子 attention 中看到的是半段序列。

## 基础用法

以 BSHD core attention 为例，每个 CP rank 先拿到本地序列切片，再把 CP style 挂到真正接收 Q/K/V 的 attention 子模块上：

```python
from hyper_parallel import ContextParallel, init_device_mesh

mesh = init_device_mesh("npu", (cp_size,), mesh_dim_names=("cp",))
cp_rank = mesh.get_local_rank("cp")

local_s = seq_len // cp_size
seq_slice = slice(cp_rank * local_s, (cp_rank + 1) * local_s)
local_tokens = global_tokens[:, seq_slice]

cp = ContextParallel(seq_dim=1, head_dim=2, ulysses_degree=1)
cp.apply(layer.attention.sdpa_core, mesh)
```

如果使用 `parallelize_module`，也可以把 CP 写进并行计划：

```python
from hyper_parallel import ContextParallel, parallelize_module

parallelize_module(
    model,
    mesh["cp"],
    {"layers.*.attention.sdpa_core": ContextParallel(seq_dim=1, head_dim=2, ulysses_degree=1)},
)
```

不同布局和模式示例：

```python
# BSHD: [batch, seq, head, dim]，Colossal CP
ContextParallel(seq_dim=1, head_dim=2, ulysses_degree=1).apply(core_attn, cp_mesh)

# BNSD: [batch, head, seq, dim]，Pure Ulysses
ContextParallel(seq_dim=2, head_dim=1).apply(core_attn, cp_mesh)

# Hybrid CP，Ulysses 子组大小为 2
ContextParallel(seq_dim=1, head_dim=2, ulysses_degree=2).apply(core_attn, cp_mesh)
```

## AsyncContextParallel

`AsyncContextParallel` 的构造参数与 `ContextParallel` 基本一致，但 `apply` 额外接收 `q_proj`、`k_proj`、`v_proj`：

```python
from hyper_parallel import AsyncContextParallel

AsyncContextParallel(seq_dim=1, head_dim=2, ulysses_degree=None).apply(
    layer.attention.sdpa_core,
    cp_mesh,
    q_proj=layer.attention.q_proj,
    k_proj=layer.attention.k_proj,
    v_proj=layer.attention.v_proj,
)
```

真实行为如下：

| 模式 | 异步重叠点 |
|------|------------|
| Pure Ulysses | Q/K/V projection 后发起 seq->head all-to-all，attention pre-hook 等待。 |
| Pure Colossal AI | K/V projection 后发起 sequence all-gather，attention pre-hook 等待；Q 仍在 attention 边界包装为 sequence shard。 |
| Hybrid | Q/K/V projection 后发起 Ulysses all-to-all；K/V 在 Colossal 子组上的 gather 仍在 attention 边界同步执行。 |

注意事项：

1. 如果 `q_proj`、`k_proj`、`v_proj` 任意一个未传入，`AsyncContextParallel.apply` 会回退为同步 `ContextParallel.apply`。
2. 传入的 projection 模块必须是 Q/K/V 进入 attention 前的最后一个可 hook 模块；projection 和 attention 之间如果还有 reshape、transpose、QK norm 等操作，异步 pre-hook 会绕过这些操作。带 QK norm 的模型应把 norm 后的模块作为 `q_proj`/`k_proj` handoff。
3. `ulysses_degree=1` 且 `load_balance=True` 时，async 版本会回退到同步 Colossal CP。

## DSA CP

DSA 这里指 DeepSeek Sparse Attention。DSA CP 不是通用 attention 的替代实现，而是给 DSA 相关算子边界准备 DTensor placement，使 `lightning_indexer`、`npu_sparse_flash_attention` 和 indexer-loss 自定义算子能够按 CP layout 分发。

DSA CP 当前只支持 `mode="colossal"`，`mode="ulysses"` 会直接报错。支持的输入布局为：

| `layout` | 序列维 |
|----------|--------|
| `"BSND"` | `seq_dim=1` |
| `"TND"` | `seq_dim=0` |

DSA style 会把传入的 `device_mesh` 当作 1-D CP mesh 使用；如果传入多维 mesh，会先按 rank list flatten。DSA CP 不走 Ulysses/Hybrid head sharding。

各 DSA style 默认按位置参数改写输入，也可以通过 `*_kwarg_name` 适配关键字参数。

| Style | 默认 forward 边界 | Query 侧 placement | Key 侧 placement |
|-------|-------------------|--------------------|------------------|
| `DSAIndexerContextParallel` | `(query, key, weights, ...)` | `query`、`weights` -> `Shard(seq)` | `key` -> `Replicate()` |
| `DSASparseAttentionContextParallel` | `(query, key, value, topk_indices, query_rope, key_rope, ...)` | `query`、`topk_indices`、`query_rope` -> `Shard(seq)` | `key`、`value`、`key_rope` -> `Replicate()` |
| `DSAIndexerLossContextParallel` | `(query, key, query_index, key_index, weights, topk_indices, softmax_max, softmax_sum, query_rope, key_rope, ...)` | `query`、`query_index`、`weights`、`topk_indices`、`query_rope` -> `Shard(seq)` | `key`、`key_index`、`key_rope` -> `Replicate()` |

同步 DSA 用法：

```python
from hyper_parallel import DSASparseAttentionContextParallel

DSASparseAttentionContextParallel(layout="BSND").apply(sparse_attention_boundary, cp_mesh)
```

异步 DSA 用法：

```python
from hyper_parallel import AsyncDSASparseAttentionContextParallel

AsyncDSASparseAttentionContextParallel(layout="BSND").apply(
    sparse_attention_boundary,
    cp_mesh,
    key_handoff=key_handoff_module,
    value_handoff=value_handoff_module,
    key_rope_handoff=key_rope_handoff_module,
)
```

异步 DSA 的 handoff 模块用于提前发起 key/value 侧 sequence all-gather。没有传对应 handoff 时，该输入会在 consumer 边界走同步 gather，输出 placement 与同步 DSA style 保持一致。

## 与 TP/FSDP/PP 组合

CP 应使用多维 mesh 中的 CP 子 mesh；TP、FSDP、PP 分别使用自己的子 mesh。Llama3 示例中通常先在 TP mesh 上对 projection/MLP 做 TP，再把 CP 挂到每层 attention 的 `sdpa_core`。

```python
from hyper_parallel import ContextParallel, init_device_mesh, parallelize_module

mesh = init_device_mesh(
    "npu",
    (cp_size, tp_size),
    mesh_dim_names=("cp", "tp"),
)

parallelize_module(model, mesh["tp"], tp_plan)

for layer in model.layers:
    ContextParallel(seq_dim=1, head_dim=2, ulysses_degree=1).apply(
        layer.attention.sdpa_core,
        mesh["cp"],
    )
```

当 Q/K/V 已经是 TP DTensor 时，CP 会在内部临时组合 CP + TP mesh 执行通信。`use_local_output=False` 时，输出会去掉 CP 维并恢复成原来的非 CP layout，例如 TP head shard；`use_local_output=True` 时则返回 local tensor。

与 FSDP 组合时，CP 并行必须放到 FSDP 上：FSDP 使用的 mesh 必须由 `mesh[("dp_shard", "cp")]` flatten 成一维 `fsdp_mesh`，不能直接传二维 `mesh[("dp_shard", "cp")]`，也不能只用纯 `dp_shard` 子 mesh。这样 CP 的反向梯度才能由 FSDP 管理。代码顺序上先挂 CP hook，再执行 FSDP；推荐顺序是 TP -> CP -> FSDP。

```python
mesh = init_device_mesh(
    "npu",
    (dp_size, cp_size, tp_size),
    mesh_dim_names=("dp_shard", "cp", "tp"),
)

fsdp_mesh = mesh[("dp_shard", "cp")].flatten(mesh_dim_name="fsdp")
cp_mesh = mesh["cp"]
tp_mesh = mesh["tp"]

parallelize_module(model, tp_mesh, tp_plan)

for layer in model.layers:
    ContextParallel(seq_dim=1, head_dim=2, ulysses_degree=1).apply(
        layer.attention.sdpa_core,
        cp_mesh,
    )

fully_shard(model, mesh=fsdp_mesh)
```

PP 场景下仍在各 stage 内按上述规则组合 FSDP 与 CP：先将 `dp_shard + cp` flatten 为一维 FSDP mesh，再在 stage 内先挂 CP 后执行 FSDP。CP hooks 不替代 FSDP 的参数/梯度管理，也不负责 pipeline stage 调度。

## 常见检查项

1. 每个 CP rank 输入的是本地序列切片，通常需要满足 `seq_len % cp_size == 0`。
2. Pure Ulysses 要满足 `num_heads % cp_size == 0`；Hybrid 要满足 `cp_size % ulysses_degree == 0` 和 `num_heads % ulysses_degree == 0`。
3. `seq_dim`、`head_dim` 必须匹配实际 Q/K/V 布局，BSHD 与 BNSD 不要混用。
4. attention forward 中 Q/K/V 不在前三个位置参数时，需要配置 `qkv_indices` 或 `qkv_kwarg_names`。
5. DSA style 只支持 `layout="BSND"` 或 `"TND"`，且只支持 `mode="colossal"`。
6. Async CP 的 projection/handoff 模块要贴近 consumer 边界；中间存在额外 tensor 变换时，应把变换后的最后一个模块作为 handoff。
