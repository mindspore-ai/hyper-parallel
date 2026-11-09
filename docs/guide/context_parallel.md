# CP Context Parallelism Guide

**English** | [简体中文](context_parallel.zh.md)

HyperParallel's Context Parallel (CP) is a `ParallelStyle` applied at attention computation boundaries. Forward hooks wrap each CP rank's local sequence shard as a DTensor and rearrange and communicate Q/K/V when entering or leaving core attention or DSA operator boundaries.

CP does not shard datasets, token IDs, or entire Transformer layer inputs. Before calling the model with CP enabled, each CP rank must receive its own sequence slice, with position IDs, the RoPE starting position, and attention masks aligned to the global sequence window.

Kimi Delta Attention uses dedicated entry points for state recurrence. For P2P, AllGather, Ulysses, and hybrid configurations, see the
[KDA Context Parallelism Guide](./kda_context_parallel.md).

DeepSeek V4.1 shared compressed attention uses a dedicated CP wrapper. For balancing policies and communication settings, see
[DeepSeek V4.1 Indexer Load Balancing and CP Communication](./deepseek_v41_indexer_cp.md).

## Scope

| Interface | Behavior |
|------|----------|
| `ContextParallel` | Register synchronous CP hooks on ordinary core attention, processing Q/K/V in forward arguments by default. |
| `AsyncContextParallel` | Register asynchronous CP hooks on ordinary core attention; supplying Q/K/V producer modules enables projection computation to overlap CP communication. |
| `DSAIndexerContextParallel` | Rewrite DTensor placements of query/key/weights at the DeepSeek Sparse Attention (DSA) indexer boundary. |
| `DSASparseAttentionContextParallel` | Rewrite placements of query/key/value/topk/rope at the DSA sparse attention boundary. |
| `DSAIndexerLossContextParallel` | Rewrite input placements at the DSA indexer-loss boundary and handle partial loss/gradient outputs. |
| `AsyncDSA*ContextParallel` | Asynchronous all-gather variants for DSA boundaries; initiate key/value gathers early when handoff modules are provided, otherwise fall back to synchronous gathers. |

## `ContextParallel` Modes

`ulysses_degree` selects the `ContextParallel` execution mode. CP size is `device_mesh.mesh.numel()` for the supplied mesh.

| Mode | Parameter | Communication and placement | Constraints |
|------|------|------------------|------|
| Pure Ulysses | `ulysses_degree=None` (default) | All-to-all Q/K/V from `Shard(seq_dim)` to `Shard(head_dim)` before attention; convert output head shards back to sequence shards. | `num_heads % cp_size == 0` |
| Pure Colossal AI | `ulysses_degree=1` | Keep Q as `Shard(seq_dim)`; all-gather K/V along the sequence dimension to `Replicate()`. | No requirement for the head count to be divisible by CP size |
| Hybrid | `1 < ulysses_degree < cp_size` | First perform seq->head all-to-all for Q/K/V within Ulysses subgroups, then gather K/V within Colossal subgroups. | `cp_size % ulysses_degree == 0` and `num_heads % ulysses_degree == 0` |

Common parameters:

| Parameter | Default | Description |
|------|--------|------|
| `seq_dim` | `1` | Sequence dimension index: `1` for BSHD, `2` for BNSD. |
| `head_dim` | `2` | Head dimension index: `2` for BSHD, `1` for BNSD. |
| `qkv_indices` | `(0, 1, 2)` | Positional argument indices of Q/K/V in attention forward. |
| `qkv_kwarg_names` | `()` | Argument names when Q/K/V are passed as keyword arguments. |
| `use_local_output` | `False` | If `True`, public output is always a local tensor. If `False`, plain tensor inputs return local tensors, CP DTensor inputs retain CP DTensors, and non-CP DTensor inputs such as TP lose the CP dimension at output while retaining their original non-CP layout. |
| `load_balance` | `False` | Supported only for `ulysses_degree=1`. Use head-tail Q exchange to balance Colossal CP load. |

Hybrid CP accepts a 1-D CP mesh directly. A 2-D mesh must have named dimensions, with the first used for Colossal subgroups and the second for Ulysses subgroups, typically `mesh_dim_names=("co", "ds")`.

`load_balance=True` splits local Q into halves and exchanges them with the peer rank using P2P, invoking attention twice internally. When constructing masks inside attention, use K's sequence length as the global length: Q sees only half the sequence in each sub-attention.

## Basic Usage

For BSHD core attention, each CP rank first receives its local sequence slice; attach the CP style to the attention submodule that actually consumes Q/K/V:

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

With `parallelize_module`, CP can also be included in the parallel plan:

```python
from hyper_parallel import ContextParallel, parallelize_module

parallelize_module(
    model,
    mesh["cp"],
    {"layers.*.attention.sdpa_core": ContextParallel(seq_dim=1, head_dim=2, ulysses_degree=1)},
)
```

Examples for different layouts and modes:

```python
# BSHD: [batch, seq, head, dim], Colossal CP
ContextParallel(seq_dim=1, head_dim=2, ulysses_degree=1).apply(core_attn, cp_mesh)

# BNSD: [batch, head, seq, dim], Pure Ulysses
ContextParallel(seq_dim=2, head_dim=1).apply(core_attn, cp_mesh)

# Hybrid CP with Ulysses subgroup size 2.
ContextParallel(seq_dim=1, head_dim=2, ulysses_degree=2).apply(core_attn, cp_mesh)
```

## AsyncContextParallel

`AsyncContextParallel` has largely the same constructor parameters as `ContextParallel`, but `apply` additionally accepts `q_proj`, `k_proj`, and `v_proj`:

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

Its behavior is:

| Mode | Asynchronous overlap point |
|------|------------|
| Pure Ulysses | Issue seq->head all-to-all after Q/K/V projection; wait in the attention pre-hook. |
| Pure Colossal AI | Issue sequence all-gather after K/V projection; wait in the attention pre-hook. Q is still wrapped as a sequence shard at the attention boundary. |
| Hybrid | Issue Ulysses all-to-all after Q/K/V projection; K/V gathers in Colossal subgroups remain synchronous at the attention boundary. |

Notes:

1. If any of `q_proj`, `k_proj`, or `v_proj` is omitted, `AsyncContextParallel.apply` falls back to synchronous `ContextParallel.apply`.
2. Each projection module must be the last hookable module before Q/K/V enter attention. The asynchronous pre-hook bypasses any intervening reshape, transpose, or QK normalization. For models with QK normalization, use the post-normalization modules as the `q_proj`/`k_proj` handoffs.
3. With `ulysses_degree=1` and `load_balance=True`, the asynchronous variant falls back to synchronous Colossal CP.

## DSA CP

DSA means DeepSeek Sparse Attention here. DSA CP prepares DTensor placements at indexer, sparse attention, and indexer-loss module boundaries, rather than replacing general attention. The boundary implementation consumes the transformed DTensor or local tensor inputs.

DSA CP currently supports only `mode="colossal"`; `mode="ulysses"` raises an error. Supported input layouts:

| `layout` | Sequence dimension |
|----------|--------|
| `"BSND"` | `seq_dim=1` |
| `"TND"` | `seq_dim=0` |

DSA styles treat the supplied `device_mesh` as a 1-D CP mesh. Multidimensional meshes are first flattened by rank list. DSA CP does not use Ulysses/Hybrid head sharding.

DSA styles rewrite positional inputs by default; `*_kwarg_name` parameters adapt them to keyword arguments.

| Style | Default forward boundary | Query-side placement | Key-side placement |
|-------|-------------------|--------------------|------------------|
| `DSAIndexerContextParallel` | `(query, key, weights, ...)` | `query`, `weights` -> `Shard(seq)` | `key` -> `Replicate()` |
| `DSASparseAttentionContextParallel` | `(query, key, value, topk_indices, query_rope, key_rope, ...)` | `query`, `topk_indices`, `query_rope` -> `Shard(seq)` | `key`, `value`, `key_rope` -> `Replicate()` |
| `DSAIndexerLossContextParallel` | `(query, key, query_index, key_index, weights, topk_indices, softmax_max, softmax_sum, query_rope, key_rope, ...)` | `query`, `query_index`, `weights`, `topk_indices`, `query_rope` -> `Shard(seq)` | `key`, `key_index`, `key_rope` -> `Replicate()` |

Synchronous DSA usage:

```python
from hyper_parallel import DSASparseAttentionContextParallel

DSASparseAttentionContextParallel(layout="BSND").apply(sparse_attention_boundary, cp_mesh)
```

Asynchronous DSA usage:

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

Asynchronous DSA handoff modules initiate key/value sequence all-gather early. Without a corresponding handoff, that input uses a synchronous gather at the consumer boundary. Output placements match the synchronous DSA style.

## Combine with TP/FSDP/PP

CP should use the CP submesh of a multidimensional mesh; TP, FSDP, and PP use their respective submeshes. Llama3 examples typically apply TP to projections/MLPs first, then attach CP to each attention layer's `sdpa_core`.

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

When Q/K/V are already TP DTensors, CP temporarily combines CP and TP meshes internally for communication. With `use_local_output=False`, output drops the CP dimension and restores the original non-CP layout, such as TP head sharding. With `use_local_output=True`, it returns a local tensor.

When combined with FSDP, the CP dimension must be included in FSDP: flatten `mesh[("dp_shard", "cp")]` into a 1-D `fsdp_mesh`. Do not pass the 2-D `mesh[("dp_shard", "cp")]` directly or use only `dp_shard`. This allows FSDP to manage CP backward gradients. Attach CP hooks before applying FSDP; the recommended order is TP -> CP -> FSDP.

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

With PP, combine FSDP and CP within each stage using the same rules: flatten `dp_shard + cp` into a 1-D FSDP mesh, then attach CP before applying FSDP inside the stage. CP hooks do not replace FSDP parameter/gradient management or schedule pipeline stages.

## Common Checks

1. Each CP rank receives a local sequence slice, typically requiring `seq_len % cp_size == 0`.
2. Pure Ulysses requires `num_heads % cp_size == 0`; Hybrid requires `cp_size % ulysses_degree == 0` and `num_heads % ulysses_degree == 0`.
3. `seq_dim` and `head_dim` must match the actual Q/K/V layout; do not mix BSHD and BNSD conventions.
4. Configure `qkv_indices` or `qkv_kwarg_names` when Q/K/V are not the first three positional arguments in attention forward.
5. DSA styles support only `layout="BSND"` or `"TND"`, and only `mode="colossal"`.
6. Async CP projection/handoff modules must be close to the consumer boundary. If additional tensor transformations intervene, use the last module after those transformations as the handoff.
