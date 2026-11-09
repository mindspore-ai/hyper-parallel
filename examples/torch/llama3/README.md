# Llama3-Style Examples (Torch: TP, FSDP2, and Context Parallelism)

**English** | [简体中文](README.zh.md)

This directory provides minimal runnable examples for **Llama3-style decoders** on the **HyperParallel Torch backend**. **Declarative tensor parallelism (TP) + sequence parallelism (SP)** uses `parallelize_module` with `ColwiseParallel`, `RowwiseParallel`, `SequenceParallel`, `PrepareModuleInput`, and related helpers. It composes with **`fully_shard` (FSDP2-style fully sharded data parallelism)** or **`ContextParallel` (context parallelism along the sequence dimension)**. TP+FSDP uses a 2-D `DeviceMesh`, selecting `mesh["tp"]` for TP and `mesh["dp"]` for FSDP. TP+CP uses a 2-D `(tp, cp)` mesh: apply `parallelize_llama3` on `mesh["tp"]` and attach `ContextParallel` to attention's BSHD SDPA submodule on `mesh["cp"]`, using Colossal mode with `ulysses_degree=1`.

> **Note:** If your documentation tree contains `examples/distributed/tensor_parallelism/README.md`, its environment, launch, and constraint sections can be read alongside this guide. **This document is the entry point for this directory.**

---

## Directories and Files

| File | Description |
|------|------|
| `model.py` | Small Llama3 architecture: `tok_embeddings`, `layers.*`, `attention` (wq/wk/wv/wo), `feed_forward` (w1/w2/w3), RMSNorm, and `output`. |
| `parallelize.py` | `parallelize_llama3()`: TorchTitan-style TP+SP, with row-sharded embeddings, sequence-dimension `Shard`, and Colwise/Rowwise attention and SwiGLU. |
| `tensor_parallel_example.py` | **TP only:** 1-D `DeviceMesh`, with `world_size` equal to the TP degree and a short training loop. |
| `fsdp_tp_example.py` | **TP + `fully_shard`:** 2-D `(dp, tp)` mesh, with TP on `mesh["tp"]` and FSDP on `mesh["dp"]`; nested `fully_shard` on sublayers and the root module. |
| `tp_cp_example.py` | **TP + CP:** Build `Llama3Model` (`n_layers=2`), apply `parallelize_llama3(..., mesh["tp"])`, then apply `ContextParallel` on `mesh["cp"]` to **every** `layer.attention.sdpa_core`. Each CP rank processes a `seq_len / cp` token window; `rope_seq_start` aligns RoPE with the post-embedding CP window. |
| `dp_tp_cp_sp_fsdp_example.py` | **Eight devices: DP + TP + CP + SP + FSDP2.** 4-D `(dp, fsdp, cp, tp)` mesh; TP+SP uses `mesh["tp"]`, CP uses `mesh["cp"]`, and FSDP2/HSDP uses `mesh[("dp", "fsdp")]`. Parameters are sharded within `fsdp` and replicated across `dp`; `dp=1` reduces to pure FSDP2. |
| `demo_utils.py` | Shared `init_dist()` / `train_steps()`, defaulting to ten steps, overridable with `LLAMA3_NUM_STEPS`. |
| `pipeline.py` | PP stage construction: `build_llama3_pp_chunk()` partitions layers by PP rank; the final stage supports per-micro-batch CE. |
| `pp_example.py` | **PP only:** 1-D `(pp,)` mesh, `Schedule1F1B`, one stage per rank; stage 0 receives tokens and the final stage computes CE. |
| `pp_tp_example.py` | **PP + TP:** 2-D `(pp, tp)` mesh, in-stage `parallelize_llama3`, and `Schedule1F1B` + P2P. |
| `pp_fsdp_tp_cp_sp_example.py` | **Eight devices: PP + FSDP2 + TP + CP + SP.** 4-D `(pp, fsdp, cp, tp)` mesh, default `(2,2,2,1)`. |
| `pp_fsdp_cp_example.py` | **Eight devices: PP + FSDP2 + CP.** 3-D `(pp, fsdp, cp)` mesh, default `(2,2,2)`. |
| `__init__.py` | Re-exports major symbols; example scripts add this directory to `sys.path`. |

---

## Environment and Dependencies

Follow the **PyTorch / CANN / HyperParallel** requirements in [`examples/README.md`](../../README.md). Activate the Ascend or virtual environment and ensure `hyper_parallel`, `torch`, and, where applicable, `torch_npu` can be imported.

| Component | Description |
|------|------|
| Python | >= 3.9 |
| CANN / driver | Match the current Ascend stack |
| HyperParallel | `import hyper_parallel` succeeds |

---

## Run 1: Tensor Parallelism Only (TP)

The process count must equal the **TP degree**; the script covers all ranks with a 1-D `DeviceMesh`.

```bash
cd examples/torch/llama3

torchrun --nnodes=1 --nproc_per_node=2 tensor_parallel_example.py
```

From the repository root:

```bash
torchrun --nnodes=1 --nproc_per_node=2 examples/torch/llama3/tensor_parallel_example.py
```

If the package is not installed, the original example also permits pointing `PYTHONPATH` at the repository root; repository development requires an editable installation as described in the main setup guide.

---

## Run 2: TP + FSDP (`fully_shard`)

The total rank count is torchrun's **`world_size`**. With TP degree `TP` (`LLAMA3_TP_SIZE`, default `2`):

- **`dp_size = world_size / TP`**, for data parallelism / FSDP along `mesh["dp"]`.
- **`world_size % TP == 0`** is required.

Example: four devices with `TP=2`, `DP=2`:

```bash
cd examples/torch/llama3
torchrun --nnodes=1 --nproc_per_node=4 fsdp_tp_example.py
```

Optional environment variables:

| Variable | Meaning | Default |
|------|------|------|
| `LLAMA3_TP_SIZE` | Tensor parallel degree | `2` |
| `LLAMA3_DEVICE_TYPE` | Device type: `npu` or `cuda` | `npu` |

---

## Run 3: TP + Context Parallelism (`ContextParallel`)

The total rank count must equal **`LLAMA3_TP_SIZE * LLAMA3_CP_SIZE`**, defaulting to `2 × 2 → 4` processes. The script builds **`Llama3Model`** from `model.py`, calls **`parallelize_llama3(model, mesh["tp"])`** with the same TP+SP plan as Run 1, then applies **`ContextParallel(..., ulysses_degree=1).apply(layer.attention.sdpa_core, mesh["cp"])`** to **every** decoder layer. This registers Colossal CP hooks on the BSHD SDPA submodule, `Llama3BshdSdpaCore` in `model.py`.

Each CP rank receives the token slice for its context-parallel group. Pass **`rope_seq_start`** to `Llama3Model.forward` to align RoPE with the global window after `tok_embeddings`; the model slices `freqs_cis` using the **local post-embedding** sequence length.

```bash
cd examples/torch/llama3
torchrun --nnodes=1 --nproc_per_node=4 tp_cp_example.py
```

Optional environment variables:

| Variable | Meaning | Default |
|------|------|------|
| `LLAMA3_TP_SIZE` | Head-dimension TP degree used by `parallelize_llama3` | `2` |
| `LLAMA3_CP_SIZE` | Context parallel degree | `2` |
| `LLAMA3_DEVICE_TYPE` | `npu` or `cuda` | `npu` |

Constraints: **`seq_len % cp == 0`**, **`(seq_len / cp) % tp == 0`** for even Rowwise embedding sequence sharding, and **both `n_heads` and `n_kv_heads` divisible by `tp`**.

---

## Run 4: Eight-Device DP + TP + CP + SP + FSDP2

`dp_tp_cp_sp_fsdp_example.py` enables the following on a 4-D `(dp, fsdp, cp, tp)` `DeviceMesh`:

| Dimension | Component | Description |
|------|------|------|
| `mesh["tp"]` | **TP + SP** | TorchTitan-style `parallelize_llama3`: Colwise/Rowwise linear layers, SequenceParallel norms, and `Shard(1)` sequence activations. |
| `mesh["cp"]` | **CP** | `ContextParallel(seq_dim=1, head_dim=2, ulysses_degree=1)` on every `layer.attention.sdpa_core`, using Colossal CP and BSHD Q/K/V. |
| `mesh[("dp", "fsdp")]` | **FSDP2 + DP** | `fully_shard` uses a 2-D HSDP slice: parameters shard within `fsdp` and replicate across `dp`; `dp=1` is pure FSDP2 and `dp>=2` is HSDP. |

Requires **`world_size == dp * fsdp * cp * tp`**. The default `(dp, fsdp, cp, tp) = (1, 2, 2, 2)` uses eight devices.

```bash
cd examples/torch/llama3
torchrun --nnodes=1 --nproc_per_node=8 dp_tp_cp_sp_fsdp_example.py
```

Alternatively, run from the repository root:

```bash
torchrun --nnodes=1 --nproc_per_node=8 examples/torch/llama3/dp_tp_cp_sp_fsdp_example.py
```

Optional environment variables and defaults:

| Variable | Meaning | Default |
|------|------|------|
| `LLAMA3_DP_SIZE` | Outer DP degree (HSDP replication) | `1` |
| `LLAMA3_FSDP_SIZE` | FSDP2 sharding degree | `2` |
| `LLAMA3_CP_SIZE` | Context parallel degree | `2` |
| `LLAMA3_TP_SIZE` | Tensor parallel degree, including SP | `2` |
| `LLAMA3_DEVICE_TYPE` | `npu` or `cuda` | `npu` |

Constraints: **`n_heads` and `n_kv_heads` divisible by `tp`**; **`seq_len % cp == 0`**; **`(seq_len / cp) % tp == 0`** for even SP sequence sharding within each CP window.

Common eight-device combinations (`dp * fsdp * cp * tp = 8`):

| `dp` | `fsdp` | `cp` | `tp` | Description |
|------|--------|------|------|------|
| 1 | 2 | 2 | 2 | Default: pure FSDP2 + CP + TP/SP. |
| 2 | 2 | 1 | 2 | HSDP (DP replication × FSDP sharding) + TP/SP, with CP disabled. |
| 1 | 4 | 1 | 2 | Larger FSDP sharding degree + TP/SP. |

> **Warning: When `fully_shard` promotes TP-DTensor weights to a ≥3-D mesh while active `(tp,)` inputs remain 1-D, as with `dp=2, fsdp=1, cp=2, tp=2`, layout inference does not yet support the weight mesh being a superset of the input mesh. It first raises an `int - tuple` type error in `parallel_embedding.infer_layout`, then also encounters the `x_mesh_shape != w_mesh_shape` check in `parallel_matmul`.** For pure DP without parameter sharding plus TP/SP/CP, use the default `(dp=1, fsdp=2, cp=2, tp=2)` or an HSDP combination above, and wait for the library fix before enabling `fsdp=1`.

The script broadcasts `tokens` / `targets` once across all ranks to ensure identical inputs. This is a smoke test like `fsdp_tp_example.py`, not a strict single-device numerical baseline.

---

## Run 5: Pipeline Parallelism (PP)

`pp_example.py` runs a Llama3-style decoder PP training loop using **`Schedule1F1B`** on a 1-D `(pp,)` `DeviceMesh`. Each PP rank owns consecutive layers: rank 0 includes `tok_embeddings`, and the final rank includes `norm` + `output` and computes per-micro-batch cross-entropy.

**`world_size` must equal `LLAMA3_PP_SIZE`**, default `2`.

```bash
cd examples/torch/llama3
torchrun --nnodes=1 --nproc_per_node=2 pp_example.py
```

Optional environment variables:

| Variable | Meaning | Default |
|------|------|------|
| `LLAMA3_PP_SIZE` | Pipeline parallel degree | `2` |
| `LLAMA3_DEVICE_TYPE` | `npu` or `cuda` | `npu` |

Constraints: **`n_layers >= pp_size`**; **batch size must be divisible by the micro-batch count**. Example defaults are `batch=8`, `micro_batches=4`.

---

## Run 6: PP + TP

`pp_tp_example.py` combines PP with TorchTitan-style TP+SP on a 2-D `(pp, tp)` mesh. Each stage calls `parallelize_llama3(..., mesh["tp"])`; stage 0 receives token batches as replicated `DTensor` inputs.

**`world_size == LLAMA3_PP_SIZE * LLAMA3_TP_SIZE`**, default `2 × 2 → 4`.

```bash
cd examples/torch/llama3
torchrun --nnodes=1 --nproc_per_node=4 pp_tp_example.py
```

Optional environment variables:

| Variable | Meaning | Default |
|------|------|------|
| `LLAMA3_PP_SIZE` | Pipeline parallel degree | `2` |
| `LLAMA3_TP_SIZE` | Tensor parallel degree | `2` |
| `LLAMA3_DEVICE_TYPE` | `npu` or `cuda` | `npu` |

Constraints: **`n_layers >= pp_size`**; **`n_heads` and `n_kv_heads` divisible by `tp`**; **`seq_len % tp == 0`**.

---

## Run 7: Eight-Device PP + FSDP2 + TP + CP + SP

`pp_fsdp_tp_cp_sp_example.py` enables five parallel modes on a 4-D `(pp, fsdp, cp, tp)` `DeviceMesh`:

| Dimension | Component | Description |
|------|------|------|
| `mesh["pp"]` | **PP** | `Schedule1F1B`, with one stage chunk per PP rank (`pipeline.py`). |
| `mesh["tp"]` | **TP + SP** | TorchTitan-style `parallelize_llama3`. |
| `mesh["cp"]` | **CP** | `ContextParallel` on every `layer.attention.sdpa_core`. |
| `mesh["fsdp"]` | **FSDP2** | `fully_shard` shards parameters along `fsdp` within each PP stage. |

Requires **`world_size == pp * fsdp * cp * tp`**. The default `(pp, fsdp, cp, tp) = (2, 2, 2, 1)` uses eight devices; setting all four axes to 2 requires 16 devices, as explained below.

```bash
cd examples/torch/llama3
torchrun --nnodes=1 --nproc_per_node=8 pp_fsdp_tp_cp_sp_example.py
```

Enable TP+SP with `tp=2` on 16 devices:

```bash
LLAMA3_TP_SIZE=2 torchrun --nnodes=1 --nproc_per_node=16 pp_fsdp_tp_cp_sp_example.py
```

Optional environment variables and defaults:

| Variable | Meaning | Default |
|------|------|------|
| `LLAMA3_PP_SIZE` | Pipeline parallel degree | `2` |
| `LLAMA3_FSDP_SIZE` | FSDP2 sharding degree for decoder blocks in each PP stage | `2` |
| `LLAMA3_CP_SIZE` | Context parallel degree | `2` |
| `LLAMA3_TP_SIZE` | Tensor parallel degree, including SP | `1` on eight devices; set `2` on 16 devices |
| `LLAMA3_DEVICE_TYPE` | `npu` or `cuda` | `npu` |

Constraints: **`n_layers >= pp`**; **`n_heads` and `n_kv_heads` divisible by `tp`**; **`seq_len % cp == 0`**; **`(seq_len / cp) % tp == 0`**. Every step uses **`micro_batch_num=1`**, compatible with PP+FSDP scheduling.

> **Eight-device note:** With `pp × fsdp × cp × tp = 8`, at least one axis must be `1`. By default, **`tp=1`** and PP/FSDP/CP are all 2. `parallelize_llama3` is still called; SP is a no-op at tp=1. FSDP wraps only **decoder blocks**, not the stage root, avoiding conflicts between scheduler-injected explicit unshard/reshard and Torch autograd. Use 16 devices with `(2,2,2,2)` if all four axes must be ≥2.
>
> **`fsdp=1` is unavailable:** With `LLAMA3_FSDP_SIZE=1`, `fully_shard` and TP have incompatible DTensor mesh dimensions (`x_mesh_shape != w_mesh_shape`). All eight-device combinations with `fsdp=1` fail. Keep **`fsdp >= 2`**.
>
> **Validated eight-device combinations (`fsdp >= 2`, one step passed for each):**
>
> | pp | fsdp | cp | tp | Description |
> |----|------|----|----|------|
> | 1 | 2 | 1 | 4 | No PP partitioning, TP=4 |
> | 1 | 2 | 2 | 2 | No PP, TP=2 and CP=2 |
> | 1 | 2 | 4 | 1 | No PP, CP=4 |
> | 1 | 4 | 1 | 2 | No PP, FSDP=4 |
> | 1 | 4 | 2 | 1 | No PP |
> | 1 | 8 | 1 | 1 | No PP, pure FSDP8 |
> | 2 | 2 | 1 | 2 | PP=2 and TP=2, no CP |
> | 2 | 2 | 2 | 1 | **Default**, PP=2 and CP=2 |
> | 2 | 4 | 1 | 1 | PP=2, FSDP=4 |
> | 4 | 2 | 1 | 1 | PP=4, one layer per stage |

---

## Run 8: Eight-Device PP + FSDP2 + CP

`pp_fsdp_cp_example.py` enables three parallel modes on a 3-D `(pp, fsdp, cp)` `DeviceMesh`:

| Dimension | Component | Description |
|------|------|------|
| `mesh["pp"]` | **PP** | `Schedule1F1B`, with one stage chunk per PP rank (`pipeline.py`). |
| `mesh["cp"]` | **CP** | `ContextParallel` on every `layer.attention.sdpa_core`. |
| `mesh["fsdp"]` | **FSDP2** | `fully_shard` shards parameters along `fsdp` inside each PP stage. |

Requires **`world_size == pp * fsdp * cp`**. Default `(pp, fsdp, cp) = (2, 2, 2)` uses eight devices.

```bash
cd examples/torch/llama3
torchrun --nnodes=1 --nproc_per_node=8 pp_fsdp_cp_example.py
```

Optional environment variables and defaults:

| Variable | Meaning | Default |
|------|------|------|
| `LLAMA3_PP_SIZE` | Pipeline parallel degree | `2` |
| `LLAMA3_FSDP_SIZE` | FSDP2 sharding degree for decoder blocks inside each PP stage | `2` |
| `LLAMA3_CP_SIZE` | Context parallel degree | `2` |
| `LLAMA3_DEVICE_TYPE` | `npu` or `cuda` | `npu` |

Constraints: **`n_layers >= pp`**; **`seq_len % cp == 0`**; **`fsdp >= 2`**. Each step uses **`micro_batch_num=1`** for PP+FSDP scheduling compatibility.

> **Note:** FSDP wraps only **decoder blocks**, not the stage root, avoiding conflicts between scheduler-injected explicit unshard/reshard and Torch autograd. `fsdp=1` is unavailable because DTensor mesh_shape values do not match.

---

## Constraints and Notes

1. **Sequence parallelism:** Activations use `Shard(1)` on the sequence dimension; **sequence length must be divisible by the TP degree**.
2. **Attention heads:** **Both `n_heads` and `n_kv_heads` must be divisible by the TP degree**, consistent with Colwise sharding of the final dimension.
3. **`parallelize_llama3`** currently implements only **`enable_sequence_parallel=True`**; `enable_sequence_parallel=False` is not implemented.
4. **Training loop:** Use `SkipDTensorDispatch` only for **`loss.backward()` and `optimizer.step()`**. **Do not** wrap the entire forward, which may break DTensor behavior along the TP path.
5. **`fsdp_tp_example.py`:** Calls **`model.set_reduce_op_type("sum")`** on the DTensor + `fully_shard` path. Ranks use **identical random batches** to smoke-test the combined stack; this is **not** a strict single-device numerical baseline.
6. The configuration is a **small educational `Llama3DemoConfig` model**; numerical results and performance refer to this repository's HyperParallel implementation.
7. **`tp_cp_example.py`** uses the real **`Llama3Model` / `Llama3Attention`** stack from `model.py`. SDPA runs inside **`Llama3BshdSdpaCore`**, allowing `ContextParallel` to hook `forward(q, k, v)` with BSHD layout. Without explicit `freqs_cis`, **`Llama3Model.forward(..., rope_seq_start=...)`** slices `self.freqs_cis` using the local post-embedding length, required when combining CP windows with TP sequence sharding.
8. **Training steps:** All examples default to **ten steps**, controlled by `demo_utils.TRAIN_STEPS` or `LLAMA3_NUM_STEPS`.

---

## Further Reading (APIs)

- Tensor parallelism: `parallelize_module` and `ParallelStyle` subclasses in `hyper_parallel`; see source code and the main project documentation.
- Fully sharded training: `hyper_parallel.fully_shard` (`core/fully_shard/api.py`).
- Context parallelism: `hyper_parallel.ContextParallel` (`core/context_parallel/context_parallel.py`).
