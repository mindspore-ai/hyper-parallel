# Operator Backend Selection and Integration

**English** | [简体中文](op_select.zh.md)

OpSelect gives high-performance modules a common operator interface across hardware backends. During training, you can let OpSelect select fused implementations for the target device or switch one operator family to its PyTorch reference implementation to diagnose numerical differences.

This stage provides backend selection, Ops implementations, YAML/CLI configuration, and first-step reporting. Migration of existing functional and module call sites, parallel wrappers, and model recipes follows in a separate PR. Settings apply to code that already calls the public Ops interfaces.

## Quick start

Add `ops` at the root of the Trainer YAML. This example pins RMSNorm to the reference implementation and uses the default selection for the other families.

```yaml
ops:
  rms_norm: native
```

| Configuration value | Behavior |
|------|------|
| Omit `ops`, set `ops: {}`, or omit a family | Uses `auto` by default. Some call sites retain a default from a legacy recipe. See [backend requirements and limits](#backend-requirements-and-limits). |
| `auto` | Selects the first available implementation for the target device and the variant being called. |
| `native` | Pins the family to its reference implementation built from PyTorch operations. |
| An implementation name such as `torch_npu`, `cann`, or `sdpa` | Pins the family to that implementation. A device, variant, or dependency mismatch raises an error when the variant is first called. |

NPU and GPU training recipes can share the same `ops` settings when they use `auto`. To pin a backend, set the implementation in each device's YAML. For example, use `attention: torch_npu` in the NPU recipe and `attention: sdpa` in the CUDA recipe.

Use `--ops.<family>=<implementation>` to add or override a selection, for example `--ops.deepseek_sparse_attention=native`. Model replacements and recipe migration follow in the integration PR. The Python examples below exercise the standalone public interfaces.

### TileKernels interleaved RoPE

[TileKernels](https://github.com/deepseek-ai/TileKernels#requirements) currently requires Ascend 950, CANN ≥9.2.0, Python ≥3.12, PyTorch ≥2.13 and TileLang ≥0.1.15 with the Ascend backend enabled.

Install and call:

```bash
git clone https://github.com/deepseek-ai/TileKernels.git
python -m pip install ./TileKernels
```

```python
import torch
import torch_npu
from hyper_parallel.components.ops import ops, rope_interleaved

ops.set_backends({"rope": "tile_kernels"})
q = torch.randn(1, 4, 16, 64, device="npu", dtype=torch.bfloat16, requires_grad=True)
k = torch.randn(1, 2, 16, 64, device="npu", dtype=torch.bfloat16, requires_grad=True)
angles = torch.randn(1, 16, 32, device="npu")
cos, sin = (t.to(q.dtype).repeat(1, 1, 2) for t in (angles.cos(), angles.sin()))
q_out, k_out = rope_interleaved(q, k, cos, sin)
(q_out.float().sum() + k_out.float().sum()).backward()
print(ops.used_backends())
```

Trainer YAML configuration (applies only to modules calling public `rope_interleaved`):

```yaml
ops:
  rope: tile_kernels
```

Supports 4-D BF16/FP32 Q/K, rotary dimensions 64/128, partial rotation and first-order input/frequency gradients without mutating inputs. Use `unsqueeze_dim=1` for `[B,H,S,D]` or `2` for `[B,S,H,D]`. The `half` variant, FP16 Q/K and second derivatives are unsupported. Automatic selection priorities are unchanged.

## How backend selection works

1. **Configuration load.** The Trainer validates family names, duplicate keys, and value types. An explicit implementation name must be registered for the family.
2. **Module call.** Each operator family corresponds to one YAML key. A variant defines a call contract, including inputs, outputs, layout, dtype, and gradient behavior. For example, `swiglu(x)` takes a packed gate and up projection, while `swiglu_clamped(gate, up, limit)` takes separate inputs. The function called by the module determines the variant. Both variants read the same `ops.swiglu` value.
3. **Call-time resolution.** Before model construction, the Trainer sets the process configuration through `ops.set_backends()`. Public operators select implementations using the actual input device, dtype, call conditions, and backend dependencies. Cached choices are specific to the contract and selection conditions. Models and threads in the same process share the configuration. `auto` skips incompatible candidates and missing optional dependencies. Kernel errors propagate once an implementation starts executing.

PyTorch device types are `npu`, `cuda`, and `cpu`. OpSelect resolves against the actual input device and dtype. A model can be constructed on `meta` and loaded onto its training device before operators execute. [Operator catalog](../../hyper_parallel/components/ops/catalog.py) · [Trainer model construction](../../hyper_parallel/trainer/base.py)

### Confirm the selected backend

After the first global training step completes, the Trainer uses `ops.used_backends()` to collect the `(Op, Variant, Backend)` triples selected across ranks since the current configuration was set. Global rank 0 deduplicates, sorts, and prints them to `stderr` regardless of the logging level. If a variant uses multiple backends during the step, each appears on its own row. [First-step reporting](../../hyper_parallel/trainer/base.py)

```text
Fused operator backends used in training step 1

Op        Variant      Backend
--------  -----------  ---------
rms_norm  fp32_affine  torch_npu
swiglu    standard     native
```

When comparing numerical results, keep the checkpoint, inputs, random seed, dtype, and parallel configuration fixed. Switch one family at a time to `native`. Confirm the backend for the target variant in the first-step report, then compare outputs, gradients, or the results of one optimizer step. If a family is absent from the report, check whether the first-step inputs exercise that branch and whether the layer uses a module or wrapper that calls OpSelect.

## Operator support matrix

Each device column lists its `auto` candidates in priority order from left to right. You can also select each listed backend explicitly. Every variant in the table supports explicit `native` selection on all three device types. On NPU, `torch_npu` is the sole `auto` candidate for RMSNorm, SwiGLU, and single grouped matrix multiplication. Select `native` explicitly to run their reference implementations. [Operator catalog](../../hyper_parallel/components/ops/catalog.py) · [Automatic selection policy](../../hyper_parallel/components/ops/auto_policy.py)

| `ops` key | Variants and usage | Ascend NPU | CUDA GPU | CPU |
|------|------|------|------|------|
| `rms_norm` | `standard`, `fp32_affine` for standard and FP32 affine scaling | `torch_npu` | `native` | `native` |
| `swiglu` | `standard`, `clamped_fp32` for packed inputs and clamping separate inputs | `torch_npu` | `native` | `native` |
| `rope` | `half`, `interleaved`, both with partial head-dimension rotation | `torch_npu`, `native` | `native` | `native` |
| `mhc_post` | `standard` mixes the transformed output and residual streams | `cann`, `omni`, `native` | `native` | `native` |
| `sinkhorn` | `standard` normalizes the mHC mixing matrix | `omni`, `native` | `native` | `native` |
| `mome` | `standard` applies masked causal convolution and a residual connection on the DSA output | `omni`, `native` | `native` | `native` |
| `grouped_matmul` | `standard` performs a single bias-free grouped matrix multiplication with expert-major inputs | `torch_npu` | `torch_grouped_mm`, `native` | `native` |
| `grouped_swiglu` | `standard` runs the gate, up, and down computations for non-quantized experts | `torch_npu`, `native` | `torch_grouped_mm`, `native` | `native` |
| `moe_router` | `deepseek_v41` performs top-k routing in V4.1 | `megagate`, `native` | `native` | `native` |
| `moe_token_reorder` | `standard`, `weighted_restore` for expert-major reordering and weighted restoration | `torch_npu`, `native` | `native` | `native` |
| `attention` | `vision`, `gqa`, `mla`, `mla_with_stats` | `torch_npu`, `native` | `sdpa`, `native` | `sdpa`, `native` |
| `gdn` | `zero_state` runs local zero-state Gated DeltaNet | `triton_ascend`, `ascendc`, `native` | `native` | `native` |
| `kda` | `zero_state` runs local zero-state Kimi Delta Attention | `triton_ascend`, `ascendc`, `native` | `native` | `native` |
| `causal_conv1d` | `standard` applies local short causal convolution and SiLU for GDN and KDA | `triton_ascend`, `ascendc`, `native` | `native` | `native` |
| `dsa_indexer` | `compressed_causal_topk`, `packed_causal_topk` for V4.1 compressed indexing and V3.2 document-local indexing | `cann`, `native` | `native` | `native` |
| `deepseek_sparse_attention` | `compressed_banks`, `absorbed_with_stats` for V4.1 compressed KV and V3.2 sparse MLA | `cann`, `native` | `native` | `native` |
| `deepseek_sparse_attention` | `absorbed_with_parameter_sink` for sparse MLA with a parameter sink | `omni`, `native` | `native` | `native` |
| `dsa_indexer_kl` | `compressed_teacher`, `packed_sparse_teacher`, `packed_dense_teacher`, including V3.2 dense warm-up | `cann`, `native` | `native` | `native` |

Attention entry points use the same full names as the module directories: `grouped_query_attention`, `multi_latent_attention` (including `multi_latent_attention_with_stats`), and `deepseek_sparse_attention_*`. The sparse attention configuration key and family file are named [`deepseek_sparse_attention`](../../hyper_parallel/components/ops/impl/deepseek_sparse_attention.py). Migrate configurations and entry points using `dsa_sparse_attention`, and entry points using `gqa_attention` or `mla_attention*`, to the corresponding new names.

### Backend requirements and limits

- `torch_grouped_mm` requires BF16 inputs, CUDA compute capability 8.0 or later, and PyTorch grouped matmul support. `sdpa` uses PyTorch's `scaled_dot_product_attention`. [Grouped matmul implementation](../../hyper_parallel/components/ops/impl/grouped_matmul.py) · [Attention implementations](../../hyper_parallel/components/ops/impl/attention.py)
- On NPU, `torch_npu`, `cann`, and `omni` require their corresponding extensions and operator symbols. Depending on the variant, `cann` uses `cann_ops_transformer`, `torch_npu`, or both. CANN mHC post requires FP16 or BF16 sublayer outputs and residuals, with FP32 mixing coefficients. `megagate` requires the standalone HyperMulticore native component, nonempty batched inputs, and `sqrtsoftplus` routing scores. [Installation guide](../installation.md) · [mHC post implementation](../../hyper_parallel/components/functional/mhc_post.py) · [MoE routing implementation](../../hyper_parallel/components/ops/impl/moe_router.py)
- The public GDN, KDA, and short-convolution interfaces provide local zero-state computation. Existing CP wrappers continue to own recurrent state and convolution halos; their OpSelect integration follows in the migration PR. [KDA context parallelism](kda_context_parallel.md) · [TND paths for GDN and KDA](linear_attention_tnd_cp.md)
- DeepSeek V4.1 CSA2 contracts require supported BF16 inputs, document segmentation, and top-k selection. Existing module options such as `use_fused_ops` and `use_grouped_mm` retain their behavior until those call sites migrate.

## Integrate OpSelect into a module

Import the operator from `hyper_parallel.components.ops` and call it directly in `forward`. The module continues to own its weights, projections, and checkpoint conversions. This snippet illustrates the convention for the subsequent module migration:

```python
from hyper_parallel.components.ops import swiglu


def forward(self, x):
    gate_up = self.linear_fc1(x)
    return self.linear_fc2(swiglu(gate_up))
```

## Set and inspect backends

Set the process configuration before model construction or compilation. Operators select their implementations from the actual inputs. [Configuration and query implementation](../../hyper_parallel/components/ops/factory.py)

| Interface | Behavior |
|------|------|
| `ops.set_backends(selections: Mapping[str, str]) -> None` | Replaces the complete process configuration and clears resolution caches and usage. Omitted families return to their defaults, usually `auto`. An empty mapping restores all defaults. |
| `ops.get_backends() -> dict[str, str]` | Returns an independent copy of the explicit configuration. Omitted families have no key in the returned dictionary. |
| `ops.used_backends() -> tuple[tuple[str, str, str], ...]` | Returns deduplicated `(op, variant, backend)` triples successfully selected since the current configuration was set. Querying preserves the accumulated records. |

Invalid settings raise `ValueError` and preserve the active configuration, caches, and usage. The input mapping and returned configuration copy can be modified independently.

```python
import torch

from hyper_parallel.components.ops import ops, swiglu

ops.set_backends({"swiglu": "native"})
x = torch.randn(2, 8, requires_grad=True)
output = swiglu(x)
output.sum().backward()

print(ops.get_backends())    # {'swiglu': 'native'}
print(ops.used_backends())   # (('swiglu', 'standard', 'native'),)
```

Each `torchrun` worker calls `ops.set_backends(config.get("ops", {}))` during initialization. Keep one configuration throughout forward, backward, and activation recomputation. For sequential A/B runs, finish the current execution before setting the next configuration. All models in the process then use the new settings. Separate processes support concurrent runs with different configurations.

`torch.compile` guards the configuration captured by a graph, so calls after a configuration change specialize for the new settings. Recapture static CUDA/NPU graphs after changing settings.

## Use with third-party training frameworks

Configure each worker in Transformers, TorchTitan, or a custom training program. User-defined computation can call the public operators without wrapping the training loop. This CPU example exercises the standalone interface:

```python
import torch

from hyper_parallel.components.ops import ops, rms_norm

ops.set_backends({"rms_norm": "native"})
hidden_states = torch.randn(2, 8, requires_grad=True)
weight = torch.ones(8, requires_grad=True)
output = rms_norm(hidden_states, weight, eps=1e-6)
output.sum().backward()
print(ops.used_backends())
```

A third-party step-end callback can query `ops.used_backends()`. Migration of existing high-performance modules, concrete model call sites, and recipes follows in the second PR, together with full Trainer examples and integration validation.

## Register an operator family or backend

1. **Define the contract from the reference implementation.** Keep the contract and implementations together in `components/ops/impl/<config_key>.py`, with the filename matching the YAML key. Use `Protocol` to describe the arguments, return values, and gradient behavior required by the reference implementation. Use `OpSpec` to name the family and variant. A family can have several variants. Bind kernel-specific arguments inside the implementation, such as the NPU group encoding for grouped matrix multiplication. [Grouped matmul contract and adapters](../../hyper_parallel/components/ops/impl/grouped_matmul.py)
2. **Register implementations.** Call `register_family` once with all variants, then use `register_impl` for each implementation of a variant. `load` is a zero-argument loader that can return a function, an object implementing `__call__`, or an `nn.Module`. An implementation can define its backward pass with `torch.autograd.Function`. The model owns trainable parameters and passes them to the operator. Use `check_compatibility(x)` to validate the primary input's device and dtype. Return `None` when compatible or a string explaining the rejection. Check shape, layout, and multi-input constraints in the public entry point and include the candidate restrictions in resolution caching. Raise `ImplementationUnavailable` from the loader when an optional package or operator is missing so `auto` can check the next candidate. [Registry API](../../hyper_parallel/components/ops/registry.py) · [Dependency checks](../../hyper_parallel/components/ops/_dependency.py)
3. **Set the default order and export the call.** Add new families to `catalog.py`. Before the catalog is sealed, use `set_auto_order` in `auto_policy.py` to set the order for each variant and device. Export the new variant's function from the package entry point. Simple contracts can use `_dispatch(SPEC)`. Reuse the existing function when adding a backend for an existing variant. [Package entry point](../../hyper_parallel/components/ops/__init__.py) · [Automatic selection policy](../../hyper_parallel/components/ops/auto_policy.py)
4. **Verify forward results, gradients, and selection.** Compare the fused and reference implementations with identical inputs across relevant dtypes, layouts, and boundary cases. Call the operator through a model module to verify the configured backend and first-step report.

`impl/swiglu.py` registers two variants for the SwiGLU family. These excerpts show the contract, implementation registration, default order, and package export for `standard`. [Full SwiGLU registration](../../hyper_parallel/components/ops/impl/swiglu.py)

```python
# Contract in impl/swiglu.py
class SwiGLUFn(Protocol):
    def __call__(self, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """Split the gate and up inputs evenly along dim and return SiLU(gate) * up."""


SWIGLU = OpSpec[SwiGLUFn](config_key="swiglu")

# Registration function in impl/swiglu.py
registry.register_family("swiglu", specs=(SWIGLU, SWIGLU_CLAMPED))
registry.register_impl(SWIGLU, name="native", devices=("cpu", "cuda", "npu"), load=_load_native)
registry.register_impl(SWIGLU, name="torch_npu", devices=("npu",), load=_load_npu)
# The same function registers the SWIGLU_CLAMPED implementations.

# auto_policy.py
registry.set_auto_order(SWIGLU, device="npu", candidates=("torch_npu",))
registry.set_auto_order(SWIGLU, device="cuda", candidates=("native",))
registry.set_auto_order(SWIGLU, device="cpu", candidates=("native",))

# components/ops/__init__.py
swiglu: SwiGLUFn = _dispatch(SWIGLU)
```
