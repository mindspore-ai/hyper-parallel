# Operator Backend Selection and Integration

**English** | [简体中文](op_select.zh.md)

OpSelect gives high-performance modules a common operator interface across hardware backends. During training, you can let OpSelect select fused implementations for the target device or switch one operator family to its PyTorch reference implementation to diagnose numerical differences.

Basic and expert module calls now use OpSelect: RMSNorm, SwiGLUMLP, SharedExpert, both GroupedExperts implementations, MegaGate, mHC Post/PipelinedMhc, and GQA/MLA RoPE. Ordinary EP/MoonEP permutation and restoration also use the public operators. Remaining attention and sequence call sites are covered by their phase-two tasks.

## Quick start

Add `ops` at the root of the Trainer YAML. This example pins RMSNorm to the reference implementation and uses the default selection for the other families.

```yaml
ops:
  rms_norm: native
```

| Configuration value | Behavior |
|------|------|
| Omit `ops`, set `ops: {}`, or omit a family | Uses `auto` with the candidate order for the current variant and device. |
| `auto` | Selects the first available implementation for the target device and the variant being called. |
| `native` | Pins the family to its reference implementation built from PyTorch operations. |
| An implementation name such as `torch_npu`, `cann`, or `sdpa` | Pins the family to that implementation. A device, variant, or dependency mismatch raises an error when the variant is first called. |

NPU and GPU training recipes can share the same `ops` settings when they use `auto`. To pin a backend, set the implementation in each device's YAML. For example, use `attention: torch_npu` in the NPU recipe and `attention: sdpa` in the CUDA recipe.

Use `--ops.<family>=<implementation>` to add or override a selection, for example `--ops.deepseek_sparse_attention=native`. Model replacements and recipe migration follow in the integration PR. The Python examples below exercise the standalone public interfaces.

## How backend selection works

Public operators do not accept a `backend` argument. Configure backends through YAML `ops:` or `ops.set_backends()`. Migrate legacy `backend=` calls for `mhc_post` (including its functional entry), `grouped_swiglu`, `causal_conv1d`, `gdn` and `kda` to the corresponding family setting. Omitted families always use `auto`.

1. **Configuration load.** The Trainer validates family names, duplicate keys, and value types. An explicit implementation name must be registered for the family.
2. **Module call.** Each operator family corresponds to one YAML key. A variant defines a call contract, including inputs, outputs, layout, dtype, and gradient behavior. For example, `swiglu(x)` takes a packed gate and up projection, while `swiglu_clamped(gate, up, limit)` takes separate inputs. The function called by the module determines the variant. Both variants read the same `ops.swiglu` value.
3. **Call-time resolution.** The Trainer sets process configuration through `ops.set_backends()`. Public functions explicitly provide the execution device. The selector caches by `(family, variant, selection, device)` without inspecting tensors, dtype, shape or argument positions. `auto` tries registered implementations in device-specific order and can continue when optional dependencies cannot load. Selected implementations validate their inputs; validation and kernel errors propagate without changing backend.

PyTorch device types are `npu`, `cuda`, and `cpu`. OpSelect resolves against the explicit execution device. A model can be constructed on `meta` and loaded onto its training device before operators execute. [Operator catalog](../../hyper_parallel/components/ops/catalog.py) · [Trainer model construction](../../hyper_parallel/trainer/base.py)

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

Configuration remains family-scoped. The public function determines the variant, and `auto` selects candidates for that variant and input device. An explicit backend must also support the variant being called. Each family appears only in its first row; blank family cells inherit the preceding family. Each variant has its own row. The table describes registration and selection capabilities, not completed hardware precision acceptance for every dtype, shape, or dependency version.

| `ops` key (family) | Variant | Usage | Ascend NPU `auto` candidates | CUDA GPU `auto` candidates | CPU `auto` candidates |
|------|------|------|------|------|------|
| `rms_norm` | `standard` | Standard RMSNorm | `torch_npu` | `native` | `native` |
|  | `fp32_affine` | FP32 normalization and affine scaling | `torch_npu` | `native` | `native` |
| `swiglu` | `standard` | SwiGLU with packed gate/up inputs | `torch_npu` | `native` | `native` |
|  | `clamped_fp32` | Separate gate/up inputs with FP32 clamping | `torch_npu` | `native` | `native` |
| `rope` | `half` | Split-half rotation with partial head dimensions | `torch_npu`, `native` | `native` | `native` |
|  | `interleaved` | Adjacent-pair rotation with partial head dimensions | `torch_npu`, `native` | `native` | `native` |
|  | `interleaved_single` | Single-tensor adjacent-pair rotation without an unused branch | `torch_npu`, `native` | `native` | `native` |
| `mhc_post` | `standard` | Mix transformed output and residual streams | `cann`, `omni`, `native` | `native` | `native` |
| `sinkhorn` | `standard` | Normalize mHC mixing matrices | `omni`, `native` | `native` | `native` |
| `aggregate_hidden` | `standard` | Causal depthwise convolution of `[S,B,H]` states with an output mask, no bias/residual; Omni requires BF16 | `omni`, `native` | `native` | `native` |
| `grouped_matmul` | `standard` | One bias-free grouped matmul with expert-major inputs | `torch_npu` | `torch_grouped_mm`, `native` | `native` |
| `grouped_swiglu` | `standard` | Gate, up and down projections for non-quantized experts | `torch_npu`, `native` | `torch_grouped_mm`, `native` | `native` |
| `moe_router` | `deepseek_v41` | V4.1 top-k routing | `megagate`, `native` | `native` | `native` |
| `moe_token_permute` | `standard` | Expert-major token reordering | `torch_npu`, `native` | `native` | `native` |
| `moe_token_unpermute` | `standard` | Route-weighted token restoration | `torch_npu`, `native` | `native` | `native` |
| `attention` | `vision` | Vision attention | `torch_npu`, `native` | `sdpa`, `native` | `sdpa`, `native` |
|  | `gqa` | Grouped-query attention | `torch_npu`, `native` | `sdpa`, `native` | `sdpa`, `native` |
|  | `mla` | Multi-latent attention over projected states | `torch_npu`, `native` | `sdpa`, `native` | `sdpa`, `native` |
|  | `mla_with_stats` | Packed causal MLA with optional teacher softmax statistics | `torch_npu`, `native` | `sdpa`, `native` | `sdpa`, `native` |
| `gdn` | `zero_state` | Local zero-state Gated DeltaNet | `triton_ascend`, `ascendc`, `native` | `native` | `native` |
| `kda` | `zero_state` | Local zero-state Kimi Delta Attention | `triton_ascend`, `ascendc`, `native` | `native` | `native` |
| `causal_conv1d` | `standard` | Local short causal convolution and SiLU for GDN/KDA | `triton_ascend`, `ascendc`, `native` | `native` | `native` |
| `dsa_indexer` | `compressed_causal_topk` | V4.1 compressed indexing | `cann`, `native` | `native` | `native` |
|  | `packed_causal_topk` | V3.2 packed document-local indexing | `cann`, `native` | `native` | `native` |
| `deepseek_sparse_attention` | `compressed_banks` | V4.1 sparse attention with compressed KV banks | `cann`, `native` | `native` | `native` |
|  | `absorbed_with_stats` | V3.2 sparse MLA with statistics | `cann`, `native` | `native` | `native` |
|  | `absorbed_with_parameter_sink` | Sparse MLA with a parameter sink | `omni`, `native` | `native` | `native` |
| `dsa_indexer_kl` | `compressed_teacher` | Indexer KL with a V4.1 compressed teacher | `cann`, `native` | `native` | `native` |
|  | `packed_sparse_teacher` | Indexer KL with a V3.2 packed sparse teacher | `cann`, `native` | `native` | `native` |
|  | `packed_dense_teacher` | Indexer KL with a V3.2 packed dense warm-up teacher | `cann`, `native` | `native` | `native` |

`moe_token_permute` and `moe_token_unpermute` are independent families, each with a `standard` variant and its own backend setting. Split the former `ops.moe_token_reorder` setting into these two keys, and replace the public `moe_token_reorder` / `moe_token_restore` calls with `moe_token_permute` / `moe_token_unpermute`. Their shared slot mapping permits mixing native and torch_npu backends.

Attention entry points use the same full names as the module directories: `grouped_query_attention`, `multi_latent_attention` (including `multi_latent_attention_with_stats`), and `deepseek_sparse_attention_*`. The sparse attention configuration key and family file are named [`deepseek_sparse_attention`](../../hyper_parallel/components/ops/impl/deepseek_sparse_attention.py). Migrate configurations and entry points using `dsa_sparse_attention`, and entry points using `gqa_attention` or `mla_attention*`, to the corresponding new names.

`functional.aggregate_hidden` delegates to the public `components.ops.aggregate_hidden` entry point and uses
its own `ops: {aggregate_hidden: native}` setting. States are `[S,B,H]`, taps `[K,H]`, and the Boolean mask
`[B,S]`. The last tap multiplies the current token; earlier positions use left zero padding. The mask zeros
output positions without excluding those states from later outputs.
`functional.mome` composes layout conversion, optional bias, output masking and a residual through the public
`ops.aggregate_hidden` entry. Configure its backend with `aggregate_hidden`; there is no separate `mome`
operator family. Aggregation restores the input dtype before bias addition, which differs from a single
rounding of fused convolution-plus-bias at low precision. Omni registration does not imply completed
precision acceptance on the target device.

### Backend requirements and limits

- `torch_grouped_mm` requires BF16 inputs, CUDA compute capability 8.0 or later, and PyTorch grouped matmul support. `sdpa` uses PyTorch's `scaled_dot_product_attention`. [Grouped matmul implementation](../../hyper_parallel/components/ops/impl/grouped_matmul.py) · [Attention implementations](../../hyper_parallel/components/ops/impl/attention.py)
- On NPU, `torch_npu`, `cann`, and `omni` require their corresponding extensions and operator symbols. Depending on the variant, `cann` uses `cann_ops_transformer`, `torch_npu`, or both. CANN mHC post requires FP16 or BF16 sublayer outputs and residuals, with FP32 mixing coefficients. `megagate` requires the standalone HyperMulticore native component, nonempty batched inputs, and `sqrtsoftplus` routing scores. [Installation guide](../installation.md) · [mHC post implementation](../../hyper_parallel/components/functional/mhc_post.py) · [MoE routing implementation](../../hyper_parallel/components/ops/impl/moe_router.py)
- The public GDN, KDA, and short-convolution interfaces provide local zero-state computation. Existing CP wrappers continue to own recurrent state and convolution halos; their OpSelect integration follows in the migration PR. [KDA context parallelism](kda_context_parallel.md) · [TND paths for GDN and KDA](linear_attention_tnd_cp.md)
- RMSNorm retains legacy member selection through `use_fused_ops`; both members now resolve their numerical backend through `ops.rms_norm`, preserving scale representation and rounding order. Other unmigrated module options retain their existing behavior.

## Integrate OpSelect into a module

SwiGLU clamp and precision policies live in `functional.swiglu`. GQA and MLA share `functional.rotary_embedding.apply_rotary_pos_emb_bsnd`, whose interleaved path uses the single-tensor variant. Use `functional.apply_rotary_pos_emb` for half-layout Q/K composition instead of the former `ops.rope_qk`. Legacy functional token permutation and unpermutation retain their two-result, int32 inverse-map and optional-probability contracts through adapters to configured ops.

Modules may call functional or ops: reuse functional for algorithm composition, layout adaptation and precision policies, or call ops directly for an atomic operator. Functional calls public ops interfaces, while backend selection remains in ops. Modules own parameters, state, projections and checkpoint conversions. This example calls an atomic operator directly:

```python
from hyper_parallel.components.ops import swiglu


def forward(self, x):
    gate_up = self.linear_fc1(x)
    return self.linear_fc2(swiglu(gate_up))
```

The final goal is for ops to own the local-operator hardware boundary. Functional and module algorithms and calls should remain unchanged across supported CPU/CUDA/NPU devices and backends. They should not load vendor extensions, call hardware kernels, import `ops.impl`, or bypass selection through device/fused switches. Ordinary Torch composition and allocations following the input device remain valid. Legacy RoPE, grouped-matmul compatibility branches and EP permutation selection still have gaps; the support matrix does not imply complete module migration. See the [hardware-boundary audit](../contributing/op_select_development.zh.md#43-当前基础与专家接入的硬件边界审视).

## Set and inspect backends

Set the process configuration before model construction or compilation. Operators select their implementations from the actual inputs. [Configuration and query implementation](../../hyper_parallel/components/ops/factory.py)

| Interface | Behavior |
|------|------|
| `ops.set_backends(selections: Mapping[str, str]) -> None` | Replaces the complete process configuration and clears resolution caches and usage. Omitted families use `auto`. An empty mapping restores `auto` for every family. |
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

Third-party step-end callbacks can inspect `ops.used_backends()`. Basic/expert modules work directly in PyTorch and Transformers Trainer without a training wrapper. See the [offline Qwen3-MoE regression](../../tests/ut/auto_models/models/qwen3_moe/test_qwen3_moe_op_select.py) and the [development and acceptance requirements](../contributing/op_select_development.zh.md).

## Register an operator family or backend

1. **Define the contract from the reference implementation.** Keep the contract and implementations together in `components/ops/impl/<config_key>.py`, with the filename matching the YAML key. Use `Protocol` to describe the arguments, return values, and gradient behavior required by the reference implementation. Use `OpSpec` to name the family and variant. A family can have several variants. Bind kernel-specific arguments inside the implementation, such as the NPU group encoding for grouped matrix multiplication. [Grouped matmul contract and adapters](../../hyper_parallel/components/ops/impl/grouped_matmul.py)
2. **Register implementations.** Call `register_family` once with all variants, then use `register_impl` for each implementation of a variant. `load` is a zero-argument loader that can return a function, an object implementing `__call__`, or an `nn.Module`. An implementation can define its backward pass with `torch.autograd.Function`. The model owns trainable parameters and passes them to the operator. The registry does not accept tensor compatibility predicates. Public entries validate general contracts and selected implementations validate backend-specific inputs on every call; dtype and shape do not participate in backend selection or caching. Raise `ImplementationUnavailable` from the loader when an optional package or operator is missing so `auto` can check the next candidate. [Registry API](../../hyper_parallel/components/ops/registry.py) · [Dependency checks](../../hyper_parallel/components/ops/_dependency.py)
3. **Set the default order and export the call.** Add new families to `catalog.py`. Before the catalog is sealed, use `set_auto_order` in `auto_policy.py` to set the order for each variant and device. Define the public function explicitly in the family file, then import it in `ops/__init__.py` and add it to `__all__`. Registration manages backend contracts and implementations; it does not generate public functions. Reuse the existing function when adding a backend for an existing variant. [Package entry point](../../hyper_parallel/components/ops/__init__.py) · [Automatic selection policy](../../hyper_parallel/components/ops/auto_policy.py)
4. **Verify forward results, gradients, and selection.** Compare the fused and reference implementations with identical inputs across relevant dtypes, layouts, and boundary cases. Call the operator through a model module to verify the configured backend and first-step report.

`impl/swiglu.py` registers two variants for the SwiGLU family. These excerpts show the contract, implementation registration, default order, and package export for `standard`. [Full SwiGLU registration](../../hyper_parallel/components/ops/impl/swiglu.py)

```python
# Contract in impl/swiglu.py
class SwiGLUFn(Protocol):
    """Split packed Gate/Up and compute SiLU(Gate) * Up; callers own projections."""

    def __call__(self, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """Split the gate and up inputs evenly along dim and return SiLU(gate) * up.

        Args:
            x: Packed gate/up projections with an even size along dim.
            dim: Axis containing the packed projection halves.

        Returns:
            Activated states with the split dimension halved.
        """

SWIGLU = OpSpec[SwiGLUFn](config_key="swiglu")

# Public dispatcher in impl/swiglu.py (full docstring omitted here)
def swiglu(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Split packed projections and compute SiLU(gate) * up."""
    operation = _select_for_call(SWIGLU, device=x.device)
    return operation(x, dim)

# Registration function in impl/swiglu.py
registry.register_family("swiglu", specs=(SWIGLU, SWIGLU_CLAMPED))
registry.register_impl(SWIGLU, name="native", devices=("cpu", "cuda", "npu"), load=_load_native)
registry.register_impl(SWIGLU, name="torch_npu", devices=("npu",), load=_load_npu)
# The same function registers the SWIGLU_CLAMPED implementations.

# auto_policy.py
registry.set_auto_order(SWIGLU, device="npu", candidates=("torch_npu",))
registry.set_auto_order(SWIGLU, device="cuda", candidates=("native",))
registry.set_auto_order(SWIGLU, device="cpu", candidates=("native",))

# Explicit export in ops/__init__.py
from .impl.swiglu import swiglu

# Call sites
from hyper_parallel.components.ops import swiglu
```

Protocols describe backend parameters, return values and implementation semantics. Public functions maintain complete type annotations and Google-style docstrings covering algorithms, public arguments, shapes, layouts, precision and outputs. Their signatures can differ: for example, public attention accepts a module while its backend contract accepts resolved causal and scaling settings.

Protocols express backend type contracts; there is currently no mandatory static type-checking gate. Registry and selector callable checks do not validate full signatures, tensor shapes or gradients. Public argument checks and contract tests validate those requirements.

Public functions call `factory._select_for_call(spec, device=execution_device)`. The operator chooses its execution device explicitly, for example from states or queries, regardless of the first argument's position or type. The cache key contains only family, variant, configured selection and the complete device. No tensor, dtype, shape or blocked-candidate signature is cached. Input validation remains active inside implementations on every invocation, including cache hits. `registry.py` owns the shared registry and `catalog.py` assembles it; the selector does not import the catalog.

IDE completion, parameter hints, navigation and documentation come directly from public function source and explicit imports in `ops/__init__.py`. When adding or removing an interface, update the family file, package imports and `__all__`, then run public API and affected operator tests. The package retains its `py.typed` marker. There is no generated `.pyi` or interface generation step in builds or CI.
