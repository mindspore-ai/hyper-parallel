# High-performance functions

`hyper_parallel.components.functional` provides reusable high-performance functions for Ascend NPU.
Some interfaces also expose an explicit `eager` reference backend; fused calls do not automatically fall back.

```python
from hyper_parallel.components.functional import rms_norm

output = rms_norm(x, weight)
```

RoPE functions keep the Transformers call contract:

```python
from hyper_parallel.components.functional import apply_rotary_pos_emb

query, key = apply_rotary_pos_emb(query, key, cos, sin)
```

Use `apply_rotary_pos_emb_interleave` for Transformers models whose source
attention calls the interleaved variant.

## MoE functions

The reusable MoE interfaces include:

- `grouped_matmul`: NPU grouped matrix multiplication with backward support.
- `moe_token_permute`: reorder routed tokens into expert-major order.
- `moe_token_unpermute`: restore expert outputs to token-major order.
- `swiglu`: fused NPU SwiGLU used by grouped experts.

```python
from hyper_parallel.components.functional import (
    moe_token_permute,
    moe_token_unpermute,
)

permuted_tokens, sorted_indices = moe_token_permute(tokens, expert_indices)
expert_outputs = experts(permuted_tokens, tokens_per_expert, routing_weights)
output = moe_token_unpermute(expert_outputs, sorted_indices, routing_probs)
```

In this example, `experts` is an instance of `hyper_parallel.components.modules.GroupedExperts`. It contains only local expert
weights and computation; routing, expert-parallel communication, and shared experts remain outside the module.
`hyper_parallel.components.modules.SharedExpert` provides the parameter-owning dense MLP used as a shared expert.

## Auxiliary-loss functions

`hyper_parallel.components.functional` also exports `aux_loss_auto_scale` and `set_aux_loss_scale` for auxiliary-loss gradient injection.

## DeepSeek V4.1 fused operators

The V4.1 adapters use the optional `cann_ops_transformer` PyTorch extension:

| Component | Required entry points |
| --- | --- |
| Indexer | `lightning_indexer` (LI V2), `lightning_indexer_metadata` when required by the device |
| Attention | `sparse_flash_mla`, `sparse_flash_mla_grad`, `sparse_flash_mla_metadata` |
| Indexer KL | `sparse_lightning_indexer_kl_loss_grad`, `sparse_lightning_indexer_kl_loss_grad_metadata` |

### Installation

Follow the official ops-transformer guides:

1. [Build and install the native operator package](https://gitcode.com/cann/ops-transformer/blob/master/docs/zh/install/compile.md).
2. [Build and install the PyTorch extension](https://gitcode.com/cann/ops-transformer/blob/master/torch_extension/README.md).

Both the native kernels and the Python extension are required. Load the CANN and
custom-operator environment as described in the guides before training.
HyperParallel imports the standard `cann_ops_transformer` package; use the
standard wheel build (`bash build.sh --torch_extension`), without `--ops` or
`--vendor_name`, which can change the Python package name at the tested revision.

### Tested versions

| Dependency | Tested configuration |
| --- | --- |
| CANN Toolkit | 9.0.0 |
| Python / PyTorch / torch-npu | 3.11 / 2.10.0 / 2.10.0 |
| ops-transformer source | `0b08075141913daeaf1e856faf7480ed393c8949` |
| Python extension | `cann_ops_transformer` 1.0.0, built from the source revision above |

The source revision identifies the tested operator interfaces; the wheel version
alone does not identify them.

### Configuration

Enable the three adapters together through the V4.1 replacement:

```yaml
plan_overrides:
  - match: "model.layers.*.self_attn"
    replace:
      kind: module
      source_type: hyper_parallel.models.deepseek_v41.modeling_deepseek_v41.DeepseekV41Attention
      implementation:
        _target_: >-
          hyper_parallel.components.modules.shared_compressed_dsa_attention.SharedCompressedDSAAttention
        use_fused_ops: true
```

`false` (default) uses the Torch reference implementation without this extension.
`true` requires the fused entry points used by the model's execution path and
raises on missing entries or unsupported inputs, without automatic fallback.
The fused path supports ordinary causal CSA with compression ratio 1 or 2;
candidate-pool selection is not supported.

## SiTU and block attention residuals

These shared functions support `backend="eager"` (default, Torch reference with FP32
intermediates) and `backend="ascendc"` (standalone CANN operators). SiTU uses
ops-nn; block attention residuals use ops-transformer.

```python
from hyper_parallel.components.functional import situ_packed, block_attention_residuals

activated = situ_packed(packed_gate_up, beta=4.0, linear_beta=25.0, backend="ascendc")
residual = block_attention_residuals(
    prefix, history, query_weight, rms_weight, norm_eps=1e-5, backend="ascendc",
)
```

`situ(gate, up, ...)` accepts separate projections of the same shape, dtype and device;
`situ_packed` accepts `[..., 2H]` with gate channels followed by up channels. Generic
SiTU defaults are `beta=1` and `linear_beta=None`; K3 supplies `4` and `25`.
`linear_beta=None` or `linear_beta <= 0` disables saturation of the up projection
in both backends, following the ops-nn contract.

Block attention residuals take `prefix: [T,H]`, `history: [T,N,H]`,
`projection: [H]` or `[1,H]`, and `norm_weight: [H]`. The model supplies `norm_eps`,
learned parameters and block/history scheduling. The wrapper promotes mixed input
dtypes differentiably and reshapes `[H]` queries to `[1,H]` for native backward.
Both fused functions support first-order gradients; empty inputs/history retain
connected gradients through tensor operations without launching a native kernel.

### Eager reference math

The eager functions implement the SiTU and block attention-residual formulas in Torch.
Their model reference is Moonshot AI's published
[Kimi K3 text implementation](https://huggingface.co/moonshotai/Kimi-K3/blob/f831ab66814297da540d832a5235f8e904f29d06/modeling_kimi_linear.py):
`SituAndMul.forward` for SiTU and `_apply_attn_res` for block attention residuals.
The public functions accept tensors directly instead of model modules.

SiTU computes `beta * tanh(gate / beta) * sigmoid(gate) * up`, with optional up
saturation. The nonpositive `linear_beta` convention follows
[ops-nn](https://gitcode.com/cann/ops-nn/blob/b7af75e68cc07c0912f9e020bce639e53de60700/activation/situ_glu/README.md);
it extends the model reference's `None` convention.

Block attention residuals normalize residuals for scoring, then mix the original
residual values. The local implementation uses elementwise multiplication and
reduction for the weighted sum; the model reference uses a batched matrix multiply.
They implement the same formula, but floating-point reduction order can differ.
Block boundaries and history updates remain the calling model's responsibility.

### CANN dependencies and errors

Install both the native operators and the matching standalone Python extension:

| Function | Python package and required callable exports | Native build selection (`--ops`) |
| --- | --- | --- |
| SiTU | `cann_ops_nn`: `situ_glu`, `situ_glu_grad` | `situ_glu,situ_glu_grad` |
| Block attention residuals | `cann_ops_transformer`: `block_attention_residuals`, `block_attention_residuals_backward` | `block_attention_residuals,block_attention_residuals_grad` |

Follow the installation instructions in [ops-nn](https://gitcode.com/cann/ops-nn)
and [ops-transformer](https://gitcode.com/cann/ops-transformer), building for the
target CANN version and SoC. Load CANN and both custom-operator environments before
training. Keep the canonical Python package names above. The revisions below compile
the PyTorch/C++ bridge on first use, so a compiler, Ninja and CANN headers are required;
use a separate `TORCH_EXTENSIONS_DIR` when changing toolchains.

The wrappers load these optional packages only when a nonempty fused path is selected
and check that both training entry points are callable. Import or dynamic-library
loading failures raise an informative `ImportError` with the original cause preserved.

- Missing Python package or API: install the matching extension with both forward and backward.
- Missing libraries or undefined symbols: check PyTorch/torch-npu/CANN compatibility,
  `ASCEND_CUSTOM_OPP_PATH` and `LD_LIBRARY_PATH` for both native packages.
- Native input, compilation and execution errors propagate unchanged. The dispatcher
  and upstream entry points validate their device, shape and dtype contracts; the
  wrapper does not duplicate those checks or silently select eager.

### Evaluated versions

The standalone operators were built with CANN `9.1.0-beta.3` and
PyTorch/torch-npu `2.10.0` at these source revisions:

| Upstream | Source revision |
| --- | --- |
| ops-nn | `b7af75e68cc07c0912f9e020bce639e53de60700` |
| ops-transformer | `a041b05638a5a3f94b79c97ce196a331738a56a1` |

## Linear attention backends

`backend="ascendc"` for local KDA/GDN and pure Ulysses requires
[fla-npu](https://github.com/flashserve/flash-linear-attention-npu), installed with
matching Python bindings, stable launcher and native operator libraries.
The adapter targets commit `2bb7e74ecf4876acbc4527070e403bffeef06e75` (including #993).
Install the wheel appropriate to the target hardware and CANN environment using
[the upstream installation guide](https://github.com/flashserve/flash-linear-attention-npu).

KDA requires `chunk_kda_fwd_prepare` save mode, `chunk_fwd_h`,
`chunk_kda_fwd_finalize`, and `chunk_kda_bwd` with the single `layout` argument
and `beta_eff` cache. `dO`, raw gate/beta and their gradients are token-major;
normalized Q/K and saved Prepare caches are head-major. The native sigmoid
backward is used where implemented; other implementations require FLA's
`fused_beta_sigmoid_bwd`. The paired FLA source is
`6b6f546f9bd10a97be1e8cf2f0120de70cfca482` with Triton-Ascend support.

GDN requires `chunk_gated_delta_rule_fwd` and `chunk_gated_delta_rule_bwd` with
`layout`, `cu_seqlens` and `chunk_indices`. Normalization and effective gate/beta
remain in the caller's autograd graph. Both wrappers use per-invocation save
mode and reject unsupported state-gradient requests. Pure Ulysses uses the
same backend for causal convolution; Ascend C additionally requires
`causal_conv1d` and `causal_conv1d_bwd` with the SiLU derivative included.
