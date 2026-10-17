# High-performance functions

`hyper_parallel.components.functional` provides reusable high-performance functions for Ascend NPU. CPU and GPU fallback implementations
are not included.

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
| Hardware / CANN Toolkit | Ascend 910B3 / 9.0.0 |
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
    module_type: hyper_parallel.models.deepseek_v41.modeling_deepseek_v41.DeepseekV41Attention
    replace_module:
      _target_: >-
        hyper_parallel.components.modules.shared_compressed_dsa_attention.SharedCompressedDSAAttention
      use_fused_ops: true
```

`false` (default) uses the Torch reference implementation without this extension.
`true` requires the fused entry points used by the model's execution path and
raises on missing entries or unsupported inputs, without automatic fallback.
The fused path supports ordinary causal CSA with compression ratio 1 or 2;
candidate-pool selection is not supported.
