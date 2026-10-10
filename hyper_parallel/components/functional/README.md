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

## Optional DeepSeek and TorchTitan-NPU Ascend operators

DeepGEMM-Ascend, DeepSelect, FlashMLA, and TileKernels are available through
lazy optional adapters. They are not imported by a plain HyperParallel import
and must be built separately for the active PyTorch, torch-npu, and CANN
environment. See the [DeepSeek Ascend operator guide](../../../docs/guide/deepseek_ascend_ops.md).

The same guide documents the lazy `torchtitan_*` adapters for TorchTitan-NPU's public AscendC, TileLang, and
Triton production operators. Installing TorchTitan-NPU is optional and is only required when one of these adapters is called.

Common GEMM, quantization, SwiGLU, MoE routing, RoPE, and Engram paths have
explicit public functions. Generic dispatch remains available for additional
public production operators as upstream packages evolve.
