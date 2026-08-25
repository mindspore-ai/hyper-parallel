# High-performance functions

`hyper_parallel.auto_models.ops` provides reusable high-performance functions for Ascend NPU. CPU and GPU fallback implementations
are not included.

```python
from hyper_parallel.auto_models.ops import rms_norm

output = rms_norm(x, weight)
```

RoPE functions keep the Transformers call contract:

```python
from hyper_parallel.auto_models.ops import apply_rotary_pos_emb

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
from hyper_parallel.auto_models.ops import (
    moe_token_permute,
    moe_token_unpermute,
)

permuted_tokens, sorted_indices = moe_token_permute(tokens, expert_indices)
expert_outputs = experts(permuted_tokens, tokens_per_expert, routing_weights)
output = moe_token_unpermute(expert_outputs, sorted_indices, routing_probs)
```

In this example, `experts` is an instance of `hyper_parallel.auto_models.modules.GroupedExperts`. It contains only local expert
weights and computation; routing, expert-parallel communication, and shared experts remain outside the module.
`hyper_parallel.auto_models.modules.SharedExpert` provides the parameter-owning dense MLP used as a shared expert.

## Auxiliary-loss functions

`hyper_parallel.auto_models.ops` also exports `aux_loss_auto_scale` and `set_aux_loss_scale` for auxiliary-loss gradient injection.
