# High-performance functions

`hyper_models.ops` provides reusable high-performance functions for Ascend NPU. CPU and GPU fallback implementations
are not included.

```python
from hyper_models.ops import rms_norm

output = rms_norm(x, weight)
```

RoPE functions keep the Transformers call contract:

```python
from hyper_models.ops import apply_rotary_pos_emb

query, key = apply_rotary_pos_emb(query, key, cos, sin)
```

Use `apply_rotary_pos_emb_interleave` for Transformers models whose source
attention calls the interleaved variant.
