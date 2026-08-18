# High-performance functions

`hyper_models.ops` provides reusable high-performance functions for Ascend NPU. CPU and GPU fallback implementations
are not included.

```python
from hyper_models.ops import rms_norm

output = rms_norm(x, weight)
```
