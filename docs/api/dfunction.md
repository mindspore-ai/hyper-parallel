# DFunction — Custom Distributed Autograd Functions

HyperParallel provides `DFunction`, a base class for writing custom distributed
functions that integrate seamlessly with the DTensor dispatch system.  Users
subclass `DFunction` (which inherits from the platform's autograd `Function`)
and implement `forward` / `backward` as plain local-tensor operations.  When the
inputs are `DTensor`s, the dispatch system transparently handles layout inference,
local-tensor extraction, and output wrapping — no changes to user code are needed
for the multi-card path.

## Core Classes

### `DFunction`

Base class for user-defined distributed autograd functions.

```python
class DFunction(platform.Function):
    _op_name: str = None          # Set this to the registered DistributedOp name

    @staticmethod
    def forward(ctx, *args, **kwargs) -> Tensor: ...

    @staticmethod
    def backward(ctx, *grad_outputs) -> ...: ...

    @classmethod
    def apply(cls, *args, **kwargs) -> Tensor | DTensor: ...
```

**Dispatch behaviour:**

| Input type | Route |
|-----------|-------|
| Plain `Tensor` | `super().apply()` — platform autograd, single-device path |
| At least one `DTensor` | `dispatch()` — layout inference + DTensor wrapping |

**`_op_name`** must be set when DTensor inputs are expected.  It must match the
`op_name` passed to a registered `DistributedOp` subclass.  Omitting `_op_name`
while passing a `DTensor` raises `ValueError`.

**`forward(ctx, *args)`** operates on **local** tensors.  When the distributed
path is taken, the dispatcher extracts the local shard from each input `DTensor`
and passes it here.  Non-tensor positional arguments are forwarded to `forward`
unchanged (see `DistributedOp.preprocess`).

**`backward(ctx, *grad_outputs)`** likewise operates on local tensors.  It is
identical to a standard `torch.autograd.Function.backward` or MindSpore's
`_Function.backward`.

---

### `DistributedOp`

Base class for local-tensor extraction and layout inference.

```python
class DistributedOp:
    def __init__(self, op_name: str): ...

    def preprocess(self, args: tuple, kwargs: dict) -> tuple: ...

    def infer_layout(self, cache_values: list) -> tuple: ...

    def get_expand_impl(self, func, infer_result, cache_values) -> None | Callable: ...
```

Instantiating a `DistributedOp` subclass **automatically registers** it under
`op_name`.  Exactly one instance must exist per `op_name` before the first call
to the corresponding `DFunction.apply`.

#### `preprocess(args, kwargs)`

Parses the call arguments, extracts the local tensors, and builds the layout
cache key.  Called once before layout inference on the **first** dispatch (cached
thereafter).  Returns `(local_args, local_kwargs, cache_values)`:

- `local_args` / `local_kwargs` — the local-tensor positional / keyword arguments
  passed to the user's `forward` (input `DTensor`s already `to_local`'d).
  Non-tensor positional arguments are forwarded here unchanged.
- `cache_values` — an ordered list of values used as the layout cache key
  (typically `Layout` objects plus scalars such as `bool`, `int`).

#### `infer_layout(cache_values)`

Computes the output layout(s) from `cache_values` and returns
`(out_layouts_tuple, None)`.  For multi-output ops, `out_layouts_tuple` holds one
`Layout` per output.

#### `get_expand_impl(func, infer_result, cache_values)`

Optional.  Returns `None` (default) or a callable that replaces the default
`func(*local_args)` call.  Use this to modify how local arguments are combined
before the computation — the canonical example is bias scaling in row-parallel
linear where each rank needs `bias / tp_size` instead of the full `bias`.

---

## Quick Start

```python
from hyper_parallel import init_device_mesh, DFunction
from hyper_parallel.core.dtensor.dtensor import distribute_tensor
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from hyper_parallel.core.shard.ops.parallel_ops import DistributedOp

# ── Step 1: Register a DistributedOp ─────────────────────────────────────────

class MyAddDistOp(DistributedOp):
    def __init__(self):
        super().__init__("MyAdd")

    def preprocess(self, args, kwargs):
        x, y = args[0], args[1]
        local_args = (x.to_local(), y.to_local())
        cache_values = [x.layout, y.layout]
        return local_args, {}, cache_values

    def infer_layout(self, cache_values):
        return ((cache_values[0],), None)  # element-wise: output layout = first input layout

MyAddDistOp()  # instantiation registers the op

# ── Step 2: Implement DFunction ───────────────────────────────────────────

class MyAdd(DFunction):
    _op_name = "MyAdd"

    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x, y)
        return x + y

    @staticmethod
    def backward(ctx, grad):
        return grad, grad

# ── Step 3: Call ─────────────────────────────────────────────────────────────

# Single-device (plain tensors)
result = MyAdd.apply(x_local, y_local)

# Multi-device (DTensors — dispatched automatically)
mesh = init_device_mesh("npu", (2, 4), mesh_dim_names=("dp", "tp"))
x_dist = distribute_tensor(x, mesh, (Shard(0), Replicate()))
y_dist = distribute_tensor(y, mesh, (Shard(0), Replicate()))
result_dist = MyAdd.apply(x_dist, y_dist)  # returns DTensor
```

---

## Usage Patterns

### Pattern 1: Element-wise op

The simplest case: the output keeps the first input's layout.  The non-tensor
`scale` argument is forwarded to `forward` through `local_args`.

```python
class _ScaleDistOp(DistributedOp):
    def __init__(self):
        super().__init__("Scale")

    def preprocess(self, args, kwargs):
        x, scale = args[0], args[1]
        local_args = (x.to_local(), scale)   # non-tensor scale forwarded here
        cache_values = [x.layout]
        return local_args, {}, cache_values

    def infer_layout(self, cache_values):
        return ((cache_values[0],), None)   # scale is element-wise

_ScaleDistOp()


class ScaleFunc(DFunction):
    _op_name = "Scale"

    @staticmethod
    def forward(ctx, x, scale):
        ctx.save_for_backward(x)
        ctx.scale = scale
        return x * scale

    @staticmethod
    def backward(ctx, grad):
        return grad * ctx.scale, None  # None for the non-tensor scale
```

---

### Pattern 2: Column-parallel Linear

`infer_layout` receives `cache_values` and returns `(out_layouts_tuple, None)`,
deriving the output layout from the input and weight layouts.

```python
class LinColDistOp(DistributedOp):
    def __init__(self):
        super().__init__("LinCol")

    def preprocess(self, args, kwargs):
        x, w = args[0], args[1]
        local_args = (x.to_local(), w.to_local())
        cache_values = [x.layout, w.layout]
        return local_args, {}, cache_values

    def infer_layout(self, cache_values):
        x_layout, w_layout = cache_values[0], cache_values[1]
        # derive output layout from x[.., in] and w[out, in] ...
        out_layout = _linear_output_layout(x_layout, w_layout)
        return ((out_layout,), None)

    def get_expand_impl(self, func, infer_result, cache_values):
        return None   # no bias → no scaling needed

LinColDistOp()


class LinColFunc(DFunction):
    _op_name = "LinCol"

    @staticmethod
    def forward(ctx, x, w):
        ctx.save_for_backward(x, w)
        return F.linear(x, w)

    @staticmethod
    def backward(ctx, grad):
        x, w = ctx.saved_tensors
        return grad @ w, grad.t() @ x
```

---

### Pattern 3: Row-parallel Linear with bias scaling (`get_expand_impl`)

When the contracting dimension is sharded across TP ranks, each rank computes a
partial sum.  Adding the full `bias` on every rank would over-count it.
`get_expand_impl` returns a replacement callable that pre-scales `bias` by
`1 / tp_size`:

```python
class LinRowDistOp(DistributedOp):
    def __init__(self):
        super().__init__("LinRow")

    def preprocess(self, args, kwargs):
        x, w, bias = args[0], args[1], args[2]
        local_args = (x.to_local(), w.to_local(), bias)
        cache_values = [x.layout, w.layout, bias is not None]
        return local_args, {}, cache_values

    def infer_layout(self, cache_values):
        x_layout, w_layout = cache_values[0], cache_values[1]
        out_layout = _linear_output_layout(x_layout, w_layout)
        return ((out_layout,), None)

    def get_expand_impl(self, func, infer_result, cache_values):
        x_layout = cache_values[0]
        bias_present = cache_values[2]
        contract = x_layout.alias_tensor_map[-1]
        if contract == "None" or not bias_present:
            return None
        # tp_size: number of TP ranks sharing the contracting dimension
        tp_size = x_layout.mesh.get_device_num_along_axis(contract)

        def expand_impl(x, w, bias):
            return func(x, w, bias / tp_size)   # scale bias down

        return expand_impl

LinRowDistOp()


class LinRowFunc(DFunction):
    _op_name = "LinRow"

    @staticmethod
    def forward(ctx, x, w, bias):
        ctx.save_for_backward(x, w, bias)
        return F.linear(x, w, bias)

    @staticmethod
    def backward(ctx, grad):
        x, w, bias = ctx.saved_tensors
        return grad @ w, grad.t() @ x, grad.sum(0)


# Usage — contracting (in) dimension sharded across TP
mesh = init_device_mesh("npu", (2, 4), mesh_dim_names=("dp", "tp"))
x_dist = distribute_tensor(x, mesh, (Replicate(), Shard(1)))
w_dist = distribute_tensor(w, mesh, (Replicate(), Shard(1)))

result = LinRowFunc.apply(x_dist, w_dist, bias)
# result is a partial DTensor; reduce before redistribution
output = result.reduce_partial()
```

---

## Backward Gradient Access

Inside distributed tests, accessing gradients for non-leaf local tensors requires
`retain_grad()` before the forward pass:

```python
x_dist = distribute_tensor(x_leaf, mesh, placements)

x_local = x_dist.to_local()   # returns the internal local tensor (non-leaf)
x_local.retain_grad()          # allow grad accumulation on non-leaf

result = MyFunc.apply(x_dist)
result.to_local().sum().backward()

print(x_local.grad)            # gradient available here
```

`x_dist.to_local()` returns the same `_local_tensor` object every call, so
calling `retain_grad()` once before the forward is sufficient.

---

## Dispatch Internals (Reference)

```text
DFunction.apply(dtensor1, ...)
  │  at least one DTensor input
  ▼
dispatch(local_callable, args, kwargs)
  │
  ├─ preprocess(args, kwargs)        → local_args, cache_values
  ├─ infer_layout(cache_values)      → output layouts
  ├─ get_expand_impl() → op_impl     (None → use local_callable directly)
  │
  └─ py_output = op_impl(*local_args)
       │
       └─ local_callable(*local_args)
            │  no DTensor input
            └─ super().apply(*local_args)   ← platform autograd, no recursion
                 └─ MyFunc.forward(ctx, local_tensor1, ...)
```

---

## Constraints

1. `_op_name` must be set on the `DFunction` subclass and match the registered
   `DistributedOp` exactly.
2. Each `op_name` can have only one registered `DistributedOp` instance at a time.
3. `cache_values` must include every value that affects layout inference so the
   layout cache key stays correct.
4. When using `get_expand_impl` and the output has partial status, call
   `result.reduce_partial()` before redistribution.
5. Both `forward` and `backward` **must** operate on local tensors — do not call
   `DFunction.apply` recursively from within them.

---

## Platform Support

`DFunction` is platform-agnostic.  It inherits from `platform.Function` which
resolves to the correct base class at import time.

| Platform | `platform.Function` | `forward` / `backward` tensor type |
|----------|--------------------|------------------------------------|
| PyTorch (GPU / NPU) | `torch.autograd.Function` | `torch.Tensor` |
| MindSpore (Ascend NPU) | `mindspore._Function` | `mindspore.Tensor` |

Cross-platform code runs identically on both backends.  The `ctx.save_for_backward`
/ `ctx.saved_tensors` contract is uniform across platforms.
