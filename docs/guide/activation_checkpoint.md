# Activation Checkpoint / Swap Guide

**English** | [简体中文](activation_checkpoint.zh.md)

HyperParallel provides flexible activation memory optimization through selective recomputation (Activation Checkpoint) and Activation Swap, which offloads intermediate activations to CPU and prefetches them during backward. Both mechanisms can be configured together.

## Core Concepts

| Mechanism | Principle | Memory savings | Compute overhead |
|------|------|----------|----------|
| Activation Checkpoint | Omit intermediate activations during forward and recompute them during backward | High (activations are not stored) | Additional forward computation during backward |
| Activation Swap | Offload activations to CPU during forward and prefetch them during backward | High (does not occupy NPU HBM) | Low (asynchronous DMA transfers) |
| Combined configuration | Checkpoint compute-intensive layers and swap transfer-intensive layers | Maximized | Optimized |

## API Overview

| Interface | Description |
|------|------|
| `checkpoint` | Functional activation checkpointing; default non-reentrant mode supports policy_fn, swap_inputs, and context_fn |
| `swap` | Functional activation swapping: offload/prefetch without recomputation |
| `checkpoint_wrapper` | Module-level checkpoint wrapper; supports `use_reentrant=True` |
| `checkpoint_exclude_wrapper` | Exclude a region inside a checkpoint from recomputation, executing its forward once |
| `swap_wrapper` | Module-level swap wrapper |
| `swap_tensor_wrapper` | Single-tensor swap wrapper |
| `CheckpointPolicy` | Recomputation policy enum: MUST_SAVE / PREFER_SAVE / MUST_RECOMPUTE / PREFER_RECOMPUTE / MUST_SWAP |
| `SwapManager` | Swap group manager (singleton) |

---

## Usage

### 1. Functional Checkpoint

```python
from hyper_parallel.core.activation_memory import checkpoint, CheckpointPolicy

# Basic usage: wrap model.layer in a checkpoint region; x is its input.
output = checkpoint(model.layer, x)

# With policy_fn: selectively save, offload, or recompute.
def my_policy(ctx, op, *args, **kwargs):
    # MUST_SAVE saves the operator output; MUST_SWAP offloads it to the host; MUST_RECOMPUTE recomputes it during backward.
    op_name_attr = getattr(op, "name", None)
    op_name = op_name_attr() if callable(op_name_attr) else (op_name_attr or str(op))
    if op_name in {"LayerNormExt", "aten.addmm.default"}:
        return CheckpointPolicy.MUST_SAVE
    return CheckpointPolicy.MUST_RECOMPUTE

output = checkpoint(model.layer, x, policy_fn=my_policy)

# With swap_inputs: also offload the checkpoint region's saved inputs to CPU.
output = checkpoint(model.layer, x, swap_inputs=True)

# With context_fn: provide separate context managers for forward and recomputation.
def context_fn():
    return forward_context(), recompute_context()

output = checkpoint(model.layer, x, context_fn=context_fn)

# Control whether recomputation stops once all tensors needed by backward are available.
output = checkpoint(model.layer, x, early_stop=True)   # Default: avoid unnecessary trailing computation.
output = checkpoint(model.layer, x, early_stop=False)  # Execute the entire recomputation region.
```

The default `use_reentrant=False` uses HyperParallel's non-reentrant implementation. `early_stop` is controlled only through
HyperParallel's per-call keyword or configuration dictionary:

```python
checkpoint_kwargs = {"early_stop": False, "preserve_rng_state": True}
output = checkpoint(model.layer, x, **checkpoint_kwargs)
```

An outer `torch.utils.checkpoint.set_checkpoint_early_stop()` does not override HyperParallel eager checkpoint arguments.
`checkpoint_wrapper(model.layer, early_stop=False)` follows the same rules. Graph compilation currently falls back to Torch's native
non-reentrant checkpoint; complete support for compile/context_fn combinations is outside the current release scope.

### Reentrant Checkpointing and Excluded Regions

In Torch eager mode, pass `use_reentrant=True` to `checkpoint` or `checkpoint_wrapper`.
A region marked with `checkpoint_exclude_wrapper` executes during the initial forward, retaining necessary outputs and its backward graph. Checkpoint
replay reuses its outputs without executing its forward again:

```python
from hyper_parallel.core.activation_memory import checkpoint_wrapper, checkpoint_exclude_wrapper

# Call the excluded region inside the checkpointed module; the parent still owns its parameters.
model.block.expensive = checkpoint_exclude_wrapper(model.block.expensive)
model.block = checkpoint_wrapper(model.block, use_reentrant=True)
```

Excluded regions always cache outputs. With `swap=True`, the current swap group can offload outputs and internal activations that cannot be recovered from
recomputation inputs; those inputs are still restored through recipes. Configure `group_swap` and `cpu_pool` on the exclude wrapper.
Reentrant checkpointing itself supports `preserve_rng_state` and a callable `context_fn`, but its outer call does not support
`policy_fn`, `swap_inputs`, `group_swap`, `cpu_pool`, nondefault `early_stop`, `debug`,
`determinism_check`, or `torch.compile`; these combinations raise errors. Do not nest reentrant checkpoints.

Ordinary PP-stage backward can use reentrant checkpointing; the stage automatically manages phased scopes without manual binding.
With `ScheduleInterleaved1F1B(overlap_b_f=True, enable_dxdw_split=True)`, dx/dw separation
retains replay state per micro-batch in non-FSDP/HSDP stages: dx propagates only input gradients, then dw computes parameter gradients,
sharing one replay between both phases. Reentrant + PP split + FSDP/HSDP and
`retain_backward_graph=True` are unsupported and explicitly rejected; ordinary FSDP without PP split is unaffected.
Outside a PP stage's phased scope, reentrant internals may accumulate gradients for closure-captured parameters even when
`autograd.grad` requests only input gradients. Use non-reentrant mode for full differentiation restricted to specified targets.
The current implementation does not guarantee higher-order gradients or arbitrary nested checkpoints.
PP reentrant stages assume parameter objects remain unchanged throughout execution; dynamically replacing parameters is unsupported.

### 2. Functional Swap

```python
from hyper_parallel.core.activation_memory import swap, CheckpointPolicy, SwapManager

# Basic usage: swap all intermediate activations to CPU.
output = swap(model.layer, x)

# With policy_fn: selectively swap activations.
def swap_policy(target):
    if target.is_small_op:
        return CheckpointPolicy.MUST_SAVE  # Do not swap small tensors.
    return CheckpointPolicy.MUST_SWAP  # Swap the remaining tensors.

output = swap(model.layer, x, policy_fn=swap_policy)

# Register inter-layer offload/prefetch relationships when using swap; layers need not be adjacent.
SwapManager().set_forward_prefetch_layer(model.layers[i], model.layers[i + 1])
```

### 3. Declarative Interfaces

```python
from hyper_parallel.core.activation_memory import (
    CheckpointPolicy,
    SwapManager,
    checkpoint_exclude_wrapper,
    checkpoint_wrapper,
    swap_wrapper,
)

# These examples show alternatives; do not nest wrappers repeatedly on the same module.

# Replace the module with checkpoint_wrapper: each forward enters a checkpoint region.
model.layers[0] = checkpoint_wrapper(model.layers[0])

# checkpoint_wrapper accepts the same keywords as checkpoint, including policy_fn, swap_inputs, and group_swap.
def recompute_policy(ctx, op, *args, **kwargs):
    return CheckpointPolicy.MUST_RECOMPUTE

model.layers[1] = checkpoint_wrapper(model.layers[1], policy_fn=recompute_policy)

# SAVE regions always cache outputs; swap=True can offload outputs and nonreconstructible internal activations to CPU.
model.expensive_a = checkpoint_exclude_wrapper(
    model.expensive_a,
    swap=True,
    group_swap=True,
)
model.expensive_b = checkpoint_exclude_wrapper(model.expensive_b)
SwapManager().set_forward_prefetch_layer(model.expensive_a, model.expensive_b)

# Without an enclosing active checkpoint, checkpoint_exclude_wrapper passes through and swap settings have no effect.

# Replace the module with swap_wrapper: intermediate activations saved during forward are offloaded to CPU according to policy.
def tensor_swap_policy(tensor):
    if tensor.numel() < 1024:
        return CheckpointPolicy.MUST_SAVE  # Keep small tensors on the device.
    return CheckpointPolicy.MUST_SWAP

model.layers[2] = swap_wrapper(model.layers[2], policy_fn=tensor_swap_policy)

# Combined use: start with an unwrapped model and select one wrapper for each layer.
for i, layer in enumerate(model.layers):
    if i % 2 == 0:
        model.layers[i] = checkpoint_wrapper(layer)
    else:
        model.layers[i] = swap_wrapper(layer)

# Register inter-layer offload/prefetch relationships when using swap; layers need not be adjacent.
SwapManager().set_forward_prefetch_layer(model.layers[i], model.layers[i + 1])

# checkpoint_wrapper and swap_wrapper accept nn.Module instances or ordinary functions.
# Wrapping a function returns a callable with the same calling convention.
def block_func(x):
    return model.block(x)

ckpt_block_func = checkpoint_wrapper(block_func)
swap_block_func = swap_wrapper(block_func)

ckpt_output = ckpt_block_func(x)
swap_output = swap_block_func(x)
```

### 4. Offload Individual Tensors

Use `swap_tensor_wrapper` to offload specific large activation tensors. Call it inside the module's `forward`; it registers eligible tensors with the current swap group for subsequent offload/prefetch scheduling.

```python
import torch.nn as nn

from hyper_parallel.core.activation_memory import SwapManager, swap_tensor_wrapper

class TransformerBlock(nn.Module):
    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        attn_out = swap_tensor_wrapper(attn_out, tag="attn_out")

        x = x + attn_out
        x = self.norm1(x)
        x = swap_tensor_wrapper(x, tag="norm1_out")

        x = x + self.ffn(x)
        return self.norm2(x)

# Register inter-layer offload/prefetch relationships for swap_tensor_wrapper as well; layers need not be adjacent.
SwapManager().set_forward_prefetch_layer(model.layers[i], model.layers[i + 1])
```

---

## Combine with Other Parallel Strategies

### FSDP + Activation Checkpoint / Activation Swap

```python
from hyper_parallel import fully_shard, init_device_mesh
from hyper_parallel.core.activation_memory import checkpoint_wrapper

mesh = init_device_mesh("npu", (dp_size,), mesh_dim_names=("dp",))

# Apply checkpoint_wrapper or swap_wrapper to selected submodules first, placing wrappers inside the FSDP boundary.
model.transformer.layer = checkpoint_wrapper(model.transformer.layer)

# Then apply FSDP around the checkpointed module.
model = fully_shard(model, mesh=mesh)
```

### TP + Activation Checkpoint / Activation Swap

```python
from hyper_parallel import ColwiseParallel, RowwiseParallel, parallelize_module, init_device_mesh
from hyper_parallel.core.activation_memory import SwapManager, swap_wrapper

tp_mesh = init_device_mesh("npu", (tp_size,), mesh_dim_names=("tp",))

# Apply TP first.
parallelize_module(model, tp_mesh, tp_plan)

# Then apply checkpoint_wrapper or swap_wrapper to selected layers or submodules.
for i, layer in enumerate(model.layers):
    model.layers[i].attn = swap_wrapper(layer.attn)

# Register inter-layer offload/prefetch relationships when using swap; layers need not be adjacent.
SwapManager().set_forward_prefetch_layer(model.layers[i], model.layers[i + 1])
```

---

## Performance Recommendations

1. **Do not swap small tensors**: DMA startup costs may outweigh the benefits for small tensors such as biases; filter them with `policy_fn`.
2. **Prioritize large activations for swapping**: Attention outputs and MLP intermediate activations are suitable candidates.
3. **Choosing checkpoint or swap**:
   - Compute-intensive layers such as softmax: use checkpoint when recomputation overhead is acceptable.
   - Memory-intensive layers such as large MLPs: use swap for lower DMA transfer overhead.
4. **Combined configuration**: Mix checkpointing and swapping to maximize memory savings while minimizing computation and transfer overhead.
