---
name: distributed
description: Rules for distributed system code (DTensor, collectives, stream sync, memory)
paths:
  - hyper_parallel/core/**
  - hyper_parallel/collectives/**
  - hyper_parallel/distributed/**
---

# Distributed Systems

## DTensor

- `is_partial()` is a **method**, not a property — always call with parentheses
- Call `reduce_partial` before `redistribute()` if layout is in partial state
- ReduceScatter must be ordered before AllReduce (see `core/dtensor/tensor_redistribution.py`)
- Use `SkipDTensorDispatch` context manager when operating on raw local tensors inside gradient hooks
- Distributed ops are registered via YAML in `core/shard/ops/yaml/`; implementations in `core/shard/ops/parallel_*.py`

## Collective Calling Conventions

The `platform/` abstraction is gone — these conventions now apply to direct Torch calls and to
the shared wrappers in `core/context_parallel/utils.py` and `core/dtensor/_utils.py`. Stream
ordering, autograd correctness, and memory-lifetime requirements below still apply everywhere.

### `differentiable_*` vs eager collectives

- Code in `TensorRedistribution` and any forward/backward computation path **must** use the `differentiable_*` helpers, not the eager ones.
- Eager `dist.all_reduce` / `dist.reduce_scatter_tensor` are only for contexts outside autograd (e.g., parameter sync, buffer broadcast).
- When adding a new collective call, ask: "Does this tensor need gradients?" — if yes, use `differentiable_*`.

### `group` vs `group_info` parameter types

- The eager helpers in `core/context_parallel/utils.py` expect a **`group_info` object** with a `.group` attribute.
- The `differentiable_*` helpers in `core/dtensor/_utils.py` expect a **raw group**.
- `create_group()` (`core/dtensor/_utils.py`) returns a **raw group** — wrap it with `SimpleNamespace(group=group)` before passing it to an eager helper.
- When in doubt, read the wrapper signature before wiring a call.

## Stream Synchronization

- `async_op=True` handles must be waited via `handle.wait()` before accessing the output tensor
- `non_blocking=True` transfers execute asynchronously — destination tensor must not be read until stream completes
- Cross-stream dependencies require events: `event.record(stream_A)` then `event.wait(stream_B)` — CPU code order does NOT guarantee GPU execution order
- `grad_sync_stream` is only used in legacy HSDP path (`HSDPSchedulerV2 + comm_async=True`)
- Activation Swap: `launch_offload/launch_load` run on `copy_stream`; must `wait_offload/wait_load` before compute stream access

## Memory Management

- Call `tensor.untyped_storage().resize_(0)` to immediately free device memory after use
- Clear communication buffers (`clear_reduce_scatter_output()` / `clear_all_reduce_output()`) after consuming reduced gradients
- Set `param.grad = None` after gradient consumed to release tensor
- Prefer buffer reuse (`resize_`) over reallocation
- `SwapGroup._storages` uses `weakref.WeakSet` — storage references auto-released on GC
- Pipeline: call `_clear_recv_buffer()` and `clear_cache()` after each micro-batch
- Activation swap: `wait_offload()` frees device storage, `wait_load()` frees CPU storage — missing either causes memory growth
