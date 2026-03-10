---
description: Rules for distributed system code (DTensor, collectives, stream sync, memory)
paths:
  - hyper_parallel/core/**
  - hyper_parallel/collectives/**
  - hyper_parallel/platform/torch/fully_shard/**
  - hyper_parallel/platform/torch/hsdp/**
---

## DTensor

- `is_partial()` is a **method**, not a property — always call with parentheses
- Call `reduce_partial` before `redistribute()` if layout is in partial state
- ReduceScatter must be ordered before AllReduce (see `tensor_redistribution.py`)
- Use `SkipDTensorDispatch` context manager when operating on raw local tensors inside gradient hooks
- Distributed ops are registered via YAML in `core/shard/ops/yaml/`; implementations in `core/shard/ops/parallel_*.py`

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
