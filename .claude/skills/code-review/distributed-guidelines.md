# Distributed System Guidelines

This document covers distributed system correctness considerations for HyperParallel PR reviews.

As a top level principle, **missing stream synchronization is the leading root cause of memory stomping and stale data bugs**. Any change involving async operations, cross-stream access, or device transfers must be reviewed with extreme care.

As a reviewer, you MUST be paranoid about stream sync and memory lifecycle. These bugs are silent — they produce incorrect results without exceptions or crashes.

## Stream Synchronization

### The Core Problem

GPU operations execute asynchronously on streams. CPU code ordering does NOT guarantee GPU execution ordering across different streams. The only way to establish GPU-side ordering across streams is via events.

### Rules

| Pattern | Requirement | Consequence of Violation |
|---------|-------------|--------------------------|
| `async_op=True` collective | `handle.wait()` before reading output | Stale or partial data |
| `non_blocking=True` transfer | Stream sync before reading destination | Reading uninitialized memory |
| Cross-stream tensor access | `event.record(src)` → `event.wait(dst)` | Memory stomping |
| Activation swap offload/load | `wait_offload/wait_load` before compute | Corrupted activations |

### Common Violations

#### 1. Missing handle.wait()

**Bad:**
```python
handle = all_gather(output, input, async_op=True)
# BUG: using output before wait
result = output.reshape(...)
```

**Good:**
```python
handle = all_gather(output, input, async_op=True)
handle.wait()  # GPU-side dependency via cudaStreamWaitEvent
result = output.reshape(...)
```

#### 2. non_blocking Without Sync

**Bad:**
```python
gpu_tensor = cpu_tensor.to(device, non_blocking=True)
# BUG: gpu_tensor may not be ready
loss = model(gpu_tensor)
```

**Good:**
```python
gpu_tensor = cpu_tensor.to(device, non_blocking=True)
torch.cuda.current_stream().synchronize()  # or use event wait
loss = model(gpu_tensor)
```

#### 3. Cross-Stream Access Without Event

**Bad:**
```python
with torch.cuda.stream(comm_stream):
    all_reduce(grad)
# BUG: default stream may read grad before comm_stream finishes
optimizer.step()
```

**Good:**
```python
with torch.cuda.stream(comm_stream):
    all_reduce(grad)
    event = torch.cuda.Event()
    event.record(comm_stream)
event.wait(torch.cuda.current_stream())
optimizer.step()
```

#### 4. Activation Swap Stream Mismatch

**Bad:**
```python
swap_group.launch_load()  # runs on copy_stream
# BUG: compute stream doesn't wait for copy_stream
output = layer(activation)
```

**Good:**
```python
swap_group.launch_load()
swap_group.wait_load()  # event wait: copy_stream → compute stream
output = layer(activation)
```

### HSDP Stream Paths

Two distinct code paths exist — do not confuse them:

| Path | Scheduler | Stream Usage | Grad Hook Type |
|------|-----------|-------------|----------------|
| Legacy | `HSDPSchedulerV2` + `comm_async=True` | Uses `grad_sync_stream` | Per-parameter |
| Current | `TorchHSDPSchedulerV2` | No `grad_sync_stream` | Module-level backward → `post_backward()` → `reduce_params()` |

## Memory Lifecycle

### The Core Problem

Device memory is scarce. Tensors used as intermediate buffers (all-gather outputs, communication buffers, gradients) must be freed immediately after consumption. Failure to free causes OOM in long training loops.

### Patterns

| Pattern | When | How |
|---------|------|-----|
| Free device memory | After consuming all-gather output | `tensor.untyped_storage().resize_(0)` |
| Clear comm buffers | After consuming reduced gradients | `clear_reduce_scatter_output()` / `clear_all_reduce_output()` |
| Null grad references | After gradient consumed | `param.grad = None` |
| Reuse buffers | When buffer size is known | `resize_(expected_size)` instead of new allocation |
| Pipeline cleanup | After each micro-batch | `_clear_recv_buffer()` + `clear_cache()` |
| Swap cleanup | After offload/load completes | `wait_offload()` frees device; `wait_load()` frees CPU |
| Weak references | Shared storage ownership | `weakref.WeakSet` for auto-release on GC |

### Common Leaks

#### 1. Missing Storage Free

**Bad:**
```python
unsharded = all_gather(sharded_param)
output = compute(unsharded)
# BUG: unsharded stays alive, wasting device memory
```

**Good:**
```python
unsharded = all_gather(sharded_param)
output = compute(unsharded)
unsharded.untyped_storage().resize_(0)  # free immediately
```

#### 2. Stale Gradient Reference

**Bad:**
```python
reduced_grad = reduce_scatter(grad)
param.data -= lr * reduced_grad
# BUG: param.grad still points to old gradient tensor
```

**Good:**
```python
reduced_grad = reduce_scatter(grad)
param.data -= lr * reduced_grad
param.grad = None  # release reference
```

#### 3. Pipeline Buffer Accumulation

**Bad:**
```python
for micro_batch in micro_batches:
    output = stage.forward(micro_batch)
    stage.backward(output)
    # BUG: recv buffers and caches accumulate across micro-batches
```

**Good:**
```python
for micro_batch in micro_batches:
    output = stage.forward(micro_batch)
    stage.backward(output)
    stage._clear_recv_buffer()
    stage.clear_cache()
```

#### 4. Incomplete Activation Swap

**Bad:**
```python
swap_group.launch_offload()
swap_group.wait_offload()
# device storage freed ✓
# ... later during backward ...
swap_group.launch_load()
swap_group.wait_load()
# BUG: CPU storage not freed — memory grows with layer count
```

The `wait_load()` implementation should free CPU storage after loading back to device. If it doesn't, memory grows linearly with model depth.

## Cross-Platform Compatibility

### Rules

1. **Platform-agnostic code must not import torch or mindspore directly** — use `get_platform()` abstraction
2. **Changes in `platform/torch/` should have matching `platform/mindspore/` changes** — or explicit justification why not
3. **New platform APIs must be defined in the abstract base class first** (`platform/platform.py`)
4. **Collective operations must go through `platform.*`** — never call raw framework collective APIs

### Common Pitfalls

| Pitfall | Example | Fix |
|---------|---------|-----|
| torch-specific API in core | `torch.cuda.synchronize()` in `core/` | Use `platform.synchronize()` |
| Device string format | Hardcoding `"cuda:0"` | Use `platform.get_device()` |
| Grad API difference | `tensor.grad` vs `.gradient()` | Use platform wrapper |
| Process group creation | Raw `dist.new_group()` | Use `platform.create_group()` |

## Review Checklist Summary

When reviewing a PR, ask these questions:

1. **Stream sync**: Does any tensor cross a stream boundary? Is there an event/wait?
2. **Memory lifecycle**: Is every intermediate buffer freed after consumption?
3. **Gradient cleanup**: Are grad references nulled after use?
4. **Platform parity**: Does the other backend need a matching change?
5. **DTensor invariants**: Is `is_partial()` called correctly? Is partial state reduced before redistribution?
6. If still unsure about correctness, **flag it** — silent bugs are worse than false positives.
