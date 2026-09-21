# Distributed System Guidelines

This document covers distributed system correctness for HyperParallel PR reviews
**with Bad/Good examples**.

**Hard-rule source of truth:** `.agent/rules/distributed.md` (shortlist also in
`.agent/rules/project-overview.md`). Do not treat this file as a second copy of
those bullets — when rules and examples disagree, **rules win**; fix this file.

As a top level principle, **missing stream synchronization is the leading root cause of memory stomping and stale data bugs**. Any change involving async operations, cross-stream access, or device transfers must be reviewed with extreme care.

As a reviewer, you MUST be paranoid about stream sync and memory lifecycle. These bugs are silent — they produce incorrect results without exceptions or crashes.

## Stream Synchronization

Canonical requirements: `.agent/rules/distributed.md` § Stream Synchronization.
Below: why it bites in review, plus Bad/Good patterns.

### The Core Problem

GPU operations execute asynchronously on streams. CPU code ordering does NOT guarantee GPU execution ordering across different streams. The only way to establish GPU-side ordering across streams is via events.

### Rules (summary — full text in `distributed.md`)

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

Canonical requirements: `.agent/rules/distributed.md` § Memory Management.
Below: why it bites in review, plus Bad/Good patterns.

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

## Collective Calling Conventions

### Rules

1. **Autograd paths go through the `differentiable_*` helpers** — `differentiable_all_reduce`,
   `differentiable_reduce_scatter`, `differentiable_all_to_all_single(_async)`,
   `differentiable_variable_all_gather` in `core/dtensor/_utils.py` and
   `core/context_parallel/utils.py`. Raw `dist.*` calls there drop the graph.
2. **Group arguments are process groups**, not rank lists — resolve once with
   `create_group(rank_list)` or `layout.get_comm_group_by_axis(dev_dim)`, then pass `group`
   down; no positional group juggling at the call site.
3. **Every collective needs its `wait()` on the async path** — an `async_op=True` work object
   that is never waited on both leaks and reorders.
4. **Bounds and shapes are settled before the call** — `torch.chunk` / `input_splits` must line
   up with `output_splits` on every rank, or the collective hangs rather than raising.

### Common Pitfalls

| Pitfall | Example | Fix |
|---------|---------|-----|
| Raw collective on an autograd path | `dist.all_reduce(grad)` in a backward-reachable fn | `differentiable_all_reduce(grad, op, group)` |
| Dropped async handle | `dist.all_gather(..., async_op=True)` never waited | `wait_async_tensor(work)` / `work.wait()` |
| Rank list where a group is expected | `all_reduce(x, [0, 1])` | `create_group([0, 1])` then pass the handle |
| Splits disagree across ranks | `input_splits` computed from local shape only | derive from the global shape, identical on all ranks |
| Sync in a hot path | `torch.npu.synchronize()` inside the step | sync only at the boundary that needs it |

## All-Reduce / Reduce-Scatter Example

**Canonical bullets:** `.agent/rules/distributed.md` § Collective Calling Conventions.
Checklist: `review-checklist.md`. Below: one Bad/Good set for review speed.

**Bad:** a raw `dist.reduce_scatter_tensor` on a tensor that still needs gradients; or passing a
rank list where the helper expects an already-resolved process group.

**Good:**
```python
from hyper_parallel.core.dtensor._utils import create_group, differentiable_reduce_scatter


def _reduce_scatter_along_dev_dim_with_axis(self, x, axis, op, layout, dev_dim):
    group = layout.get_comm_group_by_axis(dev_dim)
    return differentiable_reduce_scatter(x, dev_num, axis, op, group)


group = create_group(group_ranks)   # resolve the ranks to a group exactly once
```

## Review Checklist Summary

When reviewing a PR, ask these questions:

1. **Stream sync**: Does any tensor cross a stream boundary? Is there an event/wait?
2. **Memory lifecycle**: Is every intermediate buffer freed after consumption?
3. **Gradient cleanup**: Are grad references nulled after use?
4. **Collective semantics**: Is the autograd path on a `differentiable_*` helper? Is every async
   work object waited on? Do the splits agree on every rank?
5. **DTensor invariants**: Is `is_partial()` called correctly? Is partial state reduced before redistribution?
6. **Group plumbing**: Is the process group resolved once and threaded through, rather than
   re-created or passed as a rank list?
7. If still unsure about correctness, **flag it** — silent bugs are worse than false positives.
