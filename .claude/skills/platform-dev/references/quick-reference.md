# Platform Development Quick Reference

## File Location Guide

| Task | File Location | Notes |
|------|---------------|-------|
| Platform base class | `hyper_parallel/platform/platform.py` | All abstract methods defined here |
| PyTorch platform | `hyper_parallel/platform/torch/platform.py` | `TorchPlatform(Platform)` |
| MindSpore platform | `hyper_parallel/platform/mindspore/platform.py` | `MindSporePlatform(Platform)` |
| Torch DTensorBase | `hyper_parallel/platform/torch/dtensor.py` | Subclass of `torch.Tensor` |
| MindSpore DTensorBase | `hyper_parallel/platform/mindspore/dtensor.py` | Subclass of `ms.Tensor` |
| Torch FSDP / HSDP | `hyper_parallel/platform/torch/fully_shard/` | state, param, scheduler, hooks (HSDP shares this tree; core logic in `core/fully_shard/hsdp_*.py`) |
| MindSpore FSDP / HSDP | `hyper_parallel/platform/mindspore/fully_shard/` | state, param, scheduler, hooks (same layout as Torch) |
| Torch Pipeline | `hyper_parallel/platform/torch/pipeline_parallel/` | stage, micro-batch utils |
| MindSpore Pipeline | `hyper_parallel/platform/mindspore/pipeline_parallel/` | stage, micro-batch utils |
| Torch Activation Ckpt | `hyper_parallel/platform/torch/activation_checkpoint/` | SAC + activation swap |
| Process group utils | `hyper_parallel/platform/torch/group_utils.py` | Group creation/caching |
| Gradient clipping | `hyper_parallel/platform/torch/clip_grad.py` | Distributed grad clip |
| Backward hooks | `hyper_parallel/platform/torch/function_override.py` | DTensor backward preservation |
| Init weights (torch) | `hyper_parallel/platform/torch/init_weights.py` | init_on_device context |
| Init weights (ms) | `hyper_parallel/platform/mindspore/init_weights.py` | init_on_device context |
| Parameter init (ms) | `hyper_parallel/platform/mindspore/parameter_init.py` | Slice-based param init |
| Graph utils (ms) | `hyper_parallel/platform/mindspore/platform_graph.py` | Graph construction |
| Custom passes (ms) | `hyper_parallel/platform/mindspore/custom_pass/` | Graph optimization passes |
| Tests (torch) | `tests/torch/` | `ut/` (unit), `st/` (distributed) |
| Tests (mindspore) | `tests/mindspore/` | `ut/` (unit), `st/` (distributed) |

---

## Platform API Categories

### Tensor & Module Types

| API | Torch | MindSpore |
|-----|-------|-----------|
| `Platform.Tensor` | `torch.Tensor` | `ms.Tensor` |
| `Platform.Parameter` | `nn.Parameter` | `ms.Parameter` |
| `Platform.Module` | `nn.Module` | `nn.Cell` |
| `Platform.DTensorBase` | Custom `torch.Tensor` subclass | Custom `ms.Tensor` subclass |
| `Platform.tensor_dtype` | `torch` (dtype module) | `mstype` |
| `Platform.dtype` | `torch.dtype` | `ms.Type` |

### Collective Operations

| API | Torch Backend | MindSpore Backend |
|-----|--------------|-------------------|
| `all_gather_into_tensor()` | `dist.all_gather_into_tensor()` | `comm_func.all_gather_into_tensor()` |
| `all_reduce()` | `dist.all_reduce()` | `dist.all_reduce()` (string group) |
| `reduce_scatter_tensor()` | `dist.reduce_scatter_tensor()` | `comm_func.reduce_scatter_tensor()` |
| `all_to_all_single()` | `dist.all_to_all_single()` | `comm_func.all_to_all_single()` |
| `broadcast()` | `dist.broadcast()` | `dist.broadcast()` |
| `isend()` / `irecv()` | `dist.isend()` / `dist.irecv()` | `dist.isend()` / `dist.irecv()` |

### Differentiable Collective Operations

| API | Description |
|-----|-------------|
| `differentiable_all_gather_concat()` | All-gather with autograd support |
| `differentiable_all_to_all()` | All-to-all with autograd support |
| `differentiable_all_reduce()` | All-reduce with autograd support |
| `differentiable_reduce_scatter()` | Reduce-scatter with autograd support |

### Process Group Management

| API | Torch | MindSpore |
|-----|-------|-----------|
| Group type | `dist.ProcessGroup` object | `str` (group name) |
| Group key | `str(tuple(sorted(ranks)))` | `str(tuple(sorted(ranks)))` |
| `create_group()` | `dist.new_group(ranks)` | `new_group(rank_ids, group=name)` |
| `split_group()` | `dist.new_group(ranks=split)` | String-based split |
| Cache | `EXISTING_COMM_GROUPS` dict | `EXISTING_COMM_GROUPS` dict |

### Stream & Event Management

| API | Torch | MindSpore |
|-----|-------|-----------|
| `new_stream()` | `device_handle.Stream()` | `ms.runtime.Stream()` |
| `get_stream_context()` | `device_handle.stream` | `ms.runtime.StreamCtx` |
| `new_event()` | `device_handle.Event()` | `ms.runtime.Event()` |
| `get_current_stream()` | `device_handle.current_stream()` | `ms.runtime.current_stream()` |

### Device & RNG

| API | Torch | MindSpore |
|-----|-------|-----------|
| `device()` | `torch.device(type, idx)` | `str` ("npu"/"cpu"/"gpu") |
| `device_type()` | "npu" or "cuda" | "npu" or "gpu" |
| `meta_device` | `torch.device("meta")` | `"meta"` |
| `manual_seed()` | `torch.manual_seed()` | `ms.set_seed()` |
| `get_rng_state()` | `torch.get_rng_state()` | `ms.get_rng_state()` |

### Gradient Synchronization

| API | Description |
|-----|-------------|
| `Platform.grad_sync_stream` | Class attr: stream for gradient reduce |
| `Platform.current_grad_handle` | Class attr: current async grad handle |
| `Platform.post_grad_handle_process` | Class attr: callback after handle.wait() |
| `set_grad_reduce_handle()` | Set handle + post-process callback |
| `wait_grad_handle()` | Wait handle + execute post-process |

---

## Cross-Platform Type Mapping

| Concept | Torch | MindSpore |
|---------|-------|-----------|
| Neural network module | `nn.Module` | `nn.Cell` |
| Learnable parameter | `nn.Parameter` | `ms.Parameter` |
| Device descriptor | `torch.device` | `str` |
| Process group | `ProcessGroup` object | `str` name |
| Data types namespace | `torch.float32`, etc. | `mstype.float32`, etc. |
| dtype type | `torch.dtype` | `ms.Type` |
| Op dispatch hook | `__torch_function__` | `__fallback__` |
| Module children | `named_modules()` | `cells_and_names()` |
| Module parameters | `named_parameters()` | `parameters_and_names()` |

---

## MindSpore Unsupported Features (raise NotImplementedError)

These features are not yet implemented in MindSpore backend:

| Feature | Torch Location | Status |
|---------|---------------|--------|
| `ckpt_wrapper()` | torch activation checkpoint | Not supported |
| `create_selective_checkpoint_contexts()` | SAC | Not yet implemented |
| `async_save_on_cpu()` | Activation swap | Not yet implemented |
| `clip_grad_norm_()` | Gradient clipping | Not yet implemented |
| `get_model_state_dict()` | State dict | Not yet implemented |
| `noop_context_fn` | Checkpoint contexts | Not supported |

---

## Common Patterns

### Adding a New Platform API

```python
# 1. platform/platform.py — add abstract method
class Platform:
    def new_api(self, arg1, arg2):
        raise NotImplementedError

# 2. platform/torch/platform.py — implement (lazy import in methods)
class TorchPlatform(Platform):
    def new_api(self, arg1, arg2):
        import torch  # pylint: disable=C0415
        # torch-specific implementation
        ...

# 3. platform/mindspore/platform.py — implement (or NotImplementedError)
class MindSporePlatform(Platform):
    def new_api(self, arg1, arg2):
        import mindspore as ms  # pylint: disable=C0415
        # mindspore-specific implementation
        ...
```

### Async Collective with Handle

```python
# Start async collective
output, handle = platform.all_gather_into_tensor(input, group=group, async_op=True)

# Do other work...

# Wait before reading result
handle.wait()
use(output)
```

### Stream Synchronization

```python
# Cross-stream dependency via events
event = platform.new_event()
with platform.get_stream_context(compute_stream):
    result = compute(input)
    event.record(compute_stream)

with platform.get_stream_context(comm_stream):
    event.wait(comm_stream)
    send(result)
```

### Memory Lifecycle (resize pattern)

```python
# Free device memory after use
tensor.resize_(0)
# NEVER access tensor after resize_(0)

# Null grad after consumption
param.grad = None
```

### Lazy import in platform backends

In `platform/torch/` and `platform/mindspore/`, framework imports belong **inside methods** with `# pylint: disable=C0415`. In **non-platform** code, prefer module-level imports; local imports there are only for documented exceptions (see `.claude/rules/code-style.md`).

```python
def some_method(self):
    from hyper_parallel.platform.torch.some_module import SomeClass  # pylint: disable=C0415
    return SomeClass(...)
```

---

## Anti-Patterns (DO NOT)

| Anti-Pattern | Correct Approach |
|-------------|-----------------|
| `import torch` in platform-agnostic code | `from hyper_parallel.platform import get_platform` |
| Access tensor after `resize_(0)` | Track lifecycle, never read freed storage |
| Read async collective output without `handle.wait()` | Always `handle.wait()` first |
| Cross-stream read without event sync | `event.record(src)` → `event.wait(dst)` |
| `non_blocking=True` without stream sync | Ensure stream sync before reading |
| Add Platform API without base class method | Define in `platform/platform.py` first |
| Modify torch/ without checking mindspore/ | Always verify cross-platform parity |
| Module-top `import torch` / `import mindspore` in `platform/torch/` or `platform/mindspore/` | Prefer lazy import inside methods with `C0415` per `code-style.md` |
| Imports inside methods in **non-platform** code | Move to module top unless a documented exception applies |
| `.item()` or `.numpy()` in hot paths | Causes GPU-CPU sync, avoid in training loops |
