# Workflow 3: Backend Implementation

## Goal

Implement concrete methods in `platform/torch/` and `platform/mindspore/` for the API defined in Step 2.

## Steps

### 3.1 Implement PyTorch Backend

**File:** `hyper_parallel/platform/torch/platform.py` (or submodule files)

**Conventions:**
- **Lazy import in methods:** import `torch`, `torch.distributed`, etc. **inside** each method that needs them, with `# pylint: disable=C0415` (see `.claude/rules/code-style.md`).
- Use `torch.distributed` for collective operations
- Use `device_handle` for stream/event management (supports both CUDA and NPU)
- Return `(output, handle)` tuple for async collective ops
- Process groups are `dist.ProcessGroup` objects

**Template:**
```python
class TorchPlatform(Platform):
    def new_method(self, param1, param2=default):
        import torch  # pylint: disable=C0415
        import torch.distributed as dist  # pylint: disable=C0415
        # Implementation using torch APIs
        ...
```

### 3.2 Implement MindSpore Backend

**File:** `hyper_parallel/platform/mindspore/platform.py` (or submodule files)

**Conventions:**
- **Lazy import in methods:** import `mindspore` / `mindspore.mint.distributed`, etc. **inside** each method that needs them, with `# pylint: disable=C0415` (see `.claude/rules/code-style.md`).
- Use `mindspore.mint.distributed` for collective operations
- Process groups are `str` names (not objects)
- Device is `str` (not `torch.device`)
- `Cell` instead of `Module`

**Template:**
```python
class MindSporePlatform(Platform):
    def new_method(self, param1, param2=default):
        import mindspore as ms  # pylint: disable=C0415
        from mindspore.mint import distributed as dist  # pylint: disable=C0415
        # Implementation using MindSpore APIs
        ...
```

**If not yet implementable:**
```python
def new_method(self, param1, param2=default):
    raise NotImplementedError("new_method is not yet supported on MindSpore backend")
```

### 3.3 Submodule Implementation

For feature-specific code (FSDP, HSDP, Pipeline, Activation Checkpoint):

| Submodule | Key Files | Pattern |
|-----------|-----------|---------|
| **FSDP** | `state.py`, `param.py`, `scheduler.py`, `hook_function.py` | State manages lifecycle; Param holds per-param metadata; Scheduler handles shard/unshard timing; Hooks register forward/backward hooks |
| **HSDP** | Same as FSDP + `grad_hook.py`, `async_grad_hook.py` | Extends FSDP with gradient all-reduce across replicate dimension |
| **Pipeline** | `stage.py`, `_utils.py` | Stage defines pipeline stage; Utils handle micro-batch chunking |
| **Activation** | `sac.py`, `activation_swap.py` | SAC for selective recompute; Swap for CPU offload |

### 3.4 DTensorBase Extension

For changes to DTensorBase:

**Torch** (`platform/torch/dtensor.py`):
- Subclass of `torch.Tensor` via `_make_subclass()`
- Override `__torch_function__` for op dispatch
- Properties delegate to `_local_tensor`

**MindSpore** (`platform/mindspore/dtensor.py`):
- Subclass of `ms.Tensor` via `_make_subclass()`
- Override `__fallback__()` for op dispatch
- Support for uninitialized tensors (`has_init`)

### 3.5 Stream Safety Checklist

When implementing async operations:

- [ ] Async collective returns `(output, handle)` tuple
- [ ] `handle.wait()` called before reading output
- [ ] Cross-stream deps use event sync: `event.record(src)` → `event.wait(dst)`
- [ ] `non_blocking=True` transfers have proper stream sync
- [ ] No GPU-CPU sync in hot paths (no `.item()`, `.numpy()`)

### 3.6 Memory Safety Checklist

When implementing memory management:

- [ ] `resize_(0)` used to free device memory after consumption
- [ ] No access to storage after `resize_(0)`
- [ ] `param.grad = None` after gradient consumption
- [ ] Buffer reuse preferred over reallocation
- [ ] `_clear_recv_buffer()` + `clear_cache()` per micro-batch (pipeline)

## Output

- Updated `platform/torch/platform.py` (or submodule files) with concrete implementation
- Updated `platform/mindspore/platform.py` (or submodule files) with concrete implementation or NotImplementedError

## Next Step

Proceed to **[Workflow 4: Cross-Platform Verification](./04-cross-platform-verification.md)**
