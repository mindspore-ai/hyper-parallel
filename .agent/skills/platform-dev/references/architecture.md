# Platform Architecture Reference

## Platform Abstraction Design

### Auto-Detection Mechanism

```python
# platform/platform.py
def get_platform() -> Platform:
    # 1. Check env var HYPER_PARALLEL_PLATFORM
    # 2. Try MindSpore import (preferred if available)
    # 3. Fallback to PyTorch
    # 4. Raise error if neither available
```

**Environment Variable Control:**
```bash
export HYPER_PARALLEL_PLATFORM=torch      # Force PyTorch
export HYPER_PARALLEL_PLATFORM=mindspore  # Force MindSpore
```

### Platform Type Enum

```python
class PlatformType(Enum):
    MINDSPORE = auto()
    PYTORCH = auto()
```

### Class Hierarchy

```text
Platform (abstract base)
├── TorchPlatform
│   ├── Tensor = torch.Tensor
│   ├── Parameter = nn.Parameter
│   ├── Module = nn.Module
│   ├── DTensorBase = TorchDTensorBase
│   └── PipelineStageBase = TorchPipelineStageBase
└── MindSporePlatform
    ├── Tensor = ms.Tensor
    ├── Parameter = ms.Parameter
    ├── Module = nn.Cell
    ├── DTensorBase = MsDTensorBase
    └── PipelineStageBase = MsPipelineStageBase
```

---

## DTensorBase Dispatch Mechanism

### Torch DTensorBase

```text
User calls: dtensor + dtensor
    ↓
__torch_function__ intercepts
    ↓
Check _OP_DISPATCHER registry (core/shard/_op_dispatch.py)
    ↓
If registered: dispatch to custom op handler
    ↓
Handler: extract local tensors → compute → wrap result with layout
```

**Key Implementation Details:**
- Uses `torch.Tensor._make_subclass()` to create DTensor from local tensor
- Overrides `__torch_function__` for op interception
- Properties delegate to `_local_tensor`: `grad`, `requires_grad`, `dtype`, `device`
- In-place ops (`zero_()`, `copy_()`, `fill_()`) handle leaf tensor rebinding
- `SkipDTensorDispatch` context manager bypasses dispatch for raw tensor ops

### MindSpore DTensorBase

```text
User calls: dtensor.some_op()
    ↓
__fallback__() intercepts
    ↓
NoFallbackGuard prevents recursion
    ↓
Route to _OP_DISPATCHER
    ↓
Handler: extract local tensors → compute → wrap result
```

**Key Implementation Details:**
- Uses `ms.Tensor._make_subclass()` to create DTensor
- Overrides `__fallback__()` for op interception
- Supports uninitialized tensors (`has_init` property)
- `set_data()` handles in-place data replacement with layout preservation

---

## FSDP State Lifecycle

### Parameter Sharding Flow

```text
Model Init
    ↓
fully_shard(module) — registers hooks, creates FSDPState
    ↓
Forward Pre-Hook: unshard parameters (all-gather)
    ↓
Forward Compute (full parameters available)
    ↓
Forward Post-Hook: reshard parameters (free memory via resize_(0))
    ↓
Backward Pre-Hook: unshard parameters (all-gather again)
    ↓
Backward Compute (gradients computed on full params)
    ↓
Backward Post-Hook: reduce-scatter gradients + reshard params
    ↓
Optimizer Step (on sharded gradients)
```

### Key State Components

| Component | File | Purpose |
|-----------|------|---------|
| `FSDPState` | `fully_shard/state.py` | Manages shard/unshard lifecycle |
| `FSDPParam` | `fully_shard/param.py` | Per-parameter metadata + buffers |
| `FSDPScheduler` | `fully_shard/scheduler.py` | Shard/unshard scheduling + prefetch |
| `HookFunction` | `fully_shard/hook_function.py` | Forward/backward hook registration |

### Memory Pattern

```python
# Unshard: all-gather into full parameter buffer
full_param = all_gather(sharded_param, group)

# After forward: free full parameter
full_param.resize_(0)  # Release device memory

# Backward: all-gather again (recompute from shard)
full_param = all_gather(sharded_param, group)

# After backward: reduce-scatter gradient, free full param
grad_shard = reduce_scatter(full_grad, group)
full_param.resize_(0)
```

---

## HSDP State Lifecycle

### Gradient Reduction Flow

```text
HSDP = FSDP (intra-node) + DDP (inter-node)

Backward Compute
    ↓
Gradient Reduce-Scatter (intra-node, shard dimension)
    ↓
Gradient All-Reduce (inter-node, replicate dimension)
    ↓
Optimizer Step
```

### Key Differences from FSDP

| Aspect | FSDP | HSDP |
|--------|------|------|
| Groups | Single shard group | Shard group + Replicate group |
| Gradient | reduce-scatter only | reduce-scatter + all-reduce |
| Handle tracking | `pre_reduce_scatter_params` | `pre_reduce_scatter_params` + `pre_all_reduce_params` |
| Stream | Default stream | `grad_sync_stream` (legacy path) |

### HSDP-Specific State

```python
class HSDPState:
    pre_reduce_scatter_params   # Params pending reduce-scatter
    pre_all_reduce_params       # Params pending all-reduce
    reduce_op_type              # AVG or SUM

class HSDPParam:
    prefetch_handle             # Async all-gather handle
    reduce_scatter_handle       # Async reduce-scatter handle
    all_reduce_handle           # Async all-reduce handle
```

---

## Stream Synchronization Patterns

### Pattern 1: Async Collective with Handle

```python
output, handle = platform.all_gather_into_tensor(
    input, group=group, async_op=True
)
# ... overlap computation ...
handle.wait()  # MUST wait before reading output
result = process(output)
```

### Pattern 2: Cross-Stream Event Sync

```python
event = platform.new_event()

# Stream A: produce data
with platform.get_stream_context(stream_a):
    data = compute()
    event.record(stream_a)

# Stream B: consume data (wait for Stream A)
with platform.get_stream_context(stream_b):
    event.wait(stream_b)
    use(data)
```

### Pattern 3: Gradient Handle (HSDP Legacy)

```python
# In backward hook
platform.set_grad_reduce_handle(handle, post_process_fn)

# Before next backward / optimizer step
platform.wait_grad_handle()
# This calls: handle.wait() + post_process_fn()
```

### Pattern 4: non_blocking Transfer

```python
# D2H transfer
cpu_tensor = gpu_tensor.to("cpu", non_blocking=True)
# MUST sync stream before reading cpu_tensor
stream.synchronize()
use(cpu_tensor)
```

---

## Process Group Caching

### Global Cache

```python
# platform/platform.py
EXISTING_COMM_GROUPS = {}  # Shared across both backends

# Key format: str(tuple(sorted(rank_list)))
# Example: "(0, 1, 2, 3)"

# Value: ProcessGroup (torch) or str (mindspore)
```

### Creation Flow

```python
def create_group(rank_list):
    key = str(tuple(sorted(rank_list)))
    if key in EXISTING_COMM_GROUPS:
        return EXISTING_COMM_GROUPS[key]
    group = _create_group(rank_list)
    EXISTING_COMM_GROUPS[key] = group
    return group
```

### Torch Group Utils

```python
# platform/torch/group_utils.py
def generate_groups_from_template(template, world_size, my_rank):
    """Auto-expand group template to full group list.

    Example: template=[0,1], world_size=8
    → [[0,1], [2,3], [4,5], [6,7]]
    """

def create_sub_groups(rank_list):
    """Create ProcessGroup with validation and caching."""
```

---

## Activation Checkpoint (Torch Only)

### Selective Activation Checkpoint (SAC)

```text
Forward Pass:
    For each op:
        if policy(op) == SAVE: save activation
        if policy(op) == RECOMPUTE: skip saving

Backward Pass:
    For each op:
        if saved: use saved activation
        if recompute: re-execute forward op
```

### Activation Swap (CPU Offload)

```text
Forward Pass:
    Activation saved → async D2H copy to CPU (non_blocking)
    Record event on compute stream

Backward Pass:
    Prefetch: async H2D copy back to GPU
    Wait for event → use activation
    Free CPU buffer
```

**Key Classes:**
- `AsyncSaveOnCpu` — saved_tensors_hooks context manager
- `ActivationWrapper` — module wrapper base class
- `ActivationPolicy.SAVE` vs `ActivationPolicy.SWAP`

---

## init_on_device Pattern

Both backends implement `init_on_device(device)` context manager:

### Torch Implementation

```python
# Monkey-patches nn.Module.register_parameter and register_buffer
# to place new parameters/buffers on target device at creation time
with platform.init_on_device(device):
    model = MyModel()  # All params on `device`
```

### MindSpore Implementation

```python
# Similar monkey-patching for Cell parameter registration
with platform.init_on_device(device):
    model = MyModel()  # All params on `device`
```

Additionally, MindSpore has `parameter_init.py`:
```python
def init_parameters(cell, stage_index):
    """Initialize parameters with slice_index for distributed setup."""
    # Uses _get_slice_index() from mindspore.parallel._tensor
```
