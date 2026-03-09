# HyperParallel — CLAUDE.md

## Project Overview

HyperParallel is a **high-performance distributed parallel acceleration library** that simplifies distributed model training, inference and reinforcement learning. It provides unified abstractions for:
- **Data Parallelism (DP)** — replicate model across devices, aggregate gradients
- **Fully Sharded Data Parallelism (FSDP)** — shard parameters, gradients, and optimizer states across data-parallel ranks
- **Tensor Parallelism (TP)** — shard model weights/activations across devices
- **Expert Parallelism (EP)** — distribute MoE expert layers across devices
- **Context Parallelism (CP)** — shard sequence/context dimension across devices
- **Pipeline Parallelism (PP)** — split model stages across devices
- **Activation Checkpoint (Recomputation)** — trade compute for memory by recomputing activations
- **Parameter Offload** — offload parameters to CPU/NVMe to reduce device memory usage
- **Optimizer Offload** — offload optimizer states to CPU/NVMe to reduce device memory usage
- **Activation Swap** — swap activations to CPU memory during forward and prefetch during backward
- **Hybrid strategies** — combine the above freely

Primary target hardware: **Ascend NPU and Nvidia GPU**. Primary framework: **PyTorch and MindSpore**.

---

## Installation

```bash
pip install .[torch]       # from source (PyTorch backend)
pip install .              # from source (MindSpore backend, builds custom pass plugin)
```

Requires:
- Python >= 3.7
- numpy >= 1.20.0, < 2.0.0
- PyTorch >= 2.6 **or** MindSpore >= 2.8

---

## Repository Structure

```
hyper_parallel/
├── __init__.py                  # Public API surface
├── platform/                    # Platform abstraction layer
│   ├── platform.py              # Abstract Platform base class
│   ├── torch/                   # PyTorch implementation
│   └── mindspore/               # MindSpore implementation
├── core/
│   ├── device_mesh.py           # DeviceMesh — multi-dim device topology
│   ├── layout.py                # Layout — tensor-to-mesh mapping
│   ├── dtensor.py               # DTensor — distributed tensor abstraction
│   ├── placement_types.py       # Shard / Replicate / Partial placement types
│   ├── tensor_redistribution.py # Redistribution between layouts
│   ├── redistribute_infer.py    # Operator inference for redistribution
│   ├── shard/                   # Sharding API + op dispatch
│   │   ├── api.py               # shard_module(), shard()
│   │   ├── local_func.py        # custom_shard()
│   │   ├── _op_dispatch.py      # Distributed operator dispatch
│   │   └── ops/                 # Distributed op implementations + YAML registry
│   ├── hsdp/                    # Hybrid Shard Data Parallel
│   ├── pipeline_parallel/       # Pipeline parallelism
│   ├── fully_shard/             # Fully sharded data parallel (FSDP-style)
│   ├── activation_checkpoint/   # SAC, activation swap
│   ├── checkpoint/              # Distributed checkpoint save/load
│   └── utils.py                 # Shared utilities
├── collectives/
│   └── cc.py                    # Process group management
└── auto_parallel/
    └── fast-tuner/              # Heuristic parallel strategy search (demo)

examples/                        # Usage examples
tests/
├── common/                      # Shared test utilities, markers
├── torch/                       # PyTorch-specific tests
└── mindspore/
    ├── st/                      # System tests (distributed, msrun-based)
    └── ut/                      # Unit tests
```

---

## Core Concepts

### Platform Abstraction
Every feature must be implemented behind an abstraction layer to support both PyTorch and MindSpore; platform-specific logic goes in `platform/torch/` and `platform/mindspore/` respectively.
```python
from hyper_parallel.platform import get_platform
platform = get_platform()   # auto-detects PyTorch / MindSpore
```
- `HYPER_PARALLEL_PLATFORM` env var can force `"torch"` or `"mindspore"`
- All collective ops (`all_reduce`, `all_gather`, `reduce_scatter`, etc.) go through `platform.*`
- `DTensorBase` and `Tensor` are platform-specific tensor types

### DTensor (`core/dtensor.py`)
Distributed tensor holding local shard + metadata (DeviceMesh + Placements).
```python
from hyper_parallel import DTensor
dt = DTensor.from_local(local_tensor, device_mesh, [Shard(0), Replicate()])
dt.shape        # global shape
dt.local_shape  # local shard shape
dt.to_local()   # extract local tensor
dt.full_tensor()           # all-gather to get full tensor (expensive)
dt.redistribute(mesh, placements)  # change distribution
dt.reduce_partial()        # all-reduce/reduce-scatter pending partial state
```

#### DeviceMesh
Multi-dimensional topology of devices. Named axes (e.g., `"dp"`, `"tp"`, `"pp"`).
```python
from hyper_parallel import init_device_mesh
mesh = init_device_mesh("npu", (2, 4), mesh_dim_names=("dp", "tp"))
mesh["dp"]  # sub-mesh along dp axis
```

#### Placement Types (`core/placement_types.py`)
```python
from hyper_parallel.core.placement_types import Shard, Replicate, Partial
Shard(dim)          # split tensor along dimension
Replicate()         # fully replicated
Partial(reduce_op)  # partial result, needs reduction (sum/max/min/avg/prod/all)
```

Factory functions: `dtensor.ones()`, `dtensor.zeros()`, `dtensor.empty()`, `dtensor.full()`

### Tensor Redistribution (`core/tensor_redistribution.py`)
- Singleton `_tensor_redistribution` handles layout-to-layout transformations
- Caches computed transformation op lists in `_transform_cache` (keyed by `compact_str + rank_id`)
- Operators: `Reshape`, `AllConcat`/`all_concat`, `StridedSlice`, `all_split`, `all_to_all`

---

## Coding Conventions

- **License header**: All `.py` files start with the Apache 2.0 header (lines 1–16)
- **Style**: PEP 8, ~120-char line limit; C++ uses Google style (`.clang-format`, limit=120)
- **Naming**: Classes `PascalCase`, functions/vars `snake_case`, private `_leading_underscore`
- **Docstrings**: Google-style with `Args:`, `Returns:`, `Raises:`, `Example:`, `Note:` sections
- **Type hints**: Used on all public function signatures
- **Errors**: Raise `ValueError` with descriptive messages; validate at boundaries
- **Imports**: Lazy imports inside methods use `# pylint: disable=C0415` comment

### Docstring Example
```python
def from_local(
    local_tensor: Tensor,
    device_mesh: DeviceMesh,
    placements: Sequence[Placement]
) -> 'DTensor':
    """
    Create a DTensor from a local tensor with device mesh and placements.

    Args:
        local_tensor (Tensor): The local tensor shard on this device.
        device_mesh (DeviceMesh): The device mesh.
        placements (Sequence[Placement]): Placement strategy per mesh dim.

    Returns:
        DTensor: A new DTensor instance.

    Example:
        >>> dtensor = DTensor.from_local(local_tensor, mesh, [Shard(0), Replicate()])
    """
```

---

## Testing

**Framework**: pytest with custom markers.

**Markers** (`tests/common/mark_utils.py`):
```python
@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",        # level0 / level1
    card_mark="allcards",       # allcards / 1card / 2cards
    essential_mark="essential"
)
```

**Unit tests** (`tests/torch/ut/`, `tests/mindspore/ut/`) run without distributed setup.

**Distributed tests** use `torchrun` (PyTorch) or `msrun` (MindSpore):
```python
# PyTorch
from tests.torch.utils import torchrun_case
torchrun_case(nproc_per_node=8, file_name="test_base_shard.py", case_name="test_base_shard")
# Runs: torchrun --nproc_per_node=8 tests/torch/test_base_shard.py

# MindSpore
from tests.mindspore.st.utils import msrun_case
msrun_case(glog_v=3, file_name="base_shard.py", case_name="test_base_shard", master_port=11333)
# Runs: msrun --worker_num=8 --local_worker_num=8 ... --log_dir=./log_base_shard/test_base_shard
```

---

## Key Implementation Notes

1. **`_build_layout(device_mesh, placements, tensor_dim)`** — creates a `Layout` and calls `placement_to_tensor_map`; called on every `DTensor` construction and `redistribute()`

2. **`SkipDTensorDispatch` context manager** — disables DTensor op dispatch; used when operating on raw local tensors inside gradient hooks

3. **`no_init_parameters()` context manager** — skips weight initialization; required when creating model before sharding to avoid allocating full-size tensors

4. **`reduce_partial`** — must be called before redistribution if layout is in partial state; see `tensor_redistribution.py` for ordering rules (ReduceScatter before AllReduce)

5. **Op dispatch** — distributed ops are registered via YAML files in `core/shard/ops/yaml/`; Python implementations in `core/shard/ops/parallel_*.py`

6. **`is_partial()`** — this is a **method**, not a property (defined in `layout.py:473`); always call with parentheses: `layout.is_partial()`

7. **Memory leak prevention** — tensor allocation and deallocation must be carefully managed to avoid memory leaks; ensure tensors are properly released when no longer needed, especially in long training loops and gradient accumulation scenarios.

8. **Stream synchronization rules** — Missing stream sync is the leading root cause of memory stomping and stale data bugs. When modifying code involving async operations, verify each of the following:
   - **Async collectives**: `handle` returned by `async_op=True` must be waited via `handle.wait()` before accessing the output tensor. `handle.wait()` establishes a **GPU-side dependency** on the current stream via `cudaStreamWaitEvent`, not just a CPU block
   - **`non_blocking` transfers**: `tensor.to(device, non_blocking=True)` executes asynchronously on the current stream; the destination tensor must not be read until the stream has completed (via synchronize or event wait)
   - **Cross-stream dependencies**: Must use events to establish GPU-side ordering: `event.record(stream_A)` → `event.wait(stream_B)`. CPU-side code order does **not** guarantee GPU execution order across streams
   - **`grad_sync_stream`**: Only used in the legacy HSDP path (`HSDPSchedulerV2 + comm_async=True`, per-parameter grad hooks). The new `TorchHSDPSchedulerV2` path (module-level backward hook → `post_backward()` → `reduce_params()`) does not use `grad_sync_stream`
   - **Activation Swap**: `SwapGroup.launch_offload/launch_load` execute on `copy_stream`; tensors must not be accessed on the compute stream until `wait_offload/wait_load` completes the event wait

---

## Skills (`.agentic/skills/`)

| Skill | Description | Usage |
|-------|-------------|-------|
| **autogit** | GitCode fork workflow automation (commit, PR, status, squash), supports origin (fork) + upstream (main repo) pattern | `/skill autogit` |
| **dist-op-dev** | Distributed operator development main workflow, automatically calls analysis tools to complete the entire process from operator analysis to code push | `/skill dist-op-dev` |
| **ms-op-analysis** | [Internal] Analyze MindSpore operator primitive definitions and distributed implementations, automatically called by `dist-op-dev` | — |
| **pt-op-analysis** | [Internal] Analyze PyTorch operator interfaces and map to MindSpore operators, automatically called by `dist-op-dev` | — |
