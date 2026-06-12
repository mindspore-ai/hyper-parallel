# HyperOffload: Asynchronous Activation Offloading

## 1. Overview

HyperOffload reduces device memory (e.g., HBM) pressure during large-scale model training by **offloading intermediate activations** to pinned CPU memory asynchronously and prefetching them back on demand before backward computation.

Unlike manual swap or recomputation, HyperOffload is:

- **Zero-invasion**: Wrap your training step with `with OffloadSession(config):` — no model code modification needed.
- **Fine-grained**: Tracks individual tensor storage, not whole modules.
- **Auto-scheduled**: A greedy planner decides *which* activations to offload and *when* based on a peak memory budget.
- **Async & overlapped**: D2H copies, device releases, H2D prefetches, and host releases execute on a dedicated copy stream overlapping with computation.

![HyperOffload architecture](https://via.placeholder.com/800x400?text=HyperOffload+Architecture+Diagram)

## 2. Architecture

HyperOffload follows a **trace → plan → replay** two-phase execution model:

```text
┌────────────────────────────────────────────────────────────┐
│                       User Training Script                  │
│  config = OffloadConfig(max_resident_activation_mb=512)    │
│  session = OffloadSession(config)                          │
│  with session:                                             │
│      loss = model(x)           # Warmup: trace + evict     │
│      loss.backward()                                       │
│  with session:                                             │
│      loss = model(x)           # Replay: schedule-driven   │
│      loss.backward()                                       │
└────────────────────────────────────────────────────────────┘
```

### Layers

| Layer | Location | Responsibility |
|-------|----------|----------------|
| **API** | `api/` | `OffloadConfig`, `OffloadSession`, `skip_offload` |
| **IR** | `ir/` | `ActivationTrace`, `ResidencySchedule`, `OpGuide` |
| **Execution** | `execution/` | `WarmupExecutor`, `ReplayExecutor`, `ShadowTensor` |
| **Planning** | `planning/` | `GreedyResidencyPlanner` |
| **Runtime** | `runtime/` | `ResidencyManager`, `PinnedMemoryPool`, `BandwidthEstimator` |

## 3. Quick Start

```python
import torch
from hyper_parallel.auto_parallel.hyper_offload import OffloadConfig, OffloadSession

config = OffloadConfig(max_resident_activation_mb=512)
session = OffloadSession(config)

model = torch.nn.Linear(1024, 1024).cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for step in range(10):
    x = torch.randn(128, 1024, device="cuda")
    with session:
        loss = model(x).sum()
        loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

## 4. Key Features

- **Automatic memory budget enforcement** — Peak device memory stays below `max_resident_activation_mb`.
- **Greedy offline planner** — Selects eviction candidates by access distance and tensor size.
- **Asynchronous copy streams** — D2H / H2D transfers overlap with computation on a dedicated stream.
- **ShadowTensor** — A `torch.Tensor` subclass that lazily resolves from device or host storage.
- **Opaque region support** — `@skip_offload` decorator for dynamic control flow or third-party ops.

## 5. API Reference

### `OffloadConfig`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_resident_activation_mb` | `int` | `1024` | Device-side activation memory budget (MiB). |
| `max_offload_activation_mb` | `int` | `65536` | Pinned host memory pool size (MiB). |
| `planner` | `ResidencyPlanner \| None` | `None` | Pluggable planner; defaults to `GreedyResidencyPlanner`. |

### `OffloadSession`

Context manager that drives the warmup → replay lifecycle:

```python
session = OffloadSession(config)

# First entry: warmup mode (trace + online eviction)
with session:
    loss = model(x); loss.backward()

# Subsequent entries: replay mode (schedule-driven offload)
with session:
    loss = model(x); loss.backward()
```

### `skip_offload`

Decorator for opaque regions (dynamic control flow, custom kernels, third-party code):

```python
@skip_offload
def my_custom_block(x):
    return some_dynamic_function(x)
```

## 6. Structure

```text
hyper_offload/
├── README.md              # This file
├── __init__.py            # Public API exports
├── api/
│   ├── __init__.py
│   ├── config.py          # OffloadConfig
│   ├── opaque.py          # skip_offload
│   └── session.py         # OffloadSession
├── execution/
│   ├── __init__.py
│   ├── base.py            # BaseExecutor
│   ├── tensor.py          # ShadowTensor
│   ├── replay/
│   │   ├── __init__.py
│   │   └── executor.py    # ReplayExecutor
│   └── warmup/
│       ├── __init__.py
│       ├── executor.py    # WarmupExecutor
│       └── tracker.py     # ActivationTracker
├── ir/
│   ├── __init__.py
│   ├── replay.py          # OpGuide
│   ├── schedule.py        # ResidencySchedule, ResidencyActionType
│   └── trace.py           # ActivationTrace, TraceOp, AccessKind
├── planning/
│   ├── __init__.py
│   ├── base.py            # ResidencyPlanner (abstract)
│   └── greedy.py          # GreedyResidencyPlanner
└── runtime/
    ├── __init__.py
    ├── timer.py           # DeviceTimer
    ├── bandwidth.py       # BandwidthEstimator
    ├── pinned_memory.py   # PinnedMemoryPool
    └── residency.py       # ResidencyManager, PhysicalBuffer
```

## 7. Limitations & Future Work

- **PyTorch-only (v1)**: MindSpore backend requires a separate dispatch adapter.
- **Static graph assumption**: Warmup trace must match replay execution. Use `@skip_offload` for dynamic branches.
- **Single-device focus**: Cross-rank coordination with FSDP/TP/PP is future work.
- **Planner extensibility**: Additional planners (ILP, DP, ML-based) can be plugged in via `ResidencyPlanner`.
- **Parameter/optimizer offload**: The runtime layer is reusable for parameter and optimizer state offloading.

## 8. References

- PyTorch `TorchDispatchMode`: https://pytorch.org/docs/stable/notes/extending.html
- HyperParallel activation checkpoint swap: `hyper_parallel.platform.torch.activation_checkpoint`
