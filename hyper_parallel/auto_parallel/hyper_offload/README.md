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
| `planner` | `ResidencyPlanner \| None` | `None` | Pluggable planner; defaults to `PeakResidencyPlanner`. |

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

### Native autograd replay

HyperOffload automatically selects native autograd replay for an eligible static,
complete forward/backward step; no additional switch is required:

```python
config = OffloadConfig(max_resident_activation_mb=512)
session = OffloadSession(config)
```

Warmup still records the operator trace and uses the configured planner. If every
selected storage is an immutable forward activation saved for backward, with one
offload interval and a static save/use sequence, replay uses autograd hooks
without installing a global `TorchDispatchMode`. There is no activation-count
limit. Unsupported plans keep operator replay.

The default Peak planner prefers eligible forward-to-backward intervals covering
the current peak. It keeps the same memory budget and can choose other intervals
when eligible activations do not provide enough relief. Communication masking
uses spare residency for early H2D prefetch before delaying D2H source releases.

Only selected saves replace their original tensor with storage/view metadata.
Other saves retain detached tensors with version checks. Copies use the existing
residency manager and copy stream. D2H starts at the first selected save; source
release and individual H2D prefetch points are compiled onto the next available
save/use hook. H2D destinations are allocated on the copy stream and depend only
on the host snapshot; unrelated compute is not a producer dependency. H2D is
fenced before its consumer, and restored manager ownership
ends at the last saved-view use. These hook anchors can differ from the exact
operator anchors in the original plan.

Keep the same forward save order, view metadata and backward use order on every
entry, and complete backward before leaving the session. Activation checkpointing
can save additional aliases during backward; replay validates their position among
the recorded uses and checks them against restored storage. Repeated reads of a
saved view keep that storage until its final recorded use. Both non-reentrant and
reentrant checkpointing can use this path when every selected storage is first
saved in forward, is immutable after its first save, and has one
forward-to-backward offload interval. Initialization before the first save and
read-only split/unbind aliases are supported. A storage must have at least one
recorded backward use; extra unused saved aliases do not extend its lifetime.
Changing save/use order or control flow is unsupported. Record a new session when
the graph changes. The session checks selected metadata, storage aliases, mutation
versions and save/use counts; it does not validate every unrelated operator during
native replay.

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

- **Static graph assumption**: Warmup trace must match replay execution. Use `@skip_offload` for dynamic branches.
- **Single-device focus**: Cross-rank coordination with FSDP/TP/PP is future work.
- **Planner extensibility**: Additional planners (ILP, DP, ML-based) can be plugged in via `ResidencyPlanner`.
- **Parameter/optimizer offload**: The runtime layer is reusable for parameter and optimizer state offloading.

## 8. References

- PyTorch `TorchDispatchMode`: https://pytorch.org/docs/stable/notes/extending.html
- HyperParallel activation checkpoint swap: `hyper_parallel.core.activation_memory`
