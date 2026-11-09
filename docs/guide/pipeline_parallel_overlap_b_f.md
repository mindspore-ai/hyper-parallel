# Pipeline Parallel Compute-Communication Overlap Design

**English** | [简体中文](pipeline_parallel_overlap_b_f.zh.md)

## 1. Background and Motivation

In hybrid PP (pipeline parallelism) + EP (expert parallelism), a single micro-batch's critical path includes two high-latency communication classes:

- **Inter-stage P2P send/recv**: Transfer activations during FWD and gradients during BWD between adjacent stages.
- **MoE EP all-to-all (A2A)**: Dispatch tokens to ranks hosting the selected experts; combine expert outputs back on their original ranks.

Traditional 1F1B serializes these communications with computation, creating substantial timeline bubbles. The **`overlap_b_f` option** on `ScheduleInterleaved1F1B` pairs adjacent steady-state BWD and FWD operations into `OVERLAP_B_F` composite steps. An independent **`CommComputeOverlap`** coordinator drives them concurrently through two Python threads, allowing one side's EP A2A to run physically concurrently with the other side's noncommunication compute on the GPU.

Design goals:

1. **Correctness**: Exact numerical agreement with the non-overlap path.
2. **Composability**: Preserve `PipelineStage` / `ScheduleInterleaved1F1B` abstractions; extend through a flag and registered callbacks without complicating the scheduling core.
3. **Separate mechanism from scheduling**: `CommComputeOverlap` is an independent two-thread coordinator, potentially reusable for TP+CP or FSDP prefetch without binding it to PP.
4. **Native Torch implementation**: PP supports PyTorch only; `CommComputeOverlap` directly uses PP's internal `torch.autograd.Function` synchronization hooks.

---

## 2. Module Structure

```text
hyper_parallel/core/pipeline_parallel/
├── scheduler.py                # Scheduling core: OVERLAP_B_F, PipelineContext, custom_fn registration, overlap_b_f flag
├── stage.py                    # PipelineStage
├── hook_coordinator.py         # NEW: two-thread COMM-first rendezvous
├── comm_compute_overlap.py     # NEW:CommComputeOverlap orchestrator + A/B/C/D wrap
├── _sync_hook.py               # _SyncHookFunction (torch.autograd.Function)
└── __init__.py                 # New API exports

examples/torch/pp_overlap/
└── pp_overlap_moe_example.py   # End-to-end example: PP + EP + compute-communication overlap + MoE
```

---

## 3. Key Concepts

### 3.1 MetaStep and OVERLAP_B_F

`MetaStep` describes a scheduling unit. `MetaStepType` adds:

- `OVERLAP_F_B` / `OVERLAP_B_F`: Composite steps with `sub_steps = (first, second)`.
- `OVERLAP_B_F` combines steady-state 1F1B operations `B_i, F_{i+1}` into one step.

`ScheduleInterleaved1F1B(overlap_b_f=True).construct_stage_exec_order` transforms the original interleaved 1F1B sequence:

```text
Before: F₁, B₁, F₂, B₂, ..., F_n, B_n
After:  F₁, [B₁,F₂], [B₂,F₃], ..., [B_{n-1},F_n], B_n
```

The first FWD and last BWD remain unpaired for warmup/cooldown; each other steady-state step is `OVERLAP_B_F`.

### 3.2 PipelineContext and custom_fn Registration

`PipelineScheduleRuntime` introduces:

```python
schedule.register_custom_function(MetaStepType.OVERLAP_B_F, callback)
```

On `OVERLAP_B_F`, `run_microbatches` delegates to the callback with a `PipelineContext`:

```python
class PipelineContext:
    schedule, arg_mbs, kwarg_mbs, losses, send_handles
    fwd_recv_ops = schedule.fwd_handle_cache  # for overlap_p2p
    bwd_recv_ops = schedule.bwd_handle_cache
```

This keeps the scheduling core generic, with dual-pipe implemented as an extension.

### 3.3 P2P Overlap and `overlap_p2p`

The `PipelineScheduleRuntime(overlap_p2p: bool = False)` parameter controls:

- `False` (default): `_exec_step` issues FWD_RECV / BWD_RECV and immediately waits on the HCCL stream without blocking the host, preserving existing behavior.
- `True`: RECV caches handles in `fwd_handle_cache / bwd_handle_cache`; consumers, either FWD/BWD branches in `_exec_step` or the OVERLAP_B_F callback, pop and wait as needed.

With `overlap_p2p=True`, SEND appends to `send_handles`, drained together at the end of `run_microbatches`, without per-step blocking. HCCL automatically separates streams by process group, enabling physical overlap with the next step's computation.

### 3.4 add_send_recv and OVERLAP Column Scanning

`add_send_recv` expands each OVERLAP step into two virtual time slots (`_OverlapPhantom`), inserts SEND/RECV by column scanning, and preserves even/odd rank processing order to avoid P2P deadlocks:

- `_expand_overlap_slots`: Split each OVERLAP into `(first_half, second_half)` phantoms.
- `_column_scan_insert_comms`: Align time steps, enqueue the current step, then call `_insert_comms_for_step` to issue SEND/RECV.
- Process even ranks before odd ranks, supporting both `'loop'` and `'v'` stage-to-rank mappings.

---

## 4. EP A2A Compute-Communication Overlap

### 4.1 A/B/C/D Synchronization Hooks

Each MoE layer executes the following during forward:

```text
[A] ─► dispatch(A2A) ─► [B] ─► expert(compute) ─► [C] ─► combine(A2A) ─► [D] ─► (Attention)
```

`CommComputeOverlap` provides `wrap_dispatch` / `wrap_combine`, using `_SyncHookFunction.apply` to insert A, B, C, and D hook points into the tensor graph. Backward automatically triggers them in the symmetric D', C', B', A' order.

### 4.2 HookCoordinator:COMM-first rendezvous

A `threading.Barrier` alone is insufficient: once both threads rendezvous, **the GIL determines their release order**. Host dispatch order, and therefore GPU A2A/compute launch order, is uncontrolled.

`HookCoordinator` adds a **per-rendezvous `threading.Event`** on top of the barrier:

1. Both threads arrive and pass the barrier.
2. `COMM` immediately dispatches its A2A, then `notify_dispatched()` signals the event.
3. `COMPUTE` waits for the event and dispatches its compute kernel only after COMM has dispatched.

Every rendezvous uses an **independent** event, avoiding set/clear races between consecutive rendezvous.

`HookRole`:

- `COMM`: Dispatch first (dispatch / combine A2A).
- `COMPUTE`: Dispatch after COMM notification (expert compute / attention).

### 4.3 CommComputeOverlap.run

```python
overlap.run(fwd_fn, bwd_fn)
```

The main thread runs `fwd_fn`; a spawned daemon thread runs `bwd_fn`. `coordinator.enable()` resets the barrier and per-rendezvous events. An exception on either side immediately disables the coordinator and releases the other side's wait to avoid deadlock. **No `num_layers` argument is needed**; see §4.5.

### 4.4 Cross-Layer `D → A_next` Window

Between layers, layer k's combine (D hook) is adjacent to layer k+1's dispatch (A hook). Assigning COMM/COMPUTE roles to them lets one side's combine.bwd overlap the other side's attention computation, providing a second window beyond within-layer dispatch ↔ expert compute.

### 4.5 Final-Layer D Handling: Static `D_LAST` Annotation

FWD and BWD would each have 4N hooks (A/B/C/D per layer), but the backward entry is asymmetric. Autograd first invokes `combine_N.bwd`, an A2A, **before** any backward hook fires. This is a free dispatch: when D_N's backward hook would rendezvous, the A2A is already on the EP stream, so the rendezvous has no purpose.

Therefore, skip D_N on both sides:

- **FWD**: D_N is the last D; its next operation would be layer N+1 attention, which does not exist, so rendezvous is unnecessary.
- **BWD**: D_N is the first backward hook, and the preceding combine.bwd has already dispatched freely, so rendezvous is unnecessary.

**Implementation**: `CommComputeOverlap.wrap_combine(combine_fn, is_last_layer=True)` tags the closing D hook as `"D_LAST"`. `_TorchSyncHookFunction` returns immediately when forward/backward sees this tag:

```python
if hook_name == "D_LAST":
    return x  # forward
if hook_name == "D_LAST":
    return grad_output, None, None  # backward
```

The same hook is the last D in FWD and first D in BWD, so **one tag handles both boundary cases**. `HookCoordinator` no longer needs runtime counters. Earlier `_num_layers` / `increment_cycle` logic disabled FWD after N layers, while `_bwd_local` thread-local counting skipped the first backward D; static annotation replaces both mechanisms.

After skipping, FWD and BWD each run 4N-1 rendezvous, paired strictly by position:

```text
FWD A_1 ↔ BWD C_N    (COMM ↔ COMPUTE: FWD dispatch_1 overlaps BWD expert.bwd_N)
FWD B_1 ↔ BWD B_N
FWD C_1 ↔ BWD A_N
FWD D_1 ↔ BWD D_{N-1}
...
FWD C_N ↔ BWD A_1
```

Every pair has at least one COMM side, avoiding COMPUTE+COMPUTE deadlocks.

---

## 5. PP P2P and RECV-Compute Overlap

With `overlap_p2p=True` in the `OVERLAP_B_F` callback:

```python
# Callback entry: main thread, default stream.
fwd_recv_handles = ctx.fwd_recv_ops.pop((fwd_stage.stage_index, fwd_mi), None)
bwd_recv_handles = ctx.bwd_recv_ops.pop((bwd_stage.stage_index, bwd_mi), None)

def fwd_fn():
    if fwd_recv_handles:
        schedule._wait_p2p(fwd_recv_handles)   # Wait on the main thread's default stream.
    out = fwd_stage.forward_one_chunk(fwd_mi, ...)

def bwd_fn():
    torch.npu.set_device(device.index)         # Bind the BWD thread's device to avoid HCCL hangs.
    if bwd_recv_handles:
        schedule._wait_p2p(bwd_recv_handles)   # Wait on the BWD thread's default stream.
    bwd_stage.backward_one_chunk(bwd_mi, ...)
```

Effects:

- FWD/BWD RECV is issued to HCCL P2P streams in the preceding step, naturally overlapping current OVERLAP_B_F computation.
- Callback `wait()` places the PP-RECV completion event on the current default stream. Subsequent EP A2A obtains this dependency through the PyTorch HCCL backend's `queueEvent(currentStream)` chain, starting only after PP-RECV completes.
- SEND needs no special callback handling: `overlap_p2p=True` accumulates send handles for final draining, and HCCL orders them through producer-stream events automatically.

### 5.1 Key Design Decision: No Separate BWD Stream

An earlier version tried `with torch.npu.stream(bwd_stream):` to execute backward on a side stream for physical FWD/BWD compute overlap. **Measurements showed this did not work**:

1. `torch.autograd.backward()` dispatches backward kernels on the producer streams of tensors saved during forward, **not** the thread-local current stream selected by `with torch.npu.stream(...)`. BWD computation still runs on the main default stream.
2. `recv_buffer` is allocated on the main default stream, and PyTorch allocator-stream tracking binds it there. Even if `handle.wait()` places an event on bwd_stream, EP A2A sees the default stream through allocator dependencies, without that wait event. A2A can start prematurely and race PP-RECV.

The final design runs BWD on its thread's default stream, physically the same as the main default stream, aligning waits and buffer tracking. Losing true FWD/BWD compute concurrency has limited impact for MoE: A2A runs on EP HCCL streams and physically overlaps default-stream computation. This window comes from communication-stream separation and **does not require** a separate BWD compute stream.

---

## 6. End-to-End Timeline

A steady-state step with PP=2 / EP=2 / chunks_per_rank=2 / moe_layers=2:

```text
Default stream (compute): [BWD attention][FWD attention][BWD expert][FWD expert]...
EP HCCL stream (A2A):              [BWD combine.bwd][FWD dispatch][BWD dispatch.bwd][FWD combine]...
PP-RECV stream:           [current BWD-RECV.....] [current FWD-RECV.....]
PP-SEND stream:                                           [previous BWD-SEND.....]
```

Overlap relationships:

- EP A2A ↔ default-stream compute: the main benefit; A/B/C/D hooks guarantee dispatch order.
- PP-RECV ↔ EP A2A / default-stream compute: receives are issued one step early.
- PP-SEND ↔ next-step compute: handles are drained at the end.

---

## 7. API Overview

```python
from hyper_parallel.core.pipeline_parallel import (
    ScheduleInterleaved1F1B,           # Schedule with overlap flags.
    CommComputeOverlap,                # Two-thread orchestrator.
    HookCoordinator, HookRole,         # Low-level rendezvous primitives.
    MetaStepType, MetaStep,            # Scheduling units.
    PipelineContext,                   # Callback context.
)

# 1. Construct the schedule with B/F and P2P overlap enabled.
schedule = ScheduleInterleaved1F1B(
    stages, micro_batch_num, overlap_p2p=True, overlap_b_f=True,
)

# 2. Attach hooks to each MoE layer's dispatch/combine; mark the final layer is_last_layer=True.
overlap = CommComputeOverlap()
last_idx = len(moe_layers) - 1
for i, layer in enumerate(moe_layers):
    layer.experts.dispatch = overlap.wrap_dispatch(layer.experts.dispatch)
    layer.experts.combine  = overlap.wrap_combine(
        layer.experts.combine, is_last_layer=(i == last_idx),
    )

# 3. Register the OVERLAP_B_F callback.
def callback(step, ctx):
    bwd_step, fwd_step = step.sub_steps
    ...
    overlap.run(fwd_fn=..., bwd_fn=...)

schedule.register_custom_function(MetaStepType.OVERLAP_B_F, callback)

# 4. Run one iteration.
losses = schedule.run(*inputs)
```

See `examples/torch/pp_overlap/pp_overlap_moe_example.py` for the full example.

---

## 8. Native Torch Synchronization Hooks

`_SyncHookFunction` in `core/pipeline_parallel/_sync_hook.py` directly subclasses
`torch.autograd.Function`. During forward, it derives the COMM/COMPUTE role from `hook_name` (`A`/`B`/`C`/`D`)
and calls `coordinator.rendezvous`; backward reverses the roles symmetrically.
`D_LAST` preserves final-layer notification semantics while skipping the boundary rendezvous.

`CommComputeOverlap` calls this hook directly. Each overlap window creates and waits for a new
backward thread. PP no longer uses the Platform abstraction.

---

## 9. Limitations and Future Work

### Current Limitations

- **MoE layer alignment**: Unequal FWD/BWD chunk layer counts leave extra hooks unpaired and cause an indefinite barrier wait. The implementation assumes equal counts; otherwise register hooks only up to the shorter side, leaving excess layers unhooked. The framework does not handle this automatically.
- **No concurrent FWD/BWD compute**: Autograd stream behavior keeps BWD on the forward stream; this is currently accepted.
- **OVERLAP_F_B lacks P2P integration**: Only OVERLAP_B_F currently supports `overlap_p2p=True`; the symmetric F_B path needs a similar adaptation.

### Future Optimization Directions

1. **Separate directional FWD/BWD process groups**: Avoid serialization within an HCCL P2P group so FWD-RECV and BWD-RECV can run on separate HCCL streams concurrently.
2. **CUDA Graph / static execution**: Capture OVERLAP_B_F dispatch/combine/compute kernel sequences in a graph to further reduce host dispatch overhead.
3. **True FWD/BWD compute concurrency**: Redesign forward stream assignment, for example assigning chunks to two streams during forward. This requires substantial changes.

---

## 10. Validation

- **Numerical consistency**: With identical seeds and inputs, `overlap_p2p=False` and `True` produce exactly equal losses and parameter gradients (`atol=0, rtol=0`).
- **Profile check 1**: EP A2A operations fully overlap computation within one OVERLAP_B_F step, preserving the A/B/C/D hook chain.
- **Profile check 2**: The number of A2A operations overlapping PP-RECV within OVERLAP_B_F is ≤ the FWD A2A count, confirming that BWD A2A correctly waits for PP-RECV.
- **End to end**: `pp_overlap_moe_example.py` runs on eight devices (PP=2 × EP=4 or PP=4 × EP=2), with loss convergence as expected.
