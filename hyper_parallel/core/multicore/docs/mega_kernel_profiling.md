# MegaKernel Multicore Profiling Design and Usage

**English** | [简体中文](mega_kernel_profiling.zh.md)

---

## 1. Background and Goals

### 1.1 Problem

MegaKernel fuses communication and computation stages into one device kernel. CANN `msprof` and `torch.profiler` expose its outer
start, end, and total duration, but cannot directly show:

- Which AIC/AIV workers are waiting for dependencies.
- How GMM, SwiGLU, Dispatch, and Combine overlap across cores.
- Which expert or stage causes tail latency.
- Whether abnormal gaps occur between event triggers and downstream computation.

Internal cycle intervals must therefore be collected and reconstructed into a readable host-side timeline.

### 1.2 Design Goals

| Goal | Design choice |
|------|----------|
| Runtime enable/disable | Normal and profiled RuntimeConfig use the same operator artifact |
| Protect the disabled path | Workers branch once before the task loop, without per-task checks on the fast path |
| Generic integration | Shared records store only generic task, stage, owner, and cycle fields |
| Readable domain semantics | Stage names come from ComputeGraph nodes; modules add optional owner semantics only |
| Low-interference collection | Device writes fixed-size records; naming and JSON generation run on the host |
| Standalone use | Internal traces open directly without an outer profiler |
| Joint analysis | Pure host interfaces merge offline with native Chrome Trace |
| Trustworthy diagnostics | Explicitly report buffer overflow, unknown SoCs, misalignment, and approximate host alignment |

### 1.3 Scope

The implementation includes a standalone Torch frontend, automatic MegaMoE forward/backward integration, Chrome Trace schema v1 export, and offline merge.
`mega_kernel_profile()` does not start or wrap `torch.profiler`. Collection, standalone export, and
merging are three independent phases.

---

## 2. Overall Architecture

### 2.1 Peer Tools and the Shared Layer

Profiling and L0/L1 dump both belong to `diagnostics`. They integrate through protocols and composition, without inheriting from or calling one another:

```text
diagnostics/
├── api.py                       # prepare / MegaKernelDiagnostics.launch
├── common/
│   ├── graph.py                 # Generic graph contract and read-only task/stage/owner metadata.
│   ├── runtime.py               # Ordinary runtime and lazy metadata.
│   ├── call.py                  # Complete LaunchArguments and completion/cancellation boundaries.
│   ├── session.py               # DiagnosticSession protocol and process-wide mutual exclusion.
│   └── schedule.py              # DiagnosticAction, schedule
├── profiler/
│   ├── api.py                   # Profiling sessions, buffer pool, pending calls, and windows.
│   ├── cycle_trace.py           # Profiler-private configuration, capacity, and parsing.
│   └── trace_merger.py          # Offline merging.
└── dump/                        # Independent L0/L1 tools; does not import profiler.
```

Dependencies flow as `profiler → common ← dump`; the shared layer imports neither concrete tools nor MegaMoe.
`modules/<kernel>/diagnostics.py` supplies business-specific owner rules. Dynamic tensor-view resolvers remain in the business module's own
`tensor_dump.py`. The generic profiler has no knowledge of experts, GMM, or MoE routing.

### 2.2 Data Flow

```text
plan: create only the normal RuntimeConfig image
    │
Check diagnostics.is_active on every call
    ├─ False: call the original native wrapper directly → mode=0 → ProcessFast
    └─ True: generic launch → sole active session.prepare_call(context)
                               │
                   First profiler use of this plan:
                   Shared metadata (lazy, read-only)
                       → Private host cfg: trace IDs/owners + AIC/AIV capacity
                       → Private device image: mode=1
                               │
                   per-invocation diagnostic_buffer
                               │
                   Device cycle records
                               │
                   step/window/pending limit:
                   NPU synchronize → D2H → parse
                               │
                   Standalone trace → optional offline merge
```

Each call creates only one handle for the current tool; it does not prepare a profile handle and then overlay or replace it with dump.
The tools share the `diagnostic_buffer` ABI argument position, rather than writing the same memory simultaneously.
Their sessions are currently mutually exclusive. Neither changes business event counters, the ready handshake, workspaces, or autograd leases.

---

## 3. RuntimeConfig and Invocation Lifecycle

### 3.1 Ordinary Configuration and Tool-Private Derived Configurations

| Configuration | Preparation time | Key contents |
|---|---|---|
| normal | Plan construction | diagnostic_mode=0; original task/queue data |
| Profiler-private configuration | First selected WARMUP/RECORD/RECORD_AND_SAVE | diagnostic_mode=1; trace IDs, owners, and per-core capacity |
| L1 dump-private configuration | Every selected L1 call | diagnostic_mode=2; does not use the profiler's trace IDs/capacity |

`prepare_mega_kernel_diagnostics()` retains the complete host configuration and creates one ordinary image,
without parsing diagnostic metadata, running resolvers, or calculating the profile layout.
When disabled or action=NONE, the business path calls the ordinary wrapper directly, without entering diagnostic launch or creating diagnostic handles.

When profiling is first selected, `_PreparedProfileRuntime` copies the host configuration, encodes trace descriptors and owners in the copy,
calculates capacity, and uploads a separate profile image. Ordinary host/device configurations remain unchanged.
The session caches profiler configurations by shared runtime object and releases its cache on exit.

The shared owner representation is `GraphDiagnosticMetadata.task_owner_ids`; `profile_owner_id` is only the profiler's
private device encoding, which dump does not read. Dynamic-view resolvers and dump-directory validation do not enter profiler preparation.

### 3.2 Schedule State Machine

`schedule(skip_first, wait, warmup, active, repeat)` returns these actions for zero-based steps:

| Action | Device sampling | Retain window data | Boundary behavior |
|--------|-------------|--------------|----------|
| `NONE` | No | No | Use disabled configuration |
| `WARMUP` | Yes | No | Synchronize and discard at step end |
| `RECORD` | Yes | Yes | Retain invocation buffers and continue the window |
| `RECORD_AND_SAVE` | Yes | Yes | Complete D2H, parsing, and the window callback |

Call `profiler.step()` inside the context exactly once per logical user step. On context exit, any remaining
valid `RECORD` data completes the latest window automatically. Nested or concurrent profiler contexts within a process are rejected.

### 3.3 Per-Invocation Device Memory and Reuse

MegaMoE workspaces retain separate graph-sized symmetric `all_event_counters` for forward and backward. The prefix holds scheduling counters;
when EP > 1, the tail also holds a monotonically increasing ready generation across calls. Each launch clears only the counter prefix, preserving ready
generation.
On collection actions, the Torch profiler obtains an ordinary NPU buffer from its session-private pool by `(device_id, buffer_size)`:
`diagnostic_buffer`. Each undrained invocation owns a separate buffer, so forward/backward calls from multiple layers in one step cannot
overwrite each other. Device records go directly into the final buffer without a D2D snapshot.

D2H and parsing occur only when:

1. A warmup step ends and data must be discarded.
2. An active window ends.
3. Pending calls reach `max_pending_calls`.
4. The context exits normally with unfinished data.

Draining synchronizes the NPU first. If retaining data, each buffer is transferred directly to the host and parsed, then the device buffer returns to the current session's
pool for reuse. Exceptional exit frees pending and pooled memory without generating potentially misleading partial traces.
The pool is destroyed with the session at context exit; it does not retain device memory across models or later contexts.

---

## 4. Host/Device ABI and Buffers

### 4.1 RuntimeConfig Fields

Python `RuntimeConfigC` and Ascend C `RuntimeHeader` must agree on field order and widths in the fixed 64-byte header. Task, event,
and other arrays follow the header, serialized at actual graph capacity. Profiling reuses reserved header fields without changing dynamic-array layouts:

| Field | Type | Description |
|------|------|------|
| `diagnostic_mode` | `uint32` | 0=ordinary/L0, 1=profiling, 2=L1 dump |
| `aic_profile_record_capacity` | `uint32` | Record capacity per AIC worker |
| `aiv_profile_record_capacity` | `uint32` | Record capacity per AIV worker |

Each `TaskDesc` additionally contains:

| Field | Type | Description |
|------|------|------|
| `profile_desc_id` | `uint32` | Stage semantics; `0xFFFFFFFF` when unset |
| `profile_owner_id` | `uint32` | Domain owner; `0xFFFFFFFF` when unset |

The shared device runtime does not understand experts, layers, or other domain concepts; it passes IDs through. Stage names and owner labels remain only in host
metadata, consuming no device-record string storage. The host visits nodes through `ComputeGraph.topological_sort()`,
uses each node's `task_num` to reconstruct contiguous TaskDesc ranges, and assigns internal descIds automatically. Display names prefer
`OperatorNode.diagnostic_name`, falling back to the node's `name` converted to CamelCase. descId is internal to RuntimeConfig;
callers must not depend on specific numeric values.

### 4.2 Buffer Layout and Internal ABI

Scheduling and profiling use separate buffers with independent semantics, memory types, and lifetimes:

```text
all_event_counters (graph-sized symmetric memory)
├── scheduler event counters
└── ready generation tail (EP > 1)

diagnostic_buffer (ordinary NPU memory, allocated only for capture actions)
├── AIC slot 0
│   ├── 64-byte core header
│   └── N × 32-byte records
├── ...
└── AIV slots
```

Each AIC/AIV worker owns a slot, avoiding a global record allocator and atomic contention. Core headers store entry cycles,
valid/dropped counts, core type, block ID, and capacity. Records store start/end cycles, `desc_id`, `task_id`,
`stage_task_index` and `owner_id`. Profile slots begin at offset 0 of `diagnostic_buffer` and do not occupy symmetric
heap.

`diagnostic_buffer` is a shared diagnostic input used equally by profiling and L1 tensor dump; the two tools' sessions are currently mutually exclusive.
Profiling mode writes cycle traces, L1 dump mode writes tensor snapshots, and ordinary execution and L0 dump do not read its diagnostic contents.
It passes through the internal native ABI: forward/backward OpProto, OpDef, ACLNN GetWorkspaceSize, L0, and device
kernel inputs. Public Python `mega_moe()`, `mega_moe_grad()`, and
`MegaMoeExperts` signatures remain unchanged. The private adapter fills this argument for ordinary calls. With profiling disabled, the original
`all_event_counters` is an unread placeholder tensor, requiring no new allocation, slicing, or transfer.

### 4.3 Capacity Calculation

The host reproduces each worker's possible tasks from the complete schedule, calculating record counts for the busiest AIC and AIV separately:

```text
capacity = min(round_up(required_records, 16), 256)
```

Each ordinary task with dependencies records at most WaitDependency, Compute, and TriggerEvent. SHMEM put does not duplicate the shared
trigger interval. At the 256-record limit, recording stops and dropped counts increase without overwriting existing data.

Total profile-buffer capacity is `24 * AIC_stride + 48 * AIV_stride`, with maximum
`72 * (64 + 256 * 32) = 594432` bytes, excluding the scheduling event workspace.
`max_pending_calls` controls host-session drain frequency, independently of per-core record capacity.

### 4.4 Cycle-to-Time Conversion

The device reads the system counter with `GetSystemCycle()`:

```text
timestamp_us = (start_cycle - anchor_cycle) / cycle_frequency_mhz
duration_us  = (end_cycle - start_cycle) / cycle_frequency_mhz
```

The internal frequency map currently supports Ascend 910B/910C/910_93, with A2/A3 using 50 MHz. SoC names come from
`torch_npu.npu.get_device_name()`. Unknown models explicitly fail when capture results are parsed; the frontend does not accept user-supplied guesses.

### 4.5 Disabled Fast Path

After the ready handshake and before the `Process()` task loop, the worker reads `DiagnosticMode` and selects the fast or slow implementation. The disabled path has these constraints:

- Does not read system cycles.
- Runs no profiling barriers.
- Writes no core headers or records.
- Does not read `profile_desc_id` / `profile_owner_id`.
- Clears only the scheduling counter prefix of `all_event_counters`, preserving the ready-generation tail.
- It allocates no ordinary NPU `diagnostic_buffer` and creates no event-counter view; the device does not read the internal placeholder argument.

This keeps profiling checks out of each task's hot loop.

---

## 5. Device Sampling Semantics

### 5.1 Intervals for One Task

The sampling path records intervals in execution order:

```text
WaitDependency → Compute → TriggerEvent
```

- Tasks without dependency events produce no WaitDependency interval.
- Tasks without a domain `profile_desc_id` use `TASK_TYPE_BASE + task_type` for Compute.
- SHMEM put signals from its own compute kernel and records no additional shared TriggerEvent.
- Records preserve write order within each worker; host export then stably sorts by cycle.

### 5.2 Generic Names

Shared descIds include at least `WaitDependency` and `TriggerEvent`. Graph node stages receive internal descIds automatically.
Wait/Trigger names include the current task's graph node, such as `GMM1_WaitDependency`. Tasks outside the graph without stage metadata
use task-type fallback names.

### 5.3 Concise and Detailed Names

| `detailed_task_names` | Example | Use case |
|-----------------------|------|----------|
| `False` | `GMM1` | Inspect the overall pipeline and compute-communication overlap |
| `True` | `Expert3_GMM1_task72` | Locate expert, worker, or task tail latency |

Both modes retain original task, desc, owner, and cycle fields in event `args`; changing names loses no raw data.

---

## 6. MegaMoE Semantic Integration

### 6.1 Forward Stages

| Stage | Typical execution resource | Meaning |
|------|--------------|------|
| Dispatch | AIV / SHMEM | Dispatch tokens to target experts |
| GMM1 | AIC | gate/up projection |
| SwiGLU | AIV | Activation computation |
| GMM2 | AIC | down projection |
| Combine | AIV / SHMEM | Return expert outputs to source ranks |

### 6.2 Backward Stages

| Stage | Meaning |
|------|------|
| DispatchGrad | Gradient dispatch |
| ActGrad | Activation input-gradient GMM |
| W2Grad | Down-projection weight gradient |
| SwiGLUGrad | Activation backward |
| GateGrad | Gate/up input-gradient GMM |
| W1Grad | Gate/up weight gradient |
| CombineGrad | Return gradients to source ranks |

### 6.3 Graph-Driven Stage Semantics

MegaMoE stage order and TaskDesc ranges come directly from the forward/backward `ComputeGraph` used to generate RuntimeConfig. Each
`OperatorNode` may set `diagnostic_name` in place, keeping domain names such as GMM1, GMM2, and DispatchGrad next to
their operator definitions. Shared metadata is parsed in topological order on first capture; the profiler writes descId into its own configuration copy, without inferring stages from `task_type`, tiling positions, or input positions.

`modules/mega_moe/diagnostics.py` retains only the `Expert` owner label and global expert-ID resolver. The resolver receives both
`OperatorNode` and TaskDesc, distinguishing same-type operations such as Dispatch/Combine by node identity. Shared GMM, SwiGLU, and
AllToAll task builders do not write MegaMoE owners; model users need not supply stage names, ranks, devices, or RuntimeConfig.

### 6.4 Forward/Backward Launch Integration

```text
_MegaMoeFunction.forward / backward
    ├─ diagnostics.is_active=False → original mega_moe/mega_moe_grad wrapper
    └─ True → diagnostics.launch(runtime ABI parameters, direction, events)
                → profiler.prepare_call → complete profile arguments
                → native wrapper_with_diagnostic_buffer
                → complete / cancel on exception
```

`direction` distinguishes forward and backward invocations. After a successful launch, the handle marks completion, and the device writes directly into the invocation's exclusive buffer,
without a D2D snapshot; synchronization and readback happen at drain. Exception cancellation rolls back only this invocation's profiling state. Model users need not pass diagnostic arguments.

---

## 7. Public Torch Interface

### 7.1 Imports

```python
from hyper_parallel.core.multicore.diagnostics import profiler as multicore_profiler
```

Import concrete tools from `diagnostics.profiler`; the shared `diagnostics` package does not import tools. The project is Torch-only and has no backend dispatch.

### 7.2 Shared Schedule

```python
from hyper_parallel.core.multicore.diagnostics import schedule

schedule(
    *,
    wait: int,
    warmup: int,
    active: int,
    repeat: int = 0,
    skip_first: int = 0,
)
```

All arguments must be nonnegative integers, with `active > 0`. The returned object accepts a zero-based step and produces a `DiagnosticAction`.

### 7.3 mega_kernel_profile

```python
multicore_profiler.mega_kernel_profile(
    *,
    schedule=None,
    on_trace_ready=None,
    detailed_task_names: bool = False,
    max_pending_calls: int = 16,
)
```

| Parameter | Contract |
|------|------|
| `schedule` | Callable step schedule; omitting it retains every invocation in the context |
| `on_trace_ready` | Calls `callback(profiler)` after each completed window |
| `detailed_task_names` | Controls display names without changing raw IDs |
| `max_pending_calls` | Positive integer limiting retained invocation buffers before intermediate D2H draining |

The interface accepts no rank, device, cycle frequency, RuntimeConfig, or event buffer. These are inferred from the actual MegaKernel plan, input
tensors, and current Torch NPU device.

### 7.4 step and export

```python
with multicore_profiler.mega_kernel_profile(schedule=capture_schedule) as profiler:
    train_step(batch)
    profiler.step()

trace = profiler.export_chrome_trace("rank0_mega_kernel_trace.json")
```

`export_chrome_trace()` exports only the latest completed window. Without one, it raises `RuntimeError`; it neither reads native
framework traces nor merges implicitly.

---

## 8. Standalone Trace Schema

### 8.1 Top-Level Structure

```json
{
  "traceEvents": [],
  "megaKernelCycleTrace": {
    "schemaVersion": 1,
    "invocationCount": 1,
    "recordCount": 1411,
    "droppedRecordCount": 0,
    "warnings": []
  }
}
```

`traceEvents` follows Chromium Trace Event Format. Internal intervals use complete events (`ph: "X"`) with category
`MegaKernelInternal`; process/thread metadata creates stable tracks for each AIC/AIV worker.

### 8.2 Event args

Each complete event retains at least:

| Field | Description |
|------|------|
| `rank` / `device_id` | Data source |
| `core_type` / `block_id` | AIC/AIV worker |
| `task_id` / `desc_id` | Original task and stage identifiers |
| `stage_task_index` / `task_number` | Zero-based stage-local ID and one-based display number |
| `start_cycle` / `end_cycle` | Raw device cycles |
| `core_entry_cycle` | Current worker's entry cycle |
| `invocation_id` / `step` / `direction` | Launch, logical step, and forward/backward attribution |
| `owner_id` | Optional domain owner, such as a global expert ID |

### 8.3 Metadata Health Checks

Top-level `megaKernelCycleTrace` summarizes window information. `invocations` records each launch's kernel, direction, rank, device,
anchor, and record count. Before formal analysis, check:

```text
schemaVersion == 1
recordCount > 0
droppedRecordCount == 0
warnings == []
```

---

## 9. Offline Trace Merging

### 9.1 Interface

```python
merged = multicore_profiler.merge_chrome_traces(
    framework_trace,
    mega_kernel_trace,
    output,
    kernel_pattern=None,
    kernel_index=0,
    outer_pid=None,
    allow_host_wrapper=False,
)
```

All three trace arguments may be paths. `framework_trace` also accepts an object containing `traceEvents` or a top-level event list;
`mega_kernel_trace` accepts a standalone trace object. The merged result is written to `output` and returned as a dict.

### 9.2 Invocation Grouping

An active window may include multiple layers and forward/backward launches. The merger groups by `args.invocation_id`, then pairs each group
with an outer kernel in order of first appearance. It supports legacy single-invocation schema v1 traces without `invocation_id`, but rejects
ambiguous inputs where only some events have IDs.

### 9.3 Outer Kernel Selection

Default workflow:

1. Derive a case-insensitive regex allowing underscores from internal `kernelName`.
2. Consider only complete events with positive durations.
3. Prefer device events with NPU/AI Core/kernel evidence.
4. Sort by outer timestamps and pair with invocations starting at `kernel_index`.
5. If automatic name matching fails, allow duration-similarity fallback between the internal span and consecutive device kernels, recording a warning.
6. If an explicit `kernel_pattern` matches nothing, fail without name fallback.
7. Reject host wrappers by default; allow approximate alignment only with explicit `allow_host_wrapper=True`.

A CANN trace event can be recognized as a device kernel even with an empty category when `args["Task Type"]` is `MIX_AIC` or `MIX_AIV`.

### 9.4 Time and Track Alignment

Preserve each group's relative timestamps and durations, shifting its earliest internal event to the corresponding outer kernel's `ts`. Merged events use
the outer process ID and a new nonconflicting thread ID per `(core_type, block_id)`, named:

```text
<kernelName>/<coreType>/<blockId>
```

An earlier `thread_sort_index` places internal AIC/AIV tracks near the outer NPU kernel.

### 9.5 Merge Diagnostics

Merged output adds `megaKernelTraceMerge`:

| Field | Purpose |
|------|------|
| `mergedInvocationCount` / `mergedEventCount` | Verify merge scope |
| `usedFallbackKernelSelection` | Whether name matching fell back |
| `alignments` | Selected outer pid/tid/time and score for each invocation |
| `exceedsOuterKernelUs` | Duration by which the internal span exceeds the outer kernel |
| `warnings` | Fallback, host alignment, and boundary anomalies |

Successful merging does not guarantee trustworthy alignment; callers must inspect these diagnostics.

---

## 10. Usage

### 10.1 Steady-State Forward Collection

```python
from pathlib import Path

import torch
import torch.distributed as dist

from hyper_parallel.core.multicore.diagnostics import profiler as multicore_profiler, schedule as diagnostic_schedule

rank = dist.get_rank()
schedule = diagnostic_schedule(wait=0, warmup=0, active=1, repeat=1)

with torch.no_grad():
    # Perform real warmup with the same shape outside the context.
    experts(hidden_states, topk_ids, topk_weights, tokens_per_expert=tokens_per_expert)
    torch.npu.synchronize()
    dist.barrier()

    with multicore_profiler.mega_kernel_profile(
        schedule=schedule,
        detailed_task_names=True,
    ) as profiler:
        experts(hidden_states, topk_ids, topk_weights, tokens_per_expert=tokens_per_expert)
        profiler.step()

output = Path("./traces") / f"rank{rank}_mega_kernel_trace.json"
trace = profiler.export_chrome_trace(output)
if trace["megaKernelCycleTrace"]["droppedRecordCount"]:
    raise RuntimeError("incomplete MegaKernel profile: device records were dropped")
```

Use shapes equivalent to the target workload to avoid fragmented task views from undersized shapes. Complete NPU synchronization and rank barriers before
active windows to avoid mistaking startup skew for internal kernel gaps.

### 10.2 Training Windows

```python
from hyper_parallel.core.multicore.diagnostics import schedule as diagnostic_schedule

capture_schedule = diagnostic_schedule(
    skip_first=5,
    wait=1,
    warmup=1,
    active=2,
    repeat=1,
)

with multicore_profiler.mega_kernel_profile(schedule=capture_schedule) as profiler:
    for batch in data_loader:
        optimizer.zero_grad()
        loss = train_step(batch)
        loss.backward()
        optimizer.step()
        profiler.step()

profiler.export_chrome_trace(f"rank{rank}_train_trace.json")
```

Forward/backward launches in one window automatically record `direction=forward/backward`. Do not call `step()` after each layer or
MegaKernel launch.

### 10.3 Export Each Window

```python
from pathlib import Path
from hyper_parallel.core.multicore.diagnostics import schedule as diagnostic_schedule


def on_trace_ready(profiler):
    path = Path("./traces") / f"rank{rank}_window{profiler.step_num}.json"
    profiler.export_chrome_trace(path)


with multicore_profiler.mega_kernel_profile(
    schedule=diagnostic_schedule(wait=1, warmup=1, active=2, repeat=3),
    on_trace_ready=on_trace_ready,
) as profiler:
    for batch in data_loader:
        train_step(batch)
        profiler.step()
```

### 10.4 Merge with msprof Results

```bash
msprof --output=/path/to/dump python your_program.py
```

After locating the native `trace.json` for the current rank/device, run:

```python
from hyper_parallel.core.multicore.diagnostics import profiler as multicore_profiler

multicore_profiler.merge_chrome_traces(
    "/path/to/dump/.../trace.json",
    f"./traces/rank{rank}_mega_kernel_trace.json",
    f"./traces/rank{rank}_merged_trace.json",
)
```

Merge multiple ranks separately. Do not align rank 0's internal trace to another rank or device's outer trace.

---

## 11. Integrate a New MegaKernel

### 11.1 Default Graph Node Names

New MegaKernels using `ComputeGraph` derive stage names from `OperatorNode.name` by default; for example, `qkv_proj` displays as
`QkvProj`. Set `diagnostic_name` next to node definitions to preserve abbreviations or use clearer domain terms. Tasks outside the graph can still
show WaitDependency, TriggerEvent, TerminateTask, and task-type fallbacks. Nodes may share display names;
generic logic still assigns distinct internal descIds by topological position.

### 11.2 Add Domain Semantics

Recommended integration order:

1. Build RuntimeConfig tasks topologically from the same `ComputeGraph`, ensuring each node's `task_num` matches its actual contiguous
   TaskDesc range.
2. Set `diagnostic_name` on `OperatorNode` instances needing custom names; reuse `name` for others.
3. If owners are needed, implement a resolver receiving the node, TaskDesc, and topology in `modules/<kernel>/diagnostics.py`;
   Return `None` for nodes without owners.
4. The RuntimeConfig builder generates only ordinary execution tasks/queues and writes no diagnostic fields;
5. Call `prepare_mega_kernel_diagnostics(graph, runtime_config, ...)` in the plan. You may explicitly call
   `validate_mega_kernel_diagnostic_contract()` to check the graph/task/ABI; diagnostic metadata and the profile image are prepared lazily;
6. Thread `diagnostic_buffer` through internal OpProto, OpDef, ACLNN, L0, device kernel, and framework adapter inputs while keeping
   model-facing interfaces unchanged.
7. Check `diagnostics.is_active` first in forward and backward. When disabled, use the original wrapper directly; when enabled, use
   `with diagnostics.launch(...)` to supply complete arguments and manage completion/cancellation;
8. Add layout, metadata, Torch session, export, merge, and real-NPU tests.

Graph nodes are the sole source of stage identity; do not create another TaskDesc field-matching table. If a builder cannot guarantee topological order with contiguous tasks per
node, first record each node's task-ID ranges explicitly in the generic graph/fill layer, then apply graph metadata. Do not return to
inference from tiling or input slots. Shared task builders must not write model-specific owner semantics.

### 11.3 ABI Change Requirements

When modifying headers, records, capacities, or core types, update all of:

- Python ctypes and Ascend C structs.
- Input order in OpProto, OpDef, operator JSON, ACLNN GetWorkspaceSize, L0, and framework adapters.
- Host buffer-size and slot-offset calculations.
- Device recorder；
- Host parsers and schema versions.
- Unit tests, NPU integrity assertions, and this document.

Structural changes must advance the relevant schema; old parsers must not silently accept incompatible layouts.

---

### 11.4 Native ABI and Broadcast Queues

MegaKernels with only scheduling dependencies and tensor addresses hardcoded in C++ can lazily supply real ports through `graph_adapter`.
Place the callback in `modules/<kernel>/diagnostics.py`. It receives the original graph and private host configuration and returns a diagnostic graph; it must not modify the original graph or ordinary
device configuration. `bind_graph_tensor_ports()` sets `diagnostic_host_ports` on a graph copy. These ports come from host descriptions,
so they are not limited by the four-entry serialization capacity of each `TaskDesc.inputs/outputs` array and do not rewrite TaskDesc data used by computation.
Ports can use `shape=[]` and `is_dynamic=True` to indicate shapes resolved at call time. Actual exported dimensions follow `actualShape`
and `storedShape`. Exported ranks are job-global ranks, while MoE expert-ID/routing calculations still use EP group-local ranks;
business `owner_context` distinguishes the two, preventing multiple parallel groups from all writing to `rank_0`.

Ordinary queues use `execution_mode="event_driven"`. Paired MoE AIV workers use `vector_worker_stride=2`,
and MHC's all-AIV workers use `vector_worker_stride=1`. MegaGate uses `execution_mode="aiv_pipeline"`: each
worker executes the same task IDs on its own token-row partition, without scheduling events. The profiler calculates per-core capacity from the complete stage sequence;
L1 allocates records/slots by `(task_id, worker_id)`, and the device checks worker IDs to prevent different cores from writing the same snapshot.
Broadcast directories use `task_<id>_worker_<id>`; ordinary queues retain `task_<id>`.

MHC binds actual ABI tensors after the Torch wrapper allocates the original outputs and before launch. With diagnostics enabled, MegaGate forward uses
`mega_gate_route_out` to preallocate the same five outputs; when disabled, it retains the original native wrapper. Address lists are constructed temporarily only in the device L1 branch,
adding no diagnostic address arrays or initialization work to ordinary workers.

| MegaKernel | L0 | L1 scope and boundaries |
|---|---|---|
| MegaMoe | Graph-based capture of inputs before calls and results afterward | Dynamic expert/token views; pull can capture local receive outputs but cannot treat remote sources as local inputs; cooperative GMM outputs instead select downstream inputs whose dependencies are satisfied |
| HyperMegaMhc | Real ABI ports, rather than scheduling-graph workspace aliases | Supports contiguous token tiles; does not present CANN's circular X-cast workspace as `hc_before_norm` or expose noncontiguous Sinkhorn caches with iteration-major layouts |
| HyperMegaMhcGrad | Includes parameter gradients after completed reduction | Supports token-local inputs/outputs; parameter gradients accumulated cooperatively across cores are L0-only, and L1 must select safe ports |
| MegaGate Route | Logits, biases, masks, and five native results | Captures addressable GM ports by actual AIV row partitions; internal CANN workspaces are not exposed as ordinary tensors |
| MegaGate RouteGrad | Includes `grad_logits` after ACLNN postprocessing | Captures addressable inputs within the MegaKernel; scatter/sqrt/softplus postprocessing is outside this task sequence, and final gradients belong only to L0 |

RouteGrad postprocessing is represented by a host boundary node with `task_num=0`. It participates in L0 source/destination organization but has no device
task ID, generates no L1 record, and does not invent a nonexistent MegaKernel task on the timeline.

### 11.5 Tensor Dump Usage and Storage

```python
from hyper_parallel.core.multicore.diagnostics import dump, schedule

config = dump.TensorDumpConfig("./dump", level="L0")
with dump.tensor_dump(config, schedule=schedule(wait=0, warmup=0, active=1, repeat=1)) as dumper:
    output = model(inputs)
    dumper.step()
```

L0 copies external inputs before each selected native call and intermediate/final results after completion, rather than waiting until the entire model finishes to read
the original tensors. Snapshots are independent CPU `torch.Tensor` objects owned by the dumper's window invocation records. At window completion, `torch.save()` writes
`.pt` files preserving original dtypes such as BF16. Multiple stage-port files for the same L0 tensor use hard links. Only `stages/`
and the manifest are exposed; `before/after` are capture-timing metadata, not mandatory exposed directories. Longer windows accumulate CPU memory,
and there is currently no separate total host-memory budget, so constrain active steps, ranks, directions, and tensor filters.
Each invocation handle captures or decodes its own snapshots and delivers the invocation record to the dumper through a completion callback on success; neither reads the other's private state.
On failure, the handle still cleans up its invocation data without submitting partial results to the window.
L0 saves actual stored values at call boundaries and cannot reconstruct historical intermediates before in-place updates or workspace reuse; use safe
L1 task ports for that purpose instead of assuming stage filenames preserve every historical instant.

L1 selects capture targets through `stages`, `task_ids`, `stage_task_indices`, `owner_ids`, `core_types/core_ids`, `tensor_names`,
and `io`. Static graphs use default views; for dynamic or hardcoded address semantics, a business resolver returns
`L1TensorView(source_offset_bytes, shape)` before each kernel launch. The host validates the actual ABI tensor's dtype, contiguity, and full range,
then applies optional truncation and payload allocation; the device does not run Python resolvers.
Static contiguous slices trim tail blocks according to source offsets. The default path does not misinterpret inner-dimension slices spanning rows as contiguous byte ranges. Such noncontiguous ranges require
a business resolver to explicitly select a copyable contiguous range; changing the shape alone does not permit a direct copy.

The L1 device buffer uses version 2:

```text
[0, 32)       Header: version, valid count, region offsets, and total size
[32, 64)      Alignment padding
[64, 262208)  Record table: 4096 × 64 B; only record_count entries are valid
[262208, ...) Payload: each record has a separate slot with a 64 B-aligned start
```

Each record's ten uint32 fields describe task, phase, ABI source position/offset, destination slot, byte count, status, task index,
and worker ID, followed by 24 B of alignment padding. Records and payload slots do not share writers' data-cache lines. Ordinary queues use an any-worker
marker, while broadcast queues use specific worker IDs. The buffer grows with actual payloads; the fixed table size is not a snapshot-capacity limit.
The budget (512 MiB per call by default) and 32-bit buffer ABI limit are checked before constructing the host bytearray or allocating device memory.

The device waits for original dependencies before capturing before, performs computation and completes barriers before capturing after, then publishes the original business completion event. Capture uses generic
byte copying and cache synchronization. With diagnostics disabled, execution still enters the original `ProcessFast`, without initializing a trace recorder, scanning records, resolving views,
allocating snapshot areas, or performing D2H. L1 significantly perturbs execution time; timings with L1 enabled cannot be used to infer business performance.

---

## 12. Testing and Acceptance

### 12.1 CPU Unit Tests

```bash
pytest -v \
  tests/ut/core/multicore/test_profiling.py \
  tests/ut/core/multicore/test_torch_mega_kernel_profiler.py \
  tests/ut/core/multicore/test_trace_merger.py
```

Cover layout/parsing, schedule/session lifecycle, and offline alignment diagnostics separately. Additional workspace and Torch-session
cases must verify that disabled mode neither allocates nor slices profile buffers, pending invocations use distinct buffers, drained buffers can be
reused by device/size, and parsing starts at offset 0 without event counters.

### 12.2 Torch NPU Tests

```bash
pytest -v tests/torch/multicore/test_mega_moe_profiling.py
```

Formal validation must check at least:

- Same-shape warmup outside the profiler context.
- NPU synchronization and rank barriers after warmup.
- A separate trace for every rank.
- Expected `recordCount` and `droppedRecordCount == 0`.
- Valid cycles and correct forward/backward directions.
- Wait/Trigger names for key stages such as GMM1/GMM2.
- Correct merged invocation counts, outer device kernels, and warnings.

### 12.3 Disabled-Mode Performance

Compare a baseline with no profiling runtime against an artifact containing profiling support but using schedule action `NONE`.
Fix workloads, warmup, sample counts, and rank aggregation, and apply a predefined regression threshold rather than comparing a single mean.

### 12.4 Current Regression Coverage

Acceptance should cover ordinary forward/backward results for every MegaKernel, profiler trace completeness, L0 original dtypes/hard links, L1 task/worker
isolation and payload values, as well as failure cleanup, invocation counters, and cached-plan isolation. CPU-simulated planning/decoding tests do not replace NPU
execution, and functional regression does not replace a disabled-state performance A/B comparison.

---

## 13. Constraints and Troubleshooting

| Symptom | Checks |
|------|--------|
| `no completed ... window` | Has the active window ended, and was `step()` called or the context exited normally? |
| Empty trace | Did an active step actually run a MegaKernel, or was there only warmup data? |
| Fragmented timeline | Is the shape too small? Were real warmup, NPU synchronization, and rank barriers completed? |
| Unknown SoC | Is the current device name present in the system-counter frequency map? |
| Dropped records | Did a per-core slot reach the 256 limit? `max_pending_calls` cannot fix this. |
| No merge match | Does the native trace include device kernels? Check pattern, index, and pid. |
| Fallback warning | Verify the outer kernel chosen by duration fallback. |
| Host wrapper warning | Accept approximate results only with explicit `allow_host_wrapper=True`. |
| Incorrect multilayer pairing | Does invocation order match native outer-kernel order? |
| Context conflict | Only one nonnested active session is allowed per process. |

---

## 14. Related Documentation

- [MegaMoe Design and Usage](architecture.md)
- [Component Integration and Standalone Model Examples](../examples/README.md)
