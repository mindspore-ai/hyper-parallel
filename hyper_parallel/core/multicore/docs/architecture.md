# Multicore MoE-FFN Architecture Design

**English** | [简体中文](architecture.zh.md)

---

## 1. Background and Fusion Approach

### 1.1 Problem

The standard MoE-FFN forward flow is:

```text
AllToAll-Dispatch → GMM1(up_proj) → SwiGLU → GMM2(down_proj) → AllToAll-Combine
```

Each operator is a separate NPU kernel, with substantial HBM traffic between operators as intermediate activations are written and read back.
With Expert Parallelism (EP), AllToAll communication and GMM computation usually execute serially, creating substantial idle time.

Measured results on Ascend A3 with a DeepSeek-V3-scale MoE workload:

| Metric | Measurement |
|---|---|
| Average Cube MAC utilization | ~54%, significantly below theoretical peak |
| Critical-path Vector execution time | ~18%, for operations such as SwiGLU while Cube cores are idle |
| AllToAll share of end-to-end time | ~17%, with 61% hidden by GEMM and 39% exposed |
| Mutual Cube/Vector idleness | The two core types alternate; only one type works at any instant |

Root cause: GroupedGEMM uses only Cube cores, leaving Vector cores idle. SwiGLU runs on Vector cores while Cube cores idle. Host-driven AllToAll leaves both AIC and AIV unable to compute during communication gaps.

### 1.2 Fusion Approach

Multicore MoE-FFN **fuses the five operators into one kernel**, executed concurrently by AIC (AI Cube) and AIV (AI Vector) cores:

- **AIV cores** handle AllToAll dispatch/combine and SwiGLU.
- **AIC cores** handle GMM1 and GMM2 matrix multiplication.
- **Shared memory and event semaphores** coordinate the pipeline across cores for **fine-grained compute-communication overlap**.

Key benefits:

- A single kernel reduces host launches and synchronization boundaries, overlapping communication, GMM, and Vector computation by tile.
- Deep compute-communication overlap improves hardware utilization.
- Applies to forward and backward for end-to-end training.

### 1.3 RATR：Rank-Aware Tile Reordering

In multi-device AllToAll, simultaneous sends from multiple ranks to one destination cause congestion and expose tail latency. Rank-Aware Tile Reordering (RATR) changes AllToAll tile order to distribute destination traffic over time and avoid converging bursts. It changes no computation: only AllToAll tiles in `vector_task_indexs` are reordered. RATR and static schedules are generated offline in RuntimeConfig, adding no runtime overhead.

Backward computation is fused similarly:

```text
AllToAll-Dispatch → ┌─ GMM1(act_grad) ──────────────────────┐
                    └─ GMM4(w2_grad)                          │
                                                              ▼
                                               SwiGLU-grad → GMM2(gate_grad) → ┌─ AllToAll-Combine
                                                                                └─ GMM3(w1_grad)
```

GMM1 runs concurrently with GMM4, and AllToAll-Combine with GMM3, further reducing idle time.

---

## 2. Directory Structure

### 2.1 Design Principles

The framework uses the following layers for the AIC/AIV hardware on Ascend A3/A2 NPUs:

| Layer | Python module | C++ module | Responsibility | Excludes |
|---|---|---|---|---|
| **Graph** | `scheduler/graph.py` | — | Operator topology, split rules, shape inference | Operator-specific parameters |
| **Runtime** | `scheduler/config.py` `scheduler/scheduler.py` | — | TaskDesc generation, event graphs, RATR optimization | Framework/compiler details |
| **Task** | `tasks/` | — | Per-operator FillConfig strategies | Core scheduling logic |
| **Module** | `modules/<op>/` | — | Concrete DAGs and CLI data generation | C++/framework code |
| **Scheduler** | — | `ops/runtime/` | Generic AIC/AIV event synchronization primitives | Compute kernel implementations |
| **Operator** | — | `ops/<op_name>/` | Operator-specific kernels and KernelWorker loops | Shared cross-operator logic |
| **Torch binding** | `torch/` | `torch/csrc/` | Torch operator registration | Graph and scheduling logic |

### 2.2 Core Runtime Directories

```text
multicore/
├── __init__.py
├── _loader.py                      # Locate payload, validate OPP, and load the framework adapter on first use.
│
├── scheduler/                       # Python compiler core, without operator specialization.
│   ├── __init__.py
│   ├── config.py                    # TaskSplitValue + init_task_split_value
│   │                                # All ctypes structs, including RuntimeConfigC.
│   ├── graph.py                     # ComputeGraph / OperatorNode / TensorSpec
│   │                                # SplitSpec / OpType / propagate_splits()
│   ├── builder.py                   # build_runtime_config(graph, tsv, rank_id)
│   ├── scheduler.py                 # revise_task_queue() / revise_gmm_task_queue_bwd()
│   └── allocator.py                 # Automatic param_position allocation (reserved).
│
├── tasks/                           # Task type registration and fill strategies.
│   ├── __init__.py
│   ├── task_base.py                 # FillConfig ABC: fill(cfg, op, tsv).
│   ├── alltoall.py                  # AllToAllFillConfig（dispatch / combine）
│   ├── gmm.py                       # GmmFillConfig variants.
│   ├── swiglu.py                    # SwiGLUFillConfig + SwiGLUGradFillConfig
│   ├── utils.py                     # advance_tsv_* / add_terminate / add_dynamic_data
│   ├── tiling.py                    # Global tiling byte registry.
│   └── registry.py                  # FILL_CONFIG_REGISTRY dictionary.
│
├── modules/                         # Operator modules: graph definitions and CLI data generation.
│   └── mega_moe/
│       ├── forward/
│       │   ├── graph.py             # build_forward_graph(tsv, ...) → ComputeGraph
│       │   ├── gen_runtime_data.py  # CLI: --tp/--ep/... generates *.bin files.
│       │   └── tiling_tables.py     # GMM/SwiGLU tiling parameter tables.
│       └── backward/
│           ├── graph.py             # build_backward_graph(tsv, ...) → ComputeGraph
│           ├── gen_runtime_data.py
│           └── tiling_tables.py
│
├── ops/                             # C++ CANN Ascend C implementations.
│   ├── runtime/                     # Shared layer copied into each op_kernel/runtime/ during builds.
│   │   ├── runtime_config.hpp       # Shared constants and structs: TaskDesc / EventDesc / TaskType.
│   │   └── worker_kernel.h          # KernelWorkerBase<Derived> CRTP base class.
│   │
│   ├── hyper_mega_moe/              # MoE FFN forward operator.
│   │   ├── op_host/                 # Host-side shape inference, tiling, and APIs.
│   │   └── op_kernel/               # Device side.
│   │       ├── hyper_mega_moe.cpp  # Kernel entry point including worker_kernel.cpp.
│   │       ├── hyper_mega_moe_tiling_key.h
│   │       ├── worker_kernel.cpp    # Forward KernelWorker specialization: tiling@[23], event@[24].
│   │       │                        # #include "runtime/worker_kernel.h"
│   │       ├── swi_glu/             # Isolated build tree only: copied from ops-nn.
│   │       ├── grouped_matmul/      # Isolated build tree only: copied from ops-transformer.
│   │       ├── put_mem_signal/
│   │       │   └── put_mem_signal_kernel.cpp  # Self-contained DTYPE_DISPATCH_TARGET instantiation.
│   │       └── runtime/             # Copied from ops/runtime/ by the source assembler.
│   │           ├── runtime_config.hpp
│   │           └── worker_kernel.h
│   │
│   └── hyper_mega_moe_grad/         # MoE FFN backward operator, structured as above.
│       ├── op_host/
│       └── op_kernel/
│           ├── hyper_mega_moe_grad.cpp
│           ├── hyper_mega_moe_grad_tiling_key.h
│           ├── worker_kernel.cpp    # Backward KernelWorker specialization: tiling@[30], event@[31].
│           ├── swi_glu/
│           ├── swi_glu_grad/
│           ├── grouped_matmul/
│           ├── put_mem_signal/
│           │   └── put_mem_signal_kernel.cpp  # Self-contained DTYPE_DY instantiation.
│           └── runtime/             # Copied during build.
│               ├── runtime_config.hpp
│               └── worker_kernel.h
│
├── torch/                          # CMakeLists.txt + csrc/ + ops.py
└── shmem/                          # Private Torch one-sided communication and build support.
```

### 2.3 Core Modules

#### scheduler/graph.py — Computation Graph

Three core abstractions:

- **TensorSpec**: Describes a tensor's shape, dtype, param_position, split_dim, is_dynamic, and transpose.
- **SplitSpec**: Declares operator splitting through split_inputs, task_num_fn, and split_output_dims.
- **ComputeGraph**: DAG manager whose `propagate_splits(tsv)` computes each node's task_num and split_dim in topological order.

#### scheduler/config.py — Configuration and Topology

`TaskSplitValue` encapsulates topology hyperparameters (tp, ep, seq_size, all_expert_num, top_k) and runtime counters.

`RuntimeConfigC` defines a 64-byte common header. `scheduler/runtime.py` creates a subclass per graph, appending task,
queue, and event arrays sized to actual capacity. Host and device share the same dense layout; TaskDesc is 576 bytes.
All graphs use this structure regardless of size, without legacy-format branches, magic, or version fields.

#### tasks/ — Task Fill Strategies

Each file implements one operator task type through `FillConfig.fill(cfg, op, tsv)`, writing task/event relationships into
`RuntimeConfigC`'s `all_tasks[]`, task index arrays, and `all_event_num_triggers[]`.

Registration in `registry.py`:

```python
FILL_CONFIG_REGISTRY: Dict[OpType, Type[FillConfig]] = {
    OpType.ALLTOALL:    AllToAllFillConfig,
    OpType.GMM:         GmmFillConfig,
    OpType.SWIGLU:      SwiGLUFillConfig,
    OpType.SWIGLU_GRAD: SwiGLUFillConfig,
}
```

Adding an operator requires only one dictionary entry, without changing core scheduling logic.

#### ops/runtime/ — Shared C++ Layer

| File | Responsibility |
|---|---|
| `runtime_config.hpp` | Shared structures: `RuntimeHeader`, `TaskDesc`, `EventDesc`, `TaskType`, and accessors |
| `worker_kernel.h` | `KernelWorkerBase<Derived>` CRTP base with `Process()`, `WaitForDependency()`, and `TriggerEvent()` |

**Key point**: `ops/runtime/` is not compiled directly. The [source assembler](../_build/assemble_multicore_source.py)
copies the complete `runtime/` directory into
`hyper_mega_moe/op_kernel/runtime/` and `hyper_mega_moe_grad/op_kernel/runtime/` in an isolated temporary source tree, allowing each operator's
`worker_kernel.cpp` to access shared definitions through `#include "runtime/worker_kernel.h"`. Neither repository business sources nor upstream dependencies
are modified in place.

#### ops/<op_name>/op_kernel/worker_kernel.cpp — Operator Scheduling Loop

Each operator retains its own `KernelWorker` because its hard-coded `input_list` indices differ:

| | mega_moe (forward) | mega_moe_grad (backward) |
|---|---|---|
| Tiling parameters | `input_list[23]` | `input_list[30]` |
| Event counters | `input_list[24]` | `input_list[31]` |
| GMM workspace | `input_list[11]` | `input_list[25]` |
| SwiGLU grad workspace | — | `input_list[26]` |
| Specific compute kernel | `TASK_SWI_GLU` | `TASK_SWI_GLU_GRAD` |

### 2.4 Data Flow Overview

```text
User (Python)
    ↓ build_forward_graph(tsv)
ComputeGraph [modules/mega_moe/forward/graph.py]
    ↓ propagate_splits(tsv)
Each OperatorNode's task_num / split_dim is populated automatically.
    ↓ build_runtime_config(graph, tsv, rank_id)
    ↓  → Polymorphic FillConfig.fill() calls.
    ↓  → RATR / GMM interleaving optimization.
RuntimeConfigC (ctypes) → Serialize to runtime_config_rank_*.bin.
                                      ↓
                          Read by C++ worker_kernel.
                                      ↓
                    KernelWorker::Process() main scheduling loop.
                    AIC (even blocks) → cube_task_indexs → GMM.
                    AIV (odd blocks) → vector_task_indexs → SwiGLU/AllToAll.
                    Event synchronization: WaitForDependency / TriggerEvent.
```

---

## 3. Forward Operator

### 3.1 Computation Graph

```text
dispatch ──► up_proj (GMM1) ──► swiglu ──► down_proj (GMM2) ──► combine
```

| Operator | Type | Core | Description |
|---|---|---|---|
| dispatch | AllToAll | AIV | Route tokens to ranks by expert |
| up_proj | GroupedMatmul | AIC | `[tokens, hidden] × [E, hidden, intermediate*2]` |
| swiglu | SwiGLU | AIV | `silu(gate) * up`, producing `[tokens, intermediate]` |
| down_proj | GroupedMatmul | AIC | `[tokens, intermediate] × [E, intermediate, hidden]` |
| combine | AllToAll | AIV | Gather expert outputs back to their original ranks |

### 3.2 Task Split Parameters

| Operator | split_value | Core | task_num calculation |
|---|---|---|---|
| dispatch | 128 | AIV | `all_expert_num × (per_expert_seq_to_other // 128)` |
| up_proj | 4096 | AIC | `num_cube_cores × single_rank_expert_num` |
| swiglu | 128 | AIV | `(per_expert_seq // 128) × single_rank_expert_num` |
| down_proj | 4096 | AIC | `num_cube_cores × single_rank_expert_num` |
| combine | 128 | AIV | `all_expert_num × (per_expert_seq_to_other // 128)` |

### 3.3 Tiling Files (Forward, Shared Across Ranks)

| File | Parameter position | Description |
|---|---|---|
| `up_proj_tiling.bin` | pos 17 | GMM1 tiling; size = `num_cube_cores × tiling_entry_size` |
| `swiglu_tiling.bin` | pos 18 | SwiGLU tiling; size = `(2×num_cube_cores+1) × entry_size` |
| `down_proj_tiling.bin` | pos 19 | GMM2 tiling |

See §6.1 for generation.

---

## 4. Backward Operator

### 4.1 Computation Graph

```text
dispatch ──► act_grad (GMM1) ──────────────────────────────────────────────────────────┐
         └── w2_grad (GMM4)                                                             │
                                                                                        ▼
                                                    swiglu_grad ──► gate_grad (GMM2) ──► combine
                                                                                     └── w1_grad (GMM3)
```

| Operator | Mapping | Inputs | Outputs |
|---|---|---|---|
| dispatch | AllToAll | `dy` (input gradient) | `dispatch_target` (dispatched gradient) |
| act_grad (GMM1) | `dispatch_target @ w2.T` | `dispatch_target`, `w2` | `act_grad_y` (activation gradient) |
| w2_grad (GMM4) | `hidden.T @ dispatch_target` | `hidden` (forward SwiGLU output), `dispatch_target` | `hidden_dw` (W2 gradient) |
| swiglu_grad | SwiGLU backward | `act_grad_y`, `gate` (forward up_proj_y) | `grad_gate` (SwiGLU gradient) |
| gate_grad (GMM2) | `grad_gate @ w1.T` | `grad_gate`, `w1` | `gate_dx` (pre-combine gradient) |
| w1_grad (GMM3) | `permute_out.T @ grad_gate` | `permute_out` (forward dispatch_target), `grad_gate` | `gate_dw` (W1 gradient) |
| combine | AllToAll | `gate_dx` | `grad_x` (combined gradient) |

### 4.2 Parallel Execution

- **act_grad (GMM1) and w2_grad (GMM4) overlap**: Both depend on dispatch output `target`. GMM1 uses `advance_mode="cube_only"` to advance task counts without events; GMM4 uses `advance_mode="cube_custom"` to advance events centrally and avoid double counting.

- **combine and w1_grad (GMM3) overlap**: Both depend on gate_grad (GMM2). Combine uses `advance_mode="vector_only"`; GMM3 uses `cube_custom` for events.

### 4.3 Tiling Files (Backward, Shared Across Ranks)

| File | Parameter name | Description |
|---|---|---|
| `act_grad_tiling.bin` | `act_grad_tiling` | GMM1 backward tiling (dispatch_target @ W2.T) |
| `gate_grad_tiling.bin` | `gate_grad_tiling` | GMM2 backward tiling (grad_gate @ W1.T) |
| `w1_grad_tiling.bin` | `w1_grad_tiling` | GMM3 W1-gradient tiling |
| `w2_grad_tiling.bin` | `w2_grad_tiling` | GMM4 W2-gradient tiling |
| `swiglu_grad_tiling.bin` | `swiglu_grad_tiling` | SwiGLU backward tiling |

See §6.2 for generation.

---

## 5. Task Splitting Design

### 5.1 TaskSplitValue

`TaskSplitValue` encapsulates hardware topology parameters and runtime counters:

```python
from hyper_parallel.core.multicore.scheduler.config import TaskSplitValue

tsv = TaskSplitValue(
    tp=4,              # Tensor parallel degree.
    ep=4,              # Expert parallel degree, equal to the NPU count.
    seq_size=8192,     # Global sequence length.
    all_expert_num=32, # Total expert count.
    top_k=8,           # Top-K routing count.
)
```

Key derived read-only properties:

| Property | Calculation | Meaning |
|---|---|---|
| `single_rank_expert_num` | `all_expert_num // ep` | Experts owned by each rank |
| `seq_all` | `seq_size × ep × top_k // tp` | Total tokens after TP partitioning and Top-K expansion |
| `per_rank_seq` | `seq_all // ep` | Tokens processed by each rank |
| `per_expert_seq` | `seq_all // top_k` | Tokens in one Top-K routing slice |
| `per_expert_seq_to_other` | `seq_all // (ep × top_k)` | Tokens per destination rank and Top-K slice |
| `all_event_num` | Computed automatically | Total global events |

### 5.2 propagate_splits Algorithm

`ComputeGraph.propagate_splits(tsv)` computes each operator's `task_num` in topological order:

```text
1. Reset all TensorSpec.split_dim = -1 and split_num = 1.

2. For each operator in topological order:
   a. If split_inputs is None (source operator), task_num = task_num_fn(tsv).
   b. Otherwise, if every (input_idx, dim) satisfies op.inputs[input_idx].split_dim == dim:
      → task_num = task_num_fn(tsv)  [split]
      Otherwise task_num = 1        [unsplit]
   c. Mark output tensors: out.split_dim = split_output_dims[i] when task_num > 1.
```

**Shared TensorSpec**: Producers write `split_dim` to a shared `TensorSpec` object and consumers read it directly, propagating split information through the graph automatically.

Forward propagation example:

```text
dispatch (source)  → target.split_dim = 0
up_proj   [(0,0)]  → target.split_dim == 0 ✓ → gmm1_y.split_dim = 0
swiglu    [(0,0)]  → gmm1_y.split_dim == 0 ✓ → swiglu_out.split_dim = 0
down_proj [(0,0)]  → swiglu_out.split_dim == 0 ✓ → gmm2_y.split_dim = 0
combine   [(1,0)]  → gmm2_y.split_dim == 0 ✓
```

### 5.3 Selecting split_inputs

`split_inputs` must select **inputs with upstream producers inside the graph**, not external leaf tensors. Leaf `split_dim` resets to -1 on every `propagate_splits` call, so using a leaf as the trigger leaves `task_num` permanently at 1.

```python
# Correct: inputs[1]=target is produced by dispatch and connected within the graph.
w2_grad: split_inputs=[(1, 0)]   # inputs[1] = target

# Incorrect: inputs[0]=hidden is an external leaf whose split_dim remains -1.
w2_grad: split_inputs=[(0, 0)]   # task_num always equals 1.
```

### 5.4 Event-Driven Coordination

Multicore MoE-FFN synchronizes cores through event semaphores:

**AllToAll dispatch**: Per-expert triggering.

```text
trigger_event = pre_event_num + (task_index // expert_group_size) + 1
```

Each expert group's completed dispatch task triggers an event. GMM consumes at expert granularity without waiting for all tokens to be sent.

**AllToAll combine**: Global triggering.

```text
trigger_event = all_event_num  (one globally unique event)
```

**Inter-GMM event coordination in backward:**

| Operator | advance_mode | Description |
|---|---|---|
| act_grad (GMM1) | `cube_only` | Advance task count only, without advancing events |
| w2_grad (GMM4) | `cube_custom` | Advance events centrally, accounting for GMM1 |
| combine | `vector_only` | Advance task count only |
| w1_grad (GMM3) | `cube_custom` | Advance events centrally |

### 5.5 Task Split Parameter Principles

| Operator type | Recommended split_dim | task_num_fn | Notes |
|---|---|---|---|
| GroupedMatMul (GMM) | 0 (M dimension, expert parallelism) | `expert_num × (seq_per_expert // TILE_M)` | TILE_M must align with GMM block_size |
| AllToAll (dispatch) | 0 (token dimension) | `total_tokens // token_tile` | Per-expert events for dispatch; global events for combine |
| AllToAll (combine) | 0 | `total_tokens // token_tile` | Depends on all down_proj tiles completing: N:1 fan-in |
| SwiGLU / Add | 0 | Upstream GMM task_num (1:1) | Must match upstream task_num to avoid event misalignment |

### 5.6 Event Design Rules

| Rule | Description |
|---|---|
| **Monotonicity** | Event counters only increase, preventing deadlock |
| **1:1 dependency** | Each completed producer tile triggers the corresponding consumer tile |
| **N:1 fan-in** | Multiple upstream tiles trigger one downstream tile; set `trigger_count = N` |
| **1:N fan-out** | One upstream triggers multiple downstream tiles, each with a separate event ID |
| **Nonoverlapping IDs** | Keep dispatch, combine, and GMM event ID ranges strictly separate |

---

## 6. RuntimeConfig Generation

`RuntimeConfig` is the binary Multicore MoE-FFN scheduling configuration, containing each task's dependency events, trigger events, and tensor address offsets. It is **generated independently per rank**.

The header's first five uint32 fields are task_num, num_workers, task_capacity, event_capacity, and ready_event.
A ready_event of 0 means no cross-rank readiness is required. The next three uint32 fields are diagnostic_mode,
aic_profile_record_capacity, and aiv_profile_record_capacity, followed by completion_event and protocol_version,
The final 24 bytes provide alignment. diagnostic_mode=0 means ordinary execution, =1 profiling, and =2 L1 dump.
protocol_version=0 means push and =1 means pull; pull's completion_event
waits for all participating workers, followed by cross-rank confirmation that source storage is safe to reuse.
Task capacity covers compute tasks, terminate tasks, queue length, and the largest referenced task ID, aligned to 16, without a fixed task-count ceiling.
Event capacity is at least 1024 and covers unfused graphs, ready/completion events, and atomic-write tails. Layout size is
`128 + 588 × task_capacity + 20 × event_capacity + group_list_bytes` bytes and must be addressable with uint32 byte offsets.
Online plans and offline tools share `serialize_runtime_config()`. Device code validates protocol versions, storage, and index bounds.
The ordinary event region occupies `event_capacity × 4` bytes; when EP > 1, it is followed by `2 × (EP + 1) × 64` bytes of persistent ready/completion state.
Each call clears only ordinary events. Persistent state is initialized once on first use, followed by a host barrier.

Grouped-matmul scratch for each AIC is 128-byte aligned, with per-core stride `ceil(local_experts × 8 / 128) × 128` bytes.
`group_list_bytes = max(4096, 24 × per_core_stride + 128)` includes base-alignment slack, preventing full-cache-line writeback from overwriting neighboring cores.
There is no longer a fixed limit of 16 experts per rank. Host allocation, serialization, and device offsets use actual expert counts.
Up to 16 experts retain the original layout size; larger counts expand scratch as needed, subject to uint32 offsets and available memory.

### 6.1 Forward

```bash
# Run from the hyper-parallel repository root, containing the hyper_parallel/ package.
# Direct python gen_runtime_data.py fails with ImportError because of relative imports.
python -m hyper_parallel.core.multicore.modules.mega_moe.forward.gen_runtime_data \
    --tp 4 --ep 4 \
    --seq_size 8192 \
    --all_expert_num 32 \
    --top_k 8 \
    --hidden_size 7168 \
    --intermediate_size 2048 \
    --num_cube_cores 24 \
    --output_dir /path/to/fwd_data
```

Output files:

```text
<output_dir>/
├── up_proj_tiling.bin              # GMM1 tiling, shared across ranks.
├── swiglu_tiling.bin               # SwiGLU tiling, shared across ranks.
├── down_proj_tiling.bin            # GMM2 tiling, shared across ranks.
├── all_event_counters.bin          # Ordinary event counters and persistent ready state; requires symmetric memory.
├── gmm_workspace.bin               # 256 MiB of zeros, shared across ranks.
├── runtime_config_input_rank_0.bin # Rank 0 scheduling configuration.
├── runtime_config_input_rank_1.bin
└── ...
```

You can also invoke it directly from Python:

```python
from hyper_parallel.core.multicore.modules.mega_moe.forward.gen_runtime_data import build_config_for_rank
from hyper_parallel.core.multicore.modules.mega_moe.forward.graph import build_forward_graph
from hyper_parallel.core.multicore.scheduler.config import TaskSplitValue
from hyper_parallel.core.multicore.scheduler.runtime import serialize_runtime_config

tsv   = TaskSplitValue(tp=4, ep=4, seq_size=8192, all_expert_num=32, top_k=8)
graph = build_forward_graph(tsv, dispatch_sv=128, up_proj_sv=4096,
                            swiglu_sv=128, down_proj_sv=4096, combine_sv=128,
                            hidden_size=7168, intermediate_size=2048)
graph.propagate_splits(tsv)
cfg = build_config_for_rank(graph, tsv, rank_id=0)   # Returns RuntimeConfigC.
data = serialize_runtime_config(cfg)
```

### 6.2 Backward

```bash
python -m hyper_parallel.core.multicore.modules.mega_moe.backward.gen_runtime_data \
    --tp 4 --ep 4 \
    --seq_size 8192 \
    --all_expert_num 32 \
    --top_k 8 \
    --hidden_size 7168 \
    --intermediate_size 2048 \
    --num_cube_cores 24 \
    --output_dir /path/to/bwd_data
```

Output files:

```text
<output_dir>/
├── act_grad_tiling.bin             # GMM1 backward tiling, shared across ranks.
├── gate_grad_tiling.bin            # GMM2 backward tiling.
├── w1_grad_tiling.bin              # GMM3 W1-gradient tiling.
├── w2_grad_tiling.bin              # GMM4 W2-gradient tiling.
├── swiglu_grad_tiling.bin          # SwiGLU backward tiling.
├── all_event_counters.bin          # 4096 uint8 zeros, 4 KB, shared across ranks.
├── gmm_workspace.bin
└── runtime_config_input_rank_<i>.bin
```

---

## 7. Build and Delivery Boundaries

The Multicore entry point prepares dependencies, private SHMEM, per-SoC vendors, vendor merging, and the Torch adapter. The repository root entry point only invokes
component scripts and packages successful artifacts into the complete wheel. Build commands, environment requirements, caches, and payload paths are documented in
[Build and Delivery](build.md). CANN custom OPP activation for source and installed-wheel environments is documented in the
[HyperParallel Installation Guide](../../../../docs/installation.md).

Each SoC's CANN build generates both `HyperMegaMoe` and `HyperMegaMoeGrad`. The component validates new and cached vendors,
then revalidates the four `aclnnHyperMegaMoe*` APIs, kernel objects/metadata, SONAMEs, and runtime dependencies after merging.
Private SHMEM libraries are checked for dedicated SONAMEs. All shipped host shared libraries are checked for ELF architecture, secure linking properties, absence of
RPATH/RUNPATH, and dependency boundaries. `set_env.bash` configures library search paths before Python starts.

The Torch adapter is a CMake-built dispatcher shared library loaded by `torch.ops.load_library()`. It has no
`PyInit_*` entry point and links neither `libpython` nor `libtorch_python`. At runtime it locates its bundled vendor relative to the component and loads it by absolute path.
`set_env.bash` configures vendor OPP and library search paths according to the CANN custom OPP contract.

---

## 8. Native ABI

SHMEM's `_lifecycle.py` manages native initialization, bootstrap, references, and caller-plan-driven runtime restart.
MegaMoe's `heap_manager.py` manages capacity, workspace layout, heap budgets, and growth diagnostics; `spec.py` supplies capacity formulas,
`workspace.py` manages physical buffers and graph leases, and execution-resource objects in `module.py` handle binding and closing.
The SHMEM lifecycle does not read expert capacity or workspace layouts. See the
[SHMEM documentation](shmem.md#runtime-restart) for configuration and restart rules.

The component [README](../README.md) owns usage, input constraints, capacity, and lifecycle documentation. This section records only
the forward/backward native ABI invoked internally by `MegaMoeExperts`.

### 8.1 `mega_moe` Forward ABI

```python
multicore_ops.mega_moe(
    # AllToAll dispatch
    dispatch_target,      # [tokens, hidden]          bf16   AllToAll destination buffer; in-place output.
    dispatch_target_off,  # [all_expert_num]           int64  Per-expert remote write offset on target PE.
    dispatch_src,         # [tokens, hidden]           bf16   Send source data.
    dispatch_src_off,     # [all_expert_num]           int64  Per-expert local read offset.
    dispatch_size,        # [all_expert_num]           int32  Elements sent per expert.
    # GMM1（up_proj）
    up_proj_weight,       # [E, hidden, intermediate*2] bf16  W1 weights.
    up_proj_glist,        # [E]                        int64  Per-expert cumulative token counts.
    up_proj_y,            # [tokens, intermediate*2]  bf16   GMM1 in-place output.
    swiglu_out,           # [tokens, intermediate]    bf16   SwiGLU in-place output.
    # GMM2（down_proj）
    down_proj_weight,     # [E, intermediate, hidden] bf16   W2 weights.
    down_proj_glist,      # [E]                       int64  Per-expert cumulative token counts.
    down_proj_y,          # [tokens, hidden]          bf16   GMM2 in-place output.
    # AllToAll combine
    combine_target,       # [tokens, hidden]          bf16   AllToAll combine destination; in-place output.
    combine_target_off,   # [all_expert_num]          int64  Per-expert remote write offset.
    combine_src_off,      # [all_expert_num]          int64  Per-expert local read offset.
    combine_size,         # [all_expert_num]          int32  Elements sent per expert.
    # Configuration tensors.
    gmm_workspace,        # [256*1024*1024]           uint8  GMM workspace, 256 MiB.
    up_proj_tiling,       # uint8                     GMM1 tiling，from gen_runtime_data.py
    swiglu_tiling,        # uint8                     SwiGLU tiling
    down_proj_tiling,     # uint8                     GMM2 tiling
    runtime_config,       # uint8                     Per-rank schedule from gen_runtime_data.py.
    all_event_counters,   # uint8                       Event synchronization counters in symmetric memory.
    # Scalars.
    rank_id,              # int                       Current rank.
    ep,                   # int                       Expert parallel degree.
    expert_num,           # int                       Global expert count (all_expert_num).
    hidden_size,          # int                       Hidden dimension.
    seq_size,             # int                       Global sequence length.
)
```

**Returns**: Nothing; outputs are written into caller-provided Torch tensors.

### 8.2 `mega_moe_grad` Backward ABI

```python
multicore_ops.mega_moe_grad(
    # AllToAll dispatch for gradients.
    dispatch_target,      # [tokens, hidden]           bf16   AllToAll destination; in-place output.
    dispatch_target_off,  # [all_expert_num]            int64  Per-expert remote write offset.
    dy,                   # [tokens, hidden]            bf16   Input gradient, AllToAll send source.
    dispatch_src_off,     # [all_expert_num]            int64  Per-expert local read offset.
    dispatch_size,        # [all_expert_num]            int32  Elements sent per expert.
    # act_grad: GMM1 backward, dispatch_target @ W2.T.
    hidden,               # [tokens, intermediate]      bf16   Cached forward SwiGLU output.
    hidden_dw,            # [E, intermediate, hidden]   bf16   W2 gradient; in-place output.
    w2,                   # [E, intermediate, hidden]   bf16   W2 weights, forward down_proj_weight.
    act_grad_y,           # [tokens, intermediate]      bf16   GMM1 backward in-place output.
    # swiglu_grad
    gate,                 # [tokens, intermediate*2]   bf16   Forward up_proj_y, cached SwiGLU input.
    grad_gate,            # [tokens, intermediate*2]   bf16   SwiGLU gradient in-place output.
    # gate_grad: GMM2 backward, grad_gate @ W1.T.
    w1,                   # [E, hidden, intermediate*2] bf16  W1 weights, forward up_proj_weight.
    gate_dx,              # [tokens, hidden]            bf16   GMM2 backward in-place output.
    grad_x,               # [tokens, hidden]            bf16   AllToAll combine in-place output.
    # AllToAll combine for gradients.
    combine_target_off,   # [all_expert_num]            int64  Per-expert remote write offset.
    combine_src_off,      # [all_expert_num]            int64  Per-expert local read offset.
    combine_size,         # [all_expert_num]            int32  Elements sent per expert.
    # w1_grad（GMM3）、w2_grad（GMM4）
    permute_out,          # [tokens, hidden]            bf16   In-place intermediate buffer for W1 gradients.
    gate_dw,              # [E, hidden, intermediate*2] bf16  W1 gradient; in-place output.
    group_list,           # [E]                         int64  Per-expert cumulative token counts.
    # Configuration tensors.
    act_grad_tiling,      # uint8                       GMM1 backward tiling.
    gate_grad_tiling,     # uint8                       GMM2 backward tiling.
    w1_grad_tiling,       # uint8                       GMM3 W1-gradient tiling.
    w2_grad_tiling,       # uint8                       GMM4 W2-gradient tiling.
    swiglu_grad_tiling,   # uint8                       SwiGLU backward tiling.
    gmm_workspace,        # [256*1024*1024]             uint8  GMM workspace.
    swiglu_grad_workspace, # [64*1024*1024]             uint8  SwiGLU backward workspace.
    runtime_config,       # uint8                       Per-rank scheduling configuration.
    all_event_counters,   # uint8                       Event synchronization counters in symmetric memory.
    # Scalars.
    rank_id, ep, expert_num, hidden_size, seq_size,
)
```

---

## 9. Extension Guide

### 9.1 Add an Operator Task Type (Python)

**Step 1** — Extend `OpType` in `scheduler/graph.py`:

```python
class OpType(Enum):
    ALLTOALL    = "alltoall"
    GMM         = "gmm"
    SWIGLU      = "swiglu"
    SWIGLU_GRAD = "swiglu_grad"
    RMSNORM     = "rmsnorm"   # New entry.
```

**Step 2** — Add matching `TaskType` entries in `scheduler/config.py` and C++ `runtime_config.hpp`, then
implement `FillConfig` in `tasks/rmsnorm.py`. Use [SwiGLUFillConfig](../tasks/swiglu.py) as the Vector task reference:

- Construct task and input/output descriptions with `TaskDescC` and `TensorDescC`.
- Write tasks into `cfg.all_tasks` and indices into the corresponding `cube_task_indices`, `vector_task_indices`, or
  `mix_task_indices`.
- Set `dependent_event`, `trigger_event`, and `all_event_num_triggers`, then update `cfg.task_index_num`.
- Advance task and event counts through the `advance_tsv_*` helpers in `tasks/utils.py`.

**Step 3** — Register in `tasks/registry.py`:

```python
from .rmsnorm import RMSNormFillConfig
FILL_CONFIG_REGISTRY[OpType.RMSNORM] = RMSNormFillConfig
```

**Step 4** — Add a switch branch in C++ `ops/<op>/op_kernel/worker_kernel.cpp`:

```cpp
case TaskType::TASK_RMSNORM:
    ExecuteRMSNorm(task_desc);
    break;
```

### 9.2 Add a Model (Operator Combinations Beyond MoE FFN)

**Step 1** — Create the directory:

```text
modules/
└── attention/
    ├── forward/
    │   ├── graph.py
    │   ├── gen_runtime_data.py
    │   └── tiling_tables.py
    └── backward/
        └── ...
```

**Step 2** — Build the DAG in `graph.py`:

```python
from hyper_parallel.core.multicore.scheduler.graph import (
    ComputeGraph, OperatorNode, TensorSpec, SplitSpec, OpType
)
from hyper_parallel.core.multicore.scheduler.config import TaskSplitValue

def build_attention_fwd_graph(tsv: TaskSplitValue, ...) -> ComputeGraph:
    g = ComputeGraph()
    x       = TensorSpec("x",       shape=[tsv.seq_size, hidden], param_position=0)
    qkv_out = TensorSpec("qkv_out", shape=[tsv.seq_size, 3*hidden], param_position=1,
                         is_dynamic=True)
    qkv_proj = OperatorNode("qkv_proj", op_type=OpType.GMM,
                            inputs=[x], outputs=[qkv_out],
                            split_spec=SplitSpec(
                                split_inputs=None,
                                task_num_fn=lambda tsv: tsv.seq_size // TILE_M,
                                split_output_dims=[0]))
    g.add_op(qkv_proj)
    return g
```

**Step 3** — Implement the CLI, following `modules/mega_moe/forward/gen_runtime_data.py`:

```python
def main():
    args = parse_args()
    tsv = TaskSplitValue(tp=args.tp, ep=args.ep, ...)
    graph = build_attention_fwd_graph(tsv)
    graph.propagate_splits(tsv)
    for rank_id in range(args.tp * args.ep):
        cfg = build_config_for_rank(graph, tsv, rank_id)
        write_bin(cfg, f"{args.output_dir}/runtime_config_rank_{rank_id}.bin")
```

### 9.3 Add a C++ Compute Kernel (Ascend C)

**Step 1** — Create the operator directory:

```text
ops/
└── attn_fwd/
    ├── op_host/
    │   ├── attn_fwd_def.cpp
    │   ├── attn_fwd_infershape.cpp
    │   └── attn_fwd_tiling.cpp/h
    └── op_kernel/
        ├── attn_fwd.cpp
        ├── tiling_key.h
        ├── worker_kernel.cpp
        └── put_mem_signal/
            └── put_mem_signal_kernel.cpp
```

**Step 2** — Include the shared scheduler in `attn_fwd.cpp`:

```cpp
#include "kernel_operator.h"
#include "worker_kernel.cpp"
#include "tiling_key.h"

__global__ __aicore__ void multicore_attn_fwd(...) {
    GM_ADDR input_list[] = { ..., runtime_config, ..., tiling, all_event_counters };
    uint32_t idx = GetBlockIdx();
    worker_kernel(idx, runtime_config, input_list);
}
```

**Step 3** — Include the shared runtime in `worker_kernel.cpp`:

```cpp
#include "kernel_operator.h"
#include "runtime/worker_kernel.h"  // The build copies ops/runtime/ into op_kernel/runtime/.

class KernelWorker : public KernelWorkerBase<KernelWorker> {
 public:
    // input_list indices follow this operator's parameter layout.
    static constexpr uint32_t TILING_IDX = N;   // Replace with the actual index.
    static constexpr uint32_t EVENT_IDX  = M;

    __aicore__ inline void ExecuteComputeKernel(TaskDesc task_desc) {
        switch (task_desc.task_type) {
            // Add this operator's cases.
        }
    }
};
```

**Step 4** — Instantiate `put_mem_signal/put_mem_signal_kernel.cpp` with the operator's dtype:

```cpp
#include "kernel_operator.h"
#include "shmem.h"

template <typename T>
class PutMemSignalKernel { /* ... */ };  // Same as the existing implementation.

extern "C" inline __aicore__ void put_mem_signal_kernel(...) {
    PutMemSignalKernel<DTYPE_YOUR_TYPE> op;  // Replace with the operator's dtype macro.
    op.Init(...);
    op.Process();
}
```

**Step 5** — Register the operator in the [source assembler](../_build/assemble_multicore_source.py) so source assembly copies the shared
`runtime/`：

```python
_HYPER_OPERATORS = ("hyper_mega_moe", "hyper_mega_moe_grad", "hyper_attn_fwd")
```

---

## 10. Appendix: Common Errors and Troubleshooting

| Symptom | Possible cause | Investigation |
|---|---|---|
| `TaskDesc` event_id out of bounds | Incorrect all_event_num calculation | Check the `all_event_num` property formula in `scheduler/config.py` |
| Different task_num in two bin files | Incorrect SplitSpec task_num_fn | Print each operator's task_num after `propagate_splits()` |
| AIV hang (infinite wait) | N:1 fan-in trigger_count is too large | Verify FillConfig trigger_count equals the actual upstream tile count |
| GMM workspace out of bounds | TILE_M/N too large | Check workspace budgeting in `tiling_tables.py` |
| Missing `runtime/worker_kernel.h` | New operator absent from isolated source assembly | Check the operator list in `assemble_multicore_source.py` and `MULTICORE_SOURCE_ASSEMBLY_FAILED` logs |
| Python `ImportError: scheduler` | Running outside the package root | Run with `python -m` from the `hyper-parallel/` root |
