# MoE-FFN Multicore Operator

---

## 1. 背景与融合思路

### 1.1 问题

标准 MoE-FFN 正向计算流程为：

```text
AllToAll-Dispatch → GMM1(up_proj) → SwiGLU → GMM2(down_proj) → AllToAll-Combine
```

每个算子是独立的 NPU kernel，算子之间存在大量 HBM 读写（激活值中间结果落盘再读回）。
在 Expert Parallel（EP）场景下，AllToAll 通信与 GMM 计算通常串行执行，造成大量空闲等待。

实测数据（昇腾 A3，DeepSeek-V3 规模 MoE 负载）：

| 指标 | 测量值 |
|---|---|
| Cube 核平均 MAC 利用率 | ~54%（离理论峰值差距显著） |
| 关键路径 Vector 核执行时间 | ~18%（SwiGLU 等，此时 Cube 全闲） |
| AllToAll 通信占端到端时间 | ~17%（其中 61% 被 GEMM 掩盖，39% 暴露） |
| Cube/Vector 相互空闲 | 两类核轮流空转，任一时刻只有一类核工作 |

根因分析：GroupedGEMM 只使用 Cube 核，Vector 核在 GMM 执行期间全部空闲；SwiGLU 由 Vector 核执行时 Cube 全部空转；AllToAll 由 HOST 驱动，通信空窗期设备内 AIC 和 AIV 也无法计算。

### 1.2 融合方案

Multicore MoE-FFN 将上述五个算子**融合为一个 kernel**，由 AIC（AI Cube）和 AIV（AI Vector）核同时执行：

- **AIV 核**负责 AllToAll 通信（dispatch / combine）和 SwiGLU
- **AIC 核**负责 GMM1、GMM2（矩阵乘）
- 核间通过**共享内存 + 事件信号量**流水线协调，实现通信与计算的**细粒度重叠**

关键收益：

- 消除 AllToAll 与 GMM 之间的 HBM 中间激活读写
- 通信与计算深度重叠，提升硬件利用率
- 正向、反向均适用，支持端到端训练

### 1.3 RATR：Rank-Aware Tile Reordering

在多卡 AllToAll 场景下，各 Rank 同时向同一目标 Rank 发送数据会造成网络拥塞，导致尾延迟暴露。RATR 通过调整 AllToAll Tile 的执行顺序，使发往不同 Rank 的通信流量在时间轴上均匀分散，避免多源 Rank 同时涌向同一目标。RATR 不修改任何计算逻辑，仅对 `vector_task_indexs` 中 AllToAll Tile 重排，与静态调度配置一起在 RuntimeConfig 中离线生成，运行时零开销。

反向计算流程（同理融合）：

```text
AllToAll-Dispatch → ┌─ GMM1(act_grad) ──────────────────────┐
                    └─ GMM4(w2_grad)                          │
                                                              ▼
                                               SwiGLU-grad → GMM2(gate_grad) → ┌─ AllToAll-Combine
                                                                                └─ GMM3(w1_grad)
```

GMM1 与 GMM4 并行，AllToAll-Combine 与 GMM3 并行，进一步减少空闲等待。

---

## 2. 目录结构

### 2.1 设计原则

本框架针对昇腾 A3/A2 NPU 的 AIC/AIV 双路硬件特点制定如下分层规则：

| 层次 | Python 模块 | C++ 模块 | 职责 | 不包含 |
|---|---|---|---|---|
| **图层** | `scheduler/graph.py` | — | 算子拓扑、切分规则、形状推导 | 任何算子特定参数 |
| **运行时层** | `scheduler/config.py` `scheduler/scheduler.py` | — | TaskDesc 生成、事件图、RATR 优化 | 框架 / 编译细节 |
| **任务层** | `tasks/` | — | 每类算子的 FillConfig（填充策略） | 调度核心逻辑 |
| **模块层** | `modules/<op>/` | — | 具体 DAG + CLI 数据生成 | C++ / 框架代码 |
| **调度器层** | — | `ops/runtime/` | 通用 AIC/AIV 事件同步原语 | 计算核实现 |
| **算子层** | — | `ops/<op_name>/` | 算子特有计算核 + KernelWorker 调度循环 | 跨算子共享逻辑 |
| **适配层** | `platform/` | `platform/*/c_api/` | MindSpore/PyTorch 接口 | 图与调度逻辑 |

### 2.2 完整目录树

```text
multicore/
├── __init__.py
├── _loader.py                      # 首次调用时定位 payload、校验 OPP 并加载框架 adapter
│
├── scheduler/                       # Python 编译器核心（无算子特化）
│   ├── __init__.py
│   ├── config.py                    # TaskSplitValue + init_task_split_value
│   │                                # + 全部 ctypes structs（RuntimeConfigC 等）
│   ├── graph.py                     # ComputeGraph / OperatorNode / TensorSpec
│   │                                # SplitSpec / OpType / propagate_splits()
│   ├── builder.py                   # build_runtime_config(graph, tsv, rank_id)
│   ├── scheduler.py                 # revise_task_queue() / revise_gmm_task_queue_bwd()
│   └── allocator.py                 # param_position 自动分配（预留）
│
├── tasks/                           # 任务类型注册与填充策略
│   ├── __init__.py
│   ├── task_base.py                 # FillConfig (ABC): fill(cfg, op, tsv) 接口
│   ├── alltoall.py                  # AllToAllFillConfig（dispatch / combine）
│   ├── gmm.py                       # GmmFillConfig（各变体）
│   ├── swiglu.py                    # SwiGLUFillConfig + SwiGLUGradFillConfig
│   ├── utils.py                     # advance_tsv_* / add_terminate / add_dynamic_data
│   ├── tiling.py                    # 全局 tiling 字节注册表
│   └── registry.py                  # FILL_CONFIG_REGISTRY 字典
│
├── modules/                         # 算子模块层（图定义 + CLI 数据生成）
│   └── mega_moe/
│       ├── forward/
│       │   ├── graph.py             # build_forward_graph(tsv, ...) → ComputeGraph
│       │   ├── gen_runtime_data.py  # CLI: --tp/--ep/... 生成 *.bin 文件
│       │   └── tiling_tables.py     # GMM/SwiGLU tiling 参数表
│       └── backward/
│           ├── graph.py             # build_backward_graph(tsv, ...) → ComputeGraph
│           ├── gen_runtime_data.py
│           └── tiling_tables.py
│
├── ops/                             # C++ CANN AscendC 算子实现
│   ├── runtime/                     # ⭐ 通用共享层（build 时拷入各 op_kernel/runtime/）
│   │   ├── runtime_config.hpp       # 共享常量 + struct 定义（TaskDesc / EventDesc / TaskType 等）
│   │   └── worker_kernel.h          # KernelWorkerBase<Derived> CRTP 基类
│   │
│   ├── hyper_mega_moe/              # MoE FFN 正向算子
│   │   ├── op_host/                 # 主机侧：infershape / tiling / API
│   │   └── op_kernel/               # 设备侧
│   │       ├── hyper_mega_moe.cpp  # kernel 入口（include worker_kernel.cpp）
│   │       ├── hyper_mega_moe_tiling_key.h
│   │       ├── worker_kernel.cpp    # KernelWorker（fwd 特化：tiling@[23], event@[24]）
│   │       │                        # #include "runtime/worker_kernel.h"
│   │       ├── swi_glu/             # 仅隔离构建树：从 ops-nn 拷入
│   │       ├── grouped_matmul/      # 仅隔离构建树：从 ops-transformer 拷入
│   │       ├── put_mem_signal/
│   │       │   └── put_mem_signal_kernel.cpp  # 自包含；DTYPE_DISPATCH_TARGET 实例化
│   │       └── runtime/             # 构建时由源码组装器从 ops/runtime/ 拷入
│   │           ├── runtime_config.hpp
│   │           └── worker_kernel.h
│   │
│   └── hyper_mega_moe_grad/         # MoE FFN 反向算子（结构同上）
│       ├── op_host/
│       └── op_kernel/
│           ├── hyper_mega_moe_grad.cpp
│           ├── hyper_mega_moe_grad_tiling_key.h
│           ├── worker_kernel.cpp    # KernelWorker（bwd 特化：tiling@[30], event@[31]）
│           ├── swi_glu/
│           ├── swi_glu_grad/
│           ├── grouped_matmul/
│           ├── put_mem_signal/
│           │   └── put_mem_signal_kernel.cpp  # 自包含；DTYPE_DY 实例化
│           └── runtime/             # build 时拷入
│               ├── runtime_config.hpp
│               └── worker_kernel.h
│
├── platform/                        # 框架适配层
│   ├── mindspore/                   # CMakeLists.txt + c_api/ + framework/
│   └── torch/                       # setup.py + csrc/
```

### 2.3 核心模块说明

#### scheduler/graph.py — 计算图层

提供三个核心抽象：

- **TensorSpec**：描述一个张量（shape、dtype、param_position、split_dim、is_dynamic、transpose）
- **SplitSpec**：声明该算子如何切分（split_inputs、task_num_fn、split_output_dims）
- **ComputeGraph**：DAG 管理器，`propagate_splits(tsv)` 按拓扑序自动计算每个节点的 task_num 和 split_dim

#### scheduler/config.py — 配置与拓扑参数

`TaskSplitValue` 封装拓扑超参数（tp、ep、seq_size、all_expert_num、top_k）和运行时计数器。

`RuntimeConfigC` 是连接 Python 和 C++ 的数据桥梁，与 `ops/runtime/runtime_config.hpp` 中的 struct 定义一一对应（ctypes ↔ AscendC struct）。

#### tasks/ — 任务填充策略层

每个文件对应一类算子任务类型，实现 `FillConfig.fill(cfg, op, tsv)` 接口，向 `RuntimeConfigC` 的 `task_descs[]` 和 `event_descs[]` 写入 TaskDesc / EventDesc。

注册机制（`registry.py`）：

```python
FILL_CONFIG_REGISTRY: Dict[OpType, Type[FillConfig]] = {
    OpType.ALLTOALL:    AllToAllFillConfig,
    OpType.GMM:         GmmFillConfig,
    OpType.SWIGLU:      SwiGLUFillConfig,
    OpType.SWIGLU_GRAD: SwiGLUFillConfig,
}
```

新增算子只需在此字典中添加一行，无需修改核心调度逻辑。

#### ops/runtime/ — C++ 通用共享层

| 文件 | 职责 |
|---|---|
| `runtime_config.hpp` | 共享数据结构：`MAX_TASK_NUM`、`TaskDesc`、`EventDesc`、`TaskType` 枚举、所有 accessor 函数 |
| `worker_kernel.h` | `KernelWorkerBase<Derived>` CRTP 基类，含 `Process()`、`WaitForDependency()`、`TriggerEvent()` |

**关键**：`ops/runtime/` 不会直接参与编译。`scripts/native/assemble_multicore_source.py` 在隔离的临时源码树中，将整个 `runtime/` 目录分别拷贝到 `hyper_mega_moe/op_kernel/runtime/` 和 `hyper_mega_moe_grad/op_kernel/runtime/`，使各算子 `worker_kernel.cpp` 可以用 `#include "runtime/worker_kernel.h"` 访问共享定义。仓内业务源码和上游依赖源码均不会被原地修改。

#### ops/<op_name>/op_kernel/worker_kernel.cpp — 算子调度循环

每个算子保留自己的 `KernelWorker` 类，原因是 `input_list` 索引硬编码不同：

| | mega_moe（正向）| mega_moe_grad（反向）|
|---|---|---|
| tiling 参数 | `input_list[23]` | `input_list[30]` |
| event 计数器 | `input_list[24]` | `input_list[31]` |
| GMM workspace | `input_list[11]` | `input_list[25]` |
| SwiGLU grad workspace | — | `input_list[26]` |
| 特有计算核 | `TASK_SWI_GLU` | `TASK_SWI_GLU_GRAD` |

### 2.4 数据流总览

```text
用户 (Python)
    ↓ build_forward_graph(tsv)
ComputeGraph [modules/mega_moe/forward/graph.py]
    ↓ propagate_splits(tsv)
各 OperatorNode 的 task_num / split_dim 被自动填充
    ↓ build_runtime_config(graph, tsv, rank_id)
    ↓  → 多态调用各 FillConfig.fill()
    ↓  → RATR / GMM 交错优化
RuntimeConfigC (ctypes) → 序列化为 runtime_config_rank_*.bin
                                      ↓
                          C++ worker_kernel 读取
                                      ↓
                    KernelWorker::Process() 主调度循环
                    AIC（偶数 block）→ cube_task_indexs → GMM
                    AIV（奇数 block）→ vector_task_indexs → SwiGLU/AllToAll
                    事件同步：WaitForDependency / TriggerEvent
```

---

## 3. 正向算子

### 3.1 计算流图

```text
dispatch ──► up_proj (GMM1) ──► swiglu ──► down_proj (GMM2) ──► combine
```

| 算子 | 类型 | 执行核 | 说明 |
|---|---|---|---|
| dispatch | AllToAll | AIV | 将 token 按 expert 路由分发到各 rank |
| up_proj | GroupedMatmul | AIC | `[tokens, hidden] × [E, hidden, intermediate*2]` |
| swiglu | SwiGLU | AIV | `silu(gate) * up`，输出 `[tokens, intermediate]` |
| down_proj | GroupedMatmul | AIC | `[tokens, intermediate] × [E, intermediate, hidden]` |
| combine | AllToAll | AIV | 将 expert 结果汇聚回原始 rank |

### 3.2 Task 切分参数

| 算子 | split_value | 执行核 | task_num 计算 |
|---|---|---|---|
| dispatch | 128 | AIV | `all_expert_num × (per_expert_seq_to_other // 128)` |
| up_proj | 4096 | AIC | `num_cube_cores × single_rank_expert_num` |
| swiglu | 128 | AIV | `(per_expert_seq // 128) × single_rank_expert_num` |
| down_proj | 4096 | AIC | `num_cube_cores × single_rank_expert_num` |
| combine | 128 | AIV | `all_expert_num × (per_expert_seq_to_other // 128)` |

### 3.3 Tiling 文件（正向，各 rank 共用）

| 文件 | 参数位置 | 说明 |
|---|---|---|
| `up_proj_tiling.bin` | pos 17 | GMM1 tiling，大小 = `num_cube_cores × tiling_entry_size` |
| `swiglu_tiling.bin` | pos 18 | SwiGLU tiling，大小 = `(2×num_cube_cores+1) × entry_size` |
| `down_proj_tiling.bin` | pos 19 | GMM2 tiling |

生成方式见第 6.1 节。

---

## 4. 反向算子

### 4.1 计算流图

```text
dispatch ──► act_grad (GMM1) ──────────────────────────────────────────────────────────┐
         └── w2_grad (GMM4)                                                             │
                                                                                        ▼
                                                    swiglu_grad ──► gate_grad (GMM2) ──► combine
                                                                                     └── w1_grad (GMM3)
```

| 算子 | 映射 | 输入 | 输出 |
|---|---|---|---|
| dispatch | AllToAll | `dy`（输入梯度） | `dispatch_target`（分发后梯度） |
| act_grad (GMM1) | `dispatch_target @ w2.T` | `dispatch_target`, `w2` | `act_grad_y`（激活梯度） |
| w2_grad (GMM4) | `hidden.T @ dispatch_target` | `hidden`（正向 SwiGLU 输出）, `dispatch_target` | `hidden_dw`（W2 权重梯度） |
| swiglu_grad | SwiGLU backward | `act_grad_y`, `gate`（正向 up_proj_y） | `grad_gate`（SwiGLU 梯度） |
| gate_grad (GMM2) | `grad_gate @ w1.T` | `grad_gate`, `w1` | `gate_dx`（combine 前梯度） |
| w1_grad (GMM3) | `permute_out.T @ grad_gate` | `permute_out`（正向 dispatch_target）, `grad_gate` | `gate_dw`（W1 权重梯度） |
| combine | AllToAll | `gate_dx` | `grad_x`（汇聚后梯度） |

### 4.2 并行执行说明

- **act_grad（GMM1）与 w2_grad（GMM4）并行**：两者均依赖 dispatch 输出 `target`；GMM1 使用 `advance_mode="cube_only"`（只推进 task 计数，不推进 event），GMM4 使用 `advance_mode="cube_custom"` 统一推进 event，避免重复计数。

- **combine 与 w1_grad（GMM3）并行**：两者均依赖 gate_grad（GMM2）输出；combine 使用 `advance_mode="vector_only"`，GMM3 使用 `cube_custom` 处理 event。

### 4.3 Tiling 文件（反向，各 rank 共用）

| 文件 | 参数名 | 说明 |
|---|---|---|
| `act_grad_tiling.bin` | `act_grad_tiling` | GMM1 反向 tiling（dispatch_target @ W2.T） |
| `gate_grad_tiling.bin` | `gate_grad_tiling` | GMM2 反向 tiling（grad_gate @ W1.T） |
| `w1_grad_tiling.bin` | `w1_grad_tiling` | GMM3（W1 权重梯度）tiling |
| `w2_grad_tiling.bin` | `w2_grad_tiling` | GMM4（W2 权重梯度）tiling |
| `swiglu_grad_tiling.bin` | `swiglu_grad_tiling` | SwiGLU 反向 tiling |

生成方式见第 6.2 节。

---

## 5. Task 切分设计

### 5.1 TaskSplitValue

`TaskSplitValue` 封装硬件拓扑参数和运行时计数器：

```python
from hyper_parallel.core.multicore.scheduler.config import TaskSplitValue

tsv = TaskSplitValue(
    tp=4,              # Tensor Parallel 度
    ep=4,              # Expert Parallel 度（等于 NPU 卡数）
    seq_size=8192,     # 全局序列长度
    all_expert_num=32, # 总 expert 数
    top_k=8,           # Top-K 路由数
)
```

关键推导属性（只读 property）：

| 属性 | 计算 | 含义 |
|---|---|---|
| `single_rank_expert_num` | `all_expert_num // ep` | 每个 rank 持有的 expert 数 |
| `per_rank_seq` | `seq_size × ep × top_k // tp` | 每个 rank 处理的 token 数 |
| `per_expert_seq` | `per_rank_seq // top_k` | 每个 expert 的 token 数 |
| `per_expert_seq_to_other` | `seq_size × top_k // all_expert_num` | AllToAll 每对 rank 的 token 数 |
| `all_event_num` | 自动计算 | 全局 event 总数 |

### 5.2 propagate_splits 算法

`ComputeGraph.propagate_splits(tsv)` 按拓扑顺序自动计算每个算子的 `task_num`：

```text
1. 重置所有 TensorSpec.split_dim = -1, split_num = 1

2. 对每个算子（拓扑顺序）：
   a. 若 split_inputs is None（源算子）→ task_num = task_num_fn(tsv)
   b. 否则，若所有 (input_idx, dim) 均满足 op.inputs[input_idx].split_dim == dim
      → task_num = task_num_fn(tsv)  [切分]
      否则 task_num = 1              [不切分]
   c. 标记输出张量：out.split_dim = split_output_dims[i]（task_num > 1 时）
```

**TensorSpec 共享机制**：同一个 `TensorSpec` 对象被生产者写入 `split_dim`，消费者直接读取，切分信息自动沿图传播，无需手工传递。

正向传播示例：

```text
dispatch (源算子)  → target.split_dim = 0
up_proj   [(0,0)]  → target.split_dim == 0 ✓ → gmm1_y.split_dim = 0
swiglu    [(0,0)]  → gmm1_y.split_dim == 0 ✓ → swiglu_out.split_dim = 0
down_proj [(0,0)]  → swiglu_out.split_dim == 0 ✓ → gmm2_y.split_dim = 0
combine   [(1,0)]  → gmm2_y.split_dim == 0 ✓
```

### 5.3 split_inputs 选择原则

`split_inputs` 中必须选择**图内有上游生产者的输入**（非外部叶子张量）。叶子张量的 `split_dim` 在每次 `propagate_splits` 时重置为 -1，以叶子为触发条件将导致 `task_num` 始终为 1。

```python
# ✅ 正确：inputs[1]=target 是 dispatch 的输出，图内连通
w2_grad: split_inputs=[(1, 0)]   # inputs[1] = target

# ❌ 错误：inputs[0]=hidden（外部叶子），split_dim 永远 -1
w2_grad: split_inputs=[(0, 0)]   # task_num 始终 = 1
```

### 5.4 Event 驱动协调

Multicore MoE-FFN 中各核通过事件信号量（event）同步：

**AllToAll dispatch**：per-expert 触发

```text
trigger_event = pre_event_num + (task_index // expert_group_size) + 1
```

每个 expert group 的 dispatch task 完成后触发一个 event，GMM 以 expert 粒度消费，无需等待全部 token 发送完毕。

**AllToAll combine**：global 触发

```text
trigger_event = all_event_num  （全局唯一 event）
```

**GMM 间 event 协调**（反向）：

| 算子 | advance_mode | 说明 |
|---|---|---|
| act_grad (GMM1) | `cube_only` | 只推进 task 计数，不推进 event |
| w2_grad (GMM4) | `cube_custom` | 统一推进 event（补偿 GMM1 未推进的部分） |
| combine | `vector_only` | 只推进 task 计数 |
| w1_grad (GMM3) | `cube_custom` | 统一推进 event |

### 5.5 Task 切分参数设计原则

| 算子类型 | 推荐 split_dim | task_num_fn | 注意事项 |
|---|---|---|---|
| GroupedMatMul（GMM）| 0（M 维，expert 并行）| `expert_num × (seq_per_expert // TILE_M)` | TILE_M 必须对齐 GMM block_size |
| AllToAll（dispatch）| 0（token 维）| `total_tokens // token_tile` | event 范围：per-expert（dispatch）vs 全局（combine）|
| AllToAll（combine）| 0 | `total_tokens // token_tile` | 依赖所有 down_proj tile 完成（N:1 扇入）|
| SwiGLU / Add | 0 | = 上游 GMM task_num（1:1）| task_num 必须等于上游，否则事件错位 |

### 5.6 Event 设计规范

| 规则 | 说明 |
|---|---|
| **单调性** | event 计数器只增不减，保证无死锁 |
| **1:1 依赖** | 每个生产者 tile 完成后触发对应消费者 tile |
| **N:1 扇入** | 多个上游 tile 共同触发一个下游 tile，设置 `trigger_count = N` |
| **1:N 扇出** | 一个上游触发多个下游，每个下游分配独立 event ID |
| **ID 不重叠** | dispatch / combine / gmm 事件 ID 段需严格区分 |

---

## 6. RuntimeConfig 生成

`RuntimeConfig` 是 Multicore MoE-FFN 的调度配置二进制，包含每个 task 的依赖 event、触发 event、tensor 地址偏移等信息，**按 rank 独立生成**。

### 6.1 正向

```bash
# 必须从 hyper-parallel 仓库根目录（hyper_parallel/ 包所在目录）运行
# 直接 python gen_runtime_data.py 会报 ImportError（相对导入）
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

输出文件：

```text
<output_dir>/
├── up_proj_tiling.bin              # GMM1 tiling（各 rank 共用）
├── swiglu_tiling.bin               # SwiGLU tiling（各 rank 共用）
├── down_proj_tiling.bin            # GMM2 tiling（各 rank 共用）
├── all_event_counters.bin          # 4096 uint8 zeros，4 KB（各 rank 共用，需 symmetric memory）
├── gmm_workspace.bin               # 256 MiB zeros（各 rank 共用）
├── runtime_config_input_rank_0.bin # Rank 0 调度配置
├── runtime_config_input_rank_1.bin
└── ...
```

也可在 Python 代码中直接调用：

```python
from hyper_parallel.core.multicore.modules.mega_moe.forward.gen_runtime_data import build_config_for_rank
from hyper_parallel.core.multicore.modules.mega_moe.forward.graph import build_forward_graph
from hyper_parallel.core.multicore.scheduler.config import TaskSplitValue

tsv   = TaskSplitValue(tp=4, ep=4, seq_size=8192, all_expert_num=32, top_k=8)
graph = build_forward_graph(tsv, dispatch_sv=128, up_proj_sv=4096,
                            swiglu_sv=128, down_proj_sv=4096, combine_sv=128,
                            hidden_size=7168, intermediate_size=2048)
graph.propagate_splits(tsv)
cfg = build_config_for_rank(graph, tsv, rank_id=0)   # 返回 RuntimeConfigC
```

### 6.2 反向

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

输出文件：

```text
<output_dir>/
├── act_grad_tiling.bin             # GMM1 反向 tiling（各 rank 共用）
├── gate_grad_tiling.bin            # GMM2 反向 tiling
├── w1_grad_tiling.bin              # GMM3（W1 梯度）tiling
├── w2_grad_tiling.bin              # GMM4（W2 梯度）tiling
├── swiglu_grad_tiling.bin          # SwiGLU 反向 tiling
├── all_event_counters.bin          # 4096 uint8 zeros，4 KB（各 rank 共用）
├── gmm_workspace.bin
└── runtime_config_input_rank_<i>.bin
```

---

## 7. 编译与安装

### 7.1 依赖准备

依赖版本和 SHA256 由 `scripts/native/config/dependencies.lock.json` 锁定。`build.sh` 自动校验本地依赖
缓存，匹配时复用，缺失或不一致时下载/刷新，再进入编译阶段：

```bash
./build.sh --multicore all --shmem all --custom-ops off
```

依赖缓存位于 `build/native/deps`。用户先 source 所选 CANN >= 9.1.0 的官方 `set_env.sh`，构建脚本读取其
导出的 `ASCEND_HOME_PATH`。

### 7.2 统一 vendor 与框架 adapter

每个 SoC 的一次 CANN 调用同时编译 `HyperMegaMoe` 和 `HyperMegaMoeGrad`。多 SoC 构建分别生成并校验
vendor 输入，再合并为一个 `hyper_parallel_multicore_nn` vendor 和一个 `libcust_opapi.so`。对外 ACLNN
符号固定为：

- `aclnnHyperMegaMoe` / `aclnnHyperMegaMoeGetWorkspaceSize`
- `aclnnHyperMegaMoeGrad` / `aclnnHyperMegaMoeGradGetWorkspaceSize`

vendor 构建校验要求上述四个 `aclnnHyperMegaMoe*` 符号存在，并拒绝已知的冲突或错误大小写身份。
MindSpore adapter 使用
`ops.CustomOpBuilder`，PyTorch adapter 使用 `NpuExtension`；二者均含 CPython module，因此必须分别构建
带 cp310、cp311 或 cp312 标签的 wheel。

adapter 运行时按组件相对路径定位并绝对加载自带 vendor。host ELF 使用 `$ORIGIN` RUNPATH；`set_env.bash` 按 CANN custom OPP
契约把 vendor 的 `op_api/lib` 加入调用 shell 的 `LD_LIBRARY_PATH`。

### 7.3 wheel 与本地 PYTHONPATH

wheel 与 PYTHONPATH 共用 `build/native/payload/hyper_parallel` 中的 native payload：

```bash
./build.sh --multicore all --shmem all --custom-ops on --strict off
```

该命令总是重新组装 payload 并生成 wheel；PYTHONPATH 开发直接复用 payload。默认复用 SHMEM SDK 和按 SoC
vendor 的重型编译缓存；MindSpore/PyTorch adapter 使用按 CPython 和框架身份隔离的工作目录，每次从干净目录
重建。`--clean` 显式清理所选组件的工作与安装输出，但保留依赖下载缓存。

multicore 遵循 CANN 自定义算子包的环境契约。wheel 和 PYTHONPATH 都必须在启动业务或框架 Python 进程前 source 制品中的
`set_env.bash`：

```bash
source /usr/local/Ascend/cann/set_env.sh
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
source build/native/payload/hyper_parallel/core/multicore/lib/set_env.bash
python your_program.py
```

wheel 安装态使用安装到活动 Python 环境 `bin` 目录的真实 shell 定位脚本：

```bash
source /usr/local/Ascend/cann/set_env.sh
source "$(command -v hyper_parallel_multicore_set_env.bash)"
python your_program.py
```

脚本可重定位、重复 source 幂等，并保留、去重已有 OPP/动态库路径。未 source 时首次调用给出
`HP-NATIVE-OPP-NOT-ACTIVATED`；框架已导入时给出
`HP-NATIVE-OPP-ACTIVATION-TOO-LATE`。

### 7.4 MindSpore 接入

**使用方式**

```python
import mindspore as ms
ms.set_context(mode=ms.PYNATIVE_MODE)   # 必须：仅支持 PyNative 模式

import hyper_parallel.core.multicore as mc
mc.mega_moe(...)
mc.mega_moe_grad(...)
```

> **接口支持 PyNative 模式（动态图）**
>
> Graph 模式（`ms.GRAPH_MODE`）不属于本版本接口范围。YAML 定义中 `function: disable: True` 使 MindSpore
> 编译器不会对这两个算子进行图级追踪和下沉；算子内部依赖运行时动态 tensor 地址，不满足静态图的
> 编译期地址静态化要求。

### 7.5 PyTorch 接入

PyTorch 与 torch_npu 必须使用和所选 CANN 匹配、且 `_GLIBCXX_USE_CXX11_ABI=1` 的配套版本。
框架 adapter 通过上述 component build 生成。

---

## 8. API 接口

### 8.1 平台切换

通过环境变量 `HYPER_PARALLEL_PLATFORM` 选择后端（默认 `mindspore`）：

```python
import os
os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"   # 或 "mindspore"（默认）

import hyper_parallel.core.multicore as mc
mc.mega_moe(...)
mc.mega_moe_grad(...)
```

### 8.2 mega_moe（正向）

```python
mc.mega_moe(
    # AllToAll dispatch
    dispatch_target,      # [tokens, hidden]          bf16   AllToAll 目标缓冲区，output（in-place）
    dispatch_target_off,  # [all_expert_num]           int64  每 expert 的远端写偏移（目标 PE）
    dispatch_src,         # [tokens, hidden]           bf16   发送源数据
    dispatch_src_off,     # [all_expert_num]           int64  每 expert 的本地读偏移
    dispatch_size,        # [all_expert_num]           int32  每 expert 发送的元素数
    # GMM1（up_proj）
    up_proj_weight,       # [E, hidden, intermediate*2] bf16  W1 权重
    up_proj_glist,        # [E]                        int64  每 expert token 累积和
    up_proj_y,            # [tokens, intermediate*2]  bf16   GMM1 输出，output（in-place）
    swiglu_out,           # [tokens, intermediate]    bf16   SwiGLU 输出，output（in-place）
    # GMM2（down_proj）
    down_proj_weight,     # [E, intermediate, hidden] bf16   W2 权重
    down_proj_glist,      # [E]                       int64  每 expert token 累积和
    down_proj_y,          # [tokens, hidden]          bf16   GMM2 输出，output（in-place）
    # AllToAll combine
    combine_target,       # [tokens, hidden]          bf16   AllToAll 汇聚目标缓冲区，output（in-place）
    combine_target_off,   # [all_expert_num]          int64  每 expert 的远端写偏移
    combine_src_off,      # [all_expert_num]          int64  每 expert 的本地读偏移
    combine_size,         # [all_expert_num]          int32  每 expert 发送的元素数
    # 配置张量
    gmm_workspace,        # [256*1024*1024]           uint8  GMM 工作区（256 MiB）
    up_proj_tiling,       # uint8                     GMM1 tiling，from gen_runtime_data.py
    swiglu_tiling,        # uint8                     SwiGLU tiling
    down_proj_tiling,     # uint8                     GMM2 tiling
    runtime_config,       # uint8                     per-rank 调度配置，from gen_runtime_data.py
    all_event_counters,   # uint8  事件同步计数器（symmetric memory）
    # 标量
    rank_id,              # int  当前 rank
    ep,                   # int  Expert Parallel 度
    expert_num,           # int  全局 expert 总数（all_expert_num）
    hidden_size,          # int  隐藏层维度
    seq_size,             # int  全局序列长度
)
```

**返回值**：无（PyTorch）/ 5 个 in-place 张量的元组（MindSpore）

### 8.3 mega_moe_grad（反向）

```python
mc.mega_moe_grad(
    # AllToAll dispatch（梯度分发）
    dispatch_target,      # [tokens, hidden]           bf16   AllToAll 目标缓冲区，output（in-place）
    dispatch_target_off,  # [all_expert_num]            int64  每 expert 的远端写偏移
    dy,                   # [tokens, hidden]            bf16   输入梯度（AllToAll 发送源）
    dispatch_src_off,     # [all_expert_num]            int64  每 expert 的本地读偏移
    dispatch_size,        # [all_expert_num]            int32  每 expert 发送的元素数
    # act_grad（GMM1 反向：dispatch_target @ W2.T）
    hidden,               # [tokens, intermediate]      bf16   正向 SwiGLU 输出（缓存）
    hidden_dw,            # [E, intermediate, hidden]   bf16   W2 权重梯度，output（in-place）
    w2,                   # [E, intermediate, hidden]   bf16   W2 权重（= 正向 down_proj_weight）
    act_grad_y,           # [tokens, intermediate]      bf16   GMM1 反向输出，output（in-place）
    # swiglu_grad
    gate,                 # [tokens, intermediate*2]   bf16   正向 up_proj_y（SwiGLU 输入缓存）
    grad_gate,            # [tokens, intermediate*2]   bf16   SwiGLU 梯度输出，output（in-place）
    # gate_grad（GMM2 反向：grad_gate @ W1.T）
    w1,                   # [E, hidden, intermediate*2] bf16  W1 权重（= 正向 up_proj_weight）
    gate_dx,              # [tokens, hidden]            bf16   GMM2 反向输出，output（in-place）
    grad_x,               # [tokens, hidden]            bf16   AllToAll combine 输出，output（in-place）
    # AllToAll combine（梯度汇聚）
    combine_target_off,   # [all_expert_num]            int64  每 expert 的远端写偏移
    combine_src_off,      # [all_expert_num]            int64  每 expert 的本地读偏移
    combine_size,         # [all_expert_num]            int32  每 expert 发送的元素数
    # w1_grad（GMM3）、w2_grad（GMM4）
    permute_out,          # [tokens, hidden]            bf16   W1 梯度计算的 in-place 中间缓冲区
    gate_dw,              # [E, hidden, intermediate*2] bf16  W1 权重梯度，output（in-place）
    group_list,           # [E]                         int64  每 expert token 累积和
    # 配置张量
    act_grad_tiling,      # uint8                       GMM1 反向 tiling
    gate_grad_tiling,     # uint8                       GMM2 反向 tiling
    w1_grad_tiling,       # uint8                       GMM3（W1 梯度）tiling
    w2_grad_tiling,       # uint8                       GMM4（W2 梯度）tiling
    swiglu_grad_tiling,   # uint8                       SwiGLU 反向 tiling
    gmm_workspace,        # [256*1024*1024]             uint8  GMM 工作区
    swiglu_grad_workspace, # [64*1024*1024]             uint8  SwiGLU 反向工作区
    runtime_config,       # uint8                       per-rank 调度配置
    all_event_counters,   # uint8  事件同步计数器（symmetric memory）
    # 标量
    rank_id, ep, expert_num, hidden_size, seq_size,
)
```

---

## 9. 测试

### MindSpore

7.3 节的 CANN 和 multicore payload 环境激活完成后，执行 pytest 入口。该用例通过仓内 launcher 启动
MindSpore 多卡进程，并覆盖 HyperMegaMoe 正反向：

```bash
pytest -v tests/mindspore/st/multicore/test_moe.py
```

### PyTorch

PyTorch multicore ST 不属于本版本提供的测试范围。

### 精度容限

所有精度测试使用 bfloat16，容限为：

```text
|kernel - ref| ≤ atol + rtol × |ref|    (rtol = atol = 1e-3)
```

---

## 10. 扩展指南

### 10.1 添加新算子任务类型（Python 侧）

**Step 1** — 在 `scheduler/graph.py` 扩展 `OpType`：

```python
class OpType(Enum):
    ALLTOALL    = "alltoall"
    GMM         = "gmm"
    SWIGLU      = "swiglu"
    SWIGLU_GRAD = "swiglu_grad"
    RMSNORM     = "rmsnorm"   # ← 新增
```

**Step 2** — 在 `tasks/rmsnorm.py` 实现 FillConfig：

```python
from .task_base import FillConfig
from ..scheduler.config import RuntimeConfigC

class RMSNormFillConfig(FillConfig):
    def fill(self, cfg: RuntimeConfigC, op, tsv) -> None:
        for i in range(op.task_num):
            td = cfg.task_descs[cfg.task_num_all]
            td.task_type = TaskType.TASK_RMSNORM
            td.input_addr[0]  = op.inputs[0].param_position
            td.output_addr[0] = op.outputs[0].param_position
            td.pre_event_id   = ...
            td.trigger_event_id = ...
            cfg.task_num_all += 1
```

**Step 3** — 在 `tasks/registry.py` 注册：

```python
from .rmsnorm import RMSNormFillConfig
FILL_CONFIG_REGISTRY[OpType.RMSNORM] = RMSNormFillConfig
```

**Step 4** — 在 C++ `ops/<op>/op_kernel/worker_kernel.cpp` 的 switch 中添加分支：

```cpp
case TaskType::TASK_RMSNORM:
    ExecuteRMSNorm(task_desc);
    break;
```

### 10.2 添加新模型（MoE FFN 以外的算子组合）

**Step 1** — 建立目录：

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

**Step 2** — 在 `graph.py` 构建 DAG：

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

**Step 3** — 实现 CLI（参考 `modules/mega_moe/forward/gen_runtime_data.py`）：

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

### 10.3 添加新 C++ 计算核（AscendC 侧）

**Step 1** — 建立算子目录：

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

**Step 2** — 在 `attn_fwd.cpp` 中引用共享 scheduler：

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

**Step 3** — 在 `worker_kernel.cpp` 中引用共享 runtime：

```cpp
#include "kernel_operator.h"
#include "runtime/worker_kernel.h"  // build 脚本将 ops/runtime/ 拷入 op_kernel/runtime/

class KernelWorker : public KernelWorkerBase<KernelWorker> {
 public:
    // input_list 索引由本算子的参数布局决定
    static constexpr uint32_t TILING_IDX = N;   // 替换为实际索引
    static constexpr uint32_t EVENT_IDX  = M;

    __aicore__ inline void ExecuteComputeKernel(TaskDesc task_desc) {
        switch (task_desc.task_type) {
            // 添加本算子的 case
        }
    }
};
```

**Step 4** — 在 `put_mem_signal/put_mem_signal_kernel.cpp` 中用本算子的数据类型实例化：

```cpp
#include "kernel_operator.h"
#include "shmem.h"

template <typename T>
class PutMemSignalKernel { /* ... */ };  // 与现有实现相同

extern "C" inline __aicore__ void put_mem_signal_kernel(...) {
    PutMemSignalKernel<DTYPE_YOUR_TYPE> op;  // 替换为本算子的数据类型宏
    op.Init(...);
    op.Process();
}
```

**Step 5** — 在 `scripts/native/assemble_multicore_source.py` 中登记新算子，使源码组装阶段复制共享的 `runtime/`：

```python
_HYPER_OPERATORS = ("hyper_mega_moe", "hyper_mega_moe_grad", "hyper_attn_fwd")
```

---

## 11. 附录：常见错误与排查

| 错误现象 | 可能原因 | 排查步骤 |
|---|---|---|
| `TaskDesc` event_id 越界 | all_event_num 计算错误 | 检查 `scheduler/config.py` 中 `all_event_num` 属性公式 |
| 两 bin 文件 task_num 不一致 | SplitSpec 的 task_num_fn 写错 | 打印 `propagate_splits()` 后每个 op 的 task_num |
| AIV hang（无限等待）| N:1 扇入 trigger_count 设置偏大 | 检查 FillConfig 的 trigger_count = 实际上游 tile 数 |
| GMM workspace 越界 | TILE_M/N 设置过大 | 检查 `tiling_tables.py` 中 workspace 预算 |
| `runtime/worker_kernel.h` 找不到 | 新算子未纳入隔离源码组装 | 检查 `assemble_multicore_source.py` 的算子列表和 `MULTICORE_SOURCE_ASSEMBLY_FAILED` 日志 |
| Python `ImportError: scheduler` | 从非包根目录运行 | 从 `hyper-parallel/` 根目录用 `python -m` 方式运行 |
| MindSpore `RuntimeError`: 切换 PYNATIVE | 在 Graph 模式下调用 | `ms.set_context(mode=ms.PYNATIVE_MODE)` 放在 import 之前 |
