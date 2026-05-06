# MoE-FFN Multicore Operator

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

根因分析：GroupedGEMM 只使用 Cube 核，Vector 核在 GMM 执行期间全部空闲；SwiGLU 由 Vector 核执行时 25 个 Cube 同时空转；AllToAll 由 HOST 驱动，通信空窗期设备内 AIC 和 AIV 也无法计算。

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

在多卡 AllToAll 场景下，各 Rank 同时向同一目标 Rank 发送数据会造成网络拥塞，导致尾延迟暴露。RATR 通过调整 AllToAll Tile 的执行顺序，使发往不同 Rank 的通信流量在时间轴上均匀分散，避免多源 Rank 同时涌向同一目标。RATR 不修改任何计算逻辑，仅对 `vector_task_indexs` 中 AllToAll Tile 重排，与 SSC 静态调度配置一起在 RuntimeConfig 中离线生成，运行时零开销。

反向计算流程（同理融合）：

```text
AllToAll-Dispatch → ┌─ GMM1(act_grad)  ─────────────────┐
                    └─ GMM4(w1_grad)                     │
                                                         ▼
                                              SwiGLU-grad → GMM2(gate_grad) → ┌─ AllToAll-Combine
                                                                               └─ GMM3(w2_grad)
```

GMM1 与 GMM4 并行，AllToAll-Combine 与 GMM3 并行，进一步减少空闲等待。

---

## 2. 正向算子

### 2.1 计算流图

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

### 2.2 Task 切分参数

| 算子 | split_value | 执行核 | task_num 计算 |
|---|---|---|---|
| dispatch | 128 | AIV | `all_expert_num × (per_expert_seq_to_other // 128)` |
| up_proj | 4096 | AIC | `num_cube_cores × single_rank_expert_num` |
| swiglu | 128 | AIV | `(per_expert_seq // 128) × single_rank_expert_num` |
| down_proj | 4096 | AIC | `num_cube_cores × single_rank_expert_num` |
| combine | 128 | AIV | `all_expert_num × (per_expert_seq_to_other // 128)` |

### 2.3 Tiling 文件（正向，各 rank 共用）

| 文件 | 参数位置 | 说明 |
|---|---|---|
| `up_proj_tiling.bin` | pos 17 | GMM1 tiling，大小 = `num_cube_cores × tiling_entry_size` |
| `swiglu_tiling.bin` | pos 18 | SwiGLU tiling，大小 = `(2×num_cube_cores+1) × entry_size` |
| `down_proj_tiling.bin` | pos 19 | GMM2 tiling |

生成方式见第 5.1 节。

---

## 3. 反向算子

### 3.1 计算流图

```text
dispatch ──► act_grad (GMM1)  ──────────────────────────────────────────────────────┐
         └── w1_grad (GMM4)                                                         │
                                                                                    ▼
                                                  swiglu_grad ──► gate_grad (GMM2) ──► combine
                                                                                   └── w2_grad (GMM3)
```

| 算子 | 映射 | 输入 | 输出 |
|---|---|---|---|
| dispatch | AllToAll | `dy`（输入梯度） | `dispatch_target`（分发后梯度） |
| act_grad (GMM1) | `dispatch_target @ w2.T` | `dispatch_target`, `w2` | `act_grad_y`（激活梯度） |
| w1_grad (GMM4) | `hidden.T @ dispatch_target` | `hidden`（正向 SwiGLU 输出）, `dispatch_target` | `hidden_dw`（W2 权重梯度） |
| swiglu_grad | SwiGLU backward | `act_grad_y`, `gate`（正向 up_proj_y） | `grad_gate`（SwiGLU 梯度） |
| gate_grad (GMM2) | `grad_gate @ w1.T` | `grad_gate`, `w1` | `gate_dx`（combine 前梯度） |
| w2_grad (GMM3) | `gate.T @ grad_gate` | `gate`（正向 up_proj_y）, `grad_gate` | `gate_dw`（W1 权重梯度） |
| combine | AllToAll | `gate_dx` | `grad_x`（汇聚后梯度） |

### 3.2 并行执行说明

- **act_grad（GMM1）与 w1_grad（GMM4）并行**：两者均依赖 dispatch 输出 `target`；GMM1 使用 `advance_mode="cube_only"`（只推进 task 计数，不推进 event），GMM4 使用 `advance_mode="cube_custom"` 统一推进 event，避免重复计数。

- **combine 与 w2_grad（GMM3）并行**：两者均依赖 gate_grad（GMM2）输出；combine 使用 `advance_mode="vector_only"`，GMM3 使用 `cube_custom` 处理 event。

### 3.3 Tiling 文件（反向，各 rank 共用）

| 文件 | 参数名 | 说明 |
|---|---|---|
| `act_grad_tiling.bin` | `act_grad_tiling` | GMM1 反向 tiling（act_grad：dispatch_target @ W2.T） |
| `gate_grad_tiling.bin` | `gate_grad_tiling` | GMM2 反向 tiling（gate_grad：grad_gate @ W1.T） |
| `w2_grad_tiling.bin` | `w2_grad_tiling` | GMM3（W2 权重梯度）tiling |
| `w1_grad_tiling.bin` | `w1_grad_tiling` | GMM4（W1 权重梯度）tiling |
| `swiglu_grad_tiling.bin` | `swiglu_grad_tiling` | SwiGLU 反向 tiling |

生成方式见第 5.2 节。

---

## 4. Task 切分设计

### 4.1 TaskSplitValue

`TaskSplitValue` 封装硬件拓扑参数和运行时计数器：

```python
from hyper_parallel.core.multicore.modules.moe_ffn.common.compute_graph import TaskSplitValue

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
| `all_event_num` | 自动计算 | 全局 event 总数（combine 全局触发点） |

### 4.2 propagate_splits 算法

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

### 4.3 split_inputs 选择原则

`split_inputs` 中必须选择**图内有上游生产者的输入**（非外部叶子张量）。叶子张量的 `split_dim` 在每次 `propagate_splits` 时重置为 -1，以叶子为触发条件将导致 `task_num` 始终为 1。

```python
# ✅ 正确：inputs[1]=target 是 dispatch 的输出，图内连通
w1_grad: split_inputs=[(1, 0)]   # inputs[1] = target

# ❌ 错误：inputs[0]=hidden（外部叶子），split_dim 永远 -1
w1_grad: split_inputs=[(0, 0)]   # task_num 始终 = 1
```

### 4.4 Event 驱动协调

Multicore MoE-FFN 中各核通过事件信号量（event）同步，关键配置：

**AllToAll dispatch**：per-expert 触发

```text
trigger_event = pre_event_num + (task_index // expert_group_size) + 1
```

每个 expert group 的 dispatch task 完成后触发一个 event，GMM 以 expert 粒度消费，无需等待全部 token 发送完毕。

**AllToAll combine**：global 触发

```text
trigger_event = all_event_num  (全局唯一 event)
```

所有 combine task 触发同一个全局 event，表示本 rank 所有 expert 结果已发送完毕。

**GMM 间 event 协调**（反向）：

| 算子 | advance_mode | 说明 |
|---|---|---|
| act_grad (GMM1) | `cube_only` | 只推进 task 计数，不推进 event |
| w1_grad (GMM4) | `cube_custom` | 统一推进 event（补偿 GMM1 未推进的部分） |
| combine | `vector_only` | 只推进 task 计数 |
| w2_grad (GMM3) | `cube_custom` | 统一推进 event |

---

## 5. RuntimeConfig 生成

`RuntimeConfig` 是 Multicore MoE-FFN 的调度配置二进制，包含每个 task 的依赖 event、触发 event、tensor 地址偏移等信息，**按 rank 独立生成**。

### 5.1 正向

```bash
# 必须从 hyper-parallel 仓库根目录（hyper_parallel/ 包所在目录）运行
# gen_runtime_data.py 使用相对导入，直接 python gen_runtime_data.py 会报
# "ImportError: attempted relative import with no known parent package"
python -m hyper_parallel.core.multicore.modules.moe_ffn.forward.gen_runtime_data \
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

也可在 Python 代码中直接调用（端到端示例见第 8 节）：

```python
from hyper_parallel.core.multicore.modules.moe_ffn.forward.gen_runtime_data import build_config_for_rank
from hyper_parallel.core.multicore.modules.moe_ffn.forward.forward_graph import build_forward_graph
from hyper_parallel.core.multicore.modules.moe_ffn.common.compute_graph import TaskSplitValue

tsv   = TaskSplitValue(tp=4, ep=4, seq_size=8192, all_expert_num=32, top_k=8)
graph = build_forward_graph(tsv, dispatch_sv=128, up_proj_sv=4096,
                            swiglu_sv=128, down_proj_sv=4096, combine_sv=128,
                            hidden_size=7168, intermediate_size=2048)
graph.propagate_splits(tsv)
cfg = build_config_for_rank(graph, tsv, rank_id=0)   # 返回 RuntimeConfigC
```

### 5.2 反向

```bash
# 同上，必须从 hyper-parallel 仓库根目录运行
python -m hyper_parallel.core.multicore.modules.moe_ffn.backward.gen_runtime_data \
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
├── w2_grad_tiling.bin              # GMM3（W2 梯度）tiling
├── w1_grad_tiling.bin              # GMM4（W1 梯度）tiling
├── swiglu_grad_tiling.bin          # SwiGLU 反向 tiling
├── all_event_counters.bin          # 4096 uint8 zeros，4 KB（各 rank 共用，需 symmetric memory）
├── gmm_workspace.bin
└── runtime_config_input_rank_<i>.bin
```

---

## 6. 编译与安装

### 6.1 环境变量

Multicore MoE-FFN 依赖两个预编译 CANN vendor 包：

| 包 | 导出符号 | 说明 |
|---|---|---|
| `multicore_moe_ffn_nn` | `aclnnMulticoreMoeFfn*` | 正向 kernel |
| `multicore_moe_ffn_grad_nn` | `aclnnMulticoreMoeFfnGrad*` | 反向 kernel |

库路径解析优先级（从高到低）：

```bash
# 方式 1：显式指定每个包的 lib 目录（最高优先级）
export CANN_VENDOR_FWD_LIBDIR=/path/to/multicore_moe_ffn_nn/op_api/lib
export CANN_VENDOR_BWD_LIBDIR=/path/to/multicore_moe_ffn_grad_nn/op_api/lib

# 方式 2：指定 vendors 根目录（fwd/bwd 路径自动推导）
export HP_MULTICORE_DIR=/path/to/vendors_root

# 方式 3：遗留单库模式（fwd 和 bwd 共用同一 lib 目录）
export CANN_VENDOR_LIBDIR=/path/to/opp_vendor_root/op_api/lib

# 方式 4：无需设置（自动从 prebuild/multicore_moe_ffn.tar.gz 解压）
```

如使用预编译包（`prebuild/multicore_moe_ffn.tar.gz`），无需手动设置，导入时自动解压并检测路径。

### 6.2 PyTorch 接入

> **⚠️ 注意**
>
> PyTorch 扩展**暂不支持自动编译**，默认处于关闭状态（`scripts/build_multicore.sh` 中 `BUILD_TORCH_EXTENSION=false`）。
> 如需编译，请将该变量改为 `true`：
>
> ```bash
> # 编辑 scripts/build_multicore.sh，将第 22 行改为：
> BUILD_TORCH_EXTENSION=true
> ```
>
> 或者直接调用 `setup.py`（见下方）。

**编译步骤（手动）**

```bash
cd hyper_parallel/core/multicore/platform/torch
python setup.py build_ext --inplace
```

可选：通过环境变量启用 Ninja 加速编译：

```bash
USE_NINJA=1 python setup.py build_ext --inplace
```

**编译流程说明**

`setup.py` 使用 `torch_npu.utils.cpp_extension.NpuExtension` 将 `csrc/` 下的所有 `.cpp` 源文件编译为 Python C 扩展：

1. **vendor 库路径解析**：按优先级依次查找 `CANN_VENDOR_FWD/BWD_LIBDIR` → `CANN_VENDOR_LIBDIR` → `prebuild/multicore_moe_ffn.tar.gz`（自动解压），确定 `libcust_opapi.so` 的位置。
2. **头文件**：自动从 `torch_npu` 安装目录提取 `acl/inc` 和 `op-plugin` 头文件路径。
3. **rpath**：将 vendor 库目录写入编译产物的 rpath，运行时无需依赖 `LD_LIBRARY_PATH`。
4. **环境变量**：`ASCEND_CUSTOM_OPP_PATH` 和 `LD_LIBRARY_PATH` 由 `__init__.py` 在 import 时自动设置（仅影响当前进程），无需手动修改 `~/.bashrc`。如需跨会话持久化，按编译结束后打印的提示手动添加到 `~/.bashrc`。

**编译产物**

```text
platform/torch/hyper_parallel_multicore_moe_ffn_pta.cpython-3x-aarch64-linux-gnu.so
```

**使用方式**

```python
import hyper_parallel.core.multicore as mc
mc.moe_ffn_fwd(...)
mc.moe_ffn_bwd(...)
```

---

### 6.3 MindSpore 接入

**编译步骤**

```bash
cd hyper_parallel/core/multicore/platform/mindspore
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

**编译流程说明**

MindSpore 自定义算子需通过 `ms.ops.CustomOpBuilder` 编译，cmake 的职责是准备编译参数并驱动 Python 脚本完成实际编译：

1. **vendor 库路径解析**：与 PyTorch 路径逻辑相同，cmake 按优先级解析 `CANN_VENDOR_FWD/BWD_LIBDIR` → `HP_MULTICORE_DIR` → `CANN_VENDOR_LIBDIR` → prebuild tarball，确定 `libcust_opapi.so` 的位置。若四者均未配置且 tarball 不存在，cmake 报错退出。

2. **代码生成**：cmake 将 vendor 路径、头文件路径、链接参数等写入 `build/build_custom_with_ms.py`，该脚本调用：

   ```python
   ms.ops.CustomOpBuilder(
       name='hyper_parallel_multicore_moe_ffn_ms',
       sources=[...],          # c_api/ + framework/ 下的 .cc 源文件
       op_def=[..._op.yaml],   # moe_ffn_fwd_op.yaml + moe_ffn_bwd_op.yaml
       ...
   ).build()
   ```

   `CustomOpBuilder` 读取 YAML 算子定义，自动生成 Python 绑定文件（`gen_ops_def.py`、`gen_ops_prim.py`），并编译 C++ 源码为 `.so`。

3. **YAML 算子定义**：`c_api/moe_ffn_fwd/moe_ffn_fwd_op.yaml` 和 `c_api/moe_ffn_bwd/moe_ffn_bwd_op.yaml` 声明了算子参数类型、in-place 返回规则及 `side_effect_mem` 标记，MindSpore 框架据此生成正确的 PyBoost 调度入口和 device address 准备逻辑。YAML 中设置了 `function: disable: True` 和 `dispatch: enable: False`，**静态图编译路径被显式禁用**（原因见下方注意事项）。

4. **rpath**：链接时写入 `$$ORIGIN/../../../../lib`、vendor fwd/bwd lib 目录，使产物在 `build/lib/` 就地可用，无需安装。

5. **`ASCEND_CUSTOM_OPP_PATH`**：`__init__.py` 在首次 `import` 时自动从 vendor 路径派生并设置该环境变量，再加载 `.so`，因此**不需要**手动 `source ~/.bashrc`。

**编译产物**

```text
platform/mindspore/build/lib/
├── hyper_parallel_multicore_moe_ffn_ms.so              # C++ 扩展主体
└── hyper_parallel_multicore_moe_ffn_ms_auto_generate/
    ├── gen_ops_def.py   # 算子定义（MindSpore op registry）
    └── gen_ops_prim.py  # 算子原语（PyBoost 调用入口）
```

**使用方式**

```python
import mindspore as ms
ms.set_context(mode=ms.PYNATIVE_MODE)   # 必须：仅支持 PyNative 模式

import hyper_parallel.core.multicore as mc
mc.moe_ffn_fwd(...)
mc.moe_ffn_bwd(...)
```

> **注意：当前仅支持 PyNative 模式（动态图）**
>
> Graph 模式（`ms.GRAPH_MODE`）**暂未实现**。原因：
> - `moe_ffn_fwd` / `moe_ffn_bwd` 的 YAML 定义中设置了 `function: disable: True` 与 `dispatch: enable: False`，MindSpore 编译器无法对这两个算子进行图级追踪和下沉（lower）；
> - 算子内部依赖运行时动态 tensor 地址（AllToAll symmetric memory 指针、per-rank RuntimeConfig），不满足静态图的编译期地址静态化要求。
>
> 在 Graph 模式或 `@ms.jit` 装饰的函数内调用时，Python 层会在运行时抛出 `RuntimeError`，并提示切换到 PyNative 模式。Graph 模式支持计划在后续版本中实现。

---

## 7. API 接口

### 7.0 平台切换

通过环境变量 `HYPER_PARALLEL_PLATFORM` 选择后端（默认 `mindspore`），在 import 前设置即可：

```python
import os
os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"   # 或 "mindspore"（默认）

import hyper_parallel.core.multicore as mc
mc.moe_ffn_fwd(...)   # 自动路由到对应平台实现
mc.moe_ffn_bwd(...)
```

### 7.1 moe_ffn_fwd（正向）

```python
mc.moe_ffn_fwd(
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
    all_event_counters,   # [4096]                    uint8  事件同步计数器（symmetric memory）
    # 标量
    rank_id,              # int  当前 rank
    ep,                   # int  Expert Parallel 度
    expert_num,           # int  本 rank 持有的 expert 数（= all_expert_num // ep）
    hidden_size,          # int  隐藏层维度
    seq_size,             # int  全局序列长度
)
```

**返回值**：无（PyTorch）/ 5 个 in-place 张量的元组（MindSpore）

### 7.2 moe_ffn_bwd（反向）

```python
mc.moe_ffn_bwd(
    # AllToAll dispatch（梯度分发）
    dispatch_target,      # [tokens, hidden]           bf16   AllToAll 目标缓冲区（接收分发后的梯度），output（in-place）
    dispatch_target_off,  # [all_expert_num]            int64  每 expert 的远端写偏移
    dy,                   # [tokens, hidden]            bf16   输入梯度（AllToAll 发送源）
    dispatch_src_off,     # [all_expert_num]            int64  每 expert 的本地读偏移
    dispatch_size,        # [all_expert_num]            int32  每 expert 发送的元素数
    # act_grad（GMM1 反向：dispatch_target @ W2.T）
    hidden,               # [tokens, intermediate]      bf16   正向 SwiGLU 输出（缓存），W2 梯度的左矩阵
    hidden_dw,            # [E, intermediate, hidden]   bf16   W2 权重梯度，output（in-place）
    w2,                   # [E, intermediate, hidden]   bf16   W2 权重（= 正向 down_proj_weight）
    act_grad_y,           # [tokens, intermediate]      bf16   GMM1 反向输出（激活梯度），output（in-place）
    # swiglu_grad
    gate,                 # [tokens, intermediate*2]   bf16   正向 up_proj_y（SwiGLU 输入缓存）
    grad_gate,            # [tokens, intermediate*2]   bf16   SwiGLU 梯度输出，output（in-place）
    # gate_grad（GMM2 反向：grad_gate @ W1.T）
    w1,                   # [E, hidden, intermediate*2] bf16  W1 权重（= 正向 up_proj_weight）
    gate_dx,              # [tokens, hidden]            bf16   GMM2 反向输出（combine 前），output（in-place）
    grad_x,               # [tokens, hidden]            bf16   AllToAll combine 输出（最终激活梯度），output（in-place）
    # AllToAll combine（梯度汇聚）
    combine_target_off,   # [all_expert_num]            int64  每 expert 的远端写偏移
    combine_src_off,      # [all_expert_num]            int64  每 expert 的本地读偏移
    combine_size,         # [all_expert_num]            int32  每 expert 发送的元素数
    # w2_grad（GMM3）、w1_grad（GMM4）
    permute_out,          # [tokens, hidden]            bf16   W1 梯度计算的 in-place 中间缓冲区
    gate_dw,              # [E, hidden, intermediate*2] bf16  W1 权重梯度，output（in-place）
    group_list,           # [E]                         int64  每 expert token 累积和
    # 配置张量
    act_grad_tiling,      # uint8                       GMM1 反向 tiling，from gen_runtime_data.py
    gate_grad_tiling,     # uint8                       GMM2 反向 tiling
    w2_grad_tiling,       # uint8                       GMM3（W2 梯度）tiling
    w1_grad_tiling,       # uint8                       GMM4（W1 梯度）tiling
    swiglu_grad_tiling,   # uint8                       SwiGLU 反向 tiling
    gmm_workspace,        # [256*1024*1024]             uint8  GMM 工作区
    swiglu_grad_workspace, # [64*1024*1024]             uint8  SwiGLU 反向工作区
    runtime_config,       # uint8                       per-rank 调度配置
    all_event_counters,   # [4096]                      uint8  事件同步计数器（symmetric memory）
    # 标量
    rank_id, ep, expert_num, hidden_size, seq_size,
)
```

---

## 8. 端到端示例

测试文件同时充当端到端示例，可直接运行。

### 正向（PyTorch，2 卡 TP=2 EP=2）

```bash
torchrun --nproc_per_node=2 --master_addr=localhost --master_port=29500 \
    tests/torch/multicore/moe_ffn.py
```

### 反向（PyTorch，2 卡 TP=2 EP=2）

```bash
torchrun --nproc_per_node=2 --master_addr=localhost --master_port=29500 \
    tests/torch/multicore/moe_ffn.py
```

### MindSpore

```bash
msrun --worker_num=2 tests/mindspore/st/multicore/moe_ffn.py
```

---

## 9. 测试

### PyTorch

```bash
# 单机 2 卡（TP=2, EP=2）精度测试
torchrun --nproc_per_node=2 tests/torch/multicore/moe_ffn.py

# pytest 入口
pytest tests/torch/multicore/test_moe_ffn.py -v
```

测试内容（`tests/torch/multicore/moe_ffn.py`）：

- `test_moe_ffn_fwd_tp2ep2`：正向精度测试，跳过 AllToAll，对比 GMM1→SwiGLU→GMM2 参考实现
- `test_moe_ffn_bwd_tp2ep2`：反向精度测试，校验 act_grad / grad_gate / gate_dx 输出

### MindSpore

```bash
# msrun 2 卡
msrun --worker_num=2 tests/mindspore/st/multicore/moe_ffn.py

# pytest 入口
pytest tests/mindspore/st/multicore/test_moe_ffn.py -v
```

### 精度容限

所有精度测试使用 bfloat16，容限为：

```text
|kernel - ref| ≤ atol + rtol × |ref|    (rtol = atol = 1e-3, 即 0.1%)
```

---

## 10. 代码结构

```text
hyper_parallel/core/multicore/
├── modules/
│   ├── common/
│   │   ├── compute_graph.py        # TaskSplitValue / TensorSpec / SplitSpec / OperatorNode
│   │   └── runtime_structs.py      # RuntimeConfigC / TaskDescC / TensorDescC（ctypes）
│   └── moe_ffn/
│       ├── common/
│       │   ├── compute_graph.py    # MoE-FFN 专用图类 + init_task_split_value
│       │   ├── runtime_structs.py  # MoE-FFN RuntimeConfig 扩展
│       │   ├── task_builders.py    # fill_alltoall / fill_gmm / fill_swiglu / add_terminate
│       │   ├── task_builder_utils.py # advance_tsv_* / revise_task_queue
│       │   └── tiling_registry.py  # tiling 字节格式注册
│       ├── forward/
│       │   ├── forward_graph.py    # build_forward_graph()
│       │   ├── gen_runtime_data.py # RuntimeConfig 生成入口
│       │   └── tiling_tables.py    # get_up_proj_tiling_bytes 等
│       └── backward/
│           ├── backward_graph.py   # build_backward_graph()
│           ├── gen_runtime_data.py
│           └── tiling_tables.py
├── platform/
│   ├── torch/                      # PyTorch 接入（setup.py + pybind11）
│   └── mindspore/                  # MindSpore 接入（CMakeLists.txt + CustomOpBuilder）
├── prebuild/
│   └── multicore_moe_ffn.tar.gz   # 预编译 CANN vendor 包
└── doc/
    └── README.md                   # 本文档
```

### C++ CANN 算子源码（vendor 包）

```text
hyper_parallel/core/multicore/ops/
├── multicore_moe_ffn/       # 正向算子 CANN 源码（op_host / op_kernel / op_graph）
└── multicore_moe_ffn_grad/  # 反向算子 CANN 源码（含 swi_glu_grad/）
```

预编译产物打包为 `prebuild/multicore_moe_ffn.tar.gz`，`import` 时自动解压；如需从源码重新编译 CANN vendor 包，将`SOC_VALUE`填入`hyper-parallel\scripts\build_multicore_local.sh`结尾并取消注释，然后在根目录运行`bash scripts\build_multicore_local.sh`即可。
