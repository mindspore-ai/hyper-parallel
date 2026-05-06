# Symmetric Memory 模块使用文档

Symmetric Memory（对称内存）模块为 hyper-parallel 提供跨进程的对称内存管理、单边通信、通信函数及融合算子能力，支持高效的进程间同步与数据交互。该模块背景见[【RFC】HyperParallel Symmetric Memory 单边通信特性设计](https://gitcode.com/mindspore/hyper-parallel/issues/59)。当前模块仅对 PyTorch 提供完整支持，MindSpore 支持正在开发中。

## 目录

- [核心功能接口](#核心功能接口)
    - [基础能力](#基础能力)
    - [单边通信](#单边通信)
    - [通信函数](#通信函数)
    - [融合算子](#融合算子)
- [注意事项](#注意事项)

## 核心功能接口

所有接口的调用路径统一为：`hyper_parallel.core.symmetric_memory.xxx`（xxx 为具体函数名）。

### 基础能力

#### 1. 创建对称内存张量

```python
hyper_parallel.core.symmetric_memory.empty(shape, dtype) -> tensor
```

**功能**：创建跨进程共享的对称内存张量，分配的内存可用于单边通信、普通计算。
**参数**：

- `shape` (int...): 张量形状，支持整数序列、列表、元组。
- `dtype`: 张量数据类型，支持 `mindspore.dtype` / `torch.dtype`。

**示例**：
创建 shape=(2,3)、数据类型为 float32 的对称内存张量

```python
import torch
import hyper_parallel.core.symmetric_memory as symm_mem

symm_tensor = symm_mem.empty((2, 3), dtype=torch.float32)
```

**调整共享内存堆大小**：
首次创建共享内存时，会为各卡分配一个共享内存堆，该堆大小由环境变量 `SYMMETRIC_MEMORY_HEAP_SIZE` 控制，默认为 1G ，即单卡所有同时存在的共享内存超过 1G 时会报错。

#### 2. 进程同步屏障

```python
hyper_parallel.core.symmetric_memory.barrier() -> None
```

**功能**：阻塞当前进程，直到所有进程都到达该屏障，实现进程同步。
**参数**：无。

**示例**：
全局通信域同步

```python
import hyper_parallel.core.symmetric_memory as symm_mem

symm_mem.barrier()
```

### 单边通信

#### 1. 单边写操作（shmem_put）

```python
hyper_parallel.core.symmetric_memory.shmem_put(target, target_offset, src, src_offset, size, target_rank) -> None
```

**功能**：将本地源张量数据写入目标进程的目标张量（单边发送操作）。
**参数**：

- `target`: 目标对称内存张量。
- `target_offset`: 目标张量的字节偏移量。
- `src`: 本地源张量。
- `src_offset`: 源张量的字节偏移量。
- `size`: 传输数据的字节大小。
- `target_rank`: 目标进程的 rank 号。

#### 2. 单边读操作（shmem_get）

```python
hyper_parallel.core.symmetric_memory.shmem_get(target, target_offset, src, src_offset, size, target_rank) -> None
```

**功能**：从目标进程的源张量读取数据到本地目标张量（单边接收操作）。
**参数**：同 `shmem_put`。

#### 3. 原子更新信号值

```python
hyper_parallel.core.symmetric_memory.shmem_signal_op(signal, signal_offset, signal_value, signal_op=0, target_rank=0) -> None
```

**功能**：对对称内存中的信号值执行原子操作（支持设置/累加），用于进程间同步。
**参数**：

- `signal` (tensor): 信号张量（数据类型为 int32，对称内存张量）。
- `signal_offset`: 信号张量内的字节偏移量。
- `signal_value` (tensor): 待更新的信号值（数据类型为 int32）。
- `signal_op` (int64, 可选): 操作类型，0=设置（默认），1=累加。
- `target_rank` (int64, 可选): 目标进程 rank 号，默认 0。

#### 4. 等待信号满足条件

```python
hyper_parallel.core.symmetric_memory.shmem_wait_for_signal(depend_tensor, signal, signal_offset, compare_value, compare_op=0) -> None
```

**功能**：阻塞当前进程，直到指定信号值满足比较条件。
**参数**：

- `depend_tensor`: 依赖张量（确保操作顺序）。
- `signal` (tensor): 信号张量（数据类型为 int32，对称内存张量）。
- `signal_offset`: 信号张量内的字节偏移量。
- `compare_value` (tensor): 比较阈值（数据类型为 int32）。
- `compare_op` (int64, 可选): 比较算子，0=等于（默认），1=大于，2=小于。

#### 5. 单边写 + 原子更新信号

```python
hyper_parallel.core.symmetric_memory.shmem_put_with_signal(
    target, target_offset, src, src_offset, size,
    signal, signal_offset, signal_value, signal_op=0, target_rank=0
) -> None
```

**功能**：组合操作，先执行 `shmem_put` 写数据，再原子更新信号值，提升同步效率。
**参数**：

- 前 5 个参数同 `shmem_put`；
- 后 4 个参数同 `shmem_signal_op`。

### 通信函数

#### 1. 全聚集操作（shmem_allgather）

```python
hyper_parallel.core.symmetric_memory.shmem_allgather(output_tensor, input_tensor) -> None
```

**功能**：将所有 rank 的输入张量聚集并拼接至输出张量（仅支持 PyTorch）。
**参数**：

- `output_tensor`: 输出张量（对称内存张量），形状需为 `(world_size * local_shape)`。
- `input_tensor`: 本地输入张量（所有进程形状/类型需一致）。

**示例**：
通信域大小为 4，输出张量形状为 (8, 3)

```python
import torch
import hyper_parallel.core.symmetric_memory as symm_mem

local_tensor = torch.randn(2, 3).to("npu")
output_tensor = symm_mem.empty((8, 3), dtype=torch.float32)
symm_mem.shmem_allgather(output_tensor, local_tensor)
```

#### 2. 全交换操作（shmem_alltoall）

```python
hyper_parallel.core.symmetric_memory.shmem_alltoall(send_tensor_list, receive_tensor, receive_list) -> None
```

**功能**：全局每个 rank 向所有其他 rank 发送张量，并接收所有 rank 的张量（仅支持 PyTorch）。
**参数**：

- `send_tensor_list` (List[tensor]): 发送张量列表，长度等于通信域大小，`send_tensor_list[i]` 表示发送给 rank i 的张量。
- `receive_tensor`: 接收数据的对称内存张量。
- `receive_list` (List[int]): 接收字节数列表，`receive_list[i]` 表示从 rank i 接收的字节数。

**示例**：
<!-- markdownlint-disable MD037 -->

```python
import torch
import hyper_parallel.core.symmetric_memory as symm_mem

send_list = [torch.randn(2,3) for _ in range(4)]  # 通信域大小为 4
recv_tensor = symm_mem.empty((8, 3), dtype=torch.float32)
recv_list = [2*3*4] * 4  # float32 占 4 字节，每个 rank 接收 2*3 个元素
symm_mem.shmem_alltoall(send_list, recv_tensor, recv_list)
```

<!-- markdownlint-enable MD037 -->

### 融合算子

#### 1. 全聚集 + 矩阵乘法（fused_all_gather_matmul）

```python
hyper_parallel.core.symmetric_memory.fused_all_gather_matmul(a, b, c, gather_out, signal, block_size=None) -> None
```

**功能**：融合全聚集（allgather）和矩阵乘法 + 归约分散（reduce-scatter）操作（仅支持 PyTorch）。
**计算流程**：

1. `gather_out = allgather(a)`（聚集所有 rank 的 a 张量）；
2. `c = ReduceScatter(gather_out @ b)`（矩阵乘后归约分散）。

**参数**：

- `a`: 本地输入张量，形状 `(M_local, K)`；
- `b`: 权重矩阵，形状 `(K, N)`；
- `c`: 输出张量，形状 `(M, N)`；
- `gather_out`: 聚集 a 后的输出张量，形状 `(M, K)`（M = M_local * world_size）；
- `signal`: 信号张量（对称内存，形状 `(world_size)`，数据类型为int32）；
- `block_size` (可选): 分片计算的块大小，无输入时会将矩阵分为`min(world_size, 4)`块。

#### 2. 矩阵乘法 + 归约分散（fused_matmul_reduce_scatter）

```python
hyper_parallel.core.symmetric_memory.fused_matmul_reduce_scatter(x1, x2, symm_tensor, signal, reduce_op="sum") -> tensor
```

**功能**：融合矩阵乘法和归约分散操作（仅支持 PyTorch），公式：`output = ReduceScatter(x1 @ x2)`。
**参数**：

- `x1`: 左矩阵，形状 `(m, k)`，m 需为设备数（rank size）的整数倍；
- `x2`: 右矩阵，形状 `(k, n)`；
- `symm_tensor`: 对称内存张量，形状 `(m, n)`；
- `signal`: 信号张量（对称内存，形状 `(world_size)`，数据类型为int32）；
- `reduce_op`: 归约算子，仅支持 `sum`/`avg`，默认 `sum`。

**返回值**：输出矩阵，形状 `(m / rank_size, n)`。

## 注意事项

1. 框架支持：融合算子仅支持 PyTorch，MindSpore 支持待开发；
2. 张量要求：对称内存相关张量需通过 `symm_mem.empty()` 创建，创建时需要所有进程同步创建，且不同进程的张量大小需严格一致；
3. 信号操作：信号张量需为 int32 类型，用于进程间同步的原子操作/等待操作；
4. 通信域：所有通信函数操作需保证通信域内所有进程调用参数一致，避免死锁或数据不一致。
5. 运行效率：基于单边通信的集合通信接口和融合算子接口仅作参考，由于下发开销和算子 tilling 计算开销，实际性能可能较差，未来会进一步优化。
