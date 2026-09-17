# Multicore 私有 SHMEM

SHMEM 仅在 Multicore 内部提供对称内存和单边通信。
实现、Torch binding、算子、构建脚本和 native 制品均位于本组件的 `shmem/`：
`_runtime.py`负责Native与Torch模块的惰性访问，`_lifecycle.py`负责进程级引用生命周期，`_api.py`提供
Allocation与通信能力，`_debug.py`提供只读诊断；`ccsrc/`是Runtime、CANN适配与AllGather kernel。

## 生命周期

SHMEM以配对的模块级接口统一管理进程唯一Runtime及其本地使用者：

```python
from hyper_parallel.core.multicore import shmem

shmem.acquire()           # None 表示 dist.group.WORLD；首个引用初始化Runtime
...
shmem.release()           # 释放当前引用；最后一个引用关闭Runtime
```

- `None` selects Torch WORLD. An explicit EP group may be a noncontiguous subgroup;
  CANN PE coordinates are the group's local ranks, not Torch global ranks.
- Disjoint PP/DP groups bootstrap independently through group-local CANN unique-ID
  exchange. Initialization, allocation ordering, reuse barriers and final release
  involve only that group. One process supports one ordered active membership;
  equivalent group handles may share it, but overlapping different roots are rejected.
- 每次成功`acquire()`都必须对应一次`release()`。首个引用解析Root并初始化Native Runtime；后续等价Root
  只增加引用，不重复初始化；非最后一个`release()`只减少引用。
- 最后一个引用释放前必须完成所有相关backward和设备操作、释放全部对称Allocation，并保持初始化时冻结的
  NPU为当前设备。关闭前置条件失败时最后一个引用保留，可修正条件后重试`release()`。
- 干净关闭后同一进程可再次`acquire()`开启新生命周期，堆配置在新生命周期首次获取时重新生效。
  Native初始化失败不产生引用，可再次尝试；Native关闭失败会使进程进入不可恢复状态，之后拒绝再次获取。
- shutdown 时仍有存活分配会导致关闭失败；所有 rank 必须以一致顺序完成关闭，再销毁 HCCL 进程组。

## 接口契约

In the contracts below, **world** means the acquired CANN SHMEM root group,
which may be smaller than Torch WORLD. Other PP stages do not participate.
Expert parameters and forward activations are not symmetric workspace: multiple
microbatch graphs may remain live while serial calls reuse the same workspace.
Drain all backward/recompute work before closing experts or destroying groups.

所有操作接口都要求调用方持有一个尚未释放的引用，并且不得与最后一个`release()`并发。Allocation、free、barrier和
AllGather等collective路径必须由全world按一致顺序调用；单边Put/Get/Signal均为stream入队语义
（返回仅代表入队，完成需对stream同步），按[单边通信模式](#单边通信模式)的协议调用：

- `shmem.empty(*size, dtype=None, alignment=None)`：从对称 heap 分配连续 Tensor；
  `alignment` 为可选的分配基址字节对齐。
- `shmem.free(tensor)`：在 stream 静止后释放完整分配，view 不能独立释放。成功后本地立即把共享
  Storage 缩为 0：data pointer 清空，原 Tensor 与全部 view 不再指向已释放的对称内存，shape 元数据
  保留。重复 `free`、释放上一生命周期的 Tensor、把失效 Tensor 传入任何 SHMEM 操作
  （put/get/signal/wait_signal/all_gather）都会被拒绝。堆分配器可能为后续Allocation复用归还块，
  包括再次返回相同地址；调用方仍应在 `free` 时丢弃全部引用。既定使用模式：对称 Tensor 一次性
  分配（通常首次使用时）、关闭时一次性释放，全 rank 按一致顺序 collective 调用；不支持高频分配释放
  循环。
- `shmem.barrier(blocking=True)`：在当前 NPU stream 上 enqueue world barrier。默认阻塞：
  返回前同步当前 stream，即 barrier 与该流上所有更早工作均已完成，`free` 前调用一次即可；
  `blocking=False` 时返回仅代表入队，host 不等待完成，后续操作必须提交到同一 stream
  才能安全地排在 barrier 之后。
- `shmem.host_barrier()`：host 同步的 HCCL world barrier，不经 SHMEM 设备 barrier。
  返回时全 world 所有 rank 在各自**当前 stream** 上先于本调用入队的设备工作均已完成
  （torch_npu 将 HCCL barrier 入队到当前 stream）；其他 stream 上的既有工作不被覆盖，
  调用方须先自行同步（如 `stream.synchronize()`）。单 PE（未初始化 distributed 或
  world size 为 1）为 no-op。相对 stream barrier 引入 host 同步点，
  适用于需要确定性跨 rank 收敛语义的场景（如 workspace teardown）。
- `shmem.put(remote_dst, local_src, target_pe)`：单边 Put，把本地连续 NPU Tensor 按字节写入
  target PE 的对称地址（Tensor 或 view），两侧字节数必须一致。
- `shmem.get(local_dst, remote_src, source_pe)`：单边 Get，把 source PE 对称地址的字节读入
  本地连续 NPU Tensor。
- `shmem.signal(remote_signal, value, target_pe, operation="set")`：对 target PE 的一个
  `int32` 对称信号执行 `set` 或 `add`。
- `shmem.wait_signal(signal, value, comparison="eq")`：等待本地 `int32` 对称信号满足比较条件，
  `comparison` 支持 `eq`/`ne`/`gt`/`ge`/`lt`/`le`。
- `shmem.all_gather(output, input)`：world AllGather，`output` 必须为对称 Tensor，
  槽位按 world rank 顺序；零字节 `input` 校验后为 no-op（不发射 kernel）。
  接口本身不含任何 barrier，跨 rank 收敛由调用方负责：gather 前用 `host_barrier()`
  保证全 rank 的 input 与对称 output 就绪，本 rank stream 完成后再 `host_barrier()`
  保证全 rank 输出可读，然后才能覆写 input 或 `free` output。
  首版以接口层交付，组件内暂无消费者。
- `shmem.debug_state()`：返回本进程 Runtime 状态、生效配置（heap、timeout、engine、endpoint）、
  尚未配对`release()`的本地`reference_count`（各Rank可能不同）、堆占用（`allocated_bytes`已用请求字节、
  `remaining_bytes`可用字节、
  `max_allocated_bytes` 峰值）、活跃分配表（每项含 `allocation_id`/`allocation_base`/
  `allocation_bytes`，按 id 排序，可用 `tensor.data_ptr()` 对照 base 区间定位所属分配）、
  泄漏分配表（`leaked_allocations`，Tensor 引用丢失且未经 `shmem.free()` 的分配，按丢失顺序，
  字段同活跃分配表）与
  最近一次失败的只读快照，用于诊断；REPL 中逐行渲染便于阅读；
  干净关闭后生命周期相关字段均为 None。
  堆占用按请求字节投影：CANN 堆按 16 字节对齐取整并切分余洞，真实设备侧占用可能略高。

Runtime在初始化时冻结当前NPU设备；`empty`、`barrier`、`signal`、`wait_signal`以及实际发射设备任务的
非零`put`/`get`/`all_gather`都会拒绝其他设备的stream。零字节`put`/`get`和`all_gather`校验本地参数后
不发射设备任务。

## 单边通信模式

单边接口组合成两种数据面模式，`signal`/`wait_signal` 提供跨 PE 的完成通知：

- **Push（推）**：发送方 `put` 把数据直接写进接收方的对称槽，再 `signal` 通知；接收方
  `wait_signal` 后读自己的本地对称槽。数据落点在接收方，适合接收方持有消费缓冲的场景。
- **Pull（拉）**：发送方把数据写进自己的对称槽（本地写或收到的 put），`signal` 通知接收方；
  接收方 `wait_signal` 后 `get` 从发送方的对称槽拉取。数据落点在发送方，适合发送方
  复用同一缓冲多轮发布的场景。

两种模式都必须遵守两条协议约束，均来自 CANN 数据面的既定行为：

1. **Signal热点的64B隔离**：当多个来源PE或设备任务会并发更新同一目标PE上的多个独立Signal位置时，
   这些位置不能共享64B cacheline。热点消费者应让每个Signal独占一个`alignment=64`的分配，或在同一
   分配内按64B步长排布。没有并发写入邻居的单个Signal不因通用接口本身而要求64B基址对齐。
2. **跨 rank 初始化收敛**：任何远端可见写（`signal`、`put` 到对端槽位）之前，所有 rank
   必须完成对相应槽位的本地初始化（`zero_()`/`fill_()`）并经 `host_barrier()` 收敛；否则一个
   迟到的本地初始化会覆盖已送达的远端写，表现为信号永远等不到或数据读回全零。
   设备 barrier（`shmem.barrier()`）不得用作跨 rank 收敛点：其远端等待为无超时自旋，
   互连竞争下会触发 aicore 507015（见 bug_issue：
   `shmem_device_barrier_unbounded_spin_interconnect_contention.md`）。

## 环境变量

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `HYPER_PARALLEL_SHMEM_HEAP_SIZE` | 1073741824（1 GiB） | 每进程堆字节数，正整数；生命周期内固定，修改值随下一生命周期首次`acquire()`生效 |
| `HYPER_PARALLEL_SHMEM_BOOTSTRAP_ENDPOINT` | `tcp://127.0.0.1:8662` | Bootstrap endpoint for full Torch WORLD or non-distributed runs; subgroups use independent CANN unique IDs instead |
| `SHMEM_UID_SOCK_IFNAME` | CANN automatic selection | Optional host interface for subgroup UID bootstrap; it must be reachable by all group members across nodes |
| `HYPER_PARALLEL_SHMEM_TIMEOUT_SEC` | 120 | Runtime 超时秒数 |
| `HYPER_PARALLEL_SHMEM_DATA_ENGINE` | `mte` | 数据搬移引擎，首版仅支持 `mte` |
| `HYPER_PARALLEL_SHMEM_LOG_LEVEL` | 未设置即 `2` | Runtime 日志级别：`0`=Debug（最详细；`shmem.empty` 额外在 stderr 输出直接调用点，泄漏的 Allocation 可经 `allocation_base` 回溯到代码行）、`1`=Info、`2`=Error（仅错误）；非法值回退 `2` |

## 安全性

Subgroup UID bootstrap also uses unencrypted host sockets. Do not configure a
fixed `SHMEM_UID_SESSION_ID` for independent groups, since that bypasses unique
rendezvous allocation. The existing WORLD endpoint settings remain unchanged.

以下为 CANN `aclshmem` 实现层面的事实，部署前需据此评估威胁模型：

- bootstrap 引导走 `HYPER_PARALLEL_SHMEM_BOOTSTRAP_ENDPOINT` 指定的 TCP 端点；`aclshmemx_set_conf_store_tls(false, nullptr, 0)` 以未启用 TLS 的方式初始化 config-store。
- 对称 Heap 内任意 PE 可凭对称地址，经数据面 RMA 直接读写其他 PE 的堆；CANN 头文件对 `aclshmemx_mte_put_nbi` 等接口反复注明目标地址"must point to symmetric memory because it is translated to the corresponding address on pe"，PE 间无访问隔离。
- 数据面 MTE 传输本身无加密。

基于以上，本组件适用于可信内网、同一集群内 PE 互信的场景，不应用于跨信任域部署。

## 构建与制品

SHMEM 没有独立对外开关。启用 `--multicore on` 会构建 Torch Multicore 和必需的 SHMEM；
`--multicore off` 不交付二者的 native payload。内部 `shmem/build.sh` 由组件入口调用，
依次构建 AllGather kernel、Runtime 与 Torch binding。

私有库位于 `core/multicore/shmem/lib`，使用专属 SONAME，避免与框架自带的通用 SHMEM 库冲突。
制品不携带 RPATH/RUNPATH，运行时搜索路径由 Multicore `set_env.bash` 统一配置。详见
[构建与使用](build.md)。
