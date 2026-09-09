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

- 仅支持覆盖整个 distributed world 且 rank 顺序一致的 group；`None` 选择 WORLD。
- 每次成功`acquire()`都必须对应一次`release()`。首个引用解析Root并初始化Native Runtime；后续等价Root
  只增加引用，不重复初始化；非最后一个`release()`只减少引用。
- 最后一个引用释放前必须完成所有相关backward和设备操作、释放全部对称Allocation，并保持初始化时冻结的
  NPU为当前设备。关闭前置条件失败时最后一个引用保留，可修正条件后重试`release()`。
- 干净关闭后同一进程可再次`acquire()`开启新生命周期，堆配置在新生命周期首次获取时重新生效。
  Native初始化失败不产生引用，可再次尝试；Native关闭失败会使进程进入不可恢复状态，之后拒绝再次获取。
- shutdown 时仍有存活分配会导致关闭失败；所有 rank 必须以一致顺序完成关闭，再销毁 HCCL 进程组。

## 接口契约

所有操作接口都要求调用方持有一个尚未释放的引用，并且不得与最后一个`release()`并发。Allocation、free、barrier和
AllGather等collective路径必须由全world按一致顺序调用；单边Put/Get/Signal均为stream入队语义
（返回仅代表入队，完成需对stream同步），按[单边通信模式](#单边通信模式)的协议调用：

- `shmem.empty(*size, dtype=None, alignment=None)`：从对称 heap 分配连续 Tensor；
  `alignment` 为可选的分配基址字节对齐。
- `shmem.free(tensor)`：在 stream 静止后释放完整分配，view 不能独立释放。重复 `free`、
  释放上一生命周期的 Tensor 都会被拒绝；把已释放的 Tensor 传入任何 SHMEM 操作
  （put/get/signal/wait_signal/all_gather）同样会被确定性拒绝。但 `free` 只是簿记：映射与数据
  原样保留，Tensor 的 shape、storage 容量和 data pointer 刻意保持不变，因此 stale Tensor
  "看起来还能用"——纯本地 Torch 误用（如 `x += 1`）不做运行期防护，会静默写入已归还的堆块。
  堆分配器按 best-fit 复用归还块，下一个同尺寸 `empty` 必然拿到同一地址：stale 引用将确定性地
  别名并破坏新分配，且破坏经后续 RMA 扩散到对端。调用方必须在 `free` 时丢弃全部引用。
  既定使用模式：对称 Tensor 一次性分配（通常首次使用时）、关闭时一次性释放，全 rank 按一致顺序
  collective 调用；不支持高频分配释放循环。
- `shmem.barrier(blocking=True)`：在当前 NPU stream 上 enqueue world barrier。默认阻塞：
  返回前同步当前 stream，即 barrier 与该流上所有更早工作均已完成，`free` 前调用一次即可；
  `blocking=False` 时返回仅代表入队，host 不等待完成，后续操作必须提交到同一 stream
  才能安全地排在 barrier 之后。
- `shmem.put(remote_dst, local_src, target_pe)`：单边 Put，把本地连续 NPU Tensor 按字节写入
  target PE 的对称地址（Tensor 或 view），两侧字节数必须一致。
- `shmem.get(local_dst, remote_src, source_pe)`：单边 Get，把 source PE 对称地址的字节读入
  本地连续 NPU Tensor。
- `shmem.signal(remote_signal, value, target_pe, operation="set")`：对 target PE 的一个
  `int32` 对称信号执行 `set` 或 `add`。
- `shmem.wait_signal(signal, value, comparison="eq")`：等待本地 `int32` 对称信号满足比较条件，
  `comparison` 支持 `eq`/`ne`/`gt`/`ge`/`lt`/`le`。
- `shmem.all_gather(output, input)`：enqueue world AllGather，`output` 必须为对称 Tensor，
  槽位按 world rank 顺序；零字节 `input` 校验后为 no-op（不发射 kernel 与 barrier）。
  首版以接口层交付，组件内暂无消费者。
- `shmem.debug_state()`：返回本进程 Runtime 状态、生效配置（heap、timeout、engine、endpoint）、
  尚未配对`release()`的本地`reference_count`（各Rank可能不同）、堆占用（`allocated_bytes`已用请求字节、
  `remaining_bytes`可用字节、
  `max_allocated_bytes` 峰值）、活跃分配表（每项含 `allocation_id`/`allocation_base`/
  `allocation_bytes`，按 id 排序，可用 `tensor.data_ptr()` 对照 base 区间定位所属分配）与
  最近一次失败的只读快照，用于诊断；REPL 中逐行渲染便于阅读；
  干净关闭后生命周期相关字段均为 None。
  堆占用按请求字节投影：CANN 堆按 16 字节对齐取整并切分余洞，真实设备侧占用可能略高。

Runtime 在初始化时冻结当前 NPU 设备；在其他设备的 stream 上调用 `empty`/`barrier` 会被拒绝。

## 单边通信模式

单边接口组合成两种数据面模式，`signal`/`wait_signal` 提供跨 PE 的完成通知：

- **Push（推）**：发送方 `put` 把数据直接写进接收方的对称槽，再 `signal` 通知；接收方
  `wait_signal` 后读自己的本地对称槽。数据落点在接收方，适合接收方持有消费缓冲的场景。
- **Pull（拉）**：发送方把数据写进自己的对称槽（本地写或收到的 put），`signal` 通知接收方；
  接收方 `wait_signal` 后 `get` 从发送方的对称槽拉取。数据落点在发送方，适合发送方
  复用同一缓冲多轮发布的场景。

两种模式都必须遵守两条协议约束，均来自 CANN 数据面的既定行为：

1. **信号槽 64B 隔离**：并发写入的不同信号槽至少相距 64 字节（每个信号槽独占一个
   `alignment=64` 的分配，或在一个分配内按 64B 步长排布），否则共享 cacheline 的信号
   更新不保证互不覆盖。
2. **跨 rank 初始化收敛**：任何远端可见写（`signal`、`put` 到对端槽位）之前，所有 rank
   必须完成对相应槽位的本地初始化（`zero_()`/`fill_()`）并经 `barrier()` 收敛；否则一个
   迟到的本地初始化会覆盖已送达的远端写，表现为信号永远等不到或数据读回全零。

## 环境变量

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `HYPER_PARALLEL_SHMEM_HEAP_SIZE` | 1073741824（1 GiB） | 每进程堆字节数，正整数；生命周期内固定，修改值随下一生命周期首次`acquire()`生效 |
| `HYPER_PARALLEL_SHMEM_BOOTSTRAP_ENDPOINT` | `tcp://127.0.0.1:8662` | 引导端点；一个生命周期内各 rank 相同，重建生命周期时更换端口 |
| `HYPER_PARALLEL_SHMEM_TIMEOUT_SEC` | 120 | Runtime 超时秒数 |
| `HYPER_PARALLEL_SHMEM_DATA_ENGINE` | `mte` | 数据搬移引擎，首版仅支持 `mte` |
| `HYPER_PARALLEL_SHMEM_LOG_LEVEL` | 未设置即 `2` | Runtime 日志级别：`0`=Debug（最详细；`shmem.empty` 额外在 stderr 输出直接调用点，泄漏的 Allocation 可经 `allocation_base` 回溯到代码行）、`1`=Info、`2`=Error（仅错误）；非法值回退 `2` |

## 安全性

以下为 CANN `aclshmem` 实现层面的事实，部署前需据此评估威胁模型：

- bootstrap 引导走 `HYPER_PARALLEL_SHMEM_BOOTSTRAP_ENDPOINT` 指定的 TCP 端点；`aclshmemx_set_conf_store_tls(false, nullptr, 0)` 以未启用 TLS 的方式初始化 config-store。
- 对称 Heap 内任意 PE 可凭对称地址，经数据面 RMA 直接读写其他 PE 的堆；CANN 头文件对 `aclshmemx_mte_put_nbi` 等接口反复注明目标地址"must point to symmetric memory because it is translated to the corresponding address on pe"，PE 间无访问隔离。
- 数据面 MTE 传输本身无加密。

基于以上，本组件适用于可信内网、同一集群内 PE 互信的场景，不应用于跨信任域部署。

## 构建与制品

SHMEM 没有独立对外开关。启用 `--multicore on` 会构建 Torch Multicore 和必需的 SHMEM；
`--multicore off` 不交付二者的 native payload。内部 `shmem/build.sh` 由组件入口调用，
依次构建 AllGather kernel、Runtime 与 Torch binding。

私有库位于 `core/multicore/shmem/lib`，使用专属 SONAME 和相对 RUNPATH，
避免与框架自带的通用 SHMEM 库冲突。详见 [构建与使用](build.md)。
