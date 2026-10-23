# Multicore 私有 SHMEM

SHMEM是Multicore内部的对称内存与单边通信组件。它管理进程级Runtime和对称Allocation，提供Put、Get、
Signal、Barrier与AllGather等通信机制。

## 1. 组件定位与架构

### 1.1 依赖方向

SHMEM位于调用方与CANN实现之间，依赖方向如下：

```text
Multicore调用方
    → SHMEM Python控制面
        → SHMEM Native Runtime
            → CANN Adapter
                → Device Data Plane
```

Python控制面是资源生命周期和通信语义的边界；Native Runtime负责将稳定的内部接口映射到CANN，Device Data Plane
负责实际的数据搬移与设备同步。

代码层次如下：

```text
core/multicore/shmem/
├── _runtime.py       # Native与Torch模块的惰性访问
├── _lifecycle.py     # Runtime、Root、引用生命周期和Runtime restart
├── _api.py           # Allocation和通信接口
├── _debug.py         # Runtime只读诊断
└── ccsrc/
    ├── bindings/     # Torch binding
    ├── runtime/      # Native Runtime
    ├── cann/         # CANN Adapter
    ├── data_plane/   # 数据面实现
    ├── device/       # Device公共定义
    └── ops/          # AllGather等设备算子
```

### 1.2 组件职责

- 管理进程内唯一活动Runtime的初始化、引用计数、Root成员关系、关闭和terminal状态。
- 冻结并校验Runtime使用的NPU设备、Root坐标和Native配置。
- 管理对称Allocation的创建、释放、登记和合法性检查。
- 提供单边通信、信号、Barrier和AllGather，并定义其stream与Host完成语义。
- 执行调用方已经决定的Runtime restart事务，包括资源覆盖校验、quiesce、shutdown、bootstrap、
  initialize以及有序的Allocation释放和恢复。
- 提供Runtime、Native配置和活动Allocation的只读诊断。

### 1.3 接口边界

调用方通过SHMEM边界提供：

- Root ProcessGroup、目标NPU设备和物理heap字节数。
- 对称Allocation的shape、dtype和alignment。
- 单边通信的本地Tensor、远端对称地址和Root内PE编号。
- collective调用顺序、stream依赖和资源释放时机。
- Runtime restart所需的目标heap字节数、完整Allocation替换清单及业务状态预检钩子。

SHMEM在该边界内维护Runtime引用、Root身份、Native配置和Allocation登记，并返回对称Tensor、通信入队结果、
Runtime状态或restart阶段耗时。调用方持有自身资源描述，并在SHMEM操作成功后提交对应的上层状态。

### 1.4 演进准则

1. 新能力应使用Runtime、Root、Allocation、PE、stream和数据搬移等SHMEM概念定义输入与结果。
2. 生命周期接口应明确Root参与范围、引用与Allocation所有权、完成语义以及失败后的可用状态。
3. Runtime restart接口保持为物理heap目标、完整Allocation替换清单和业务状态快照组成的事务接口。
4. `debug_state()`保持为Runtime、Native配置和Allocation的事实快照。
5. 数据面接口应明确PE坐标系、对称地址范围、跨rank调用顺序和stream完成语义。
6. Python控制面、Native Runtime、CANN Adapter和Device Data Plane分别在所属层级验证其合同。

## 2. Runtime模型与生命周期

### 2.1 Root、world与PE坐标

本文中的**world**是当前获取的CANN SHMEM Root group，可能小于Torch WORLD。PE编号使用该Root group的
局部rank，不是Torch global rank。

- `acquire(None)`选择Torch WORLD；显式EP group可以是非连续子组。
- 不相交的PP或DP group使用各自group-local CANN unique ID独立bootstrap。
- 初始化、Allocation顺序、复用Barrier和最终释放只涉及Root成员，不要求无关WORLD成员参与。
- 一个进程只支持一个有序的活动membership。等价group handle可以共享Runtime；Root重叠但成员或顺序不同的
  group会被拒绝。

### 2.2 引用生命周期

SHMEM通过配对接口管理进程内唯一Runtime：

```python
from hyper_parallel.core.multicore import shmem

shmem.acquire()  # 首个引用初始化Runtime
try:
    ...
finally:
    shmem.release()  # 最后一个引用关闭Runtime
```

- 每次成功`acquire()`必须对应一次`release()`。
- 首个引用解析Root并初始化Native Runtime；后续等价Root只增加引用，不重复初始化。
- 非最后一个`release()`只减少引用。最后一个引用负责关闭Runtime。
- `root_members()`返回Root的Torch global ranks，不能把该结果当作CANN PE编号。

### 2.3 关闭与失败状态

最后一个引用释放前，调用方必须完成相关backward和设备操作、释放全部对称Allocation，并保持初始化时冻结的
NPU为当前设备。所有Root rank必须按一致顺序关闭，再销毁HCCL ProcessGroup。

- 关闭前置条件失败时保留最后一个引用，调用方修正条件后可以重试`release()`。
- Native初始化失败不产生引用，可以重新尝试。
- Native关闭或Runtime restart的破坏性阶段失败会将进程置于terminal状态，之后拒绝新的操作和引用。
- 干净关闭后，同一进程可以再次`acquire()`开启新生命周期；heap配置在新生命周期首次获取时重新生效。

## 3. 接口合同

所有操作都要求调用方持有尚未释放的Runtime引用，并且不得与最后一个`release()`并发。SHMEM只跟踪通过
`shmem.empty()`创建的对称Allocation；调用方负责确保使用这些Allocation的计算已经完成，再释放Allocation、
关闭Runtime或销毁ProcessGroup。

### 3.1 对称内存

- `shmem.empty(*size, dtype=None, alignment=None)`：从对称heap分配连续Tensor；`alignment`是可选的
  分配基址字节对齐。全world必须以一致顺序调用。
- `shmem.free(tensor)`：在相关stream静止后释放完整Allocation，view不能独立释放。成功后本地共享Storage
  缩为0，data pointer清空，原Tensor和全部view均失效，但shape元数据保留。

重复`free`、释放上一生命周期的Tensor或把失效Tensor传给SHMEM操作都会被拒绝。分配器可能复用已释放地址，
调用方仍应在`free`后丢弃原Tensor及全部view。既定模式是首次使用时分配、关闭时释放，不支持高频分配释放循环。

### 3.2 同步与集合操作

- `shmem.barrier(blocking=True)`：在当前NPU stream上入队world barrier。默认在返回前同步当前stream；
  `blocking=False`只保证入队，后续依赖操作必须提交到同一stream。全world按一致顺序调用阻塞Barrier，可以在
  `free()`前收敛各rank当前stream上的既有工作。
- `shmem.host_barrier()`：执行Host同步的HCCL world barrier，不经过SHMEM设备Barrier。返回时，各rank当前
  stream上早于本调用入队的工作已经完成；其他stream不在保证范围内。单PE场景为no-op。
- `shmem.all_gather(output, input)`：执行world AllGather，`output`必须为对称Tensor，槽位按world rank排列。
  零字节输入完成校验后为no-op。接口不隐含Barrier：gather前用`host_barrier()`确保各rank的input和对称output
  已经就绪；本rank stream完成后再次用`host_barrier()`确保全rank输出可读，然后才能覆写input或释放output。

Allocation、free、Barrier和AllGather等collective路径必须由全world以一致顺序调用。

### 3.3 单边通信

- `shmem.put(remote_dst, local_src, target_pe)`：把本地连续NPU Tensor按字节写入target PE的对称地址；
  源和目标字节数必须一致。
- `shmem.get(local_dst, remote_src, source_pe)`：从source PE的对称地址读取到本地连续NPU Tensor。
- `shmem.signal(remote_signal, value, target_pe, operation="set")`：对target PE的一个`int32`对称信号执行
  `set`或`add`。
- `shmem.wait_signal(signal, value, comparison="eq")`：等待本地`int32`对称信号满足`eq`、`ne`、`gt`、
  `ge`、`lt`或`le`条件。

Put、Get和Signal均为stream入队语义，返回不表示远端消费已经完成。调用方必须通过同一stream顺序、Signal协议或
显式同步建立完成关系。

### 3.4 设备归属

Runtime初始化时冻结当前NPU设备。`empty`、`barrier`、`signal`、`wait_signal`以及实际发射设备任务的非零
`put`、`get`和`all_gather`会拒绝其他设备的stream。零字节Put、Get和AllGather完成本地参数校验后不发射任务。

### 3.5 诊断

`shmem.debug_state()`只返回Runtime和Allocation事实：

- Runtime状态、Root、引用数和生效配置。
- heap的请求字节占用、剩余字节和峰值请求字节。
- 按Allocation ID排序的活动Allocation及其base和bytes。
- Tensor引用丢失但未经`free()`的泄漏Allocation。
- 最近一次失败的只读快照。

干净关闭后生命周期字段为`None`。占用量按请求字节投影；CANN heap按16字节对齐并切分余洞，真实设备侧占用
可能略高。

## 4. Runtime restart

### 4.1 SHMEM restart机制

SHMEM lifecycle负责进程级Runtime引用、Root一致性和安全restart。内部调用方提交目标heap字节数、完整的
Allocation替换清单和只读业务状态预检钩子；lifecycle验证这些Allocation完整覆盖活动Runtime后，按以下顺序
执行集体事务：

```text
preflight
→ quiesce
→ release allocations
→ shutdown
→ bootstrap
→ initialize
→ restore allocations
→ commit
```

SDK bootstrap是全局状态，必须先finalize旧Runtime，再获取fresh unique ID。restart保留逻辑引用计数和
HCCL ProcessGroup，只在Root成员间collective，不要求无关WORLD成员参与。

restart前必须确认：

- 发起者是Runtime的唯一user，替换清单恰好覆盖全部活动Allocation。
- 各Root rank的业务状态和Allocation manifest只包含确定性的全局事实，不包含本地地址或timing。
- 没有活跃调用或graph capture阻止Allocation失效。
- 调用方已经停止可能访问旧物理Storage的计算。

资源释放前的失败保留原Runtime和Allocation。进入破坏性阶段后的失败将Runtime标记为terminal。
vendor collective卡住和进程退出仍依赖通信库及launcher超时。

### 4.2 调用方集成合同

调用方可以通过`acquire(root_group, heap_size_bytes=...)`指定初始物理heap。活动Runtime上的后续引用只能请求
不超过当前heap的大小。

需要改变物理heap时，内部调用方先确定目标字节数，并为每个活动Allocation声明当前完整Tensor和目标shape、dtype、
alignment。SHMEM统一收敛业务状态与物理manifest，释放旧Allocation并按声明顺序创建新Tensor。restart成功后，
旧物理Tensor Storage及其view失效；调用方负责将返回Tensor绑定到稳定的业务对象，再提交自身状态。逻辑Runtime
引用和HCCL ProcessGroup在restart期间保持不变。

## 5. 单边通信协议

### 5.1 Push与Pull

- **Push**：发送方通过`put`把数据写入接收方对称槽，再通过`signal`通知；接收方`wait_signal`后读取自己的
  本地对称槽。该模式适合接收方拥有消费buffer的场景。
- **Pull**：发送方把数据写入自己的对称槽并通知接收方；接收方`wait_signal`后通过`get`拉取。该模式适合
  发送方复用同一buffer多轮发布的场景。

单边通信省去对端配对的接收调用，但不改变物理链路带宽，也不自动提供跨PE完成关系。

### 5.2 协议约束

1. **Signal热点的64 B隔离**：多个来源PE或设备任务并发更新同一目标PE上的多个Signal位置时，这些位置不能
   共享64 B cacheline。热点消费者应让每个Signal独占一个`alignment=64`的Allocation，或在同一Allocation中
   按64 B步长排布。不存在邻居并发写的单个Signal不需要额外的64 B基址对齐。
2. **跨rank初始化收敛**：任何远端可见写之前，所有rank必须完成相应槽位的本地初始化，并通过
   `host_barrier()`收敛。迟到的`zero_()`或`fill_()`会覆盖已经送达的远端写，表现为Signal永久等待或数据全零。
3. **不要用设备Barrier替代初始化收敛**：`shmem.barrier()`的远端等待可能无超时自旋，互连竞争下会触发
   AICore 507015。

## 6. 配置与日志

| 变量 | 默认值 | 语义 |
| --- | --- | --- |
| `HYPER_PARALLEL_SHMEM_HEAP_SIZE` | 1073741824（1 GiB） | `acquire()`未显式指定heap大小时使用的默认字节数 |
| `HYPER_PARALLEL_SHMEM_BOOTSTRAP_ENDPOINT` | `tcp://127.0.0.1:8662` | 普通WORLD或单进程初始化endpoint；子组及显式heap初始化使用独立unique ID |
| `SHMEM_UID_SOCK_IFNAME` | CANN自动选择 | 子组UID bootstrap使用的Host网卡，所有group成员必须可达 |
| `HYPER_PARALLEL_SHMEM_TIMEOUT_SEC` | 120 | Runtime超时秒数 |
| `HYPER_PARALLEL_SHMEM_DATA_ENGINE` | `mte` | 数据搬移引擎，当前只支持`mte` |
| `HYPER_PARALLEL_SHMEM_LOG_LEVEL` | 2 | `0`=Debug、`1`=Info、`2`=Error；非法值回退到2 |

底层日志由Runtime转发到stderr并使用同一日志级别过滤。Debug级别下，`shmem.empty`额外记录直接调用点，
可以结合`allocation_base`定位泄漏来源。初始化失败时，`source=shmem`日志保留底层原因。

## 7. 安全边界

Subgroup UID bootstrap使用未加密的Host socket。不要为相互独立的group设置固定`SHMEM_UID_SESSION_ID`，否则会
绕过独立的rendezvous分配。

CANN `aclshmem`具有以下安全属性：

- config-store通过`HYPER_PARALLEL_SHMEM_BOOTSTRAP_ENDPOINT`指定的TCP endpoint引导，当前使用
  `aclshmemx_set_conf_store_tls(false, nullptr, 0)`，未启用TLS。
- Root内任意PE可以凭对称地址，通过RMA访问其他PE的heap，不提供PE间访问隔离。
- MTE数据面传输不加密。

因此，本组件只适用于可信内网和同一集群内PE互信的场景，不应用于跨信任域部署。

## 8. 构建与制品

SHMEM没有独立对外构建开关。`--multicore on`会构建Torch Multicore及其必需的SHMEM；`--multicore off`
不交付两者的Native payload。内部`shmem/build.sh`依次构建AllGather kernel、Runtime和Torch binding。

A2/A3 使用 `ascend910b,ascend910_93` 构建目标，对应 Ascend910B backend；
A5 使用 `ascend950`，对应 Ascend950 backend，要求所选 CANN 提供 `libhcomm.so`。
950 的 AllGather 设备代码使用 dav-c310-vec，Runtime 初始化时检查当前设备与构建 backend 是否匹配，
不匹配时明确报错并给出重建目标。一次组件构建只选择其中一代，不将两代 SDK 或 native 制品混用。
当前数据搬移使用 MTE；选用 A5 backend 不表示自动启用 UDMA 或跨节点传输。

私有库位于 `core/multicore/shmem/lib`，使用专属 SONAME，避免与框架自带的通用 SHMEM 库冲突。
制品不携带 RPATH/RUNPATH，运行时搜索路径由 Multicore `set_env.bash` 统一配置。详见
[构建与使用](build.md)。
