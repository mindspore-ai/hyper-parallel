# MegaKernel Multicore Profiling Design and Usage

---

## 1. 背景与目标

### 1.1 问题

MegaKernel 把多个通信与计算阶段融合进一个 Device kernel。CANN `msprof` 或 `torch.profiler` 能看到外层
kernel 的开始、结束和总耗时，但无法直接回答以下问题：

- 哪个 AIC/AIV worker 正在等待依赖；
- GMM、SwiGLU、Dispatch 和 Combine 在不同核上如何重叠；
- 哪个 expert 或阶段造成长尾；
- event trigger 与下游计算之间是否存在异常空洞。

因此需要在多核运行时内部采集 cycle 区间，并在 Host 侧恢复为可读的 timeline。

### 1.2 设计目标

| 目标 | 设计选择 |
|------|----------|
| 运行时启停 | 正常和 profiling RuntimeConfig 共用同一算子产物 |
| 保护关闭路径 | worker 在任务循环前一次分流，快路径不做逐 task 判断 |
| 通用接入 | 公共 record 只保存 task、stage、owner 和 cycle 等通用字段 |
| 业务可读性 | 阶段名来自 ComputeGraph 节点；业务模块只补充可选 owner 语义 |
| 低干扰采集 | Device 只写定长 record，名称解析和 JSON 生成放到 Host |
| 可独立使用 | 内部 trace 可直接打开，不依赖外层 profiler |
| 可联合分析 | 通过纯 Host 接口与原生 Chrome Trace 离线融合 |
| 诊断可信 | buffer 溢出、未知 SoC、错误对齐和 Host 近似对齐均显式报告 |

### 1.3 范围

当前实现包含 Torch 独立前端、MegaMoe 正反向自动接入、Chrome Trace schema v1 导出和离线 merge，
且不在 `mega_kernel_profile()` 内启动或包装 `torch.profiler`。采集、独立导出和
融合是三个彼此独立的阶段。

---

## 2. 总体架构

### 2.1 并列工具与公共层

profiling 与 L0/L1 dump 同属于 `diagnostics`，通过协议与组合接入，不相互继承或调用：

```text
diagnostics/
├── api.py                       # prepare / MegaKernelDiagnostics.launch
├── common/
│   ├── graph.py                 # 通用图契约与只读 task/stage/owner metadata
│   ├── runtime.py               # 普通 runtime 与惰性 metadata
│   ├── call.py                  # 完整 LaunchArguments 与完成/取消边界
│   ├── session.py               # DiagnosticSession 协议、进程互斥
│   └── schedule.py              # DiagnosticAction、schedule
├── profiler/
│   ├── api.py                   # profiling session、buffer 池、pending、窗口
│   ├── cycle_trace.py           # profiler 私有配置、容量、解析
│   └── trace_merger.py           # 离线融合
└── dump/                        # 独立的 L0/L1 工具；不导入 profiler
```

依赖方向是 `profiler → common ← dump`；公共层不导入具体工具或 MegaMoe。
`modules/<kernel>/diagnostics.py` 提供业务 owner 规则，动态 Tensor 视图 resolver 留在业务自己的
`tensor_dump.py`。通用 profiler 不认识 Expert、GMM 或 MoE 路由。

### 2.2 数据流

```text
plan：只创建 normal RuntimeConfig image
    │
每次调用检查 diagnostics.is_active
    ├─ False：直接调用原 native wrapper → mode=0 → ProcessFast
    └─ True：通用 launch → 唯一活动 session.prepare_call(context)
                               │
                   profiler 首次使用该 plan：
                   公共 metadata（惰性、只读）
                       → 私有 Host cfg：trace ID/owner + AIC/AIV capacity
                       → 私有 Device image：mode=1
                               │
                   per-invocation diagnostic_buffer
                               │
                   Device cycle records
                               │
                   step/window/pending limit：
                   NPU synchronize → D2H → parse
                               │
                   独立 trace → 可选离线 merge
```

每次只产生当前工具的一个句柄；不存在先准备 profile handle 再叠加或覆盖 dump 的过程。
共享的是 `diagnostic_buffer` ABI 参数位置，而不是同时写入同一内存。
两个工具的 session 当前互斥，均不改变业务 event counter、ready handshake、workspace 或 autograd 租约。

---

## 3. RuntimeConfig 与调用生命周期

### 3.1 普通配置与工具私有派生配置

| 配置 | 准备时机 | 关键内容 |
|---|---|---|
| normal | plan 构造 | diagnostic_mode=0；原有 task/queue 数据 |
| profiler 私有配置 | 首次选中 WARMUP/RECORD/RECORD_AND_SAVE | diagnostic_mode=1；trace ID、owner 与每核容量 |
| L1 dump 私有配置 | 每次选中 L1 调用 | diagnostic_mode=2；不使用 profiler 的 trace ID/容量 |

`prepare_mega_kernel_diagnostics()` 保存完整 Host 配置并创建一个普通 image，
不解析诊断 metadata、不执行 resolver、不计算 profile layout。
关闭或 action=NONE 时，业务直接走普通 wrapper，不进入 diagnostic launch，不创建诊断句柄。

首次选中 profiler 时，`_PreparedProfileRuntime` 复制 Host 配置，在副本中编码 trace 描述符、owner，
计算容量并上传独立的 profile image。普通 Host/Device 配置均保持不变。
session 按公共 runtime 对象缓存 profiler 配置，退出时释放该 session 的缓存。

公共 owner 是 `GraphDiagnosticMetadata.task_owner_ids`；`profile_owner_id` 只是 profiler 在私有配置中的
Device 编码，dump 不读取它。动态视图 resolver 和 dump 目录校验不会进入 profiler 准备路径。

### 3.2 Schedule 状态机

`schedule(skip_first, wait, warmup, active, repeat)` 对零起始 step 返回以下动作：

| Action | Device 采样 | 保留窗口数据 | 边界行为 |
|--------|-------------|--------------|----------|
| `NONE` | 否 | 否 | 使用关闭配置 |
| `WARMUP` | 是 | 否 | step 结束同步并丢弃 |
| `RECORD` | 是 | 是 | 保留调用 buffer，继续当前窗口 |
| `RECORD_AND_SAVE` | 是 | 是 | 完成 D2H、解析和窗口回调 |

`profiler.step()` 必须在 context 内调用，并且每个用户逻辑 step 只调用一次。离开 context 时，如果仍有
有效 `RECORD` 数据，会自动完成最近窗口。嵌套或并发的进程内 profiler context 会被拒绝。

### 3.3 每次调用的普通显存与资源复用

MegaMoe workspace 中正向、反向各保留一块按图容量分配的对称 `all_event_counters`。前缀承载调度计数器；
EP 大于 1 时，尾部还保存跨调用持续递增的 ready generation。每次 launch 只清零计数器前缀，不破坏 ready
generation。
进入采集 action 后，Torch profiler 按 `(device_id, buffer_size)` 从 session 私有池取得普通 NPU
`diagnostic_buffer`。每个尚未 drain 的 invocation 独占一块 buffer，因此同一 step 的多层正反向调用不会互相
覆盖；Device 直接把 record 写入最终 buffer，不再创建 D2D snapshot。

D2H 和解析只发生在：

1. warmup step 结束且数据需要丢弃；
2. active 窗口结束；
3. pending 调用数达到 `max_pending_calls`；
4. context 正常退出并仍有待完成数据。

drain 先同步 NPU；需要保留时直接把每块 buffer 搬到 Host 并解析，随后将 Device buffer 回收到当前 session
的池中，供后续 invocation 复用。异常退出时释放 pending 和池内显存，不生成容易误判的半成品 trace。
profiler context 退出后池随 session 销毁，不跨模型或下一次 context 长期持有显存。

---

## 4. Host/Device ABI 与 Buffer

### 4.1 RuntimeConfig 字段

Python `RuntimeConfigC` 与 AscendC `RuntimeHeader` 必须保持 64 Byte 固定头的字段顺序和宽度一致。Task、event
等数组按图的实际容量紧随固定头序列化；profiling 复用固定头的保留字段，不改变动态数组的布局：

| 字段 | 类型 | 说明 |
|------|------|------|
| `diagnostic_mode` | `uint32` | 0=普通/L0，1=profiling，2=L1 dump |
| `aic_profile_record_capacity` | `uint32` | 每个 AIC worker 的 record 容量 |
| `aiv_profile_record_capacity` | `uint32` | 每个 AIV worker 的 record 容量 |

每个 `TaskDesc` 额外包含：

| 字段 | 类型 | 说明 |
|------|------|------|
| `profile_desc_id` | `uint32` | 阶段语义；未配置时为 `0xFFFFFFFF` |
| `profile_owner_id` | `uint32` | 业务归属；未配置时为 `0xFFFFFFFF` |

公共 Device runtime 不理解 Expert、layer 等业务概念，只透传 ID。阶段名称和 owner label 只保存在 Host
metadata 中，不占用 Device record 的字符串空间。Host 按 `ComputeGraph.topological_sort()` 遍历节点，
使用每个节点的 `task_num` 还原其连续 TaskDesc 区间，并自动分配内部 descId；展示名优先取
`OperatorNode.diagnostic_name`，未设置时由节点 `name` 转成驼峰形式。descId 是 RuntimeConfig 内部标识，
调用方不应依赖具体数值。

### 4.2 Buffer 布局与内部 ABI

调度和 profiling 使用两块语义、内存类型及生命周期都独立的 buffer：

```text
all_event_counters（按图容量分配的对称内存）
├── scheduler event counters
└── ready generation tail（EP > 1）

diagnostic_buffer（普通 NPU 显存，仅采集 action 分配）
├── AIC slot 0
│   ├── 64-byte core header
│   └── N × 32-byte records
├── ...
└── AIV slots
```

每个 AIC/AIV worker 拥有独立 slot，避免全局 record allocator 和原子抢占。core header 保存 entry cycle、
有效/丢弃数量、core 类型、block ID 和容量；record 保存起止 cycle、`desc_id`、`task_id`、
`stage_task_index` 和 `owner_id`。profile slot 从 `diagnostic_buffer` 的 offset 0 开始，不占用 symmetric
heap。

`diagnostic_buffer` 是 profiling 与 L1 Tensor dump 平等使用的公共诊断入参；当前两种工具的 session 互斥。
profiling 模式写入 cycle trace，L1 dump 模式写入 Tensor 快照，普通执行与 L0 dump 不读取其诊断内容。
它通过内部 native ABI 传递：正反向 OpProto、OpDef、ACLNN GetWorkspaceSize、L0 和 Device
kernel 的输入列表都显式传递它，但 Python 公共 `mega_moe()`、`mega_moe_grad()` 和
`MegaMoeExperts` 签名保持不变。普通调用通过私有 adapter 补齐参数；profiling 关闭时用原
`all_event_counters` 作为未读取的占位 tensor，不产生新的申请、切片或数据搬运。

### 4.3 容量计算

Host 根据完整调度表复现每个 worker 可能执行的任务，分别计算最忙 AIC 和 AIV 的记录数：

```text
capacity = round_up(required_records, 16)
```

每个有依赖的普通 task 最多记录 WaitDependency、Compute 和 TriggerEvent 三段；SHMEM put 不重复记录公共
trigger 区间。每核容量须满足 Device ABI 的 uint32 stride 边界，超限在 Host 准备时明确报错。
Device 达到该调用分配的容量后停止写入新 record 并累计 dropped count，不覆盖既有数据。

令 `C = max(24, ceil(num_workers / 2))`，profile buffer 总容量为
`C * AIC_stride + 2 * C * AIV_stride`，不包含调度 event workspace。
Host 分配、Device 写入与 Host parser 共用当前执行计划的槽位容量；950 不固定为 24/48 核。
`max_pending_calls` 控制 Host session 的 buffer drain 频率，与这里的每核 record 容量无关。

### 4.4 Cycle 与时间换算

Device 使用 `GetSystemCycle()` 读取 system counter：

```text
timestamp_us = (start_cycle - anchor_cycle) / cycle_frequency_mhz
duration_us  = (end_cycle - start_cycle) / cycle_frequency_mhz
```

内部频率映射支持 Ascend 910B/910C/910_93 和 950 系列；A2/A3 使用 50 MHz，A5 使用 1000 MHz。
950PR/DT 子型号归一到 950 家族，不以 AI Core 主频代替系统计数频率。SoC 名称来自
`torch_npu.npu.get_device_name()`；未知型号会在解析采集结果时显式失败，不接受用户从前端传入猜测值。
Device 事件轮询也按相同代际使用 50 或 1000 cycles/µs，轮询间隔参数仍以实际微秒表示。

### 4.5 关闭态快路径

worker 在 ready handshake 之后、`Process()` 的任务循环之前读取 `DiagnosticMode`，并在进入任务循环前选择快慢实现。关闭路径具有以下约束：

- 不读取 system cycle；
- 不执行 profiling barrier；
- 不写 core header 或 record；
- 不读取 `profile_desc_id` / `profile_owner_id`；
- 只清零 `all_event_counters` 的调度计数器前缀，保留 ready generation 尾部；
- 不申请普通 NPU `diagnostic_buffer`，也不创建 event-counter view；内部占位参数不会被 Device 读取。

这样可避免把 profiling 判断放进每个 task 的热循环。

---

## 5. Device 采样语义

### 5.1 单个 Task 的区间

采样路径对一个 task 按实际执行顺序记录：

```text
WaitDependency → Compute → TriggerEvent
```

- task 没有依赖 event 时不产生 WaitDependency；
- task 没有业务 `profile_desc_id` 时，Compute 使用 `TASK_TYPE_BASE + task_type` 兜底；
- SHMEM put 的信号触发由自身计算核完成，不额外记录公共 TriggerEvent；
- 同一 worker 内 record 保持写入顺序，Host 导出时再按 cycle 稳定排序。

### 5.2 通用名称

公共 descId 至少覆盖 `WaitDependency` 和 `TriggerEvent`。图中的节点阶段会自动获得内部 descId；
Wait/Trigger 名称会携带当前 task 对应的图节点名称，例如 `GMM1_WaitDependency`。图外且没有阶段信息的
task 仍使用 task type 兜底名称。

### 5.3 简洁与详细名称

| `detailed_task_names` | 示例 | 适用场景 |
|-----------------------|------|----------|
| `False` | `GMM1` | 查看整体流水和通算掩盖 |
| `True` | `Expert3_GMM1_task72` | 定位 expert、worker 或具体 task 长尾 |

两种模式都在事件 `args` 中保留原始 task、desc、owner 和 cycle 字段；切换名称不会丢失原始数据。

---

## 6. MegaMoe 语义适配

### 6.1 正向阶段

| 阶段 | 典型执行资源 | 含义 |
|------|--------------|------|
| Dispatch | AIV / SHMEM | 将 token 分发到目标 expert |
| GMM1 | AIC | gate/up projection |
| SwiGLU | AIV | 激活计算 |
| GMM2 | AIC | down projection |
| Combine | AIV / SHMEM | 将 expert 输出返回源 rank |

### 6.2 反向阶段

| 阶段 | 含义 |
|------|------|
| DispatchGrad | 梯度分发 |
| ActGrad | 激活输入梯度 GMM |
| W2Grad | down projection 权重梯度 |
| SwiGLUGrad | 激活反向 |
| GateGrad | gate/up 输入梯度 GMM |
| W1Grad | gate/up 权重梯度 |
| CombineGrad | 梯度回传源 rank |

### 6.3 图驱动阶段语义

MegaMoe 的阶段顺序和 TaskDesc 区间直接取自生成 RuntimeConfig 的正反向 `ComputeGraph`。每个
`OperatorNode` 可就地设置 `diagnostic_name`；GMM1、GMM2、DispatchGrad 等业务名称与对应算子定义放在
一起。公共 metadata 在首次采集时按拓扑顺序解析；profiler 将 descId 写入自己的配置副本，不再通过 `task_type`、tiling position 或 input position 反向猜测阶段。

`modules/mega_moe/diagnostics.py` 只保留 `Expert` owner label 和全局 expert ID resolver。resolver 同时接收
`OperatorNode` 与其 TaskDesc，因此 Dispatch/Combine 等同类型算子可按节点身份区分。公共 GMM、SwiGLU 和
AllToAll task builder 不写 MegaMoe owner，模型用户也无需传阶段名、rank、device 或 RuntimeConfig。

### 6.4 正反向 launch 接入

```text
_MegaMoeFunction.forward / backward
    ├─ diagnostics.is_active=False → 原 mega_moe/mega_moe_grad wrapper
    └─ True → diagnostics.launch(runtime ABI parameters, direction, events)
                → profiler.prepare_call → 完整 profile 参数
                → native wrapper_with_diagnostic_buffer
                → complete / 异常 cancel
```

`direction` 区分正反向 invocation。成功 launch 后句柄标记完成，Device 直接写入本次独占的 buffer，
无需 D2D snapshot；drain 才同步与回读。异常取消只撤销本次 profiling 状态，模型用户无需传诊断参数。

---

## 7. Torch 公共接口

### 7.1 导入

```python
from hyper_parallel.core.multicore.diagnostics import profiler as multicore_profiler
```

具体工具从 `diagnostics.profiler` 导入；公共 `diagnostics` 不导入工具。项目为 Torch-only，不设 backend 分派。

### 7.2 公共 schedule

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

所有参数必须是非负整数，且 `active > 0`。返回对象接受零起始 step 并产生 `DiagnosticAction`。

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

| 参数 | 契约 |
|------|------|
| `schedule` | 可调用的 step schedule；省略时保留 context 内全部调用 |
| `on_trace_ready` | 每个窗口完成后调用 `callback(profiler)` |
| `detailed_task_names` | 控制展示名，不改变原始 ID |
| `max_pending_calls` | 正整数；控制中间 D2H drain 前同时保留的调用 buffer 上限 |

接口不接受 rank、device、cycle 频率、RuntimeConfig 或 event buffer。这些信息从实际 MegaKernel plan、输入
Tensor 和当前 Torch NPU device 自动解析。

### 7.4 step 与 export

```python
with multicore_profiler.mega_kernel_profile(schedule=capture_schedule) as profiler:
    train_step(batch)
    profiler.step()

trace = profiler.export_chrome_trace("rank0_mega_kernel_trace.json")
```

`export_chrome_trace()` 只导出最近完成的窗口。如果没有完整窗口会抛出 `RuntimeError`；它不会读取原生
framework trace，也不会隐式 merge。

---

## 8. 独立 Trace Schema

### 8.1 顶层结构

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

`traceEvents` 遵循 Chromium Trace Event Format。每个内部区间使用 complete event（`ph: "X"`），category
固定为 `MegaKernelInternal`；process/thread metadata 为每个 AIC/AIV worker 创建稳定 track。

### 8.2 Event args

每个 complete event 至少保留：

| 字段 | 说明 |
|------|------|
| `rank` / `device_id` | 数据来源 |
| `core_type` / `block_id` | AIC/AIV worker |
| `task_id` / `desc_id` | 原始任务和阶段标识 |
| `stage_task_index` / `task_number` | 阶段内零起始 ID 和一起始展示序号 |
| `start_cycle` / `end_cycle` | Device 原始 cycle |
| `core_entry_cycle` | 当前 worker 的入口 cycle |
| `invocation_id` / `step` / `direction` | launch、逻辑 step 和正反向归属 |
| `owner_id` | 可选业务归属，例如全局 expert ID |

### 8.3 Metadata 健康检查

顶层 `megaKernelCycleTrace` 汇总窗口级信息，`invocations` 保存每次 launch 的 kernel、direction、rank、device、
anchor 和 record 数。任何正式分析都应先检查：

```text
schemaVersion == 1
recordCount > 0
droppedRecordCount == 0
warnings == []
```

---

## 9. 离线 Trace 融合

### 9.1 接口

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

三个 trace 参数都可使用路径；`framework_trace` 也接受含 `traceEvents` 的对象或顶层 event list，
`mega_kernel_trace` 接受独立 trace 对象。融合结果会写入 `output` 并作为 dict 返回。

### 9.2 Invocation 分组

一个 active 窗口可以包含多层、多次正向和反向 launch。merger 先按 `args.invocation_id` 分组，再按每组
首次出现顺序与外层 kernel 一一配对。兼容没有 `invocation_id` 的旧单 invocation schema v1 trace，但拒绝
一部分事件有 ID、另一部分没有 ID 的歧义输入。

### 9.3 外层 Kernel 选择

默认流程：

1. 从内部 `kernelName` 生成允许下划线的大小写不敏感正则；
2. 只考虑持续时间为正的 complete event；
3. 优先选择具有 NPU/AICore/kernel 证据的 Device event；
4. 按外层时间排序，从 `kernel_index` 开始与 invocation 配对；
5. 自动名称未命中时，可按内部 span 与连续 Device kernel 的 duration 相似度降级选择并写 warning；
6. 显式 `kernel_pattern` 未命中时直接失败，不做名称降级；
7. 默认拒绝 Host wrapper，只有显式 `allow_host_wrapper=True` 才允许近似对齐。

CANN trace 中即使 category 为空，只要 `args["Task Type"]` 为 `MIX_AIC` 或 `MIX_AIV`，也可识别为
Device kernel。

### 9.4 时间与 Track 对齐

每组内部事件保持相对时间和持续时长不变，将最早内部事件平移到对应外层 kernel 的 `ts`。融合后的事件使用
外层 process ID，并为每个 `(core_type, block_id)` 分配不冲突的新 thread ID，名称为：

```text
<kernelName>/<coreType>/<blockId>
```

同时写入更靠前的 `thread_sort_index`，使 AIC/AIV 内部 track 靠近外层 NPU kernel 显示。

### 9.5 Merge 诊断

融合结果增加 `megaKernelTraceMerge`：

| 字段 | 用途 |
|------|------|
| `mergedInvocationCount` / `mergedEventCount` | 核对融合规模 |
| `usedFallbackKernelSelection` | 是否发生名称匹配降级 |
| `alignments` | 每个 invocation 选择的外层 pid/tid/时间和评分 |
| `exceedsOuterKernelUs` | 内部 span 超出外层 kernel 的时长 |
| `warnings` | 降级、Host 对齐和边界异常 |

merge 成功不等于对齐一定可信；调用方仍应检查以上诊断字段。

---

## 10. 使用方式

### 10.1 稳态正向采集

```python
from pathlib import Path

import torch
import torch.distributed as dist

from hyper_parallel.core.multicore.diagnostics import profiler as multicore_profiler, schedule as diagnostic_schedule

rank = dist.get_rank()
schedule = diagnostic_schedule(wait=0, warmup=0, active=1, repeat=1)

with torch.no_grad():
    # Context 外使用相同 shape 做真实 warmup。
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

使用与目标业务等价的 shape，能够避免小 shape 导致的 task 碎片化视图。所有 rank 在 active window 前完成
NPU synchronize 和 barrier，能够避免启动偏差被误认为 kernel 内部空洞。

### 10.2 训练窗口

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

同一个窗口中的正反向 launch 自动写入 `direction=forward/backward`。不要在每层或每个 MegaKernel launch 后
调用 `step()`。

### 10.3 每个窗口导出

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

### 10.4 与 msprof 结果融合

```bash
msprof --output=/path/to/dump python your_program.py
```

找到当前 rank/device 对应的原生 `trace.json` 后执行：

```python
from hyper_parallel.core.multicore.diagnostics import profiler as multicore_profiler

multicore_profiler.merge_chrome_traces(
    "/path/to/dump/.../trace.json",
    f"./traces/rank{rank}_mega_kernel_trace.json",
    f"./traces/rank{rank}_merged_trace.json",
)
```

多 rank 结果必须逐 rank 融合。不要把 rank 0 内部 trace 对齐到其他 rank 或 device 的外层 trace。

---

## 11. 新 MegaKernel 接入

### 11.1 默认图节点名称

接入 `ComputeGraph` 的新 MegaKernel 默认以 `OperatorNode.name` 生成阶段名，例如 `qkv_proj` 显示为
`QkvProj`。需要保留缩写或使用更易读的业务术语时，在节点定义旁设置 `diagnostic_name`。图外 task 仍可
显示 WaitDependency、TriggerEvent、TerminateTask 和 task-type fallback。多个节点可以使用相同展示名，
公共逻辑仍按拓扑位置为它们分配不同的内部 descId。

### 11.2 增加业务语义

推荐按以下顺序接入：

1. 使用同一份 `ComputeGraph` 按拓扑顺序构造 RuntimeConfig task，并保证每个节点的 `task_num` 与实际连续
   TaskDesc 区间一致；
2. 在需要自定义名称的 `OperatorNode` 上填写 `diagnostic_name`；其余节点直接复用 `name`；
3. 需要 owner 时，在 `modules/<kernel>/diagnostics.py` 实现接收节点、TaskDesc 和 topology 的 resolver；
   不属于任何 owner 的节点返回 `None`；
4. RuntimeConfig builder 只生成普通执行的 task/queue，不写诊断字段；
5. 在 plan 中调用 `prepare_mega_kernel_diagnostics(graph, runtime_config, ...)`，可显式调用
   `validate_mega_kernel_diagnostic_contract()` 检查图/task/ABI；诊断 metadata 与 profile image 自动惰性准备；
6. 在内部 OpProto、OpDef、ACLNN、L0、Device kernel 和框架 adapter 中贯通 `diagnostic_buffer` 输入，但保持
   模型用户接口不变；
7. 正反向先检查 `diagnostics.is_active`；关闭时直接走原 wrapper，开启时通过
   `with diagnostics.launch(...)` 提供完整参数并管理完成/取消；
8. 增加布局、metadata、Torch session、导出、merge 和真实 NPU 测试。

图节点是阶段身份的唯一来源，不再另建 TaskDesc 字段匹配表。若某个 builder 不能保证“按图拓扑顺序、每个
节点连续写入 task”，应先在通用构图/填充层显式记录每个节点的 task ID 区间，再应用图 metadata，不能退回
依赖 tiling 或输入槽位猜测。公共 task builder 不得写具体模型的 owner 语义。

### 11.3 ABI 变更要求

修改 header、record、容量或 core 类型时，必须同步更新：

- Python ctypes 与 AscendC struct；
- OpProto、OpDef、算子 JSON、ACLNN GetWorkspaceSize、L0 和框架 adapter 的输入顺序；
- Host buffer size 和 slot offset 计算；
- Device recorder；
- Host parser 和 schema 版本；
- 单元测试、NPU 完整性断言和本文档。

结构性变更必须升级相应 schema，不得让旧解析器静默接受不兼容布局。

---

### 11.4 Native ABI 与广播队列

只有调度依赖、在 C++ 内硬编码 Tensor 地址的 MegaKernel，可通过 `graph_adapter` 惰性补充真实端口。
回调放在 `modules/<kernel>/diagnostics.py`，接收原图和私有 Host 配置，返回诊断图；不能修改原图或普通
Device 配置。`bind_graph_tensor_ports()` 在图副本上设置 `diagnostic_host_ports`，此时端口来自 Host 描述，
不受 `TaskDesc.inputs/outputs` 各 4 项的序列化容量限制，也不会改写计算使用的 TaskDesc。
这些端口可用 `shape=[]`、`is_dynamic=True` 表示形状在调用时解析；导出文件的实际维度以 `actualShape`
和 `storedShape` 为准。导出 rank 使用 job-global rank，MoE 的 Expert ID/路由计算仍使用 EP 组内 rank，
二者通过业务 `owner_context` 区分，避免多个并行组都写入 `rank_0`。

普通队列以 `execution_mode="event_driven"` 接入；MoE 配对 AIV worker 使用 `vector_worker_stride=2`，
MHC 全 AIV worker 使用 `vector_worker_stride=1`。MegaGate 使用 `execution_mode="aiv_pipeline"`：每个
worker 在自己的 token 行分片上执行同一组 task ID，没有调度事件。Profiler 按完整阶段序列计算每核容量；
L1 按 `(task_id, worker_id)` 分配 record/slot，并由 Device 检查 worker ID，防止不同核写入同一份快照。
广播目录使用 `task_<id>_worker_<id>`，普通队列仍使用 `task_<id>`。

MHC 在 Torch wrapper 分配原有输出之后、发射之前绑定真实 ABI Tensor。MegaGate 正向在诊断开启时通过
`mega_gate_route_out` 预分配同样的 5 个输出；关闭时仍调用原 native wrapper。地址列表只在 Device L1 分支
临时构造，不给普通 worker 增加诊断地址数组或初始化工作。

| MegaKernel | L0 | L1 范围与边界 |
|---|---|---|
| MegaMoe | 按图采集调用前输入和调用后结果 | 动态 Expert/token 视图；pull 可采本地接收输出，不可将远端来源当作本地输入；GMM 协作输出改选已满足依赖的下游输入 |
| HyperMegaMhc | 真实 ABI 端口，而非调度图的工作区别名 | 支持连续 token tile；不把 CANN 环形 X-cast 工作区冒充 `hc_before_norm`，不暴露迭代优先布局的非连续 Sinkhorn cache |
| HyperMegaMhcGrad | 包括完成归约后的参数梯度 | 支持 token 局部输入/输出；多核协作累加的参数梯度只允许 L0，L1 必须筛选安全端口 |
| MegaGate Route | logits、bias、mask 和 5 个 native 结果 | 按真实 AIV 行分片采集可寻址的 GM 端口；内部 CANN 工作区不作为普通 Tensor 暴露 |
| MegaGate RouteGrad | 包括 ACLNN 后处理完成的 `grad_logits` | 采集 MegaKernel 内可寻址的输入；scatter/sqrt/softplus 后处理不是该 task 序列，最终梯度只属于 L0 |

RouteGrad 的后处理由一个 `task_num=0` 的 Host 边界节点描述。它可参加 L0 的来源/去向整理，但没有 Device
task ID，不生成 L1 record，也不会在 timeline 中伪造一个不存在的 MegaKernel task。

### 11.5 Tensor dump 的使用与存储

```python
from hyper_parallel.core.multicore.diagnostics import dump, schedule

config = dump.TensorDumpConfig("./dump", level="L0")
with dump.tensor_dump(config, schedule=schedule(wait=0, warmup=0, active=1, repeat=1)) as dumper:
    output = model(inputs)
    dumper.step()
```

L0 在每次选中的 native 调用之前复制外部输入，在调用完成之后复制中间/最终结果，不等整个模型结束才读取
原 Tensor。快照是独立 CPU `torch.Tensor`，由 dumper 的窗口调用记录持有；窗口结束时 `torch.save()` 为
`.pt`，保留 BF16 等原始 dtype。同一 L0 Tensor 的多个 stage 端口文件使用硬链接。对外只呈现 `stages/`
和 manifest；`before/after` 是记录采集时机的 metadata，不是必须暴露的目录。较长窗口会累积 CPU 内存，
目前没有独立 Host 总量预算，应控制 active step、rank、方向和 Tensor 筛选范围。
每次调用的句柄负责采集或解码其快照，成功后通过完成回调将调用记录交给 dumper；两者不互读私有状态。
失败时仍由句柄清理本次调用的数据，不向窗口提交半成品。
L0 保存的是调用边界上的实际存储值，不能还原原地更新或工作区复用前的历史中间值；这类需求应使用安全的
L1 task 端口，而不是根据 stage 文件名假定每个历史时刻都被保留下来。

L1 通过 `stages`、`task_ids`、`stage_task_indices`、`owner_ids`、`core_types/core_ids`、`tensor_names`
及 `io` 选择采集对象。静态图使用默认视图；动态或硬编码地址语义由业务 resolver 在每次 Kernel 发射前返回
`L1TensorView(source_offset_bytes, shape)`。Host 校验真实 ABI Tensor 的 dtype、连续性、完整范围后，
再执行可选截断和 payload 分配；Device 不执行 Python resolver。
静态连续切片会根据源偏移裁剪尾块；默认路径不把跨行的内维切片误当作连续字节段。此类非连续范围需要
业务 resolver 明确选择可复制的连续范围，不能仅改变 shape 后直接复制。

L1 Device buffer 使用版本 2：

```text
[0, 32)       Header：版本、有效条数、区域偏移和总大小
[32, 64)      对齐填充
[64, 262208)  Record table：4096 × 64 B，只有 record_count 条有效
[262208, ...) Payload：每条 record 对应独立 slot，slot 起点按 64 B 对齐
```

每条 record 的 10 个 uint32 字段描述 task、phase、ABI 源位置/偏移、目标 slot、字节数、状态、task index
和 worker ID，余下 24 B 对齐填充。record 和 payload slot 不共享写入者的数据缓存行；普通队列使用任意
worker 标记，广播队列使用具体 worker ID。buffer 随实际 payload 增长，固定 table 大小不是快照容量上限。
在构造 Host bytearray 和 Device 分配前检查预算（默认单次 512 MiB）及 32 位 buffer ABI 上限。

Device 等待原依赖后采 before，执行计算并完成 barrier 后采 after，最后发布原业务完成事件。采样使用通用
字节复制和缓存同步；诊断关闭时仍进入原 `ProcessFast`，不会初始化 trace recorder、扫描 record、解析视图、
分配快照区或进行 D2H。L1 会显著扰动执行时间，不能用开启 L1 时的耗时推断业务性能。

---

## 12. 测试与验收

### 12.1 CPU 单元测试

```bash
pytest -v \
  tests/ut/core/multicore/test_profiling.py \
  tests/ut/core/multicore/test_torch_mega_kernel_profiler.py \
  tests/ut/core/multicore/test_trace_merger.py
```

分别覆盖 layout/解析、schedule/session 生命周期和离线对齐诊断。额外的 workspace 与 Torch session
用例需要断言：关闭态不申请或切片 profile buffer；多个 pending invocation 使用不同 buffer；drain 后可以
按 device/size 复用；解析 offset 从 0 开始且不包含 event counters。

### 12.2 Torch NPU 测试

```bash
pytest -v tests/torch/multicore/test_mega_moe_profiling.py
```

正式验证至少应检查：

- profiler context 外完成同 shape warmup；
- warmup 后执行 NPU synchronize 和 rank barrier；
- 每个 rank 都生成独立 trace；
- `recordCount` 与预期相符且 `droppedRecordCount == 0`；
- cycle 合法，正反向 direction 正确；
- GMM1/GMM2 等关键阶段的 Wait/Trigger 名称存在；
- merged trace 的 invocation 数、外层 Device kernel 和 warnings 正确。

### 12.3 关闭态性能

关闭态性能必须比较“完全没有 profiling runtime 的基线”和“包含能力但 schedule action 为 `NONE` 的产物”。
测试应固定 workload、预热、采样轮数和 rank 聚合方式，并使用预先定义的回退门槛判断，而不是只比较一次均值。

### 12.4 当前回归结果

验收应覆盖各 MegaKernel 的普通正反向结果、Profiler trace 完整性、L0 原始 dtype/硬链接、L1 task/worker
隔离和 payload 数值，以及失败清理、调用计数器和缓存 plan 的隔离。CPU 模拟的规划/解码测试不能替代 NPU
执行；功能回归也不替代关闭态性能 A/B。

---

## 13. 约束与故障排查

| 现象 | 检查项 |
|------|--------|
| `no completed ... window` | active 窗口是否结束，是否调用了 `step()` 或正常退出 context |
| trace 为空 | active step 内是否真的执行 MegaKernel；是否只有 warmup 数据 |
| timeline 零散 | shape 是否过小；是否完成真实 warmup、NPU synchronize 和 rank barrier |
| 未知 SoC | 当前 device name 是否在 system counter 频率映射中 |
| dropped record | 检查实际任务计划与每核分配容量、Host/Device 布局是否一致；不能用 `max_pending_calls` 修复 |
| merge 无匹配 | 原生 trace 是否含 Device kernel；检查 pattern、index 和 pid |
| fallback warning | 复核 duration 降级选择的外层 kernel 是否正确 |
| Host wrapper warning | 仅在明确设置 `allow_host_wrapper=True` 时接受近似结果 |
| 多层 trace 配错 | 检查 invocation 顺序是否与原生外层 kernel 顺序一致 |
| context 冲突 | 同一进程只允许一个非嵌套 active session |

---

## 14. 相关文档

- [MegaMoe Design and Usage](architecture.md)
- [MegaMoe Torch 示例](../examples/mega_moe/README.md)
