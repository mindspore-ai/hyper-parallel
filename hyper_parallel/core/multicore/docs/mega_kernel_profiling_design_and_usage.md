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
| 业务可读性 | 每个 MegaKernel 在自己的 Host 规则文件中声明阶段和 owner 语义 |
| 低干扰采集 | Device 只写定长 record，名称解析和 JSON 生成放到 Host |
| 可独立使用 | 内部 trace 可直接打开，不依赖外层 profiler |
| 可联合分析 | 通过纯 Host 接口与原生 Chrome Trace 离线融合 |
| 诊断可信 | buffer 溢出、未知 SoC、错误对齐和 Host 近似对齐均显式报告 |

### 1.3 范围

当前实现包含 Torch 独立前端、MegaMoe 正反向自动接入、Chrome Trace schema v1 导出和离线 merge。
不实现 MindSpore 前端，也不在 `mega_kernel_profile()` 内启动或包装 `torch.profiler`。采集、独立导出和
融合是三个彼此独立的阶段。

---

## 2. 总体架构

### 2.1 分层

| 层次 | 主要文件 | 职责 |
|------|----------|------|
| 公共 API | `core/multicore/profiler.py` | schedule、context 和 merge 统一入口 |
| Torch session | `core/multicore/profiler.py` | active session、快慢配置选择、普通显存 buffer 池、D2H 和窗口导出 |
| Host profiling runtime | `core/multicore/profiling.py` | 容量计算、SoC 频率、buffer 解析和 Chrome Trace 构造 |
| Kernel 语义适配 | `core/multicore/modules/<kernel>/profiling.py` | 阶段匹配、展示名和 owner 规则 |
| MegaMoe 调用链 | `modules/mega_moe/plan.py`、`function.py` | 准备双 RuntimeConfig，并在正反向 launch 前接入 session |
| Host/Device ABI | `scheduler/config.py`、`ops/runtime/runtime_config.hpp` | 固定宽度配置和 task profiling 元数据 |
| Device recorder | `ops/runtime/cycle_trace_recorder.h` | 每核 header、record 和 dropped 计数 |
| Device worker | `ops/runtime/worker_kernel.h` | 快慢路径分流及 wait/compute/trigger 采样 |
| 离线融合 | `core/multicore/trace_merger.py` | invocation 分组、外层 kernel 选择、时间对齐和诊断报告 |

### 2.2 数据流

```text
RuntimeConfig builder
    ├─ normal RuntimeConfig: cycle_profiling_enabled = 0
    └─ profiled RuntimeConfig: enable = 1 + AIC/AIV capacity
                              │
                              ▼
MegaMoe forward / backward launch
    └─ _prepare_mega_kernel_call(direction)
        ├─ no active capture ─► normal RuntimeConfig + 4096 B event counters
        │                        └─ Device fast path；不申请/读取 profile buffer
        └─ active capture ────► profiled RuntimeConfig
                                + per-invocation ordinary NPU profile_buffer
                                           │
                                           ▼
                                Device direct-writes cycle records
                                           │
                         step/window boundary or pending limit
                                           ▼
                              NPU synchronize + direct D2H + parse
                                           │
                         ┌─────────────────┴─────────────────┐
                         ▼                                   ▼
                 standalone trace JSON             merge with outer trace
```

MegaMoe 正反向在 launch 前调用公共内部适配器。Profiler hook 根据当前 session 选择本次 launch 的
RuntimeConfig，并在采集开启时取得内部 `profile_buffer`；调度 event counters 仍是原来的固定 buffer。
这些参数只穿过内部 native 调用链，不改变 `MegaMoeExperts` 等业务接口。

### 2.3 目录关系

```text
hyper_parallel/
├── core/multicore/
│   ├── profiler.py
│   ├── profiling.py
│   ├── trace_merger.py
│   ├── scheduler/config.py
│   ├── modules/mega_moe/
│   │   ├── profiling.py
│   │   ├── plan.py
│   │   └── function.py
│   └── ops/runtime/
│       ├── runtime_config.hpp
│       ├── cycle_trace_recorder.h
│       └── worker_kernel.h
tests/
├── ut/core/multicore/
│   ├── test_profiling.py
│   ├── test_torch_mega_kernel_profiler.py
│   └── test_trace_merger.py
└── torch/expert_parallel/
    ├── test_mega_moe_profiling.py
    └── _test_mega_moe_profiling.py
```

---

## 3. RuntimeConfig 与调用生命周期

### 3.1 双配置模型

每个 shape-specific plan 保留普通配置，并按需准备开启 profiling 的配置：

| 配置 | 用途 | 关键字段 |
|------|------|----------|
| normal RuntimeConfig | context 外、`NONE` action 和关闭态快路径 | `cycle_profiling_enabled = 0` |
| profiled RuntimeConfig | `WARMUP`、`RECORD`、`RECORD_AND_SAVE` | enable 为 1，并写入 AIC/AIV 容量 |

两个配置按以下时机创建和使用：

1. 构造 MegaMoe plan 时，只把 `cycle_profiling_enabled` 设为 0 的 RuntimeConfig 序列化成
`normal_tensor`；此时 `profile_tensor` 还不存在。
2. 第一次执行 `WARMUP`、`RECORD` 或 `RECORD_AND_SAVE` action 时， `_prepare_mega_kernel_call()` 首次访问 `profile_tensor`。该属性复制一份 `normal_tensor`，只把副本中的enable 字段改为 1，并缓存这份开启版供后续采集调用复用。
3. profiler context 外或 action 为 `NONE` 时始终选择 `normal_tensor`；采集 action 才选择`profile_tensor`。

`normal_tensor` 在整个 plan 生命周期内保持关闭且不再修改，`profile_tensor` 作为独立副本按需创建。正反向 launch 均通过 `_prepare_mega_kernel_call()` 根据当前 schedule action 取得本次调用所需的RuntimeConfig。这样 action 可在普通执行与采集之间切换，同时避免不同调用共享可变的 enable 状态；profilingcontext 结束后的普通调用会继续使用关闭配置。

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

MegaMoe workspace 中正向、反向各保留一块固定 4096 Byte 的对称 `all_event_counters`，只承载调度事件。
进入采集 action 后，Torch profiler 按 `(device_id, buffer_size)` 从 session 私有池取得普通 NPU
`profile_buffer`。每个尚未 drain 的 invocation 独占一块 buffer，因此同一 step 的多层正反向调用不会互相
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

Python `RuntimeConfigC` 与 AscendC `RuntimeConfig` 必须保持字段顺序和宽度一致：

| 字段 | 类型 | 说明 |
|------|------|------|
| `cycle_profiling_enabled` | `uint32` | 当前 launch 是否进入采样路径 |
| `aic_profile_record_capacity` | `uint32` | 每个 AIC worker 的 record 容量 |
| `aiv_profile_record_capacity` | `uint32` | 每个 AIV worker 的 record 容量 |

每个 `TaskDesc` 额外包含：

| 字段 | 类型 | 说明 |
|------|------|------|
| `profile_desc_id` | `uint32` | 阶段语义；未配置时为 `0xFFFFFFFF` |
| `profile_owner_id` | `uint32` | 业务归属；未配置时为 `0xFFFFFFFF` |

公共 Device runtime 不理解 Expert、layer 等业务概念，只透传 ID。阶段名称和 owner label 只保存在 Host
metadata 中，不占用 Device record 的字符串空间。

### 4.2 Buffer 布局与内部 ABI

调度和 profiling 使用两块语义、内存类型及生命周期都独立的 buffer：

```text
all_event_counters（对称内存，固定 4096 Byte）
└── scheduler event counters

profile_buffer（普通 NPU 显存，仅采集 action 分配）
├── AIC slot 0
│   ├── 64-byte core header
│   └── N × 32-byte records
├── ...
└── AIV slots
```

每个 AIC/AIV worker 拥有独立 slot，避免全局 record allocator 和原子抢占。core header 保存 entry cycle、
有效/丢弃数量、core 类型、block ID 和容量；record 保存起止 cycle、`desc_id`、`task_id`、
`stage_task_index` 和 `owner_id`。profile slot 从 `profile_buffer` 的 offset 0 开始，不占用 symmetric
heap。

`profile_buffer` 是内部 native ABI 输入：正反向 OpProto、OpDef、ACLNN GetWorkspaceSize、L0 和 Device
kernel 的输入列表都显式传递它，但 Python 公共 `mega_moe()`、`mega_moe_grad()` 和
`MegaMoeExperts` 签名保持不变。普通调用通过私有 adapter 补齐参数；profiling 关闭时用原
`all_event_counters` 作为未读取的占位 tensor，不产生新的申请、切片或数据搬运。

### 4.3 容量计算

Host 根据完整调度表复现每个 worker 可能执行的任务，分别计算最忙 AIC 和 AIV 的记录数：

```text
capacity = min(round_up(required_records, 16), 256)
```

每个有依赖的普通 task 最多记录 WaitDependency、Compute 和 TriggerEvent 三段；SHMEM put 不重复记录公共
trigger 区间。达到 256 条上限后停止写入新 record 并累计 dropped count，不覆盖既有数据。

profile buffer 总容量为 `24 * AIC_stride + 48 * AIV_stride`，最大值为
`72 * (64 + 256 * 32) = 594432` Byte；不再包含 4096 Byte event counters。
`max_pending_calls` 控制 Host session 的 buffer drain 频率，与这里的每核 record 容量无关。

### 4.4 Cycle 与时间换算

Device 使用 `GetSystemCycle()` 读取 system counter：

```text
timestamp_us = (start_cycle - anchor_cycle) / cycle_frequency_mhz
duration_us  = (end_cycle - start_cycle) / cycle_frequency_mhz
```

当前内部频率映射支持 Ascend 910B/910C/910_93 系列，A2/A3 使用 50 MHz。SoC 名称来自
`torch_npu.npu.get_device_name()`；未知型号会在开启 RuntimeConfig 前显式失败，不接受用户从前端传入猜测值。

### 4.5 关闭态快路径

worker 在 `Process()` 入口读取一次 enable 字段，并在进入任务循环前选择快慢实现。关闭路径具有以下约束：

- 不读取 system cycle；
- 不执行 profiling barrier；
- 不写 core header 或 record；
- 不读取 `profile_desc_id` / `profile_owner_id`；
- 只清零 `all_event_counters` 原有的 4096 Byte 调度区；
- 不申请普通 NPU `profile_buffer`，也不创建 event-counter view；内部占位参数不会被 Device 读取。

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

公共 descId 至少覆盖 `WaitDependency` 和 `TriggerEvent`。未接入业务 spec 的 MegaKernel 仍可显示 task type
兜底名称；接入 spec 后，Wait/Trigger 名称会携带相邻业务阶段，例如 `GMM1_WaitDependency`，避免只看
`Expert` 序号时产生歧义。

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

### 6.3 Profile spec

MegaMoe 的阶段匹配、descId、`Expert` owner label 和全局 expert ID 计算集中在
`modules/mega_moe/profiling.py`。RuntimeConfig builder 构造完全部 task 后调用一次规则应用函数，将匹配结果
写入 `TaskDesc` 并保存 Host-only 展示 metadata。

GMM、SwiGLU 和 AllToAll 等公共 task builder 不写 MegaMoe owner，防止业务语义散落到通用调度层。
用户始终调用公共 `multicore.profiler.mega_kernel_profile()`，无需传阶段名、rank、device 或 RuntimeConfig。

### 6.4 正反向 launch 接入

Profiling 接入点为：

```text
_MegaMoeFunction.forward / backward
    └── _prepare_mega_kernel_call(runtime, direction, event_counters)
        ├── 当前 session 不采集 ──► normal RuntimeConfig + ABI placeholder
        └── WARMUP / RECORD ─────► profiled RuntimeConfig + private profile_buffer
```

`direction` 区分正反向 invocation。成功 launch 后调用 handle 的 `complete()` 完成 pending 记账；
Device 已直接写入该调用独占的 buffer，不需要再复制。异常路径调用 `cancel()` 撤销 pending 状态并释放
buffer。模型调用方无需传 profiling 参数。

---

## 7. Torch 公共接口

### 7.1 导入

```python
from hyper_parallel.core.multicore import profiler as multicore_profiler
```

`core.multicore.profiler` 延迟加载 Torch backend，公共 `core/` 不直接导入 Torch。

### 7.2 schedule

```python
multicore_profiler.schedule(
    *,
    wait: int,
    warmup: int,
    active: int,
    repeat: int = 0,
    skip_first: int = 0,
)
```

所有参数必须是非负整数，且 `active > 0`。返回对象接受零起始 step 并产生 `ProfilerAction`。

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

from hyper_parallel.core.multicore import profiler as multicore_profiler

rank = dist.get_rank()
schedule = multicore_profiler.schedule(wait=0, warmup=0, active=1, repeat=1)

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
capture_schedule = multicore_profiler.schedule(
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


def on_trace_ready(profiler):
    path = Path("./traces") / f"rank{rank}_window{profiler.step_num}.json"
    profiler.export_chrome_trace(path)


with multicore_profiler.mega_kernel_profile(
    schedule=multicore_profiler.schedule(wait=1, warmup=1, active=2, repeat=3),
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
from hyper_parallel.core.multicore import profiler as multicore_profiler

multicore_profiler.merge_chrome_traces(
    "/path/to/dump/.../trace.json",
    f"./traces/rank{rank}_mega_kernel_trace.json",
    f"./traces/rank{rank}_merged_trace.json",
)
```

多 rank 结果必须逐 rank 融合。不要把 rank 0 内部 trace 对齐到其他 rank 或 device 的外层 trace。

---

## 11. 新 MegaKernel 接入

### 11.1 无业务定制

公共 runtime 可以为没有 spec 的新 MegaKernel 导出 WaitDependency、TriggerEvent 和 task-type fallback。
这种方式适合先验证采样链路，但 timeline 不具备业务阶段名和 owner。

### 11.2 增加业务语义

推荐按以下顺序接入：

1. 在 `modules/<kernel>/profiling.py` 定义一份 `_ProfileSpec`；
2. 使用已有稳定字段匹配阶段，例如 `task_type`、`tiling_data_position` 或 input position；
3. 需要 owner 时实现从已构造 `TaskDesc` 和 topology 计算 ID 的 resolver；
4. 在 RuntimeConfig builder 完成全部 task 和调度修订后调用一次 `_apply_mega_kernel_profile_spec()`；
5. 在 plan 中用 `_prepare_mega_kernel_runtime_config()` 准备 normal/profiled 配置和 Host metadata；
6. 在内部 OpProto、OpDef、ACLNN、L0、Device kernel 和框架 adapter 中贯通 `profile_buffer` 输入，但保持
   模型用户接口不变；
7. 在正反向 launch 前调用 `_prepare_mega_kernel_call()`，成功后 `complete()`，异常路径 `cancel()`；
8. 增加布局、metadata、Torch session、导出、merge 和真实 NPU 测试。

多个匹配条件是逻辑与关系。应选择能够稳定区分阶段的已有字段，避免只依赖全局 task 顺序；公共 task builder
不得写具体模型的 owner 语义。

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

### 12.4 2026-09-09 真实 NPU 回归

独立 `profile_buffer` 改造后在 Ascend 910B3 上完成以下验证：

- 34 个定向 UT 和 5 个 subtest 全部通过；
- Torch multicore 的 Ascend 910B/910_93 正反向 Device kernel、OpProto/OpDef、ACLNN、L0 和 Torch adapter
  均编译成功；
- 2-rank 完整正向+反向精度用例和共享资源/交替 stream 用例通过，覆盖 profiling 关闭路径；
- 2-rank 正向 profiling 每 rank 仍为 1 个 invocation、1411 条事件、0 dropped；
- 同次 `msprof` 采集中 rank 0/device 6 与 rank 1/device 7 各融合 1411 条内部事件，均未 fallback、无
  warning，内部 span 未越过外层 Device kernel。

---

## 13. 约束与故障排查

| 现象 | 检查项 |
|------|--------|
| `no completed ... window` | active 窗口是否结束，是否调用了 `step()` 或正常退出 context |
| trace 为空 | active step 内是否真的执行 MegaKernel；是否只有 warmup 数据 |
| timeline 零散 | shape 是否过小；是否完成真实 warmup、NPU synchronize 和 rank barrier |
| 未知 SoC | 当前 device name 是否在 system counter 频率映射中 |
| dropped record | 每核 slot 是否达到 256 上限；不能用 `max_pending_calls` 修复 |
| merge 无匹配 | 原生 trace 是否含 Device kernel；检查 pattern、index 和 pid |
| fallback warning | 复核 duration 降级选择的外层 kernel 是否正确 |
| Host wrapper warning | 仅在明确设置 `allow_host_wrapper=True` 时接受近似结果 |
| 多层 trace 配错 | 检查 invocation 顺序是否与原生外层 kernel 顺序一致 |
| context 冲突 | 同一进程只允许一个非嵌套 active session |

---

## 14. 相关文档

- [MegaKernel 多核并行 Profiling 使用指南](../../../../docs/guide/mega_kernel_profiling.md)
- [MegaMoe Design and Usage](mega_moe_design_and_usage.md)
- [MegaMoe Torch 示例](../examples/mega_moe/README.md)
