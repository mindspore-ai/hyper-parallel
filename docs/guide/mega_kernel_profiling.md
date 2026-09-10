# MegaKernel 多核并行 Profiling 使用指南

HyperParallel 提供面向 Torch 的 MegaKernel 内部 Profiling 工具，用于观察一个融合 kernel 内部各 AIC/AIV
worker 的依赖等待、计算和事件触发区间。工具导出标准 Chrome Trace JSON，可直接用 Perfetto 或
Chrome Trace Viewer 打开，也可以与 `msprof`、`torch.profiler` 等工具产生的外层 trace 离线融合。

## 核心概念

常规 NPU profiling 通常只能看到 MegaKernel 的整体边界，无法区分内部的 Dispatch、GMM、SwiGLU 和
Combine。MegaKernel profiler 在同一算子产物中保留轻量设备侧打点，通过运行时 schedule 决定是否开启：

| 能力 | 说明 |
|------|------|
| 独立采集 | 只采集 MegaKernel 内部事件，不启动 `torch.profiler` |
| 分阶段展示 | 展示 WaitDependency、Compute 和 TriggerEvent 区间 |
| 多核视图 | AIC 和 AIV worker 分别显示为独立 timeline track |
| 调度窗口 | 使用 `wait`、`warmup`、`active`、`repeat` 控制采集 step |
| 独立导出 | 每个 rank 导出一份 Chrome Trace JSON |
| 离线融合 | 将内部 trace 对齐到原生 trace 中对应的外层 Device kernel |

内部调度计数器仍使用原有的 4096 Byte 对称内存；只有处于采集 action 时，profiler 才为每次
MegaKernel 调用取得一块精确大小的普通 NPU `profile_buffer`。Device 直接写入该 buffer，窗口 drain 时再
同步并搬到 Host。该内部参数不出现在 `MegaMoeExperts` 公共接口中；关闭 profiling 时也不会申请它。

当前采集前端只支持 Torch。MindSpore 前端不在支持范围；该工具也不会自动包装或启动原生
`torch.profiler`。

## 接口概览

统一从 `hyper_parallel.core.multicore.profiler` 使用以下接口：

| 接口 | 说明 |
|------|------|
| `schedule()` | 创建按 step 推进的采集计划 |
| `mega_kernel_profile()` | 创建进程内独立 MegaKernel profiler context |
| `profiler.step()` | 结束当前逻辑 step 并推进 schedule |
| `profiler.export_chrome_trace()` | 导出最近一个完整采集窗口 |
| `merge_chrome_traces()` | 离线融合原生 trace 和 MegaKernel 内部 trace |

---

## 使用前准备

先按照 [MoE 多核并行使用指南](../../hyper_parallel/core/multicore/README.md) 构建 Torch multicore 与 SHMEM payload：

```bash
source /path/to/cann/set_env.sh
./build.sh --multicore on --custom-ops off
python -m pip install -e .
source build/native/payload/hyper_parallel/core/multicore/lib/set_env.bash
```

在导入 Torch、torch_npu 或 HyperParallel 前设置平台，并确保所有 EP rank 使用一致的 SHMEM 配置：

```bash
export HYPER_PARALLEL_PLATFORM=torch
export SHMEM_IP_PORT=127.0.0.1:18888
```

Profiling 不改变 `MegaMoeExperts` 的构造和调用方式。先确认普通正反向能够正常执行，再开启采集。

---

## 基础使用

### 1. 采集一次稳定的正向

正式采集前先使用相同 shape 完成一次 warmup，再同步 NPU 并让所有 rank 对齐。这样可以避开首次加载、
编译和各 rank 启动抖动，使 timeline 更接近稳态执行：

```python
from pathlib import Path

import torch
import torch.distributed as dist

from hyper_parallel.core.multicore import profiler as multicore_profiler

rank = dist.get_rank()
output_dir = Path("./mega_kernel_traces")
capture_schedule = multicore_profiler.schedule(
    wait=0,
    warmup=0,
    active=1,
    repeat=1,
)

with torch.no_grad():
    # 使用与正式采集相同的 MegaMoe workload 完成 warmup。
    experts(
        hidden_states,
        topk_ids,
        topk_weights,
        tokens_per_expert=tokens_per_expert,
    )
    torch.npu.synchronize()
    dist.barrier()

    with multicore_profiler.mega_kernel_profile(
        schedule=capture_schedule,
        detailed_task_names=True,
    ) as profiler:
        experts(
            hidden_states,
            topk_ids,
            topk_weights,
            tokens_per_expert=tokens_per_expert,
        )
        profiler.step()

trace_path = output_dir / f"rank{rank}_mega_kernel_trace.json"
trace = profiler.export_chrome_trace(trace_path)
```

每个 rank 都应独立执行并导出自己的文件。`profiler.step()` 表示一个用户侧逻辑 step 的结束，不是一个
MegaKernel launch 的结束；一次逻辑 step 内可以包含多个 MegaKernel 调用。

### 2. 配置采集窗口

```python
capture_schedule = multicore_profiler.schedule(
    skip_first=10,
    wait=2,
    warmup=1,
    active=2,
    repeat=1,
)
```

| 参数 | 含义 |
|------|------|
| `skip_first` | 首次进入周期前完全跳过的 step 数 |
| `wait` | 每个周期开始前不采集的 step 数 |
| `warmup` | 执行设备侧采样、但在 step 结束后丢弃的 step 数 |
| `active` | 每个周期保留的 step 数，必须大于 0 |
| `repeat` | 周期重复次数；`0` 表示持续重复到 context 退出 |

一个周期依次经历 `NONE → WARMUP → RECORD → RECORD_AND_SAVE`。窗口最后一个 active step 调用
`profiler.step()` 后，trace 才成为可导出的完整窗口。
### 3. 采集训练正反向

把 `profiler.step()` 放在一个完整训练 step 的末尾。正向和反向 MegaKernel 会自动写入不同的
`direction`，不需要分别创建 profiler：

```python
with multicore_profiler.mega_kernel_profile(
    schedule=multicore_profiler.schedule(
        wait=1,
        warmup=1,
        active=2,
        repeat=1,
    ),
    detailed_task_names=False,
) as profiler:
    for batch in data_loader:
        optimizer.zero_grad()
        loss = train_step(batch)
        loss.backward()
        optimizer.step()
        profiler.step()

profiler.export_chrome_trace(f"rank{rank}_train_mega_kernel_trace.json")
```

`detailed_task_names=False` 使用 `GMM1`、`SwiGLU` 等简洁名称；设为 `True` 时追加 Expert 和 task
序号，更适合定位单个 worker 或 expert。

### 4. 按窗口立即导出

多次 repeat 时，`export_chrome_trace()` 只保留最近完成的窗口。使用 `on_trace_ready` 为每个窗口保存
不同文件：

```python
output_dir = Path("./mega_kernel_traces")


def trace_handler(profiler):
    path = output_dir / f"rank{rank}_window{profiler.step_num}.json"
    profiler.export_chrome_trace(path)


with multicore_profiler.mega_kernel_profile(
    schedule=multicore_profiler.schedule(
        wait=1,
        warmup=1,
        active=2,
        repeat=3,
    ),
    on_trace_ready=trace_handler,
    max_pending_calls=16,
) as profiler:
    for batch in data_loader:
        train_step(batch)
        profiler.step()
```

`max_pending_calls` 控制一次 Host drain 前允许同时保留的调用 buffer 数量，不是设备侧每核 record 容量。
一般保持默认值即可。

### 5. 不使用 schedule

不传 schedule 时，context 内所有 MegaKernel 调用属于同一个窗口，退出 context 时完成同步、解析和导出准备：

```python
with multicore_profiler.mega_kernel_profile() as profiler:
    train_step(batch)

profiler.export_chrome_trace("mega_kernel_internal.json")
```

---

## 与原生 Trace 融合

### 1. 采集原生 Trace

可单独使用 CANN `msprof` 采集包含外层 Device kernel 的 trace，例如：

```bash
msprof --output=/path/to/dump python your_program.py
```

原生 trace 和 MegaKernel 内部 trace 可以分别采集，也可以在同一次运行中各自导出；融合本身是纯 Host
后处理，不依赖 Torch、NPU Runtime 或设备。

### 2. 按 rank 融合

每个 rank 的内部 trace 必须与对应 rank/device 的原生 trace 分别融合：

```python
from hyper_parallel.core.multicore import profiler as multicore_profiler

report = multicore_profiler.merge_chrome_traces(
    "/path/to/framework_trace.json",
    f"./mega_kernel_traces/rank{rank}_mega_kernel_trace.json",
    f"./mega_kernel_traces/rank{rank}_merged_trace.json",
)
```

常用可选参数：

| 参数 | 说明 |
|------|------|
| `kernel_pattern` | 外层 MegaKernel 名称正则；默认从内部 metadata 自动推导 |
| `kernel_index` | 从第几个匹配的外层 Device kernel 开始配对 |
| `outer_pid` | 限定外层 trace 的 process ID |
| `allow_host_wrapper` | 允许 Host wrapper 近似对齐；仅建议诊断时使用 |

融合按 invocation 顺序将内部事件对齐到外层 Device kernel。显式传入 `kernel_pattern` 后不会进行名称降级；
默认也不会把 Host wrapper 当作可靠的 Device kernel。

---

## 检查结果

### 独立 Trace

打开 JSON 后应看到 `MegaMoe/AIC/<block_id>` 和 `MegaMoe/AIV/<block_id>` 等 track。MegaMoe 正向阶段
包括 Dispatch、GMM1、SwiGLU、GMM2 和 Combine；反向阶段包括 DispatchGrad、ActGrad、W2Grad、
SwiGLUGrad、GateGrad、W1Grad 和 CombineGrad。

重点检查顶层 `megaKernelCycleTrace`：

| 字段 | 正常表现 |
|------|----------|
| `invocationCount` | 与 active 窗口内的 MegaKernel 正反向调用数一致 |
| `recordCount` | 大于 0 |
| `droppedRecordCount` | 应为 0 |
| `warnings` | 通常为空 |
| `invocations[].direction` | 能区分 `forward` 和 `backward` |

### 融合 Trace

融合结果增加顶层 `megaKernelTraceMerge`。重点检查：

- `mergedInvocationCount` 是否与预期调用数一致；
- `usedFallbackKernelSelection` 是否为 `false`；
- `alignments` 中选择的外层 kernel、pid、tid 和 direction 是否正确；
- `warnings` 是否为空；
- `exceedsOuterKernelUs` 是否为 0 或仅有可解释的时钟/边界误差。

---

## 常见问题

### Trace 为空或无法导出

确认 active step 内实际执行了 MegaKernel，并在窗口边界调用了 `profiler.step()`。如果只执行到 warmup，
记录会被主动丢弃；如果在窗口完成前调用 `export_chrome_trace()`，接口会抛出异常。

### Timeline 看起来零散

先使用真实或等价 workload shape，并在 profiler context 外完成至少一次 MegaMoe warmup；随后执行
`torch.npu.synchronize()` 和 rank barrier，再开始 active window。小 shape、首次编译或 rank 启动不同步
都会放大任务间空洞。需要定位具体阶段时再开启 `detailed_task_names=True`。

### Merge 找不到外层 kernel

确认输入的是包含 NPU Device kernel 的原生 trace，而不是只有 Host API 的文件。必要时传入
`kernel_pattern`、`kernel_index` 或 `outer_pid`。只有在明确接受近似对齐时才使用
`allow_host_wrapper=True`，并检查输出 warnings。

### 出现 dropped record

`droppedRecordCount > 0` 表示至少一个 worker 的固定 record slot 已满。此时结果不完整，不能当作完整
性能分析依据；应缩小单次 kernel 内任务规模或扩展设备侧容量设计，而不是调整 `max_pending_calls`。

### 多个 profiler context 冲突

同一进程不支持嵌套或并发 MegaKernel profiler context。多 rank 场景中每个 rank 各自创建一个 context 和
输出文件即可。

---

## 更多参考

- [MoE 多核并行使用指南](../../hyper_parallel/core/multicore/README.md)
- [MegaMoe 方案设计及使用说明](../../hyper_parallel/core/multicore/docs/architecture.md)
- [MegaKernel Profiling 详细方案设计及使用说明](../../hyper_parallel/core/multicore/docs/mega_kernel_profiling_design_and_usage.md)
