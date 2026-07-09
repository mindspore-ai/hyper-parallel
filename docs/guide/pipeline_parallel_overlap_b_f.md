# Pipeline Parallel 通算掩盖设计文档

## 1. 背景与动机

在 PP（Pipeline Parallelism）+ EP（Expert Parallelism）混合并行场景下，单 micro-batch 内的关键路径包含两类高延迟通信：

- **跨 stage 的 P2P send/recv**：上下游 stage 之间传递 activation（FWD）/ gradient（BWD）。
- **MoE 层的 EP all-to-all（A2A）**：dispatch 把 token 发到对应 expert 所在 rank，combine 把 expert 输出聚合回原 rank。

在传统 1F1B 调度里，这些通信与 compute 串行，timeline 上构成显著的 bubble。本 PR 在 `ScheduleInterleaved1F1B` 上加了 **`overlap_b_f` 选项**，把稳态期相邻的 BWD 与 FWD 配对成 `OVERLAP_B_F` 复合 step；运行时由独立的 **`CommComputeOverlap`** 协调器以两个 Python 线程并发驱动，让一侧的 EP A2A 与另一侧的非通信 compute 在 GPU 上真正并发执行。

设计目标：

1. **正确性**：与非 overlap 路径数值完全一致。
2. **可组合**：保持现有 `PipelineStage` / `ScheduleInterleaved1F1B` 抽象，新增 flag + 通过 callback 注册扩展，不污染调度核心。
3. **机制与调度解耦**：`CommComputeOverlap` 是独立的双线程协调器，未来可用于 TP+CP、FSDP prefetch 等其他场景，不绑定 PP。
4. **跨平台基础**：上层抽象（`HookCoordinator` / `CommComputeOverlap`）平台无关；同步原语下沉到 `platform.differentiable_sync_hook`。

---

## 2. 模块结构

```text
hyper_parallel/core/pipeline_parallel/
├── scheduler.py                # 调度核心（OVERLAP_B_F、PipelineContext、custom_fn 注册、overlap_b_f flag）
├── stage.py                    # PipelineStage
├── hook_coordinator.py         # NEW：双线程 COMM-first rendezvous
├── comm_compute_overlap.py     # NEW：CommComputeOverlap orchestrator + A/B/C/D wrap
└── __init__.py                 # 导出新增 API

hyper_parallel/platform/
├── platform.py                 # 抽象基类新增 differentiable_sync_hook
└── torch/platform.py           # 增加 _TorchSyncHookFunction (autograd.Function)

examples/torch/pp_overlap/
└── pp_overlap_moe_example.py   # 端到端示例：PP + EP + 通算掩盖 + MoE
```

---

## 3. 关键概念

### 3.1 MetaStep 与 OVERLAP_B_F

`MetaStep` 描述调度单元，`MetaStepType` 增加：

- `OVERLAP_F_B` / `OVERLAP_B_F`：复合 step，`sub_steps = (first, second)`。
- `OVERLAP_B_F` 把 1F1B 的稳态期 `B_i, F_{i+1}` 合并成一拍。

`ScheduleInterleaved1F1B(overlap_b_f=True).construct_stage_exec_order` 在原有 interleaved 1F1B 基础上做替换：

```text
原:    F₁, B₁, F₂, B₂, ..., F_n, B_n
新:    F₁, [B₁,F₂], [B₂,F₃], ..., [B_{n-1},F_n], B_n
```

第一个 FWD 与最后一个 BWD 不配对（暖热/冷却），其余每拍稳态都是一个 `OVERLAP_B_F`。

### 3.2 PipelineContext 与 custom_fn 注册

`PipelineScheduleRuntime` 引入：

```python
schedule.register_custom_function(MetaStepType.OVERLAP_B_F, callback)
```

`run_microbatches` 遇到 `OVERLAP_B_F` 时把控制权交给 callback，传入 `PipelineContext`：

```python
class PipelineContext:
    schedule, arg_mbs, kwarg_mbs, losses, send_handles
    fwd_recv_ops = schedule.fwd_handle_cache  # for overlap_p2p
    bwd_recv_ops = schedule.bwd_handle_cache
```

这样调度核心保持泛化，dual-pipe 是一个外挂插件。

### 3.3 P2P 重叠与 `overlap_p2p`

新增 `PipelineScheduleRuntime(overlap_p2p: bool = False)` 参数：

- `False`（默认）：FWD_RECV / BWD_RECV 在 `_exec_step` 里 issue 并立即 wait（HCCL 流侧 wait，不阻塞 host），与原行为兼容。
- `True`：RECV 把 handle 缓存到 `fwd_handle_cache / bwd_handle_cache`，由消费者（`_exec_step` 的 FWD/BWD 分支或 OVERLAP_B_F callback）按需 pop+wait。

SEND 在 `overlap_p2p=True` 下追加到 `send_handles` 列表，在 `run_microbatches` 末尾统一 drain，不做 per-step 阻塞——HCCL 按 PG 自动分流，与下一拍 compute 物理并发。

### 3.4 add_send_recv 与 OVERLAP 列扫描

`add_send_recv` 把 OVERLAP 步骤展开成 2 个虚拟 time slot（`_OverlapPhantom`），按列扫描穿插插入 SEND/RECV，并保持偶/奇 rank 处理顺序避免 P2P 死锁。具体：

- `_expand_overlap_slots`：每个 OVERLAP 拆成 `(first_half, second_half)` 两个 phantom。
- `_column_scan_insert_comms`：对齐时间步，先把当前 step 入队，再调 `_insert_comms_for_step` 发 SEND/RECV。
- 偶数 rank 先于奇数 rank 处理，`'loop'` / `'v'` 两种 stage→rank 映射。

---

## 4. EP A2A 通算掩盖

### 4.1 A/B/C/D 同步钩子

每个 MoE 层在 forward 时执行：

```text
[A] ─► dispatch(A2A) ─► [B] ─► expert(compute) ─► [C] ─► combine(A2A) ─► [D] ─► (Attention)
```

`CommComputeOverlap` 提供 `wrap_dispatch` / `wrap_combine`，调用 `platform.differentiable_sync_hook` 把 A、B、C、D 4 个钩子点插到张量计算图里。Backward 自动得到对称的 D'、C'、B'、A' 触发顺序。

### 4.2 HookCoordinator：COMM-first rendezvous

仅靠 `threading.Barrier` 不够——两个线程到达 rendezvous 后**释放顺序由 GIL 决定**，host 侧 dispatch 顺序不可控，GPU 侧的 A2A 与 compute 启动顺序也就不可控。

`HookCoordinator` 在 Barrier 之上叠加 **per-rendezvous `threading.Event`**：

1. 两线程都到 → Barrier 通过。
2. `COMM` 侧立即 dispatch 它的 A2A，然后 `notify_dispatched()` 触发 Event。
3. `COMPUTE` 侧 wait Event，确认 COMM 已 dispatch 后再 dispatch 自己的 compute kernel。

每次 rendezvous 用**独立** Event，避免单 Event 在连续 rendezvous 间出现 set/clear race。

`HookRole`：

- `COMM`：先 dispatch（dispatch / combine A2A）
- `COMPUTE`：等 COMM 通知后再 dispatch（expert compute / Attention）

### 4.3 CommComputeOverlap.run

```python
overlap.run(fwd_fn, bwd_fn)
```

主线程跑 `fwd_fn`，持久 daemon worker 线程跑 `bwd_fn`。`coordinator.enable()` 把 Barrier + per-rendezvous Event 状态重置；任一侧异常会立即 `disable` 解除另一侧 wait，避免死锁。**不需要传 `num_layers`**——见 4.5。

### 4.4 跨层 `D → A_next` 窗口

层与层之间，第 k 层的 combine（D 钩）与第 k+1 层的 dispatch（A 钩）相邻。现实现把它们各自配 COMM/COMPUTE 角色，让一侧的 combine.bwd 与另一侧的 Attention 计算并发——这是除"层内 dispatch ↔ expert compute"之外的第二个掩盖窗口。

### 4.5 最后一层 D 的特殊处理：`D_LAST` 静态标注

FWD 与 BWD 一拍下来 hook 数本应都是 4N（每层 A/B/C/D），但 BWD 的入口结构有非对称：autograd 进入反向时第一个被调用的就是 `combine_N.bwd`（一个 A2A），它**先于**任何 hook 的 backward 触发，是个"自由 dispatch"——D_N 的 backward hook 想去 rendezvous 时 A2A 已经在 EP 流上了，rendezvous 也无意义。

所以 D_N 这个 hook 在两侧都要跳过：

- **FWD 一侧**：D_N 是最后一个 D，next op 是"layer N+1 的 Attention"，根本不存在 → rendezvous 没意义
- **BWD 一侧**：D_N 是 BWD 第一个 hook，前面的 combine.bwd 已经自由 dispatch 过了 → rendezvous 也没意义

**实现**：在 `CommComputeOverlap.wrap_combine(combine_fn, is_last_layer=True)` 时把闭合的 D 钩 tag 成 `"D_LAST"`，`_TorchSyncHookFunction` 的 forward / backward 看到 `D_LAST` 直接 return：

```python
if hook_name == "D_LAST":
    return x  # forward
if hook_name == "D_LAST":
    return grad_output, None, None  # backward
```

同一个 hook 实例在 FWD 是"最后一个 D"、在 BWD 是"第一个 D"，**一个 tag 同时关掉两侧的特判**。这样 `HookCoordinator` 就不需要再维护 runtime 计数器（早期版本有 `_num_layers` / `increment_cycle` 让 FWD 计数到 N 自动 disable，以及 `_bwd_local` thread-local 计数让 BWD 跳过第一个 D，这两个机制现在都被静态标注取代）。

跳过后 FWD/BWD 各跑 4N-1 个 rendezvous，按位置严格配对：

```text
FWD A_1 ↔ BWD C_N    (COMM ↔ COMPUTE，FWD dispatch_1 与 BWD expert.bwd_N 重叠)
FWD B_1 ↔ BWD B_N
FWD C_1 ↔ BWD A_N
FWD D_1 ↔ BWD D_{N-1}
...
FWD C_N ↔ BWD A_1
```

每对至少有一侧是 COMM，没有 COMPUTE+COMPUTE 死锁风险。

---

## 5. PP P2P 与 RECV-Compute 重叠

`overlap_p2p=True` + `OVERLAP_B_F` callback 中：

```python
# callback 入口（main thread, default stream）
fwd_recv_handles = ctx.fwd_recv_ops.pop((fwd_stage.stage_index, fwd_mi), None)
bwd_recv_handles = ctx.bwd_recv_ops.pop((bwd_stage.stage_index, bwd_mi), None)

def fwd_fn():
    if fwd_recv_handles:
        schedule._wait_p2p(fwd_recv_handles)   # 主线程默认流 wait
    out = fwd_stage.forward_one_chunk(fwd_mi, ...)

def bwd_fn():
    torch.npu.set_device(device.index)         # BWD 线程绑设备，否则 HCCL hang
    if bwd_recv_handles:
        schedule._wait_p2p(bwd_recv_handles)   # BWD 线程默认流 wait
    bwd_stage.backward_one_chunk(bwd_mi, ...)
```

效果：

- FWD/BWD RECV 在前一拍就被 issue 到 HCCL P2P 流，与本拍 OVERLAP_B_F 的 compute 自然并发。
- callback 内的 `wait()` 把 PP-RECV stream 完成事件挂到当前默认流上 → 后续的 EP A2A 通过 PyTorch HCCL backend 的 `queueEvent(currentStream)` 链取到这个等待事件 → A2A 正确等到 PP-RECV 完成才启动。
- SEND 不需要 callback 内特殊处理：`overlap_p2p=True` 路径把 send handle 攒到末尾 drain，HCCL 自动用生产者流上的 event 排序。

### 5.1 关键设计决策：BWD 不开独立 stream

早期版本曾尝试 `with torch.npu.stream(bwd_stream):` 把 BWD 跑到独立侧流，期望 FWD/BWD compute 物理并发。**实测发现该方案不工作**：

1. `torch.autograd.backward()` 派发反向 kernel 时使用的是 forward 期间 saved tensor 的 producer stream，**不看** `with torch.npu.stream(...)` 的 thread-local current stream——BWD compute 实际上还是落在主线程默认流上。
2. `recv_buffer` 由主线程默认流分配，PyTorch tensor allocator-stream 跟踪把 buffer 与默认流绑定。即便我们把 `handle.wait()` 的 event 挂到 bwd_stream 上，EP A2A 在读 buffer 时通过 allocator 看到的依赖流是默认流（不含 wait_event），导致 A2A 提前启动，与 PP-RECV 形成数据竞争。

最终设计：BWD 跑在 BWD 线程的默认流（与主线程默认流物理相同），整个 wait 链路与 buffer 跟踪对齐。失去的"FWD/BWD compute 真并行"在 MoE 模型里收益本就有限——MoE 的 A2A 在 EP HCCL 流，与默认流的 compute 物理并发，这部分掩盖窗口靠流隔离实现，**不依赖**独立的 BWD compute 流。

---

## 6. 端到端时间线

PP=2 / EP=2 / chunks_per_rank=2 / moe_layers=2 的稳态某拍：

```text
默认流(compute):  [BWD attention][FWD attention][BWD expert][FWD expert]...
EP HCCL 流(A2A):           [BWD combine.bwd][FWD dispatch][BWD dispatch.bwd][FWD combine]...
PP-RECV 流:        [本拍 BWD-RECV.....]  [本拍 FWD-RECV.....]
PP-SEND 流:                                          [上拍 BWD-SEND.....]
```

掩盖关系：

- EP A2A ↔ 默认流 compute（核心收益，由 A/B/C/D 钩子保证 dispatch 顺序）
- PP-RECV ↔ EP A2A / 默认流 compute（recv 提前一拍 issue）
- PP-SEND ↔ 下一拍 compute（末尾 drain）

---

## 7. API 概览

```python
from hyper_parallel.core.pipeline_parallel import (
    ScheduleInterleaved1F1B,           # 调度（带 overlap flag）
    CommComputeOverlap,                # 双线程 orchestrator
    HookCoordinator, HookRole,         # 底层 rendezvous 原语
    MetaStepType, MetaStep,            # 调度单元
    PipelineContext,                   # callback 上下文
)

# 1. 构造 schedule，启用 B/F overlap + P2P overlap
schedule = ScheduleInterleaved1F1B(
    stages, micro_batch_num, overlap_p2p=True, overlap_b_f=True,
)

# 2. 给每个 MoE 层的 dispatch / combine 装钩；最后一层标 is_last_layer=True
overlap = CommComputeOverlap()
last_idx = len(moe_layers) - 1
for i, layer in enumerate(moe_layers):
    layer.experts.dispatch = overlap.wrap_dispatch(layer.experts.dispatch)
    layer.experts.combine  = overlap.wrap_combine(
        layer.experts.combine, is_last_layer=(i == last_idx),
    )

# 3. 注册 OVERLAP_B_F callback
def callback(step, ctx):
    bwd_step, fwd_step = step.sub_steps
    ...
    overlap.run(fwd_fn=..., bwd_fn=...)

schedule.register_custom_function(MetaStepType.OVERLAP_B_F, callback)

# 4. 跑一个 iteration
losses = schedule.run(*inputs)
```

完整示例见 `examples/torch/pp_overlap/pp_overlap_moe_example.py`。

---

## 8. 平台抽象

新增 `platform.differentiable_sync_hook(tensor, hook_name, coordinator)`：

- Torch backend：`_SyncHookFunction(autograd.Function)`，前向时根据 `hook_name`（`A`/`B`/`C`/`D`）计算 COMM/COMPUTE role 并调 `coordinator.rendezvous`；反向时角色对称翻转。
- MindSpore backend：默认实现为 noop（不支持 dual-pipe overlap，将来可扩展）。

通过 `get_platform()` 获取，符合 `code-style.md` 平台抽象约定。

---

## 9. 限制与未来工作

### 当前限制

- **MoE 层数对齐**：FWD 与 BWD chunk 层数不等时，多余的 hook 落在没有配对的一侧 → barrier 死等。当前实现假设 FWD/BWD chunk 层数一致；不一致时需要在装钩时按短边对齐（多余层不装钩），框架不会自动处理。
- **MindSpore 暂不支持**：仅做了 noop 占位，需要补 autograd 兼容层。
- **FWD/BWD compute 不并行**：autograd 的 stream 行为决定了 BWD 必然跑在 forward 流上，目前接受这个事实。
- **OVERLAP_F_B 未对 P2P 适配**：本 PR 只完成了 OVERLAP_B_F 路径的 `overlap_p2p=True` 适配，对称的 F_B 需类似改造。

### 后续优化方向

1. **FWD/BWD 双向独立 PG**：避免 HCCL P2P group 内序列化，让 FWD-RECV 与 BWD-RECV 在两条 HCCL 流上真正并发。
2. **CUDA Graph / 静态化**：把 OVERLAP_B_F 内的 dispatch/combine/compute kernel 序列做成 graph，进一步降低 host 侧 dispatch overhead。
3. **MindSpore 适配**：实现 `differentiable_sync_hook` MindSpore 版（autograd 兼容层）。
4. **FWD compute 与 BWD compute 真并行**：方案上需要重新设计 forward 时的 stream 分配（比如 forward 时就把 chunk 分到两条流上），属于较大改动。

---

## 10. 验证

- **数值一致性**：`overlap_p2p=False` vs `True` 同 seed 同 input，loss / 各参数 grad 完全相等（atol=0, rtol=0）。
- **profile 验证 1**：单拍 OVERLAP_B_F 内 EP A2A 数量与 compute 完整重叠（A/B/C/D 钩子链不破）。
- **profile 验证 2**：PP-RECV 与 OVERLAP_B_F 内的 A2A 重叠数 ≤ FWD 的 A2A 数（BWD A2A 正确等待 PP-RECV）。
- **端到端**：`pp_overlap_moe_example.py` 在 8 卡（PP=2 × EP=4 或 PP=4 × EP=2）下跑通，loss 收敛符合预期。
