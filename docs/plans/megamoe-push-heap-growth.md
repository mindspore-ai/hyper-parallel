# MegaMoE Push：容量溢出时在线重建 SHMEM heap

日期：2026-09-21。状态：核心实现完成，已通过单测、2/4卡回归和8卡 Qwen 验证。

实施结果与测量边界见[验证报告](megamoe-push-heap-growth-validation.md)。以下保留设计依据与验收范围。

- 基线：`megamoe-push-pull`，提交 `84519c6f055352d5627688f2a5e0601f306dbde7`。
- 开发分支：`feat/megamoe-push-heap-growth`。
- Worktree：`/home/feiran/hyper-parallel-megamoe-push-heap-growth`。
- 项目约束：[代码风格](../../.agent/rules/code-style.md)、[分布式规则](../../.agent/rules/distributed.md)、[测试规则](../../.agent/rules/testing.md)。

## 1. 目标与结论边界

让 push 从较小的接收容量开始运行；当本次路由的最大接收量超过当前容量时，所属 EP 组的所有 rank
在进入 MegaMoE kernel 前暂停提交、完成旧 heap 上的设备工作、销毁并重建更大的 SHMEM heap，
然后继续执行同一次 forward。模型参数、优化器状态、路由选择及 token 数量保持原有语义，不丢 token。

解决的是“为最坏情况长期预留大 heap，或采用有限容量后一旦溢出就报错”的问题。
扩容后的所有 rank 仍使用相同的对称布局；一次极端热点仍可能把 heap 推到很大，扩容也不能突破物理 HBM 上限。
第一版只增长、不自动缩容，以避免负载波动造成反复 finalize/init。

此前 Qwen 实验的不同后端各自进行了28次参数更新，逐步路由未对齐；
因此“负载不均衡时 pull 一定没有性能或整卡 HBM 优势”尚不能由那组结果证明。
本计划把动态 push 作为独立方案验证，使用受控路由及相同参数状态重新建立比较依据。

## 2. 已核对的现状

以下路径均相对仓库根目录。

| 位置 | 当前行为 | 对改造的影响 |
| --- | --- | --- |
| `hyper_parallel/core/multicore/modules/mega_moe/spec.py` | `None` 配置 `EP × T × TopK` 的接收容量；有限因子按128行对齐；spec 冻结 | 构造时容量策略与运行时已分配容量需要分开 |
| `hyper_parallel/core/multicore/modules/mega_moe/route.py` | count all-gather 后，所有 rank 都得到目的 rank 的负载；`_expert_capacity` 对有限容量执行一致的越界检查 | 可以复用现有统计触发扩容，正常路径不增加一次负载 all-reduce |
| `hyper_parallel/core/multicore/modules/mega_moe/workspace.py` | 一次确定总 heap；workspace 不能改变容量；关闭时同步、释放并使旧 tensor storage 失效 | 需要可重绑定的 workspace 和独立于模块关闭的 heap 重建流程 |
| `hyper_parallel/core/multicore/modules/module.py` | 管理多个独立或共享的 execution resource group；部分资源延迟绑定 | 重建清单必须覆盖整个 SHMEM root，而非只覆盖溢出的模块 |
| `hyper_parallel/core/multicore/shmem/_lifecycle.py` | 一个进程只有一个活动 SHMEM root；引用计数归零才 shutdown；成员关系按有序 rank 集合判断 | 必须有统一的重建协调者，不能让单个模块擅自 finalize |
| `hyper_parallel/core/multicore/shmem/ccsrc/runtime/runtime.cpp` | shutdown 要求 active allocations 为0；成功后回到 Uninitialized；allocation ID 跨生命周期继续递增 | 可保留已有 stale allocation 防护；shutdown 失败后不能假装回到可运行状态 |
| `hyper_parallel/core/multicore/modules/mega_moe/function.py` | ctx 保存 plan/workspace；push 保存独立的 dispatch 副本和输出；backward 重新读取 workspace buffer | 有机会允许“旧 forward → heap 扩容 → 旧 backward”，但必须验证全部地址与别名 |

现有设备用例 `tests/torch/multicore/shmem/_test_lifecycle.py::test_binding_reinit_with_different_heap_sizes`
已经描述同一进程内64 → 128 → 64 MiB的干净生命周期重建，launcher 使用2卡。
本轮已通过该2卡用例；它本身不覆盖 MegaMoE 通信后重建、保存的 autograd 图或多个资源组，
这些场景由后续 MegaMoE 设备回归补充。

## 3. 建议的接口和容量策略

### 3.1 收敛后的配置接口

先选择 `dispatch_mode="push" | "pull"`；push 使用以下两个容量因子，pull 不接受这两个数值参数：

```python
experts = MegaMoeExperts(
    local_num_tokens=4096,
    hidden_size=5120,
    intermediate_size=1792,
    num_experts=48,
    top_k=8,
    ep_size=8,
    dispatch_mode="push",
    initial_capacity_factor=1.25,
    capacity_growth_factor=1.25,
)
```

- push 默认初始因子1.25、增长因子1.25；因子必须为有限数且不小于1。
- 不再提供静态/动态策略开关；push 超限时自动扩容。初始因子设为 EP 大小即可预留无损上界。
- 增长因子1.0表示只分配当次需求；任意配置均受 EP 无损上界约束。
- 两个因子纳入资源共享兼容性和各 rank 布局一致性检查；移除旧 `expert_capacity_factor`、`capacity_policy`。
- 模型 shape、dtype、device、EP 成员及专家数仍为静态。

### 3.2 增长规则

对每个唯一资源组维护当前对称接收容量 `C`，与本次实际本地接收行数 `R_rank` 分离。
设 `N = local_num_tokens × top_k`，`R_max = max(R_rank)`，初始 `C = align128(ceil(factor × N))`。

```text
R_max <= C:
    使用原 heap，不做重建 collective

R_max > C:
    upper = align128(EP × N)                  # 沿用当前保守无损上界
    C_new = min(upper, align128(max(R_max, ceil(capacity_growth_factor × C))))
```

增长系数默认1.25，可显式配置，以权衡预留空间和重建频率。
当几何增长余量超过显式预算、但 `align128(R_max)` 能放下时，缩减余量后重算，不能仅因增长余量报 OOM。
资源清单、目标容量、最终 heap 字节数必须由所有 EP rank 校验一致。

总 heap 预算按“每个唯一 execution resource group 一份”累加数据区、事件区、512字节分配对齐余量，
最终按2 MiB取整；共享的多层只计算一次。其他资源组保留各自已有容量，不全部跟随溢出组放大。
提前为已登记但尚未执行的资源组计入预算，避免下一层第一次绑定时落回旧的固定 heap 假设。

对于 H5120 / T4096 / TopK8 / BF16、单个共享资源组，公式示例为：

| 接收容量 | 接收区 + 返回区及开销后的 heap | 说明 |
| --- | --- | --- |
| `1.0 × N` | 642 MiB | 较小初始容量 |
| `1.25 × N` | 722 MiB | 当前默认初始容量 |
| `1.5 × N` | 802 MiB | 示例容量档位 |
| `2.25 × N` | 1042 MiB | 示例容量档位 |
| `8.0 × N` | 2882 MiB | 当前 EP8 静态无损预算 |

这些是 heap 预算，不是整网峰值 HBM；实际增长可能直接跨过多个档位。
扩容期间普通 HBM 中的激活、mapping、参数、梯度仍然存活，也必须计入内存可行性分析。

### 3.3 环境变量与真实容量

当前自动配置会写入 `HYPER_PARALLEL_SHMEM_HEAP_SIZE`，不能在下一次调用时误判为用户显式指定。
需要将配置来源、运行时实际 heap 和用户预算分别记录，避免靠修改环境变量作为扩容协议。

- 用户显式设置该变量时，首版保留原有固定 heap 预算承诺；所需布局超过该预算便在销毁前一致报错。
  grow 可以在预算内增加逻辑容量；物理 heap 保持显式大小，仍可统一重建布局。
- 未显式设置时，由协调者管理实际 heap 大小。建议给 native 初始化增加内部可选的 heap bytes 参数，
  由结构化配置传入；既有普通初始化继续支持环境变量。
- benchmark 和诊断从 `shmem.debug_state()` / 协调者读取实际字节数，不能继续把环境变量值当作峰值 heap。

## 4. 所有权与重建协议

### 4.1 一个 SHMEM root，一个协调者

建议在 MegaMoE 层新增内部 `heap_manager.py`，与现有 resource manager 配合，登记：

- SHMEM root 的有序全局 rank 成员、device、heap epoch、当前物理字节数、配置来源。
- 确定顺序的资源组清单及每组容量、布局、已绑定状态、workspace 句柄。
- 本地活动执行租约、重建状态、失败状态以及扩容统计。

不能只用 Python `id(ep_group)` 认定 root 身份：当前 SHMEM 允许不同 ProcessGroup 对象代表相同有序成员。
对跨 rank 清单使用一致的逻辑分配顺序和规格，不能传进程地址或依赖不同进程里偶然相同的对象 ID。

同一 root 下其他 managed workspace 都受重建影响，包括静态 push 或 pull；它们的逻辑容量不变，
但旧地址与事件需要一起替换。若存在未纳管的 SHMEM 引用或 allocation，在释放任何旧资源之前统一拒绝扩容。
首版不为任意外部 SHMEM 数据做隐式搬迁，也不能通过强行将引用计数归零绕过其他 owner。

### 4.2 扩容触发位置

把 route 的“收集负载并构造元数据”与“执行容量策略”分开：

1. 启动已有 count all-gather，与 push 的普通 HBM permutation 重叠。
2. 等待 count work 完成，得到 `R_rank`、`R_max` 和本轮独立的 mapping / 路由元数据。
3. push 与协调者的当前容量比较，必要时进入重建；pull 按实际接收量分配普通 HBM。
4. 成功后才取得当前 workspace 的执行租约并启动本轮 forward kernel。

push 的 permutation 输出不在 SHMEM 中，可以跨越此次重建继续使用。
触发扩容的这次 forward 尚未发起 MegaMoE PUT，不需要运行一遍失败的 kernel，也不重复计算或随机生成路由。
不要在 `shmem.empty` 的单 rank OOM 异常里临时决定扩容，因为此时其他 rank 可能已经进入不同的 collective。

### 4.3 状态机

```text
READY(epoch=e)
  └─ 所有 rank 观察到同一次容量不足
       → AGREE：核对 owner / 布局 / 容量 / 预算 / epoch
       → QUIESCE：阻止新提交，完成所有旧 heap 使用者
       → FREE：按统一顺序释放全部 managed SHMEM allocations
       → FINALIZE：结束旧 native SHMEM 生命周期
       → INIT：初始化新 heap 和新的通信状态
       → ALLOCATE：按统一顺序重建全部已绑定 workspace
       → COMMIT：所有 rank 成功后发布 epoch=e+1
       → READY：继续原 forward

破坏性步骤前失败 → 保留旧 READY 状态并让本次调用一致报错
旧 heap 开始释放后失败 → FAILED，停止使用所有旧 / 半建地址
```

具体要求：

1. **协商。** 重建期间冻结 owner 清单、模块绑定和关闭；校验调用序号、触发资源、各组容量和目标字节数。
   普通不扩容路径沿用已有 count 交换；额外的 manifest / 错误一致性 collective 只放在慢路径。
2. **完成设备工作。** `workspace.in_use=False` 只表示 Host 已提交结束，不能表示 NPU 或远端访问已经完成。
   等待已登记 completion event，并在慢路径对当前 device 同步；所有 EP 成员到达 HCCL Host barrier 后才能释放。
   Host 上尚在执行的并发调用首版按既有串行限制一致拒绝，不在持有锁时等待另一个调用进入 collective。
3. **释放。** 按相同顺序释放所有 managed SHMEM tensor，复用已有 storage 失效处理。
   验证 native active allocation 为0，再 finalize。普通激活、参数、optimizer state 和未反向图的独立存储保留。
4. **重新初始化。** 在 SHMEM 生命周期层实现内部受控重建入口，维护逻辑 owner 引用关系及失败状态，
   不调用 `MegaMoeExperts.close()`，不破坏 Python 模块成员关系。
   重建期间 acquire/release/分配/提交必须被同一状态门控；需要明确锁顺序，避免新增递归锁死锁。
5. **重新引导。** 保持原 EP 成员和组内 rank；重建代际需要独立 rendezvous。
   优先验证每次重建生成新的 unique ID，并使用现有 HCCL 组分发；覆盖 WORLD 和非连续子组。
   不假定复用旧 endpoint 的 vendor 状态在任意异常后都可重用。
6. **重新分配。** 对清单中的已绑定资源重建 buffer，对未绑定资源保留预算；所有 rank 使用相同顺序。
   forward/backward event 区重新初始化，ready/completion generation 不继承旧 heap 的值。
7. **发布。** 所有 rank 确认初始化、分配和事件准备完成后，才发布新 epoch 和可提交状态。
   逻辑 workspace 对象保持稳定，内部 buffer 更新；容量、指针、ready 标志及调试信息作为一个状态切换。

整个慢路径使用所属 EP/root 的控制通信，不能加 WORLD barrier 把无关 PP/DP 组绑在一起。
HCCL ProcessGroup 在 SHMEM finalize/init 之间保持存活，作为重建协调的控制通道。

## 5. 跨扩容的 forward/backward 与地址有效性

必须支持实际训练中的顺序：`forward A → forward B 触发扩容 → backward B → backward A`。
若只能在没有任何待反向图时扩容，共享多层和多 micro-batch 场景会失去主要用途。

实现时逐项审计 `_save_forward_state`、profiling runtime、native adapter 与 device SHMEM context：

- 保存的 dispatch、激活、mapping、offset/count metadata 和返回给上层的输出，不能持有旧 heap 的 storage。
  当前 push 的 dispatch 副本、combine clone 是已有基础，但还需要存储别名断言和真实设备验证。
- `ctx.workspace` 应指向稳定的逻辑 owner；反向从该 owner 取得当前代的 buffer，不保存或复用旧物理地址。
- `MegaMoeSpec.receive_capacity` 当前是静态字段。新增独立的运行时容量状态，不能让旧 plan 的 spec
  将已扩容 workspace 判为“不兼容”。本地计算尺寸继续使用本次路由的实际接收量。
- 当前 plan 主要由静态 shape / topology 构建；优先证明它与 heap 地址、容量无关后复用。
  如果发现嵌入指针或代际相关状态，必须在提交时解析当前代或刷新相应状态，并覆盖旧 ctx 的反向。
- 新容量只增长，因此旧 forward 所需的对称接收空间不会超过新容量；反向无需因为旧路由再收缩或重建。
- 区分“旧物理 buffer 非法”与“旧逻辑 forward 仍可反向”。不能一律以 epoch 不同拒绝旧计算图。
- `retain_graph=True`、重复 backward、checkpoint 重算、交替 stream、外部权重均须保持原有语义。
  首版禁止在图捕获 / 已捕获图回放内部动态重建；固定地址假设需要单独的图兼容方案。

先完成旧代所有在途设备访问，再清零新事件区；不能仅靠 epoch 数字或 Host barrier 替代设备完成。
底层已有跨生命周期单调 allocation ID，应继续保留，覆盖释放后相同地址被重新分配的情况。

## 6. 失败语义

| 失败阶段 | 要求 |
| --- | --- |
| 布局不一致、预算不足、未纳管 owner、并发租约等预检查 | 所有 rank 在释放前得到一致错误，旧资源保留 |
| 设备同步或控制组通信失败 | 停止本次重建；不继续释放仍可能被使用的存储；按通信故障处理 |
| 已释放部分 buffer、finalize 失败、新 heap 初始化 / 分配失败 | 整组进入 FAILED，禁止继续提交或使用旧句柄；记录失败阶段及各 rank 状态 |
| rank 退出或卡在 vendor collective 中 | 使用现有通信 / vendor 超时及 launcher 超时终止该进程组；不宣称 Python 异常捕获能保证所有 rank 正常返回 |

旧 heap 一旦被销毁，不承诺无条件回滚。第一版不尝试在部分 rank 成功、部分失败时继续训练，
也不偷偷切换 pull 或丢弃 token。失败前后同步错误状态时必须考虑 peer 是否仍能进入 HCCL collective，
不能在已故障的通信组上无限追加 barrier。

内存预检查可参考所有 rank 的可用 HBM，并计入旧 heap 将释放的字节及仍存活的普通 HBM 数据；
该检查只是早期拒绝条件，不能保证后续 vendor 分配必定成功。
采用先销毁旧 heap 再分配新 heap，避免持有两份完整对称 heap；记录真实扩容期间峰值验证这一点。

## 7. 分阶段实施与文件范围

每阶段保留可独立审阅的提交，不修改既有已推送提交。

| 阶段 | 工作 | 主要位置 | 完成标准 |
| --- | --- | --- | --- |
| P0：验证底层重建 | 重跑现有2卡变 heap 用例；增加真实 PUT/GET 或 MegaMoE 通信后重复重建、fresh bootstrap、子组验证 | SHMEM lifecycle 既有 worker / launcher | 多次重建后跨卡数据正确，旧 tensor 拒绝，allocation / 引用计数无泄漏 |
| P1：容量策略 | 新增显式 grow 策略，分离初始与当前容量，复用 gathered counts，增长和总布局预算纯逻辑 | `mega_moe/{spec,route,workspace,module}.py` | 边界、对齐、几何增长、预算余量回退、跨 rank 判定一致的 UT 通过 |
| P2：root 重建协调 | owner 清单、运行时状态门控、native heap 参数、统一释放/初始化/分配、epoch 和失败状态 | `modules/module.py`、新 `mega_moe/heap_manager.py`、`shmem/_lifecycle.py`、native bindings / runtime 按需修改 | 多资源组、混合 managed workspace、延迟绑定及未知 owner 拒绝通过 |
| P3：autograd 与事件 | 保持逻辑 workspace，反向取新 buffer，审计 plan / native context / profiling，完善 ready 重置 | `mega_moe/{function,plan,workspace}.py` 及有证据需要修改的 native 路径 | 跨扩容旧图、重复 backward、checkpoint、交替 stream 数值正确 |
| P4：设备回归与故障 | 溢出增长、慢 rank、子组、异常阶段、长期循环 | 复用现有 MegaMoE memory / ready / resources worker 和 SHMEM lifecycle worker | 无死锁、无越界、无旧地址复用、稳定路由无额外扩容 |
| P5：Qwen 验收与文档 | 初始/逐步对齐审计、受控负载性能和内存、扩容成本及实际 heap 观测 | Qwen benchmark / README、multicore docs | 报告稳态、扩容步和摊销结果，给出是否适合默认启用的依据 |

测试优先合并到现有文件和参数化矩阵，不复制出大量独立 worker。
实现前按需读取 `add-unit-test` skill；ST launcher 继续保持无框架导入。
本次计划阶段不需要切换共享 Python 环境；开始实现和运行前，为新 worktree 准备独立环境 / editable 安装，
核对 `hyper_parallel.__file__` 和 native 制品来源，防止测到旧 worktree。

## 8. 验证矩阵与验收条件

### 8.1 功能和生命周期

| 场景 | 核心断言 |
| --- | --- |
| `R_max=C`、`R_max=C+1`、跨多个增长档位 | 等于容量不扩容；超过时所有 rank 选择同一目标；当次 token 无丢失 |
| 均衡 → 热点 → 更强热点 → 均衡 | 按需增长、回落不缩容、恢复后结果正确且不重复重建 |
| 热点 rank 切换、部分 rank 零接收 | 所有 rank 都参与重建；空接收仍满足 native 非空 ABI 占位约定 |
| 两层共享一个 workspace；两个不同规格的独立资源组 | shared 只计一次；另一资源组在重建后继续正确运行 |
| 尚未绑定的后续模块；静态 push / pull 共存 | 预算与统一分配顺序一致；非触发组旧地址得到更新 |
| 同一 root 的额外 acquire 或外部 allocation | 在旧资源未释放前一致拒绝，外部数据保持有效 |
| 两个 forward 后才反向，扩容发生在两者之间 | 新旧图的输入、Router、专家梯度均与固定大容量基准一致 |
| retain_graph、重复 backward、checkpoint、交替 stream | 更新后 workspace 可串行复用；mapping 与保存激活不被重建破坏 |
| 非连续子组 `[0,2]`、`[1,3]` | 一个组扩容不要求另一个组进入同一重建 barrier；组内 rank 映射正确 |
| rank 延迟、配置不一致、预算不足、init/finalize 故障 | 预检查保留旧状态；破坏性失败后不再使用旧地址；进程组有有界退出方式 |
| 重复增长与最终 close | 无 allocation / reference 泄漏，旧 tensor 不能写入后续复用的地址 |

单测落到现有 `tests/ut/core/multicore/modules/mega_moe/test_{route,workspace,module,function}.py`
和 `tests/ut/core/multicore/shmem/test_runtime.py`；协调状态测试仅在现有文件难以承载时新增一个专用文件。
设备回归优先扩展 `_test_mega_moe_memory.py`、`_test_mega_moe_ready.py`、`_test_mega_moe_resources.py`
及 `shmem/_test_lifecycle.py`，复用对应 launcher。

### 8.2 公平的性能与 HBM 比较

固定规格：H5120 / I1792 / EP8 / E48 / seq4096 / TopK8，BF16，2层，8卡。
对照静态无损 push、grow push、pull，native 作为整网参考；先验证每个 rank 的输入和规范化权重 hash。

分成三类实验，结果不能混在同一个“稳态延迟”数字中：

1. **相同状态的整网训练步。** 每次计时前从同一份模型及 FP32 master / AdamW 状态恢复，恢复时间不计入训练步。
   同时检查每层 TopK 和每 rank 接收量；相同权重不自动保证后层 TopK 完全相同，差异必须记录。
   用少量逐步对照定位初始数值分歧，再开始性能测量。
2. **明确固定的路由回放。** 相同 token 激活、专家权重、TopK IDs/weights 和梯度，比较静态 / grow push 的稳态与扩容步；
   可纳入 pull，但标明这是受控专家计算/通信实验，不冒充学习式整网轨迹。
3. **真实连续训练轨迹。** 保留实际参数更新，逐步记录路由、容量、重建次数和 loss。
   用于验证无损语义、长期运行和摊销成本；不同后端路由分化后，不再用单个均值归因通信收益。

负载矩阵至少包括均衡、约1.5倍/2倍/3倍偏斜、突发热点以及热点 rank 轮换；
按实际生成的 TopK 统计标注不均衡度，不只报告请求的目标值。

必须记录：

- 每步 rank-max 耗时、逐层接收分布、`R_max / mean`、当前容量与 heap epoch。
- 扩容步总延迟，以及协商 / quiesce / free / finalize / init / allocation 各阶段耗时。
- 不含扩容的稳态中位数和尾延迟；包含扩容的总时间、扩容次数、每步摊销时间。
- Torch peak allocated、peak reserved、实际 SHMEM bytes、设备 HBM 采样峰值。
  heap 可变后，按代际 / 时间对齐 allocator 与 heap，不能把两个不同时刻的独立峰值直接相加。
- 分别统计初始化、稳态和扩容窗口的峰值；测量脚本的恢复快照、双模型对照或回放缓存不能混入单后端 HBM。
  若恢复数据有不可避免的常驻开销，单独量化并注明，设备峰值以另一个无这些辅助常驻数据的运行交叉验证。
- 设备任务干扰、软件 / 制品版本、实际配置与所有原始样本。

验收要求：数值和 token 数量与固定大容量 push 一致；溢出前不进入非法通信；稳定负载不反复重建；
无扩容路径不新增负载 collective；受控路由下稳态开销落在实测重复噪声范围内。
分别回答“小 heap 节省多少”“一次重建多贵”“多久摊销回来”，不预设最终性能胜负。

## 9. 首先要完成的三件事

1. 在新 worktree 的独立环境中运行现有变 heap lifecycle 用例，并验证通信后重建和 fresh bootstrap。
2. 画清同一 SHMEM root 的全部 owner / allocation 清单，落实可保持逻辑 workspace 的受控重建入口。
3. 以2卡、两个 outstanding forward 的最小用例完成“第二次 forward 扩容，随后两次 backward”，
   再接入多资源组及8卡 Qwen；该用例不过，不进入性能结论阶段。

本文件的容量增长系数、接口名及验证顺序是建议实施决策；没有把未运行的设备验证写成已完成结果。
