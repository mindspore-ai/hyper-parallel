# 已有训练模型的接入与运行检查

本文补充 [组件迁移教程](../examples/README.md#已有模块与权重迁移)之外的训练边界。
已有的 `from_module`、自动 token 容量和参数持有方式继续使用；诊断接入现有
`diagnostics`，与 [profiler 和 Tensor dump](mega_kernel_profiling.md)
共用会话和零起始 step 调度，不需要修改业务入口文件或管理恢复 manifest。

## 先检查，再采样

在已经构造的模型上读取执行配置，不会绑定 SHMEM、加载 native 库、materialize 参数或执行 collective：

```python
import json
from hyper_parallel.core.multicore import diagnostics

report = diagnostics.inspect_execution(model, optimizer=optimizer)
print(json.dumps(report, indent=2))
```

报告包含实际模块类型、参数 Tensor 类型/dtype、逻辑 shape/mesh/placement、残留 meta 参数、优化器各组的 fused/foreach、
当前已加载 native 路径，以及每个专家执行器的参数模式、EP 成员顺序和资源共享状态。
未绑定执行器的实际容量、heap 大小为 `null`；构造配置不是已分配内存。
首次执行才加载的 native 库会在对应 step 报告中更新，不以启动时的空列表判断运行后仍未加载。
多个配置兼容的独立组会产生提示，只有确认层串行执行后才调用 `share_execution_resources`。
静态报告不能证明 empty 参数已正确初始化，也不能自动证明 FSDP 参数归属正确。

对原训练循环加一个明确的 context 和 step 边界，即可采集 Host 运行信息：

```python
with diagnostics.execution_monitor(
    model,
    optimizer=optimizer,
    output_dir="moe_execution",
    schedule=diagnostics.schedule(wait=2, warmup=0, active=3, repeat=1),
) as monitor:
    for batch in loader:
        optimizer.zero_grad()
        loss = model(batch)
        loss.backward()
        optimizer.step()
        monitor.step()
```

例中采集零起始 step 2、3、4。`step()` 在完整训练 step 结束后调用，包括未采样的 step；
从 checkpoint 恢复时，这个编号是 session 内部序号，不自动等于训练全局步号。
不加 context 时保持普通执行。该会话与 profiler/dump 互斥，需分轮采集。

每进程创建独立 `rank_*_pid_*` 子目录，包含 `identity.json`、选中 step 的 `step_*.json`
和关闭状态 `session.json`。不同轮次使用不同根目录；多节点需要收齐所有节点的子目录。

| 观察内容 | 解释 |
| --- | --- |
| `route` | 复用已交换并读回的来源 T 和 owner 接收量；最大/均值可定位热点 |
| `plan` | 当前 bucket、覆盖行数、命中或构建、构建 CPU 包络；不引入新的计划策略 |
| `moe` | 来源本地 T、计划行数、接收缓冲容量及共享组标识 |
| `execution_modules` | 本 step 结束时已提交容量、缓存 key、接收容量和 SHMEM heap 字节数 |

Host 观察不会为路由再执行一次 collective、Tensor 拷贝或 NPU 同步，也不修改 kernel 参数。
它仍有 Python 记录和文件输出开销，正式性能另行关闭诊断测量。
计划构建时间包含在前反向之内，不能与前反向包络相加；该报告不提供整步设备计时。
`shmem_heap_bytes` 属于 root heap，同一组下多个层显示的值不能累加；它不包含待反向激活或 optimizer state。
`moe.local_receive_capacity` 包含空 owner 的至少一行 ABI 占位；真实接收负载使用 `route.received_rows`，
不能把保留容量或占位行当作实际 token 数。
默认每 step 最多 10000 条事件，超过会记录 `dropped_events`；按需调整 `max_events_per_step`。
异常中断时不会把未调用 `step()` 的当前记录当成完整样本。

汇总时显式给出所有预期 rank 和 step，避免只检查已产生的采样步：

```python
coverage = diagnostics.summarize_execution_reports(
    "moe_execution", world_size=16, expected_steps=[2, 3, 4],
)
print(json.dumps(coverage, indent=2))
```

`complete` 只表示覆盖完整且无重复/截断，不表示数值或性能验收通过。
`observations` 按 step/rank 汇总计划构建次数和 CPU 包络、owner 最大/均值和实际本地 T。
也可以直接执行一行离线汇总，结果写入根目录的 `summary.json`；不完整时退出码为 2：

```bash
python -m hyper_parallel.core.multicore.diagnostics summarize ./moe_execution --world-size 16 --steps 2,3,4
```

单组件入口已有一行采样命令，采集其一次前向/反向，并非整网 optimizer step：

```bash
python -m hyper_parallel.core.multicore.examples.minimal_moe --diagnostic-output ./moe_execution
```

## 参数、初始化与并行归属

- `from_module` 转换实际存在的权重，需在正确 EP 切片和 materialization/初始化/加载之后、
  构造优化器之前调用。meta 权重会明确报错；优化器状态迁移仍按原教程单独处理。
- 保留已有 FSDP/参数管理器时，使用 `create_parameters=False`，在原参数所有者 forward 中传入
  当前已 unshard 的本地专家权重。不要同时保留两套可训练专家参数，也不要缓存跨更新的 BF16 转换副本。
- Dense FSDP 与专家 EP 可以用于同一组卡上的不同参数。纯 EP owner 独占的专家应排除在 dense FSDP 之外；
  专家有 EDP/FSDP 副本时，仅在同一 owner 的副本组分片，并在执行前收齐局部专家。
- `ep_group` 是实际通信拓扑。外部框架的 YAML EP 字段不一定直接驱动它；避免框架与适配层重复应用 EP。
  checkpoint 专家切片使用组内 rank 顺序，不用 WORLD rank 替代子组 rank。
- 在首次 forward 前共享配置兼容且串行执行的专家执行器。共享 workspace 不共享参数，也不消除
  每个 autograd 图保存的激活；热点接收和多层在途反向仍占普通 HBM。
- `to_empty()` 后需运行模型定义的初始化/加载流程，覆盖自定义 norm、router 和派生 buffer。
  检查有限值或固定 seed 都不能证明全局初始化一致，不要通过统一清零或训练后重置 norm 掩盖遗漏。

梯度归一化属于训练目标。纯 EP16、无专家副本、目标为 16 个来源 loss 均值时，owner 汇总的专家 dW
需要且只需要对应的一次平均；如果 loss 或框架已经缩放，不能再除 16。dense/router/shared expert
沿用各自归约规则。梯度累积、token 加权 loss 或专家副本需要重新推导，clip 按唯一参数计数。

MoE 的 fused 配置与 AdamW 的 fused/foreach 独立。核对实际参数、grad、FP32 master、moment 的
dtype 与 DTensor layout；普通 Tensor 能运行 fused AdamW，不代表 DTensor 或混合参数路径已支持。
不在 multicore 中静默切换优化器或绕过分布式 dispatch。

## 动态输入与性能对照

执行器按实际输入自动维护容量，不需要 token 上限或动态/静态模式开关。
旧 `local_num_tokens`、`max_local_num_tokens` 关键字仅兼容接收并忽略，两者同时传入也不报错；
修改外部 max_seq_len 不会自动更新显式构造参数。source 容量、调度 bucket、owner 接收量是不同概念。
先看报告中的 miss 和实际 T，再分析构建开销；没有 miss 时检查 owner 热点和其他训练阶段。

先用相同全局初始权重与输入验证输出、dX、router/expert dW、clip、optimizer state 和更新后参数，
再扩大层数、重算和动态输入范围。性能比较统一 FSDP、精度、优化器、EMA、确定性及 launch blocking，
保留实际 packed 段长度和文本/图像工作量。EP1 收齐权重与 EP16 路由 token 的对比包含策略变化，
相同 EP owner 布局下的 native/Mega 对照才可进一步隔离执行后端。
