# Profiling 输出文件清单与表头含义

基于公开的 MindStudio **msprof** 与 `torch_npu.profiler`。本页登记的文件名与
表头是在 **CANN 9.0.0 + torch_npu 2.9.0.post2 / Ascend 910** 上实采核对过的，
不是按通配猜的。版本差异处标 `[待对齐]`。

## 目录结构

```
<out>/<host>_<pid>_<ts>_ascend_pt/
├── ASCEND_PROFILER_OUTPUT/      ← 解析后的产物，breakdown.py 只读这一层
├── FRAMEWORK/                   ← torch 侧原始落盘（op_mark / op_range / python_tracer）
├── PROF_<n>_<ts>_<id>/          ← CANN 侧原始落盘（host/ device_<n>）
├── logs/
├── profiler_info.json           ← 单卡；分布式为 profiler_info_<rank>.json
└── profiler_metadata.json
```

`breakdown.py --prof-dir` 可传 `ASCEND_PROFILER_OUTPUT` 本身、`*_ascend_pt`
目录、或装着多个 rank 目录的父目录（多卡时逐 rank 解析）。

## ASCEND_PROFILER_OUTPUT 实测文件

| 文件 | 内容 | 用于 |
|---|---|---|
| `step_trace_time.csv` | 迭代级时间切分 | **步时加和拆解**（见下） |
| `op_statistic.csv` | 按算子类型聚合的 device 耗时 | 计算维度分类、AI_CPU 回退热点 |
| `kernel_details.csv` | 逐 kernel 明细（含 aic_*/aiv_* 流水线指标） | 单算子深挖、cube_utilization |
| `operator_details.csv` | torch 层区间（`aten::*`、autograd、`Optimizer.step#*`） | **优化器耗时**、框架视角 |
| `communication.json` | 逐 step 的 collective/p2p 明细 | 通信维度、链路带宽 |
| `communication_matrix.json` | 通信对矩阵 | 通信域拓扑 |
| `api_statistic.csv` | host 侧 ACL API 统计 | 下发瓶颈 |
| `trace_view.json` | chrome tracing 时间线 | 人工看重叠/气泡 |
| `ascend_pytorch_profiler*.db` / `analysis.db` | 同数据的 DB 形态 | 大 trace 时替代 CSV |
| `analyse.done` | 解析完成标记（空文件） | 判断采集是否完整落盘 |

> **没有 `op_summary` 这类文件。** 早期资料里的 `*_op_summary*.csv` 在该版本
> 不存在；按它通配会一个文件都匹配不到。

## step_trace_time.csv

```
Device_id,Step,Computing,Communication(Not Overlapped),Overlapped,Communication,
Free,Stage,Bubble,Communication(Not Overlapped and Exclude Receive),Preparing
```

**加和关系（实测恒等式，breakdown.py 会校验并报偏差）：**

```
Stage = Computing + Communication(Not Overlapped) + Free
```

| 列 | 含义 | 是否在加和内 |
|---|---|---|
| `Computing` | 计算耗时（**含优化器**） | 是 |
| `Communication(Not Overlapped)` | 未被计算掩盖的通信 | 是 |
| `Free` | 空闲/下发气泡 | 是 |
| `Stage` | 上三者之和 | 合计 |
| `Communication` | 通信总耗时 | 否（参考） |
| `Overlapped` | 被计算掩盖的通信 | 否（参考）|
| `Bubble` | PP 气泡；无 PP 时为 0 | 否（参考）|
| `Preparing` | step 前的 host 准备（取数等） | 否（参考）|

掩盖率 = `Overlapped / Communication`。该值低且 `Communication(Not Overlapped)`
占比高，才是真的通信瓶颈。

## 优化器耗时的取法

`operator_details.csv` 里有 torch 的 `record_function` 区间，**每个采集 step 恰好
一条** `Optimizer.step#<优化器名>.step`（如 `Optimizer.step#AdamW.step`），其
`Device Total Duration(us)` 即该步优化器的 device 耗时（`Device Self` 为 0，
因为它是父区间）。

**不要按算子名求和代替它**：AdamW 的 foreach 实现落成 `ForeachMulScalar` /
`ForeachLerpScalar` / `ForeachAddcmulScalar` / `ForeachSqrt` /
`ForeachDivScalarList` / `ForeachAddScalar` / `Addcdiv` / `Pows` 等一串算子，
其中 `Pows`（优化器的 bias correction）与 `Pow`（模型里的幂运算）只差一个字母。
实测两种口径相差约 2%，但按名求和既易错又不可移植。

`Optimizer.zero_grad#*` 的 device 耗时为 0（`set_to_none=True` 只动 host）。

**优化器是 `Computing` 的子集，不是第四个加和项。** 报告里单列它是为了看占比。

## 计算维度分类

`op_statistic.csv` 的 `Core Type` 与 `kernel_details.csv` 的 `Accelerator Core`
取同一取值集合，实测为：

| 取值 | 归类 |
|---|---|
| `AI_CORE` | Cube |
| `AI_VECTOR_CORE` | Vector |
| `MIX_AIV` | 混合 |
| `AI_CPU` | AI CPU（**含通信 launch kernel，见下**）|
| `COMMUNICATION` | 通信 kernel（`Type` 形如 `hcom_allReduce_`）|

> 早期资料里的 `Task Type` 列**不存在**，按它判 AI_CPU 会全部落空。

**AI_CPU 里要先剔掉通信**：HCCL 每个集合通信都会起一个 AI CPU 侧的 launch
kernel（`allreduceAicpuKernel` 等 `*AicpuKernel`）。它跑在 AI CPU 上，但属于
通信，不是「需要消除的算子回退」。实测一次双卡采集里它占 AI_CPU 总时间的
17%，不剔掉就会误报。`breakdown.py` 把它计入 AI_CPU 类别总量、但排除在回退
热点表之外。

真回退（如整型 ArgSort 落 AI CPU）才是要交给 perf-playbook 的。

## communication.json

结构：`{"step<N>": {"collective": {...}, "p2p": {...}}}`，内层键为
`<算子名>@<通信域 hash>`，外加一个聚合键 **`Total Op Info`**。

- 取每步总量读 `Total Op Info`；**枚举单个通信算子时必须跳过它**，否则总量翻倍。
- `Communication Time Info`：`Elapse Time(ms)` / `Transit Time(ms)` /
  `Wait Time(ms)` / `Synchronization Time(ms)` / `Idle Time(ms)` 及对应 Ratio。
  `Wait` 占 `Elapse` 高说明瓶颈在同步等待而非带宽。
- `Communication Bandwidth Info`：按 `RDMA` / `HCCS` / `PCIE` / `SDMA` / `SIO`
  分链路给 `Transit Time(ms)` 与 `Transit Size(MB)`。
  **`MB/ms` 即 `GB/s`**，不要再除 1000。实测单机双卡 allreduce 走 SIO/SDMA，
  约 162 GB/s；`SDMA` 与 `SIO` 两项数值相同（同一路径的两种记法），不要相加。
- 交叉校验：`Total Op Info` 的 `Elapse Time(ms)×1000` 与 `step_trace_time.csv`
  的 `Communication` 列实测逐微秒吻合，可用来判断解析是否正确。

## 采集元信息（provenance）

- `profiler_info<_rank>.json`：`cann_version`、`torch_npu_version`、`rank_id`、
  `config.common_config.schedule`（`wait`/`warmup`/`active`/`repeat`）、
  `config.experimental_config._profiler_level`。
  **单卡与分布式不同**：单卡写 `profiler_info.json` 且**没有 `rank_id` 字段**；
  分布式写 `profiler_info_<rank>.json` 且带 `rank_id`。
- `profiler_metadata.json`：`ENV_VARIABLES` 与 **`parallel_group_info`**
  （`group_name` / `group_rank` / `global_ranks`）——这就是通信域→并行轴映射的
  来源。只有一个通信组时可直接对应；多组时需把组名与 `communication.json` 的
  `@hash` 对上，该 join key 在 metadata 中暂未暴露，标 `[待对齐]`。

`schedule` 还让「跳过 warmup / 稳态至少 2 步」这条硬规则可被机器校验，
`breakdown.py` 把它作为一致性校验项输出。

## 采集注意

- 跳过 warmup，采稳态若干步；单步未重复不可靠（已成 breakdown.py 的校验项）。
- 采集本身有开销，拆解占比以相对值为准，不把带 profiling 开销的绝对步时当作
  真实步时。
- `kernel_details.csv` 的 `Start Time(us)` 单元格带尾部制表符，`Input Shapes`
  是被 CSV 二次转义的引号串（`"""1,2048,128"""`）；数值解析要 `strip()`。
