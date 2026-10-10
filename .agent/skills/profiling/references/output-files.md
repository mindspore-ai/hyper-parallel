# Profiling 输出文件清单与表头含义

基于公开的 MindStudio **msprof** 与 `torch_npu.profiler`。采集后得到一个
输出目录，内含多类文件。本页登记常用文件的内容与关键表头含义，供
`breakdown.py` 解析与人工核对。随工具版本与实践补充；版本差异标 `[待对齐]`。

## 典型输出集（msprof / torch_npu profiler）

| 文件 / 目录 | 内容 | 用于 |
|---|---|---|
| `*_op_summary*.csv` | 逐算子耗时汇总（Device 侧） | 计算维度拆解、算子视角 |
| `*_op_statistic*.csv` | 算子类型聚合统计 | 按算子类型归类（Cube/Vector/FA） |
| `*communication*.json` / `*.csv` | 集合通信算子耗时与通信域 | 通信维度拆解、未掩盖通信 |
| `*step_trace*.csv` | 迭代级时间线（前向/反向/优化器/通信段） | 步时四维切分 |
| `*timeline*.json`（trace view） | 可视化时间线（chrome tracing 格式） | 人工看重叠/气泡 |
| `*memory*` | 显存相关（本 skill 不主用，见 memory-analysis） | 交叉参考 |

> 具体文件名前缀随 msprof 版本/采集配置变化；`breakdown.py` 用通配匹配、
> 列名防御式解析，缺列时标 `[待解析]` 而非臆测。

## 关键表头（op_summary 常见列）

| 列 | 含义 |
|---|---|
| `Op Name` | 算子实例名 |
| `OP Type` | 算子类型（用于归类计算class） |
| `Task Duration(us)` | 该算子 device 执行耗时 |
| `Task Type` | AI Core / AI CPU / HCCL 等（AI CPU 多为回退热点）|

## 关键表头（communication 常见项）

| 项 | 含义 |
|---|---|
| 通信算子类型 | AllReduce / AllGather / ReduceScatter / All2All 等 |
| 通信域 / group | 对应的并行轴（需与并行配置映射，见 breakdown-dimensions）|
| 耗时 / 是否被计算掩盖 | 未掩盖通信计入「通信（未掩盖）」维度 |

## 采集注意

- 跳过 warmup，采稳态若干步；单步未重复不可靠。
- 采集本身有开销，拆解占比以相对值为准，不把带 profiling 开销的绝对步时
  当作真实步时。
- `Task Type=AI_CPU` 的算子是常见后处理热点（如整型 ArgSort、
  scatter_reduce 回退），标出来供 perf-playbook 选手段。
