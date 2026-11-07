# 知识源地图

各类知识在本仓的落点。检索按「规则/技能 → 文档 → 代码 → 测试 → 官网」的
顺序，先权威后广泛。

## 仓内

| 想知道 | 去哪 |
|---|---|
| 项目总则、技能/代理/规则目录 | `AGENTS.md`（索引，顺指针往下走）|
| 硬性约束（代码风格、分布式、测试、精度验收等）| `.agent/rules/**` |
| 做某件事的流程（提交/门禁/评审/实验/调试/精度/性能/显存/特性文档）| `.agent/skills/**` 的 `SKILL.md`，细节在其 `references/` |
| 特性行为、对外接口、使用限制 | `docs/**`（按主题分目录：`docs/guide/**` 使用指南、`docs/api/**` 接口、`docs/feature_status.md` 支持矩阵）|
| 并行实现（DTensor/TP/FSDP/PP/EP/CP、激活、checkpoint）| `hyper_parallel/core/**` |
| 并行策略的**对外装配入口**（`apply` / builder / 各并行的 wrapper）| `hyper_parallel/distributed/**`（`expert_parallel/`、`context_parallel/`、`activation_checkpoint.py` 等；与 `core/**` 的关系是装配 vs 实现）|
| 自动并行 / 策略搜索 / offload 规划 | `hyper_parallel/auto_parallel/**`（`sapp_nd`、`sapp_ppb`、`hyper_offload`、`config_adapter`）|
| 模型适配（各模型族）| `hyper_parallel/models/<family>/adapter/**` |
| 共享组件/算子封装 | `hyper_parallel/components/**` |
| 训练器与配置字段 | `hyper_parallel/trainer/**`（配置字段是回答"开关在哪"的关键）|
| 数据集 / 分词 / batching / 数据并行读取 | `hyper_parallel/data/**` |
| 图编译与 pass | `hyper_parallel/compile/**`（`passes/`、`tracer/`、`pass_plan.py`）|
| 进程组 / 通信域构建 | `hyper_parallel/collectives/cc.py` |
| 外部框架集成 | `hyper_parallel/integration/**`（当前为 llamafactory）|
| 推理侧（generation / kv cache / sampler）| `hyper_parallel/infer/**` |
| 通用工具（日志、实验环等，不属于任何并行实现）| `hyper_parallel/tools/**` |
| RL | **走 `.agent/rules/hyper-rl.md`**，它是 RL 的唯一入口；不要直接从本表进 `hyper_parallel/rl/**` |
| 契约与期望行为 | `tests/ut/**`（CPU 可跑）、`tests/**` 的 ST |
| 可跑的 recipe / 启动方式 | `examples/**` |

> 上表按 master 的顶层目录逐个对过，`hyper_parallel/**` 下无未登记的目录。
> 漏登记一个目录的代价不是"查得慢一点"，而是**查不到就去编 API**——
> `distributed/**` 与 `auto_parallel/**` 曾不在表内，而后者是本仓第二大目录。

## 仓外（仓内查不到再去）

| 想知道 | 去哪 |
|---|---|
| 框架官方文档 | MindSpore / HyperParallel 官网文档 |
| 算子与 CANN 行为 | Ascend 官方文档、算子仓 |
| 上游模型结构 | 上游模型仓（HF 等）|

## 常见问题的最短路径

| 问题 | 路径 |
|---|---|
| "这个开关怎么配" | `hyper_parallel/trainer/config/**` 找字段 → 再看 `examples/**` 的 recipe 用法 |
| "X 特性支持吗" | `docs/feature_status.md` / `docs/**` → 再到对应模块代码确认 |
| "这个报错什么意思" | 搜报错片段：先 `.agent/skills/*/references/`（已沉淀的坑）→ 再代码抛出点 |
| "这个模块的契约是什么" | 对应 `tests/ut/**` 的用例名与断言 |
| "为什么这么实现" | 代码 docstring + 对应 skill 的 references（设计取舍常记在那里）|

## 纪律

- 文档与代码冲突时以代码为准，但要**明说冲突**，不要静默选一边。
- 一条匹配行只是线索，不是答案；读到定义/函数/用例级别再回答。
- 仓里查不到就说查不到，不要编 API。
