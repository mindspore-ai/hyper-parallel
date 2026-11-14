# 缩写与中英对照

检索时把缩写**展开**、把全称**缩写**，中文词映射到代码里的英文标识符——
同一概念在文档与代码里常是不同字符串，只搜一种必然漏。

## 并行与分布式

| 缩写 | 全称 | 中文 | 代码里常见写法 |
|---|---|---|---|
| TP | tensor parallel | 张量并行 | `tensor_parallel`, `tp_size` |
| DP | data parallel | 数据并行 | `data_parallel`, `dp` |
| PP | pipeline parallel | 流水线并行 | `pipeline_parallel`, `pp_size` |
| EP | expert parallel | 专家并行 | `expert_parallel`, `ep` |
| CP | context parallel | 上下文/序列并行 | `context_parallel`, `cp_size` |
| FSDP / HSDP | (hybrid) fully sharded data parallel | 全/混合分片 | `fully_shard`, `hsdp` |
| DTensor | distributed tensor | 分布式张量 | `dtensor`, `placement`, `Shard`, `Replicate` |

## 激活与显存

| 缩写 | 全称 | 中文 | 代码里常见写法 |
|---|---|---|---|
| AC | activation checkpoint | 激活重算 | `activation_checkpoint`, `recompute` |
| SAC | selective activation checkpoint | 选择性重算 | `selective`, `make_selective_checkpoint_context_fn` |
| swap | activation swap | 激活交换/换出 | `activation_swap`, `offload` |
| OOM | out of memory | 显存不足 | `OutOfMemoryError` |

## 计算与算子

| 缩写 | 全称 | 中文 | 代码里常见写法 |
|---|---|---|---|
| FA | flash attention | 融合注意力 | `flash_attention`, `fused_attention` |
| MoE | mixture of experts | 混合专家 | `moe`, `router`, `experts` |
| CSA | compressed sparse attention | 压缩稀疏注意力 | `compressed_*`, `shared_compressed_dsa_attention` |
| MTP | multi-token prediction | 多 token 预测 | `mtp`, drafter 相关 |
| UT / ST | unit / system test | 单元 / 系统测试 | `tests/ut`, `tests/.../st` |

## 常见中文词 → 代码英文

| 中文 | 代码 |
|---|---|
| 重算 / 重计算 | `recompute`, `checkpoint` |
| 显存 | `memory`（设备侧，非 host RAM）|
| 精度 | `precision`, `dtype`, `accuracy` |
| 并行度 | `*_size`（`tp_size` 等）、`parallel` |
| 切分 / 分片 | `shard`, `split`, `Shard` |
| 通信域 | `group`, `process_group` |
| 开关 / 配置项 | `config` 字段（不是环境变量）|
| 门禁 | gate / CI（`.agent/skills/gate-doctor`）|
| 首错 | first error（集群诊断用语）|

## 用法

- 搜中文词去 `docs/**`，搜英文标识符去 `hyper_parallel/**`。
- 缩写与全称都搜一遍（`SAC` 与 `selective activation checkpoint`）。
- 标识符试多种拼写：`snake_case` / `CamelCase` / 连字符。
- 本表随实践补充；新缩写进项目时在此登记。
