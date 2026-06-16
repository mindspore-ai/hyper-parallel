# 自动并行使用指南

HyperParallel 提供 AutoParallel（自动并行）能力，主要包含 SAPP-ND 和 SAPP-PPB：SAPP-ND 用于多维混合并行策略搜索，SAPP-PPB 用于 Pipeline Parallel stage 负载均衡和 offset/recompute 联合调优。

## 核心概念

| 模块 | 说明 | 适用场景 |
|------|------|----------|
| SAPP-ND | N Dimensional Parallelism 并行策略生成，提供 top T 个最优并行配置 | DP, TP, PP, EP 多维并行自动配置 |
| SAPP-PPB | Pipeline Parallelism Balancing，自动平衡 PP stage 的计算和内存负载 | PP stage offset 和 recompute 联合自动调优 |

---

## SAPP-ND：ND 搜索

ND 提供 N 维并行策略的生成。它会根据模型配置、硬件规模、全局 batch size 和内存预算生成候选并行配置，先通过内存估算过滤 OOM 配置，再通过性能估算对剩余配置排序。由于内存和性能都采用解析式估算，ND 搜索不依赖在线 profiling，可以在 CPU-only 环境中完成规划。

```bash
python -m hyper_parallel.auto_parallel.sapp_nd.nd.run_nd \
    -y <yaml> \
    -l DP MP PP EP MB MBS \
    -d 1024 \
    -b 2048 \
    -t 10
```

常用输入包括：

- 模型类型和模型超参数
- 待搜索的并行维度
- 硬件类型和设备数
- global batch size 和内存预算

SAPP-ND 开放的并行维度：

- `DP`：数据并行
- `TP`：模型并行或张量并行
- `SP`：Megatron sequence parallel
- `EP`：专家并行
- `PP`：流水线并行
- `MB`：micro-batch 数
- `MBS`：micro-batch size
- `VPP`：virtual pipeline parallelism
- `CP`：context parallelism (适配中)
- `OP`：优化器或 ZeRO-DP 并行 (适配中)

SAPP-ND 模块包含：

- `nd/`：ND 搜索入口、并行空间构造和排序逻辑
- `memory_estimation/`：内存估算
- `perf_estimation/`：性能估算

ND 的输出会给出满足内存约束的候选配置和排序结果，并生成全局并行策略的性能对比图，便于选择当前约束下的最优并行配置。

![SAPP-ND 输出示例](../../hyper_parallel/auto_parallel/sapp_nd/figures/plot_example.png)

图中展示候选并行配置的性能排序和对比结果。

---

## SAPP-PPB：Pipeline Parallelism Balancing

SAPP-PPB 用于 Pipeline Parallel stage 负载均衡。它读取每类 layer 在不同重计算或 swap 策略下的耗时和内存信息，以及 PP stage 数、micro-batch 数、interleave 数、调度策略等流水线配置，然后构造整数线性规划问题，由求解器返回 layer 分配、offset 和 recompute 配置，并可通过 simulator 查看流水线效果。

```bash
python hyper_parallel/auto_parallel/sapp_ppb/run_pipeline_balance.py \
    -m llama2_70b \
    -s 16 \
    -mb 32 \
    -i 2 \
    -O 2
```

SAPP-PPB 的核心流程：

1. 解析 layer 描述和 pipeline 配置，生成 ILP 问题。
2. 使用第三方求解器求解；默认使用 PuLP，可选 Gurobi。
3. 输出 balancing 结果，并通过 pipeline simulator 生成可视化结果。

SAPP-PPB 模块包含：

- `run_pipeline_balance.py`：PPB 命令行入口
- `sapp/`：问题构造和 ILP 求解逻辑
- `simulator/`：pipeline block 级调度模拟和可视化
- `layers/`：layer 描述样例
- `cfgs/`：手动配置样例

PPB 的输出包括 pipeline simulator 可视化图，以及可直接落到训练配置里的 layer 分配、offset 和 recompute 结果。

![SAPP-PPB 输出示例](../../hyper_parallel/auto_parallel/sapp_ppb/figures/ex_result.png)

图中展示不同 PP stage 的执行时间线和负载分布。

```yaml
offset:    [[0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,1,1], [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,-2]]
recompute: [[3,1,1,1, 1,0,0,0, 0,0,0,0, 0,0,0,0], [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0, 0]]
```

`offset` 表示各 virtual stage 的层数偏移，`recompute` 表示各 stage 采用的重计算策略。

---

## 性能建议

1. **ND 并行配置调优用 SAPP-ND**：先用 ND 搜索 DP/MP/PP/EP/MB/MBS 等多维组合，得到满足内存约束并按性能排序的候选配置。
2. **Pipeline 调优用 SAPP-PPB**：当 PP stage 分配、VPP、offset 或 recompute 成为主要问题时，用 PPB 做流水线负载均衡。
3. **建议先定全局并行配置，再细调流水线**：优先用 ND 确认较优的整体并行配置，再基于该配置使用 PPB 调整 pipeline offset 和 recompute。
4. **当前为 demo 特性**：相关功能正在逐渐完善及扩展。
