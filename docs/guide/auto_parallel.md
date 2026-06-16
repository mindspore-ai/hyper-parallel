# 自动并行使用指南

HyperParallel 提供 AutoParallel（自动并行）能力，自动搜索最优的多维混合并行策略。包括 Fast-Tuner（黑盒代价模型搜索）和 SAPP 系列（Pipeline Parallelism Balancing + ND 搜索）。

## 核心概念

| 模块 | 说明 | 适用场景 |
|------|------|----------|
| Fast-Tuner | 基于 profiling 的黑盒代价模型搜索 | 全局多维混合并行策略 |
| SAPP-PPB | Pipeline Parallelism Balancing | PP stage 分配平衡 |
| SAPP-ND | ND 搜索（内存估算 + 性能估算） | 多维并行策略搜索 |

---

## Fast-Tuner

基于 profiling 信息构建黑盒代价模型，通过枚举、剪枝、搜索自动生成多维混合并行策略。

> **注意**：Fast-Tuner 目前为 demo 特性，仍在持续优化。

```python
# Fast-Tuner 通过 profiling 信息自动搜索最优并行策略
# 详见 hyper_parallel/auto_parallel/fast-tuner/
```

---

## SAPP-PPB：Pipeline Parallelism Balancing

自动平衡 Pipeline Parallel 的 stage 分配，确保各 stage 计算负载均衡。

```python
from hyper_parallel.auto_parallel.sapp_ppb import run_pipeline_balance

# 自动计算最优 PP stage 分配
balance_result = run_pipeline_balance(
    model_config=model_config,
    num_stages=pp_size,
    num_devices=device_count,
)
```

SAPP-PPB 模块包含：

- `sapp/sapp_solver.py`：搜索求解器
- `sapp/sapp_pipeline.py`：Pipeline 平衡逻辑
- `simulator/`：性能模拟器
- `cfgs/`：配置模板

---

## SAPP-ND：ND 搜索

ND 搜索模块包含内存估算和性能估算，支持多维并行策略搜索。

```python
from hyper_parallel.auto_parallel.sapp_nd import NDSearch

# ND 搜索自动估算内存和性能
search = NDSearch(
    model_config=model_config,
    device_info=device_info,
)

result = search.search(
    max_memory=memory_limit,
    target_perf=perf_target,
)
```

SAPP-ND 模块包含：

- `memory_estimation/`：内存估算
- `perf_estimation/`：性能估算
- `nd/`：ND 搜索算法

---

## 性能建议

1. **先 profiling**：使用 Fast-Tuner 前先对模型进行 profiling，获取准确的计算/通信时间
2. **PP 优先用 SAPP-PPB**：Pipeline stage 分配问题直接使用 SAPP-PPB
3. **结合手动调优**：AutoParallel 提供初始策略，可根据实际训练效果微调
4. **当前为 demo 特性**：建议在验证环境充分测试后再应用到生产训练