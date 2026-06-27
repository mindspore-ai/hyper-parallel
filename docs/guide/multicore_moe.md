# MoE 多核并行使用指南

HyperParallel 提供芯片内多核 MPMD 并行能力，结合核级内存语义单边通信，增强 MoE 通算掩盖和 MAC 利用率。

## 核心概念

多核并行是 HyperMPMD 的核心能力之一，从集群级 MPMD（Pipeline 并行）扩展到芯片内多核 MPMD：

- **O0**：通过框架层 host CPU 侧的调度，支持 cube、vector、单边通信算子分核执行
- **O1**：调度下沉到 AICore，支持 cube、vector、单边通信算子分核执行，进一步提升性能

当前已基于多核并行实现 MoE 通算掩盖（Multicore MoE-FFN）：将 MoE-FFN 的五个算子（AllToAll-Dispatch、GMM1、SwiGLU、GMM2、AllToAll-Combine）融合为一个 kernel，由 AIC（AI Cube）和 AIV（AI Vector）核同时执行，实现通信与计算的细粒度重叠。

## 接口概览

多核并行模块位于 `hyper_parallel/core/multicore/`，包含以下组件：

| 组件 | 说明 |
|------|------|
| `modules/` | 多核并行模块实现 |
| `ops/` | 多核并行算子 |
| `scheduler/` | 多核并行调度器 |
| `tasks/` | 任务编排 |
| `platform/` | 平台抽象 |
| `prebuild/` | 预构建 |

---

## 基础使用

### MoE 多核通算掩盖

基于多核并行优化 MoE FFN 的通算掩盖，将 dispatch（AllToAll）与 expert compute（GMM）在不同核上并发执行。调度配置（RuntimeConfig）按 rank 离线生成后，通过 `mc.mega_moe` / `mc.mega_moe_grad` 调用正反向算子：

```python
import hyper_parallel.core.multicore as mc

# 正向：融合 dispatch → up_proj → swiglu → down_proj → combine
mc.mega_moe(...)
# 反向
mc.mega_moe_grad(...)
```

完整的参数说明、RuntimeConfig 生成方式与编译步骤见下方详细文档。

---

## 详细说明

完整的 MoE-FFN 多核并行说明文档：

[MOE-FFN 说明](../../hyper_parallel/core/multicore/doc/README.md)

---

## 性能建议

1. **dispatch ↔ compute 掩盖**：MoE 的 AllToAll dispatch 与 expert compute 在不同核上并发，是最核心的掩盖收益
2. **单边通信**：基于内存语义的单边通信（Symmetric Memory）避免传统集合通信的同步开销
3. **RATR 通信重排**：通过 Rank-Aware Tile Reordering 将 AllToAll 流量在时间轴上均匀分散，避免多源 Rank 同时涌向同一目标，降低尾延迟
4. **O0 vs O1**：O0 通过 host CPU 调度，O1 调度下沉到 AICore，性能更高但实现难度更大
