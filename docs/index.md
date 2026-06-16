# HyperParallel 文档

昇腾超节点亲和的分布式并行加速库 — 文档中心

## 快速入门

| 文档 | 说明 |
|------|------|
| [安装指南](./getting_started/installation.md) | 源码构建、依赖安装、环境配置 |
| [快速开始](./getting_started/quick_start.md) | 最小可运行示例、设计理念、版本信息 |

## 特性使用指南

| 文档 | 说明 |
|------|------|
| [HSDP / FSDP 数据并行](./guide/fsdp.md) | fully_shard、HSDPModule、overlap、梯度缩放 |
| [TP 张量并行](./guide/tensor_parallel.md) | ColwiseParallel、RowwiseParallel、parallelize_module、Loss Parallel |
| [PP 流水线并行](./guide/pipeline_parallel.md) | PipelineStage、Schedule、overlap_b_f、PP+FSDP、P2P prefetch |
| [CP 上下文并行](./guide/context_parallel.md) | ContextParallel、AsyncContextParallel、DSA 系列 |
| [EP 专家并行](./guide/expert_parallel.md) | ExpertParallel、MoE 构建模块、负载均衡 |
| [Activation Checkpoint / Swap](./guide/activation_checkpoint.md) | checkpoint_wrapper、swap_wrapper、协同配置 |
| [Optimizer](./guide/optimizer.md) | AdamW、Muon、ChainedOptimizer、学习率调度器 |
| [DCP 分布式检查点](./guide/distributed_checkpoint.md) | 检查点保存/加载、异步 staging、离线转换 |
| [自动并行](./guide/auto_parallel.md) | Fast-Tuner、SAPP-PPB、SAPP-ND |
| [MoE 多核并行](./guide/multicore_moe.md) | 多核 MPMD、MoE 通算掩盖 |

## API 参考

| 文档 | 说明 |
|------|------|
| [API Reference](./api/api_reference.md) | 按特性模块组织的完整接口说明 |
| [DFunction](./api/dfunction.md) | 自定义分布式 autograd 函数详细文档 |

## FAQ 与故障排查

| 文档 | 说明 |
|------|------|
| [FAQ & 故障排查](./faq.md) | 安装问题、运行时问题、通信问题、内存问题、调试技巧 |

## 社区贡献

| 文档 | 说明 |
|------|------|
| [开发环境搭建](./contributing/dev_environment.md) | 开发环境配置、依赖安装 |
| [测试流程规范](./contributing/testing.md) | 测试框架、标记规范、分布式测试 |
| [发布流程规范](./contributing/release.md) | 版本发布流程、文档更新规范 |

## 其他

| 文档 | 说明 |
|------|------|
| [Release Notes v1.0.0](../hyper_parallel_v1.0.0_release_notes.md) | 版本变更记录 |
| [PP overlap_b_f 设计文档](./guide/pipeline_parallel_overlap_b_f.md) | 通算掩盖详细设计（原 docs 保留） |
| [TP Styles 文档](./guide/tensor_parallel_styles.md) | TP Styles 原有特性文档（原 docs 保留） |
| [agent_workflow.md](./agent_workflow.md) | AI Agent 工作流参考 |