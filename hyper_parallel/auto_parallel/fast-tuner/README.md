# 大模型训练并行策略自动搜索工具(DEMO FEATURE)

## 1 介绍

### 1.1 总体架构

（1）工具的作用简介：

大规模LLM的训练有多种并行范式，在不同的并行配置和优化特性的混搭下性能差别很大。本工具用于大模型训练并行配置搜索，根据给定的模型信息和硬件信息，提供建议的并行训练配置、优化特性的启用与否以及流水线负载均衡的配置。

本工具具有以下优势：

- 使用容易，无需调优经验，一键提升模型开箱性能。
- 快速调优，小时级匹配专家配置性能。
- 泛化支持，支持常用热门模型。

（2）使用工具的总体流程

用户提供模型网络脚本和一定的测试资源，经ND配置搜索剪枝算法后会生成若干需要做profile的配置文件，做完profile之后（多机场景需要人工介入，单机场景工具支持自动化）工具将会自动生成各个配置的推荐流水线和重计算配置以及相应的Step Time估值，估值越小的配置性能越优。

### 1.2 关键特性

已支持模型：

- [x] deepseekv3
- [x] llama2/3
- [x] qwen3-30B-MOE

已支持特性：

- [x] Data Parallel (DP)
- [x] Tensor Parallel (TP)
- [x] Pipeline Parallel (PP)
- [x] Expert Parallel (EP)
- [x] Sequence Parallel (SP)
- [x] Optimizer Parallel (OP)
- [x] Fully Sharded Data Parallel (FSDP)
- [x] Context Parallel (CP)
- [x] Activation Checkpointing (重计算)
- [x] Virtual Pipeline (VPP)

### 1.3 即将推出的功能

计划支持特性

- [ ] Per-batch Size (PBS)
- [ ] Swap Optimizer (swap)
- [ ] 多机场景一键式运行

## 2 安装

支持通过源代码安装工具，安装步骤如下：

```bash
git clone https://gitee.com/mindspore/hyper-parallel.git
cd hyper-parallel/hyper_parallel/auto_parallel/fast-tuner
pip install -e .
```

## 3 使用指南

对于大语言模型训练任务，本工具支持当前开源社区的三个应用广泛的训练框架：Mindspore Transformers， Mindspeed LLM， TorchTitan。无论基于哪个训练框架，用户都可以通过配置参数，执行命令两步轻松完成并行策略自动搜索。

[基于Mindformers](./docs/mindformers.md) 使用文档\
[基于Torchtitan](./docs/torchtitan.md) 使用文档 \
[基于Mindspeed](./docs/mindspeed.md) 使用文档
