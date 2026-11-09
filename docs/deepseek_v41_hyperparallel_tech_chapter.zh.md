# 三、HyperParallel 的关键实现：把模型结构接入分布式训练

[English](deepseek_v41_hyperparallel_tech_chapter.md) | **简体中文**

面向 DeepSeek-V4.1，HyperParallel 的适配工作不仅是复现模型结构，还要把跨层状态、稀疏参数和多模态计算纳入可组合的分布式训练流程。模型适配器将参数分片、TP/CP/EP、重计算、外部状态、视觉编码器并行和 loss 等能力接入统一 Trainer，模型特有逻辑则通过适配器与配置完成组合。

## 1. CSA2：管理跨层共享状态与并行通信

CSA2 的 Full、Reindex、Reuse 路径会在不同层之间共享压缩 KV、Indexer Key 和 Top-K 等状态。HyperParallel 按注意力角色管理这些状态：上下文并行时，各 rank 保留本地 Query，通过异步可微 AllGather 获取计算所需的 KV 与 Indexer Key；张量并行时，先汇总各分片 Indexer Head 的分数，再完成全局 Top-K 选择。

激活重计算也遵循相同的状态边界。重放时复用前向已经选出的离散索引，并避免再次写入共享状态，使 CSA2 的重计算与跨层状态生命周期保持一致。

## 2. Engram：CPU 常驻、EP 分片和稀疏优化

Engram 表按专家并行 rank 分配行区间，并可将 FP32 参数保存在 CPU。前向按需查表，反向只记录访问行的梯度；训练步结束前再合并重复行、同步稀疏梯度，并交由 SparseAdam 更新。Checkpoint 通过独立 sidecar 保存和恢复各 rank 的表数据及稀疏优化器状态。

这条路径减少了超大条件记忆表对设备常驻内存的占用。本文的裁剪模型实验用于验证训练链路，并不代表已加载完整规模的 Engram 参数。

## 3. mHC 与 MoE：接入昇腾融合执行路径

HyperParallel 将 DeepSeek-V4.1 的 mHC 模块映射到 HyperMegaMhc，在层间边界执行融合计算；同时提供 MegaGate 路由替换和 routed/shared expert 并行计算路径。模型参数映射与训练框架的并行策略由适配层连接，避免把模型改造成一套独立的训练入口。

## 4. 复用融合算子并完成模型级集成

DeepSeek-V4.1 的昇腾执行路径也复用了 [torchtitan-npu](https://github.com/torchtitan-npu/torchtitan-npu) 仓中的部分融合算子实现。HyperParallel 的独特工作在于将这些算子接入自身的 functional 组件和模型适配层：处理模型张量布局与序列长度元数据，衔接算子前后的路由、稀疏索引和输出重排，并使其参与对应的并行计算与自动微分路径。这样模型可以复用已有的 NPU 高性能内核，同时仍通过 HyperParallel 的适配器和 Trainer 组织训练。

仓库中可见的接入路径包括 DSA 的 Lightning Indexer 与稀疏 Flash Attention、参数 sink attention，以及 MoE 的 grouped matmul、SwiGLU 和 token 排列/还原。相关封装负责适配 TND 等算子布局、保留注意力反向所需的 softmax 统计量、连接 grouped GEMM 的权重梯度与梯度累积，并将专家计算放入 EP 路径。文章应将这项贡献表述为“融合算子复用 + HyperParallel 的模型和分布式训练集成”；底层算子来源与框架侧适配分别说明，不把复用的内核称为 HyperParallel 原创算子。

## 5. 多模态与 loss：按计算结构拆分工作

多模态路径为 DeepSeek-V4.1 视觉编码器提供并行适配，并保留图像元数据到视觉工作单元的映射。训练 loss 则支持按序列块计算交叉熵，避免在整个序列上一次性物化完整词表 logits，从而控制大词表训练的中间张量开销。

需要区分代码支持范围与本文实测范围：文中的吞吐和显存结果仍来自随机初始化的裁剪模型与短程训练；未合入主线的 PR 改动不计入这些结果。
