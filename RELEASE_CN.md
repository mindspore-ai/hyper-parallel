# HyperParallel v1.0.0 Release Notes

HyperParallel 是面向昇腾超节点的分布式并行加速库，解耦模型代码与系统优化，提供从集群级 MPMD 到芯片内多核并行的统一分布式训练能力。v1.0.0 为首个正式发布版本，核心特性如下：

## 核心特性

- **DTensor** `[STABLE]`：基于分布式张量抽象，提供统一的 Stateless 编程模型，支持自动 layout 推导与跨设备 redistribute，实现本地/远程张量的透明化操作。

- **FSDP** `[STABLE]`：全切片数据并行，将参数、梯度和优化器状态在设备间分片，支持 reshard_after_forward、通信融合与 overlap 模式，大幅降低单卡显存占用。

- **Tensor Parallel（TP）** `[DEMO]`：声明式张量并行，提供 ColwiseParallel、RowwiseParallel、SequenceParallel 等策略，支持 Loss Parallel，兼容 PyTorch 分布式张量 API。

- **Context Parallel（CP）** `[STABLE]`：面向长序列训练（128K+）的序列维度切分，提供同步 CP、异步 CP 和 DSA（Dense Sparse Attention）三种变体，支持计算-通信异步 overlap。

- **Pipeline Parallel（PP）** `[STABLE]`：支持 GPipe、1F1B、VPP 等多种流水线调度策略，支持 PP+FSDP 融合、P2P 预取 overlap、Mpipe 多模态转置，并提供前反向 overlap 实现 EP all-to-all 通信掩盖。

- **Expert Parallel（EP）** `[DEMO]`：面向 MoE 模型的专家并行，支持 EP+TP 二维并行，提供 GroupedExperts、TokenChoiceTopKRouter 等构建模块，支持负载均衡与辅助损失。

- **重计算（Activation Checkpoint）** `[STABLE]`：通过 checkpoint_wrapper 实现选择性重计算，以计算换显存，支持策略化选择重计算层，与 swap 机制协同工作。

- **Swap（激活卸载）** `[STABLE]`：通过 swap_wrapper 将激活异步卸载至 CPU 并在反向时预取回 NPU，提供 SwapManager 实现层级别的 offload/prefetch 协调管理。

- **自动并行搜索（SAPP）** `[DEMO]`：SAPP-ND 提供 DP/TP/PP/EP 等多维并行策略自动搜索与内存估算；SAPP-PPB 实现流水线阶段负载均衡与重计算联合调优。

- **Mpipe 多模态并行** `[DEMO]`：提供Mpipe VLM多模态Transpose调度。

- **分布式 Checkpoint（DCP）** `[STABLE]`：按 rank 分片保存模型状态，支持不同并行策略间的 reshard 加载、异步暂存和离线格式转换，消除单卡内存瓶颈。

- **分布式优化器** `[STABLE]`：提供 AdamW 和 Muon 优化器，支持 ChainedOptimizer 混合参数组训练、梯度缩放与 FSDP 分片优化器状态。

- **芯片内多核并行** `[DEMO]`：提供 O0（Host CPU 调度）和 O1（AICore 调度）两级片上 MPMD 并行，结合多核分发与单边通信，提升 MoE 场景通信掩盖能力和 MAC 利用率。

- **DFunction** `[STABLE]`：自定义分布式 Autograd 函数接口，支持自动 DTensor dispatch、layout 推导和输出封装，方便用户扩展自定义分布式算子。

- **权重延后初始化** `[STABLE]`：支持模型权重的延后初始化，先在 meta device 上构建模型结构再按需物化参数，降低大模型初始化阶段的峰值内存占用。

## 平台支持

- 支持 PyTorch 2.6 / 2.7 / 2.9 及 MindSpore 后端。
- 支持 pip 安装与源码编译，可配置 multicore、symmetric memory、custom ops 等原生扩展。
