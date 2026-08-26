# 特性清单

本文记录 HyperParallel 当前已支持与规划中的模型和系统能力。`[x]` 表示当前版本已支持，`[ ]` 表示规划中。

## 支持模型

- [x] Qwen3-30B-A3B

## HyperShard

### DTensor

- [x] DTensor basic
- [x] DTensor redistribute
- [x] manual_seed 分布式随机数种子控制
- [ ] DTensor centric communication
- [ ] Cross Mesh DTensor redistribution

### HSDP / FSDP

- [x] Parameter 与 Optimizer 切分
- [x] Parameter、Optimizer 与 Gradient 切分
- [x] Overlap（全 overlap 模式）
- [x] gradient_scaling_factor 梯度缩放因子
- [ ] 动转静

### Shard / TP

- [x] 分布式算子支持列表
- [x] 自定义分布式算子注册（YAML registry + Python impl）
- [x] Custom Shard
- [x] DFunction 自定义分布式 autograd 函数
- [x] parallelize_value_and_grad
- [x] Loss Parallel（TP 训练损失并行）

#### TP Styles

- [x] ColwiseParallel / RowwiseParallel / SequenceParallel
- [x] parallelize_module 声明式 TP 接口
- [x] 1D
- [ ] 高维TP，2D/2.5D/3D

#### EP

- [x] ExpertParallel / ExpertTensorParallel 基础流程
- [x] MoE 构建模块（GroupedExperts / TokenChoiceTopKRouter / MoE）
- [x] 负载均衡（expert_bias + aux_loss + AutoScaler）
- [x] MoE zero-overhead activation storage
- [x] MoE + EP token dispatch 解耦
- [ ] Dropless 基础流程
- [ ] 通信与计算重叠
- [ ] 专家热迁移 / 热点专家副本

#### CP

- [x] ContextParallel 基础上下文并行
- [x] AsyncContextParallel 异步上下文并行
- [x] DSA 系列（Indexer / Loss / SparseAttention）
- [x] TP DTensor local rewrap
- [ ] DeepSpeed Ulysses
- [ ] Ring Attention
- [ ] 3D 序列并行

- [ ] Overlap

### 分布式随机数

- [x] manual_seed 随机数种子控制
- [ ] Dropout

## HyperMPMD

### Pipeline 并行

- [x] GPipe
- [x] 1F1B
- [x] VPP（ScheduleInterleaved1F1B）
- [x] PP+FSDP（MetaStep集成）
- [x] PipelineStage dx/dw计算
- [x] 通算掩盖 overlap_b_f（CommComputeOverlap双线程协调器）
- [x] batched P2P transport / overlap_p2p
- [x] PP Activation Swap
- [x] variable-layer + mixed-recompute under overlap_b_f
- [ ] ZBV
- [ ] SeqPP
- [ ] 每个 PP Stage 分配不同卡数

### Mpipe 多模态并行

- [x] Mpipe Transpose 调度
- [ ] Mpipe data-reordering

### 子图切分

- [ ] 多模态 encoder / decoder 切分到不同设备

### 多核并行

- [ ] 多核并行 - O0
- [ ] 多核并行 - O1
- [x] 基于多核并行优化 MoE 通算掩盖
- [ ] 基于多核并行优化 PP 1F1B 通算掩盖

## HyperOffload

- [x] Activation Checkpoint（checkpoint / checkpoint_wrapper / CheckpointPolicy）
- [x] Activation Swap（swap / swap_wrapper / swap_tensor_wrapper / SwapManager）
- [x] Activation Swap 与 Checkpointing 协同配置
- [x] Swap fusion
- [ ] SAS（Selective Activation Swap）
- [ ] SPO（Selective Parameter/Gradient/Optimizer Offload）
- [ ] 基于内存语义的Offload
- [ ] 自动Activation Swap策略生成

## Optimizer

- [x] AdamW
- [x] Muon（momentum-based optimizer）
- [x] ChainedOptimizer（Muon + AdamW 链式组合）
- [x] get_hyper_optimizer / get_hyper_lr_scheduler
- [x] 分片优化器（FSDP / HSDP 集成）
- [x] gradient scaling factor + clip_grad 增强

## AutoParallel

- [x] SAPP-ND：ND 搜索（内存估算 + 性能估算）
- [x] SAPP-PPB：Pipeline Parallelism Balancing
- [ ] SAPP-Omni

## 单边通信

- [x] Symmetric Memory
- [x] AllGather
- [x] AllGatherMatmul / MatmulReduceScatter（MC2 融合通信算子）
- [ ] AllToAll
- [ ] AllReduce
- [ ] ReduceScatter
- [ ] 低精通信高精累加

## 故障快速恢复

- [x] DCP（Distributed Checkpoint）
  - [x] 分布式检查点保存/加载
  - [x] 异步 staging 保存
  - [x] 离线格式转换
  - [ ] 支持 Hugging Face 格式
  - [ ] 支持不同切分策略倒换
- [ ] 基础故障恢复流程
- [ ] 进程级故障快速恢复
- [ ] 临终遗言
- [ ] SDC检测

## Trainer

- [x] AutoModels TextTrainer 文本训练
- [x] AutoModels VLMTrainer 多模态训练
- [x] Callbacks（Logging / MoeMonitor）
- [x] AutoModels YAML Trainer 配置与并行维度配置

## 生态集成

- [x] LlamaFactory 集成（activation recompute & swap + HSDP）

## 工具

### 精度监控

- [ ] global norm
- [ ] local norm
- [ ] local loss

### DryRun

- [ ] 内存开销分析
- [ ] 单卡模拟集群执行

[返回项目 README](../README.md)
