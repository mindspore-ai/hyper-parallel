# HyperParallel v1.0.0 Release Notes

**发布日期：2026-06-30**

HyperParallel v1.0.0 是项目的首个正式发布版本，标志着从快速迭代阶段进入稳定交付阶段。本版本提供了昇腾超节点亲和的分布式并行加速库的完整功能集，覆盖从集群级分布式并行到芯片内多核并行，从数据并行到多维混合并行，从训练到推理的全场景加速能力。

---

## 核心特性（全量）

### HyperShard：模型与系统优化解耦

**DTensor 分布式张量**

- DTensor 基础功能：分布式张量抽象，封装 local shard + DeviceMesh + Placements
- DTensor redistribute：按需重分布，支持跨 mesh 维度重排布，缓存优化（compact_str + rank_id）
- DeviceMesh：多维设备拓扑管理，支持 `init_device_mesh` 创建任意形状 mesh
- Layout：张量到 mesh 的排布映射，声明式定义并行策略
- `manual_seed`：分布式随机数种子控制（PyTorch parity）
- `init_parameters` / `init_empty_weights` / `init_on_device`：分片参数初始化工具

**HSDP/FSDP 数据并行**

- `fully_shard`：参数/梯度/优化器状态全切分，降低单卡内存占用
- `HSDPModule`：混合分片数据并行模块封装
- `hsdp_sync_stream`：梯度同步流管理
- HSDP Overlap：全 overlap 模式，通信与计算并发
- `set_gradient_scaling_factor`：梯度缩放因子配置
- 混合并行参数切分：TP+DP 场景下 FSDP 流正确处理 mixed parallel params
- MindSpore 参数 hook 迁移与修复

**Shard 分布式算子与张量并行**

- `shard_module`：声明式并行策略接口，通过 sharding_plan 配置输入/输出/参数排布
- `custom_shard`：自定义并行接入 DTensor 并行流程
- `DFunction`：自定义分布式 autograd 函数基类，集成 DTensor dispatch 系统
- `parallelize_value_and_grad`：并行化的值与梯度计算
- `SkipDTensorDispatch`：梯度 hook 中绕过 DTensor dispatch，直接操作 local tensor
- 分布式算子注册（YAML registry + Python impl）
- 分布式算子列表：AllGatherMatmul / MatmulReduceScatter (MC2)、RotaryPositionEmbedding 等
- MindSpore replicate allreduce overlap

**Tensor Parallel（声明式 TP Styles）**

- `ColwiseParallel`：列切分并行
- `RowwiseParallel`：行切分并行
- `SequenceParallel`：序列并行
- `PrepareModuleInput` / `PrepareModuleOutput` / `PrepareModuleInputOutput`：模块输入/输出准备钩子
- `parallelize_module`：声明式 TP 应用接口，支持 fnmatch glob pattern
- Loss Parallel：TP 训练场景的 loss 并行支持

**Expert Parallel（专家并行）**

- `ExpertParallel`：标准 all-to-all EP，每个 rank 持有 num_experts / ep_degree 个本地专家
- `TensorParallel`（EP 内）：TP-only 权重切分
- `ExpertTensorParallel`：EP+TP 二维并行（2-D [ep, tp] mesh）
- MoE 构建模块：`FeedForward`、`GroupedExperts`、`TokenChoiceTopKRouter`、`MoE`
- 负载均衡：`expert_bias` + `tokens_per_expert` + auxiliary load-balance loss (`MoEAuxLossAutoScaler`)
- MoE+EP token dispatch 解耦：token permutation 和 dispatch 从 forward 中解耦
- PP+EP 多 micro-batch 状态安全：`DispatchContext` 防止实例共享问题

**Context Parallel（上下文并行）**

- `ContextParallel`：基础上下文并行
- `AsyncContextParallel`：异步上下文并行
- DSA 系列：
  - `DSAIndexerContextParallel` / `AsyncDSAIndexerContextParallel`
  - `DSAIndexerLossContextParallel` / `AsyncDSAIndexerLossContextParallel`
  - `DSASparseAttentionContextParallel` / `AsyncDSASparseAttentionContextParallel`
- TP DTensor local rewrap 支持

---

### HyperMPMD：集群 MPMD + 多核 MPMD

**Pipeline Parallel（流水线并行）**

- `PipelineStage`：流水线 stage 封装，支持 dx/dw 计算
- `Schedule1F1B`：1F1B 调度
- `ScheduleInterleaved1F1B`：交错 1F1B 调度（VPP）
- `ScheduleGPipe`：GPipe 调度
- `MetaStep` / `MetaStepType` / `BatchDimSpec`：调度单元抽象
- PP+FSDP 支持：`MetaStep` 集成 FSDP metastep，PP stage 内参数切分
- PP 通算掩盖（overlap_b_f）：
  - B/F 复合 step（`OVERLAP_B_F`）
  - `CommComputeOverlap` 双线程协调器
  - `HookCoordinator` COMM-first rendezvous
  - EP A2A 与 compute 并发
  - PP P2P prefetch：`overlap_p2p=True`，batched P2P transport（duplex default under overlap_b_f）
  - forward-side P2P prefetch via `batch_isend_irecv`
- PP Activation Swap：pipeline parallel 场景下的 activation swap
- Variable-layer + mixed-recompute under overlap_b_f
- SAPP-PPB：Pipeline Parallelism Balancing 自动平衡模块
- Batch size 校验：MindSpore 后端拒绝 batch size 不整除 micro_batch_num

**MoE 多核并行**

- 多核 MPMD 并行：芯片内多核并行 + 核级内存语义单边通信
- 基于多核并行优化 MoE 通算掩盖：将 AllToAll-Dispatch / GMM1 / SwiGLU / GMM2 / AllToAll-Combine 融合为单 kernel，AIC 与 AIV 核细粒度重叠
- RATR（Rank-Aware Tile Reordering）：AllToAll Tile 重排，离线生成、运行时零开销

---

### HyperOffload：计算状态分离

**Activation Checkpoint / Swap**

- `checkpoint` / `checkpoint_wrapper`：选择性激活重计算（SAC）
- `swap` / `swap_wrapper` / `swap_tensor_wrapper`：激活 swap（offload to CPU, prefetch on backward）
- `CheckpointPolicy`：重计算策略配置
- `SwapManager`：Swap 管理器
- Activation Swap 和 Checkpointing 协同配置
- Swap fusion：swap 操作融合优化
- 共享函数排除：callable overlap tracking 排除 shared functions
- LlamaFactory 集成：activation recompute & swap 支持
- MindSpore recompute independent schedule 支持

---

### Mpipe 多模态并行

- Mpipe VLM 多模态 Transpose 调度

### Optimizer（优化器）

- `AdamW`：标准 AdamW 优化器
- `Muon`：Muon 优化器（momentum-based optimizer）
- `ChainedOptimizer`：链式优化器（Muon+AdamW 组合）
- `get_hyper_optimizer`：优化器工厂函数
- `get_hyper_lr_scheduler`：学习率调度器工厂函数
- 分片优化器：与 FSDP/HSDP 集成的分片优化器状态管理
- gradient scaling factor 支持
- clip_grad 增强：对齐 clip_grad_norm_ reduction 与各 grad 的 process group

---

### DCP 分布式检查点

- 分布式检查点保存/加载（planner + storage + reshard）
- `async_staging`：异步 staging 保存
- `offline_transform`：离线格式转换
- Huggingface 格式支持
- 不同切分策略倒换

---

### AutoParallel（自动并行）

- **SAPP-ND**：ND 搜索模块，包含内存估算和性能估算
- **SAPP-PPB**：Pipeline Parallelism Balancing，自动平衡 pipeline stage 分配

---

### 单边通信

- `Symmetric Memory`：对称内存语义单边通信
- `AllGather`：单边 AllGather 操作
- `AllGatherMatmul` / `MatmulReduceScatter`（MC2）：融合通信算子

---

### Process Group（进程组管理）

- `init_process_group` / `destroy_process_group`
- `get_process_group_ranks` / `get_backend`
- `split_group` / `get_group_local_rank` / `mark_created_groups`

---

### Models（模型支持）

- DeepSeekV3
- Qwen3 系列：
  - Qwen3.5-0.8B-Base
  - Qwen3.5-35B-A3B-Base
  - Qwen3-VL-30B-A3B-Instruct

---

### Integration（集成）

- LlamaFactory 集成：activation recompute & swap + HSDP 支持
- MindSpore llama3 TP+FSDP demo
- Torch llama3 PP+FSDP+CP demo
- PP+EP 1F1B 正确性验证示例
- MoE Expert Parallel 正确性 demo

---

### Trainer（训练框架）

- `LLMTrainer`：LLM 训练框架
- `VLTrainer`：多模态视觉语言训练框架
- Callbacks：LoggingCallback、MoeMonitorCallback
- `parallel_dims`：并行维度配置

---

### Platform Abstraction（平台抽象）

- 双后端支持：PyTorch（GPU/NPU）+ MindSpore（Ascend NPU）
- `get_platform()` 统一抽象接口
- Platform-specific 实现：
  - FSDP/HSDP：Torch + MindSpore 双后端
  - Pipeline Parallel：Torch + MindSpore 双后端
  - Activation Checkpoint/Swap：Torch + MindSpore 双后端
  - Process Group：Gloo + HCCL
- Wheel 打包：Python/arch tag + CXX11 ABI 统一

---

## v0.2.0 → v1.0.0 重要变更（增量）

### 新增特性

- **Optimizer 模块**：Muon+AdamW 链式优化器、学习率调度器、分片优化器
- **Activation Checkpoint 重构**：`checkpoint_wrapper` / `swap_wrapper` / `swap_tensor_wrapper` 新接口
- **PP 增强**：FSDP metastep 集成、batched P2P transport、overlap_b_f 通算掩盖、activation swap for PP
- **Context Parallel 增强**：async CP、DSA 系列（Indexer/Loss/SparseAttention）、TP DTensor local rewrap
- **新增 Qwen3 系列模型支持**：Qwen3.5/Qwen3.5-MoE/Qwen3-VL-MoE
- **MoE+EP 增强**：token dispatch 解耦、load balance aux_loss + expert_bias sync、zero-overhead activation storage
- **Distributed Ops**：AllGatherMatmul / MatmulReduceScatter (MC2)、RotaryPositionEmbedding、Detach、StopGradient
- **HSDP Overlap**：全 overlap 模式
- **Loss Parallel**：TP 训练 loss 并行
- **gradient_scaling_factor**：梯度缩放因子配置
- **Mpipe**：多模态流水线并行异构调度
- **SAPP-PPB / SAPP-ND**：自动 pipeline balance + ND 搜索
- **LlamaFactory 集成**：activation 优化 + HSDP
- **Trainer 框架**：LLMTrainer + VLTrainer

### 重要修复

- **DTensor**：in-place random ops 返回 self、strided shard full_tensor concat 修复、parameter casts 保持 DTensor identity
- **FSDP/HSDP**：tp>1/dp=1 场景 FSDP meta 上下文 device mesh 构造修复、non-continuous all-reduce 修复、mixed parallel params 切分修复、param hook 迁移修复
- **Shard**：replicate params 状态跟踪修复、DTensor identity 保持 for in-place ops on dispatch bypass、MindSpore replicate allreduce overlap
- **Activation Checkpoint**：shared functions 排除 callable overlap tracking、recompute packing sharded param views 修复
- **Pipeline**：PipelineStage 设备默认值修复、batch size 整除校验
- **Context Parallel**：async CP preserve TP layout 修复
- **Optimizer**：clip_grad_norm_ reduction 对齐各 grad process group
- **Packaging**：wheel python/arch tag + CXX11 ABI 统一
- **Codecheck**：66 文件补充 docstring（C0116 warnings）

### 测试覆盖提升

- Activation Checkpoint UT：17.9% → 95% coverage
- Shard UT/ST：覆盖提升 + 结构重构 + Gloo CPU backend 迁移（206 cases）
- Context Parallel UT：CP 单元覆盖
- PP+FSDP+TP 复合精度测试 vs 单卡基线
- Process Group worker path 修复 + world-group teardown 安全
- Trainer / trainer/utils / core/utils UT + ST 补充

---

## 贡献者

感谢以下贡献者对 v1.0.0 版本的贡献（按 commit 数排序）：

| 贡献者 | Commit 数 |
|--------|----------|
| xuxinglei | 26 |
| yide12 | 20 |
| Meng107 | 14 |
| lichen666 | 12 |
| DavidFFFan | 12 |
| david-he91 | 12 |
| ch-l | 10 |
| changzherui1 / changzherui | 7 |
| 宋佳琪 | 5 |
| silkage_jiajia | 5 |
| yao_yf | 4 |
| wangleiit03 | 4 |
| hedongdong | 4 |
| zhuangqinghe | 2 |
| zhangyuguo | 2 |
| yuzhenhua666 | 2 |
| yanxr123 | 2 |
| wang_hua_2025 | 2 |
| rongyue | 2 |
| liuchongming74 / liuchongming | 2 |
| Liang_Ziyi | 2 |
| lei_xxx | 2 |
| jinxiaoxian | 2 |
| guangpengz | 2 |
| Cui-yshoho / celiaccui | 2 |
| zhangbuxue | 1 |
| yuzhenhua | 1 |
| SHUYUAN6 | 1 |
| sargerasking | 1 |
| qhzhuang11111111 | 1 |
| pengshuyuan | 1 |
| niujunhao | 1 |
| liuyanwei / liu-yanwei6 | 1 |
| huilan_li / huilan li | 1 |
| duchengyi | 1 |
| buxue | 1 |
| bj-wang1 / bj-wang | 1 |
| alpha-junh | 1 |

**合计：187 commits，1040 files changed，103628 insertions，23160 deletions**

---

## 已合并 PR

| PR | 描述 |
|----|------|
| !460 | feat: add activation recompute & swap support for LlamaFactory integration |
| !524 | feat: SAPP-PPB for Pipeline Parallelism Balancing |
| !622 | [feature][pipeline_parallel] pp support fsdp |
| !627 | [feature][fully_shard] support set_gradient_scaling_factor |
| !631 | merge feat-sapp-nd-pr1 into master |
| !635 | full overlap for hsdp |
| !636 | merge test-gloo into master |
| !637 | swap_fusion |
| !639 | fix(shard): skip unsharded replicate params |
| !640 | merge feat/moe-load-balance into master |
| !647 | feat(moe): zero-overhead activation storage via internal score weighting |
| !649 | [Add Test]: 为 trainer / trainer/utils / core/utils 补充 UT + ST |
| !656 | feat: add distributed ops AllGatherMatmul and MatmulReduceScatter (MC2) |
| !657 | bugfix: add CMake pre-build support for mindspore custom_ops |
| !659 | feat(dtensor): add DTensor manual_seed (PyTorch parity) |
| !660 | feat(context-parallel): add DSA context parallel style |
| !661 | feat: add distributed op RotaryPositionEmbedding with UT and MindSpore ST |
| !663 | feat(context_parallel): add async colossal cp coverage |
| !664 | refactor ms pipeline_stage |
| !665 | test: improve UT coverage for core/shard and custom_ops/experimental |
| !667 | 多核并行代码结构整改 |
| !668 | merge bugfix_strided_shard_master into master |
| !669 | feat(gate-doctor): autonomous PR gate doctor skill |
| !670 | feat(examples): MindSpore llama3 TP+FSDP demo and local embedding |
| !671 | feat: add pipeline parallel comm/compute overlap (B/F + lazy a2a) |
| !673 | disable_dispatch |
| !675 | merge master into master |
| !677 | test: add MindSpore PP+EP+overlap PoC test |
| !678 | merge dmodule_protocols_part1 into master |
| !680 | refactor(test): overhaul shard ST test structure and file naming |
| !682 | refactor(moe+ep): decouple token permutation and dispatch from forward |
| !683 | fix: avoid recompute packing sharded param views |
| !684 | [task][fully_shard] improve fully_shard ut cov rate |
| !685 | feat: register Detach and StopGradient distributed ops in element_wise_ops.yaml |
| !687 | feat(moe): 新增负载均衡辅助损失、expert_bias 分布式同步及 sequence_partition_group |
| !689 | fix(dtensor): keep parameter casts as dtensor |
| !690 | multicore megamoe rename w1_grad and w2_grad; add test cases |
| !691 | migrate distributed op |
| !693 | fix: package sapp nd memory estimation |
| !694 | perf(pipeline): use platform.empty for recv buffer allocation |
| !695 | refactor: add dist-op-analysis & dist-op-dev skills; simplify dev workflow docs |
| !697 | refactor: preserve tensor identity in activation swap for keep_graph |
| !699 | merge feat-sapp-nd-pr2 into master |
| !700 | merge pp-swap into master |
| !703 | fix(dtensor): size dim_map by tensor rank in _calc_shard_info |
| !704 | fix: add loss parallel support for tensor parallel training |
| !705 | merge pp-swap into master |
| !706 | feat(examples): add Llama3 PP demos and unify training steps |
| !707 | [fix][fully_shard] adjust prefetch, unshard position |
| !710 | [feature][pipeline_parallel] pipelinestage support dx,dw compute |
| !713 | test(context-parallel): add cp unit coverage |
| !720 | feat(examples): add MoE Expert Parallel correctness demo |
| !723 | fix(shard): overlap MindSpore replicate allreduce |
| !724 | test(torch): add PP+FSDP+TP composite accuracy tests vs single-card baseline |
| !725 | bug_fix |
| !726 | feat: support tp dtensor local rewrap for context parallel |
| !729 | feat(pipeline): overlap_bf_recompute_fixes |
| !730 | merge fix-grad-norm into master |
| !734 | test: add comprehensive UT for core/activation_checkpoint (17.9% → 95% coverage) |
| !737 | feat(example): add PP(1F1B)+FSDP2+CP demo for Llama3 on Torch backend |
| !738 | chore: make agent tooling assistant-agnostic for any AI coding assistant |
| !740 | feat(moe): add PP+EP 1F1B correctness verification example |
| !741 | ms_support_recompute_independent_schedule |
| !744 | fix fsdp hook clear |
| !749 | merge fix/ms-fsdp-param-hook-migration into master |
| !752 | fix(init_weights): preserve parameter attributes when moving to target device |
| !754 | fix(activation_checkpoint): exclude shared functions from callable overlap tracking |
| !755 | merge sapp_ut_coverage1 into master |
| !756 | fix(dtensor): return self from MindSpore in-place random ops |
| !757 | [shard] Preserve DTensor identity for in-place ops on the dispatch bypass |
| !759 | 【bugfix】修复 tp>1/dp=1 时 FSDP 在 meta 上下文构造 device mesh 及 compat all-reduce 非连续报错 |
| !766 | test: fix process_group worker path and avoid double world-group teardown |
| !767 | fix dcp level st |
| !768 | feat(pipeline_parallel): forward-side P2P prefetch for overlap_b_f via batch_isend_irecv |
| !770 | fix dcp level st |
| !771 | fix(fsdp2): use assign=True when loading full state dict to fix optimizer device check on torch >= 2.8 |
| !775 | fix(packaging): tag wheel per python/arch and unify cxx11 abi per backend |
| !777 | fix(codecheck): add missing docstrings for remaining C0116 warnings |
| !779 | fix(dtensor): refine MindSpore random op inplace dispatch |
| !782 | merge fix/ms-param-shape-error into master |
| !783 | muon, Hp |
| !786 | fix(pipeline): default PipelineStage device to current accelerator |
| !787 | fix(pipeline): reject batch size not divisible by micro_batch_num on MindSpore |
| !788 | test: reduce NPU level0 ST gate load with gloo level0 coverage |
| !790 | [feature][pipeline_parallel] torch backend pp support FSDP metastep |
| !795 | merge fix_named_parameters into master |
| !803 | fix: remove duplicate empty decorator in sparse_grad_kl_loss test |
| !806 | [bugfix] fix reduce type |
| !808 | Add MHC pre clamp sinkhorn custom op |

---

## 已知限制

- **MindSpore CP**：DSA 系列 async CP 仅 PyTorch 后端支持，MindSpore 为 noop 占位
- **DTensor**：centric communication、Cross Mesh redistribution 待实现
- **EP**：dropless 基础流程、通算 overlap、专家热迁移/热点专家副本待实现
- **PP**：ZBV、SeqPP 调度，每个 PP Stage 分配不同卡数待实现
- **HyperOffload**：SAS/SPO/SAC 用户配置接口、内存语义 Offload、自动策略生成待实现
- **AutoParallel**：Fast-Tuner 和 PARADISE 当前为 demo 特性，持续优化中
- **单边通信**：AllToAll、AllReduce、ReduceScatter、低精通信高精累加待实现

---

## 升级指南

从 v0.2.0 升级到 v1.0.0：

1. **Activation Checkpoint API 变更**：新增 `checkpoint_wrapper` / `swap_wrapper` / `swap_tensor_wrapper` 接口，推荐使用新接口替代旧 API
2. **Optimizer 模块**：新增 `Muon` / `ChainedOptimizer` / `get_hyper_optimizer` / `get_hyper_lr_scheduler`，可直接集成到训练流程
3. **PP API 扩展**：`PipelineStage` 新增 dx/dw 计算、FSDP metastep 支持；`ScheduleInterleaved1F1B` 新增 `overlap_b_f` / `overlap_p2p` 参数
4. **CP API 扩展**：新增 `AsyncContextParallel` 及 DSA 系列接口
5. **Wheel 打包变更**：whl 文件名包含 python/arch tag + CXX11 ABI 标识，需匹配目标环境

---

## 反馈与贡献

- Issue 反馈：https://gitcode.com/mindspore/hyper-parallel/issues
- 代码贡献：欢迎加入 [Parallel Training System SIG](https://www.mindspore.cn/sig/Parallel%20Training%20System)
- 许可证：Apache 2.0
