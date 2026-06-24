# HyperParallel

昇腾超节点亲和的分布式并行加速库，简化超节点编程，释放算力潜能。

HyperParallel提供昇腾超节点亲和的分布式并行加速能力，在保障易用性的前提下，针对昇腾超节点资源池化、对等架构、网络拓扑分层多样、FP8低精格式等架构特点，实现了集群的分布式并行到芯片内多核并行，支持CPU
DRAM和NPU HBM的池化统一管理，支持拓扑感知调度和通信路径规划，支持FP8混合精度训练等昇腾超节点亲和的加速能力。

关键设计思路:

**模型和系统优化解耦**
：随着LLM和多模态算法新架构的不断演进，性能优化技术也随之向前持续创新，传统的算法和系统优化融合架构给算法迭代和系统长期维护带来了困难。通过HyperParallel，支持编程模型从系统优化内嵌到模型脚本演进到模型和系统优化解耦，隐式注入并行、重计算、offload等系统优化；支持并行范式从SPMD演进到MPMD，进一步支持集群MPMD和多核MPMD协同优化；支持存算关系从Stateful演进到Stateless计算状态分离。支持大语言模型、多模态大模型训练及强化学习等能力。

**全流程确定性**
：为了进一步保障训练稳定性和精度可复现性，HyperParallel支持了全流程的确定性，包括高性能确定性计算、通信、数据预处理、以及随机数的确定性，支持浮点bitwise对齐，所有支持的模型均会用确定性进行验证。虽然有部分的性能劣化，但出于对训练精度可复现性、SDC的快速发现、bug识别，仍然建议训练开启确定性。

**训推一体**：随着Reasoning RL和Agentic
RL任务越来越复杂，训推不一致问题导致强化学习训练难以收敛的问题愈发突出。HyperParallel会探索训推一体架构，通过一套加速框架同时实现训练和推理的性能优化，加强训推一致性，保障RL收敛。

**动静混合**
：基于静态图的优化是进一步提升性能的重要手段，比如基于静态图的通算并发、内存分析、执行序编排等能力可以有效优化性能，在动态图模式下并不容易实现。但动转静的编译支持难度非常大，目前还不能实现完全的动转静。HyperParallel会通过一些语法的约束，支持局部的动转静，使用MindSpore高阶图优化能力，进一步提升性能。

## 架构简介

<div align="center"> <img src="./docs/images/HyperParallel.jpg" width = 60% /> </div>

### HyperShard：编程模型演进，系统优化内嵌到模型 -> 模型和系统优化解耦

- SuperPoD Layout：Tensor切分、Device映射、通信路径统一建模，实现超节点单卡抽象;
- 声明式HSDP/TP/CP/EP：并行、重计算、offload等优化隐式注入到模型，实现模型代码和系统优化代码解耦，提高算法开发效率;

### HyperMPMD：并行范式演进，SPMD -> 集群MPMD -> 集群+多核MPMD

- 集群MPMD：支持异构模型切分，支持模型切片任意分配卡数；<br>
- 多模态MPMD (Mpipe)：支持多模态流水线并行异构调度，解锁超节点在 MLLM 下的利用率；<br>
- 芯片内多核MPMD：芯片内多核MPMD并行，结合核级内存语义单边通信，增强通算掩盖和MAC利用率；

### HyperOffload：算存关系演进，Stateful -> Stateless计算状态分离

- 远端和本地Tensor统一编程：支持tensor位置分配，隐藏远端数据传输，提升集群内存利用率；
- 远端Tensor预取和缓存，全模型Offload：DDP/HSDP+Offload替换DP/TP/PP/CP/SP/EP等复杂并行模式，简化系统设计，提升性能；

## 关键特性

- Models
    - [x] DeepSeekV3
    - [x] Qwen3.5-0.8B-Base
    - [x] Qwen3.5-35B-A3B-Base (MoE)
    - [x] Qwen3-VL-30B-A3B-Instruct (MoE)
    - [ ] DeepSeekV3.2
    - [ ] Qwen3-Omni

- HyperShard
    - DTensor
        - [x] DTensor basic
        - [x] DTensor redistribute
        - [x] manual_seed 分布式随机数种子控制
        - [ ] DTensor centric communication
        - [ ] Cross Mesh DTensor redistribution
    - HSDP / FSDP
        - [x] Parameter&Optimizer切分
        - [x] Parameter&Optimizer&Gradient切分
        - [x] Overlap（全 overlap 模式）
        - [x] gradient_scaling_factor 梯度缩放因子
        - [ ] 动转静
    - Shard / TP
        - [x] 分布式算子支持列表
        - [x] 自定义分布式算子注册（YAML registry + Python impl）
        - [x] Custom Shard
        - [x] DFunction 自定义分布式 autograd 函数
        - [x] parallelize_value_and_grad
        - [x] Loss Parallel（TP训练 loss 并行）
        - TP Styles
            - [x] ColwiseParallel / RowwiseParallel / SequenceParallel
            - [x] parallelize_module 声明式 TP 接口
            - [x] 1D
            - [ ] 高维TP，2D/2.5D/3D
        - EP
            - [x] ExpertParallel / ExpertTensorParallel 基础流程
            - [x] MoE构建模块（GroupedExperts / TokenChoiceTopKRouter / MoE）
            - [x] 负载均衡（expert_bias + aux_loss + AutoScaler）
            - [x] MoE zero-overhead activation storage
            - [x] MoE+EP token dispatch解耦
            - [ ] dropless基础流程
            - [ ] 通算Overlap
            - [ ] 专家热迁移 / 热点专家副本
        - CP
            - [x] ContextParallel 基础上下文并行
            - [x] AsyncContextParallel 异步上下文并行
            - [x] DSA 系列（Indexer / Loss / SparseAttention）
            - [x] TP DTensor local rewrap
            - [ ] DeepSpeed Ulysses
            - [ ] Ring Attention
            - [ ] 3D序列并行
        - [ ] Overlap
    - 分布式随机数
        - [x] manual_seed 随机数种子控制
        - [ ] DropOut

- HyperMPMD
    - Pipeline并行
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
        - [ ] 每个PP Stage分配不同卡数
    - Mpipe 多模态并行
        - [x] Mpipe Transpose 调度
        - [ ] Mpipe data-reordering
    - 子图切分
        - [ ] 多模态encoder/decoder切分到不同卡
    - 多核并行
        - [ ] 多核并行 - O0
        - [ ] 多核并行 - O1
        - [x] 基于多核并行优化MoE通算掩盖
        - [ ] 基于多核并行优化PP 1F1B通算掩盖

- HyperOffload
    - [x] Activation Checkpoint（checkpoint / checkpoint_wrapper / CheckpointPolicy）
    - [x] Activation Swap（swap / swap_wrapper / swap_tensor_wrapper / SwapManager）
    - [x] Activation Swap和Checkpointing协同配置
    - [x] Swap fusion
    - [ ] SAS（Selective Activation Swap）
    - [ ] SPO（Selective Parameter/Gradient/Optimizer Offload）
    - [ ] 基于内存语义的Offload
    - [ ] 自动Activation Swap策略生成

- Optimizer
    - [x] AdamW
    - [x] Muon（momentum-based optimizer）
    - [x] ChainedOptimizer（Muon+AdamW链式组合）
    - [x] get_hyper_optimizer / get_hyper_lr_scheduler
    - [x] 分片优化器（FSDP/HSDP集成）
    - [x] gradient scaling factor + clip_grad增强

- AutoParallel
    - [x] SAPP-ND：ND 搜索（内存估算 + 性能估算）
    - [x] SAPP-PPB：Pipeline Parallelism Balancing
    - [ ] SAPP-Omni

- 单边通信
    - [x] Symmetric Memory
    - [x] AllGather
    - [x] AllGatherMatmul / MatmulReduceScatter（MC2融合通信算子）
    - [ ] AllToAll
    - [ ] AllReduce
    - [ ] ReduceScatter
    - [ ] 低精通信高精累加

- 故障快速恢复
    - [x] DCP（Distributed Checkpoint）
        - [x] 分布式检查点保存/加载
        - [x] 异步staging保存
        - [x] 离线格式转换
        - [ ] 支持Huggingface格式
        - [ ] 支持不同切分策略倒换
    - [ ] 基础故障恢复流程
    - [ ] 进程级故障快速恢复
    - [ ] 临终遗言
    - [ ] SDC检测

- Trainer
    - [x] LLMTrainer 训练框架
    - [x] VLTrainer 多模态训练框架
    - [x] Callbacks（Logging / MoeMonitor）
    - [x] parallel_dims 并行维度配置

- Integration
    - [x] LlamaFactory集成（activation recompute & swap + HSDP）

- 工具
    - 精度监控
        - [ ] global norm
        - [ ] local norm
        - [ ] local loss

    - DryRun
        - [ ] 内存开销分析
        - [ ] 单卡模拟集群执行

## 安装教程

完整安装方式、源码构建参数和环境要求见 [安装指南](docs/installation.md)。

## 快速开始

1. 使用 `fully_shard` 进行数据并行参数切分

```python
from hyper_parallel import fully_shard, init_device_mesh

mesh = init_device_mesh(device_type="npu", mesh_shape=(dp_size,), mesh_dim_names=("dp",))
model = fully_shard(model, mesh=mesh)
```

2. 使用 `shard_module` 进行张量并行

```python
from mindspore.nn.utils import no_init_parameters
from hyper_parallel import DTensor, fully_shard, init_device_mesh, init_parameters, shard_module
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from hyper_parallel.core.shard.sharding_plan import ShardingPlan

# 定义设备网格和 placement
mesh = init_device_mesh(device_type="npu", mesh_shape=(dp_size, tp_size), mesh_dim_names=("dp", "tp"))
x_placement = (Shard(0), Shard(1))
w_placement = (Replicate(), Shard(0))
out_placement = (Shard(0), Replicate())

# 网络权重延后初始化
with no_init_parameters():
    model = SimpleModel()

# 对网络输入/输出/权重做切分配置
sharding_plan = ShardingPlan(
    input_plan={"input": x_placement},
    output_plan={"output": out_placement},
    plan={"weight": w_placement},
)
model = shard_module(model, device_mesh=mesh, sharding_plan=sharding_plan)

# 可以进一步配置fully_shard
model = fully_shard(model, mesh=mesh["dp"])

# 权重分片初始化
model = init_parameters(model)

# 执行
x = DTensor.from_local(local_x, mesh, x_placement)
run_model(x, model)
```

3. 使用声明式 TP Styles 进行张量并行

```python
from hyper_parallel import ColwiseParallel, RowwiseParallel, parallelize_module, init_device_mesh

tp_mesh = init_device_mesh("npu", (tp_size,), mesh_dim_names=("tp",))

parallelize_module(
    model,
    tp_mesh,
    {
        "attn.q_proj": ColwiseParallel(),
        "attn.k_proj": ColwiseParallel(),
        "attn.v_proj": ColwiseParallel(),
        "attn.o_proj": RowwiseParallel(),
        "mlp.gate_proj": ColwiseParallel(),
        "mlp.up_proj": ColwiseParallel(),
        "mlp.down_proj": RowwiseParallel(),
    },
)
```

4. 使用 `PipelineStage` 和 `PipelineSchedule` 进行流水线并行

```python
from hyper_parallel import DTensor
from hyper_parallel.core.pipeline_parallel import PipelineStage, Schedule1F1B

# 将切分后的module封装成PipelineStage
stage = PipelineStage(split_model, stage_index, stage_num=4)

# 选择流水线并行的调度
schedule = Schedule1F1B(stage, micro_batch_num=8)

# 执行
x = DTensor.from_local(local_x, input_mesh, input_placements)
schedule.run(x)
```

5. 使用 Optimizer 进行 Muon+AdamW 链式优化

```python
from hyper_parallel.core.optimizer import get_hyper_optimizer

optimizer = get_hyper_optimizer(
    model=model,
    muon_params=muon_param_groups,
    adamw_params=adamw_param_groups,
    muon_kwargs={"lr": 0.02, "momentum": 0.95},
    adamw_kwargs={"lr": 3e-4, "weight_decay": 0.1},
)
```

6. 使用 Activation Checkpoint/Swap 进行内存优化

```python
from hyper_parallel.core.activation_checkpoint import checkpoint_wrapper, swap_wrapper

model.layers[0] = checkpoint_wrapper(model.layers[0])
model.layers[1] = swap_wrapper(model.layers[1])
```

7. 使用基于多核并行的MoE通算掩盖算子

详见[MOE-FFN说明](./hyper_parallel/core/multicore/doc/README.md)。

## 文档

- [文档中心](./docs/index.md) — 文档索引与导航
- [安装指南](docs/installation.md) — 源码构建、依赖安装
- [特性使用指南](./docs/guide/) — 10 个核心特性使用指南
- [API参考](./docs/api/api_reference.md) — 按特性模块组织的接口说明
- [FAQ与故障排查](./docs/faq.md) — 常见问题与解决方案
- [社区贡献](./docs/contributing/) — 开发环境、测试规范、发布流程
- [Release Notes](./hyper_parallel_v1.0.0_release_notes.md) — 版本变更记录

## 参与贡献

1. Fork 本仓库
2. 新建 Feat_xxx 分支
3. 提交代码
4. 新建 Pull Request

如果您对HyperParallel有任何建议，请通过issue与我们联系，我们将及时处理。
如果对HyperParallel的技术感兴趣，或者想参与贡献代码，欢迎加入[Parallel Training System SIG](https://www.mindspore.cn/sig/Parallel%20Training%20System)。

## 许可证

[Apache 2.0许可证](LICENSE)
