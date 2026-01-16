# MindSpore HyperParallel
昇腾超节点亲和的分布式并行加速库，简化超节点编程，释放算力潜能。
#### 介绍
HyperParallel提供昇腾超节点亲和的MindSpore原生分布式并行加速能力，在保障易用性的前提下，针对昇腾超节点资源池化、对等架构、网路拓扑分层多样、FP8低精格式等架构特点，实现了集群的分布式并行到芯片内多核并行，支持CPU DRAM和NPU HBM的池化统一管理，支持拓扑感知调度和通信路径规划，支持FP8混合精度训练等昇腾超节点亲和的加速能力。  
关键设计思路：    
**模型和系统优化解耦**：随着LLM和多模态算法新架构的不断演进，性能优化技术也随之向前持续创新，传统的算法和系统优化融合架构给算法迭代和系统长期维护带来了困难。我们希望通过HyperParallel，支持编程模型从系统优化内嵌到模型脚本演进到模型和系统优化解耦，隐式注入并行、重计算、offload等系统优化；支持并行范式从SPMD演进到MPMD，进一步支持集群MPMD和多核MPMD协同优化；支持存算关系从Stateful演进到Stateless计算状态分离。支持大语言模型、多模态大模型训练及强化学习等能力。  
**全流程确定性**：为了进一步保障训练稳定性和精度可复现性，HyperPrallel支持了全流程的确定性，包括高性能确定性计算、通信、数据预处理、以及随机数的确定性，支持浮点bitwise对齐，所有支持的模型均会用确定性进行验证。虽然有部分的性能劣化，但出于对训练精度可复现性、SDC的快速发现、bug识别，我们仍然建议训练开启确定性。  
**训推一体**：随着Reasoning RL和Agentic RL任务越来越复杂，训推不一致问题导致强化学习训练难以收敛的问题愈发突出。HyperParallel会探索训推一体架构，通过一套加速框架同时实现训练和推理的性能优化，加强训推一致性，保障RL收敛。  
**动静混合**：基于静态图的优化是进一步提升性能的重要手段，比如基于静态图的通算并发、内存分析、执行序编排等能力可以有效优化性能，在动态图模式下并不容易实现。但动转静的编译支持难度非常大，目前还不能实现完全的动转静。HyperParallel会通过一些语法的约束，支持局部的动转静，使用MindSpore高阶图优化能力，进一步提升性能。
   
**本项目正在快速迭代，相关特性会持续开源，欢迎共建。**

#### 架构简介
<div align="center"> <img src="./docs/images/HyperParallel.jpg" width = 60% /> </div>

##### HyperShard：编程模型演进，系统优化内嵌到模型 -> 模型和系统优化解耦
- SuperPoD Layout：Tensor切分、Device映射、通信路径统一建模，实现超节点单卡抽象;<br>
- 声明式HSDP/TP/CP/EP：并行、重计算、offload等优化隐式注入到模型，实现模型代码和系统优化代码解耦，提高算法开发效率；<br>
##### HyperMPMD：并行范式演进，SPMD -> 集群MPMD -> 集群+多核MPMD
- 分布式MPMD：支持异构模型切分，支持模型切片任意分配卡数；<br>
- 多核MPMD：芯片内多核MPMD并行，结合核级内存语义单边通信，增强通算掩盖和MAC利用率；  
##### HyperOffload：算存关系演进，Stateful -> Stateless计算状态分离
- 远端和本地Tensor统一编程：支持tensor位置分配，隐藏远端数据传输，提升集群内存利用率；  
- 远端Tensor预取和缓存，全模型Offload：DDP/HSDP+Offload替换DP/TP/PP/CP/SP/EP等复杂并行模式，简化系统设计，提升性能；

#### 关键特性
- Models
  - [x] DeepSeekV3
  - [ ] DeepSeekV3.2
  - [ ] Qwen3
  - [ ] Qwen3-MoE
  - [ ] Qwen3-VL
  - [ ] Qwen3-Omini
  
- HyperShard
  - DTensor
    - [x] DTensor basic
    - [x] DTensor redistribute
    - [ ] DTensor centric communication
    - [ ] Cross Mesh DTensor redistribution
  - HSDP
    - [x] Parameter&Opitmizer切分
    - [x] Parameter&Optimizer&Gradient切分
    - [x] Overlap
    - [ ] 动转静
  - Shard
    - [x] 分布式算子支持列表（待补充）
    - [x] 自定义分布式算子注册
    - [x] Custom Shard，支持用户手写并行接入dtensor并行流程
    - TP
      - [x] 1D
      - [ ] 高纬TP，2D/2.5D/3D
    - EP
      - [ ] dropless基础流程
      - [ ] 通算Overlap
      - [ ] 专家热迁移，解决负载不均
      - [ ] 热点专家副本，解决负载不均
    - CP
      - [ ] DeepSpeed Ulysses
      - [ ] Ring Attention
      - [ ] 3D序列并行
    - [ ] Overlap
  - 分布式随机数
    - [ ] 随机数生成
    - [ ] DropOut

- HyperMPMD
  - PipeLine并行
    - [x] Gpie
    - [x] 1F1B
    - [x] VPP
    - [ ] ZBV
    - [ ] SeqPP
    - [ ] Overlap
    - [ ] 每个PP Stage分配不同卡数
  - 子图切分
    - [ ] 多模态encoder/decoder切分到不同卡
  - 多核并行
    - [ ] 多核并行 - O0：通过框架层host cpu侧的调度，支持cube、vector、单边通信算子分核执行；
    - [ ] 多核并行 - O1：调度下沉到AICore，支持cube、vector、单边通信算子分核执行，进一步提升性能；
    - [ ] 基于多核并行优化MoE通算掩盖
    - [ ] 基于多核并行优化PP 1B1F通算掩盖

- HyperOffload
  - [ ] SAS（Selective Activation Swap）：支持用户配置任意activation swap到host cpu dram；
  - [ ] SPO（Selective Parameter/Gradient/Optimizer Offload）：支持用户配置任意P/G/O offload到host cpu dram；
  - [ ] SAC（Selective Activation Checkpointing）：支持用户任意配置重计算范围；
  - [ ] Activation Swap和Chechpointing协同配置
  - [ ] 基于内存语义的Offload
  - [ ] 自动Activation Swap策略生成

- AutoParallel
  - [x] Fast-Tunner：基于profiling信息，构建黑盒代价模型，通过枚举、剪枝、搜索，自动生成多维混合并行策略，目前是demo特性，仍在持续优化;
  - [ ] PARADISE：基于Symbolic代价模型模拟内存和计算通信代价，高效生成最优多维混合并行策略，目前是demo特性，仍在持续优化;

- 单边通信
  - [ ] Symmetry memory
  - [ ] AllToAll
  - [ ] AllGather
  - [ ] AllReduce
  - [ ] ReduceScatter
  - [ ] 低精通信高精累加
  
- 故障快恢
  - [ ] DCP（Distributed Checkpoint）
    - [ ] 支持Huggingface格式
    - [ ] 支持不同切分策略倒换
  - [ ] 基础故障恢复流程
  - [ ] 进程级故障快恢
  - [ ] 临终遗言，故障触发ckpt保存，保存故障step的ckpt，实现0回退；
  - [ ] SDC检测

- 工具
  - 精度监控
    - [ ] global norm
    - [ ] local norm
    - [ ] local loss

  - DryRun
    - [ ] 内存开销分析
    - [ ] 单卡模拟集群执行

#### 安装教程

当前仅支持从源码安装，你需要执行：

```
git clone https://gitee.com/mindspore/hyper-parallel.git
cd hyper-parallel
pip install .
```
HyperParallel 依赖深度学习框架，在使用HyperParallel前，你需要：<br>
- 安装深度学习框架
- 推荐安装的MindSpore版本 >= 2.8，最好使用最新的MindSpore版本

#### 使用说明

1.  使用hsdp进行数据并行或zero切分优化
```
from hyper_parallel import hsdp

# 配置数据并行
model = hsdp(model, shard_size=1)

# 或者配置zero切分
model = hsdp(model, shard_size=dp_size, optimizer_level="level1")
```

2.  使用shard进行张量并行
```
from mindspore.nn.utils import no_init_parameters
from hyper_parallel import DTensor, Layout, hsdp, init_parameters, shard

# 定义张量排布
layout = Layout((dp, mp), ("dp", "mp"))
x_layout = layout("dp", "mp")
w_layout = layout("mp", "None")
out_layout = layout()

# 网络权重延后初始化
with no_init_parameters():
    model = SimpleModel()

# 对网络输入/输出/权重做切分配置
sharding_plan = { "forward": { "input": (x_layout,), "output": (out_layout,)},
                "parameter": {"weight": w_layout}}
model = shard(model, sharding_plan)

# 可以进一步配置hsdp
model = hsdp(model, shard_size=hsdp_shard_size, threshold=0)

# 权重分片初始化
model = init_parameters(model)

# 执行
x = DTensor.from_local(local_x, x_layout)
run_model(x, model)
```

3.  使用PipelineStage和PipelineSchedule进行流水线并行
```
from hyper_parallel import PipelineStage, Schedule1F1B

# 将切分后的module封装成PipelineStage
stage = PipelineStage(splited_model, stage_index, stage_num=4)

# 选择流水线并行的调度
schedule = Schedule1F1B（stage, micro_batch_num=8）

# 执行
x = DTensor.from_local(local_x, x_layout)
schedule.run(x)
```

#### 参与贡献

1.  Fork 本仓库
2.  新建 Feat_xxx 分支
3.  提交代码
4.  新建 Pull Request

