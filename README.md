# MindSpore HyperParallel
昇腾超节点亲和的分布式并行加速库，简化超节点编程，释放算力潜能。
#### 介绍
HyperParallel提供昇腾超节点亲和的MindSpore原生分布式并行加速能力，在保障易用性的前提下，针对昇腾超节点资源池化、对等架构、网路拓扑分层多样、FP8低精格式等架构特点，实现了集群的分布式并行到芯片内多核并行，支持CPU DRAM和NPU HBM的池化统一管理，支持拓扑感知调度和通信路径规划，支持FP8混合精度训练等昇腾超节点亲和的加速能力。  
通过HyperParallel，支持编程模型从系统优化内嵌到模型脚本演进到模型和系统优化解耦；支持并行范式从SPMD演进到MPMD，进一步支持集群MPMD和多核MPMD协同优化；支持存算关系从Stateful演进到Stateless计算状态分离。支持大语言模型、多模态大模型训练及强化学习等能力。  
   
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
from hyper_parallel.platform.mindspore.pipeline_parallel.stage import PipelineStage
from hyper_parallel.platform.mindspore.pipeline_parallel.schedule import Schedule1F1B

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

