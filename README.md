# HyperParallel

昇腾超节点亲和的分布式并行加速库，简化超节点编程，释放算力潜能。

HyperParallel 面向昇腾超节点提供分布式并行加速能力。针对资源池化、对等架构、分层网络拓扑和 FP8 低精度格式等特点，框架覆盖从集群分布式并行到芯片内多核并行，并支持 CPU DRAM 与 NPU HBM 池化管理、拓扑感知调度、通信路径规划和 FP8 混合精度训练。

核心设计：

**模型与系统优化解耦**：HyperParallel 将并行、重计算和 Offload 等系统优化从模型代码中分离，并通过声明式接口注入模型；并行范式覆盖集群 SPMD、集群 MPMD 和芯片内多核 MPMD，面向大语言模型和多模态模型训练。

**兼容 Hugging Face 生态**：HyperModels 直接读取 Hugging Face 模型配置和预训练权重，并复用 Transformers 中已有的模型实现；模型创建后，再由 HyperParallel 应用并行、重计算和算子替换等系统优化，无需将分布式逻辑写入模型代码。

**全流程确定性**：HyperParallel 支持计算、通信、数据预处理和随机数等环节的确定性，并对已支持模型进行 bitwise 对齐验证。确定性模式可能带来性能开销，但有助于复现训练结果、发现 SDC 和定位问题。

**动静混合**：静态图有利于通算并发、内存分析和执行序编排，但完整动转静对模型语法和编译能力要求较高。HyperParallel 在动态图基础上支持受约束的局部动转静，并使用 MindSpore 静态图优化能力提升执行效率。

## 最新消息

- [Aug. 25, 2026]: [Qwen/Qwen3-30B-A3B 模型支持](examples/training_demo/train.yaml)
- [Aug. 25, 2026]: [Online 数据动态组批训练支持](examples/training_demo/train_parallel_online.yaml)
- [Aug. 25, 2026]: [Indexed 数据训练支持](examples/training_demo/train_parallel_offline.yaml)

## 架构

<div align="center"> <img src="./docs/images/HyperParallel.jpg" width = 60% /> </div>

#### 使用入口

- [HyperModels](hyper_parallel/auto_models/README.md)：面向训练用户，通过 YAML 配置 Hugging Face 模型、数据、优化器、并行策略和保存与恢复方式。
- [HyperParallel Core](docs/api/api_reference.md)：面向模型作者和框架开发者，提供 DeviceMesh、DTensor、FSDP/HSDP、TP、CP、EP、PP 和自定义分布式算子。

#### HyperShard：从系统优化内嵌到模型与系统优化解耦

- SuperPoD Layout：统一建模 Tensor 切分、设备映射和通信路径，实现超节点单卡抽象。
- 声明式 HSDP / TP / CP / EP：将并行、重计算和 Offload 等优化注入模型，降低模型代码与系统优化的耦合。

#### HyperMPMD：从 SPMD 到集群与芯片内 MPMD

- 集群 MPMD：支持异构模型切分，并允许为不同模型切片分配不同数量的设备。
- 多模态 MPMD（Mpipe）：支持多模态流水线并行的异构调度，提高 MLLM 场景下的资源利用率。
- 芯片内多核 MPMD：结合核级内存语义和单边通信，增强通信与计算重叠及 MAC 利用率。

#### HyperOffload：从 Stateful 到 Stateless 的存算分离

- 远端与本地 Tensor 统一编程：支持 Tensor 位置配置并隐藏远端数据传输，提高集群内存利用率。
- Tensor 预取与缓存：结合 DDP/HSDP 和 Offload，简化 DP、TP、PP、CP、SP、EP 等并行策略的组合。

## 能力覆盖

- **模型与训练**：兼容 Hugging Face 模型配置和预训练权重，提供 TextTrainer、VLMTrainer 与 YAML 训练入口。
- **分布式 Tensor 与切分**：以 DeviceMesh、DTensor 和 Shard 表达设备拓扑、Tensor 布局与模型切分。
- **并行策略组合**：支持 FSDP/HSDP、TP、CP、EP、PP、SP 及其组合。
- **MoE 训练**：提供 Expert Parallel、Expert Tensor Parallel、路由、负载均衡与 token dispatch 能力。
- **流水线与异构执行**：支持 GPipe、1F1B、VPP、多模态 Mpipe、集群 MPMD 与芯片内多核 MPMD。
- **内存优化**：支持 Activation Checkpoint、Activation Swap、协同配置与 Swap fusion。
- **通信与自动并行**：支持 Symmetric Memory、MC2 融合通信、通信重叠、SAPP-ND 与 SAPP-PPB。
- **训练优化器**：支持 AdamW、Muon、链式优化器、分片优化器与梯度缩放。
- **保存与可靠性**：支持分布式检查点、异步 staging、离线格式转换和确定性训练。

当前支持能力与未来规划详见 [特性清单](docs/feature_status.md)。

## 安装

完整安装方式、源码构建参数和环境要求见 [安装指南](docs/installation.md)。

## 快速开始

### 使用 HyperModels 启动训练

HyperModels 使用一份 YAML 配置统一声明模型、数据、优化器、并行策略、训练参数和保存与恢复策略。仓库提供了可直接启动的基础示例 [`examples/training_demo/train_base.yaml`](examples/training_demo/train_base.yaml)：

```bash
NPROC=8 bash examples/training_demo/run_base.sh
```

附加命令行参数可按 YAML 字段路径覆盖配置值，无需修改配置文件：

```bash
NPROC=8 bash examples/training_demo/run_base.sh \
  --model.pretrained_model_name_or_path=/path/to/model \
  --training.train_iters=10
```

配置结构、字段解析和更多训练示例见 [HyperModels](hyper_parallel/auto_models/README.md)。

### 使用 HyperParallel Core API

HyperParallel Core 提供面向模型开发与系统扩展的底层并行接口。以下示例在数据并行 DeviceMesh 上应用 `fully_shard`，实现模型参数分片：

```python
from hyper_parallel import fully_shard, init_device_mesh

dp_mesh = init_device_mesh("npu", (dp_size,), mesh_dim_names=("dp",))
model = fully_shard(model, mesh=dp_mesh)
```

TP、CP、EP、PP 等组合方式见 [特性使用指南](docs/guide/)，公开接口见 [API 参考](docs/api/api_reference.md)。

## 项目结构

```text
HyperParallel/
├── hyper_parallel/
│   ├── auto_models/
│   │   ├── config/                     # YAML、TrainerConfig 和 CLI override 解析
│   │   ├── trainer/                    # TextTrainer、VLMTrainer 和训练生命周期
│   │   ├── _transformers/              # Transformers 模型构建与预训练权重加载
│   │   └── components/                 # 数据、优化器、并行计划、checkpoint 和 loss
│   ├── core/
│   │   ├── dtensor/                    # DeviceMesh、Layout、placement 和 DTensor
│   │   ├── shard/                      # sharding plan、自定义 shard 和 DFunction
│   │   ├── fully_shard/                # FSDP/HSDP 参数与执行调度
│   │   ├── tensor_parallel/            # TP styles 与 loss parallel
│   │   ├── context_parallel/           # Context Parallel
│   │   ├── expert_parallel/            # Expert Parallel
│   │   ├── pipeline_parallel/          # Pipeline stage 与调度
│   │   └── distributed_checkpoint/     # 分布式保存、加载与 reshard
│   ├── collectives/                    # 集合通信接口与实现
│   └── platform/                       # PyTorch、MindSpore 与设备后端适配
├── examples/
│   ├── training_demo/                  # HyperModels YAML Trainer 示例
│   ├── recipes/                        # 模型与并行拓扑配置
│   └── distributed/                    # HyperParallel Core API 组合示例
├── docs/
│   ├── guide/                          # 使用指南
│   └── api/                            # API 参考
└── tests/                              # HyperModels、Core 和后端测试
```

## 文档

- [文档中心](./docs/index.md) — 文档索引与导航
- [HyperModels](hyper_parallel/auto_models/README.md) — YAML Trainer、训练组件与专题指南入口
- [安装指南](docs/installation.md) — 源码构建、依赖安装
- [特性使用指南](./docs/guide/) — 并行与训练能力指南
- [API 参考](./docs/api/api_reference.md) — 按特性模块组织的接口说明
- [FAQ 与故障排查](./docs/faq.md) — 常见问题与解决方案
- [AI 辅助开发](./AGENTS.md) — AI 辅助开发说明
- [社区贡献](./docs/contributing/) — 开发环境、测试规范和发布流程
- [版本说明](./hyper_parallel_v1.0.0_release_notes.md) — 版本变更记录

## 加入我们

欢迎提交 Issue 和 Pull Request。开发环境、测试规范和提交流程见 [社区贡献指南](./docs/contributing/)。

### Parallel Training System SIG

如果对 HyperParallel 的技术方向感兴趣，欢迎加入 [Parallel Training System SIG](https://www.mindspore.cn/sig/Parallel%20Training%20System)。

扫描下方二维码加入 Parallel Training System SIG 微信交流群，与社区开发者交流使用经验、技术方案和项目进展。

<img src="./docs/images/parallel_training_system_sig_wechat.png" alt="Parallel Training System SIG 微信交流群二维码" width="220" />

## 许可证

[Apache 2.0许可证](LICENSE)
