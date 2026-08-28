<h1 align="center">HyperParallel</h1>

<p align="center"><strong>昇腾超节点亲和的分布式并行加速库</strong></p>

---

## 简介

HyperParallel 面向昇腾超节点提供分布式并行加速能力。针对资源池化、对等架构、分层网络拓扑和 FP8 低精度格式等特点，框架覆盖从集群分布式并行到芯片内多核并行，并支持 CPU DRAM 与 NPU HBM 池化管理、拓扑感知调度、通信路径规划和 FP8 混合精度训练。

---

## 核心设计

- **模型与系统优化解耦**：HyperParallel 将并行、重计算和 Offload 等系统优化从模型代码中分离，并通过声明式接口注入模型；并行范式覆盖集群 SPMD、集群 MPMD 和芯片内多核 MPMD，面向大语言模型和多模态模型训练。

- **兼容 Hugging Face 生态**：AutoModels 直接读取 Hugging Face 模型配置和预训练权重，并复用 Transformers 中已有的模型实现；模型创建后，由 HyperParallel 以声明式方式将并行、重计算和算子替换等分布式逻辑注入模型。

- **全流程确定性**：HyperParallel 支持计算、通信、数据预处理和随机数等环节的确定性，并对已支持模型进行 bitwise 对齐验证。确定性模式可能带来性能开销，但有助于复现训练结果、发现 SDC 和定位问题。

- **动静混合**：静态图有利于通算并发、内存分析和执行序编排，但完整动转静对模型语法和编译能力要求较高。HyperParallel 在动态图基础上支持受约束的局部动转静。

---

## 📣 最新消息

- [Aug. 25, 2026]: [Qwen/Qwen3-30B-A3B 模型支持](examples/demo_trainer/train.yaml)

---

## 使用入口

### AutoModels

> [AutoModels](hyper_parallel/auto_models/README.md) 面向训练用户，通过 YAML 配置 Hugging Face 模型、数据、优化器、并行策略和保存与恢复方式。
>
> **支持模型**
>
> - [Qwen3-30B-A3B](examples/demo_trainer/train.yaml)

### HyperParallel Core

> [HyperParallel Core](docs/api/api_reference.md) 面向模型作者和框架开发者，提供从分布式模型构建到训练执行与状态管理的底层能力。
>
> **核心能力**
>
> - **模型切分与并行组合**：以 DeviceMesh、DTensor 和 Shard 描述设备拓扑与 Tensor 布局，并组合 FSDP/HSDP、TP、CP、EP、PP 等并行策略。
> - **MoE 与异构执行**：覆盖专家切分、路由和 token dispatch，以及流水线调度、集群 MPMD 和芯片内多核 MPMD。
> - **内存与性能优化**：提供 Activation Checkpoint、Activation Swap、通信与计算重叠、通信融合和自动并行能力。
> - **训练与状态管理**：提供优化器、分布式检查点、异步保存、离线格式转换和确定性训练能力。
>
> <details>
> <summary><strong>相关文档</strong></summary>
>
> - **模型切分**：[DTensor](docs/api/api_reference.md#dtensor-分布式张量) · [FSDP/HSDP](docs/guide/fsdp.md) · [TP](docs/guide/tensor_parallel.md)
> - **并行执行**：[CP](docs/guide/context_parallel.md) · [EP / MoE](docs/guide/expert_parallel.md) · [PP / 集群 MPMD](docs/guide/pipeline_parallel.md) · [多核 MPMD](docs/guide/multicore_moe.md)
> - **内存与性能**：[Activation Checkpoint / Swap](docs/guide/activation_checkpoint.md) · [通信与计算重叠](docs/guide/pipeline_parallel_overlap_b_f.md) · [通信融合](docs/guide/fsdp.md#通信融合) · [自动并行](docs/guide/auto_parallel.md)
> - **训练与状态**：[Optimizer](docs/guide/optimizer.md) · [DCP](docs/guide/distributed_checkpoint.md) · [确定性调试](docs/faq.md#3-确定性模式调试)
>
> </details>
>
> <br>
>
> <details>
> <summary><strong>未来计划</strong></summary>
>
> </details>

---

## 安装

完整安装方式、源码构建参数和环境要求见 [安装指南](docs/installation.md)。

---

## 🚀 快速开始

### 使用 AutoModels 启动训练

运行仓库提供的单机 8 卡训练示例 [`examples/demo_trainer/train.yaml`](examples/demo_trainer/train.yaml)：

```bash
torchrun --standalone --nproc_per_node=8 \
  scripts/train_lm.py examples/demo_trainer/train.yaml
```

附加命令行参数可按 YAML 字段路径覆盖配置值，无需修改配置文件：

```bash
torchrun --standalone --nproc_per_node=8 \
  scripts/train_lm.py examples/demo_trainer/train.yaml \
  --model.pretrained_model_name_or_path=/path/to/model \
  --training.train_iters=10
```

配置结构、字段解析和更多训练示例见 [AutoModels](hyper_parallel/auto_models/README.md)。

### 使用 HyperParallel Core API

以下示例在数据并行 DeviceMesh 上应用 `fully_shard`，实现模型参数分片：

```python
from hyper_parallel import fully_shard, init_device_mesh

dp_mesh = init_device_mesh("npu", (dp_size,), mesh_dim_names=("dp",))
model = fully_shard(model, mesh=dp_mesh)
```

TP、CP、EP、PP 等组合方式见 [特性使用指南](docs/guide/)，公开接口见 [API 参考](docs/api/api_reference.md)。

---

## 📖 文档

- [文档中心](./docs/index.md) — 文档索引与导航
- [AutoModels](hyper_parallel/auto_models/README.md) — YAML Trainer、训练组件与专题指南入口
- [安装指南](docs/installation.md) — 源码构建、依赖安装
- [特性使用指南](./docs/guide/) — 并行与训练能力指南
- [API 参考](./docs/api/api_reference.md) — 按特性模块组织的接口说明
- [FAQ 与故障排查](./docs/faq.md) — 常见问题与解决方案
- [AI 辅助开发](./AGENTS.md) — AI 辅助开发说明
- [社区贡献](./docs/contributing/) — 开发环境、测试规范和发布流程
- [版本说明](./hyper_parallel_v1.0.0_release_notes.md) — 版本变更记录

---

## 🗂️ 项目结构

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
│   ├── demo_trainer/                   # AutoModels 单机 8 卡训练示例
│   ├── recipes/                        # 模型与并行拓扑配置
│   └── distributed/                    # HyperParallel Core API 组合示例
├── docs/
│   ├── guide/                          # 使用指南
│   └── api/                            # API 参考
└── tests/                              # AutoModels、Core 和后端测试
```

---

## 🤝 加入我们

欢迎提交 Issue 和 Pull Request。开发环境、测试规范和提交流程见 [社区贡献指南](./docs/contributing/)。

### Parallel Training System SIG

如果对 HyperParallel 的技术方向感兴趣，欢迎加入 [Parallel Training System SIG](https://www.mindspore.cn/sig/Parallel%20Training%20System)。

扫描下方二维码加入 Parallel Training System SIG 微信交流群，与社区开发者交流使用经验、技术方案和项目进展。

<img src="./docs/images/parallel_training_system_sig_wechat.png" alt="Parallel Training System SIG 微信交流群二维码" width="220" />

---

## 📄 许可证

[Apache 2.0许可证](LICENSE)
