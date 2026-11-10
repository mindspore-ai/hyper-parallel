<!-- markdownlint-disable MD033 MD041 -->
<!-- The branded README header and QR images use HTML for layout and sizing. -->

<div align="center">
  <img src="docs/images/hyperparallel-logo.png" alt="HyperParallel Logo" title="HyperParallel logo" width="50%">

[![WeChat 微信交流群](https://img.shields.io/badge/WeChat-64A519?style=flat&logo=wechat&logoColor=07C160)](#parallel-training-system-sig)
[![小红书 HyperParallel（小红书号：hyperparallel）](https://img.shields.io/badge/%E5%B0%8F%E7%BA%A2%E4%B9%A6-FF2442?style=flat&logo=xiaohongshu&logoColor=white)](#xiaohongshu-qrcode)
[![Document 文档中心](https://img.shields.io/badge/Document-0088CC?style=flat&logo=readthedocs&logoColor=white)](docs/index.zh.md)
[![Papers](https://img.shields.io/badge/Papers-red?style=flat&logo=arxiv&logoColor=white)](#technical-reports)
  <h3>简化超节点编程 &nbsp;·&nbsp; 释放并行算力</h3>

  <p><a href="README.md">English</a> | <strong>简体中文</strong></p>

  <p>
    <a href="#-快速开始">快速开始</a>
    &emsp;·&emsp;
    <a href="#supported-models">支持模型</a>
    &emsp;·&emsp;
    <a href="docs/index.zh.md">文档中心</a>
    &emsp;·&emsp;
    <a href="docs/api/api_reference.zh.md">API 参考</a>
    &emsp;·&emsp;
    <a href="#technical-reports">技术报告</a>
  </p>
</div>

---

## 简介

HyperParallel 面向昇腾超节点提供分布式并行加速能力。针对资源池化、对等架构、分层网络拓扑和 FP8 低精度格式等特点，框架覆盖从集群分布式并行到芯片内多核并行，并支持 CPU DRAM 与 NPU HBM 池化管理、拓扑感知调度、通信路径规划和 FP8 混合精度训练。

---

## 核心设计

- **模型与系统优化解耦**：HyperParallel 将并行、重计算和 Offload 等系统优化从模型代码中分离，并通过声明式接口注入模型；并行范式覆盖集群 SPMD、集群 MPMD 和芯片内多核 MPMD，面向大语言模型和多模态模型训练。

- **兼容 Hugging Face 生态**：HyperParallel 支持直接读取 Hugging Face 模型配置和预训练权重，并复用 Transformers 中已有的模型实现；模型创建后，由 HyperParallel 以声明式方式将并行、重计算和算子替换等分布式逻辑注入模型。

- **全流程确定性**：HyperParallel 支持计算、通信、数据预处理和随机数等环节的确定性，并对已支持模型进行 bitwise 对齐验证。确定性模式可能带来性能开销，但有助于复现训练结果、发现 SDC 和定位问题。

- **动静混合**：静态图有利于通算并发、内存分析和执行序编排，但完整动转静对模型语法和编译能力要求较高。HyperParallel 在动态图基础上支持受约束的局部动转静。

---

## 📣 最新消息

- [Oct. 15, 2026]: [DeepSeek-V4.1-Flash 模型支持](docs/deepseek_v41_article/README.zh.md)
- [Oct, 9, 2026]: [moonshotai/Kimi-K2.6 模型支持](hyper_parallel/models/kimi_k26/README.zh.md)
- [Aug. 25, 2026]: [Qwen/Qwen3-30B-A3B 模型支持](examples/training_demo/README.md#full-pretrained-model)

---

<a id="supported-models"></a>
<a id="支持模型"></a>

## 🤖 支持模型

详细能力与在研进展见[模型路线图](https://github.com/mindspore-ai/hyper-parallel/issues/1305)。

| 模型 | 示例配置 / 文档 |
|---|---|
| Qwen3 | [模型入口](hyper_parallel/models/qwen3) |
| Qwen3-MoE | [recipes](hyper_parallel/models/qwen3_moe/recipes) |
| Qwen3-VL-MoE | [模型适配](hyper_parallel/models/qwen3_vl_moe) |
| Qwen3.5 Dense | [模型入口](hyper_parallel/models/qwen3_5) |
| DeepSeek-V3 | [recipes](hyper_parallel/models/deepseek_v3/recipes) |
| DeepSeek-V3.2 | [recipes](hyper_parallel/models/deepseek_v32/recipes) |
| DeepSeek-V4.1 | [文本配置](examples/training_demo/deepseek_v41/train_deepseek_v41_online.yaml) · [多模态配置](examples/training_demo/deepseek_v41/train_deepseek_v41_vlm_online.yaml) |
| Kimi-K2.6 | [recipes](hyper_parallel/models/kimi_k26/recipes) |
| openPangu-2.0 | [代码仓](https://gitcode.com/ascend-tribe/openPangu-2.0-Training) |

---

<a id="安装"></a>

## 🛠️ 安装

完整安装方式、源码构建参数和环境要求见 [安装指南](docs/installation.zh.md)。

---

## 🚀 快速开始

### 使用 HyperParallel 启动模型训练

运行仓库提供的单机 8 卡训练示例 [`hyper_parallel/models/qwen3_moe/recipes/train.yaml`](hyper_parallel/models/qwen3_moe/recipes/train.yaml)：

```bash
torchrun --standalone --nproc_per_node=8 \
  scripts/train_lm.py hyper_parallel/models/qwen3_moe/recipes/train.yaml
```

附加命令行参数可按 YAML 字段路径覆盖配置值，无需修改配置文件：

```bash
torchrun --standalone --nproc_per_node=8 \
  scripts/train_lm.py hyper_parallel/models/qwen3_moe/recipes/train.yaml \
  --model.pretrained_model_name_or_path=/path/to/model \
  --training.train_iters=10
```

字段定义与解析规则见 [训练配置与组件](hyper_parallel/models/README.zh.md)。

### 使用 HyperParallel Core API

以下示例在数据并行 DeviceMesh 上应用 `fully_shard`，实现模型参数分片：

```python
from hyper_parallel import fully_shard, init_device_mesh

dp_mesh = init_device_mesh("npu", (dp_size,), mesh_dim_names=("dp",))
model = fully_shard(model, mesh=dp_mesh)
```

TP、CP、EP、PP 等组合方式见 [特性使用指南](docs/guide)，公开接口见 [API 参考](docs/api/api_reference.zh.md)。

---

## 📖 文档

- [文档中心](./docs/index.zh.md) — 文档索引与导航
- [训练配置与组件](hyper_parallel/models/README.zh.md) — TrainerConfig 字段、模块结构与组件文档入口
- [安装指南](docs/installation.zh.md) — 源码构建、依赖安装
- [特性使用指南](./docs/guide) — 并行与训练能力指南
- [API 参考](./docs/api/api_reference.zh.md) — 按特性模块组织的接口说明
- [FAQ 与故障排查](./docs/faq.zh.md) — 常见问题与解决方案
- [AI 辅助开发](./AGENTS.md) — AI 辅助开发说明
- [社区贡献](CONTRIBUTING.zh.md) — 贡献流程、测试、评审和版本发布
- [版本说明](./hyper_parallel_v1.0.0_release_notes.zh.md) — 版本变更记录

---

## 🤝 加入我们

欢迎提交 Issue 和 Pull Request。开发环境、测试、评审和社区发布流程见[社区贡献指南](CONTRIBUTING.zh.md)。

### Parallel Training System SIG

如果对 HyperParallel 的技术方向感兴趣，欢迎加入 [Parallel Training System SIG](https://www.mindspore.cn/sig/Parallel%20Training%20System)。

扫描下方二维码加入 Parallel Training System SIG 微信交流群，与社区开发者交流使用经验、技术方案和项目进展。

<img src="./docs/images/parallel_training_system_sig_wechat.png" alt="Parallel Training System SIG 微信交流群二维码" width="220" />

### HyperParallel 技术博客

扫描下方二维码关注 HyperParallel 小红书账号（小红书号：`hyperparallel`），获取技术博客与实践分享。

<a id="xiaohongshu-qrcode"></a>
<img src="./docs/images/hyperparallel_xiaohongshu.png" alt="HyperParallel 小红书技术博客二维码" width="220" />

---

<a id="technical-reports"></a>

## 📄 技术报告

- HyperParallel-FSDP: Topology-Aware Fully Sharded Training with Layout-Driven Muon on Ascend SuperPods [[paper](https://arxiv.org/abs/2609.21594)]
- HyperParallel-Mpipe: A Composable Algebra System for Optimizing MLLM Training over Supernode Clusters [[paper](https://arxiv.org/abs/2607.03229)]
- HyperParallel-MoE: Multi-Core Interleaved Scheduling for Fast MoE Training on Ascend NPUs [[paper](https://arxiv.org/abs/2605.23764)]
- HyperParallel: A Supernode-Affinity AI Framework [[paper](https://arxiv.org/abs/2603.03731)]
- HyperOffload: Graph-Driven Hierarchical Memory Management for Large Language Models on SuperNode Architectures [[paper](https://arxiv.org/abs/2602.00748)]

---

## 致谢

感谢以下合作伙伴与项目的出色工作和支持：

- [鹏城实验室（Pengcheng Laboratory）](https://www.pcl.ac.cn/)
- [LlamaFactory](https://github.com/hiyouga/LlamaFactory)

---

## ⚖️ 许可证

[Apache 2.0许可证](LICENSE)
