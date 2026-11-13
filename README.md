<!-- markdownlint-disable MD033 MD041 -->
<!-- The branded README header and QR images use HTML for layout and sizing. -->

<div align="center">
  <img src="docs/images/hyperparallel-logo.png" alt="HyperParallel Logo" title="HyperParallel logo" width="50%">

[![WeChat community](https://img.shields.io/badge/WeChat-64A519?style=flat&logo=wechat&logoColor=07C160)](#parallel-training-system-sig)
[![HyperParallel on REDnote](https://img.shields.io/badge/REDnote-FF2442?style=flat&logo=xiaohongshu&logoColor=white)](#xiaohongshu-qrcode)
[![Documentation](https://img.shields.io/badge/Document-0088CC?style=flat&logo=readthedocs&logoColor=white)](docs/index.md)
[![Papers](https://img.shields.io/badge/Papers-red?style=flat&logo=arxiv&logoColor=white)](#technical-reports)
  <h3>Simplify SuperPoD Programming &nbsp;·&nbsp; Unleash Parallel Computing Power</h3>

  <p><strong>English</strong> | <a href="README.zh.md">简体中文</a></p>

  <p>
    <a href="#-quick-start">Quick Start</a>
    &emsp;·&emsp;
    <a href="#supported-models">Supported Models</a>
    &emsp;·&emsp;
    <a href="docs/index.md">Documentation</a>
    &emsp;·&emsp;
    <a href="docs/api/api_reference.md">API Reference</a>
    &emsp;·&emsp;
    <a href="#technical-reports">Technical Reports</a>
  </p>
</div>

---

## Introduction

HyperParallel provides distributed parallel acceleration for Ascend SuperPoDs. Designed for resource pooling, peer-to-peer architectures, hierarchical network topologies, and FP8 low-precision formats, it spans cluster-level distributed parallelism and on-chip multicore parallelism. It supports pooled management of CPU DRAM and NPU HBM, topology-aware scheduling, communication path planning, and FP8 mixed-precision training.

---

## Core Design

- **Decoupling model code from system optimization**: HyperParallel separates parallelism, recomputation, and offload from model code and injects them through declarative interfaces. Its parallel paradigms cover cluster SPMD, cluster MPMD, and on-chip multicore MPMD for large language and multimodal model training.

- **Hugging Face ecosystem compatibility**: HyperParallel reads Hugging Face model configurations and pretrained weights directly and reuses existing Transformers model implementations. After model creation, HyperParallel declaratively injects distributed logic such as parallelism, recomputation, and operator replacement.

- **End-to-end determinism**: HyperParallel supports determinism across computation, communication, data preprocessing, and random number generation, with bitwise alignment validation for supported models. Deterministic mode may incur a performance cost, but helps reproduce training results, detect silent data corruption (SDC), and diagnose issues.

- **Hybrid dynamic and static execution**: Static graphs facilitate compute-communication overlap, memory analysis, and execution scheduling, but full dynamic-to-static conversion places substantial demands on model syntax and compiler capabilities. HyperParallel supports constrained, local dynamic-to-static conversion on top of eager execution.

---

## 📣 Latest News

- [Nov. 12, 2026]: [Qwen/Qwen3-Omni-30B-A3B model support](hyper_parallel/models/qwen3_omni/recipes/train.yaml)
- [Oct. 15, 2026]: [DeepSeek-V4.1-Flash model support](docs/deepseek_v41_article/README.md)
- [Oct. 9, 2026]: [moonshotai/Kimi-K2.6 model support](hyper_parallel/models/kimi_k26/recipes/train.yaml)
- [Aug. 25, 2026]: [Qwen/Qwen3-30B-A3B model support](hyper_parallel/models/qwen3_moe/recipes/train.yaml)

---

## Where to Start

### HyperParallel Model Training

> [HyperParallel Models](hyper_parallel/models/README.md) lets training users configure Hugging Face models, data, optimizers, parallel strategies, and checkpoint saving and restoration through YAML.
>
> See [Supported Models](#supported-models) for available models and configuration entry points.

### HyperParallel Core

> [HyperParallel Core](docs/api/api_reference.md) provides model authors and framework developers with foundational capabilities, from distributed model construction to training execution and state management.
>
> **Core capabilities**
>
> - **Model sharding and parallel composition**: Describe device topology and tensor layouts with DeviceMesh, DTensor, and Shard, and combine FSDP/HSDP, TP, CP, EP, and PP strategies.
> - **MoE and heterogeneous execution**: Expert sharding, routing, and token dispatch, along with pipeline scheduling, cluster MPMD, and in-chip multicore MPMD.
> - **Memory and performance optimization**: Activation checkpointing, activation swapping, compute-communication overlap, communication fusion, and automatic parallelism.
> - **Training and state management**: Optimizers, distributed checkpoints, asynchronous saving, offline format conversion, and deterministic training.
>
> <details>
> <summary><strong>Related documentation</strong></summary>
>
> - **Model sharding**: [DTensor](docs/api/api_reference.md#dtensor-distributed-tensors) · [FSDP/HSDP](docs/guide/fsdp.md) · [TP](docs/guide/tensor_parallel.md)
> - **Parallel execution**: [CP](docs/guide/context_parallel.md) · [EP / MoE](docs/guide/expert_parallel.md) · [PP / Cluster MPMD](docs/guide/pipeline_parallel.md) · [Multicore MPMD](hyper_parallel/core/multicore/README.md)
> - **Memory and performance**: [Activation Checkpoint / Swap](docs/guide/activation_checkpoint.md) · [Compute-communication overlap](docs/guide/pipeline_parallel_overlap_b_f.md) · [Communication fusion](docs/guide/fsdp.md#communication-fusion) · [Automatic parallelism](docs/guide/auto_parallel.md)
> - **Training and state**: [Optimizer](docs/guide/optimizer.md) · [DCP](docs/guide/distributed_checkpoint.md) · [Deterministic debugging](docs/faq.md#3-debug-in-deterministic-mode)
>
> </details>
>
> <br>
>
> <details>
> <summary><strong>Future Plans</strong></summary>
>
> </details>
>>>>>>> theirs

---

<a id="supported-models"></a>

## 🤖 Supported Models

See the [model roadmap](https://github.com/mindspore-ai/hyper-parallel/issues/1305) for detailed capabilities and ongoing work.

| Model | Example configuration / documentation |
|---|---|
| Qwen3 | [Model entry](hyper_parallel/models/qwen3) |
| Qwen3-MoE | [Recipes](hyper_parallel/models/qwen3_moe/recipes) |
| Qwen3-VL-MoE | [Model adapter](hyper_parallel/models/qwen3_vl_moe) |
| Qwen3-Omni | [Model adapter](hyper_parallel/models/qwen3_omni) |
| Qwen3.5 Dense | [Model entry](hyper_parallel/models/qwen3_5) |
| DeepSeek-V3 | [Recipes](hyper_parallel/models/deepseek_v3/recipes) |
| DeepSeek-V3.2 | [Recipes](hyper_parallel/models/deepseek_v32/recipes) |
| DeepSeek-V4.1 | [Text configuration](examples/training_demo/deepseek_v41/train_deepseek_v41_online.yaml) · [Multimodal configuration](examples/training_demo/deepseek_v41/train_deepseek_v41_vlm_online.yaml) |
| Kimi-K2.6 | [Recipes](hyper_parallel/models/kimi_k26/recipes) |
| openPangu-2.0 | [Repository](https://gitcode.com/ascend-tribe/openPangu-2.0-Training) |

---

<a id="installation"></a>

## 🛠️ Installation

See the [Installation Guide](docs/installation.md) for installation methods, source build options, and environment requirements.

---

## 🚀 Quick Start

### Launch Model Training with HyperParallel

Run the single-node, eight-device training example provided in [`hyper_parallel/models/qwen3_moe/recipes/train.yaml`](hyper_parallel/models/qwen3_moe/recipes/train.yaml):

```bash
torchrun --standalone --nproc_per_node=8 \
  scripts/train_lm.py hyper_parallel/models/qwen3_moe/recipes/train.yaml
```

Additional command-line arguments override configuration values by YAML field path, without modifying the configuration file:

```bash
torchrun --standalone --nproc_per_node=8 \
  scripts/train_lm.py hyper_parallel/models/qwen3_moe/recipes/train.yaml \
  --model.pretrained_model_name_or_path=/path/to/model \
  --training.train_iters=10
```

See [Training Configuration and Components](hyper_parallel/models/README.md) for field definitions and parsing rules.

### Use the HyperParallel Core API

The following example applies `fully_shard` on a data-parallel DeviceMesh to shard model parameters:

```python
from hyper_parallel import fully_shard, init_device_mesh

dp_mesh = init_device_mesh("npu", (dp_size,), mesh_dim_names=("dp",))
model = fully_shard(model, mesh=dp_mesh)
```

See the [Feature Guides](docs/index.md#feature-guides) for TP, CP, EP, and PP combinations, and the [API Reference](docs/api/api_reference.md) for public interfaces.

---

## 📖 Documentation

- [Documentation Center](./docs/index.md) — Documentation index and navigation
- [Training Configuration and Components](hyper_parallel/models/README.md) — TrainerConfig fields, module structure, and component documentation
- [Installation Guide](docs/installation.md) — Source builds and dependencies
- [Feature Guides](./docs/index.md#feature-guides) — Parallelism and training capabilities
- [API Reference](./docs/api/api_reference.md) — Interfaces organized by feature
- [FAQ and Troubleshooting](./docs/faq.md) — Common issues and solutions
- [AI-Assisted Development](./AGENTS.md) — Instructions for AI-assisted development
- [Contributing](CONTRIBUTING.md) — Contribution workflow, testing, review, and releases
- [Release Notes](./hyper_parallel_v1.0.0_release_notes.md) — Version history

---

## 🤝 Join Us

Issues and pull requests are welcome. Start with the [Contribution Guide](CONTRIBUTING.md) for development setup, testing, review, and the community release workflow.

### Parallel Training System SIG

If you are interested in HyperParallel's technical direction, join the [Parallel Training System SIG](https://www.mindspore.cn/sig/Parallel%20Training%20System).

Scan the QR code below to join the Parallel Training System SIG WeChat group and discuss usage, technical approaches, and project progress with community developers.

<img src="./docs/images/parallel_training_system_sig_wechat.png" alt="Parallel Training System SIG WeChat group QR code" width="220" />

### HyperParallel Technical Blog

Scan the QR code below to follow HyperParallel on REDnote (account: `hyperparallel`) for technical articles and practical experience.

<a id="xiaohongshu-qrcode"></a>
<img src="./docs/images/hyperparallel_xiaohongshu.png" alt="HyperParallel REDnote technical blog QR code" width="220" />

---

<a id="technical-reports"></a>

## 📄 Technical Reports

- HyperParallel-FSDP: Topology-Aware Fully Sharded Training with Layout-Driven Muon on Ascend SuperPods [[paper](https://arxiv.org/abs/2609.21594)]
- HyperParallel-Mpipe: A Composable Algebra System for Optimizing MLLM Training over Supernode Clusters [[paper](https://arxiv.org/abs/2607.03229)]
- HyperParallel-MoE: Multi-Core Interleaved Scheduling for Fast MoE Training on Ascend NPUs [[paper](https://arxiv.org/abs/2605.23764)]
- HyperParallel: A Supernode-Affinity AI Framework [[paper](https://arxiv.org/abs/2603.03731)]
- HyperOffload: Graph-Driven Hierarchical Memory Management for Large Language Models on SuperNode Architectures [[paper](https://arxiv.org/abs/2602.00748)]

---

## Acknowledgments

We thank the following partners and projects for their work and support:

- [Pengcheng Laboratory](https://www.pcl.ac.cn/)
- [LlamaFactory](https://github.com/hiyouga/LlamaFactory)

---

## ⚖️ License

[Apache 2.0 License](LICENSE)
