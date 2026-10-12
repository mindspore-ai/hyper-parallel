# HyperParallel 模型训练

HyperParallel 1.1.0 支持加载 Hugging Face 模型配置和预训练权重，也支持通过自定义构建函数接入自有模型。Trainer 负责数据处理、前向与反向计算、优化器更新以及检查点保存与恢复；并行和内存优化由 HyperParallel Core 提供。

本分支提供 `TextTrainer` 和 `VLMTrainer`，现有训练配方面向文本模型。多模态任务需配置对应的 processor、样本转换、collator 和模型输入接口，接入方式见 [二次开发指南](../../docs/guide/trainer/custom_component.md#4-新增任务-trainer)。

Hugging Face 模型通过 `HyperAutoModelForCausalLM` 或 `HyperAutoModelForImageTextToText` 等入口加载。自有模型通过 `model._target_` 接入自定义构建函数，并按需声明模型族 adapter。模型所支持的并行组合、模块替换和数据协议由模型实现与 adapter 决定，接入要求见 [二次开发指南](../../docs/guide/trainer/custom_component.md#31-接入自有模型)。

## 使用入口

| 任务 | 文档 |
|---|---|
| 准备框架依赖、CANN 与通信环境 | [安装指南](../../docs/installation.md) |
| 启动单机 8 卡 Qwen3-30B-A3B 训练 | [项目快速开始](../../README.md#-快速开始) |
| 修改 YAML 或通过命令行覆盖参数 | [YAML Trainer 配置结构](../../docs/guide/trainer/yaml_config.md) |
| 接入自有模型、替换模块或扩展训练组件 | [二次开发指南](../../docs/guide/trainer/custom_component.md) |
| 配置 tokenizer/processor、数据集和 DataLoader | [数据配置](../data/README.md) |
| 保存检查点并恢复训练 | [Trainer 检查点配置](../../docs/guide/distributed_checkpoint.md#在-trainer-中保存与恢复) |

## 训练配方

从与模型对应的配方开始，按实际设备、模型权重和数据修改配置：

| 模型 | 配方与说明 |
|---|---|
| Qwen3-30B-A3B | [文本训练配方](qwen3_moe/recipes/train.yaml) |
| DeepSeek-V3.2 | [文本训练配方](deepseek_v32/recipes/train.yaml) |

## TrainerConfig 配置

[`TrainerConfig`](../trainer/config/trainer.py) 定义 YAML 的根字段。常用配置及其文档如下：

| 配置字段 | 作用 | 相关文档 |
|---|---|---|
| `model` / `model_init_dtype` | 配置模型构建入口、预训练权重及最终初始化数据类型 | [模型接入](../../docs/guide/trainer/custom_component.md#31-接入自有模型) |
| `training` | 配置训练步数、batch、随机种子、梯度裁剪及日志和评估周期 | — |
| `accelerator` / `fsdp_config` | 配置并行拓扑和 FSDP/HSDP 参数分片 | [并行特性指南](../../docs/index.md#特性使用指南) |
| `fsdp_config.mix_precision` / `optimizer.fp32_main_params` | 配置 FSDP 混合精度与优化器 FP32 主参数 | [FSDP](../../docs/guide/fsdp.md)、[Optimizer](../../docs/guide/optimizer.md) |
| `plan_overrides` | 按模块匹配规则替换模型组件或覆盖并行计划 | [模块替换与并行计划](../../docs/guide/trainer/custom_component.md#2-替换模型子模块或增加并行计划) |
| `activation_checkpoint` | 配置激活重计算 | [Activation Checkpoint](../../docs/guide/activation_checkpoint.md) |
| `activation_swap` / `compile` | 配置激活换入换出和模型编译 | — |
| `dataset` / `dataloader` | 配置模型资产、样本转换、数据集读取、batch 组装和模型输入 | [Dataset / DataLoader](../data/README.md) |
| `loss_fn` / `optimizer` / `lr_scheduler` | 配置损失计算、参数分组、优化器和学习率策略 | [组件扩展](../../docs/guide/trainer/custom_component.md#3-接入新的组件实现)、[Optimizer](../../docs/guide/optimizer.md) |
| `checkpoint` | 配置模型、优化器和训练进度的保存与恢复 | [Distributed Checkpoint](../../docs/guide/distributed_checkpoint.md) |
| `debug` / `profiling` / `wandb` | 配置数据与数值检查、性能采集和训练指标上报 | — |

`mixed_precision.enabled` 当前未参与 Trainer 的混合精度配置，相关策略应通过上表中的具体字段设置。
`peft` 字段为预留配置；当前默认 [模型构建流程](./_transformers/model_builder.py) 尚未实现 PEFT 注入。

## 模块结构

```text
hyper_parallel/
├── models/          # 模型族适配层（adapter + recipes/train.yaml）、对外入口与 _transformers 模型构建
├── trainer/         # YAML 解析、TrainerConfig、组件构建、训练循环与生命周期
├── components/      # 优化器、损失函数、checkpoint、高性能 modules/functional 与量化
├── data/            # 数据集读取、样本转换与 dataloader
└── distributed/     # 分布式模型构建、并行计划应用与激活管理
```

模型族的模块替换、CP/EP 规则和训练配方位于 `models/<family>/` 下；具体目录职责见 [二次开发指南](../../docs/guide/trainer/custom_component.md#2-替换模型子模块或增加并行计划)。
