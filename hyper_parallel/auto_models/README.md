# AutoModels

AutoModels 提供由 YAML 配置驱动的文本与多模态 Trainer。`parse_training_args()` 将 YAML 和 CLI dotted overrides 解析为 `TrainerConfig`，`TextTrainer` 或 `VLMTrainer` 再根据配置构建训练组件并执行训练。分布式张量、模型分片和集合通信等底层能力由 HyperParallel Core 提供。

## 使用入口

- 安装要求、源码构建参数、CANN 与通信环境配置见 [安装指南](../../docs/installation.md)。
- 训练启动方式见 [项目快速开始](../../README.md#快速开始)。

## TrainerConfig 配置

[`TrainerConfig`](trainer/config.py#L720-L753) 中由当前 Trainer 主路径使用的配置区域如下：

| 配置字段 | 作用 | 相关文档 |
|---|---|---|
| `model` / `peft` | 配置模型构建入口、预训练权重、数据类型、注意力实现和 PEFT | — |
| `training` | 配置训练步数、batch、随机种子、梯度裁剪及日志和评估周期 | — |
| `accelerator` / `fsdp_config` | 配置并行拓扑和 FSDP/HSDP 参数分片 | — |
| `mixed_precision` | 配置 Trainer 混合精度开关 | — |
| `plan_overrides` | 按模块匹配规则替换模型组件或覆盖并行计划 | — |
| `activation_checkpoint` | 配置激活重计算 | [Activation Checkpoint](../../docs/guide/activation_checkpoint.md) |
| `activation_swap` / `compile` | 配置激活换入换出和模型编译 | — |
| `dataset` / `dataloader` | 配置模型资产、样本转换、数据集读取、batch 组装和模型输入 | [Dataset / DataLoader](components/datasets/README.md) |
| `loss_fn` / `optimizer` / `lr_scheduler` | 配置损失计算、参数分组、优化器和学习率策略 | — |
| `checkpoint` | 配置模型、优化器和训练进度的保存与恢复 | [Distributed Checkpoint](../../docs/guide/distributed_checkpoint.md) |
| `debug` / `profiling` / `wandb` | 配置数据与数值检查、性能采集和训练指标上报 | — |

字段解析、`Target` 和 CLI dotted override 规则见 [YAML Trainer 配置结构](../../docs/guide/trainer/yaml_config.md)；外部组件的接入方式见 [AutoModels 二次开发指南](../../docs/guide/trainer/custom_component.md)。

## 模块结构

```text
hyper_parallel/auto_models/
├── config/          # YAML 解析与 CLI dotted overrides
├── trainer/         # 组件构建、训练循环与生命周期
├── _transformers/   # Transformers 模型构建、并行计划与权重加载
├── components/      # 数据、优化器、损失函数、checkpoint、编译与分布式组件
├── modules/         # 可直接使用或通过模型替换计划注入的高性能模块
└── ops/             # 面向昇腾 NPU 的底层高性能函数
```
