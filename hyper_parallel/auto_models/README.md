# HyperModels

HyperModels 提供由 YAML 配置驱动的文本与多模态 Trainer。YAML 声明模型构建、数据处理、优化器、并行配置、训练参数及保存与恢复方式；`parse_training_args()` 将 YAML 与 CLI overrides 解析为 `TrainerConfig`，`TextTrainer` 或 `VLMTrainer` 据此构建运行对象并执行训练。分布式张量、分片与集合通信等底层能力由 HyperParallel Core 提供。

```text
YAML + CLI overrides
  -> parse_training_args(): TrainerConfig
  -> TextTrainer / VLMTrainer: build model, loss, data pipeline, optimizer,
                               lr scheduler, training context and callbacks
  -> trainer.train(): execute the training lifecycle
```

## 使用入口

安装要求、源码构建参数、CANN 与通信环境配置见[安装指南](../../docs/installation.md)。

训练启动方式见[项目快速开始](../../README.md#快速开始)。基础文本训练配置见 [`train_base.yaml`](../../examples/training_demo/train_base.yaml)，其他文本与多模态训练配置见下方“训练示例”。

## Trainer 配置结构

`TrainerConfig` 的常用配置区域包括：

| 配置区域 | 职责 | 用户指南 |
|---|---|---|
| `model` | Transformers 模型构建与预训练权重加载参数 | [HyperAutoModel 与预训练权重](../../docs/guide/trainer/hyper_auto_model_from_pretrained.md) |
| `training` | 训练步数、batch、随机种子、梯度裁剪和训练循环参数 | — |
| `accelerator` / `fsdp_config` / `plan_overrides` | 并行维度、FSDP/HSDP 配置、模块替换与分片计划 | [Trainer 分布式组件教程](../../docs/guide/trainer/components_distributed_tutorial.md) |
| `activation_checkpoint` / `activation_swap` / `compile` / `mixed_precision` | 激活重计算、激活换入换出、编译和混合精度配置 | [Activation Checkpoint](../../docs/guide/activation_checkpoint.md) |
| `dataset` / `dataloader` | 模型资产、样本转换、数据集、batch 组装和训练输入构造 | [Online Dataset](../../docs/guide/trainer/online_dataset.md) / [Indexed Dataset](../../docs/guide/trainer/index_dataset_tutorial.md) |
| `loss_fn` / `optimizer` / `lr_scheduler` | 损失函数、参数分组、优化器和学习率策略 | [Optimizer 指南](../../docs/guide/optimizer.md) |
| `checkpoint` | 模型、优化器与训练进度的保存和恢复 | [Distributed Checkpoint](../../docs/guide/distributed_checkpoint.md) |
| `debug` / `profiling` / `wandb` | 数据与数值检查、性能采集和训练指标上报 | — |

解析器主要处理两种基础表示：

- **Dataclass 配置节点**：按照字段和类型注解解析，不写 `_target_`。
- **`Target` 延迟构建节点**：通过 `_target_` 指定 Python callable；解析器保存 callable 和参数，由 Trainer 在需要运行对象时调用 `Target.build()`。

`DatasetConfig` 和 `DataLoaderConfig` 是包含 `Target` 的 dataclass 包装层，分别组织数据处理和 batch 组装所需的嵌套配置。

```yaml
training:
  train_iters: 25
  global_batch_size: 8
  micro_batch_size: 1

model:
  _target_: hyper_parallel.auto_models._transformers.HyperAutoModelForCausalLM.from_pretrained
  pretrained_model_name_or_path: Qwen/Qwen3-30B-A3B
  torch_dtype: bfloat16
```

字段与类型校验、嵌套节点解析、`Target.build()` 和 CLI dotted override 规则见 [YAML Trainer 配置结构](../../docs/guide/trainer/yaml_config.md)。

## 训练示例

`examples/training_demo/` 提供可直接运行和修改的训练配置：

| 配置 | 用途 |
|---|---|
| [`train.yaml`](../../examples/training_demo/train.yaml) | Qwen3 与 WikiText-2 端到端训练配置 |
| [`train_base.yaml`](../../examples/training_demo/train_base.yaml) | TP×CP×EP×FSDP 固定 batch 基础训练 |
| [`train_parallel_offline.yaml`](../../examples/training_demo/train_parallel_offline.yaml) | Indexed 数据并行读取 |
| [`train_parallel_online.yaml`](../../examples/training_demo/train_parallel_online.yaml) | Online 数据动态 batch 并行读取 |
| [`train_vlm.yaml`](../../examples/training_demo/train_vlm.yaml) | Qwen3.5 多模态 SFT |

启动脚本和数据准备步骤位于同一目录。Online 和 Indexed 数据处理分别见 [Online Dataset 指南](../../docs/guide/trainer/online_dataset.md)和 [Indexed Dataset 教程](../../docs/guide/trainer/index_dataset_tutorial.md)。

## 模块结构

```text
hyper_parallel/auto_models/
├── config/              # YAML 加载、CLI overrides 和类型解析入口
├── trainer/
│   ├── config.py        # TrainerConfig 及各配置节点
│   ├── base.py          # 共享的构建阶段、训练状态和生命周期能力
│   ├── text_trainer.py  # 文本数据处理、train_step 和训练循环
│   ├── vlm_trainer.py   # 多模态数据处理、train_step 和训练循环
│   └── callbacks/       # 日志、checkpoint、profiling 等生命周期回调
├── _transformers/       # 构建 Transformers 模型、应用并行计划并加载权重
├── components/          # 数据、优化器、loss、checkpoint 和分布式训练组件
├── modules/             # 可直接使用或通过 YAML 替换的高性能模型模块
└── ops/                 # 面向昇腾 NPU 的底层高性能函数
```
