# YAML Trainer 配置使用指南

HyperParallel 可以把一份 YAML 解析为有类型的 `TrainerConfig`。YAML 负责选择模型和训练组件，CLI override 用于修改单次实验参数。

本功能当前提供配置解析接口：

```text
YAML + CLI override
  -> parse_training_args()
  -> TrainerConfig
```

本接口目前用于配置编写、解析和检查；`train_lm.py` 和 `train_vl.py` 仍使用现有训练入口。

## 1. 编写 YAML

YAML 根节点直接列出本次训练需要的配置组。`model` 是必填项；其余配置组可以省略。省略普通参数组时使用 `TrainerConfig` 中的默认值，省略 optimizer、scheduler 或 loss 时对应字段为 `None`。

每个写入 YAML 的一级配置组都需要使用 `_target_` 选择具体的 Config 类：

```yaml
model:
  _target_: hyper_parallel.trainer.config.ModelConfig
  name: qwen3_5
  weights_path: /models/Qwen3.5-0.8B-Base

optimizer:
  _target_: hyper_models.components.optim.AdamW.Config
  lr: 0.0002
  weight_decay: 0.1
  betas: [0.9, 0.95]

lr_scheduler:
  _target_: hyper_models.components.optim.CosineWithWarmup.Config
  warmup_ratio: 0.05
  min_lr: 0.00001

loss:
  _target_: hyper_models.components.loss.CausalLMLoss.Config
  ignore_index: -100

training:
  _target_: hyper_models.trainer.config.TrainingConfig
  max_steps: 100
  global_batch_size: 8
  init_device: meta
  loss_aggregation: token_weighted

accelerator:
  _target_: hyper_models.trainer.config.AcceleratorConfig
  tp_size: 2
  dp_shard_size: 4

mixed_precision:
  _target_: hyper_models.trainer.config.MixedPrecisionConfig
  enabled: true

gradient_checkpointing:
  _target_: hyper_models.trainer.config.GradientCheckpointingConfig
  activation_checkpoint: full

debug:
  _target_: hyper_models.trainer.config.DebugConfig
  check_nan_inf: true
```

根节点不写 `_target_`。解析器会将这些配置组组合成一个 `TrainerConfig`。

## 2. 解析配置

配置解析调用 `parse_training_args()`：

```python
from hyper_models.config.manager import parse_training_args

config = parse_training_args()
```

该函数从命令行读取第一个位置参数作为 YAML 路径。当前可以直接检查解析结果（例如）：

```bash
python -c \
  'from hyper_models.config.manager import parse_training_args; print(parse_training_args())' \
  configs/qwen3_5.yaml
```

解析完成后，各节点已经是具体的 Config 对象：

```python
config.model             # ModelConfig
config.optimizer         # AdamW.Config
config.lr_scheduler      # CosineWithWarmup.Config
config.loss              # CausalLMLoss.Config
config.training          # TrainingConfig
config.accelerator       # AcceleratorConfig
```

此时尚未创建真正的 optimizer、scheduler 或 loss 运行对象。

## 3. 配置错误

以下问题会在配置解析阶段报错：

| 问题 | 示例 |
| --- | --- |
| 缺少必填配置 | YAML 没有 `model` |
| 缺少组件 target | `optimizer` 中没有 `_target_` |
| target 不存在 | Python 路径无法导入 |
| 字段不存在 | `accelerator.tp_szie` 拼写错误 |
| 参数类型错误 | `tp_size: two` |
| override 路径不存在 | `--accelerator.tp_szie=4` |

错误信息包含配置路径，例如：

```text
$.accelerator.tp_size: expected int, got str
```

这样可以在模型、分布式环境和训练组件创建之前发现输入问题。

CLI 参数修改见 [CLI override 使用指南](cli_override.md)，组件替换与扩展见 [训练组件扩展指南](custom_component.md)。
