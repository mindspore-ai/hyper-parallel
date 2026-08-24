# 训练组件扩展指南

`TrainerConfig` 的 optimizer、scheduler 和 loss 字段保存组件 Config。切换已有组件只需要修改 YAML 中的 `_target_`；增加新组件时，需要先定义该组件及其 Config。

下面以新增学习率 scheduler 为例。

## 1. 定义组件

在自己的 Python package 中定义 `MyWarmup`：

```python
from dataclasses import dataclass

from hyper_parallel.auto_models.components.optim import LRScheduler


class MyWarmup(LRScheduler):
    @dataclass
    class Config(LRScheduler.Config):
        warmup_steps: int = 500

    def __init__(self, config: "MyWarmup.Config", *, optimizer) -> None:
        self.config = config
        self.optimizer = optimizer
```

`MyWarmup.Config` 保存 YAML 可修改的参数。继承 `LRScheduler.Config` 后，它可以放入 `TrainerConfig.lr_scheduler`。

## 2. 在 YAML 中选择组件

假设组件位于 `my_project/scheduler.py`，YAML 写成：

```yaml
lr_scheduler:
  _target_: my_project.scheduler.MyWarmup.Config
  warmup_steps: 1000
```

解析器会完成以下检查：

```text
导入 MyWarmup.Config
  -> 检查它属于 LRScheduler.Config
  -> 检查 warmup_steps 字段与类型
  -> 创建 MyWarmup.Config
  -> 写入 TrainerConfig.lr_scheduler
```

## 3. 构建运行对象

解析阶段得到的是 Config：

```python
from hyper_parallel.auto_models.config.manager import parse_training_args

config = parse_training_args()
scheduler_config = config.lr_scheduler
scheduler = scheduler_config.build(optimizer=optimizer)
```

Trainer 取得 optimizer 等运行时依赖后，再调用该 Config 的 `build(...)` 创建 scheduler。YAML 不保存 optimizer 实例或其他运行时对象。

## 4. 扩展其他组件

新增 optimizer 时继承 `Optimizer`，新增 loss 时继承 `Loss`：

```python
from hyper_parallel.auto_models.components.loss import Loss
from hyper_parallel.auto_models.components.optim import Optimizer
```

对应 Config 分别继承 `Optimizer.Config` 和 `Loss.Config`。`TrainerConfig` 使用组件基类 Config 作为字段类型，因此新增实现不需要给 Trainer 增加新的类型分支。

YAML 根配置和解析方法见 [YAML Trainer 配置使用指南](yaml_config.md)，单次参数修改见 [CLI override 使用指南](cli_override.md)。
