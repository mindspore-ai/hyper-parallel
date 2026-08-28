# AutoModels 二次开发指南

AutoModels 将 YAML 解析为 `TrainerConfig`，再由任务 Trainer 构建运行对象。`Target` 和 CLI dotted override 的解析规则见 [YAML Trainer 配置结构](yaml_config.md)。

## 扩展入口

| 需求 | 推荐入口 | 是否修改 Trainer |
|---|---|---|
| 调整已有组件参数 | 修改 YAML 字段或使用 CLI override | 否 |
| 替换模型子模块或增加并行计划 | 在 `plan_overrides` 中配置匹配规则、替换函数或分片契约 | 否 |
| 增加模型构建、数据处理、损失函数、优化器或学习率策略 | 实现 Python 类或函数，并通过 `_target_` 引用 | 否 |
| 支持新的模态资产、模型输入协议或训练循环 | 新增任务 Trainer，复用 `BaseTrainer` 的共享阶段 | 是 |

`plan_overrides` 中的 `_target_` 修改模型内部结构或计算；组件节点的 `_target_` 构建 Trainer 直接使用的模型、数据、Loss、Optimizer 或 Scheduler 对象。

## 1. 调整已有组件参数

改动位置：训练 YAML 或启动命令。

```yaml
training:
  train_iters: 10
```

同一字段可以由 CLI 覆盖：

```bash
bash examples/demo_trainer/launch_1node_8dies.sh \
  --training.train_iters=10
```

字段路径从 `TrainerConfig` 根节点开始；字段不存在或值无法转换为声明类型时，解析失败。

## 2. 替换模型子模块或增加并行计划

模块替换通过 `plan_overrides.replace_module` 配置；完整字段见 [`plan_overrides` 定义](../../../hyper_parallel/auto_models/trainer/config.py#L246-L353)。

```yaml
plan_overrides:
  - match: "*.language_model.layers.*.input_layernorm"
    module_type: transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe.Qwen3VLMoeTextRMSNorm
    exact_type: true
    replace_module:
      _target_: my_project.modules.replace_rms_norm
```

YAML 解析结果为 `list[PlanOverride]`；其中的模块替换条目由 [`_apply_module_replacement_actions()`](../../../hyper_parallel/auto_models/_transformers/infrastructure.py#L540-L601) 应用。接入必须满足：

1. `module_type` 指向可导入的 `torch.nn.Module` 类型；`match` 中的每个路径模式至少命中一个模型子模块，且命中模块符合该类型。
2. `replace_module._target_` 指向训练环境可导入的自定义函数或类。
3. target 使用 `@module_replacement`，声明 `module`、`module_fqn`、`context` 三个参数，并返回 `torch.nn.Module`。

完整实现可参考 [`RMSNorm`](../../../hyper_parallel/auto_models/modules/rms_norm.py#L30-L79)。

## 3. 接入新的组件实现

开发者可以在外部编写新的组件实现逻辑，再通过训练 YAML 的 `_target_` 接入，无需修改框架源码。

### 3.1 以自定义 Loss 实现为例

[`TrainerConfig.loss_fn`](../../../hyper_parallel/auto_models/trainer/config.py#L720-L729) 是可选的 `Target`。[`BaseTrainer._build_loss()`](../../../hyper_parallel/auto_models/trainer/base.py#L395-L404) 根据该字段是否配置，确定训练实际使用的 Loss 模块：

- 配置 `loss_fn` 时，调用 `config.loss_fn.build()` 构建指定实现。
- 未配置 `loss_fn` 时，创建默认的 [`ModelOutputLoss`](../../../hyper_parallel/auto_models/components/loss/model_output.py#L26-L55)；该模块从 `model_output.loss` 获取 Loss。

无论使用配置实现还是默认实现，构建结果都必须是 `torch.nn.Module`，并保存为 `self.loss_fn`。训练时，[`BaseTrainer.postforward()`](../../../hyper_parallel/auto_models/trainer/base.py#L595-L607) 按以下方式调用该模块：

```python
self.loss_fn(model_output=outputs, labels=labels)
```

因此，自定义 `loss_fn` 只需要满足三个条件：

1. `_target_` 构建出的对象是 `torch.nn.Module`。
2. `forward()` 接受 `model_output` 和 `labels` 两个关键字参数。
3. `forward()` 返回 `torch.Tensor` 或 `dict[str, torch.Tensor]`。

以下列 `my_project/losses.py` 为例：

```python
from typing import Any

import torch


class ScaledModelOutputLoss(torch.nn.Module):
    def __init__(self, scale: float = 1.0) -> None:
        super().__init__()
        self.scale = scale

    def forward(
        self,
        *,
        model_output: Any,
        labels: torch.Tensor | None,
    ) -> torch.Tensor:
        del labels
        return model_output.loss * self.scale
```

在训练 YAML 中配置该实现：

```yaml
loss_fn:
  _target_: my_project.losses.ScaledModelOutputLoss
  scale: 0.5
```

### 3.2 以自定义 Optimizer 实现为例

[`TrainerConfig.optimizer`](../../../hyper_parallel/auto_models/trainer/config.py#L724-L728) 是必填的 `Target`。[`BaseTrainer._build_optimizer()`](../../../hyper_parallel/auto_models/trainer/base.py#L485-L489) 调用 `config.optimizer.build(model=self.model)` 构建配置的 Optimizer 组件，并由 Trainer 传入已构建的模型。

target 构建完成后，Trainer 调用其 `get_optimizer()`，并将返回值保存为 `self.optimizer`。

因此，自定义 Optimizer target 只需要满足两个条件：

1. 接受 `model` 关键字参数。
2. 构建出的对象提供 `get_optimizer()`，返回训练循环使用的 Optimizer。

以下列 `my_project/optimizers.py` 为例：

```python
import torch


class SGDOptimizer:
    def __init__(self, lr: float, *, model: torch.nn.Module) -> None:
        self.optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    def get_optimizer(self) -> torch.optim.Optimizer:
        return self.optimizer
```

在训练 YAML 中配置该实现：

```yaml
optimizer:
  _target_: my_project.optimizers.SGDOptimizer
  lr: 1.0e-3
```

完整实现可参考 [`AdamW`](../../../hyper_parallel/auto_models/components/optim/optimizer/optimizer.py#L60-L152)。

### 3.3 其他可配置组件

| 配置字段 | 构建位置 | Trainer 提供的运行时参数 |
|---|---|---|
| `model` | [`BaseTrainer._build_model()`](../../../hyper_parallel/auto_models/trainer/base.py#L374-L383) | `distributed_setup`、`peft_config`、`activation_checkpoint`、`activation_swap`、`compile_config` |
| `dataset` | [`BaseTrainer._build_dataset()`](../../../hyper_parallel/auto_models/trainer/base.py#L414-L421) | `transform`、`tokenizer`、`mesh_context`、`training_config` |
| `dataloader` | [`BaseTrainer._build_dataloader()`](../../../hyper_parallel/auto_models/trainer/base.py#L434-L446) → [`build_dataloader()`](../../../hyper_parallel/auto_models/components/datasets/batching/build_dataloader.py#L246-L254) | `dataset`、构建后的 `collate_fn`、`batch_sampler`、`batch_size`、`dp_world_size`、`max_seq_len`、`seed` |
| `optimizer` | [`BaseTrainer._build_optimizer()`](../../../hyper_parallel/auto_models/trainer/base.py#L485-L489) | `model` |
| `lr_scheduler` | [`BaseTrainer._build_lr_scheduler()`](../../../hyper_parallel/auto_models/trainer/base.py#L491-L497) | `optimizer`、`train_iters` |
| `loss_fn` | [`BaseTrainer._build_loss()`](../../../hyper_parallel/auto_models/trainer/base.py#L395-L404) | 无 |

其他组件的配置与教程见 [AutoModels README](../../../hyper_parallel/auto_models/README.md)。

## 4. 新增任务 Trainer

新增一种训练任务通常涉及两个位置：

1. **配置**：任务需要独立配置区域时，在 [`TrainerConfig`](../../../hyper_parallel/auto_models/trainer/config.py#L720-L755) 中增加根字段及对应 dataclass；能够归入现有 `model`、`dataset`、`dataloader` 或 `training` 配置时，不需要增加根字段。
2. **任务 Trainer**：在 `hyper_parallel/auto_models/trainer/` 中新增 `<task>_trainer.py`，复用 [`BaseTrainer`](../../../hyper_parallel/auto_models/trainer/base.py#L276-L321) 的共享构建阶段，并实现该任务所需的模型资产、样本转换、collator、`get_batch` 和训练步骤。

以[`VLMTrainer`](../../../hyper_parallel/auto_models/trainer/vlm_trainer.py#L34-L298) 为例，VLMTrainer 复用现有 `TrainerConfig` 配置，直接在任务 Trainer 中完成：

- [`_build_model_assets()`](../../../hyper_parallel/auto_models/trainer/vlm_trainer.py#L66-L85)：根据模型路径构建 processor，并取得 processor 中的 tokenizer。
- [`_build_data_transform()`](../../../hyper_parallel/auto_models/trainer/vlm_trainer.py#L87-L97) 和 [`_build_collate_fn()`](../../../hyper_parallel/auto_models/trainer/vlm_trainer.py#L99-L110)：将 processor 传给多模态样本转换，并构建多模态 collator。
- [`_build_get_batch()`](../../../hyper_parallel/auto_models/trainer/vlm_trainer.py#L112-L127)：根据并行 mesh 创建 `parallel_context`，再构建 DataLoader 到模型输入的 batch adapter。
- [`train_step()`](../../../hyper_parallel/auto_models/trainer/vlm_trainer.py#L177-L205)：从 `get_batch` 取得 `model_inputs` 和 `loss_inputs`，分别用于模型前向/反向和 Loss token 统计。
