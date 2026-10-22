# 二次开发指南

HyperParallel 模型训练将 YAML 解析为 `TrainerConfig`，再由任务 Trainer 构建运行对象。`Target` 和 CLI dotted override 的解析规则见 [YAML Trainer 配置结构](yaml_config.md)。

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
torchrun --standalone --nproc_per_node=8 \
  scripts/train_lm.py hyper_parallel/models/qwen3_moe/recipes/train.yaml \
  --training.train_iters=10
```

字段路径从 `TrainerConfig` 根节点开始；字段不存在或值无法转换为声明类型时，解析失败。

## 2. 替换模型子模块或增加并行计划

模块替换通过 `plan_overrides.replace_module` 配置；完整字段见 [`plan_overrides` 定义](../../../hyper_parallel/trainer/config/parallelism.py)。

```yaml
plan_overrides:
  - match: "*.language_model.layers.*.input_layernorm"
    module_type: transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe.Qwen3VLMoeTextRMSNorm
    exact_type: true
    replace_module:
      _target_: my_project.modules.replace_rms_norm
```

YAML 解析结果为 `list[PlanOverride]`；Trainer 侧先归一化为替换规则，再由 [`_apply_module_replacement_actions()`](../../../hyper_parallel/models/replacement.py) 应用。接入必须满足：

1. `module_type` 指向可导入的 `torch.nn.Module` 类型；`match` 中的每个路径模式至少命中一个模型子模块，且命中模块符合该类型。
2. `replace_module._target_` 指向训练环境可导入的自定义函数或类。
3. target 使用 `@module_replacement`，声明 `module`、`module_fqn`、`context` 三个参数，并返回 `torch.nn.Module`。

完整实现可参考 [`RMSNorm`](../../../hyper_parallel/components/modules/rms_norm.py)。

模型族 adapter 按职责统一划分：模块/权重转换位于 `adapter/conversion/`，数据编码、processor、
transform 和 runtime-input 位于 `adapter/data/`，模型前向合约位于 `adapter/runtime/`，CP/EP
实现位于 `adapter/distributed/`，并行与重计算策略位于 `adapter/policies/`，验收声明位于
`adapter/validation/`。没有对应职责的模型无需创建空目录。YAML 通过 `_target_` 直接引用具体实现，
唯一训练入口为 `models/<family>/recipes/train.yaml`。

## 3. 接入新的组件实现

开发者可以在外部编写新的组件实现逻辑，再通过训练 YAML 的 `_target_` 接入，无需修改框架源码。

### 3.1 接入自有模型

自有模型可按配置和权重加载方式选择以下入口，训练配方见 [模型训练](../../../hyper_parallel/models/README.md)。

**兼容 Hugging Face 配置与加载协议的模型**：使用
[`register_custom_model()`](../../../hyper_parallel/models/registry.py) 将
`config.architectures[0]` 映射到可导入的模型类。模型类按所用路径实现 `from_pretrained()` 或
`from_config()`，并按需注册权重转换。使用 `from_pretrained()` 时，不要设置 `force_hf: true`，
否则会强制选择 Transformers 原生实现。

模型族通过 [`ModelAdapterSpec`](../../../hyper_parallel/models/adapter_spec.py) 声明参数分片、
FSDP 边界、CP/EP、重计算和验证等能力。仓库内的
`models/<family>/adapter/registration.py` 会被按目录约定发现；外部模型包需要在构建前显式导入
自己的注册模块。注册实现可参考
[DeepSeek-V4.1](../../../hyper_parallel/models/deepseek_v41/adapter/registration.py)。

**自定义构建函数**：通过 `model._target_` 指定可导入的模型构建函数，例如：

```yaml
model:
  _target_: my_project.models.build_model
  pretrained_model_name_or_path: /path/to/my_model
```

`my_project.models.build_model` 是由用户实现并安装到训练环境中的函数，示例仅展示配置方式。
复用现有 `TextTrainer` / `VLMTrainer` 时，需要满足：

1. 返回 `torch.nn.Module`，并提供 Trainer 使用的 `model.config`，包括 `to_dict()` 等配置接口。
   模型还需满足所选 optimizer、数据组件和 checkpoint 使用的属性与状态协议。
2. 接收并处理所需运行时参数，返回完成初始化、权重加载及目标并行配置的模型。参数列表见
   [其他可配置组件](#34-其他可配置组件)；`Target.build()` 按函数签名传参，构建函数负责应用相应能力。
3. `forward()` 接受 `dataloader.get_batch` 产生的 `model_inputs`，输出与 `loss_fn` 约定一致。
   未配置 `loss_fn` 时，默认从模型输出的 `.loss` 读取损失。
4. 配套提供 tokenizer/processor、数据转换、collator 与 get-batch runtime；新增输入协议或训练
   循环时，按本文第 4 节扩展任务 Trainer。

接入完成后，按 [模型接入验证使用指南](model_integration_validation.md) 检查权重覆盖、模块精度、
目标并行组合和断点恢复。普通 PyTorch 模型如需保留原有训练循环，也可以直接使用
[HyperParallel Core API](../../api/api_reference.md)。

### 3.2 以自定义 Loss 实现为例

[`TrainerConfig.loss_fn`](../../../hyper_parallel/trainer/config/trainer.py) 是可选的 `Target`。[`BaseTrainer._build_loss()`](../../../hyper_parallel/trainer/base.py) 根据该字段是否配置，确定训练实际使用的 Loss 模块：

- 配置 `loss_fn` 时，调用 `config.loss_fn.build()` 构建指定实现。
- 未配置 `loss_fn` 时，创建默认的 [`ModelOutputLoss`](../../../hyper_parallel/components/losses/model_output.py)；该模块从 `model_output.loss` 获取 Loss。

无论使用配置实现还是默认实现，构建结果都必须是 `torch.nn.Module`，并保存为 `self.loss_fn`。训练时，[`BaseTrainer.postforward()`](../../../hyper_parallel/trainer/base.py) 按以下方式调用该模块：

```python
self.loss_fn(model_output=outputs, labels=labels)
```

普通 post-forward 自定义 `loss_fn` 只需要满足三个条件：

1. `_target_` 构建出的对象是 `torch.nn.Module`。
2. `forward()` 接受 `model_output` 和 `labels` 两个关键字参数。
3. `forward()` 返回 `torch.Tensor` 或 `dict[str, torch.Tensor]`。

需要在完整模型输出生成前介入的 loss（例如不生成 full logits 的
[Chunk Loss](../chunk_loss.md)）还可以实现两个可选生命周期接口：

1. `bind_model(model, distributed_setup)`：Loss 构建后调用一次，用于绑定已完成 FSDP 的模型族
   adapter；
2. `prepare_model_inputs(model_inputs, loss_inputs)`：每个 micro-batch 前调用，把 get-batch runtime
   产生的独立 loss-only 字段转换为模型内 loss 协议。

普通 loss 不实现这两个方法时，Trainer 行为保持不变。

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

### 3.3 以自定义 Optimizer 实现为例

[`TrainerConfig.optimizer`](../../../hyper_parallel/trainer/config/trainer.py) 是必填的
[`OptimizerConfig`](../../../hyper_parallel/trainer/config/optimization.py)，其中 `target` 保存优化器构建对象。
[`BaseTrainer._build_optimizer()`](../../../hyper_parallel/trainer/base.py) 调用
`config.optimizer.target.build(model=self.model)` 构建配置的 Optimizer 组件，并由 Trainer 传入已构建的模型。

target 构建完成后，Trainer 调用其 `get_optimizer()`，再按 `optimizer.fp32_main_params` 与
`optimizer.swap` 配置应用相应包装，并将结果保存为 `self.optimizer`。

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

完整实现可参考 [`AdamW`](../../../hyper_parallel/components/optim/builders.py)。

### 3.4 其他可配置组件

| 配置字段 | 构建位置 | Trainer 提供的运行时参数 |
|---|---|---|
| `model` | [`BaseTrainer._build_model()`](../../../hyper_parallel/trainer/base.py) | `distributed_setup`、`peft_config`、`activation_checkpoint`、`activation_checkpoint_selection`、`swap_inputs`、`activation_swap`、`compile_config`、`model_init_dtype` |
| `dataset` | [`BaseTrainer._build_dataset()`](../../../hyper_parallel/trainer/base.py) | `transform`、`tokenizer`、`mesh_context`、`training_config` |
| `dataloader` | [`BaseTrainer._build_dataloader()`](../../../hyper_parallel/trainer/base.py) → [`build_dataloader()`](../../../hyper_parallel/data/batching/build_dataloader.py) | `dataset`、构建后的 `collate_fn`、`batch_sampler`、`batch_size`、`dp_world_size`、`max_seq_len`、`seed` |
| `optimizer` | [`BaseTrainer._build_optimizer()`](../../../hyper_parallel/trainer/base.py) | `model` |
| `lr_scheduler` | [`BaseTrainer._build_lr_scheduler()`](../../../hyper_parallel/trainer/base.py) | `optimizer`、`train_iters` |
| `loss_fn` | [`BaseTrainer._build_loss()`](../../../hyper_parallel/trainer/base.py) | 构建时无；可选生命周期接收模型、并行配置和 loss inputs |

其他组件的配置与教程见 [HyperParallel 模型训练](../../../hyper_parallel/models/README.md)。

## 4. 新增任务 Trainer

新增一种训练任务通常涉及两个位置：

1. **配置**：任务需要独立配置区域时，在 [`TrainerConfig`](../../../hyper_parallel/trainer/config/trainer.py) 中增加根字段及对应 dataclass；能够归入现有 `model`、`dataset`、`dataloader` 或 `training` 配置时，不需要增加根字段。
2. **任务 Trainer**：在 `hyper_parallel/trainer/` 中新增 `<task>_trainer.py`，复用 [`BaseTrainer`](../../../hyper_parallel/trainer/base.py) 的共享构建阶段，并实现该任务所需的模型资产、样本转换、collator、`get_batch` 和训练步骤。

以[`VLMTrainer`](../../../hyper_parallel/trainer/vlm_trainer.py) 为例，VLMTrainer 复用现有 `TrainerConfig` 配置，直接在任务 Trainer 中完成：

- [`_build_model_assets()`](../../../hyper_parallel/trainer/vlm_trainer.py)：通过 `dataset.model_assets` 构建 processor，并取得 processor 中的 tokenizer。
- [`_build_data_transform()`](../../../hyper_parallel/trainer/vlm_trainer.py)：将 processor 传给 `dataset.data_transform`，构建 `OmniDataTransform`。
- [`_build_collate_fn()`](../../../hyper_parallel/trainer/vlm_trainer.py)：向 collator target 传入 `mesh_context`，并计算梯度累积所需的 micro-batch 数。
- [`_build_get_batch()`](../../../hyper_parallel/trainer/vlm_trainer.py)：向 get-batch target 传入 `mesh_context`、`device`、`data_config` 和 `pp_shared_data`，构建 DataLoader 到模型输入的 runtime。
- [`train_step()`](../../../hyper_parallel/trainer/vlm_trainer.py)：从 `get_batch` 取得 `model_inputs` 和 `loss_inputs`，分别用于模型前向/反向和 Loss token 统计。
