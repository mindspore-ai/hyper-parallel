# Extension Guide

**English** | [简体中文](custom_component.zh.md)

HyperParallel model training parses YAML into `TrainerConfig`, then a task Trainer constructs runtime objects. See [YAML Trainer Configuration](yaml_config.md) for `Target` and CLI dotted override parsing rules.

## Extension Points

| Requirement | Recommended entry point | Modify Trainer? |
|---|---|---|
| Adjust existing component parameters | Edit YAML fields or use CLI overrides | No |
| Replace model submodules or add parallel plans | Configure matching rules, replacement functions, or sharding contracts in `plan_overrides` | No |
| Add model construction, data processing, losses, optimizers, or learning rate policies | Implement a Python class or function and reference it through `_target_` | No |
| Support new modality assets, model input protocols, or training loops | Add a task Trainer reusing shared `BaseTrainer` phases | Yes |

A `_target_` in `plan_overrides` modifies model internals or computation; a component node's `_target_` constructs a model, data, loss, optimizer, or scheduler object used directly by the Trainer.

## 1. Adjust Existing Component Parameters

Edit the training YAML or launch command.

```yaml
training:
  train_iters: 10
```

The same field can be overridden through the CLI:

```bash
torchrun --standalone --nproc_per_node=8 \
  scripts/train_lm.py hyper_parallel/models/qwen3_moe/recipes/train.yaml \
  --training.train_iters=10
```

Field paths start at the `TrainerConfig` root. Parsing fails if a field does not exist or its value cannot be converted to the declared type.

## 2. Replace Model Submodules or Add Parallel Plans

Configure module replacements through `plan_overrides.replace`; see the [`PlanOverride` definition](../../../hyper_parallel/trainer/config/parallelism.py) for all fields.

```yaml
plan_overrides:
  - match: "*.language_model.layers.*.input_layernorm"
    replace:
      kind: module
      implementation: my_project.modules.replace_rms_norm
```

YAML parsing produces `list[PlanOverride]`. The Trainer normalizes this into replacement rules (§15.2.6), then [`_apply_module_replacement_actions()`](../../../hyper_parallel/models/replacement.py) applies them. Integration requires:

1. Each path pattern in `match` must match at least one model submodule; the framework obtains the actual type from matched objects. Set `replace.source_type` only when an additional expected-type assertion is needed, optionally with `exact_type: true`.
2. `replace.implementation` points to a replacement factory importable in the training environment. Supply static arguments with a `{_target_: path, ...}` mapping.
3. Decorate the factory with `@module_replacement`, declare `module`, `module_fqn`, and `context` parameters, and return a `torch.nn.Module`.

See [`RMSNorm`](../../../hyper_parallel/components/modules/rms_norm.py#L31-L82) for a complete implementation.

Model-family adapters are organized by responsibility: module/weight conversion in `adapter/conversion/`; data encoding, processors,
transforms, and runtime inputs in `adapter/data/`; model forward contracts in `adapter/runtime/`; CP/EP
implementations in `adapter/distributed/`; parallel and recomputation policies in `adapter/policies/`; and acceptance declarations in
`adapter/validation/`. Do not create empty directories for responsibilities a model does not have. YAML references concrete implementations directly through `_target_`.
The sole training entry point is `models/<family>/recipes/train.yaml`.

## 3. Integrate New Components

Developers can implement components externally and reference them through `_target_` in training YAML without modifying framework source code.

### 3.1 Custom Loss Example

[`TrainerConfig.loss_fn`](../../../hyper_parallel/trainer/config/trainer.py#L54) is an optional `Target`. [`BaseTrainer._build_loss()`](../../../hyper_parallel/trainer/base.py#L315-L325) selects the loss module based on whether this field is configured:

- When `loss_fn` is configured, `config.loss_fn.build()` constructs the specified implementation.
- Otherwise, the default [`ModelOutputLoss`](../../../hyper_parallel/components/losses/model_output.py#L26-L58) reads the loss from `model_output.loss`.

In either case, the constructed object must be a `torch.nn.Module` and is stored as `self.loss_fn`. During training, [`BaseTrainer.postforward()`](../../../hyper_parallel/trainer/base.py#L517-L530) invokes it as follows:

```python
self.loss_fn(model_output=outputs, labels=labels)
```

An ordinary custom post-forward `loss_fn` has three requirements:

1. `_target_` constructs a `torch.nn.Module`.
2. `forward()` accepts the keyword arguments `model_output` and `labels`.
3. `forward()` returns `torch.Tensor` or `dict[str, torch.Tensor]`.

Losses that intervene before complete model outputs are generated, such as
[Chunk Loss](../chunk_loss.md), which avoids full logits, may implement two optional lifecycle interfaces:

1. `bind_model(model, distributed_setup)`: Called once after loss construction to bind the model-family
   adapter after FSDP has been applied;
2. `prepare_model_inputs(model_inputs, loss_inputs)`: Called before each micro-batch to convert the get-batch runtime's
   separate loss-only fields into the model-integrated loss protocol.

Trainer behavior is unchanged for ordinary losses that do not implement these methods.

For example, in `my_project/losses.py`:

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

Configure the implementation in training YAML:

```yaml
loss_fn:
  _target_: my_project.losses.ScaledModelOutputLoss
  scale: 0.5
```

### 3.2 Custom Optimizer Example

[`TrainerConfig.optimizer`](../../../hyper_parallel/trainer/config/trainer.py#L51) is a required `Target`. [`BaseTrainer._build_optimizer()`](../../../hyper_parallel/trainer/base.py#L405-L414) calls `config.optimizer.build(model=self.model)` to construct the configured optimizer component, supplying the already-built model.

After target construction, the Trainer calls `get_optimizer()` and stores its result as `self.optimizer`.

A custom optimizer target therefore has two requirements:

1. Accept the `model` keyword argument.
2. Provide `get_optimizer()` on the constructed object, returning the optimizer used by the training loop.

For example, in `my_project/optimizers.py`:

```python
import torch


class SGDOptimizer:
    def __init__(self, lr: float, *, model: torch.nn.Module) -> None:
        self.optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    def get_optimizer(self) -> torch.optim.Optimizer:
        return self.optimizer
```

Configure the implementation in training YAML:

```yaml
optimizer:
  _target_: my_project.optimizers.SGDOptimizer
  lr: 1.0e-3
```

See [`AdamW`](../../../hyper_parallel/components/optim/builders.py#L37-L107) for a complete implementation.

### 3.3 Other Configurable Components

| Configuration field | Construction site | Runtime arguments supplied by the Trainer |
|---|---|---|
| `model` | [`BaseTrainer._build_model()`](../../../hyper_parallel/trainer/base.py#L285-L314) | `distributed_setup`, `peft_config`, `activation_checkpoint`, `activation_swap`, `compile_config` |
| `dataset` | [`BaseTrainer._build_dataset()`](../../../hyper_parallel/trainer/base.py#L334-L349) | `transform`, `tokenizer`, `mesh_context`, `training_config` |
| `dataloader` | [`BaseTrainer._build_dataloader()`](../../../hyper_parallel/trainer/base.py#L354-L370) → [`build_dataloader()`](../../../hyper_parallel/data/batching/build_dataloader.py#L171-L269) | `dataset`, constructed `collate_fn`, `batch_sampler`, `batch_size`, `dp_world_size`, `max_seq_len`, `seed` |
| `optimizer` | [`BaseTrainer._build_optimizer()`](../../../hyper_parallel/trainer/base.py#L405-L414) | `model` |
| `lr_scheduler` | [`BaseTrainer._build_lr_scheduler()`](../../../hyper_parallel/trainer/base.py#L415-L422) | `optimizer`, `train_iters` |
| `loss_fn` | [`BaseTrainer._build_loss()`](../../../hyper_parallel/trainer/base.py#L315-L325) | None during construction; optional lifecycle hooks receive the model, parallel configuration, and loss inputs |

See [HyperParallel Model Training](../../../hyper_parallel/models/README.md) for other component configurations and tutorials.

## 4. Add a Task Trainer

A new training task typically requires changes in two places:

1. **Configuration**: Add a root field and its dataclass to [`TrainerConfig`](../../../hyper_parallel/trainer/config/trainer.py#L47-L105) only when the task needs an independent configuration section. No new root field is needed if the settings fit existing `model`, `dataset`, `dataloader`, or `training` sections.
2. **Task Trainer**: Add `<task>_trainer.py` under `hyper_parallel/trainer/`, reuse shared construction phases from [`BaseTrainer`](../../../hyper_parallel/trainer/base.py#L105), and implement the task's model assets, sample transformation, collator, `get_batch`, and training steps.

For example, [`VLMTrainer`](../../../hyper_parallel/trainer/vlm_trainer.py#L34-L298) reuses `TrainerConfig` and implements these operations directly in the task Trainer:

- [`_build_model_assets()`](../../../hyper_parallel/trainer/vlm_trainer.py#L66-L85): Build the processor from the model path and obtain its tokenizer.
- [`_build_data_transform()`](../../../hyper_parallel/trainer/vlm_trainer.py#L87-L97) and [`_build_collate_fn()`](../../../hyper_parallel/trainer/vlm_trainer.py#L99-L110): Pass the processor to the multimodal sample transformation and build the multimodal collator.
- [`_build_get_batch()`](../../../hyper_parallel/trainer/vlm_trainer.py#L112-L127): Create `parallel_context` from the parallel mesh, then build the get-batch runtime that converts DataLoader output into model inputs.
- [`train_step()`](../../../hyper_parallel/trainer/vlm_trainer.py#L177-L205): Obtain `model_inputs` and `loss_inputs` from `get_batch` for model forward/backward and loss token statistics respectively.
