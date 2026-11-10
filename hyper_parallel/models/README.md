# HyperParallel Model Training

**English** | [简体中文](README.zh.md)

HyperParallel model training provides YAML-driven text and multimodal trainers. `parse_training_args()` parses YAML and CLI dotted overrides into `TrainerConfig`; `TextTrainer` or `VLMTrainer` then builds the training components and runs training according to that configuration. HyperParallel Core provides the underlying distributed tensors, model sharding, and collective communication.

## Entry Points

- See the [Installation Guide](../../docs/installation.md) for installation requirements, source build options, and CANN and communication environment configuration.
- See the [Project Quick Start](../../README.md#-quick-start) for training launch instructions.

For numerical validation and checkpoint recovery acceptance when integrating a new Hugging Face or custom model, see the [Model Integration Validation Guide](../../docs/guide/trainer/model_integration_validation.md).

## TrainerConfig Configuration

The current main Trainer path uses the following configuration sections in [`TrainerConfig`](../trainer/config/trainer.py#L47-L114):

| Configuration field | Purpose | Related documentation |
|---------------------|---------|-----------------------|
| `model` / `peft` | Model construction entry point, pretrained weights, dtype, attention implementation, and PEFT | — |
| `training` | Training steps, batches, random seed, gradient clipping, logging, and evaluation intervals | — |
| `accelerator` / `fsdp_config` | Parallel topology and FSDP/HSDP parameter sharding | — |
| `mixed_precision` | Trainer mixed-precision switches | — |
| `plan_overrides` | Replace model components or override parallel plans using module matching rules | — |
| `activation_checkpoint` | Activation recomputation | [Activation Checkpoint](../../docs/guide/activation_checkpoint.md) |
| `activation_swap` / `compile` | Activation swapping and model compilation | — |
| `dataset` / `dataloader` | Model assets, sample transformation, dataset reading, batch assembly, and model inputs | [Dataset / DataLoader](../data/README.md) |
| `loss_fn` / `optimizer` / `lr_scheduler` | Loss computation, parameter groups, optimizers, and learning rate policies | — |
| `checkpoint` | Save and restore models, optimizers, and training progress | [Distributed Checkpoint](../../docs/guide/distributed_checkpoint.md) |
| `debug` / `profiler` / `wandb` | Data and numerical checks, profiling, and training metric reporting | — |

See [YAML Trainer Configuration](../../docs/guide/trainer/yaml_config.md) for field parsing, `Target`, and CLI dotted override rules, and the [Extension Guide](../../docs/guide/trainer/custom_component.md) for integrating external components.

## Module Structure

```text
hyper_parallel/
├── models/          # Model-family adapters and recipes/train.yaml, public entry points, and _transformers construction
├── trainer/         # YAML parsing, TrainerConfig, component construction, training loops, and lifecycle
├── components/      # Optimizers, losses, checkpoints, high-performance modules/functional, and quantization
├── data/            # Dataset reading, sample transformations, and dataloaders
└── distributed/     # Distributed model construction, parallel plan application, and activation management
```

Each model family's replacement factories, CP/EP rules, and training entry points reside under `models/<family>/`. The repository currently ships `models/qwen3_moe/recipes/train.yaml`, `models/deepseek_v32/recipes/train.yaml`, and `models/kimi_k26/recipes/train.yaml`. The Kimi-K2.6 multimodal recipe targets four SuperPoDs / 512 devices; see the [Kimi-K2.6 model-family guide](kimi_k26/README.md).
