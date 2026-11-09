# YAML Trainer Configuration

**English** | [简体中文](yaml_config.zh.md)

HyperParallel model training parses YAML according to the fields of [`TrainerConfig`](../../../hyper_parallel/trainer/config/trainer.py#L47-L105). Field names determine the accepted configuration keys; field types determine how values are parsed. Parsing produces a `TrainerConfig` object, then CLI dotted overrides update its fields.

```python
@dataclass
class TrainerConfig:
    model: Target[Any]
    optimizer: Target[Optimizer]
    training: TrainingConfig = field(default_factory=TrainingConfig)
    accelerator: AcceleratorConfig = field(default_factory=AcceleratorConfig)
    ...
```

`training` and `accelerator` are parsed as their respective dataclasses; `model` and `optimizer` are parsed as `Target` objects.

## 1. Dataclass Nodes

When a `TrainerConfig` field is declared as a dataclass, its YAML mapping is parsed according to that dataclass's fields, without `_target_`:

```yaml
training:
  train_iters: 25
  global_batch_size: 8
  micro_batch_size: 1
  backend: hccl
  max_grad_norm: 1.0
  seed: 42

accelerator:
  tp_size: 1
  cp_size: 1
  ep_size: 1
  pp_size: 1
  sequence_parallel: false
  loss_parallel: false
```

The two mappings above become `TrainingConfig` and `AcceleratorConfig`. Following the dataclass definitions, the parser:

- Validates that YAML keys correspond to declared fields.
- Fills defaults for omitted fields and checks required fields without defaults.
- Converts scalars and containers according to field types, including `Optional`, `Literal`, lists, tuples, and nested dataclasses.
- Preserves full paths in type and field errors, such as `$.accelerator.tp_size: expected int, got str`.

The outer field's type annotation determines the dataclass for each nested mapping. The root YAML therefore accepts only fields declared by `TrainerConfig`; nested configurations accept only fields declared by the corresponding dataclass.

### Online NPU Operator Stress Testing

Configure stress testing through [`StressDetectConfig`](../../../hyper_parallel/trainer/config/stress_detect.py),
alongside `training`. Detection runs after a complete optimizer step:

```yaml
stress_detect:
  op_stress_detect:
    detection_interval: 100        # Step interval; defaults to 0, which creates no stress-testing callback.
    num_detections: 3              # Detections per type per round; defaults to 1.
    detect_type: all               # aic (default), hccs, or all.
    exit_training_on_sdc: false    # Whether a numerical anomaly stops training; defaults to false.
```

`aic` tests AI Cores, `hccs` tests inter-device links within a node, and `all` runs AIC first, then HCCS.
Results are handled immediately after each detection returns:

- Numerical anomaly: log ERROR; with `exit_training_on_sdc: true`, immediately raise an exception, stopping training and remaining detections.
- Execution failure or unknown return value: log WARNING and continue training.
- Low-level exceptions, such as voltage-restoration failures: propagate directly.

Only NPU is supported, and approximately 10 GB of device memory must be reserved. Use identical configuration on all ranks; HCCS additionally requires initialized distributed communication.
For other hardware and runtime constraints, see the [torch_npu API documentation](https://www.hiascend.com/document/detail/zh/Pytorch/700/apiref/apilist/ptaoplist_001321.html).
Set `HP_LOG_CONFIG=HP:INFO` to see INFO logs for detection start and success.

`model_stress_detect` is reserved configuration and cannot currently be enabled.

## 2. Target Nodes

[`Target`](../../../hyper_parallel/trainer/config/target.py#L54) is a configuration object for deferred invocation of a Python class or function. It stores the imported target, original import path, and YAML arguments. `build()` merges runtime arguments supplied by the Trainer and creates the component.

For example, consider model configuration:

```yaml
model:
  _target_: hyper_parallel.models.HyperAutoModelForCausalLM.from_pretrained
  pretrained_model_name_or_path: Qwen/Qwen3-30B-A3B
  ignore_mismatched_sizes: true
  torch_dtype: bfloat16
  attn_implementation: sdpa
  force_hf: true
```

The framework parses the configuration through [`_resolve_target()`](../../../hyper_parallel/trainer/config/resolver.py#L347-L382): the resolver imports the object referenced by `_target_`, validates and normalizes YAML arguments, then constructs:

```python
config.model = Target(
    HyperAutoModelForCausalLM.from_pretrained,
    target_path=(
        "hyper_parallel.models."
        "HyperAutoModelForCausalLM.from_pretrained"
    ),
    **normalized_args,
)
```

`normalized_args` contains the YAML arguments and defaults from [`HyperAutoModelForCausalLM.from_pretrained()`](../../../hyper_parallel/models/_transformers/auto_model.py#L77-L152). This creates only a configuration object; it neither calls `from_pretrained` nor loads model weights.

[`BaseTrainer._build_model()`](../../../hyper_parallel/trainer/base.py#L285-L314) constructs the model after distributed environment initialization:

```python
self.model = self.config.model.build(
    distributed_setup=self.distributed_setup,
    peft_config=self.peft_config,
    activation_checkpoint=self.config.activation_checkpoint.mode,
    activation_swap=self.config.activation_swap,
    compile_config=self.config.compile,
)
```

After merging YAML arguments with runtime arguments supplied by the Trainer, this is equivalent to calling `HyperAutoModelForCausalLM.from_pretrained()`:

```python
self.model = HyperAutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path="Qwen/Qwen3-30B-A3B",
    ignore_mismatched_sizes=True,
    torch_dtype="bfloat16",
    attn_implementation="sdpa",
    force_hf=True,
    distributed_setup=self.distributed_setup,
    peft_config=self.peft_config,
    activation_checkpoint=self.config.activation_checkpoint.mode,
    activation_swap=self.config.activation_swap,
    compile_config=self.config.compile,
)
```

[`TrainerConfig`](../../../hyper_parallel/trainer/config/trainer.py#L47-L105) declares each node's parsing behavior through its field type: `Target[...]` fields become `Target` objects; dataclass fields become their corresponding dataclasses.

When parsing a `Target`, the parser checks:

1. Whether `_target_` can be imported and the imported object is callable.
2. Whether every YAML argument name can be passed to the target as a keyword argument.
3. Whether YAML values can be validated and converted according to target argument annotations, where present.

The parser also fills in target function defaults. Required arguments supplied by the Trainer during construction need not be included in YAML.

See the [Extension Guide](custom_component.md) for integrating external models and training components through `_target_`.

Performance tracing uses the root `profiler` node; see [Training profiler](profiler.md) for fields and migration instructions.

Allocator memory history is configured through the root `memory` node. For fields, step boundaries, and exception-cleanup semantics, see
[Memory snapshot configuration](memory_profiler.md).

## 3. CLI Dotted Override

CLI overrides run after YAML has been parsed into `TrainerConfig`, using `--field.path=value` to locate and replace values in the configuration tree:

```bash
--training.train_iters=100
--model.pretrained_model_name_or_path=/path/to/model
--optimizer.adamw_config.adamw_lr=2.0e-5
--profiler.enable=true
```

Values are first parsed with `yaml.safe_load()`, then converted according to the target node's type:

```bash
--optimizer.no_decay_params='[bias, norm, ln_]'
--compile.options='{trace.enabled: true}'
--profiler.enable=true
--checkpoint.restore_from=null
```

Path replacement follows the configuration tree's structure:

- Use `--field.path=value`; the separate form `--field.path value` is not accepted.
- Dataclass paths must refer to existing fields.
- Ordinary mappings allow updates only to existing keys.
- `Target` arguments must already be configured or appear in the callable's signature; callables with `**kwargs` may accept new arguments.
- `_target_` is not replaceable; CLI overrides do not change the callable.
- An optional component whose current value is `None` has no configuration nodes to traverse; YAML must provide its structure.
