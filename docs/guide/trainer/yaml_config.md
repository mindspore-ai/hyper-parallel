# YAML Trainer 配置结构

AutoModels 按 [`TrainerConfig`](../../../hyper_parallel/trainer/config/trainer.py#L47-L105) 的字段定义解析 YAML。字段名决定 YAML 可以包含哪些配置项，字段类型决定对应的值如何解析。解析完成后得到 `TrainerConfig` 对象，随后通过 CLI dotted overrides 更新配置字段。

```python
@dataclass
class TrainerConfig:
    model: Target[Any]
    optimizer: Target[Optimizer]
    training: TrainingConfig = field(default_factory=TrainingConfig)
    accelerator: AcceleratorConfig = field(default_factory=AcceleratorConfig)
    ...
```

`training` 和 `accelerator` 按各自的 dataclass 解析；`model`、`optimizer`解析为 `Target`。

## 1. Dataclass 节点

当 `TrainerConfig` 字段声明为 dataclass 时，对应的 YAML mapping 按该 dataclass 的字段解析，不写 `_target_`：

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

上例中的两个 mapping 分别解析为 `TrainingConfig` 和 `AcceleratorConfig`。解析器按照 dataclass 定义：

- 校验 YAML key 是否对应字段。
- 为缺省字段填入默认值，并检查无默认值的必填字段。
- 根据字段类型转换标量和容器，包括 `Optional`、`Literal`、list、tuple 和嵌套 dataclass。
- 在类型或字段错误中保留完整路径，例如 `$.accelerator.tp_size: expected int, got str`。

外层字段的类型注解确定嵌套 mapping 对应的 dataclass。因此根 YAML 只接受 `TrainerConfig` 声明的字段，嵌套配置也只接受对应 dataclass 声明的字段。

## 2. Target 节点

[`Target`](../../../hyper_parallel/trainer/config/target.py#L54) 是一个延迟调用 Python 类或函数的配置对象。它保存导入后的目标对象、原始导入路径和 YAML 参数；调用 `build()` 时，再合并 Trainer 提供的运行时参数并创建组件。

以模型配置为例：

```yaml
model:
  _target_: hyper_parallel.models.HyperAutoModelForCausalLM.from_pretrained
  pretrained_model_name_or_path: Qwen/Qwen3-30B-A3B
  ignore_mismatched_sizes: true
  torch_dtype: bfloat16
  attn_implementation: sdpa
  force_hf: true
```

框架完成对于配置的解析 [`_resolve_target()`](../../../hyper_parallel/trainer/config/resolver.py#L347-L382)：Resolver 导入 `_target_` 指向的对象，校验并规范化 YAML 参数，然后构造：

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

`normalized_args` 包含 YAML 参数和 [`HyperAutoModelForCausalLM.from_pretrained()`](../../../hyper_parallel/models/_transformers/auto_model.py#L77-L152) 的默认参数。此时只生成配置对象，不调用 `from_pretrained`，也不加载模型权重。

[`BaseTrainer._build_model()`](../../../hyper_parallel/trainer/base.py#L285-L314) 在分布式环境初始化完成后构建模型：

```python
self.model = self.config.model.build(
    distributed_setup=self.distributed_setup,
    peft_config=self.peft_config,
    activation_checkpoint=self.config.activation_checkpoint.mode,
    activation_swap=self.config.activation_swap,
    compile_config=self.config.compile,
)
```

将 YAML 参数与 Trainer 提供的运行时参数合并后，这一构建过程等价于调用 `HyperAutoModelForCausalLM.from_pretrained()` 函数：

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

[`TrainerConfig`](../../../hyper_parallel/trainer/config/trainer.py#L47-L105) 通过字段类型声明各配置节点的解析方式：`Target[...]` 字段解析为 `Target`，dataclass 字段解析为对应的 dataclass。

解析 `Target` 时会检查：

1. `_target_` 的路径能否导入，以及导入结果是否可调用。
2. YAML 中的每个参数名能否作为关键字参数传给目标对象。
3. 目标参数存在类型注解时，YAML 值能否按该类型校验和转换。

解析器还会补入目标函数的默认参数。由 Trainer 在构建阶段提供的必填参数不需要写入 YAML。

基于 `_target_` 接入外部模型和训练组件的方法见 [AutoModels 二次开发指南](custom_component.md)。

## 3. CLI Dotted Override

CLI override 在 YAML 已经解析为 `TrainerConfig` 后执行，使用 `--field.path=value` 定位并替换配置树中的值：

```bash
--training.train_iters=100
--model.pretrained_model_name_or_path=/path/to/model
--optimizer.adamw_config.adamw_lr=2.0e-5
--profiling.enabled=true
```

Value 先由 `yaml.safe_load()` 解析，再按照目标节点的类型转换：

```bash
--optimizer.no_decay_params='[bias, norm, ln_]'
--compile.options='{trace.enabled: true}'
--profiling.enabled=true
--checkpoint.restore_from=null
```

路径替换遵循配置树本身的结构：

- 参数使用 `--field.path=value`，不接受分离的 `--field.path value`。
- Dataclass 路径必须对应已有字段。
- 普通 Mapping 只能修改已有 key。
- `Target` 参数必须已在配置中，或存在于 callable 签名中；带 `**kwargs` 的 callable 可以接收新增参数。
- `_target_` 不参与替换，CLI override 不改变 callable。
- 当前值为 `None` 的可选组件没有可继续遍历的配置节点，组件结构需要由 YAML 提供。
