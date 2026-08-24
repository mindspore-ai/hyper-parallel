# YAML Trainer 配置结构

HyperModels 配置层将 YAML 根 mapping 解析为带类型的 `TrainerConfig`。根节点的预期类型固定为
`TrainerConfig`，其字段类型继续决定每个子节点的解析方式。

```text
YAML mapping
  -> resolve_root(expected_type=TrainerConfig)
  -> typed TrainerConfig tree
  -> apply CLI dotted overrides
  -> final TrainerConfig
```

配置树包含两类节点：保存参数的 dataclass，以及保存 callable 和调用参数的 `Target`。

## 1. Dataclass 节点

预期类型为 dataclass 时，YAML mapping 直接解析为该类型，不写 `_target_`：

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
- 在类型或字段错误中保留完整路径，例如 `$.accelerator.tp_size`。

Dataclass 字段的预期类型来自外层类型注解。根节点因此只接受 `TrainerConfig` 声明的字段，嵌套 dataclass 也按其
自身字段继续解析。

## 2. Target 节点

预期类型为 `Target` 时，节点必须提供 `_target_`。它是 Python 类、函数或其他 callable 的 dotted import path；
其余 key 是调用该 callable 的配置参数：

```yaml
model:
  _target_: hyper_parallel.auto_models._transformers.HyperAutoModelForCausalLM.from_pretrained
  pretrained_model_name_or_path: Qwen/Qwen3-30B-A3B
  torch_dtype: bfloat16
  attn_implementation: sdpa
  force_hf: true

optimizer:
  _target_: hyper_parallel.auto_models.components.optim.optimizer.AdamW
  no_decay_params: [bias, norm, ln_]
  adamw_config:
    adamw_lr: 1.0e-5
    adamw_weight_decay: 0.01
    adamw_betas: [0.9, 0.95]
    adamw_eps: 1.0e-8
```

解析器按以下顺序构造 `Target`：

1. 导入 `_target_` 指向的对象并检查其可调用性。
2. 检查 callable 是否支持 keyword 调用，以及配置参数是否存在于签名中。
3. 根据 callable 参数的类型注解转换配置值；例如带 dataclass 注解的参数会继续解析为 dataclass。
4. 保存 callable、原始导入路径和已经解析的 keyword 参数。

解析阶段不调用 callable。运行阶段调用 `Target.build(**runtime_kwargs)` 时：

1. 若 callable 没有 `**kwargs`，先丢弃其签名不接受的运行时参数。
2. 将 YAML 参数与运行时参数合并；同名参数以运行时值为准。
3. 以 keyword arguments 调用保存的 callable，并返回运行对象。

`_target_` 的识别同样由预期类型驱动。普通 Mapping 中即使存在 `_target_`，也不会自动解析为 `Target`；只有外层
字段或 callable 参数的类型注解要求 `Target` 时，才进入这条解析路径。

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

Dataclass 字段和带类型注解的 `Target` 参数执行类型转换；Mapping、`Any`、无注解参数和 `**kwargs` 参数保留
`yaml.safe_load()` 产生的值。完整规则见 [CLI Override 使用指南](cli_override.md)。

## 4. 解析结果与运行边界

Resolver 输出完整的 `TrainerConfig` 数据树。它校验 YAML 结构、导入路径、参数名和可静态确定的类型，并在错误中
报告对应配置路径：

```text
$.accelerator.tp_size: expected int, got str
```

组件实例、分布式环境、模型权重和优化器都在 Trainer 消费配置后创建。依赖运行时对象才能检查的参数，以及
callable 自身的运行约束，在对应的 `Target.build(...)` 调用阶段报告。
