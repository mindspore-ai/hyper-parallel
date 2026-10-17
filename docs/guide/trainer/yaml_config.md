# YAML Trainer 配置结构

HyperParallel 模型训练按 [`TrainerConfig`](../../../hyper_parallel/trainer/config/trainer.py) 的字段定义解析 YAML。字段名决定 YAML 可以包含哪些配置项，字段类型决定对应的值如何解析。解析完成后得到 `TrainerConfig` 对象，随后通过 CLI dotted overrides 更新配置字段。

```python
@dataclass
class TrainerConfig:
    model: Target[Any]
    optimizer: OptimizerConfig
    training: TrainingConfig = field(default_factory=TrainingConfig)
    accelerator: AcceleratorConfig = field(default_factory=AcceleratorConfig)
    ...
```

配置节点主要分为三类：

| 类型 | 示例 | YAML 写法 |
|---|---|---|
| 普通 dataclass | `training`、`accelerator` | 按字段填写参数 |
| `Target` | `model`、`loss_fn`、`lr_scheduler` | 用 `_target_` 指定可导入的类或函数，并填写构建参数 |
| 包含 `Target` 的组件配置 | `optimizer`、`dataset`、`dataloader` | 在同一节点中填写 `_target_`、构建参数和组件专属字段 |

## 1. Dataclass 节点

`training`、`accelerator` 等普通 dataclass 节点按字段解析，不写 `_target_`：

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

根 YAML 只接受 `TrainerConfig` 声明的字段。普通 dataclass 节点按字段名校验；包含 `Target` 的组件节点还可以填写构建参数，详见下文。

## 2. Target 节点

[`Target`](../../../hyper_parallel/trainer/config/target.py) 是一个延迟调用 Python 类或函数的配置对象。它保存导入后的目标对象、原始导入路径和 YAML 参数；调用 `build()` 时，再合并 Trainer 提供的运行时参数并创建组件。

以模型配置为例：

```yaml
model:
  _target_: hyper_parallel.models.HyperAutoModelForCausalLM.from_pretrained
  pretrained_model_name_or_path: Qwen/Qwen3-30B-A3B
  torch_dtype: bfloat16
  attn_implementation: sdpa
  force_hf: true
```

[`Resolver`](../../../hyper_parallel/trainer/config/resolver.py) 导入 `_target_` 指向的对象并构造 `Target`，同时检查：

1. `_target_` 的路径能否导入，以及导入结果是否可调用。
2. YAML 中的每个参数名能否作为关键字参数传给目标对象。
3. 目标参数存在类型注解时，YAML 值能否按该类型校验和转换。

解析器会补入目标函数的默认参数，此时不调用模型构建函数，也不加载权重。

分布式环境初始化后，[`BaseTrainer._build_model()`](../../../hyper_parallel/trainer/base.py) 调用
`config.model.build(...)`，传入分布式配置、重计算策略等运行时参数。`Target.build()` 根据目标
函数签名筛选运行时参数，再与 YAML 参数合并；同名参数以运行时值为准。
各组件的运行时参数见 [二次开发指南](custom_component.md#34-其他可配置组件)，无需重复写入 YAML。

### 包含 Target 的组件配置

`optimizer` 解析为 [`OptimizerConfig`](../../../hyper_parallel/trainer/config/optimization.py)：
其中 `target` 保存优化器构建对象，`fp32_main_params` 和 `swap` 保存 Trainer 管理的策略。
这些字段在 YAML 中写在同一个 `optimizer` 节点下，例如：

```yaml
optimizer:
  _target_: hyper_parallel.components.optim.AdamW
  adamw_config:
    adamw_lr: 1.0e-5
  fp32_main_params: false
  swap:
    enabled: false
```

Resolver 将 `_target_` 和 `adamw_config` 归入 `target`，将其余两个字段解析为组件策略，
无需额外写 `optimizer.target`。使用 `fp32_main_params: true` 时，
`fsdp_config.mix_precision.reduce_dtype` 必须为 `float32`。

`dataset` 和 `dataloader` 同样将构建参数与组件专属字段写在同一节点下，具体字段见 [数据配置](../../../hyper_parallel/data/README.md)。

基于 `_target_` 接入外部模型和训练组件的方法见 [二次开发指南](custom_component.md)。

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
- 普通 dataclass 路径必须对应已有字段；包含 `Target` 的组件会将构建参数转交给内部 target，例如 `--optimizer.adamw_config.adamw_lr=2.0e-5`。
- 普通 Mapping 只能修改已有 key。
- `Target` 参数必须已在配置中，或存在于 callable 签名中；带 `**kwargs` 的 callable 可以接收新增参数。
- `_target_` 不参与替换，CLI override 不改变 callable。
- 当前值为 `None` 的可选组件没有可继续遍历的配置节点，组件结构需要由 YAML 提供。
