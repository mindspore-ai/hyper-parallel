# CLI Override 使用指南

CLI override 用于在不修改 YAML 的情况下调整一次实验的少量参数。基础配置仍由 YAML 提供，override 只修改解析后已经存在的 Config 字段。

## 1. 命令格式

每个 override 使用 `--字段路径=值`：

```bash
python -c \
  'from hyper_models.config.manager import parse_training_args; print(parse_training_args())' \
  configs/qwen3_5.yaml \
  --training.max_steps=200 \
  --accelerator.tp_size=4 \
  --optimizer.lr=0.0003
```

字段路径从 `TrainerConfig` 根节点开始。`accelerator.tp_size` 对应 `config.accelerator.tp_size`，`optimizer.lr` 对应 `config.optimizer.lr`。

当前接口只接受带等号的写法：

```text
--accelerator.tp_size=4
```

下面的写法不会被接受：

```text
--accelerator.tp_size 4
```

## 2. 值的类型

解析器根据目标字段的类型处理值：

| 命令行值 | 目标字段 | 解析结果 |
| --- | --- | --- |
| `4` | `int` | `4` |
| `0.0003` | `float` | `0.0003` |
| `true` | `bool` | `True` |
| `null` | `Optional` | `None` |
| `[2, 2]` | `list[int]` | `[2, 2]` |

不符合目标类型的值会直接报错：

```text
--accelerator.tp_size=abc
```

## 3. 可修改范围

override 只能修改 YAML 已经选择的组件。例如 YAML 中存在：

```yaml
optimizer:
  _target_: hyper_models.components.optim.AdamW.Config
  lr: 0.0002
```

才可以使用：

```text
--optimizer.lr=0.0003
```

如果 YAML 没有选择 optimizer，`config.optimizer` 为 `None`，CLI 不能用字段覆盖临时创建一个 optimizer。切换组件类型需要修改 YAML 中的 `_target_`。

完整 YAML 写法见 [YAML Trainer 配置使用指南](yaml_config.md)。
