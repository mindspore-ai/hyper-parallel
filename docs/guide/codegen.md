# Codegen 使用指南

Codegen 根据训练 YAML 中的模型和并行配置生成模型代码及差异文件，方便检查并行配置对模型代码的影响。
生成过程独立于训练，可在启动训练前单独完成。

## 准备环境

在 HyperParallel 仓库根目录安装当前代码，并确认 Python 导入的是当前工作树：

```bash
pip install -e .
python -c "import hyper_parallel; print(hyper_parallel.__file__)"
```

准备一份可正常解析的训练 YAML。YAML 引用的模型配置、模型源码和其他结构文件需要在当前机器上可访问。生成过程不会加载训练数据、优化器或模型权重。

## 生成代码

使用 Python 单进程运行：

```bash
python -m hyper_parallel.codegen --verbose generate \
  --config path/to/train.yaml
```

Codegen 会读取 YAML 中的模型结构、模块替换和 TP、CP、EP 等并行配置。

生成成功后，命令返回 `0`；失败时返回非零状态，并在日志中说明原因。

### 指定输出目录

```bash
python -m hyper_parallel.codegen --verbose generate \
  --config path/to/train.yaml \
  --output /path/to/generated
```

未指定 `--output` 时，已支持的模型通常写入：

```text
hyper_parallel/models/<model_family>/generated/
```

例如：

```text
hyper_parallel/models/qwen3_moe/generated/
hyper_parallel/models/deepseek_v41/generated/
```

如果无法识别模型类型，则写入训练 YAML 所在目录下的 `generated/`。

## 检查生成结果

生成完成后，使用相同的 YAML 和配置覆盖参数执行检查：

```bash
python -m hyper_parallel.codegen check \
  --config path/to/train.yaml
```

检查成功时返回 `0`；文件缺失、内容被修改或配置不匹配时返回非零状态。


## 产物说明

典型生成目录如下：

```text
generated/
├── __init__.py
├── codegen_meta.json
├── modeling_<Model>_gen_npu.py
└── modeling_<Model>_gen_npu.py.diff
```

- `modeling_*_gen_npu.py`：生成后的模型源码。
- `modeling_*_gen_npu.py.diff`：生成源码相对于原始源码的差异。
- `codegen_meta.json`：本次生成使用的模型来源、并行配置和文件校验信息。
- `__init__.py`：生成目录的 Python 包标识。

某些配置不需要改写模型源码，此时生成目录可能只有配置记录和包文件。这并不一定是错误，可查看 `codegen_meta.json` 和命令日志确认结果。

## 复用已有结果

模型源码、模型结构和并行配置均未变化时，Codegen 会复用已有结果。以下变化会触发重新生成：

- 模型源码或模型 config 变化；
- 模块替换规则变化；
- TP、CP、EP 等并行配置变化；
- 生成器版本或产物格式变化。

需要重新生成时，建议先备份并移走整个 `generated/` 目录，再运行 `generate`。不要只删除目录中的单个文件，以免新旧产物混在一起。

## 与训练的关系

生成代码和训练是两个独立操作。先运行 Codegen，再按项目现有方式启动训练即可：

```bash
# 生成并检查代码，不占用训练设备
python -m hyper_parallel.codegen generate \
  --config path/to/train.yaml

# 启动训练；具体命令以项目的训练入口为准
torchrun --standalone --nproc_per_node=8 path/to/train_entry.py \
  path/to/train.yaml
```

普通训练不会读取或执行 Codegen 产物。生成目录缺失、损坏或过期也不会改变训练行为。

## 常见问题

### 生成需要 NPU/GPU 吗？

不需要。Codegen 使用单个 CPU 进程读取模型结构，不会初始化 NPU/GPU 通信，也不会创建分布式进程组。

### 生成需要使用 `torchrun` 吗？

不需要。请直接使用 `python -m hyper_parallel.codegen generate`。如果使用 `torchrun`，命令会提示改用单进程方式，避免多个进程同时写入同一目录。

### 生成需要模型权重或训练数据吗？

不需要加载模型权重或训练数据。模型配置、模型源码，以及构建模型结构时引用的文件仍需在当前机器上可访问。

### `check` 提示找不到 `codegen_meta.json`

确认 `generate` 已成功完成，并且检查时使用了相同的 YAML 和配置覆盖参数。如果生成时指定了 `--output`，请确认该目录存在且包含完整产物。

### `check` 提示 hash drift

生成后的文件被修改、缺失，或者目录中混入了旧文件。请先备份并移走整个 `generated/` 目录，然后重新运行 `generate`。

### 生成成功但没有 modeling 文件

当前配置可能不需要改写模型源码。查看命令日志和 `codegen_meta.json`；如果生成和检查命令都成功，该结果可以正常使用。
