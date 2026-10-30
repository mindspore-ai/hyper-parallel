# Codegen 使用指南

Codegen 可以根据训练 YAML 配置生成加入并行策略、模块替换patch后的模型代码，产物仅用于检查和交付生成结果，不会被普通训练自动加载。训练始终使用原生
模型构建和并行化流程。

## 准备环境

请先以 editable 方式安装当前代码，并确认 Python 导入的是当前工作树：

```bash
pip install -e .
python -c "import hyper_parallel; print(hyper_parallel.__file__)"
```

生成时使用普通训练 YAML。YAML 中不要添加 `codegen: true`。

YAML 内引用的模型配置、权重或其他本地资产必须可以正常访问。Codegen 会按照 YAML
中的并行配置构建原生模型，因此启动卡数应与配置中的并行规模一致。

## 生成 Codegen 产物

### 多卡生成

以 8 卡任务为例：

```bash
torchrun --standalone --nproc_per_node=8 \
  -m hyper_parallel.codegen --verbose generate \
  --config path/to/train.yaml
```

该命令只完成模型构建和产物生成，成功后立即退出，不会开始训练。

如果配置只需要单进程，也可以直接运行：

```bash
python -m hyper_parallel.codegen --verbose generate \
  --config path/to/train.yaml
```

### 指定输出目录

```bash
torchrun --standalone --nproc_per_node=8 \
  -m hyper_parallel.codegen --verbose generate \
  --config path/to/train.yaml \
  --output /path/to/generated
```

不指定 `--output` 时，已注册模型默认写入：

```text
hyper_parallel/models/<model_family>/generated/
```

例如：

```text
hyper_parallel/models/qwen3_moe/generated/
hyper_parallel/models/deepseek_v41/generated/
```

无法识别模型 family 时，产物写入 YAML 所在目录下的 `generated/`。

### 使用配置覆盖参数

CLI 支持与训练配置相同的 dotted override：

```bash
torchrun --standalone --nproc_per_node=8 \
  -m hyper_parallel.codegen generate \
  --config path/to/train.yaml \
  --accelerator.tp_size=2 \
  --accelerator.ep_size=4
```

覆盖后的并行规模必须与实际启动的进程数匹配。

## 检查生成产物

生成完成后，使用相同 YAML 执行检查：

```bash
python -m hyper_parallel.codegen check \
  --config path/to/train.yaml
```

检查成功时命令返回 `0`；检查失败时返回非零状态，并输出具体原因。

检查目录下的全部 YAML：

```bash
python -m hyper_parallel.codegen check \
  --all \
  --root path/to/configs
```

`check --all` 会跳过 `generated`、`.git`、虚拟环境和临时目录。

检查 Transformers 版本约束：

```bash
python -m hyper_parallel.codegen check \
  --transformers-pin '==5.15.0'
```

## 生成产物说明

一个典型 bundle 包含：

```text
generated/
├── __init__.py
├── codegen_meta.json
├── modeling_<Model>_gen_npu.py
└── modeling_<Model>_gen_npu.py.diff
```

如果原始模型使用 remote code，目录中还可能包含它依赖的 Python 源码文件。

各文件用途：

- `codegen_meta.json`：模型来源、并行计划、签名和输出文件校验信息。
- `modeling_*_gen_npu.py`：生成后的 modeling 源码。
- `modeling_*_gen_npu.py.diff`：生成源码相对于原始源码的差异。
- `__init__.py`：标识该目录为 Codegen bundle。

对于不需要生成并行代码的配置，Codegen 会生成 audit-only bundle。此时目录可能只包含
`codegen_meta.json` 和 `__init__.py`，属于正常结果。

## 复用与重新生成

相同模型、配置、源码和并行计划会得到相同签名。已有产物与当前签名一致时，`generate`
会直接复用，不会重写文件。

以下变化通常会触发重新生成：

- 模型源码变化；
- replacement 或并行策略变化；
- TP、CP、EP、PP 等并行配置变化；
- 原生构建得到的实际并行计划变化。

需要强制 fresh generation 时，请先备份并移走对应的 `generated/` 目录，再重新运行
`generate`。不要只删除目录中的单个文件。

## 与训练配合使用

生成和训练是两个独立操作：

```bash
# 1. 单独生成并检查产物
torchrun --standalone --nproc_per_node=8 \
  -m hyper_parallel.codegen generate \
  --config path/to/train.yaml

python -m hyper_parallel.codegen check \
  --config path/to/train.yaml

# 2. 按项目原有方式启动普通训练
torchrun --standalone --nproc_per_node=8 path/to/train_entry.py \
  path/to/train.yaml
```

普通训练不会读取或执行刚生成的 bundle。删除、损坏或过期的 Codegen 产物也不应改变
原生训练行为。

## 从旧配置迁移

训练侧 Codegen 开关和 generated backend 已删除。请进行以下迁移：

- 从 YAML 删除 `codegen: true`；
- 删除 AutoModel 调用中的 `codegen`、`codegen_config`、
  `codegen_artifact_dir` 和 `codegen_session` 参数；
- 删除 `backend="gen"`；
- 使用 `python -m hyper_parallel.codegen generate` 单独生成产物；
- 使用 `python -m hyper_parallel.codegen check` 检查产物。

旧参数不会静默降级，会直接提示迁移错误。

## 常见问题

### 生成时长时间等待或出现通信错误

确认以下条件：

- 启动进程数与 YAML 并行规模一致；
- 所有 rank 使用相同 YAML、源码和本地资产；
- 设备未被其他任务占用；
- 上一次失败的 `torchrun` 进程已经退出；
- rendezvous 端口没有被其他任务占用。

### `check` 提示找不到 `codegen_meta.json`

先确认 `generate` 已成功完成，并使用与生成时相同的 YAML 和 CLI override。若生成时使用
了 `--output`，请直接确认该目录中的 bundle；当前 `check --config` 默认按模型 family
或 YAML 位置解析标准输出目录。

### `check` 提示 hash drift

生成后的文件被修改、缺失，或者 bundle 中混入了旧文件。请备份并移走整个
`generated/` 目录，然后重新执行 fresh generation。

### 生成成功但没有 modeling 文件

检查 `codegen_meta.json` 中的状态。如果状态为 `not_required`，说明当前配置不需要生成
并行 modeling 源码，audit-only bundle 是预期结果。

### 生成是否需要模型权重

Codegen 复用原生模型构建入口。是否读取权重取决于 YAML 使用的 model target 及其配置。
请确保该 target 在当前环境中能够正常完成模型构建。
