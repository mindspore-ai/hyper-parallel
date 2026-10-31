# Codegen 使用指南

Codegen 根据模型配置和并行策略生成可审阅的模型源码。生成过程与训练分离：无需启动训练，
也不需要使用 `torchrun`。

## 准备环境

在 HyperParallel 仓库根目录安装当前代码，并确认 Python 导入的是当前工作树：

```bash
pip install -e .
python -c "import hyper_parallel; print(hyper_parallel.__file__)"
```

生成时需要能够访问 YAML 引用的模型 config、模型源码及其结构文件，但不会加载模型权重、
训练数据或优化器状态。

仓库提供的 Codegen 配置统一位于模型自己的 `recipes` 目录：

```text
hyper_parallel/models/<model_name>/recipes/<model_name>_codegen.yaml
```

例如：

```text
hyper_parallel/models/kimi_k3/recipes/kimi_k3_codegen.yaml
hyper_parallel/models/deepseek_v32/recipes/deepseek_v32_codegen.yaml
hyper_parallel/models/qwen3_5/recipes/qwen3_5_codegen.yaml
```

## 生成模型代码

使用单个 Python 进程运行：

```bash
python -m hyper_parallel.codegen --verbose generate \
  --config hyper_parallel/models/kimi_k3/recipes/kimi_k3_codegen.yaml
```

Codegen 会读取 YAML 中的模型结构和 TP、CP、EP 等并行配置，在 meta device 上构造模型并
生成代码。命令成功时返回 `0`，失败时返回非零状态并在日志中说明原因。

不要使用 `torchrun` 启动 Codegen。Codegen 不创建分布式进程组，也不占用 NPU/GPU 进行训练。

### 指定输出目录

```bash
python -m hyper_parallel.codegen --verbose generate \
  --config path/to/model_codegen.yaml \
  --output /path/to/generated
```

未指定 `--output` 时，已支持的模型写入：

```text
hyper_parallel/models/<model_name>/generated/
```

## 检查生成结果

使用生成时相同的 YAML 执行检查：

```bash
python -m hyper_parallel.codegen check \
  --config hyper_parallel/models/kimi_k3/recipes/kimi_k3_codegen.yaml
```

检查会验证产物是否完整、源码是否可编译、文件哈希和生成签名是否与当前配置一致。检查成功
返回 `0`；文件缺失、内容被修改或配置发生变化时返回非零状态。

检查仓库内发现的全部 Codegen 配置：

```bash
python -m hyper_parallel.codegen check --all --root .
```

## 产物说明

典型产物目录如下：

```text
generated/
├── __init__.py
├── codegen_meta.json
├── modeling_<Model>_gen_npu.py
└── modeling_<Model>_gen_npu.py.diff
```

- `modeling_*_gen_npu.py`：应用并行策略后的模型源码，也可直接作为训练入口。
- `modeling_*_gen_npu.py.diff`：生成源码相对于原始源码的差异，便于代码审阅。
- `codegen_meta.json`：模型来源、并行配置、生成签名和文件校验信息。
- `__init__.py`：生成目录的 Python 包标识。

如果模型的官方 remote code 本身由多个 modeling 文件组成，Codegen 会保留这种结构。例如
Kimi K3 的多模态入口和语言模型分别位于两个 modeling 文件，因此会生成两份对应源码和 diff。
这些文件共同组成一个 Kimi K3 bundle，并不是两个独立模型。

Bundle 还可能包含 `configuration_*.py` 等官方源码依赖。生成后的 modeling 文件使用 bundle 内
相对导入，可以在新的 Python 进程中加载，不依赖 Hugging Face 动态模块缓存中的包名。

## 修改并行配置

如需尝试其他并行组合，复制模型目录中的 Codegen recipe，再调整 TP、CP、EP、PP、FSDP 及
`plan_overrides`。建议让文件名继续以 `_codegen.yaml` 结尾，便于识别和批量检查。

模型源码、模型 config、并行策略或生成器版本变化后，应重新运行 `generate`。Codegen 会根据
生成签名复用一致的产物；签名变化时会原子更新整个 bundle，避免新旧文件混合。

## 与训练的关系

Codegen 和训练是独立操作。普通 Trainer 不读取或执行 `generated/` 中的文件，生成目录缺失、
损坏或过期也不会改变原生训练流程。

如果需要训练生成后的模型，直接执行该模型 bundle 中的主 `*_gen_npu.py` 文件，并传入普通训练
YAML。YAML 不负责定位 `generated/`，也不需要增加 Codegen 专属字段；训练入口就是当前执行的
模型文件。例如单进程启动：

```bash
python hyper_parallel/models/deepseek_v41/generated/modeling_Deepseek_V4_gen_npu.py \
  path/to/train.yaml
```

多卡训练仍使用标准 `torchrun`：

```bash
torchrun --standalone --nproc_per_node=8 \
  hyper_parallel/models/deepseek_v41/generated/modeling_Deepseek_V4_gen_npu.py \
  path/to/train.yaml
```

生成文件只提供已经写入源码的模型结构和 forward 改写。设备 mesh、实时并行计划、参数分片、
FSDP、PP、优化器和数据流程仍由 HyperParallel 原生 Trainer 根据训练 YAML 构建。
`codegen_meta.json` 用于记录生成来源和校验产物，不作为训练计划输入。

## 常见问题

### 是否需要 NPU/GPU？

不需要。Codegen 使用 CPU/meta 模型收集结构和并行计划，不会执行训练计算或建立 HCCL/NCCL
通信。

### 是否需要模型权重或训练数据？

不需要加载模型权重和训练数据。模型 config、模型源码以及构造模型时引用的结构文件必须可访问。

### `check` 提示找不到 `codegen_meta.json`

确认 `generate` 已成功完成，并且检查时使用了相同的 YAML 和配置覆盖参数。如果生成时指定了
`--output`，检查和产物管理也应使用对应目录。

### `check` 提示 hash drift

产物文件可能被修改、缺失，或者目录中混入了旧文件。请重新运行 `generate`，不要手工修改
generated 文件。

### 为什么一个模型有多个 modeling 文件？

部分官方模型将顶层模型和语言模型拆在不同源码文件中。Codegen 保留官方依赖关系，并对每个
被修改的 modeling 文件生成对应 diff；它们仍属于同一个模型 bundle。
