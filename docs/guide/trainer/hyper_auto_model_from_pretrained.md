# HyperAutoModel `from_pretrained` 使用指南

本文介绍 `hyper_parallel.auto_models._transformers.auto_model` 中 `from_pretrained` 接口的功能、参数、模型加载流程，
以及如何在不使用 Trainer 的情况下直接构造一个可训练的分布式模型。

## 1. 功能概述

`HyperAutoModel*.from_pretrained()` 是 HyperParallel 提供的 Hugging Face 兼容模型加载入口。它不只是调用
Transformers 加载权重，还会把以下步骤组合成一次原子操作：

1. 读取 Hugging Face 模型配置；
2. 选择 HyperParallel 自定义实现或 Hugging Face 原生实现；
3. 在需要时使用 Meta Device 创建空模型；
4. 应用 Pipeline、TP、CP、EP 等模型并行布局；
5. 应用 `torch.compile` 和 FSDP2；
6. 把预训练权重加载到最终的分片参数中；
7. 校验缺失权重、意外权重和 tied weights。

因此，在分布式训练场景中，该接口应当被视为一个完整的：

```text
构造模型 -> 应用并行策略 -> 物化参数 -> 加载预训练权重
```

入口。它返回的模型可以直接参与 forward、backward 和优化器构造，不需要再次调用 Trainer 侧的模型并行化接口。

当前提供以下三个入口类：

| 类 | 使用场景 |
| --- | --- |
| `HyperAutoModelForCausalLM` | 因果语言模型和大语言模型训练 |
| `HyperAutoModelForImageTextToText` | 图文到文本、多模态生成模型 |
| `HyperAutoModelForSequenceClassification` | 序列分类模型 |

可以从 `_transformers` 包直接导入：

```python
from hyper_parallel.auto_models._transformers import (
    HyperAutoModelForCausalLM,
    HyperAutoModelForImageTextToText,
    HyperAutoModelForSequenceClassification,
)
```

## 2. 接口定义

```python
from_pretrained(
    pretrained_model_name_or_path: str,
    *model_args,
    distributed_setup=None,
    backend=None,
    peft_config=None,
    torch_dtype="auto",
    attn_implementation="sdpa",
    force_hf=False,
    validate_placement=False,
    qat_config=None,
    fp8_config=None,
    compile_config=None,
    freeze_config=None,
    **kwargs,
)
```

返回值是已经完成设备放置、并行化和权重加载的 `transformers.PreTrainedModel`。

接口返回前会调用 `model.train()`。如果模型用于推理，需要显式调用：

```python
model.eval()
```

## 3. 单卡使用

单卡或不需要模型分片时，可以省略 `distributed_setup`：

```python
import torch

from hyper_parallel.auto_models._transformers import HyperAutoModelForCausalLM


model = HyperAutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-30B-A3B",
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa",
    force_hf=True,
)
```

也可以加载本地 Hugging Face 模型目录：

```python
model = HyperAutoModelForCausalLM.from_pretrained(
    "./models/my_model",
    torch_dtype="auto",
    local_files_only=True,
    force_hf=True,
)
```

本地目录需要包含可被 `AutoConfig.from_pretrained()` 识别的 `config.json` 和对应权重。

## 4. 直接构造分布式模型

### 4.1 参数传递方式

当前接口把 `distributed_setup` 定义为关键字参数。因此，概念上的：

```python
model = ModelClass.from_pretrained(tgt_model_name, parallelize_config)
```

在当前版本中应写成：

```python
model = ModelClass.from_pretrained(
    tgt_model_name,
    distributed_setup=parallelize_config,
)
```

不能把 `parallelize_config` 作为第二个位置参数传入，因为位置参数会进入底层模型的 `model_args`，不会被解释为
HyperParallel 的并行配置。

这里的 `parallelize_config` 应当是一个已经根据当前进程组构造完成的 `DistributedSetup`，而不是普通字典。

### 4.2 可运行的 FSDP 示例

下面的示例不依赖 HyperModels Trainer。每个进程独立执行同一段代码，程序会完成：

1. 初始化 HCCL/NCCL 进程组和当前设备；
2. 根据 `world_size` 创建纯数据并行拓扑；
3. 使用整个数据并行域作为 FSDP shard 域；
4. 在 Meta Device 构造模型；
5. 包装 FSDP2 后直接把 Safetensors 权重加载到各 rank 的最终分片；
6. 返回可以直接 forward/backward 的分布式模型。

```python
import argparse
from types import SimpleNamespace

import torch
import torch.distributed as dist
from transformers import AutoTokenizer

from hyper_parallel.auto_models._transformers import HyperAutoModelForCausalLM
from hyper_parallel.auto_models.components.distributed.config import FSDP2Config
from hyper_parallel.auto_models.components.distributed.infrastructure import (
    DistributedSetup,
    create_distributed_setup_from_config,
    destroy_process_group,
    initialize_distributed,
)
from hyper_parallel.auto_models.trainer.config import AcceleratorConfig


def build_parallelize_config() -> DistributedSetup:
    """Build a pure-FSDP distributed setup for the current process group."""
    world_size = dist.get_world_size()
    config = SimpleNamespace(
        accelerator=AcceleratorConfig(
            tp_size=1,
            cp_size=1,
            ep_size=1,
            pp_size=1,
            sequence_parallel=False,
            loss_parallel=False,
        ),
        fsdp_config=FSDP2Config(
            dp_shard_size=world_size,
            reshard_after_forward=True,
            requires_grad_sync=True,
        ),
        plan_overrides=[],
    )
    return create_distributed_setup_from_config(config)


def main() -> None:
    """Build a distributed pretrained model and run one training forward."""
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name_or_path")
    args = parser.parse_args()

    initialize_distributed(backend="hccl")
    parallelize_config = build_parallelize_config()

    model = HyperAutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        distributed_setup=parallelize_config,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        force_hf=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    batch = tokenizer("HyperParallel distributed training", return_tensors="pt")
    model_device = next(model.parameters()).device
    input_ids = batch["input_ids"].to(model_device)

    outputs = model(
        input_ids=input_ids,
        labels=input_ids,
    )
    outputs.loss.backward()

    destroy_process_group()


if __name__ == "__main__":
    main()
```

将代码保存为 `build_distributed_model.py`，在 8 张 Ascend NPU 上启动：

```bash
torchrun \
    --nproc_per_node=8 \
    --module build_distributed_model \
    /path/to/huggingface-model
```

也可以传入 Hugging Face Hub 模型 ID：

```bash
torchrun \
    --nproc_per_node=8 \
    --module build_distributed_model \
    Qwen/Qwen3-30B-A3B
```

如果使用 NVIDIA GPU，把示例中的初始化改为：

```python
initialize_distributed(backend="nccl")
```

上述示例是纯 FSDP 配置，因此 `dp_shard_size=world_size`。加入 TP、CP 或 PP 后，数据并行大小不再等于
`world_size`，需要满足：

```text
dp_size * cp_size * tp_size * pp_size = world_size
```

以及：

```text
dp_replicate_size * dp_shard_size = dp_size * cp_size
```

开启 TP、CP 或 EP 时，还需要为目标模型提供能够被 `ShardingPlanner` 识别的布局，以及必要的
`plan_overrides` 计算区域配置。不要只修改 `tp_size`、`cp_size` 或 `ep_size` 就假定模型已经能够正确并行运行。

## 5. Trainer YAML 使用方式

训练任务更推荐通过 HyperModels Trainer 使用。Trainer 会负责初始化进程组、创建 DeviceMesh，并在模型构造时自动
注入 `distributed_setup`：

```yaml
model:
  _target_: hyper_parallel.auto_models._transformers.HyperAutoModelForCausalLM.from_pretrained
  pretrained_model_name_or_path: "Qwen/Qwen3-30B-A3B"
  torch_dtype: bfloat16
  attn_implementation: sdpa
  force_hf: true

training:
  backend: hccl
  init_device: meta

accelerator:
  tp_size: 1
  cp_size: 1
  ep_size: 1
  pp_size: 1
  sequence_parallel: false
  loss_parallel: false

fsdp_config:
  dp_shard_size: 8
  replicate_params: []
  reshard_after_forward: true
  requires_grad_sync: true
```

完整示例见 `examples/training_demo/train.yaml`。

## 6. 参数说明

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `pretrained_model_name_or_path` | 必填 | Hugging Face Hub 仓库 ID 或本地模型目录 |
| `distributed_setup` | `None` | 已构造的 DeviceMesh、FSDP 和模型并行策略；Trainer 会自动注入 |
| `torch_dtype` | `"auto"` | 参数类型，例如 `torch.bfloat16`、`"bfloat16"` 或 `"auto"` |
| `attn_implementation` | `"sdpa"` | Attention 实现，例如 `"sdpa"`、`"eager"` 或 `"flash_attention_2"` |
| `force_hf` | `False` | 是否强制使用 Hugging Face 原生模型实现 |
| `validate_placement` | `False` | 应用 TP/CP/EP 分片计划时是否校验输入输出 Placement |
| `compile_config` | `None` | 非 `None` 时作为 `torch.compile(model, **compile_config)` 的参数 |
| `peft_config` | `None` | PEFT 配置；当前分支尚未真正实现 |
| `qat_config` | `None` | QAT 配置；当前分支尚未真正实现 |
| `fp8_config` | `None` | FP8 配置；当前分支尚未真正实现 |
| `freeze_config` | `None` | 参数冻结配置；当前分支尚未真正实现 |
| `backend` | `None` | 预留参数；当前加载链路没有实际使用 |
| `model_args` | 无 | 传递给底层模型构造函数的额外位置参数 |
| `kwargs` | 无 | 传递给 `AutoConfig.from_pretrained()` 和底层模型加载接口的额外参数 |

常用 Hugging Face 参数可以通过 `kwargs` 传入：

```python
model = HyperAutoModelForCausalLM.from_pretrained(
    "organization/model-name",
    revision="main",
    cache_dir="/path/to/cache",
    local_files_only=False,
    trust_remote_code=True,
    token="...",
    torch_dtype=torch.bfloat16,
)
```

`trust_remote_code=True` 会执行模型仓库中的远程代码，只应对可信仓库启用。

## 7. 模型实现选择

接口读取 `config.architectures[0]`，然后查询 HyperParallel 模型注册表：

```text
force_hf=True
  -> 始终使用 Hugging Face 原生实现

force_hf=False
  -> 注册且成功导入 HyperParallel 自定义模型：使用自定义实现
  -> 未注册或导入失败：回退到 Hugging Face 原生实现
```

当前注册表声明了以下架构：

- `LlamaForCausalLM`；
- `Qwen3_5ForCausalLM`；
- `Qwen3_5MoeForConditionalGeneration`；
- `Qwen3_5ForConditionalGeneration`；
- `Qwen3VLMoeForConditionalGeneration`。

当前分支中，注册表所指向的 `hyper_parallel.auto_models.components.models.*` 实现模块尚不存在。这些架构会记录 warning，
然后回退到 Hugging Face 原生实现。当前建议显式传入 `force_hf=True`，以清楚表达使用 HF 原生模型的意图。

## 8. 分布式权重加载

在多进程场景中，只要没有传入 `quantization_config`，接口会使用 Meta Device 延迟加载：

```text
Meta Device 创建空模型
  -> 应用 TP/CP/EP 分片计划
  -> 应用 FSDP2
  -> 在当前设备物化本 rank 参数
  -> 读取 checkpoint tensor
  -> 按目标 DTensor Layout 提取本地分片
  -> 复制到最终参数存储
```

这种加载顺序可以避免先在设备上构造完整模型再替换参数，并保持 FSDP/DTensor 包装后参数的对象身份不变。

### 8.1 支持的分布式权重格式

当前延迟加载器只支持 Safetensors：

```text
model.safetensors
```

或者：

```text
model.safetensors.index.json
model-00001-of-000xx.safetensors
...
```

分布式 Meta Device 路径目前不支持：

```text
pytorch_model.bin
pytorch_model.bin.index.json
```

从 Hugging Face Hub 加载时，延迟加载器也只会下载 `*.safetensors` 和对应索引。普通单进程、HF 原生、
非 Meta Device 的路径由 Transformers 自己加载，其格式支持范围由当前 Transformers 版本决定。

## 9. `compile_config`

传入字典后，接口会在参数布局应用之后、FSDP2 包装之前调用 `torch.compile`：

```python
model = HyperAutoModelForCausalLM.from_pretrained(
    "./models/my_model",
    torch_dtype=torch.bfloat16,
    force_hf=True,
    compile_config={
        "backend": "inductor",
        "dynamic": True,
    },
)
```

即使传入空字典也会启用编译。如果不需要编译，应保留默认值 `None`。

## 10. 当前限制和注意事项

- 该接口基于 PyTorch 和 Transformers，不是 MindSpore 模型加载入口；
- 返回模型始终处于训练模式，推理前需要调用 `model.eval()`；
- `peft_config`、`qat_config`、`fp8_config` 和 `freeze_config` 当前只会输出未实现警告；
- `backend` 当前是预留参数，没有实际效果；
- 自定义模型注册表已定义，但对应实现模块在当前分支缺失，实际会回退到 Hugging Face；
- 分布式延迟加载只支持 Safetensors；
- 不建议传入 Hugging Face `device_map`，设备放置和参数分片应交给 HyperParallel；
- 不应设置 `output_loading_info=True`，当前接口假定底层只返回一个 `PreTrainedModel`；
- `quantization_config` 会关闭 Meta Device 延迟加载路径，量化模型与 TP/FSDP 的组合需要单独验证；
- CP/EP 开启时通常需要通过 `plan_overrides` 提供模型相关的计算区域实现。

## 11. 内部实现流程

完整调用链可以概括为：

```text
from_pretrained
  -> 创建或读取 DistributedSetup
  -> 创建 ShardingPlanner / FSDP2Manager / AutoPipeline
  -> AutoConfig.from_pretrained
  -> 根据 architectures 和 force_hf 选择模型实现
  -> 在 Meta Device 或真实设备上构造模型
  -> Pipeline 切分
  -> 规划并应用 TP/CP/EP Layout
  -> torch.compile
  -> FSDP2 包装
  -> 物化参数并加载 Safetensors
  -> 校验 missing/unexpected/tied weights
  -> model.train()
```
