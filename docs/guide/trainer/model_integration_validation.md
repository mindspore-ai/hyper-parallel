# 模型接入验证使用指南

## 使用 Agent 开始接入验证

最少只需告诉 Agent 模型来源和可用设备。模型来源可以是本地路径、Hugging Face repo ID，
也可以是 `https://huggingface.co/...` URL。没有现成数据时，让 Agent 自动制作一份；并行
策略、序列长度等有明确要求时，再追加一句即可。

```text
请使用 hf-model-integration 和 hf-model-precision-validation，接入并验证
[模型名称]。权威实现/模型来源为 [本地路径、HF repo ID 或 HF URL]，
当前有 [卡数] 张 [NPU/GPU]。

如果没有数据，请自动制作小型、确定性的本地验证数据，并让所有 case 复用相同输入。
实现前请扫描仓库现有通用高性能模块、replacement、CP/EP 通信与已有模型 adapter，输出
能力匹配表和复用决策；不要重复实现已有 kernel 或直接依赖其他模型族的私有 adapter。
对每个声明为高性能的 replacement、CP 或 EP 路径，使用相同权重、输入、dtype、拓扑和
Trainer 路径完成替换前后精度对拍、目标设备模块微基准和同拓扑端到端 A/B；另行报告
CP/EP 扩展性。没有公平 A/B 证据时标记 PERFORMANCE_INCONCLUSIVE，不要宣称加速。
先构造覆盖全部关键层角色的裁剪模型做快速验证；如具备完整权重和足够资源，最后加载
完整规格权重做短程续训与断点恢复验证。请修复接入问题直至 PASS，无法继续则报告
BLOCKED。不要在框架中硬编码模型，也不要未经允许联网下载。最后给出实际运行的配置、
命令、结果、证据目录和未覆盖项。
```

例如只需追加：`验证 TP1/TP2、CP1/CP2、EP1/EP8，序列长度 4K。` 未指定时，Agent
会根据模型结构和实际设备生成最小但有区分度的验证矩阵。

如果完整规格权重已经下载，只需追加本地路径：

```text
裁剪验证通过后，请加载本地完整规格模型 [checkpoint 目录]，在 [目标卡数/并行策略] 下
验证短程续训；不要重新下载权重。请在第 K 步保存完整训练状态，恢复后继续 N 步，并与
连续 K+N 步对齐。若资源不足以承载完整模型，请报告 BLOCKED。
```

Agent 会检查该目录中的完整 config、权重分片及索引、tokenizer/processor 和 checkpoint
coverage。完整权重路径只用于第二阶段；第一阶段仍基于同一发布 config 构造裁剪模型。

如果权重尚未下载，并且希望 Agent 下载后验证，则追加：

```text
裁剪验证通过后，允许将 [HF repo ID 或 URL] 的完整权重下载到 [本地目录]；请使用完整
发布 config 和权重，在 [目标卡数/并行策略] 下验证短程续训，并在第 K 步保存完整训练状态，
恢复后继续 N 步，与连续 K+N 步对齐。若资源不足以承载完整模型，请报告 BLOCKED，不要
再次裁剪后宣称完整规格通过。
```

只给出 Hugging Face URL 并不代表允许下载大权重。Agent 应先用 source-only 方式取得本地
原始仓：`GIT_LFS_SKIP_SMUDGE=1 git clone --filter=blob:none <URL> <DIR>`，固定解析后的
commit，并确认权重文件仍是 Git LFS pointer，而不是 payload。训练 launcher 使用这个本地
目录读取 config、tokenizer 和源码，不在多卡启动过程中临时访问网络。只有用户明确授权并
给出目标目录后，才可另行下载权重内容。

### 默认的两阶段验证

第一阶段是裁剪模型快速验证。Agent 从完整 config 出发，保留 dense/MoE、不同 attention
角色、共享状态以及视觉路径等结构差异，使用 `from_config` 初始化小模型，完成最终模块
parity、并行/重计算精度自洽和裁剪模型的 checkpoint resume。该阶段通过只能证明
`cropped_due_to_resource` 范围。

第二阶段是完整规格权重续训。对应 Trainer recipe 必须切换到完整 config 和
`from_pretrained`，不得继续使用裁剪 builder。HF-native 模型的加载形式例如：

```yaml
model:
  _target_: hyper_parallel.models.HyperAutoModelForCausalLM.from_pretrained
  pretrained_model_name_or_path: /local/path/to/full-hf-checkpoint
```

自定义架构则由 custom-model registry 解析正式模型类，由 checkpoint mapping 加载同一个本地完整
checkpoint；model adapter 另外暴露并行、FSDP、重计算和验证能力。完整规格验收至少检查权重
coverage、第一步 native/candidate 对齐和短程 loss/norm。

需要注意，Hugging Face 模型仓通常只提供模型权重。用它启动训练是“从预训练权重开始续训”，
不等于“恢复完整训练状态”。断点续训验证还要从该权重启动一条连续 K+N 步基线，同时运行
K 步后保存 HyperParallel DCP，再恢复模型、FP32 main parameters、optimizer、scheduler、
global step、RNG 和 dataloader cursor 继续 N 步；两条路径必须使用相同输入并对齐结果。

### Agent 默认如何制作验证数据

用户没有提供数据时，Agent 应先制作可重复的小数据集，不要求用户理解 Dataset、collator
或 label shift 的细节。

文本模型默认生成本地 JSONL。此前 DeepSeek V4.1 验证使用了 128 条带唯一编号的长文本，
重复固定句子使 Online tokenizer 和 packing 能稳定形成 4K 序列：

```bash
python -m examples.training_demo.deepseek_v41.prepare_deepseek_v41_online_data \
  --output output/training_demo/deepseek_v41/online_4k.jsonl \
  --num-samples 128 \
  --sequence-length 4096
```

每行只有一个通用字段，例如 `{"text": "..."}`。模型自己的 transform 负责 tokenize、
packing 和 labels；固定文件、随机种子和 sampler 后，各并行 case 必须得到相同的
`global_input_sha256`。

多模态模型默认制作少量图片和 OpenAI messages 风格 JSONL。此前 DeepSeek V4.1 VLM
验证从固定 revision 的 ChartQA `val` 中选择固定行，保存图片，并在每条记录中写入数据集
revision、row index 和图片 SHA256：

```json
{"id":"chartqa_val_1656","messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"images/train/chartqa_val_1656.jpg"}},{"type":"text","text":"图中的答案是什么？"}]},{"role":"assistant","content":"China"}],"source":{"dataset":"HuggingFaceM4/ChartQA","revision":"固定 revision","row_index":1656,"image_sha256":"..."}}
```

如果不允许联网，Agent 可以生成固定尺寸和颜色的本地测试图片及对应问答；这足以验证图片
读取、processor、视觉塔、融合与反向链路，但不能替代真实数据质量结论。只有用户明确允许
联网时，Agent 才下载固定 revision 的小型公开数据子集。

无论文本还是多模态数据，Online 数据只有在所有 case 的全局输入哈希一致时，才可作为
正式精度证据；否则只能记为 smoke。需要与外部基线严格逐 token 对齐时，Agent 会把已经
tokenize 的 batch 固化为 Offline replay。

### 裁剪验证和全量验证的结论边界

大模型在有限设备上通常只能先完成裁剪验证。裁剪不是降低验收标准，而是缩小了验收结论
所覆盖的模型实例：同样需要逐模块前向/反向对齐、多步精度自洽、重计算、并行和 checkpoint
验证，但不能把裁剪实例的结果外推为全量模型已经通过。

工具状态始终是 `PASS`、`FAIL` 或 `BLOCKED`。Agent 会根据是否改动发布 config 自动在报告
中标明 `cropped_due_to_resource` 或 `full_pretrained`，用户无需在提示词中填写该字段。

| `validation_scope` 与状态 | 可以证明 | 不能证明 |
|---|---|---|
| `cropped_due_to_resource` + `PASS` | 已覆盖模块和层角色的数学、梯度、数据、并行与恢复路径在所执行配置下通过 | 完整 checkpoint 全覆盖、未进入裁剪图的层角色、全量显存/性能、完整规模通信行为和训练收敛 |
| `full_pretrained` + `PASS` | 完整发布结构和 checkpoint 在实际执行矩阵下通过规定门禁 | 未运行的拓扑、序列长度、硬件或精度配置 |
| `BLOCKED` | 已有证据不足以作出通过或失败结论 | 不能通过放宽阈值、缩小模型或跳过 case 改写为 PASS |

即使裁剪模型保留了全部角色，也仍应标明 `validation_scope=cropped_due_to_resource`。模块 parity 可以使用
与权威实现相同的模块尺寸；端到端 crop 则负责在现有设备上验证这些模块组合后的真实
Trainer 生命周期。获得足够资源后，再使用完整 config/checkpoint 复用相同清单和证据门，
完成 `validation_scope=full_pretrained` 的验证。

## 教程目标

模型接入验证用于回答四个不同层次的问题：最终构建出来的模型结构是否安全、替换后的关键模块是否与权威实现一致、优化路径相对公平基线是否同时满足精度和性能目标、同一训练状态在不同并行策略下是否保持精度自洽。它不会替代普通训练配置，也不会在运行时猜测某个模型族的语义。

模型主体语义由正式 architecture 实现；模型特定的验证契约和安全重计算边界分别由
`ModelAdapterSpec.validation` 与 `ModelAdapterSpec.recompute` 声明。通用工具只负责执行、采证、比较和
报告。新增模型不需要在 Trainer、CLI 或诊断模块中加入模型名、类名或 FQN 特判。

本文统一使用三个容易混淆的术语：

| 术语 | 含义 |
|---|---|
| 正式 reference path | production architecture 自带的未优化、可执行 fallback；也可以作为性能 A/B baseline |
| 正确性 oracle | 与生产实现独立的发布仓实现或公式，用于证明输出、梯度和中间量正确 |
| 优化 candidate | recipe 显式选择的 replacement、CP/EP wrapper 或 fused kernel |

reference path 与优化 candidate 可以共享语义基类来避免状态机漂移，但共享代码不能充当独立 oracle；
公共数学仍要与发布仓实现或独立公式对拍。性能 baseline 还必须与 candidate 位于相同设备、dtype、
输入和拓扑，不能拿 CPU correctness oracle 与 NPU candidate 计算加速比。

## 1. 验证边界

验证流程分为四个证据门：

1. `DISCOVERED`：记录清单、源码环境和候选/参考结构。
2. `STRUCTURE_VALIDATED`：检查正式 architecture 的直接可执行树、可选替换后的最终树、materialization、FSDP 参数归属、checkpoint 覆盖和探针覆盖。
3. `MODULE_PARITY_PASSED`：比较权威模块与生产路径最终模块的前向、中间量、输入梯度和参数梯度。
4. `MATRIX_PASSED`：比较不同 TP/CP/EP/FSDP、重计算深度和断点恢复用例的输入、loss、norm、参数生命周期及 checkpoint layout。

状态只能按顺序前进；`FAILED` 和 `BLOCKED` 是终态。终态目录不得继续复用，应修复问题后使用新的输出目录。

高性能能力发现与 A/B 是贯穿这些状态的设计/实验门禁，不新增状态机枚举。核心模型精度
状态与优化性能状态分别报告；只有 `PRECISION_PASS + PERFORMANCE_PASS` 才能把一个目标
称为已验收的高性能实现。

训练侧只有三个开关值：

| `model_integration.mode` | 行为 | 适用场景 |
|---|---|---|
| `off` | 使用 no-op session，不采集证据 | 正常训练，默认值 |
| `build` | 构建完成后执行结构、布局、数据和 checkpoint 契约检查 | 低成本接入检查 |
| `runtime` | 包含 `build`，并逐步采集输入、梯度、参数、优化器、共享状态、性能和 checkpoint layout | 正式精度矩阵 |

`debug.check_fsdp_runtime` 是独立的 FSDP 执行顺序诊断开关，不由 `model_integration.mode` 隐式开启。

## 2. 准备工作

从当前检出目录安装并确认实际导入路径：

```bash
pip install -e .
python -c "import hyper_parallel; print(hyper_parallel.__file__)"
```

模型、权重、tokenizer、processor 和参考仓可以由用户提供本地路径，也可以提供 Hugging Face
来源并授权 Agent 下载到指定目录。数据可以由用户提供，也可以按本文开头的默认方式交给
Agent 制作。验证流程不会未经允许联网下载资源，也不会替用户设置设备、通信库或 Python
环境变量。

建议先准备一个能被标准 Trainer 正常启动的 recipe。正式精度验证建议固定以下策略：

```yaml
model_init_dtype: float32

fsdp_config:
  mix_precision:
    param_dtype: bfloat16
    reduce_dtype: float32
    cast_forward_inputs: false

optimizer:
  fp32_main_params: true

model_integration:
  mode: off  # validate 命令会对矩阵子进程覆盖为 runtime
```

除非模型合约明确要求，不配置 `fsdp_config.mix_precision.output_dtype`，从而保留模型产生的 FP32 scalar loss。

### 2.1 高性能能力发现与复用设计

在创建 adapter 或编写优化代码前，Agent 必须从权威实现提取 attention、norm、位置编码、
共享/递归状态、router、expert activation、多模态路径和目标 dtype/backend，然后扫描：

```bash
rg --files hyper_parallel/components/modules
rg --files hyper_parallel/distributed/context_parallel
rg --files hyper_parallel/distributed/expert_parallel
rg --files hyper_parallel/models | \
  rg 'adapter/(conversion|distributed|policies)|registration.py'
rg -n 'replacements=|context_parallel=|expert_parallel=' \
  hyper_parallel/models/*/adapter/registration.py
```

搜索结果必须形成 `analysis/optimized_capability_inventory.json`，而不是只写在 Agent 对话里。
每个待优化的最终 FQN、CP 路径和 EP 路径至少记录：

- 权威源码和数学/状态契约；
- 可复用的通用组件、collective、routing/packing/dispatch 原语；
- 可作为设计参考的已有模型 adapter；
- forward/backward、state dict、materialization、checkpoint 和 TP/CP/EP/FSDP 兼容性；
- 目标硬件、kernel/backend 依赖和 fallback；
- `direct_reuse`、`thin_wrapper`、`extract_generic`、`new_generic`、`model_owned` 或
  `unsupported` 决策，以及被拒绝候选的具体原因；
- 预期改善的是计算、通信、重叠、显存还是扩展性；
- 对应精度和性能实验 ID。

复用优先级是：直接复用通用组件；组合通用原语并增加模型薄 wrapper；把已有模型私有实现
中的共性抽到通用组件；新增模型无关组件；最后才保留不可约的模型私有逻辑。已有模型
adapter 只作为参考，不能被另一个模型族直接依赖。名称或 tensor shape 相似不代表可以复用，
mask、参数 packing、状态生命周期、activation 和 checkpoint identity 都必须逐项一致。

实现前还要写 `analysis/optimization_plan.yaml`，声明 correctness reference、performance
baseline、候选实现、相同拓扑 Trainer A/B、模块微基准 shape、warm-up/测量策略、精度容差、
性能阈值、fallback/通信重叠证据和结论范围。性能阈值来自用户或项目目标，不假设一个通用
加速比。没有现成目标时，先运行标记为 pilot 的 baseline 估计重复运行波动区间，随后冻结
正式 plan；candidate 的改善必须超过该区间且不能有无法解释的 p90 或显存回退，不能在看到
candidate 结果后再选择阈值。

### 2.2 模型主体与高性能实现的边界

正式注册的 architecture 必须直接构造可执行的模型主体，并在不应用 recipe replacement 的
条件下通过小型 forward/backward。Attention、MoE/Router、Engram/Memory、残差连接和跨层
状态等数学语义不能只存在于高性能替换函数中。正式模块可以和优化模块共享不含设备优化的
语义基类，但不能在构造时默认把“优化 replacement”当成模型主体。设备 fused kernel 可以是
模块内部可关闭的分支，也可以是 recipe replacement；无论哪种形式，关闭优化不能使模型失去
基本语义。

验收必须分成两个阶段：第一阶段完全不应用 recipe replacement，验证 production architecture 的
direct forward/backward、参数初始化和 checkpoint FQN；第二阶段应用优化 replacement，验证替换前后
state-dict identity、数值精度、TP/CP/EP/FSDP 能力和同拓扑性能。第二阶段通过不能替代第一阶段，
否则 Placeholder 或 mandatory wrapper 仍可能被误当成正式模型。

只有在 planner 或 loader 必须先观察官方参数树、且直接构造无法保持 FQN、参数 alias、初始化、
meta materialization 或 checkpoint 契约时，才允许临时参数 Placeholder。此时必须记录该
生命周期约束，证明替换前后 state-dict/FQN identity，并保证最终生产 builder 返回前完成
finalization；structure gate 发现遗留 Placeholder 必须失败。为了开发方便而用 Placeholder
拼出模型，不属于可接受理由。

直接套用通用 wrapper 时还要复现源类的类型特定初始化。例如 wrapper 使模块不再是上游
Attention/HyperConnection 类型后，上游 `post_init()` 可能不会清零 attention sinks 或残差
mixing base，也不会把 scale 置为 1。除类型断言外，必须检查这些特殊参数的初始化值，并用
端到端反向确认 loss 和梯度有限。

## 3. 声明模型侧验证契约

### 3.1 注册入口

每个模型族在 `hyper_parallel/models/<family>/adapter/registration.py` 完成三类互相独立的登记。注册模块
由目录约定惰性发现，中央 registry 无需修改：

| 注册机制 | 登记的关系 | 不负责什么 |
|---|---|---|
| `register_custom_model()` | `config.architectures[0]` 到正式 Python 类 | 不构造模型、不应用 recipe、不选择并行策略 |
| checkpoint mapping 注册 | 发布 checkpoint key 到正式参数 key 的转换 | 不创建参数、不决定 placement、不下载权重 |
| `register_model_adapter()` | model type/architecture 到模型能力 provider | 仅使 provider 可发现；不代表 provider 已调用或优化已启用 |

一个完整入口的骨架如下。checkpoint 注册 API 取决于上游权重系统，因此由 family 函数封装：

```python
from hyper_parallel.models.adapter_spec import ModelAdapterSpec
from hyper_parallel.models.registry import register_custom_model, register_model_adapter

from .conversion.checkpoint_mapping import register_example_checkpoint_mapping


def _load_validation():
    from .validation.model_validation_spec import get_validation_spec

    return get_validation_spec()


MODEL_ADAPTER_SPEC = ModelAdapterSpec(
    architecture="ExampleForCausalLM",
    model_type="example",
    validation=_load_validation,
)
register_custom_model(
    "ExampleForCausalLM",
    "hyper_parallel.models.example.modeling_example",
    "ExampleForCausalLM",
)
register_example_checkpoint_mapping()
register_model_adapter(MODEL_ADAPTER_SPEC)
```

import 这个 registration 模块只完成登记，不会立即解析模型类、调用 provider、安装 replacement 或启动
并行。`validation` 必须惰性加载，这样普通训练不会导入权威参考仓、测试算子或验证专用依赖。
`replacements/context_parallel/expert_parallel` 在当前 recipe 驱动路径中属于能力目录，实际启用仍由
`replace_module._target_`、`inner_wrapper._target_` 或 `local_compute_fn._target_` 显式声明；sharding、
FSDP、recompute 和 validation provider 则由对应 planner/Trainer/工具直接消费。

### 3.2 最小验证声明

`ModelValidationSpec` 的所有字段均可选；根据模型风险逐项声明：

```python
from hyper_parallel.models.validation_spec import (
    DataValidationSpec,
    ModelValidationSpec,
    ParameterProbeSpec,
    StateInvariantSpec,
)


def _canonical_tree(model, _context):
    invalid = [
        name for name, module in model.named_modules()
        if type(module).__name__.endswith("Placeholder")
    ]
    return None if not invalid else {"non_executable_modules": invalid}


def get_validation_spec():
    return ModelValidationSpec(
        parameter_probes=(
            ParameterProbeSpec("model.layers.*.experts.*.weight"),
        ),
        state_invariants=(
            StateInvariantSpec(
                name="example.canonical_executable_tree",
                checker=_canonical_tree,
                phase="structure",
                error_code="HP-REPLACE-002",
            ),
        ),
        data=DataValidationSpec(
            required_forward_fields=("input_ids", "labels"),
            labels_are_shifted=True,
        ),
    )
```

常用声明如下：

| 声明 | 用途 |
|---|---|
| `ModuleParityCase` | 定义权威模块与最终生产模块的比较方法 |
| `ObservationSpec` | 增加路由结果、索引、共享状态等中间量比较 |
| `ParameterProbeSpec` | 按最终 FQN 选择参数及生命周期阶段 |
| `StateInvariantSpec` | 在 structure/materialization/checkpoint/runtime 阶段检查模型语义 |
| `SharedStateValidationSpec` | 验证跨层状态只发布一次且消费者读取正确来源 |
| `DataValidationSpec` | 声明 forward 字段、运行时字段、模态字段和 label shift |
| `CheckpointValidationSpec` | 声明训练专用 target 和推理专用 source 的合法缺失 |
| `TopologyConstraint` | 在启动前拒绝模型不支持的并行组合 |

### 3.3 模块精度对齐

进程内 parity 的 candidate 必须来自正式构建/materialization 路径；若实验对象是可选
replacement，再让 candidate 经过该真实替换路径。reference 必须是独立实现。下面示例比较
一个最终线性模块：

```python
def build_candidate(context):
    return build_final_production_model(context.manifest.model)


def build_reference(_context):
    return AuthoritativeLinear(16, 32)


def build_inputs(_context):
    return (torch.randn(2, 16),)


def copy_weights(reference, candidate, _context):
    with torch.no_grad():
        reference.weight.copy_(candidate.weight)
    return {"weight": "weight"}


case = ModuleParityCase(
    name="linear",
    candidate_selector="model.layers.0.proj",
    candidate_builder=build_candidate,
    reference_builder=build_reference,
    input_builder=build_inputs,
    weight_adapter=copy_weights,
)
```

当参考仓与生产环境存在模块名冲突或只能在独立解释器中导入时，使用 `execution="isolated_process"` 和 `isolated_runner`。DeepSeek-V4.1 的 Engram、MoE 和 CSA2 就采用这一方式。

### 3.4 安全重计算区域

共享 KV、Indexer 状态或其他跨层状态的 producer 不应被整层 replay。模型 adapter 应声明可重计算子模块：

```python
from hyper_parallel.models.adapter_spec import RecomputePolicy


def build_recompute_policy():
    return RecomputePolicy(
        safe_module_patterns=(
            "model.layers.*.input_layernorm",
            "model.layers.*.post_attention_layernorm",
            "model.layers.*.mlp",
        ),
        no_replay_module_patterns=("model.layers.*.self_attn",),
    )
```

Trainer recipe 通过通用配置选择该策略：

```yaml
activation_checkpoint:
  mode: selective
  selection:
    source: model_adapter_safe_regions
    layer_count: 16
```

`layer_count` 按所有已发现 layer container 的展平顺序，从下标 `0` 开始连续选择指定数量的层；例如 `layer_count: 2` 选择第 `0、1` 层，`0` 表示不选择任何层。需要选择非连续层时使用与其互斥的 `layer_indices`：

```yaml
activation_checkpoint:
  mode: selective
  selection:
    source: model_adapter_safe_regions
    layer_indices: [0, 4, 8, 12]
```

`layer_indices` 从 `0` 开始，必须非空、非负且不重复。`source: default` 不接受这两个字段。

## 4. 编写验证清单

验证清单只描述模型身份、标准 Trainer 启动方式、实验矩阵和验收阈值。模型结构、数据集、dtype、tokenizer 和本地资产应留在 Trainer recipe；环境变量、源码 revision 和 hash 会自动进入证据，不在 YAML 中重复维护。

```yaml
schema_version: 1

model:
  adapter: example
  # inspect/check 需要最终生产模型 builder；kwargs 原样传入。
  builder:
    _target_: your_package.build_final_model
    config_path: /local/model/config.json

reference:
  # parity 使用的本地权威仓；不会隐式下载。
  source_path: /local/reference/repository

launcher:
  module: examples.training_demo.train_text
  config: /local/recipes/train_example.yaml
  timeout_seconds: 7200

matrix:
  devices: 16
  steps: 10
  resume_split_step: 5
  performance_warmup_steps: 2
  # 先由 baseline initialize launch 完成 1 步并保存 step-1 warm-start DCP，再让正式 baseline 和
  # 所有候选恢复同一模型、优化器、scheduler 与 RNG，并从相同配置的数据起点请求 replay。
  shared_initial_checkpoint: true
  baseline:
    tp: 1
    cp: 1
    ep: 1
    fsdp: 16
    sequence_parallel: false
    recompute: {layer_count: 0}
  axes:
    ep: [1, 16]
  combined:
    - {tp: 2, cp: 1, ep: 16, fsdp: 8, sequence_parallel: true}
    - {tp: 1, cp: 2, ep: 16, fsdp: 16, sequence_parallel: false}
  recompute_selections:
    - {layer_count: 0}
    - {layer_count: 2}
    - {layer_indices: [0, 1, 2, 3]}
  recompute_topology: {tp: 1, cp: 1, ep: 16, fsdp: 16}
  same_topology_resume: true
  cross_topology_resume: {tp: 2, cp: 1, ep: 16, fsdp: 8}

acceptance:
  same_topology:
    loss_max_abs: 5.0e-4
    norm_max_abs: 2.0e-3
  cross_topology:
    loss_max_abs: 5.0e-3
    loss_max_rel: 5.0e-3
    norm_max_abs: 1.25e-1
    norm_max_rel: 5.0e-3
    # 分别对 loss 和 norm 使用“绝对或相对误差满足其一”；loss 与 norm 仍必须都通过。
    combination: any
  parameters:
    # 基础档用于模块对拍和同拓扑；两者不能被跨拓扑 BF16 容差放宽。
    max_abs: 1.0e-6
    relative_l2: 1.0e-6
    combination: all
    summary:
      l2_norm_relative: 1.0e-6
    cross_topology:
      max_abs: 1.0e-3
      relative_l2: 2.0e-2
      # 只对记录了完整 values 的张量生效。
      combination: any
      summary:
        # 大张量只有聚合摘要时，只能判断 L2 范数本身的相对漂移。
        l2_norm_relative: 2.0e-2
```

### 4.1 Matrix 字段

| 字段 | 说明 |
|---|---|
| `devices` | 每个 `torchrun` 子进程的 `--nproc_per_node` |
| `steps` | 普通用例训练步数，也是 resume 恢复后的最终步数 |
| `resume_split_step` | resume prepare 阶段保存步数，必须满足 `0 < split < steps` |
| `shared_initial_checkpoint` | 生成公共 step-1 warm-start DCP；baseline、候选及 resume prepare 恢复同一模型/优化器/scheduler/RNG，但不恢复 warm-start 的 dataloader cursor，而是请求从配置的数据起点 replay；是否得到相同逻辑输入仍由 global input hash 门禁判定 |
| `baseline` | 所有用例的参考拓扑 |
| `axes` | 每次只覆盖一个字段，重复 baseline 的值会去重 |
| `combined` | 显式覆盖多个 baseline 字段的混合策略 |
| `recompute_selections` | 安全区域重计算的精确层数或层下标；每项只能包含 `layer_count` 或 `layer_indices` |
| `recompute_topology` | 运行重计算选择组时覆盖的固定拓扑 |
| `production_validate_pair` | 同一拓扑分别设置 `model.validate_placement=false/true` |
| `same_topology_resume` | baseline 拓扑保存并恢复 |
| `cross_topology_resume` | baseline 拓扑保存、目标拓扑恢复 |

标准 launcher 只接受 `tp`、`cp`、`ep`、`fsdp`、`sequence_parallel` 和 `recompute`。无法映射到 Trainer 命令的字段会直接报错，不会生成“命令相同但看似通过”的空转用例。

编写 matrix 前必须先建立 coverage ledger：逐项列出模型实际支持的并行轴、关键混合拓扑、重计算层选择、同/跨拓扑恢复、Production/Validate，以及多模态模型的 language/vision/projector/fusion 子树。每项只能是正式用例或带证据的排除项。`--generate-only` 后将 ledger 与 `cases/resolved_cases.json` 逐项对账，并分别统计 baseline、策略泛化、重计算、resume 和 Production/Validate；重计算和 resume 不计入策略泛化数量。只执行了一个策略候选时，即使另有两个重计算和两个 resume 用例，也只能报告一组策略泛化。

多模态模型还必须记录各子树的实际 placement。ViT 在含 TP 的进程组里完成前向反向、但其参数沿 TP 为 `Replicate()` 时，只能证明“多模态路径与语言 TP 可组合”，不能报告为 ViT TP 覆盖。算子支持也按 dispatcher 截获的精确 callable key 审计；`npu_rms_norm` 不能替代 `rms_norm`，`conv3d` 不能替代 `conv1d`/`conv2d`。

验收档位根据实际启动的 TP/CP/EP/FSDP/sequence-parallel 与 baseline 是否相同自动选择。重计算层选择本身不构成并行拓扑变化；如果 `recompute_topology` 改了 EP 等字段，则该组使用 `cross_topology` 阈值。`cross_topology_resume` 必须真正改变至少一个并行字段，当前不提供“只切换重计算策略后恢复”的用例。

标量支持绝对阈值与相对阈值。默认 `combination: all`；显式设为 `any` 时，比较器分别判断 loss 和 norm 的“绝对或相对误差满足其一”，然后仍要求 loss、裁剪前 norm、裁剪后 norm、输入身份、LR 和有限性全部通过。不能用 loss 的相对阈值替代 norm 失败，也不能用 `any` 绕过数据身份门禁。

参数探针必须区分拓扑档位和证据强度。`parameters` 根字段是模块对拍与同拓扑的严格基础档；`parameters.module_parity`、`parameters.same_topology`、`parameters.cross_topology` 可分别覆盖它，未声明的字段继承基础档。因此 BF16 跨拓扑的尺度容差不会意外放宽模块对拍或同拓扑恢复。记录了完整 `values` 的小张量才能计算真实逐元素 `max_abs` 和张量 `relative_l2`，其默认组合规则是 `all`；BF16 跨拓扑可在预先声明且有数值依据时使用 `parameters.cross_topology.combination: any`。只记录 `sum/l2/min/max` 的大张量没有逐元素证据，比较器会将 `max_abs` 和 `relative_l2` 报为 `n/a`，仅按当前档位的 `summary.l2_norm_relative`（以及可选的 `l2_norm_max_abs`）判断 L2 范数摘要漂移。聚合统计量之差不得再标成逐元素最大误差。两种模式始终单独检查有限性和全局 shape。

### 4.2 复用 parity handoff

可以在同一个输出目录依次执行所有门，也可以让精度清单引用之前生成的 handoff：

```yaml
integration_handoff: /local/evidence/integration_handoff.yaml
```

handoff 必须是 schema-v1、`status: PASS`，并且 `family` 与 `model.adapter` 相同。它只证明结构与模块 parity 已通过，不替代本次 matrix 的 Trainer recipe 和输入证据。

### 4.3 高性能替换的成对实验

每个被称为高性能的 replacement、CP 或 EP 路径需要四层证据：

| 实验 | 固定条件 | 需要回答的问题 |
|---|---|---|
| 模块精度对拍 | 相同权重、输入、状态、dtype | 输出、中间量、输入梯度和参数梯度是否与独立权威实现一致 |
| 目标设备模块微基准 | 相同 shape、dtype、autograd、compile 和同步边界 | forward/backward p50/p90、吞吐、峰值显存和 kernel fallback 是否达到预声明目标 |
| 同拓扑 Trainer A/B | 相同起始/预热步完整 DCP、数据 hash、拓扑、精度、优化器、重计算和步数 | 优化路径是否在真实训练生命周期保持精度并改善端到端性能 |
| CP/EP 扩展性 | 相同全局 workload 和模型语义 | CP/EP degree 增加后的通信、负载均衡、显存和扩展效率如何 |

正确性 reference 与性能 baseline 不一定相同：权威 eager 公式可以用于精度，但 CPU
reference 不能与 NPU candidate 计算加速比；性能 baseline 与 candidate 必须使用相同设备、拓扑、shape、
dtype、compile 和同步边界。普通 validation matrix 只切换并行拓扑与
recompute，不能切换任意 replacement provider；因此端到端 A/B 使用两个只在目标实现选择上
不同的 recipe 或 `plan_overrides`，分别写入隔离证据目录，并共享同一个已记录实际步号的
完整 DCP。标准 matrix launcher 当前生成 step-1 warm-start DCP，不能把它写成 step 0。

这里还有一个必须在 `--generate-only` 后预检的限制：DCP 能重分片模型/优化器，不代表 stateful
dataloader cursor 也能跨 DP world size 重分片。标准 launcher 已区分两个场景：共享 warm-start 恢复
模型/优化器/scheduler/RNG，但显式不恢复 initialize 步骤消费后的 cursor，让 measured case 从配置的数据
起点重新请求 replay；真正 K→K+N resume 则仍恢复 K 步后的 cursor。重新从起点不自动等于相同输入：若
Online loader 的 rank 分片、随机化或 packing 依赖 DP world size，TP/CP 改变 DP degree 后 global input hash
仍可能不同，此时该 case 只能证明拓扑可执行，不能比较数值。正式跨拓扑精度应使用 DP-invariant 的固定
Offline replay；当前 Online dataloader 在真正 resume 场景遇到 DP world size 改变也会显式失败，需要能按
全局样本位置重建 cursor 的 loader，否则将能力记录为 `BLOCKED`/带证据排除。不能删除 cursor 后仍宣称
输入连续，也不能把输入不一致或训练前失败记成 TP/CP 模型精度失败。

CP 的实现 A/B 必须在相同 CP/TP degree、全局序列、mask 和 attention kernel 下比较同步/参考
与优化通信；CP1 对 CPn 只能称扩展性实验。EP 同理，需要在相同 EP degree、learned router、
expert layout、capacity 和 grouped-kernel 设置下比较 dispatch/compute/combine；EP1 对 EPn
不能替代实现 A/B。

结果写入 `analysis/optimization_results.json`，逐目标保存精度误差、p50/p90、吞吐、峰值
显存、collective bytes/time、异步 launch/wait 与 overlap、实际 kernel/backend、fallback、
样本数/离散度和范围。分别给出：

- `PRECISION_PASS` 或 `PRECISION_FAIL`；
- `PERFORMANCE_PASS`、`PERFORMANCE_FAIL` 或 `PERFORMANCE_INCONCLUSIVE`。

只有 `PRECISION_PASS + PERFORMANCE_PASS` 才能称该目标为已验收的高性能实现。裁剪模型
端到端结果只适用于 crop；使用完整 shape 的模块微基准也不能外推成完整模型吞吐或通信
扩展性。

当前通用 `report` 命令不会执行或读取任意 replacement A/B。保留它生成的 `summary.md` 和
`summary.zh-CN.md`，再由 Agent 从 inventory、plan 和 results 生成相邻的英文
`optimization-summary.md` 与中文 `optimization-summary.zh-CN.md`；不能手工改写矩阵首行
状态来伪装性能验收。

## 5. 执行流程

以下命令应使用同一个绝对 `--output-dir`。相对路径按 manifest 所在目录解析，不按当前 shell
目录解析，因此建议在正式运行中始终写绝对路径：

```bash
RUN_DIR=output/model_integration/example
MANIFEST=/local/manifests/example_validation.yaml

python -m hyper_parallel.tools.model_integration inspect \
  --manifest "$MANIFEST" --output-dir "$RUN_DIR"

python -m hyper_parallel.tools.model_integration check \
  --manifest "$MANIFEST" --output-dir "$RUN_DIR"

python -m hyper_parallel.tools.model_integration parity \
  --manifest "$MANIFEST" --output-dir "$RUN_DIR" \
  --device cpu --dtype float32

# 按 analysis/optimization_plan.yaml 执行生产 dtype 模块对拍、目标设备
# 微基准，以及隔离目录中的 reference/optimized 同拓扑 Trainer A/B。
# 这些实验由模型的 builder、recipe 和 plan_overrides 决定，没有通用的
# 隐式 replacement 开关；结果写入 analysis/optimization_results.json。

python -m hyper_parallel.tools.model_integration validate \
  --manifest "$MANIFEST" --output-dir "$RUN_DIR" --generate-only

python -m hyper_parallel.tools.model_integration validate \
  --manifest "$MANIFEST" --output-dir "$RUN_DIR"

python -m hyper_parallel.tools.model_integration report \
  --output-dir "$RUN_DIR"
```

先用 coverage ledger 检查 `cases/resolved_cases.json` 的分类数量和完整拓扑列表，再占用多卡资源。正式用例中移除 `--generate-only`。

命令语义：

| 命令 | 输入 | 主要产物 |
|---|---|---|
| `inspect` | 本地路径或显式 builder | reference/candidate inventory、结构差异、资源指纹 |
| `scaffold` | 模型 identity | 最小 adapter 注册、validation provider、manifest 和注册测试骨架 |
| `check` | 最终生产模型 builder | findings、参数归属、FSDP unit、checkpoint 覆盖 |
| `parity` | adapter validation provider + reference | 模块比较和 `integration_handoff.yaml` |
| `validate` | Trainer recipe + matrix | 每个 case 的训练证据、跨 case 比较、性能摘要 |
| `report` | 已存在的 evidence dir | 首行稳定为 PASS/FAIL/BLOCKED 的英文 `summary.md` 与中文 `summary.zh-CN.md` |

退出码为 `0=PASS`、`1=FAIL 或执行错误`、`2=BLOCKED 或 manifest 错误`。

## 6. 只启用 Trainer 内检查

不运行矩阵时，也可以直接在普通 Trainer recipe 中开启：

```yaml
model_integration:
  mode: build
```

或：

```yaml
model_integration:
  mode: runtime
```

若未设置 `HYPER_PARALLEL_MODEL_INTEGRATION_OUTPUT_DIR`，证据默认写到 `<checkpoint_dir>/model_integration`。矩阵 launcher 会为每个 case 自动设置独立目录，普通用户无需在 YAML 增加输出路径字段。

`runtime` 会在优化器更新前检查梯度和布局；因此发现错误时不会先污染参数。它会增加诊断开销，只应用于验证运行。

## 7. 证据目录

典型目录如下：

```text
<run>/
├── manifest.resolved.yaml
├── environment.json
├── integration_state.json
├── source_fingerprints.json
├── inventory/
│   ├── reference.json
│   ├── candidate.json
│   └── structure_diff.json
├── check/
│   ├── findings.json
│   ├── parameter_ownership.csv
│   ├── fsdp_units.json
│   └── optimizer_layout_initial.json
├── checkpoint/
│   └── coverage.json
├── module_parity/
│   ├── comparison.json
│   └── <case>/comparison.json
├── integration_handoff.yaml
├── analysis/
│   ├── operator_support.json
│   ├── parallel_ownership.json
│   ├── issues.json
│   ├── optimized_capability_inventory.json
│   ├── optimization_plan.yaml
│   └── optimization_results.json
├── cases/
│   ├── resolved_cases.json
│   └── <case>/
│       ├── preflight.json
│       ├── stdout.log
│       ├── stderr.log
│       └── cases/runtime/
│           ├── metrics.jsonl
│           └── performance.jsonl
├── comparison.json
├── summary.md
├── summary.zh-CN.md
├── optimization-summary.md
└── optimization-summary.zh-CN.md
```

参数探针和 checkpoint layout 会按 rank 写文件。比较器按逻辑 global shape、mesh/placement 和 logical shard identity 聚合，不通过把完整大参数聚集到单卡来比较。

普通并行矩阵中的性能数据是描述信息：loss、norm、输入 identity、参数探针、布局或
checkpoint continuation 失败仍决定核心精度状态；吞吐不会改变该状态。对明确声明为高性能
的 replacement、CP 或 EP，另行使用预声明的同拓扑成对实验作性能门禁，其结果只决定该
优化目标的 `PERFORMANCE_PASS/FAIL/INCONCLUSIVE`，不能掩盖精度失败。

## 8. DeepSeek-V4.1 示例

可从 [DeepSeek-V4.1 validation manifest](../../../examples/training_demo/deepseek_v41/deepseek_v41_validation.yaml) 开始。该示例保持清单简洁：模型结构、Online 数据、4K 序列和 dtype 在相邻 Trainer recipe 中，shell 环境和本地资产由用户指定。

DeepSeek adapter 的验证声明包括：

- Engram `q_weight`、`k_weight`、`wkv.weight` 参数生命周期探针；
- CSA2 sinks 与 MoE gate bias 探针；
- 正式模型直接构造可执行的 Attention、Engram、mHC 和 Router，不依赖 mandatory replacement；
- 直接 wrapper 覆盖的 attention sinks、mHC base/scale 等特殊初始化必须正确且梯度有限；
- scratch crop 的 Engram q/k 必须完成官方初始化；
- `compressed_kv`、`index_key`、`topk_indices`、`candidate_blocks` 跨层 publish/consume 检查；
- packed runtime 字段、视觉输入字段及 vision/aligner 梯度检查；
- 独立进程中的原生 Engram/MoE/CSA2 前向与反向比较；
- attention 不 replay，只对 norm/MLP/MoE 安全区域做分层重计算。

DeepSeek-V4.1 的模型主体和优化路径是一一对应、但彼此不互相替代的：

| 模型语义 | 正式 reference 模块 | recipe 可选 candidate | 选择方式 |
|---|---|---|---|
| pipelined mHC | `DeepseekV41PipelinedHyperConnection` | `PipelinedMhcModule` | `replace_module._target_` |
| Engram | `DeepseekV41Engram` | `EngramModule` | `replace_module._target_` |
| CSA2 Attention | `DeepseekV41Attention` | `SharedCompressedDSAAttention` | `replace_module._target_` |

两种 Attention 共享 `SharedCompressedDSAAttentionBase`：正式模块固定
`use_optimized_sparse_attention=False`，candidate 固定为 `True`。provider 只登记 factory；当前 demo 是 recipe
显式选择三项 replacement。删掉这些条目时模型仍能 forward/backward，并回退到未优化 reference path。

示例 `recompute_topology` 使用 EP16，而 baseline 使用 EP1，因此重计算组刻意按跨拓扑阈值比较。如果目标是单独验证重计算开关，应让 `recompute_topology` 的并行字段与 baseline 相同。

## 9. 常见失败与处理

| 现象 | 含义 | 处理方式 |
|---|---|---|
| `HP-REPLACE-001/002` | 可选替换未命中，或最终树仍有非可执行 Placeholder/forward-state 契约破坏 | 优先把模型主体迁入正式 architecture；确有生命周期约束时再修正 finalization、adapter pattern 或显式 mapping |
| `HP-FSDP-004/005/006` | 一个 FSDP unit 混合 dense/expert 域、专家落入 root 或 alias 归属冲突 | 在 adapter 声明同质 child unit 和正确边界 |
| `HP-MAT-001/002/003` | meta tensor、derived buffer 或模型初始化未完成 | 修正 materialization hook/模型初始化，不用 checkpoint 掩盖派生状态 |
| `HP-CKPT-001/002` | checkpoint key 未覆盖或转换不可逆 | 增加 mapping/transform/reverse transform 或显式合法排除 |
| `HP-DATA-001/002` | forward 字段所有权冲突、字段缺失、label shift/模态梯度不符 | 修正 Omni dataset transform、get-batch 或 runtime input adapter 声明 |
| `HP-PREC-001` | required probe 未匹配最终参数 | 使用替换后的最终 FQN 更新 adapter 声明 |
| `HP-STATE-001/002/003` | 跨层状态缺 producer/consumer、producer replay 或 trace 超限 | 缩小重计算边界，修正共享状态来源和 forward 生命周期 |
| Online dataloader 拒绝跨 DP world-size cursor，或跳过 cursor 后 input hash 仍不同 | DCP 能重分片模型/优化器，不代表 loader 能重建全局样本位置；从同一配置起点也不保证 DP-invariant packing | warm-start 可用 `restore_dataloader_state=false` 请求 replay，但必须检查 global input hash；正式跨拓扑精度改用固定 Offline replay，真正续训改用支持全局 cursor 重建的 loader，否则标为 `BLOCKED` |
| `BLOCKED` | 证据、设备、资产或独立 oracle 不可用 | 补齐前置条件；不要放宽阈值或静默缩小实验 |

如果两个 case 的 `global_input_sha256` 不同，应先修复数据重放，再判断 loss/norm。若聚合 loss/norm 通过但参数探针失败，应以参数探针为准，它用于发现小参数、专家局部参数和 optimizer state 的局部错误。

## 10. 新模型接入完成条件

一个接入只有同时满足以下条件才可判定完成：

- 中央框架中没有新增模型名、模型类、模型 FQN 或模型配置字段特判；
- 高性能能力 inventory 覆盖所有 required optimized modules、CP 和 EP 路径，并记录搜索范围、复用决策、被拒绝候选和 fallback；
- 每个高性能声明都有预先制定的 optimization plan、独立精度 oracle、目标设备模块微基准和同拓扑端到端 A/B；
- 优化目标只有在 `PRECISION_PASS + PERFORMANCE_PASS` 时称为已验收；缺少公平 baseline 或稳定证据时明确标记 `PERFORMANCE_INCONCLUSIVE`；
- 最终生产模块而非测试替身参与 structure check 和 parity；
- 正式 architecture 在不应用 recipe replacement 时可直接完成有限的 forward/backward，高性能实现不承担模型主体语义；
- checkpoint 每个 source/target key 都有明确分类；
- dense/expert 参数归属和 DTensor global layout 可证明；
- coverage ledger 与 `resolved_cases.json` 完全对账，报告按互斥类别计数；
- 多模态各子树的实际 TP/CP/EP/FSDP placement 单独记录，不以启动拓扑代替真实切分证据；
- 所需 DTensor 算子按精确 callable key 验证实现、签名与注册；
- 输入 identity、loss、pre/post-clip norm、关键参数/梯度/优化器状态均比较；
- 共享跨层状态和重计算边界由 adapter 明确声明；
- 多模态输入到达模型时，声明的视觉参数获得有限梯度；
- 同拓扑、跨拓扑、EP1 对照、重计算和 resume 覆盖满足模型实际使用场景；
- 中英文最终报告首行为 `PASS`，并在人工结论中明确标注裁剪或全量验证范围。
