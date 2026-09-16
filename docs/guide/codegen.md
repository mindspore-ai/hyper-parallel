# Codegen 使用指南

Codegen 把并行规划与高性能模块替换从训练运行期提前到**训练启动前的生成阶段**。它读取训练配置
YAML，把其中的并行契约、模块替换和可静态展开的通信逻辑写入 `generated/` 产物。用户可以直接打开
generated model 文件和 diff，看到训练实际导入的模型代码，定位问题时不再只能从运行时 planner 的日志里反推。

当前推荐使用的是 inline 产物形态：Codegen 会先基于 YAML 完成模块替换，再在生成模型文件中展开已经支持的
TP/CP/EP 策略。以 Qwen3-MoE 为例，RMSNorm、GQA attention、CP attention 通信、EP routed MoE dispatch
都会出现在 `modeling_Qwen3_Moe_gen_npu.py` 里；训练启动后只安装一份很薄的并行状态，真正的 forward 逻辑由
generated model 文件执行。

## 核心概念

| 概念 | 说明 |
|------|------|
| 产物 bundle | 位于训练 YAML 同级的 `generated/` 目录，包含 generated model 文件、`.diff` 差异文件、`codegen_meta.json` 等 |
| signature | 由会影响产物的配置和原始 modeling source 摘要生成；训练步数、日志级别等运行参数变化不会触发重新生成 |
| `codegen: true` | 打开 Codegen 生成与检查流程；未显式指定其他后端且 `force_hf=false` 时，后端解析为 `gen` |
| `modeling_backend: gen` | 显式指定用 generated model 构建模型，推荐在 YAML 中明确配置 |
| 生成文件名 | 由 HuggingFace config 的 `model_type` 推导，例如 `qwen3_moe` 生成 `modeling_Qwen3_Moe_gen_npu.py` |
| `force_hf` | 逃生门，优先级最高；设为 `true` 会回到原始 HF modeling，不会使用 generated model |

## 前置条件

- 一份可被训练入口解析的 YAML。标准训练入口会记录 YAML 路径，Codegen 用它定位 `generated/` 目录。
- 模型来源可解析。离线环境推荐让 `pretrained_model_name_or_path` 指向本地 checkpoint；如果使用模型 ID，当前
  Transformers 环境必须能取得 config 和 modeling source。
- 分布式启动方式要和 YAML 中的并行规模匹配。例如配置了 `tp_size=2`、`cp_size=2`，手动执行 `generate` /
  `check` 时也应通过 `python -m torch.distributed.run` 启动，让 Codegen 能看到正确的 world size。

## 快速开始

### 1. 在 YAML 中打开 Codegen

推荐从已经验证过的 `examples/training_demo/train_codegen_qwen3_moe.yaml` 开始。自定义 YAML 时，需要同时打开
Codegen、选择 `gen` 后端，并关闭 `force_hf`：

```yaml
codegen: true
modeling_backend: gen

model:
  _target_: hyper_parallel.models._transformers.HyperAutoModelForCausalLM.from_pretrained
  pretrained_model_name_or_path: /path/to/Qwen3-30B-A3B
  force_hf: false
  torch_dtype: bfloat16
  attn_implementation: sdpa
```

Codegen 的输入就是这份 YAML 本身，不需要第二份配置。signature 会覆盖模型来源、`accelerator` 中的并行规模、
`plan_overrides` 中的模块替换与策略配置、`codegen` 和 `modeling_backend`。优化器、训练步数、日志级别和
YAML 注释不属于 generated model 的语义输入。

### 2. 生成与检查产物

训练启动时会自动准备产物：

- 产物不存在：生成；
- 产物存在但 signature 不一致：重新生成；
- 产物存在且 signature 一致：复用并做文件校验。

也可以手动管理产物。若 YAML 中的 TP/CP/EP 等配置要求多卡，手动命令也要用分布式方式启动：

```bash
# 生成或复用产物；--verbose 会显示 artifact 路径和 signature
python -m torch.distributed.run --nproc_per_node=4 \
  --module hyper_parallel.codegen --verbose generate \
  --config examples/training_demo/train_codegen_qwen3_moe.yaml

# 完整检查：signature、文件哈希、meta、import 与 generated forward 结构
python -m torch.distributed.run --nproc_per_node=4 \
  --module hyper_parallel.codegen --verbose check \
  --config examples/training_demo/train_codegen_qwen3_moe.yaml

# 只检查产物是否与当前配置匹配；不触发重新生成
python -m torch.distributed.run --nproc_per_node=4 \
  --module hyper_parallel.codegen --verbose check \
  --config examples/training_demo/train_codegen_qwen3_moe.yaml \
  --require-fresh
```

`clean` 只删除某一 YAML 对应的 bundle，不执行并行配置预检，可以单进程运行：

```bash
python -m hyper_parallel.codegen --verbose clean \
  --config examples/training_demo/train_codegen_qwen3_moe.yaml
```

补充说明：

- `generate` 和 `check` 遇到 `codegen:false` 时会直接跳过；加上 `--verbose` 才会显示跳过原因。
- 分布式启动时只有 rank0 写入 bundle，其他 rank 等待同一份 signature 完成后返回。
- `check --require-fresh` 适合 CI 或上线前检查：只接受已经存在且与当前 YAML 匹配的产物。
- `check` 不实例化完整训练模型，真实权重覆盖检查仍发生在训练加载 generated model 时。

### 3. 启动训练

用 `python -m torch.distributed.run` 启动训练。不要依赖 PATH 上的同名 `torchrun`，它可能被其他 CLI 应用遮蔽，
导致 PyTorch 分布式参数无法识别。

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HYPER_PARALLEL_PLATFORM=torch

python -m torch.distributed.run --nproc_per_node=8 \
  --module examples.training_demo.train_text \
  examples/training_demo/train_codegen_qwen3_moe.yaml \
  --model.pretrained_model_name_or_path=/path/to/Qwen3-30B-A3B \
  --model.force_hf=false \
  --dataset.data_path=./outputs/.../wikitext103_qwen3_4k_text_document \
  --training.train_iters=5
```

首次启动时应看到生成日志；复用时会显示复用日志。以示例 YAML 为例，产物会生成在 YAML 同级目录：

```text
codegen: generating artifact examples/training_demo/generated (signature f463aefff4a4)
```

### 4. 判断是否真正使用 generated model

生成 artifact 只表示 Codegen 执行过，不等于训练模型一定来自 generated model。需要同时确认以下日志。

模型构建阶段应看到 generated modeling 文件被导入：

```text
codegen: imported generated modeling file .../modeling_<model>_gen_npu.py
```

并行化阶段应看到 generated module 接管 sharding：

```text
codegen: sharding applied by the generated module; source_shard_info keys=<N>
```

权重加载和训练 step 也应继续正常推进：

```text
Loaded <N> model tensors from ...
step=1 ... training/foundation_loss=...
```

如果只看到 `generating artifact` 或 `reuse artifact`，不能证明训练已经使用 generated model。`force_hf=true`
时，artifact 仍可能生成或复用，但模型后端会回到原始 HF。

## 产物 bundle

生成物位于训练 YAML 同级的 `generated/` 目录下。当前默认规则是“YAML 所在目录直接拥有一份 generated
bundle”，不会再创建以 YAML 文件名命名的子目录：

```text
/home/hyperparallel/examples/training_demo/train_codegen_qwen3_moe.yaml
→ /home/hyperparallel/examples/training_demo/generated/
```

同一目录下如果放多份 Codegen YAML，默认会共用这个 `generated/` 目录，后生成的 bundle 会按当前 YAML 的
signature 覆盖旧 bundle。需要并存多份产物时，应为不同 YAML 使用不同目录，或在集成代码里显式传入
`codegen_artifact_dir`。

```text
examples/training_demo/
├── train_codegen_qwen3_moe.yaml
└── generated/
    ├── __init__.py
    ├── codegen_meta.json
    ├── modeling_Qwen3_Moe_gen_npu.py
    ├── modeling_Qwen3_Moe_gen_npu.py.diff
    └── configuration_*.py / source_*.py / 其他同级依赖文件（可选）
```

`generated/` 是整份 bundle 的根目录。训练时 loader 会在该目录下查找唯一的 `modeling_*_gen_npu.py`，并把
这个目录注册成隔离的 synthetic Python package，避免不同 artifact 的生成文件在 `sys.modules` 中互相覆盖。

`codegen_meta.json` 是 bundle 的索引和校验文件。它记录 signature、YAML 摘要、原始 modeling source 摘要、
并行维度、冻结后的 sharding plan、module override 记录、输出文件哈希和覆盖声明。训练前的 preflight 会读取
它判断 artifact 是否属于当前配置；训练时 runtime 也会使用其中的 frozen plan 完成参数分片和状态安装。

`modeling_Qwen3_Moe_gen_npu.py` 是训练真正导入的 generated model 文件。当前 inline 形态会把可静态展开的
模块替换和并行通信直接写进这个文件，例如：

- `RMSNorm` 替换 `Qwen3MoeRMSNorm`，原始 RMSNorm 类会从 generated model 中移除；
- `GQAAttention` 替换 `Qwen3MoeAttention`，attention 的 TP all-gather、CP all-gather 和 TP reduce-scatter
  会出现在 `forward()` 中；
- `GroupedExperts` 包裹 `Qwen3MoeExperts`，原始 experts 类会保留，因为它仍是 fused wrapper 的内部模块；
- `Qwen3MoeSparseMoeBlock.forward()` 会展开 EP dispatch、`ep_all_to_all`、本地 experts 计算和输出合并。

`modeling_Qwen3_Moe_gen_npu.py.diff` 是审计文件。它展示原始 modeling source 和 generated model 的统一
diff，用于检查 Codegen 到底替换了哪些类、插入了哪些 import、哪些 forward 被改写，以及哪些通信算子被展开。

可选的 `configuration_*.py`、`source_*.py` 或其他同级 Python 文件用于支持 remote-code 或相对导入场景。
当原始 modeling 文件依赖同级模块时，Codegen 会把必要依赖复制进 bundle，使 generated model 离开原包目录后
仍然可以 import。

bundle 以原子目录切换写入：先写进临时兄弟目录，再整目录换入，所以训练不会导入到半成品。

## generated model 里能看到什么

Codegen 产物的目标是让用户读到接近生产执行路径的模型代码，而不是只看到一个抽象的重分发接口。以
Qwen3-MoE 为例，generated model 中会出现如下结构。

### 模块替换说明

每个已静态展开的模块替换前都会有注释，说明 generated model 中的新模块来自哪个原始模块：

```python
# Codegen replacement: RMSNorm replaces Qwen3MoeRMSNorm.
# Codegen replacement: GQAAttention replaces Qwen3MoeAttention.
# Codegen replacement: GroupedExperts wraps Qwen3MoeExperts; the original class is kept as the wrapper input.
```

被完全替换且不再调用的原始类会从 generated model 中删除，例如 `Qwen3MoeRMSNorm` 和
`Qwen3MoeAttention`。仍作为 wrapper 输入使用的类会保留，例如 `Qwen3MoeExperts`。

### TP/CP attention 展开

`GQAAttention.forward()` 会直接读取外部安装的并行状态，然后显式调用通信算子：

```python
ps = get_parallel_state()
if ps.tp_enabled:
    hidden_states = ps.tp.all_gather(hidden_states, dim=1)

...

if ps.cp_enabled:
    cp_mesh = ps.cp_mesh
    key_states, value_states = flex_cp_allgather(
        key_states.contiguous(), value_states.contiguous(), 2, cp_mesh
    )
    attention_mask = _cp_offset_causal_mask(...)

...

if ps.tp_enabled:
    attn_output = ps.tp.reduce_scatter(attn_output, dim=1)
```

`TPOperators` 是 generated model 中的一层薄封装，内部直接调用 platform 暴露的可微 collective，例如
all-gather、all-reduce 和 reduce-scatter。这样产物中既能看到 production 通信形态，也能保留不同后端
对底层算子的适配能力。

### EP routed MoE 展开

EP 的 routed MoE 逻辑会写入 `Qwen3MoeSparseMoeBlock.forward()`。用户可以直接看到路由、dispatch、
all-to-all、本地 experts 计算和输出合并：

```python
ps = get_parallel_state()
if not ps.ep_enabled:
    return self._forward_impl(hidden_states)

ep_group = ps.ep_group
local_expert_count = self.experts.local_expert_count

topk_indices, topk_weights = MOE_ROUTER_ADAPTERS["qwen3moe"](self, hidden_states)
source_token_indices, flattened_expert_weights, dispatch_order, dispatched_states, \
    dispatched_expert_indices, send_counts, receive_counts = _prepare_ep_dispatch(...)

received_states = ep_all_to_all(dispatched_states, send_counts, receive_counts, ep_group)
received_indices = ep_all_to_all(dispatched_expert_indices, send_counts, receive_counts, ep_group).squeeze(-1)
local_outputs = self.experts(received_states, received_indices - expert_offset)
combined_expert_outputs = ep_all_to_all(local_outputs.contiguous(), receive_counts, send_counts, ep_group)
```

如果 YAML 中没有开启对应策略，或者当前模型/策略组合尚未支持静态展开，generated model 会保留原始计算路径或
使用当前支持的降级路径。是否真正展开，可以直接看 generated model 和 `.diff`。

## Codegen 如何根据 YAML 改写模型

Codegen 的输入仍然是现有 YAML 中的 `plan_overrides`，不需要额外扩展 YAML schema。生成阶段会按如下顺序处理：

1. 读取原始 Transformers modeling source；
2. 根据 `replace_module` 目标匹配已支持的静态替换规则，把构造函数替换成 generated model 中的新模块；
3. 删除已完全替换、且不再被调用的原始类；
4. 根据 CP/EP 等策略目标，对匹配到的模块做二次 forward 改写；
5. 写出 generated model、diff 和 `codegen_meta.json`；
6. 训练加载 generated model，并安装并行状态、参数分片和必要的 runtime 连接。

以示例 YAML 中的 Qwen3-MoE 配置为例：

```yaml
plan_overrides:
  - match:
      - "*.input_layernorm"
      - "*.post_attention_layernorm"
      - "model.norm"
    module_type: transformers.models.qwen3_moe.modeling_qwen3_moe.Qwen3MoeRMSNorm
    replace_module:
      _target_: hyper_parallel.models.qwen3_moe.adapter.replacements.replace_qwen3_moe_rms_norm

  - match: "*.self_attn"
    module_type: transformers.models.qwen3_moe.modeling_qwen3_moe.Qwen3MoeAttention
    replace_module:
      _target_: hyper_parallel.models.qwen3_moe.adapter.replacements.replace_qwen3_moe_flash_attention

  - match: "*.mlp"
    when: ep
    region_dispatch: false
    local_compute_fn:
      _target_: hyper_parallel.models.qwen3_moe.adapter.distributed.expert_parallel.qwen3moe_ep_compute_fn
```

上述配置会让 Codegen 在 generated model 中替换 RMSNorm 和 attention，并把 EP MoE 的 local compute 逻辑展开到
对应 `forward()` 中。其他模型也走同一套机制：是否能静态展开，取决于该 `replace_module` 或策略 target 是否已经
在 Codegen 中实现了对应的 source-level 规则。

## 注意事项

### 1. `force_hf` 覆盖了 Codegen

后端优先级是 `force_hf` > `modeling_backend` > `codegen`。如果 YAML 里 `force_hf: true`，即使设置了
`codegen:true`，模型也会回到原始 HF modeling。

日志会出现：

```text
codegen is enabled but force_hf=True, so the generated modeling file will not be used
```

修法是在 Codegen 训练中关闭 `force_hf`：

```bash
--model.force_hf=false
```

### 2. checkpoint 提示部分模型权重没有加载

如果 generated model 可以成功 import，sharding 也已经执行，但加载 checkpoint 时出现类似日志：

```text
did not load <N> owned model tensors
```

这表示当前 rank 应持有的一部分预训练参数没有从 checkpoint 正确恢复。此时不要继续用 loss 判断训练是否正常；
这次启动应视为无效。

先排查用户配置：

1. `pretrained_model_name_or_path` 是否指向与 YAML 模型类型一致的 checkpoint；
2. hyperparallel、PyTorch、torch-npu 和 Transformers 版本是否是项目声明支持的组合；
3. 是否复用了旧产物，可先清理后重新生成：

```bash
python -m hyper_parallel.codegen --verbose clean \
  --config examples/training_demo/train_codegen_qwen3_moe.yaml

python -m hyper_parallel.codegen --verbose generate \
  --config examples/training_demo/train_codegen_qwen3_moe.yaml
```

如果同一份 YAML 的原始 HF 路径可以正常加载，而 Codegen 路径仍报该问题，通常说明当前模型结构或权重转换规则
还缺少适配。请提交 issue，并附上模型名称、训练 YAML、checkpoint 加载日志、`codegen_meta.json`，以及原始 HF
路径可正常加载同一 checkpoint 的日志片段。

### 3. 修改配置后仍复用旧产物

修改训练步数、日志、优化器等运行参数不会改变 generated model，复用旧产物是正常行为。修改模型来源、并行规模、
`plan_overrides` 或 `replace_module` 后，应触发重新生成。

如果不确定当前产物是否正确，推荐清理后重新生成：

```bash
python -m hyper_parallel.codegen --verbose clean \
  --config examples/training_demo/train_codegen_qwen3_moe.yaml

python -m torch.distributed.run --nproc_per_node=4 \
  --module hyper_parallel.codegen --verbose generate \
  --config examples/training_demo/train_codegen_qwen3_moe.yaml
```

然后执行 fresh 检查：

```bash
python -m torch.distributed.run --nproc_per_node=4 \
  --module hyper_parallel.codegen --verbose check \
  --config examples/training_demo/train_codegen_qwen3_moe.yaml \
  --require-fresh
```

重新启动训练时，应确认日志出现新的生成或导入记录：

```text
codegen: generating artifact ...
codegen: imported generated modeling file ...
```

### 4. 自定义入口找不到 artifact

Codegen 按训练 YAML 的位置生成和查找 artifact：

```text
<yaml目录>/generated/
```

标准训练入口会自动完成 artifact 准备和路径传递。如果自定义入口绕过 YAML 解析，直接创建配置对象，可能会出现：

```text
modeling_backend='gen' requires codegen_artifact_dir
```

普通用户推荐始终通过 YAML 启动训练。如果必须使用自定义入口，需要在模型构建前显式准备 artifact，并把
artifact 目录传给模型构建参数。集成完成后至少确认以下日志：

```text
codegen: imported generated modeling file ...
codegen: sharding applied by the generated module; source_shard_info keys=<N>
```
