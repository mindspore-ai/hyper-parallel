# Codegen 使用指南

codegen 把并行规划与高性能模块替换显示的写入模型文件。codegen读取训练 YAML配置，离线推导 TP/CP/EP 等并行契约，将参数布局、边界重分发、前向边界改写和模块替换冻结进 artifact（`codegen_meta.json` + 生成版 modeling 文件）。训练时各 rank 加载同一份 artifact，由 runtime 读取其中冻结的 sharding plan 并执行分片。

codegen 同时承载两类能力：一类是将 sharding plan 固化为生成模型中的参数分片和边界通信逻辑；另一类是将 `replace_module` 等模型改写下沉到生成文件。当 YAML 用 `replace_module` 把标准注意力或 MoE 替换成融合实现（flash attention、grouped GEMM 专家、CP wrapper）时，codegen 会把替换入口写入模型类的 `__init__`，保证替换在模型构造阶段生效。

## 核心概念

| 概念 | 说明 |
|------|------|
| 产物 bundle | 位于训练 YAML 同级的 `generated/` 目录，含生成版 modeling 文件、`.diff` 差异、`codegen_meta.json` |
| signature | 对影响 artifact 的 canonical spec、modeling source 摘要和 Transformers 版本取 sha256 前 12 位；无关的 YAML 字段变化不会触发重新生成 |
| `codegen: true` | 打开 artifact 生成与预检；未显式指定其他后端且 `force_hf=false` 时，后端解析为 `gen` |
| `modeling_backend: gen` | 显式指定用生成物构建模型。**可选**——只要 `codegen: true` 且 `force_hf` 不为 true，后端就已经自动是 `gen`，不写也成立；写上只是让意图更明确 |
| `force_hf` | 逃生门，优先级最高，会把 `gen` 覆盖回 `hf`——见下方「注意事项」 |

## 前置条件

- 一份训练 YAML。离线训练推荐让 `pretrained_model_name_or_path` 指向本地 checkpoint；模型 ID 也可以使用，但必须能被当前 Transformers 环境和 source resolver 正常解析，并能取得对应的 config 与 modeling source。
- 一份能被 `parse_training_args` 解析的 YAML——codegen 通过它记录「这份 YAML 在哪」，从而定位产物目录。

---

## 快速开始

### 1. 在 YAML 里打开 codegen

推荐直接从已经验证过的 `examples/training_demo/train_codegen_qwen3_moe.yaml` 开始。自定义 YAML 时，**真正必需的是 `codegen: true` 并关闭模型配置里的 `force_hf`**——开了 `codegen` 后端就会自动是 `gen`，`modeling_backend: gen` 不是必需的。下面的示例保留了它（demo 里也这么写），只是为了把意图写得更明确：

```yaml
# train_codegen_qwen3_moe.yaml
codegen: true
modeling_backend: gen   # 可省：codegen:true 已隐含 gen

model:
  _target_: hyper_parallel.models._transformers.HyperAutoModelForCausalLM.from_pretrained
  pretrained_model_name_or_path: /path/to/Qwen3-30B-A3B
  force_hf: false
  torch_dtype: bfloat16
  attn_implementation: sdpa
```

codegen 的输入就是这份 YAML 本身，不需要第二份配置。signature 投影覆盖模型来源、`accelerator`（`tp/cp/ep/pp_size`、`sequence_parallel`、`loss_parallel`）、`plan_overrides`（包括其中的 `replace_module`）、`codegen` 和 `modeling_backend`。优化器、日志级别、训练步数和 YAML 注释等不会改变 artifact。

### 2. 生成产物（或直接交给 trainer）

训练 YAML 本身就会触发生成：训练启动时，trainer 在 `_setup` 与 `_build_model` 之间执行 codegen manager——产物不存在就生成，存在但签名变了就覆盖生成，存在且签名一致就复用（跳过生成，耗时约等于一次文件校验）。

也可以手动管理产物：

```bash
# 生成 / 复用产物（--verbose 用于显示 artifact 与 signature）
python -m hyper_parallel.codegen --verbose generate \
  --config examples/training_demo/train_codegen_qwen3_moe.yaml

# 预飞完整性校验：signature、文件哈希、meta、import 与冻结计划结构
python -m hyper_parallel.codegen --verbose check \
  --config examples/training_demo/train_codegen_qwen3_moe.yaml

# 只检查 artifact 是否与当前配置匹配，不触发重新生成
python -m hyper_parallel.codegen --verbose check \
  --config examples/training_demo/train_codegen_qwen3_moe.yaml \
  --require-fresh

# 删除某一 YAML 的产物
python -m hyper_parallel.codegen --verbose clean \
  --config examples/training_demo/train_codegen_qwen3_moe.yaml
```

`generate` 与 `check` 在 `codegen:false` 时都是 no-op；加上 `--verbose` 才会显示对应的跳过信息。`check` 不实例化完整模型，真实模型参数覆盖检查会在训练加载 generated model 时继续执行。

### 3. 正常启动训练

用 `python -m torch.distributed.run` 启动（注意：不要依赖 PATH 上的同名 `torchrun`，它可能被其他 CLI 应用遮蔽，导致 `--nproc_per_node` 等 PyTorch 分布式参数不可识别；用 `python -m torch.distributed.run` 最稳）：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python -m torch.distributed.run --nproc_per_node=8 \
  --module examples.training_demo.train_text \
  examples/training_demo/train_codegen_qwen3_moe.yaml \
  --model.pretrained_model_name_or_path=/path/to/Qwen3-30B-A3B \
  --model.force_hf=false \
  --dataset.data_path=./outputs/.../wikitext103_qwen3_4k_text_document \
  --training.train_iters=5
```

首次启动时应看到生成日志；复用时则显示 `reuse artifact`：

```text
codegen: generating artifact <yaml_dir>/generated (signature f463aefff4a4)
```

### 4. 判断是否真正使用 codegen 启动成功

生成 artifact 只表示 codegen manager 已经执行过，不等于训练模型一定来自 generated modeling。判断训练是否**真正使用 gen 后端**，需要同时满足以下日志条件。

首先，模型构建阶段应看到 generated modeling 文件被导入：

```text
codegen: imported generated modeling file .../modeling_<model>_gen_npu.py
```

这说明 `HyperAutoModel` 选择了 `modeling_backend=gen`，并从 artifact 目录加载生成后的 Python 模型文件。

其次，并行化阶段应看到 generated module 接管 sharding：

```text
codegen: sharding applied by the generated module; source_shard_info keys=<N>
```

这说明训练没有进入 native planner/applier 主链，而是由 generated modeling 文件接管分片：runtime 读取 artifact 中冻结的 sharding plan 并执行（generated 文件若定义了 `hyper_parallelize()` 入口则优先调用它），并向 FSDP/训练框架返回 `source_shard_info`。

最后，权重加载和训练 step 应继续正常推进：

```text
Loaded <N> model tensors from ...
step=1 ... training/foundation_loss=...
```

如果只看到 `generating artifact` 或 `reuse artifact`，不能证明模型使用了生成文件。`force_hf=true` 时 artifact 仍可能生成或复用，但建模后端会回到 HF。只有同时看到 generated modeling import、generated sharding 应用、checkpoint 正常加载和 step 推进，才可以判定 codegen 启动成功。

---

## 产物 bundle

生成物位于训练 YAML 同级的 `generated/` 目录下。当前目录规则是“每个示例目录一份 generated bundle”，因此同一示例目录下默认只保留一个 codegen 产物目录：

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

`generated/` 是整份 bundle 的根目录。训练时 loader 会在该目录下查找唯一的 `modeling_*_gen_npu.py`，并把这个目录注册成一个隔离的 synthetic Python package，避免不同 YAML 的生成文件在 `sys.modules` 中互相覆盖。

`__init__.py` 让 `generated/` 具备包目录形态。它通常不承载业务逻辑，主要用于让 generated modeling 和可选同级依赖在同一个 package 下解析相对导入。

`codegen_meta.json` 是 bundle 的索引和校验文件。它记录本次生成对应的 signature、YAML 摘要、原始 modeling source 摘要、并行维度、入口点、冻结后的 sharding plan、module override 记录、输出文件哈希和 covered 声明。训练前的 preflight 会读取它判断 artifact 是否属于当前配置；训练时 runtime 也会消费其中的 frozen plan。

`modeling_Qwen3_Moe_gen_npu.py` 是真正被训练导入的生成模型文件。它以原始 Transformers modeling 源码为基础，追加生成来源 banner，把可静态下沉的边界 forward 改写为显式 `hyper_redistribute()` 调用，并把 `replace_module` 下沉到模块的构造点。生成文件不再写入冻结计划常量或 `hyper_parallelize()` 入口——冻结计划保存在 `codegen_meta.json`，由 runtime 读取后执行分片；模型构建时，`replace_module` 相关 patch 直接由这个文件里的 `__init__` 构造逻辑生效。

`modeling_Qwen3_Moe_gen_npu.py.diff` 是审计文件。它展示原始 modeling 源码和 generated modeling 的统一 diff，用于检查 codegen 到底改了哪些位置：新增了哪些 import、哪些类的 forward 被下沉为 `hyper_redistribute()` 调用，以及哪些模块构造点被替换成融合实现。

可选的 `configuration_*.py`、`source_*.py` 或其他同级 Python 文件用于支持 remote-code 或相对导入场景。当原始 modeling 文件依赖同级模块时，codegen 会把必要依赖复制进 bundle，使 generated modeling 离开原包目录后仍然可以 import。

bundle 以**原子目录切换**写入：先写进临时兄弟目录，再整目录换入，所以永远不会导入到半成品。

---

## codegen 如何打入 patch

codegen 的 patch 不是在训练过程中动态改 Python 源码，而是在生成阶段把改写结果写入 generated modeling 文件。训练 import 的就是已经改写好的文件。

以普通 attention boundary 为例，原始 Transformers modeling 通常是一个直接执行计算的 `forward`：

```python
class Qwen3MoeAttention(nn.Module):
    def forward(
        self,
        hidden_states,
        position_embeddings,
        attention_mask,
        **kwargs,
    ):
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights
```

生成后，codegen 会保留原始实现为 `_forward_impl()`，再把公开的 `forward()` 改成“边界入口重分发 → 原始计算 → 边界出口重分发”的结构：

```python
_HYPER_BOUNDARY_Qwen3MoeAttention = {
    "in_src": {
        "hidden_states": {"tp": "S(1)", "cp": "S(1)", "ep": "R"},
        "position_embeddings": {"tp": "R", "cp": "R", "ep": "R"},
    },
    "in_dst": {
        "hidden_states": {"tp": "R", "cp": "S(1)", "ep": "R"},
        "position_embeddings": {"tp": "R", "cp": "R", "ep": "R"},
    },
    "out_src": {"output": {"tp": "P(sum)", "cp": "S(1)", "ep": "R"}},
    "out_dst": {"output": {"tp": "S(1)", "cp": "S(1)", "ep": "R"}},
}


class Qwen3MoeAttention(nn.Module):
    def _forward_impl(
        self,
        hidden_states,
        position_embeddings,
        attention_mask,
        **kwargs,
    ):
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    def forward(
        self,
        hidden_states,
        position_embeddings,
        attention_mask,
        **kwargs,
    ):
        args, kwargs = hyper_redistribute(
            ((hidden_states, position_embeddings, attention_mask), kwargs),
            _HYPER_BOUNDARY_Qwen3MoeAttention,
            mesh_context,
            _HYPER_MESH_DIM_NAMES,
            module=self,
        )
        outputs = self._forward_impl(*args, **kwargs)
        outputs = hyper_redistribute(
            outputs,
            _HYPER_BOUNDARY_Qwen3MoeAttention,
            mesh_context,
            _HYPER_MESH_DIM_NAMES,
        )
        return outputs
```

这段改写的核心含义是：原始计算逻辑仍在 `_forward_impl()` 中，codegen 只把并行边界通信显式插到它的前后。这样做可以把“当前 rank 输入是什么布局、进入算子前应该是什么布局、算子输出后应该变回什么布局”固定进文件，训练时不再重新推导。

对于 `replace_module`，原生路径是在 trainer 的动态 applier 中匹配模块并替换；codegen 在生成阶段就把替换下沉到源码里的模块构造点，也就是 `self.<attr> = <原始构造调用>(...)` 这条赋值本身，因此生成的 `__init__` 直接构造的就是高性能实现，模型构建时不再需要跑一遍替换：

```python
# 原始 modeling 源码
class Qwen3MoeDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.self_attn = Qwen3MoeAttention(config, layer_idx)

# 生成后：构造点替换为通用融合实现，原类作为 wrapper 输入保留
class Qwen3MoeDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.self_attn = GQAAttention(
            module=Qwen3MoeAttention(config, layer_idx),
            module_fqn='',
            context=None,
            attention_interface=run_qwen3_moe_flash_attention,
        )
```

命中的 FQN 仍会记录进 `codegen_meta.json` 的 `module_overrides`（并置 `covered.module_overrides=true`），供 preflight 核对，同时告诉 trainer 不必再对同一批规则执行自己的替换步骤。

在本文 demo 使用的 Qwen3-MoE 配置中，RMSNorm、attention fusion、CP wrapper、MoE local compute 等 patch 都由 generated artifact 承载。训练启动后只需要 import generated modeling，runtime 会读取 artifact 中冻结的 sharding plan 并执行分片，不需要再把同一批 `replace_module` 规则交给 native applier 重复处理。其他模型是否包含同类 patch，取决于对应 YAML 中的 `plan_overrides` 和 codegen 当前支持的静态下沉能力。

---

## 注意事项

### 1. `force_hf` 把 codegen 盖掉了

后端优先级：`force_hf` > `modeling_backend` > `codegen`。若 YAML 里 `force_hf: true`（native/offline 示例通常如此），即使加了 `codegen:true`，后端仍解析为 `hf`——训练照常跑、照常收敛，但**用的根本不是生成的 modeling 文件**。这是最危险的一种静默失败。

日志会出现：

```text
codegen is enabled but force_hf=True, so the generated modeling file will not be used
```

修法：跑 codegen 时关掉 `force_hf`，而不要只加 `codegen:true`：

```bash
--model.force_hf=false
```

### 2. checkpoint 提示部分模型权重没有加载

**问题现象**

生成模型可以成功 import，sharding 也已经执行，但加载 checkpoint 时出现类似日志：

```text
did not load <N> owned model tensors
```

这表示当前 rank 应该持有的一部分预训练参数没有从 checkpoint 正确恢复。此时不要继续用后续 loss 判断训练是否正常；即使训练流程没有立刻退出，这次启动也应视为无效。

**问题根因**

先区分两类情况。

如果 checkpoint 路径、模型类型、Transformers 版本或 YAML 中的 `replace_module` 配置写错，属于用户配置问题。典型表现是模型文件和权重本来就不匹配，native 路径也无法稳定加载同一份 checkpoint。

如果同一份 YAML 的 native 路径可以正常加载，而 codegen 路径出现该日志，通常属于 HyperParallel codegen 对当前模型结构或权重转换规则的支持不足。常见原因是生成模型中已经应用了结构融合或模块替换，但 codegen 尚未为这种生成后的结构注册对应的 checkpoint 转换规则，导致原始 checkpoint 中的参数名无法映射到生成模型中的目标参数名。

**解决方案**

用户侧只需要完成基础配置排查：

1. 确认 `pretrained_model_name_or_path` 指向与 YAML 模型类型一致的 checkpoint；
2. 确认使用项目声明支持的 HyperParallel 与 Transformers 版本；
3. 清理旧 artifact 后重新生成，避免复用历史产物：

```bash
python -m hyper_parallel.codegen --verbose clean \
  --config examples/training_demo/train_codegen_qwen3_moe.yaml

python -m hyper_parallel.codegen --verbose generate \
  --config examples/training_demo/train_codegen_qwen3_moe.yaml
```

如果上述条件都满足，native 路径可加载而 codegen 仍报该问题，请在 HyperParallel 仓库提交 issue，并附上以下信息：模型名称、训练 YAML、checkpoint 加载日志、generated artifact 中的 `codegen_meta.json`，以及 native 路径可以正常加载同一 checkpoint 的日志片段。

### 3. 修改配置后仍复用旧产物

**问题现象**

修改 YAML 中的模型来源、并行策略或模块替换配置后重新启动训练，日志仍显示：

```text
codegen: reuse artifact ...
```

或者训练仍然导入旧的 generated modeling 文件，表现为生成文件内容、运行行为或训练结果没有体现本次修改。

**问题根因**

通常有三种情况：

1. 修改的是训练步数、日志、优化器等运行参数。这类配置不会改变 generated model，复用旧产物是正常行为。
2. 修改的是模型来源、并行规模、`plan_overrides` 或 `replace_module`。这类配置应该触发重新生成；如果没有重新生成，说明当前 artifact 与预期不一致。
3. 直接编辑了 `generated/` 下的文件。generated 文件是产物，不是源码入口，手工修改不会成为稳定配置，后续检查或重新生成会覆盖这些改动。

**解决方案**

如果修改的是模型或并行相关配置，推荐先清理旧产物，再重新生成：

```bash
python -m hyper_parallel.codegen --verbose clean \
  --config examples/training_demo/train_codegen_qwen3_moe.yaml

python -m hyper_parallel.codegen --verbose generate \
  --config examples/training_demo/train_codegen_qwen3_moe.yaml
```

重新生成后执行检查：

```bash
python -m hyper_parallel.codegen --verbose check \
  --config examples/training_demo/train_codegen_qwen3_moe.yaml \
  --require-fresh
```

重新启动训练时，应确认日志中出现新的生成或导入记录：

```text
codegen: generating artifact ...
codegen: imported generated modeling file ...
```

如果只是修改训练步数、日志级别、学习率等运行参数，不需要清理或重新生成 artifact。

### 4. 自定义入口找不到 artifact

**问题现象**

使用自定义训练入口时，如果绕过标准 YAML 解析流程，直接在代码里创建 `TrainerConfig`，即使设置了
`codegen=True` 和 `modeling_backend="gen"`，也可能出现无法定位 artifact 的错误。典型日志包括：

```text
modeling_backend='gen' requires codegen_artifact_dir
```

或提示找不到 generated modeling 文件。

**问题根因**

codegen 按训练 YAML 的位置生成和查找 artifact：

```text
<yaml目录>/generated/
```

标准训练入口会从 YAML 读取配置，并自动完成 artifact 的生成、检查和路径传递。直接手写 `TrainerConfig`
时，配置里只有字段值，没有 YAML 文件位置，因此 codegen 不知道应该从哪个 `generated/` 目录加载产物。

**解决方案**

普通用户推荐始终通过 YAML 启动训练：

```bash
python -m torch.distributed.run --nproc_per_node=8 \
  --module examples.training_demo.train_text \
  examples/training_demo/train_codegen_qwen3_moe.yaml
```

如果必须使用自定义入口，需要在调用模型构建前显式准备 artifact，并把返回的 artifact 目录传给模型构建参数。
这属于集成开发场景，不是普通 YAML 训练的推荐用法。

集成完成后，至少确认以下日志：

```text
codegen: imported generated modeling file ...
codegen: sharding applied by the generated module; source_shard_info keys=<N>
```

如果自定义入口无法提供 YAML 路径和 artifact 目录，请改用标准 YAML 入口启动 codegen。
