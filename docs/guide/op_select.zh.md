# 算子后端选型与接入

[English](op_select.md) | **简体中文**

OpSelect 让高性能模块通过统一的算子接口调用不同硬件上的实现。训练时可按设备自动选择融合算子，也可用同一份模型配置把单个算子族切换到基础 PyTorch 参考实现，用于精度定位。

## 快速开始

在 Trainer YAML 的根节点添加 `ops`。下面的配置将 RMSNorm 固定到参考实现，其余算子族按各自的默认规则选择。

```yaml
ops:
  rms_norm: native
```

| 配置值 | 行为 |
|------|------|
| 省略 `ops`、写 `ops: {}` 或省略某个族 | 默认使用 `auto`。兼容旧配方的个别调用点可能提供自己的默认值，见下文的使用限制。 |
| `auto` | 针对当前设备和实际调用的变体，按优先级选择首个可用实现。 |
| `native` | 固定使用基础 PyTorch 算子组成的参考实现。 |
| `torch_npu`、`cann`、`sdpa` 等实现名 | 固定使用指定实现。设备、变体或依赖不匹配时，在该变体首次调用时报错。 |

使用 `auto` 时，NPU 和 GPU 的训练配方可复用同一组 `ops` 配置。需要固定后端时，分别在各自的 YAML 中填写对应实现名。例如 NPU 配方可写 `attention: torch_npu`，CUDA 配方可写 `attention: sdpa`。

也可以使用 `--ops.<算子族>=<实现>` 新增或覆盖一个算子族的配置。以 DeepSeek V4.1 文本示例为例，下面只切换稀疏注意力，Indexer 与 Indexer KL 继续使用 YAML 中各自的配置。[DeepSeek V4.1 启动指南](../../examples/training_demo/deepseek_v41/deepseek_v41_quick_start.md)

```bash
bash examples/training_demo/deepseek_v41/run_deepseek_v41_online.sh \
  /path/to/DeepSeek-V4.1-Flash \
  --ops.dsa_sparse_attention=native
```

`ops` 对模型中已经调用 OpSelect 的算子生效。`plan_overrides` 可以为模型安装由 OpSelect 驱动的高性能模块或并行 wrapper。例如 Qwen3-MoE 配方先用 `plan_overrides` 安装 RMSNorm 和 GQA 替换，再通过 `ops.rms_norm` 和 `ops.attention` 设置后端。[Qwen3-MoE 训练配方](../../hyper_parallel/models/qwen3_moe/recipes/train.yaml)

## 选型如何生效

1. **配置加载。** Trainer 校验算子族名、重复键和值类型。显式实现名必须在该族的注册目录中存在。
2. **模块调用。** 算子族对应一个 YAML 键。每个变体对应一种调用契约，规定输入、输出、布局、数据类型和梯度。例如 `swiglu(x)` 接收拼接后的 Gate/Up，`swiglu_clamped(gate, up, limit)` 接收分开的输入。模块通过函数入口确定变体，`ops.swiglu` 为两个变体提供同一个选择值。
3. **调用时解析。** Trainer 在模型构建前通过 `ops.set_backends()` 设置当前进程的配置。公共算子根据实际输入的设备、dtype、调用条件和后端依赖选择实现，缓存键包含契约和影响选型的条件。同进程的模型和线程共享配置。自动选择跳过已判定不兼容或缺少可选依赖的候选。实现开始执行后，内核错误直接报出。

Torch 设备类型使用 `npu`、`cuda` 和 `cpu`。解析时使用当前输入张量的设备和 dtype，支持模型先在 `meta` 上构建后再加载到训练设备。[算子目录与运行时](../../hyper_parallel/components/ops/catalog.py) · [Trainer 模型构建](../../hyper_parallel/trainer/base.py)

### 确认实际后端

第 1 个全局训练步完成后，Trainer 通过 `ops.used_backends()` 汇总各 rank 当前配置生效后已用的 `(Op, Variant, Backend)`，由全局 rank 0 去重、排序并打印到 `stderr`。这张表独立于日志级别，同一变体本步使用了多个后端时会分行列出。[首步输出实现](../../hyper_parallel/trainer/base.py)

```text
Fused operator backends used in training step 1

Op        Variant      Backend
--------  -----------  ---------
rms_norm  fp32_affine  torch_npu
swiglu    standard     native
```

做精度对照时，保持 checkpoint、输入、随机种子、dtype 和并行配置一致，每次只把一个算子族改成 `native`。先用首步表格确认改动命中了目标变体，再比较输出、梯度或一步更新结果。如果目标族没有出现在表格中，检查首步输入是否覆盖对应分支，以及该层是否安装了调用 OpSelect 的模块或 wrapper。

## 算子支持矩阵

各设备列按优先级从左到右列出 `auto` 候选。列出的后端均可显式填写。表中所有变体也都可在三种设备上显式选择 `native`。RMSNorm、SwiGLU 和单次 Grouped Matmul 在 NPU 上的 `auto` 只有 `torch_npu` 候选，需要参考实现时请显式填写 `native`。[注册目录](../../hyper_parallel/components/ops/catalog.py) · [自动选择顺序](../../hyper_parallel/components/ops/auto_policy.py)

| `ops` 配置键 | 变体与用途 | Ascend NPU | CUDA GPU | CPU |
|------|------|------|------|------|
| `rms_norm` | `standard`、`fp32_affine`，常规和 FP32 仿射缩放 | `torch_npu` | `native` | `native` |
| `swiglu` | `standard`、`clamped_fp32`，拼接输入和独立输入裁剪 | `torch_npu` | `native` | `native` |
| `rope` | `half`、`interleaved`，均支持部分 head 维度旋转 | `torch_npu`, `native` | `native` | `native` |
| `mhc_post` | `standard`，变换输出与残差流混合 | `cann`, `omni`, `native` | `native` | `native` |
| `sinkhorn` | `standard`，mHC 混合矩阵归一化 | `omni`, `native` | `native` | `native` |
| `mome` | `standard`，DSA 输出侧的带掩码因果卷积与残差 | `omni`, `native` | `native` | `native` |
| `grouped_matmul` | `standard`，按专家排列的单次无 bias 分组矩阵乘 | `torch_npu` | `torch_grouped_mm`, `native` | `native` |
| `grouped_swiglu` | `standard`，非量化专家的 Gate、Up 和 Down 计算 | `torch_npu`, `native` | `torch_grouped_mm`, `native` | `native` |
| `moe_router` | `deepseek_v41`，V4.1 TopK 路由 | `megagate`, `native` | `native` | `native` |
| `moe_token_reorder` | `standard`、`weighted_restore`，专家顺序重排和加权还原 | `torch_npu`, `native` | `native` | `native` |
| `attention` | `vision`、`gqa`、`mla`、`mla_with_stats` | `torch_npu`, `native` | `sdpa`, `native` | `sdpa`, `native` |
| `gdn` | `zero_state`，Gated DeltaNet 局部零状态计算 | `triton_ascend`, `ascendc`, `native` | `native` | `native` |
| `kda` | `zero_state`，Kimi Delta Attention 局部零状态计算 | `triton_ascend`, `ascendc`, `native` | `native` | `native` |
| `causal_conv1d` | `standard`，GDN 和 KDA 的局部因果短卷积与 SiLU | `triton_ascend`, `ascendc`, `native` | `native` | `native` |
| `dsa_indexer` | `compressed_causal_topk`、`packed_causal_topk`，V4.1 压缩索引和 V3.2 文档内索引 | `cann`, `native` | `native` | `native` |
| `dsa_sparse_attention` | `compressed_banks`、`absorbed_with_stats`，V4.1 压缩 KV 和 V3.2 稀疏 MLA | `cann`, `native` | `native` | `native` |
| `dsa_sparse_attention` | `absorbed_with_parameter_sink`，带 parameter sink 的稀疏 MLA | `omni`, `native` | `native` | `native` |
| `dsa_indexer_kl` | `compressed_teacher`、`packed_sparse_teacher`、`packed_dense_teacher`，含 V3.2 dense warm-up | `cann`, `native` | `native` | `native` |

### 后端依赖与使用限制

- `torch_grouped_mm` 要求 BF16 输入、CUDA compute capability 8.0 或更高，以及 PyTorch 提供的 grouped matmul。`sdpa` 使用 PyTorch 的 `scaled_dot_product_attention`。[Grouped Matmul 实现](../../hyper_parallel/components/ops/families/grouped_matmul.py) · [Attention 实现](../../hyper_parallel/components/ops/families/attention.py)
- NPU 上的 `torch_npu`、`cann` 和 `omni` 需要对应的可选扩展和算子符号。`cann` 按变体使用 `cann_ops_transformer`、`torch_npu` 或两者。mHC post 的 CANN 路径要求 FP16 或 BF16 的子层输出与残差，以及 FP32 的混合系数。`megagate` 需要独立的 HyperMulticore 原生组件，并要求非空的批量输入和 `sqrtsoftplus` 路由打分。[安装指南](../installation.zh.md) · [mHC post 实现](../../hyper_parallel/components/functional/mhc_post.py) · [MoE 路由实现](../../hyper_parallel/components/ops/families/moe_router.py)
- GDN、KDA 和短卷积的融合实现各有 dtype、形状与依赖约束。`ops` 作用于本地零状态计算和接入的 Ulysses 路径。跨 rank 递归状态和卷积 halo 由 CP wrapper 管理。纯 Ulysses 配方里的旧 `backend` 字段可为未设置的族提供默认值，显式 `ops` 值优先。[KDA 上下文并行](kda_context_parallel.zh.md) · [GDN 与 KDA 的 TND 路径](linear_attention_tnd_cp.md)
- DeepSeek V4.1 的 CSA2 融合实现检查 BF16、文档分段和 TopK 等条件。让已有的 `SharedCompressedDSAAttention` 替换使用 `ops` 时，请省略旧的 `use_fused_ops` 字段。分块或容量受限的 EP 使用参考 token 重排。[DeepSeek V4.1 启动指南](../../examples/training_demo/deepseek_v41/deepseek_v41_quick_start.md)
- RMSNorm 与 OffsetRMSNorm 的显式 `ops.rms_norm` 配置对本地和 NPU 成员均生效。省略该族时，`use_fused_ops=False` 的本地成员使用 Torch 参考计算，`use_fused_ops=True` 保留 NPU 输入约束。[RMSNorm 模块](../../hyper_parallel/components/modules/rms_norm.py)
- 对 `components.modules.moe.GroupedExperts`，旧参数 `use_grouped_mm=False` 会让未配置的 `grouped_swiglu` 默认使用参考实现。显式 `ops.grouped_swiglu` 值优先。[分组专家实现](../../hyper_parallel/components/modules/moe.py)

## 在高性能模块中接入

从 `hyper_parallel.components.ops` 导入对应函数，在 `forward` 里直接调用。模块继续管理权重、投影和 checkpoint 转换。以下是 `SwiGLUMLP` 的调用方式。[SwiGLUMLP 实现](../../hyper_parallel/components/modules/swiglu_mlp.py)

```python
from hyper_parallel.components.ops import swiglu


def forward(self, x):
    gate_up = self.linear_fc1(x)
    return self.linear_fc2(swiglu(gate_up))
```

## 设置和查询后端

Python 入口在模型构建或编译前设置当前进程的后端，算子按实际输入选型。[配置与查询实现](../../hyper_parallel/components/ops/factory.py)

| 接口 | 语义 |
|------|------|
| `ops.set_backends(selections: Mapping[str, str]) -> None` | 完整替换当前进程配置，并清理解析缓存和使用统计。省略的族恢复默认选择，通常为 `auto`。`{}` 恢复全部默认选择。 |
| `ops.get_backends() -> dict[str, str]` | 返回显式配置的独立副本，缺省的族以字典中缺少该键表示。 |
| `ops.used_backends() -> tuple[tuple[str, str, str], ...]` | 返回当前配置生效后成功选用的 `(op, variant, backend)` 三元组，结果去重。查询保留累计记录。 |

配置无效时抛出 `ValueError`，原配置、缓存和使用统计继续有效。输入字典及查询副本可独立修改。

```python
import torch

from hyper_parallel.components.ops import ops, swiglu

ops.set_backends({"swiglu": "native"})
x = torch.randn(2, 8, requires_grad=True)
output = swiglu(x)
output.sum().backward()

print(ops.get_backends())    # {'swiglu': 'native'}
print(ops.used_backends())   # (('swiglu', 'standard', 'native'),)
```

在 `torchrun` 的每个 worker 中执行 `ops.set_backends(config.get("ops", {}))`。一次完整的前向、反向及 checkpoint 重计算使用同一配置。顺序 A/B 在当前执行完成后设置下一份配置，同进程内所有模型会使用新配置。并发运行不同后端配置可使用独立进程。

`torch.compile` 在捕获时对配置状态建立 guard，切换配置后的调用匹配新配置。CUDA/NPU 静态图在切换后重新捕获。

## 在第三方训练框架中使用

### Transformers

下面构建一个本地小型 Llama，用高性能 `RMSNorm` 接管对应层并复用原权重，之后使用 Transformers 自身的 `Trainer` 训练一步。示例在 CPU 上通过 `native` 验证接入。NPU 训练时将 `use_cpu` 设为 `False`，并把后端设为 `auto` 或 `torch_npu`。

```python
from transformers import AutoModelForCausalLM, LlamaConfig, Trainer, TrainingArguments
from transformers.models.llama.modeling_llama import LlamaRMSNorm

from hyper_parallel.components.modules.rms_norm import RMSNorm
from hyper_parallel.components.ops import ops

ops.set_backends({"rms_norm": "native"})
config = LlamaConfig(
    vocab_size=32, hidden_size=16, intermediate_size=32,
    num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
)
model = AutoModelForCausalLM.from_config(config)
for name, module in list(model.named_modules()):
    if isinstance(module, LlamaRMSNorm):
        model.set_submodule(name, RMSNorm(module=module))

tokens = list(range(8))
trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir="./output/op_select",
        max_steps=1,
        per_device_train_batch_size=1,
        use_cpu=True,
        report_to="none",
    ),
    train_dataset=[{"input_ids": tokens, "labels": tokens}],
)
trainer.train()
print(ops.used_backends())
```

原生激活重计算再次调用公共算子时会使用同一进程配置。模型的 checkpoint、优化器和训练调度通过 Transformers 的接口设置。

### TorchTitan 和自定义 PyTorch 程序

在各 worker 的模型构建入口设置后端，在模型定义或 recipe 的现有模块替换位置创建对应高性能层。

```python
from hyper_parallel.components.modules.rms_norm import RMSNorm
from hyper_parallel.components.ops import ops

# 每个训练 worker 在模型构建前执行。
ops.set_backends({"rms_norm": "auto", "swiglu": "auto"})

# 模型定义中使用对应高性能层。
norm = RMSNorm(hidden_size=128, eps=1e-6)
```

随后由原有流程完成模型构建、并行策略、checkpoint、编译和训练。模型内的 SwiGLU 等计算可直接导入 `hyper_parallel.components.ops` 中的函数。配置字段由接入方的配置系统加载，Python 接口接收算子族到实现名的映射。

第三方框架可在已有 step-end callback 读取 `ops.used_backends()` 并汇总打印。HyperParallel Trainer 自带的首步输出使用同一查询接口。

## 注册新算子族或后端

1. **从参考实现确定契约。** 在 `components/ops/families/<配置键>.py` 同时维护契约与实现，文件名与 YAML 键一致。用 `Protocol` 描述参考实现所需的入参、返回值和梯度语义，用 `OpSpec` 指定族名和变体名。同族可以有多个变体。内核专有的参数由实现绑定，例如 Grouped Matmul 的 NPU 分组编码。[Grouped Matmul 契约与适配](../../hyper_parallel/components/ops/families/grouped_matmul.py)
2. **登记实现。** 用 `register_family` 一次登记族内全部变体，再对每个变体调用 `register_impl`。`load` 是无参加载器，可返回函数、实现了 `__call__` 的对象或 `nn.Module`。可在实现内部封装 `torch.autograd.Function` 来定义反向。训练参数由模型模块持有，在调用时传入。`check_compatibility(x)` 根据主输入的设备和 dtype 判断兼容性，返回 `None` 表示可用，返回字符串给出拒绝原因。形状、布局及多个输入之间的约束由公共入口检查，并把候选限制纳入解析缓存。可选包或算子缺失时，加载器使用 `ImplementationUnavailable` 告知自动选择继续检查下一候选。[注册接口](../../hyper_parallel/components/ops/registry.py) · [依赖检查](../../hyper_parallel/components/ops/_dependency.py)
3. **设置默认顺序并导出。** 新族在 `catalog.py` 中组装，目录封存前用 `auto_policy.py` 的 `set_auto_order` 设置每个变体和设备的默认顺序。新变体的函数从包入口导出，简单契约可用 `_dispatch(SPEC)`。只新增同一变体的后端时，复用现有入口。[包入口](../../hyper_parallel/components/ops/__init__.py) · [自动选择策略](../../hyper_parallel/components/ops/auto_policy.py)
4. **验证前向、反向和选型。** 用相同输入对比融合与参考实现的结果和梯度，覆盖相关 dtype、布局和边界输入，再通过一个实际模块调用检查配置与首步输出。

以 SwiGLU 为例，`families/swiglu.py` 同时登记两个变体。以下片段展示 `standard` 变体的实现登记、默认顺序和包入口。[SwiGLU 完整注册](../../hyper_parallel/components/ops/families/swiglu.py)

```python
# families/swiglu.py 中的契约
class SwiGLUFn(Protocol):
    def __call__(self, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """沿 dim 均分 Gate 和 Up，返回 SiLU(Gate) * Up。"""


SWIGLU = OpSpec[SwiGLUFn](config_key="swiglu")

# families/swiglu.py 中的注册函数
registry.register_family("swiglu", specs=(SWIGLU, SWIGLU_CLAMPED))
registry.register_impl(SWIGLU, name="native", devices=("cpu", "cuda", "npu"), load=_load_native)
registry.register_impl(SWIGLU, name="torch_npu", devices=("npu",), load=_load_npu)
# 同一注册函数继续登记 SWIGLU_CLAMPED 的实现。

# auto_policy.py
registry.set_auto_order(SWIGLU, device="npu", candidates=("torch_npu",))
registry.set_auto_order(SWIGLU, device="cuda", candidates=("native",))
registry.set_auto_order(SWIGLU, device="cpu", candidates=("native",))

# components/ops/__init__.py
swiglu: SwiGLUFn = _dispatch(SWIGLU)
```
