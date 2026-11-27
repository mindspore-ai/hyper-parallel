# 算子后端选型与接入

[English](op_select.md) | **简体中文**

OpSelect 让高性能模块通过统一的算子接口调用不同硬件上的实现。训练时可按设备自动选择融合算子，也可用同一份模型配置把单个算子族切换到基础 PyTorch 参考实现，用于精度定位。

已接入 RMSNorm、SwiGLUMLP、SharedExpert、两套 GroupedExperts、MegaGate、mHC Post/PipelinedMhc，以及 GQA/MLA 的 RoPE 调用。普通 EP/MoonEP 的重排与恢复也通过公共接口选型。注意力和序列算子的其余调用迁移由第二阶段的对应任务推进；配置只作用于已经调用公共 Ops 的路径。

## 快速开始

在 Trainer YAML 的根节点添加 `ops`。下面的配置将 RMSNorm 固定到参考实现，其余算子族按各自的默认规则选择。

```yaml
ops:
  rms_norm: native
```

| 配置值 | 行为 |
|------|------|
| 省略 `ops`、写 `ops: {}` 或省略某个族 | 统一使用 `auto`，按当前 variant 和设备的候选顺序选择。 |
| `auto` | 针对当前设备和实际调用的变体，按优先级选择首个可用实现。 |
| `native` | 固定使用基础 PyTorch 算子组成的参考实现。 |
| `torch_npu`、`cann`、`sdpa` 等实现名 | 固定使用指定实现。设备、变体或依赖不匹配时，在该变体首次调用时报错。 |

使用 `auto` 时，NPU 和 GPU 的训练配方可复用同一组 `ops` 配置。需要固定后端时，分别在各自的 YAML 中填写对应实现名。例如 NPU 配方可写 `attention: torch_npu`，CUDA 配方可写 `attention: sdpa`。

也可以使用 `--ops.<算子族>=<实现>` 新增或覆盖一个算子族的配置，例如 `--ops.deepseek_sparse_attention=native`。现有模型的模块替换及配方迁移随后续接入 PR 提供；本阶段可以通过下文的 Python 示例验证选型。

## 选型如何生效

公共算子不接收 `backend` 参数，后端统一由 YAML 的 `ops:` 或 `ops.set_backends()` 配置。旧 `mhc_post`（包括 functional 入口）、`grouped_swiglu`、`causal_conv1d`、`gdn`、`kda` 的 `backend=` 调用应迁移为对应 family 配置；未配置的族统一使用 `auto`。

1. **配置加载。** Trainer 校验算子族名、重复键和值类型。显式实现名必须在该族的注册目录中存在。
2. **模块调用。** 算子族对应一个 YAML 键。每个变体对应一种调用契约，规定输入、输出、布局、数据类型和梯度。例如 `swiglu(x)` 接收拼接后的 Gate/Up，`swiglu_clamped(gate, up, limit)` 接收分开的输入。模块通过函数入口确定变体，`ops.swiglu` 为两个变体提供同一个选择值。
3. **调用时解析。** Trainer 通过 `ops.set_backends()` 设置进程配置。公共函数显式传入执行设备，选择器按 family、variant、配置后端和完整 device 构造四字段缓存 key，不读取 Tensor，也不按 dtype、shape 或参数位置选型。`auto` 按该 variant、设备的注册顺序尝试加载依赖。后端选定后，输入限制由实现检查；输入或内核错误直接报出，不切换实现。

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

配置仍按 family 填写；公共函数决定调用的 variant，`auto` 按该 variant 和输入设备选择候选。显式后端也必须支持本次调用的 variant。下表每个 family 仅在首行填写，后续首列留空表示沿用该 family；每个 variant 独立一行，列出注册与选型能力，不代表所有 dtype、shape 和依赖版本均已完成硬件精度验收。

| `ops` 配置键（family） | variant | 用途 | Ascend NPU `auto` 候选 | CUDA GPU `auto` 候选 | CPU `auto` 候选 |
|------|------|------|------|------|------|
| `rms_norm` | `standard` | 常规 RMSNorm | `torch_npu` | `native` | `native` |
|  | `fp32_affine` | FP32 归一化与仿射缩放 | `torch_npu` | `native` | `native` |
| `swiglu` | `standard` | 拼接 Gate/Up 输入的 SwiGLU | `torch_npu` | `native` | `native` |
|  | `clamped_fp32` | 独立 Gate/Up 输入、FP32 裁剪 | `torch_npu` | `native` | `native` |
| `rope` | `half` | 半分布局旋转，支持部分 head 维度 | `torch_npu`, `native` | `native` | `native` |
|  | `interleaved` | 相邻对布局旋转，支持部分 head 维度 | `torch_npu`, `native` | `native` | `native` |
|  | `interleaved_single` | 单 Tensor 相邻对旋转，避免重复计算 | `torch_npu`, `native` | `native` | `native` |
| `mhc_post` | `standard` | 变换输出与残差流混合 | `cann`, `omni`, `native` | `native` | `native` |
| `sinkhorn` | `standard` | mHC 混合矩阵归一化 | `omni`, `native` | `native` | `native` |
| `aggregate_hidden` | `standard` | `[S,B,H]` 带输出掩码的因果深度卷积，无 bias/残差；Omni 限 BF16 | `omni`, `native` | `native` | `native` |
| `grouped_matmul` | `standard` | 按专家排列的单次无 bias 分组矩阵乘 | `torch_npu` | `torch_grouped_mm`, `native` | `native` |
| `grouped_swiglu` | `standard` | 非量化专家的 Gate、Up 和 Down 计算 | `torch_npu`, `native` | `torch_grouped_mm`, `native` | `native` |
| `moe_router` | `deepseek_v41` | V4.1 TopK 路由 | `megagate`, `native` | `native` | `native` |
| `moe_token_permute` | `standard` | 按专家顺序重排 token | `torch_npu`, `native` | `native` | `native` |
| `moe_token_unpermute` | `standard` | 路由权重加权与 token 还原 | `torch_npu`, `native` | `native` | `native` |
| `attention` | `vision` | 视觉注意力 | `torch_npu`, `native` | `sdpa`, `native` | `sdpa`, `native` |
|  | `gqa` | 分组查询注意力 | `torch_npu`, `native` | `sdpa`, `native` | `sdpa`, `native` |
|  | `mla` | 投影后状态的多头潜在注意力 | `torch_npu`, `native` | `sdpa`, `native` | `sdpa`, `native` |
|  | `mla_with_stats` | Packed 因果 MLA，可返回 teacher softmax 统计量 | `torch_npu`, `native` | `sdpa`, `native` | `sdpa`, `native` |
| `gdn` | `zero_state` | Gated DeltaNet 局部零状态计算 | `triton_ascend`, `ascendc`, `native` | `native` | `native` |
| `kda` | `zero_state` | Kimi Delta Attention 局部零状态计算 | `triton_ascend`, `ascendc`, `native` | `native` | `native` |
| `causal_conv1d` | `standard` | GDN/KDA 的局部因果短卷积与 SiLU | `triton_ascend`, `ascendc`, `native` | `native` | `native` |
| `dsa_indexer` | `compressed_causal_topk` | V4.1 压缩索引 | `cann`, `native` | `native` | `native` |
|  | `packed_causal_topk` | V3.2 packed 文档内索引 | `cann`, `native` | `native` | `native` |
| `deepseek_sparse_attention` | `compressed_banks` | V4.1 压缩 KV 稀疏注意力 | `cann`, `native` | `native` | `native` |
|  | `absorbed_with_stats` | V3.2 稀疏 MLA 与统计量 | `cann`, `native` | `native` | `native` |
|  | `absorbed_with_parameter_sink` | 带 parameter sink 的稀疏 MLA | `omni`, `native` | `native` | `native` |
| `dsa_indexer_kl` | `compressed_teacher` | V4.1 压缩 teacher 的 Indexer KL | `cann`, `native` | `native` | `native` |
|  | `packed_sparse_teacher` | V3.2 packed 稀疏 teacher 的 Indexer KL | `cann`, `native` | `native` | `native` |
|  | `packed_dense_teacher` | V3.2 packed dense warm-up teacher 的 Indexer KL | `cann`, `native` | `native` | `native` |

`moe_token_permute` 和 `moe_token_unpermute` 是两个独立算子族，均使用 `standard` variant，可分别配置后端。原 `ops.moe_token_reorder` 配置需拆为这两个键；原公共接口 `moe_token_reorder` / `moe_token_restore` 分别改为 `moe_token_permute` / `moe_token_unpermute`。两者共享槽位映射契约，可以混用 native 与 torch_npu 后端。

Attention 公共入口与模块目录使用一致的完整名称：`grouped_query_attention`、`multi_latent_attention`（含 `multi_latent_attention_with_stats`）和 `deepseek_sparse_attention_*`。稀疏注意力配置键及族文件统一为 [`deepseek_sparse_attention`](../../hyper_parallel/components/ops/impl/deepseek_sparse_attention.py)，原 `dsa_sparse_attention` 配置和入口、`gqa_attention` 和 `mla_attention*` 入口需迁移到对应的新名称。

`functional.aggregate_hidden` 转调公共 `components.ops.aggregate_hidden`，使用独立的
`ops: {aggregate_hidden: native}` 配置。输入为 `[S,B,H]`，滤波器为 `[K,H]`，布尔掩码为 `[B,S]`；
最后一个滤波器 tap 对应当前位置，左侧越界按零填充。掩码只将对应输出清零，不屏蔽这些位置对后续输出的贡献。
`functional.mome` 组合布局适配、可选 bias、输出掩码和残差，仅调用公共 `ops.aggregate_hidden`，
后端通过 `aggregate_hidden` 配置，不再维护独立的 `mome` 算子族。聚合先恢复输入 dtype，再加 bias；
低精度下不等同于卷积与 bias 融合后仅舍入一次。Omni 的注册不代表目标设备精度已验收。

### 后端依赖与使用限制

- `torch_grouped_mm` 要求 BF16 输入、CUDA compute capability 8.0 或更高，以及 PyTorch 提供的 grouped matmul。`sdpa` 使用 PyTorch 的 `scaled_dot_product_attention`。[Grouped Matmul 实现](../../hyper_parallel/components/ops/impl/grouped_matmul.py) · [Attention 实现](../../hyper_parallel/components/ops/impl/attention.py)
- NPU 上的 `torch_npu`、`cann` 和 `omni` 需要对应的可选扩展和算子符号。`cann` 按变体使用 `cann_ops_transformer`、`torch_npu` 或两者。mHC post 的 CANN 路径要求 FP16 或 BF16 的子层输出与残差，以及 FP32 的混合系数。`megagate` 需要独立的 HyperMulticore 原生组件，并要求非空的批量输入和 `sqrtsoftplus` 路由打分。[安装指南](../installation.zh.md) · [mHC post 实现](../../hyper_parallel/components/functional/mhc_post.py) · [MoE 路由实现](../../hyper_parallel/components/ops/impl/moe_router.py)
- GDN、KDA 和短卷积的公共接口提供本地零状态计算。跨 rank 递归状态和卷积 halo 仍由原 CP wrapper 管理，其接入迁移在后续 PR 中提供。[KDA 上下文并行](kda_context_parallel.zh.md) · [GDN 与 KDA 的 TND 路径](linear_attention_tnd_cp.md)
- DeepSeek V4.1 的 CSA2 算子契约检查 BF16、文档分段和 TopK 等条件。RMSNorm 的 `use_fused_ops` 保留旧成员选择，两个成员的实际算子后端均由 `ops.rms_norm` 控制；权重表示及舍入顺序保持不变。尚未迁移的模块配置仍保持原有行为。
- `moe_token_unpermute` 保留调用者传入的路由权重精度；native 按输入 dtype 的类型提升规则完成加权和累加，最终输出 dtype 与 expert states 一致，NPU 后端直接传入原始权重。需要权重舍入时，在调用前显式使用 `weights.to(states.dtype)`；算子内部不再预先舍入权重。

## 在高性能模块中接入

SwiGLU 的 clamp 与精度策略统一在 `functional.swiglu`；GQA/MLA 的 BSND 旋转统一复用 `functional.rotary_embedding.apply_rotary_pos_emb_bsnd`，其 interleaved 分支调用单 Tensor 变体。Q/K 半布局组合使用 `functional.apply_rotary_pos_emb`，替代原 `ops.rope_qk`。旧 functional token permutation/unpermutation 通过适配保留双返回值、int32 逆映射及 `probs=None` 语义，内部同样使用进程后端配置。

module 可以调用 functional 或 ops：需要复用算法组合、布局适配或精度策略时调用 functional，仅需要原子算子时可直接调用 ops。functional 通过 ops 公共接口执行算子，后端选择统一留在 ops。模块继续管理权重、状态、投影和 checkpoint 转换。以下展示原子算子的直接调用：

```python
from hyper_parallel.components.ops import swiglu


def forward(self, x):
    gate_up = self.linear_fc1(x)
    return self.linear_fc2(swiglu(gate_up))
```

最终目标是由 ops 完成本地算子的硬件解耦：functional/module 的算法与接口调用不随 CPU/CUDA/NPU 或后端切换而改变，不直接加载厂商扩展、调用硬件内核、导入 `ops.impl` 或按设备/fused 开关绕过选型。普通 Torch 组合和跟随输入设备分配张量可以保留。当前仍有旧 RoPE、Grouped Matmul 兼容分支及 EP 重排选择等缺口，不能将支持矩阵解读为完整模块迁移已完成；详见[硬件边界审视](../contributing/op_select_development.zh.md#43-当前基础与专家接入的硬件边界审视)。

## 设置和查询后端

Python 入口在模型构建或编译前设置当前进程的后端，算子按实际输入选型。[配置与查询实现](../../hyper_parallel/components/ops/factory.py)

| 接口 | 语义 |
|------|------|
| `ops.set_backends(selections: Mapping[str, str]) -> None` | 完整替换当前进程配置，并清理解析缓存和使用统计。省略的族统一使用 `auto`。`{}` 恢复所有族的 `auto` 选择。 |
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

在 Transformers、TorchTitan 或自定义训练程序的每个 worker 中设置进程配置，用户自己的计算代码可以直接调用公共算子，无需包裹训练流程。以下 CPU 示例验证这一独立入口：

```python
import torch

from hyper_parallel.components.ops import ops, rms_norm

ops.set_backends({"rms_norm": "native"})
hidden_states = torch.randn(2, 8, requires_grad=True)
weight = torch.ones(8, requires_grad=True)
output = rms_norm(hidden_states, weight, eps=1e-6)
output.sum().backward()
print(ops.used_backends())
```

第三方 step-end callback 可以读取 `ops.used_backends()`。基础与专家模块可直接用于普通 PyTorch 和 Transformers Trainer，无需训练流程 wrapper。离线 Qwen3-MoE 的替换、梯度及 Trainer 单步回归见[接入测试](../../tests/ut/auto_models/models/qwen3_moe/test_qwen3_moe_op_select.py)，真实后端与模块的验收要求见[开发与验收标准](../contributing/op_select_development.zh.md)。

## 注册新算子族或后端

完整开发流程、两层精度基线、测试接入和交付标准见 [OpSelect 算子开发流程与验收标准](../contributing/op_select_development.zh.md)。下面介绍注册接口与目录约定。

1. **从参考实现确定契约。** 在 `components/ops/impl/<配置键>.py` 同时维护契约与实现，文件名与 YAML 键一致。用 `Protocol` 描述参考实现所需的入参、返回值和梯度语义，用 `OpSpec` 指定族名和变体名。同族可以有多个变体。内核专有的参数由实现绑定，例如 Grouped Matmul 的 NPU 分组编码。[Grouped Matmul 契约与适配](../../hyper_parallel/components/ops/impl/grouped_matmul.py)
2. **登记实现。** 用 `register_family` 一次登记族内全部变体，再对每个变体调用 `register_impl`。`load` 是无参加载器，可返回函数、实现了 `__call__` 的对象或 `nn.Module`。可在实现内部封装 `torch.autograd.Function` 来定义反向。训练参数由模型模块持有，在调用时传入。不登记基于 Tensor 的兼容性回调。公共入口检查通用契约，所选后端检查其特有限制；dtype、shape 和布局不参与选型或缓存 key。可选包或算子缺失时，加载器使用 `ImplementationUnavailable` 告知自动选择继续检查下一候选。[注册接口](../../hyper_parallel/components/ops/registry.py) · [依赖检查](../../hyper_parallel/components/ops/_dependency.py)
3. **设置默认顺序并导出。** 新族在 `catalog.py` 中组装，目录封存前用 `auto_policy.py` 的 `set_auto_order` 设置每个变体和设备的默认顺序。在族文件中显式定义公共函数，再在 `ops/__init__.py` 中显式导入并加入 `__all__`。注册表只管理后端契约和实现，不生成公共函数。只新增同一变体的后端时，复用现有入口。[包入口](../../hyper_parallel/components/ops/__init__.py) · [自动选择策略](../../hyper_parallel/components/ops/auto_policy.py)
4. **验证前向、反向和选型。** 用相同输入对比融合与参考实现的结果和梯度，覆盖相关 dtype、布局和边界输入，再通过一个实际模块调用检查配置与首步输出。

以 SwiGLU 为例，`impl/swiglu.py` 同时登记两个变体。以下片段展示 `standard` 变体的实现登记、默认顺序和包入口。[SwiGLU 完整注册](../../hyper_parallel/components/ops/impl/swiglu.py)

```python
# impl/swiglu.py 中的契约
class SwiGLUFn(Protocol):
    """将 Gate/Up 投影均分，计算 SiLU(Gate) * Up；投影权重由调用者管理。"""

    def __call__(self, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """沿 dim 均分 Gate 和 Up，返回 SiLU(Gate) * Up。

        Args:
            x: 沿 dim 拼接的 Gate/Up 投影，该维度长度须为偶数。
            dim: 拼接投影所在维度。

        Returns:
            dim 维度减半的激活结果。
        """


SWIGLU = OpSpec[SwiGLUFn](config_key="swiglu")

# Public dispatcher in impl/swiglu.py (full docstring omitted here)
def swiglu(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Split packed projections and compute SiLU(gate) * up."""
    operation = _select_for_call(SWIGLU, device=x.device)
    return operation(x, dim)

# impl/swiglu.py 中的注册函数
registry.register_family("swiglu", specs=(SWIGLU, SWIGLU_CLAMPED))
registry.register_impl(SWIGLU, name="native", devices=("cpu", "cuda", "npu"), load=_load_native)
registry.register_impl(SWIGLU, name="torch_npu", devices=("npu",), load=_load_npu)
# 同一注册函数继续登记 SWIGLU_CLAMPED 的实现。

# auto_policy.py
registry.set_auto_order(SWIGLU, device="npu", candidates=("torch_npu",))
registry.set_auto_order(SWIGLU, device="cuda", candidates=("native",))
registry.set_auto_order(SWIGLU, device="cpu", candidates=("native",))

# Explicit export in ops/__init__.py
from .impl.swiglu import swiglu

# Call sites
from hyper_parallel.components.ops import swiglu
```

Protocol 描述后端实现接收的参数、返回值和实现语义。公共函数独立维护完整的类型标注和 Google 风格 docstring，说明算法、公共参数、形状、布局、精度和返回值。两者签名可能不同，例如 Attention 的公共入口接收 module，后端契约接收已解析的 causal、scale 等参数。

Protocol 用于表达后端类型契约，当前没有强制的静态类型检查门禁。注册器和选择器的 callable 检查不校验完整签名、Tensor 形状或梯度；这些约束由公共入口和契约测试验证。

公共函数调用 `factory._select_for_call(spec, device=execution_device)`，执行设备由算子语义明确指定，例如 `x.device` 或 `query.device`；第一个公共参数可以是 module 或 Python 标量。缓存 key 为 `(family, variant, selection, device)`，不保存 Tensor、dtype、shape 或排除候选。每次读取配置，配置替换时清空缓存。输入验证在每次实际执行中生效，不因缓存命中而跳过。`registry.py` 持有共享实例，`catalog.py` 组装并封存，factory 不导入 catalog，避免循环依赖。

IDE 补全、参数提示、跳转和文档直接来自公共函数源码及 `ops/__init__.py` 中的显式导入。新增或删除接口时，同步修改族文件、包导入和 `__all__`，并运行公共 API 和相关算子测试。保留并打包 `py.typed`，不再生成或维护 `.pyi`，构建和 CI 无需执行接口生成步骤。
