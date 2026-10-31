# Module Family

Module Family 在构建阶段为同一种模块选择具体实现。用户通过统一的 replacement 入口填写选择参数，框架根据模块族声明的白名单规则选出实现，再复用现有的 Module Replacement 完成模型层匹配和实例安装。

当前内置 `GroupedExperts` Family（默认实现、MXFP8 和融合 MXFP8）以及 GQA/gated-GQA 的并行声明。更多模块可通过注册自己的 Family 接入。

## 使用 GroupedExperts Family

将以下片段加入完整训练配置。`match` 使用模型中实际的模块路径。需要限定源类型时，在 `replace.source_type` 中填写可导入的模块类。

```yaml
accelerator:
  tp_size: 1
  cp_size: 1
  ep_size: 1
  pp_size: 1

plan_overrides:
  - match: "model.layers.*.mlp.experts"
    replace:
      kind: module
      implementation:
        _target_: hyper_parallel.components.modules.GroupedExperts
        use_mxfp8: true
        use_fused_mxfp8: true
```

两个开关均为严格的布尔类型，默认值均为 `false`：

| `use_mxfp8` | `use_fused_mxfp8` | 构建结果 |
| --- | --- | --- |
| `false` | `false` | `GroupedExperts`，沿用源模型的参数 dtype |
| `true` | `false` | `MXFP8GroupedSwiGLU` |
| `true` | `true` | `FusedMXFP8GroupedSwiGLU` |
| `false` | `true` | 配置错误 |

省略两个开关等价于选择默认实现。`true` 与字符串 `"true"`、整数 `1` 是不同类型，后两者不能替代布尔值。

- `match` 确定要替换的实例；可选的 `replace.source_type` 校验源类型，`replace.exact_type: true` 排除子类。
- `replace.implementation._target_` 指定已注册的 baseline；选择参数确定本次使用的具体实现。
- MXFP8 变体支持三维 packed、无 bias、标准 SwiGLU，并要求路由权重在第二次 GMM 后由外层应用。隐藏维度和专家中间维度须为 32 的倍数。
- 当前 MXFP8 变体要求 `TP=CP=EP=PP=1`。Family 选择本身不增加并行支持；其他模块原有的并行能力保持各自契约。
- 运行环境需提供 MXFP8 GMM；融合另需 `npu_swiglu_mx_quant_with_dual_axis` 和 `_npu_swiglu_backward_mx_quant_with_dual_axis`。缺少能力时会报错。

这里的两个开关直接选择固定 MXFP8 路径，不通过 `training.low_precision` 选择数值策略。配置中不要再给同一替换条目绑定 W4/fake 等命名 dtype scheme。Scheme 驱动的 QAT 接口不属于本页描述的入口。

默认实现和低精实现共用 [GroupedExperts](grouped_experts.py) 的持久权重布局与 checkpoint 转换：Gate/Up 为 `[E,H,2I]`，Down 为 `[E,I,H]`。非 meta 源参数需要转置时复制已有值；布局已匹配时复用原 Parameter；meta 参数通过原 checkpoint 或初始化流程填值。应在替换完成后建立优化器。

## 配置如何生效

```mermaid
flowchart TD
    A[解析 replace.implementation] --> B[导入 baseline 并查询 Family]
    B --> C[补齐 selector 默认值并检查类型]
    C --> D{匹配完整参数组合}
    D -->|完整默认组合| E[选择 baseline]
    D -->|唯一合法规则| F[按 variants 路径导入实现]
    D -->|其余组合| X[抛出配置错误]
    E --> G[ReplacementTarget 保存选型]
    F --> G
    G --> H[原 replacement 匹配源模型层]
    H --> I[Target.build 注入源模块和 context]
    I --> J[检查参数与调用契约并安装实例]
    J --> K[forward 调用已选实例的计算方法]
```

选择发生在配置构建阶段。训练 forward 使用已经安装的模块，不再解释规则。没有注册 Family 的普通 replacement 继续使用原 Target 解析和执行流程。

`ReplacementTarget` 同时保留用户请求的 baseline、显式选择参数和最终 callable。重新选择或序列化时保留配置意图；真正构建时，选择参数已从普通构造参数中移除。

```python
from hyper_parallel.components.modules import GroupedExperts
from hyper_parallel.trainer.config.replacement import ReplacementTarget

target = ReplacementTarget(
    GroupedExperts,
    target_path="hyper_parallel.components.modules.GroupedExperts",
    use_mxfp8=True,
)
fused = target.replace(use_fused_mxfp8=True)
baseline = fused.replace(use_mxfp8=False, use_fused_mxfp8=False)
serialized = fused.to_dict()
```

`.replace()` 返回新的构建配置，不修改已有模型实例。关闭低精与融合时应同时恢复合法组合。`to_dict()` 保存原 baseline 路径和显式选择参数，省略的 selector 不补写。

## Family 声明与规则

[family.py](family.py) 提供以下声明：

| 类型／字段 | 含义 |
| --- | --- |
| `SelectorSpec(value_type, default)` | 选择参数的精确类型与 baseline 默认值 |
| `VariantRule(when, variant)` | 完整的具名参数组合与对应变体名 |
| `ModuleFamily.name` | 模块族名称 |
| `ModuleFamily.baseline` | 对外入口 callable，通常是模块类 |
| `ModuleFamily.variants` | 变体名到可导入 callable 路径的映射 |
| `ModuleFamily.selector_specs` | 本模块族接受的选择参数 |
| `ModuleFamily.variant_rules` | 明确允许的非默认参数组合 |

允许集合由完整默认组合和全部显式规则组成。缺省字段先补默认值；非默认组合必须完整匹配一条规则。没有部分匹配、规则优先级或未命中后回退。

注册时检查规则字段完整性、参数类型、变体引用和条件重复。每条规则须包含全部且仅有已声明的 selectors，默认组合无需重复登记。选择参数采用 `bool`、`int`、有限 `float` 或 `str`；矩阵大小、配置对象等普通构造参数由选中的实现处理。

注册表保存声明的只读副本，并按 baseline 对象身份查询。一个子类不会自动取得父类的 Family。`variants` 使用字符串路径延迟导入；导入失败会报错。构造接口中的 `module`、`module_fqn`、`context` 等运行时字段不能声明为 selector。

## 扩展一个 Family

### 1. 实现具有相同契约的替换模块

下面的 CPU Linear 示例展示注册方法，可放入接入方的 `my_project/linear_family.py`。它只匹配原生 `torch.nn.Linear`，两种实现共用源 weight、bias 和训练状态。

```python
from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from hyper_parallel.components.modules.family import (
    ModuleFamily,
    SelectorSpec,
    VariantRule,
    register_module_family,
)
from hyper_parallel.models.replacement import module_replacement


@module_replacement
class ExampleLinear(nn.Linear):
    """Baseline replacement preserving native Linear parameters."""

    def __init__(
        self,
        *,
        module: nn.Linear,
        module_fqn: str = "",
        context: Mapping[str, Any] | None = None,
    ) -> None:
        # Avoid allocating and initializing a second set of weights.
        nn.Module.__init__(self)
        self.in_features = module.in_features
        self.out_features = module.out_features
        self.register_parameter("weight", module.weight)
        self.register_parameter("bias", module.bias)
        self.train(module.training)


@module_replacement
class MatmulLinear(ExampleLinear):
    """Alternative local computation with the same forward signature."""

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = input @ self.weight.t()
        return output if self.bias is None else output + self.bias


register_module_family(ModuleFamily(
    name="example_linear",
    baseline=ExampleLinear,
    variants={"matmul": "my_project.linear_family.MatmulLinear"},
    selector_specs={"implementation": SelectorSpec(str, "reference")},
    variant_rules=(
        VariantRule({"implementation": "matmul"}, "matmul"),
    ),
))
```

继承可以复用参数与外部流程；也可以让 Family 选择满足同一 replacement 契约的工厂函数。`@module_replacement` 声明构造入口，实际替换仍由原执行器完成。

### 2. 配置模块路径与选择参数

确保 `my_project` 在训练进程中可导入。解析以下 target 时会导入上面的文件并执行注册：

```yaml
plan_overrides:
  - match: "model.projection"
    replace:
      kind: module
      implementation:
        _target_: my_project.linear_family.ExampleLinear
        implementation: matmul
```

省略 `implementation` 或填写 `reference` 选择 baseline；填写 `matmul` 选择另一实现；其他值报错。新增这个 Family 无需修改通用 resolver。

如果注册放在独立文件，应用需在解析配置前导入该文件。每个 baseline 只注册一次，不应在 forward 或逐层构造循环中重复注册。

### 3. 验证共享契约

替换实现应保持源模块的 forward 调用协议、训练／冻结状态，以及参数和 buffer 的注册契约。参数名称或布局变化时，通过 `make_transforms()` 提供 checkpoint 转换，并保留已加载源参数的正确数值。保留 hook、别名和参数身份的条件由原 replacement 执行器检查。

新增变体时，应在真实配置解析和 replacement 路径中验证合法／非法组合、选中 callable、初始参数、输出、输入／权重梯度及更新；测试助手不应在替换后额外填权重来弥补构造行为。数值算法有变化时，为该实现单独定义精度和设备验收条件。

给已有 Family 增加 selector 时，更新所有规则，使每条条件仍然完整；默认值应保持旧配置的选择结果。增加变体只能扩展已验证的计算组合，不自动改变并行或底层算子的支持范围。

## 代码与测试入口

| 位置 | 职责 |
| --- | --- |
| [family.py](family.py) | 注册、白名单匹配与构造参数检查 |
| [grouped_experts.py](grouped_experts.py) | 内置 Family 登记、参数与路由契约 |
| [mxfp8_grouped_swiglu.py](mxfp8_grouped_swiglu.py) | 两种低精模块的本地计算 |
| [trainer/config/replacement.py](../../trainer/config/replacement.py) | 选型配置、复制与序列化 |
| [trainer/config/resolver.py](../../trainer/config/resolver.py) | replacement 专用解析入口 |
| [trainer/config/parallelism.py](../../trainer/config/parallelism.py) | 配置到替换规格的接线 |
| [models/replacement.py](../../models/replacement.py) | 原模块匹配、构造检查与实例安装 |
| [Family 单元测试](../../../tests/ut/components/modules/test_module_family_selectors.py) | 规则与注册契约 |
| [配置集成测试](../../../tests/ut/trainer/test_module_family_replacement.py) | 实际 replacement、参数和配置往返 |
| [MXFP8 单设备测试](../../../tests/torch/expert_parallel/test_family_mxfp8_local.py) | 非融合与融合的设备计算验证 |

通用高性能模块的安装与替换说明见 [modules README](README.md)。


## Family 的 CP / EP 适配声明

`ModuleFamily.parallel_adapters` 将具体类型绑定到适配函数的导入路径。Family 保留选型与绑定声明，
适配实现按职责放在现有 `distributed/context_parallel/` 和 `distributed/expert_parallel/` 目录。
它们采用与模型 adapter 相同的 wrapper / factory 格式，不依赖某个具体模型的 adapter。

| Family | 适配入口 | 复用能力 |
| --- | --- | --- |
| GQAAttention / GatedGQAAttention | `context_parallel/gqa_attention.py::gqa_attention_cp_wrapper` | 已有 CP 通信、mask 处理与 forward rewriter |
| GroupedExperts | `expert_parallel/experts.py::grouped_experts_contract` | 已有 EP dispatch/combine、分块和本地专家计算 |

Family 的 `parallel_adapters` 按最终具体类型查询，查询时才导入实现。CP 返回 wrapper，
EP 返回 Experts 契约提供者。变体不自动继承 baseline 的并行能力；缺少依赖或不满足支持条件时明确报错。

CP 通过已有 `inner_wrapper` / `inner_target="self"` 注入 GQA 边界，保留选定的
`attention_interface` 以及投影、归一化、RoPE 和输出流程。首版使用已有 all-gather 策略；
外部 mask 须覆盖全局 K/V 并包含模型所需的因果限制。
cache、packed sequence、sliding window 和 attention weights 输出暂不支持。
显式 CP wrapper 优先；与 `region_dispatch=True` 冲突时报错。

EP 契约声明专家参数轴、全局专家数和本地计算入口。原生和 GroupedExperts 复用已有
`_local_swiglu_expert_forward`，GroupedExperts 沿其已有分支调用 `forward_expert_major`。现有 Planner 在已有 MoE
边界内将专家参数归属到 `.experts` 子边界，复用参数切分、forward 安装和通信流程。
`Experts.forward(hidden_states, top_k_index, top_k_weights)` 接收原模型的路由结果；
父级 MoE 保留原有 Router、共享专家、辅助输出和返回协议，无需增加模型专属 Router/combine 回调。
路由权重由现有 combine 应用一次。本地计算在 Experts 自己的受管理 forward 内执行，
保留模块调用 hooks；不会递归调用自身 forward。

替换后的 GroupedExperts 通过现有 Family 声明契约。未加入 Family 的原生 Experts 可以显式注册：

```python
from hyper_parallel.distributed.expert_parallel.experts import (
    register_experts_contract, packed_swiglu_experts_contract,
)

register_experts_contract(MyPackedExperts, packed_swiglu_experts_contract)
```

此示例仅适用于明确采用 `activation(gate) * up`、down 投影后应用路由权重的原生实现，
权重布局分别为 `[E, 2I, H]` 和 `[E, H, I]`，入口参数需与上述 Experts 接口一致。
注册是语义声明，不通过参数形状猜测模型行为。不同契约应提供自己的 `ExpertsContract`。
当前替换契约仅支持三维 packed 权重且无 bias；不扩展 MXFP8 等数值变体。
原生注册按具体类型匹配，不自动继承。

当 `ep_size > 1` 且 Experts 已声明契约时自动绑定。父级 MoE 的旧 EP factory、实例 forward
替换或通信 dispatcher 会与 Experts EP 冲突，构建时明确报错，避免重复通信。
未声明契约的模块仍沿用已有模型适配路径；并行度为 1 时不绑定。
`derive=False` 仍由用户完整声明方案。Qwen3-MoE 的 grouped-experts 训练配置移除旧父级 EP 注入。

验证入口：

```bash
python -m pytest -q tests/ut/components/modules/test_family_parallel_adapters.py
python -m pytest -q tests/ut/core/expert_parallel/test_ep_npu_token_dispatch.py tests/ut/core/expert_parallel/test_ep_dispatch_chunks.py
python -m pytest -q tests/ut/dual_mode_dtensor/test_ep_moe_local_region.py
python -m pytest -q tests/torch/expert_parallel/test_module_parallel_adapters.py
```

测试使用实际 GQAAttention / GroupedExperts，设备 GMM 和 attention 采用 CPU 计算。
两进程 builder → Planner → apply 检查覆盖 CP、原生/替换 Experts 自动 EP、二维/三维输入，
并比较输出与输入、路由器、共享专家及专家参数梯度；覆盖 production/validate、辅助输出和 hooks。
EP-only 且无活跃 dense 轴时，普通子边界沿用本地执行，避免普通参数与 DTensor 输入混用。
原有模型级 EP 路径另行覆盖不均匀/空路由、
同步/分离 dispatch、1/2 个 chunk 和模块 hooks。
真实 Ulysses 通信、设备异步重叠、完整 HF 模型训练/checkpoint/FSDP 与 NPU/GPU 整模精度性能仍待验证。
