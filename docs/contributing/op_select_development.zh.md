# OpSelect 算子开发流程与验收标准

本文面向接入算子族、扩展融合后端、整改高性能模块或补充算子精度测试的开发者。完整流程为：

**定义契约 → 建立 native 基线 → 接入后端 → 整改模块 → 精度验收 → 集成与性能验收。**

已有契约下新增后端时，复用公共接口和参考实现；仅补测试时，围绕现有实现扩展用例。验收范围由本次改动及声明的支持能力决定。

接入可以分步交付选型流程、算子实现、functional/module 调用点及整网配置，但必须逐项声明实际完成范围。当前基础与专家主路径已做部分调用迁移；这不代表同族全部兼容入口、相邻模块或完整网络已经实现硬件解耦，详见第 4.3 节。

用户配置、支持矩阵和注册 API 见[算子后端选型与接入](../guide/op_select.zh.md)。本文规定开发交付物及其验证方式。Agent 执行入口为仓库 skill [op-select-dev](../../.agent/skills/op-select-dev/SKILL.md)。

## 1. 开发前明确支持范围

先列出本次涉及的 `(op, variant, backend, device, dtype)`，补充实际模块使用的代表性形状、布局、可选参数及依赖版本。设备名称使用 PyTorch 的 `cpu`、`cuda`、`npu`。

区分三类工作：

| 工作 | 必须维护的内容 |
| --- | --- |
| 新增算子族或 variant | 契约、native、公共入口、注册、自动选择策略、精度用例及文档 |
| 为已有 variant 新增后端 | 等价适配、能力限制、注册、共享硬件用例；需要时调整自动选择顺序 |
| 整改模块或补测试 | 公共调用点或共享用例；保留既定契约，并验证受影响的执行路径 |

OpSelect 管理本地算子调用。DTensor 分布式规则开发遵循 [dist-op-dev](../../.agent/skills/dist-op-dev/SKILL.md)。模块涉及 TP、CP、EP 等真实通信时，再按[测试规则](../../.agent/rules/testing.md)和[分布式规则](../../.agent/rules/distributed.md)验证相关路径。

## 2. 从 native 确定契约

### 2.1 定义可观察行为

在 `components/ops/impl/<配置键>.py` 中用 `Protocol` 描述接口，用 `OpSpec` 定义算子族和 variant。文件名必须与 YAML 的 `ops` 配置键一致。

契约应明确：

- 数学计算、输入输出结构、shape、dtype、布局及广播规则。
- 可选参数及默认值、裁剪和掩码语义、packed 文档边界和前缀偏移。
- 哪些输入和参数参与求导，哪些返回值必须 detach，以及未使用参数返回 `None` 还是零梯度。
- 空输入、空专家、无有效候选、分块尾部和非法输入的行为。
- 是否允许修改输入，以及随机算子的 dropout 等行为。

公共入参按 native 计算所需的能力确定。影响通用语义的参数应在契约中体现；内核专有的编码和控制参数由适配实现固定或转换，以实现同一契约。例如，当前 Grouped Matmul 是无 bias 的分组乘法，NPU 的 `group_type=0`、`group_list_type=0` 留在后端适配中。若要增加通用 bias 能力，应同步扩展 native、Protocol、实现及测试。

输入协议、输出语义、精度行为或梯度语义存在实质差异时，定义不同 variant 和公共函数。仅更换内核实现时复用原 variant。不同 variant 可以使用不同 Protocol；YAML 仍按算子族选择后端，公共函数确定实际 variant。

### 2.2 先验证参考实现

native 使用基础 PyTorch 运算表达契约，并通过独立公式或独立算法验证。独立参考不能直接调用待验证的 native，也不能与其共用关键计算函数。

对真正支持 FP64 的参考路径补充 `gradcheck`。内部固定 FP32 或经过显式低精度舍入的路径，使用独立解析梯度、递推算法等方法验证，不把转成 FP64 当作已经提高了内部计算精度。

边界输入也应检查梯度连接。例如，空专家的权重梯度按契约应为零时，需要保留计算图；teacher 输入按契约不参与求导时，应保持梯度隔离。

## 3. 适配后端并维护选型

算子族文件同时维护声明、native 和后端适配。实现登记与设备优先级分别维护：

| 位置 | 职责 |
| --- | --- |
| [impl/](../../hyper_parallel/components/ops/impl/) | Protocol、OpSpec、显式公共函数、参数适配、实现及注册函数 |
| [registry.py](../../hyper_parallel/components/ops/registry.py) | 共享注册表实例、`register_family`、`register_impl` 和解析接口 |
| [factory.py](../../hyper_parallel/components/ops/factory.py) | 进程级配置、统一实现选择、缓存及实际后端记录 |
| [catalog.py](../../hyper_parallel/components/ops/catalog.py) | 组装并封存内置算子目录 |
| [auto_policy.py](../../hyper_parallel/components/ops/auto_policy.py) | 每个 variant、设备的 `auto` 候选顺序 |
| [包入口](../../hyper_parallel/components/ops/__init__.py) | 显式导入公共函数并维护静态 `__all__`；调用级约束归属族文件 |

共享注册表位于 registry，依赖方向为 `catalog → impl → factory → registry`；factory 不导入 catalog，避免族文件调用选择器时形成循环。Protocol 约束后端类型，公共函数维护调用者签名，两者分别保留；当前没有强制静态类型检查门禁，运行时 callable 检查不能代替签名、shape 和数值验收。

`register_impl` 的 `load` 是无参加载器，可返回函数、可调用对象或 `nn.Module`，实现内部可以使用 `torch.autograd.Function` 定义自定义反向。构造参数由加载器绑定；训练参数由模型模块持有，并在调用时传入，避免隐藏在全局缓存的实现对象中。

选型与输入验证遵循以下分工：

1. 公共函数显式传入执行设备：`_select_for_call(spec, device=x.device)`；不按第一个参数推断。
2. 缓存 key 只有 `(family, variant, selection, device)`，不使用 dtype、shape 或输入约束排除候选。
3. 公共入口检查通用契约，所选实现逐次检查后端限制；缓存命中不绕过输入验证。
4. `auto` 按设备注册顺序加载候选，依赖缺失可以继续。输入不支持和内核运行错误直接报出，不自动改用 native。
5. 注册表不接受 `check_compatibility`；选择器不接受 `native_only` 或 `blocked_backends`。

注册片段见[注册新算子族或后端](../guide/op_select.zh.md#注册新算子族或后端)。调整 `auto` 优先级时，同时记录所依据的正确性、能力范围和性能证据。

公共接口在 `components/ops/impl/<配置键>.py` 中显式定义，由 `components/ops/__init__.py` 显式导入并列入 `__all__`。Protocol 保留为后端契约，公共函数维护完整的签名、算法说明及 Google 风格参数文档，并调用统一选择器完成适配和后端调用。

新增或删除公共接口时，同步维护族文件、包导入和 `__all__`，运行 `tests/ut/components/ops/test_factory.py` 的静态导出检查及相关算子测试。源码直接提供 IDE 补全、跳转和文档；构建继续打包 `py.typed`，无需生成声明文件或安装提交钩子。详见[注册与公共 API](../guide/op_select.zh.md#注册新算子族或后端)。

## 4. 在模块中使用公共接口

module 可以从 functional 和 ops 导入，functional 可以从 ops 导入。职责按以下方式划分：

| 层 | 职责 | 调用方式 |
| --- | --- | --- |
| module | 参数、状态、checkpoint 转换和模块流程 | 组合功能复用 functional；原子算子可直接调用 ops |
| functional | 无状态算法组合、布局适配、精度策略和旧接口适配 | 通过 ops 公共接口执行算子，不导入 module |
| ops | Protocol、公共算子契约、后端选择及具体实现 | 不反向依赖 functional 的组合逻辑 |

### 4.1 最终目标：ops 完成本地算子的硬件解耦

切换已声明支持的执行设备或后端时，functional/module 中的算子调用和算法组合应保持不变。ops 负责识别设备、读取后端配置、加载可选扩展、转换内核专有格式及参数、调用内核和检查后端能力。上层只表达算法契约，不根据 CPU/CUDA/NPU 或后端名称选择实现。

具体边界如下：

- functional/module 可以用原生 Torch 表达硬件无关的组合、布局、精度策略及状态处理，也可用 `x.device` 跟随输入分配张量；这些操作无需全部注册为算子。
- functional/module 不通过 `device.type`、硬件可用性、`backend` 或 fused 开关切换已接入算子的实现，不直接调用厂商内核，也不导入 `ops.impl` 中的实现或加载器。历史兼容入口同样应满足这一要求。
- 因数学语义产生的状态、mask、dtype 或 shape 分支可以保留，但不能借此绕过后端配置；内核专有的对齐、编码和限制在 ops 中处理。所选实现不支持输入时显式报错，不由上层改用另一实现。
- 模块继续管理训练参数、checkpoint、辅助损失和调度。DTensor 布局与通信基础设施按各自模块规则治理；其中的本地算子不能以硬件判断代替 ops 选型。第三方源模块识别与参数迁移应单独记录，不与算子执行混为一项验收结论。

### 4.2 复用组合并维护兼容入口

不要为了统一 import 路径，在不同模块重复实现组合逻辑。以标准 SwiGLU 原子调用为例：

```python
import torch

from hyper_parallel.components.ops import swiglu


class Activation(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return swiglu(x)
```

需要 clamp 或激活精度策略时，模块复用 `functional.swiglu.local_swiglu` 或 `swiglu_activation`；BSND 布局的单分支 RoPE 复用 `functional.rotary_embedding.apply_rotary_pos_emb_bsnd`。该适配调用 `ops.rope_interleaved_single`，不通过重复传入同一 Tensor 计算两份旋转结果。Q/K 的半布局组合由 `functional.apply_rotary_pos_emb` 管理，原 `ops.rope_qk` 调用应迁移到该入口。

已有 functional 名称可以作为兼容入口保留，但应转调 ops 并显式维护历史契约。例如，functional token permutation 返回两个结果，逆映射保持 int32；ops 返回三个结果，逆映射使用 int64。functional unpermutation 的 `probs=None` 按 top-k=1 适配。不要同时保留直连旧内核和经过 OpSelect 的两套实现。

外部模块无需感知 `components/ops/impl/` 内部路径、调用 `ops.create` 或在初始化时绑定后端函数。后端仅通过已有 YAML 配置或 `ops.set_backends` 在每个 worker 中设置；公共接口不接收 `backend`，选择器不接收 `default_backend`，省略的族统一使用 `auto`。不需要包裹 Trainer 的训练流程。

`ops.set_backends` 完整替换当前进程配置，并清理解析缓存和使用统计。测试必须保存并恢复原配置；一次前向和反向应在同一配置下完成，结束后再切换到另一组对照实现。

整改已有模块时，保留原权重所有权、参数命名、checkpoint 语义和计算顺序。确认首步实际执行覆盖了目标分支，并通过 `ops.used_backends()` 检查命中的 `(op, variant, backend)`。

### 4.3 当前基础与专家接入的硬件边界审视

2026-11-28，基于 `7534aada` 的静态调用链审视。结论是**已迁移的主路径大多满足目标，完整接入范围尚未全部完成硬件解耦**。本表不以算子注册或已有精度测试通过代替调用链审视，也不代表此次重新执行了硬件测试。

| 路径 | 当前结论 | 证据与剩余工作 |
| --- | --- | --- |
| RMSNorm / OffsetRMSNorm | 已接入计算路径满足 | [模块](../../hyper_parallel/components/modules/common/rms_norm.py)仅处理 scale、舍入顺序并调用公共 ops。旧 `use_fused_ops` / `npu` 成员名仍用于兼容构造，两成员实际使用同一进程选型，不决定执行后端 |
| SwiGLUMLP 及显式精度策略 | 已接入路径满足 | [functional 策略](../../hyper_parallel/components/functional/swiglu.py)组合公共 SwiGLU；[MLP](../../hyper_parallel/components/modules/mlp/swiglu_mlp.py)管理投影与参数，普通/FP32 clamp 的选择表达数学契约 |
| 旧 GroupedExperts / SharedExpert 的激活适配 | 仍有配置绕过 | [旧专家](../../hyper_parallel/components/modules/grouped_experts.py)和[共享专家](../../hyper_parallel/components/modules/moe/shared_expert.py)的源模块适配仍检查 `use_fused_swiglu`；关闭时可直接执行 `act_fn(gate) * up`。已识别的标准 SiLU 应统一经过公共 SwiGLU；自定义激活另行维护数学契约 |
| Grouped Matmul 标准无 bias 路径 | 已接入路径满足，兼容分支未完成 | [functional 兼容入口](../../hyper_parallel/components/functional/grouped_matmul.py)在标准条件下调用公共 ops；bias、group_type/group_list_type、gemm_fusion 或缺省 group_list 等其他分支仍导入并调用 `ops.impl` 的 NPU 实现。需为保留能力定义公共契约、后端适配及测试，不能只移动 import |
| Grouped SwiGLU | 公共契约内满足 | [族实现](../../hyper_parallel/components/ops/impl/grouped_swiglu.py)封装 native/CUDA/NPU 差异，上层不为该接口选择设备实现；量化专家成员不在本次非量化契约内 |
| Token permute / unpermute 的公共 ops、functional 和新专家模块 | 已接入路径满足 | [functional 重排](../../hyper_parallel/components/functional/moe_token_permute.py)、[恢复](../../hyper_parallel/components/functional/moe_token_unpermute.py)仅适配历史返回值、索引 dtype 和可选概率；[新专家](../../hyper_parallel/components/modules/moe/grouped_experts/grouped_experts.py)调用公共 ops |
| MoonEP / HF-native EP 的本地重排 | 尚未完全满足 | [MoonEP](../../hyper_parallel/distributed/expert_parallel/moonep_context.py)和[HF-native EP](../../hyper_parallel/distributed/expert_parallel/experts.py)仍按 `device.type == "npu"` 决定是否调用公共重排接口。需保留路由、ragged 语义和通信顺序，移除硬件决定的本地实现选择 |
| MegaGate 路由计算 | 已接入执行路径满足，源适配仍有外部依赖 | [路由模块](../../hyper_parallel/components/modules/moe/router.py)通过公共 `moe_router` 执行；`_validate_source` 仍识别并导入 HyperMulticore 源类，属于源适配依赖，不是前向内核选择；不能据此称整个模块无厂商依赖 |
| mHC Post / 普通 PipelinedMhc / Sinkhorn | 已接入主路径满足 | [Post](../../hyper_parallel/components/modules/mhc/mhc_post.py)和[普通 pipelined 路径](../../hyper_parallel/components/modules/mhc/pipelined_mhc.py)调用公共 ops。`functional.sinkhorn_knopps` 仍直接调用 native 实现，不读取配置；虽不限定硬件，也不能作为普通配置选型路径 |
| aggregate_hidden / functional MOME | 已接入路径满足 | [聚合入口](../../hyper_parallel/components/functional/aggregate_hidden.py)转调公共 ops；[MOME](../../hyper_parallel/components/functional/mome.py)仅组合 padding、布局、bias、mask 和残差，Omni autograd 在 ops 内 |
| GQA / MLA 使用的 BSND RoPE 及 Q/K 公共组合 | 已接入路径满足，旧入口未完成 | [共享位置适配](../../hyper_parallel/components/functional/rotary_embedding.py)调用公共 half / interleaved_single；同文件的 `apply_rotary_pos_emb_single` 仍判断设备和 fused 开关，`apply_rotary_pos_emb_single_freqs` 仍直接加载 torch_npu。迁移时需保留 SBND、相邻对/半带输出及频率广播语义 |

以下是相邻的既有硬件路径，本次迁移不能被解释为它们也已完成：

- GQA/MLA 的完整 attention 仍使用 `functional.npu_fusion_attention_forward`，该函数直接调用 torch_npu；[ops attention](../../hyper_parallel/components/ops/impl/attention.py)也仍复用该硬件实现。后续应由 ops 承担内核实现，functional 保留通用 mask、packed 元数据及返回值适配。
- `NativeShiftedPipelinedMhcModule.advance` 仍检查 NPU 并直接调用 HyperMulticore；mHC Pre、MegaMhc 等既有融合路径也不属于本次普通 Post/Sinkhorn 接入的完成范围。
- GDN、KDA、DSA 等现有序列算子的部分 ops 实现仍依赖 functional 中的硬件实现。公共导出改为显式函数，不代表这些算子的完整调用链已完成分层迁移。

后续优先处理本次范围内的配置绕过和兼容分支，再按各自契约迁移相邻路径。每项整改都需同时验证调用链、独立算法参考、配置命中、输出及梯度；未声明支持的组合应明确报错。将算法功能、硬件后端和源模块兼容性分开验收，避免用成员名、普通 Torch 操作或单设备结果推断硬件解耦已经完成。

## 5. 精度测试采用两层基线

### 5.1 native 对齐独立参考

CPU UT 不初始化分布式环境，覆盖契约实际支持的 dtype、常规形状和相关边界。使用固定局部随机生成器，输入和上游梯度可复现，不依赖全局随机状态或用例执行顺序。

每次验证检查全部输出及所有可导输入、参数和系数的梯度。比较前先检查元数据和梯度连接，再计算浮点误差，避免统一转成 FP32 后掩盖 dtype 或结构错误。

### 5.2 融合后端对齐同设备 native

验收单位是 `(op, variant, backend, device, dtype)`，在声明支持的目标设备上运行真实内核。

对照过程使用相同的已量化输入、初始参数和随机上游梯度，两侧各自创建独立叶子张量。先完成 native 的前向和反向，再显式切换目标后端，最后断言实际选型命中了目标。不能用 `auto` 的成功结果推断目标内核已经执行。

严格精度比较设置 `dropout=0`；非零 dropout 的随机性和行为另行验证。TopK 等返回顺序只有在契约允许无序时才能规范化，关联权重必须同步重排；稳定重排等有序契约仍需严格检查。

### 5.3 复用共享用例

| 文件 | 维护内容 |
| --- | --- |
| [op_precision.py](../../tests/common/op_precision.py) | `PrecisionCase`、独立叶子和上游梯度、选型断言、元数据及精度判定 |
| [op_precision_cases.py](../../tests/common/op_precision_cases.py) | 点算子、专家算子、注意力和线性注意力的输入及独立公式 |
| [op_precision_dsa.py](../../tests/common/op_precision_dsa.py) | DSA 的独立选择、稀疏注意力及 KL 参考 |
| [op_precision_suite.py](../../tests/common/op_precision_suite.py) | `FAMILY_BUILDERS`、共享用例入口和注册项清单 |
| [CPU UT](../../tests/ut/components/ops/) | 各族的 native 精度、边界、选型及覆盖审计 |
| [单卡 worker](../../tests/torch/ops/_test_backend_precision.py) | 注册后端的真实硬件对齐、依赖预检和测试 dtype 范围 |
| [单卡启动器](../../tests/torch/ops/test_backend_precision.py) | 平台标记和单进程启动，不导入 Torch 或 HyperParallel |

新增族时扩展合适的共享 builder，并登记 `FAMILY_BUILDERS`；新增 variant 时扩展该族的 case。用 `hardware=True` 提供符合融合内核能力的样例，同时保留 CPU 边界用例。

下面的 CPU 示例展示如何用独立表达式构造一个 `PrecisionCase`：

```python
import torch

from hyper_parallel.components.ops import swiglu
from tests.common.op_precision import PrecisionCase, check_case


def reference(x: torch.Tensor) -> torch.Tensor:
    gate, up = x.chunk(2, dim=-1)
    return gate * gate.sigmoid() * up


generator = torch.Generator().manual_seed(17)
x = torch.randn(2, 5, 128, generator=generator, requires_grad=True)
case = PrecisionCase(
    family="swiglu",
    variant="standard",
    name="regular_fp32",
    operation=swiglu,
    reference=reference,
    args=(x,),
    precision_dtype=torch.float32,
)
check_case(case)
```

已有 case 的硬件验证使用 `check_case(case, backend="torch_npu", device="npu:0")` 等显式调用，由 worker 负责硬件和依赖检查。低精度 case 的独立公式还需保留契约要求的中间舍入行为，避免参考计算图引入额外低精度梯度累加。

`expert_parameters` 指定按专家维度额外检查梯度的参数位置，例如 Grouped SwiGLU 的 `(1, 2, 3)`。packed 场景应覆盖独立文档参考、文档边界和 reset 行为；需要逐文档误差门槛时，在用例中补充分段断言。

注册覆盖审计检查 native 及融合 variant 是否有共享样例。worker 枚举注册项，但测试 dtype 和依赖预检需要显式维护；审计通过并不证明所有 dtype、形状和布局已经实测。

## 6. 数值与行为验收标准

### 6.1 数值门槛

每个输出和梯度分别满足逐元素误差与 relative L2：

```text
abs(actual - reference) <= atol + rtol * abs(reference)
relative_l2 = norm(actual - reference, 2) / norm(reference, 2)
```

当前公共门槛与 [op_precision.py 的 _LIMITS](../../tests/common/op_precision.py) 保持一致：

| 精度档位 | 前向 atol / rtol | 前向 relative L2 | 反向 atol / rtol | 反向 relative L2 |
| --- | --- | --- | --- | --- |
| FP64 | `1e-10 / 1e-8` | ≤ `1e-9` | `1e-9 / 1e-7` | ≤ `1e-8` |
| FP32 | `1e-5 / 1e-4` | ≤ `1e-5` | `1e-4 / 1e-3` | ≤ `1e-4` |
| FP16 | `1e-3 / 1e-3` | ≤ `1e-3` | `3e-3 / 3e-3` | ≤ `3e-3` |
| BF16 | `1e-2 / 1e-2` | ≤ `5e-3` | `2e-2 / 2e-2` | ≤ `1e-2` |

`precision_dtype` 按计算路径显式选择，不能仅按返回张量的 dtype 推断。例如 BF16 计算链路中的 FP32 系数梯度仍需考虑中间舍入；内部全 FP32 计算的变体则需据其契约判断使用的门槛。

参考张量全零时不计算相对误差，直接要求结果精确全零。局部结构性零值，如未使用专家的梯度和被屏蔽行，应单独断言。`max_abs`、`max_rel`、cosine 及最差位置用于诊断，其中 cosine 目前不作为额外通过条件。

这些值是现有公共测试门槛，硬件支持结论仍需真实设备证据。若某个契约确需单独门槛，应说明误差来源、影响张量及硬件测量结果，限定调整范围并同步文档；不能通过放宽全局门槛处理单个用例失败。

### 6.2 行为和边界

| 检查项 | 通过条件 |
| --- | --- |
| 返回契约 | 返回结构、shape、dtype、device、布局语义和 `requires_grad` 一致 |
| 梯度 | 全部可导输入和参数对齐，`None` 与零梯度语义一致，teacher 梯度隔离 |
| 离散结果 | 整数、布尔值和索引按契约严格比较 |
| 专家路径 | 空 token、未使用专家、路由权重及逐专家参数梯度符合契约 |
| 序列路径 | 相关 packed 文档、前缀偏移、因果和滑窗掩码、跨 chunk 与尾部行为一致 |
| 数值稳定性 | 对合法有限输入检查输出及梯度的非有限值；覆盖相关裁剪、近零和较大幅值场景 |
| 副作用 | 未声明为原地操作时，输入与参数保持不变 |
| 选型 | 显式后端命中目标；自动顺序、拒绝原因及缓存隔离符合策略 |

只覆盖算子相关的边界，不为无此能力的算子增加无意义的变体或配置项。

## 7. 模块集成与性能验收

整改实际模块后，使用相同权重和输入比较输出、梯度及一步参数更新。涉及激活重计算或编译时，补充对应路径；涉及真实通信时，按模块规则运行相应 ST。单步更新是模块接入检查，完整模型数值验收遵循[模型接入验证指南](../guide/trainer/model_integration_validation.zh.md)。

精度与性能分别给结论。声称性能提升或调整自动选择优先级时，在目标设备上比较：

- 同一契约的融合后端与 native，衡量融合收益。
- 整改后的公共接口与原有直接内核调用，检查封装和模块迁移的影响。

保持形状、dtype、参数、上游梯度及设备环境一致。执行必要预热和设备同步，分别记录首次加载与稳定执行的前向、反向、合计耗时和峰值设备内存，保留重复测量与统计口径。

性能目标由接入范围和代表性 workload 确定，暂不设置跨算子的统一加速比。没有可靠性能数据时，报告“性能未验证”，不把精度通过写成性能通过。模块集成与性能验证目前需要按改动范围补充，共享精度 runner 不会自动完成这两项。

## 8. 执行与交付

### 8.1 验证命令

在仓库根目录、已正确安装依赖的环境中执行；若 editable 指向其它目录，先重新安装：

```bash
python -m pip install -e .
python -c "import hyper_parallel; print(hyper_parallel.__file__)"
```

先运行本次涉及的族及覆盖审计，必要时扩展到相关模块和功能测试：

```bash
python -m pytest -q tests/ut/components/ops/test_swiglu.py
python -m pytest -q tests/ut/components/ops/test_precision_checks.py
python -m pytest -q tests/ut/components/ops
```

单卡测试通过轻量启动器执行。直接运行 worker 可查看每个后端的通过、失败及跳过原因：

```bash
python -m pytest -q tests/torch/ops/test_backend_precision.py
python -m pytest -q -rs tests/torch/ops/_test_backend_precision.py
```

启动器会自行创建 worker，不要再额外包一层 `torchrun`。缺少设备或可选包时可以跳过并记录未验证；支持范围外的条件可以明确排除。已安装依赖缺少算子符号、内核运行异常或数值失败按失败处理。查看实际 worker 结果，启动器退出成功可能包含全部 worker 用例被跳过的情况。

按[代码规范](../../.agent/rules/code-style.md)检查修改文件。涉及 Markdown 时还需检查链接并运行目录校验：

```bash
python3 .agent/scripts/check_agents_catalog.py
```

### 8.2 验收记录

在 PR 或开发记录中填写以下信息，不将一次运行结果写成长期支持承诺：

| 项目 | 记录内容 |
| --- | --- |
| 契约 | op、variant、公共签名及语义变化 |
| 支持范围 | backend、设备、dtype、形状、布局及明确限制 |
| 环境 | 被测提交、硬件、PyTorch、扩展和驱动等相关版本 |
| native 验证 | 独立参考、梯度验证方式、命令和结果 |
| 后端验证 | 实际选型、前向和反向误差、边界用例及原始结果位置 |
| 模块验证 | 调用点、权重兼容、一步更新，以及适用的重计算、编译和并行验证 |
| 性能 | 对照路径、测量条件、时间和内存结果，或未验证原因 |
| 未完成项 | skip、未执行、失败和支持范围外项，逐项说明原因 |

验收按范围分别给结论：CPU native 通过、目标后端精度通过、模块集成通过、性能通过或未验证。**skip 不计为通过，也不能用 CPU 通过替代目标设备上的真实内核验收。**
