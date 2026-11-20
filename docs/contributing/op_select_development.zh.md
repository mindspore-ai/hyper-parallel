# OpSelect 算子开发流程与验收标准

本文面向接入算子族、扩展融合后端、整改高性能模块或补充算子精度测试的开发者。完整流程为：

**定义契约 → 建立 native 基线 → 接入后端 → 整改模块 → 精度验收 → 集成与性能验收。**

已有契约下新增后端时，复用公共接口和参考实现；仅补测试时，围绕现有实现扩展用例。验收范围由本次改动及声明的支持能力决定。

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

在 `components/ops/families/<配置键>.py` 中用 `Protocol` 描述接口，用 `OpSpec` 定义算子族和 variant。文件名必须与 YAML 的 `ops` 配置键一致。

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
| [families/](../../hyper_parallel/components/ops/families/) | Protocol、OpSpec、实现和该族的注册函数 |
| [registry.py](../../hyper_parallel/components/ops/registry.py) | `register_family`、`register_impl` 和解析接口 |
| [catalog.py](../../hyper_parallel/components/ops/catalog.py) | 组装并封存内置算子目录 |
| [auto_policy.py](../../hyper_parallel/components/ops/auto_policy.py) | 每个 variant、设备的 `auto` 候选顺序 |
| [包入口](../../hyper_parallel/components/ops/__init__.py) | 对外导出算子函数并处理调用级约束 |

`register_impl` 的 `load` 是无参加载器，可返回函数、可调用对象或 `nn.Module`，实现内部可以使用 `torch.autograd.Function` 定义自定义反向。构造参数由加载器绑定；训练参数由模型模块持有，并在调用时传入，避免隐藏在全局缓存的实现对象中。

兼容性处理遵循以下分工：

1. `check_compatibility(x)` 根据主输入设备、dtype 等判定能力，返回 `None` 表示可用，字符串表示拒绝原因。
2. 形状、布局和多个输入之间的约束在公共入口检查，将不兼容候选传入解析逻辑，使结果与解析缓存保持一致。
3. 可选包和算子按需加载；依赖缺失使用 `ImplementationUnavailable` 表达。普通包导入无需安装所有硬件扩展。
4. `auto` 只在既定候选中选择；是否允许 native 回退取决于该 variant 的设备策略。显式后端不可用或不兼容时应报错。
5. 实际内核执行异常直接暴露，不能捕获所有异常后静默改用 native。

注册片段见[注册新算子族或后端](../guide/op_select.zh.md#注册新算子族或后端)。调整 `auto` 优先级时，同时记录所依据的正确性、能力范围和性能证据。

## 4. 在模块中使用公共接口

模块直接从 `components.ops` 导入算子，在 `forward` 中调用。以标准 SwiGLU 为例：

```python
import torch

from hyper_parallel.components.ops import swiglu


class Activation(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return swiglu(x)
```

外部模块无需感知 `families` 路径、调用 `ops.create` 或在初始化时绑定后端函数。后端通过已有 YAML 配置或 `ops.set_backends` 在每个 worker 中设置，不需要包裹 Trainer 的训练流程。

`ops.set_backends` 完整替换当前进程配置，并清理解析缓存和使用统计。测试必须保存并恢复原配置；一次前向和反向应在同一配置下完成，结束后再切换到另一组对照实现。

整改已有模块时，保留原权重所有权、参数命名、checkpoint 语义和计算顺序。确认首步实际执行覆盖了目标分支，并通过 `ops.used_backends()` 检查命中的 `(op, variant, backend)`。

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
