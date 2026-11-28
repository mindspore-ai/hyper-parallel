---
name: op-select-dev
description: >
  Develop or extend OpSelect operator families and fused backends under
  hyper_parallel/components/ops, migrate high-performance module call sites,
  and add native/backend precision tests. Use for local operator integration
  and acceptance; DTensor shard-rule development belongs to dist-op-dev.
---

# OpSelect 算子开发

交付符合公共契约的算子接入或测试改动，并分别报告 native、真实后端、模块和性能验证结果。按用户本次授权范围执行；仅补测试时复用既定 API，新增同一 variant 的后端时复用既定契约。

## 读取入口

1. 读取仓库 `AGENTS.md` 和 [code-style.md](../../rules/code-style.md)。
2. 读取[开发流程与验收标准](../../../docs/contributing/op_select_development.zh.md)，以其中的契约、两层基线、精度门槛和交付要求作为执行依据。数值门槛只维护在该文档及其指向的公共判定代码中，不在 skill 中复制。
3. 接入实现时查阅[注册与公共 API](../../../docs/guide/op_select.zh.md#注册新算子族或后端)，并读取相近的实际族、`catalog.py`、`auto_policy.py` 和包入口。
4. 编写测试时读取 [testing.md](../../rules/testing.md)、[unit-test.md](../../rules/unit-test.md) 和 [test-assertion-style.md](../../rules/test-assertion-style.md)，复用现有精度工具与族用例。

## 执行流程

### 确定改动范围

- 从现有调用点和参考实现列出 `(op, variant, backend, device, dtype)` 及相关形状、布局、梯度和边界。
- 确认是新增族、扩展 variant、接入后端、模块整改还是补精度测试；复用已有能力，避免扩大为完整模型接入或分布式算子开发。
- 遵循会话中已确定的契约与偏好，直接推进已授权工作；需要澄清时先完成不依赖答案的分析。

### 硬件解耦目标

- ops 是本地算子的硬件边界。切换已支持的 CPU/CUDA/NPU 或后端时，functional/module 的算子调用保持不变；设备识别、可选扩展加载、内核调用、内核专有参数及能力限制归属 ops。
- functional/module 可使用普通 Torch 运算完成硬件无关的组合、分配和数据布局处理；`x.device` 用于跟随输入分配张量不等于硬件选型。不得根据 `device.type`、硬件可用性、后端名称或 fused 开关决定某个已接入算子的实现，也不得导入 `ops.impl` 或直接调用厂商内核。
- dtype、shape、mask、状态和数学布局可以影响算法流程，但不能用于绕过 ops、选择硬件实现或静默回退。特定硬件的格式编码及对齐限制应由 ops 实现适配和检查。
- 历史参数、旧名称和源模块适配逐项审查：保留数学语义与兼容性，不把硬件开关继续作为执行选型来源。第三方源模块识别与参数迁移区别于算子执行，记录其剩余依赖，不能据此宣称整个模块完成硬件解耦。

### 实现契约与后端

- 公共入参由 native 所需能力决定。影响通用语义的参数进入 Protocol；内核专有编码由实现固定或转换以保持等价。
- 在 `components/ops/impl/<YAML配置键>.py` 统一声明与实现。协议或行为有实质差异时使用独立 variant 和公共入口；相同契约更换实现不增加 variant。
- native 先通过独立公式或算法验证。适用时使用 FP64 gradcheck；内部固定 FP32 的实现不能仅靠输入转 FP64 验证。
- 公共接口不接收 `backend`，统一读取进程级 `ops` 配置；未配置的 family 使用 `auto`，不增加调用点默认后端或 `default_backend`。
- 使用现有 `register_impl` 和按需加载方式。执行设备由公共函数显式传入，key 仅为 family、variant、配置后端、完整 device；不按 dtype/shape 过滤后端。通用契约和所选后端输入限制分别检查，执行错误不触发回退。
- 分层遵循 `module → functional → ops`，同时允许 module 直接调用 ops。module 管理训练参数、状态、checkpoint 和模块流程；functional 复用无状态算法组合、布局适配与精度策略；ops 管理算子契约、后端选型和实现。不要要求所有模块绕过 functional。
- 多处模块使用同一组合逻辑时，先复用或扩展 functional；仅需要原子算子的模块可直接导入 ops 公共接口。functional 不导入 module，ops 不反向导入 functional 的组合逻辑；既有后端适配依赖按实际迁移范围处理。
- 兼容 functional 入口应转调 ops，显式适配历史返回值、索引 dtype、可选参数和精度边界，不保留一套直连内核的重复路径。单 Tensor 调用使用匹配的算子契约，避免传入两次相同输入再丢弃一份结果。
- 仅在本次需要时维护目录组装、自动选择顺序、支持矩阵及配置。调整优先级应有对应设备和 workload 的证据。

### 维护显式公共接口

- 保留 Protocol 作为后端契约，在对应族文件中显式定义公共函数；签名、公共参数适配、调用级限制和完整 docstring 均在该函数维护，调用 `factory._select_for_call(spec, device=execution_device)` 完成统一选型。
- 在 `components/ops/__init__.py` 显式导入公共函数并维护 `__all__`。新增或删除接口时同步修改导出，运行 `tests/ut/components/ops/test_factory.py` 的公共接口检查及受影响算子测试。
- IDE 直接读取源码签名和文档。保留 `py.typed` 的版本管理和构建打包；无需生成或维护 `.pyi`，不增加生成脚本、构建生成动作或提交钩子。详见[注册与公共 API](../../../docs/guide/op_select.zh.md#注册新算子族或后端)。

### 审视调用链的硬件边界

- 对声明接入的每个入口追踪 `module → functional/ops → 所选实现`；检查所有分支、默认值、兼容入口及 EP 的本地计算，不能只检查普通输入下的一条路径。
- 检查上层的设备判断、厂商扩展导入、直接内核调用、`ops.impl` 依赖和 fused/backend 控制；区分通信基础设施、普通张量分配、源模块适配及实际算子选型。
- 分别报告“已接入路径满足”“同族兼容入口仍有缺口”“相邻路径未迁移”。新增注册、公共函数或单设备精度通过，均不能代替调用链完成硬件解耦的证据。当前审视记录见[开发文档](../../../docs/contributing/op_select_development.zh.md#43-当前基础与专家接入的硬件边界审视)。

### 补充与运行验证

- 优先扩展 `tests/common/op_precision_cases.py` 或 `op_precision_dsa.py`，在 `op_precision_suite.py` 中维护 builder 映射，再补 `tests/ut/components/ops/test_<族>.py`。
- 复用 `PrecisionCase` 和 `check_case`，验证独立参考对 native、同设备 native 对真实后端。两侧使用相同已量化输入、独立叶子和相同上游梯度；显式选择并断言实际 backend。
- 验证全部输出、可导输入与参数梯度、`None`/零梯度语义、teacher 隔离、离散结果和输入不变性。专家梯度使用 `expert_parameters` 逐专家检查，相关 packed/reset/chunk 等边界补针对性断言。
- 测试前检查 editable 指向当前 checkout。前向和反向完成后再切换进程配置，测试结束恢复原配置。
- 运行受影响族、覆盖审计及必要相关回归。新增族或变体应通过 `test_precision_checks.py` 的清单审计。
- 真正的后端精度在目标设备运行 `tests/torch/ops/_test_backend_precision.py` 或其轻量启动器。维护 worker 的 dtype 范围、可选依赖预检及兼容样例，不能只依赖注册清单自动枚举。
- 模块整改时补实际模块输出、梯度及一步更新；按相关能力验证重计算、编译和真实通信。声称加速或调整优先级时，再完成文档规定的性能对照。

## 失败处理与交付

发现差异时先检查契约、输入量化、舍入顺序、梯度连接及实际选型，再定位独立参考或实现错误。不要通过降低全局精度门槛、缩减已声明覆盖范围或静默回退掩盖失败。

硬件或可选包缺失时继续完成可执行工作，将受影响组合报告为未验证；支持范围外的组合说明排除依据。依赖已安装但符号缺失、内核异常和数值错误按失败处理。启动器成功退出不代表所有 worker 用例通过，需查看 skip 明细。

完成时给出改动文件、契约和支持范围、测试命令与结果、数值与性能证据，以及未验证或失败项。区分 native 精度、目标后端精度、模块接入和性能结论。不得将本地无硬件的 skip 写成后端验收通过。

本 skill 不隐含提交、推送或发布授权；用户要求这些操作时遵循仓库 Git 工作流。
