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

### 实现契约与后端

- 公共入参由 native 所需能力决定。影响通用语义的参数进入 Protocol；内核专有编码由实现固定或转换以保持等价。
- 在 `components/ops/families/<YAML配置键>.py` 统一声明与实现。协议或行为有实质差异时使用独立 variant 和公共入口；相同契约更换实现不增加 variant。
- native 先通过独立公式或算法验证。适用时使用 FP64 gradcheck；内部固定 FP32 的实现不能仅靠输入转 FP64 验证。
- 使用现有 `register_impl` 和按需加载方式。兼容限制按主输入检查、公共入口调用级检查分工处理，并纳入解析缓存条件；避免捕获执行异常后静默回退。
- 模块从 `hyper_parallel.components.ops` 直接导入函数，在 `forward` 调用；训练参数保持在模型模块中。无需新增 Trainer wrapper、`ops.create` 调用或硬件分支。
- 仅在本次需要时维护目录组装、自动选择顺序、支持矩阵及配置。调整优先级应有对应设备和 workload 的证据。

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
