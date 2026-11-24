# OpSelect开发事项

## 分步合入

| 阶段 | 内容 |
| --- | --- |
| PR 1：基础流程与 Ops | 算子契约、注册与选型、各族实现、YAML/CLI 配置、首步报告及算子级测试 |
| PR 2：调用链接入 | functional 调整、module/CP/EP 调用迁移、整网配置与示例，以及对应集成测试；待 PR 1 合入后再提交 |

本地 `master_only_ops` 准备第一阶段，完整接入版本保留在 `master_hyper_ops`。下文的分工按两阶段推进，模块与整网任务在第二阶段交付。

## PR 简介

PR #1586 引入 OpSelect，让高性能模块通过统一的 `components.ops` 接口调用算子，按设备和 `ops` 配置选择融合后端，并可切换到 native 进行精度定位。PR 已接入主要算子族，提供 native 精度测试及首步实际后端输出，本轮在此基础上完成多后端与调用链验收。

配套资料：

- **用户指南**：[中文](../guide/op_select.zh.md) / [English](../guide/op_select.md)，介绍配置、选型机制、支持矩阵和第三方框架接入。
- **开发文档**：[算子开发流程与验收标准](op_select_development.zh.md)，说明契约、注册、测试接入及精度和性能要求。
- **开发 skill**：[op-select-dev](../../.agent/skills/op-select-dev/SKILL.md)，可通过 `$op-select-dev` 引用，指导算子接入、模块整改和验收。

## 代码版本

OpSelect 代码尚未合入主仓库 `master`。合入前，两位开发者均从以下 PR 的来源分支建立各自工作分支，基于同一版本开展整改和验证。

| 项目 | 基准 |
| --- | --- |
| PR | [mindspore-ai/hyper-parallel #1586](https://github.com/mindspore-ai/hyper-parallel/pull/1586) |
| 来源分支 | [liuluobin/hyper-parallel:master_hyper_ops](https://github.com/liuluobin/hyper-parallel/tree/master_hyper_ops) |
| 拆分前的 PR 提交 | `b695ca52fb6ab0c888922a86969395aaa16f445d` |
| 本地 rebase 基准（2026-11-23） | `upstream/master`：`11646fd9622ec8bd1c42ed6937ffa4d56ea21f2c` |

当前已在本地按上述 upstream 基准重新拆分；远端 PR 尚未更新。开始前核对实际交付分支和最新提交，双方约定同一基准，并在验收结果中记录被测 commit SHA。PR 更新后统一同步基准，重新验证受影响事项。

## 目标与范围

完成已有 **17 个算子族、29 个 variant** 的多后端验收，以及 **function → module → 网络** 调用链的整改。最终能够通过 `ops` 配置切换后端和 native，并有对应的精度、兼容性及性能结果。

本轮围绕已有能力查漏补缺；mHC Pre、量化算子等新增能力另行规划。开发和精度标准统一遵循[算子开发流程与验收标准](op_select_development.zh.md)。

## 分工

两人按算子领域负责实现、测试和问题修复，人员 A 同时维护公共测试设施。

| 负责人 | 算子范围 | 当前真实后端验收范围 |
| --- | --- | --- |
| **人员 A：基础与专家算子** | RMSNorm、SwiGLU、RoPE、mHC Post、Sinkhorn、MOME、Grouped Matmul、Grouped SwiGLU、MoE Router、Token Reorder/Restore | 15 个 NPU、2 个 CUDA 组合 |
| **人员 B：注意力与序列算子** | Attention、GDN、KDA、Causal Conv1d、DSA Indexer、DSA Sparse Attention、DSA Indexer KL | 18 个 NPU、4 个 CUDA 组合 |

组合按 `(op, variant, backend, device)` 统计，dtype、shape 和布局按各后端支持能力进一步展开。

## 具体任务

| 编号 | 负责人 | 事项与交付物 |
| --- | --- | --- |
| **A1** | A | **公共测试与 CI**：将共享用例按领域拆分，保持 `PrecisionCase`、`check_case` 兼容；完善后端/dtype 测试矩阵和逐项结果，确认目标平台 CI 实际执行预期组合。 |
| **A2** | A | **function 与后端验收**：完成本人 17 个后端组合的前向、反向和边界验证，修复适配问题；重点覆盖裁剪、partial RoPE、混合精度系数、空专家和路由权重。记录融合/native、公共入口/原直接调用的性能对照。 |
| **A3** | A | **module 整改与验证**：检查 RMSNorm、SwiGLUMLP、SharedExpert、GroupedExperts、mHC 的调用点，补齐公共入口和后端切换测试；验证梯度、一步参数更新、参数复用、checkpoint 及相关重计算路径。 |
| **A4** | A | **网络和第三方接入**：验证 Qwen3-MoE 的配置与模块替换；固化独立 PyTorch、Transformers Trainer 的接入回归，有环境时补 TorchTitan。汇总双方结果，更新用户支持矩阵与指南。 |
| **B1** | B | **V3.2 CP 调用整改**：替换直接 functional 调用及固定 `backend="cann"`，通过公共算子选型；验证 native/CANN 切换、packed 前缀、dense warm-up 和 KL 衔接，保持通信语义。 |
| **B2** | B | **function 与后端验收**：完成本人 22 个后端组合的前向、反向和边界验证，修复适配问题；重点覆盖 mask、TopK、统计量、teacher 梯度隔离、packed reset 和 chunk 尾部。采用统一方式记录性能对照。 |
| **B3** | B | **module 整改与验证**：检查 GQA/MLA、DSA、GDN/KDA 等模块的公共调用，补后端 A/B、梯度和一步更新测试；验证适用的编译、重计算和 CP 路径。 |
| **B4** | B | **网络接入与文档收尾**：验证 DeepSeek V3.2/V4.1 的配置和端到端路径，按实际使用覆盖 CP/EP；核查 GDN/KDA 相关配方。同步设计文档和 RFC 的路径、支持范围与默认顺序，向 A 提供验收结果。 |

主要入口：[族实现](../../hyper_parallel/components/ops/impl/)、[模块](../../hyper_parallel/components/modules/)、[CPU 测试](../../tests/ut/components/ops/)、[硬件 worker](../../tests/torch/ops/_test_backend_precision.py)。B1 的整改位置为 [DSA 模块](../../hyper_parallel/components/modules/attention/deepseek_sparse_attention/deepseek_sparse_attention.py)及其 [CP 成员](../../hyper_parallel/components/modules/attention/deepseek_sparse_attention/cp_deepseek_sparse_attention.py)。

## 协作方式

1. **先完成公共准备**：A 完成 A1 中的用例拆分和验收表约定，双方基于同一版本推进；B 可同时开展 B1 的调用点分析和整改。
2. **按领域并行交付**：各自修改所属族、模块和测试，并修复自己领域的失败。公共精度工具、用例映射、硬件 worker，以及 `components/ops` 的包入口、`catalog.py`、`auto_policy.py` 由 A 统一合入；B 提供所需变更及验证依据。
3. **联合网络验收**：模块验证后再进行网络对照，记录实际使用的 backend；跨领域问题交给对应负责人修复，双方交叉评审。硬件未就绪时继续完成代码和 CPU 测试，硬件结果保持未验证。

## 完成标准

- **function**：目标后端实际执行，与同设备 native 的全部输出及梯度对齐，边界和接口契约检查通过。
- **module**：公共算子调用与配置生效，参数、checkpoint 和相关重计算/编译行为兼容，完成一步更新对照。
- **网络**：固定权重和输入验证配置及训练路径，正式精度验收按[模型验证指南](../guide/trainer/model_integration_validation.zh.md)比较 loss、梯度范数等结果；性能单独给结论。
- **交付记录**：每项保留被测版本、设备及依赖、执行命令、精度和性能结果、失败或未验证原因。**skip 不计为通过**；双方确认支持矩阵和文档与结果一致。
