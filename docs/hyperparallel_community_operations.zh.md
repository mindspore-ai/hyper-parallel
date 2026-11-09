# HyperParallel接入VeOmni提案

[English](hyperparallel_community_operations.md) | **简体中文**

**2026 年第四季度 · 面向贡献者、框架维护者和下游集成团队 · 更新于 2026 年 11 月 5 日**

一项加速能力如何进入主线、通过多卡验证，再作为可选后端交付给 VeOmni 的昇腾训练用户。

HyperParallel 是基于原生 PyTorch 的分布式加速库，面向昇腾（Ascend）NPU 和 NVIDIA GPU。训练框架可以按需使用其参数分片、张量并行、流水线并行、上下文并行、专家并行和分布式检查点能力。社区通过公开的 Issue 和 PR 讨论需求、评审接口并记录验证结果。[项目主页](https://github.com/mindspore-ai/hyper-parallel) · [Q4 技术路线图](https://github.com/mindspore-ai/hyper-parallel/issues/1158)

社区按**约每月一个 Beta、约每季度一个 Stable**组织发布。Beta 供贡献者和下游提前适配，Stable 提供经过验收、可复现的集成基线。2026 年 Q4，社区计划将 CI/CD 扩展到 PyTorch 2.9 至最新稳定版，覆盖 Ascend NPU 和 NVIDIA GPU，并推进与 PyTorch 生态项目的协作。

## HyperParallel接入VeOmni社区提案

1. **请求批准的能力。** 本次提请 VeOmni 批准通过可选 Python 包依赖接入 HyperParallel FSDP、Muon 优化器、高性能模块和 MegaMoE。GPU 后端暂不在此次支持范围内。RFC 将逐项列出高性能模块的名称、启用方式和适用模型。
2. **可复现的收益与正确性。** 以现有报告中的实验为基础，按 VeOmni 社区提出的场景和验收要求补测。完整报告记录模型与配置、双方代码版本、Ascend 环境与卡数、原生后端对照、性能收益及数值误差。RFC 与合入 PR 均附完整报告，PR 所附结果对应最终被测提交。[现有实验记录](https://github.com/mindspore-ai/hyper-parallel/issues/637)
3. **首批支持矩阵与用户边界。** 首批面向字节内部使用 VeOmni 在 Ascend NPU 上训练的用户。RFC 的支持矩阵按模型列出已验证的加速能力、Ascend 设备、VeOmni 与 HyperParallel 版本、PyTorch、torch-npu 和 CANN 版本，并链接对应的实验结果。
4. **评审人可运行的最小示例。** 在 VeOmni 代码仓随适配 PR 提交《使用 HyperParallel 作为 Ascend 加速后端》指南，给出安装已验证版本的命令、最小训练配置、启动命令和预期输出。评审人可据此从干净环境启动多卡训练，确认所选加速能力已生效。
5. **双边维护责任和故障处理。** HyperParallel 承担 Ascend 后端的日常维护和故障首接，负责维护安装包、适配代码和 VeOmni 仓内使用指南，提供 Ascend 测试资源，并每周验证与 VeOmni 的兼容性。

### 接入方式与适配进展

HyperParallel 的 Q4 路线图已列入 VeOmni 可选 FSDP2 后端和 Muon 优化器。现有实现位于适配分支，向 VeOmni 上游合入仍是跟踪事项。VeOmni 保留训练编排和 DeviceMesh 构建，HyperParallel 提供用户显式选择的加速能力。双方可保留各自版本号，通过实测兼容表向用户说明哪些组合可以一起安装。[HyperParallel 接入跟踪单](https://github.com/mindspore-ai/hyper-parallel/issues/1124) · [VeOmni 发布记录](https://github.com/ByteDance-Seed/VeOmni/releases)

**接入方式采用可选 Python 包依赖。** HyperParallel 独立构建和发布 wheel，调用其公共 API 的适配代码随 VeOmni 发布。正式接入时，在 VeOmni 的 `pyproject.toml` 中声明可选依赖，并把验证过的版本写入 `uv.lock`。用户按所启用的能力安装相应依赖，默认 PyTorch 路径可独立使用。[VeOmni 依赖配置](https://github.com/ByteDance-Seed/VeOmni/blob/main/pyproject.toml)

### 2026 Q4 路线图交集

VeOmni 的总路线图覆盖多个硬件平台，Ascend 专项进一步列出了 NPU 上需要解决的问题。HyperParallel 的 Q4 路线图明确提出向 VeOmni 提供可选的 FSDP2 和 Muon 支持。下表对照双方的技术目标；VeOmni 是否采用 HyperParallel 的实现，仍需通过其社区评审。[VeOmni Q4 总路线图](https://github.com/ByteDance-Seed/VeOmni/issues/1365) · [VeOmni Ascend 专项](https://github.com/ByteDance-Seed/VeOmni/issues/1405) · [HyperParallel Q4 路线图](https://github.com/mindspore-ai/hyper-parallel/issues/1158) · [接入提案](https://github.com/mindspore-ai/hyper-parallel/issues/1124)

| 共同方向 | VeOmni 的 Q4 目标 | HyperParallel 的 Q4 目标与对接边界 |
| --- | --- | --- |
| FSDP2 低精度通信与训练 | 在 Ascend 上对参数 AllGather 做量化，为 HCCL 适配低精度 ReduceScatter；总路线图还包括 FP8 计算和量化感知训练（QAT）。[VeOmni 低精度 RFC](https://github.com/ByteDance-Seed/VeOmni/issues/1360) | 推进 FSDP2 的 FP8 权重通信和 MXFP4 QAT。双方最直接的交集是降低参数 AllGather 的通信开销；HCCL ReduceScatter 和量化导出需要单独验证。[FP8 权重通信](https://github.com/mindspore-ai/hyper-parallel/issues/1165) · [MXFP4 QAT](https://github.com/mindspore-ai/hyper-parallel/issues/1170) |
| Muon 状态卸载 | 在 Ascend 上异步卸载 AdamW 和 Muon 状态，要求传输与计算重叠，同时保持更新精度和检查点可恢复性。[VeOmni NPU 内存优化](https://github.com/ByteDance-Seed/VeOmni/issues/1405) | 将 Muon 动量保存在锁页 CPU 内存，更新时搬回设备。可共用一条多卡训练用例，对照峰值显存、单步耗时、参数更新及恢复后的优化器状态。[Muon 状态卸载 RFC](https://github.com/mindspore-ai/hyper-parallel/issues/1077) |
| MoE 专家负载均衡与执行 | 缓解 token 分布不均造成的专家热点和显存峰值。Qwen3.5-MoE 已有通过临时专家副本分担热点的工作。[VeOmni 专家副本方案](https://github.com/ByteDance-Seed/VeOmni/pull/1112) | 规划冷热专家迁移与负载均衡；MegaMoE 提供专家分发、计算与合并的融合执行能力。接入时需约定由谁决定专家布局，分别测量调度收益和融合算子收益。[专家迁移 RFC](https://github.com/mindspore-ai/hyper-parallel/issues/1129) · [MegaMoE RFC](https://github.com/mindspore-ai/hyper-parallel/issues/1121) |
| 长上下文并行 | 扩展 Ring Attention 与 Ulysses 混合并行，并在 Ascend 稀疏注意力模型上验证 KV-AllGather 上下文并行（CP）。[Ascend 混合并行](https://github.com/ByteDance-Seed/VeOmni/pull/1183) · [KV-AllGather CP](https://github.com/ByteDance-Seed/VeOmni/pull/1290) | 同样规划 Ring + Ulysses、KV AllGather 和多模态 CP。双方可复用变长打包边界、梯度与显存测试，并验证与 FSDP、专家并行的组合。[混合 CP](https://github.com/mindspore-ai/hyper-parallel/issues/1107) · [多模态 CP](https://github.com/mindspore-ai/hyper-parallel/issues/1118) |
| 选择性重计算与激活卸载 | 扩展层级和算子级重计算，增加异步激活卸载；另计划在适用场景跳过冗余反向计算。[VeOmni 选择性重计算](https://github.com/ByteDance-Seed/VeOmni/pull/1166) · [Ascend 专项](https://github.com/ByteDance-Seed/VeOmni/issues/1405) | 规划 MoE 细粒度重计算、激活 Swap 优化和重计算边界显存优化。可共测同一配置下的梯度、峰值显存和单步耗时；跳过反向计算另行验收。[MoE 重计算](https://github.com/mindspore-ai/hyper-parallel/issues/1161) · [激活卸载](https://github.com/mindspore-ai/hyper-parallel/issues/1079) |

#### 可共用的模型和测试

- **模型与高性能模块。** DeepSeek V4.1-Flash、Kimi K3 和 Qwen3 系列的 MoE/VL 场景可作为共同验证对象。VeOmni 提供训练入口和模型配置，HyperParallel 提供可替换的注意力、专家或 MLP 模块。每次固定一个模型和替换位置，记录替换前后的数值与性能。[VeOmni 模型计划](https://github.com/ByteDance-Seed/VeOmni/issues/1405) · [HyperParallel 模型清单](https://github.com/mindspore-ai/hyper-parallel/issues/1166) · [高性能模块 RFC](https://github.com/mindspore-ai/hyper-parallel/issues/1126)
- **多模态数据均衡。** 双方都关注长短样本和多模态输入造成的 DP rank 负载差异。可使用同一组样本测量最慢 rank 的耗时和显存。这与 MoE 专家之间的 token 均衡属于不同的调度环节。[VeOmni 多模态数据均衡](https://github.com/ByteDance-Seed/VeOmni/issues/1074) · [HyperParallel 数据均衡](https://github.com/mindspore-ai/hyper-parallel/issues/1155)
- **大模型加载与检查点恢复。** VeOmni 规划按目标参数分片读取，HyperParallel 规划 DCP 并行读取、广播和恢复。可共用加载耗时、内存峰值和续训一致性测试；检查点格式兼容性需另行验证。[VeOmni 分片加载](https://github.com/ByteDance-Seed/VeOmni/issues/1297) · [HyperParallel DCP 加载](https://github.com/mindspore-ai/hyper-parallel/issues/1178) · [恢复场景](https://github.com/mindspore-ai/hyper-parallel/issues/1179)
- **编译与真机回归。** VeOmni 关注动态长度注意力的编译兼容性，以及模型、数值和检查点的 Ascend 真机验证；HyperParallel 推进专家并行与上下文并行的整图编译。代表性训练用例可复用于每周联调和 PR 验证。[VeOmni 编译目标](https://github.com/ByteDance-Seed/VeOmni/issues/1365) · [VeOmni 真机验证](https://github.com/ByteDance-Seed/VeOmni/issues/1405) · [HyperParallel EP 编译](https://github.com/mindspore-ai/hyper-parallel/issues/1112) · [CP 编译](https://github.com/mindspore-ai/hyper-parallel/issues/1113)

**建议优先联调两组 Ascend 场景。** 第一组用 FSDP2 + Muon 多卡训练核对数值、显存和断点恢复。第二组在一个 MoE 模型上分别启用高性能模块和 MegaMoE，保留 VeOmni 的训练入口与专家布局，测量替换前后的收益。首版仍按 FSDP、Muon、高性能模块、MegaMoE 四项提请评审；低精度 QAT、长上下文 CP 等方向留待双方另行确认。

## HyperParallel社区代码合入流程

| 环节 | 贡献者和维护者分别做什么 |
| --- | --- |
| 提出问题或方案 | 缺陷报告附版本、硬件和最小复现步骤。新增功能或涉及公共接口的改动，通过 Feature Issue 或 RFC 说明使用场景、兼容性和验收方法，由相关模块维护者评审设计。 |
| 编码与规范自检 | 作者按仓库编程规范实现改动，提交前检查代码风格、复杂度和文档格式。 |
| 提交 PR | 每个 PR 聚焦一个可独立验证的变更，整理为一个提交，按仓库约定使用 `feat:`、`fix:`、`docs:` 等前缀。作者关联 Issue，填写改动理由、影响范围、实测结果；涉及 API 或用户行为时同步更新文档。 |
| 提供测试证据 | 单元测试（UT）覆盖可独立验证的逻辑。涉及通信、分片、并行组合或断点续训的改动，补充分布式系统测试（ST），写明设备、卡数、软件版本和结果。性能改动给出同一配置下的基线和测量方法。尚未验证的场景明确列出。 |
| 评审与合入 | 根据 CODEOWNERS 请求模块负责人评审。当前 GitHub 主线要求 Code Owner 批准、最近一次推送获得独立批准、评审讨论已解决，并通过 CI 和评审人数两项检查。作者推送新提交后，按新的提交重新取得有效结果。 |

**编程规范与提交前自检**

- **语言与格式。** Python 文件以 Apache 2.0 许可头开篇，遵循 PEP 8，行宽约 120 列。类名用 PascalCase，函数和变量用 snake_case。C++ 遵循仓库的 .clang-format，行宽为 120 列。
- **接口与实现。** 公共函数须有类型标注，公共函数和方法使用 Google 风格文档字符串。导入默认放在模块顶部，依次排列标准库、第三方库和项目模块。分布式实现直接使用原生 PyTorch API。
- **提交前自检。** 安装仓库的 pre-commit 钩子后，提交时会对适用文件运行 pylint、lizard 和 markdownlint。作者修复检查发现的问题；确需保留例外时，在 PR 中说明理由。评审人核对公共接口的文档和类型是否与实现一致。

[仓库编程规范](https://github.com/mindspore-ai/hyper-parallel/blob/master/.agent/rules/code-style.md) · [提交前检查配置](https://github.com/mindspore-ai/hyper-parallel/blob/master/.pre-commit-config.yaml)

贡献入口可直接使用仓库模板。PR 模板把设计、测试、效果验证、公共接口和文档列为五项自检内容。[Issue 模板](https://github.com/mindspore-ai/hyper-parallel/issues/new/choose) · [PR 模板](https://github.com/mindspore-ai/hyper-parallel/blob/master/.github/PULL_REQUEST_TEMPLATE.zh-CN.md) · [代码归属](https://github.com/mindspore-ai/hyper-parallel/blob/master/CODEOWNERS) · [仓库贡献规则](https://github.com/mindspore-ai/hyper-parallel/blob/master/AGENTS.md)

### 现行合入门禁

GitHub 主线有两项独立的必需检查。Jenkins 执行测试并把结果回传到 PR，GitHub 检查结果是否仍然有效；另一个工作流检查当前提交是否得到足够的正式批准。[主线生效规则](https://api.github.com/repos/mindspore-ai/hyper-parallel/rules/branches/master) · [评审和合并流程](https://github.com/mindspore-ai/hyper-parallel/blob/master/docs/contributing/github_auto_merge_zh.md)

- **测试门禁 `label-check`。** 作者按模板完成描述检查，通过后可以在 PR 查看 Jenkins 的测试、覆盖率及构建详情。通过标签 `ci-pipeline-passed` 或 `ci-pipeline-passed-but-overtime` 自添加起有效 24 小时。超时通过的标签保留了耗时异常这一信息。推送新提交或通过结果过期后，需要为待合入代码重新取得有效结果。[标签有效期规则](https://github.com/mindspore-ai/hyper-parallel/pull/1365) · [实际 Jenkins 回传示例](https://github.com/mindspore-ai/hyper-parallel/pull/1438#issuecomment-6604110750)
- **评审门禁 `all-reviewers-approved`。** 一位评审人时需一人批准，两位及以上时需两位不同评审人批准。批准必须针对当前提交，作者不计入人数。尚未解除的修改请求、待处理的团队评审请求仍会阻止通过。达到人数后，还须满足主线的 Code Owner 等原生规则。[评审规则及示例](https://github.com/mindspore-ai/hyper-parallel/blob/master/docs/contributing/github_auto_merge_zh.md)
- **失败处理。** 作者从失败阶段和日志定位问题，代码回归由模块维护者协助修复，运行环境异常交由 CI 维护者处理。重测后以最新提交的记录为准。主项目 UT 和分布式 ST 有不同的执行条件；RL 系统测试目前单独运行，其结果需另行核对。[测试规范](https://github.com/mindspore-ai/hyper-parallel/blob/master/docs/contributing/testing.md) · [RL 测试范围](https://github.com/mindspore-ai/hyper-parallel/blob/master/AGENTS.md)

## VeOmni 前置验证

Q4 计划在 **HyperParallel CI/CD** 中增加 VeOmni 兼容性流水线。每周任务负责跟进 VeOmni，每个 HyperParallel PR 则用本次提交构建的安装包跑 VeOmni 训练，在合入前发现影响下游的回归。

| 触发时机 | 流水线执行内容 | 通过条件 |
| --- | --- | --- |
| 每周更新一次 VeOmni | 获取 VeOmni 上游主线的具体提交，记录对应的发布标签（如有）。适配尚未合入上游时，同时固定与该提交配套的适配补丁。用 HyperParallel 主线安装包运行集成测试。 | 全部通过后，将该组合设为 PR 的测试基线。失败时保留上一次通过的基线，记录失败组合、日志和负责人，修复后重新验证。 |
| 每个 HyperParallel PR 合入前 | 从 PR 最新提交构建 wheel，在干净环境中与固定版本的 VeOmni 一起安装。在 Ascend 多卡上验证 FSDP2/HSDP 前反向、参数更新、同拓扑保存与恢复，并与原生后端作数值对照。对已接入的 Muon、高性能模块和 MegaMoE 分别运行 RFC 约定的用例。 | 将双方提交、PyTorch 版本、设备和各用例结果回传 PR。必测项失败、跳过或缺少结果时阻止合入。PR 新增提交或 VeOmni 测试基线变更后，重新验证待合入组合。 |
| Beta / Stable 发版前 | 在干净环境中安装候选包，验证导入、快速入门和声明支持的组合。Stable 再核验升级路径和代表性下游任务。 | 交付安装包、精确依赖、兼容表、已知限制和升级说明。 |

## 支持PyTorch多版本与多硬件

Q4 的目标是覆盖 **PyTorch 2.9 至当期最新稳定版**，并在 Ascend NPU 和 NVIDIA GPU 上提供可查的测试结果。本次 VeOmni 接入仅验收 Ascend 组合。每条发布线应列出具体 PyTorch 小版本。NPU 同时记录设备型号、torch-npu 与 CANN 版本，GPU 记录设备型号与 CUDA 版本。遇到某个平台的运行时尚未适配新 PyTorch 时，兼容表单列“待验证”及跟踪 Issue。

满足 PyTorch Ecosystem Project 标准是 Q4 目标之一。在兼容性方面，社区将持续跟进 PyTorch 最近两个发布版本，并为每种声明支持的架构提供可复核的 CI 结果。正式收录以生态工作组的评审结果为准。[PyTorch 官方标准](https://github.com/pytorch-fdn/tac/blob/main/docs/governance/PyTorch_Ecosystem_Process.md#ecosystem-projects-requirements)

| 测试对象 | 硬件 | Q4 验证范围 |
| --- | --- | --- |
| HyperParallel | Ascend NPU | 从 PyTorch 2.9 到当期最新稳定版，按具体小版本和 NPU 运行时安排基础回归。 |
| HyperParallel | NVIDIA GPU | 在相同的 PyTorch 版本范围内，按具体小版本和 CUDA 环境安排基础回归。 |
| HyperParallel + VeOmni | Ascend NPU | 从两项目标版本和运行时均匹配的组合中选取实测基线，执行本次集成验收。 |
| HyperParallel + VeOmni | NVIDIA GPU | 不在本次 VeOmni 支持范围内。 |

每次验证记录模型、启用的能力、双方版本或提交、PyTorch 版本、卡型与卡数、运行时、测试结果和报告链接。实际支持组合以当次报告为准。

### 测试安排

| 建议的执行时机 | 验证范围 | 交付给社区的结果 |
| --- | --- | --- |
| 每个 PR | 检查受影响模块的代码风格、UT 和必要的多卡 ST。优先覆盖改动触及的框架版本与硬件。 | 当前提交的检查结果、失败日志及复现条件。 |
| 主线定期回归 | 在两类硬件上轮转执行版本矩阵，重点比较数值精度、通信正确性、训练吞吐、峰值显存和检查点恢复。 | 各组合最近一次成功记录、回归项和负责模块。 |
| Beta / Stable 发布前 | 用候选安装包在干净环境执行安装、导入和快速入门；对本次声明支持的组合执行多卡回归。Stable 再核验升级路径和代表性下游任务。 | 可安装的制品、精确依赖、兼容表、已知限制及升级说明。 |

这张表给出 Q4 的实施建议。用户选择环境时，以相应版本发布说明中的实测组合为准。仓库已有正式发布的严格构建、安装包验证和人工评审流程，可作为矩阵扩展的基础。[发布流程](https://github.com/mindspore-ai/hyper-parallel/blob/master/docs/contributing/release.md)

## 版本发布节奏

双方独立编号和发布。每次提交检查当前 PR，每周计划刷新 VeOmni 联调基线；HyperParallel 约每月发布 Beta、每季度发布 Stable，具体日期由候选版本的验收结果决定。

| 发布通道 | 主要使用者 | 社区如何使用这个窗口 |
| --- | --- | --- |
| **Beta · 约每月一次** | 新特性贡献者、硬件适配者、下游框架维护者 | 提供可安装的候选版本，列出新增能力、接口变动和已知限制。下游可在下一个 Stable 前反馈集成问题，并附版本、环境和复现用例。 |
| **Stable · 约每季度一次** | 需要可重复部署的训练团队和下游正式版 | 根据 Beta 反馈收敛发布范围。完成目标组合的验证和人工评审后，生成版本标签、安装包和 Release Notes，公开增量变更、已知限制、升级指南、贡献者与合入 PR。 |

版本号采用 `vMAJOR.MINOR.PATCH`：不兼容 API 变更升主版本，向后兼容的新功能升次版本，向后兼容的缺陷修复升补丁版本。季度和月度是计划窗口，具体日期、功能清单和支持范围随候选版本确认。发布说明还会区分能力成熟度，例如某项能力仍标为 `DEMO` 时，即使随 Stable 交付，也应按该能力自己的使用边界集成。[版本命名与发布规范](https://github.com/mindspore-ai/hyper-parallel/blob/master/docs/contributing/release.md) · [正式版本说明示例](https://github.com/mindspore-ai/hyper-parallel/blob/master/RELEASE_CN.md)

### 下游版本协作

1. **Beta 联调。** 双方各指定维护者，选定 VeOmni 提交与 HyperParallel Beta 安装包。以昇腾多卡 FSDP2/HSDP 为首批场景，验证原生后端的数值对照，以及相同并行拓扑下模型和优化器状态的保存与恢复。Muon 单独记录验收结果。
2. **Stable 定版。** HyperParallel 发布通过验收的安装包；VeOmni 在含适配代码的候选版本复测。双方在发布说明中列出两个项目的精确版本、安装来源、PyTorch 和硬件运行时、支持的模型与并行拓扑。
3. **后续升级。** 一方发布兼容补丁，另一方复测原有版本，通过后更新兼容表即可。遇到公共接口或检查点格式变更，在月度 Beta 提前通知对方，并为仍使用旧组合的用户保留升级说明。

**HyperMulticore 独立建仓。** 作为 2026 年 Q4 社区计划的一部分，HyperParallel 已于 10 月 28 日为 HyperMulticore 建立独立仓库。该仓库集中维护 MegaMoE、MegaGate 和 MegaMHC 的昇腾多核算子及 PyTorch 运行时。贡献者可在新仓库查阅构建指南、运行示例或提交相关 Issue 和 PR。[HyperMulticore 仓库](https://github.com/mindspore-ai/hyper-multicore)

## VeOmni社区维护

HyperParallel 承担 Ascend 后端的日常维护与故障首接。VeOmni 维护者审查其仓库的适配变更，并决定合入和发版。

建议在 RFC 中登记双方的主责人和备份人。HyperParallel 指定一位接入维护者，负责用户报障、复现和跟踪直至关闭。VeOmni 指定一位接口维护者，负责评审其仓库内的配置、训练入口和适配变更。涉及 MegaMoE 实现问题，由 HyperParallel 接入维护者协调 HyperMulticore 修复，用户沿用原工单。

| 场景 | 牵头人和处理方式 | 恢复条件 |
| --- | --- | --- |
| HyperParallel PR 引入回归 | HyperParallel PR 作者修复，接入维护者提供 Ascend 资源与日志。为失败场景补回归用例。 | 当前 PR 提交通过所需检查，方可合入 HyperParallel。 |
| 每周跟进 VeOmni 新版本失败 | HyperParallel 接入维护者保留上一次通过的组合，登记失败提交并复现；涉及 VeOmni 接口时，向其接口维护者提交最小复现和适配方案。 | 新组合通过同一用例后切换基线，兼容表记录验证日期。 |
| 发布后 · 训练中断或数值异常 | Ascend硬件下的问题由HyperParallel承接，GPU硬件下问题由故障模块的维护者牵头。 | 在原工单记录根因、受影响版本、修复版本及回归结果，双方确认后恢复推荐。 |

HyperParallel 负责维护 wheel、适配代码、VeOmni 仓内指南和 Ascend 测试资源，并执行每周兼容验证。VeOmni 保留其仓库的合入和发版权；双方各自决定本项目的阻塞范围。建议在首次发布前明确报障入口和当值联系人，并用一次模拟故障演练从报告、回退到复测的全过程。
