# 参与 HyperParallel 社区贡献

[English](CONTRIBUTING.md) | **简体中文**

HyperParallel 欢迎问题反馈、文档改进、测试、模型接入、性能优化和新并行能力贡献。无需上游仓库写权限，即可通过 Fork 参与开发。
请使用公开 Issue 和 Pull Request 记录方案、评审决定和验证结果，便于社区共同维护。

## 选择贡献方向

- **反馈问题**：先搜索[已有 Issue](https://github.com/mindspore-ai/hyper-parallel/issues)，再使用[问题模板](https://github.com/mindspore-ai/hyper-parallel/issues/new/choose)提交。
- **改进文档或测试**：小范围修正可以直接提交聚焦的 PR。英文文档与同目录的 `.zh.md` 中文版本同步更新。
- **提出新功能或框架接入**：较大实现开始前，先提交 Feature Issue 或 RFC，说明使用场景、接口、兼容性影响、硬件范围和验收标准，并请[相关模块维护者](CODEOWNERS)评审方案。
- **优化性能**：说明工作负载和瓶颈，在相同模型、硬件、精度、并行配置及测量方法下，提供可复现的正确性与性能对比。

不确定修改归属时，可在 Issue 中描述需求，并参考 [CODEOWNERS](CODEOWNERS) 联系模块维护者。
请保持尊重、回应技术意见，并将决定记录在 Issue 或 PR 中。

## 准备开发环境

先在 GitHub 上 Fork [HyperParallel](https://github.com/mindspore-ai/hyper-parallel)，替换以下命令中的 `YOUR_GITHUB_USERNAME`，从目标上游分支创建开发分支。
普通开发面向 `master`；版本修复应先与发布维护者确认目标版本分支。

```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/hyper-parallel.git
cd hyper-parallel
git remote add upstream https://github.com/mindspore-ai/hyper-parallel.git
git fetch upstream
git switch -c feat/your-change upstream/master
```

使用自己习惯的工具（如 `venv` 或 Conda）创建并激活独立 Python 环境，Python 版本需符合安装指南要求，环境名称自行选择。
按照[安装指南](docs/installation.zh.md)准备支持的 PyTorch 与硬件运行时组合，再在该环境中执行以下命令，以 editable 模式安装当前代码：

```bash
python -m pip install -e .
python -m pip install pytest
python -c "import hyper_parallel; print(hyper_parallel.__file__)"
```

输出路径必须指向当前代码目录。若指向其他工作区或旧安装，请在目标仓库重新安装。
原生组件还需按照安装指南执行构建及环境激活，单独的 Python editable 安装不会构建这些组件。

依赖详情见[开发环境](docs/contributing/dev_environment.zh.md)，身份认证与账号配置见 [GitHub 协作手册](docs/contributing/github_development_manual.zh.md)。
涉及 RL 的工作从 [RL 入口规则](.agent/rules/hyper-rl.md)开始，并完成其独立运行时配置。

## 遵循编程与文档规范

修改前阅读 [AGENTS.md](AGENTS.md) 和[编程规范](.agent/rules/code-style.md)。具体要求及例外以编程规范为准。

- Python 文件添加 Apache 2.0 头，遵循 PEP 8，行宽约 120 字符，公开 API 提供类型标注和 Google 风格文档字符串。C++ 遵循仓库 `.clang-format` 及 C++ 规范。
- 默认在模块顶部导入，直接使用原生 PyTorch API。涉及分片、通信、流或内存管理时，遵循[分布式实现规则](.agent/rules/distributed.md)。
- 保持修改聚焦。行为、公开接口、依赖和检查点兼容性变化需要配套测试与文档。
- 默认文档使用英文，同步修改对应 `.zh.md`。检查相对链接、锚点、示例命令和 API 签名。
- 修复 lint 和测试失败的根因；必要例外及未验证配置应在 PR 中说明。

在已激活的环境中安装本地提交钩子并执行检查：

```bash
bash scripts/pre-commit/install.sh
python .agent/skills/autogit/scripts/autogit.py check
python .agent/scripts/check_agents_catalog.py
git diff --check
```

检查会根据文件类型选择 pylint、lizard、C++／Shell 检查器、codespell、markdownlint 等工具。
修改 Markdown 必须运行目录检查；该检查只校验技能与代理目录，因此文档链接需单独检查。
安装方式与工具覆盖见 [pre-commit 指南](scripts/pre-commit/README.md)。
`autogit check` 会临时暂存修改以识别检查范围，建议在只包含本次任务修改的工作区运行。
AutoGit 的远程 PR 流程面向 GitCode；提交 GitHub PR 时使用 Git 和 GitHub 工具。

## 验证修改

根据变化的行为选择测试。目录、标记和启动方式见[测试规则](.agent/rules/testing.md)与[测试指南](docs/contributing/testing.zh.md)。

| 修改类型 | 所需验证证据 |
| --- | --- |
| 纯文档 | Markdown lint、目录检查、修改的链接和锚点，以及适用的命令或示例验证。说明运行时测试不适用的原因。 |
| 可独立测试的逻辑 | `tests/ut/` 中受影响的单元测试，问题修复应包含回归用例。 |
| 分片、集合通信、并行组合或检查点恢复 | 在受影响硬件上执行相关分布式 ST，并按场景验证数值与状态恢复。 |
| 性能或内存优化 | 正确性检查，以及相同配置下修改前后的对比；按场景记录吞吐、单步时间、峰值内存及测量条件。 |
| 安装包或依赖修改 | 在声明支持的组合上验证构建、干净环境安装、导入和相关示例。 |

例如执行受影响的 shard 单元测试；影响范围需要时再运行全量单元测试：

```bash
python -m pytest -vs tests/ut/core/shard/
python -m pytest -vs tests/ut/
```

单元测试不能初始化分布式任务。分布式 ST 使用测试规则中指定的仓库启动器，遵循选定用例的设备要求，不要统一再套一层 `torchrun`。
RL ST 独立于主项目 PR 门禁运行，主项目 CI 通过不代表 RL ST 已完成验收。

在 PR 中记录被测提交、命令、模型与配置、Python 和 PyTorch 版本、设备型号与数量及结果。
Ascend 还需记录 `torch-npu`、CANN 和驱动版本；NVIDIA GPU 需记录 CUDA 和驱动版本。
明确列出缺少的硬件与未运行用例，并与维护者协调必要的硬件验证。跳过测试或仅成功解析配置不能记为硬件验证通过。

## 提交并迭代 Pull Request

1. 每个 PR 聚焦一个可独立评审的修改。使用 Conventional Commit，例如 `docs: clarify installation`，英文主题使用祈使句，长度约 80 字符。
2. 每个 PR 保持一个提交。提交 PR 前压缩开发过程中的临时提交，后续修复 amend 到同一提交。改写历史前先备份；仅更新自己的 PR 源分支，并使用带预期远端 SHA 的 `--force-with-lease`。禁止强推共享主分支或版本分支。
3. 推送至个人 Fork，并向正确的上游分支发起 PR。使用仓库[默认模板](.github/PULL_REQUEST_TEMPLATE.md)或[中文模板](.github/PULL_REQUEST_TEMPLATE.zh-CN.md)，说明问题、修改后行为、范围、兼容性影响与实际测试结果。关联真实 Issue；没有适用 Issue 时明确说明，如实填写自检项。
4. 设计、实现或必要证据尚未完成时使用 Draft PR。准备好后，向 [CODEOWNERS](CODEOWNERS) 中相关维护者请求评审。
5. 回应评审意见，同步更新测试与中英文文档，并为最新提交取得有效 CI 和评审结果。相关问题得到处理后再解决评审讨论。

对于与 GitCode 配对的 PR，更新约定的源分支，由同步服务处理对应请求，避免在两端产生竞争修改。
遵循目标分支的实时[仓库规则](https://github.com/mindspore-ai/hyper-parallel/rules)以及适用的身份与 CLA 检查。

## CI、评审与合入

Jenkins 执行项目流水线并向 PR 回报测试及构建结果。GitHub 工作流检查 CI 标签有效期和评审票数。
当前 `master` 门禁包括：

| 门禁 | 贡献者需要了解的要求 |
| --- | --- |
| `label-check` | CI 通过标签必须有效。`ci-pipeline-passed` 和 `ci-pipeline-passed-but-overtime` 在添加后 24 小时过期；新增提交需要新的 CI 结果。超时标签保留耗时异常信息。 |
| `all-reviewers-approved` | 一名评审人时需一次批准；两名及以上评审人时需两名不同评审人批准。批准必须针对当前提交，作者本人不计入。未解决的修改请求或待处理团队评审仍会阻断。 |
| 原生分支规则 | 最新推送需要独立批准，推送后旧批准失效，评审讨论需解决。其他要求以目标分支实时规则为准。 |

详细策略见[评审与合入规则](docs/contributing/github_auto_merge.zh.md)。通过某一门禁不能替代其他门禁，维护者根据有效分支规则判断合入条件。

验证失败时先查看失败阶段与日志。代码回归由作者在模块维护者协助下修复，执行资源、环境和结果回报问题由 CI 维护者处理。
修复根因后，通过仓库支持的入口申请重测，确认结果对应当前提交。不要手工添加通过标签或关闭检查来促成合入。

## 版本节奏与兼容性建设

社区计划约**每月一个 Beta、每季度一个 Stable**。Beta 用于提前试用和收集下游反馈，Stable 提供通过验收的部署基线。
实际时间与范围取决于候选版本验证和维护者评审。SemVer、打包、发布说明、文档、已知限制及升级指引见[发布流程](docs/contributing/release.zh.md)。

**2026 年 Q4 CI/CD 计划**将验证范围扩展到 Ascend NPU 与 NVIDIA GPU 上的 PyTorch 2.9 至最新稳定版本，并跟进最新两个 PyTorch 版本。
这是路线图目标，实际支持组合以已发布验证结果为准。运行时尚未适配的组合标记为待验证，并关联跟踪 Issue。
即使随 Stable 发布，标为 `DEMO` 的能力仍受其声明的限制约束。

### 下游接入与 VeOmni

下游以可选安装包依赖方式使用 HyperParallel，各项目独立发布。
接入 PR 应提供安装方式、用户配置、可运行示例、兼容边界及可复现的正确性与性能结果，并与适配代码同步更新下游使用指南。

计划中的 VeOmni 流水线每周更新固定的上游版本基线，在 HyperParallel 修改合入前执行验证。
新基线通过后再切换；失败则保留上一个通过组合并跟踪问题。必要接入用例需记录两个项目的提交及精确环境。
VeOmni 首版接入面向 Ascend，GPU 后续支持。这些计划中的检查在接入中启用后成为合入要求，未有运行证据前不能描述为已经执行。

## 日常维护与问题跟进

可复现缺陷或跨 PR 问题通过[公开 Issue](https://github.com/mindspore-ai/hyper-parallel/issues/new/choose)提交，PR 特定问题保留在该 PR 中。
提供预期与实际行为、版本、设备、配置、最小复现和脱敏日志。数值或性能问题补充比较方法与基线，附件中移除凭据和私有训练数据。

模块维护者评审方案与实现，PR 作者处理其引入的回归，CI 维护者处理基础设施问题。
在原 Issue 中记录根因、受影响与修复版本以及回归结果。

对于拟接入的 VeOmni，HyperParallel 负责安装包、适配代码、VeOmni 仓内使用指南、Ascend 测试资源及每周兼容验证。
VeOmni 维护者保留其仓内评审、合入与发布决定权。

## 许可证

贡献遵循仓库的 [Apache 2.0 许可证](LICENSE)。引入第三方代码时，保留必要署名与许可信息，并按需更新[第三方软件声明](Third_Party_Open_Source_Software_Notice)。
