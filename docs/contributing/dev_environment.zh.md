# 开发环境搭建

[English](dev_environment.md) | **简体中文**

## 前置条件

纯文档修改按[贡献指南](../../CONTRIBUTING.zh.md#准备开发环境)准备工具，除非修改的示例需要，否则无需框架运行时和原生构建工具。
Python 开发需要独立环境、Git、支持的 PyTorch 安装及受影响模块的依赖；下列编译器、SDK 和构建工具适用于原生组件构建：

- Python 3.10 / 3.11 / 3.12
- 当前 Python 的开发头文件，以及 `setuptools`、`wheel`、`pybind11`
- Git
- GCC/G++（支持 C++17）、GNU Make、binutils、coreutils、tar、sed、awk
- CANN >= 9.1.0 的完整 toolkit/ops 开发环境（Ascend NPU 开发需要）
- CMake ≥ 3.18
- Ninja（native target 需要）

## 克隆个人 GitHub Fork

先 Fork 上游仓库并替换 `YOUR_GITHUB_USERNAME`。克隆个人 Fork 时会自动设置 `origin`：

```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/hyper-parallel.git
cd hyper-parallel
git remote add upstream https://github.com/mindspore-ai/hyper-parallel.git
git fetch upstream
git switch -c feat/your-change upstream/master
```

已有克隆先用 `git remote -v` 检查，再按需调整远端。`origin` 指向个人 Fork，`upstream` 指向主仓库。
配对的 GitCode 请求按[提交指南](../../CONTRIBUTING.zh.md#提交并迭代-pull-request)处理。

## 构建开发版本

```bash
# 仅 Python 源码开发（不编译 optional native 组件）
pip install -e .

# native 本地开发；先 source 用户选择的 CANN 官方环境
source /usr/local/Ascend/cann/set_env.sh
./build.sh --multicore on --soc-list ascend910b,ascend910_93
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
source build/native/payload/hyper_parallel/core/multicore/lib/set_env.bash
python your_program.py

# 该命令同时生成 wheel；安装 build.sh 最后打印的精确路径
wheel_path=/absolute/path/printed/by/build.sh
pip install "${wheel_path}"
source "$(command -v hyper_parallel_multicore_set_env.bash)"
```

直接执行 `python setup.py bdist_wheel` 只生成 core-only wheel，不在 setup/pip 阶段下载或编译 native 依赖。
其他公共构建选项见[安装指南](../installation.zh.md)。Multicore 的依赖和 CANN custom OPP 激活方式见
[Multicore 构建与使用](../../hyper_parallel/core/multicore/docs/build.zh.md)。

## 安装开发依赖

```bash
python -m pip install pytest pre-commit PyYAML
bash scripts/pre-commit/install.sh
```

安装脚本会准备已配置的 pylint、lizard 和基于 Node 的 markdownlint 钩子环境。纯文档贡献者可省略 pytest。
其他检查工具见 [pre-commit 指南](../../scripts/pre-commit/README.md)。

## 代码规范

项目遵循以下代码规范，完整要求及例外以[编程规范](../../.agent/rules/code-style.md)为准：

- **License**：所有 `.py` 文件需包含 Apache 2.0 header（前 16 行）
- **Style**：PEP 8，~120 字符行宽
- **Naming**：类 `PascalCase`，函数/变量 `snake_case`，私有 `_leading_underscore`
- **Docstrings**：Google 风格（`Args:`、`Returns:`、`Raises:`、`Example:`）
- **Type hints**：公共函数签名必须包含类型提示
- **Errors**：使用 `ValueError` 配合描述性消息，仅在边界处验证
- **Imports**：在 `core/`、`collectives/`、`tests/` 中默认于模块顶部导入，直接使用 Torch，不做后端分发。可选依赖按编程规范中列出的例外处理。

## Git 工作流

- **Commits**：Conventional Commits 格式（`feat:`、`fix:`、`docs:`），~80 字符标题，祈使语气
- **Branch**：功能分支命名 `feat/xxx` 或 `fix/xxx` 或 `docs/xxx`
- **Squash**：打开 PR 前压缩 WIP commits

## AI Agent 工具

项目配置了 AI Agent skills（`.agent/skills/`），支持：

| Skill | 用途 |
|-------|------|
| `autogit` | GitCode fork 工作流自动化（commit、PR、status、squash） |
| `code-review` | 分布式正确性、stream sync、内存安全审查 |
| `dist-op-dev` | 分布式算子开发流程 |
