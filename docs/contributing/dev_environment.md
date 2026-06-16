# 开发环境搭建

## 前置条件

- Python 3.10 / 3.11 / 3.12
- Git
- CANN 工具链（Ascend NPU 开发需要）
- CMake ≥ 3.18

## 克隆仓库

```bash
git clone https://gitcode.com/mindspore/hyper-parallel.git
cd hyper-parallel
```

## Fork 工作流

采用 origin（fork）+ upstream（主仓库）模式：

```bash
# 在 GitCode 上 fork 仓库后
git remote add origin https://gitcode.com/<your-username>/hyper-parallel.git
git remote add upstream https://gitcode.com/mindspore/hyper-parallel.git
```

## 构建开发版本

```bash
# MindSpore 后端
python setup.py bdist_wheel
pip install -e .

# 同时构建 PyTorch + MindSpore 后端
BUILD_TORCH_EXTENSION=true pip install -e .
```

## 安装开发依赖

```bash
pip install pytest pylint markdownlint
```

## 代码规范

项目遵循以下代码规范（详见 `.agent/rules/code-style.md`）：

- **License**：所有 `.py` 文件需包含 Apache 2.0 header（前 16 行）
- **Style**：PEP 8，~120 字符行宽
- **Naming**：类 `PascalCase`，函数/变量 `snake_case`，私有 `_leading_underscore`
- **Docstrings**：Google 风格（`Args:`、`Returns:`、`Raises:`、`Example:`）
- **Type hints**：公共函数签名必须包含类型提示
- **Errors**：使用 `ValueError` 配合描述性消息，仅在边界处验证
- **Imports**：
  - `core/`、`collectives/`、`tests/` 等：模块顶部 import
  - `platform/torch/**`、`platform/mindspore/**`：方法内 lazy import torch/mindspore

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
| `platform-dev` | 平台抽象层开发 |