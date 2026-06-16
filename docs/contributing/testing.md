# 测试流程规范

## 测试框架

使用 pytest，配合项目自定义标记 `@arg_mark`（定义在 `tests/common/mark_utils.py`）。

## 测试目录结构

| 目录 | 说明 | 是否需要分布式环境 |
|------|------|-------------------|
| `tests/ut/` | 单元测试 | 不需要 |
| `tests/st/torch/` | PyTorch 分布式系统测试 | 需要（torchrun） |
| `tests/st/mindspore/` | MindSpore 分布式系统测试 | 需要（msrun，8卡） |

## 运行测试

### 单元测试（本地单卡）

```bash
pytest tests/ut/
```

### PyTorch 分布式测试

```bash
torchrun --nproc_per_node=8 -m pytest tests/st/torch/
```

### MindSpore 分布式测试

```bash
msrun_case()  # 项目封装的 msrun 启动函数，默认 8 卡
```

### 按模块运行

```bash
# 仅运行 shard 相关 UT
pytest tests/ut/core/shard/

# 仅运行 activation_checkpoint UT
pytest tests/ut/core/activation_checkpoint/
```

## 测试标记

项目使用 `@arg_mark` 标记系统，支持：

- **level0**：基础覆盖测试，Gloo CPU backend 可运行
- **level1**：需要 NPU 的测试
- **feature 标记**：按特性模块标记（如 `@arg_mark(arg0=['arg0'], arg1=['arg1'])`）

## 测试编写规范

1. **新增功能必须配套 UT**：覆盖率目标 ≥ 80%
2. **分布式测试需要同时覆盖 MindSpore 和 PyTorch 后端**（如适用）
3. **使用项目封装的分布式启动函数**：`torchrun_case()` 和 `msrun_case()`
4. **优先使用 Gloo CPU backend 进行 UT**：减少 NPU 资源占用，`tests/common/` 中提供了 Gloo 适配器

## 测试覆盖率目标

| 模块 | 覆盖率目标 |
|------|-----------|
| `core/activation_checkpoint/` | ≥ 95% |
| `core/shard/` | ≥ 80% |
| `core/dtensor/` | ≥ 80% |
| `core/fully_shard/` | ≥ 80% |
| `core/pipeline_parallel/` | ≥ 70% |
| `core/optimizer/` | ≥ 80% |

## CI 流程

PR 提交时会自动运行：
1. **Lint 检查**：pylint + markdownlint（通过 pre-commit hook）
2. **Level0 UT**：Gloo CPU backend 基础覆盖
3. **Codecheck**：静态代码检查（C0116 docstring 等）