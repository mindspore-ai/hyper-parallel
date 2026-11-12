# 测试流程规范

[English](testing.md) | **简体中文**

## 测试框架

使用 pytest，配合项目自定义标记 `@arg_mark`（定义在 `tests/common/mark_utils.py`）。

## 测试目录结构

| 目录 | 说明 | 是否需要分布式环境 |
|------|------|-------------------|
| `tests/ut/` | 单元测试 | 不需要 |
| `tests/torch/` | PyTorch 分布式系统测试 | 需要（torchrun） |

## 运行测试

### 单元测试（不初始化分布式任务）

```bash
python -m pytest -vs tests/ut/core/shard/
# 影响范围需要时，再运行全量单元测试。
python -m pytest -vs tests/ut/
```

### PyTorch 分布式测试

直接运行选定的 pytest 启动用例，由它通过 `torchrun_case()` 或 `parallel_run()` 创建所需工作进程。
例如，完成当前仓库的 editable 安装后，从仓库根目录执行以下已有 Gloo ST：

```bash
(
  export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
  export HYPER_PARALLEL_TEST_DEVICE_TYPE=cpu
  cd tests/torch/shard
  python -m pytest -vs test_base_dtensor_init.py::test_base_dtensor_init_group1_gloo
)
```

该启动器按当前工作目录解析工作进程文件名，因此需进入 `tests/torch/shard` 执行。
将仓库根目录加入 `PYTHONPATH` 是为子进程提供 `tests.common` 等测试辅助模块，不能替代项目的 editable 安装。
需显式选择 `cpu`，因为 `_gloo` 测试名及平台标记不会选择后端。子 Shell 退出后会恢复原目录和环境。

不要在启动用例外再套 `torchrun`，否则会在每个 rank 上重复启动整组任务。
执行 NPU/GPU ST 前，检查用例的平台标记、进程数和运行时要求。缺少硬件时明确记录未测试项，并与维护者协调验证。
启动器与工作进程的职责以[测试规则](../../.agent/rules/testing.md)为准。

### 按模块运行

```bash
# 仅运行 shard 相关 UT
python -m pytest -vs tests/ut/core/shard/

# 仅运行激活内存相关 UT
python -m pytest -vs tests/ut/core/activation_memory/
```

## 测试标记

使用 [mark_utils.py](../../tests/common/mark_utils.py) 支持的参数：

```python
@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
```

`level_mark` 表示测试分层，不用于选择硬件；平台要求使用 `plat_marks`，分布式资源需求还需查看启动器配置。
CPU/Gloo 多进程检查属于 ST，不属于 UT。

## 测试编写规范

1. **新增功能必须配套 UT**：覆盖率目标 ≥ 80%
2. **分布式测试覆盖真实的通信路径**：UT 用 mock 校验调用契约，ST 用多进程校验数值
3. **使用项目封装的分布式启动函数**：`torchrun_case()`
4. **优先编写仅需 CPU 的 UT**：按[测试规则](../../.agent/rules/testing.md)，分布式初始化和 Gloo 多进程检查放入 ST。

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

提交前运行适用的本地 lint 和测试。本地 pre-commit 钩子检查暂存文件，不会启动远程流水线。
Jenkins 执行已配置的代码检查与测试套件，并在 PR 中回报实际阶段；GitHub Actions 单独核对 CI 标签有效期和评审批准。

Jenkins 重测及门禁排障见[贡献指南](../../CONTRIBUTING.zh.md#门禁失败后的处理)。PR 中记录被测提交、命令、环境、结果及未运行用例。
本地检查通过不能替代目标分支要求的远程检查。
