# Testing Standards

**English** | [简体中文](testing.zh.md)

## Test Framework

Use pytest with the project's `@arg_mark` decorator, defined in `tests/common/mark_utils.py`.

## Test Directory Structure

| Directory | Description | Distributed environment required |
|-----------|-------------|----------------------------------|
| `tests/ut/` | Unit tests | No |
| `tests/torch/` | PyTorch distributed system tests | Yes (torchrun) |

## Run Tests

### Unit Tests (Local, Single Device)

```bash
pytest tests/ut/
```

### PyTorch Distributed Tests

```bash
torchrun --nproc_per_node=8 -m pytest tests/torch/
```

### Run by Module

```bash
# Run only shard unit tests.
pytest tests/ut/core/shard/

# Run only activation_checkpoint unit tests.
pytest tests/ut/core/activation_memory/
```

## Test Markers

The project's `@arg_mark` system supports:

- **level0**: Basic coverage runnable with the Gloo CPU backend.
- **level1**: Tests requiring an NPU.
- **Feature markers**: Mark tests by feature, for example `@arg_mark(arg0=['arg0'], arg1=['arg1'])`.

## Test Authoring Standards

1. **New features must include unit tests**: Target coverage ≥ 80%.
2. **Distributed tests must exercise real communication paths**: Use mocks in UT to verify call contracts and multiple processes in ST to verify numerical behavior.
3. **Use the project's distributed launcher**: `torchrun_case()`.
4. **Prefer CPU-only UT**: Keep distributed initialization and Gloo multiprocess checks in ST, as required by the [testing rules](../../.agent/rules/testing.md).

## Coverage Targets

| Module | Target coverage |
|--------|-----------------|
| `core/activation_checkpoint/` | ≥ 95% |
| `core/shard/` | ≥ 80% |
| `core/dtensor/` | ≥ 80% |
| `core/fully_shard/` | ≥ 80% |
| `core/pipeline_parallel/` | ≥ 70% |
| `core/optimizer/` | ≥ 80% |

## CI Workflow

The following checks run when a PR is submitted:

1. **Lint**: pylint and markdownlint through the pre-commit hook.
2. **Level0 UT**: Basic CPU coverage.
3. **Codecheck**: Static checks, including C0116 docstring checks.
