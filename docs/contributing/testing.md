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

### Unit Tests (No Distributed Setup)

```bash
python -m pytest -vs tests/ut/core/shard/
# Run the full unit suite when the change scope requires it.
python -m pytest -vs tests/ut/
```

### PyTorch Distributed Tests

Run the selected pytest launcher directly; it starts the required workers through `torchrun_case()` or `parallel_run()`.
For example, run this existing Gloo ST from the repository root with an editable installation of this checkout:

```bash
(
  export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
  export HYPER_PARALLEL_TEST_DEVICE_TYPE=cpu
  cd tests/torch/shard
  python -m pytest -vs test_base_dtensor_init.py::test_base_dtensor_init_group1_gloo
)
```

This launcher resolves worker filenames relative to its working directory, so run it in `tests/torch/shard`.
The root on `PYTHONPATH` makes the repository's `tests.common` helpers available to child processes; it does not replace
the editable package installation. Explicitly select `cpu` because the `_gloo` test name and platform marker do not
select a backend. The subshell restores your original directory and environment when it exits.

Do not wrap launcher tests in an outer `torchrun`, which would start the launcher once per rank and duplicate jobs.
Check the selected case's platform markers, worker count, and runtime before running NPU/GPU ST. Missing hardware must
be reported as untested and coordinated with maintainers. The [testing rules](../../.agent/rules/testing.md) define
launcher and worker responsibilities.

### Run by Module

```bash
# Run only shard unit tests.
python -m pytest -vs tests/ut/core/shard/

# Run only activation-memory unit tests.
python -m pytest -vs tests/ut/core/activation_memory/
```

## Test Markers

Use the arguments accepted by [mark_utils.py](../../tests/common/mark_utils.py):

```python
@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
```

`level_mark` records the test tier; it does not select hardware. Use `plat_marks` for platform requirements and check
the launcher's worker configuration for distributed resource needs. CPU/Gloo multiprocess checks are ST, not UT.

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

Run applicable lint and tests locally before submission. Local pre-commit hooks check staged files; they do not start
the remote pipeline. Jenkins runs the configured code checks and test suites and reports the actual stages on the PR.
GitHub Actions separately reconciles CI-label validity and review approvals.

Use the [contributing guide](../../CONTRIBUTING.md#when-a-gate-fails) for Jenkins retests and gate troubleshooting.
Include the tested commit, commands, environment, results, and unrun cases in the PR. A green local check does not
replace the target branch's required remote checks.
