# Development Environment

**English** | [简体中文](dev_environment.zh.md)

## Prerequisites

For documentation-only changes, follow the [contributing guide](../../CONTRIBUTING.md#prepare-your-development-environment);
framework runtimes and native build tools are unnecessary unless a changed example needs them.
For Python development, use an isolated environment, Git, a supported PyTorch installation, and the affected module's
dependencies. The compiler, SDK, and build tools below apply when building native components:

- Python 3.10 / 3.11 / 3.12
- Development headers for the current Python, plus `setuptools`, `wheel`, and `pybind11`
- Git
- GCC/G++ with C++17 support, GNU Make, binutils, coreutils, tar, sed, and awk
- A complete CANN >= 9.1.0 toolkit/ops development environment for Ascend NPU development
- CMake ≥ 3.18
- Ninja for native targets

## Clone Your GitHub Fork

Fork the upstream repository first and replace `YOUR_GITHUB_USERNAME`. Cloning your fork sets `origin` automatically:

```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/hyper-parallel.git
cd hyper-parallel
git remote add upstream https://github.com/mindspore-ai/hyper-parallel.git
git fetch upstream
git switch -c feat/your-change upstream/master
```

For an existing clone, inspect `git remote -v` before changing remotes. Keep `origin` pointing to your fork and
`upstream` to the main repository. Paired GitCode requests follow the [submission guidance](../../CONTRIBUTING.md#submit-and-iterate-on-a-pull-request).

## Build a Development Version

```bash
# Python source development only, without optional native components.
pip install -e .

# Local native development: first source your selected official CANN environment.
source /usr/local/Ascend/cann/set_env.sh
./build.sh --multicore on --soc-list ascend910b,ascend910_93
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
source build/native/payload/hyper_parallel/core/multicore/lib/set_env.bash
python your_program.py

# The build also produces a wheel; install the exact path printed by build.sh.
wheel_path=/absolute/path/printed/by/build.sh
pip install "${wheel_path}"
source "$(command -v hyper_parallel_multicore_set_env.bash)"
```

Running `python setup.py bdist_wheel` directly produces a core-only wheel. The setup/pip phase does not download or compile native dependencies. See the [Installation Guide](../installation.md) for other public build options, and [Multicore Build and Usage](../../hyper_parallel/core/multicore/docs/build.md) for Multicore dependencies and CANN custom OPP activation.

## Install Development Dependencies

```bash
python -m pip install pytest pre-commit PyYAML
bash scripts/pre-commit/install.sh
```

The installer prepares the configured pylint, lizard, and Node-based markdownlint hook environments. Documentation-only
contributors can omit pytest. See the [pre-commit guide](../../scripts/pre-commit/README.md) for other lint tools.

## Coding Standards

The project follows these standards; see [code-style.md](../../.agent/rules/code-style.md) for the authoritative rules:

- **License**: Every `.py` file must include the Apache 2.0 header in its first 16 lines.
- **Style**: PEP 8, with lines around 120 characters.
- **Naming**: `PascalCase` for classes, `snake_case` for functions and variables, and `_leading_underscore` for private names.
- **Docstrings**: Google style (`Args:`, `Returns:`, `Raises:`, `Example:`).
- **Type hints**: Required on public function signatures.
- **Errors**: Use `ValueError` with descriptive messages; validate at boundaries.
- **Imports**: Use module-level imports in `core/`, `collectives/`, and `tests/`. Import Torch directly without backend dispatch. Optional dependencies follow the exceptions in the coding rules.

## Git Workflow

- **Commits**: Conventional Commits (`feat:`, `fix:`, `docs:`), with an imperative subject around 80 characters.
- **Branch**: Name feature branches `feat/xxx`, `fix/xxx`, or `docs/xxx`.
- **Squash**: Squash work-in-progress commits before opening a PR.

## AI Agent Tools

The project provides AI agent skills in `.agent/skills/`:

| Skill | Purpose |
|-------|---------|
| `autogit` | Automate the GitCode fork workflow: commit, PR, status, and squash |
| `code-review` | Review distributed correctness, stream synchronization, and memory safety |
| `dist-op-dev` | Distributed operator development workflow |
