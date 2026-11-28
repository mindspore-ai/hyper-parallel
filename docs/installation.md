# Installation Guide

**English** | [简体中文](installation.zh.md)

HyperParallel provides two installation methods:

- **pip installation**: Install a prebuilt `hyper-parallel` package and select runtime deep learning framework dependencies through extras.
- **Source build**: Use `./build.sh` to generate a wheel.

Prefer `pip install` when you only need a published package. Build from source when you need to generate a wheel locally or customize native extension builds.

## 1. Install with pip

Install the published package into an environment with your chosen PyTorch stack:

```bash
python -m pip install hyper-parallel
```

To choose a TorchNPU version when installing a local wheel, select one NPU extra from the table in
[Manage development dependencies with uv](#5-manage-development-dependencies-with-uv). In this example,
`./wheels` contains CPU PyTorch, CPU torchvision, TorchNPU, and HyperMulticore wheels that match the target
Python version and architecture.

```bash
python -m pip install --find-links ./wheels 'torch==2.10.0+cpu' 'torchvision==0.25.0+cpu' \
    './hyper_parallel-xxx.whl[torch-npu-2-10]'
```

For the HyperParallel adapter in an existing VeRL environment, install the local wheel as `'./hyper_parallel-xxx.whl[verl]'`
to include OmegaConf.

## 2. Build a Wheel from Source

The indexed Dataset C++ helper is always compiled.

```bash
# Install the uv version used by this project.
python -m pip install uv==0.11.16
```

```bash
uv sync --locked --python 3.12 --only-group build
source .venv/bin/activate
./build.sh --help
./build.sh
# Install the exact wheel path printed by this build.
pip install /absolute/path/to/the-built-wheel.whl
```

General build requirements: Python 3.10–3.12, matching development headers, uv, pybind11, and a C++17 host toolchain. Each wheel targets one Python ABI and host architecture. Each native component defines its own SDK, framework, and device requirements. Runtime environments for source and prebuilt wheels must also satisfy the glibc, framework ABI, and SDK constraints recorded in the final ELF binaries.

Each unified build regenerates `build/native/payload/hyper_parallel` and ships only components that succeeded in that build. `uv build --wheel` does not trigger native compilation; without an explicit payload, it packages Python sources only.
To use Indexed Dataset from a source-only wheel, install `pybind11>=2.12.0` and prepare the C++ toolchain before the first import.

## 3. Optional HyperMulticore Integration

MegaMoE, MegaGate and MegaMHC replacements use the separately installed
[HyperMulticore](https://github.com/mindspore-ai/hyper-multicore) package. The NPU extras install HyperMulticore. In an existing pip-managed environment with a base HyperParallel
installation, activate your local CANN environment, install a compatible wheel, and activate the custom OPP before starting Python:

```bash
pip install /absolute/path/to/hyper_multicore.whl
source "$(command -v hyper_multicore_set_env.bash)"
python your_program.py
```

HyperMulticore's build guide describes source builds and framework compatibility. Ordinary HyperParallel operations
and models without these replacements do not require HyperMulticore.

## 4. Verify the Installation

After installation, first verify that the core module imports successfully:

```python
import hyper_parallel as hp
import importlib.metadata as metadata

print(metadata.version("hyper_parallel"))
```

## 5. Manage Development Dependencies with uv

On Linux Ascend hosts, choose one NPU extra. Each installs CPU builds of PyTorch and torchvision, the matching
[TorchNPU archive wheel](https://ascend.devcloud.huaweicloud.com/pypi/torch-npu/), and
[HyperMulticore `0.1.0.dev0`](https://repo.mindspore.cn/pypi/nightly/simple/hyper-multicore/).
The supported environment is CPython 3.10–3.12 on `aarch64` or `x86_64` with glibc 2.28 or later.

| uv extra | PyTorch | torchvision | torch-npu |
| --- | --- | --- | --- |
| `torch-npu-2-9` | `2.9.0` | `0.24.0` | `2.9.0.post9` |
| `torch-npu-2-10` | `2.10.0` | `0.25.0` | `2.10.0.post7` |
| `torch-npu-2-11` | `2.11.0` | `0.26.0` | `2.11.0.post3` |
| `torch-npu-2-12` | `2.12.0` | `0.27.0` | `2.12.0.post3` |
| `torch-npu-2-13` | `2.13.0` | `0.28.0` | `2.13.0` |
| `torch-npu-2-14` | `2.14.0` | `0.29.0` | `2.14.0` |

From the repository root, synchronize an isolated Ascend development environment. This example selects PyTorch 2.10.

```bash
UV_PROJECT_ENVIRONMENT=.venv-uv uv sync --locked --python 3.12 --extra torch-npu-2-10 --dev

# Activate the Python environment in a shell with CANN already active.
source .venv-uv/bin/activate
```

`requirements.txt` remains the source of runtime dependency ranges for pip installation.
After changing dependency declarations, run `uv lock` and commit the updated `uv.lock`.
