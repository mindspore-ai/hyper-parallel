# Installation Guide

**English** | [简体中文](installation.zh.md)

HyperParallel provides two installation methods:

- **pip installation**: Install a prebuilt `hyper-parallel` package and select runtime deep learning framework dependencies through extras.
- **Source build**: Use `./build.sh` to generate a wheel.

Prefer `pip install` when you only need a published package. Build from source when you need to generate a wheel locally or customize native extension builds.

## 1. Install with pip

The extras for `pip install` control Python runtime dependencies only, not native extension compilation.

| Command | Installed dependencies | Use case |
|---------|------------------------|----------|
| `pip install hyper-parallel` | Common dependencies only; no deep learning framework | You manage framework versions yourself or only use framework-independent capabilities |
| `pip install 'hyper-parallel[torch]'` | Common dependencies + `torch==2.9.1` + `torch-npu==2.9.1` | Default PyTorch 2.9 backend |
| `pip install 'hyper-parallel[torch26]'` | Common dependencies + `torch==2.6.0` + `torch-npu==2.6.0.post3` | PyTorch 2.6 backend |
| `pip install 'hyper-parallel[torch27]'` | Common dependencies + `torch==2.7.1` + `torch-npu==2.7.1` | PyTorch 2.7 backend |
| `pip install 'hyper-parallel[torch29]'` | Common dependencies + `torch==2.9.1` + `torch-npu==2.9.1` | Explicit PyTorch 2.9 backend |
| `pip install 'hyper-parallel[all]'` | Common dependencies + default PyTorch 2.9 | Default PyTorch backend |

In shells such as zsh, quote package names containing extras so that `[]` is not interpreted as a wildcard.

## 2. Build a Wheel from Source

The indexed Dataset C++ helper is always compiled.

```bash
./build.sh --help
./build.sh
# Install the exact wheel path printed by this build.
pip install /absolute/path/to/the-built-wheel.whl
```

General build requirements: Python 3.10–3.12, matching development headers, setuptools, wheel, pybind11, and a C++17 host toolchain. Each wheel targets one Python ABI and host architecture. Each native component defines its own SDK, framework, and device requirements. Runtime environments for source and prebuilt wheels must also satisfy the glibc, framework ABI, and SDK constraints recorded in the final ELF binaries.

Each unified build regenerates `build/native/payload/hyper_parallel` and ships only components that succeeded in that build. `python setup.py bdist_wheel` does not trigger native compilation; without an explicit payload, it packages Python sources only.

## 3. Optional HyperMulticore Integration

MegaMoE, MegaGate and MegaMHC replacements use the separately installed
[HyperMulticore](https://github.com/mindspore-ai/hyper-multicore) package. HyperParallel does not
install or compile it automatically. Install a wheel matching the training environment and activate
its CANN custom OPP before starting Python:

```bash
pip install /absolute/path/to/hyper_multicore.whl
source /usr/local/Ascend/cann/set_env.sh
source "$(command -v hyper_multicore_set_env.bash)"
python your_program.py
```

The CANN activation path follows the local installation. HyperMulticore's build guide describes
source builds and framework compatibility. Ordinary HyperParallel operations and models without
these replacements do not require HyperMulticore.

## 4. Verify the Installation

After installation, first verify that the core module imports successfully:

```python
import hyper_parallel as hp
import importlib.metadata as metadata

print(metadata.version("hyper_parallel"))
```
