# 安装指南

[English](installation.md) | **简体中文**

HyperParallel 提供两种安装方式：

- **pip 安装**：安装已经构建好的 `hyper-parallel` 包，并通过 extras 选择运行时深度学习框架依赖。
- **源码构建**：使用 `./build.sh` 生成 whl 包。

如果只需要安装已发布的包，优先使用 `pip install`。如果需要在本机生成 whl，或需要调整 native 扩展构建配置，再使用源码构建。

## 1. 使用 pip 安装

如果已有可用的 PyTorch 环境，可在该环境中安装发行包：

```bash
python -m pip install hyper-parallel
```

安装本地 wheel 时，可从[用 uv 管理开发依赖](#5-用-uv-管理开发依赖)的版本表中选择一个 NPU extra。
以下示例假设 `./wheels` 中已备齐与目标 Python 版本和架构匹配的 CPU 版 PyTorch 和 torchvision，
以及 TorchNPU 和 HyperMulticore wheel。

```bash
python -m pip install --find-links ./wheels 'torch==2.10.0+cpu' 'torchvision==0.25.0+cpu' \
    './hyper_parallel-xxx.whl[torch-npu-2-10]'
```

如果已有 VeRL 环境，安装本地 wheel 时可指定 `'./hyper_parallel-xxx.whl[verl]'`，一并安装 HyperParallel 适配器需要的 OmegaConf。

## 2. 从源码编译 whl 包

indexed Dataset C++ helper 每次都会编译。

```bash
# 安装项目使用的 uv 版本。
python -m pip install uv==0.11.16
```

```bash
uv sync --locked --python 3.12 --only-group build
source .venv/bin/activate
./build.sh --help
./build.sh
# 安装本次命令打印的精确 wheel 路径
pip install /absolute/path/to/the-built-wheel.whl
```

通用构建要求：Python 3.10–3.12、对应开发头文件、uv、pybind11、C++17 主机工具链。
每个 wheel 对应一个 Python ABI 和主机架构；各 native 组件的 SDK、框架和设备要求由组件自行定义。
源码和预编译 wheel 的运行环境还必须满足最终 ELF 记录的 glibc、框架 ABI 及 SDK 约束。

每次统一构建重新生成 `build/native/payload/hyper_parallel`；只交付本次成功的组件。
`uv build --wheel` 不触发 native 编译。无显式 payload 时仅组装 Python 源码。
如果用这个 wheel 运行 Indexed Dataset，首次导入前需在运行环境中安装 `pybind11>=2.12.0` 并准备 C++ 工具链。

## 3. 按需使用 HyperMulticore

MegaMoE、MegaGate 和 MegaMHC 替换使用独立安装的
[HyperMulticore](https://github.com/mindspore-ai/hyper-multicore)。
选择 NPU extra 时会同时安装 HyperMulticore。如果已有的 pip 训练环境只安装了 HyperParallel 基础包，
先激活本机的 CANN 环境，再安装匹配的 wheel，并在启动 Python 前激活自定义 OPP：

```bash
pip install /absolute/path/to/hyper_multicore.whl
source "$(command -v hyper_multicore_set_env.bash)"
python your_program.py
```

源码构建和框架兼容要求见 HyperMulticore 构建指南。
普通 HyperParallel 功能及未启用这些替换的模型不要求安装 HyperMulticore。

## 4. 验证安装

安装完成后，可以先验证核心模块是否可导入：

```python
import hyper_parallel as hp
import importlib.metadata as metadata

print(metadata.version("hyper_parallel"))
```

## 5. 用 uv 管理开发依赖

在 Linux 昇腾主机上，选择下表中的一个 NPU extra。每个选项都会安装 CPU 版 PyTorch 和 torchvision、
对应的 [TorchNPU 归档 wheel](https://ascend.devcloud.huaweicloud.com/pypi/torch-npu/)，以及
[HyperMulticore `0.1.0.dev0`](https://repo.mindspore.cn/pypi/nightly/simple/hyper-multicore/)。
支持的环境为 CPython 3.10–3.12、`aarch64` 或 `x86_64` 架构，要求 glibc 2.28 或更高版本。

| uv extra | PyTorch | torchvision | torch-npu |
| --- | --- | --- | --- |
| `torch-npu-2-9` | `2.9.0` | `0.24.0` | `2.9.0.post9` |
| `torch-npu-2-10` | `2.10.0` | `0.25.0` | `2.10.0.post7` |
| `torch-npu-2-11` | `2.11.0` | `0.26.0` | `2.11.0.post3` |
| `torch-npu-2-12` | `2.12.0` | `0.27.0` | `2.12.0.post3` |
| `torch-npu-2-13` | `2.13.0` | `0.28.0` | `2.13.0` |
| `torch-npu-2-14` | `2.14.0` | `0.29.0` | `2.14.0` |

在仓库根目录按锁文件同步独立的 Ascend 开发环境。以下示例选择 PyTorch 2.10。

```bash
UV_PROJECT_ENVIRONMENT=.venv-uv uv sync --locked --python 3.12 --extra torch-npu-2-10 --dev

# 在已激活 CANN 的 shell 中，激活 Python 环境。
source .venv-uv/bin/activate
```

`requirements.txt` 继续提供 pip 安装所用的运行时依赖范围。
变更依赖声明后，运行 `uv lock` 并提交更新后的 `uv.lock`。
