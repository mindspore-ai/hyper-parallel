# 安装指南

[English](installation.md) | **简体中文**

HyperParallel 提供两种安装方式：

- **pip 安装**：安装已经构建好的 `hyper-parallel` 包，并通过 extras 选择运行时深度学习框架依赖。
- **源码构建**：使用 `./build.sh` 生成 whl 包。

如果只需要安装已发布的包，优先使用 `pip install`。如果需要在本机生成 whl，或需要调整 native 扩展构建配置，再使用源码构建。

## 1. 使用 pip 安装

`pip install` 的 extras 只控制 Python 运行时依赖，不控制 native 扩展的编译。

| 命令                                        | 安装内容                                             | 适用场景                          |
|-------------------------------------------|--------------------------------------------------|-------------------------------|
| `pip install hyper-parallel`              | 仅安装通用依赖，不安装深度学习框架                                | 已自行管理框架版本，或只使用不依赖框架的能力        |
| `pip install 'hyper-parallel[torch]'`     | 通用依赖 + `torch==2.9.1` + `torch-npu==2.9.1`       | 使用默认 PyTorch 2.9 后端           |
| `pip install 'hyper-parallel[torch26]'`   | 通用依赖 + `torch==2.6.0` + `torch-npu==2.6.0.post3` | 使用 PyTorch 2.6 后端             |
| `pip install 'hyper-parallel[torch27]'`   | 通用依赖 + `torch==2.7.1` + `torch-npu==2.7.1`       | 使用 PyTorch 2.7 后端             |
| `pip install 'hyper-parallel[torch29]'`   | 通用依赖 + `torch==2.9.1` + `torch-npu==2.9.1`       | 显式使用 PyTorch 2.9 后端           |
| `pip install 'hyper-parallel[all]'`       | 通用依赖 + 默认 PyTorch 2.9                            | 使用默认 PyTorch 后端                |

zsh 等 shell 下建议给带 extras 的包名加引号，避免 `[]` 被解释为通配符。

## 2. 从源码编译 whl 包

indexed Dataset C++ helper 每次都会编译。

```bash
# 安装项目使用的 uv 版本。
python -m pip install uv==0.11.16
```

```bash
./build.sh --help
./build.sh
# 安装本次命令打印的精确 wheel 路径
pip install /absolute/path/to/the-built-wheel.whl
```

通用构建要求：Python 3.10–3.12、对应开发头文件、uv、pybind11、C++17 主机工具链。
每个 wheel 对应一个 Python ABI 和主机架构；各 native 组件的 SDK、框架和设备要求由组件自行定义。
源码和预编译 wheel 的运行环境还必须满足最终 ELF 记录的 glibc、框架 ABI 及 SDK 约束。

每次统一构建重新生成 `build/native/payload/hyper_parallel`；只交付本次成功的组件。
`uv build --wheel` 不触发 native 编译；无显式 payload 时仅组装 Python 源码。

## 3. 按需使用 HyperMulticore

MegaMoE、MegaGate 和 MegaMHC 替换使用独立安装的
[HyperMulticore](https://github.com/mindspore-ai/hyper-multicore)。
在已有的 pip 训练环境中，安装匹配的 wheel，并在启动 Python 前激活 CANN custom OPP：

```bash
pip install /absolute/path/to/hyper_multicore.whl
source /usr/local/Ascend/cann/set_env.sh
source "$(command -v hyper_multicore_set_env.bash)"
python your_program.py
```

CANN 激活脚本路径以本机安装为准。源码构建和框架兼容要求见 HyperMulticore 构建指南。
普通 HyperParallel 功能及未启用这些替换的模型不要求安装 HyperMulticore。

## 4. 验证安装

安装完成后，可以先验证核心模块是否可导入：

```python
import hyper_parallel as hp
import importlib.metadata as metadata

print(metadata.version("hyper_parallel"))
```

## 5. 用 uv 管理开发依赖

当前草稿使用 HyperMulticore 下载占位地址。运行下列同步示例前，需等匹配的 wheel 发布后重新生成锁文件。

在仓库根目录按锁文件同步 CPU 开发环境。

```bash
# 将项目和 CPU 开发依赖安装到独立环境。
UV_PROJECT_ENVIRONMENT=.venv-uv uv sync --locked --python 3.12 --extra cpu --dev
```

Linux CUDA 12.8 开发可在独立环境中选择 `--extra gpu`。
在受支持的 Linux Ascend 主机上，`npu` extra 包含 PyTorch、TorchNpu 和 HyperMulticore。

```bash
# 同步锁定的 Ascend 依赖，包括 HyperMulticore wheel。
UV_PROJECT_ENVIRONMENT=.venv-npu uv sync --locked --python 3.12 --extra npu --dev

# 启动训练前，激活已安装的 CANN 和 HyperMulticore 环境。
source .venv-npu/bin/activate
source /usr/local/Ascend/cann/set_env.sh
source .venv-npu/bin/hyper_multicore_set_env.bash
```

`requirements.txt` 继续提供 pip 安装所用的运行时依赖范围。
变更依赖声明后，运行 `uv lock` 并提交更新后的 `uv.lock`。
