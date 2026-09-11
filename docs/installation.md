# 安装指南

HyperParallel 提供两种安装方式：

- **pip 安装**：安装已经构建好的 `hyper-parallel` 包，并通过 extras 选择运行时深度学习框架依赖。
- **源码构建**：使用 `./build.sh` 生成 whl 包，并按构建参数决定是否编译 native 扩展。

如果只需要安装已发布的包，优先使用 `pip install`。如果需要在本机生成 whl，或需要调整 native 扩展构建配置，再使用源码构建。

## 1. 使用 pip 安装

`pip install` 的 extras 只控制 Python 运行时依赖，不控制 native 扩展的编译。

| 命令                                        | 安装内容                                             | 适用场景                          |
|-------------------------------------------|--------------------------------------------------|-------------------------------|
| `pip install hyper-parallel`              | 仅安装通用依赖，不安装深度学习框架                                | 已自行管理框架版本，或只使用不依赖框架的能力        |
| `pip install 'hyper-parallel[mindspore]'` | 通用依赖 + `mindspore>=2.10`                         | 使用受支持的 MindSpore 后端          |
| `pip install 'hyper-parallel[torch]'`     | 通用依赖 + `torch==2.9.1` + `torch-npu==2.9.1`       | 使用默认 PyTorch 2.9 后端           |
| `pip install 'hyper-parallel[torch26]'`   | 通用依赖 + `torch==2.6.0` + `torch-npu==2.6.0.post3` | 使用 PyTorch 2.6 后端             |
| `pip install 'hyper-parallel[torch27]'`   | 通用依赖 + `torch==2.7.1` + `torch-npu==2.7.1`       | 使用 PyTorch 2.7 后端             |
| `pip install 'hyper-parallel[torch29]'`   | 通用依赖 + `torch==2.9.1` + `torch-npu==2.9.1`       | 显式使用 PyTorch 2.9 后端           |
| `pip install 'hyper-parallel[all]'`       | 通用依赖 + MindSpore + 默认 PyTorch 2.9                  | 同一环境需要同时使用两种后端             |

zsh 等 shell 下建议给带 extras 的包名加引号，避免 `[]` 被解释为通配符。

## 2. 从源码编译 whl 包

统一入口调用所选组件的构建脚本，再将成功组件的产物组装为一个 wheel。indexed Dataset C++ helper
每次都会编译；Multicore 包含其私有 SHMEM 依赖，不再提供独立 SHMEM 开关。

| 参数 | 默认值 | 可选值 | 说明 |
|------|--------|--------|------|
| `--multicore` | `on` | `on`、`off` | 编译 Torch Multicore 及其私有 SHMEM。 |
| `--soc-list` | `ascend910b,ascend910_93` | SoC 列表 | 选择 Multicore kernel 目标。 |
| `--strict` | `off` | `on`、`off` | 所选 optional 组件失败时是否终止。 |
| `--jobs` | `nproc` | 正整数 | 设置 native 编译并行度。 |
| `--clean` | 关闭 | 无参数开关 | 重建所选组件的工作和安装目录。 |

```bash
./build.sh --help
./build.sh --multicore on --soc-list ascend910b,ascend910_93
./build.sh --strict on --jobs 24
# 安装本次命令打印的精确 wheel 路径
pip install /absolute/path/to/the-built-wheel.whl
```

通用构建要求：Python 3.10–3.12、对应开发头文件、setuptools、wheel、pybind11、C++17 主机工具链。
每个 wheel 对应一个 Python ABI 和主机架构；各 native 组件的 SDK、框架和设备要求由组件自行定义。
源码和预编译 wheel 的运行环境还必须满足最终 ELF 记录的 glibc、框架 ABI 及 SDK 约束。

每次统一构建重新生成 `build/native/payload/hyper_parallel`；只交付本次成功的组件。
`--strict on` 在所选组件失败时终止，适用于验收与发布；默认宽松模式允许 optional 组件失败后继续组包，
因此 wheel 生成不代表所有 optional 组件可用。
`python setup.py bdist_wheel` 不触发 native 编译；无显式 payload 时仅组装 Python 源码。
启用 Multicore 后，必须在启动 Python 前激活其 CANN custom OPP。

## 3. 激活 Multicore CANN custom OPP

源码或 editable 开发使用统一构建生成的 payload：

```bash
source /usr/local/Ascend/cann/set_env.sh
source build/native/payload/hyper_parallel/core/multicore/lib/set_env.bash
python your_program.py
```

wheel 安装后，使用安装到当前 Python 环境的定位脚本：

```bash
source /usr/local/Ascend/cann/set_env.sh
source "$(command -v hyper_parallel_multicore_set_env.bash)"
python your_program.py
```

激活脚本必须在启动业务 Python 进程前执行。脚本设置已交付 Multicore vendor 所需的
`ASCEND_CUSTOM_OPP_PATH` 和动态库路径。wheel 安装态不依赖源码仓中的 `build/` 目录；源码或 editable
开发态则从同一 checkout 的 `build/native/payload` 加载构建产物。

## 4. 验证安装

安装完成后，可以先验证核心模块是否可导入：

```python
import hyper_parallel as hp

print(hp.get_platform())
```
