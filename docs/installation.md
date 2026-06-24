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
| `pip install 'hyper-parallel[mindspore]'` | 通用依赖 + `mindspore>=2.10`                         | 使用 MindSpore 后端               |
| `pip install 'hyper-parallel[torch]'`     | 通用依赖 + `torch==2.9.1` + `torch-npu==2.9.1`       | 使用默认 PyTorch 2.9 后端           |
| `pip install 'hyper-parallel[torch26]'`   | 通用依赖 + `torch==2.6.0` + `torch-npu==2.6.0.post3` | 使用 PyTorch 2.6 后端             |
| `pip install 'hyper-parallel[torch27]'`   | 通用依赖 + `torch==2.7.1` + `torch-npu==2.7.1`       | 使用 PyTorch 2.7 后端             |
| `pip install 'hyper-parallel[torch29]'`   | 通用依赖 + `torch==2.9.1` + `torch-npu==2.9.1`       | 显式使用 PyTorch 2.9 后端           |
| `pip install 'hyper-parallel[all]'`       | 通用依赖 + MindSpore + 默认 PyTorch 2.9                | 同一环境需要同时使用两种后端，PyTorch 默认 2.9 |

zsh 等 shell 下建议给带 extras 的包名加引号，避免 `[]` 被解释为通配符。

## 2. 从源码编译 whl 包

基于源码构建 hyper-parallel 可选择编译 `multicore`、`symmetric memory`、`custom ops` 三个 native 模块。

通过 `build.sh` 构建 whl 支持以下编译参数：

| 参数             | 默认值         | 可选值                                                   | 说明                                                                              |
|----------------|-------------|-------------------------------------------------------|---------------------------------------------------------------------------------|
| `--multicore`  | `mindspore` | `off`、`mindspore`、`ms`、`torch`、`pytorch`、`all`、`both` | 控制 multicore 模块编译范围；`ms` 等价于 `mindspore`，`pytorch` 等价于 `torch`，`both` 等价于 `all` |
| `--shmem`      | `all`       | `off`、`mindspore`、`ms`、`torch`、`pytorch`、`all`、`both` | 控制 symmetric memory 模块编译范围；`all` 同时编译公共库、MindSpore wrapper 和 PyTorch wrapper    |
| `--custom-ops` | `on`        | `on`、`off`                                            | 控制 MindSpore custom ops 是否编译                                                    |
| `--strict`     | `on`        | `on`、`off`                                            | 控制 native 模块构建失败时是否终止 whl 构建；`off` 会记录 warning 并继续打包                             |

源码构建 hyper-parallel 环境要求如下：

| 环境项               | 要求                                    | 说明                                                                            |
|-------------------|---------------------------------------|-------------------------------------------------------------------------------|
| Python            | 3.10、3.11 或 3.12                      | 构建出的 whl 仅可装到对应 Python 小版本的解释器上                                               |
| 主机 GCC            | [7.3.0, 11.3.0]                       | 与 MindSpore 编译策略一致                                                            |
| CMake             | ≥ 3.18                                | native 扩展构建需要                                                                 |
| CANN 工具链          | 启用 native 扩展时需要可用的 `ASCEND_HOME_PATH` | 脚本会尝试 source `/usr/local/Ascend/cann/set_env.sh`                              |
| MindSpore         | ≥ 2.10                                | 当 `--custom-ops on`，`--multicore mindspore/all`，或 `--shmem all/mindspore` 时需要 |
| PyTorch 及 NPU 适配包 | PyTorch 后端对应版本                        | 当 `--multicore torch/all`，或 `--shmem all/torch` 时需要                           |

```bash
git clone https://gitcode.com/mindspore/hyper-parallel.git
cd hyper-parallel
./build.sh
./build.sh --multicore all --shmem all --custom-ops on
./build.sh --multicore torch --shmem torch --strict off
./build.sh --multicore off --shmem off --custom-ops off
pip install dist/hyper_parallel-*.whl
```

> 注意事项：构建出的 whl 对运行环境的 glibc 版本有要求，安装环境的 glibc 需不低于编译环境的 glibc 版本。
> 如需部署到旧系统，请在更老的发布镜像内编译；例如在 OpenEuler 22.03（glibc 2.34）编出的 whl 无法在 glibc < 2.34 的环境运行。

## 3. 验证安装

安装完成后，可以先验证核心模块是否可导入：

```python
import hyper_parallel as hp

print(hp.get_platform())
```
