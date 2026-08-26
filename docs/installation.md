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

基于源码构建 hyper-parallel 可选择编译 `multicore`、`symmetric memory`、`custom ops` 三个 native 模块。

通过 `build.sh` 构建 whl 支持以下编译参数：

| 参数             | 默认值         | 可选值                                                   | 说明                                                                              |
|----------------|-------------|-------------------------------------------------------|---------------------------------------------------------------------------------|
| `--multicore`  | `all`       | `off`、`mindspore`、`ms`、`torch`、`pytorch`、`all`、`both` | 控制 multicore 模块编译范围；`ms` 等价于 `mindspore`，`pytorch` 等价于 `torch`，`both` 等价于 `all` |
| `--shmem`      | `all`       | `off`、`mindspore`、`ms`、`torch`、`pytorch`、`all`、`both` | 控制 symmetric memory 模块编译范围；`all` 同时编译公共库、MindSpore wrapper 和 PyTorch wrapper    |
| `--custom-ops` | `on`        | `on`、`off`                                            | 启用或关闭 MindSpore custom ops 编译                                                 |
| `--soc-list`   | `ascend910b,ascend910_93` | `ascend910b`、`ascend910_93`、`ascend950` 的逗号分隔组合 | 控制 wheel 携带的 kernel；支持 `ascend910b`（910B）和 `ascend910_93`（910C），选择 `ascend950` 时记录 optional failure |
| `--strict`     | `off`       | `on`、`off`                                            | `off` 保留 wheel 并记录结构化 warning；显式 strict 开发构建使用 `on`                              |
| `--jobs`       | `nproc`     | 正整数                                                  | 控制 native 编译并行度                                                                  |
| `--clean`      | 关闭          | 无参数开关                                                 | 清理所选组件的工作和安装输出后重编译；保留依赖下载缓存                                                  |

源码构建 hyper-parallel 环境要求如下：

| 环境项               | 要求                                    | 说明                                                                            |
|-------------------|---------------------------------------|-------------------------------------------------------------------------------|
| Python            | 3.10、3.11 或 3.12                      | 构建出的 whl 仅可装到对应 Python 小版本的解释器上                                               |
| 主机 GCC            | >= 7.3.0                              | 7.3.0--11.3.0 为验证目标范围；更高版本仅告警                                          |
| CMake             | ≥ 3.18                                | native 扩展构建需要                                                                 |
| GNU Make          | 可从 `PATH` 找到                           | CMake 与 CANN 算子构建流程使用                                                        |
| CANN 工具链          | CANN 9.1.0                            | 预先 source 所选 CANN 的 `set_env.sh`；默认路径 `/usr/local/Ascend/cann/set_env.sh` 可由 `build.sh` 自动激活 |
| MindSpore         | >= 2.10                               | 当 `--custom-ops on`，`--multicore mindspore/all`，或 `--shmem all/mindspore` 时需要        |
| PyTorch 及 NPU 适配包 | 相互配套且 `_GLIBCXX_USE_CXX11_ABI=1` 的版本   | 当 `--multicore torch/all` 或 `--shmem all/torch` 时需要；构建使用活动环境中安装的配套版本       |

```bash
git clone https://gitcode.com/mindspore/hyper-parallel.git
cd hyper-parallel

# 默认 CANN 安装会在需要时自动 source；自定义安装路径需要在 build.sh 前显式 source。
./build.sh
./build.sh --multicore all --shmem all --custom-ops on --soc-list ascend910b,ascend910_93
./build.sh --multicore torch --shmem torch --strict off
./build.sh --multicore off --shmem off --custom-ops off
# 安装 build.sh 最后打印的精确 wheel 路径。
wheel_path=/absolute/path/printed/by/build.sh
pip install "${wheel_path}"
```

`build.sh` 每次都从组件安装目录重新组装 `build/native/payload/hyper_parallel`、生成 wheel，并打印本次
wheel 的精确路径；PYTHONPATH 开发直接复用同一 payload。单独执行某个组件脚本
成功后会刷新该组件的 payload 子目录，可直接用于局部增量开发。默认保留耗时的 SHMEM 和按 SoC 的 vendor
编译缓存；轻量 framework adapter 每次从按框架身份隔离的干净目录重编。`--clean` 用于显式全量重编所选
组件。锁定依赖缓存正确时直接复用，缺失或不一致时自动下载/刷新。

multicore 多 SoC 构建会让 ops-nn 分别编译各 SoC kernel。流程使用固定优先级
（`ascend910_93`/910C 优先于 `ascend910b`/910B）选择唯一的 host 制品，不受 `--soc-list` 顺序影响；
其他 vendor 在公共 host 构建输入身份和 ABI 校验一致后，只合入各自的 kernel/config。

> 注意事项：构建出的 whl 对运行环境的 glibc 版本有要求，安装环境的 glibc 需不低于编译环境的 glibc 版本。
> 如需部署到 glibc 较低的系统，请在满足目标 glibc 基线的发布镜像内编译；例如在 OpenEuler 22.03
>（glibc 2.34）编出的 whl 无法在 glibc < 2.34 的环境运行。
> 正式发布构建需使用版本指定的 glibc 基线，并通过 Level 1/全量发布用例。产物所需的最低运行时 glibc
> 由最终 ELF 依赖决定。

native 源码构建和正式预编译 wheel 均要求 CANN 9.1.0。构建前 source 所选 CANN 的 `set_env.sh`，
构建脚本读取其导出的 `ASCEND_HOME_PATH`；默认 CANN 路径由 `build.sh` 自动激活。

## 3. 使用 multicore 前激活自定义算子环境

wheel 和 PYTHONPATH 开发态都必须在启动业务或框架 Python 进程前 source multicore 制品自带的
`set_env.bash`，为调用 shell 激活相邻的 CANN custom OPP vendor。

PYTHONPATH 开发态：

```bash
source /usr/local/Ascend/cann/set_env.sh
export PYTHONPATH=/path/to/hyper-parallel:${PYTHONPATH:-}
source /path/to/hyper-parallel/build/native/payload/hyper_parallel/core/multicore/lib/set_env.bash
python application.py
```

wheel 安装态：

```bash
source /usr/local/Ascend/cann/set_env.sh
source "$(command -v hyper_parallel_multicore_set_env.bash)"
python application.py
```

wheel 安装完成后，从活动 Python 环境的 `bin` 目录 source 定位脚本，再启动业务或框架 Python 进程。

脚本必须早于 `import mindspore` 或 `import torch/torch_npu` 执行。未激活时报
`HP-NATIVE-OPP-NOT-ACTIVATED`；框架已经导入后才发现未激活时，报
`HP-NATIVE-OPP-ACTIVATION-TOO-LATE`，需要退出该 Python 进程、source 脚本后重新运行。

## 4. 验证安装

安装完成后，可以先验证核心模块是否可导入：

```python
import hyper_parallel as hp

print(hp.get_platform())
```
