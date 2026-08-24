# HyperParallel Docker 环境

本目录提供 Ascend NPU Docker 环境配置，用于安装和验证 HyperParallel。

## 快速部署

### 1. 获取已构建镜像

可以直接从 Quay.io 获取 PyTorch 2.9 NPU 镜像：

```bash
docker pull quay.io/hyper-parallel/hyper-parallel-npu:cann9.1.0-torch29
```

使用已拉取的镜像启动容器：

```bash
IMAGE=quay.io/hyper-parallel/hyper-parallel-npu:cann9.1.0-torch29 \
bash docker/run_hyper-parallel.sh
```

### 2. 从源码构建镜像

如果需要使用当前源码或自定义依赖，可以使用构建脚本：

```bash
bash docker/build_hyper-parallel_npu.sh hyper-parallel:npu
```

构建完成后启动本地镜像：

```bash
IMAGE=hyper-parallel:npu bash docker/run_hyper-parallel.sh --name hyper-parallel-npu --cards auto
```

### 3. 进入容器并验证

```bash
docker exec -it hyper-parallel-npu bash
source /usr/local/Ascend/cann/set_env.sh
python3 -c "import hyper_parallel as hp; print(hp.get_platform())"
```

## 文件说明

| 文件 | 说明 |
| --- | --- |
| `Dockerfile.hyper-parallel-npu` | 通用 HyperParallel NPU 镜像，默认 PyTorch 2.9 后端 |
| `Dockerfile.torch` | PyTorch 后端镜像，安装 `hyper_parallel[torch29]` |
| `Dockerfile.mindspore` | MindSpore 后端镜像，安装 `hyper_parallel[mindspore]` |
| `build_hyper-parallel_npu.sh` | 简化构建脚本，构建后自动做 import smoke test |
| `run_hyper-parallel.sh` | Ascend NPU 容器启动脚本 |

基础镜像默认使用：

```bash
quay.io/ascend/cann:9.1.0-a3-ubuntu22.04-py3.12
```

## 构建镜像

默认构建通用 NPU 镜像。该命令需要在仓库根目录执行：

```bash
bash docker/build_hyper-parallel_npu.sh hyper-parallel:npu
```

构建 PyTorch 环境：

```bash
DOCKERFILE=docker/Dockerfile.torch \
HP_EXTRA=torch29 \
bash docker/build_hyper-parallel_npu.sh hyper-parallel:torch
```

构建 MindSpore 环境：

```bash
DOCKERFILE=docker/Dockerfile.mindspore \
HP_EXTRA=mindspore \
bash docker/build_hyper-parallel_npu.sh hyper-parallel:mindspore
```

`HP_EXTRA` 对应 `docs/installation.md` 中的 extras：

```text
torch26 | torch27 | torch29 | torch | mindspore | all
```

## 启动容器

启动默认镜像并进入 shell。脚本默认挂载当前仓库目录，并删除同名旧容器；`--cards auto` 会自动
探测主机上的 NPU 卡，适配 A2/A3 及非 8 卡机器：

```bash
bash docker/run_hyper-parallel.sh
```

指定 NPU 卡：

```bash
bash docker/run_hyper-parallel.sh --cards 0,1
```

也可以通过环境变量指定卡列表或硬件对应的默认列表：

```bash
CARDS=auto DEFAULT_CARDS=0,1,2,3 bash docker/run_hyper-parallel.sh
```

执行验证命令：

```bash
bash docker/run_hyper-parallel.sh --cards 0,1,2,3,4,5,6,7 -- \
  python3 -c "import hyper_parallel as hp; print(hp.get_platform())"
```

## Native 扩展构建

Dockerfile 支持以下构建参数：

```text
BUILD_MULTICORE_EXTENSION=off|mindspore|torch|all
BUILD_SHMEM_EXTENSION=off|mindspore|torch|all
BUILD_CUSTOM_OPS_EXTENSION=off|on
HYPER_PARALLEL_BUILD_STRICT=off|on
```

示例：

```bash
docker build -f docker/Dockerfile.torch \
  -t hyper-parallel:torch-native \
  --build-arg TORCH_EXTRA=torch29 \
  --build-arg BUILD_MULTICORE_EXTENSION=torch \
  --build-arg BUILD_SHMEM_EXTENSION=torch \
  --build-arg HYPER_PARALLEL_BUILD_STRICT=off \
  .
```

## 验证

镜像内验证：

```bash
source /usr/local/Ascend/cann/set_env.sh
python3 -c "import hyper_parallel as hp; print(hp.get_platform())"
```

查看包版本：

```bash
python3 - <<'PY'
import importlib.metadata as md
for name in ("hyper_parallel", "torch", "torch-npu", "mindspore"):
    try:
        print(name, md.version(name))
    except md.PackageNotFoundError:
        pass
PY
```

## 基础版本参数

构建脚本支持通过环境变量覆盖基础镜像版本和硬件架构：

```bash
CANN_VERSION=9.1.0 CANN_ARCH=a2 PYTHON_VERSION=3.12 \
bash docker/build_hyper-parallel_npu.sh hyper-parallel:a2
```

其中 `CANN_ARCH` 用于选择基础镜像标签（例如 `a2` 或 `a3`，具体以镜像仓库实际提供的标签为准）。
若基础镜像使用 `/usr/local/Ascend/ascend-toolkit` 或带版本号的 CANN 目录，Dockerfile 会自动
创建统一的 `/usr/local/Ascend/cann` 路径。
