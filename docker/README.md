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
IMAGE=hyper-parallel:npu bash docker/run_hyper-parallel.sh
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

启动默认镜像并进入 shell。脚本会自动挂载当前仓库的父目录，并删除同名旧容器：

```bash
bash docker/run_hyper-parallel.sh
```

指定 NPU 卡：

```bash
bash docker/run_hyper-parallel.sh --cards 0,1
```

执行验证命令：

```bash
bash docker/run_hyper-parallel.sh --cards 0 -- \
  python3 -c "import hyper_parallel as hp; print(hp.get_platform())"
```

切换仓库路径或工作目录：

```bash
bash docker/run_hyper-parallel.sh \
  --workdir /workspace/hyper-parallel
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