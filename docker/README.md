# HyperParallel Docker Environment

**English** | [简体中文](README.zh.md)

This directory provides an Ascend NPU Docker environment for installing and validating HyperParallel.

## Quick Deployment

### 1. Pull a Prebuilt Image

Pull the PyTorch 2.9 NPU image directly from Quay.io:

```bash
docker pull quay.io/hyper-parallel/hyper-parallel-npu:cann9.1.0-torch29
```

Start a container from the downloaded image:

```bash
IMAGE=quay.io/hyper-parallel/hyper-parallel-npu:cann9.1.0-torch29 \
bash docker/run_hyper-parallel.sh
```

### 2. Build an Image from Source

Use the build script to include the current source or custom dependencies:

```bash
bash docker/build_hyper-parallel_npu.sh hyper-parallel:npu
```

Start the local image after building:

```bash
IMAGE=hyper-parallel:npu bash docker/run_hyper-parallel.sh --name hyper-parallel-npu --cards auto
```

### 3. Enter the Container and Verify

```bash
docker exec -it hyper-parallel-npu bash
source /usr/local/Ascend/cann/set_env.sh
python3 -c "import importlib.metadata as md; print(md.version('hyper_parallel'))"
```

## Files

| File | Description |
| --- | --- |
| `Dockerfile.hyper-parallel-npu` | General HyperParallel NPU image, defaulting to the PyTorch 2.9 backend |
| `Dockerfile.models-npu-a3` | Ascend A3 / Ubuntu 22.04 / Python 3.11 training image with dependencies for high-performance fused operators |
| `Dockerfile.torch` | PyTorch backend image installing `hyper_parallel[torch29]` |
| `build_hyper-parallel_npu.sh` | Simplified build script that runs an import smoke test after building |
| `run_hyper-parallel.sh` | Ascend NPU container launch script |

The default base image is:

```bash
quay.io/ascend/cann:9.1.0-a3-ubuntu22.04-py3.12
```

## Build Images

The default build creates the general NPU image. Run this command from the repository root:

```bash
bash docker/build_hyper-parallel_npu.sh hyper-parallel:npu
```

Build the PyTorch environment:

```bash
DOCKERFILE=docker/Dockerfile.torch \
HP_EXTRA=torch29 \
bash docker/build_hyper-parallel_npu.sh hyper-parallel:torch
```

`HP_EXTRA` corresponds to the extras in `docs/installation.md`:

```text
torch26 | torch27 | torch29 | torch | all
```

## Start Containers

Start the default image and enter its shell. By default, the script mounts the current repository and removes an existing container with the same name. `--cards auto`
automatically detects the host's NPU devices, supporting A2/A3 and hosts with device counts other than eight:

```bash
bash docker/run_hyper-parallel.sh
```

Select NPU devices explicitly:

```bash
bash docker/run_hyper-parallel.sh --cards 0,1
```

Alternatively, set the device list or hardware-specific defaults through environment variables:

```bash
CARDS=auto DEFAULT_CARDS=0,1,2,3 bash docker/run_hyper-parallel.sh
```

Run a verification command:

```bash
bash docker/run_hyper-parallel.sh --cards 0,1,2,3,4,5,6,7 -- \
  python3 -c "import importlib.metadata as md; print(md.version('hyper_parallel'))"
```

## Native Extension Builds

The Dockerfile supports these build arguments:

```text
BUILD_MULTICORE_EXTENSION=off|torch|all
BUILD_SHMEM_EXTENSION=off|torch|all
HYPER_PARALLEL_BUILD_STRICT=off|on
```

Example:

```bash
docker build -f docker/Dockerfile.torch \
  -t hyper-parallel:torch-native \
  --build-arg TORCH_EXTRA=torch29 \
  --build-arg BUILD_MULTICORE_EXTENSION=torch \
  --build-arg BUILD_SHMEM_EXTENSION=torch \
  --build-arg HYPER_PARALLEL_BUILD_STRICT=off \
  .
```

## Verification

Verify inside the image:

```bash
source /usr/local/Ascend/cann/set_env.sh
python3 -c "import importlib.metadata as md; print(md.version('hyper_parallel'))"
```

Check the package version:

```bash
python3 - <<'PY'
import importlib.metadata as md
for name in ("hyper_parallel", "torch", "torch-npu"):
    try:
        print(name, md.version(name))
    except md.PackageNotFoundError:
        pass
PY
```

## Base Version Parameters

The build script allows environment variables to override the base image version and hardware architecture:

```bash
CANN_VERSION=9.1.0 CANN_ARCH=a2 PYTHON_VERSION=3.12 \
bash docker/build_hyper-parallel_npu.sh hyper-parallel:a2
```

`CANN_ARCH` selects the base image tag, such as `a2` or `a3`; available values depend on the tags actually published in the image registry.
If the base image uses `/usr/local/Ascend/ascend-toolkit` or a versioned CANN directory, the Dockerfile automatically
creates the unified `/usr/local/Ascend/cann` path.
