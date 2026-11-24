# HyperParallel 昇腾镜像

在昇腾上使用 HyperParallel 运行分布式 PyTorch 训练。每个镜像都为目标 Python 和 CPU 架构编译了 Indexed Dataset 原生扩展。镜像使用 AscendHub 的 CANN 或 TorchNPU（PTA）基座，按芯片、Python 和 PyTorch 版本选择。

## 快速参考

| 项目 | 入口 |
| --- | --- |
| 构建配方与脚本 | [Dockerfile](./Dockerfile) · [build.sh](./build.sh) |
| 已发布镜像 | [Quay 上的 HyperParallel 标签](https://quay.io/repository/hyper-parallel/hyper-parallel-npu?tab=tags) |
| 上游镜像 | [AscendHub CANN](https://www.hiascend.com/developer/ascendhub/detail/17da20d1c2b6493cb38765adeba85884) · [AscendHub TorchNPU](https://www.hiascend.com/developer/ascendhub/detail/1bb1e5d9c8c64f6ea8db0d0ec5061531) · [AscendHub MindSpeed-LLM](https://www.hiascend.com/developer/ascendhub/detail/e26da9266559438b93354792f25b2f4a) |

## 支持的构建组合

<!-- BEGIN generated support matrix -->

| CANN | PyTorch | torch_npu (PTA) | Python | 操作系统 | 芯片 |
| --- | --- | --- | --- | --- | --- |
| 9.2.0 | 2.9.0 | 2.9.0.post9 | 3.10, 3.11, 3.12 | ubuntu22.04, openeuler24.03 | 910b, a3, 950 |
| 9.2.0 | 2.7.1 | 2.7.1.post11 | 3.10, 3.11, 3.12 | ubuntu22.04, openeuler24.03 | 910b, a3, 950 |
| 9.2.0 | 2.10.0 | 2.10.0.post7 | 3.10, 3.11, 3.12 | ubuntu22.04, openeuler24.03 | 910b, a3, 950 |
| 9.2.0 | 2.11.0 | 2.11.0.post3 | 3.10, 3.11, 3.12 | ubuntu22.04, openeuler24.03 | 910b, a3, 950 |
| 9.2.0 | 2.12.0 | 2.12.0.post3 | 3.10, 3.11, 3.12 | ubuntu22.04, openeuler24.03 | 910b, a3, 950 |
| 9.2.0 | 2.13.0 | 2.13.0 | 3.10, 3.11, 3.12 | ubuntu22.04, openeuler24.03 | 910b, a3, 950 |
| 9.2.0 | 2.14.0 | 2.14.0 | 3.10, 3.11, 3.12 | ubuntu22.04, openeuler24.03 | 910b, a3, 950 |
| 9.0.0 | 2.9.0 | 2.9.0.post2 | 3.11 | ubuntu22.04, openeuler24.03 | 910b, a3 |

脚本默认使用 Python 3.12 和 PTA 2.9.0.post9。 [HyperParallel 完整 Python 依赖](https://github.com/mindspore-ai/hyper-parallel/blob/c84594c4578c4ad7b048db40fa0c4998c776c0e4/requirements.txt)

<!-- END generated support matrix -->

作为参考，MindSpeed-LLM 26.2.0 已发布六个 CANN 9.2.0 标签。均使用 Python 3.12、PyTorch 2.10.0 和 PTA 2.10.0.post7，覆盖 `910b`、`a3`、`950` 与两种操作系统，每个标签同时含 Linux arm64 和 amd64。HyperParallel 按 CPU 架构分别构建。950 对应 950PR/DT 系列，910B 对应 Atlas A2。可拉取的成品以 Quay 为准。[MindSpeed-LLM 官方发布矩阵](https://gitcode.com/Ascend/MindSpeed-LLM/blob/224267cad7dc68045e13753017a75fa80ea7e8e0/docker/OVERVIEW.zh.md)

HyperParallel 固定源码声明 `Requires-Python: >=3.10,<3.13`。镜像使用基座自带的解释器，并从原始源码编译 wheel。[上游版本约束](https://github.com/mindspore-ai/hyper-parallel/blob/c84594c4578c4ad7b048db40fa0c4998c776c0e4/setup.py)

| CANN 9.2 的 Python | 构建方式 |
| --- | --- |
| 3.10、3.11 | 对应 Python 和芯片的官方 CANN 基座，安装与 ABI 匹配的 PyTorch CPU wheel 和 TorchNPU wheel。 |
| 3.12 | 官方 PTA 基座已包含配套的 PyTorch 和 TorchNPU。 |

[Ascend PyTorch 兼容表](https://github.com/Ascend/pytorch/blob/master/COMPATIBILITY.md) · [PyTorch 官方 CPU wheel](https://download.pytorch.org/whl/cpu/torch/) · [TorchNPU 安装包](https://pypi.org/project/torch-npu/)

### 镜像标签

`v{HyperParallel版本}-cann{CANN版本}-torch_npu{PTA版本}-{芯片}-{操作系统}-py{Python版本}-{CPU架构}-g{源码提交}`

各字段依次表示 HyperParallel 源码版本、CANN 工具包版本、TorchNPU 包版本、芯片、操作系统发行版、Python 主次版本、Linux CPU 架构，以及编译 HyperParallel **wheel 的源码提交号**前 12 位。全部标签共用一份 [Dockerfile](./Dockerfile)。

## 快速开始

### 在 Linux CPU 主机上构建

主机构建需要 Bash 4+、GNU coreutils 和 Docker。文档生成、校验与双系统推送还需 awk 和 GNU diffutils（`cmp`、`diff`）。通常按目标 CPU 原生构建，跨架构时需先为构建器配置模拟。以下命令在 `docker/` 的上一级目录执行，构建时无需 NPU 或宿主机 Python。

<!-- BEGIN generated build example -->

```bash
bash docker/build.sh -t 910b -o ubuntu22.04 -a arm64 --cann-version 9.2.0 --python-version 3.12 --pta-version 2.9.0.post9
IMAGE=quay.io/hyper-parallel/hyper-parallel-npu:v1.2.0.dev0-cann9.2.0-torch_npu2.9.0.post9-910b-ubuntu22.04-py3.12-arm64-gc84594c4578c
```

<!-- END generated build example -->

| 脚本参数 | 取值与默认行为 |
| --- | --- |
| `-t`、`--npu-type`（也可用 `-c`、`--chip`） | 构建必填。A2 用 `a2` 或 `910b`，A3 用 `a3`，A5 用 `a5` 或 `950`。 |
| `-o`、`--os` | `ubuntu22.04` 或 `openeuler24.03`。省略时构建双系统。 |
| `-a`、`--arch` | 默认使用宿主机架构。支持 `arm64`/`aarch64`、`amd64`/`x86_64`。 |
| `--python-version` | 3.10、3.11 或 3.12。默认采用所选 PTA 基座的 Python。 |
| `--hyper-parallel-branch NAME` 或 `--hyper-parallel-commit SHA` | 选择官方 HyperParallel 分支或完整的 40 位提交号。默认使用配方固定的提交。分支在一次构建中只解析一次，双系统使用同一份源码。 |
| `--hp-version V` | 指定预期的 wheel 版本。自定义源码时，脚本默认从该提交的 `setup.py` 读取版本。 |
| `--cann-version` | 默认 `9.2.0`。`9.0.0` 使用 Python 3.11、PyTorch 2.9.0、PTA 2.9.0.post2，仅覆盖 A2/A3。 |
| `--torch-version` | 指定所选 CANN 行中的 PyTorch 版本，脚本自动匹配 PTA。 |
| `--pta-version`、`--torch-npu-version` | 同一参数的两种写法，均使用完整 PTA 包版本。同时指定 PyTorch 时两者须匹配。 |
| `--tags`、`--print-command` | 分别打印完整镜像名、可直接执行的 `docker build` 命令。命令包含基座摘要和解析后的完整源码提交。 |
| `--no-cache`、`--network MODE` | 禁用 Docker 构建缓存，或设置构建期间的网络模式（例如 `host`）。 |
| `--push` | 双系统都构建成功后推送到 Quay。需要写权限，请省略 `--os`。 |
| `--docs`、`--check`、`--export DIR` | 分别用于更新仓内两份概述、检查是否与配方一致、向新目录或空目录导出配方及每个标签独立的中英文概述。 |
| `-h`、`--help` | 显示帮助。 |

下面用 `master` 选源码，构建与上表配套的 arm64 镜像。保存打印的命令，可按同一提交复现构建。也可以将 `--hyper-parallel-branch master` 换成 `--hyper-parallel-commit` 加完整 SHA。

```bash
bash docker/build.sh --print-command --hyper-parallel-branch master -t 910b -o ubuntu22.04 -a arm64 --torch-version 2.10.0 > build-hp.sh
bash build-hp.sh
```

直接使用 `docker build` 时，`BASE_IMAGE` 选择具体芯片、系统和 Python ABI 的官方基座，`--platform` 选择 CPU 架构。`CANN_VERSION`、`PYTHON_VERSION` 和框架参数校验实际软件栈。Docker 18.09 原生构建可省略 `--platform`，脚本会自动处理。源码参数使用 `HYPER_PARALLEL_COMMIT`（完整提交号）或 `HYPER_PARALLEL_BRANCH`（例如 `master`），`HP_VERSION` 须与 wheel 版本一致。直接传入移动分支时，用 `--no-cache` 获取新提交，并用 `-t` 为自定义镜像命名。每个成品的 `/etc/hyper-parallel-revision` 记录实际 40 位提交号。

<!-- BEGIN generated docker build -->

```bash
docker build --platform linux/arm64 \
  --build-arg BASE_IMAGE=swr.cn-south-1.myhuaweicloud.com/ascendhub/torch-npu:2.9.0.post9-cann9.2.0-910b-ubuntu22.04-py3.12@sha256:90773b3183c1264211506be5624b6c17475bc478a182190bca885e3e21010464 \
  --build-arg CANN_VERSION=9.2.0 --build-arg TORCH_VERSION=2.9.0 \
  --build-arg PYTHON_VERSION=3.12 --build-arg TORCH_NPU_VERSION=2.9.0.post9 \
  --build-arg HYPER_PARALLEL_COMMIT=c84594c4578c4ad7b048db40fa0c4998c776c0e4 \
  --build-arg HP_VERSION=1.2.0.dev0 \
  -f docker/Dockerfile -t quay.io/hyper-parallel/hyper-parallel-npu:v1.2.0.dev0-cann9.2.0-torch_npu2.9.0.post9-910b-ubuntu22.04-py3.12-arm64-gc84594c4578c docker/
```

<!-- END generated docker build -->

### 软件依赖与网络访问

| 阶段 | 软件或资源 | 需要访问的站点 |
| --- | --- | --- |
| 拉取基座（Docker 守护进程） | 按摘要固定的公开 AscendHub CANN 或 PTA 镜像 | HTTPS 443，`swr.cn-south-1.myhuaweicloud.com` 及镜像层重定向的 `*.obs.cn-south-1.myhuaweicloud.com` |
| 获取源码、编译 wheel | 主机解析自定义源码时使用 Git。构建阶段使用 Git、C++、Make、setuptools、wheel 和 pybind11。 | HTTPS 443，`github.com`、`gitcode.com`。Git 优先访问 GitHub，失败时使用官方 GitCode 镜像。 包含可选 Multicore 组件的源码还会从 GitCode 获取 `cann/ops-nn`。 |
| 安装框架（使用 CANN 基座时） | 指定版本和 ABI 的 CPU 版 PyTorch 与昇腾 TorchNPU | HTTPS 443，`download.pytorch.org` 及其下载节点 `download-r2.pytorch.org`，以及下行的 PyPI 站点。 |
| 安装 Python 包 | 公开 PyPI 构建工具及 NumPy、torchdata、PyYAML、datasets、transformers 等运行依赖。安装时锁定所选 torch 和 torch_npu，执行 `pip check`。 | HTTPS 443，`pypi.org`、`files.pythonhosted.org`。 |
| 补装系统工具（按需） | 发行版签名的 Ubuntu 或 openEuler 软件包 | Ubuntu 通常通过 HTTP 80 访问 `ports.ubuntu.com`（arm64）或 `archive.ubuntu.com`、`security.ubuntu.com`（amd64）。openEuler 通过 HTTPS 443 访问 `repo.openeuler.org`、`repo.openeuler.openatom.cn`、`mirrors.openeuler.org` 及其 metalink 返回的软件源。 |
| 拉取或推送已发布镜像 | Quay 及其镜像分发服务。推送需要写凭据。 | HTTPS 443，`quay.io`、`*.quay.io`。本地构建、运行均无需访问 Quay。 |
| 在 NPU 主机运行 | 配套的昇腾硬件和宿主机驱动。Ascend Docker Runtime 可自动挂载设备，标准 runc 可手动映射。 | 本地镜像可断网执行单卡示例。多机训练需配置参与主机间的通信。 |

构建机与设备机分开时，可用 `docker save -o hp.tar "$IMAGE"` 导出，传输后用 `docker load -i hp.tar` 导入。拉取镜像需要代理时，请为 Docker 守护进程配置代理。构建容器内的下载可使用 Docker 客户端代理配置，或在原生 `docker build` 命令中传入 `--build-arg HTTPS_PROXY`、`--build-arg HTTP_PROXY`。

### 交互式启动与进入容器

入口脚本在容器启动时显示 HyperParallel 版本和镜像许可。 `-it` 保持标准输入打开并分配终端。在宿主机用 `docker image ls "$IMAGE"` 确认上一步得到的 `REPOSITORY:TAG`。仅查看镜像里的软件时，可在普通 Linux CPU 主机打开 Bash。进入后运行 `python3 -I -m pip show hyper-parallel torch torch-npu` 查看安装版本，用 `exit` 退出。

```bash
docker run -it --rm --runtime=runc "$IMAGE" /bin/bash
```

在有配套驱动和 Ascend Docker Runtime 的设备机上，先确认目标卡可分配给新容器。可用 `npu-smi info` 查看负载，使用独占设备时还需检查现有容器的设备分配。以下以物理卡 1 为例，从业务项目目录启动命名容器，并把当前源码挂到 `/workspace`。

```bash
docker run -it --rm --name hp-dev --runtime=ascend -e ASCEND_VISIBLE_DEVICES=1 -v "$PWD:/workspace" -w /workspace "$IMAGE" bash
```

保持首个终端运行。在宿主机的第二个终端进入同一容器，退出首个 Bash 后 Docker 会自动删除容器。

```bash
docker exec -it hp-dev bash
```

在容器内查看设备并验证计算。单卡示例中 `npu:0` 是容器内的逻辑卡号，Python 输出 `1 6`。

```bash
npu-smi info -l
python3 -c 'import torch, hyper_parallel; print(torch.npu.device_count(), torch.arange(4, device="npu:0").sum().item())'
```

#### 使用标准 runc 手动挂载 NPU

也可从宿主机用 `runc` 启动。以下是 A2/910B、物理卡 `/dev/davinci1`、驱动安装于 `/usr/local/Ascend/driver` 的示例。按宿主机的设备节点和安装路径调整，进入后可执行上面的设备检查。

```bash
docker run -it --rm --runtime=runc \
  --device=/dev/davinci1 --device=/dev/davinci_manager \
  --device=/dev/devmm_svm --device=/dev/hisi_hdc \
  -v /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64:ro \
  -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info:ro \
  -v /usr/local/dcmi:/usr/local/dcmi:ro \
  -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi:ro \
  -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
  -v "$PWD:/workspace" -w /workspace "$IMAGE" /bin/bash
```

| 按需参数 | 用途 |
| --- | --- |
| `-v /data/my-dataset:/data:ro` | 将示例源路径换成自己的数据目录。代码通过 `$PWD:/workspace` 读写。 |
| `--network host`、`--ipc=host` | 分别共享宿主机网络和 IPC，按分布式作业的网络与共享内存要求选择。 |
| `--privileged` | 在允许特权容器的主机上授予全部宿主设备访问权，可配合 `-e ASCEND_RT_VISIBLE_DEVICES=1` 指定 PyTorch 使用物理卡 1。`npu-smi` 仍可列出全部卡。 |

要从宿主机执行断网检查，以下示例使用物理卡 0，输出 `6`。

```bash
docker run --rm --network=none --runtime=ascend -e ASCEND_VISIBLE_DEVICES=0 "$IMAGE" \
  python3 -c 'import torch, hyper_parallel; print(torch.arange(4, device="npu").sum().item())'
```

### 二次开发

在含有 `train.py` 的业务项目中，将模型所需的额外 Python 包写入 `requirements.txt`（没有额外依赖时可用空文件），按所选镜像的框架版本固定依赖。将下面的内容保存为 `Dockerfile.hp`。`python3 -I -m pip` 在应用解释器的包环境中安装和校验依赖。

```dockerfile
ARG HP_IMAGE
FROM ${HP_IMAGE}
WORKDIR /workspace
COPY requirements.txt .
RUN python3 -I -m pip install --no-cache-dir --index-url https://pypi.org/simple -r requirements.txt && python3 -I -m pip check
COPY . .
CMD ["python3", "train.py"]
```

在宿主机项目目录构建。第一条 `docker run` 执行训练脚本，第二条打开子镜像的交互式 Bash。手动映射设备时，可在上面的 `runc` 命令中用 `hp-training:local` 替换 `"$IMAGE"`。

```bash
docker build -f Dockerfile.hp --build-arg HP_IMAGE="$IMAGE" -t hp-training:local .
docker run --rm --runtime=ascend -e ASCEND_VISIBLE_DEVICES=1 hp-training:local
docker run -it --rm --runtime=ascend -e ASCEND_VISIBLE_DEVICES=1 hp-training:local /bin/bash
```

修改 HyperParallel 自身时，在原生构建命令中将 `HYPER_PARALLEL_COMMIT` 和 `HP_VERSION` 换为官方仓库可获取的源码提交和对应包版本，即可为目标 Python ABI 重新编译 wheel。

## 完整构建标签

<!-- BEGIN generated image tags -->

| 芯片 | CPU | 完整版本标签 |
| --- | --- | --- |
| 910b | arm64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.9.0.post9-910b-ubuntu22.04-py3.12-arm64-gc84594c4578c` |
| 910b | amd64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.9.0.post9-910b-ubuntu22.04-py3.12-amd64-gc84594c4578c` |
| 910b | arm64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.9.0.post9-910b-openeuler24.03-py3.12-arm64-gc84594c4578c` |
| 910b | amd64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.9.0.post9-910b-openeuler24.03-py3.12-amd64-gc84594c4578c` |
| a3 | arm64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.9.0.post9-a3-ubuntu22.04-py3.12-arm64-gc84594c4578c` |
| a3 | amd64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.9.0.post9-a3-ubuntu22.04-py3.12-amd64-gc84594c4578c` |
| a3 | arm64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.9.0.post9-a3-openeuler24.03-py3.12-arm64-gc84594c4578c` |
| a3 | amd64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.9.0.post9-a3-openeuler24.03-py3.12-amd64-gc84594c4578c` |
| 950 | arm64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.9.0.post9-950-ubuntu22.04-py3.12-arm64-gc84594c4578c` |
| 950 | amd64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.9.0.post9-950-ubuntu22.04-py3.12-amd64-gc84594c4578c` |
| 950 | arm64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.9.0.post9-950-openeuler24.03-py3.12-arm64-gc84594c4578c` |
| 950 | amd64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.9.0.post9-950-openeuler24.03-py3.12-amd64-gc84594c4578c` |
| 910b | arm64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.7.1.post11-910b-ubuntu22.04-py3.12-arm64-gc84594c4578c` |
| 910b | amd64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.7.1.post11-910b-ubuntu22.04-py3.12-amd64-gc84594c4578c` |
| 910b | arm64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.7.1.post11-910b-openeuler24.03-py3.12-arm64-gc84594c4578c` |
| 910b | amd64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.7.1.post11-910b-openeuler24.03-py3.12-amd64-gc84594c4578c` |
| a3 | arm64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.7.1.post11-a3-ubuntu22.04-py3.12-arm64-gc84594c4578c` |
| a3 | amd64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.7.1.post11-a3-ubuntu22.04-py3.12-amd64-gc84594c4578c` |
| a3 | arm64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.7.1.post11-a3-openeuler24.03-py3.12-arm64-gc84594c4578c` |
| a3 | amd64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.7.1.post11-a3-openeuler24.03-py3.12-amd64-gc84594c4578c` |
| 950 | arm64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.7.1.post11-950-ubuntu22.04-py3.12-arm64-gc84594c4578c` |
| 950 | amd64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.7.1.post11-950-ubuntu22.04-py3.12-amd64-gc84594c4578c` |
| 950 | arm64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.7.1.post11-950-openeuler24.03-py3.12-arm64-gc84594c4578c` |
| 950 | amd64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.7.1.post11-950-openeuler24.03-py3.12-amd64-gc84594c4578c` |
| 910b | arm64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.10.0.post7-910b-ubuntu22.04-py3.12-arm64-gc84594c4578c` |
| 910b | amd64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.10.0.post7-910b-ubuntu22.04-py3.12-amd64-gc84594c4578c` |
| 910b | arm64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.10.0.post7-910b-openeuler24.03-py3.12-arm64-gc84594c4578c` |
| 910b | amd64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.10.0.post7-910b-openeuler24.03-py3.12-amd64-gc84594c4578c` |
| a3 | arm64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.10.0.post7-a3-ubuntu22.04-py3.12-arm64-gc84594c4578c` |
| a3 | amd64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.10.0.post7-a3-ubuntu22.04-py3.12-amd64-gc84594c4578c` |
| a3 | arm64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.10.0.post7-a3-openeuler24.03-py3.12-arm64-gc84594c4578c` |
| a3 | amd64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.10.0.post7-a3-openeuler24.03-py3.12-amd64-gc84594c4578c` |
| 950 | arm64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.10.0.post7-950-ubuntu22.04-py3.12-arm64-gc84594c4578c` |
| 950 | amd64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.10.0.post7-950-ubuntu22.04-py3.12-amd64-gc84594c4578c` |
| 950 | arm64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.10.0.post7-950-openeuler24.03-py3.12-arm64-gc84594c4578c` |
| 950 | amd64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.10.0.post7-950-openeuler24.03-py3.12-amd64-gc84594c4578c` |
| 910b | arm64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.11.0.post3-910b-ubuntu22.04-py3.12-arm64-gc84594c4578c` |
| 910b | amd64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.11.0.post3-910b-ubuntu22.04-py3.12-amd64-gc84594c4578c` |
| 910b | arm64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.11.0.post3-910b-openeuler24.03-py3.12-arm64-gc84594c4578c` |
| 910b | amd64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.11.0.post3-910b-openeuler24.03-py3.12-amd64-gc84594c4578c` |
| a3 | arm64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.11.0.post3-a3-ubuntu22.04-py3.12-arm64-gc84594c4578c` |
| a3 | amd64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.11.0.post3-a3-ubuntu22.04-py3.12-amd64-gc84594c4578c` |
| a3 | arm64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.11.0.post3-a3-openeuler24.03-py3.12-arm64-gc84594c4578c` |
| a3 | amd64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.11.0.post3-a3-openeuler24.03-py3.12-amd64-gc84594c4578c` |
| 950 | arm64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.11.0.post3-950-ubuntu22.04-py3.12-arm64-gc84594c4578c` |
| 950 | amd64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.11.0.post3-950-ubuntu22.04-py3.12-amd64-gc84594c4578c` |
| 950 | arm64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.11.0.post3-950-openeuler24.03-py3.12-arm64-gc84594c4578c` |
| 950 | amd64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.11.0.post3-950-openeuler24.03-py3.12-amd64-gc84594c4578c` |
| 910b | arm64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.12.0.post3-910b-ubuntu22.04-py3.12-arm64-gc84594c4578c` |
| 910b | amd64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.12.0.post3-910b-ubuntu22.04-py3.12-amd64-gc84594c4578c` |
| 910b | arm64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.12.0.post3-910b-openeuler24.03-py3.12-arm64-gc84594c4578c` |
| 910b | amd64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.12.0.post3-910b-openeuler24.03-py3.12-amd64-gc84594c4578c` |
| a3 | arm64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.12.0.post3-a3-ubuntu22.04-py3.12-arm64-gc84594c4578c` |
| a3 | amd64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.12.0.post3-a3-ubuntu22.04-py3.12-amd64-gc84594c4578c` |
| a3 | arm64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.12.0.post3-a3-openeuler24.03-py3.12-arm64-gc84594c4578c` |
| a3 | amd64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.12.0.post3-a3-openeuler24.03-py3.12-amd64-gc84594c4578c` |
| 950 | arm64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.12.0.post3-950-ubuntu22.04-py3.12-arm64-gc84594c4578c` |
| 950 | amd64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.12.0.post3-950-ubuntu22.04-py3.12-amd64-gc84594c4578c` |
| 950 | arm64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.12.0.post3-950-openeuler24.03-py3.12-arm64-gc84594c4578c` |
| 950 | amd64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.12.0.post3-950-openeuler24.03-py3.12-amd64-gc84594c4578c` |
| 910b | arm64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.13.0-910b-ubuntu22.04-py3.12-arm64-gc84594c4578c` |
| 910b | amd64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.13.0-910b-ubuntu22.04-py3.12-amd64-gc84594c4578c` |
| 910b | arm64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.13.0-910b-openeuler24.03-py3.12-arm64-gc84594c4578c` |
| 910b | amd64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.13.0-910b-openeuler24.03-py3.12-amd64-gc84594c4578c` |
| a3 | arm64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.13.0-a3-ubuntu22.04-py3.12-arm64-gc84594c4578c` |
| a3 | amd64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.13.0-a3-ubuntu22.04-py3.12-amd64-gc84594c4578c` |
| a3 | arm64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.13.0-a3-openeuler24.03-py3.12-arm64-gc84594c4578c` |
| a3 | amd64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.13.0-a3-openeuler24.03-py3.12-amd64-gc84594c4578c` |
| 950 | arm64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.13.0-950-ubuntu22.04-py3.12-arm64-gc84594c4578c` |
| 950 | amd64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.13.0-950-ubuntu22.04-py3.12-amd64-gc84594c4578c` |
| 950 | arm64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.13.0-950-openeuler24.03-py3.12-arm64-gc84594c4578c` |
| 950 | amd64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.13.0-950-openeuler24.03-py3.12-amd64-gc84594c4578c` |
| 910b | arm64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.14.0-910b-ubuntu22.04-py3.12-arm64-gc84594c4578c` |
| 910b | amd64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.14.0-910b-ubuntu22.04-py3.12-amd64-gc84594c4578c` |
| 910b | arm64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.14.0-910b-openeuler24.03-py3.12-arm64-gc84594c4578c` |
| 910b | amd64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.14.0-910b-openeuler24.03-py3.12-amd64-gc84594c4578c` |
| a3 | arm64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.14.0-a3-ubuntu22.04-py3.12-arm64-gc84594c4578c` |
| a3 | amd64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.14.0-a3-ubuntu22.04-py3.12-amd64-gc84594c4578c` |
| a3 | arm64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.14.0-a3-openeuler24.03-py3.12-arm64-gc84594c4578c` |
| a3 | amd64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.14.0-a3-openeuler24.03-py3.12-amd64-gc84594c4578c` |
| 950 | arm64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.14.0-950-ubuntu22.04-py3.12-arm64-gc84594c4578c` |
| 950 | amd64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.14.0-950-ubuntu22.04-py3.12-amd64-gc84594c4578c` |
| 950 | arm64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.14.0-950-openeuler24.03-py3.12-arm64-gc84594c4578c` |
| 950 | amd64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.14.0-950-openeuler24.03-py3.12-amd64-gc84594c4578c` |
| 910b | arm64 | `v1.2.0.dev0-cann9.0.0-torch_npu2.9.0.post2-910b-ubuntu22.04-py3.11-arm64-gc84594c4578c` |
| 910b | amd64 | `v1.2.0.dev0-cann9.0.0-torch_npu2.9.0.post2-910b-ubuntu22.04-py3.11-amd64-gc84594c4578c` |
| 910b | arm64 | `v1.2.0.dev0-cann9.0.0-torch_npu2.9.0.post2-910b-openeuler24.03-py3.11-arm64-gc84594c4578c` |
| 910b | amd64 | `v1.2.0.dev0-cann9.0.0-torch_npu2.9.0.post2-910b-openeuler24.03-py3.11-amd64-gc84594c4578c` |
| a3 | arm64 | `v1.2.0.dev0-cann9.0.0-torch_npu2.9.0.post2-a3-ubuntu22.04-py3.11-arm64-gc84594c4578c` |
| a3 | amd64 | `v1.2.0.dev0-cann9.0.0-torch_npu2.9.0.post2-a3-ubuntu22.04-py3.11-amd64-gc84594c4578c` |
| a3 | arm64 | `v1.2.0.dev0-cann9.0.0-torch_npu2.9.0.post2-a3-openeuler24.03-py3.11-arm64-gc84594c4578c` |
| a3 | amd64 | `v1.2.0.dev0-cann9.0.0-torch_npu2.9.0.post2-a3-openeuler24.03-py3.11-amd64-gc84594c4578c` |
| 910b | arm64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.9.0.post9-910b-ubuntu22.04-py3.10-arm64-gc84594c4578c` |
| 910b | amd64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.9.0.post9-910b-ubuntu22.04-py3.10-amd64-gc84594c4578c` |
| 910b | arm64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.7.1.post11-910b-ubuntu22.04-py3.10-arm64-gc84594c4578c` |
| 910b | amd64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.7.1.post11-910b-ubuntu22.04-py3.10-amd64-gc84594c4578c` |
| 910b | arm64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.10.0.post7-910b-ubuntu22.04-py3.10-arm64-gc84594c4578c` |
| 910b | amd64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.10.0.post7-910b-ubuntu22.04-py3.10-amd64-gc84594c4578c` |
| 910b | arm64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.11.0.post3-910b-ubuntu22.04-py3.10-arm64-gc84594c4578c` |
| 910b | amd64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.11.0.post3-910b-ubuntu22.04-py3.10-amd64-gc84594c4578c` |
| 910b | arm64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.12.0.post3-910b-ubuntu22.04-py3.10-arm64-gc84594c4578c` |
| 910b | amd64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.12.0.post3-910b-ubuntu22.04-py3.10-amd64-gc84594c4578c` |
| 910b | arm64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.13.0-910b-ubuntu22.04-py3.10-arm64-gc84594c4578c` |
| 910b | amd64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.13.0-910b-ubuntu22.04-py3.10-amd64-gc84594c4578c` |
| 910b | arm64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.14.0-910b-ubuntu22.04-py3.10-arm64-gc84594c4578c` |
| 910b | amd64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.14.0-910b-ubuntu22.04-py3.10-amd64-gc84594c4578c` |
| 910b | arm64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.9.0.post9-910b-ubuntu22.04-py3.11-arm64-gc84594c4578c` |
| 910b | amd64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.9.0.post9-910b-ubuntu22.04-py3.11-amd64-gc84594c4578c` |
| 910b | arm64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.7.1.post11-910b-ubuntu22.04-py3.11-arm64-gc84594c4578c` |
| 910b | amd64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.7.1.post11-910b-ubuntu22.04-py3.11-amd64-gc84594c4578c` |
| 910b | arm64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.10.0.post7-910b-ubuntu22.04-py3.11-arm64-gc84594c4578c` |
| 910b | amd64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.10.0.post7-910b-ubuntu22.04-py3.11-amd64-gc84594c4578c` |
| 910b | arm64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.11.0.post3-910b-ubuntu22.04-py3.11-arm64-gc84594c4578c` |
| 910b | amd64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.11.0.post3-910b-ubuntu22.04-py3.11-amd64-gc84594c4578c` |
| 910b | arm64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.12.0.post3-910b-ubuntu22.04-py3.11-arm64-gc84594c4578c` |
| 910b | amd64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.12.0.post3-910b-ubuntu22.04-py3.11-amd64-gc84594c4578c` |
| 910b | arm64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.13.0-910b-ubuntu22.04-py3.11-arm64-gc84594c4578c` |
| 910b | amd64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.13.0-910b-ubuntu22.04-py3.11-amd64-gc84594c4578c` |
| 910b | arm64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.14.0-910b-ubuntu22.04-py3.11-arm64-gc84594c4578c` |
| 910b | amd64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.14.0-910b-ubuntu22.04-py3.11-amd64-gc84594c4578c` |
| 910b | arm64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.9.0.post9-910b-openeuler24.03-py3.10-arm64-gc84594c4578c` |
| 910b | amd64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.9.0.post9-910b-openeuler24.03-py3.10-amd64-gc84594c4578c` |
| 910b | arm64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.7.1.post11-910b-openeuler24.03-py3.10-arm64-gc84594c4578c` |
| 910b | amd64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.7.1.post11-910b-openeuler24.03-py3.10-amd64-gc84594c4578c` |
| 910b | arm64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.10.0.post7-910b-openeuler24.03-py3.10-arm64-gc84594c4578c` |
| 910b | amd64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.10.0.post7-910b-openeuler24.03-py3.10-amd64-gc84594c4578c` |
| 910b | arm64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.11.0.post3-910b-openeuler24.03-py3.10-arm64-gc84594c4578c` |
| 910b | amd64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.11.0.post3-910b-openeuler24.03-py3.10-amd64-gc84594c4578c` |
| 910b | arm64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.12.0.post3-910b-openeuler24.03-py3.10-arm64-gc84594c4578c` |
| 910b | amd64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.12.0.post3-910b-openeuler24.03-py3.10-amd64-gc84594c4578c` |
| 910b | arm64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.13.0-910b-openeuler24.03-py3.10-arm64-gc84594c4578c` |
| 910b | amd64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.13.0-910b-openeuler24.03-py3.10-amd64-gc84594c4578c` |
| 910b | arm64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.14.0-910b-openeuler24.03-py3.10-arm64-gc84594c4578c` |
| 910b | amd64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.14.0-910b-openeuler24.03-py3.10-amd64-gc84594c4578c` |
| 910b | arm64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.9.0.post9-910b-openeuler24.03-py3.11-arm64-gc84594c4578c` |
| 910b | amd64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.9.0.post9-910b-openeuler24.03-py3.11-amd64-gc84594c4578c` |
| 910b | arm64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.7.1.post11-910b-openeuler24.03-py3.11-arm64-gc84594c4578c` |
| 910b | amd64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.7.1.post11-910b-openeuler24.03-py3.11-amd64-gc84594c4578c` |
| 910b | arm64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.10.0.post7-910b-openeuler24.03-py3.11-arm64-gc84594c4578c` |
| 910b | amd64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.10.0.post7-910b-openeuler24.03-py3.11-amd64-gc84594c4578c` |
| 910b | arm64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.11.0.post3-910b-openeuler24.03-py3.11-arm64-gc84594c4578c` |
| 910b | amd64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.11.0.post3-910b-openeuler24.03-py3.11-amd64-gc84594c4578c` |
| 910b | arm64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.12.0.post3-910b-openeuler24.03-py3.11-arm64-gc84594c4578c` |
| 910b | amd64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.12.0.post3-910b-openeuler24.03-py3.11-amd64-gc84594c4578c` |
| 910b | arm64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.13.0-910b-openeuler24.03-py3.11-arm64-gc84594c4578c` |
| 910b | amd64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.13.0-910b-openeuler24.03-py3.11-amd64-gc84594c4578c` |
| 910b | arm64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.14.0-910b-openeuler24.03-py3.11-arm64-gc84594c4578c` |
| 910b | amd64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.14.0-910b-openeuler24.03-py3.11-amd64-gc84594c4578c` |
| a3 | arm64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.9.0.post9-a3-ubuntu22.04-py3.10-arm64-gc84594c4578c` |
| a3 | amd64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.9.0.post9-a3-ubuntu22.04-py3.10-amd64-gc84594c4578c` |
| a3 | arm64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.7.1.post11-a3-ubuntu22.04-py3.10-arm64-gc84594c4578c` |
| a3 | amd64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.7.1.post11-a3-ubuntu22.04-py3.10-amd64-gc84594c4578c` |
| a3 | arm64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.10.0.post7-a3-ubuntu22.04-py3.10-arm64-gc84594c4578c` |
| a3 | amd64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.10.0.post7-a3-ubuntu22.04-py3.10-amd64-gc84594c4578c` |
| a3 | arm64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.11.0.post3-a3-ubuntu22.04-py3.10-arm64-gc84594c4578c` |
| a3 | amd64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.11.0.post3-a3-ubuntu22.04-py3.10-amd64-gc84594c4578c` |
| a3 | arm64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.12.0.post3-a3-ubuntu22.04-py3.10-arm64-gc84594c4578c` |
| a3 | amd64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.12.0.post3-a3-ubuntu22.04-py3.10-amd64-gc84594c4578c` |
| a3 | arm64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.13.0-a3-ubuntu22.04-py3.10-arm64-gc84594c4578c` |
| a3 | amd64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.13.0-a3-ubuntu22.04-py3.10-amd64-gc84594c4578c` |
| a3 | arm64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.14.0-a3-ubuntu22.04-py3.10-arm64-gc84594c4578c` |
| a3 | amd64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.14.0-a3-ubuntu22.04-py3.10-amd64-gc84594c4578c` |
| a3 | arm64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.9.0.post9-a3-ubuntu22.04-py3.11-arm64-gc84594c4578c` |
| a3 | amd64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.9.0.post9-a3-ubuntu22.04-py3.11-amd64-gc84594c4578c` |
| a3 | arm64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.7.1.post11-a3-ubuntu22.04-py3.11-arm64-gc84594c4578c` |
| a3 | amd64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.7.1.post11-a3-ubuntu22.04-py3.11-amd64-gc84594c4578c` |
| a3 | arm64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.10.0.post7-a3-ubuntu22.04-py3.11-arm64-gc84594c4578c` |
| a3 | amd64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.10.0.post7-a3-ubuntu22.04-py3.11-amd64-gc84594c4578c` |
| a3 | arm64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.11.0.post3-a3-ubuntu22.04-py3.11-arm64-gc84594c4578c` |
| a3 | amd64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.11.0.post3-a3-ubuntu22.04-py3.11-amd64-gc84594c4578c` |
| a3 | arm64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.12.0.post3-a3-ubuntu22.04-py3.11-arm64-gc84594c4578c` |
| a3 | amd64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.12.0.post3-a3-ubuntu22.04-py3.11-amd64-gc84594c4578c` |
| a3 | arm64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.13.0-a3-ubuntu22.04-py3.11-arm64-gc84594c4578c` |
| a3 | amd64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.13.0-a3-ubuntu22.04-py3.11-amd64-gc84594c4578c` |
| a3 | arm64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.14.0-a3-ubuntu22.04-py3.11-arm64-gc84594c4578c` |
| a3 | amd64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.14.0-a3-ubuntu22.04-py3.11-amd64-gc84594c4578c` |
| a3 | arm64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.9.0.post9-a3-openeuler24.03-py3.10-arm64-gc84594c4578c` |
| a3 | amd64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.9.0.post9-a3-openeuler24.03-py3.10-amd64-gc84594c4578c` |
| a3 | arm64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.7.1.post11-a3-openeuler24.03-py3.10-arm64-gc84594c4578c` |
| a3 | amd64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.7.1.post11-a3-openeuler24.03-py3.10-amd64-gc84594c4578c` |
| a3 | arm64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.10.0.post7-a3-openeuler24.03-py3.10-arm64-gc84594c4578c` |
| a3 | amd64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.10.0.post7-a3-openeuler24.03-py3.10-amd64-gc84594c4578c` |
| a3 | arm64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.11.0.post3-a3-openeuler24.03-py3.10-arm64-gc84594c4578c` |
| a3 | amd64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.11.0.post3-a3-openeuler24.03-py3.10-amd64-gc84594c4578c` |
| a3 | arm64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.12.0.post3-a3-openeuler24.03-py3.10-arm64-gc84594c4578c` |
| a3 | amd64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.12.0.post3-a3-openeuler24.03-py3.10-amd64-gc84594c4578c` |
| a3 | arm64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.13.0-a3-openeuler24.03-py3.10-arm64-gc84594c4578c` |
| a3 | amd64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.13.0-a3-openeuler24.03-py3.10-amd64-gc84594c4578c` |
| a3 | arm64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.14.0-a3-openeuler24.03-py3.10-arm64-gc84594c4578c` |
| a3 | amd64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.14.0-a3-openeuler24.03-py3.10-amd64-gc84594c4578c` |
| a3 | arm64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.9.0.post9-a3-openeuler24.03-py3.11-arm64-gc84594c4578c` |
| a3 | amd64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.9.0.post9-a3-openeuler24.03-py3.11-amd64-gc84594c4578c` |
| a3 | arm64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.7.1.post11-a3-openeuler24.03-py3.11-arm64-gc84594c4578c` |
| a3 | amd64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.7.1.post11-a3-openeuler24.03-py3.11-amd64-gc84594c4578c` |
| a3 | arm64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.10.0.post7-a3-openeuler24.03-py3.11-arm64-gc84594c4578c` |
| a3 | amd64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.10.0.post7-a3-openeuler24.03-py3.11-amd64-gc84594c4578c` |
| a3 | arm64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.11.0.post3-a3-openeuler24.03-py3.11-arm64-gc84594c4578c` |
| a3 | amd64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.11.0.post3-a3-openeuler24.03-py3.11-amd64-gc84594c4578c` |
| a3 | arm64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.12.0.post3-a3-openeuler24.03-py3.11-arm64-gc84594c4578c` |
| a3 | amd64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.12.0.post3-a3-openeuler24.03-py3.11-amd64-gc84594c4578c` |
| a3 | arm64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.13.0-a3-openeuler24.03-py3.11-arm64-gc84594c4578c` |
| a3 | amd64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.13.0-a3-openeuler24.03-py3.11-amd64-gc84594c4578c` |
| a3 | arm64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.14.0-a3-openeuler24.03-py3.11-arm64-gc84594c4578c` |
| a3 | amd64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.14.0-a3-openeuler24.03-py3.11-amd64-gc84594c4578c` |
| 950 | arm64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.9.0.post9-950-ubuntu22.04-py3.10-arm64-gc84594c4578c` |
| 950 | amd64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.9.0.post9-950-ubuntu22.04-py3.10-amd64-gc84594c4578c` |
| 950 | arm64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.7.1.post11-950-ubuntu22.04-py3.10-arm64-gc84594c4578c` |
| 950 | amd64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.7.1.post11-950-ubuntu22.04-py3.10-amd64-gc84594c4578c` |
| 950 | arm64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.10.0.post7-950-ubuntu22.04-py3.10-arm64-gc84594c4578c` |
| 950 | amd64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.10.0.post7-950-ubuntu22.04-py3.10-amd64-gc84594c4578c` |
| 950 | arm64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.11.0.post3-950-ubuntu22.04-py3.10-arm64-gc84594c4578c` |
| 950 | amd64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.11.0.post3-950-ubuntu22.04-py3.10-amd64-gc84594c4578c` |
| 950 | arm64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.12.0.post3-950-ubuntu22.04-py3.10-arm64-gc84594c4578c` |
| 950 | amd64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.12.0.post3-950-ubuntu22.04-py3.10-amd64-gc84594c4578c` |
| 950 | arm64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.13.0-950-ubuntu22.04-py3.10-arm64-gc84594c4578c` |
| 950 | amd64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.13.0-950-ubuntu22.04-py3.10-amd64-gc84594c4578c` |
| 950 | arm64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.14.0-950-ubuntu22.04-py3.10-arm64-gc84594c4578c` |
| 950 | amd64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.14.0-950-ubuntu22.04-py3.10-amd64-gc84594c4578c` |
| 950 | arm64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.9.0.post9-950-ubuntu22.04-py3.11-arm64-gc84594c4578c` |
| 950 | amd64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.9.0.post9-950-ubuntu22.04-py3.11-amd64-gc84594c4578c` |
| 950 | arm64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.7.1.post11-950-ubuntu22.04-py3.11-arm64-gc84594c4578c` |
| 950 | amd64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.7.1.post11-950-ubuntu22.04-py3.11-amd64-gc84594c4578c` |
| 950 | arm64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.10.0.post7-950-ubuntu22.04-py3.11-arm64-gc84594c4578c` |
| 950 | amd64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.10.0.post7-950-ubuntu22.04-py3.11-amd64-gc84594c4578c` |
| 950 | arm64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.11.0.post3-950-ubuntu22.04-py3.11-arm64-gc84594c4578c` |
| 950 | amd64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.11.0.post3-950-ubuntu22.04-py3.11-amd64-gc84594c4578c` |
| 950 | arm64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.12.0.post3-950-ubuntu22.04-py3.11-arm64-gc84594c4578c` |
| 950 | amd64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.12.0.post3-950-ubuntu22.04-py3.11-amd64-gc84594c4578c` |
| 950 | arm64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.13.0-950-ubuntu22.04-py3.11-arm64-gc84594c4578c` |
| 950 | amd64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.13.0-950-ubuntu22.04-py3.11-amd64-gc84594c4578c` |
| 950 | arm64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.14.0-950-ubuntu22.04-py3.11-arm64-gc84594c4578c` |
| 950 | amd64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.14.0-950-ubuntu22.04-py3.11-amd64-gc84594c4578c` |
| 950 | arm64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.9.0.post9-950-openeuler24.03-py3.10-arm64-gc84594c4578c` |
| 950 | amd64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.9.0.post9-950-openeuler24.03-py3.10-amd64-gc84594c4578c` |
| 950 | arm64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.7.1.post11-950-openeuler24.03-py3.10-arm64-gc84594c4578c` |
| 950 | amd64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.7.1.post11-950-openeuler24.03-py3.10-amd64-gc84594c4578c` |
| 950 | arm64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.10.0.post7-950-openeuler24.03-py3.10-arm64-gc84594c4578c` |
| 950 | amd64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.10.0.post7-950-openeuler24.03-py3.10-amd64-gc84594c4578c` |
| 950 | arm64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.11.0.post3-950-openeuler24.03-py3.10-arm64-gc84594c4578c` |
| 950 | amd64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.11.0.post3-950-openeuler24.03-py3.10-amd64-gc84594c4578c` |
| 950 | arm64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.12.0.post3-950-openeuler24.03-py3.10-arm64-gc84594c4578c` |
| 950 | amd64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.12.0.post3-950-openeuler24.03-py3.10-amd64-gc84594c4578c` |
| 950 | arm64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.13.0-950-openeuler24.03-py3.10-arm64-gc84594c4578c` |
| 950 | amd64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.13.0-950-openeuler24.03-py3.10-amd64-gc84594c4578c` |
| 950 | arm64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.14.0-950-openeuler24.03-py3.10-arm64-gc84594c4578c` |
| 950 | amd64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.14.0-950-openeuler24.03-py3.10-amd64-gc84594c4578c` |
| 950 | arm64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.9.0.post9-950-openeuler24.03-py3.11-arm64-gc84594c4578c` |
| 950 | amd64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.9.0.post9-950-openeuler24.03-py3.11-amd64-gc84594c4578c` |
| 950 | arm64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.7.1.post11-950-openeuler24.03-py3.11-arm64-gc84594c4578c` |
| 950 | amd64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.7.1.post11-950-openeuler24.03-py3.11-amd64-gc84594c4578c` |
| 950 | arm64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.10.0.post7-950-openeuler24.03-py3.11-arm64-gc84594c4578c` |
| 950 | amd64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.10.0.post7-950-openeuler24.03-py3.11-amd64-gc84594c4578c` |
| 950 | arm64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.11.0.post3-950-openeuler24.03-py3.11-arm64-gc84594c4578c` |
| 950 | amd64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.11.0.post3-950-openeuler24.03-py3.11-amd64-gc84594c4578c` |
| 950 | arm64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.12.0.post3-950-openeuler24.03-py3.11-arm64-gc84594c4578c` |
| 950 | amd64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.12.0.post3-950-openeuler24.03-py3.11-amd64-gc84594c4578c` |
| 950 | arm64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.13.0-950-openeuler24.03-py3.11-arm64-gc84594c4578c` |
| 950 | amd64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.13.0-950-openeuler24.03-py3.11-amd64-gc84594c4578c` |
| 950 | arm64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.14.0-950-openeuler24.03-py3.11-arm64-gc84594c4578c` |
| 950 | amd64 | `v1.2.0.dev0-cann9.2.0-torch_npu2.14.0-950-openeuler24.03-py3.11-amd64-gc84594c4578c` |

<!-- END generated image tags -->

## 许可证与免责声明

HyperParallel 采用 [Apache 2.0 许可证](https://github.com/mindspore-ai/hyper-parallel/blob/master/LICENSE)。拉取和使用镜像适用[华为容器镜像许可协议](https://www.hiascend.com/legal/ascendhub-download)。镜像内各组件适用各自的许可证。
