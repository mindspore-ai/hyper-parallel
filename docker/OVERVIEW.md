# HyperParallel on Ascend

Run distributed PyTorch workloads on Ascend with HyperParallel. Each image contains the native Indexed Dataset extension compiled for its Python interpreter and CPU architecture. The recipe selects official AscendHub CANN or TorchNPU (PTA) bases by chip, Python ABI and PyTorch version.

## Quick reference

| Resource | Where to find it |
| --- | --- |
| Build recipe and CLI | [Dockerfile](./Dockerfile) · [build.sh](./build.sh) |
| Published images | [HyperParallel tags on Quay](https://quay.io/repository/hyper-parallel/hyper-parallel-npu?tab=tags) |
| Upstream images | [AscendHub CANN](https://www.hiascend.com/en/developer/ascendhub/detail/17da20d1c2b6493cb38765adeba85884) · [AscendHub TorchNPU](https://www.hiascend.com/en/developer/ascendhub/detail/1bb1e5d9c8c64f6ea8db0d0ec5061531) · [AscendHub MindSpeed-LLM](https://www.hiascend.com/en/developer/ascendhub/detail/e26da9266559438b93354792f25b2f4a) |

## Supported build targets

<!-- BEGIN generated support matrix -->

| CANN | PyTorch | torch_npu (PTA) | Python | OS | Chips |
| --- | --- | --- | --- | --- | --- |
| 9.2.0 | 2.9.0 | 2.9.0.post9 | 3.10, 3.11, 3.12 | ubuntu22.04, openeuler24.03 | 910b, a3, 950 |
| 9.2.0 | 2.7.1 | 2.7.1.post11 | 3.10, 3.11, 3.12 | ubuntu22.04, openeuler24.03 | 910b, a3, 950 |
| 9.2.0 | 2.10.0 | 2.10.0.post7 | 3.10, 3.11, 3.12 | ubuntu22.04, openeuler24.03 | 910b, a3, 950 |
| 9.2.0 | 2.11.0 | 2.11.0.post3 | 3.10, 3.11, 3.12 | ubuntu22.04, openeuler24.03 | 910b, a3, 950 |
| 9.2.0 | 2.12.0 | 2.12.0.post3 | 3.10, 3.11, 3.12 | ubuntu22.04, openeuler24.03 | 910b, a3, 950 |
| 9.2.0 | 2.13.0 | 2.13.0 | 3.10, 3.11, 3.12 | ubuntu22.04, openeuler24.03 | 910b, a3, 950 |
| 9.2.0 | 2.14.0 | 2.14.0 | 3.10, 3.11, 3.12 | ubuntu22.04, openeuler24.03 | 910b, a3, 950 |
| 9.0.0 | 2.9.0 | 2.9.0.post2 | 3.11 | ubuntu22.04, openeuler24.03 | 910b, a3 |

The script defaults to Python 3.12 and PTA 2.9.0.post9. [HyperParallel Python requirements](https://github.com/mindspore-ai/hyper-parallel/blob/c84594c4578c4ad7b048db40fa0c4998c776c0e4/requirements.txt)

<!-- END generated support matrix -->

The MindSpeed-LLM 26.2.0 reference publishes six CANN 9.2.0 tags for Python 3.12, PyTorch 2.10.0 and PTA 2.10.0.post7: three chips (`910b`, `a3`, `950`) on two operating systems. Each upstream tag covers Linux arm64 and amd64. HyperParallel builds a separate tag for each CPU architecture. The 950 label covers the 950PR/DT series and 910B denotes Atlas A2. Quay lists the images available to pull. [MindSpeed-LLM release matrix](https://gitcode.com/Ascend/MindSpeed-LLM/blob/224267cad7dc68045e13753017a75fa80ea7e8e0/docker/OVERVIEW.md)

The pinned HyperParallel source requires Python `>=3.10,<3.13`. The image builds the wheel from that source with the interpreter supplied by its base. [Upstream Python requirement](https://github.com/mindspore-ai/hyper-parallel/blob/c84594c4578c4ad7b048db40fa0c4998c776c0e4/setup.py)

| Python on CANN 9.2 | How the image is built |
| --- | --- |
| 3.10, 3.11 | Use the CANN image for that interpreter and chip. Install the matching PyTorch CPU and TorchNPU wheels. |
| 3.12 | Inherit the framework packages from the official PTA image. |

[Ascend PyTorch compatibility](https://github.com/Ascend/pytorch/blob/master/COMPATIBILITY.en.md) · [Official PyTorch CPU wheels](https://download.pytorch.org/whl/cpu/torch/) · [TorchNPU packages](https://pypi.org/project/torch-npu/)

### Image tags

`v{HyperParallel}-cann{CANN}-torch_npu{PTA}-{chip}-{OS}-py{Python}-{arch}-g{revision}`

In order, the fields identify the HyperParallel source version, CANN toolkit, TorchNPU package, target chip, OS release, Python major/minor version, Linux CPU architecture and first 12 characters of the HyperParallel **wheel source** commit. All tags use the same [Dockerfile](./Dockerfile).

## Quick start

### Build on a Linux CPU host

Build on the target CPU architecture with Bash 4+, GNU coreutils and Docker. Generating or checking the documentation and pushing an OS pair also require awk and GNU diffutils (`cmp` and `diff`). Cross-building requires a builder with emulation for that architecture. Run the following from the parent of `docker/`. Building the wheel requires no NPU or host Python.

<!-- BEGIN generated build example -->

```bash
bash docker/build.sh -t 910b -o ubuntu22.04 -a arm64 --cann-version 9.2.0 --python-version 3.12 --pta-version 2.9.0.post9
IMAGE=quay.io/hyper-parallel/hyper-parallel-npu:v1.2.0.dev0-cann9.2.0-torch_npu2.9.0.post9-910b-ubuntu22.04-py3.12-arm64-gc84594c4578c
```

<!-- END generated build example -->

| Script option | Values and defaults |
| --- | --- |
| `-t`, `--npu-type` (also `-c`, `--chip`) | Required for builds. A2: `a2` or `910b`. A3: `a3`. A5: `a5` or `950`. |
| `-o`, `--os` | `ubuntu22.04` or `openeuler24.03`. Omit to build both. |
| `-a`, `--arch` | Defaults to the host. `arm64`/`aarch64` or `amd64`/`x86_64`. |
| `--python-version` | 3.10, 3.11 or 3.12. Defaults to the Python in the selected PTA base. |
| `--hyper-parallel-branch NAME` or `--hyper-parallel-commit SHA` | Select an official HyperParallel branch or a full 40-character commit. The default is the commit pinned in the recipe. The script resolves a branch once for both OS images. |
| `--hp-version V` | Expected wheel version. For custom sources, the script reads it from that commit's `setup.py` by default. |
| `--cann-version` | Defaults to `9.2.0`. `9.0.0` selects Python 3.11, PyTorch 2.9.0 and PTA 2.9.0.post2 on A2 and A3. |
| `--torch-version` | Select a PyTorch release for the chosen CANN version. The script resolves the matching PTA package. |
| `--pta-version`, `--torch-npu-version` | Aliases for the exact PTA package version. If you also select PyTorch, the two versions must match. |
| `--tags`, `--print-command` | Print image names or ready-to-run `docker build` commands with the resolved base digest and full source commit. |
| `--no-cache`, `--network MODE` | Disable the Docker build cache or select the build network (for example, `host`). |
| `--push` | Build both OS images before pushing them to Quay. Requires write access. Omit `--os`. |
| `--docs`, `--check`, `--export DIR` | Regenerate the two repository overviews, check them for drift, or export the recipe and one bilingual pair per tag to a new or empty directory. |
| `-h`, `--help` | Print CLI help. |

This example selects `master` and an arm64 stack. Save the printed command to rebuild the same commit later. To select a specific commit, replace `--hyper-parallel-branch master` with `--hyper-parallel-commit` and the full SHA.

```bash
bash docker/build.sh --print-command --hyper-parallel-branch master -t 910b -o ubuntu22.04 -a arm64 --torch-version 2.10.0 > build-hp.sh
bash build-hp.sh
```

For native Docker builds, `BASE_IMAGE` selects the official base for the chip, OS and Python ABI. `--platform` selects the CPU architecture. `CANN_VERSION`, `PYTHON_VERSION` and the framework arguments validate the installed stack. For a native build on Docker 18.09, omit `--platform`. The script handles this automatically. Select the source with either `HYPER_PARALLEL_COMMIT` (full SHA) or `HYPER_PARALLEL_BRANCH` (for example, `master`). Set `HP_VERSION` to the wheel version. Use `--no-cache` to refresh a moving branch when passing it directly to Docker, and choose a custom image name with `-t`. Every image records the full commit in `/etc/hyper-parallel-revision`.

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

### Software and network access

| Stage | Dependencies | Outbound access |
| --- | --- | --- |
| Pull the base (Docker daemon) | Public AscendHub CANN or PTA image pinned by digest | HTTPS 443 to `swr.cn-south-1.myhuaweicloud.com` and its layer redirects under `*.obs.cn-south-1.myhuaweicloud.com` |
| Fetch source and build the wheel | Git on the host when resolving custom sources. Git, a C++ compiler, Make, setuptools, wheel and pybind11 in the build stage. | HTTPS 443 to `github.com` and `gitcode.com`. Git uses GitHub first and falls back to the official GitCode mirror. Source revisions that build optional Multicore operators also fetch `cann/ops-nn` from GitCode. |
| Install the framework (CANN bases) | PyTorch CPU and Ascend TorchNPU wheels for the selected version and ABI | HTTPS 443 to `download.pytorch.org`, its download host `download-r2.pytorch.org`, and the PyPI sites below. |
| Install Python packages | Public PyPI build tools and HyperParallel's runtime dependencies, including NumPy, torchdata, PyYAML, datasets and transformers. The build pins the selected torch and torch_npu versions and runs `pip check`. | HTTPS 443 to `pypi.org` and `files.pythonhosted.org`. |
| Install missing OS tools (conditional) | Distribution-signed Ubuntu or openEuler packages | Ubuntu uses `ports.ubuntu.com` (arm64) or `archive.ubuntu.com` and `security.ubuntu.com` (amd64), typically HTTP 80. openEuler uses `repo.openeuler.org`, `repo.openeuler.openatom.cn`, `mirrors.openeuler.org` and mirrors returned by its metalink, over HTTPS 443. |
| Pull or push a published image | Quay and its image CDN. Push requires registry credentials. | HTTPS 443 to `quay.io` and `*.quay.io`. A local build and a local run need no Quay access. |
| Run on an NPU host | Matching Ascend hardware and a compatible host driver. Use Ascend Docker Runtime for automatic device injection or standard runc for manual mapping. | The single-device example runs offline from a local image. Multi-host training needs connectivity between the participating hosts. |

For separate build and NPU hosts, transfer the image with `docker save -o hp.tar "$IMAGE"` and `docker load -i hp.tar`. Configure the Docker daemon's proxy for registry pulls if needed. For downloads inside a build, configure Docker client proxy settings or pass `--build-arg HTTPS_PROXY` and `--build-arg HTTP_PROXY` to `docker build`.

### Start and enter an interactive container

The entrypoint prints the HyperParallel version and container license at startup. `-it` keeps standard input open and allocates a terminal. On the host, run `docker image ls "$IMAGE"` to confirm the local `REPOSITORY:TAG` selected above. To inspect the installed software, open Bash on a Linux CPU host. Run `python3 -I -m pip show hyper-parallel torch torch-npu` inside and type `exit` when finished.

```bash
docker run -it --rm --runtime=runc "$IMAGE" /bin/bash
```

On a device host with a compatible driver and Ascend Docker Runtime, choose a physical device available to the new container. Check load with `npu-smi info` and check container assignments when the device is exclusive. From your project directory, start a named container on device 1 and mount your code at `/workspace`.

```bash
docker run -it --rm --name hp-dev --runtime=ascend -e ASCEND_VISIBLE_DEVICES=1 -v "$PWD:/workspace" -w /workspace "$IMAGE" bash
```

Leave the first terminal open. From a second host terminal, open another shell in the same container. Docker removes the container when the first shell exits.

```bash
docker exec -it hp-dev bash
```

Inside the container, list the device and check a tensor operation. In this single-device example, `npu:0` is the device index inside the container and Python prints `1 6`.

```bash
npu-smi info -l
python3 -c 'import torch, hyper_parallel; print(torch.npu.device_count(), torch.arange(4, device="npu:0").sum().item())'
```

#### Map an NPU manually with runc

You can also use the standard `runc` runtime. This example uses an A2/910B host, physical device `/dev/davinci1` and a driver installed at `/usr/local/Ascend/driver`. Adjust the nodes and paths to match your host. Run the same device check inside.

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

| Optional flag | When to use it |
| --- | --- |
| `-v /data/my-dataset:/data:ro` | Replace the source path with your dataset directory. The project mount at `/workspace` is writable. |
| `--network host`, `--ipc=host` | Share the host network or IPC namespace when a distributed job requires it. |
| `--privileged` | On a host that permits privileged containers, this grants access to all host devices. Add `-e ASCEND_RT_VISIBLE_DEVICES=1` to select physical device 1 for PyTorch. `npu-smi` can still list all devices. |

To run an offline check from the host, this example uses physical device 0 and prints `6`.

```bash
docker run --rm --network=none --runtime=ascend -e ASCEND_VISIBLE_DEVICES=0 "$IMAGE" \
  python3 -c 'import torch, hyper_parallel; print(torch.arange(4, device="npu").sum().item())'
```

### Extend the image

In the project containing `train.py`, put the model’s additional Python dependencies in `requirements.txt`. Use an empty file if the base already provides them. Pin dependencies compatible with the selected framework versions. Save the following as `Dockerfile.hp`. Using `python3 -I -m pip` installs and checks dependencies in the application interpreter’s package environment.

```dockerfile
ARG HP_IMAGE
FROM ${HP_IMAGE}
WORKDIR /workspace
COPY requirements.txt .
RUN python3 -I -m pip install --no-cache-dir --index-url https://pypi.org/simple -r requirements.txt && python3 -I -m pip check
COPY . .
CMD ["python3", "train.py"]
```

Build from the project directory on the host. The first `docker run` executes the training script. The second opens an interactive shell. For manual device mapping, substitute `hp-training:local` for `"$IMAGE"` in the `runc` example above.

```bash
docker build -f Dockerfile.hp --build-arg HP_IMAGE="$IMAGE" -t hp-training:local .
docker run --rm --runtime=ascend -e ASCEND_VISIBLE_DEVICES=1 hp-training:local
docker run -it --rm --runtime=ascend -e ASCEND_VISIBLE_DEVICES=1 hp-training:local /bin/bash
```

To modify HyperParallel itself, set `HYPER_PARALLEL_COMMIT` and `HP_VERSION` in the native build command to your public project commit and its package version, then rebuild the wheel for the selected Python ABI.

## Full build tags

<!-- BEGIN generated image tags -->

| Chip | CPU | Full image tag |
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

## License and disclaimer

HyperParallel is licensed under [Apache 2.0](https://github.com/mindspore-ai/hyper-parallel/blob/master/LICENSE). Pulling and using the image is subject to the [Huawei Container License Agreement](https://www.hiascend.com/en/legal/ascendhub-download). Each bundled component retains its own license.
