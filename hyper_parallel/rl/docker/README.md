# Hyper-RL Runtime Image

**English** | [简体中文](README.zh.md)

ARM64 ordinary RL, MoE R3, Codex Agent, DeepSeek Harness, and Qwen3.5/3.8 controllers all use
`swr.cn-east-3.myhuaweicloud.com/huawei-hyper-rl/hyper-rl:v0.23.0.post1-code-agent-arm64`.
GSM8K launch scripts, Search-R1 training containers, and RL system tests use this image by default, with explicit image overrides still available.

## Download and Verification

The image is publicly readable without authentication:

```bash
image=swr.cn-east-3.myhuaweicloud.com/huawei-hyper-rl/hyper-rl:v0.23.0.post1-code-agent-arm64
docker pull "${image}"
docker image inspect --format '{{.Id}} {{index .RepoDigests 0}} {{.Os}}/{{.Architecture}}' "${image}"
```

| Item | Value |
| --------------- | ------------------------------------------------------------------------------------------------------------------- |
| Platform | `linux/arm64`, Ascend NPU |
| Manifest digest | `sha256:0d1e8af0cbc8eb558497958b827260fa6d9a635fc20e4ec74916587b0348731d`                                         |
| Image ID        | `sha256:9d796c354631648532be7c5c775860dab3653359fd249423f074abf35341a2e1`                                         |
| Size | Approximately 18.9 GB unpacked and 7.25 GB total compressed layers |
| Training and inference | Torch `2.10.0+cpu`, torch-npu `2.10.0.post4`, CANN `9.1`, Transformers `5.5.4`, vLLM `0.23.0+empty`, vLLM-Ascend `0.23.0.post1` |
| Numerical consistency dependencies | `batch_invariant_ops==1.0.0`, `flash-attn-npu==0.2.0b1` |
| Agent           | Codex CLI `0.152.1`;DeepSeek Harness SDK/runtime `0.1.1rc1`                                                     |

Codex uses a standalone binary package without Node.js. DeepSeek refers to the agent harness, not new model support.
Check agent dependencies:

```bash
docker run --rm "${image}" /bin/bash -lc '
set -e
codex --version
python -c "import deepseek_harness; from importlib.metadata import version; print(version(\"deepseek-harness-sdk\"))"
'
```

## Independent Codex Candidate Containers for Search-R1

Training/inference continues to use the runtime image above. Search-R1 candidates use the separate `hyper-rl-codex:0.152.1-arm64` image,
containing only the unmodified Codex binary, Python, Bash, Git, and required search tools; it excludes Torch, vLLM, CANN, and model weights.
This tag is a local build artifact, not a published image address. Build it on the candidate execution host:

```bash
docker build --network host -f hyper_parallel/rl/docker/Dockerfile.codex \
  -t hyper-rl-codex:0.152.1-arm64 hyper_parallel/rl/docker
```

The Dockerfile pins the base image and Codex source, verifies the CLI binary's SHA256 and version, and accepts existing offline images through
`BASE_IMAGE` and `AGENT_IMAGE` build arguments. Record the actual Image ID; images from different sources are not equivalent builds.
`--network host` applies only during the build; candidates still run with `network=none`.
The launcher's `--image` selects the training image; `--candidate-image` selects the candidate image. Candidate tools are verified separately before training,
with Image ID, size, and versions recorded in `candidate-runtime.json`. All candidates in the run use the resolved immutable ID.
The image build precompiles the Python standard library so read-only candidates do not repeatedly compile modules at startup.

Candidates share `DockerWorkspace` and the Codex execution relay with repository tasks. Each episode still creates a clean container,
without a container pool. No NPU or Docker socket is mounted: only the current question's workspace, a separate Codex home, and read-only articles.
CPU candidates explicitly use `runc` to avoid inheriting the host's accelerator runtime; training/inference containers retain their existing runtime.
The host relay forwards model requests through the current session's Unix socket; containers cannot access training administration endpoints or external networks.
Isolation probes run inside the actual candidate container, without a separate probe container, and still check file scope, read-only articles, and non-root identity per question.
`execution-timing.json` records queueing, container startup, relay, isolation checks, CLI execution, and cleanup durations.
The host executor writes raw output to the session's `container/` directory for collection by the controller; they do not share writable log files.

Candidate environments explicitly set `SHELL=/bin/bash`. When `SHELL` is absent, the pinned CLI's `portable-pty 0.9.0` dependency
uses the non-thread-safe `getpwuid` to find the default shell; concurrent TTY creation can cause native crashes.
The generic Codex launcher preserves existing `SHELL` settings, and the candidate executor supplies missing values; neither rewrites tool arguments nor disables TTYs.
Signal termination preserves the signal name and negative exit code, returning `128 + signal`; container-boundary records retain Docker `State`.

R3 collection and model computation occur in the training/inference container; see [MoE R3](../docs/moe_code_agent.md#moe-r3).
The model-free test `tests/st/search_r1_isolation.py` validates tools and request relay boundaries; it does not replace NPU training acceptance.

## SandboxFusion Image for Single-Turn Code

The sandbox service is independent of training and inference images and requires no NPU. Public image:
`swr.cn-east-3.myhuaweicloud.com/huawei-hyper-rl/sandboxfusion-python:v2-arm64`.

```bash
docker pull swr.cn-east-3.myhuaweicloud.com/huawei-hyper-rl/sandboxfusion-python:v2-arm64
```

| Item | Value |
| --- | --- |
| Platform | `linux/arm64`, Python 3.12.13 / Debian bookworm slim |
| SWR manifest digest | `sha256:edcb45e18b4b7d4f608df13ee9436a504087fb7d7213cabdac42b26654086069` |
| Image ID | `sha256:5f5c291ba3d15d9ec20bc3b5695f87f107ee26d39c86817d92283fbd79a0f856` |
| Unpacked size | Approximately 192 MB |
| Base image ARM64 digest | `sha256:3f9c7f75b916e8279e24136d7a57769aba9e193715be52570417698c52d6eed0` |
| SandboxFusion revision | `469de0845cb7d665bc68044d541aa3aa28c5b30a` |

Download the published image directly; no build in this repository is required. Verify its identity after pulling:

```bash
sandbox_image=swr.cn-east-3.myhuaweicloud.com/huawei-hyper-rl/sandboxfusion-python:v2-arm64
docker image inspect --format '{{.Id}} {{.Os}}/{{.Architecture}} {{json .RepoDigests}}' "$sandbox_image"
```

The image contains only Python HTTP service and executor dependencies.
The service uses `/opt/sandbox-server`; candidates use `/opt/candidate`, containing only the standard library and no pip.
Runtime system tools are bash, coreutils, util-linux, mount, sudo, iproute2, cgroup-tools, procps, and tini for reaping child processes.
Compilers are used only at build time and excluded from the final image. The image contains no Torch, vLLM, CANN, Conda, or other language runtimes.

Candidates run as non-root, with no bridge network, a 1 MiB limit on combined raw stdout/stderr,
and cleanup of timed-out processes. The service accepts only Python requests, loads no dataset/database backend, and imports no training code.
`agentic.code.runtime_version` uses the Image ID above, not the registry manifest or base-image digest.
When replacing the image explicitly, update the actual Image ID and revalidate isolation and execution contracts.
The complete system package list is stored at `/opt/sandboxfusion/system-packages.lock` in the image.

See the [code example](../examples/code/README.md#sandboxfusion-deployment) for startup, cgroups, and isolation requirements.
Real ST covers correct/incorrect answers, exceptions, timeouts, output limits, candidate permissions/dependencies/network, and cross-request file isolation.
Ordinary CPU UT does not start the service. Sandbox image replacements can validate this execution contract without rerunning NPU training.

## Host Requirements

These requirements apply to training containers; see the code deployment guide above for sandbox isolation permissions and cgroups.

- Linux ARM64, Docker, and a compatible Ascend NPU driver; `npu-smi info` must work.
- Idle, healthy NPUs appropriate to the scenario, model/data directories, and at least 30 GB free Docker storage.
- A matching source checkout. Launch scripts mount code, models, data, and results; they are not packaged into this image.

The container entry point loads CANN; launch scripts invoke `docker/install_runtime.sh` to install RL and register the vLLM plugin.
After manual `docker exec` or overriding `PYTHONPATH`, rerun
`source /usr/local/Ascend/ascend-toolkit/set_env.sh` in the shell that launches the model, preserving CANN Python paths so `import acl` works.
`docker/patches/patch_sampler.py` applies the sampling tensor `record_stream` fix for validated vLLM-Ascend versions,
verifies source hashes before/after modification, supports repeated execution, and logs results; unknown versions are rejected. The GSM8K launcher invokes it inside the container.

`docker/patches/vllm-dp-coordinator-timeout.patch` extends vLLM DP Coordinator startup timeout from 30 to 120 seconds.
The GSM8K TP launch script applies it inside the container. Confirm the patch still applies after upgrading vLLM.
See the [Hyper-RL README](../README.md) for execution and the [ST guide](../README.md#system-tests) for system tests.
Custom images must retain launcher `HYPER_*_IMAGE` and test `RL_ST_*_IMAGE` overrides.

## Repository Code Agent and the Qwen3.5 Family

ARM64 controllers use the unified image above; [Download and Verification](#download-and-verification) defines its identity and versions.
Runtime mounts supply source, models, and data; install the editable main package and RL plugin from the current checkout.
Prepare CLI/repository images for remote amd64 task hosts separately as described in [SWE-Smith](../examples/swe_smith/README.md#image-preparation-and-identity).
SWE-Smith candidates and graders continue using task-specific repository images; the unified training image does not replace their project dependencies.

The 9B and 27B controllers are based on official `quay.io/ascend/vllm-ascend:v0.23.0.post1`.
`Dockerfile.qwen3_8` is a minimal layer adding only pinned Codex CLI; published versions also include prepared RL dependencies and reuse the official inference stack.
The accepted environment uses Torch 2.10.0+cpu, torch-npu 2.10.0.post4, CANN 9.1,
Transformers 5.5.4, vLLM 0.23.0, vLLM-Ascend 0.23.0.post1, Codex CLI 0.152.1;
SWE-Smith does not depend on the SWE-bench evaluator. Current inference uses native-vLLM; see the
[inference runtime guide](../docs/vllm_rollout.md) for deployment requirements.

The [SWE-Smith Dockerfile](../examples/swe_smith/Dockerfile) adds CLI and an independent Python standard library to the selected repository's pinned task image.
Task images and CLI must match the host architecture. `prepare_images.py` does not pull missing images by default. Trusted controllers hold the Docker client and socket; candidates and graders
have no network, NPU, or socket and use the same pinned image. The host Docker client may also need `libltdl.so.7`.

Runtime configuration specifies models, task data, per-task image manifests, and result directories.
Check disk space, dependency versions, and editable import paths before building or launching.

Nonprivileged NPU controllers map required `davinci`, `davinci_manager`, `devmm_svm`, and `hisi_hdc` devices
and mount drivers read-only. Users need device/socket group permissions, a matching UID entry, and writable home/cache directories.
Check read access to CANN and operator directories as well. `ASCEND_RT_VISIBLE_DEVICES` uses in-container enumeration:
if only four physical devices are mapped, container IDs are 0–3. Validate a small operator and actual device occupancy under the same user before launch.
In the current v0.23 image, specify `ASCEND_RT_VISIBLE_DEVICES` in ascending order; nonascending lists may be detected as zero devices.
Before model loading, ensure `torch.npu.device_count()` matches the planned count to prevent accidental CPU mesh construction after device discovery failure.

Candidates and graders use `pids_limit=1024`, leaving room for CLI and pytest threads.
CPU-only controllers receive no NPU and set `TORCH_DEVICE_BACKEND_AUTOLOAD=0`.
SWE preparation, training/resume commands, graph configuration, and acceptance boundaries are maintained in the
[Code Agent example](../examples/swe_smith/README.md).
