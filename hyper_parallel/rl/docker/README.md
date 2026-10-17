# Hyper-RL 运行镜像

ARM64 的普通 RL、MoE R3、Codex Agent、DeepSeek Harness 及 Qwen3.5/3.8 controller 统一使用
`swr.cn-east-3.myhuaweicloud.com/huawei-hyper-rl/hyper-rl:v0.23.0.post1-code-agent-arm64`。
GSM8K 启动脚本、Search-R1 的训练与候选容器、RL 系统测试均默认使用此镜像，仍保留显式镜像覆盖接口。

## 下载与校验

镜像公开可读，无需登录：

```bash
image=swr.cn-east-3.myhuaweicloud.com/huawei-hyper-rl/hyper-rl:v0.23.0.post1-code-agent-arm64
docker pull "${image}"
docker image inspect --format '{{.Id}} {{index .RepoDigests 0}} {{.Os}}/{{.Architecture}}' "${image}"
```

| 项目            | 值                                                                                                                  |
| --------------- | ------------------------------------------------------------------------------------------------------------------- |
| 平台            | `linux/arm64`，Ascend NPU                                                                                         |
| Manifest digest | `sha256:0d1e8af0cbc8eb558497958b827260fa6d9a635fc20e4ec74916587b0348731d`                                         |
| Image ID        | `sha256:9d796c354631648532be7c5c775860dab3653359fd249423f074abf35341a2e1`                                         |
| 大小            | 展开约 18.9 GB，压缩层合计约 7.25 GB                                                                                  |
| 训练与推理      | Torch `2.10.0+cpu`、torch-npu `2.10.0.post4`、CANN `9.1`、Transformers `5.5.4`、vLLM `0.23.0+empty`、vLLM-Ascend `0.23.0.post1` |
| 数值一致性依赖  | `batch_invariant_ops==1.0.0`、`flash-attn-npu==0.2.0b1`                                                         |
| Agent           | Codex CLI`0.152.1`；DeepSeek Harness SDK/runtime `0.1.1rc1`                                                     |

Codex 使用独立二进制包，无需 Node.js。DeepSeek 指 Agent harness，不表示新增模型支持。
检查 Agent 依赖：

```bash
docker run --rm "${image}" /bin/bash -lc '
set -e
codex --version
python -c "import deepseek_harness; from importlib.metadata import version; print(version(\"deepseek-harness-sdk\"))"
'
```

## Search-R1 的独立 Codex 候选容器

训练/推理容器与候选容器使用同一镜像，但权限和职责独立。
候选容器只运行 Codex CLI 0.152.1、Python 标准库请求转发程序，以及 Bash、`rg`、`sed`、`cat` 等搜索工具；
不加载 vLLM、Torch、CANN 或模型权重，不分配 NPU，也不挂载 Docker socket。
候选启动器将镜像已有的 `/opt/codex/codex-path` 加入自己的 `PATH`，以使用随 CLI 打包的 `rg`；
不安装额外软件，也不修改宿主或训练进程的工具路径。
保留无外网、只读根目录、候选文章只读及会话隔离的原边界。

R3 采集与模型计算发生在训练/推理容器中，见 [MoE R3](../docs/moe_code_agent.md#moe-r3)。
无模型隔离测试 `tests/st/search_r1_isolation.py` 验证工具和请求转发边界，不代替 NPU 训练验收。

## 单轮 code 的 SandboxFusion 镜像

沙箱服务独立于训练和推理镜像，不需要 NPU。公开镜像：
`swr.cn-east-3.myhuaweicloud.com/huawei-hyper-rl/sandboxfusion-python:v2-arm64`。

```bash
docker pull swr.cn-east-3.myhuaweicloud.com/huawei-hyper-rl/sandboxfusion-python:v2-arm64
```

| 项目 | 值 |
| --- | --- |
| 平台 | `linux/arm64`，Python 3.12.13 / Debian bookworm slim |
| SWR manifest digest | `sha256:edcb45e18b4b7d4f608df13ee9436a504087fb7d7213cabdac42b26654086069` |
| Image ID | `sha256:5f5c291ba3d15d9ec20bc3b5695f87f107ee26d39c86817d92283fbd79a0f856` |
| 展开大小 | 约 192 MB |
| 基础镜像 ARM64 digest | `sha256:3f9c7f75b916e8279e24136d7a57769aba9e193715be52570417698c52d6eed0` |
| SandboxFusion revision | `469de0845cb7d665bc68044d541aa3aa28c5b30a` |

直接下载上述已发布镜像，无需在本仓库构建。拉取后核对身份：

```bash
sandbox_image=swr.cn-east-3.myhuaweicloud.com/huawei-hyper-rl/sandboxfusion-python:v2-arm64
docker image inspect --format '{{.Id}} {{.Os}}/{{.Architecture}} {{json .RepoDigests}}' "$sandbox_image"
```

镜像仅保留 Python HTTP 服务与执行器依赖。
服务使用 `/opt/sandbox-server`，候选使用仅含标准库、无 pip 的 `/opt/candidate`。
运行系统工具为 bash、coreutils、util-linux、mount、sudo、iproute2、cgroup-tools、procps，以及回收退出子进程的 tini；
编译工具仅在构建阶段使用，不进入最终镜像。镜像不包含 Torch、vLLM、CANN、Conda 或其他语言运行时。

沙箱以非 root 身份执行候选程序，使用无桥网络隔离，限制合并 stdout/stderr 原始输出为 1 MiB，
并清理超时进程；服务仅开放 Python 请求，不加载数据集/数据库后端，也不引入训练代码。
`agentic.code.runtime_version` 使用上表的 Image ID，不使用 registry manifest 或基础镜像 digest。
若显式更换镜像，须同步实际 Image ID 并重新验证隔离和执行合同。
完整系统包清单保存在镜像 `/opt/sandboxfusion/system-packages.lock`。

启动方式、cgroup 与隔离要求见 [code 示例](../examples/code/README.md#sandboxfusion-部署)。
真实 ST 覆盖正确/错误答案、异常、超时、输出限制、候选权限/依赖/网络及跨请求文件隔离；
普通 CPU UT 不启动服务。更换沙箱镜像无需重新执行 NPU 训练即可验证此执行合同。

## 宿主要求

以下要求针对训练容器；沙箱的隔离权限与 cgroup 条件见上述 code 部署说明。

- Linux ARM64、Docker、兼容的 Ascend NPU driver；`npu-smi info` 正常。
- 按运行场景准备空闲、健康的 NPU，模型和数据目录，以及至少 30 GB Docker 可用空间。
- 使用匹配的仓库源码；代码、模型、数据和结果由启动脚本挂载，不打入此镜像。

容器入口加载 CANN，启动脚本调用 `docker/install_runtime.sh` 安装 RL 包并注册 vLLM 插件。
手动 `docker exec` 或覆盖 `PYTHONPATH` 后，须在启动模型的同一 shell 中重新执行
`source /usr/local/Ascend/ascend-toolkit/set_env.sh`，保留 CANN Python 路径，确保 `import acl` 可用。
`docker/patches/vllm-dp-coordinator-timeout.patch` 将 vLLM DP Coordinator 的启动等待时间从 30 秒延长到 120 秒；
GSM8K 的 TP 与一致性启动脚本会在容器中应用它。更新 vLLM 后应先确认补丁仍适用。
具体运行步骤见 [Hyper-RL README](../README.md)，系统测试见 [ST 指南](../README.md#系统测试)。
需要自定义镜像时，保留各启动脚本的 `HYPER_*_IMAGE` 和测试的 `RL_ST_*_IMAGE` 覆盖方式。

## 仓库型 Code Agent 与 Qwen3.5 家族

ARM64 controller 使用上文的统一镜像，身份及版本以 [下载与校验](#下载与校验)为准。
仓库源码、模型和数据由运行时挂载提供，并按当前 checkout 安装 editable 主包及 RL 插件。
远端 amd64 任务机的 CLI/仓库镜像仍按 [SWE-Smith 说明](../examples/swe_smith/README.md#镜像准备与身份) 单独准备。
SWE-Smith 候选和 grader 仍使用按仓库任务准备的专用镜像，不用统一训练镜像替代其项目依赖。

9B、27B 的 controller 基于官方 `quay.io/ascend/vllm-ascend:v0.23.0.post1`，
`Dockerfile.qwen3_8` 是仅加入固定 Codex CLI 的最小叠层入口；发布版本还包含已准备的 RL 依赖，复用官方推理栈。
已验收运行环境为 Torch 2.10.0+cpu、torch-npu 2.10.0.post4、CANN 9.1、
Transformers 5.5.4、vLLM 0.23.0、vLLM-Ascend 0.23.0.post1、Codex CLI 0.152.1；
SWE-Smith 不依赖 SWE-bench evaluator。Qwen3 dense 一致性按版本选择对应的固定配方，见
[一致性运行说明](../docs/qwen3_training_inference_consistency.md#运行)。

[SWE-Smith Dockerfile](../examples/swe_smith/Dockerfile) 为选中仓库的固定任务镜像加入 CLI 和独立 Python stdlib。
任务镜像与 CLI 必须匹配宿主架构；`prepare_images.py` 默认不拉取缺失镜像。可信 controller 持有 Docker 客户端及 socket，候选与 grader
无网络、无 NPU、无 socket，使用同一固定镜像。宿主 Docker 客户端可能还需 `libltdl.so.7`。

模型、任务数据、逐任务镜像清单和结果目录均由运行配置指定。
构建或启动前核对磁盘空间、依赖版本及 editable 导入路径。

非特权 NPU controller 映射所需 `davinci`、`davinci_manager`、`devmm_svm` 和 `hisi_hdc`，
只读挂载驱动；用户须有设备及 socket 的属组权限、匹配 UID 的用户记录和可写家目录/缓存。
同时核对 CANN 和算子目录读取权限。`ASCEND_RT_VISIBLE_DEVICES` 使用容器内枚举编号；
例如只映射四张物理卡时，容器内编号为 0–3。启动前用同一用户核验小算子及实际占卡。
当前 v0.23 镜像中，`ASCEND_RT_VISIBLE_DEVICES` 应按编号升序指定；非升序列表可能被识别为零设备。
模型加载前核对 `torch.npu.device_count()` 与计划卡数一致，避免设备发现失败后意外构建 CPU 网格。

候选和 grader 的 `pids_limit` 使用 1024，为 CLI 与 pytest 线程留出空间。
纯 CPU controller 不分配 NPU，设置 `TORCH_DEVICE_BACKEND_AUTOLOAD=0`。
SWE 准备、训练/恢复命令、图配置和验收边界统一见
[Code Agent 示例](../examples/swe_smith/README.md)。
