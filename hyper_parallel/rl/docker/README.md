# Hyper-RL 运行镜像

普通 RL、Codex Agent 和 DeepSeek Harness 共用一个镜像，启动脚本和系统测试均默认使用此版本。

## 下载与校验

镜像公开可读，无需登录：

```bash
image=swr.cn-east-3.myhuaweicloud.com/huawei-hyper-rl/hyper-rl:v0.22.1rc1-unified-arm64
docker pull "${image}"
docker image inspect --format '{{index .RepoDigests 0}} {{.Os}}/{{.Architecture}}' "${image}"
```

| 项目            | 值                                                                                                                  |
| --------------- | ------------------------------------------------------------------------------------------------------------------- |
| 平台            | `linux/arm64`，Ascend NPU                                                                                         |
| Manifest digest | `sha256:450b32a4a1d7818d80b832335665a7d8bece5b91827f8ee80a9f62609e939944`                                         |
| 展开大小        | 约 18.9 GB                                                                                                          |
| 训练与推理      | Torch`2.10.0+cpu`、torch-npu `2.10.0`、Transformers `5.5.4`、vLLM `0.22.1+empty`、vLLM-Ascend `0.22.1rc1` |
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

## 宿主要求

- Linux ARM64、Docker、兼容的 Ascend NPU driver；`npu-smi info` 正常。
- 按运行场景准备空闲、健康的 NPU，模型和数据目录，以及至少 30 GB Docker 可用空间。
- 使用匹配的仓库源码；代码、模型、数据和结果由启动脚本挂载，不打入此镜像。

容器入口加载 CANN，启动脚本调用 `docker/install_runtime.sh` 安装 RL 包并注册 vLLM 插件。
`docker/patches/vllm-dp-coordinator-timeout.patch` 将 vLLM DP Coordinator 的启动等待时间从 30 秒延长到 120 秒；
GSM8K 的 TP 与一致性启动脚本会在容器中应用它。更新 vLLM 后应先确认补丁仍适用。
具体运行步骤见 [Hyper-RL README](../README.md)，系统测试见 [ST 指南](../docs/hyper-rl-st.md)。
需要自定义镜像时，保留各启动脚本的 `HYPER_*_IMAGE` 和测试的 `RL_ST_*_IMAGE` 覆盖方式。
