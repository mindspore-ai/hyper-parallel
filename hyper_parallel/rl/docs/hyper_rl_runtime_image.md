# Hyper-RL 运行镜像

## 镜像信息

| 项目 | 值 |
| --- | --- |
| 公共镜像 | `swr.cn-east-3.myhuaweicloud.com/huawei-hyper-rl/hyper-rl:v0.22.1rc1-arm64` |
| Launcher 默认值 | 同上，无需本地改名 |
| 源码构建默认标签 | `hyper-parallel/hyper-rl:v0.22.1rc1` |
| 平台 | `linux/arm64` |
| Manifest digest | `sha256:601be16c16fc6a105154bc600a5c9b25420ae192982ea36e379a32e63b9b6e72` |
| 展开大小 | 约 18.4 GB |
| 基础镜像 | `hyper-parallel/unified-e2-dev:v0.22.1rc1` |

镜像包含 CANN、Torch/torch-npu、Transformers、vLLM/vLLM-Ascend、`batch_invariant_ops==1.0.0` 和
`flash-attn-npu==0.2.0b1`。仓库代码、模型、数据、结果和开发 patch 不写入镜像，由 launcher 在运行时挂载。

## 下载

镜像公开可读，无需登录：

```bash
docker pull \
  swr.cn-east-3.myhuaweicloud.com/huawei-hyper-rl/hyper-rl:v0.22.1rc1-arm64
```

如果需要同时禁用环境变量和 Docker CLI 继承的代理：

```bash
env -u http_proxy -u https_proxy \
  -u HTTP_PROXY -u HTTPS_PROXY \
  -u ALL_PROXY -u all_proxy \
  docker pull \
  swr.cn-east-3.myhuaweicloud.com/huawei-hyper-rl/hyper-rl:v0.22.1rc1-arm64
```

该命令不修改 Docker daemon 自身的代理配置。

## 校验

```bash
docker image inspect \
  --format '{{index .RepoDigests 0}} {{.Os}}/{{.Architecture}}' \
  swr.cn-east-3.myhuaweicloud.com/huawei-hyper-rl/hyper-rl:v0.22.1rc1-arm64
```

输出应包含上述 digest 和 `linux/arm64`。正式 launcher 直接使用公共镜像地址，不需要创建本地标签。

校验固定依赖：

```bash
docker run --rm \
  swr.cn-east-3.myhuaweicloud.com/huawei-hyper-rl/hyper-rl:v0.22.1rc1-arm64 \
  /bin/bash -lc '
python - <<"PY"
from importlib.metadata import version

packages = (
    "torch-npu",
    "transformers",
    "vllm",
    "vllm-ascend",
    "batch_invariant_ops",
    "flash-attn-npu",
)
print({name: version(name) for name in packages})
PY
'
```

镜像 entrypoint 自动加载 CANN。正式 launcher 会设置源码 `PYTHONPATH`、挂载 driver、模型、数据和结果，不要求在容器内
安装额外依赖。

如果使用源码构建的本地镜像，通过 `HYPER_QWEN3_IMAGE` 或 `HYPER_QWEN3_TP_IMAGE` 指定即可，无需修改 launcher。

## 宿主要求

- Linux ARM64 与兼容的 Ascend NPU driver；
- `npu-smi info` 能正常显示设备；
- 只选择 `Health=OK` 且无运行进程的 NPU；
- 至少 30 GB Docker 可用空间，模型、数据和结果空间另计；
- 与镜像版本匹配的 HyperParallel 仓库；
- 模型和数据路径位于宿主机，并由 launcher 只读挂载。

具体运行命令见 [Hyper-RL README](../README.md)。

## 从源码构建

仓库构建入口为：

```bash
./hyper_parallel/rl/docker/build_image.sh
```

构建脚本验证 flash-attn-npu wheel SHA256 和所有固定依赖版本。默认网络代理为宿主
`http://127.0.0.1:8991`，可通过 `HYPER_RL_BUILD_PROXY` 覆盖。开发中的代码和 patch 仍不进入镜像。

## 安全

vLLM RLHF/refit development endpoints 使用不安全序列化。镜像只能运行在受信任、隔离的训练网络，不应暴露到公网。
