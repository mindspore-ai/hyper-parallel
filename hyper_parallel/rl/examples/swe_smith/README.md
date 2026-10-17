# Code Agent：SWE-Smith

使用 Codex 修改仓库，通过独立测试评分，支持推理、GRPO 训练和 checkpoint 恢复。
默认使用本机 Docker socket；远端沙盒通过 `--docker-host` 指定。

## 环境准备

准备 Docker、模型权重、空闲 Ascend NPU，以及[固定版本的 RL controller 镜像](../../docker/README.md#仓库型-code-agent-与-qwen35-家族)。
宿主启动入口依赖 Python 和 PyYAML；数据准备在已安装本项目及 RL 依赖的环境中执行。
以下命令均从仓库根目录运行，路径替换为实际位置：

```bash
export PYTHONPATH="$PWD/hyper_parallel/rl:$PWD${PYTHONPATH:+:$PYTHONPATH}"
```

## 模型配方

| 模型 | 配方 | NPU 数量 |
| --- | --- | --- |
| Qwen3.5-9B（默认） | [qwen3_5_9b.yaml](configs/qwen3_5_9b.yaml) | 2 |
| Qwen3.8-27B | [qwen3_8_27b.yaml](configs/qwen3_8_27b.yaml) | 4 |

均为纯文本 GRPO，默认 eager。切换配方使用 `--config`，同时调整 `--devices`。

## 镜像准备与身份

镜像必须匹配沙盒主机的 CPU 架构。下面三个已发布镜像用于 ARM64 九例验证；
远端 amd64 或其他任务需准备对应镜像和清单，见[远端运行](#远端沙盒)。

### 本地 ARM64 九例镜像

```bash
SWE_REGISTRY=swr.cn-east-3.myhuaweicloud.com/huawei-hyper-rl/swe-smith
docker --host unix:///var/run/docker.sock pull "$SWE_REGISTRY:schema-arm64-codex-0.152.1"
docker --host unix:///var/run/docker.sock pull "$SWE_REGISTRY:boltons-arm64-codex-0.152.1"
docker --host unix:///var/run/docker.sock pull "$SWE_REGISTRY:oauthlib-arm64-codex-0.152.1"
```

若需要认证，先用有权限的账号执行 `docker login swr.cn-east-3.myhuaweicloud.com`。
镜像已包含源码、测试依赖和 Codex；直接使用 [images_arm64.json](configs/images_arm64.json)，无需构建。
该清单固定 Image ID 和远端摘要；原始数据集名称中的 `x86_64` 仅用于身份映射。

## 准备子集或全量数据

下载 agent-lightning 的[筛选数据包](https://drive.google.com/file/d/1q19DP53l4rldvBR2dkUhbaPI_mHVBVL1/view)，
使用其中的两个 JSONL。下面准备[固定九例](configs/selected_ids.txt)：六例训练、三例验证。

```bash
python -m examples.swe_smith.prepare_data \
  --train-jsonl /data/train_dataset_mixed.jsonl \
  --validation-jsonl /data/val_dataset_filtered.jsonl \
  --instance-ids-file hyper_parallel/rl/examples/swe_smith/configs/selected_ids.txt \
  --output /path/to/swe_smith/data
```

输出目录须为空，保存 Parquet、实例记录和选择清单。省略 `--instance-ids-file` 可准备全量数据，
但仍需提供所有任务对应的镜像，这三个镜像不覆盖完整数据集。

## 本地与远端统一运行入口

在宿主机设置路径和空闲 NPU 编号；数据准备若在容器内完成，这里填写对应的宿主路径：

```bash
SWE_DATA=/path/to/swe_smith/data
SWE_IMAGES="$PWD/hyper_parallel/rl/examples/swe_smith/configs/images_arm64.json"
SWE_MODEL=/path/to/Qwen3.5-9B
SWE_RUNS=/path/to/swe_smith/runs
SWE_CHECKPOINTS=/path/to/swe_smith/checkpoints
swe_common=(--data-dir "$SWE_DATA" --images-path "$SWE_IMAGES"
            --model-dir "$SWE_MODEL" --devices 0,1)
```

默认连接 `unix:///var/run/docker.sock`。入口自动填写配置、挂载目录并启动 controller；
模型和 NPU 始终位于 controller 宿主。需要特权设备访问时，显式追加
`swe_common+=(--privileged-controller)`，该选项与沙盒运行在本地还是远端无关。

### 短预算推理

```bash
python -m examples.swe_smith.launch --mode infer "${swe_common[@]}" \
  --output-dir "$SWE_RUNS/infer-short" \
  --checkpoint-dir "$SWE_CHECKPOINTS/infer-unused"
```

默认运行已准备的九例，每例最多两次调用、每次最多 512 输出 token。
可追加 `--instance-id ID` 选择单例。推理不保存 checkpoint，但入口仍要求该目录参数。

### 训练一步与新进程恢复

```bash
python -m examples.swe_smith.launch --mode train "${swe_common[@]}" \
  --steps 1 --output-dir "$SWE_RUNS/train-step1" \
  --checkpoint-dir "$SWE_CHECKPOINTS/train-step1"

python -m examples.swe_smith.launch --mode train "${swe_common[@]}" \
  --steps 2 --resume "$SWE_CHECKPOINTS/train-step1/step_1" \
  --output-dir "$SWE_RUNS/resume-step2" \
  --checkpoint-dir "$SWE_CHECKPOINTS/resume-step2"
```

`--steps` 是总步数，恢复到 2 只追加一步。恢复须保持模型、配方及数据/镜像清单身份一致。
每次使用新的结果目录，为完整 checkpoint 预留空间；训练和恢复分别保存。
日志见 `runtime.log`，推理结果见 `predictions.jsonl`，调用和评分记录见 `sessions/`。
入口结束后清理本次 controller 和临时目录，保留结果及 checkpoint；强制中断后需核查本次沙盒残留。

## 远端沙盒

远端主机需有 Docker、对应架构的任务镜像；宿主需已解锁的 SSH-agent 和已验证的 host key。
ARM64 远端可下载上面的镜像；amd64 远端须使用其对应的镜像和清单。
已有目标架构的任务镜像和 CLI 运行镜像时，可在 RL 环境中生成远端清单：

```bash
python -m examples.swe_smith.prepare_images \
  --instances /path/to/swe_smith/data/instances.jsonl \
  --output /path/to/swe_smith/images_remote.json \
  --docker-host ssh://USER@HOST:PORT --cli-image TARGET_CLI_IMAGE
```

已有完整清单可跳过此步；缺失任务镜像时显式加 `--pull`。
启动时将上面的 `SWE_IMAGES` 换成远端清单的宿主路径，重新构造 `swe_common`，再追加：

```bash
swe_common+=(--docker-host ssh://USER@HOST:PORT)
```

其余推理和训练命令相同。远端连接失败会报错，不会切换本地执行。

## 图执行与验证范围

9B 短预算 eager 已覆盖本地 ARM64 和远端 amd64 沙盒的推理、训练与新进程恢复。
真实零奖励、零梯度仅证明流程运行；学习验收仍需真实混合奖励和非零更新。
参考预算推理使用 [qwen3_5_9b_infer.yaml](configs/qwen3_5_9b_infer.yaml)，通过 `launch --mode infer --config` 选择。
评分合同、图配置、资源参数、27B 及全量验证限制见[开发说明](../../docs/swe_smith.md)。
