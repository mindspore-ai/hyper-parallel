# Qwen3-MoE 训推一致性

四层 Qwen3-MoE 运行两步正常 GRPO，验证不同训推并行配置下的实际采样 token log-prob 完全一致。
模型保留原始 128 专家、top-k8 和权重，只裁剪 decoder 层数。

```text
consistency/
├── configs/qwen3_moe_4l_rl.yaml
└── scripts/
    ├── run_qwen3_moe_docker.sh   # 本机 Docker 入口
    ├── prepare_checkpoint.py    # 裁剪、校验并复用四层权重
    └── run_consistency.py       # 正常 RL 循环 + 更新前一致性检查
```

## 本地复现

以下命令在同一个 Bash 终端中、从仓库根目录执行。实验使用本机 Docker，无需远程服务器。

### 1. 准备镜像、模型和数据

宿主机需要 Linux ARM64、可用的 Docker 和 Ascend 910 系列设备，`npu-smi info` 能正常显示设备。
运行实验需要四张空闲 NPU；构建 wheel 不需要 NPU。宿主无需安装 Torch、CANN Toolkit 等 Python/编译依赖，
工具链由镜像提供；宿主仍需安装兼容的 NPU 驱动。

```bash
image=swr.cn-east-3.myhuaweicloud.com/huawei-hyper-rl/hyper-rl:consistency-v0.30.0-cann9.2.1-arm64
docker pull "$image"
docker image inspect --format '{{.Id}} {{.Architecture}}' "$image"
npu-smi info
```

构建和运行脚本均校验镜像 ID：
`sha256:f139fa4776cf71fa1ebf00c47fa1f4c313deec48267c2102a1a42d968634fb08`。
依赖版本见[运行环境](../../rl/unified_execution/README.md#运行环境)。镜像已包含采样器的
`q.record_stream(...)` 修复，不需要再执行 `patch_sampler.py`。

准备以下文件；路径相对于仓库根目录：

| 路径                            | 内容                                                      |
| ------------------------------- | --------------------------------------------------------- |
| `../models/Qwen3-30B-A3B/`    | 完整原始模型，包括配置、tokenizer、safetensors 权重及索引 |
| `../data/gsm8k/train.parquet` | GSM8K 训练数据，与现有 GSM8K 示例使用同一格式             |
| `../data/gsm8k/test.parquet`  | GSM8K 测试数据                                            |

YAML 从 parquet 的 `prompt` 列读取提示、`extra_info` 列读取答案信息。
无需提前裁剪模型：启动器从完整权重中提取前四层及公共参数，保存到结果目录的 `Qwen3-MoE-4L/`，
不修改原始模型。再次运行时，校验裁剪配置与文件哈希后复用。

### 2. 准备 V4 补丁 wheel

基础镜像中的 V4 尚未包含 canonical rowsum 补丁，运行时需要加载 `0.5.0+hpbi1`。
**缺少 wheel 时先构建；构建成功、校验通过后，再运行第三步实验。** 以下两种方式选一种。

**已有原验收 wheel：** 按实际位置设置路径，使用对应的固定校验和：

```bash
export V4_WHEEL="$(realpath ../wheels/flash_attn_npu-0.5.0+hpbi1-cp312-cp312-linux_aarch64.whl)"
export V4_WHEEL_SHA256=5ef04a872b1be2c812bb0c36c17ba4776a4b5f9b9eabb24eab0a56a139a497c9
printf '%s  %s\n' "$V4_WHEEL_SHA256" "$V4_WHEEL" | sha256sum --check
```

**没有 wheel：** 使用仓库提供的脚本，在现有镜像中构建：

```bash
bash hyper_parallel/rl/docker/build_v4_wheel.sh ../wheels/v4-build
```

脚本下载固定提交的 FlashAttention NPU 和 CATLASS，应用仓库中的补丁，只编译 Ascend 910 的 V4。
它不修改基础镜像，不向宿主安装依赖，也不占用 NPU；临时源码与编译缓存随容器删除。
输出目录必须为空；重新构建时换一个空目录，已有成功产物可直接复用，无需再次编译。

构建成功后，输出目录包含 wheel、`SHA256SUMS`、`build.json`、`build.log` 和 `environment.txt`。
执行下面的命令，校验并选用本次产物：

```bash
wheel_dir=$(realpath ../wheels/v4-build)
(cd "$wheel_dir" && sha256sum --check SHA256SUMS)
export V4_WHEEL="$wheel_dir/flash_attn_npu-0.5.0+hpbi1-cp312-cp312-linux_aarch64.whl"
export V4_WHEEL_SHA256=$(cut -d ' ' -f 1 "$wheel_dir/SHA256SUMS")
```

重新编译可能改变 wheel 的文件哈希，因此要使用本次 `SHA256SUMS` 中的值，不要填入原验收 wheel 的哈希。
请保留 `build.json`，其中记录了源码提交、补丁及产物身份。
下载使用宿主的 HTTP/HTTPS 代理设置；已有完整 Git 源码时，也可通过 `SOURCE_DIR` 离线构建，
详见[构建说明](../../docker/README.md#构建-v4-补丁-wheel)。

### 3. 运行两步 RL 实验

保留第二步导出的 `V4_WHEEL` 和 `V4_WHEEL_SHA256`，设置本地输入、设备与结果目录：

```bash
export MODEL_ROOT="$(realpath ../models/Qwen3-30B-A3B)"
export DATA_ROOT="$(realpath ../data/gsm8k)"
export DEVICES=0,1,2,3
export RESULT_ROOT=/tmp/hyper-rl-consistency/train-sp

bash hyper_parallel/rl/examples/consistency/scripts/run_qwen3_moe_docker.sh
```

`DEVICES` 必须列出四张不同的空闲卡，可按本机情况修改。`RESULT_ROOT` 首次运行至少预留 10 GiB，
用于裁剪权重和实验记录；需要持久保存时改为自己的数据目录。Docker 存储还需有镜像、构建及容器运行空间。

启动器依次检查输入、wheel 校验和、镜像身份和设备占用，创建独立容器，安装当前 checkout 与 RL 插件，
将补丁 wheel 安装到 `/opt/hp-v4`，供训练进程和 vLLM worker 共同加载，然后裁剪权重并执行两步训练。
源码、完整模型和数据以只读方式挂载；结果保存在宿主 `RESULT_ROOT`，容器退出后保留。
首次启动会进行图编译与捕获，需要等待；过程日志写入 `run/train.log`。

默认使用 [qwen3_moe_4l_rl.yaml](configs/qwen3_moe_4l_rl.yaml)，四卡共卡运行：

| 实验         | 训练配置                  | 推理配置                |
| ------------ | ------------------------- | ----------------------- |
| 默认配置     | TP4 / SP / EP4            | TP2 / DP2 / 无 SP / EP4 |
| 下方对照配置 | TP2 / FSDP2 / 无 SP / EP4 | TP4 / SP / EP4          |

共卡表示两端复用相同四张物理卡；各自按自己的 TP、DP、EP 配置组织进程组。
`CONFIG=/absolute/path/to/config.yaml` 可选择另一份 YAML；模型、数据的宿主路径通过上面的环境变量修改。

### 4. 验证另一组并行配置

默认实验结束后，保持相同模型、数据和 wheel，使用另一个结果目录运行对照：

```bash
export RESULT_ROOT=/tmp/hyper-rl-consistency/infer-sp
bash hyper_parallel/rl/examples/consistency/scripts/run_qwen3_moe_docker.sh \
  --train.accelerator.tp=2 --train.accelerator.dp_shard=2 \
  --train.accelerator.sequence_parallel=false \
  --rollout.vllm.tensor_parallel_size=4 --rollout.vllm.data_parallel_size=1 \
  --rollout.vllm.sequence_parallel=true
```

两组实验分别记录结果，方便对照训练开启 SP、推理开启 SP 时的训推一致性。

## 检查结果

配方启用统一模型与 Attention、Level 3（关闭未初始化内存填充）、普通 row-parallel PIK、
固定容量 EP；保留 `FULL_AND_PIECEWISE`、prefix caching、chunked prefill 和权重发布。
完整配置位于 [YAML](configs/qwen3_moe_4l_rl.yaml)，实现见 [Unified Execution](../../rl/unified_execution/README.md)。

每步更新前，使用同一权重版本，比较 rollout 实际采样 log-prob 与 Actor 重算结果；
仅统计有效 action token，排除 padding，要求非空、有限且最大绝对差为 **0**。
不满足条件立即失败，不修改 token、reward 或 Actor 计算结果来通过检查。
两步训练与权重发布正常完成后，`run/consistency.json` 才写入 `passed: true`；
`run/train.log` 保存过程，`run/environment.txt` 保存依赖版本。
检查针对实际采样 token；梯度正确性与非零参数更新由独立算子/模型回归覆盖。

实验命令成功退出后，检查当前结果目录：

```bash
cat "$RESULT_ROOT/run/consistency.json"
rg 'CONSISTENCY_CHECK|CONSISTENCY_PASS' "$RESULT_ROOT/run/train.log"
```

验收时应看到 `passed: true`、`completed_steps: 2`，`checks` 中包含第 1、2 步；
每步 `log_probs_valid_tokens > 0`，`rollout_log_probs_diff_max` 与 `rollout_log_probs_diff_mean` 均为 `0`。
若启动或一致性检查失败，先查看终端输出及日志；本次成功报告只在两步完整完成后生成。

| 结果文件                            | 用途                       |
| ----------------------------------- | -------------------------- |
| `run/consistency.json`            | 两步验收结论和每步数值差异 |
| `run/train.log`                   | 训练、推理及一致性检查日志 |
| `run/config.yaml`                 | 包含命令行覆盖的实际配置   |
| `run/environment.txt`             | 容器中的 Python 依赖版本   |
| `run/v4-wheel.sha256`             | 本次使用的 V4 wheel 校验和 |
| `Qwen3-MoE-4L/crop_manifest.json` | 裁剪权重的来源和校验记录   |
