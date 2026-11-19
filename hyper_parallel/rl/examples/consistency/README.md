# Qwen3-MoE 训推一致性

四层 Qwen3-MoE 运行两步正常 GRPO，验证不同训推并行配置下的实际采样 token log-prob 完全一致。
模型保留原始 128 专家、top-k8 和权重，只裁剪 decoder 层数。

```text
consistency/
├── configs/qwen3_moe_4l_rl.yaml      # TP/SP/DP/EP 基线
├── configs/qwen3_moe_4l_cp_rl.yaml   # 训练 CP、推理 PCP、跨 PCP EP
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
依赖版本见[运行环境](../../docs/unified_execution_guide.md#运行环境)。镜像已包含采样器的
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
固定镜像缺少当前 HP 要求的 PEFT 依赖，启动器会下载固定版本 `peft==0.21.2` 和
`accelerate==1.12.0`，以 `--no-deps` 安装并执行 `pip check`；不会升级 Torch/NPU/V4 等数值依赖。
首次启动会进行图编译与捕获，需要等待；过程日志写入 `run/train.log`。

默认使用普通并行基线配置。两份 YAML 都显式包含完整替换配方，不通过隐藏的配置继承补充训练参数。
运行其他组合时只覆盖下面列出的拓扑字段，保持共享数值实现不变。

## 配置与验证矩阵

所有组合使用四张物理卡。训练的进程数为 `DP × TP × CP`，推理 worker 数为 `DP × TP × PCP`；
不能只改一个并行度而忽略总 worker 数和 EP 整除约束。表中的 DP2 训练指 `dp_shard=2` 的 FSDP 数据并行。

| 用例 | 配置 | 训练 | 推理 | 主要验证点 |
| --- | --- | --- | --- | --- |
| A | 普通基线 | TP4/SP/EP4/CP1 | TP2/DP2/PCP1/EP4，无 SP | 两端 TP 不同，训练开启 SP |
| B | 普通基线＋覆盖 | TP2/DP2/EP4/CP1，无 SP | TP4/DP1/PCP1/EP4/SP | FSDP 数据并行与推理 SP |
| C | CP/PCP 配置 | TP2/CP2/EP2/SP | TP2/DP1/PCP2/EP4，无 SP | 跨 CP 前向与跨 PCP 专家域 |
| D | CP/PCP 配置＋覆盖 | TP2/CP2/EP2/SP | TP2/DP2/PCP1/EP2，无 SP | 只在训练侧开启 CP |
| E | CP/PCP 配置＋覆盖 | TP2/CP2/EP2/SP | TP2/DP1/PCP2/EP2，无 SP | 相同 PCP 下不同专家组大小 |
| F | CP/PCP 配置＋覆盖 | TP2/CP2/EP2/SP | TP2/DP1/PCP2/EP4/SP，budget=512 | 推理 SP 与较大合法 batch |
| G | CP/PCP 配置＋覆盖 | TP2/DP2/CP1/EP2/SP | TP2/DP1/PCP2/EP4，无 SP | 只在推理侧开启 PCP |

两份文件是 [普通基线](configs/qwen3_moe_4l_rl.yaml) 和 [CP/PCP 配置](configs/qwen3_moe_4l_cp_rl.yaml)。
后者的 CP wrapper 声明为 `when: cp`，因此 G 中 CP1 不安装该 wrapper。
EP2 自然在 TP2 组内；EP4 跨两个 PCP 平面。两端 CP 不必相同。

保留上文模型、数据、wheel 和 DEVICES 设置；每个用例使用独立 RESULT_ROOT，**依次**运行：

```bash
runner=hyper_parallel/rl/examples/consistency/scripts/run_qwen3_moe_docker.sh
base=hyper_parallel/rl/examples/consistency/configs/qwen3_moe_4l_rl.yaml
context=hyper_parallel/rl/examples/consistency/configs/qwen3_moe_4l_cp_rl.yaml

# A：默认的不同 TP/SP 组合。
CONFIG="$base" RESULT_ROOT=/tmp/hyper-rl-consistency/A bash "$runner"

# B：训练 FSDP2，推理开启 SP。
CONFIG="$base" RESULT_ROOT=/tmp/hyper-rl-consistency/B bash "$runner" \
  --train.accelerator.tp=2 --train.accelerator.dp_shard=2 \
  --train.accelerator.sequence_parallel=false \
  --rollout.vllm.tensor_parallel_size=4 --rollout.vllm.data_parallel_size=1 \
  --rollout.vllm.sequence_parallel=true

# C：训练 CP2，推理 PCP2，专家域跨 PCP。
CONFIG="$context" RESULT_ROOT=/tmp/hyper-rl-consistency/C bash "$runner"

# D：训练 CP2，推理 PCP1。
CONFIG="$context" RESULT_ROOT=/tmp/hyper-rl-consistency/D bash "$runner" \
  --rollout.vllm.context_parallel_size=1 --rollout.vllm.data_parallel_size=2 \
  --rollout.vllm.expert_parallel_size=2

# E：推理 PCP2/EP2，每个 TP2 组拥有一份完整专家集合。
CONFIG="$context" RESULT_ROOT=/tmp/hyper-rl-consistency/E bash "$runner" \
  --rollout.vllm.expert_parallel_size=2

# F：推理 SP，调度预算大于默认 MC2 容量 256。
CONFIG="$context" RESULT_ROOT=/tmp/hyper-rl-consistency/F bash "$runner" \
  --rollout.vllm.sequence_parallel=true --rollout.vllm.max_num_batched_tokens=512

# G：训练 CP1/FSDP2，推理仍为 PCP2。
CONFIG="$context" RESULT_ROOT=/tmp/hyper-rl-consistency/G bash "$runner" \
  --train.accelerator.cp=1 --train.accelerator.dp_shard=2
```

这些是可分别复现的验证组合，不表示一次启动会穷举全部拓扑。
F 提高调度上界，但短 prompt、max_num_seqs 和实际流量决定是否真的超过 MC2 容量；
不能仅凭 budget=512 就声称已执行 AllGather。后端切换和大 bucket 图回放由下面的 EP 设备测试强制覆盖。

## EP 后端与图模式对照

两份配方都使用 `unified_ep_compute_fn`，默认 `mc2_capacity: 256`、`inference_backend: auto`。
训练保持动态路由；推理在 MC2 可用且当前 bucket 适合时使用 MC2，其余合法尺寸由 AllGather 承接。
具体条件与数值合并规则见 [MoE 实现](../../docs/unified_execution_moe.md#后端选择与容量)。

对照 eager 与图模式，可对任一上述命令增加 `--rollout.vllm.enforce_eager=true`，并使用新的结果目录。
两种模式使用相同的 MC2/AllGather 算法选择。
需要显式选择 AllGather 或修改 MC2 容量时，修改选中 YAML 的 MoE replacement 条目：

```yaml
implementation:
  _target_: rl.unified_execution.runtimes.hyper_parallel.moe.unified_ep_compute_fn
  mc2_capacity: 256
  inference_backend: allgather
```

CLI loader 不支持通过列表下标覆盖 `plan_overrides`；不要使用 `--train.plan_overrides.4...`。
修改后使用独立结果目录，实际解析配置会保存在 `run/config.yaml` 中。

## 检查结果

每步更新前比较实际 rollout log-prob 与 Actor 重算结果，要求使用同一权重版本和 token 前缀。
仅统计有效 action token，排除 padding，要求非空、有限且最大/平均差值均为 **0**。
不使用 Actor 概率回填推理结果，不修改采样 token 或 reward 来通过检查。

两步正常 RL 循环与权重发布完成后才写出 `passed: true`：

```bash
RESULT_ROOT=/tmp/hyper-rl-consistency/C
cat "$RESULT_ROOT/run/consistency.json"
rg 'CONSISTENCY_CHECK|CONSISTENCY_PASS' "$RESULT_ROOT/run/train.log"
```

用例命令中的 `RESULT_ROOT=... bash ...` 是该命令的局部环境变量；检查时请先将 RESULT_ROOT
设为要检查的目录，例如 `RESULT_ROOT=/tmp/hyper-rl-consistency/C`。
报告需要包含 `completed_steps: 2`、第 1/2 步，以及每步正的 `log_probs_valid_tokens`。

| 文件 | 内容 |
| --- | --- |
| `run/consistency.json` | 两步采样概率检查与实际训练/推理配置 |
| `run/train.log` | 图编译、推理、训练、发布及一致性检查日志 |
| `run/config.yaml` | 包含 CLI 覆盖的完整实际配置 |
| `run/environment.txt`、`run/v4-wheel.sha256` | 依赖与数值内核身份 |
| `Qwen3-MoE-4L/crop_manifest.json` | 裁剪权重的来源和校验信息 |

默认短响应可能产生全零任务奖励，因此示例的成功不能单独证明非零梯度或学习效果。
需要非零更新时使用下述测试夹具：它只提供明确的交替奖励，采样、模型计算、optimizer 和发布仍走真实路径。
梯度正确性另由完整参数对照检查，不由两步采样概率检查代替。

## 组件与非零更新检查

在已安装本仓及固定数值依赖的环境中，显式选择空闲设备：

```bash
# 真实 EP 路由、MC2/AllGather、大小 bucket 切换、图回放与训练梯度。
RL_EP_ST_DEVICES=0,1,2,3 RL_EP_ST_SIZE=4 \
  pytest -q hyper_parallel/rl/tests/st/test_unified_expert_parallel.py

# CP Attention、完整测试模型的 CP1/CP2 logits 与逐参数梯度。
RL_CP_ST_DEVICES=0,1,2,3 \
  pytest -q hyper_parallel/rl/tests/st/test_context_parallel.py -k 'attention or model'

# 跨 PCP 专家域、权重映射及分页 Attention。
RL_PCP_ST_DEVICES=0,1,2,3 RL_UNIFIED_ST_MODEL=/path/to/Qwen3-MoE-4L \
  pytest -q hyper_parallel/rl/tests/st/test_prefill_context_parallel.py -k 'mesh or attention'

# 两次非零更新及发布；该 YAML 内的模型、数据和结果路径必须对当前环境有效。
RL_PCP_ST_DEVICES=0,1,2,3 \
  RL_PCP_ST_RL_CONFIG=hyper_parallel/rl/examples/consistency/configs/qwen3_moe_4l_cp_rl.yaml \
  pytest -q hyper_parallel/rl/tests/st/test_prefill_context_parallel.py -k parallel_rl
```

执行范围和梯度容差见 [统一执行验收](../../docs/unified_execution_guide.md#支持范围与验收)。
这两份配置聚焦不同并行布局的前向一致性；跨机、大 EP、长上下文性能需要单独验收。
