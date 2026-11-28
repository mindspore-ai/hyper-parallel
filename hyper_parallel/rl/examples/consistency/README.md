# Qwen3 训推一致性验收

每个用例运行两步 GRPO。在每次 Actor 更新前，独立比较 rollout 和 Actor 对有效采样 token 给出的 log-prob；要求有效 token 数大于零、统计量有限、最大和平均绝对差均为零。两步均通过后，[验收入口](scripts/run_consistency.py) 写出包含 `passed: true` 的 JSON 报告。

| 模型 | 设备 | 验收配置 |
| --- | --- | --- |
| Qwen3-4B | 2 NPU | [完整 Dense 模型，训练 TP/SP](configs/qwen3_4b_rl.yaml) |
| Qwen3-30B-A3B 前四层 | 4 NPU | [TP/SP/DP/EP 基线](configs/qwen3_moe_4l_rl.yaml) · [CP/PCP 与跨 PCP 的 EP](configs/qwen3_moe_4l_cp_rl.yaml) |

用例复用 GSM8K 的数据格式、Agent 和评分。完整模型的 native/unified 50 步训练命令及 reward 对照见 [GSM8K 训练示例](../gsm8k/README.zh.md)。

## Qwen3-4B

[稠密 4B 配方](configs/qwen3_4b_rl.yaml) 使用完整 Qwen3-4B 权重，两张 NPU 同置运行两步 GRPO。默认训练 TP2/SP、推理 TP2，均为 BF16、Level 3 和 V4。PIK 的 `num_leaves: auto` 在此配置解析为 2，覆盖 Attention 的 `o_proj` 和 MLP 的 `down_proj`。在已安装 Hyper-RL 和 V4 的运行容器中，从仓库根目录运行：

```bash
torchrun --standalone --local-addr=127.0.0.1 --nproc-per-node=2 \
  hyper_parallel/rl/examples/consistency/scripts/run_consistency.py \
  hyper_parallel/rl/examples/consistency/configs/qwen3_4b_rl.yaml \
  --report /results/qwen3_4b_consistency.json
```

权重和数据分别挂载到 `/models/Qwen3-4B` 和 `/data/gsm8k`。

## Qwen3-MoE

四层 Qwen3-MoE 运行两步正常 GRPO，验证不同训推并行配置下的实际采样 token log-prob 完全一致。
模型保留原始 128 专家、top-k8 和权重，只裁剪 decoder 层数。

[四卡 Docker 启动器](scripts/run_qwen3_moe_consistency_docker.sh)调用[权重裁剪脚本](scripts/prepare_checkpoint.py)准备四层模型，再运行两步验收。

## 本地复现

以下命令在同一个 Bash 终端中、从仓库根目录执行。实验使用本机 Docker

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

依赖版本见[运行环境](../../docs/unified_execution_guide.md#运行环境)。

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

训练和推理均加载 `flash_attn_npu` 的 `0.5.0+hpbi1` wheel。可使用已有 wheel，也可从源码构建。

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

bash hyper_parallel/rl/examples/consistency/scripts/run_qwen3_moe_consistency_docker.sh
```

可通过 `IMAGE` 指定含所需数值算子和 vLLM 接口的本地镜像。
`DEVICES` 必须列出四张不同的空闲卡，可按本机情况修改。`RESULT_ROOT` 首次运行至少预留 10 GiB，
用于裁剪权重和实验记录；需要持久保存时改为自己的数据目录。Docker 存储还需有镜像、构建及容器运行空间。
同卡部署若在创建 rollout 通信组时出现 HCCL `EI0020` 监听端口冲突，可以给本次运行追加
`--rollout.vllm.server_hccl_if_base_port=64000 --rollout.vllm.server_hccl_npu_socket_port_range=64000-64150`。
请按本机可用端口选择，与训练进程的 HCCL 端口段错开。

启动器依次检查输入、wheel 校验和、镜像身份和设备占用，创建独立容器，安装当前 checkout 与 RL 插件，
将补丁 wheel 安装到 `/opt/hp-v4`，供训练进程和 vLLM worker 共同加载，然后裁剪权重并执行两步训练。
源码、完整模型和数据以只读方式挂载；结果保存在宿主 `RESULT_ROOT`，容器退出后保留。
启动器下载并安装 `peft==0.21.2` 和 `accelerate==1.12.0`，随后执行 `pip check`。
首次启动会进行图编译与捕获，需要等待；过程日志写入 `run/train.log`。

启动器在容器内、导入 Torch 和启动 torchrun 之前计算 CPU 线程预算，结果保存为
`run/cpu-threads.json`。它读取进程 affinity（包含 cpuset 限制）、SMT 物理核心关系及
可见 cgroup v1/v2 的最紧上级 CPU 配额。开启训练 CPU offload 时，默认取可用核心的
50%（向下取整），按本机训练进程数均分，每进程最多 8 线程、至少 1 线程。
例如 192 核、8 个训练 rank 时，每 rank 分配 8 线程，总计 64 线程。
关闭 offload 时自动使用 1 线程；显式设置的 OMP/MKL 线程数不受自动分配上限约束。
CPU offload 的判断使用与训练入口相同的 YAML 和命令行覆盖。

可用下列环境变量覆盖线程预算：

| 变量 | 含义 |
| --- | --- |
| `OMP_NUM_THREADS` | 手动指定每个训练进程的线程数；未设置或 `auto` 时自动分配 |
| `MKL_NUM_THREADS` | 显式数值优先于 OMP；启动器将两个变量统一为最终训练线程数 |
| `HYPER_RL_CPU_RESERVE` | 自动分配时预留的核心预算，默认预留一半；非负整数只可进一步增加预留，不能突破 50% 预算 |
| `HYPER_RL_VLLM_CPU_THREADS` | 独立的推理 OMP/MKL 线程数，启动器默认 1，不继承较大的训练线程数 |

例如 `OMP_NUM_THREADS=8 bash hyper_parallel/rl/examples/consistency/scripts/run_qwen3_moe_consistency_docker.sh`
可保留手动训练预算（显式设置了 MKL 时以 MKL 为准）。未使用该启动器的入口可在容器内、
torchrun 之前调用 `python -m rl.utils.cpu_threads --config CONFIG --local-world-size N`：
标准输出是最终线程数，标准错误输出包含资源计算依据。将结果设置到 OMP/MKL 后再启动训练。

这是启动时的计算线程预算，不是运行中抢占空闲 CPU 或分配互斥核集合，也不改变 NUMA 绑定。
同一启动器的共享 affinity 只除以本机进程数一次；不要在已经分别绑核的 worker 中再次均分。
共享机器应通过容器限额或手动预算限制资源。自动值不保证是内存带宽约束下的性能最优值。

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
用例 G 采用本地 GQA；CP2 由 HP 的 CP GQA 成员执行序列交换。
EP2 自然在 TP2 组内；EP4 跨两个 PCP 平面。两端 CP 不必相同。

两份配方的 PIK 使用 `num_leaves: auto`，按本次训练和推理的最大 TP 确定共享 G。
TP 必须为 2 的幂（包括 1）。若要在不同最大 TP 的独立实验间比较逐位结果，应显式固定同一个 G；
自动模式不保证不同 G 的实验互相逐位相同。

保留上文模型、数据、wheel 和 DEVICES 设置；每个用例使用独立 RESULT_ROOT，**依次**运行：

```bash
runner=hyper_parallel/rl/examples/consistency/scripts/run_qwen3_moe_consistency_docker.sh
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

# F：推理 SP，提高每轮调度的 token 预算。
CONFIG="$context" RESULT_ROOT=/tmp/hyper-rl-consistency/F bash "$runner" \
  --rollout.vllm.sequence_parallel=true --rollout.vllm.max_num_batched_tokens=512

# G：训练 CP1/FSDP2，推理仍为 PCP2。
CONFIG="$context" RESULT_ROOT=/tmp/hyper-rl-consistency/G bash "$runner" \
  --train.accelerator.cp=1 --train.accelerator.dp_shard=2
```

这些是可分别复现的验证组合，不表示一次启动会穷举全部拓扑。
F 提高调度上界，实际送入 MoE 的行数取决于 prompt、并发数和调度结果。

## EP 与图模式对照

两份训练配方均选择 HP 的 `GroupedExperts` 模块族。训练使用动态 AllToAll，推理使用 vLLM-Ascend 原生专家。TP、SP、PCP 与 DP 改变源行和专家的归属。对照 eager 与图模式时，可给上述命令增加 `--rollout.vllm.enforce_eager=true` 并更换结果目录。两种模式执行同一个专家入口。[MoE 实现](../../docs/unified_execution_guide.md#推理端的专家计算与图回放)

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
# 真实 EP 路由、不等长及空源行、图回放与训练梯度。
RL_EP_ST_DEVICES=0,1,2,3 RL_EP_ST_SIZE=4 RL_EP_ST_MODEL=/path/to/Qwen3-MoE-4L \
  pytest -q hyper_parallel/rl/tests/st/test_unified_execution.py::test_unified_expert_parallel

# CP Attention、完整测试模型的 CP1/CP2 logits 与逐参数梯度。
RL_CP_ST_DEVICES=0,1,2,3 \
  pytest -q hyper_parallel/rl/tests/st/test_context_parallel.py -k 'test_context_parallel_attention or test_context_parallel_model'

# 跨 PCP 专家域、权重映射及分页 Attention。
RL_PCP_ST_DEVICES=0,1,2,3 RL_UNIFIED_ST_MODEL=/path/to/Qwen3-MoE-4L \
  pytest -q hyper_parallel/rl/tests/st/test_context_parallel.py -k 'prefill_context_parallel_mesh or prefill_context_parallel_attention'

# 两次非零更新及发布；该 YAML 内的模型、数据和结果路径必须对当前环境有效。
RL_PCP_ST_DEVICES=0,1,2,3 \
  RL_PCP_ST_RL_CONFIG=hyper_parallel/rl/examples/consistency/configs/qwen3_moe_4l_cp_rl.yaml \
  pytest -q hyper_parallel/rl/tests/st/test_context_parallel.py::test_prefill_context_parallel_rl
```

执行范围和梯度容差见 [统一执行验收](../../docs/unified_execution_guide.md#支持范围与验收)。
这两份配置聚焦不同并行布局的前向一致性；跨机、大 EP、长上下文性能需要单独验收。
