# vLLM 在线 Rollout 故障记录与验收状态

> 状态日期：2026-08-11
> 适用版本：vLLM 0.22.1、vLLM-Ascend 0.22.1rc1、Torch/torch-npu 2.10.0
> 权威边界：当前代码、可重复脚本和真实两卡执行证据

本文记录 Hyper-RL 将训练和 vLLM rollout 接入同一 GRPO 闭环时遇到的进程初始化故障、排查证据、最终实现和
当前验收状态。固定镜像、模型、数据哈希、插件和历史 adapter 证据不在本文重复，统一见
[Qwen3.5 vLLM 兼容与实验基线](vllm_compatibility.md)。

## 状态定义

| 状态 | 含义 |
|---|---|
| implemented | 实现存在，并有契约或单元测试 |
| smoke | 受限硬件场景至少成功执行一次 |
| accepted | 固定环境中的可重复门禁成功，且结果已记录 |
| failed | 固定环境中的门禁已执行，但至少一项验收条件不满足 |
| historical | 历史 profile 曾成功，但当前 action 已变化，不能作为当前可重复门禁 |
| not run | 路径存在，但当前拓扑没有执行证据 |
| not implemented | 所需实现或安全边界尚不存在 |

## 当前结论

单独运行 vLLM 一直是正常的。故障只发生在 Trainer 已经初始化 torch-npu/HCCL 和训练模型后，再从同一 Python
进程中动态修改 `ASCEND_RT_VISIBLE_DEVICES` 并构造 vLLM library client 的路径。

分卡基线采用以下边界：

```text
torchrun Trainer process                       vLLM server process group
ASCEND_RT_VISIBLE_DEVICES=0,1                  ASCEND_RT_VISIBLE_DEVICES=1
logical npu:0 -> physical NPU0                 logical npu:0 -> physical NPU1

Actor + Reference + optimizer                  API server + EngineCore + Hyper adapter
        |                                                       |
        +------- loopback HTTP: token/control plane ------------+
        +------- stateless HCCL: weight data plane -------------+
```

Trainer 不再构造生产用 `vllm.LLM` 或 vLLM execution engine。它只持有 HTTP/HCCL 代理，并按需 import
trainer-side weight-transfer API；`vllm serve` 由全新 Python 解释器启动，在任何 torch-npu/vLLM 初始化前获得固定
推理卡环境。

当前还支持单节点强同步共卡拓扑：每个 FSDP rank 在同一物理 NPU 启动一个独立 TP1 vLLM server，rollout 阶段
FSDP 参数、梯度和 optimizer state 位于 CPU，训练阶段 vLLM 使用 level 1 sleep 释放权重和 KV cache。更新后的完整
Actor state 在每个 rank staging 到本地 NPU，rank 0 汇总 IPC handles 并 fan-out 到全部推理 DP replicas；全部 replica
完成 `finish_weight_update`、KV wake 和 resume 后才提交 policy version。

## 故障记录：Trainer 内启动 vLLM 卡在 HCCL 初始化

### 现象

最小两步 GRPO 配置能够完成以下阶段：

- torchrun world-size 1 初始化；
- Actor 在 NPU0 加载；
- Reference 在 NPU0 加载；
- optimizer 和 rollout manager 构造。

随后进程停在 vLLM EngineCore 初始化附近，典型最后一条关键日志为：

```text
world_size=1 rank=0 local_rank=0 distributed_init_method=tcp://127.0.0.1:<port> backend=hccl
```

同时观察到：

- NPU0 保持训练模型显存占用；
- NPU1 只有基础占用，长时间没有进入模型加载；
- AICore 没有持续计算；
- 没有进入第一次 rollout、backward 或 optimizer step；
- 外层 600 秒主动超时最终终止作业。

### 已确认的非根因

以下路径均独立通过，因此问题不在 Qwen3.5 adapter、普通 HCCL 或单独 vLLM 功能：

| 检查 | 结果 | 说明 |
|---|---|---|
| Hyper vLLM TP1 rollout | passed | 模型注册、权重加载、prefill/decode 正常 |
| Hyper vLLM TP2 rollout | passed | 两 rank HCCL、Hyper TP shard 和生成正常 |
| CPU safetensors refit | passed | reload、tied embedding/lm-head 和 cache reset 正常 |
| 父进程 NPU0 HCCL + spawned 子进程 NPU1 HCCL | passed | 两个独立 world-size-1 HCCL group 可以共存 |
| 子进程环境继承 | passed | 子进程看到 `ASCEND_RT_VISIBLE_DEVICES=1`、`device_count=1`、`current_device=0` |

这组证据排除了“设备损坏”“两张卡不能同时初始化 HCCL”“vLLM adapter 不能生成”和“refit 本身失败”。

### 进一步诊断

将 vLLM worker 在 HCCL 前执行的预初始化步骤加入最小诊断后，曾复现：

```text
Allocator for npu is not a DeviceAllocator
```

触发顺序是先初始化 NPU runtime，再首次加载部分 vLLM/vLLM-Ascend platform、Triton/Inductor 或 allocator 注册。
真实 GRPO 路径没有稳定抛出同一个异常，而是表现为 EngineCore 初始化阻塞；该异常是初始化顺序不安全的旁证，
不是单独宣称的唯一底层根因。

### 根因边界

原实现试图在 `VLLMGenerationEngine._ensure_client()` 中临时执行：

```python
os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "1"
client = LLM(...)
```

然后恢复父进程环境。该方案的问题是：

1. Trainer 已经初始化 torch-npu、训练 HCCL 和 NPU0 context；
2. vLLM 作为 library 使用时，API 进程、平台插件、EngineCore 和 worker 的初始化边界并不等于一个干净的 `exec`；
3. 动态环境修改只能影响之后读取环境的代码和子进程，不能回滚父进程中已建立的 accelerator/runtime 全局状态；
4. vLLM 官方 multiprocessing 文档明确将“accelerator 已初始化后再从 library 启动 multiprocessing”列为已知易失败场景；
5. Ascend plugin 还会在 worker 初始化前注册 custom op、Triton/Inductor 和 allocator，进一步放大初始化顺序差异。

因此根因不是 `spawn` 环境变量没有继承，而是生产推理 runtime 的隔离边界放错在一个已经初始化 accelerator 的
Trainer 进程内部。继续增加动态环境修改或 sleep 时间不能形成可靠修复。

### 未采用或已关闭的方向

- 仅设置 `VLLM_WORKER_MULTIPROC_METHOD=spawn`：必要但不充分，无法清理 Trainer 父进程已初始化的 runtime。
- 在 `_ensure_client()` 内临时切换 `ASCEND_RT_VISIBLE_DEVICES`：已证实不能提供完整进程隔离。
- 禁用 vLLM V1 multiprocessing：会把 EngineCore 放回已初始化的 Trainer 进程，不能实现 NPU0/NPU1 runtime 隔离。
- 直接复制 RL2 的 `multiprocessing.Process`：RL2 使用 CUDA/NCCL + SGLang，且没有固定 `spawn`；不能作为 Ascend 模板。
- 分卡使用 NPU IPC：NPU IPC 要求 sender 和 receiver 位于同一物理 NPU，不适用于 NPU0 训练、NPU1 推理。
- 首个闭环直接做共卡：会同时引入显存时分、sleep/wake 和 IPC 安全边界，无法隔离当前问题。

## 最终解决方法

### 1. 外部 server 进程

`VLLMGenerationEngine` 使用 `subprocess.Popen(..., shell=False, start_new_session=True)` 启动：

```text
python -m vllm.entrypoints.cli.main serve <model> ...
```

子进程环境在 `exec` 前完成：

- 设置 `ASCEND_RT_VISIBLE_DEVICES` 为 rollout 专用物理卡；
- 设置 `VLLM_WORKER_MULTIPROC_METHOD=spawn`；
- 设置 `VLLM_SERVER_DEV_MODE=1`，启用官方 RL 控制接口；
- 设置 `VLLM_ASCEND_ENABLE_NZ=0`，避免 RL reload 的 NZ 权重风险；
- 清除 `RANK`、`LOCAL_RANK`、`WORLD_SIZE`、`MASTER_ADDR`、`MASTER_PORT` 和 TorchElastic 身份；
- 清除 vLLM DP rank/master 环境，避免继承其他作业的拓扑。

server 只监听 `127.0.0.1`。开发控制接口不能暴露到非可信网络。

### 2. HTTP token/control plane

生成请求使用 `/v1/completions`，直接发送预分词 token IDs，并要求：

```text
return_token_ids=true
logprobs=1              # 算法需要 old log-probs 时
add_special_tokens=false
```

这样 Trainer 不需要在返回文本上重新 tokenize，避免特殊 token 和 tokenizer normalization 导致 token 序列漂移。

权重生命周期使用官方接口：

```text
GET  /get_world_size
POST /init_weight_transfer_engine
POST /pause?mode=abort&clear_cache=true
POST /start_weight_update
POST /update_weights
POST /finish_weight_update
POST /resume
```

### 3. Stateless HCCL weight plane

分卡场景使用：

```json
{"backend": "nccl"}
```

这是上游配置名称；vLLM-Ascend plugin 会把它替换为 `HCCLWeightTransferEngine`。不要写成
`{"backend": "hccl"}`。

HCCL transfer group 独立于 Trainer 默认 process group：

- Trainer 是 rank 0；
- vLLM worker 从 `rank_offset=1` 开始；
- rendezvous 使用新的 loopback port；
- HTTP init/update 请求在后台线程中等待；
- Trainer 同时执行 HCCL init/broadcast，避免两端互相等待。

### 4. Hyper strict loader 兼容

Hyper Qwen3.5 的 `load_weights()` 负责 checkpoint 名称映射、TP transform 和 tied embedding/lm-head 合成，不能改用
`model.get_parameter(name).copy_()`。vLLM-Ascend 默认 layerwise reload wrapper 与该 strict loader 不兼容，因此 plugin 仅对
`HyperQwen3_5ForCausalLM` 跳过 layerwise parameter wrapper，同时仍让 HCCL receiver 调用 Hyper `load_weights()`。

当前 packed buffer 设为完整策略大小加 128 MiB，并使用一个 buffer。原因是 strict loader 在一次调用末尾校验所有
参数；若 HCCL 将模型拆成多个 callback，单个 callback 会被误判为不完整 checkpoint。该实现是当前 0.8B
correctness-first 边界，不是大模型最终内存方案。

### 5. Stream 和版本提交顺序

vLLM-Ascend packed producer 使用自己的 NPU stream。optimizer step 返回后，Trainer 在 HCCL send 前显式执行当前 NPU
stream synchronize，确保 packing stream 不会读取尚未完成写入的参数。

只有以下步骤全部成功后才更新 `policy_version`：

```text
optimizer write complete
→ pause/cache reset
→ start weight update
→ trainer stream synchronize
→ HCCL receive/load/synchronize
→ finish
→ resume
→ policy_version commit
```

失败时 Trainer 不确认新版本，并在统一清理阶段终止整个 vLLM process group。

### 6. 共卡 FSDP + NPU IPC

共卡配置使用：

```yaml
rollout:
  engine: vllm
  vllm:
    deployment: colocated
    tensor_parallel_size: 1
train:
  accelerator:
    dp_shard: 2
    cpu_offload: true
    reshard_after_forward: true
```

`visible_devices` 由 Trainer 的 `LOCAL_RANK` 和父进程 `ASCEND_RT_VISIBLE_DEVICES` 自动映射，不能在共卡配置中
手工指定。每个 server 启用 `--enable-sleep-mode`、`backend="ipc"` 和 `weight_nz_mode=0`，并只监听 loopback。
HTTP IPC 需要 `VLLM_ALLOW_INSECURE_SERIALIZATION=1`，因此该路径不能暴露到不可信网络。

一次强同步 step 的顺序为：

```text
local rollout on every DP replica
→ sleep(level=1, mode=wait)
→ FSDP reference/actor forward + backward + optimizer
→ wake weights only + pause
→ full Actor state staging to each rank-local NPU
→ merged NPU IPC handles fan-out to every replica
→ all replicas finish
→ FSDP reshard + optimizer CPU residency check + empty_cache
→ wake KV + resume
→ policy_version commit
```

CPU-offloaded gradient norm需要 CPU 和 NPU 双 backend。若用户配置保持 `comm_backend: hccl` 或未设置，Trainer 会
解析为 `cpu:gloo,npu:hccl`；CPU scalar/state collective 走 Gloo，FSDP tensor collective 继续走 HCCL。

Checkpoint 只有在所有 rank 完成 model、optimizer、RNG 和 dataloader state 后才原子写入
`checkpoint_complete.json`。恢复前会校验该完成标记、world size 和全部 rank-local 文件；旧的半成品或没有完成标记的
目录会被拒绝，不会进行部分恢复。

## 参考实现和文件

### 当前仓库

| 文件 | 作用 |
|---|---|
| `hyper_parallel/rl/rl/roles/rollout/vllm.py` | server 启动、HTTP client、HCCL/NPU IPC refitter、sleep/wake、版本提交和进程清理 |
| `hyper_parallel/rl/rl/roles/rollout/vllm_plugin.py` | Hyper model 注册、secure reload RPC、Hyper 专用 weight-update lifecycle |
| `hyper_parallel/rl/rl/roles/rollout/vllm_qwen3_5.py` | Qwen3.5 adapter、严格权重映射、TP transform、tied weight 合成 |
| `hyper_parallel/rl/rl/trainer.py` | rollout topology 校验、训练/推理设备不重叠校验、退出清理 |
| `hyper_parallel/rl/examples/configs/local_qwen3_5_0_8b_gsm8k_vllm_smoke.yaml` | 两样本、两 response、四 token、两 step 门禁 |
| `hyper_parallel/rl/examples/configs/local_qwen3_5_0_8b_gsm8k_vllm_colocated_dp_smoke.yaml` | FSDP DP2 + rollout DP2/TP1 共卡门禁 |
| `hyper_parallel/rl/examples/scripts/run_qwen3_5_vllm_docker.sh` | 固定镜像中的 rollout/refit/GRPO 统一入口和主动超时 |
| `tests/torch/rl/vllm/validate_qwen3_5_rollout.py` | 独立 TP1/TP2 rollout 门禁 |
| `tests/torch/rl/vllm/validate_qwen3_5_refit.py` | 独立 CPU refit、cache reset 和版本门禁 |
| `hyper_parallel/rl/tests/test_architecture.py` | server 环境隔离、HTTP/生成、stream 同步和版本契约 |
| `hyper_parallel/rl/tests/test_colocated_vllm.py` | 共卡配置、rank-local server、IPC fan-out 和延迟版本提交契约 |
| `tests/ut/rl/vllm/test_plugin.py` | plugin 注册和 Hyper lifecycle patch 契约 |

### 外部参考

| 仓库或文档 | 参考点 |
|---|---|
| RL2 `RL2/trainer/ppo.py`、`RL2/workers/rollout.py` | Trainer 持有 rollout 代理，推理 server 独立运行，HTTP 控制 |
| veRL `verl/workers/rollout/vllm_rollout/vllm_async_server.py` | server actor 在构造 vLLM 前设置 Ascend 可见设备 |
| veRL `verl/checkpoint_engine/hccl_checkpoint_engine.py` | 独立 checkpoint HCCL group 和 packed transfer |
| vLLM-Ascend `examples/rl/rlhf_http_hccl.py` | 官方外部 server + HTTP + HCCL RLHF 流程 |
| vLLM-Ascend `examples/rl/rlhf_http_npu_ipc.py` | 同物理 NPU 的 IPC 约束和 insecure serialization 要求 |
| vLLM-Ascend `tests/e2e/pull_request/two_card/test_hccl_weight_transfer.py` | NPU0/NPU1 分卡官方端到端门禁 |
| [vLLM Python Multiprocessing](https://docs.vllm.ai/en/latest/design/multiprocessing.html) | `fork`/`spawn` 和 library 使用限制 |
| [vLLM Troubleshooting](https://docs.vllm.ai/en/latest/usage/troubleshooting.html#python-multiprocessing) | accelerator 已初始化后的 multiprocessing 故障边界 |

RL2 不包含 vLLM、torch-npu 或 HCCL rollout 实现。它只证明独立服务进程和 HTTP 控制面是成熟的 RL runtime 边界；
其 CUDA IPC、`torch.cuda` API 和默认 Linux `fork` 行为没有复制到 Hyper-RL。

## 当前验收矩阵

| 门禁 | 状态 | 当前证据 |
|---|---|---|
| Hyper adapter contracts | accepted | focused adapter/plugin contracts 通过；完整 pytest 本轮按要求未重跑 |
| Hyper-RL contracts | accepted | focused contracts、静态检查和真实硬件门禁通过；完整 pytest 本轮按要求未重跑 |
| 独立 TP1 rollout | accepted | Hyper/native greedy 8 tokens 完全一致，token mismatch 为 0 |
| 独立 TP2 rollout | accepted | `rollout-tp2` 生成的 `hyper-tp2.json` 报告 |
| 独立 CPU refit | accepted | Hyper/native 均发生可观测输出变化并提交 version 1，61 个 replicated norm 探针一致 |
| Trainer NPU0 + server NPU1 startup | accepted | EngineCore 记录 `visible_npus=[1]` 并完成模型加载 |
| 第一步 rollout/update/refit | accepted | rollout、backward、optimizer 和完整 HCCL lifecycle 成功 |
| 第二步使用更新后策略 rollout | accepted | 第二轮 completion 在第一次 refit 后成功，随后第二次 refit 成功 |
| server/EngineCore cleanup | accepted | 日志记录 `Shutdown complete`，无实验容器残留 |
| 在线 rollout TP2 | not run | adapter TP2 已通过，但外部 server + HCCL refit TP2 未单独验收 |
| 多训练 rank HCCL refit | not implemented | 当前配置在 `dp_shard != 1` 时 fail-fast |
| 共卡 FSDP DP2 + rollout DP2/TP1 | accepted | 两步 rollout/sleep/train/NPU IPC/wake 全部成功 |
| 共卡 vLLM sleep mode | accepted | 每个 replica 每次 sleep 释放约 31.03 GiB |
| 共卡 NPU IPC refit | accepted | 两步中全部 `/update_weights` 与 `/finish_weight_update` 返回 200 |
| 共卡推理 TP | not implemented | 当前 `deployment: colocated` 明确限制 `tensor_parallel_size=1` |
| M3 真实学习门禁 | accepted | Hyper/native DP2 两步均有 mixed reward、非零梯度、norm 探针变化和 version 1/2 |
| 20-step correctness soak | accepted | Hyper/native 均 exit 0，连续提交 version 1..20，显存峰值稳定 |
| production canary（未启用 batch-invariant） | historical | Hyper/native DP2 单步均通过，但当前 action 已切换到新 profile |
| production canary（启用 batch-invariant） | not run | 当前 profile 已启用新开关，colocated refit/sleep-wake canary 待重跑 |
| production serial/concurrent A/B | failed | batch-invariant 后 native token exact；Hyper 11/12 条 exact；两者 log-prob 门禁均失败 |
| 完整 GSM8K production 训练 | not run | 1,868-step profile 已实现，尚未执行 |

## 2026-08-10 两步实验记录

执行命令：

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1 \
HYPER_VLLM_TIMEOUT_SECONDS=600 \
./hyper_parallel/rl/examples/scripts/run_qwen3_5_vllm_docker.sh grpo-smoke
```

固定 workload：

```text
GSM8K samples        2
responses per prompt 2
max new tokens       4
training steps       2
rollout TP           1
training ranks       1
```

最后一次验证的关键结果：

| 项目 | Step 1 | Step 2 |
|---|---:|---:|
| generated tokens | 8 | 8 |
| generation seconds | 78.8643 | 3.73475 |
| tokens/second | 0.10144 | 2.14204 |
| gradient norm | 0.0151978 | 0.00650024 |
| optimizer steps | 1 | 1 |
| HCCL update | passed | passed |
| cache reset / resume | passed | passed |

Step 1 时间包含 vLLM API server、EngineCore、插件和模型冷启动；Step 2 是热 server。该小样本恰好得到全零 accuracy
reward，但 `kl_coef=0.001`，因此仍产生有限 KL loss、非零 gradient norm 和 optimizer update。它证明系统闭环，不能用来
声明 GSM8K 收敛或训练质量。

实验日志由 `HYPER_VLLM_RESULT_ROOT` 控制，默认按实现写入 `.rollout-results/hyper-grpo-smoke.log` 或
`.rollout-results/native-grpo-smoke.log`。临时目录中的日志不是长期
事实来源；可重复脚本、固定配置和本文记录共同定义验收边界。

## 2026-08-10 共卡两步实验记录

执行命令：

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1 \
HYPER_VLLM_TIMEOUT_SECONDS=900 \
./hyper_parallel/rl/examples/scripts/run_qwen3_5_vllm_docker.sh grpo-colocated-dp-smoke
```

固定拓扑为训练 FSDP DP2、推理 DP2/TP1，每个训练 rank 和一个 vLLM replica 共享同一物理 NPU。两步各完成
16 个 action token 和一次 optimizer step；训练进程最大 allocated/reserved memory 分别约 6.14/7.49 GiB。
Step 1 包含两个 server 冷启动，生成约 74.94 秒；Step 2 热生成约 0.47 秒。该小样本两组 reward 均为零，
因此 gradient norm 为零；门禁验证的是系统和权重版本闭环，不声明有效学习信号。

## 2026-08-11 M3 两步与 20-step 记录

M3 使用固定的 4 个 GSM8K prompt，每个 prompt 生成 4 个独立 seed response。训练拓扑是 FSDP DP2，两个 rank
各处理一个 prompt，因此每步共有 8 条 trajectory；20 步会循环该小数据集 10 次。该 workload 用于反复执行
rollout、sleep、GRPO update、NPU IPC refit、replicated norm 探针校验和 wake，不用于声明 GSM8K 收敛。

两步真实学习门禁结果：

| 实现 | Step 1 gradient norm | Step 2 gradient norm | policy version | 结果 |
|---|---:|---:|---|---|
| Hyper | 12.3125 | 9.0 | 1, 2 | passed |
| native | 12.0625 | 6.53125 | 1, 2 | passed |

20-step correctness soak 最终重跑结果：

| 实现 | 容器退出码 | 最终 step/version | allocated peak | reserved peak | 末步生成吞吐 |
|---|---:|---:|---:|---:|---:|
| Hyper | 0 | 20 / 20 | 6.65694 GiB | 8.52344 GiB | 25.7699 token/s |
| native | 0 | 20 / 20 | 6.65694 GiB | 8.52148 GiB | 29.4751 token/s |

soak 不要求每一步都有非零梯度。在线采样可能令一组 response 全部正确或全部错误，此时 GRPO group advantage 和
gradient 为零是预期行为；权重传输、worker norm 探针一致性、policy version 和 sleep/wake 状态机仍然逐步执行。

### 并行作业 HCCL 端口冲突

首次并行启动 Hyper 和 native soak 时，即使两项实验使用不同 NPU，Hyper 仍在第一次 FSDP all-gather 初始化时报错：

```text
Communication_Error_Bind_IP_Port
192.168.2.198/.199:16666 already bound
```

原因是 Docker bridge 和 `ASCEND_RT_VISIBLE_DEVICES` 只隔离进程与设备选择，不隔离 Ascend 物理通信网卡的监听端口。
两个 torchrun 作业都使用 HCCL 默认端口 16666，因此发生冲突。最终并行重跑使用互不重叠的范围：

```text
Hyper:  HCCL_IF_BASE_PORT=62000 HCCL_NPU_SOCKET_PORT_RANGE=62000-62100 NPU=0,1
native: HCCL_IF_BASE_PORT=62200 HCCL_NPU_SOCKET_PORT_RANGE=62200-62300 NPU=2,3
```

两项重跑均 exit 0。native 首次运行曾在 step 12 后收到外部 `SIGKILL`，Docker 状态为 `OOMKilled=false`，没有
Python、HCCL 或 NPU OOM 异常；独立端口重跑未复现并完成 20 步，因此没有将该单次现象归因于模型或 refit 实现。

### Correctness 与 production performance 的关系

二者不是两套 rollout 模块。它们共享同一个 `VLLMGenerationEngine`、Hyper/native adapter、vLLM scheduler、
PagedAttention、NPU IPC refitter 和 policy-version 状态机。区别只应由配置和受控开关表达：

- correctness profile 固定 seed、eager、functional 数值 profile、保守 KV/并发和逐步 norm 探针，用于可重复故障定位；
- production profile 提高并发请求和 `max_num_seqs`，扩大 rollout batch，并降低经验证可省略的诊断频率，用于完整
  GSM8K 训练吞吐。首个 profile 只开放请求并发、prompt batch、scheduler 容量和 PagedAttention KV；graph/NZ
  与探针降频仍需单独门禁。

保留 correctness profile 的原因是并发调度、graph/NZ 和更大的 batch 会改变请求完成顺序、数值路径和资源峰值。
这些性能开关应建立在相同实现上逐项验收，而不是覆盖唯一可复现的正确性基线或复制另一套推理代码。

当前 production profile 配置位于
`hyper_parallel/rl/examples/configs/qwen3_5_0_8b_gsm8k_vllm_production.yaml`，通过统一 launcher 的
`grpo-production` action 启动。每 rank 一次加载 2 个 prompt，并将 12 个独立 seed response 并发提交给本地 TP1
vLLM replica；`max_num_seqs=16`，KV cache 固定为 2 GiB。production profile 显式启用 vLLM-Ascend 官方
batch-invariant 算子，并跳过 text-only workload 不需要的多模态显存 profile；M3 继续显式使用
`request_concurrency=1`，且不启用 batch-invariant。

### Production 串行/并发 A/B 结果

2026-08-11 首次在未启用 batch-invariant 的 Hyper adapter 上使用 2 个 prompt、每 prompt 6 个 response、最大 300 个
新 token、2 GiB KV cache、`max_num_seqs=12` 和固定 per-request seed，对同一热 server 分别执行两轮
concurrency 1 与 12。结果为：

| 模式 | 两轮生成 token | 总耗时 | 吞吐 | 轮间 replay |
|---|---:|---:|---:|---|
| serial | 3,566 | 255.8552 s | 13.9376 token/s | token/log-prob 完全一致 |
| concurrent | 3,096 | 45.1258 s | 68.6081 token/s | token/log-prob 完全一致 |

并发吞吐提升为 4.9225 倍，scheduler 日志达到 `Running: 12 reqs`、`Waiting: 0 reqs`。但 serial 与 concurrent
产生了稳定且不同的 sampled trajectory：12 条 response 中只有 3 条 token 序列完全一致，共有 1,338 个 token
mismatch。在首个 token 分歧前的 553 个可比位置上，sampled-token log-prob 最大/平均绝对差分别为
0.230051/0.011223；分歧后的 log-prob 来自不同上下文，不纳入数值比较。

客户端 seed 和顺序映射没有变化：第 `row` 条 singleton HTTP 请求始终使用 `base_seed + row`，并发 futures 按提交
顺序收集。serial 和 concurrent 各自两轮都完全复现，说明结果不是 future 完成顺序或 seed 置换导致。并发调度改变
vLLM packed token batch，BF16 GEMM、PagedAttention 和 GDN kernel 的 shape/累加路径随之变化；较小的 logits 差异可
跨过 sampling CDF 边界，之后被自回归上下文放大。因此本次 exact sampled-trajectory 门禁记为 failed，不能用 4.92
倍吞吐结果声明 production A/B 已整体验收。

`production-benchmark` 会在 `HYPER_VLLM_RESULT_ROOT` 下生成 `hyper-production-benchmark.json` 和同名日志，保存
每轮完整 token IDs、sampled-token log-probs、digest、轮间 replay 和 common-prefix 数值差异，并在写完报告后以
非零状态拒绝未通过的门禁。原始报告属于本地运行产物，不提交仓库；本文记录可复现的结果数值。

随后启用 vLLM-Ascend 官方 `VLLM_BATCH_INVARIANT=1`，保持其余 workload 和 exact token、0.05 log-prob、1.2 倍
吞吐门限不变，分别运行 Hyper 和 native 控制实验：

| 实现 | serial 吞吐 | concurrent 吞吐 | speedup | exact response | token mismatch | log-prob 最大/平均差 |
|---|---:|---:|---:|---:|---:|---:|
| Hyper | 12.6529 | 87.2491 | 6.8956 倍 | 11/12 | 106 | 0.118699 / 0.000340 |
| native | 13.5920 | 85.9186 | 6.3213 倍 | 12/12 | 0 | 0.232431 / 0.000603 |

两种实现的 serial 和 concurrent 模式内部两轮 replay 都是 token/log-prob 完全一致。native 控制实验表明官方
batch-invariant 已使 12 条 sampled token trajectory 在串行与并发间完全一致，但 sampled-token log-prob 仍不是逐位
不变，因此 0.05 log-prob 综合门禁失败。Hyper 的 mismatch 从 1,338 降至 106，且只剩 response 0 在第 28 个 token
首次分岔；其余 11 条 token 序列完全一致。Hyper common-prefix log-prob 平均差也从 0.011223 降至 0.000340。

这组控制实验支持“batch shape 是主要误差来源”，但不能仅凭 native token exact 就将 Hyper 剩余差异唯一归因于
Hyper Attention/GDN。native 本身仍有最高 0.232431 的 log-prob 漂移，而采样是否跨过 CDF 边界还取决于具体 logits
分布；同类漂移在 native 没有改变 token，在 Hyper response 0 上改变了 token。按当前要求，本轮不启用历史 parity GDN
兼容模式，继续保留 functional Hyper adapter 路径。

batch-invariant 实验分别生成 `hyper-production-benchmark.json` 和 `native-production-benchmark.json`；日志均记录官方
`Enabling batch-invariant mode for vLLM on Ascend NPU` 和正常 EngineCore shutdown。原始报告由各自的
`HYPER_VLLM_RESULT_ROOT` 保存，不提交仓库。

## 快速排查顺序

再次遇到 vLLM 启动、HCCL 或 refit 阻塞时，按以下顺序排查，不要先修改模型代码：

1. 核对 `vllm_compatibility.md` 中的镜像 digest、Torch、torch-npu、vLLM 和 vLLM-Ascend 版本。
2. 先运行 `rollout-tp1`。如果失败，问题在 adapter/plugin/模型加载，不要进入 GRPO 联调。
3. 再运行 `refit`。如果失败，检查权重名称、tied weight、reload RPC 和 cache reset。
4. 运行 `grpo-smoke`，确认 server 日志出现 `Resolved architecture: HyperQwen3_5ForCausalLM`。
5. 确认 EngineCore 日志出现 `visible_npus=[1]`；若是 NPU0，检查子进程启动前的 `visible_devices` 和父环境。
6. 确认 server 配置打印 `WeightTransferConfig(backend='nccl')`，worker 打印创建 `HCCLWeightTransferEngine`。
7. 确认 `/get_world_size`、`/init_weight_transfer_engine`、`/pause`、`/update_weights` 均返回 200。
8. 若独立 rollout 正常、GRPO 仍在 EngineCore init 阶段阻塞，检查是否退回了进程内 `LLM(...)` 或动态环境切换。
9. 若 HCCL init 阻塞，检查 server 是否继承 `RANK/WORLD_SIZE/MASTER_*`，并确认 transfer rendezvous 使用新端口。
10. 若 update 阻塞，检查 HTTP receive 是否与 Trainer broadcast 并发启动，且 names/dtypes/shapes/packed buffer 完全一致。
11. 若 reload 报 incomplete checkpoint，检查完整策略是否装入同一 packed callback，以及 Hyper lifecycle patch 是否加载。
12. 若下一轮疑似使用旧权重，确认 HCCL send 前执行 Trainer current-stream synchronize，且成功后才推进版本。
13. 若退出后仍有 EngineCore，检查 `start_new_session=True` 和 process-group `SIGTERM/SIGKILL` 清理路径。
14. 如果通过 `tee` 写日志时报权限错误，将 `HYPER_VLLM_RESULT_ROOT` 指向宿主用户可写目录。

## 已知非阻塞告警

固定镜像中可能出现以下日志，当前成功门禁也会出现：

- `Unknown vLLM environment variable: VLLM_ASCEND_ENABLE_NZ`：上游 vLLM env scanner 不认识 Ascend 扩展变量，但
  vLLM-Ascend 0.22.1rc1 会读取该变量；成功日志会显示 `weight_nz_mode ... value 0`。
- `Qwen2VLImageProcessorFast is deprecated`：Transformers 导入告警，当前 text-only rollout 不受影响。
- `barrier(): using the device under current context`：world-size 1 清理阶段告警，不影响本次结果。
- `resource_tracker ... leaked semaphore`：vLLM multiprocessing 退出告警；当前 EngineCore 和容器均已正常退出。若出现
  持续子进程或显存占用，则不能按该已知告警处理。

## 当前限制和下一步

当前已完成 correctness-first 分卡和单节点共卡闭环，不是完整生产 rollout 集群：

1. 在线 HCCL refitter 只支持一个训练 rank；多 rank 必须设计 Trainer/TP/DP 到 inference ranks 的明确 mapping。
2. 完整策略单 buffer 会增加约一个模型大小的临时 NPU 内存；大模型需要让 strict loader 支持跨 bucket 增量校验。
3. PyHCCL broadcast 是阻塞调用；production action 提供 72 小时容器内总 watchdog，但仍需更细粒度进度检测。
4. 开发控制接口依赖 `VLLM_SERVER_DEV_MODE=1`，目前只允许 loopback；生产部署需要认证或独立可信控制网络。
5. 共卡路径当前是一训练 rank 对应一 TP1 vLLM replica；推理 TP 需要新增 FSDP full state 到 TP shard 的在线映射。
6. 共卡仅支持单节点；多节点 endpoint、host UUID 和跨节点 DP 控制尚未实现。
7. NPU IPC update 没有 server-side rollback；任一 replica 失败后整个 rollout process set 会被关闭，尚未实现自动重启。
8. 在线 TP2、多节点、完整 production 训练和故障注入仍未执行；Hyper/native production A/B 已执行，但完整
   token/log-prob 综合门禁仍失败。
