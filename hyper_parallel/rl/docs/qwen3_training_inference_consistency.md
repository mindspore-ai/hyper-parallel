# Qwen3 训练-推理一致性

## 声明范围

一致性 profile 的目标组合为（当前 master 的实际验收见下文最新记录）：

- Trainer：Transformers Qwen3 + `models/qwen3` 的 `Qwen3AutoModel` / HyperParallel，TP1 FSDP、pure TP2 或 `FSDP-shard2×TP2`；
- Rollout：[`HyperQwen3ForCausalLM`](../rl/roles/rollout/models/qwen3/model.py) + vLLM/vLLM-Ascend，TP degree 与 Trainer 相同；
- Deployment：colocated NPU IPC 或 disjoint HCCL；
- 平台：单节点 Ascend NPU、BF16、eager；
- 比较时点：使用同一 policy version，在每次 optimizer update 前执行 Actor replay。

该组合建立在训推一体模型 contract 上：Trainer 与 Hyper-vLLM 共享 Transformers Qwen3 参数语义和 HyperParallel TP
sharding plan；训练侧使用可反向传播的 attention 路径，推理侧使用 vLLM paged attention/KV cache。Bit-exact 门禁验证这两个
执行边界在固定 Ascend recipe 下产生相同的 selected-token raw logprobs。

该结论不覆盖 Native-vLLM、其他模型、TP4/TP8、backward、gradient、optimizer state、更新后参数或收敛表现。

## Bit-Exact 定义

Rollout 使用 policy V 生成 response token IDs，并保存逐 token raw logprobs。Trainer 在 optimizer update 前使用相同权重、
token、mask 和概率定义执行 forward：

```text
raw logits
-> FP32
-> log_softmax(dim=-1)
-> gather by authoritative response token ID
```

Comparator 只比较 `loss_action_mask` 选中的 tokens，并要求：

```text
dtype = FP32
shape exactly equal
valid token count > 0
all values finite
policy version equal
int32 bit-pattern mismatch_count = 0
max_abs_diff = 0
mean_abs_diff = 0
```

FP32 tensor 按 `int32` bit pattern 逐元素比较，不使用近似容差。失败必须让所有 Trainer ranks 同步退出，不执行 optimizer
update，也不发布下一 policy version。

禁止复制 rollout logprobs、缩小 mask、截断长度、忽略异常 token、decode/re-encode、阻断梯度或永久关闭生产 cache 功能。

## Token 与 Mask

- Rollout token IDs 是 Trainer 输入的唯一权威来源。
- `response_mask` 只选择实际生成的 response token，包括策略生成的 EOS。
- Prompt、padding、环境内容和 EOS 后 token 不参与比较或 policy loss。
- 两侧使用相同 next-token shift：位置 `i` 的 logits 预测 token `i + 1`。
- Token、attention mask、action mask 和 rollout logprobs 必须保持 request boundary、顺序和长度一致。

## Consistency Profile

```yaml
consistency:
  enabled: true
```

默认配置为 `false`。开启后只允许 Qwen3 Hyper-vLLM，并要求 Trainer/rollout matched TP；普通模式不受该限制。Profile 是
进程级安装，同一进程启用后不能切回 off。

| 侧 | Profile 行为 |
| --- | --- |
| Trainer | BF16 forward、FP32 reduce、packed THD、trainable FA2 varlen、batch-invariant ops、NPU RMSNorm |
| Hyper-vLLM | Qwen3 Hyper adapter、FA3 KV-cache attention、batch-invariant、raw logprobs |
| Scheduler | 保留 Prefix Cache/Chunked Prefill，并恢复 discarded partial-prefill 的 RNG offset |
| 共同设置 | 相同 checkpoint/tokenizer、selected-token FP32 log-softmax 和 deterministic collectives |

Trainer 的 FA2 backward 路径与 rollout 的 FA3 KV-cache 路径不是同一个算子，但它们是固定 recipe 中成对验收的训练/decode
实现。固定依赖已安装在运行镜像中：

| Package | Version |
| --- | --- |
| Transformers | 5.5.4 |
| vLLM | 0.22.1 |
| vLLM-Ascend | 0.22.1rc1 |
| batch-invariant-ops | 1.0.0 |
| flash-attn-npu | 0.2.0b1 |

缺少依赖、版本不符、非 eager、非 Qwen3、非 Hyper-vLLM 或 TP degree 不匹配时 fail closed。

## 运行

TP1 默认使用两张 NPU：

```bash
export HYPER_QWEN3_MODEL_ROOT=/absolute/path/to/Qwen3-4B
export HYPER_QWEN3_DATA_ROOT=/absolute/path/to/gsm8k
export HYPER_QWEN3_RESULT_ROOT=/home/mwl/project/hyper/hyper-parallel-master/hyper_parallel/rl/output/qwen3-consistency-smoke
export HYPER_QWEN3_VISIBLE_DEVICES=0,1

./hyper_parallel/rl/examples/scripts/run_qwen3_consistency_docker.sh
```

Colocated TP2 使用四张 NPU：

```bash
export HYPER_QWEN3_VISIBLE_DEVICES=0,1,2,3
export HYPER_QWEN3_TP=2

./hyper_parallel/rl/examples/scripts/run_qwen3_consistency_docker.sh colocated
```

默认 TP2 recipe 使用 full-gather。验证 direct：

```bash
export HYPER_QWEN3_WEIGHT_SYNC_STRATEGY=direct_reshard

./hyper_parallel/rl/examples/scripts/run_qwen3_consistency_docker.sh colocated
```

Disjoint TP2 使用同一 launcher、YAML 和 consistency recipe。完整
`FSDP-shard2×TP2→DP2×TP2` 拓扑需要 8 张 NPU：

```bash
export HYPER_QWEN3_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HYPER_QWEN3_TP=2
export HYPER_QWEN3_TRAINER_COUNT=4
export HYPER_QWEN3_ROLLOUT_DP=2

./hyper_parallel/rl/examples/scripts/run_qwen3_consistency_docker.sh disjoint
```

正式配置位于 `examples/configs/qwen3_4b_gsm8k_vllm_tp2_consistency.yaml`。普通 Hyper/Native 或 TP degree mismatch 使用
`run_qwen3_tp_docker.sh`；该入口显式关闭 consistency，不能用于声明 bit-exact。

## 显式归约修复后的验证（2026-09-13）

目录重组工作区及未修改的 `d3cd261f` 基线均在首次 Actor 更新遇到
`numeric_mask.sum()` 缺少 batch-invariant 算子必需参数 `dim` 的错误。
Actor/Critic 的有效 token 计数已恢复旧 RL 实现的 `flatten().sum(dim=0).detach()`；
不修改 mask、奖励、attention 或 logprob 计算，也不增加全局算子补丁。
CPU 回归增加显式维度约束，覆盖 GRPO/PPO Actor 和 Critic。

固定 Hyper-RL 镜像（image ID `f13967683d44`）、真实 Qwen3-4B/GSM8K，
四卡 colocated：Trainer FSDP2×TP2，Hyper-vLLM DP2×TP2，BF16 eager、CPU offload、
完整 activation checkpoint，Prefix Cache/Chunked Prefill 保持开启。
每个 DP rank 每批 2 个 prompt，每个 prompt 4 个 response，生成上限 512，学习率 `1e-6`。

| 同步方式 | step | 有效比较 token | mismatch / max abs / mean abs | 梯度范数 | 发布版本 |
| --- | ---: | ---: | --- | ---: | ---: |
| full-gather | 1 | 7296 | 0 / 0 / 0 | 0.984375 | 1 |
| full-gather | 2 | 7714 | 0 / 0 / 0 | 0.000499725 | 2 |
| direct-reshard | 1 | 7296 | 0 / 0 / 0 | 0.984375 | 1 |
| direct-reshard | 2 | 7663 | 0 / 0 / 0 | 0.00044632 | 2 |

full-gather 每步 73 个 bucket 全部 ACK/release；direct 两步均确认实际策略为 direct-reshard。
更新后对旧策略的 bit-pattern 差异分别为 full-gather 2988/3946、direct 2991/4123，负对照有效。
相关 Docker CPU 回归 52 passed，实验结束后容器和 NPU 进程已清理。
该验收仅覆盖上述 GRPO 四卡 matched TP2 组合，不扩展到 PPO bit-exact、TP1、disjoint 或恢复续训。

## master 隔离版本历史验证（2026-09-12）

使用固定镜像 `swr.cn-east-3.myhuaweicloud.com/huawei-hyper-rl/hyper-rl:v0.22.1rc1-arm64`，
挂载当前 master 源码及本地 Qwen3-4B / GSM8K，并安装当前 RL 的 vLLM 注册入口。
四卡 colocated：Trainer FSDP-shard2×TP2，Hyper-vLLM DP2×TP2，BF16 eager，
完整 activation checkpoint，CPU offload，学习率 `1e-6`。
测试派生配置启用 `consistency.enabled=true`、batch-invariant、prefix cache 和 chunked prefill；
每个 DP rank 每批 2 个 prompt，每个 prompt 4 个 response，最多生成 512 token，运行两步。
没有关闭 cache、缩小比较 mask、使用误差容忍或复制 rollout logprob。

| 同步方式 | step | 有效比较 token | mismatch / max abs / mean abs | 梯度范数 | 已发布版本 |
| --- | ---: | ---: | --- | ---: | ---: |
| full-gather | 1 | 7296 | 0 / 0 / 0 | 0.984375 | 1 |
| full-gather | 2 | 7683 | 0 / 0 / 0 | 0.000453949 | 2 |
| direct-reshard | 1 | 7296 | 0 / 0 / 0 | 0.984375 | 1 |
| direct-reshard | 2 | 7688 | 0 / 0 / 0 | 0.000576019 | 2 |

full-gather 结果目录：`rl_tests/st/results/dense-tp2-consistency-full-80ee17d7a5/`，
内含开启一致性的 `phase-1.yaml`、完整 `phase-1.log` 与通过状态的 `result.json`。
每步 73 个 bucket 全部 ACK/release。更新后相对旧策略分别出现 2964 / 3810 个
logprob bit-pattern 差异，负对照有效；第二步更新前再次归零验证了 V1 权重同步后的对应关系。

direct-reshard 结果目录：`rl_tests/st/results/dense-tp2-consistency-direct-0f9f0582a5/`，
同样保存测试 YAML、日志和 passed 报告。两步均确认实际完成的同步策略为 direct-reshard；
更新后相对旧策略的 logprob bit-pattern 差异分别为 2963 / 3844，负对照有效。
两种方式均通过完整两步验收，测试容器正常清理，结束时 NPU 均无运行进程。

此前排除的一致性 CPU 单测另有 10 passed（宿主 CPU 环境），不能代替上述 Docker/NPU 结果。
当前结果不代表 TP1、八卡 disjoint、checkpoint resume 或 agentic 的一致性均已验收，
也不代表此前的 optimizer 恢复问题已修复。

后续 PPO 接入已将标准构建适配移入 `models/qwen3`。此次 GRPO 四卡回归目录为
`/tmp/hyper-ppo-st/grpo-tp2-consistency-regression-6de951f437`，两步分别比较 7296 / 7674
个有效 token，三项误差均为 0，完成非零更新和 V1/V2 发布。
optimizer 恢复的局部修复及双角色真机恢复结果见 [PPO](ppo.md)；该回归不代表 PPO bit-exact 已验收。

## 迁移来源历史验证矩阵

下表保留迁移来源的既有记录，不作为当前 master 隔离适配的通过依据。
Full-gather 在 2026-09-11 改为完整参数 packed 传输；当前版本的复验以本页上一节为准，
不能沿用旧 fragment full-gather 的结论。

| Deployment/拓扑 | 验证内容 | 结果 |
| --- | --- | --- |
| Colocated TP1 FSDP → Hyper-vLLM DP×TP1 | Direct 两步 generation/update/publication | 所有有效 token `0/0/0` |
| Colocated `FSDP-shard2×TP2→DP2×TP2` | Direct、scheduler/cache 和 DCP resume | 所有有效 token `0/0/0` |
| Disjoint `pure TP2→DP1×TP2` | Direct、LR matrix 和 DCP resume | 所有有效 token `0/0/0` |
| Disjoint `FSDP-shard2×TP2→DP2×TP2` | 8 卡 direct 两步真实更新 | 所有有效 token `0/0/0` |
| Colocated/disjoint packed full-gather | 完整参数合桶、IPC/HCCL、worker `load_weights()` | CPU 合同通过；NPU bit-exact 待重跑 |
| Disjoint 同步失败 | 错误在所有 Trainer rank 传播并终止运行 | 无版本发布、无继续训练 |
| Disjoint Prefix Cache/Chunked Prefill | Long/short 混排、partial prefill 和稳定顺序 | 两步 `0/0/0` |
| Disjoint DCP destroy/resume/publication | 四 rank checkpoint，fresh process 在 generation 前发布恢复权重 | `0/0/0` |

`0/0/0` 依次表示 mismatch count、max absolute diff 和 mean absolute diff。在线同步以 worker 提交的整数版本作为运行时边界；设备侧权重内容正确性由 direct/full-gather 集成测试分别验证。

## 修改门禁

修改以下任一范围后必须重跑对应 matched TP 门禁：

- tokenizer、token/mask 或 packing；
- attention、RMSNorm、LM head 或 log-softmax；
- request grouping、scheduler、Prefix Cache 或 Chunked Prefill；
- weight publication、cache lifecycle、checkpoint resume；
- Trainer/rollout parallel layout。

TP2 主门禁包括四卡 colocated 和八卡 disjoint。多节点、Native-vLLM bit-exact 和 TP4/TP8 尚未验收。
