# Qwen3 训练-推理一致性

## 声明范围

当前 bit-exact 组合为：

- Trainer：Transformers Qwen3 + HyperAutoModel/HyperParallel，TP1 FSDP、pure TP2 或 `FSDP-shard2×TP2`；
- Rollout：`HyperQwen3ForCausalLM` + vLLM/vLLM-Ascend，TP degree 与 Trainer 相同；
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
policy version/fingerprint equal
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
export HYPER_QWEN3_RESULT_ROOT=/absolute/path/to/results
export HYPER_QWEN3_VISIBLE_DEVICES=0,1

./hyper_parallel/rl/examples/scripts/run_qwen3_consistency_docker.sh
```

Colocated TP2 使用四张 NPU：

```bash
export HYPER_QWEN3_VISIBLE_DEVICES=0,1,2,3
export HYPER_QWEN3_TP=2

./hyper_parallel/rl/examples/scripts/run_qwen3_consistency_docker.sh colocated
```

默认 TP2 recipe 使用 full-gather。验证 direct 与 fallback：

```bash
export HYPER_QWEN3_WEIGHT_SYNC_STRATEGY=direct_reshard
export HYPER_QWEN3_WEIGHT_SYNC_FALLBACK=full_gather

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

## 已验证矩阵

| Deployment/拓扑 | 验证内容 | 结果 |
| --- | --- | --- |
| Colocated TP1 FSDP → Hyper-vLLM DP×TP1 | 两步 generation/update/refit | 所有有效 token `0/0/0` |
| Colocated `FSDP-shard2×TP2→DP2×TP2` | Full/direct、scheduler/cache 和 DCP resume | 所有有效 token `0/0/0` |
| Disjoint `pure TP2→DP1×TP2` | Full/direct/fallback、LR matrix 和 DCP resume | 所有有效 token `0/0/0` |
| Disjoint `FSDP-shard2×TP2→DP2×TP2` | 8 卡 full/direct 两步真实更新 | 所有有效 token `0/0/0` |
| Disjoint direct receive failure → full | Abort、全量覆盖、cache reset 和两步 replay | `0/0/0`，版本原子发布 |
| Disjoint direct 与 fallback 双失败 | 四 rank 同错、admission 保持关闭 | 无版本发布、无 resume |
| Disjoint Prefix Cache/Chunked Prefill | Long/short 混排、partial prefill 和稳定顺序 | 两步 `0/0/0` |
| Disjoint DCP destroy/resume/refit | 四 rank checkpoint，fresh process 在 generation 前 refit | `0/0/0` |

`0/0/0` 依次表示 mismatch count、max absolute diff 和 mean absolute diff。完整参数发布由每个 worker 的 source-derived
manifest 验证，不以 logprob 相等代替 weight correctness。

## 修改门禁

修改以下任一范围后必须重跑对应 matched TP 门禁：

- tokenizer、token/mask 或 packing；
- attention、RMSNorm、LM head 或 log-softmax；
- request grouping、scheduler、Prefix Cache 或 Chunked Prefill；
- weight publication、cache lifecycle、checkpoint resume；
- Trainer/rollout parallel layout。

TP2 主门禁包括四卡 colocated 和八卡 disjoint。多节点、Native-vLLM bit-exact 和 TP4/TP8 尚未验收。
