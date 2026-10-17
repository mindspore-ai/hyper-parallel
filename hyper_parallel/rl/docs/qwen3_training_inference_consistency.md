# Qwen3 训练-推理一致性

## 声明范围

一致性检查面向以下组合：

- Trainer：Transformers Qwen3 + `models/qwen3` 的 `Qwen3AutoModel` / HyperParallel，TP1 FSDP、pure TP2 或 `FSDP-shard2×TP2`；
- Rollout：[`HyperQwen3ForCausalLM`](../rl/roles/rollout/consistency_models/qwen3/model.py) + vLLM/vLLM-Ascend，TP degree 与 Trainer 相同；
- Deployment：colocated NPU IPC 或 disjoint HCCL；
- 平台：单节点 Ascend NPU、BF16、eager；
- 比较时点：使用同一 policy version，在每次 optimizer update 前执行 Actor replay。

该组合建立在训推一体模型 contract 上：Trainer 与 Hyper-vLLM 共享 Transformers Qwen3 参数语义和 HyperParallel TP
sharding plan；训练侧使用可反向传播的 attention 路径，推理侧使用 vLLM paged attention/KV cache。Bit-exact 门禁验证这两个
执行边界在固定 Ascend recipe 下产生相同的 selected-token raw logprobs。

检查仅覆盖更新前的 selected-token raw logprobs，不覆盖 Native-vLLM、其他模型、TP4/TP8、
梯度、optimizer state、更新后参数或收敛表现。各拓扑的验收以对应[系统测试](../README.md#系统测试)为准。

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

| 侧         | Profile 行为                                                                                  |
| ---------- | --------------------------------------------------------------------------------------------- |
| Trainer    | BF16 forward、FP32 reduce、packed THD、trainable FA2 varlen、batch-invariant ops、NPU RMSNorm |
| Hyper-vLLM | Qwen3 model adapter、FA3 KV-cache attention、batch-invariant、raw logprobs                    |
| Scheduler  | 保留 Prefix Cache/Chunked Prefill，并恢复 discarded partial-prefill 的 RNG offset             |
| 共同设置   | 相同 checkpoint/tokenizer、selected-token FP32 log-softmax 和 deterministic collectives       |

训练侧 FA2 backward 与推理侧 FA3 KV-cache 是不同的执行路径，需要在固定配置下成对验证。
固定依赖与镜像下载见[运行镜像](../docker/README.md)。

缺少依赖、版本不符、非 eager、非 Qwen3、非 Hyper-vLLM 或 TP degree 不匹配时 fail closed。

### 0.23 概率归一化配方

运行时自动选择固定版本配方；原有 0.22.1 / 0.22.1rc1 配方保持原合同。
`qwen3_ascend_consistency_v023` 要求 vLLM 0.23.0、vLLM-Ascend 0.23.0.post1、
Torch 2.10.0、torch-npu 2.10.0.post4、Transformers 5.5.4、
flash-attn-npu 0.2.0b1 和 batch-invariant-ops 1.0.0；其他版本组合拒绝安装。

0.23 配方让 Trainer 与 rollout 对原始二维 logits 使用显式 FP32 输出 log-softmax。
Trainer 的非持久 RoPE 频率在 CPU 上重建后传到目标设备，与 Hyper-vLLM 的初始化保持相同；
避免设备上的 FP32 幂运算微小差异在较大位置改变 BF16 cos/sin。
此外，默认 `aten::sum` 重载桥接到批次无关轴归约，保留 mask 的整数类型和损失梯度。
这些处理不会复制采样概率，也不会改变有效 token mask 或放宽 bit-exact 门禁。
实际支持的拓扑和长度仍须由真实系统测试证明。

Checkpoint 保存 `training_consistency_profile`，恢复时要求相同配方；不能直接将 off 配方的优化器状态
载入一致性模型，因为两者的参数结构可能不同。

## 运行

TP1 默认使用两张 NPU：

```bash
export HYPER_QWEN3_MODEL_ROOT=/absolute/path/to/Qwen3-4B
export HYPER_QWEN3_DATA_ROOT=/absolute/path/to/gsm8k
export HYPER_QWEN3_RESULT_ROOT=/absolute/path/to/results/qwen3-consistency
export HYPER_QWEN3_VISIBLE_DEVICES=0,1

./hyper_parallel/rl/examples/gsm8k/scripts/run_qwen3_consistency_docker.sh
```

Colocated TP2 使用四张 NPU：

```bash
export HYPER_QWEN3_VISIBLE_DEVICES=0,1,2,3
export HYPER_QWEN3_TP=2

./hyper_parallel/rl/examples/gsm8k/scripts/run_qwen3_consistency_docker.sh colocated
```

默认 TP2 recipe 使用 full-gather。验证 direct：

```bash
export HYPER_QWEN3_WEIGHT_SYNC_STRATEGY=direct_reshard

./hyper_parallel/rl/examples/gsm8k/scripts/run_qwen3_consistency_docker.sh colocated
```

Disjoint TP2 使用同一 launcher、YAML 和 consistency recipe。完整
`FSDP-shard2×TP2→DP2×TP2` 拓扑需要 8 张 NPU：

```bash
export HYPER_QWEN3_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HYPER_QWEN3_TP=2
export HYPER_QWEN3_TRAINER_COUNT=4
export HYPER_QWEN3_ROLLOUT_DP=2

./hyper_parallel/rl/examples/gsm8k/scripts/run_qwen3_consistency_docker.sh disjoint
```

TP1/TP2 共用 [基础 YAML](../examples/gsm8k/configs/qwen3_4b_gsm8k_vllm_production.yaml)，
上述一致性脚本根据 `HYPER_QWEN3_TP` 覆盖对应的拓扑、容量和预算，保留原 TP2 配方的行为。
普通 Hyper/Native 或 TP degree mismatch 使用 `run_qwen3_tp_docker.sh`；该入口显式关闭 consistency，
不能用于声明 bit-exact。
