# Qwen3 训练-推理一致性

## 范围与定义

当前一致性能力只覆盖以下组合：

- Trainer：Transformers Qwen3 + HyperModels/HyperParallel FSDP Actor；
- Rollout：`HyperQwen3ForCausalLM` + vLLM/vLLM-Ascend；
- 平台：Torch NPU、BF16、eager；
- 拓扑：单节点 FSDP，多个 colocated Hyper-vLLM TP1 replicas。

本文中的一致性是指：rollout 使用 policy version `V` 生成 response token 并保存逐 token raw logprobs，Trainer 在该批
数据第一次 optimizer update 前，使用相同权重、token 和概率定义重新执行真实 Actor forward；response mask 内每个 FP32
logprob 的 bit pattern 必须完全相同。

该合同不要求 backward、梯度、optimizer state 或更新后的参数 bit-exact，也不要求 Native-vLLM 与 Hyper-vLLM 生成相同
token。Native-vLLM 和普通 Hyper-vLLM 只作为诊断与性能基线。

## 严格合同

### Token 与 Mask

- Rollout 原始 token IDs 是 Trainer 输入的唯一权威来源，禁止 decode/re-encode。
- `response_mask` 只选择策略实际生成的 response token，包括策略生成的 EOS。
- Prompt、padding、环境内容和 EOS 后 token 不参与比较或 policy loss。
- Token、attention mask、action mask 和 rollout logprobs 必须保持请求边界、顺序和长度一致。
- `action_mask` 必须是 `attention_mask` 的子集，任何 padding 被选为 action 都立即失败。
- Trainer 与 rollout 必须使用相同的 next-token shift：位置 `i` 的 logits 预测 token `i + 1`。

### Logprob 定义

两侧比较同一种 raw policy distribution：

```text
raw logits
→ 转为 FP32
→ log_softmax(dim=-1)
→ 按 rollout 生成的 token ID gather
```

不比较经过 temperature、top-k、top-p、penalty 或 grammar processor 修改后的 sampling distribution。当前 profile 固定
temperature `1.0`、top-p `1.0`、top-k `0`。

### Bit-Exact 门禁

Optimizer 前 comparator 必须满足：

```text
dtype = FP32
shape 完全一致
有效 token 数 > 0
所有值 finite
worker policy version 完全一致
worker fingerprint 完全一致
mismatch_count = 0
max_abs_diff = 0.0
mean_abs_diff = 0.0
```

Comparator 将 FP32 tensor 视为 `int32` bit pattern 后逐元素比较。发生错误时记录 rank、trajectory、token ID、response
offset、两侧浮点值和 bit pattern，并让所有 DP ranks 同步失败。门禁失败时不得执行 optimizer update 或发布下一 version。

禁止复制 rollout logprobs、缩小 mask、按较短长度截断、忽略异常 token、阻断 Trainer 梯度或静默改用近似容差。

## 版本化 Consistency Profile

配置开关为：

```yaml
consistency:
  profile: qwen3_ascend_fa3_batch_invariant_v1
```

通用配置默认 `off`，不会导入可选 NPU kernel 或修改普通 Trainer/rollout。Qwen3 production recipe 显式启用已验证
profile。Profile 是进程级不可逆安装，同一进程不能在启用后切回 `off`。

| 侧         | Profile 行为                                                                                                               |
| ---------- | -------------------------------------------------------------------------------------------------------------------------- |
| Trainer    | BF16 参数/forward、FP32 reduce、右 padding 转 packed THD、trainable FA2 varlen、batch-invariant ATen ops、`npu_rms_norm` |
| Hyper-vLLM | `HyperQwen3ForCausalLM`、FA3 KV-cache attention、`VLLM_BATCH_INVARIANT=1`、block size 128、raw logprobs                |
| Scheduler  | 保留 Prefix Cache 与 Chunked Prefill；恢复被丢弃 partial-prefill 请求的真实 RNG offset                                     |
| 共同设置   | 相同 checkpoint/tokenizer、Qwen3 语义、FP32 selected-token log-softmax 和 deterministic HCCL/LCCL                          |

Trainer 使用可 backward 的 FA2/varlen，rollout 使用带 KV cache 的 FA3；两者不是同一个 Python 函数，而是同一
profile 中成对验证的训练和 decode 实现。Batch-invariant mode 同时覆盖 linear/matmul、reduction、RMSNorm、activation、
LM head 和 log-softmax，不能只替换 attention。

Profile 固定依赖：

| Package             | Version   |
| ------------------- | --------- |
| Transformers        | 5.5.4     |
| vLLM                | 0.22.1    |
| vLLM-Ascend         | 0.22.1rc1 |
| batch-invariant-ops | 1.0.0     |
| flash-attn-npu      | 0.2.0b1   |

`flash-attn-npu` wheel 使用 source commit `c7528a4bdb0a33f21e181cd25108c1e60c11d061`，验证 SHA256 为：

```text
9f58e114b77f72079111e2f86fa9750d3be39d1ec9324b309588a540a3e9e12b
```

缺少固定依赖、版本不符、非 Qwen3、非 Torch NPU、非 Hyper-vLLM、非 TP1 或非 eager 时 fail closed，不允许静默
fallback。两张 NPU 的可复现 smoke 命令见 [README](../README.md#smokeqwen3-训推一致性)。

## Batch、Decode 与 Cache

同一请求的有效 token logprobs 不应因以下因素改变：

- batch size、请求顺序和 DP replica 分配；
- padding 长度、packing 布局和 `max_num_seqs`；
- 与不同长度请求共同 continuous batch；
- fresh prefill、cached decode 和 chunked prefill；
- Prefix Cache cold/warm 状态。

Trainer 将右 padding batch 转为 packed THD，为每条请求重置 position IDs，并用 `cu_seqlens` 保留独立 causal 边界。
Rollout 保留 Prefix Cache、Chunked Prefill 和 KV-cache decode，不通过永久关闭生产能力换取一致性。

vLLM-Ascend partial prefill 可能先消耗 seeded generator，再丢弃尚未完成 prefill 的 sample。Profile 在 sampling 前保存
generator offset，并只为被丢弃请求恢复 offset，避免 scheduler 切分方式改变后续采样序列。

## Policy 与 Refit 生命周期

```text
rollout(V)
→ generation 前后验证 worker-owned V/fingerprint
→ Trainer(V) optimizer 前 bit-exact comparator
→ optimizer update
→ pause admission，wake weight residency
→ NPU IPC start/update/finish，worker commit V+1
→ fingerprint/version 校验
→ wake KV cache，hard reset request/prefix cache
→ pending identity 校验
→ resume admission 和 residency check
→ controller 发布 V+1
```

Generation 前后 identity 必须保持不变。任一 worker refit、fingerprint、cache reset 或 resume 失败时，controller 不发布
下一 version。当前 fingerprint 算法只覆盖稳定的 language-model norm 权重，是生命周期 canary，不是 full-policy digest。

## Correctness 结果

### 2-NPU 两步门禁

| 指标                |    Step 1 |    Step 2 |
| ------------------- | --------: | --------: |
| Rollout version     |         0 |         1 |
| Published version   |         1 |         2 |
| 有效 response token |      2048 |      2048 |
| Pre-update mismatch |         0 |         0 |
| Max/mean abs diff   | 0.0 / 0.0 | 0.0 / 0.0 |
| Fingerprint changed |      true |      true |

该门禁证明数值和版本生命周期。Step 2 的固定样本 reward 为零、gradient norm 为零，因此它不证明两步都有新学习信号；
独立 mixed-reward smoke 已验证非零梯度和连续 V1/V2。

### 2-NPU 可复现单步 Smoke

2026-08-19 在包含 Qwen3 dual-EOS 处理的 clean commit `6860bb9b` 上重新执行 README launcher，进程 exit 0。
Selector 重新生成并验证固定 mixed-reward 数据，输出 parquet SHA256 为
`33323088f60ae101c6e9771f167fe6ded93edb0e6c5fbf4bd2314f3f2903232f`。

| 指标                     | 结果 |
| ------------------------ | ---: |
| 有效 response token      | 2048 |
| Pre-update mismatch      |    0 |
| Max/mean abs diff        | 0 / 0 |
| Reward min/max           | 0 / 1 |
| Optimizer steps          |    1 |
| Gradient norm            | 7.40625 |
| Fingerprint changed      | true |
| Published policy version |    1 |

### 8-NPU Production-Shaped 门禁

拓扑为 FSDP2 `dp_shard=8`、八个 colocated Hyper-vLLM TP1 replicas。每 rank 使用 2 prompts × 8 responses，
每 replica 16 requests、并发 12，Prefix Cache 和 Chunked Prefill 均启用。

| 指标                 |    Step 1 |    Step 2 |
| -------------------- | --------: | --------: |
| Response 数          |       128 |       128 |
| 有效 response token  |     62794 |     64790 |
| Pre-update mismatch  |         0 |         0 |
| Max/mean abs diff    | 0.0 / 0.0 | 0.0 / 0.0 |
| Optimizer mini-steps |         2 |         2 |
| Gradient norm        |  0.679688 |  0.589844 |
| Published version    |         1 |         2 |

可以声明：固定单节点 8-NPU、八个 TP1 rollout replicas、BF16 eager 配置下，两步真实 RL optimizer-pre-update
逐 token bit-exact correctness 门禁通过。不能将该结论扩大为完整 production、收敛或长期稳定性验收。

## 性能结果

公平 standalone workload 使用单 NPU、TP1、BF16 eager、16 requests、并发 12、每 response 固定 512 token、raw
logprobs、2 GiB KV cache 和五轮 cold-cache measured rounds。每轮固定生成 8192 token。

| Arm                        | 平均 token/s | 标准差 |             95% CI | 相对 Native |
| -------------------------- | -----------: | -----: | -----------------: | ----------: |
| Native-vLLM                |      111.405 |  0.534 | [110.741, 112.068] |    baseline |
| Hyper-vLLM                 |      103.296 |  0.262 | [102.970, 103.621] |     -7.279% |
| Hyper-vLLM + FA3           |       98.468 |  1.432 |  [96.690, 100.246] |    -11.613% |
| Hyper-vLLM batch invariant |       89.515 |  1.366 |   [87.819, 91.212] |    -19.649% |

Hyper-BI 相对普通 Hyper 下降 `13.341%`。FA3-only Arm 相对普通 Hyper 下降 `4.674%`；完整 Hyper-BI 相对
FA3-only 再下降 `9.092%`。后一差值同时包含 batch-invariant 数值路径和 consistency identity RPC，不能全部归因于
batch-invariant kernel。从端点绝对 token/s 损失看，FA3-only 占 `35.035%`，完整 Hyper-BI 的其余增量占
`64.965%`。

Profiler 中 batch-invariant matmul/reduce 占设备计算时间 `48.966%`，其中 `MatMulV3BatchInvariant` 单项占
`48.811%`；FA3 占 `6.938%`。设备侧首要优化目标是 batch-invariant matmul/reduce，但 identity RPC 的端到端占比仍需
独立 ablation。Profiler 比例只描述 Hyper-BI 内部组成，不能直接解释相对 Hyper 的全部性能差值。

一步 8-NPU 真实 RL 的观测结果为：

| Arm         | 有效 token | Rollout token/s | Step 时间 | Mean/Max logprob diff |
| ----------- | ---------: | --------------: | --------: | --------------------: |
| Native-vLLM |      62125 |         536.032 | 315.284 s |   0.010385 / 0.876897 |
| Hyper-vLLM  |      62861 |         574.617 | 265.438 s |  0.00951668 / 1.17954 |
| Hyper-BI    |      62794 |         524.138 | 308.661 s |             0.0 / 0.0 |

RL 使用 natural EOS，各 Arm 的 token 和动态 batch shape 不完全相同，因此该表是系统观测，不是公平吞吐 A/B。只有
Hyper-BI 满足严格 bit-exact 门禁。

## 实现入口

- `rl/consistency/qwen3_dense.py`：版本化 profile、Trainer packed forward 和依赖校验。
- `rl/consistency/gates.py`：跨 rank optimizer 前 FP32 bit comparator。
- `rl/roles/policy/actor.py`：Trainer selected-token logprob 和真实 backward。
- `rl/agentic/runner.py`：token、response mask、rollout logprobs 和 worker identity 数据链。
- `rl/roles/rollout/vllm.py`：Hyper-vLLM generation、raw logprobs 和 process isolation。
- `rl/roles/weight_sync/`：refit、worker commit、cache reset 和 policy publication。
- `examples/configs/qwen3_4b_gsm8k_vllm_production.yaml`：8-NPU 已验证 recipe。
- `tests/torch/rl/vllm/benchmark_qwen3_consistency_performance.py`：四 Arm standalone benchmark。

## 限制

- 不覆盖 graph、TP2、多节点、异步 rollout、长期 soak、完整故障注入、收敛或其他模型。
- 当前没有 full-policy digest，norm fingerprint 不能替代完整参数 manifest。
- 性能结果来自 runtime source digest
  `155c49c69d45b348bc93248047cf02040df2bb7398d6c3b28b8491f7ac220ce1` 对应的 dirty worktree；当前未提交完整
  source manifest，因此这些数字是已审计的本地实验结果，不是可由 clean commit 独立重建的 release benchmark。
- 发布新 kernel、runtime 或 profile 后必须重新执行 benchmark 和真实 RL 门禁。
- vLLM RLHF/refit development endpoints 使用不安全序列化，只能运行在受信任、隔离的训练网络。
