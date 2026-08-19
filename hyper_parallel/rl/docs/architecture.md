# Hyper-RL 架构

## 范围

Hyper-RL 是单节点强同步 LLM 强化学习运行时。当前端到端支持 GRPO，Trainer 使用 Transformers、HyperModels 和
HyperParallel FSDP，rollout 统一使用 vLLM。Qwen3、Qwen3.5 与 Native/Hyper-vLLM 四种 TP1 组合均已完成真实两步
learning smoke。

当前主要拓扑为：

```text
Trainer rank 0 ── colocated vLLM TP1 replica 0
Trainer rank 1 ── colocated vLLM TP1 replica 1
...
Trainer rank N ── colocated vLLM TP1 replica N
```

多个 rollout replicas 构成数据并行服务，每个 replica 独立管理 scheduler、请求状态和 KV cache。当前不支持 Trainer
TP、TP2 Hyper-vLLM、多节点 colocated 或异步 rollout。

## 组件职责

| 组件 | 职责 | 代码入口 |
| --- | --- | --- |
| 配置 | YAML、CLI override、模型身份、拓扑和 profile 校验 | `rl/config.py` |
| Trainer | 角色构建、同步训练循环、评估、checkpoint 和资源生命周期 | `rl/trainer.py` |
| Dataset | Parquet、tokenization、分布式采样和 batch assembly | `rl/dataset/` |
| Agentic | 环境 session、turn、reward、EOS 和 token-first trajectory | `rl/agentic/` |
| Rollout | vLLM 服务进程、HTTP generation、并发请求和模型 adapter | `rl/roles/rollout/` |
| Actor | selected-token logprobs、loss、backward 和 optimizer | `rl/roles/policy/actor.py` |
| Algorithm | GRPO/PPO requirements、advantages、targets 和 loss | `rl/algorithm/` |
| Weight sync | CPU/HCCL/NPU IPC transfer、worker identity 和 publication | `rl/roles/weight_sync/` |
| Checkpoint | Actor DCP、rank-local optimizer/RNG/dataloader state | `rl/roles/weight_sync/checkpoint.py` |
| Consistency | Qwen3 数值 profile 和 optimizer 前 bit-exact gate | `rl/consistency/` |
| Monitoring | Console/W&B、训练诊断和 learning gate | `rl/utils/monitoring/` |

Trainer 是唯一顶层编排者。Algorithm 只声明所需数据和数学计算，不持有模型或 optimizer。Reference Actor 是独立冻结
模型，不是当前 Actor 的临时 eval 模式。

## 模型与 Rollout

`rollout.engine` 当前统一为 `vllm`。`rollout.vllm.model_implementation` 只选择服务进程中的模型实现：

- `native`：使用 vLLM 原生模型实现。
- `hyper`：使用基于 Transformers 模型定义的 Hyper-vLLM adapter。

Hyper-vLLM adapter 处理模型注册、position、KV cache、weight mapping 和 logprob 接口适配。Scheduler、continuous
batching、sampling、request state、Prefix Cache、Chunked Prefill 和 sleep/wake 仍由 vLLM/vLLM-Ascend 管理。

Checkpoint `config.json` 是模型身份的权威来源。Model registration 负责解析模型 family、architecture、text config 和
训练/推理权重名称映射，避免调用方复制模型特判。

## 数据合同

### PromptRecord

`PromptRecord` 保存稳定的 `prompt_id`、结构化 messages、ground truth、metadata 和 tokenized prompt。显式配置的
`data.answer_column` 是权威答案来源；未配置时依次检查 `reward_model.ground_truth`、`extra_info`、`answer` 和
`solution`。

### GenerationResult

Rollout 返回：

- 完整 prompt + response token IDs；
- 只覆盖实际生成 token 的 `response_mask`；
- 与 response token 对齐的 FP32 rollout logprobs；
- generation 时间；
- 可选 worker-owned policy version 和 fingerprint。

### Trajectory

`Trajectory` 是单次环境 episode 的不可变 token-first 记录。`token_ids`、`attention_mask` 和 `action_mask` 必须等长，
且 `action_mask` 不得选择 padding。Rollout logprobs 对应 next-token positions，因此长度必须为 `token_count - 1`。

策略生成的 EOS 保留在 token、action mask 和 logprob 中；EOS 后 padding 或环境内容不参与 policy loss。Rollout token IDs
直接进入 Trainer，不允许通过 decode/re-encode 重建训练输入。

### ExperienceBatch

Batch builder 将 trajectories 右 padding 为二维 tensor，并保留：

```text
sequences            [responses, tokens]
attention_mask       [responses, tokens]
action_mask          [responses, tokens]
loss_action_mask     action_mask[:, 1:]
old_log_probs        [responses, tokens - 1]
advantages           [responses, tokens - 1]
reference_log_probs  [responses, tokens - 1]
```

Worker policy version 和 fingerprint 必须成对出现，并在所有 trajectories 与 batch 间保持一致。

## 同步训练生命周期

```text
加载配置和 consistency profile
→ 初始化 FSDP Actor、Reference、rollout replicas
→ rollout 使用 policy V 生成 token 和 old logprobs
→ vLLM sleep，释放训练阶段不需要的 residency
→ 可选 Trainer pre-update consistency forward
→ Reference logprobs、rewards 和 advantages
→ Actor backward 与 optimizer step
→ 将 V+1 权重同步到全部 rollout replicas
→ worker identity 校验和 hard cache reset
→ 恢复 KV residency，resume admission
→ controller 发布 policy V+1
→ metrics、evaluation 和 checkpoint
```

Policy version 只有在 weight transfer、worker commit、fingerprint、cache reset、resume 和 residency check 全部成功后
才对下一批 rollout 可见。任一 rank 或 replica 失败都会阻止版本发布。

## Colocated 显存生命周期

FSDP Trainer 与 vLLM 共享同一组 NPU，不能长期同时持有完整训练状态、推理权重和 KV cache。Colocated 模式使用阶段化
residency：

```text
rollout：vLLM weights + KV cache active
training：vLLM level-1 sleep，FSDP/optimizer active
refit：scheduler paused，按顺序 wake weights、transfer、wake KV cache
```

Trainer 在唤醒 rollout 前检查 FSDP reshard、optimizer CPU residency 和 allocator cache 释放，避免跨阶段显存泄漏。

## 在线权重发布

`PolicySnapshot` 包含严格递增 version、模型名、Actor payload 和 metadata。Colocated 模式使用 NPU IPC：

1. Pause rollout admission。
2. 提取并映射 FSDP Actor 权重。
3. Start、update、finish worker weight transaction。
4. Worker commit pending version。
5. 校验所有 replicas 的 fingerprint 和 version。
6. Wake KV cache 并 hard reset request/prefix cache。
7. Resume admission 并检查 residency。
8. Controller 发布下一 version。

当前 fingerprint 算法 `qwen_norms_f32_v3` 只覆盖稳定的 language-model norm 权重。它是生命周期 canary，不是
full-policy digest，不能替代完整参数 manifest。

## Checkpoint

Actor 权重通过 collective DCP 保存。Optimizer、scheduler、CPU/NPU RNG 和 stateful dataloader state 按 rank 独立
保存。`checkpoint_complete.json` 最后原子发布；缺少完成标记或 world size 不一致时拒绝 resume。

Resume 后若 Trainer step 高于 rollout 初始 version，Trainer 必须先 refit rollout，之后才能接受新 generation。

## 已验证能力

- Qwen3/Qwen3.5 × Native/Hyper-vLLM 四组合两步 GRPO learning smoke。
- Mixed reward、非零 gradient norm、Actor fingerprint change 和 policy V1/V2。
- Colocated NPU IPC refit、hard cache reset 和重复 sleep/wake。
- Qwen3 Hyper-vLLM 单节点 8-NPU、八个 TP1 replicas 的 optimizer 前 bit-exact 门禁。
- Prefix Cache、Chunked Prefill 和每 replica 12 请求 continuous batching。

Qwen3 一致性的数值合同、结果和性能见
[Qwen3 训练-推理一致性](qwen3_training_inference_consistency.md)。

## 限制与安全

- GRPO 已端到端验证；PPO/Critic 当前只具备数学与接口测试。
- 不支持 Trainer TP、TP2 Hyper-vLLM、多节点 colocated 或异步 rollout。
- Native 与 Hyper 实现之间不承诺 token 或 logprob exact。
- 尚未完成 graph、长期 soak、完整故障注入、full-policy identity 或收敛验收。
- vLLM RLHF/refit development endpoints 使用不安全序列化，只能运行在受信任、隔离的训练网络。
