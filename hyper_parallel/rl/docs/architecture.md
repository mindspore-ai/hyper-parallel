# Hyper-RL 架构

## 范围

Hyper-RL 是单节点 Torch NPU 上的同步大语言模型强化学习运行时。Trainer 使用 Transformers、HyperAutoModel 与
HyperParallel FSDP/TP；rollout 使用 vLLM/vLLM-Ascend。当前端到端算法和模型范围为 Qwen3 GRPO。

当前实现已迁移到 `upstream/master@b9fa61a9` 的 HyperAutoModel/Trainer 基线。训练基础能力直接使用 master 的模型加载、
distributed setup、optimizer、gradient clipping 和 checkpoint，不保留 trainer_dev 或旧 RL Trainer 兼容路径。

`model_implementation=hyper|native` 只选择 vLLM worker 中的模型实现，不代表两套 rollout backend。Trainer、request、
ownership、policy lifecycle 和 weight-sync controller 共用同一套实现。

## 训推一体模型

Trainer 与 Hyper-vLLM 都从 Transformers Qwen3 定义获得模块命名、参数边界、tied embedding 和模型语义。两侧进一步复用
HyperParallel `ShardingPlanner` 与 `apply_sharding_plan`，因此 row/column-parallel placement、TP degree、source layout 和
checkpoint-to-TP mapping 使用同一套 contract。

Hyper-vLLM 不是训练模型的原样执行副本。它在保持 Qwen3 模型结构和 HyperParallel 分布式切分的同时，将 attention 执行叶子
替换为 vLLM paged attention，并接入 KV cache、scheduler 和 worker lifecycle：

```text
Transformers Qwen3 definition
    -> shared HyperParallel sharding plan
        +-> Trainer: trainable forward/backward + FSDP/optimizer
        +-> Hyper-vLLM: paged attention + KV cache + generation workers
```

这使权重同步可以直接使用同源 parameter/layout contract，而无需维护另一套手写推理模型或分布式映射。Native-vLLM 继续使用
vLLM 原生 Qwen3 实现，因此不属于这一训推一体模型声明。

## 组件

| 组件 | 职责 | 位置 |
| --- | --- | --- |
| Config | YAML/CLI、模型身份、并行度、deployment 和 consistency 校验 | `rl/config.py` |
| Trainer | 同步训练循环、评估、checkpoint 和资源生命周期 | `rl/trainer.py` |
| Dataset/Agentic | Parquet、tokenization、session、reward 和 trajectory | `rl/dataset/`、`rl/agentic/` |
| Actor/Algorithm | Selected-token logprob、GRPO loss、backward 和 optimizer | `rl/roles/policy/`、`rl/algorithm/` |
| Rollout | Shared vLLM server、HTTP generation 和 Qwen3 adapter | `rl/roles/rollout/` |
| Weight sync | Source/destination layout、transport、transaction 和 publication | `rl/roles/weight_sync/` |
| Consistency | Qwen3 Ascend recipe 与 optimizer-pre-update comparator | `rl/consistency/` |

Trainer 是唯一顶层编排者。Algorithm 只声明数学与数据需求；Reference Actor 是独立冻结模型，不是当前 Actor 的临时
evaluation mode。

## Shared Deployment

```text
FSDP / FSDP+TP Trainer ranks
    -> one shared vLLM endpoint
        -> upstream DP router
            -> DP engine 0 -> TP workers
            -> DP engine 1 -> TP workers
            -> ...
```

- Trainer rank 0 是唯一 server owner；其他 Trainer ranks 连接同一 endpoint。
- vLLM 创建 rollout DP/TP groups，并决定 request 对应的 DP engine。
- Hyper-RL 不实现额外 Router，不固定 Trainer rank 到 DP engine 或 TP worker。
- Frontend 数由 vLLM upstream 决定。
- Rank-local server、per-rank port 和 `rollout.vllm.topology` 已删除。

| Deployment | 设备关系 | Residency | Weight transport |
| --- | --- | --- | --- |
| Colocated | Trainer 与 rollout 共用完整 NPU 集合 | Training/rollout 分阶段切换 | NPU IPC |
| Disjoint | Rollout 使用显式且不与 Trainer 重叠的 NPU | Rollout 长期 resident | HCCL |

Colocated 要求 `rollout_dp × rollout_tp = trainer_dp_shard × trainer_tp`。Disjoint 的 rollout world size 与 Trainer
world size 独立，只要求显式设备数量等于 `rollout_dp × rollout_tp`。

两种 deployment 使用同一个配置 schema、Trainer、rollout controller 和 policy transaction 接口；`deployment` 只选择
设备 ownership、residency 和 transport。当前已验证的完整 TP2 拓扑为：

| Deployment | Trainer | Rollout | 权重发布 | Consistency |
| --- | --- | --- | --- | --- |
| Colocated | `FSDP-shard2×TP2`，4 NPU | `DP2×TP2`，共享 4 NPU | IPC full/direct/fallback | `0/0/0` |
| Disjoint | `FSDP-shard2×TP2`，NPU 0–3 | `DP2×TP2`，NPU 4–7 | HCCL full/direct/fallback | `0/0/0` |

Disjoint 还通过了双失败不发布、Prefix Cache/Chunked Prefill 和四 rank DCP destroy/resume/refit 门禁。这里的
`0/0/0` 表示 selected-token raw logprob 的 mismatch count、max absolute diff 和 mean absolute diff 均为 0。

## 数据合同

`PromptRecord` 保存稳定的 prompt identity、messages、ground truth 和 tokenized prompt。Rollout 返回的 token IDs 是
Trainer 输入的唯一权威来源，不允许 decode/re-encode。

`GenerationResult` 包含：

- prompt + response token IDs；
- 仅覆盖实际生成 response token 的 mask；
- 与 response token 对齐的 FP32 raw sampled-token logprobs；
- generation 时间与 worker policy identity。

策略生成的 EOS 属于 action；EOS 后 token、padding 和环境内容不参与 policy loss。

Batch builder 将 trajectory 转换为：

```text
sequences            [responses, tokens]
attention_mask       [responses, tokens]
action_mask          [responses, tokens]
loss_action_mask     action_mask[:, 1:]
old_log_probs        [responses, tokens - 1]
advantages           [responses, tokens - 1]
reference_log_probs  [responses, tokens - 1]
```

同一 batch 的 trajectories 必须属于同一 committed policy version/fingerprint。

## 训练生命周期

```text
validate resume
-> rollout(V) generation
-> verify generation identity
-> Reference / reward / advantage
-> Actor forward / backward / optimizer
-> publish PolicySnapshot(V+1)
-> verify workers and reset cache
-> expose V+1
-> metrics / evaluation / checkpoint
```

Colocated 在 Actor training 前让 rollout sleep，发布时依次 wake weights、传输权重、wake KV cache。Disjoint 不执行
sleep/wake，只在 publication transaction 期间暂停 admission。

## 权重发布

```text
Trainer FSDP/FSDP+TP source layout
-> rollout DP/TP destination layout
-> bounded transfer plan
-> NPU IPC or HCCL
-> worker transaction
-> identity/manifest verification
-> controller publication
```

| 策略 | TP1 | Qwen3 TP2 |
| --- | --- | --- |
| Full-gather | 重建完整 logical tensors | 正常策略、oracle 或 fallback |
| Direct-reshard | 自动退化为 full-gather | 只传输 source/destination region intersection |

Policy version 严格递增。Direct 失败时先 abort pending transaction，再由 full-gather 完整覆盖；fallback 也失败时
保持 admission 关闭，不恢复 rollout，也不发布新版本。详细合同见 [vLLM Rollout](vllm_rollout.md)。

## Checkpoint

Actor 使用 collective DCP。Optimizer、scheduler、CPU/NPU RNG 和 stateful dataloader state 按 rank 保存，
`checkpoint_complete.json` 最后发布。缺少完成标记或 world size 不一致时拒绝 resume；如果 Trainer step 高于 rollout
初始版本，generation 前必须先 refit rollout。该流程已在 disjoint `FSDP-shard2×TP2→DP2×TP2` 四 rank checkpoint
拓扑中完成 fresh-process 验收。

## 一致性边界

普通训练允许 Trainer/rollout TP degree 不同。Qwen3 consistency 只在显式开启时安装，并要求 Hyper-vLLM matched TP。
比较对象是同一 policy version、同一 token/mask 下 optimizer update 前的 FP32 raw selected-token logprobs。

数值定义见 [Qwen3 训练-推理一致性](qwen3_training_inference_consistency.md)。

## 安全边界

- 当前不支持多节点、异步/off-policy rollout、动态扩缩容或透明 generation retry。
- Native-vLLM 不属于 bit-exact 声明。
- Worker norm fingerprint 是 lifecycle canary；完整参数验收使用 source-derived manifest。
- vLLM development endpoints 只能部署在受信任、隔离的训练网络。
