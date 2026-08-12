# Hyper-RL、Slime、RL2、Molt 架构对比

对比基于本地 reference 快照：Slime `fb42ae4`、RL2 `9161ede`、Molt `3880946`。结论描述的是这些快照，不泛化到未来版本。

## 一览

| 维度 | Hyper-RL（本次重构） | Slime | RL2 | Molt |
|---|---|---|---|---|
| 设计中心 | Hyper 原生最小同步闭环 | Megatron + SGLang 大规模吞吐与 Data Buffer | Ray-less、易读、FSDP/Megatron | agentic-first、Ray + vLLM + AutoModel/FSDP2 |
| 训练引擎 | Hyper-Parallel FSDP | Megatron 主路径 | FSDP 或 Megatron | NVIDIA AutoModel + FSDP2 |
| rollout | Hyper 已验证；vLLM 使用 Hyper Qwen3.5 adapter 和外部 server | 核心只选 SGLang，深度透传其能力 | SGLang async engine/router | vLLM，Ray 管理 engine 与 router |
| 调度 | `torchrun`、共卡、同步 | Ray placement + 同步/异步 rollout/Data Buffer | `torchrun`，无 Ray；管理本地 rollout 进程 | Ray placement、异步队列、partial rollout |
| 算法扩展 | 完整 GRPO/PPO Recipe + requirements；Recipe 内复用组件 | 统一训练核，按 estimator/flags 分支 | 一个参数化在线 trainer，按 estimator/flags 分支 | advantage estimator 与 surrogate registry |
| Critic | GRPO 不分配；PPO requirements 创建独立 Qwen value model | `advantage_estimator == "ppo"` 时启用 | `adv.estimator == "gae"` 时启用 | `advantage.estimator == "gae"` 时创建独立 Critic group |
| 数据契约 | `PromptRecord → Trajectory → ExperienceBatch`，全序列多段 action mask | Sample/RolloutBatch；agent 路径另有 token/action-range Trajectory | tensor dict + `action_mask`/packed sequences | token-first Trajectory → Experience |
| Agentic | AgentRunner（框架控制）+ ProgramAgentRunner（用户控制） | custom rollout/generate function、agent harness、sandbox | `env_path` async step 与 GEM | `Env`/`StepEnvRunner` 或 `ChatAgent` 两条一等路径 |
| 权重同步 | Hyper 零拷贝发布；vLLM 分卡使用 HCCL、共卡 DP 使用 NPU IPC | SGLang broadcast/disk/delta 等成熟路径 | actor `update_rollout()` | FSDP → vLLM broadcast/refit，支持 partial/async |

## 不同算法是怎么处理的

### Hyper-RL

Trainer 从 `ALGORITHMS` 构建完整 Recipe，只读取 `algorithm.requirements`，不出现算法名称枚举。GRPO 声明：

```text
roles: actor + rollout + reference, no critic
data: rollout_log_probs + reference_log_probs + grouped_responses + action_mask
```

`ExperienceBuilder` 按 requirements 准备 reference log-probs、values、advantages/returns；Actor/Critic Manager 负责当前模型 forward、全局 token 归一化、backward 和 optimizer。用户注册完整 `GRPOAlgorithm` / `PPOAlgorithm`，内部才复用 AdvantageEstimator、PolicyObjective、Regularizer，不把任意组件组合直接暴露给配置。

PPO/GAE 和按需 Critic 已有 CPU 契约测试；当前限制是 Critic checkpoint 和 Qwen PPO NPU 端到端尚未完成，因此不能把它描述为生产就绪。

### Slime

Slime 的做法更接近“一个高性能训练核 + 配置化 estimator”，而不是每个算法一套独立 Trainer：

- `slime/utils/arguments.py` 把 `use_critic` 推导为 `advantage_estimator == "ppo"`；
- `slime/backends/megatron_utils/loss.py` 在统一 loss/advantage 路径中对 GRPO、GSPO、CISPO、PPO、REINFORCE++ 等分支；
- rollout 侧通过 `rollout_function_path`、`custom_generate_function_path` 与 Data Buffer 承载普通生成和 agentic workflow；
- 核心 rollout 后端刻意只深度支持 SGLang，而非维持多后端最小公分母。

它的优点是新算法直接继承成熟的 Megatron/SGLang、异步、权重同步和大规模调度；代价是算法选择、参数校验和 loss 分支分布在 arguments、rollout 与 backend loss 中，开发者需要跨文件追踪。

### RL2

RL2 也复用一个名为 `PPOTrainer` 的在线训练循环：rollout 后，根据配置按需计算 reference log-probs、critic values、old log-probs，再统一调用 `compute_advantages()` 与 actor update。Critic 仅在 `adv.estimator == "gae"` 创建；GRPO/Dr.GRPO 通过 advantage、KL 类型和归一化组合配置出来。

它非常直接、Ray-less，便于读懂完整控制流；但算法入口仍以 trainer 内条件分支和共享 tensor dict 为主，字段依赖没有独立 requirements 契约。相比 Hyper-RL，本次重构更强调插件声明；相比 RL2，Hyper-RL 目前远未覆盖其 FSDP/Megatron、SGLang、packing、partial rollout 和 GEM 能力。

### Molt

Molt 是四者中 agentic 与 token-first 契约最完整的一种：用户选择 `Env + StepEnvRunner`（框架拥有生成循环）或 `ChatAgent + ChatAgentRunner`（用户拥有循环），最终都交付 Trajectory。训练侧把 Trajectory 转为 Experience。

算法上，Molt 对 advantage estimator 使用 registry，已有 GRPO、Dr.GRPO、RLOO、GAE 等；reference 在 KL 系数大于零时创建，Critic 在 estimator 为 GAE 时创建。它没有为每个 XPO 复制完整训练流程，而是在一个 actor-centric async runtime 中组合 estimator、policy surrogate、KL 和可选 Critic。

这和 Hyper-RL 的方向最接近，但公开算法边界不同：Molt 更偏运行时组合 estimator/surrogate；Hyper-RL 选择完整 Recipe 作为用户 API、组件只在 Recipe 内复用。Molt 已包含 Ray placement、vLLM router、异步队列、partial rollout、高吞吐权重广播和 MoE routing replay；Hyper-RL 仍是强同步 runtime，vLLM 数据面支持单训练 rank 分卡 HCCL，以及单节点 FSDP 多 rank 到共卡 TP1 replicas 的 NPU IPC fan-out。

## 关键取舍

Hyper-RL 现在不是 Slime/RL2/Molt 的功能平替。本轮从 Molt 最值得借鉴并已落地的是四点：

1. `ExperienceBuilder` 独立于 Actor update，按 requirements 准备训练字段；
2. requirements 真正驱动 Critic 创建，GRPO 无 Critic，PPO 才创建；
3. Agentic 同时支持框架拥有循环和用户拥有循环，最终统一为 Trajectory；
4. rollout 绑定单调版本的 `PolicySnapshot`，vLLM 未真实 refit 就不能确认新版本。

因此本次没有加入 Ray、异步队列、工具沙箱或 partial rollout。外部 server、HCCL 和 NPU IPC refitter 保持 `PolicySnapshot` 接口不变；共卡推理 TP 仍需扩展 FSDP 到 TP shard mapping。

## 建议演进顺序

1. 补 Critic checkpoint/save-resume，并跑 Qwen PPO NPU 端到端；
2. 扩展共卡 refitter 的 TP rank mapping，并增加跨拓扑版本测试；
3. 在真实工具 Agent 上补超时、错误、资源边界和 chat-template 责任划分；
4. 有真实吞吐瓶颈后，再引入异步队列、背压、partial rollout 与 off-policy 修正。

这个顺序保留基础设施复用，也避免为了“支持各种 XPO/Agentic RL”的名义一次性加入不可验证的空壳。
