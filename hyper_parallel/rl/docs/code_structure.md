# Hyper-RL 代码目录结构设计

更新时间：2026-08-06。

本文以《第一级目录》和 RL2 的“工程根目录 + 同名 Python 包”方式为参考，把当前已跑通的 Qwen3.5 + GRPO 代码放入 `hyper_parallel/rl/` 工程和内层 `rl/` 包。内层包一级只保留 `dataset`、`roles`、`algorithm`、`agentic`、`utils` 五个职责域。更完整的长期目标见 [架构设计](architecture.md)。

## 1. 当前定位

当前版本是一个最小同步 RL runtime：

- 使用 Hyper-Parallel 构建、FSDP 切分、加载和训练 Qwen3.5；
- 默认使用同一个 Hyper actor 做原生生成；
- 对外保留完整 `GRPOAlgorithm` / `PPOAlgorithm` Recipe，内部复用数学组件；
- 通过 requirements 创建冻结 reference 和可选 Critic，GRPO 不创建 Critic；
- 由 `dataset.ExperienceBuilder` 准备 reference log-probs、values、advantages/returns；
- 单轮与多轮 rollout 都产出 token-first `Trajectory/ExperienceBatch`；
- 注册 vLLM 适配器和版本化 refitter 契约，但在提供具体 refitter 前不允许 Trainer 选择；
- 同时提供框架控制的 `AgentRunner` 和用户控制的 `ProgramAgentRunner`。

本次没有实现 Ray、异步队列、工具沙箱、partial rollout、具体 vLLM refitter 或 Critic checkpoint。PPO/Critic 已完成代码与 CPU 契约测试，但尚未做 Qwen PPO NPU 端到端声明。

## 2. 目录结构

```text
hyper-parallel/
└── hyper_parallel/
    └── rl/                             # Hyper-RL 工程根目录，类似 RL2/
        ├── README.md
        ├── rl/                         # 可安装/导入的 Python 包，类似 RL2/RL2/
        │   ├── contracts.py            # PromptRecord/Trajectory/ExperienceBatch
        │   ├── dataset/
        │   │   ├── adapters.py
        │   │   ├── data_source.py
        │   │   └── batch_builder.py
        │   ├── roles/                  # 训推引擎、并行角色与权重同步
        │   │   ├── model.py            # Hyper/vLLM 共享模型注册
        │   │   ├── policy/             # actor、按需 critic、value adapter
        │   │   ├── rollout/            # worker、Hyper infer、vLLM、registry
        │   │   └── weight_sync/        # versioned PolicySnapshot
        │   ├── algorithm/
        │   │   ├── components/         # Recipe 内部数学复用
        │   │   ├── reward/             # GSM8K 等 task reward
        │   │   ├── grpo.py             # 完整公开 GRPO Recipe
        │   │   └── ppo.py              # 完整公开 PPO Recipe
        │   ├── agentic/                # Environment、session 与两种 runner
        │   ├── utils/
        │   │   └── monitoring/         # console/W&B fan-out
        │   ├── trainer.py              # 已实现的 SyncTrainer
        │   └── async_trainer.py        # Ray 异步能力未实现，构造时明确失败
        ├── examples/                   # 配置、入口与 launcher
        ├── tests/                      # CPU/契约测试
        └── docs/                       # 架构、对比与复现文档
```

### 2.1 为什么这样拆

| 层 | 拥有什么 | 明确不拥有什么 |
|---|---|---|
| `contracts.py` | 跨 dataset/roles/algorithm/agentic 的稳定数据协议 | 模型、注册副作用、业务流程 |
| `dataset/` | 外部数据适配、数据源、轨迹到训练 batch | reward、模型 forward、optimizer |
| `roles/` | Policy、Rollout、模型注册、WeightSync | 算法名称分支、数据集解析 |
| `algorithm/` | 完整 GRPO/PPO Recipe、数学组件、task reward | Role 生命周期、分布式编排 |
| `agentic/` | Environment 协议、episode/session 与 runner | 第二套训练格式、optimizer |
| `utils/` | monitoring 等跨层无状态工具 | Trainer 控制流、算法语义 |
| `trainer.py` | 显式同步主循环和生命周期 | GRPO/PPO 数学细节 |

`hyper_parallel/rl/` 是独立 RL 工程边界，内层 `hyper_parallel/rl/rl/` 才是 Python import 根。这样既符合 RL2 的布局，也让 RL 能力归入 Hyper-Parallel 主包目录，而不把源代码、示例、测试和文档散落在仓库根部。

### 2.2 当前不创建的目录

`async_trainer.py` 只定义明确的能力边界，当前构造会抛出 `NotImplementedError`，不会把同步实现伪装成异步。`buffer.py`、`router.py`、`checkpoint_manager.py`、`reward_model.py` 和 `dapo.py` 仍不创建，等对应能力真正实现后再进入上述五个职责域。

## 3. 核心依赖方向

```text
Trainer
 ├── Algorithm Registry ──► GRPOAlgorithm | PPOAlgorithm
 ├── Model Registry ──────► ModelRegistration
 ├── Engine Registry ─────► HyperGenerationEngine
 ├── RolloutManager ──────► AgentRunner ──────► Trajectory
 ├── dataset.ExperienceBuilder ─► requirements-driven ExperienceBatch
 ├── ActorManager ────────► algorithm.compute_actor_loss()
 └── CriticManager ───────► algorithm.compute_critic_loss()（按需）

Algorithm ─X─► Trainer / Role Worker / Hyper-Parallel / optimizer
AgentRunner ──────► GenerationEngine + environment registry
AgentSession ─────► 同一种 Trajectory
```

依赖约束：

1. 算法层只依赖张量和稳定 contract；
2. 分布式、模型和 optimizer 由 role/trainer 层拥有；
3. `agentic` 控制流只依赖 `GenerationEngine`、`Environment` 和稳定 contracts；
4. vLLM 是可选导入，不影响 Hyper 默认环境；
5. Agentic 路径不定义第二种训练数据格式。
6. `trainer` 可以依赖各能力层，各能力层不能反向依赖 `trainer`。

## 4. 一次 GRPO step

```text
Parquet row
  ↓
PromptDataset + chat template
  ↓
PromptRecord + 精确初始 observation tokens
  ↓
GSM8KEnvironment.reset()
  ↓
AgentRunner → HyperGenerationEngine.generate()
  ├─ fixed-iteration Hyper decode
  └─ rollout policy log-probs
  ↓
Action → GSM8KEnvironment.step() → rule reward
  ↓
Trajectory × group_size
  ↓
Rollout ExperienceBatch
  ├─ full sequences/attention_mask/action_mask
  ├─ rewards
  └─ full next-token-position old_log_probs
  ↓
dataset.ExperienceBuilder
  ├─ frozen reference log-probs
  ├─ GRPOAlgorithm.build_targets()
  └─ immutable prepared ExperienceBatch
  ↓
ActorManager
  ├─ GRPOAlgorithm.compute_actor_loss()
  └─ FSDP backward/clip/optimizer
```

`action_mask` 始终与完整 sequence 等长：initial observation、tool/environment observation 与 padding 为 0，每一段 assistant action 为 1，EOS 和其后 token 为 0。`old/current/reference_log_probs` 统一覆盖 `[B, T-1]` 的 next-token positions，训练使用 `ExperienceBatch.loss_action_mask == action_mask[:, 1:]`；因此多段 action 可以一次 teacher-forcing，observation token 不进入 loss。

## 5. 算法插件

`RLAlgorithm` 当前最小接口：

```python
name: str
requirements: AlgorithmRequirements
compute_advantages(rewards, group_ids)
build_targets(rewards, action_mask, group_ids, values)
compute_actor_loss(current, old, reference, advantages, action_mask)
compute_critic_loss(current_values, old_values, returns, action_mask)
```

算法返回未归一化 token sums。ActorManager 根据全局有效 token 数完成 FSDP 正确缩放，再负责 backward 和 optimizer；算法不能调用 `optimizer.step()`。

当前 GRPO requirements：

```text
Actor          required
Rollout        required
Reference      required
Critic         not required
Grouped reward required
Old/ref log-p  required
Values         not required
Action mask    required
```

因此 GRPO 构建 actor + frozen reference，`critic_model` 保持 `None`。PPO requirements 声明 `critic=true, values=true, returns=true`，Trainer 才创建独立 Qwen Critic；value head 从已注册的末层 hidden state 计算标量 value，不从 logits 伪造。

用户注册的是完整 Recipe。`components/` 中的 estimator、objective、regularizer 只用于 Recipe 内部代码复用，不提供 YAML 任意拼装入口；因此算法语义、requirements、默认参数和测试始终绑在一起。

## 6. Role worker

### ActorModel

在 Hyper-Parallel 模型外提供两个通用能力：

- `generate()`：继承 Hyper `GenerateMixin`；
- `sequence_log_probs()`：一次 teacher forcing 计算全序列 chosen-token log-probs；
- `response_log_probs()`：仅供单次 GenerationEngine 收集新 action 的 log-probs。

它不知道算法名称。

### ActorManager

拥有 actor、optimizer、scheduler 和分布式组。它负责：

- 验证 ExperienceBatch；
- 分块计算 current log-probs；
- mini/micro batch 与梯度同步；
- 全局 token mean 缩放；
- 梯度裁剪、optimizer、scheduler；
- 返回通用 `UpdateMetrics`。

它通过注入的 `RLAlgorithm` 获取数学，不 import GRPO 实现。

### dataset.ExperienceBuilder 与 CriticManager

`ExperienceBuilder` 位于 rollout 和 optimizer 之间，只按 Recipe requirements 收集字段：GRPO 请求 reference log-probs 与 group advantages；PPO 额外请求 Critic values，并生成 GAE advantages/returns。`CriticManager` 仅在 `critic=true` 时存在，负责 value forward、分布式 token 缩放和 optimizer；value loss 仍由完整 Recipe 计算。

### RolloutManager

它只是 role 层的配置 facade：把 tokenizer、generation settings、environment 名称和 group size 交给 `AgentRunner`。reward 属于 Environment，不再写死在 manager；manager 也不直接持有 actor。

## 7. Rollout engine 与模型注册

`ModelRegistration` 把逻辑名、Hyper model spec、weights path、tokenizer path 绑定一次。Hyper 与 vLLM adapter 使用同一个 registration，避免两套模型路径配置漂移。

`GenerationEngine` 统一输入：padded token ids、attention mask、generation settings；统一输出：sequences、可选 rollout log-probs、耗时和可选显式 response mask。每个引擎还维护已加载的 `policy_version`，只接受单调递增的 `PolicySnapshot`。

- Hyper：共用 actor，无权重复制；更新只推进 snapshot 版本；固定 decode 次数确保 FSDP ranks collective 对齐；已双卡验证。
- vLLM：延迟 import、支持变长 mask 和采样 log-probs 适配；只有真实 `VLLMWeightRefitter.refit()` 成功后才推进版本。

当前同步 Trainer 会拒绝 `rollout.engine=vllm`，原因是仓库没有绑定某一种 vLLM 部署拓扑的具体 refitter。版本与 stale-policy 校验已经存在，但不能用空 refitter 冒充已加载权重。

## 8. Agentic 最小主路径

```text
ENVIRONMENTS.build(name, PromptRecord)
  ↓
AgentSession.start() → Environment.reset()
  ↓
AgentRunner 批量调用 GenerationEngine
  ↓
Action → AgentSession.apply() → Environment.step()
  ↓ 重复到 done/truncated/max_turns
AgentSession.close() → Trajectory → ExperienceBatch
```

`AgentSession` 只维护一个 episode 的 token、turn、reward 与终止状态；`AgentRunner` 负责批量生成、EOS mask 和固定 turn 调度。为保证共卡 FSDP rank 的 collective 顺序一致，每个 rank 都执行恰好 `max_turns` 次 generation；已经结束的 session 生成结果会被丢弃。

`ProgramAgentRunner` 是第二条一等入口：用户的 `AgentProgram.run()` 自己决定模型调用、工具、沙箱和结束条件，框架只验证其 `Trajectory` 的 prompt/policy version 并进入同一个 batching/ExperienceBuilder。两种 runner 共享 `dataset/batch_builder.py`，不会产生第二套训练格式。

内置 `gsm8k` Environment 是单步 episode，所以现有 Qwen + GRPO 也真实经过该链路。测试中的确定性两轮 Environment 验证 `user → assistant → tool → assistant → environment`，且只有两段 assistant token 进入 loss。新增环境只需注册 builder 并返回 tokenized Observation；每轮非初始 Observation 受 `agentic.max_observation_tokens` 硬限制。当前不替用户实现工具沙箱或 chat-template 拼接。

## 9. 配置命名

训练配置使用算法无关名称：

- `response_mini_batch_size`
- `policy_update_epochs`
- 指标 `old_policy_kl`

源代码和 YAML 中没有遗留的 PPO 专属字段。算法自己的特定配置放在 `algorithm` 节，例如 GRPO 的 group advantage、clip 和 reference KL 系数。

## 10. 测试与验证

CPU/契约测试覆盖：

- group advantage、clipping、reference KL、梯度；
- 多 mini-batch 使用更新后的 current policy；
- Hyper engine 模式恢复与 old log-probs；
- vLLM lazy registration、变长 response mask 与“refit 成功才推进版本”；
- Trajectory/ExperienceBatch token alignment；
- GSM8K Environment reward 与 close；
- 两轮 AgentRunner、全序列多段 action mask 与 log-prob 对齐；
- ProgramAgentRunner 用户控制流与统一 batching；
- GRPO/PPO Recipe、GAE、clipped value loss、Qwen hidden-state value head；
- ExperienceBuilder requirements 与 Critic optimizer；
- data adapters/source、environment reward、monitoring。

端到端验证覆盖：

- 双卡 Qwen3.5-0.8B Hyper FSDP；
- AgentRunner 主路径的快速 2×32 smoke；
- AgentRunner 主路径的 6×300 有效 GRPO 更新，非零 reward、policy loss 和 gradient。

复现命令和本次指标见 [docs/REPRODUCE_GRPO.md](docs/REPRODUCE_GRPO.md)。

## 11. 下一步边界

当前最值得借鉴的四点已落到代码：requirements-driven ExperienceBuilder、按需 Critic/PPO Recipe、用户拥有控制流的 ProgramAgentRunner、版本化 PolicySnapshot/refitter 契约。下一步严格按以下顺序继续：

1. 补齐 Critic checkpoint/save-resume，并跑 Qwen PPO NPU 端到端；
2. 为确定的 vLLM 部署方式实现真实 refitter，并增加两步 stale-policy 测试；
3. 增加真实工具 Agent 的超时、错误和资源边界；
4. 有吞吐证据后再做异步 queue、backpressure、partial rollout 与 off-policy correction。
