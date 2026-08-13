# Hyper-RL 可扩展强化学习框架设计

## 1. 文档目标

本文定义 `Hyper-RL` 从当前最小 GRPO Demo 演进为通用大模型强化学习框架的目标架构。

框架重点满足以下需求：

1. Actor、Rollout、分布式训练、Checkpoint、日志等基础设施在不同算法之间复用；
2. 算法通过显式能力声明决定是否创建 Critic、Reference、Reward Model 等角色；
3. 同时支持单轮文本 RL 和可扩展的 Agentic RL；
4. 用户可以低成本开发新的 XPO 算法，不需要修改 Trainer 和 Role Worker 核心代码；
5. 支持同步训练，并为异步 Rollout、Experience Buffer 和策略版本控制保留清晰接口；
6. 算法实现保持独立语义，避免在 GRPO 中暴露大量 `ppo_*` 命名。

本文同时描述目标架构与已经固定的公开边界。当前代码结构和验证状态参见 [代码结构](code_structure.md)。

截至 2026-08-10，当前实现已有完整 GRPO/PPO Recipe、requirements-driven ExperienceBuilder、按需 Qwen Critic、两种 Agent runner、统一 Trajectory/action mask 和版本化 PolicySnapshot。GRPO 已完成双卡 NPU 闭环；vLLM 已接入 Hyper Qwen3.5 adapter、独立 server、分卡 HCCL 和共卡 DP/NPU IPC refitter；PPO/Critic 目前完成 CPU 契约验证。Critic checkpoint、共卡推理 TP、真实 Tool Router 和异步训练仍属于后续阶段。

## 2. 设计原则

### 2.1 Role 负责能力，Algorithm 负责数学

```text
Role Worker
    负责“可以执行什么”
    例如 forward、generate、compute_values、optimizer_step

Algorithm
    负责“为什么执行、如何计算”
    例如 group advantage、GAE、policy objective、value loss
```

Role Worker 不判断当前算法是 PPO、GRPO、DAPO 还是 GSPO。

Algorithm 不直接管理 FSDP、设备、进程组、模型权重和底层 optimizer 实现。

### 2.2 基础设施复用，完整 Recipe 对外，组件在内部复用

不同 XPO 通常共享：

- Rollout；
- Token/Mask 处理；
- Actor log probability；
- Reference log probability；
- Mini/Micro Batch；
- 分布式梯度同步；
- Checkpoint；
- Weight Sync；
- Tracking。

真正变化的主要是：

- Reward 后处理；
- Advantage/Return Estimator；
- Importance Ratio 粒度；
- Policy Objective；
- KL/Entropy Regularization；
- Loss Reduction；
- 是否需要 Critic；
- Rollout Sampling/Filtering 策略。

因此框架采用“公共执行骨架 + 完整公开 Recipe + Recipe 内部组件复用”。用户选择和注册的是 `GRPOAlgorithm`、`PPOAlgorithm`、`DAPOAlgorithm` 这类语义完整的算法，不直接在 YAML 中任意拼 Advantage、Objective 和 Regularizer。这样既避免复制 Trainer，也避免产生未经验证的算法组合。

### 2.3 能力声明驱动角色创建

Algorithm 必须声明运行所需角色：

```python
RoleRequirements(
    actor=True,
    rollout=True,
    reference=False,
    critic=False,
    reward_model=False,
    judge=False,
)
```

Runtime 根据声明创建角色，不在配置解析和 Trainer 中散落算法名称判断。

### 2.4 数据契约优先

模块之间通过稳定的数据契约交互：

```text
PromptRecord
    ↓
Trajectory
    ↓
ExperienceBatch
    ↓
LossOutput / UpdateMetrics
```

单轮 RL 和 Agentic RL 使用同一种 `Trajectory` 表达方式。

### 2.5 同步和异步共享语义，不共享所有控制流

同步 Trainer 与异步 Trainer 可以拥有独立控制流，但共同使用：

- Algorithm Plugin；
- Role Worker；
- Trajectory/Experience；
- Rollout Runtime；
- Checkpoint；
- Tracking；
- Weight Sync。

### 2.6 插件优先，核心代码保持封闭

用户新增算法时，原则上只需要：

1. 实现一个完整 Algorithm Recipe，内部可复用现有组件；
2. 注册算法名称；
3. 添加算法配置；
4. 编写算法契约测试。

不应该要求用户修改：

- `Trainer.train()`；
- Actor Worker；
- Rollout Engine；
- Checkpoint Manager；
- 分布式初始化代码。

## 3. 总体架构

```text
                         ┌──────────────────────┐
                         │ Config / CLI         │
                         └──────────┬───────────┘
                                    │
                         ┌──────────▼───────────┐
                         │ Algorithm Registry   │
                         │ Complete Recipes     │
                         └──────────┬───────────┘
                                    │
                         ┌──────────▼───────────┐
                         │ Runtime Builder      │
                         │ 根据能力声明创建角色 │
                         └──────────┬───────────┘
                                    │
              ┌─────────────────────┼─────────────────────┐
              │                     │                     │
   ┌──────────▼─────────┐ ┌─────────▼──────────┐ ┌────────▼─────────┐
   │ Trainer Runtime    │ │ Role Workers       │ │ Rollout Runtime  │
   │ Sync / Async       │ │ Actor/Critic/Ref   │ │ Single/Agentic   │
   └──────────┬─────────┘ └─────────┬──────────┘ └────────┬─────────┘
              │                     │                     │
              └─────────────────────┼─────────────────────┘
                                    │
                         ┌──────────▼───────────┐
                         │ Trajectory / Buffer │
                         │ Experience Builder  │
                         └──────────┬───────────┘
                                    │
                         ┌──────────▼───────────┐
                         │ Algorithm Recipe    │
                         │ Advantage/Objective │
                         └──────────┬───────────┘
                                    │
                         ┌──────────▼───────────┐
                         │ Actor/Critic Update │
                         └──────────────────────┘
```

横向公共能力：

```text
Checkpoint | Weight Sync | Tracking | Distributed | Fault Handling
```

## 4. 长期完整目录蓝图

本节描述能力全部落地后的扩展蓝图，不等于当前仓库会预建这些模块。当前可运行代码及其目录所有权以 [代码目录结构设计](code_structure.md) 为准；未实现的异步、RewardModel、checkpoint 和更多 engine 不创建可导入空壳。

```text
hyper-parallel/hyper_parallel/rl/rl/
├── dataset/
│   ├── adapters.py
│   ├── data_source.py
│   ├── batch_builder.py
│   ├── experience_buffer.py
│   └── sampler.py
│
├── roles/
│   ├── model.py
│   ├── policy/
│   │   ├── actor.py
│   │   ├── critic.py
│   │   ├── reference.py
│   │   ├── reward_model.py
│   │   └── judge.py
│   ├── rollout/
│   │   ├── base.py
│   │   ├── registry.py
│   │   ├── worker.py
│   │   ├── hyper_infer.py
│   │   ├── vllm.py
│   │   ├── sglang.py
│   │   └── router.py
│   └── weight_sync/
│       ├── base.py
│       ├── in_process.py
│       ├── collective.py
│       ├── tensor_stream.py
│       ├── checkpoint.py
│       └── delta.py
│
├── algorithm/
│   ├── base.py
│   ├── requirements.py
│   ├── context.py
│   ├── output.py
│   │
│   ├── advantage/
│   │   ├── base.py
│   │   ├── gae.py
│   │   ├── group_relative.py
│   │   ├── group_centered.py
│   │   ├── leave_one_out.py
│   │   └── reinforce.py
│   │
│   ├── objective/
│   │   ├── base.py
│   │   ├── clipped_token.py
│   │   ├── clipped_sequence.py
│   │   ├── dual_clip.py
│   │   ├── cispo.py
│   │   └── reinforce.py
│   │
│   ├── regularization/
│   │   ├── base.py
│   │   ├── reference_kl.py
│   │   ├── reward_kl.py
│   │   ├── adaptive_kl.py
│   │   └── entropy.py
│   │
│   ├── reward_processing/
│   │   ├── base.py
│   │   ├── group_normalization.py
│   │   ├── reward_shaping.py
│   │   └── dynamic_sampling.py
│   │
│   ├── reducers/
│   │   ├── base.py
│   │   ├── token_mean.py
│   │   ├── sample_mean.py
│   │   └── sequence_mean.py
│   │
│   ├── reward/
│   │   ├── base.py
│   │   └── rule.py
│   └── recipes/
│       ├── ppo.py
│       ├── grpo.py
│       ├── dr_grpo.py
│       ├── dapo.py
│       ├── gspo.py
│       ├── cispo.py
│       └── rloo.py
│
├── agentic/
│   ├── base.py
│   ├── registry.py
│   ├── runner.py
│   ├── program_runner.py
│   ├── session.py
│   ├── trajectory_builder.py
│   ├── tool_router.py
│   ├── timeout.py
│   ├── tool_environment.py
│   ├── sandbox_environment.py
│   └── multi_agent_environment.py
│
├── utils/
│   ├── config/
│   │   ├── schema.py
│   │   ├── loader.py
│   │   └── validation.py
│   ├── checkpoint/
│   │   ├── manager.py
│   │   └── state.py
│   ├── monitoring/
│   │   ├── tracker.py
│   │   ├── metrics.py
│   │   └── trajectory_logger.py
│   └── distributed/
│       ├── topology.py
│       ├── groups.py
│       └── reductions.py
│
├── contracts.py
├── trainer.py
└── async_trainer.py
```

## 5. Algorithm Plugin 设计

### 5.1 RoleRequirements

算法通过能力声明描述依赖：

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class RoleRequirements:
    actor: bool = True
    rollout: bool = True
    reference: bool = False
    critic: bool = False
    reward_model: bool = False
    judge: bool = False
    environment: bool = False
```

还可以声明所需数据：

```python
@dataclass(frozen=True)
class DataRequirements:
    rollout_log_probs: bool = True
    reference_log_probs: bool = False
    values: bool = False
    grouped_responses: bool = False
    token_rewards: bool = False
    action_mask: bool = True
```

完整算法需求：

```python
@dataclass(frozen=True)
class AlgorithmRequirements:
    roles: RoleRequirements
    data: DataRequirements
```

### 5.2 RLAlgorithm 协议

```python
from typing import Protocol


class RLAlgorithm(Protocol):
    name: str

    @property
    def requirements(self) -> AlgorithmRequirements:
        ...

    def build_targets(
        self,
        rewards: Tensor,
        action_mask: Tensor,
        group_ids: tuple[str | None, ...] | None,
        values: Tensor | None,
    ) -> TargetOutput:
        ...

    def compute_actor_loss(
        self,
        current_log_probs: Tensor,
        old_log_probs: Tensor,
        reference_log_probs: Tensor | None,
        advantages: Tensor,
        action_mask: Tensor,
    ) -> LossOutput:
        ...

    def compute_critic_loss(
        self,
        current_values: Tensor,
        old_values: Tensor,
        returns: Tensor,
        action_mask: Tensor,
    ) -> CriticLossOutput:
        ...
```

Algorithm 不执行 `optimizer.step()`，以免绑定具体并行框架。它只输出可反向传播的 loss 和指标。

### 5.3 完整 Algorithm Recipe 是公开 API

每个公开算法是一个完整类，固定自己的 requirements、默认组件和约束：

```python
class GRPOAlgorithm:
    name = "grpo"
    requirements = GRPO_REQUIREMENTS

    def __init__(self, config: GRPOConfig):
        self._advantage_estimator = GroupRelativeAdvantageEstimator(...)
        self._policy_objective = ClippedPolicyObjective(...)
        self._regularizer = LowVarianceKLRegularizer(...)

    def build_targets(...): ...
    def compute_actor_loss(...): ...
```

`PPOAlgorithm` 同样是完整 Recipe，内部组合 GAE、clipped policy objective、KL regularizer 和 clipped value objective。用户不能绕过 Recipe 直接把 GRPO advantage 与任意 value loss 拼在一起。

对于控制流明显不同的算法，允许直接实现 `RLAlgorithm`，不强制套入 Recipe。

### 5.4 内部组件协议

组件协议是代码复用边界，不是最终用户配置 API。新增或替换组件后，必须由一个完整 Recipe 选择它，并由该 Recipe 的契约/数值测试覆盖。

Advantage：

```python
class AdvantageEstimator(Protocol):
    def compute(
        self,
        trajectories: list[Trajectory],
        rewards: Tensor,
        values: Tensor | None,
        masks: Tensor,
    ) -> AdvantageOutput:
        ...
```

Policy Objective：

```python
class PolicyObjective(Protocol):
    def compute(
        self,
        current_log_probs: Tensor,
        old_log_probs: Tensor,
        advantages: Tensor,
        action_mask: Tensor,
    ) -> ObjectiveOutput:
        ...
```

Regularizer：

```python
class Regularizer(Protocol):
    def compute(
        self,
        experience: ExperienceBatch,
        actor_output: ActorOutput,
    ) -> RegularizationOutput:
        ...
```

Loss Reducer：

```python
class LossReducer(Protocol):
    def reduce(
        self,
        token_losses: Tensor,
        action_mask: Tensor,
        group_ids: Tensor | None,
    ) -> Tensor:
        ...
```

### 5.5 算法注册

```python
ALGORITHMS = Registry("algorithm")


@ALGORITHMS.register("grpo")
def build_grpo(config: Mapping[str, Any]) -> RLAlgorithm:
    return GRPOAlgorithm(GRPOConfig.from_mapping(config))
```

Runtime 只依赖注册表：

```python
algorithm = ALGORITHMS.build(
    config.algorithm.name,
    config.algorithm,
)
```

核心 Trainer 不出现算法枚举：

```python
if name == "grpo":
    ...
elif name == "ppo":
    ...
```

## 6. Critic 按需创建

### 6.1 Runtime Builder

```python
class RuntimeBuilder:
    def build(self, config, algorithm):
        requirements = algorithm.requirements

        actor = self.build_actor(config.actor)
        rollout = self.build_rollout(config.rollout)

        reference = None
        if requirements.roles.reference:
            reference = self.build_reference(config.reference)

        critic = None
        if requirements.roles.critic:
            critic = self.build_critic(config.critic)

        reward_model = None
        if requirements.roles.reward_model:
            reward_model = self.build_reward_model(config.reward_model)

        return RLRuntime(
            algorithm=algorithm,
            actor=actor,
            rollout=rollout,
            reference=reference,
            critic=critic,
            reward_model=reward_model,
        )
```

### 6.2 GRPO Requirements

```python
GRPO_REQUIREMENTS = AlgorithmRequirements(
    roles=RoleRequirements(
        actor=True,
        rollout=True,
        reference=True,
        critic=False,
    ),
    data=DataRequirements(
        rollout_log_probs=True,
        reference_log_probs=True,
        values=False,
        grouped_responses=True,
    ),
)
```

### 6.3 PPO Requirements

```python
PPO_REQUIREMENTS = AlgorithmRequirements(
    roles=RoleRequirements(
        actor=True,
        rollout=True,
        reference=True,
        critic=True,
    ),
    data=DataRequirements(
        rollout_log_probs=True,
        reference_log_probs=True,
        values=True,
        grouped_responses=False,
    ),
)
```

### 6.4 配置校验

配置校验应基于 requirements：

```python
if requirements.roles.critic and config.critic is None:
    raise ValueError(
        f"Algorithm '{algorithm.name}' requires critic configuration"
    )
```

不需要 Critic 时，即使配置中存在 Critic 字段，也默认不创建；可以选择警告用户配置未生效。

## 7. Role Worker 设计

### 7.1 ActorWorker

ActorWorker 只暴露通用模型执行能力：

```python
class ActorWorker:
    def compute_log_probs(
        self,
        sequences: Tensor,
        attention_mask: Tensor,
        action_mask: Tensor,
    ) -> ActorOutput:
        ...

    def backward(
        self,
        loss: Tensor,
        sync_gradients: bool,
    ) -> None:
        ...

    def optimizer_step(self) -> OptimizerMetrics:
        ...

    def zero_grad(self) -> None:
        ...

    def state_dict(self) -> Mapping[str, Any]:
        ...
```

ActorWorker 不提供 `update_grpo()` 或 `update_ppo()`。

### 7.2 CriticWorker

```python
class CriticWorker:
    def compute_values(
        self,
        sequences: Tensor,
        attention_mask: Tensor,
        action_mask: Tensor,
    ) -> CriticOutput:
        ...

    def backward(
        self,
        loss: Tensor,
        sync_gradients: bool,
    ) -> None:
        ...

    def optimizer_step(self) -> OptimizerMetrics:
        ...
```

Critic 的 Value Loss 由 Algorithm 计算，CriticWorker 只负责 Value 模型执行和优化。

### 7.3 ReferenceWorker

```python
class ReferenceWorker:
    def compute_log_probs(
        self,
        sequences: Tensor,
        attention_mask: Tensor,
        action_mask: Tensor,
    ) -> ReferenceOutput:
        ...
```

Reference 可以由不同后端实现：

- 独立冻结模型；
- Actor 初始权重快照；
- CPU Offload；
- 远端 Reference Service；
- 无 Reference。

### 7.4 RewardWorker 和 JudgeWorker

```python
class RewardWorker:
    def score(
        self,
        trajectories: list[Trajectory],
    ) -> RewardOutput:
        ...
```

Reward 支持：

- Rule Reward；
- Reward Model；
- Process Reward；
- LLM Judge；
- Environment Reward；
- 多种 Reward 加权组合。

## 8. 核心数据契约

### 8.1 PromptRecord

```python
@dataclass
class PromptRecord:
    prompt_id: str
    messages: list[Message]
    ground_truth: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
```

### 8.2 Turn

```python
@dataclass
class Turn:
    role: Literal[
        "system",
        "user",
        "assistant",
        "tool",
        "environment",
    ]
    content: Any
    token_start: int
    token_end: int
    trainable: bool
    tool_call: ToolCall | None = None
    observation: Any | None = None
```

### 8.3 Trajectory

```python
@dataclass
class Trajectory:
    trajectory_id: str
    prompt_id: str
    group_id: str | None
    policy_version: int

    turns: list[Turn]
    token_ids: Tensor
    attention_mask: Tensor
    action_mask: Tensor

    rollout_log_probs: Tensor | None

    reward: float
    reward_components: dict[str, float]

    done: bool
    truncated: bool
    terminal_reason: str

    metadata: dict[str, Any]
```

`action_mask` 是统一 RL 训练掩码。

单轮 RL：

```text
Prompt Tokens     0 0 0 0
Response Tokens   1 1 1 1
```

Agentic RL：

```text
System Tokens       0
User Tokens         0
Assistant Action    1
Tool Observation    0
Assistant Action    1
Environment Result  0
```

### 8.4 ExperienceBatch

```python
@dataclass
class ExperienceBatch:
    trajectory_ids: list[str]
    group_ids: Tensor | None
    policy_versions: Tensor

    sequences: Tensor
    attention_mask: Tensor
    action_mask: Tensor

    rewards: Tensor
    advantages: Tensor | None
    returns: Tensor | None
    values: Tensor | None

    old_log_probs: Tensor | None
    reference_log_probs: Tensor | None

    metadata: dict[str, Any]
```

所有算法字段允许 Optional，但 Algorithm Requirements 必须声明所需字段，Experience Builder 在训练前完成校验。

### 8.5 RewardOutput

```python
@dataclass
class RewardOutput:
    total_rewards: Tensor
    components: dict[str, Tensor]
    metadata: list[dict[str, Any]]
```

不应只保留最终总 Reward。组件级 Reward 是训练诊断和 Reward Hacking 分析的重要依据。

## 9. Experience Builder

Experience Builder 连接 Role Outputs 和 Algorithm：

```text
Trajectories
    │
    ├── Actor/Old Log Probs
    ├── Reference Log Probs（可选）
    ├── Critic Values（可选）
    └── Reward Components
    │
    ▼
Algorithm.process_trajectories()
    │
    ├── Reward Processing
    ├── Advantage/Return
    └── Batch Validation
    │
    ▼
ExperienceBatch
```

建议接口：

```python
class ExperienceBuilder:
    def build(
        self,
        trajectories: list[Trajectory],
        algorithm: RLAlgorithm,
        roles: RoleSet,
    ) -> ExperienceBatch:
        requirements = algorithm.requirements
        role_outputs = self.collect_required_outputs(
            trajectories,
            requirements,
            roles,
        )
        return algorithm.process_trajectories(
            trajectories,
            role_outputs,
        )
```

## 10. Rollout Runtime

### 10.1 统一接口

```python
class RolloutRuntime(Protocol):
    async def rollout(
        self,
        requests: list[RolloutRequest],
        policy_version: int,
    ) -> list[Trajectory]:
        ...
```

Rollout Runtime 负责生成轨迹，不负责算法 Advantage 和 Policy Loss。

### 10.2 RolloutRequest

```python
@dataclass
class RolloutRequest:
    request_id: str
    prompt: PromptRecord
    num_samples: int
    generation_config: GenerationConfig
    environment_name: str | None = None
    max_turns: int = 1
    timeout_seconds: float | None = None
```

### 10.3 Engine Adapter

```python
class GenerationEngine(Protocol):
    async def generate(
        self,
        requests: list[GenerationRequest],
    ) -> list[GenerationResult]:
        ...

    async def update_weights(
        self,
        weight_handle: WeightHandle,
    ) -> None:
        ...
```

后端可以是：

- Hyper 原生推理；
- vLLM；
- SGLang；
- 远端 OpenAI-compatible Service；
- 测试 Fake Engine。

## 11. Agentic RL 扩展

### 11.1 Environment 协议

```python
class Environment(Protocol):
    async def reset(
        self,
        request: EnvRequest,
    ) -> Observation:
        ...

    async def step(
        self,
        action: Action,
    ) -> Transition:
        ...

    async def close(self) -> None:
        ...
```

```python
@dataclass
class Transition:
    observation: Observation
    reward: float
    done: bool
    truncated: bool
    info: dict[str, Any]
```

### 11.2 AgentSession

```python
class AgentSession:
    async def run(
        self,
        request: RolloutRequest,
        policy_version: int,
    ) -> Trajectory:
        observation = await self.environment.reset(
            EnvRequest.from_rollout(request)
        )

        builder = TrajectoryBuilder(request, policy_version)
        builder.append_observation(observation)

        for turn_id in range(request.max_turns):
            generation = await self.engine.generate(
                [builder.next_generation_request()]
            )
            action = self.action_parser.parse(generation[0])
            builder.append_action(action, generation[0])

            transition = await self.environment.step(action)
            builder.append_transition(transition)

            if transition.done or transition.truncated:
                break

        await self.environment.close()
        return builder.build()
```

### 11.3 Agentic Runtime 必须处理

- Tool Call 解析失败；
- Tool 超时；
- Sandbox 资源限制；
- 最大轮次；
- 上下文长度限制；
- Observation 截断；
- 环境异常和重试；
- Episode 取消；
- 多 Agent 消息路由；
- Partial Rollout；
- Terminal Reward 和 Step Reward；
- 只有 Agent Action Token 参与训练。

### 11.4 Agentic Credit Assignment

初始版本支持 Episode-level Reward：

```text
整个轨迹一个 Reward
    ↓
广播到所有 Action Token
```

后续通过独立组件扩展：

```text
credit_assignment/
├── episode_broadcast.py
├── discounted_step_reward.py
├── process_reward.py
├── turn_level_advantage.py
└── counterfactual.py
```

Credit Assignment 不应硬编码在 Environment 或 Rollout Engine 中。

## 12. Trainer 与 Execution Plan

### 12.1 ExecutionPlan

Algorithm 将 requirements 转换为执行计划：

```python
@dataclass
class ExecutionPlan:
    collect_rollout_log_probs: bool
    compute_reference_log_probs: bool
    compute_values: bool
    update_actor: bool
    update_critic: bool
    actor_update_epochs: int
    critic_update_epochs: int
```

Trainer 根据计划执行，不判断算法名称。

### 12.2 Sync Trainer

```python
class SyncTrainer:
    def train_step(self) -> StepMetrics:
        prompts = self.prompt_source.next_batch()

        trajectories = self.rollout_runtime.rollout_sync(
            self.request_builder.build(prompts),
            policy_version=self.state.model_version,
        )

        experience = self.experience_builder.build(
            trajectories,
            algorithm=self.algorithm,
            roles=self.roles,
        )

        actor_metrics = self.actor_updater.update(
            self.algorithm,
            experience,
        )

        critic_metrics = None
        if self.algorithm.requirements.roles.critic:
            critic_metrics = self.critic_updater.update(
                self.algorithm,
                experience,
            )

        self.state.model_version += 1
        self.weight_sync.publish(self.state.model_version)

        return merge_metrics(actor_metrics, critic_metrics)
```

### 12.3 Async Trainer

```text
Prompt Source
    │
    ▼
Rollout Producers ───────────────┐
    │                            │
    ▼                            │
Experience Buffer               │
    │                            │
    ▼                            │
Learner                         │
    │                            │
    ├── Actor/Critic Update      │
    └── Weight Publisher ────────┘
```

异步训练必须检查策略版本：

```python
staleness = current_model_version - trajectory.policy_version

if staleness > max_policy_staleness:
    reject_or_downweight(trajectory)
```

### 12.4 Experience Buffer

```python
class ExperienceBuffer(Protocol):
    async def put(
        self,
        trajectories: list[Trajectory],
    ) -> None:
        ...

    async def get(
        self,
        requirements: BatchRequirements,
    ) -> list[Trajectory]:
        ...
```

Buffer 支持：

- 容量限制；
- Backpressure；
- Policy Version 过滤；
- Group 完整性检查；
- Priority Sampling；
- Episode 去重；
- 失败轨迹隔离。

## 13. Actor/Critic Updater

Updater 是 Algorithm 与 Role Worker 的桥梁，负责通用优化流程：

```python
class ActorUpdater:
    def update(
        self,
        algorithm: RLAlgorithm,
        experience: ExperienceBatch,
    ) -> UpdateMetrics:
        for epoch in range(self.config.update_epochs):
            for mini_batch in self.batch_iterator(experience):
                self.actor.zero_grad()

                for micro_batch in mini_batch.micro_batches():
                    actor_output = self.actor.compute_log_probs(
                        micro_batch.sequences,
                        micro_batch.attention_mask,
                        micro_batch.action_mask,
                    )

                    loss_output = algorithm.compute_actor_loss(
                        micro_batch,
                        actor_output,
                    )

                    self.actor.backward(
                        loss_output.loss,
                        sync_gradients=micro_batch.is_last,
                    )

                optimizer_metrics = self.actor.optimizer_step()

        return aggregate_metrics(...)
```

通用 Updater 负责：

- Mini/Micro Batch；
- Gradient Accumulation；
- FSDP Sync 开关；
- Global Token Count；
- Gradient Clipping；
- Optimizer Step；
- Scheduler Step；
- Non-finite 检查。

Algorithm 负责：

- Advantage；
- Policy Objective；
- Regularization；
- Loss Reduction；
- 算法指标。

## 14. XPO Recipe 示例

本节的组件列表用于说明各完整 Recipe 的内部组成，不代表向用户暴露通用 `AlgorithmRecipe(...)` 构造器。实际注册项应分别是 `GRPOAlgorithm`、`PPOAlgorithm`、`DAPOAlgorithm` 等完整类。

### 14.1 GRPO

```python
class GRPOAlgorithm:
    name = "grpo"
    requirements = GRPO_REQUIREMENTS
    _advantage_estimator = GroupRelativeAdvantageEstimator(...)
    _policy_objective = TokenClippedObjective(clip_low=0.2, clip_high=0.2)
    _regularizer = ReferenceK3KL(coef=0.001)
```

### 14.2 Dr.GRPO

```python
class DrGRPOAlgorithm:
    name = "dr_grpo"
    requirements = DR_GRPO_REQUIREMENTS
    _advantage_estimator = GroupCenteredAdvantage(divide_std=False)
    _policy_objective = TokenClippedObjective(...)
```

### 14.3 PPO

```python
class PPOAlgorithm:
    name = "ppo"
    requirements = PPO_REQUIREMENTS
    _advantage_estimator = GAE(gamma=1.0, gae_lambda=0.95)
    _policy_objective = TokenClippedObjective(clip_low=0.2, clip_high=0.2)
    _regularizer = ReferenceK3KL(coef=0.001)
    _value_objective = ClippedValueObjective(value_clip=0.2)
```

### 14.4 DAPO

```python
class DAPOAlgorithm:
    name = "dapo"
    requirements = DAPO_REQUIREMENTS
    _advantage_estimator = GroupRelativeAdvantageEstimator(...)
    _policy_objective = TokenClippedObjective(clip_low=0.2, clip_high=0.28)
    _rollout_filter = DynamicSamplingFilter()
```

### 14.5 GSPO

```python
class GSPOAlgorithm:
    name = "gspo"
    requirements = GSPO_REQUIREMENTS
    _advantage_estimator = GroupRelativeAdvantageEstimator(...)
    _policy_objective = SequenceClippedObjective(...)
    _regularizer = ReferenceK3KL(coef=0.001)
```

这些示例用于说明组件组合边界，具体公式和默认超参数应由对应算法实现与测试确定。

## 15. 用户开发新算法

### 15.1 最简单情况：新增完整 Recipe 并复用内部组件

用户新增一个语义完整的算法类：

```python
@ALGORITHMS.register("my_xpo")
def build_my_xpo(config):
    return MyXPOAlgorithm(MyXPOConfig.from_mapping(config))


class MyXPOAlgorithm:
    name = "my_xpo"
    requirements = AlgorithmRequirements(...)

    def __init__(self, config):
        self._advantage = ExistingAdvantageEstimator(...)
        self._objective = ExistingPolicyObjective(...)
        self._regularizer = ExistingRegularizer(...)

    def build_targets(...): ...
    def compute_actor_loss(...): ...
```

无需修改 Trainer。

### 15.2 新增 Advantage Estimator

```python
class LeaveOneOutAdvantage:
    def compute(self, trajectories, rewards, values, masks):
        grouped = group_by_id(rewards, trajectories)
        baseline = leave_one_out_mean(grouped)
        advantages = grouped - baseline
        return AdvantageOutput(advantages=advantages)
```

随后由 `RLOOAlgorithm` 或另一个完整 Recipe 显式选择该 estimator；不单独把它注册为用户可选算法。

### 15.3 新增 Policy Objective

```python
class MyClippedISObjective:
    def compute(
        self,
        current_log_probs,
        old_log_probs,
        advantages,
        action_mask,
    ):
        ratio = (current_log_probs - old_log_probs).exp()
        token_losses = ...
        return ObjectiveOutput(
            token_losses=token_losses,
            metrics={...},
        )
```

同样必须由完整 Recipe 固定参数语义、requirements 与测试后才能公开。

### 15.4 完全自定义算法

控制流差异较大时，直接实现 `RLAlgorithm`：

```python
@ALGORITHMS.register("custom_agent_xpo")
class CustomAgentXPO:
    requirements = AlgorithmRequirements(...)

    def process_trajectories(...):
        ...

    def compute_actor_loss(...):
        ...

    def compute_critic_loss(...):
        ...
```

### 15.5 插件发现

完整 Recipe 支持两种发现方式：

1. Python import path：

```yaml
algorithm:
  factory: my_package.algorithms:build_algorithm
```

2. Python entry point：

```toml
[project.entry-points."rl.algorithm"]
my_xpo = "my_package.algorithms:build_algorithm"
```

框架启动时加载 entry points，但必须对重复名称和版本兼容性进行校验。

## 16. 配置设计

```yaml
algorithm:
  name: grpo

  reward_processor:
    type: group_normalization
    subtract_mean: true
    divide_std: true
    epsilon: 1.0e-6

  advantage:
    type: sequence_reward_broadcast

  objective:
    type: token_clipped
    clip_low: 0.2
    clip_high: 0.2
    dual_clip: 3.0

  regularizers:
    - type: reference_kl
      estimator: k3
      coefficient: 0.001

  reducer:
    type: token_mean

roles:
  actor:
    model_path: /path/to/model
    optimizer:
      lr: 1.0e-6

  reference:
    backend: snapshot

  critic: null

rollout:
  runtime: single_turn
  engine: hyper
  num_samples_per_prompt: 6
  max_new_tokens: 300

trainer:
  mode: sync
  response_mini_batch_size: 3
  response_micro_batch_size: 3
  update_epochs: 1

environment: null
```

PPO 选择完整 Recipe；Critic 由 requirements 自动创建，不由用户手工选择 estimator/objective：

```yaml
algorithm:
  name: ppo
  gamma: 1.0
  gae_lambda: 0.95
  normalize_advantages: true
  clip_ratio: 0.2
  value_clip_ratio: 0.2
  kl_coef: 0.001
  loss_aggregation: token-mean

train:
  critic_update_epochs: 1
  checkpoint:
    # 当前 Critic checkpoint 尚未实现，必须显式关闭。
    save_final: false
    save_steps: 0
    load_path: null
```

Agentic RL：

```yaml
rollout:
  runtime: agentic
  engine: sglang
  max_turns: 16
  timeout_seconds: 600

environment:
  type: code_sandbox
  max_cpu_seconds: 30
  max_memory_mb: 4096
```

## 17. Weight Sync 和策略版本

### 17.1 统一协议

```python
class WeightSyncBackend(Protocol):
    def publish(
        self,
        actor: ActorWorker,
        version: int,
    ) -> WeightHandle:
        ...

    async def update_target(
        self,
        target: GenerationEngine,
        handle: WeightHandle,
    ) -> None:
        ...
```

### 17.2 版本字段

必须同时记录：

```text
TrainerState.model_version
RolloutEngine.model_version
Trajectory.policy_version
Checkpoint.model_version
```

同步训练要求：

```text
Trajectory.policy_version == 当前训练的 old policy version
```

异步训练允许有限 staleness，但算法必须声明是否支持 off-policy 数据。

### 17.3 Weight Sync 后端

- 同进程共享实例；
- Collective Tensor Transfer；
- Full Checkpoint；
- Delta Checkpoint；
- Remote Weight Service。

## 18. Checkpoint 设计

Checkpoint 必须保存完整训练状态：

```python
@dataclass
class RLCheckpointState:
    actor: Any
    actor_optimizer: Any
    actor_scheduler: Any

    critic: Any | None
    critic_optimizer: Any | None
    critic_scheduler: Any | None

    algorithm_state: Any
    trainer_state: Any
    dataloader_state: Any
    buffer_state: Any | None
    rng_state: Any

    model_version: int
    rollout_version: int
    resolved_config: Mapping[str, Any]
```

恢复策略默认 fail-fast：

- 任一必需组件恢复失败时终止；
- 不允许 Actor 已恢复但 Optimizer 静默保持初始状态；
- 算法插件需要声明 checkpoint schema version；
- 允许通过显式配置启用部分恢复。

## 19. Tracking 和可观测性

统一指标命名：

```text
train/actor/*
train/critic/*
algorithm/*
reward/*
rollout/*
environment/*
buffer/*
weight_sync/*
system/*
```

公共指标不使用特定算法名称：

```text
old_policy_kl      # 不叫 ppo_kl
policy_clip_ratio
policy_loss
action_token_count
```

算法专属指标放入：

```text
algorithm/grpo/zero_std_groups
algorithm/ppo/value_clip_fraction
algorithm/gspo/sequence_ratio
```

Agentic RL 额外记录：

- Episode Reward；
- Turn 数；
- Tool Call 数；
- Tool Error/Timeout；
- Token Cost；
- Terminal Reason；
- 环境延迟；
- Reward Components；
- 完整或脱敏后的 Trajectory。

## 20. 错误处理

### 20.1 Role 失败

- 任一分布式 rank 出现 non-finite loss，应通知所有 rank 终止当前 step；
- 不允许单 rank 抛错后其他 rank 永久等待 collective；
- Optimizer Step 前完成全局健康检查。

### 20.2 Environment 失败

Environment 异常转换为结构化终止状态：

```python
terminal_reason = "tool_timeout"
truncated = True
reward_components["timeout"] = penalty
```

可恢复错误不直接破坏整个训练任务。

### 20.3 算法输入校验

Algorithm 在训练前校验：

- Group Size；
- Group 完整性；
- Required Role Outputs；
- Tensor Shape；
- Action Mask 非空；
- Policy Version；
- Advantage/Return 有限性。

## 21. 测试策略

### 21.1 组件单测

- Advantage Estimator 数学；
- Policy Objective 数学；
- KL Estimator；
- Loss Reducer；
- Reward Processor；
- Mask 和 Group 处理。

### 21.2 Algorithm Contract Test

每个算法必须通过统一契约测试：

```text
requirements 与字段一致
输入不被意外修改
输出 shape 正确
loss 可反向传播
无效输入明确失败
指标名称稳定
```

### 21.3 Role Worker Test

- Actor log probability 对齐；
- Critic Value 对齐；
- Reference 无梯度；
- Micro-batch 与 full-batch 梯度一致；
- FSDP global token mean 一致。

### 21.4 Agentic Test

- Action/Observation Mask；
- Tool timeout；
- 最大轮次；
- Environment close；
- Partial Trajectory；
- Reward Components；
- 多轮 Token Offset；
- Policy Version 标注。

### 21.5 端到端矩阵

至少包含：

```text
GRPO + Single Turn + Rule Reward
PPO + Critic + Single Turn
GRPO + Agentic Tool Environment
Async GRPO + Policy Staleness Filter
Checkpoint Save/Resume Determinism
```

## 22. 从当前 Demo 的迁移计划

截至 2026-08-10 的状态：阶段一与统一 Trajectory 已完成；完整 GRPO/PPO Recipe、ExperienceBuilder、按需 Critic、双 Agent runner、Hyper Qwen3.5 vLLM adapter、外部 server、分卡 HCCL 和共卡 DP/NPU IPC refitter 已落地。以下保留为演进检查表。

### 阶段一：建立边界，不改变行为

1. 新建 `algorithms/`；
2. 将旧 `utils/algorithms.py` 移为 GRPO 内部组件；
3. 将 `ActorManager.update()` 拆成 ActorWorker 和 ActorUpdater；
4. 引入 `AlgorithmRequirements`；
5. 将 `ppo_mini_batch_size` 改为 `response_mini_batch_size`；
6. 将 `ppo_epochs` 改为 `policy_update_epochs`；
7. 将 `ppo_kl` 改为 `old_policy_kl`；
8. 保持现有 GRPO 数学和训练结果不变。

### 阶段二：验证 XPO 可扩展性

按顺序实现：

1. GRPO（已完成）；
2. PPO（CPU 契约已完成）；
3. Dr.GRPO；
4. DAPO；
5. GSPO；
6. CISPO。

验收标准：新增算法不修改 Trainer 和 Role Worker。

### 阶段三：引入 Critic

1. 实现 CriticWorker（已完成）；
2. 实现 Qwen Value Model（已完成）；
3. 实现 GAE（已完成）；
4. 实现 Clipped Value Objective（已完成）；
5. 通过 requirements 按需创建 Critic（已完成）；
6. 接入完整 PPO Recipe（CPU 契约已完成）；
7. 完整保存和恢复 Critic 状态（待完成）；
8. Qwen PPO NPU 端到端（待完成）。

### 阶段四：统一 Trajectory

1. 用 `Trajectory` 替换当前 `RolloutBatch` 的业务语义；
2. 引入 `action_mask`；
3. 保留单轮 Rollout Adapter；
4. 引入 Reward Components；
5. 增加 Policy Version。

### 阶段五：Agentic RL

1. Environment Protocol（已完成）；
2. AgentSession（已完成）；
3. 框架控制与用户控制两种 runner（已完成）；
4. Multi-turn Action Mask（已完成）；
5. Tool Router；
6. Timeout/Truncation 生产策略；
7. 真实工具 Agent 端到端测试。

### 阶段六：异步训练

1. Experience Buffer；
2. Rollout Producer；
3. Learner Runtime；
4. Weight Sync；
5. Policy Staleness；
6. Backpressure 和容错。

## 23. 架构决策总结

### 采用

- 将 Actor/Critic/Rollout 能力收拢到 `rl.roles`；
- 公共基础设施复用；
- 完整 Algorithm Recipe 作为公开 API、数学组件只在内部复用；
- Critic 由 requirements 按需创建；
- 单轮和 Agentic RL 使用统一 Trajectory；
- 同步和异步 Trainer 分开；
- Weight Sync 与策略版本独立建模；
- 用户通过注册表开发完整新算法 Recipe。

### 不采用

- 每种 XPO 完整复制一套 Trainer；
- 在 ActorWorker 内部判断具体算法；
- 在一个巨型 `loss.py` 中持续添加算法分支；
- 将所有算法都简单归类为 Advantage Estimator；
- Agentic RL 继续使用单段 Response Mask；
- Checkpoint 部分恢复失败后静默继续训练。

## 24. 最终目标

理想的用户体验应当是：

```text
开发一个新 XPO：
实现完整 Algorithm Recipe
        ↓
在 Recipe 内复用/新增 Advantage + Objective + Regularizer
        ↓
声明 Role/Data Requirements 并注册
        ↓
编写配置和契约测试
        ↓
直接运行现有 Sync/Async Trainer
```

开发 Agentic RL 任务：

```text
选择实现 Environment 或 AgentProgram
        ↓
框架拥有循环 / 用户拥有循环
        ↓
配置 Agentic Rollout Runtime
        ↓
选择现有或自定义 Algorithm
        ↓
生成统一 Trajectory 并训练
```

框架核心保持稳定，算法、环境、推理后端和部署拓扑可以独立演进。
