# Hyper-RL 代码结构重构方案

> 状态：架构决策已确认，尚未执行任何 Python 代码重构。
>
> 范围：`hyper_parallel/rl/rl`。
>
> 核心约束：保持现有 GRPO/PPO 数学、Qwen3.5 + GSM8K 训练行为、vLLM rollout、权重同步、checkpoint、日志指标和 YAML schema 不变。

## 1. 重构目标

本轮重构解决代码职责和阅读路径问题，不重新设计训练功能。

1. 每个模块只有一个清晰的主要职责。
2. 开发者从 `trainer.py` 顶部即可顺序读完强化学习训练流程。
3. 新增一个使用现有 Actor/Reference/Critic 能力的算法时，不需要修改 trainer 主循环。
4. 删除当前实施范围内没有实际价值的重复 registry 和兼容性代码；agentic/rollout 相关清理延后评估。
5. 不因为单纯行数较长而拆散内部高度内聚的实现。
6. 每个迁移阶段都可独立测试和回退，避免一次性大规模搬迁。

## 2. 当前基线

当前 `rl` 包含 45 个 Python 文件，约 8153 个物理行。

主要大文件：

| 文件 | 行数 | 当前主要职责 |
| --- | ---: | --- |
| `trainer.py` | 1563 | 配置、构建、训练、评估、checkpoint、显存驻留、日志 |
| `roles/weight_sync/transfer.py` | 737 | CPU/HCCL/NPU IPC 权重传输 |
| `roles/rollout/vllm_qwen3_5.py` | 674 | Hyper Qwen3.5 的 vLLM 模型适配 |
| `roles/rollout/vllm.py` | 636 | vLLM client、进程启动、生成和生命周期 |
| `algorithm/loss.py` | 560 | 协议、requirements、registry、loss、GRPO/PPO |
| `roles/policy/actor.py` | 315 | Actor forward、log-prob、反向和更新 |

当前完整 CPU 回归基线为 `107 passed, 16 warnings`，当前约定 smoke 为：

```bash
cd /home/mwl/project/hyper/hyper-rl
HYPER_VLLM_TIMEOUT_SECONDS=3600 \
./hyper_parallel/rl/examples/scripts/run_qwen3_5_vllm_docker.sh \
grpo-colocated-dp-smoke
```

## 3. 当前正确且应保留的设计

以下部分不应在结构重构中推翻：

1. `ExperienceBatch` 是 immutable 的跨角色训练数据契约。
2. `RLAlgorithm` 只负责 requirements、target 和 loss 数学，不拥有模型或 optimizer。
3. Reference log probabilities 由冻结 `Actor` 实例计算。
4. Critic values 由 Critic role 计算。
5. Trainer 显式编排各 role，不由 dataset 或 algorithm 发起模型 forward。
6. Actor 保持单类结构，按 `sequence_log_probs -> compute_log_probs -> forward_backward -> update` 阅读。
7. Rollout 只使用 vLLM backend，vLLM 内仍支持 `model_implementation=hyper|native`。
8. Policy Actor 到 rollout 的权重同步继续由 `roles/weight_sync` 独立负责。
9. Monitoring backend 不进入 Actor/Critic 的优化控制流程。

## 4. 主要结构问题

### 4.1 Trainer 职责过载

`SyncTrainer` 从第 371 行延续到第 1563 行，`train()` 位于第 1418 行。开发者必须先越过配置校验、对象构建、评估、checkpoint 和显存处理，才能看到真正的强化学习流程。

当前 Trainer 同时承担：

- YAML 配置结构与语义校验。
- Hyper-Parallel `HyperTrainerConfig` 转换。
- Tokenizer、dataset 和 dataloader 构建。
- Actor、Reference、Critic 和 optimizer 构建。
- vLLM 与 agent rollout 构建。
- GRPO/PPO step 编排。
- 分布式 validation。
- rollout metrics 和 learning gate。
- checkpoint manifest、resume 和 reload 验证。
- colocated training/rollout 显存驻留切换。
- tracker 和进程组清理。

### 4.2 Dataset Batch Builder 的跨层职责（本轮保留现状）

`dataset/batch_builder.py` 同时 import：

- `rl.algorithm.RLAlgorithm`
- `rl.roles.rollout.base.GenerationSettings`

这意味着 `batch_builder.py` 不是纯数据源模块，而是训练 batch assembly 模块。根据本次审阅意见，本轮接受这一定位：继续同时保留 `build_experience_batch()` 和 `ExperiencePreparer`，不新增 `experience.py`，也不调整其公开接口。

为避免扩大这项例外，`dataset/contracts.py` 和 `dataset/data_source.py` 仍保持底层数据职责；只有 `dataset/batch_builder.py` 可以依赖 algorithm 和 rollout contract。

### 4.3 Agentic 与 Rollout 有重复包装（后续阶段再评估）

`roles/rollout/worker.py` 的 `RolloutManager` 只负责构造 `GenerationSettings`，随后全部委托给 `AgentRunner`。

同时：

- `RolloutManager -> AgentRunner`
- `AgentRunner -> roles.rollout.base`

虽然当前没有形成 Python import SCC，但概念依赖是双向的，而且多了一层 facade。根据本次审阅意见，本轮不删除 `RolloutManager`，不移动 `AgentRunner`，不调整 agentic/rollout 的目录或调用接口，仅把这个问题记录为后续候选项。

### 4.4 Weight Sync 存在真实循环依赖

当前存在：

```text
weight_sync.transfer -> weight_sync.sync
weight_sync.sync -> weight_sync.transfer
```

原因是 `transfer.py` 使用 snapshot/client/error contract，而 `sync.py` 又在运行时 import fingerprint 验证函数。

### 4.5 扩展接口与实际能力不完全一致

- `RoleRequirements` 中 `actor/rollout/reward_model/environment` 当前没有调用者。
- `DataRequirements.action_mask` 是固定数据契约，不是可选 requirement。
- Reward 已有 registry，但 GSM8K environment 直接调用具体 reward 函数。
- `ModelRegistry` 只注册当前 YAML 中的一个模型，项目中没有通过 registry 查找模型。
- `algorithm/loss.py` 中 GRPO/PPO 的 Actor loss 组装有可消除的重复。

### 4.6 Barrel import 和兼容层扩大耦合

内部代码多处通过以下包级入口互相导入：

```python
from rl.algorithm import ...
from rl.dataset import ...
from rl.roles import ...
from rl.agentic import ...
from rl.utils.monitoring import ...
```

这些 import 会加载整个 `__init__.py`，把 registration side effect 和内部依赖隐藏起来。

当前冗余兼容层还包括：

- `roles/rollout/vllm_policy.py`
- `CPUStateDictRefitter/HCCLWeightRefitter/NPUIPCWeightRefitter`
- `map_policy_state_dict`
- `refit()` 到 `transfer()` 的兼容入口
- `rl.__init__` 中动态注册的 `rl.contracts` 模块别名

## 5. 参考 MOLT 与 RL2 后的取舍

### 5.1 借鉴 RL2 的部分

RL2 最值得借鉴的是 Trainer 中角色调用顺序直接可见：

```text
rollout
-> reference log-probs
-> critic values
-> advantages
-> actor update
-> critic update
-> rollout weight update
```

Hyper-RL 应保留这种显式编排，不引入通用 pipeline executor、事件总线或隐式 task graph。

### 5.2 不照搬 RL2 的部分

不采用 RL2 的可变 `tensor_dict`、role 内直接 W&B logging，以及由大量 config 分支决定数据字段的方式。当前 immutable contract 和 requirements 机制更适合做可靠扩展。

### 5.3 借鉴 MOLT 的部分

保留当前 MOLT 风格的 decorator registry 和纯算法组件，但不让每个模块重复实现相同 registry 容器。

## 6. 目标目录结构

```text
rl/
├── __init__.py
├── trainer.py                 # RL 主循环、对象组合、生命周期编排
├── config.py                  # 配置校验和 HyperTrainerConfig 转换
├── evaluation.py              # 分布式评估及评估结果汇总
├── registry.py                # 无业务依赖的通用 Registry
├── async_trainer.py           # 保留 fail-fast 异步边界
├── dataset/
│   ├── __init__.py
│   ├── contracts.py           # Prompt/Trajectory/ExperienceBatch
│   ├── data_source.py         # parquet、prompt normalization、collate
│   └── batch_builder.py       # rollout batching + target preparation
├── algorithm/
│   ├── __init__.py
│   ├── advantage.py
│   ├── loss.py
│   └── reward.py
├── agentic/
│   ├── __init__.py
│   ├── base.py
│   ├── registry.py
│   ├── gsm8k.py
│   ├── session.py
│   ├── runner.py
│   └── program_runner.py
├── roles/
│   ├── model.py               # immutable ModelRegistration
│   ├── policy/
│   │   ├── actor.py
│   │   ├── critic.py
│   │   └── value.py
│   ├── rollout/
│   │   ├── base.py
│   │   ├── registry.py
│   │   ├── vllm.py
│   │   ├── vllm_plugin.py
│   │   ├── vllm_policy.py     # 本轮暂不调整
│   │   ├── vllm_qwen3_5.py
│   │   └── worker.py          # RolloutManager，本轮暂不调整
│   └── weight_sync/
│       ├── __init__.py
│       ├── checkpoint.py      # checkpoint、manifest、resume 验证
│       ├── sync.py
│       ├── transfer.py
│       └── vllm_worker.py
└── utils/monitoring/
    ├── config.py
    ├── metrics.py
    ├── tracker.py
    └── backends/
```

## 7. 目标依赖规则

### 7.1 分层方向

以下箭头表示“左侧 import 或调用右侧”：

```text
trainer -> config / evaluation / dataset / algorithm / roles / monitoring

dataset.data_source -> dataset.contracts
dataset.batch_builder -> dataset.contracts / algorithm / roles.rollout.base

roles.policy -> algorithm / dataset.contracts / monitoring
roles.rollout -> roles.weight_sync
roles.weight_sync.transfer -> roles.weight_sync.sync
trainer -> roles.weight_sync.checkpoint
```

`dataset.batch_builder` 是本轮明确保留的跨层训练 batch assembly 模块。`roles.weight_sync` 的职责相应扩展为两部分：在线 Actor-to-rollout 权重同步，以及持久化训练状态的 checkpoint 管理。`checkpoint.py` 与 `sync.py/transfer.py` 共享目录，但不互相调用。

### 7.2 强制规则

1. `dataset/contracts.py` 和 `dataset/data_source.py` 不得 import algorithm、agentic、roles 或 trainer。
2. `dataset/batch_builder.py` 可以依赖 algorithm 和 `roles.rollout.base`，但不得依赖具体 vLLM 实现或 policy role。
3. `algorithm` 不得 import dataset、agentic、roles 或 trainer。
4. `weight_sync.sync/transfer/checkpoint` 不得反向 import rollout。
5. `roles.policy` 不得 import trainer、rollout 或 agentic。
6. `trainer` 是唯一可以组合所有 role 和服务的顶层模块。
7. 本轮修改到的内部模块使用 leaf-module import；agentic/rollout 原有 import 暂不做结构迁移。
8. 新增 architecture test，至少检查无新增 cycle，并消除现有 `weight_sync.sync <-> transfer` SCC。

## 8. Trainer 目标阅读顺序

`trainer.py` 中的方法按开发者阅读顺序排列：

```text
SyncTrainer.__init__
-> train
-> _train_step
-> _complete_step
-> runtime builders
-> BaseTrainer compatibility helpers
-> memory/lifecycle helpers
-> cleanup
```

不新增 `_start_training()`、`_finish_training()`、`_cleanup_training()`、
`_collect_role_outputs()` 或 `_publish_policy()`。训练启动、最终 checkpoint、barrier
和 cleanup 直接保留在 `train()`；role 计算和 policy 发布直接保留在
`_train_step()`。

### 8.1 目标主循环

```python
def train(self) -> None:
    completed = False
    try:
        self.checkpoints.validate_resume()
        self.checkpoints.begin(self.state)
        # checkpoint resume 后的初始 rollout 权重同步直接保留在这里。
        data_iterator = iter(self.train_dataloader)
        while self.state.global_step < self.state.max_steps:
            batch, data_iterator = self._next_batch(data_iterator)
            self._train_step(batch)
        self.checkpoints.finalize(self.state.global_step)
        completed = True
    finally:
        if completed:
            platform.barrier()
        self._cleanup_distributed()
```

### 8.2 目标单步流程

```python
def _train_step(self, batch: Mapping[str, Any]) -> None:
    next_step = self.state.global_step + 1
    rollout = self.rollout_manager.generate(
        prompt_records=self._build_prompt_records(batch),
        policy_version=self.state.global_step,
    )

    self.rollout_engine.prepare_for_training()

    requirements = self.algorithm.requirements.data
    reference_log_probs = None
    if requirements.reference_log_probs:
        reference_log_probs = self.reference_actor.compute_log_probs(rollout)

    values = None
    if requirements.values:
        values = self.critic.compute_values(rollout)

    experience = self.experience_preparer.prepare(
        rollout,
        reference_log_probs=reference_log_probs,
        values=values,
    )

    actor_update = self.actor.update(experience)
    critic_update = (
        self.critic.update(experience)
        if self.critic is not None
        else None
    )

    self.rollout_engine.update_weights(
        PolicySnapshot(
            version=next_step,
            model_name=self.model_registration.name,
            payload=self.actor.actor_model,
            metadata={"optimizer_steps": actor_update.optimizer_steps},
        )
    )
    self._release_training_state_for_rollout()
    self.rollout_engine.prepare_for_rollout()

    self._complete_step(
        step=next_step,
        batch=batch,
        rollout=rollout,
        actor_update=actor_update,
        critic_update=critic_update,
    )
```

`_train_step()` 不返回 tuple 或额外 result dataclass。`_complete_step()` 负责推进
step 状态、metrics、learning gate、日志、按现有策略触发 evaluation 和 checkpoint。
这段代码必须保持显式，不能把 requirements 判断、role 调用顺序或 weight
publication 隐藏到通用 pipeline 对象中。

## 9. 逐模块调整方案

### 9.1 `config.py`

迁入当前 `trainer.py` 顶部的配置辅助和校验：

- `_required_mapping`
- `_optional_mapping`
- `_path_value`
- model/data/evaluation/train/topology/logging/checkpoint 校验
- `_uses_colocated_vllm`
- `HyperTrainerConfig` 转换

调整原则：

1. 不新增 `ResolvedConfig` 等包装类，也不返回复杂 tuple。
2. 提供 `validate_config(config, algorithm)`、`build_base_config(config)`、
   `build_model_registration(config)` 和 `uses_colocated_vllm(config)`。
3. Trainer 只构建一次 algorithm，再把实例传入配置校验。
4. 配置校验本身不创建目录；checkpoint output directory 在
   `RLCheckpointManager` 初始化时创建。
5. vLLM 专属校验按现有语义迁入 `config.py`；本轮不改 rollout registry 或 backend validation 接口。
6. 不新增必填 YAML 字段，不改变现有默认值和错误约束。

### 9.2 `trainer.py`

保留：

- Runtime composition。
- RL 主循环和单步角色调用顺序。
- `train()` 中显式的 resume、初始 rollout 权重同步、final checkpoint、barrier 和 cleanup。
- BaseTrainer 必需的 `self.model/self.optimizer/self.lr_scheduler` 兼容接口。
- Colocated training/rollout phase 切换。
- 最终 cleanup。

迁出：

- 配置校验与转换。
- validation 实现。
- checkpoint manifest 和 reload 验证，迁入 `roles/weight_sync/checkpoint.py`。
- rollout metric 聚合与 sample round-robin。
- learning gate 指标验收，迁入 monitoring。

`train()`、`_train_step()` 和 `_complete_step()` 放在类前部。`_train_step()` 中
直接显示 reference log-probs、critic values、experience preparation、Actor/Critic
update 和 rollout weight publication，不添加隐藏这些调用的 helper。

简化模型构建中的临时别名切换：

```python
def _build_optimizer_for(self, model):
    self.model = model
    self.optimizer = None
    self.lr_scheduler = None
    self._build_optimizer()
    self._build_lr_scheduler()
    return self.optimizer, self.lr_scheduler
```

所有 BaseTrainer 兼容字段在 role 构建结束后统一恢复为 Actor。

### 9.3 `dataset/batch_builder.py`

本轮保持当前结构和行为，不新增 `experience.py`：

- `build_experience_batch()` 继续负责 Trajectory padding、mask、old log-probs、rewards、responses 和 metadata。
- `ExperiencePreparer` 继续接收 algorithm 以及 reference/critic role 输出。
- `ExperiencePreparer.prepare()` 继续校验 requirements、调用 `algorithm.build_targets()`，并返回补齐后的 immutable `ExperienceBatch`。
- 保留当前 `GenerationSettings` 和 `RLAlgorithm` import。
- 不调整函数签名、类名、导出位置或现有测试归属。

本轮只要求其内部代码保持清晰，不做职责迁移。

### 9.4 `agentic` 与 `roles/rollout`（本轮暂不调整）

本轮明确不执行以下原提案：

- 不删除 `roles/rollout/worker.py`。
- 不删除或替换 `RolloutManager`。
- 不让 Trainer 直接持有 `AgentRunner`。
- 不移动 `AgentRunner/AgentSession/ProgramAgentRunner`。
- 不调整 rollout registry、vLLM backend 或 train/eval rollout 构建接口。
- 不删除 `roles/rollout/vllm_policy.py` 或 vLLM 中的 compatibility wrapper。

Trainer 继续使用 `self.rollout_manager.generate()`，Evaluator 继续使用独立的 evaluation `RolloutManager`。该区域仅在本轮其他修改需要更新 import 时做最小适配，不进行结构重构。

### 9.5 `roles/policy`

Actor 当前结构保持不拆分。

将 `CriticModel + CriticManager` 合并为单个 `Critic(platform.Module)`，与 Actor 对齐：

```text
__init__
-> sequence_values
-> compute_values
-> forward_backward
-> update
```

`value.py` 只保留模型特定能力：

- `attach_value_head()`
- Qwen3.5 final hidden state/value head 适配

`Critic` 持有 `critic_model`、optimizer 和 scheduler；原
`CriticModel.sequence_values()` 与梯度同步逻辑迁入 `Critic`。删除
`CriticModel`、`CriticManager` 旧名称，不保留兼容别名。

不提取通用 Actor/Critic optimizer 基类。当前少量重复能让两个训练流程保持可直接阅读。

### 9.6 `algorithm`

继续保留已约定的四个文件：

```text
advantage.py
loss.py
reward.py
__init__.py
```

不恢复 `base.py/components/grpo.py/ppo.py`。

内部优化：

1. `loss.py` 按 protocol/registry/common loss/GRPO/PPO 分区。
2. 抽取 GRPO/PPO 共用的 clipped objective + reference KL 结果组装函数。
3. `RoleRequirements` 只保留 Trainer 真正支持的 optional roles。
4. `DataRequirements` 只保留真正可选的数据依赖。
5. 保留 `register_algorithm/register_policy_loss/register_advantage_estimator/register_reward` 外部扩展形式。
6. `__init__.py` 只导出稳定扩展 API，不导出所有内部 helper。

确认精简后的 requirements：

```python
@dataclass(frozen=True)
class RoleRequirements:
    reference: bool = False
    critic: bool = False

@dataclass(frozen=True)
class DataRequirements:
    rollout_log_probs: bool = True
    reference_log_probs: bool = False
    values: bool = False
    returns: bool = False
    grouped_responses: bool = False
```

### 9.7 `registry.py`

提供一个约 40 到 60 行的 typed registry，统一以下重复机制：

- algorithm builders
- advantage estimators
- policy losses
- rewards

Environment 和 rollout engine registry 本轮保持原实现，不迁移到通用 Registry。

每个业务模块继续保留自己语义化的 decorator，例如：

```python
def register_reward(name):
    return REWARDS.register(name)
```

通用 Registry 只负责：

- key normalization。
- duplicate validation。
- unknown-key error。
- deterministic names。
- builder lookup/build。

不把模型注册放进这个通用 Registry。当前单模型配置直接解析成 immutable `ModelRegistration` 即可。
删除项目内部未被查询使用的 `ModelRegistry`、`MODEL_REGISTRY` 和
`register_configured_model()`；这不影响 `vllm_plugin.py` 使用 vLLM 自己的
`ModelRegistry.register_model()`。

### 9.8 Reward 与 Environment（本轮暂不调整调用关系）

`algorithm/reward.py` 可以复用通用 Registry，但 GSM8K Environment 本轮继续直接调用现有 reward 函数，不改 agentic 调用关系。

后续调整 agentic 时，再考虑改为：

```python
reward_fn = get_reward("gsm8k")
reward = reward_fn(action.content, ground_truth)
```

`extract_answer()` 仍可直接用于 metadata，因为它不是 reward selection。

该变化不属于本轮实施范围。

### 9.9 `roles/weight_sync`

按本次审阅意见，将 checkpoint 管理放入该目录：

```text
checkpoint.py
sync.py
transfer.py
vllm_worker.py
__init__.py
```

该目录的模块定义扩展为“训练权重与状态生命周期”：

- `checkpoint.py` 负责持久化 checkpoint、manifest、resume 和 reload 验证。
- `sync.py/transfer.py/vllm_worker.py` 负责在线 Actor-to-rollout 权重同步。
- `checkpoint.py` 不调用在线 transfer，在线 sync 也不调用 checkpoint。

调整依赖：

- `checkpoint.py`：checkpoint save policy、config snapshot、manifest、resume 和 reload verification。
- `sync.py`：snapshot、client contract、同步错误、fingerprint contract、同步生命周期。
- `transfer.py`：CPU/HCCL/NPU IPC concrete transfer 和 builder。
- `vllm_worker.py`：vLLM server worker hooks。
- `transfer.py -> sync.py`，禁止 `sync.py -> transfer.py`。

不因为 `transfer.py` 较长就再次拆成三个 strategy 文件。三种实现属于同一个可替换边界，且用户此前已要求合并整理。

### 9.10 Monitoring

`utils/monitoring/metrics.py` 继续负责：

- Actor/Critic update metrics。
- Actor metric accumulator。
- system metrics。
- public metric key mapping。
- rollout statistics 和 bounded samples 汇总。
- learning gate 检查。

Trainer 通过 `summarize_rollout()`、`build_training_metrics()` 和
`enforce_learning_gate()` 调用这些逻辑，只保留调用时机。

`tracker.py` 只负责 backend fan-out。

Actor/Critic 不直接调用 console、W&B 或 `TrainingTracker`。

### 9.11 `evaluation.py`

新增 `Evaluator`，接收已构建依赖：

- dataset
- collate function
- evaluation `RolloutManager`
- device
- batch/limit/progress config

公开接口：

```python
metrics, samples = evaluator.run(step)
```

它负责当前 Trainer 中的 padded DP batches、generation、local record、all-gather、summary 和 sample selection。
Trainer 构建并持有一个轻量 `Evaluator`，只在现有 checkpoint 保存步骤调用
`evaluator.run(step)`；本轮不增加 `evaluation_steps` 或其他 YAML 字段。

### 9.12 `roles/weight_sync/checkpoint.py`

新增 `RLCheckpointManager`，由 Trainer 组合使用，负责：

- checkpoint save policy。
- config snapshot。
- manifest invalidation/finalization。
- resume path validation。
- reload verification。
- periodic/final checkpoint bookkeeping。

`RLCheckpointManager` 内部复用现有 `CheckpointCallback`，不重新实现 Actor、
optimizer 或 scheduler 的底层保存格式。它在初始化时创建 output directory。
Trainer 只保留 BaseTrainer 需要的 `dispatch_save_event()` 和
`dispatch_load_event()` 转发入口。

Trainer 保留 BaseTrainer callback 要求的少量转发方法，不把 checkpoint 细节留在 RL 主循环中。该文件虽然位于 `weight_sync`，但与在线 rollout weight transfer 保持代码和调用隔离。
colocated 模式的 training/rollout 驻留切换仍由 Trainer 显式执行，
`RLCheckpointManager` 不持有 rollout engine。PPO Critic checkpoint 本轮仍保持
未实现限制，现有报错条件和语义不变。

## 10. 已确认的公共 API 与兼容性决策

### 10.1 数据契约

- 删除 `rl.__init__` 中动态构造的 `rl.contracts` 模块别名。
- 项目内部和测试统一从 `rl.dataset.contracts` 导入。
- 根包仍便捷导出 `ExperienceBatch`、`Message`、`PromptRecord`、
  `Trajectory`、`Turn`，因此 `from rl import ExperienceBatch` 继续可用。

### 10.2 Algorithm API

`algorithm/__init__.py` 只导出稳定的算法构建和扩展接口：

- `build_algorithm` 和语义化的 `register_*` / `get_*`。
- `RLAlgorithm`、requirements 和公开 output 类型。
- `GRPOAlgorithm`、`PPOAlgorithm` 及对应 Config。
- reward 的稳定公共函数。

registry 容器、内部 helper、具体 estimator/objective 和独立 recipe builder
不再作为包级稳定 API，内部代码使用 leaf-module import。

### 10.3 Role API

`rl.roles` 顶层只导出 `Actor`、`Critic` 和 `ModelRegistration`。
`attach_value_head()` 从 `rl.roles.policy.value` 直接导入。删除
`CriticModel`、`CriticManager` 和 `register_configured_model` 旧名称，不保留兼容别名。

### 10.4 本轮保留的兼容入口

- 保留旧 `*Refitter`、`refit()` 和 `map_policy_state_dict`。
- 保留 `roles/rollout/vllm_policy.py`。
- 保留 `async_trainer.py` 及其 fail-fast 测试；不实现异步 runtime。

### 10.5 内部导入规则

项目内部使用定义所在的 leaf module，例如：

```python
from rl.dataset.contracts import ExperienceBatch
from rl.roles.policy.actor import Actor
```

各目录 `__init__.py` 只用于稳定公共 API，不作为内部模块互相依赖的入口。

## 11. 分阶段实施计划

### Phase 0：冻结行为

目标：先把当前行为写成可回归 contract。

操作：

- 保留现有 107 个 CPU 测试基线。
- 增加 config characterization tests。
- 增加 validation summary tests。
- 增加 checkpoint phase/manifest tests。
- 增加 AST import dependency tests。
- 固化 GRPO/PPO trainer role call 顺序。

完成标准：仅新增测试，不改运行实现；全部现有测试继续通过。

### Phase 1：让 Trainer 主流程可读

目标：优先解决入口阅读问题。

操作：

- 新增 `config.py`。
- 新增 `evaluation.py`。
- 新增 `roles/weight_sync/checkpoint.py`。
- 将 `train()` 和 `_train_step()` 移到 `SyncTrainer` 前部。
- 新增 `_complete_step()`；`_train_step()` 返回 `None`，不增加 result 类型。
- 把 metrics/evaluation/checkpoint 委托出 Trainer。
- algorithm 只构建一次。
- 封装 BaseTrainer optimizer 临时别名切换。
- 保持现有 `RolloutManager` 和 `ExperiencePreparer` 调用不变。
- 启动、结束、role 计算和 policy publication 继续在 `train()` / `_train_step()`
  中显式呈现。

完成标准：Trainer 主流程在类定义后约 150 行内可见，调用顺序测试不变。

### Phase 2：统一 Policy Role

目标：Actor/Critic 具有一致、可顺序阅读的角色接口。

操作：

- 合并 `CriticModel + CriticManager -> Critic`。
- `value.py` 只保留 value capability adapter。
- 保留 Actor 现有单类结构。
- 保持 optimizer、micro-batch、DP reduction 和 metrics 行为不变。

完成标准：PPO values/returns/critic update 数值测试不变。

### Phase 3：Algorithm Registry 和局部 import DAG

目标：降低重复代码并锁定单向依赖。

操作：

- 新增通用 `registry.py`。
- 迁移 algorithm、advantage、policy loss 和 reward registries。
- 删除未使用 `ModelRegistry`。
- 本轮修改涉及的内部 import 改为 leaf module。
- 缩小各 `__init__.py` 的 `__all__`。
- 添加 import DAG contract test。
- 不调整 environment registry、rollout registry 或 GSM8K Environment 调用方式。
- 收紧 `algorithm`、`roles` 等包级公共导出，删除 `rl.contracts` 动态别名。

完成标准：algorithm 相关 registry 错误文本、可用 names 和 build 行为保持一致；不在 agentic/rollout 中引入新变化。

### Phase 4：Weight Sync 与 Checkpoint

目标：让在线权重同步保持单向依赖，并把 checkpoint 从 Trainer 迁入指定模块。

操作：

- 将 fingerprint shared contract 归入 `sync.py`。
- 保持 `transfer.py -> sync.py` 单向。
- 完成 `roles/weight_sync/checkpoint.py` 与 Trainer callback 的集成验证。
- 不改 CPU/HCCL/NPU IPC transfer 的执行顺序。
- 保留 rollout/vLLM compatibility wrapper，不执行决策 B/C 的删除项。

完成标准：checkpoint 行为不变；weight sync phase、fingerprint、错误同步和 vLLM worker hooks 测试全部通过；`sync <-> transfer` SCC 消失。

### Phase 5：完整回归与 Smoke

目标：确认结构调整没有改变真实训练功能。

操作：

- 完整 CPU tests。
- `git diff --check`。
- import DAG 检查。
- 最终只跑 `grpo-colocated-dp-smoke`。
- 不跑 production。

每个 Phase 完成后运行对应定向测试并更新 `recode.md`，但不自动创建 Git
commit；最终由维护者决定提交边界。

## 12. 必须保持的行为不变量

1. GRPO requirements 仍创建 reference，不创建 critic。
2. PPO requirements 仍创建 reference 和 critic。
3. Rollout old log-probs 必须来自 vLLM generation。
4. Reference Actor inference 使用 no-grad、eval、micro-batch，并恢复原状态。
5. Critic old values inference 使用 no-grad、eval、micro-batch，并恢复原状态。
6. Advantages/returns/reference log-probs/values 在 ExperienceBatch 中保持 detached。
7. Actor 先更新，Critic 后更新，随后才能发布 rollout policy。
8. Policy version 只在成功 weight transfer 后递增。
9. Colocated vLLM sleep/wake/transfer/resume 顺序不变。
10. `train/*`、`critic/*`、`rollout/*`、`validation/*`、`system/*`、`policy/*` 指标 key 不变。
11. Checkpoint 目录、配置快照、manifest 和 resume 语义不变。
12. YAML schema、默认值、smoke 配置和启动命令不变。
13. GSM8K answer extraction 和 strict rule reward 数值不变。
14. PromptDataset 对 HF 与 Megatron/verl parquet 的兼容行为不变。

## 13. 验证计划

### 13.1 定向测试

```bash
python3 -m pytest -q -p no:cacheprovider \
  hyper_parallel/rl/tests/test_trainer_orchestration.py \
  hyper_parallel/rl/tests/test_experience_preparer.py \
  hyper_parallel/rl/tests/test_batch_builder.py \
  hyper_parallel/rl/tests/test_actor.py \
  hyper_parallel/rl/tests/test_critic.py \
  hyper_parallel/rl/tests/test_algorithm_registry.py \
  hyper_parallel/rl/tests/test_monitoring.py \
  hyper_parallel/rl/tests/test_architecture.py
```

测试文件会随类名和文件归属同步重命名，但覆盖行为不减少。

### 13.2 完整 CPU 回归

```bash
python3 -m pytest -q -p no:cacheprovider hyper_parallel/rl/tests
git diff --check
```

### 13.3 最终 Smoke

```bash
cd /home/mwl/project/hyper/hyper-rl
HYPER_VLLM_TIMEOUT_SECONDS=3600 \
./hyper_parallel/rl/examples/scripts/run_qwen3_5_vllm_docker.sh \
grpo-colocated-dp-smoke
```

## 14. 代码量优化

先完成已确认的职责和依赖重构，再统一
统计整个 `rl` 包的物理行数并规划第二轮精简，避免为压行数破坏功能或修改本轮
明确冻结的 agentic/rollout。

本轮仍遵守以下约束：

- 新文件主要承接迁出的既有逻辑，不能靠复制导致总量无意义膨胀。
- 优先删除重复 registry、重复 loss 组装、无效 requirements 和未使用
  `ModelRegistry`。
- 不新增通用 `runtime.py`、pipeline framework、manager 基类或无明确职责的 utils 文件。
- 不仅为行数拆分 `Actor`、`vllm_qwen3_5.py` 或三个 weight transfer strategy。

## 15. 风险控制

1. 每个 Phase 单独验证并记录，不自动创建 Git commit，不跨 Phase 混合数学修改。
2. 文件迁移阶段先做纯移动，再做简化，便于 diff 审查。
3. 不同时修改算法公式和调用结构。
4. 不同时修改 YAML schema 和 config parser。
5. 每次真实修改、增加、删除和验证结果继续追加到 `hyper_parallel/rl/recode.md`。
6. 任一 Phase 不能通过完整 CPU tests 时，不进入下一 Phase。
7. Smoke 仅在所有 CPU tests 和格式检查通过后运行。

## 16. 最终决策清单

- [x] 保留 `dataset/batch_builder.py` 当前职责，不新增 `experience.py`。
- [x] 本轮不调整 `agentic` 与 `roles/rollout` 结构。
- [x] 新增 `config.py`、`evaluation.py`、`registry.py` 和
  `roles/weight_sync/checkpoint.py`。
- [x] checkpoint manager 复用 `CheckpointCallback`，不控制 rollout 生命周期。
- [x] Trainer 阅读顺序为 `train -> _train_step -> _complete_step -> builders/helpers`。
- [x] 不增加 step result、`_start_training()`、`_publish_policy()` 等额外包装。
- [x] reference、critic 和 policy publication 在 `_train_step()` 中直接可见。
- [x] `CriticModel + CriticManager` 合并为 `Critic`，不保留旧别名。
- [x] 通用 Registry 只先服务 algorithm、advantage、policy loss 和 reward。
- [x] 精简 AlgorithmRequirements 未使用或非可选字段。
- [x] 提取 GRPO/PPO 共用 Actor loss 组装，但不新增算法基类。
- [x] 删除未使用的项目级 `ModelRegistry`，不影响 vLLM ModelRegistry。
- [x] 删除 `rl.contracts` 动态别名，保留 `from rl import ...` 根包契约导出。
- [x] 内部使用 leaf-module import，并增加轻量 AST 架构测试。
- [x] monitoring 接管 rollout metrics、样本选择和 learning gate。
- [x] `Evaluator` 仅在现有 checkpoint 保存步骤运行。
- [x] `weight_sync` 消除 `sync <-> transfer` 循环，保持具体传输行为不变。
- [x] 本轮保留旧 Refitter API 和 `roles/rollout/vllm_policy.py`。
- [x] 保留 `async_trainer.py` 及其 fail-fast 测试。
- [x] 维持 PPO Critic checkpoint 未实现限制。
- [x] 按 Phase 0 到 Phase 5 顺序验证，更新 `recode.md`，不自动创建 commit。
- [x] 本轮暂不设置代码量硬指标，完成结构调整后再统一精简。
