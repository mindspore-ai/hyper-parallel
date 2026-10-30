# HyperParallel-RL 架构与主项目边界

本文维护模块职责与执行关系；[功能导航](rl-navigation.md)提供配置、代码和测试映射，
[模块索引](../.agent/rules/rl/module-map.md)定位文件，[RL 规则](../.agent/rules/hyper-rl.md)约束开发流程。
路径相对 `hyper_parallel/rl/`，明确写出的 `hyper_parallel/` 路径相对仓库根目录。

## 主项目与 RL 的分工

| 层次 | 职责 | 主要位置 |
| --- | --- | --- |
| 主项目模型与分布式能力 | 模型加载、算子替换、DTensor、TP、FSDP/HSDP、EP/CP/PP、优化器和 checkpoint 基础能力 | `hyper_parallel/models/`、`hyper_parallel/core/`、`hyper_parallel/distributed/`、`hyper_parallel/trainer/` |
| RL 角色运行时 | 构造 Actor/Reference/Critic，准备经验、更新角色、管理驻留和发布 | `rl/trainer.py::RLRoleRuntime`、`rl/roles/` |
| 同步编排 | 按已发布策略生成，更新后发布，再提交训练步 | `rl/trainer.py::SyncTrainer` |
| 异步编排 | 分离生成与训练设备，管理完整组缓冲、策略滞后、发布与恢复 | `rl/async_training/` |
| 算法与数据 | GRPO/PPO/GSPO、可选蒸馏目标；真实 token、动作 mask、episode/call 身份 | `rl/algorithm/`、`rl/dataset/` |
| 生成与发布 | vLLM DP/TP 服务、模型适配、权重传输和版本核对 | `rl/roles/rollout/`、`rl/weight_sync/` |
| Agent 与任务 | 通用协议、环境和程序执行；任务 prompt、工作区及评分留在示例 | `rl/agentic/`、`examples/` |

主项目能力不等于 RL 已完成接入。模型、算法和拓扑由 `rl/config.py` 校验，
验证范围由[支持说明](../hyper_parallel/rl/README.md#支持范围)及各任务文档维护。
DeepSeek Harness 表示程序接入方式，不代表支持 DeepSeek-V3 模型。
主项目与 RL 均使用 Torch；共享模块仍遵循 [AGENTS.md](../AGENTS.md) 和各模块规则。

## 同步训练链路

```text
train_rl.py → 配置校验、RLRoleRuntime 与 SyncTrainer
    ↓
PromptRecord → rollout / Agent 交互 → 真实轨迹与 ExperienceBatch
    ↓
按配置选择 old log-prob → Reference / PPO values → 优势与目标
    ↓
Actor / PPO Critic 更新 → Actor 权重发布与版本确认
    ↓
提交 global_step → 指标、评估、checkpoint → 下一轮
```

- `train.old_log_prob_source` 默认 `rollout`；设为 `actor` 时，更新前重算并冻结训练侧概率。
  日志诊断本身不改变训练分母；独立采样概率用于训推差异分析，缺失时不能构造差异。
  GSM8K 正式配方使用 Actor 重算；蒸馏的重算与教师概率见[蒸馏合同](../hyper_parallel/rl/examples/distillation/README.md)。
- 算法声明角色和目标需求；Reference 按需求启用，PPO 另计算 values、bootstrap、GAE 和 Critic 更新。
  动态 token 分批改变微批组织，不改变优化器 mini-batch 成员；超预算单条成批不保证一定能放入显存。
- 训练 CPU offload 与生成阶段卸载独立。`phase_offload.py` 复用 HSDP 转换，
  `ActorRolloutWeightSync` 协调推理睡眠和训练恢复；详细顺序见[驻留合同](../hyper_parallel/rl/docs/architecture.md#master-模型接入)。
- 发布先阻止新请求，再传输、核验和恢复服务；异常不自动回退。
  `full_gather`/`direct_reshard` 与 colocated IPC/disjoint HCCL 的组合见[vLLM 合同](../hyper_parallel/rl/docs/vllm_rollout.md)。
- checkpoint 保存训练角色、优化器、数据位置与 RNG；恢复后重新发布策略。
  训练指标 `reward/` 与评估指标 `validation/` 分开，评估间隔可独立于 checkpoint。

## 异步执行与同步准备重叠

`examples/async_training/launch.py` 启动 `FullyAsyncTrainer`，复用 `RLRoleRuntime`，不修改同步循环。
当前配置校验限定 Qwen3 dense GRPO、disjoint vLLM、internal 单轮或 Codex 程序；
拒绝 PPO、MoE、Qwen3.5、内部多轮、RM、蒸馏及同步 consistency，Codex 内联评估也须关闭。
完整组准入、原始行为概率、逐调用版本、策略滞后、发布排空和 checkpoint 提交由
[异步合同](../hyper_parallel/rl/docs/architecture.md#rayless-asynchronous-grpo)定义；实现存在不等于学习效果或多节点已验收。

`agentic.codex.overlap_preparation` 则仍属于同步模式：`SyncTrainer` 预取一批 CPU 准备，
`ProgramAgentRunner` 管理消费与取消，Gateway 等目标策略发布后才放行模型请求。
它不跨评估、checkpoint 或 epoch 边界预取，也不等于异步采样训练。

## 模型与任务边界

Qwen3 的 RL 构建兼容位于 `rl/roles/qwen3_builder.py`，主项目调用仍使用公共 AutoModel。
Qwen3/Qwen3.5 的 RL 概率投影与重计算统一放在 `rl/roles/policy/token_log_probs*.py`。
RM 管理冻结评分服务，OPD/MOPD 管理外部 Teacher；二者不参与学生 optimizer 或 Actor 权重发布。

Search-R1 与 SWE-Smith 复用 `DockerWorkspace` 和 `codex/container_execution.py`。
Search-R1 的 `examples/search_r1/container_runtime.py` 负责文章隔离、宿主交接和搜索反馈；
SWE-Smith 保留仓库与独立 grader 生命周期。通用层不按任务名决定奖励，
异常、预算终态与真实 token 的处理见[Agent 合同](../hyper_parallel/rl/docs/agentic_rl.md)。

## 安装与验证

`hyper_parallel/rl/` 是 `rl.*` 的源码根；依赖安装和插件注册见[运行镜像](../hyper_parallel/rl/docker/README.md)。
UT 在 `tests/ut/rl/`，ST 在 `hyper_parallel/rl/tests/st/`，共用配方在 `tests/common/rl_st_cases.py`。
UT 不依赖独立 ST 目录；NPU ST 需显式运行，默认 PR 门禁不代表已完成模型、通信或学习验收。
详见[测试入口](../hyper_parallel/rl/README.md#系统测试)。
