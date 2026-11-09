# Hyper-RL 架构

[English](architecture.md) | **简体中文**

## 系统视图

`train_rl.py` 的 `main()` 调用 `rl.config.loader.load_config()` 加载配置。`trainer/build.py` 校验配置并组装资源，按 YAML 选择同步、分离异步或共卡异步循环。
构建对象只在初始化期间存在；循环直接持有所需组件，不继承共享角色运行时。
主项目继续负责模型加载、并行计划、HSDP、优化器和 DCP；RL 保留算法、轨迹、奖励及训练调度。
`trainer/` 表示训练循环与控制流程；`workers/` 表示 Actor/Critic 等执行角色，
其中 `engine/` 负责共享的模型、优化器、驻留和卸载机制。

| 职责 | 代码 |
| --- | --- |
| 类型与基本字段 | `rl/config/schema.py` |
| HP 配置转换、自动容量与支持矩阵 | `rl/config/resolve.py`、`limits.py`、`validate.py` |
| YAML 加载与严格覆盖 | `rl/config/loader.py`；外部入口为 `train_rl.py` |
| Checkpoint 与评估 | `rl/trainer/checkpoint.py`、`rl/trainer/evaluation.py` |
| 工具证据、XML 校验与诊断 | `rl/tool_protocol/protocol.py`、`xml.py`、`diagnostics.py` |
| 共享扩展点注册 | `rl/utils/registry.py` |
| 唯一组装入口 | `rl/trainer/build.py` |
| 三种训练循环 | `rl/trainer/sync.py`、`fully_async.py`、`colocate_async.py` |
| 进度、失败同步、服务清理 | `rl/trainer/state.py`、`distributed.py`、`lifecycle.py` |
| Actor、Critic、固定训练目标 | `rl/workers/actor.py`、`critic.py`、`targets.py` |
| 模型、优化器、驻留 | `rl/workers/engine/` |
| token 分批、DP 重排、模型适配 | `rl/workers/input/`、`adapters/` |
| 轨迹、提示、episode 和经验 | `rl/data/` |
| native-vLLM 推理 | `rl/rollout/` |
| 权重发布及 IPC/HCCL 传输 | `rl/weight_sync/` |
| CPU producer、版本化样本、partial service | `rl/async_runtime/` |

## 控制面

配置先验证算法、模型、设备拓扑和路径，再启动分布式环境。启动失败与训练失败都通过 `RunLifecycle` 清理已拥有的资源，保留原始异常。
`RewardScorer` 和 `TeacherScorer` 负责真实评分流程；评估回调捕获这些组件，不捕获构建器。

### master 模型接入

`train.plan_overrides` 必须在本次 RL YAML 中显式声明；空列表表示不替换组件。
示例直接列出共享 HP 的 RMSNorm、Attention 或 MoE 替换规则，不在运行时读取另一份训练 recipe，也不按 family 隐式补规则。
`train.reference.plan_overrides`、`train.critic.plan_overrides` 可替换各角色的完整规则列表，独立归一化，不修改 Actor 的计划。
Qwen3 适配仅处理 tied 参数身份、物化缓冲与加载 provenance；不注册自定义 vLLM 模型。

Actor/Critic 共用 `workers/engine/optimizer.py`：同步流、准备主参数梯度、裁剪、执行 step、清理梯度。
每个 optimizer mini-batch 的成员、有效 action-token 归一化和 optimizer 次数不变；scheduler 仍由各角色 update 的既定边界推进。
动态分批复用 HP `FirstFitPackingSelector`，预算按实际 padded rectangle 计费，超预算单行单独成批，不截断。
DP rank 的 microbatch 调用次数对齐，不能通过复制有效训练样本对齐。

训练 CPU offload 与生成阶段 offload 分别配置。`TrainingResidency` 在发布前后重分片、校验优化器状态并同步设备流。
`TrainingStateOffload` 保持参数和 optimizer 对象身份；搬运要求梯度已清理。IPC 接收确认后才释放 bucket 和缓存，再唤醒 rollout。
`sequence_parallel`、`loss_parallel`、`reshard_after_backward`、`requires_grad_sync` 已显式映射到 HP 配置；支持矩阵仍拒绝未验证的值组合。

## 边界合同

### Token-first 不变量

轨迹中的 token IDs 是训练输入的权威来源；禁止 decode 后重新 tokenize。
`sequences`、`attention_mask`、`action_mask` 为 `[batch, tokens]`，概率、优势和 value 目标对齐 next-token 位置。
loss mask 为 `action_mask[:, 1:]`。Prompt、padding、工具观察及 EOS 之后的 token 不参与 policy loss。
episode/call ID、worker policy version、行为概率和完整上下文随重排一起迁移。

### Agent 边界

internal runner 驱动环境与工具，Codex/DeepSeek harness 保留多轮程序和逐调用轨迹。
完整 episode 奖励可以附着到多个调用，但不能把不连续的模型调用拼成一条虚假 token 序列。

### Agentic Harness

任务 prompt、容器工作区和评分函数留在 examples。通用 Agent 层不按任务名分支。
同步 CPU preparation overlap 不改变策略版本，不跨 evaluation、checkpoint 或 epoch 边界预取。

## 一个训练步

1. 按已提交策略生成真实轨迹，完成 RM/Teacher 评分，休眠 rollout 并恢复训练驻留。
2. `TargetPreparer` 按算法需求计算 Actor anchor、Reference 概率、PPO values/bootstrap 和固定目标。
   启用 DP balancing 时，先在完整 GRPO 组上计算目标，再重排，最后计算对应 anchor/reference；不能在残组上重算优势。
3. 更新 Actor 与可选 Critic；发布 Actor 权重并确认所有 worker 的版本，然后推进 global_step、记录指标、评估和保存。

`old_log_prob_source=actor` 强制重算并冻结更新前概率；独立 rollout 概率仍保留用于诊断和蒸馏。
可选诊断本身不改变训练分母。旧 consistency profile 和 bit-exact 门禁已经删除；蒸馏的概率有效性门禁继续保留。

## 状态所有权

Actor/Critic 各自拥有模型、optimizer 和 scheduler。进度属于训练循环，worker 发布版本属于权重同步控制器。
`TargetPreparer` 只持有目标计算依赖，`TrainingResidency` 只负责存储生命周期，`PolicyPublication` 负责训练到推理的切换顺序。
`CheckpointResources` 明确列出持久化所需资源，不持有 trainer、rollout 或构建器回引用。

## 两种 Deployment

colocated 使用 NPU IPC，disjoint 使用 HCCL；具体模型和并行组合以配置校验为准。
仅支持 native-vLLM 和流式 full_gather。`model_implementation=hyper`、`direct_reshard`、顶层 `consistency` 都会明确报错。

## 权重发布与失败原子性

发布依次准备布局、暂停 admission、开始更新、逐 bucket 传输、完成加载、核验 committed version。
仅保留一个 full-gather 发布实现。一次最多有一个 bucket 在途，收到同步复制确认后释放临时存储。
单个参数可能超过 bucket 大小；该参数独占一个 bucket。统计分别记录最大 bucket、确认和释放数量。
任一阶段失败不提交新版本、不恢复采样，也不隐式切换传输策略。

## Checkpoint 与恢复

新格式 `schema=2`。不兼容历史 checkpoint；缺失或不匹配的版本在加载模型前报错。
HP DCP 保存模型及 rank-local optimizer、scheduler、dataloader 和 RNG；完成标记最后提交。
恢复保持 FP32 主参数精度、并行拓扑、Critic 所有权及 Teacher/RM 身份校验；恢复 RNG 放在其他状态加载之后。
异步完成标记链接样本状态校验和和配置指纹，样本使用版本化字段表示，不保存 RL 类的 Python 导入路径。
恢复要求 producer 已 drain、没有在途或 selected 未提交组；损坏和不兼容快照不能被悄悄接受。

## 扩展点

算法需求驱动可选角色和目标。任务扩展使用现有 Agent/environment 接口。
新增模型必须明确验证训练映射、native 权重加载、完整轨迹和支持矩阵；不预留 IsoExec、多后端或已删除策略的占位实现。

## 演进边界（规划）

IsoExec 不属于本次实现。当前代码与配置可运行不代表新组合或学习效果已经完成设备验收。

## 阅读代码

从 `trainer/build.py` 进入对应循环，再阅读 `workers/targets.py`、Actor/Critic update、`weight_sync/publication.py` 和 checkpoint。
配置、源码、示例与测试索引见[功能导航](../../../docs/rl-navigation.zh.md)。

## Rayless asynchronous GRPO

分离异步循环在训练线程执行 optimizer 和发布 collective；CPU producer 负责完整候选组。
先完成 update，才提交消费额度；随后按 freshness 或发布间隔更新策略。
`policy_version = source_update + 1`，不能用 global_step 直接替代发布版本。缓冲容量计入 selected 组，完整组 admission、lag 和重试上限不变。
仅保留现有 GRPO、模型及拓扑支持范围，不扩展为异步 PPO。

## 共卡异步 GRPO

共卡循环显式执行 pause/drain、推理休眠、训练驻留、更新、发布和 partial resume。
失败不能误提交 lease 或版本。Codex 评估前 producer 释放 gateway，评估后关闭评估 gateway，再按恢复快照重启 producer。

### 共卡 off-policy 校正

校正保持默认关闭，启用时遵守现有算法支持范围和逐行为段概率合同。
拒绝的组不进入 optimizer，训练 batch 在恢复 rollout 前释放。

### 共卡发布与恢复

保存前 drain 完整组并冻结 source，模型 checkpoint 和异步快照提交为同一成功更新边界。
恢复后先重新发布实际 Actor，再放行 producer；不得复用旧进程中的推理版本状态。
