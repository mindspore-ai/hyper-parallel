# HyperParallel-RL 架构与主项目边界

本文描述 RL 在 HyperParallel 中的位置、模块职责和执行链路。详细接口由各子系统文档维护；
功能、配置、代码符号、指标与测试的对应关系见 [功能导航](rl-navigation.md)。
开发规则见 [Hyper-RL 入口](../.agent/rules/hyper-rl.md)，子系统索引见
[模块归属](../.agent/rules/rl/module-map.md)。下文代码路径均相对仓库根目录。

## 1. 主项目与 RL 的能力边界

HyperParallel 主项目提供分布式模型构建、训练和并行能力；RL 通过模型与并行组件组合自己的同步训练流程。
主项目具有某项能力，不表示 RL 已完成该能力的接入或验收。

| 层次 | 当前职责与边界 | 代码依据 |
| --- | --- | --- |
| 主项目 | DTensor、TP、FSDP/HSDP、CP、EP、PP、checkpoint 等基础能力；各模块后端范围分别维护 | `hyper_parallel/core/`、`hyper_parallel/distributed/`、`hyper_parallel/platform/` |
| RL 运行时 | 同步在线 GRPO/PPO；当前模型注册只接受 Qwen3 dense，公开配方面向单节点 Ascend NPU | `hyper_parallel/rl/rl/trainer.py`、`hyper_parallel/rl/rl/roles/model_setup.py` |
| 训练拓扑 | `dp_replicate=1`、`tp=1` 或 `2`、`cp=pp=ep=1`；`dp_shard` 为正整数 | `hyper_parallel/rl/rl/config.py::_trainer_topology`、`hyper_parallel/rl/rl/config.py::_validate_trainer_ep` |
| Rollout | 一个共享 vLLM 服务，支持 DP 与 TP；colocated 共用训练设备，disjoint 使用独立设备集；dense 模型拒绝 EP/EPLB | `hyper_parallel/rl/rl/config.py::_validate_vllm`、`hyper_parallel/rl/rl/roles/rollout/topology.py` |
| 验证范围 | UT 验证合同及局部计算；真实模型、通信与学习效果由明确的 NPU 配方验证 | [功能盘点](../hyper_parallel/rl/docs/current_feature_inventory.md)、[PPO](../hyper_parallel/rl/docs/ppo.md)、[ST](../hyper_parallel/rl/docs/hyper-rl-st.md) |

算法注册支持扩展，但内置算法为 GRPO/PPO。模型层对 Qwen3 家族身份的接受，不代表任意模型规模、设备或拓扑已验收。
DeepSeek Harness 是 Agent 程序接入方式，不表示支持 DeepSeek-V3 模型。多节点、MoE、异步/off-policy 等能力
不应从主项目接口或外部目录中的文档推断为当前 RL 已支持。

## 2. 模块位置与职责

| 层次 | 路径 | 职责 |
| --- | --- | --- |
| 模型加载与注册 | `hyper_parallel/models/`、`hyper_parallel/models/_transformers/` | 主项目模型注册、Transformers 模型加载与替换 |
| Qwen3 模型适配 | `hyper_parallel/models/qwen3/adapter/` | 模型身份、算子替换和 attention 接口；主项目通过通用 AutoModel 构建 |
| RL Qwen3 构建兼容 | `hyper_parallel/rl/rl/roles/qwen3_builder.py` | RL 使用的参数绑定、物化、并行适配和默认算子选择 |
| 并行策略应用 | `hyper_parallel/distributed/` | 并行布局、策略构建与应用；RL 复用主项目分布式设置 |
| 并行基础能力 | `hyper_parallel/core/`、`hyper_parallel/collectives/` | DTensor、分片、通信与进程组等基础实现 |
| 高性能模块 | `hyper_parallel/components/` | 模型可组合的函数及模块 |
| 主项目 Trainer | `hyper_parallel/trainer/` | 通用配置、优化器及训练组件；RL 拥有独立的训练主循环 |
| RL 配置与编排 | `hyper_parallel/rl/rl/config.py`、`hyper_parallel/rl/rl/trainer.py` | 校验 YAML、构造运行配置、编排 rollout、更新、发布、评估与保存 |
| RL 退出清理 | `hyper_parallel/rl/rl/process_cleanup.py` | 关闭 tracker、rollout 服务，销毁进程组并清理分布式缓存 |
| RL 算法与角色 | `hyper_parallel/rl/rl/algorithm/`、`hyper_parallel/rl/rl/roles/policy/` | GRPO/PPO 目标与损失；Actor/Reference/Critic 持有模型和计算职责 |
| RL 数据与交互 | `hyper_parallel/rl/rl/dataset/`、`hyper_parallel/rl/rl/agentic/` | prompt、trajectory、token mask、经验批次、环境、工具与外部 Agent 程序 |
| RL 生成与发布 | `hyper_parallel/rl/rl/roles/rollout/`、`hyper_parallel/rl/rl/roles/weight_sync/` | vLLM 服务、模型适配、IPC/HCCL 传输及策略版本提交 |
| RL 持久化与观测 | `hyper_parallel/rl/rl/checkpoint.py`、`hyper_parallel/rl/rl/evaluation.py`、`hyper_parallel/rl/rl/utils/monitoring/` | 保存恢复、独立评估、指标及 console/W&B 输出 |

Qwen3 主项目构建使用通用 `HyperAutoModelForCausalLM` 和 `models/qwen3/adapter/`。
RL 的实例级构建兼容位于 `hyper_parallel/rl/rl/roles/qwen3_builder.py`，价值头和 Critic 工厂位于
`hyper_parallel/rl/rl/roles/policy/critic.py`。当前 rollout 的 Hyper Qwen3 实现在
`hyper_parallel/rl/rl/roles/rollout/consistency_models/qwen3/`，由 `vllm_plugin.py` 注册。
主项目调用边界和隔离适配理由见 [Qwen3 适配说明](../hyper_parallel/rl/docs/qwen3_master_adaptation.md)。

## 3. 一步同步训练如何执行

```text
YAML → 配置校验 → 分布式设置、角色、数据源与 vLLM 服务
                           ↓
prompt → rollout / Agentic 交互 → ExperienceBatch（含 rollout old_log_probs）
                           ↓
Reference logprobs + PPO Critic values/bootstrap → 优势与 returns
                           ↓
Actor 更新 → PPO Critic 更新 → 发布 Actor 策略 → 提交 policy_version
                           ↓
指标 / 按保存边界评估 / checkpoint → 下一步 rollout
```

- `SyncTrainer._prepare_experience` 按算法需求计算 Reference logprobs、Critic values 和目标；
  Actor 重算用于诊断及可选的一致性门禁，训练的 `old_log_probs` 仍来自 rollout。
- GRPO 使用分组奖励优势；PPO 已接入价值头、GAE、bootstrap、Actor/Critic 更新与双角色 checkpoint。
  两者当前都需要 Reference。算法计算目标和损失，角色执行反向与优化器步进。
- `SyncTrainer._publish_policy` 发布 Actor 模型。`ActorRolloutWeightSync` 管理训练与生成驻留状态，
  `WeightPublisher` 执行暂停、传输、完成及版本核对；发布异常会向上传播，不自动切换策略。
- `full_gather` 按完整参数组桶，并把完整参数交给 vLLM 加载；单个大参数可以超过配置的桶大小。
  `direct_reshard` 按源/目标布局规划传输。colocated 走 IPC，disjoint 走 HCCL。
  两种策略的合同见 [vLLM rollout](../hyper_parallel/rl/docs/vllm_rollout.md)。
- `RLCheckpointManager` 保存角色、优化器、调度器、数据加载器、步数及 RNG 状态。恢复后训练器按恢复步数
  重新发布策略。评估输出使用 `validation/` 指标，和训练 rollout 的 `reward/` 指标分开。

## 4. Torch 与平台边界

RL 和 `hyper_parallel/models/qwen3/` 直接使用 Torch 与 Torch distributed API，不通过 Platform 或
`get_platform()` 调度。主项目仍处于分模块移除平台抽象的阶段，不能把 RL 的规则扩大到全部主项目模块，
也不能把旧的全局 Platform 禁令重新施加给 Torch 原生组件。

主项目规则以 [AGENTS.md](../AGENTS.md)、[代码风格](../.agent/rules/code-style.md) 和
[分布式规则](../.agent/rules/distributed.md) 为准。尤其是 Multicore、Pipeline 和 DFunction 已有明确的
Torch 原生规则；RL 文档不另行声明它们必须经过 Platform，也不引用旧的 lint baseline 作为现行门禁。

## 5. 入口、安装与测试

- 运行入口：`hyper_parallel/rl/train_rl.py`；配方：`hyper_parallel/rl/examples/gsm8k/configs/`。
  环境安装和 vLLM 插件注册见
  [运行镜像](../hyper_parallel/rl/docker/README.md)。`hyper_parallel/rl/` 是 `rl.*` 的源码根，
  不能假设只安装主项目就已满足 vLLM 服务及所有 RL 运行依赖。
- UT：`tests/ut/rl/`，其中 `conftest.py` 定位当前安装包中的 RL 源码，并收集 `agentic_ut.py`。
  这些用例包括真实 CPU 计算和外部依赖替身，不等同于 NPU 全流程验收。
- ST：`tests/torch/rl/test_rl_st.py` 调用 `st_runtime.py`；实际 NPU 配方及配置生成由
  `tests/common/rl_st_cases.py` 共享。该文件同时供独立部署的 UT 读取，UT 不依赖 `tests/torch/` 目录。
- 真实 RL ST 标记为 `level1`、`allcards`，需要指定模型、数据和设备。PR 默认门禁通过不能证明所有 RL ST 配方
  已执行；以 [ST 操作说明](../hyper_parallel/rl/docs/hyper-rl-st.md) 中的条件与结果为准。
- 文档检查：目录校验器仅检查 AGENTS 中的 Skills/Agents 清单；Markdown 链接、代码符号、配置及测试对应关系
  需要单独核对，不能把目录校验通过当作语义验证。
