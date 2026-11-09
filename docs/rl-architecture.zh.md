# HyperParallel-RL 架构与主项目边界

[English](rl-architecture.md) | **简体中文**

主项目负责模型、并行、优化器及 DCP；RL 负责算法、轨迹、奖励与训练调度。
本次重构保留独立发行包 `hyper-parallel-rl` 和顶层导入 `rl`，不改变主项目实现。

`train_rl.main → trainer/build.py` 是组装入口。
同步、分离异步和共卡异步循环位于 `trainer/`，分别持有明确的组件依赖，不继承通用角色运行时。
配置拆分为 schema、resolve、limits 和 validate；模型替换规则直接在本次 RL YAML 中声明。
数据合同在 `data/`，算法和角色在 `algorithm/`、`workers/`，生成与权重发布在 `rollout/`，CPU 异步协议在 `async_runtime/`。

仅保留 native-vLLM 和流式 full_gather，按部署选择 IPC/HCCL。
旧 consistency、Hyper 推理模型和 direct_reshard 已删除；新版 checkpoint 显式采用 schema 2，不兼容历史快照。
GRPO/PPO/GSPO、MoE、RM、Teacher、Agent 及两种异步模式继续遵守原有支持矩阵；目录迁移不扩展支持范围。

详细顺序与状态合同见[运行时架构](../hyper_parallel/rl/docs/architecture.zh.md)，
配置及验证入口见[功能导航](rl-navigation.zh.md)，开发边界见[RL 规则](../.agent/rules/hyper-rl.md)。
CPU 合同验证、设备短流程和学习效果应分别记录。
