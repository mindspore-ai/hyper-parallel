# Hyper-RL

Hyper-Parallel 原生的 LLM 强化学习运行时。当前以 Qwen3.5 + GRPO 为最小可运行主线，同时保留按需 Critic、vLLM 模型注册和 Agentic RL 扩展边界。

## 状态

| 能力 | 当前状态 |
|---|---|
| Qwen3.5-0.8B + GSM8K + GRPO | 双 Ascend NPU 端到端验证 |
| Hyper 原生训练与生成 | 默认路径 |
| GRPO / PPO | 独立公开 Recipe，内部复用算法组件 |
| Critic | 由 Recipe requirements 按需创建；GRPO 不创建 |
| vLLM | 延迟注册和权重版本/refitter 契约已实现 |
| Agentic RL | 框架控制与用户控制两种 runner，共用 Trajectory |
| 异步、Ray、工具沙箱 | 暂未纳入最小版本 |

## 快速运行

仓库、模型和数据放在同一父目录：

```text
workspace/
├── hyper-parallel/
├── models/Qwen3.5-0.8B/
└── gsm8k/
    ├── train-00000-of-00001.parquet
    └── test-00000-of-00001.parquet
```

快速 smoke 会生成每个 prompt 2 个 response、每条最多 32 tokens，用于验证完整链路：

```bash
cd hyper-parallel
HYPER_RL_CONFIG=/workspace/hyper-parallel/hyper_parallel/rl/examples/configs/local_qwen3_5_0_8b_gsm8k_smoke.yaml \
  ./hyper_parallel/rl/examples/scripts/run_qwen3_5_0_8b_gsm8k_docker.sh
```

有效更新 smoke 默认生成每个 prompt 6 个 response、每条最多 300 tokens：

```bash
./hyper_parallel/rl/examples/scripts/run_qwen3_5_0_8b_gsm8k_docker.sh
```

Docker launcher 会根据仓库目录名生成容器内路径；如果仓库不叫 `hyper-parallel`，可让 `HYPER_RL_CONFIG` 保持未设置，使用自动推导的默认配置。

“6 份 rollout”是 GRPO 的 group size，不是 6 个 worker。快速配置用 2 验证链路，默认配置用 6 提供更有意义的组内相对奖励。

## 结构

```text
hyper_parallel/
└── rl/               # Hyper-RL 工程
    ├── README.md
    ├── rl/           # Python 包
    │   ├── dataset/  # adapters、data source、batch builder
    │   ├── roles/    # Policy、Rollout、WeightSync
    │   ├── algorithm/# GRPO/PPO Recipe、组件与 reward
    │   ├── agentic/  # Environment、AgentRunner、ProgramAgentRunner
    │   ├── utils/    # monitoring 等公共工具
    │   ├── trainer.py
    │   └── async_trainer.py
    ├── examples/
    ├── tests/
    └── docs/
```

核心数据流：

```text
PromptRecord
  → AgentRunner / ProgramAgentRunner
  → Trajectory
  → ExperienceBuilder
  → GRPOAlgorithm / PPOAlgorithm
  → ActorManager (+ CriticManager when required)
```

算法是完整 Recipe，而不是用户任意拼装数学组件。Trainer 只读取 Recipe 的 requirements，因此新增 XPO 一般只需实现并注册 Recipe，不需要修改训练主循环。

## 验证

最终代码已通过：

- 51 项 CPU/契约测试；
- 双卡 Qwen3.5-0.8B + GRPO 快速 smoke；
- 128 个 action token，1 次 optimizer step，峰值分配显存约 7.60 GiB；
- 默认 6×300 配置曾得到非零 reward、policy gradient 和 optimizer update。

短 smoke 的随机 batch 可能全部同奖励，此时 GRPO advantage 与梯度为零是正确结果；验证有效学习信号请运行默认配置。

## 当前边界

PPO/Critic 已覆盖 CPU 数学和能力契约，但尚未声明 Qwen PPO NPU 端到端完成。Critic checkpoint 未接入，因此 PPO 保存/恢复会 fail-fast。vLLM adapter 已注册，但同步 Trainer 会在没有真实部署 refitter 时拒绝多步在线训练，避免把未同步权重误标为新策略版本。

## 文档

- [架构设计](docs/architecture.md)
- [代码结构](docs/code_structure.md)
- [与 Slime、RL2、Molt 对比](docs/comparison.md)
- [Qwen3.5 + GRPO 复现](docs/reproduce_grpo.md)
