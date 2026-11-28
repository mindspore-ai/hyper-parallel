# Unified Execution

HyperParallel 与 vLLM 复用模型、参数语义和关键数值计算，各自保留训练和推理运行时。
接入新模型时，从[训推一致实现指导大纲](../../docs/unified_execution_design.md)开始。
[Qwen3 统一执行实现](../../docs/unified_execution_guide.md) 沿一次 RL 更新说明模型加载、Attention、TP/SP、CP/PCP、MoE 与权重发布。关注专家计算时可以直接阅读其中的 [MoE 与专家并行](../../docs/unified_execution_guide.md#moe-与专家并行)。

[Dense、MoE 配方及拓扑验证矩阵](../../examples/consistency/README.md)。

## 代码导航

| 入口 | 职责 |
| --- | --- |
| [contract.py](contract.py) · [execution_plan.py](execution_plan.py) · [batch_invariant.py](batch_invariant.py) | 校验配置、解析替换计划、配置进程数值策略 |
| [models/](models/) | 注册模型族和模型类型，两侧建模后用 [structure.py](models/structure.py) 核对实际结构 |
| [modules/](modules/) | YAML 选用的共享 GQA、PIK 行投影及两端调用的 FlashAttention V4 接口 |
| [training/](training/) | 训练 GQA 的 V4/CP 适配，以及 HP 本地专家的空输入处理；推理 EP1 复用后者 |
| [vllm/](vllm/) | worker 建模、并行域、权重加载、PCP 概率，以及分页 Attention 和专家执行 |

训练从 [角色构建](../trainer/build.py) 进入，按配方构造 HP 模型并安装训练适配。
推理从 [编译入口](vllm/entrypoints.py) 进入，[Qwen3 模型适配器](vllm/qwen3.py) 用 [model_builder.py](vllm/model_builder.py) 构造独立的 HP 模型，再绑定 vLLM 的分页缓存和专家运行时。
新增模型族从 [models/registry.py](models/registry.py) 与 [models/qwen3.py](models/qwen3.py) 开始。共享数值实现放在 `modules/`，依赖 vLLM 的实现放在对应的 [attention/](vllm/attention/) 或 [moe/](vllm/moe/) 下。
