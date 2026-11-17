# Unified Execution

HyperParallel 与 vLLM 复用模型、参数语义和关键数值计算，各自保留训练和推理运行时。
实现说明集中在两篇文档：

- [统一执行核心流程](../../docs/unified_execution_guide.md)：模型加载与替换、执行契约、Level 3、PIK、
  Attention、CP/PCP、权重切分与发布，以及支持范围。
- [MoE 与 EP 实现](../../docs/unified_execution_moe.md)：数值合并规则、训练动态路由、推理 MC2/AllGather、
  图执行、容量和专家所有权。

[运行两份配置及不同拓扑的验证矩阵](../../examples/consistency/README.md)。

## 代码导航

| 目录 | 职责 |
| --- | --- |
| [core/](core/) | 执行契约、显式配方、进程数值策略、公共结构校验 |
| [models/](models/) | 模型组成与共享 PIK/专家/router 绑定 |
| [ops/](ops/) | V4 Attention、PIK、训练 EP 与推理 MC2/AllGather |
| [runtimes/hyper_parallel/](runtimes/hyper_parallel/) | HP 构建、训练 Attention/CP、可微 EP 接入 |
| [runtimes/vllm/](runtimes/vllm/) | 推理协议、并行域、MoE 图执行、分页 Attention、权重加载 |

共用绑定不导入运行时。模型完整 forward、参数切分与布局仍由 HP 负责；调度、KV cache 和图回放
仍由 vLLM 负责。各运行时向公共结构校验提供自己的 EP 角色检查。
