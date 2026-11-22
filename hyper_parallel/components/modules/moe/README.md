# MoE 模块与训练特性

本页说明新的 `Gate`、`GroupedExperts` 和共享专家如何组合。
专家接口和张量约定见 [GroupedExperts 模块说明](grouped_experts/README.md)。
**本次保留 Qwen、DeepSeek-V4.1、Kimi 的旧模型路径，不自动替换原生 gate，也不修改其原生辅助损失。**
下面介绍公共 Gate 的开关语义；模型接入新 Gate 时须传入模型级训练配置。
旧 Router 仍按各模型自己的参数映射和实现执行。
仅替换 `mlp.experts` 不会把旧 router 的训练逻辑迁移过来。

### fix-router

[balanced_expert_indices](router.py) 沿用已有 owner-major 均匀目的模式。
全局专家数 `E`、物理 EP owner 数 `P`、`C=E/P`，flatten slot `s` 的目的为：

```text
(s % P) * C + (s // P) % C
```

例如 `E=8,P=4` 为 `0,2,4,6,1,3,5,7`。要求 `P` 整除 `E`。
绑定时 `P` 从现有 EP mesh 取得，包含 TP-extend-EP；未绑定的单卡组件按 `P=1`。

### aux-loss

复用 sequence balancing 公式，每个 logical sequence 计算：
`E * sum(实际 assignment 比例 * 无偏归一化 score 的均值)`，再对非空样本平均。
`token_mask` 排除 padding。普通 `[B,S,H]` 输入按 B 区分样本；
packed/跨 rank 分片输入可传 `sequence_ids` 与 `num_sequences` 明确样本边界。

通过已有 `aux_loss_auto_scale(weights, coeff * loss)` 注入辅助梯度，forward weights 数值不变。
`last_aux_loss` 为 detach 后的原始指标，**不要再次加进 task loss**。
Trainer 原有 `aux_loss_scale_context` / `bind_aux_loss_scale` 继续负责梯度累积和 loss scaling。
