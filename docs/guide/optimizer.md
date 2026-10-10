# Optimizer 使用指南

HyperParallel 提供 Muon、Sinkhorn、AdamW 的链式组合和学习率调度器，支持分片优化器与 FSDP/HSDP 集成。

## 核心概念

| 优化器 | 说明 | 适用场景 |
|--------|------|----------|
| AdamW | 标准 AdamW 优化器 | 通用训练 |
| Muon | Momentum-based optimizer | 大模型训练 |
| Sinkhorn | Momentum + 行列归一化 | Embedding 与预测头 |
| ChainedOptimizer | 具名优化器链式组合 | 混合参数组训练 |

Muon 优化器针对矩阵参数组（如 projection 层）使用 momentum-based 更新策略，其余参数使用 AdamW，实现混合优化。

## 接口概览

| 接口 | 说明 |
|------|------|
| `AdamW` | 标准 AdamW 优化器 |
| `Muon` | Momentum-based 优化器 |
| `Sinkhorn` | 带 momentum 的 Sinkhorn 平衡更新 |
| `ChainedOptimizer` | 具名优化器链式组合 |
| `get_hyper_optimizer` | 优化器工厂函数 |

---

## 基础使用

### 1. 使用 get_hyper_optimizer 创建链式优化器

```python
from hyper_parallel.core.optimizer import get_hyper_optimizer

optimizer = get_hyper_optimizer(
    model=model,
    muon_params=muon_param_groups,   # Muon 参数组
    adamw_params=adamw_param_groups,  # AdamW 参数组
    muon_kwargs={"lr": 0.02, "momentum": 0.95},
    adamw_kwargs={"lr": 3e-4, "weight_decay": 0.1},
)
```

`get_hyper_optimizer` 内部创建 Muon 和 AdamW 实例，并用 `ChainedOptimizer` 将它们组合。参数分组：

- `muon_params`：Muon 优化的参数组（通常为大型 projection 参数）
- `adamw_params`：AdamW 优化的参数组（通常为 embedding、bias 等参数）
- 任一参数组为空列表时，对应优化器不创建

### 2. 参数分组示例

```python
# 将模型参数分为 Muon 和 AdamW 两组
muon_params = []
adamw_params = []

for name, param in model.named_parameters():
    if not param.requires_grad:
        continue
    # 大型 projection 层用 Muon
    if any(key in name for key in ["q_proj", "k_proj", "v_proj", "o_proj",
                                    "gate_proj", "up_proj", "down_proj"]):
        muon_params.append(param)
    else:
        # embedding、bias、norm 等用 AdamW
        adamw_params.append(param)

optimizer = get_hyper_optimizer(
    model=model,
    muon_params=muon_params,
    adamw_params=adamw_params,
    muon_kwargs={"lr": 0.02, "momentum": 0.95},
    adamw_kwargs={"lr": 3e-4, "weight_decay": 0.1},
)
```

### 3. 训练循环

```python
for epoch in range(num_epochs):
    for batch in dataloader:
        output = model(batch)
        loss = criterion(output)
        loss.backward()

        # ChainedOptimizer 自动处理两个优化器的 step
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
```

---

## 分片优化器（FSDP/HSDP 集成）

与 FSDP/HSDP 配合使用时，优化器状态自动分片：

```python
from hyper_parallel import fully_shard, init_device_mesh
from hyper_parallel.core.optimizer import get_hyper_optimizer

mesh = init_device_mesh("npu", (dp_size,), mesh_dim_names=("dp",))
model = fully_shard(model, mesh=mesh)

# FSDP 分片后，优化器状态也自动分片
optimizer = get_hyper_optimizer(
    model=model,
    muon_params=muon_param_groups,
    adamw_params=adamw_param_groups,
    ...
)
```

---

## gradient scaling factor + clip_grad

```python
from hyper_parallel import fully_shard, init_device_mesh

mesh = init_device_mesh("npu", (dp_size,), mesh_dim_names=("dp",))
model = fully_shard(model, mesh=mesh)

# 设置梯度缩放因子
model.set_gradient_scaling_factor(scale_factor=0.5)

# clip_grad 增强：对齐 clip_grad_norm_ reduction 与各 grad 的 process group
# 确保在混合并行场景下 clip_grad 正确工作
```

---

## 性能建议

1. **Muon 参数选择**：通常将大型 projection 层参数分配给 Muon，embedding/bias/norm 分配给 AdamW
2. **学习率配置**：Muon lr 通常比 AdamW lr 大（如 0.02 vs 3e-4）
3. **warmup**：建议使用 warmup_steps，从 0 线性增加到目标 lr
4. **weight_decay**：AdamW 的 weight_decay 建议设为 0.1
5. **与 FSDP 配合**：FSDP 下优化器状态自动分片，无需额外配置

## DeepSeek V4.1：Muon、head-wise Muon 与 Sinkhorn

以下配置对应 DeepSeek-V4.1 报告 §2.5 的优化器组合。
新配置接口按模型参数自动分组，旧的显式参数组接口仍可使用：

```python
optimizer = get_hyper_optimizer(
    model,
    muon={
        "lr": 2.6e-4,
        "matched_adamw_rms": 0.18,
        "head_wise": True,
        "head_dim": 128,  # 按实际模型设置；未提供时尝试从 model.config 推导
    },
    sinkhorn={"lr": 2.6e-4},
    adamw={"lr": 2.6e-4, "betas": (0.9, 0.95), "eps": 1e-20, "weight_decay": 0.1},
)
```

- `None` 表示禁用该优化器，`{}` 表示启用并使用默认配置。
- Sinkhorn 默认接收 `nn.Embedding.weight` 和 `lm_head`、`output`、`output_layer` 的权重。
  未启用 Sinkhorn 时，这些参数交给 AdamW。归一化模块的参数交给 AdamW，其他矩阵交给 Muon，其余参数交给 AdamW。
- 冻结参数排除，绑定权重按对象身份去重；显式选择冲突或存在未分配的可训练参数时会报错。
- 每类配置可添加 `param_patterns` 正则列表，显式选择优先于默认分类。
  例如 `sinkhorn={"param_patterns": [r"engram\.tables\..*weight$"]}` 可补充模型专用表。
- AdamW 的 bias、非归一化模块的向量/标量及名称为 `scale`、`scales`、`scaling_factor` 的参数不做衰减；
  normalization weight 使用配置的衰减。

Sinkhorn 默认使用 `momentum=0.95`、`correction=0.18`、`steps=11`、`tau=1e-3`、`eps=1e-20`，
不做 weight decay。矩阵方向必须是 `[token_count, feature_count]`，不按长边自动转置。
行/列分片只归约统计量；梯度须已完成归约，同一矩阵各 shard 的梯度存在性须一致。
不支持 sparse gradient 或 Partial placement。

需要 Engram 独立的 5 倍学习率时，可使用显式参数组：

```python
optimizer = get_hyper_optimizer(
    model,
    muon_params=muon_groups,
    adamw_params=adamw_groups,
    sinkhorn_params=[
        {"params": token_and_head_parameters, "lr": base_lr},
        {"params": engram_table_parameters, "lr": 5 * base_lr},
    ],
    muon_kwargs={"head_wise": True, "head_dim": 128, "matched_adamw_rms": 0.18},
    sinkhorn_kwargs={"lr": base_lr},
)
```

自动配置字典不能与 `*_params` / `*_kwargs` 混用。

Head-wise 默认匹配独立的 `q_proj`、`k_proj`、`q_b_proj`、`wq`、`wk` 的 `.weight`，
将 `[heads * head_dim, input_dim]` 看作多个独立矩阵。自定义名称或不同 Q/K 维度可替换匹配规则：

```python
muon_config = {
    "head_wise": True,
    "head_wise_patterns": {
        r"attention\.query\.weight$": 128,
        r"attention\.key\.weight$": 64,
    },
}
```

每个被选参数只能匹配一条规则，行数必须能整除对应的 head dimension。
融合 QKV、交错存储等特殊布局请显式使用已有 `ns_transform_fn`；它与内置 `head_wise` 不同时启用。
直接构造 `Muon` 时需给参数绑定 `model_name`，工厂会自动绑定。
恢复训练时需使用相同的 head-wise 配置重新构造优化器，正则与 head dimension 属于运行时布局配置。

`ChainedOptimizer` 保留 `{name: optimizer}` 接口，支持任意数量的互斥优化器，
无需额外传递重复的参数组列表。参数名称可通过 `param_names_by_optimizer` 查询，
取代原有的 `muon_keys` / `no_muon_keys` 二分类元数据。

### Trainer YAML builders

The Trainer builds optimizer components through `build(model=...).get_optimizer()`.
`AdamW`, `Muon`, `Sinkhorn`, and `ComposedOptimizer` are implemented in
`hyper_parallel.components.optim.builders` and exported from `hyper_parallel.components.optim`.
They delegate runtime construction to `get_hyper_optimizer`; the Trainer entry point is unchanged.

For a three-family composition, use:

```yaml
optimizer:
  _target_: hyper_parallel.components.optim.ComposedOptimizer
  fp32_main_params: true
  muon:
    lr: 3.0e-5
    head_wise: true
    head_dim: 128
  sinkhorn:
    lr: 3.0e-5
    steps: 11
    tau: 1.0e-3
    param_groups:
      - param_patterns: ['\.engram\.embed\.weight$']
        lr: 1.5e-4
  adamw:
    lr: 1.0e-5
    weight_decay: 0.01
```

Set head widths or `head_wise_patterns` to match the model. Each family's optional `param_groups`
list selects parameters within that family by `param_patterns` and supplies ordinary optimizer-group
options such as an absolute `lr` or `weight_decay`. The example gives Engram tables `lr: 1.5e-4`
while other Sinkhorn parameters retain `lr: 3.0e-5`. Omit this group when the model has no Engram table.
Unmatched, overlapping, or wrong-family group selectors fail before optimizer construction; tied
parameters are selected through any alias and optimized once. Parameters outside explicit groups
retain family defaults and decay exclusions. LR scheduling preserves the relative group rates.

For Sinkhorn on embeddings/output weights with AdamW on remaining parameters, use:

```yaml
optimizer:
  _target_: hyper_parallel.components.optim.Sinkhorn
  fp32_main_params: true
  sinkhorn_config:
    sinkhorn_lr: 3.0e-5
    sinkhorn_steps: 11
    sinkhorn_tau: 1.0e-3
  adamw_config:
    adamw_lr: 1.0e-5
    adamw_weight_decay: 0.01
```

This builder accepts both prefixed and unprefixed hyperparameter keys. Add `param_patterns` to
`sinkhorn_config` to select other matrix weights explicitly. Both new builders use the core semantic
routing/decay rules; they do not use the legacy Muon builder's `no_decay_params` keyword rules.
