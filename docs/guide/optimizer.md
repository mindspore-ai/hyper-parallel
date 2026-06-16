# Optimizer 使用指南

HyperParallel 提供 Muon+AdamW 链式优化器和学习率调度器，支持分片优化器与 FSDP/HSDP 集成。

## 核心概念

| 优化器 | 说明 | 适用场景 |
|--------|------|----------|
| AdamW | 标准 AdamW 优化器 | 通用训练 |
| Muon | Momentum-based optimizer | 大模型训练 |
| ChainedOptimizer | Muon+AdamW 链式组合 | 混合参数组训练 |

Muon 优化器针对特定参数组（如 embedding、projection 层）使用 momentum-based 更新策略，其余参数使用 AdamW，实现混合优化。

## 接口概览

| 接口 | 说明 |
|------|------|
| `AdamW` | 标准 AdamW 优化器 |
| `Muon` | Momentum-based 优化器 |
| `ChainedOptimizer` | 链式优化器（Muon+AdamW 组合） |
| `get_hyper_optimizer` | 优化器工厂函数 |
| `get_hyper_lr_scheduler` | 学习率调度器工厂函数 |

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

### 2. 使用学习率调度器

```python
from hyper_parallel.core.optimizer import get_hyper_lr_scheduler

scheduler = get_hyper_lr_scheduler(
    optimizer=optimizer,
    scheduler_type="cosine",
    total_steps=10000,
    warmup_steps=500,
)
```

### 3. 参数分组示例

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

### 4. 训练循环

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