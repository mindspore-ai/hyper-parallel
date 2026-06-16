# DCP 分布式检查点使用指南

HyperParallel 提供 DCP（Distributed Checkpoint）能力，支持分布式检查点的保存和加载，包括异步 staging 保存和离线格式转换。

## 核心概念

分布式检查点将模型参数按分布式切分策略保存，每个 rank 只保存本地 shard，加载时根据目标切分策略重新分配。避免集中式保存时的单卡内存瓶颈和通信开销。

## 接口概览

DCP 模块位于 `hyper_parallel/core/distributed_checkpoint/`，包含以下组件：

| 组件 | 说明 |
|------|------|
| `api.py` | 保存/加载 API |
| `planner.py` / `standard_planner.py` | 检查点规划器 |
| `saver.py` | 检查点保存器 |
| `loader.py` | 检查点加载器 |
| `storage.py` / `filesystem_storage.py` | 存储后端 |
| `metadata.py` | 元数据管理 |
| `reshard.py` | 重分片（不同切分策略倒换） |
| `async_staging.py` | 异步 staging 保存 |
| `offline_transform.py` | 离线格式转换 |
| `layout.py` | 布局管理 |

---

## 基础使用

### 1. 保存分布式检查点

```python
from hyper_parallel.core.distributed_checkpoint import save, StandardPlanner

planner = StandardPlanner()
save(
    state_dict=model.state_dict(),
    path="/path/to/checkpoint",
    planner=planner,
)
```

### 2. 加载分布式检查点

```python
from hyper_parallel.core.distributed_checkpoint import load, StandardPlanner

planner = StandardPlanner()
state_dict = load(
    path="/path/to/checkpoint",
    planner=planner,
)
model.load_state_dict(state_dict)
```

### 3. 异步 staging 保存

异步 staging 将检查点保存操作 offload 到独立线程，避免阻塞训练计算：

```python
# 使用 async_staging 进行异步保存
# 训练继续的同时后台线程异步写磁盘
```

### 4. 离线格式转换

离线将不同切分策略的检查点进行转换，例如从 TP=8 转为 TP=4：

```python
from hyper_parallel.core.distributed_checkpoint import offline_transform

# 从源切分策略转换到目标切分策略
offline_transform(
    src_path="/path/to/checkpoint_tp8",
    dst_path="/path/to/checkpoint_tp4",
    src_layout=src_layout,
    dst_layout=dst_layout,
)
```

---

## reshard（不同切分策略倒换）

当需要更改并行策略（如从 TP=8 改为 TP=4）时，需要将检查点 reshard：

```python
from hyper_parallel.core.distributed_checkpoint.reshard import reshard

# 将检查点从一种切分策略重新分布到另一种
reshard(
    src_path="/path/to/checkpoint",
    dst_path="/path/to/resharded_checkpoint",
    src_mesh=old_mesh,
    dst_mesh=new_mesh,
)
```

---

## 与 FSDP/HSDP 配合

DCP 与 FSDP/HSDP 天然配合，因为 FSDP 切分后的参数已经是分布式状态：

```python
from hyper_parallel import fully_shard, init_device_mesh

mesh = init_device_mesh("npu", (dp_size,), mesh_dim_names=("dp",))
model = fully_shard(model, mesh=mesh)

# FSDP 模型的 state_dict 已是分布式格式
# DCP 保存时每个 rank 只保存本地 shard
save(model.state_dict(), "/path/to/checkpoint")
```

---

## 性能建议

1. **异步 staging**：使用 async_staging 避免检查点保存阻塞训练
2. **离线 reshard**：更改并行策略时使用 offline_transform 而非在线 reshard
3. **规划器选择**：StandardPlanner 适用于大多数场景，自定义规划器可用于特殊需求
4. **存储后端**：默认使用 filesystem_storage，支持本地文件系统和分布式文件系统