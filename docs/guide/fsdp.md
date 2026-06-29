# FSDP / HSDP 数据并行使用指南

HyperParallel 提供 `fully_shard` 接口实现 FSDP（Fully Sharded Data Parallel）和 HSDP（Hybrid Sharded Data Parallel），将参数、梯度、优化器状态切分到数据并行域的各个设备上，显著降低单卡内存占用。

## 核心概念

`fully_shard` 支持 FSDP 和 HSDP 两种切分模式，通过传入 `mesh` 的维度不同来应用不同模式。两者的区别在于数据并行域是否被划分为「切分组」和「复制组」两级：

| 模式 | mesh | 参数切分范围 | 梯度通信 |
|------|------|--------------|----------|
| FSDP | 1 维 `(shard,)` | 在整个 DP 域内切分，每张卡只持有 `1/N` 的参数 | ReduceScatter（整个 DP 域内） |
| HSDP | 2 维 `(replicate, shard)` | 仅在「切分组」内切分，「复制组」之间持有相同的参数副本 | 切分组内 ReduceScatter + 复制组间 AllReduce |

- **FSDP**：参数在所有 DP rank 间均匀切分，切分粒度最细、内存节省最大，但每次前/反向都需要在整个 DP 域内做 AllGather 还原参数，跨节点时通信开销较大。
- **HSDP**：把 DP 域划分为若干「复制组」，组内做 FSDP 式切分，组间做参数复制。AllGather 只在切分组（通常落在单节点内）进行，跨节点只需对梯度做一次 AllReduce，从而在内存节省与通信成本之间取得平衡。`(replicate, shard)` 即「复制组数 × 每组切分数」，例如 `(2, 4)` 表示 2 个复制组、每组 4 路切分。

一般来说，单节点或通信不是瓶颈时用 FSDP；跨多节点、希望把 AllGather 限制在节点内时用 HSDP。

## 接口概览

切分入口为 `fully_shard`，将模块应用 FSDP/HSDP 切分，原地改写并返回。混合精度与 offload 行为通过策略类（`MixedPrecisionPolicy`、`OffloadPolicy` / `CPUOffloadPolicy`）作为参数传入。切分后优化器状态会自动随参数切分，无需额外配置。

```python
fully_shard(
    module,                       # 要切分的模块，或模块列表（列表内合并为一个切分单元）
    *,
    mesh=None,                    # DeviceMesh：1 维 → FSDP，2 维 → HSDP
    reshard_after_forward=True,   # 正向后是否 reshard
    shard_placement_fn=None,      # 自定义每个参数的切分方式
    mp_policy=MixedPrecisionPolicy(),   # 混合精度策略
    offload_policy=OffloadPolicy(),     # Offload 策略
    ignored_params=None,          # 完全排除在切分与梯度同步之外的参数
    replicate_params=None,        # 仍纳入管理但保持复制、梯度仍做AllReduce 的参数
    comm_fusion=False,            # 是否开启 AllGather / ReduceScatter 融合
)
```

| 参数 | 说明 |
|------|------|
| `mesh` | 数据并行拓扑。1 维触发 FSDP，2 维 `(replicate, shard)` 触发 HSDP；`None` 时按全局 world size 构建默认 1 维 mesh |
| `reshard_after_forward` | 正向后是否 reshard，见下文性能优化章节 |
| `shard_placement_fn` | 自定义每个参数的切分方式，传入一个返回 `Shard` 的可调用对象以指定切分维度；`None` 表示按默认维度 0 切分 |
| `mp_policy` | 混合精度策略，控制参数计算 dtype 与梯度归约 dtype |
| `offload_policy` | 传入 `CPUOffloadPolicy()` 可将切分后的参数/梯度卸载到 CPU，进一步省显存 |
| `ignored_params` | 这些参数留在原模块、不切分、不参与梯度同步 |
| `replicate_params` | 这些参数不切分但仍受管理，梯度以 DDP 式 AllReduce 同步，区别于 `ignored_params` |
| `comm_fusion` | AllGather / ReduceScatter / AllReduce 融合 |

### 运行期控制

`fully_shard` 调用后，模块被动态扩展出一组运行期控制方法，用于在训练过程中调整切分与通信行为：

| 方法 | 说明 |
|------|------|
| `model.set_modules_to_forward_prefetch(modules)` | 设置正向预取的模块列表 |
| `model.set_modules_to_backward_prefetch(modules)` | 设置反向预取的模块列表 |
| `model.set_requires_gradient_sync(flag)` | 控制本次反向是否做梯度同步，用于梯度累积 |
| `model.set_gradient_scaling_factor(factor)` | 设置梯度缩放因子 |
| `model.set_reduce_op_type("avg" / "sum")` | 设置梯度归约算子类型，默认 `avg` |
| `model.set_reshard_after_forward(flag)` | 运行期切换正向后是否 reshard |
| `model.unshard()` / `model.reshard()` | 手动还原 / 重新切分参数 |

---

## 基础使用

### 1. 最小 FSDP 示例

```python
from hyper_parallel import fully_shard, init_device_mesh

# 1 维 mesh = FSDP，在整个 DP 域内切分
mesh = init_device_mesh(device_type="npu", mesh_shape=(dp_size,), mesh_dim_names=("dp",))
model = fully_shard(model, mesh=mesh)

# 正常训练流程
output = model(input)
loss = criterion(output)
loss.backward()
optimizer.step()
optimizer.zero_grad()
```

### 2. HSDP 示例

```python
# 2 维 mesh = HSDP，(复制组数, 每组切分数)
mesh = init_device_mesh(
    device_type="npu",
    mesh_shape=(replicate_size, shard_size),
    mesh_dim_names=("replicate", "shard"),
)
# 直接传入 2 维 mesh，无需取子 mesh
model = fully_shard(model, mesh=mesh)
```

### 3. 混合精度

```python
from hyper_parallel.core.fully_shard.utils import MixedPrecisionPolicy
import torch

mp_policy = MixedPrecisionPolicy(
    param_dtype=torch.bfloat16,    # 参数计算精度
    reduce_dtype=torch.float32,    # 梯度归约精度
)
model = fully_shard(model, mesh=mesh, mp_policy=mp_policy)
```

### 4. 梯度缩放因子

CP/PP 场景下，可能需要对梯度做缩放：

```python
model = fully_shard(model, mesh=mesh)
model.set_gradient_scaling_factor(0.5)   # None 表示关闭缩放
```

---

## FSDP 性能优化

`fully_shard` 通过**通信与计算并发**来掩盖参数 AllGather 和梯度归约的开销，主要包括预取与梯度通信并发两类手段。

### 正/反向预取

为了让本层计算与下一层参数的 AllGather 并发，可以提前把后续模块的参数 unshard 出来。预取需要显式注册，由用户给出本模块计算时应提前还原哪些模块：

```python
# 正向：进入 layer_i 时，提前 AllGather layer_{i+1} 的参数
layer_i.set_modules_to_forward_prefetch([layer_i+1])

# 反向：进入 layer_i 反向时，提前 AllGather layer_{i-1} 的参数
layer_i.set_modules_to_backward_prefetch([layer_i-1])
```

注册后，当前模块计算所触发的参数 AllGather 会与上一/下一模块的计算重叠，从而隐藏通信延迟。

### 反向梯度通信与计算并发

反向过程中，每层算完梯度后的归约通信以异步方式下发，与后续层的反向计算并发执行，无需等待通信完成即可继续计算：

- **FSDP**：梯度的 ReduceScatter 异步下发，与计算并发。
- **HSDP**：梯度先在切分组内做 ReduceScatter，再在复制组间做 AllReduce。两段通信被流水化，使得某层的 AllReduce 可与下一层的 ReduceScatter、以及反向计算同时进行，进一步掩盖跨组通信开销。

### 通信融合

`comm_fusion=True` 会把多个小通信合并为一次大通信，减少通信次数；在 Host Bound 或跨机训练等场景下可能带来收益。

```python
model = fully_shard(model, mesh=mesh, comm_fusion=True)
```

### reshard_after_forward

`reshard_after_forward=True`（默认）在正向结束后立即把参数重新切分，释放正向到反向之间的参数内存；设为 `False` 可保留完整参数以减少反向时的 AllGather，用显存换取更少的FSDP通信次数。

---

## 与其他并行策略组合

### FSDP + TP

构造 `(dp, tp)` 二维 mesh，先在 `tp` 子维度应用张量并行，再在 `dp` 子维度应用 FSDP：

```python
from hyper_parallel import init_device_mesh, fully_shard, parallelize_module

model = ...  # model that developer created

mesh = init_device_mesh("npu", (dp_size, tp_size), mesh_dim_names=("dp", "tp"))

# 先应用 TP
parallelize_module(model, mesh["tp"], tp_plan)

# 再在 DP 子维度应用 FSDP
model = fully_shard(model, mesh=mesh["dp"])
```

### FSDP + PP

构造包含 `dp` 维度的 mesh，对每个 PP stage 子模块分别应用 FSDP：

```python
from hyper_parallel import PipelineStage, fully_shard, init_device_mesh

split_models = ...  # PP stage models that developer split

mesh = init_device_mesh("npu", (pp_size, dp_size), mesh_dim_names=("pp", "dp"))

# 每个 PP stage 内部应用 FSDP
for stage_model in split_models:
    fully_shard(stage_model, mesh=mesh["dp"])

stage = PipelineStage(stage_model, stage_index, stage_num=pp_size)
```

### FSDP + EP

构造 `(fsdp, ep)` 二维 mesh，先在 `ep` 子维度对专家层应用专家并行，再对整个 MoE 模块应用 FSDP：

```python
from hyper_parallel import init_device_mesh, fully_shard
from hyper_parallel.core.expert_parallel import ExpertParallel

moe = ...  # MoE model that developer created

mesh = init_device_mesh("npu", (fsdp_size, ep_size), mesh_dim_names=("fsdp", "ep"))

# EP 在专家维度切分专家层
ExpertParallel().apply(moe.experts, mesh["ep"])

# FSDP 在 fsdp 维度切分整个 MoE 模块
fully_shard(moe, mesh=mesh["fsdp"])
```
