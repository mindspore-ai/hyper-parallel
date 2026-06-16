# PP 流水线并行使用指南

HyperParallel 提供 Pipeline Parallel（流水线并行）能力，将模型按层切分为多个 stage 分布到不同设备上，支持 GPipe、1F1B、VPP 等调度策略，并提供通算掩盖（overlap_b_f）、PP+FSDP、P2P prefetch 等高级特性。

## 核心概念

| 调度策略 | 说明 | Bubble 比例 |
|----------|------|-------------|
| GPipe | 所有 micro-batch 先正向再反向 | 较大 |
| 1F1B | 一个正向紧跟一个反向 | 较小 |
| VPP (Interleaved 1F1B) | 交错分配，每个 rank 多个 virtual stage | 最小 |
| MPipe (demo 版本) | 将多模态大模型流水拆分为 Encoder Stage 和 LLM Backbone Stage，并将 Encoder 计算转置到多个 PP rank 上并行执行 | 消除模态 bubble |

## 接口概览

| 接口 | 说明 |
|------|------|
| `PipelineStage` | 流水线 stage 封装，支持 dx/dw 计算 |
| `Schedule1F1B` | 1F1B 调度 |
| `ScheduleGPipe` | GPipe 调度 |
| `ScheduleInterleaved1F1B` | VPP（交错 1F1B）调度 |
| `ScheduleMPipeTranspose` | MPipe Transpose 调度 |
| `MetaStep` / `MetaStepType` / `BatchDimSpec` | 调度单元抽象 |
| `CommComputeOverlap` | 双线程通算掩盖协调器 |
| `HookCoordinator` / `HookRole` | COMM-first rendezvous 原语 |

---

## 基础使用

### 1. 标准 1F1B 调度

```python
from hyper_parallel import PipelineStage, Schedule1F1B

# 将切分后的 module 封装成 PipelineStage
stage = PipelineStage(split_model, stage_index, stage_num=4)

# 选择流水线并行的调度
schedule = Schedule1F1B(stage, micro_batch_num=8)

# 执行
x = DTensor.from_local(local_x, x_layout)
losses = schedule.run(x)
```

### 2. VPP (Interleaved 1F1B) 调度

```python
from hyper_parallel import PipelineStage, ScheduleInterleaved1F1B

# VPP 场景下每个 rank 有多个 virtual stage
schedule = ScheduleInterleaved1F1B(stages, micro_batch_num=8)
losses = schedule.run(*inputs)
```

### 3. MPipe (Transpose) 调度

```python
from hyper_parallel import PipelineStage, ScheduleMPipeTranspose

# MPipe 将 Encoder 或 dataload-only identity 放到每个 PP rank 上
stage = PipelineStage(stage_module, stage_index=rank, stage_num=pp_size)

schedule = ScheduleMPipeTranspose(
    [stage],
    micro_batch_num=8,
    preprocess_module=preprocess_module,
    num_transpose_layers=transpose_layer_num,
)

losses = schedule.run(local_batch)
```

---

## PP+FSDP 集成

PP stage 内部可以使用 FSDP 进行参数切分，进一步降低单卡内存占用：

```python
from hyper_parallel import PipelineStage, Schedule1F1B, fully_shard, init_device_mesh

mesh = init_device_mesh("npu", (pp_size, dp_size), mesh_dim_names=("pp", "dp"))

# 每个 PP stage 内部应用 FSDP
for stage_model in split_models:
    fully_shard(stage_model, mesh=mesh["dp"])

# PipelineStage 支持 MetaStep 集成 FSDP
stage = PipelineStage(stage_model, stage_index, stage_num=pp_size)
```

---

## 通算掩盖 (overlap_b_f)

overlap_b_f 将稳态期的 BWD 和 FWD 配对为复合 step，通过双线程协调器实现 EP A2A 与 compute 的物理并发：

```python
from hyper_parallel import ScheduleInterleaved1F1B, MetaStepType

# 1. 构造 schedule，启用 B/F overlap + P2P overlap
schedule = ScheduleInterleaved1F1B(
    stages, micro_batch_num, overlap_p2p=True, overlap_b_f=True,
)

# 2. 给每个 MoE 层装钩子
overlap = CommComputeOverlap()
for i, layer in enumerate(moe_layers):
    layer.experts.dispatch = overlap.wrap_dispatch(layer.experts.dispatch)
    layer.experts.combine = overlap.wrap_combine(
        layer.experts.combine, is_last_layer=(i == len(moe_layers) - 1),
    )

# 3. 注册 OVERLAP_B_F callback
def callback(step, ctx):
    bwd_step, fwd_step = step.sub_steps
    overlap.run(fwd_fn=..., bwd_fn=...)

schedule.register_custom_function(MetaStepType.OVERLAP_B_F, callback)

# 4. 执行
losses = schedule.run(*inputs)
```

### P2P Prefetch

`overlap_p2p=True` 让 PP stage 间的 P2P send/recv 提前一拍 issue，与本拍的 compute 自然并发：

```python
schedule = ScheduleInterleaved1F1B(
    stages, micro_batch_num, overlap_p2p=True, overlap_b_f=True,
)
```

### batched P2P transport

overlap_b_f 下默认使用 duplex P2P（batch_isend_irecv），减少 P2P 通信次数：

```python
# 默认启用 duplex batched P2P
# forward-side P2P prefetch via batch_isend_irecv
```

---

## PP Activation Swap

Pipeline Parallel 场景下可以配合 Activation Swap，将 stage 内的激活 offload 到 CPU：

```python
from hyper_parallel.core.activation_checkpoint import swap_wrapper

# PP stage 内部使用 swap
for layer in stage_model.layers:
    stage_model.layers[layer] = swap_wrapper(layer)
```

---

## PipelineStage dx/dw 计算

PipelineStage 支持 dx（输入梯度）和 dw（权重梯度）的独立计算：

```python
stage = PipelineStage(split_model, stage_index, stage_num=4)
# dx/dw 在 backward 时自动计算
```

---

## variable-layer + mixed-recompute

overlap_b_f 支持不同 PP stage 包含不同层数，并支持混合重计算策略：

```python
# 不同 stage 可包含不同层数
# mixed-recompute 在 overlap_b_f 下正确处理
```

---

## 性能建议

1. **micro_batch_num 选择**：通常设为 2×pp_size 或 4×pp_size，增大可减少 bubble 但增加内存
2. **overlap_b_f**：PP+EP 场景下开启 overlap_b_f 可显著减少通算串行等待
3. **overlap_p2p**：配合 overlap_b_f 开启 overlap_p2p 可进一步减少 P2P 通信延迟
4. **PP+FSDP**：stage 内部使用 FSDP 切分参数可大幅降低单卡内存占用
5. **batch size 整除**：MindSpore 后端要求 batch size 必须整除 micro_batch_num

---

## 更多参考

- [PP overlap_b_f 设计文档](./pipeline_parallel_overlap_b_f.md) — 通算掩盖详细设计
- [API 参考](../api/api_reference.md) — PP 模块接口详细说明
