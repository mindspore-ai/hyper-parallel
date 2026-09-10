# MoE 多核并行使用指南

Multicore 是独立的 Torch-only 组件，提供芯片内多核 MPMD 并行能力，结合核级内存语义单边通信，增强 MoE 通算掩盖和 MAC 利用率。

用户从独立组件 `hyper_parallel.core.multicore` 显式导入 `MegaMoeExperts`。
SHMEM 由 Multicore 在内部管理，不提供独立用户接口。

- [构建与交付](docs/build.md)
- [架构设计与扩展](docs/architecture.md)
- [MegaKernel Profiling 使用指南](../../../docs/guide/mega_kernel_profiling.md)
- [MegaKernel Profiling 设计与实现](docs/mega_kernel_profiling_design_and_usage.md)
- [私有 SHMEM 生命周期](docs/shmem.md)
- [完整 MegaMoE 样例](examples/mega_moe/README.md)

## 核心概念

多核并行是 HyperMPMD 的核心能力之一，从集群级 MPMD（Pipeline 并行）扩展到芯片内多核 MPMD：

- **O0**：通过框架层 host CPU 侧的调度，支持 cube、vector、单边通信算子分核执行
- **O1**：调度下沉到 AICore，支持 cube、vector、单边通信算子分核执行，进一步提升性能

HyperParallel 基于多核并行实现 MoE 通算掩盖（Multicore MoE-FFN）：将 MoE-FFN 的五个算子
（AllToAll-Dispatch、GMM1、SwiGLU、GMM2、AllToAll-Combine）融合为一个 kernel，由 AIC（AI Cube）和
AIV（AI Vector）核同时执行，实现通信与计算的细粒度重叠。

## 接口概览

多核并行模块位于 `hyper_parallel/core/multicore/`，包含以下组件：

| 组件 | 说明 |
|------|------|
| `modules/` | 多核并行模块实现 |
| `ops/` | 多核并行算子 |
| `scheduler/` | 多核并行调度器 |
| `tasks/` | 任务编排 |
| `torch/` | Torch native binding |
| `shmem/` | 仅供本组件使用的私有单边通信 |

源码构建生成一个同时包含正反向 kernel 的
`hyper_parallel_multicore_nn` vendor，并与构建环境对应的框架 adapter 一起进入 wheel 或本地 native payload。
Torch adapter 是通过 `torch.ops.load_library()` 加载的普通共享库。

---

## Torch managed API

Torch 模型通过 `MegaMoeExperts` 执行 Router 选出的专家，当前支持 Ascend NPU 上的 Torch BF16 训练。
先按[构建与交付](docs/build.md)选择 `--multicore on`，激活 CANN 和 native payload，
并为所有 EP rank 配置相同的 `SHMEM_IP_PORT`。

在业务进程中选定当前 NPU、初始化 HCCL 进程组后，以下为两卡 EP 示例：

```python
import torch
import torch.distributed as dist
from hyper_parallel.core.multicore import MegaMoeExperts

experts = MegaMoeExperts(
    local_num_tokens=1024,
    hidden_size=512,
    intermediate_size=128,
    num_experts=4,
    top_k=2,
    ep_size=dist.get_world_size(),
    ep_group=dist.group.WORLD,
).to(device="npu", dtype=torch.bfloat16)

output = experts(hidden_states, topk_ids, topk_weights)
```

### 输入与权重

- EP 必须覆盖整个默认 world；每 rank 的 token 数 `T` 固定且为 128 的倍数，专家数 `E` 可被 EP 整除。
  执行计划采用静态 tiling，构造后的 `T/H/I/E/K/EP` 固定；其他 shape、芯片及后端需单独验证。
- `hidden_states` 扁平后的 token 数为 `T`，`topk_ids`、`topk_weights` 均为 `[T, K]`，
  ID 使用全局专家编号。
- 可选的 `tokens_per_expert` 是同 device 上 `[E]` 的精确本 rank histogram。
  省略时内部统计；传入时由调用方保证与当前 ID 一致。contiguous INT32 可直接使用，
  INT64 或非连续输入会转换。
- 本地参数布局为 `gate_up_weight: [E/EP, H, 2I]`、`down_weight: [E/EP, I, H]`。
  普通 gate/up/down 权重转换时，gate、up 分别转置后在末维拼接，down 转置最后两维。

Router 已有精确计数时，可直接传入：

```python
output = experts(
    hidden_states,
    topk_ids,
    topk_weights,
    tokens_per_expert=tokens_per_expert,
)
```

### 容量

`expert_capacity_factor=None` 是默认值，接收容量为 `EP * T * K` 向上对齐到 128，保证 lossless。
SHMEM workspace 和每个待反向 forward 保存的 dispatch clone、up-projection、activation 均受该容量影响。

显式设置不小于 1 的有限 factor 时，容量改为 `ceil(T * K * factor)` 再对齐。
超过容量时，所有 EP rank 在进入 native kernel 前报 `capacity overflow`。
应根据显存和路由负载选择容量，确保显式容量覆盖实际接收量。

### 资源共享与关闭

相同配置的串行层可在首次 forward 前共享 workspace，各层参数、梯度和待反向激活仍独立：

```python
MegaMoeExperts.share_execution_resources(layer.mlp.experts for layer in model.layers)
```

共享资源只允许串行提交；跨 stream 时调用方须建立输入 tensor 的依赖，不支持并发线程调用。
checkpoint/recompute 和 `retain_graph=True` 暂未验证。所有 backward 完成后，各 rank 按相同顺序调用
每层的幂等 `close()`，并在销毁进程组前完成关闭。

Multicore 内部的多个 SHMEM owner 共用进程级 runtime，分别通过各自的 `close()` 关闭；
最后一个 owner 关闭才 finalize。
重新开启生命周期时，所有 rank 完成关闭后使用新的 `SHMEM_IP_PORT`。

完整 Qwen 接入及启动方式见 [MegaMoe 示例](examples/mega_moe/README.md)。MegaKernel 内部阶段采集方式见
[MegaKernel Profiling 使用指南](../../../docs/guide/mega_kernel_profiling.md)。

---

## 性能建议

1. **dispatch ↔ compute 掩盖**：MoE 的 AllToAll dispatch 与 expert compute 在不同核上并发，是最核心的掩盖收益
2. **单边通信**：基于内存语义的单边通信（Symmetric Memory）避免传统集合通信的同步开销
3. **RATR 通信重排**：通过 Rank-Aware Tile Reordering 将 AllToAll 流量在时间轴上均匀分散，避免多源 Rank 同时涌向同一目标，降低尾延迟
4. **O0 vs O1**：O0 通过 host CPU 调度，O1 调度下沉到 AICore，性能更高但实现难度更大
