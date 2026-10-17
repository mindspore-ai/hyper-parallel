# MoE 多核并行使用指南

Multicore 是独立的 Torch-only 组件，提供芯片内多核 MPMD 并行能力，结合核级内存语义单边通信，增强 MoE 通算掩盖和 MAC 利用率。

用户从独立组件 `hyper_parallel.core.multicore` 显式导入 `MegaGate` 或 `MegaMoeExperts`。
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

### MegaGate

`MegaGate`提供可独立构造的Gate参数和返回值。NPU上的`sqrtsoftplus`路径使用native Route/RouteGrad，
投影MatMul由Torch和CANN选型；其他评分函数及非NPU环境使用Torch实现。使用native路径前须以
`--multicore on`构建并激活生成的multicore payload。

```python
import torch

from hyper_parallel.core.multicore import MegaGate

gate = MegaGate(
    hidden_size=5120,
    num_experts=384,
    top_k=6,
    scoring_func="sqrtsoftplus",
    routed_scaling_factor=1.0,
    vision_enabled=True,
).to(device="npu", dtype=torch.bfloat16)

logits, routing_weights, expert_indices = gate(hidden_states, image_mask=image_mask)
```

`weight`默认使用标准差为0.02的Normal分布初始化，可通过`initializer_range`调整；文本和视觉correction
bias初始化为零。`hidden_states`形状为
`[batch, sequence, H]`。`logits`保持Projection输出dtype，`routing_weights`为FP32，专家索引为INT64。
传入`image_mask`时，mask形状须为`[batch, sequence]`；启用视觉bias时还须是同一NPU上的连续BOOL Tensor。
外层BF16 autocast开启时Projection产生BF16 logits；MegaGate将其转换为FP32后调用native Route。

### MegaMoeExperts

Torch 模型通过 `MegaMoeExperts` 执行 Router 选出的专家，当前支持 Ascend NPU 上的 Torch BF16 训练。
先按[构建与交付](docs/build.md)选择 `--multicore on`，激活 CANN 和 native payload，
并为所有 EP rank 配置相同的 `HYPER_PARALLEL_SHMEM_BOOTSTRAP_ENDPOINT`（如 `tcp://<rank-zero-ip>:<port>`）。

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

`ep_group` may also be a PP/DP-local EP subgroup: pass its size as `ep_size`.
MegaMoE uses group-local expert ownership and does not take a TP-size argument.
Subgroups exchange independent CANN bootstrap IDs internally and do not use a
shared WORLD rendezvous port. See the [SHMEM lifecycle](docs/shmem.md) for
cross-node interface selection and group-local shutdown requirements.

### DSV4.1 的 TP/SP 与 CP 集成

DSV4.1 适配器在 push 和 pull 模式下均支持 CP，以及配合序列并行使用的 TP。
开启 TP 时要求 `sequence_parallel: true` 且 `ep_size > 1`；PP 必须保持为 1。
框架从展平后的 dense DP/CP/TP 域生成 EP mesh。
路由专家仅沿专家维度分片，保留完整的隐藏维度和中间维度；
稠密层及共享专家沿用各自的 TP 布局，不支持对路由专家矩阵做张量并行。

MoE 边界直接消费经过 CP 和 TP/SP 划分后的 token。
VLM 路由时，适配器将 CP 局部图像 mask 对齐到 TP 局部隐藏状态区间。
共享专家的 TP 通信仍由其嵌套边界负责。

例如，16 个 rank 使用 TP2、CP2、EP16 和 micro-batch size 1 时，可配置为：

```yaml
accelerator:
  tp_size: 2
  cp_size: 2
  ep_size: 16
  pp_size: 1
  sequence_parallel: true
fsdp_config:
  dp_shard_size: 8
  edp_shard_size: 1
training:
  global_batch_size: 4
  micro_batch_size: 1
```

该配置对应 dense DP4、dense FSDP8 和 expert FSDP1。
将这些字段合并到现有 DSV4.1 配置中，保留其模块替换和 compute 声明。
长度为 4096 的序列均匀划分后，每个 rank 向 MegaMoe 提交 1024 行。
`local_num_tokens` 仍表示经过 CP/SP 划分后的实际行数上限，要求为 128 的倍数，并覆盖最大输入。
仅减少实际行数、保持容量配置不变，并不会缩小预留容量。
TP 不会进一步切分路由专家矩阵；本地序列变短也可能降低 MegaMoe 的执行效率。

激活 NPU native payload 后，可执行以下分布式精度对齐测试矩阵：

```bash
pytest -q tests/torch/multicore/test_deepseek_v41_megamoe.py -k megamoe_parallel
```

该 8-rank 矩阵覆盖 TP2/CP1、TP1/CP2、TP2/CP2、push/pull、文本/VLM 路由、
动态本地行数，以及输出、输入梯度、路由梯度、专家梯度和共享专家梯度。
验证范围为 MoE 模块及实际接入边界，不代表整网训练或性能验收。

### 输入与权重

- EP 可以使用 WORLD 或子组，`E` 必须能被 EP 整除。
  `local_num_tokens=T` 指定固定 token 数，要求 `T` 是 128 的倍数；
  `max_local_num_tokens=T_cap` 则允许各 rank 的实际 token 数不同，也允许空输入。
  动态模式按对齐后的上限预留存储，按实际输入长度生成调度计划。
  这两个 token 数参数互斥；动态执行仅支持 eager 模式，不会添加填充 token。
  包括空输入 rank 在内，所有 EP rank 必须按相同顺序进入每次正反向，并采用一致的
  输入、专家权重和路由权重梯度参与策略。实际长度和梯度参与策略随路由计数一起交换。
  调度范围取组内最大实际长度并向上对齐到 128；全组输入为空时，最小调度范围仍为 128。
  每个资源组最多缓存四份调度计划；autograd 会保留每次调用所用的计划，直到对应反向完成。
  pull 模式下，空输入仍保留其对称内存基址；仅改变实际输入长度不会重建 SHMEM。
- `hidden_states` 扁平后的 token 数为 `T`，`topk_ids`、`topk_weights` 均为 `[T, K]`，
  ID 使用全局专家编号，同一个 token 的 K 个 ID 必须互不重复。
- 可选的 `tokens_per_expert` 是同 device 上 `[E]` 的精确本 rank histogram。
  省略时内部统计；传入时由调用方保证与当前 ID 一致。contiguous INT32 可直接使用，
  INT64 或非连续输入会转换。
- 本地参数布局为 `gate_up_weight: [E/EP, H, 2I]`、`down_weight: [E/EP, I, H]`。
  内部在进入自定义 autograd Function 前对两份权重调用 `.contiguous()`：连续权重直接复用原存储，
  转置、切片等非连续权重会逐次复制为连续布局，并通过 autograd 将梯度回传到调用方的原始权重。
  执行器不缓存外部权重的连续副本；非连续布局每次调用都会产生复制开销。
  普通 gate/up/down 权重转换时，gate、up 分别转置后在末维拼接，down 转置最后两维即可。

Router 已有精确计数时，可直接传入：

```python
output = experts(
    hidden_states,
    topk_ids,
    topk_weights,
    tokens_per_expert=tokens_per_expert,
)
```

### 通信模式与容量

构造时通过 `dispatch_mode="push"`（默认）或 `dispatch_mode="pull"` 选择 dispatch；
两种模式的 combine 均使用 PUT。模式必须在所有 EP rank 上一致，运行中不可修改。
不同模式可以在同一进程中串行使用，但不能共享同一 workspace。

push 默认从较小的接收容量开始，在路由溢出时在线扩容；配置只有两个 push 专用因子：

| 参数 | push 默认值 | 作用 |
| --- | --- | --- |
| `initial_capacity_factor` | `1.25` | 初始接收容量相对于本地 `T * K` 路由行数的倍数 |
| `capacity_growth_factor` | `1.25` | 超限时相对于当前接收容量的增长倍数；`1.0` 表示仅满足当次需求 |

两个因子都要求为有限数且不小于1。省略或传入 `None` 表示采用 push 默认值。
pull 不需要容量因子，显式传入任一数值都会报错。旧 `expert_capacity_factor` 和 `capacity_policy` 参数已移除。

```python
# Pass these keywords alongside the model's fixed shape and EP topology.
push_options = dict(dispatch_mode="push", initial_capacity_factor=1.25, capacity_growth_factor=1.25)
pull_options = dict(dispatch_mode="pull")
```

初始容量为 `align128(ceil(initial_capacity_factor * T * K))`，不超过 `EP * T * K` 的保守无损上界。
需要一次性预留最坏情况时，将 `initial_capacity_factor` 设为 EP 大小。
当最大目的 rank 接收量 `R_max` 超过当前容量 `C` 时：

```text
C_new = min(EP * T * K, align128(max(R_max, ceil(capacity_growth_factor * C))))
```

同一 EP root 的所有 rank 完成在途工作，释放全部受管对称 buffer，finalize 旧 runtime，
再使用 fresh bootstrap ID 初始化更大的 heap 并重建 workspace，继续本次 forward；回落时不缩容。
该路径复用已有路由 counts，未溢出的 forward 不增加负载 collective。
增长倍数作用于接收容量，不是整个 heap；返回区仍固定为 `T * K` 行，总 heap 向上取整到2 MiB。

pull 的 SHMEM 仅包含 `T * K` 行发送区和返回区，接收区位于普通 HBM、按实际接收量分配。
两种模式的计算中间张量和待反向保存的 dispatch、up-projection、activation 都按本 rank 实际接收量分配；
零接收保留一行 ABI 占位。通信任务默认128行，可通过 `dispatch_split`、`combine_split` 分别配置。
路由不丢 token，热点的普通 HBM 开销仍存在。

共享层、独立的 push/pull workspace 和尚未绑定的层统一计入预算。
逻辑 workspace 对象保持稳定，因此扩容前的 forward 支持延迟或重复 backward、checkpoint 重算及串行交替 stream。
每个 root 要求串行执行；push 不支持图捕获，同一 root 下的其他层也不能保留使用旧地址的捕获图。
外部持有的专家参数不属于 heap，扩容不会移动它们。

若显式设置 `HYPER_PARALLEL_SHMEM_HEAP_SIZE`，物理 heap 直接按此固定预算初始化；push 只在预算内增加逻辑容量。
当增长余量放不下、而实际需求能放下时，退到最小必要容量。预算不足、未知 owner/allocation 或活动租约
会在销毁前阻止扩容；开始释放后若重建失败，runtime 不可继续使用，需退出并重启训练进程。
自动预算不写入环境变量；实际大小读取 `shmem.debug_state()["config"]["heap_size_bytes"]`。
扩容有一次性同步和初始化开销，可通过增加初始因子或增长因子减少重建次数。

### 资源共享与关闭

相同配置的串行层可在首次 forward 前共享 workspace，各层参数、梯度和待反向激活仍独立：

```python
MegaMoeExperts.share_execution_resources(layer.mlp.experts for layer in model.layers)
```

共享资源只允许串行提交；跨 stream 时调用方须建立输入 tensor 的依赖，不支持并发线程调用。
两条路径已验证 non-reentrant checkpoint 和 `retain_graph=True`。默认由运行时托管 workspace，无需显式
关闭。模块 GC 仅登记使用权释放，不执行 collective；下一个资源组
首次 forward 时，所有 SHMEM Root rank 一起核对资源表，只复用各 rank 都无成员且无待反向图的兼容
workspace，其余可回收的孤儿 workspace 当场释放。此协调不在已绑定模型的每步 forward 中执行。
GC 时机不一致或仍有反向图会延后回收；持续保持活模型/图引用仍会占用内存，heap 受显式预算或物理 HBM 容量约束。

自动清理适配在首次绑定时安装：拦截 Torch `ProcessGroup.shutdown`（旧版本使用
`distributed_c10d._shutdown_backend`），在 WORLD 或 SHMEM Root 被关闭前先回收资源。
因此标准 `dist.destroy_process_group()` 及其提前导入的函数别名无需额外适配；不支持的 Torch
版本在 native 资源创建前报错。直接调用底层 HCCL/C++ 销毁接口不在此 Python 适配范围内。
另注册先于 torch_npu 退出钩子的 `atexit` 清理；所有 rank 正常退出时自动尝试回收。
这些边界仍要求所有 rank 同序进入且没有在途调用/待反向图。异常、Ctrl+C、kill 或通信故障
不承诺安全或及时退出，不从 signal handler 中执行 collective；SIGKILL 无法执行退出钩子。

需要提前释放时，仍可调用原有 `MegaMoeExperts.close()`，示例中的 `model.close()` 保持逐层关闭；
共享 workspace 在最后一个成员关闭时才释放，不释放普通模型参数。所有 SHMEM Root rank 须在无在途
调用及待反向图时同序关闭。关闭失败会保留句柄供安全条件下重试，部分关闭的模块不能再次执行。
只有所有 rank 均成功后才移除资源记录；对称内存部分释放状态不一致或原生 shutdown 失败须重启。
自动清理失败会阻止显式通信域销毁继续执行；进程退出时仅记录失败并继续框架退出流程，不能保证释放成功。

SHMEM Python层以进程级引用计数统一管理Runtime生命周期。每个MegaMoe执行资源组建立时配对调用一次
内部`shmem.acquire()`，关闭时在workspace释放全部对称Tensor后调用一次`shmem.release()`；
`share_execution_resources`的相同配置层共享同一组及workspace，因此只形成一个资源组SHMEM引用。
资源池另外持有一个Runtime引用，使不同配置的孤儿workspace替换不会触发Runtime反复终结/重建；
该引用存活时保留原 heap manager；更大模型首次绑定会根据实际物理 heap 容量集体扩容，
包括旧 owner 已全部回收后的 push 或 pull 模型替换。
显式关闭全部已绑定资源或自动关停时释放该引用。非最后一个
`release()`只减少本地计数，进程内最后一个引用才执行跨rank关闭（仅丢弃模块对象不会触发释放）。未来
MegaMHC、MegaDSA等Multicore特性复用同一SHMEM Runtime时，也通过同一配对接口共享这套进程级计数，
不在各消费者内重复实现生命周期协调。
重新开启生命周期时，所有 rank 完成关闭后使用新的 `HYPER_PARALLEL_SHMEM_BOOTSTRAP_ENDPOINT`，
`HYPER_PARALLEL_SHMEM_HEAP_SIZE` 等堆配置在下一生命周期首次获取引用时重新生效。

完整 Qwen 接入及启动方式见 [MegaMoe 示例](examples/mega_moe/README.md)。MegaKernel 内部阶段采集方式见
[MegaKernel Profiling 使用指南](../../../docs/guide/mega_kernel_profiling.md)。

---

## 性能建议

1. **dispatch ↔ compute 掩盖**：MoE 的 AllToAll dispatch 与 expert compute 在不同核上并发，是最核心的掩盖收益
2. **单边通信**：基于内存语义的单边通信（Symmetric Memory）避免传统集合通信的同步开销
3. **RATR 通信重排**：通过 Rank-Aware Tile Reordering 将 AllToAll 流量在时间轴上均匀分散，避免多源 Rank 同时涌向同一目标，降低尾延迟
4. **O0 vs O1**：O0 通过 host CPU 调度，O1 调度下沉到 AICore，性能更高但实现难度更大
