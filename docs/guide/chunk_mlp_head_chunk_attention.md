# Chunk MLP 与 Head Chunk Attention

本文介绍两个显式 opt-in 的实验性独立组件：`ChunkedSwiGLUMLP` 和
`HeadChunkGQAAttention`。它们缩小长序列训练中的临时激活工作集，不是默认训练路径，
也不会自动接入 Trainer 或正式模型 recipe。调用方负责模块选择、权重转换、CP mesh 注入，
以及在目标模型和硬件上的数值、显存与吞吐验证。

| 组件 | 分块维度 | 主要缩小的工作集 | 仍然保留的完整量 |
| --- | --- | --- | --- |
| Chunk MLP | 当前 rank 的 token | Gate/Up、SwiGLU mixed 激活 | 输入、输出、权重梯度 accumulator |
| Head Chunk GQA | query head | Q projection、Q A2A payload、单次 FA 工作集 | 当前组的 shared K/V、输出、参数梯度；FA 保存状态可留在设备或逐 stage offload |

## 1. Chunk MLP

### 1.1 参数布局和前向

`ChunkedSwiGLUMLP` 继承当前 `SwiGLUMLP`，保持同一参数布局和 `make_transforms()`：

```text
linear_fc1.weight [2I, H] = [Gate; Up]
linear_fc2.weight [H, I]
```

source checkpoint 中的 `gate_proj`、`up_proj` 按 Gate/Up 顺序拼接；`down_proj` 只重命名。
模块替换应先于 sharding、FSDP 和 checkpoint load，不应先替换为普通 `SwiGLUMLP` 再替换一次。
当前父类会在构造时打包已有 Gate/Up 参数并复用 Down 参数；checkpoint 加载仍须遵循声明的
weight transforms，不能把 source 参数名直接当作 packed 参数名。

输入 `x [..., H]` 展平为 `[T, H]`，每个 token 块 `Tc <= chunk_size` 执行：

```text
gate_up[Tc,2I] = x_chunk @ W_fc1.T
mixed[Tc,I]    = npu_swiglu(gate_up)
output[Tc,H]   = mixed @ W_fc2.T
```

完整输出持续存在，Gate/Up 和 mixed 的临时工作集被限制在 token chunk 内。

### 1.2 反向和梯度累加

custom autograd forward 保存输入与两份权重，不保存每块的 Gate/Up 和 mixed 激活。
backward 逐块重算这些激活，调用 `npu_swiglu_backward` 计算 packed Gate/Up 梯度，
并仅计算 `ctx.needs_input_grad` 要求的梯度。因此输入 detach 或单独冻结一份投影权重均可使用。

两份权重梯度通过 `npu_grouped_matmul_add_` 累加到 FP32 buffer，结束后只转换一次到参数 dtype。
实现不在循环内写 `Parameter.grad`，也不按 token chunk 触发 FSDP gradient hook。
两份完整 accumulator 的元素数仍为 `3IH`；短序列或小模型不一定有明显显存收益。

### 1.3 使用方式与边界

```python
from hyper_parallel.components.modules import ChunkedSwiGLUMLP

mlp = ChunkedSwiGLUMLP(
    module=source_mlp,
    module_fqn="model.layers.0.mlp",
    context=None,
    chunk_size=512,
)
```

training 且启用梯度时使用分块路径，要求 Ascend NPU、BF16/FP16、bias-free SwiGLU，
并要求环境提供 `npu_swiglu_backward` 和权重梯度所需的 `npu_grouped_matmul_add_`。
evaluation 或 `no_grad` 使用父类的普通 forward。首版不支持 TP 或二阶梯度。

## 2. Head Chunk GQA

### 2.1 模块接口和权重转换

`HeadChunkGQAAttention` 继承当前 `GQAAttention`，保持 packed/interleaved `linear_qkv`、
`o_proj`、Q/K norm 的参数布局，以及 `(output, attention_weights)` 返回形式。
constructor 提供 head 预算、可选的分阶段 FA 状态 offload 和默认关闭的通信流水开关，不提供旧别名：

旧实验接口的 `expand_kv_heads` 和 `attention_interface` 参数已移除；本版本固定使用
shared-KV GQA Ulysses 与原生 FA-v3，不接受 KV 扩展分支或外部 attention callback。

```python
from hyper_parallel.components.modules import HeadChunkGQAAttention

attn = HeadChunkGQAAttention(
    module=source_attention,
    module_fqn="model.layers.0.self_attn",
    context=None,
    head_chunk_size=4,
    offload_saved_stats=True,
    saved_stats_prefetch_slots=2,
    saved_stats_host_limit_bytes=8 * 1024**3,
    pipeline_output_a2a=False,
    pipeline_dq_a2a=False,
    early_saved_stats=False,
)
```

这个构造步骤只建立目标 attention 结构：尤其 `linear_qkv` 是新建参数，不会自动复制 source
Q/K/V 权重。必须通过 `make_transforms()` 声明的 checkpoint conversion 加载 source 权重，
再做数值比较或训练；不能将尚未转换权重的实例当成 source 的等价替身。
source 可以采用独立 Q/K/V 或当前父类支持的 fused QKV 布局，转换后的目标布局一致。

设 `Nq` 为 query head 数，`Nkv` 为 KV head 数，`G=Nq/Nkv`，`D` 为共同的 head dim。
每个 packed KV group 的行布局为：

```text
[G * D query rows | D key rows | D value rows]
```

`head_chunk_size` 指全局 query head 上限 `C`，`P` 为 CP degree。要求 `Nq`、`Nkv`、`C` 为
正整数，`Nq % Nkv == 0`、`Nkv % P == 0`、`C % P == 0`，且 `C <= Nq`。
允许尾块，不要求 `Nq % C == 0`；query stage 也不必覆盖完整的 query/KV 分组比 `G`。

### 2.2 固定的 shared-KV 执行路径

首版始终使用 shared-KV native GQA Ulysses。planner 先按 CP destination 分组选择物理 KV
head，每组覆盖的 KV head 数由每 rank query 容量 `C/P` 和 query/KV 比 `G` 决定。
同组内的 query stages 复用该组 K/V，不保证全部 KV head 总共只执行一次通信。
每组 K/V 各进行一次正向 sequence-to-head A2A，保持真实 KV head 数量，不将 K/V 重复扩展成
query head 数，也不提供 `expand_kv_heads` 分支。
Q projection、Q/K 标准 RMSNorm、caller 提供的 half-RoPE、Ulysses 通信和输出投影共同遵循
packed 布局；每个 query stage 的 O-projection 结果累加到一个完整输出 buffer。
该 buffer 使用输入 dtype：BF16 输入时，输出投影及跨 stage 累加仍为 BF16，不是 FP32
输出累加。FP32 master parameters 或梯度规约配置不会自动改变这个中间 buffer 的 dtype。

各 head stage 的 FA 计算串行执行，即 G1 调度；这里的 G1 不是 GQA 的 query/KV 分组比 `G`。
默认通信也是串行的；可选 offload 使用独立 copy stream 搬运保存状态。第 2.5 节的显式开关
可将一阶段通信与后续计算重叠，但不并发执行多个 FA。所有 CP rank 必须使用一致的 shape、
chunk、开关配置和 collective 顺序。

forward 保存各 stage 的 FA output、softmax max/sum、seed/offset。backward 每个 KV 组只
重算一次 K/V projection、norm、RoPE 和正向 A2A，query 部分按 stage 重算；直接把保存的
统计量交给 `npu_fusion_attention_grad_v3`，不会重新执行 FA forward。同组各 stage 的 dK/dV
以 FP32 累加后转换回 kernel dtype，各做一次逆向 A2A 和本地 VJP。
各 stage 的输入与共享参数梯度累加后，一次性交回 autograd。
其中 dHidden 在输入 dtype 的完整 buffer 中按原 stage 顺序累加，BF16 输入时仍为 BF16；
Q/K norm 的梯度使用 FP32 accumulator，结束时转换回对应参数 dtype。这些累加精度不同，
不能统称为“所有梯度都以 FP32 累加”。
FSDP 仍在参数边界处理梯度同步，不按 head stage 发起参数梯度通信。

### 2.3 CP mesh 与支持范围

constructor 不创建 ProcessGroup。需要 CP 时，由调用方使用通用 wrapper 注入已有的框架
DeviceMesh：

```text
hyper_parallel.distributed.context_parallel.head_chunk.head_chunk_ulysses_cp_wrapper
```

不注入 CP mesh 时使用 CP=1。首版只支持 pure Ulysses 的连续等长序列 shard，且 TP=1；
不包含任何模型专用 recipe 或 Trainer 配置项。

training、evaluation 与 `no_grad` 都使用相同 shared-KV 路径，没有父类 attention fallback。
Head Chunk 的数值执行路径要求：

- Ascend NPU，BF16/FP16；
- 环境提供 `torch_npu.npu_fusion_attention_v3`、`npu_fusion_attention_grad_v3` 与
  `npu_rotary_mul`；使用 HP RMSNorm 时还要求 `npu_rms_norm`；
- bias-free、标准 causal self-attention，Q/K/V head dim 相同且为偶数；
- 标准 Q/K RMSNorm，仅支持组件明确识别的 HyperParallel RMSNorm 和 Qwen3RMSNorm，
  保持各自的计算语义，不支持 offset RMSNorm 或其他 norm 变体；
- caller 提供 full-head half-RoPE 的 `(cos, sin)`，形状为 `[S_local,D]` 或 `[B,S_local,D]`
  （batch 可为 1），与 hidden 使用相同设备和 dtype，且不要求位置嵌入梯度；
- dropout=0，无 sliding window、KV cache、显式 attention mask、packed/varlen sequence；
- 不返回 attention weights，不支持 TP、二阶梯度或额外 attention 策略。

这些限制不是静默回退为完整 QKV 的条件；不满足时应修正 caller 的配置或使用普通 attention。
BF16 与 FP16 都是接口接受的 dtype，但不代表两者具有同等验证覆盖；当前主要训练验证采用
BF16，FP16 需要在目标模型、算子版本及并行配置下单独验证。

### 2.4 分阶段 offload 与预取

`offload_saved_stats` 默认为 `False`。开启后，每个 stage 前向结束将 FA 的输出 O、
softmax max/sum 异步复制到 pinned host 内存；seed/offset 仍通过 autograd 保存。
不 offload 输入 hidden、参数或共享 KV，也不是 FSDP parameter offload。

`saved_stats_prefetch_slots=1` 使用一个设备槽，按需加载；`2` 使用双槽，使下一 stage 的
H2D 可以与当前 stage 的计算重叠。设备槽按分配给该槽的最大 stage 定容，尾块不会额外
增加槽位。跨流 event 约束 producer、H2D 和 consumer 顺序；槽位须在当前 consumer
使用结束后才可覆盖。异步提交不保证传输全部被隐藏，实际收益须在目标形状上测量。
本接口不进行跨层预取，每个 forward 的 store、copy stream 和设备槽相互独立。

`saved_stats_host_limit_bytes` 是每个 Attention 模块共享 pinned pool 的硬容量上限，
计入对齐后的 buffer 容量和缓存；同一模块的多份未完成 forward 共同使用这个预算。
它不是整个模型或所有 ranks 合计的预算。超限会报错，不静默回退或超配；复用仍在传输的
host buffer 时，池可能等待对应 event，因此这也不是完全无阻塞的分配策略。

backward 完成后及时归还 host 租约并释放该次 invocation 的设备槽；池可保留已经空闲的
pinned buffer 供后续迭代复用。前向异常或未反向的图被销毁时也回收该次状态。
`saved_stats_offload_statistics()` 返回纯整数的传输次数/字节数、活跃 store、host 保存量、
设备槽容量及池容量，便于区分尚未消费的状态和可复用缓存。

offload 图只支持一次 backward。第一次消费后即使设置过 `retain_graph=True`，也不能再次
使用该图反向；需要重复反向时设置 `offload_saved_stats=False`。多个独立 forward 则各有
自己的状态，可以按不同顺序各反向一次。`no_grad` 或所有输入/参数都不需要梯度时不分配
offload store，也不保存 FA 状态。
当前接口面向单 Python 线程顺序发起的模块调用，不支持同一模块跨线程并发首次 forward。

### 2.5 可选通信流水与提前搬运

以下三个开关均默认为 `False`，可独立归因；它们仍是实验性路径，不能由原串行路径的
验证结果推导出设备正确性或性能收益：

- `pipeline_output_a2a`：组内提前提交下一 stage 的 Q 正向 A2A，再提交当前 O 逆向 A2A；
  当前 O 在下一 FA 后退役并按原 stage 顺序完成输出投影累加。组尾排空，不跨 KV 组流水。
- `pipeline_dq_a2a`：先准备下一 stage 的 Q/O/dO 通信，再提交当前 dQ 逆向 A2A；
  下一 FA backward 后退役前一 dQ 的本地 norm/RoPE VJP。各 query VJP、dK/dV 的 FP32
  累加和组末 KV VJP 的顺序保持不变。
- `early_saved_stats`：必须同时开启 `offload_saved_stats`。将 D2H 提前到 FA 返回后、
  O 逆向 A2A 之前；双槽时将下一 stage 的 H2D 提前到当前 Q replay 之前。单槽仍须等待
  当前 consumer 释放槽位，不增加槽数。不开此项时，O 流水在该 O 退役、投影后保存状态。

CP=1 或某 KV 组只有一个 query stage 时，两个通信流水开关均回退为该组的原串行调度；
CP=1 只能验证提前搬运，不能覆盖通信流水。通信使用已有 CP ProcessGroup，不增加通信用
线程、ProcessGroup 或自定义 kernel，也不改变 norm 数学或重复 KV。
同一 CP 组的所有 rank 必须使用一致的 stage 划分及开关配置；接口不会额外发起 collective
核对配置。配置不一致会造成 collective 次序或形状错配，可能导致错误或挂起。

通信票据显式持有 source、packed send 和 recv，读取前等待 work；默认路径还向消费流记录使用。
前向最多同时持有一份 Q 票据和一份 O 票据；反向最多一份 dQ 票据及两份本地 Q replay 图。
若一个 stage 的通信 payload 为 `M = B * S_local * C_stage * D * element_size`，
每张票据至多持有三份这样的 tensor 容量（部分可能 alias），重建布局还可能分配临时量。
因此额外显存不仅是 offload 的单/双槽；它还包括这些通信 buffer、下一 stage 的 Q 图及
尚未退役的 O/梯度，不能把槽位统计当作总显存峰值或保证流水一定省显存。
票据退役后，跨流使用对应的 allocator block 仍可能等待安全回收；有限数量的 Python
票据不等于只有相同数量的物理缓冲代数。分析容量时还需要观察 active、reserved 及释放事件，
不能只统计仍被 Python 对象引用的 tensor。

每次 invocation 的票据独立管理，正常组尾和异常退出都会等待已提交的 work 后释放引用；
异常清理不补发 collective，不能修复跨 rank 不匹配或通信后端失效。进入下一层前已排空
当前 forward 的通信，backward 返回前也排空本次通信，没有自动跨层协调。

另有默认关闭的 `trim_a2a_stream_records` 生命周期实验开关。它只作用于上面的私有 NPU
通信流水：在 `work.wait()` 已建立通信到计算流的依赖后，若 launch 与 wait 位于同一流，
省去对 source/send/recv 的额外消费流登记。内部 Q、FA 输出、dQ 的 storage 在该流生成；
packed send 若物化则在该流分配，若 alias source 则继承其 storage 生命周期，recv 由该流新建。
重建结果也可能 alias recv，返回 Tensor 会继续持有该 storage。成功 trim 时，重建布局和后续
读取必须留在同一流；这是封闭调用点的约定，不是运行时查询出的 allocation-stream 信息。
跨流 wait 保守回退原登记方式，CPU/CUDA 与 CP1 不做该精简。
它不删除通信等待、不改变 HCCL 自身的内存保护，也不改变 pack 或梯度累加次序。
如果以后增加外部 tensor 或跨流 producer，必须重新审查这项约定。

此开关不是固定 send/recv buffer 池，也不修改全局 allocator 环境。`work.wait()` 负责
设备依赖，`record_stream` 负责 allocator 的安全回收，二者不可互相替代；只比较 Python
引用数量不能证明显存收益。验证时应分别记录 active、allocated、reserved 和进程 HBM，
并在 fresh process 中固定 TorchNPU 的多流复用配置。

另有两个相互独立、默认关闭的生命周期开关，不改变 FA、norm 或梯度累加数学：

- `retire_d2h_sources`：要求开启 offload。最多持有一阶段的私有 FA O/max/sum 源存储，
  在下一阶段 FA 已提交后，让原计算流等待上一阶段 D2H 完成，再擦除对应 copy-stream
  marker 并释放源引用。forward 返回前清理尾票，不把这些设备源张量保留到下一层或反向。
  仅适用于内部同流分配、contiguous 且存储相互独立的 FA 输出；不满足或跨流时保留原
  allocator 保护。这里是设备事件依赖，不是新增 host synchronize，但仍可能影响流水重叠。
  此项额外要求运行时提供 `torch_npu.erase_stream`；缺少该接口时保持此开关关闭，
  使用原 allocator 生命周期保护路径。
- `release_a2a_source`：仅私有 NPU transport，在 packed send 与 source 的底层存储独立时，
  不再让通信票据额外持有 source；send、recv、work 仍保留至 wait。pack 与 source 分配在
  同一计算流，HCCL 只读取 send。共享存储、CP1、CPU/CUDA 保留原 source ownership。
  该开关不删除图、offload 等其他所有者；释放引用的累计字节数不等于 allocator 净节省。

这两项分别针对 D2H 完成后的延迟回收和通信期间的冗余活跃引用，不是固定通信 buffer 池，
也不保证峰值位于同一阶段。尾 stage 的 pack 可能是 alias，原生 FA workspace 也可能随形状
改变；应分别比较多种序列长度下的 allocated、active、reserved、HBM 和时间线。

## 3. 与整层 checkpoint 配合

Head Chunk 自身已实现 saved-stats backward。若外层整层 checkpoint 在 backward 重放这个
attention forward，会再次执行 head stage 与 FA，抵消部分节省。使用 HyperParallel 的
checkpoint exclusion 保留该区域保存的 autograd 张量及输出，使外层 replay 复用原 attention
输出，再由原 custom backward 完成计算：

```python
from hyper_parallel.core.activation_memory import (
    checkpoint_exclude_wrapper,
    checkpoint_wrapper,
)

# 先完成 attention 替换、权重转换与 CP mesh 配置。
block.self_attn = checkpoint_exclude_wrapper(attn, save_output=True)
block = checkpoint_wrapper(block, use_reentrant=False)
```

这里必须使用 HyperParallel 的 `checkpoint_wrapper`，不能直接以 PyTorch checkpoint 替代。
该组合要求 eager、non-reentrant checkpoint，且不支持嵌套 exclusion wrapper。
部分 Hugging Face 模型在配置 `activation_checkpoint.mode: full` 时会优先使用 native
reentrant checkpoint；仅加 exclusion wrapper 不会自动切换这一选择。集成时必须核实实际
checkpoint 类型，并以 forward/FA 调用计数确认 Attention 没有在外层 replay 中再次执行。
`save_output=True` 会保留完整 attention 输出，是避免外层重复计算的显式显存开销；
不要把它改为 `False` 作为通用省显存选项。两种 wrapper 的细节见
[Activation Checkpoint / Swap](./activation_checkpoint.md)。
多层和多份 outstanding forward 使用各自的 offload store 与通信票据；启用流水不能替代
checkpoint exclusion，也不会改变 offload 图仅支持一次 backward 的边界。

## 4. 数值与性能验证口径

本文不把旧环境、旧实现或未提交实验的数值当作当前结果，也不声明新的设备性能结论。
启用这些组件前，应在同一模型、权重、输入、dtype、CP/FSDP 和 checkpoint 配置下建立普通
packed 模块基线，并分开记录以下项目：

- 输出、dInput 与各类参数梯度是否齐全、有限，整体相对误差和方向是否一致；
- BF16 下连续多步的 loss、梯度范数及参数更新轨迹，避免仅凭一次 forward 下结论；
- warmup 后相同测量窗口的峰值 allocated/reserved、完整 step 时间与吞吐。

BF16 的分块累加顺序不同，不以所有元素逐一满足极紧阈值作为唯一放行条件。
整体数值和多步训练行为应共同判断，同时单独调查明显异常或非有限值。
小型 CPU oracle 只能检查分块数学和张量布局，不能替代 NPU kernel、CP collective 或
CP+FSDP 的真实设备验证。三个调度开关需要在目标硬件上分别验证单项与组合，并检查
实际进入的 CP 流水分支，不能以 CP=1 的回退结果代替。调度分支与串行 Head Chunk
数值一致，也不等于已经完成原生全-head Attention 与 Head Chunk 的整网精度对照。

不开 offload 时，Head Chunk 在设备保留所有 stage 的 FA output/统计量；开启后保存量转移
到 host，并在反向保留单槽或双槽加载容量。shared K/V 和完整参数梯度仍在设备；Chunk MLP
保留 FP32 权重梯度 accumulator。它们都不是按 chunk 数等比例减少总显存。
较小 chunk 还会增加 kernel/collective 次数和重算开销，因此应在目标长序列工作负载上选择，
不能将单层短序列结果外推为完整训练收益。

## 5. 提交范围

本版本包含 Chunk MLP、Head Chunk GQA、分阶段 FA 状态 offload/预取、必要 functional/module
导出、通用 CP wrapper 及本文档。Trainer 可通过本地显式替换与 CP 配置验证，但本版本不修改
默认 Trainer 行为或正式模型 recipe；不新增 UT 文件。包含显式 opt-in 的有界 stage 通信流水，不包含跨层预取或跨层通信协调。
