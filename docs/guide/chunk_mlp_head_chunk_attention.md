# Chunk MLP 与 Head Chunk Attention

本文说明两个面向长序列训练峰值显存的实验组件。二者只提供独立模块、通用 CP mesh 注入和
checkpoint conversion，不接入 Trainer，也不修改任何正式模型 recipe。模型族如何选择替换目标、
配置数据 mask 和分块参数，仍由模型 adapter 负责。

| 组件 | 分块维度 | 主要缩小的工作集 | 必须保留的完整量 |
| --- | --- | --- | --- |
| Chunk MLP | 当前 rank 的 token | Gate/Up、SwiGLU mixed 激活 | 输入、输出、参数梯度 accumulator |
| Head Chunk GQA | 全局 query head | QKV projection、A2A payload、FA 单次 workspace | 输出 arena、参数梯度、各 stage 的 FA 统计量 |

## 1. Chunk MLP

### 1.1 参数布局和前向

`ChunkedSwiGLUMLP` 直接继承当前 `SwiGLUMLP`，因此两者有完全相同的参数名和布局：

```text
linear_fc1.weight [2I, H] = [Gate; Up]
linear_fc2.weight [H, I]
```

它也继承 `make_transforms()`：source checkpoint 中的 `gate_proj`、`up_proj` 按上述顺序拼接，
`down_proj` 只重命名。替换必须发生在 sharding、FSDP 和 checkpoint load 之前，不能先替换成
普通 `SwiGLUMLP` 后再做第二次替换。

输入 `x [..., H]` 先展平为 `[T, H]`，其中 `T` 是 batch 与本地序列维的乘积。每个 token 块
`Tc <= chunk_size` 执行：

```text
gate_up[Tc,2I] = x_chunk @ W_fc1^T
mixed[Tc,I]    = npu_swiglu(gate_up)
output[Tc,H]   = mixed @ W_fc2^T
```

只有完整输出 `[T,H]` 持续存在；`gate_up` 和 `mixed` 在进入下一块前即可释放。因此主要 MLP
activation 从 `O(T*I)` 限制为 `O(Tc*I)`。

### 1.2 反向和 accumulator

custom autograd forward 保存输入和两份权重，不保存各块中间激活。backward 对每块重算
`gate_up`、`mixed`，再执行：

1. `dW_fc2 += dOutput_chunk.T @ mixed`；
2. `dMixed = dOutput_chunk @ W_fc2`；
3. `npu_swiglu_backward(dMixed, gate_up)` 得到 packed `dGateUp`；
4. `dW_fc1 += dGateUp.T @ input_chunk`；
5. 仅在上游需要时写入该块 `dInput`。

两份 dWeight 通过 `npu_grouped_matmul_add_` 直接累加到 FP32 buffer，循环结束后只 cast 一次到
参数 dtype 并作为普通参数梯度返回。实现根据 `ctx.needs_input_grad` 懒分配 `dInput`、`dW_fc1`
和 `dW_fc2`，因此支持输入 detach 或单独冻结一份投影权重；它不会在循环内写
`Parameter.grad`，也不会为每块触发一次 FSDP gradient hook。

Chunk MLP 没有消除两份完整 FP32 dWeight accumulator：其元素数为 `2IH + HI = 3IH`。模型较小
或本地 token 较短时，这一固定项会降低分块的相对收益。

### 1.3 使用边界

```python
from hyper_parallel.components.modules import ChunkedSwiGLUMLP

replacement = ChunkedSwiGLUMLP(
    module=source_mlp,
    module_fqn="model.layers.0.mlp",
    context=None,
    chunk_size=512,
)
```

当前生产路径要求 Ascend NPU、BF16/FP16、bias-free SwiGLU。evaluation 或 `no_grad` 使用父类的
普通 fused forward。custom backward 只保证一阶梯度，显式拒绝 double backward。TP 下 packed
weight 可能是局部 DTensor shard，尚未定义 token chunk 与 TP layout 的组合，因此首版不支持 TP。

## 2. Head Chunk GQA

### 2.1 为什么不能搬旧实现

历史实现保存裸 ProcessGroup、拥有独立 `q_proj/k_proj/v_proj`，并从 `position_ids` 内部生成
RoPE。当前主干的 `GQAAttention` 已经改为：

- `linear_qkv` 的 per-KV-head interleaved packed weight；
- 由 `make_transforms()` 负责 source checkpoint 转换；
- caller 传入 `(cos, sin)` `position_embeddings`；
- attention kernel 通过 `attention_interface` 解耦；
- CP 拓扑由 `inner_wrapper` 注入同一个框架 DeviceMesh。

`HeadChunkGQAAttention` 因此继承当前 `GQAAttention`，不恢复旧参数结构。普通 packed GQA 和
Head Chunk 的 state dict、参数对象类型、forward 参数以及 `(output, attention_weights)` 返回 ABI
保持一致，A/B 不再混入 QKV fusion layout 变化。

这里的 `attention_interface` 兼容参数只供 CP=1 evaluation/no-grad 时调用父类普通 GQA 路径。
分块 training 路径为了在 backward 直接消费 forward 的 max/sum/output/seed/offset，固定调用
`npu_fusion_attention_v3` 及其 grad v3 ABI；它不会静默调用 adapter 传入的其他 attention
interface。显式 mask、packed/varlen、非因果或其他 sparse/window 参数会在 collective 前报错。

### 2.2 packed 行映射

令：

- `Nq`：query head 数；
- `Nkv`：KV head 数；
- `G=Nq/Nkv`：每个 KV head 对应的 query head 数；
- `Dq`、`Dv`：QK 与 V head dim。

`linear_qkv.weight` 按 KV group 排列，每组连续行是：

```text
[G * Dq query rows | Dq key rows | Dv value rows]
```

一个 head stage 必须包含完整 KV group，所以 `head_chunk_size` 必须是 `G` 的倍数，并且首版要求
整除 `Nq`。若 stage 有 `Cq` 个 query head，则只读取对应的
`Ckv=Cq/G` 个 packed group；不会构造完整 Q/K/V，也不会在分块路径创建另一份独立 Q/K/V 参数。

### 2.3 完整前向

每个 stage 严格按相同顺序在所有 CP rank 执行：

```text
hidden[B,S_local,H]
  -> F.linear(当前 packed QKV 行)
  -> Q/K RMSNorm
  -> caller 已生成的 RoPE
  -> Q/K/V: sequence-to-head Ulysses A2A
  -> npu_fusion_attention_v3
  -> attention output: head-to-sequence A2A
  -> 当前 O-projection 列
  -> addmm_ 到唯一 output arena[B,S_local,H]
```

forward 为每个 FA stage 只保存 `attention output`、`softmax max`、`softmax sum`、`seed`、`offset`。
Q/K/V projection 激活和 A2A 输入不跨越 forward/backward 保存。

当 `Ckv` 不能被 CP degree 整除时，可以设置 `expand_kv_heads=true`：K/V activation 在当前 stage
内按 `G` 重复到 `Cq` 个逻辑 head，再做 A2A。它不复制参数；反向中的 `repeat_interleave` VJP
会把逻辑槽梯度求和回真实 KV head。代价是单 stage K/V payload 变大，因此只在较小 head stage
需要它时打开。

### 2.4 完整反向

反向不重新执行 FA forward。对每个 stage：

1. 由保存的 FA output 重建本地 attention output；
2. 计算当前 O 列的 `dW_o`，并把完整 `dOutput` 投影成该 stage 的 dAttention；
3. dAttention 做 sequence-to-head A2A；
4. 重算该 stage 的 packed QKV projection、Q/K norm、RoPE 和正向 A2A；
5. 以保存的 max/sum/output/seed/offset 直接调用 `npu_fusion_attention_grad_v3`；
6. dQ/dK/dV 做逆 A2A；
7. 用局部 `torch.autograd.grad` 完成 projection、RMSNorm 和 RoPE VJP；
8. 将 dHidden、共享 Q/K norm 梯度累加，将 packed dQKV 行和 dW_o 列写回各自完整梯度。

custom Function 最后一次性把普通梯度交回 autograd。FSDP 仍在参数边界完成 materialize、
reduce-scatter 与 reshard；stage 循环本身不调用 FSDP，也不按 stage 发起参数梯度通信。

### 2.5 CP mesh 注入

组件 constructor 不创建 ProcessGroup。`head_chunk_ulysses_cp_wrapper` 由当前 recipe builder 注入
已有 `cp_mesh`，同时在第一步 collective 前检查：

- query head stage 能被 CP degree 整除；
- 未扩展 K/V 时，KV stage 也能被 CP degree 整除；
- TP degree 必须为 1。

这保证所有 rank 的 stage 数量和 collective 顺序一致。Qwen3-MoE 等具体模型仍需在本地 adapter
中选择 source module、attention mask 和 `expand_kv_heads`；本 PR 不把该选择写入正式 recipe。
wrapper 可由 plan override 使用完整目标路径
`hyper_parallel.distributed.context_parallel.head_chunk.head_chunk_ulysses_cp_wrapper` 引用，无需组件
自行导入或创建通信组。

### 2.6 首版限制

Head Chunk 当前只支持：

- Ascend NPU BF16/FP16 training；
- bias-free standard causal self-attention；
- Q/K RMSNorm、相同 QK/V head dim；
- cosine/sine 与 hidden 使用同一设备和 dtype，且不要求位置嵌入梯度；
- dropout=0、无 sliding window；
- 无 KV cache、显式 attention mask、packed/varlen sequence 或 attention weights；
- TP=1；CP 使用 pure Ulysses 连续等长序列 shard；
- 一阶梯度。

这些条件均在进入 kernel/collective 前检查，不会静默回退为 full QKV。外层 full-layer activation
checkpoint 会额外 replay 整层 forward，而 Head Chunk backward 自身还会重算 stage projection；
组合使用前必须单独测吞吐。

## 3. 显存和性能应怎样解读

Head Chunk 把单次 QKV 与 A2A/FA workspace 从完整 head 数限制到 `Cq`。但所有 stage 保存的 FA
output 与统计量总和未必按 `Nq/Cq` 缩小，完整 output arena 和参数梯度也仍存在。因此，小 hidden、
单层或短序列场景可能只看到很小的 allocator 改善；必须以完整模型的峰顶 snapshot 为准。

性能方面，若 stage 数 `K=Nq/Cq`，QKV/O 总 FLOPs 接近不变，但 FA 调用和 Q/K/V A2A 从一次变成
`K` 次，backward 还要重算 projection/norm/RoPE。当前 saved-stats 实现已经避免额外 FA forward，
仍会在小模型上明显通信/launch-bound。

## 4. 当前新主干本地验证

以下都是 PR 外验证脚本的实测，不是正式 recipe 的端到端吞吐声明。

### 4.1 Chunk MLP

单卡 BF16，`T=2048,H=512,I=4096,C=512`，warmup 2、repeat 5：

| 指标 | 普通 packed SwiGLU | Chunk MLP | 变化 |
| --- | ---: | ---: | ---: |
| max allocated | 123,737,088 B | 96,476,672 B | -22.03% |
| max reserved | 150,994,944 B | 123,731,968 B | -18.05% |
| forward+backward 中位数 | 1.2549 ms | 4.6381 ms | 3.696x |

输出、dInput、`dW_fc1`、`dW_fc2` 全部存在；相对 L2 误差最坏为 `8.72e-5`。参数逐元素相同，
非整除 `113 token / chunk 37` 的额外冒烟也通过。

### 4.2 Head Chunk GQA

两卡 CP2 BF16，真实 Qwen3-MoE attention source，packed `linear_qkv`，
`global S=2048,H=512,Nq=16,Nkv=4,Cq=4`，warmup 2、repeat 5：

| 指标 | 普通 packed Ulysses | Head Chunk saved-stats | 变化 |
| --- | ---: | ---: | ---: |
| max allocated/rank0 | 131,636,224 B | 128,115,200 B | -2.67% |
| max reserved/rank0 | 222,298,112 B | 203,423,744 B | -8.49% |
| forward+backward 中位数 | 7.0364 ms | 45.4449 ms | 6.459x |

在较小的 `global S=256,H=256,Nq=8,Nkv=4,Cq=2` 上，普通 CP2 和 CP2+FSDP 两组均完成正反向。
输出、dInput、packed QKV、O projection、Q/K norm 梯度以及 FSDP local shards 全部存在；相对 L2
误差最坏为 `5.41e-3`。CP2+FSDP 的 max allocated 为
`111,294,464 -> 41,773,056 B`，但该小模型峰值受首次 kernel/allocator 和固定项影响，不能直接
外推到大模型。

为了确认性能结构，还比较过两个未保留方案：完全 checkpoint 重放约 `53.80 ms`，选择性缓存
FA op 约 `91.07 ms`。二者均已从提交实现移除；最终版本直接使用 saved stats，约比完全重放快
15.5%，但相对普通 Ulysses 仍有明显差距。下一步优化对象应是 stage 数、A2A 合并/复用和
较大 head chunk，而不是重新引入 FA forward 重算。

## 5. 提交范围

本变更提交：

- `ChunkedSwiGLUMLP`、`chunked_swiglu` 与 `swiglu_backward`；
- packed/interleaved `HeadChunkGQAAttention`；
- 通用 `head_chunk_ulysses_cp_wrapper`；
- 本文档和公共导出。

按照当前集成边界，不提交 Trainer 接线、正式模型 recipe、Qwen3-MoE 专用 replacement，也不在
该 PR 增加 UT。真实 Qwen3-MoE 接入、CP/FSDP launcher 和设备产物保留在 PR 外用于复核。
