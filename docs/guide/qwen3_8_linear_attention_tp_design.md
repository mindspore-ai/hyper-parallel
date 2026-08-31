# Qwen3.8 LinearAttention Tensor Parallel 设计

## 1. 目标与核心结论

本文设计 Qwen3.8 `LinearAttention`（Gated DeltaNet，以下简称 GDN）的 Tensor Parallel（TP）接入方案。
方案通过标准 `ShardingPlan` 描述参数布局，由 `ShardingApplier` 完成参数重排和分片，不复制 Transformers 的
`Qwen3_5MoeGatedDeltaNet.forward()`。

Qwen3.8 GDN 的普通投影可以使用 ColumnWise/RowWise 规则，但以下两个参数不能直接使用普通连续 `Shard(0)`：

- `in_proj_qkv.weight` 按 `[Q | K | V]` 打包，需要分别切分三个逻辑块；
- `conv1d.weight` 的 channel 同样按 `[Q | K | V]` 打包，必须复用完全一致的 Q/K/V head 映射。

因此标准 plan placement 需要表达 packed/blockwise sharding：

```text
ShardingPlanner
  → 识别 GDN boundary 与参数角色
  → 生成包含 PackedShard 的 ModuleShardingSpec.params
  → ShardingApplier 将每个逻辑块分别切给 TP rank
  → 重新拼成 rank-local Tensor/DTensor
  → Transformers 原始 forward 使用 TP-local 参数计算
```

`PackedShard` 是 `ShardingPlan` 层的参数布局描述，不是新的通信原语。它最终仍然产生沿 TP mesh 分片的参数；特殊之处
仅在于本地 shard 不是全局 tensor 的一个连续区间，而是多个逻辑块的 rank-local slice 拼接结果。

## 2. Qwen3.8 GDN 参数结构

一个 LinearAttention 层包含以下 TP 相关参数：

```text
linear_attn
├─ in_proj_qkv.weight    [Q | K | V] projection
├─ in_proj_z.weight      output gate projection
├─ in_proj_b.weight      beta projection
├─ in_proj_a.weight      alpha projection
├─ conv1d.weight         [Q | K | V] depthwise convolution channels
├─ conv1d.bias           optional
├─ A_log                 per-value-head decay parameter
├─ dt_bias               per-value-head time-step bias
├─ norm.weight           per-head normalization parameter
└─ out_proj.weight       value channels → hidden size
```

设：

```text
H     = hidden_size
Nk    = num_key_heads
Nv    = num_value_heads
Dk    = key_head_dim
Dv    = value_head_dim
Kdim  = Nk × Dk
Vdim  = Nv × Dv
T     = TP size
```

主要参数形状和 TP 目标如下：

| 参数 | 全局形状 | TP-local 形状 | TP 语义 |
| --- | --- | --- | --- |
| `in_proj_qkv.weight` | `[2Kdim+Vdim, H]` | `[(2Kdim+Vdim)/T, H]` | Packed ColumnWise |
| `in_proj_z.weight` | `[Vdim, H]` | `[Vdim/T, H]` | ColumnWise |
| `in_proj_b.weight` | `[Nv, H]` | `[Nv/T, H]` | ColumnWise |
| `in_proj_a.weight` | `[Nv, H]` | `[Nv/T, H]` | ColumnWise |
| `conv1d.weight` | `[2Kdim+Vdim, 1, K]` | `[(2Kdim+Vdim)/T, 1, K]` | Packed channel shard |
| `conv1d.bias` | `[2Kdim+Vdim]` | `[(2Kdim+Vdim)/T]` | Packed channel shard |
| `A_log` | `[Nv]` | `[Nv/T]` | value-head shard |
| `dt_bias` | `[Nv]` | `[Nv/T]` | value-head shard |
| `norm.weight` | `[Dv]` | `[Dv]` | Replicate |
| `out_proj.weight` | `[H, Vdim]` | `[H, Vdim/T]` | RowWise |

`K` 表示 convolution kernel size。`norm.weight` 作用于每个 value head 内部的 `Dv`，而不是所有 heads 拼接后的
`Vdim`，因此每个 TP rank 都保留完整的 `[Dv]`。

## 3. 为什么普通 Shard(0) 不能处理 packed 参数

### 3.1 `in_proj_qkv.weight`

Transformers 中该参数的第一维布局为：

```text
[Q_all_heads | K_all_heads | V_all_heads]
```

以 TP=2 为例，直接连续平分整个第一维可能得到：

```text
rank 0: fused tensor 的前半段
rank 1: fused tensor 的后半段
```

当 Q、K、V 的块大小不同，连续分割点可能落在逻辑块内部。即使总维度能够被 TP size 整除，每个 rank 也不一定同时
得到相互对应的 Q、K、V heads。GDN 后续按 Q/K/V 固定 section 拆分 tensor，会把错误的通道解释成对应分量。

正确切分为：

```text
rank r:
  Q_r = Q[r × Kdim/T : (r+1) × Kdim/T]
  K_r = K[r × Kdim/T : (r+1) × Kdim/T]
  V_r = V[r × Vdim/T : (r+1) × Vdim/T]

  local_in_proj_qkv = concat(Q_r, K_r, V_r, dim=0)
```

因此本地 tensor 仍然保持 Transformers forward 期望的：

```text
[Q_local | K_local | V_local]
```

### 3.2 `conv1d.weight`

GDN 的 Conv1d 是 depthwise convolution：

```text
groups = 2Kdim + Vdim
```

每个卷积 channel 与 `in_proj_qkv` 的一个输出 channel 一一对应。其 TP 切分必须复用相同的三个 slice：

```text
local_conv1d = concat(conv_Q_r, conv_K_r, conv_V_r, dim=0)
```

如果 projection 和 Conv1d 使用不同的 rank-to-head 映射，shape 可能完全合法，但 Q/K/V channel 会使用错误的卷积核，
形成无报错的数值错误。

## 4. Megatron-LM 的对应设计

Megatron 将 GDN 的输入投影进一步融合为：

```text
in_proj.weight = [Q | K | V | Z | Beta | Alpha]
```

运行时使用 `TELayerNormColumnParallelLinear`，每个 TP rank 的逻辑布局是：

```text
rank r = [Q_r | K_r | V_r | Z_r | Beta_r | Alpha_r]
```

Conv1d、`A_log` 和 `dt_bias` 在模块构造阶段直接创建为 TP-local shape：

```text
conv1d channels = (2Kdim + Vdim) / T
A_log shape     = Nv / T
dt_bias shape   = Nv / T
```

`out_proj` 使用 Row Parallel。各 rank 使用自己的 value-channel shard 做局部矩阵乘法，再在 TP group 上归并输出。

Megatron 的 checkpoint 层不会把 fused tensor 当作普通连续 shard，而是将其拆成逻辑子项：

```text
in_proj.weight:
  query / key / value / z / beta / alpha

conv1d.weight:
  query / key / value
```

这样 TP=M 保存的 checkpoint 可以按逻辑块重分片到 TP=N，不会跨越 Q/K/V 边界。

HyperParallel 与 Megatron 的目标布局一致，但接入点不同：

| 项目 | Megatron-LM | HyperParallel AutoModel |
| --- | --- | --- |
| 模型构造 | 直接创建 TP-local Megatron module | 保留 Transformers module |
| 输入投影 | 六路融合的 ColumnParallelLinear | Transformers 四个 projection 参数 |
| packed 切分 | module 构造与 checkpoint factory | ShardingPlan placement 与 ShardingApplier |
| forward | Megatron GDN forward | Transformers 原始 forward |
| checkpoint | logical split factory | loader 根据 plan 执行 logical block slice |

## 5. ShardingPlan placement 设计

### 5.1 Placement 表达

普通参数继续使用已有 placement：

```python
Shard(0)       # ColumnWise weight
Shard(1)       # RowWise weight
Replicate()    # TP replicated parameter
```

packed 参数增加标准 plan placement：

```python
PackedShard(
    dim=0,
    sections=("q", "k", "v"),
    section_sizes=(key_dim, key_dim, value_dim),
)
```

这里的 `section_sizes` 描述全局逻辑块大小。`ShardingApplier` 根据模块属性和 TP mesh 验证每个 section 能否整除，并
计算当前 rank 在每个 section 内的局部 slice。

`PackedShard` 应作为 `NamedPlacement` 能够携带的 plan placement，和 `Shard`、`Replicate` 位于同一参数 contract
中，而不是通过 `plan_overrides` 或架构专用旁路使能。只有 `tp_size > 1` 时 planner 才会生成有效的 TP 分片；TP=1
时 placement 的执行结果保持完整参数。

### 5.2 ModuleShardingSpec 示例

以一个 Qwen3.8 LinearAttention boundary 为例：

```python
ModuleShardingSpec(
    params={
        "in_proj_qkv.weight": {
            "tp": PackedShard(
                dim=0,
                sections=("q", "k", "v"),
                section_sizes=(key_dim, key_dim, value_dim),
            ),
        },
        "in_proj_z.weight": {"tp": Shard(0)},
        "in_proj_b.weight": {"tp": Shard(0)},
        "in_proj_a.weight": {"tp": Shard(0)},
        "conv1d.weight": {
            "tp": PackedShard(
                dim=0,
                sections=("q", "k", "v"),
                section_sizes=(key_dim, key_dim, value_dim),
            ),
        },
        "A_log": {"tp": Shard(0)},
        "dt_bias": {"tp": Shard(0)},
        "norm.weight": {"tp": Replicate()},
        "out_proj.weight": {"tp": Shard(1)},
    },
    in_src={"hidden_states": {"tp": Replicate()}},
    in_dst={"hidden_states": {"tp": Replicate()}},
    out_src={"output": {"tp": Partial()}},
    out_dst={"output": {"tp": Replicate()}},
)
```

如果 `conv1d.bias` 存在，则使用与 `conv1d.weight` 相同的 `PackedShard` section。实际字段名称以
`Qwen3_5MoeGatedDeltaNet.named_parameters()` 为准，planner 不应依赖参数在 `state_dict()` 中的遍历顺序。

### 5.3 参数角色识别

Planner 根据 boundary 模块类型和相对参数名派生 placement：

| 相对参数 FQN | 参数角色 | placement |
| --- | --- | --- |
| `in_proj_qkv.weight` | GDN packed input projection | `PackedShard(dim=0, Q/K/V)` |
| `in_proj_z.weight` | GDN value gate projection | `Shard(0)` |
| `in_proj_b.weight` | GDN beta projection | `Shard(0)` |
| `in_proj_a.weight` | GDN alpha projection | `Shard(0)` |
| `conv1d.weight` | GDN packed depthwise channels | `PackedShard(dim=0, Q/K/V)` |
| `conv1d.bias` | GDN packed depthwise bias | `PackedShard(dim=0, Q/K/V)` |
| `A_log` | GDN per-value-head parameter | `Shard(0)` |
| `dt_bias` | GDN per-value-head parameter | `Shard(0)` |
| `norm.weight` | per-head inner norm | `Replicate()` |
| `out_proj.weight` | GDN output projection | `Shard(1)` |

该识别规则进入标准架构模板和参数角色派生阶段。由此生成的参数必须参与 plan 的唯一性、整除性和完整 coverage 检查。

## 6. ShardingApplier 执行流程

### 6.1 普通参数

`Shard(0)`、`Shard(1)` 和 `Replicate()` 沿用现有参数分发流程：

```text
ModuleShardingSpec.params
  → resolve NamedPlacement against active mesh
  → distribute parameter
  → production 模式提取 local tensor
  → 保存 source_shard_info 供 FSDP 使用
```

### 6.2 PackedShard 参数

`PackedShard` 不能直接传给只理解 PyTorch `Shard` 的通用 `distribute_tensor()`。Applier 需要先解释逻辑分块，再构造
rank-local 参数：

```python
def apply_packed_shard(parameter, placement, tp_mesh):
    rank = tp_mesh.get_local_rank()
    world_size = tp_mesh.size()
    local_parts = []
    offset = 0

    for section_size in placement.section_sizes:
        validate_divisible(section_size, world_size)
        local_size = section_size // world_size
        start = offset + rank * local_size
        local_parts.append(parameter.narrow(placement.dim, start, local_size))
        offset += section_size

    return concat(local_parts, dim=placement.dim).contiguous()
```

概念执行顺序为：

```text
global [Q | K | V]
  → 根据 section_sizes 找出 Q/K/V 边界
  → 各逻辑块内部按 TP rank 切分
  → concat([Q_rank, K_rank, V_rank])
  → 安装为 rank-local parameter
  → 记录 global logical layout 与 TP source layout
```

Applier 还需要同步更新依赖参数 shape 的 module metadata：

```text
in_proj_qkv.out_features
conv1d.in_channels
conv1d.out_channels
conv1d.groups
```

这些属性的 TP-local 值应由 plan 显式允许的 `tp_divide_attrs` 或 GDN 标准模板声明，不能依赖 forward 第一次执行时再
动态猜测。

## 7. Activation 与通信

TP rank 输入相同的 hidden states：

```text
hidden_states: [B, S, H], TP Replicate
```

四个输入投影分别产生 TP-local activation：

```text
qkv:   [B, S, (2Kdim+Vdim)/T]
z:     [B, S, Vdim/T]
beta:  [B, S, Nv/T]
alpha: [B, S, Nv/T]
```

Conv1d 和 Gated Delta Rule 都只计算本 rank 的 heads，不需要在 GDN core 前做 TP AllGather。`norm` 作用于每个
value head 内部，因此也不需要 TP 通信。

`out_proj` 是 RowWise：

```text
local value output
  → local matmul with out_proj weight shard
  → Partial hidden-state contribution
  → TP AllReduce(SUM)
  → Replicated hidden states
```

这里的 `Partial()` 表示每个 TP rank 只持有最终结果的一部分加和贡献。它不是 collective；从 `Partial()` 转换到
`Replicate()` 时才触发 AllReduce。

## 8. Checkpoint 加载与保存

### 8.1 加载

checkpoint loader 需要读取 ShardingPlan 中的 packed placement，按逻辑块直接加载本 rank slice：

```text
checkpoint full tensor
  → 读取 plan 中 Q/K/V section metadata
  → 分别读取 Q_rank、K_rank、V_rank
  → concat 成 Transformers 期望的 local packed ordering
```

如果 loader 当前先加载完整权重、随后由 `apply_sharding_plan()` 分片，可以先完成正确性实现；但 27B 模型的正式路径
应支持 load-time slicing，避免每个 rank 暂存完整 packed 参数。

### 8.2 保存与 TP reshard

保存时不能只记录 local packed tensor 是全局 dim 0 的普通连续 shard。必须保留逻辑 section metadata：

```text
in_proj_qkv.weight → query / key / value
conv1d.weight      → query / key / value
```

从 TP=M 加载到 TP=N 时，先按逻辑 section 恢复全局映射，再在每个 section 内按新的 TP size 重分片。不能把旧的 local
packed tensors 按 rank 简单拼接后再连续切分。

## 9. 与 FSDP 的组合

TP 先定义参数在 TP mesh 上的 source layout，FSDP 再沿 DP shard mesh 切分每个 TP-local 参数：

```text
global GDN parameter
  → TP PackedShard/Shard
  → TP-local parameter
  → FSDP 沿 dp_shard 对 TP-local parameter fully shard
```

例如 `in_proj_qkv.weight`：

```text
全局: [Q | K | V]
TP rank r: [Q_r | K_r | V_r]
FSDP rank d: shard([Q_r | K_r | V_r], dp_shard)
```

`source_shard_info` 必须把 packed 参数标记为 TP-sharded source，保证 FSDP 不把它误判为 TP Replicate 参数。Backward
中先得到 TP-local weight gradient；FSDP 沿 DP shard group ReduceScatter。`out_proj` forward 的 TP AllReduce 与
FSDP gradient ReduceScatter 属于不同 mesh 和不同语义。

## 10. 与 Sequence Parallel、Loss Parallel 和 CP 的组合

### 10.1 TP + Sequence Parallel

Sequence Parallel（SP）切 sequence activation，GDN 参数 placement 不变。进入 GDN 前需要按照其计算 contract 将
hidden states 从 sequence shard 转成 TP Replicate，或提供支持 sequence-sharded input 的 GDN wrapper。GDN 输出完成
TP reduction 后再恢复 sequence shard。

首批 TP 接入可使用：

```text
SP shard hidden states
  → AllGather(sequence)
  → GDN TP-local head computation
  → out_proj TP AllReduce
  → ReduceScatter(sequence)
```

### 10.2 TP + Loss Parallel

GDN 本身不接触 logits 或 Cross Entropy。Loss Parallel 只影响末端 `lm_head` 和 loss：

```text
GDN/decoder output: TP Replicate 或 SP Shard
lm_head: vocab Shard(-1)
loss: distributed Cross Entropy
```

因此 GDN packed parameter placement 与 Loss Parallel 正交。

### 10.3 TP + CP

TP 先将 global heads 切为 TP-local heads，CP 再把 TP-local head 集合转换为完整 sequence、进一步局部 heads：

```text
global heads
  → TP-local heads
  → CP-to-HP AllToAll
  → TP×CP-local heads + global sequence
```

需要满足：

```text
Nk % TP == 0
Nv % (TP × CP) == 0
```

TP 参数 plan 和 CP inner wrapper 各自负责不同对象：

| 维度 | 处理对象 | 处理位置 |
| --- | --- | --- |
| TP | projection、Conv1d、A_log、dt_bias、out_proj 参数及 head activation | ShardingPlan + ShardingApplier |
| CP | sequence/head activation layout 与 causal history | GDN CP inner wrapper |

## 11. Planner 与 Apply 阶段校验

Planner 在生成计划时校验：

```text
Kdim % TP == 0
Vdim % TP == 0
Nk % TP == 0
Nv % TP == 0
每个 PackedShard section size % TP == 0
PackedShard section_sizes 之和等于参数对应维长度
GDN 所有 trainable parameters 均被 params contract 覆盖
同一参数只属于一个 boundary
```

Apply 阶段校验实际模型结构：

```text
in_proj_qkv.weight.ndim == 2
conv1d.weight 的 shard dim 与 depthwise channel dim 一致
projection 与 Conv1d 使用相同的 Q/K/V section metadata
local conv1d groups == local conv channel 数
local A_log/dt_bias 长度 == Nv/TP
local out_proj input dim == Vdim/TP
```

Transformers 升级造成参数名称、shape 或 packed ordering 改变时必须 fail-fast，不能退化成普通 `Shard(0)`。

## 12. 测试方案

### 12.1 PackedShard UT

构造不同大小的三个逻辑块并验证：

- TP=1 返回完整 tensor；
- TP=2/4 分别得到每个 section 的 rank-local slice；
- 各 rank local shard 按逻辑块重建后等于原 tensor；
- section 不能整除 TP 时抛出明确错误；
- section sizes 与参数维度不一致时抛错；
- forward 和 backward 的 local slice 映射一致；
- 非零 shard dim 和 1D/2D/3D 参数行为正确。

### 12.2 单层 GDN 数值对拍

固定随机性、权重和输入，对比 TP=1 与 TP=2：

- `in_proj_qkv` 拆分后的 Q/K/V；
- Conv1d output；
- `z`、`beta`、`alpha`；
- Gated Delta Rule output；
- `out_proj` AllReduce 后输出；
- input gradient；
- 每个参数的 local gradient 与 TP=1 对应 slice；
- grad norm。

### 12.3 Checkpoint 对拍

验证：

```text
TP=1 checkpoint → TP=2 load
TP=2 checkpoint → TP=1 load
TP=2 checkpoint → TP=4 load
```

对 `in_proj_qkv` 和 `conv1d` 单独检查 Q/K/V section，不只比较最终 loss。

### 12.4 Qwen3.8 训练验证

按以下顺序验证：

1. `TP=1, CP=1` FSDP baseline；
2. `TP=2, CP=1`；
3. 固定随机性运行多个 step，对比 loss、grad norm 和参数更新；
4. 对比峰值 allocated/reserved memory；
5. 验证 `TP=2, CP=2`；
6. 验证 checkpoint save/load 和断点续训。

适配完成的最低标准是 TP=2 能完成 plan、权重加载、forward、backward、optimizer step 和 checkpoint round-trip，且
关键中间 tensor、loss 和 gradient 与 TP=1 baseline 在既定容差内一致。仅通过 coverage check 或仅完成 forward 不能
视为适配完成。

### 12.5 当前 TP=1/TP=2 精度验证结果

以下结果来自 Qwen3.8-27B 的单步因果诊断。两组实验使用相同的 8 条 token sequence：TP=1 使用 FSDP size 8 和一个
micro-batch，TP=2 使用 FSDP size 4 和两个 micro-batch。为控制诊断成本，每条 sequence 截取前 16 tokens；这些数值
用于定位误差来源，不能直接代替 1024 tokens 训练的最终验收结果。

#### 12.5.1 原始前向对拍

固定输入和权重后，TP=2 两个 TP rank 在每个 decoder layer 的通信后输出、final norm 输出及 logits 上完全一致，说明
对应 boundary 的 TP collective 没有漏通信。TP=2 与 TP=1 的相对 RMS 误差沿网络逐步累积：

| 对拍点 | TP=2 vs TP=1 relative RMS |
| --- | ---: |
| Layer 0 output | 0.2602% |
| Layer 15 output | 0.7455% |
| Layer 31 output | 1.1579% |
| Layer 47 output | 1.2903% |
| Layer 63 output | 1.4739% |
| Final norm output | 1.6270% |
| Final logits | 0.7559% |

在 8172 个有效 token 的训练样本对拍中，TP=1 与 TP=2 的 Cross Entropy 分别为 `2.4429018497` 和
`2.4426879883`；token loss 的 mean absolute delta 为 `0.021459`，max absolute delta 为 `0.392774`。

对 RowWise projection 的两个 TP partial 做 FP32 手工求和后，结果仍与 TP=1 完整 GEMM 存在误差；改成 BF16 求和
只会额外增加一部分误差。例如 Layer 0：

| 算子 | FP32 partial SUM vs TP=1 | BF16 partial SUM vs TP=1 |
| --- | ---: | ---: |
| GDN `out_proj` | 0.2106% | 0.2891% |
| MLP `down_proj` | 0.7322% | 0.7547% |

这说明前向误差不能全部归因于 BF16 AllReduce；TP 分块 GEMM 与完整 GEMM 的累加顺序本身已经产生差异。

#### 12.5.2 激活注入因果实验

TP=2 每层 forward 仍执行真实 TP 计算，但在 boundary 输出处使用：

```python
corrected = tp2_hidden + (tp1_hidden - tp2_hidden).detach()
```

由此 forward value 等于 TP=1，而 backward 保留 TP=2 的局部 Jacobian。结果为：

| 场景 | Loss | Logits relative RMS | Trainer grad norm |
| --- | ---: | ---: | ---: |
| TP=1 | 4.7247419357 | 0 | 78.82117 |
| TP=2，未注入 | 4.7299818993 | 0.8183% | 79.15486 |
| TP=2，注入全部 layer output | 4.7247419357 | 0 | 78.92594 |

把前向值修正为 TP=1 后，本实验中的 grad norm 差距由约 0.423% 降至约 0.129%。因此前向数值误差确实会通过
backward 改变梯度，但仅修正 forward value 不能消除 TP 分块 Jacobian 的差异。

独立 grad norm 使用每个参数 local gradient 的 FP32 平方和，根据参数 source placement 去除 TP replica 重复项，再
跨 rank 求和重建。TP=2 激活注入组中，Trainer 内部 grad norm 为 `78.92594`，独立结果为 `78.93059`，相差约
0.0059%。因此剩余差异不是 grad norm 聚合公式、TP replica 重复计数或漏计数造成的。

#### 12.5.3 GDN 参数完整梯度重建

`conv1d.weight` 和 `in_proj_qkv.weight` 的最终 placement 为 FSDP `StridedShard(0, 2)` 与 TP
`PackedShard(0, Q/K/V)`。对保存的 FSDP gradient shards 先沿 FSDP 维重建 TP-local packed gradient，再按 Q/K/V
section 恢复全局逻辑顺序。Layer 13 的结果为：

| 参数或 section | TP=1 norm | TP=2 norm | relative L2 |
| --- | ---: | ---: | ---: |
| `conv1d.weight` | 5.96926 | 6.31900 | 7.8208% |
| `conv1d.weight/Q` | 2.44604 | 2.43568 | 5.1631% |
| `conv1d.weight/K` | 5.30027 | 5.69540 | 8.4609% |
| `conv1d.weight/V` | 1.24746 | 1.24884 | 2.3898% |
| `in_proj_qkv.weight` | 4.83262 | 5.01927 | 7.6826% |

`conv1d.weight` 的最大元素误差位于全局 K section 的第 640 个通道、卷积核第 4 个位置：TP=1 为 `4.75`，TP=2
为 `5.125`。最大的 12 个元素贡献约 91.77% 的误差平方和，说明 5.86% 的参数 norm 差主要由少量梯度尖峰主导，
不是整个 K section 等比例偏移。

#### 12.5.4 同时注入激活与 boundary gradient

为区分“Layer 13 局部 backward 错误”和“下游梯度误差传播到 Layer 13”，TP=2 在注入 TP=1 layer output 的基础上，
进一步在每个 decoder boundary backward 注入 TP=1 `grad_output`。TP=2 的两个 micro-batch 使用对应 TP=1 gradient
的二分之一，以保持 loss scaling 一致。

| 场景 | 独立重建 grad norm |
| --- | ---: |
| TP=1 | 78.82919 |
| TP=2，仅注入激活 | 78.93059 |
| TP=2，注入激活与 `grad_output` | 78.83354 |

双注入后整体 grad norm 与 TP=1 相差约 0.0055%。Layer 13 的变化为：

| 指标 | 仅注入激活 | 同时注入 `grad_output` |
| --- | ---: | ---: |
| `conv1d.weight` norm | 6.31900 | 5.99228 |
| 相对 TP=1 的 norm 差 | 5.86% | 0.39% |
| 逐元素 relative L2 | 7.82% | 0.63% |
| 最大元素误差 | 0.375 | 0.03125 |

该实验说明 Layer 13 的大部分参数梯度尖峰来自其收到的上游 gradient，而不是 Layer 13 `PackedShard`、FSDP
ReduceScatter 或 GDN local backward 单独产生。

#### 12.5.5 顶部 backward 与 64 层边界对拍

在 logits 已严格对齐的控制组中，依次比较 CE、lm_head、final norm 和 decoder layer boundary。`lm_head` 与 layer
boundary 的 hook 捕获 TP-local partial gradient，因此必须先将两个 TP rank 的 partial 做 SUM，再与 TP=1 比较；不能
把单个 TP rank 的 hook 结果直接乘以 TP size。

| 对拍点 | TP=2 vs TP=1 relative L2 | norm ratio |
| --- | ---: | ---: |
| `dlogits` | 约 0 | 1.000000 |
| `lm_head` backward 输出 | 0.1882% | 1.000021 |
| Final norm backward 输出 | 0.2156% | 1.000025 |
| Layer 63 backward 输出 | 0.5596% | 0.999988 |
| Layer 62 backward 输出 | 0.7035% | 1.000143 |
| Layer 60 backward 输出 | 0.8761% | 1.000417 |
| Layer 48 backward 输出 | 1.5173% | 1.000168 |
| Layer 32 backward 输出 | 1.9662% | 0.999868 |
| Layer 13 backward 输出 | 2.4274% | 1.000837 |
| Layer 0 backward 输出 | 2.5833% | 0.999653 |

从 layer output gradient 到 layer input gradient 的最大单层 relative L2 增量依次包括：Layer 63 `+0.3439%`、
Layer 62 `+0.1440%`、Layer 13 `+0.1210%` 和 Layer 61 `+0.1044%`。Layer 63 是 full-attention decoder layer，
Layer 62、61、13 是 linear-attention decoder layer。

### 12.6 当前结果的归因边界与后续验证

当前结果不能归纳为“误差都来自 Attention 累积”，原因如下：

1. 原始 forward 已经存在逐层误差。误差同时经过 GDN/full attention、MoE/MLP、residual 和 norm，现有 layer boundary
   数据没有把这些分支拆开。
2. 激活注入只固定 forward value，没有把 TP 分块算子的 Jacobian 改成 TP=1。相同输出值并不意味着 backward 路径
   相同。
3. Layer 63 的 boundary relative L2 增量最大，只能说明第一个 decoder layer 级明显放大发生在 Layer 63。该层同时
   包含 full attention、MoE/MLP、residual 和 normalization，不能据此只归因于 attention。
4. `lm_head` backward 的 relative L2 为 0.1882%，但 norm ratio 接近 1；它是顶部误差种子之一，不足以解释后续
   2.5% 的方向误差和个别参数的梯度尖峰。
5. 参数 gradient norm 对少量大元素很敏感。Layer 13 的 5.86% norm 差不能直接理解为该层全部 gradient 普遍相差
   5.86%。

因此当前可确认的结论是：

```text
TP forward 分块 GEMM/collective 产生数值差异
  + TP backward 从 lm_head 开始产生小的 partial-SUM 差异
  → 差异经过多个完整 decoder layer 的 Jacobian 逐步传播和放大
  → 少量 GDN 参数元素形成明显 gradient spike
```

其中“哪个子模块贡献最大”仍未确定。下一步应在 Layer 63 和 Layer 62 内增加以下边界对拍：

```text
decoder grad_output
  → residual 分支
  → MoE/MLP grad_input
  → attention/GDN out_proj grad_input
  → Q/K/V 或 GDN recurrent core grad_input
  → decoder grad_input
```

每个点都按 source placement 重建完整逻辑 gradient，并同时比较 relative L2、norm ratio、cosine 和 top-k error power。
只有在 attention 分支的输入输出之间观察到独立于 MLP/residual 的显著增量，才能把对应误差归因于 Attention。

## 13. 实施顺序

1. 在标准 plan placement 中定义 `PackedShard` 及其序列化表示；
2. 为 Qwen3.8 GDN 增加参数角色识别和 `ModuleShardingSpec.params` 派生；
3. 在 ShardingApplier 中实现 packed section 的 rank-local 重排与分片；
4. 同步更新 Conv1d 和 projection 的 TP-local module attributes；
5. 让 checkpoint loader 使用 plan metadata 执行 load-time packed slicing；
6. 支持 packed 参数的保存和 TP reshard；
7. 增加 PackedShard、单层 GDN 和 checkpoint UT；
8. 在 Qwen3.8-27B 上验证 `TP=2, CP=1`；
9. 验证 `TP=2, CP=2` 与 FSDP 组合场景。
