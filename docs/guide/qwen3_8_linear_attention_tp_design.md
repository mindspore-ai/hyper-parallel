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
