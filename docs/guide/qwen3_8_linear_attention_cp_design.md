# Qwen3.8 LinearAttention Context Parallel 设计

## 1. 目标与核心结论

本文设计 Qwen3.8 `LinearAttention`（Gated DeltaNet，以下简称 GDN）的 Context Parallel（CP）接入方案。
首批实现聚焦 `TP=1, CP>1` 的训练场景，并为后续 `TP>1, CP>1` 保留组合能力。

接入不复制 Transformers 的 `Qwen3_5MoeGatedDeltaNet.forward()`。参考现有 HF attention CP wrapper 对
`attention_interface()`/SDPA primitive 的处理方式，GDN CP 只拦截 forward 内部已有的计算接口：

- 拦截 `causal_conv1d_fn()`，为局部 sequence shard 补充前序 rank 的 convolution halo；
- 拦截 Gated Delta Rule 接口，在调用前执行 CP-to-HP AllToAll，在调用后执行 HP-to-CP AllToAll；
- 原始投影、QKV 拆分、gate 计算、norm、`out_proj` 和返回值均继续由 Transformers forward 执行。

```text
Transformers 原始 GDN forward
        │
        ├─ projections / QKV ordering             原样执行
        │
        ├─ causal_conv1d_fn(...)                  inner wrapper 拦截
        │     └─ previous-rank halo + 原始 Conv1d
        │
        ├─ query/key/value/g/beta preparation     原样执行
        │
        ├─ gated_delta_rule(...)                  inner wrapper 拦截
        │     ├─ CP-to-HP AllToAll
        │     ├─ 原始 Gated Delta Rule
        │     └─ HP-to-CP AllToAll
        │
        └─ gated norm / out_proj                  原样执行
```

这种方式保留 Transformers forward 作为唯一模型语义来源，避免 Transformers 升级后 HyperParallel 中的复制版本
与上游实现发生偏移。

## 2. 为什么 GDN 需要特殊 CP 处理

GDN 的 recurrent state 沿 sequence 递归：

```text
state[t] = update(state[t-1], key[t], value[t], gate[t])
output[t] = read(state[t], query[t])
```

如果 CP rank 0 计算 `[0, S/2)`，rank 1 独立计算 `[S/2, S)`，rank 1 缺少 rank 0 结束时的 state，结果不等价于
单卡计算。

Megatron 的解决方式是将布局从：

```text
局部 sequence × 完整 head/channel
```

转换为：

```text
完整 sequence × 局部 head/channel
```

每个 CP rank 因而能按照正确的全局 token 顺序执行一部分 heads 的完整 recurrence。GDN 计算完成后，再把布局转换
回局部 sequence。

## 3. 与 AttentionInterface 方案的对应关系

HF attention forward 在完成 Q/K/V 投影和 reshape 后调用 `attention_interface()`。CP wrapper 不复制 attention
forward，而是在 attention interface 前后加入通信：

```text
local Q/K/V
  → CP layout transform
  → original attention interface
  → inverse CP layout transform
  → local attention output
```

Qwen3.8 GDN 当前没有统一的 `linear_attention_interface()`，但 forward 内部已经调用独立 GDN primitive：

```python
core_attn_out, last_recurrent_state = torch_chunk_gated_delta_rule(
    query,
    key,
    value,
    g=g,
    beta=beta,
    ...,
)
```

fast path 对应 FLA `chunk_gated_delta_rule`。因此可以把 Gated Delta Rule 视为 GDN 的内部计算接口：

```text
local query/key/value/g/beta
  → CP-to-HP AllToAll
  → original gated_delta_rule
  → HP-to-CP AllToAll
  → local core_attn_out
```

Wrapper 只临时替换 Transformers modeling module 中的 primitive symbol，并在 `try/finally` 中恢复，与当前
`sdpa_hf_ulysses_cp_wrapper` 临时替换 `F.scaled_dot_product_attention` 的结构一致。

## 4. GDN rule 前后的 AllToAll

### 4.1 接口输入布局

进入 Gated Delta Rule 前，Transformers 已完成：

- input projections；
- packed QKVZ/BA ordering；
- Q/K/V causal Conv1d；
- Q/K/V reshape；
- `beta` 和 decay `g` 计算；
- Q/K head 到 value head 的 repeat-interleave。

训练路径上的输入布局为：

```text
query: [B, S_local, N_v_heads, D_k]
key:   [B, S_local, N_v_heads, D_k]
value: [B, S_local, N_v_heads, D_v]
g:     [B, S_local, N_v_heads]
beta:  [B, S_local, N_v_heads]
```

其中 `S_local = S / CP`。

### 4.2 CP-to-HP

对五个输入执行一致的 Ulysses AllToAll：

```text
query/key/value:
  [B, S/CP, N_v, D] → [B, S, N_v/CP, D]

g/beta:
  [B, S/CP, N_v]    → [B, S, N_v/CP]
```

伪代码：

```python
query = ulysses_seq_to_head(query, seq_dim=1, head_dim=2, cp_mesh=cp_mesh)
key = ulysses_seq_to_head(key, seq_dim=1, head_dim=2, cp_mesh=cp_mesh)
value = ulysses_seq_to_head(value, seq_dim=1, head_dim=2, cp_mesh=cp_mesh)
g = ulysses_seq_to_head(g, seq_dim=1, head_dim=2, cp_mesh=cp_mesh)
beta = ulysses_seq_to_head(beta, seq_dim=1, head_dim=2, cp_mesh=cp_mesh)
```

这里不再需要处理 packed QKV。Transformers 已经完成 Q/K/V 拆分和 value-head 对齐，wrapper 面对的是明确的
GDN rule 接口参数。

### 4.3 原始 GDN rule

Wrapper 使用转换后的输入调用捕获的原始接口：

```python
core_attn_out, last_recurrent_state = original_gated_delta_rule(
    query,
    key,
    value,
    g=g,
    beta=beta,
    **kwargs,
)
```

每个 CP rank 此时拥有完整 sequence，因此 recurrence 与 CP=1 的 token 顺序一致；每个 rank 只计算
`N_v_heads/CP` 个 heads。

### 4.4 HP-to-CP

原始 GDN rule 返回：

```text
core_attn_out: [B, S, N_v/CP, D_v]
```

立即执行反向 AllToAll：

```python
core_attn_out = ulysses_head_to_seq(
    core_attn_out,
    seq_dim=1,
    head_dim=2,
    cp_mesh=cp_mesh,
)
```

恢复为：

```text
[B, S/CP, N_v, D_v]
```

随后回到原始 forward。原始 `z` 也是 `[B, S/CP, N_v, D_v]`，因此 Transformers 可以继续执行：

```text
gated RMSNorm → reshape → out_proj → return
```

## 5. Conv1d 的边界处理

不能只包装 Gated Delta Rule 而忽略 Conv1d。原始 causal depthwise Conv1d 发生在 GDN rule 之前；如果直接在
每个 sequence shard 上独立卷积，除 CP rank 0 外，每个 rank 开头的 token 都缺少前一 rank 的历史窗口。

没有必要为了 Conv1d 把完整 sequence AllToAll 到每个 rank。卷积只依赖有限窗口，可以单独拦截
`causal_conv1d_fn()`，交换宽度为：

```text
halo_width = (kernel_size - 1) × dilation
```

的前序 token：

```text
rank 0: [zero halo | local sequence 0]
rank 1: [rank 0 tail | local sequence 1]
rank 2: [rank 1 tail | local sequence 2]
...
```

伪代码：

```python
def cp_aware_causal_conv1d(
    hidden_states,
    weight,
    bias=None,
    activation=None,
    **kwargs,
):
    halo = exchange_previous_rank_halo(
        hidden_states,
        width=weight.shape[-1] - 1,
        cp_mesh=cp_mesh,
    )
    conv_input = concat_halo(halo, hidden_states)
    return original_causal_conv1d(
        conv_input,
        weight,
        bias,
        activation,
        padding=0,
        output_length=hidden_states.shape[-1],
        **kwargs,
    )
```

halo 通信必须可微，且只发送到后一 CP rank。首个 rank 使用零 halo。当前 AutoModel dataloader 使用连续 sequence
切分，因此前序 rank 就对应前序 token 区间。

这一处理与 GDN rule 的 AllToAll 分工如下：

| 计算区域 | CP 处理 | 原因 |
| --- | --- | --- |
| Causal Conv1d | 前序 rank halo | 只需要有限历史窗口 |
| Gated Delta Rule | CP-to-HP/HP-to-CP AllToAll | recurrence 需要完整历史 |
| Norm、gate、out_proj | 不增加通信 | 输入输出已经恢复为 local sequence |

## 6. Inner wrapper 设计

ShardingPlan 仍以 `linear_attn` 作为 boundary，但 inner wrapper 不替换它的计算逻辑，只在调用原始 forward 期间
临时替换两个 primitive。

```python
@inner_wrapper
def qwen3_8_gdn_ulysses_cp_wrapper(
    target_module,
    mesh,
    tp_mesh,
    cp_mesh,
    ep_mesh,
):
    del mesh, ep_mesh
    _validate_qwen3_8_gdn_cp(target_module, tp_mesh, cp_mesh)

    original_forward = target_module.forward
    modeling_module = resolve_modeling_module(target_module)
    original_conv = modeling_module.causal_conv1d_fn
    original_chunk_rule = modeling_module.torch_chunk_gated_delta_rule

    @functools.wraps(original_forward)
    def cp_forward(*args, **kwargs):
        fired = {"conv": False, "gdn": False}

        def cp_conv(*conv_args, **conv_kwargs):
            fired["conv"] = True
            return _cp_causal_conv1d(
                original_conv,
                cp_mesh,
                *conv_args,
                **conv_kwargs,
            )

        def cp_gdn_rule(query, key, value, *, g, beta, **rule_kwargs):
            fired["gdn"] = True
            query, key, value, g, beta = _gdn_cp_to_hp(
                query,
                key,
                value,
                g,
                beta,
                cp_mesh,
            )
            output, state = original_chunk_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                **rule_kwargs,
            )
            output = _gdn_hp_to_cp(output, cp_mesh)
            return output, state

        modeling_module.causal_conv1d_fn = cp_conv
        modeling_module.torch_chunk_gated_delta_rule = cp_gdn_rule
        try:
            output = original_forward(*args, **kwargs)
        finally:
            modeling_module.causal_conv1d_fn = original_conv
            modeling_module.torch_chunk_gated_delta_rule = original_chunk_rule

        if not fired["conv"] or not fired["gdn"]:
            raise RuntimeError(
                "Qwen3.8 GDN CP wrapper did not intercept all required primitives"
            )
        return output

    target_module.forward = cp_forward
```

实际实现必须同时识别 eager 和 fast-path 使用的 GDN rule symbol，不能假设所有 Transformers 环境只调用
`torch_chunk_gated_delta_rule`。首批可以明确只支持当前服务器实际使用的 eager 路径，对其他路径 fail-fast；后续再扩展
FLA fast path。

Wrapper 必须具有 misfire detection。如果 Transformers 版本升级后不再调用预期 primitive，应立即报错，不能在没有
CP 通信的情况下继续训练。

## 7. 为什么不包装更小或更大的范围

### 7.1 不包装整个 GDN forward

复制整个 forward 会重复以下上游逻辑：

- padding mask；
- cache 分支；
- packed projection ordering；
- eager/fast kernel选择；
- norm、gate 和 output projection；
- Transformers 新版本增加的参数与行为。

版本升级时很容易出现 shape 合法但数值错误。

### 7.2 不只包装 Conv1d

Conv1d halo 只能修复有限卷积窗口，不能给后续 GDN recurrent state 补充全部历史。

### 7.3 不只包装 GDN rule

GDN rule 前后的 AllToAll 可以修复 recurrence，但 Conv1d 已经在局部 shard 上错误计算，因此仍然存在 CP boundary
数值误差。Conv1d primitive 和 GDN rule primitive 必须同时拦截。

### 7.4 不在投影后立即做 AllToAll

Megatron 在 input projection 后立即做 CP-to-HP，并让 Conv1d 与 GDN 都在完整 sequence、局部 channel 上执行。
Transformers 没有暴露“投影结束、Conv1d 开始”这一统一接口。如果坚持在该点注入，就必须复制 forward 或侵入式
修改 Transformers module。

本文把它等价拆为：

```text
Conv1d：局部 sequence + halo
GDN rule：完整 sequence + 局部 heads
```

这样保留原始 forward，同时保持两部分计算的 causal 正确性。

## 8. ShardingPlan 接入

ShardingPlanner 应根据 FQN 和模块类型识别 Qwen3.8 LinearAttention boundary：

```text
FQN:       model.layers.<layer_id>.linear_attn
module:    Qwen3_5MoeGatedDeltaNet
condition: cp_size > 1
```

标准派生 plan 声明：

```python
ModuleShardingSpec(
    in_src={"hidden_states": {"cp": Shard(1)}},
    in_dst={"hidden_states": {"cp": Shard(1)}},
    out_src={"output": {"cp": Shard(1)}},
    out_dst={"output": {"cp": Shard(1)}},
    inner_target="self",
    inner_wrapper="qwen3_8_gdn_ulysses_interface",
    region_dispatch=False,
)
```

`inner_target="self"` 表示 wrapper 安装在 `linear_attn` 上，但 wrapper 中的 `cp_forward` 只负责 primitive 的临时
替换和恢复，实际模型 forward 始终调用捕获的 `original_forward`。

该配置应由标准 ShardingPlanner 模板派生，不通过 `plan_overrides` 临时启用。

纯 CP 不要求每个模型参数都具有 TP/EP placement。当前全参数 coverage check 应按并行轴收窄：

```python
if tp_size > 1 or ep_size > 1:
    self._check_all_trainable_params_covered(plan, model)
```

CP 仍需校验 LinearAttention boundary、sequence placement 和 inner wrapper 是否完整。

## 9. 参数、FSDP 与梯度

该方案不按 CP 永久切分参数，也不创建新的 `nn.Parameter`：

- input projection 继续使用完整权重处理本 rank 的 local sequence；
- Conv1d 继续使用完整 channel 权重处理 local sequence 和 halo；
- GDN rule 不包含模型参数，只对 activation heads 做临时 CP layout 转换；
- norm 与 `out_proj` 继续使用完整权重处理恢复后的 local sequence；
- checkpoint 参数名称和 shape 保持不变。

AllToAll 和 halo 通信必须使用可微 collective。Backward 自动执行逆通信，参数梯度继续交给 FSDP 在 DP×CP mesh 上
归约和分片。

与“投影后立即 CP-to-HP”的 Megatron 实现相比，本方案不会为 Conv1d、`A_log` 和 `dt_bias` 创建 CP-local 参数
view；它们继续在 local sequence 上按完整 channels 计算。两种方案的数学结果相同，但计算与通信切分点不同。

## 10. 支持范围与约束

首批支持：

- training；
- contiguous sequence shard；
- `use_cache=False`；
- non-packed sequence；
- eager GDN rule；
- `TP=1, CP>1`。

首批 fail-fast：

- cached decode 或 `use_cache=True`；
- packed sequence；
- FLA fast path 未完成接口识别；
- non-contiguous/head-tail/zigzag sequence shard；
- value heads 不能整除 CP；
- Conv1d 或 GDN rule wrapper 未触发。

Wrapper apply 阶段至少校验：

```text
num_value_heads % (TP × CP) == 0
local sequence length >= convolution halo width
GDN rule input sequence dimension == 1
GDN rule input head dimension == 2
```

当前 AutoModel dataloader 按 CP rank 连续切分 sequence，因此按 CP rank 顺序执行 CP-to-HP 可以恢复原始 token
顺序。如果以后支持 head-tail/zigzag load balance，必须在 GDN rule 前恢复 chronological order，并在返回前恢复原
layout。

## 11. TP+CP 扩展

后续启用 TP+CP 时，GDN rule 接口输入已经是 TP-local heads。CP AllToAll 在该布局上继续切分：

```text
global heads
  → TP-local heads
  → TP×CP-local heads + global sequence
```

因此要求：

```python
num_value_heads % (tp_size * cp_size) == 0
```

CP wrapper 已接收 `tp_mesh` 和 `cp_mesh`，不需要增加新的注入接口。TP 对投影和参数的处理仍由标准 TP plan
负责；CP wrapper 只处理 TP-local activation。

## 12. 测试方案

### 12.1 Collective helper UT

验证：

- CP-to-HP 后得到完整、顺序正确的 sequence 和 local heads；
- HP-to-CP 是 CP-to-HP 的逆变换；
- `query/key/value/g/beta` 使用相同的 head-to-rank 映射；
- backward 梯度与单卡一致；
- halo exchange 的 rank 0 零填充和相邻 rank 边界正确。

### 12.2 Primitive wrapper UT

使用假的原始 forward 和 primitive 验证：

- wrapper 调用原始 forward，不包含复制的 GDN forward；
- Conv1d 和 GDN rule 均被拦截一次；
- `try/finally` 在 forward 异常时也能恢复原始 primitive；
- primitive 未触发时 fail-fast；
- 不支持的 cache、packed 或 fast-path 分支 fail-fast。

### 12.3 单层数值对拍

固定权重、输入和随机性，对比 `CP=1` 与 `CP=2`：

- GDN output；
- input gradient；
- 每个参数的 gradient；
- CP shard 起始 token 的 Conv1d output；
- GDN rule output；
- grad norm。

### 12.4 Qwen3.8 训练验证

按以下顺序验证：

1. `TP=1, CP=1` FSDP baseline；
2. `TP=1, CP=2`；
3. 固定随机性运行多个 step，对比 loss 和 grad norm；
4. 对比峰值 allocated/reserved memory；
5. 后续增加 `TP=2, CP=2`。

适配完成的最低标准是 CP=2 能完成权重加载、forward、backward 和 optimizer step，并且 loss/gradient 与 CP=1
baseline 在既定容差内一致。只通过 planner 或只完成 forward 不能视为适配完成。

## 13. 实施顺序

1. 实现可微 previous-rank Conv1d halo helper；
2. 实现 GDN rule CP-to-HP/HP-to-CP helper；
3. 实现 primitive-interception inner wrapper 和 misfire detection；
4. 将 wrapper 注册到 `INNER_WRAPPER_REGISTRY`；
5. ShardingPlanner 自动识别 LinearAttention boundary 并派生 wrapper；
6. 将全参数 coverage check 收窄到 TP/EP；
7. 增加 helper、wrapper 和单层数值 UT；
8. 在 Qwen3.8-27B 上验证 `TP=1, CP=2`；
9. 验证通过后扩展 fast path 和 `TP=2, CP=2`。
