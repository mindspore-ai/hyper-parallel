# KDA 上下文并行使用指南

Kimi Delta Attention（KDA）通过递推状态连接相邻序列片段。本指南介绍 KDA 的 P2P、AllGather（AG）、
Ulysses 及其混合方式，适用于 Kimi K3 的 KDA 层和库内 `KimiDeltaAttention` 模块。

KDA 的 AG 收集的是状态摘要与转移矩阵，不是整段 K/V。模型调用前仍需将输入按序列连续切分到各个
CP rank，并保持 CP mesh 中的 rank 顺序与全局 token 顺序一致。

## 并行方式

设 CP 总大小为 `P`，Ulysses 大小为 `U`，状态 CP 大小为 `R = P / U`，组内 AG 大小为 `g`。

| 方式 | 执行过程 | 适用考虑 |
|------|----------|----------|
| P2P | 各 rank 生成本地摘要，沿序列顺序传递边界状态；反向沿相反方向传递边界梯度。 | 通信工作区较小，依赖链随状态 CP 大小增长。 |
| AG | 各 rank 收集状态摘要和转移矩阵，通过融合算子合并所需的前缀或反向后缀。 | 消除逐 rank 的通信依赖，临时收集缓冲区随状态 CP 大小增长。 |
| Ulysses | AllToAll 将序列分片转换为 head 分片，计算后恢复原序列分片。 | 减少每个 rank 的 head 数，要求 head 数能被 U 整除。 |
| Ulysses + P2P / AG | 先在 Ulysses 组内交换序列和 head，再在状态 CP 组内执行 P2P 或 AG。 | 用 head 分工减少状态通信量，并减小状态 CP 的规模。 |
| AG + P2P | 连续 g 个状态 rank 组内 AG，由各组最后一个 rank 组成 P2P 链，再向组内广播输入边界。 | 将 AG 临时缓冲区限制到组宽 g，保留组间的串行通信依赖。 |
| Ulysses + AG + P2P | 先做 Ulysses，再在 R 个状态 rank 上执行组内 AG 和组间 P2P。 | 同时调整 head 分工、收集缓冲区和组间链长。 |

组内 AG + 组间 P2P 的前向先收集 S/M；组 owner 接收上一组边界后，按顺序应用组内摘要并发送到下一组。
组内各 rank 根据广播的组入口状态恢复自己的输入状态。反向收集 G/M，按相反的组和 rank 顺序执行。
组间链有 `R/g` 个 owner，但 owner 仍需顺序应用组内摘要，不能将通信跳数的减少直接等同为计算量的减少。

## 配置入口

在训练配置的 `plan_overrides` 中，为 KDA 层选择统一 wrapper：

```yaml
plan_overrides:
  - match: "*.self_attn"
    when: cp
    region_dispatch: false
    inner_target: self
    inner_wrapper:
      _target_: hyper_parallel.models.kimi_k3.adapter.distributed.context_parallel.kimi_delta_attention_cp_wrapper
      backend: triton
      state_cp_method: grouped_allgather_p2p
      ulysses_degree: 2
      allgather_group_size: 2
```

`match` 匹配模型 `named_modules()` 给出的模块完整路径（FQN），采用大小写敏感的 glob，
不是模块类型判断，也不是正则表达式。`*` 可以跨越路径中的点，因此 `"*.self_attn"` 会匹配
`model.layers.0.self_attn` 和 `model.layers.12.self_attn`，但不匹配根级的 `self_attn`；
后者应写 `match: "self_attn"`。模型同时包含 KDA 和其他注意力、且都使用 `self_attn` 名称时，
应为实际 KDA 路径配置精确匹配或更窄的 glob，不能用该通配符自动识别 KDA。
`when: cp` 表示仅在 CP 大于1时启用；`inner_target: self` 作用于匹配模块本身，
`region_dispatch: false` 让 wrapper 执行其本地张量路径。

上例在 CP8 下形成 U2、状态 CP4、组内 AG2 和两个组 owner 的 P2P 链。
CP 总大小由训练的并行配置决定。`state_cp_method` 只选择状态 CP 方式，完整方案还需结合
`ulysses_degree` 和 `allgather_group_size` 判断；例如 `state_cp_method: p2p` 配合 `ulysses_degree: 2`
表示 Ulysses + P2P。统一入口内部固定使用 64-token chunk，不暴露 `chunk_size` 配置；
`eager` 和 `triton` 均采用这一大小。

| 参数 | 默认值 | 含义与约束 |
|------|--------|------------|
| `backend` | `triton` | 本地 KDA 后端。AG 和分组 AG 需要 `triton`；已有 P2P/Ulysses 支持 `eager`。 |
| `state_cp_method` | `p2p` | 状态 CP 方式，可选 `p2p`、`allgather`、`grouped_allgather_p2p`。 |
| `ulysses_degree` | `1` | U；正整数，必须整除 CP 大小，且 query/key head 数和 value head 数都必须能被 U 整除。 |
| `allgather_group_size` | `1` | g；每个组内 AllGather 包含的状态 rank 数，必须整除 R。仅用于 `grouped_allgather_p2p`；纯 `allgather` 自动收集整个 R 组，此项保持1。 |

CP8 的常用组合：

| 方式 | `state_cp_method` | `ulysses_degree` | `allgather_group_size` |
|------|---------------------|------------------|--------------|
| P2P | `p2p` | 1 | 1 |
| AG | `allgather` | 1 | 1 |
| Ulysses + P2P | `p2p` | 2 | 1 |
| Ulysses + AG | `allgather` | 2 | 1 |
| AG + P2P | `grouped_allgather_p2p` | 1 | 2 |
| Ulysses + AG + P2P | `grouped_allgather_p2p` | 2 | 2 |
| 纯 Ulysses | `p2p` | 8 | 1 |

`U=1` 时不做 Ulysses；`U=P` 时状态 CP 大小为 1，只运行本地 KDA。
分组协议在 `g=1` 时退化为 P2P，在 `g=R` 时退化为 AG。
已有 `kimi_delta_attention_p2p_cp_wrapper` 和 `kimi_delta_attention_ulysses_cp_wrapper` 继续可用，默认行为不变。
它们及底层 eager 算子的 `chunk_size` 参数保留，供直接调用和参考验证使用。

## 输入与 mesh 约束

- 当前融合 AG 路径面向 Ascend 上的 BF16 训练，key/value head 维度为 128，使用 lower-bounded gate。
- 各 CP rank 的本地序列长度相同，Triton 路径要求完整的 64-token chunk。
- 使用一维 CP mesh，rank 自然递增且与序列顺序一致。可以传入根 mesh 的 CP 子 mesh，例如 DP×CP；
  不要求 CP 等于 WORLD，也不要求子组的全局 rank 连续。
- 混合方式需要保留 CP 子 mesh 与根 mesh 的关联，以便一致地创建子组；不支持任意置换 rank，
  用于拆分的最终根 mesh 必须覆盖 WORLD；仅保留一个局部 root 仍不足以安全创建所有兄弟组。
- ShortConv 在原始 CP 序列邻居之间交换 halo，之后才执行 Ulysses，保证卷积边界不随 head 重排而改变。
- 当前入口不支持 packed/变长序列、推理缓存或同时启用 TP 和 CP。CP 不负责参数梯度归约，
  训练仍需使用框架相应的数据并行与梯度管理配置。

## 显存与选择建议

S 是本地零初态的输出状态，M 是本地片段的状态转移矩阵，G 是反向零边界的梯度摘要。
AG 和分组 AG 都只为每次在途调用保留一份本地 M，反向重新收集 G 与该 M；
前向收集到的 S/M 不跨前反向缓存。融合边界算子只改变合并的执行方式，不改变本地 KDA 的 S/M/G 算术。

以 `B=1、H=96、K=V=128` 为例，一份 FP32 状态为 6 MiB；不使用 Ulysses 时，一份 S/M 打包数据为
12 MiB。纯 AG 的接收缓冲区随 R 增长，分组 AG 的接收缓冲区随 g 增长；Ulysses 将每份状态的 head 数
缩小到原来的 `1/U`。这些是边界缓冲区的规模，不代表整层激活显存或整次训练峰值。

选择时应同时考虑 head 整除限制、边界通信的串行长度和临时收集缓冲区：显存充足、状态 CP 较小时可考虑 AG；
需要限制收集规模时可选择分组 AG + P2P；head 数允许时可先加入 Ulysses，缩小状态 CP 的规模。
不同组合会改变浮点计算顺序，不承诺与单卡递推逐位相同；具体训练规模仍需按模型的精度要求选择。

## 实现说明

配置与执行的逐函数说明、CP8三维分组示例，以及融合状态扫描和缓存的关系见
[KDA 上下文并行实现说明](kda_context_parallel_internals.md)。
