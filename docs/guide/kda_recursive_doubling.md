# KDA 递归倍增与 Ulysses 混合上下文并行

Kimi Delta Attention（KDA）可以将各序列片段的状态写成仿射摘要 `S + M @ h`。
递归倍增（Recursive Doubling，RD）按距离1、2、4等合成相邻区间，
用对数级通信阶段代替 P2P 的逐片段传递。区间必须保持原有 token 顺序。

本功能建立在 [KDA 混合上下文并行](kda_context_parallel.md) 的公共 Layer/Core 入口上。
可以单独使用 RD，也可以先通过 Ulysses 交换 token/head，再对每个 head 分片执行 RD。
本地 prepare、S/M/G、FLA h/dhu 及投影/卷积沿用公共实现，改变的是状态交换与调用缓存。
默认 state_cp_method 仍为 p2p；RD 需要显式选择。

当前实现仍有整层数值兼容性待验收，适用于显式评估，尚不建议作为训练默认方案。
固定摘要的 FP64 检查和调用缓存检查已通过，但弱遗忘输入的整层梯度对照仍超出既有兼容预算；
Ulysses×RD 的完整 NPU 验收也须单独完成。具体状态见关联 PR 和 issue，不能由协议检查推导训练资格。

## 配置

```yaml
plan_overrides:
  - match: "*.self_attn"
    when: cp
    region_dispatch: false
    inner_target: self
    inner_wrapper:
      _target_: hyper_parallel.models.kimi_k3.adapter.distributed.context_parallel.kimi_delta_attention_cp_wrapper
      backend: triton
      state_cp_method: recursive_doubling
      ulysses_degree: 2
```

match 是模型模块名称的匹配规则，只应匹配真实的 KDA 层；混合其他注意力类型时需要收窄范围。
ulysses_degree=1 表示纯 RD，大于1表示 Ulysses×RD；它必须整除总 CP 大小以及 query/value head 数。
RD 不使用 allgather_group_size，该参数保持默认1。局部融合计算固定64-token chunk。

例如总 CP=8、Ulysses=2，自然 rank 顺序下 U 组为 [0,1]、[2,3]、[4,5]、[6,7]，
两个 RD 组分别为 [0,2,4,6] 和 [1,3,5,7]。本地 core 处理两倍 token 和半数 head，
输出再通过 Ulysses 恢复原始 token 分片。ShortConv halo 仍沿原始序列邻居交换；
A_log/dt_bias 按 U 的 head 分片使用。输入切分和模型参数梯度归约由训练框架负责。

## 前反向与缓存

前向各 rank 生成 S/M，按距离递增合成有序区间，最后通过相邻交接得到 exclusive 输入状态。
反向使用本地 G，并逆序执行前向扫描的伴随操作，应用各阶段保存的 M 转置。
前向非末轮消息包含 S/M，末轮、相邻交接和反向消息只包含一份状态。

协议对象只保存通信元数据。每次 forward 返回独立的转移矩阵缓存，
由该调用的 autograd 上下文保存，支持多次 forward 在途、倒序 backward 和非重入 checkpoint。
AG/分组协议使用同一个缓存接口，但只保留本地 M；RD 保留所需的各轮 M。
缓存保持紧凑存储，不因一个连续视图而保留整块生产者工作区。
所有 rank 仍须以一致顺序执行通信；独立缓存不意味着可以任意重排调用顺序。

矩阵乘法及合成顺序沿用缓存 RD，打包、加法和紧凑矩阵写出使用点式融合 kernel。
不使用低精度通信、低秩近似或历史截断。

## 支持范围

- Ascend NPU、BF16、K=V128、lower-bounded gate、完整64-token chunk、各 rank 等长序列。
- 有序的一维状态 CP mesh，支持保留 WORLD root 的 DP×CP 子mesh；全局 rank 可以不连续。
  mesh 逻辑顺序必须与通信组的 slot 顺序和 token 顺序一致。
- 当前为 eager 通信执行；不包含图捕获、NCCL、变长 packed 输入、推理缓存或 TP×CP。
- Torch/torch-npu 2.10.0、CANN 9.1.0-beta.3、Ascend910B3 的已验证组合使用 HCCL coalescing，
  其他环境回退公开的 batch_isend_irecv；回退不等于该环境已通过训练验证。
- CPU Gloo 仅用于协议检查，完整融合 KDA 路径仍要求 NPU。

## 时间与显存取舍

设总 CP 为 P、Ulysses 为 U、状态 CP 为 R=P/U，
单份 FP32 状态大小 `W_U = B × (H/U) × 128 × 128 × 4` 字节。
R>1时每方向有 ceil(log2 R) 个倍增阶段和一次 exclusive 交接；R=1走本地 KDA。

| 项目 | P2P | 缓存 RD |
|------|-----|---------|
| 状态依赖深度 | O(R) | O(log R) |
| 状态组全部通信/合成工作 | O(R) | O(R log R) |
| 最坏 rank 每调用转移缓存 | O(W_U) | O(W_U log R) |

B1/H96/K=V128、无 U 时，一份状态6MiB，CP8最坏 rank 转移缓存18MiB，CP256为48MiB。
CP256/U2时状态组128、每份状态3MiB、最大缓存21MiB；U4时分别为64、1.5MiB、9MiB。
这些数值不含通信工作区、模型激活、编译器workspace及多层在途调用。

RD 阶段少不保证整层更快：前向消息更大，M 合成增加计算；并发通信、摘要就绪时间和轮间交接也影响尾部。
U 增大会减少状态字节及RD轮数，但增加本地递推长度、降低head并行度并引入token级AllToAll。
选型应比较同数值配置下最慢 rank 的整层时间及实际峰值显存，不能只比较log R和R。

## 数值语义

S 与固定 FLA Ascend h 的零入口结果配对，G 与 dhu 的零末端梯度结果配对；
M 显式形成每个 chunk 的转移并以 FP32 合成。RD 与其他方式共用这些本地算子。
浮点结合顺序不同仍可能产生差异；原生非零状态 dhu 的低精度反馈也并非严格仿射。
因此固定摘要的独立 FP64 检查、整层同输入比较和缓存生命周期检查分别验证不同的性质，不能相互替代。
