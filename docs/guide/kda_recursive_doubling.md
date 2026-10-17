# KDA 递归倍增上下文并行

Kimi Delta Attention（KDA）的递归倍增（Recursive Doubling，RD）在各 rank 的本地状态摘要上进行
有序扫描，用对数级通信阶段替代 P2P 的逐 rank 状态传递。输入 token 保持原来的序列分片，
本地 KDA 仍使用现有 S/M/G 摘要算子与 FLA Triton-Ascend 核心。

当前 RD 为显式启用的实验性选项。固定摘要的前后向公式和缓存生命周期已有验证，但弱遗忘输入下的
整层梯度与 P2P 仍有超过现有兼容门槛的差异，尚不能作为已经完成训练精度验收的默认方案。

## 执行方式

每个 rank 生成本地状态摘要 S 和转移矩阵 M，对输入状态 h 的作用写作 `S + M @ h`。
前向按距离 1、2、4 等依次合成相邻的有序区间；区间顺序不能交换。最后将 inclusive 结果向下一个
rank 传递，得到本地片段实际需要的 exclusive 输入状态。

反向先交换本地梯度摘要 G，再逆序遍历前向的合成阶段，使用各阶段保存的 M 转置传递边界梯度。
每次 forward 返回自己的缓存，交给对应的 autograd 上下文保存；协议对象不保存调用相关的张量。
因此同一层有多次在途调用时，不会用后一轮 forward 的 M 覆盖前一轮所需的数据。
非重入 activation checkpoint 会按相同 rank 顺序重新执行前向通信。

RD 的打包、状态相加和紧凑矩阵写出使用点式融合 kernel。矩阵乘法及其加法顺序与原缓存 RD 一致，
不使用低精度通信、历史截断或近似低秩。

## 配置

通过模型并行计划为 KDA 层选择统一入口：

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
```

`match` 必须匹配真实的 KDA 层。模型包含其他注意力类型时，只对 KDA 层应用该规则。
输入仍由训练代码按全局 token 顺序切分；CP wrapper 不负责切分数据集或归约模型参数梯度。

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `state_cp_method` | `p2p` | 状态 CP 方法；本增量支持 `p2p`、`recursive_doubling`。 |
| `backend` | `triton` | 本地 KDA 后端；RD 要求 `triton`，既有 P2P 也支持 `eager`。 |

统一入口内部固定使用 64-token chunk，不提供 `chunk_size` 配置。
旧 `kimi_delta_attention_p2p_cp_wrapper`、`kimi_delta_attention_ulysses_cp_wrapper` 及底层算子的参数保持兼容。
RD 的方法名表示状态 CP 这一维度；本增量不提供 Ulysses×RD 或 summary head 分工配置。

## 输入和运行条件

- 当前整层接入使用 Ascend NPU、BF16、本地 key/value head 维度128、lower-bounded gate。
  各 rank 的本地序列长度相同，并由完整的64-token chunk组成。
- 使用一维、有序的 CP mesh，mesh 内的 rank 顺序与全局序列片段顺序一致。
  可以使用 DP×CP 的 CP 子组，全局 rank 不必连续；不支持任意 rank 置换。
- ShortConv 仍在原始序列邻居之间交换 halo；输入投影、输出投影与模型参数归属不变。
- 所有 rank 必须按一致顺序调用同一 CP 方法。每次调用拥有独立缓存，不意味着可以在不同 rank
  任意交换两个调用的通信次序。
- 当前为 eager 通信执行，不支持图捕获；NCCL、packed/变长序列、推理缓存和 TP×CP 不在本次接入范围。
- HCCL 在已验证的 Torch/torch-npu 2.10.0、CANN 9.1.0-beta.3、Ascend910B3 环境下使用 coalescing；
  其他环境回到公开的 `batch_isend_irecv`。选择公开接口不代表新的设备和软件组合已经通过训练验证。
  CPU Gloo 路径用于独立协议检查，不是 CPU 上的融合 KDA 后端。

## 时间和显存取舍

设 CP 大小为 P，单份 FP32 状态大小为 `W = B × H × 128 × 128 × 4` 字节。
P大于1时，每个方向包含 `ceil(log2(P))` 个倍增阶段和一次 exclusive 边界交接。
CP8 的前反向合计为8个阶段；P2P则有14个相邻通信跳。
阶段数量不等于全组消息数，也不直接等于整层耗时。

| 项目 | P2P | 当前缓存 RD |
|------|-----|-------------|
| 边界依赖深度 | O(P) | O(log P) |
| 全组通信与合成工作 | O(P) | O(P log P) |
| 最坏 rank 跨前反向的 M 缓存 | O(W) | O(W log P) |

RD 的非末轮前向消息包含 S/M，约2W；末轮、exclusive交接和反向消息为状态大小W。
以 B1/H96/K=V128 为例，W为6 MiB，CP8 最坏 rank 的转移缓存为18 MiB，CP256为48 MiB。
这些值只计算每次在途调用的 M 缓存，不包括通信工作区、模型激活或多层累积。
缓存使用紧凑独立存储，避免一个连续 M view 意外保留整个生产者工作区。

小 CP 时，较大的消息、矩阵合成和每轮启动可能抵消 RD 的轮数优势。
大 CP 时还需要考虑通信并发、最慢 rank 的摘要就绪时间和多层缓存。
目前没有大规模超节点实测，不能仅凭复杂度承诺整层加速。

## 数值适用范围

有序矩阵合成在实数算术下等价，但浮点乘加不满足严格结合律。RD 与 P2P 改变合成顺序后，
小的边界差异可能在 BF16 状态反馈及后续梯度中产生更大的互差。
固定 S/M/G 相对独立 FP64 的边界检查、同 RD 的缓存生命周期检查，以及整层相对 P2P 的兼容检查
是三种不同的验证，不能互相替代。

当前敏感整层检查保留原相对 L2 门槛0.003；不通过时不能宣称具备正式训练资格。
与 P2P 不一致本身也不能判断哪一条路径更接近独立参考。默认仍选择 P2P。
