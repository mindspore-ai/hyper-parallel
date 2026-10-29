# KDA / Gated DeltaNet 的 TND 上下文并行

TND 将多个独立文档的 token 连续拼接。文档边界必须同时约束 ShortConv、状态递推及反向状态梯度；
仅改变张量形状或向最终 attention 算子传入长度，不能防止前面的卷积混入其他文档。

## 支持范围

| 层与执行方式 | TND 支持 |
|---|---|
| KDA 本地层、P2P、纯 Ulysses | 支持；可选 eager 参考与 Triton-Ascend |
| KDA AllGather、分组 AllGather + P2P、Recursive Doubling | 支持；使用 Triton-Ascend |
| KDA Ulysses + P2P / AllGather / 分组 AllGather + P2P / RD | 支持；保持原有并行配置入口 |
| GDN 本地层、P2P、纯 Ulysses | 支持；可选 eager 参考与 Triton-Ascend |

层入口接受 `[T, hidden_size]` 或 `[1, T, hidden_size]`，输出保持输入维数。
投影后的算子接受 `[T, H, D]` 或 `[1, T, H, D]`。TND 的批维为 1；多个文档由累计长度表示。
所有 CP rank 按时间顺序持有等长、连续、非空的 token 区间。
可在已有 DP×CP mesh 的 CP 子域中执行，不要求全局 rank 连续。Ulysses 仍要求相关 head 数能被其大小整除。

融合路径沿用当前后端的 BF16 输入、FP32 gate / 状态、K=V=128 和 chunk_size=64 约束。
TND 的真实文档和局部片段不必是 64 的倍数；末尾不足一个 chunk 的摘要使用显式零填充。
未提供 TND 元数据时保留定长路径；定长序列末尾同样允许不足64的 chunk。

此入口不覆盖 zigzag 分片、不等长或空 rank、TP 与 CP 同时开启、推理缓存，以及任意 `seq_idx` 或
Megatron `packed_seq_params` 的自动转换。GDN 的可选无 gate 分支也不属于本特性。
混合模型中的 full-attention 层仍需要单独配置支持同一 TND 分片方式的后端；仅替换线性注意力层不足以
使整个模型支持 TND。

## 文档边界与局部片段

模型或整层 wrapper 的 `cu_seq_lens` 表示**当前 CP 组完整 token 流**的累计长度，包含开头的 0 和总长度。
例如 CP4、每 rank 4096 token：

```python
output = layer(local_hidden, cu_seq_lens=(0, 6145, 12288, 16383, 16384))
```

第二个 rank 的区间为 `[4096, 8192)`，局部累计长度为 `(0, 2049, 4096)`：
第一段继续前一 rank 的文档，第二段从真实文档边界重新开始。CP 切口本身不是文档 reset。

模型级别接受 `cu_seq_lens`、`cu_seqlens`、`cu_seq_lens_q/k`，多个别名同时出现时必须一致。
已有批处理 adapter 可以提前构造 `PackedSequenceInfo`，以 `packed_sequence_info` 传入，避免逐层读取
设备上的累计长度。对象保存不可变的全局边界和本地位置，按设备、chunk 大小缓存只读内核索引。
不要在调用后改写其设备索引张量。

直接调用 `chunk_kda` / `chunk_gated_delta_rule` 时，`cu_seqlens` 则是**传入算子这一段 token**的局部累计长度；
不要将这个底层契约与模型级全局边界混用。Ulysses 后的局部边界由原始全局区间重新推导，不能直接拼接
各 rank 的局部累计长度，否则会把人工 CP 切口误当成文档开始。

边界必须严格递增。若数据管道为对齐增加尾部 padding，应将它声明为独立的非空末尾样本，并在数据管道中
屏蔽对应 loss；不能将 padding 混入最后一个真实文档，或仅传递一个含 0 的 attention_mask 来代替文档边界。
跨文档 labels、position_ids 和 loss_mask 继续由数据管道负责。

## 配置

KDA 沿用 [KDA 上下文并行](kda_context_parallel.md) 的 wrapper、`state_cp_method`、`ulysses_degree` 和
`allgather_group_size`。TND 不增加一套 CP 方法，也不在每个 batch 创建通信组。
例如 CP8 的 U2 + 组内 AG2 + 组间 P2P：

```yaml
plan_overrides:
  - match: "model.layers.0.self_attn.forward"
    when: cp
    region_dispatch: false
    replace:
      kind: method
      implementation:
        _target_: hyper_parallel.models.kimi_k3.adapter.distributed.context_parallel.kimi_delta_attention_cp_wrapper
        backend: triton
        state_cp_method: grouped_allgather_p2p
        ulysses_degree: 2
        allgather_group_size: 2
```

`replace.kind: method` 使用实际模块 FQN 的 glob 加 `.forward` 方法后缀。以上具体层名只是示例；
应为模型里的所有 KDA 层配置相应规则，
不能把它应用到同名的 full-attention 层。

在 `TextParallelBatch` 的现有配置中增加模型 runtime adapter：

```yaml
dataloader:
  get_batch:
    _target_: hyper_parallel.data.batching.TextParallelBatch
    source_type: online
    attention_mode: compressed
    runtime_input_adapter:
      _target_: hyper_parallel.models.kimi_k3.adapter.data.runtime.KimiK3PackedRuntime
```

这里展示的是配置增量，token packing、并行大小和 full-attention 后端仍使用模型自己的训练配置。
`attention_mode: compressed` 保留紧凑累计长度而不创建平方大小的 dense mask，并不选择或实现 full attention。
Qwen3.5 使用 `hyper_parallel.models.qwen3_5.adapter.data.runtime.Qwen35PackedRuntime`。
这两个 adapter 从全局 `cu_seq_lens` 和当前 CP 坐标构造元数据，并保留供 full-attention 层消费的全局长度别名。

Qwen3.5 的线性 decoder 还需要显式转发元数据。为实际线性层的 decoder 和 mixer 分别配置 wrapper：

```yaml
plan_overrides:
  - match: "model.layers.0.linear_attn.forward"
    when: cp
    region_dispatch: false
    replace:
      kind: method
      implementation:
        _target_: hyper_parallel.models.qwen3_5.adapter.distributed.context_parallel.qwen3_5_gdn_p2p_cp_wrapper
        backend: triton
  - match: "model.layers.0.forward"
    region_dispatch: false
    replace:
      kind: method
      implementation:
        _target_: hyper_parallel.models.qwen3_5.adapter.distributed.context_parallel.qwen3_5_packed_decoder_wrapper
```

纯 Ulysses 将 mixer wrapper 换成 `qwen3_5_gdn_ulysses_cp_wrapper`。CP1 的 packed 控制使用
`qwen3_5_gdn_packed_local_wrapper`，且不设置 `when: cp`。decoder wrapper 只能匹配线性 decoder；
不能用 `model.layers.*.forward` 无区别地覆盖 full-attention 层。它保留参数归属与 FQN，并让非 packed 调用继续
使用模型原来的 forward。以上不是完整 Trainer 配置，新增适配器本身也不代表所有 Transformers 版本均已验收。

## 纯 Ulysses 的后端选择

KDA 与 GDN 的纯 Ulysses 整层 wrapper 用一个 `backend` 同时选择 attention core 和 ShortConv，
定长 dense 与 packed 共用该配置：

```python
from hyper_parallel.distributed.context_parallel.kimi_delta_attention import KimiDeltaAttentionLayerUlyssesCP
from hyper_parallel.distributed.context_parallel.gated_delta_net import GatedDeltaNetUlyssesCP

kda_layer = KimiDeltaAttentionLayerUlyssesCP(original_kda_layer, cp_mesh, backend="triton")
gdn_layer = GatedDeltaNetUlyssesCP(original_gdn_layer, cp_mesh, backend="triton")
```

`eager` 使用 Torch 参考实现；`triton` 使用 FLA 的注意力与卷积算子，需要兼容的
FLA / Triton-Ascend 环境。显式选择的后端不做静默回退。
融合卷积要求 stride=1、dilation=1、depthwise 因果零填充和 SiLU。

Dense 卷积保留 `[B,S,C]` 的 batch 隔离；packed 由同一公共适配传递真实文档 offsets，
并保留已有 halo/reset 语义。各调用独立保存反向所需张量和元数据，不共享可写工作区。

## 状态交换与生命周期

局部前向摘要 S 只描述最后一个文档片段的零初态输出；反向摘要 G 只描述第一个片段的零终态梯度。
如果本 rank 内存在真实 reset，入口状态对出口的有效转移为零，不能用末片段 M 的转置替代首片段的反向关系。

- Triton P2P：真实文档边界恰好落在 rank 切口时，两侧成对跳过状态收发；内部 reset 允许先发送独立的 S/G，
  再在本地实际需要入口状态的消费者处等待。仍使用初始化时创建的 CP 通信组。
- AllGather / 分组 AllGather：用显式零转移表示 reset，保留原有正向、逆向有序扫描。每个在途调用仅保存
  本地 M；收集的 S/M 不跨前反向保留。反向重新收集本地 G/M。
- Recursive Doubling：根据真实文档边界构造不可变的逐调用计划，只执行有消费者的倍增收发与合成，
  backward 沿同一计划的反向边消费必要转移缓存。Ulysses 后先重建文档区间，再沿 state 子域规划。
  尾文档不延续到下一状态 rank 时不生产 S；只有入口会影响该出口时才生产 M。头文档延续时仍需 G，
  所以 rank 内部 reset 不等于整张卡都无需通信。没有活跃倍增轮的 rank 跳过 S/M 打包；计划不创建新通信组。
- ShortConv：保留既有 halo 交换，按全局文档边界屏蔽越界 tap；该 mask 同时阻断越界的反向梯度。
- backward：仅在内核调用处将紧凑入口状态、出口梯度展开到局部片段槽位，元数据和本地 M 由各自 autograd
  调用持有。不同 microbatch、反序 backward 和非重入 checkpoint 不共享可写通信缓冲区。

剪枝只依据真实文档边界和消费者，不依据矩阵数值大小，不改变保留计算的合成或舍入顺序。
同一元数据计划可以只读复用，各次调用的状态张量和反向缓存仍独立持有；没有活跃边不代表可以跳过本地 KDA。

AG 保存的本地 M 不随文档数量线性增长；RD 缓存随实际参与轮数增长。原生 varlen 内核的临时状态槽、chunk 索引、conv 中间量和 AG 临时缓冲区
仍有各自成本；不能将“仅保存一份 M”解释为整层峰值恒定。

## 数值语义

本特性要求不同文档之间的导数严格为零，并保持同一 CP 配置在重算和多个在途调用下的一致性。
FP64 / eager 对照用于检查分段、halo 和通信语义；NPU 对照还需覆盖实际低精度内核及反向。

RD 可通过既有 `state_cp_method: recursive_doubling` 选择，`allgather_group_size` 必须为 1。
当前限制为 Triton-Ascend、固定 K=V128、等长连续分片；不扩展 GDN 的协议范围。
此组合沿用 dense RD 的有限精度合成顺序，不承诺不同协议逐位一致。RD 需显式选择；
多文档 reset 用例与单篇跨域长文档须分别验收，当前不据此启用自动协议选择或宣称完整模型训练收敛。

KDA 状态摘要沿用已有的有限精度数值契约；GDN 的 G 将每个 chunk 的有符号更新先相减、再加到状态，
与同库原生 dhu 的舍入位置配对。此处实数公式不变，定长 GDN 也使用这一配对顺序。带 BF16 舍入的原生状态反馈不是严格仿射程序，因此换 CP 切分或
比较仿射摘要与整文档原生递推时，不承诺逐位一致。本文档不把历史小模型多步实验等同于完整模型训练收敛验收。

## 底层 GDN 状态算子的可选输入

GDN 前向状态算子同时包含 key-gate 的布局和地址修复：`gk` 在入口转成与 kernel 一致的 head-major
布局，packed 分支使用全局 token 长度和文档起点寻址；`g=None`、`save_new_value=False` 不再操作空对象。
这项修复不改变本文常规 scalar-gate GDN 的递推算术，也不扩大公开训练接口：仅 key-gate 的完整 backward
尚未在本特性中获得支持，不能根据前向可用推断整条训练路径可用。
