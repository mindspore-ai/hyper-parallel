# KDA 上下文并行实现说明

KDA 的状态 CP 在保持 token 序列分片的同时，计算每段的输入递推状态以及反向所需的末端状态梯度。
本文面向评审和维护代码的开发者，解释 P2P、AllGather、Ulysses 及分组组合的函数职责、通信顺序和存储约束。
配置示例见[使用指南](kda_context_parallel.md)。

## 代码职责与命名

| 文件 | 职责 |
|------|------|
| [adapter/distributed/context_parallel.py](../../hyper_parallel/models/kimi_k3/adapter/distributed/context_parallel.py) | 将模型配置变成原子 forward 改写请求，不接管模型参数。 |
| [distributed/context_parallel/kimi_delta_attention.py](../../hyper_parallel/distributed/context_parallel/kimi_delta_attention.py) | 整层投影、ShortConv、Ulysses 置换、状态 CP 调用和输出恢复。 |
| [distributed/context_parallel/kimi_delta_attention_state_cp.py](../../hyper_parallel/distributed/context_parallel/kimi_delta_attention_state_cp.py) | CP mesh 拆分、状态通信执行器、AG 和分组 AG＋P2P。 |
| [functional/kimi_delta_attention.py](../../hyper_parallel/components/functional/kimi_delta_attention.py) | 本地融合核心和自定义 autograd；将状态通信插入 S/M/G 与 FLA 算子的调用链。 |
| [_triton/kimi_delta_attention/cp_merge.py](../../hyper_parallel/components/functional/_triton/kimi_delta_attention/cp_merge.py) | 已收集摘要的前缀状态与反向后缀梯度扫描。 |

旧名称 `boundary` 表示序列片段之间的状态边界，并非网络边界或设备拓扑边界。
执行器文件与对象现在使用 `state_cp`；`AllGatherStateCP` 和 `GroupedAllGatherP2PStateCP`
直接表达状态 CP 方式。底层数学说明中的“边界状态”仍指片段输入/输出的递推状态。
配置 `allgather_group_size` 表示组内 AG 宽度，不是 Ulysses 大小，也不是单独的 head 分块数。

## 摘要、状态扫描和 M 的生产

对 rank i 的片段，实数域的摘要写成：

```text
h_out = M_i @ h_in + S_i
D_in  = M_i.T @ D_out + G_i
```

S 是零初态的局部输出；M 是局部状态转移；G 是末端状态梯度为零时的首端梯度摘要。
M 只依赖该次前向的局部转移，反向复用它；G 随本次 do 等反向输入变化，必须重新生成。
以上公式描述仿射合并关系，混合精度局部递推与不同合并次序不保证逐位相同。

`kda_cached_affine_merge_kernel` 输出一个实际状态或状态梯度，不输出整个组合后的仿射映射。
前向从全局零初态开始，顺序应用 rank 0 到 rank i−1 的 `M @ h + S`。
若写为整个前缀的 `A_prefix @ h0 + B_prefix`，当 h0=0时，结果确实等于 B_prefix；
但 kernel 每一步仍需要 M，并非忽略了转移矩阵。
即使提供非零 h0，也可以从实际 h0 顺序递推，无须显式形成 A_prefix。
当前纯 AG 融合入口固定零外部边界；分组路径的 `_scan_maps` 可以接收非零组入口。

局部 M 已由现有 `kda_state_summary_forward_from_prepared` 及其 Triton S/M 算子生成，
并不是缺少计算 A 的实现。只有需要合并两个完整映射、将组合映射继续向上传递时，
才需要 `M_right @ M_left`；本 AG 顺序扫描不需要这个额外输出。

纯 AG 前向临时收集 S/M，反向临时收集 G/本地缓存M；前向的完整收集缓冲区不跨 forward/backward 保存。
每次调用只保留自己的本地 M，S 不进入反向缓存。执行器对象只持有通信元数据，没有调用相关张量。
这使两次 forward 后倒序 backward 或非重入 checkpoint 不会相互覆盖状态。

## 从配置到执行的调用关系

```text
plan_overrides
  -> kimi_delta_attention_cp_wrapper
  -> _build_kda_cp_rewrite / _ForwardRewriteRequest
  -> LayerP2PCP 或 LayerHybridCP
  -> 投影与原始 CP 邻居的 ShortConv halo
  -> 可选 Ulysses sequence-to-head
  -> fused_chunk_kda_p2p / _KDAStateP2PFunction
       -> FLA prepare -> S/M -> state_cp.forward -> FLA h/output
       -> backward 重算 -> G -> state_cp.backward -> FLA dhu/WY/intra
  -> 可选 Ulysses head-to-sequence
  -> output gate / norm / o_proj
```

旧纯 Ulysses 整层 wrapper 使用单次五路输入打包，在收齐整个序列的本地 head 上做 ShortConv。
新混合 wrapper 则先在原始 CP 邻居上完成 halo，再做 Ulysses；两者不能仅按类名假设调用顺序相同。

## CP mesh 如何拆成三种并行

设原始 CP 大小 P，Ulysses 大小 U，状态 CP 大小 R=P/U，组内 AllGather 大小 g，外层组数 Q=R/g。
逻辑顺序可写为 `[Q, g, U]`，原 CP 局部编号为 `r = (q*g + a)*U + u`。

`split_kda_mesh(mesh, degree, names)` 不移动 token，也不改变全局 rank 编号；它将当前的一维轴
重解释为 `[原大小/degree, degree]`，再用根 mesh 的 rank map 构造子通信组。
第一次 degree=U 得到 `[kda_state, kda_ulysses]`；随后在每条状态轴上以 degree=g 拆出
`[kda_outer, kda_group]`。两次都是对相应的一维轴拆分，不是把原始 mesh 连续截成三个列表。

以 P=8、U=2、g=2，原 rank 为0到7为例：

| 外层组 q | AG 位置 a | U 位置 u=0 | U 位置 u=1 |
|----------|-----------|------------|------------|
| 0 | 0 | 0 | 1 |
| 0 | 1 | 2 | 3 |
| 1 | 0 | 4 | 5 |
| 1 | 1 | 6 | 7 |

- Ulysses 组：`[0,1]`、`[2,3]`、`[4,5]`、`[6,7]`。
- 状态 CP 组：`[0,2,4,6]` 和 `[1,3,5,7]`，分别拥有相同的 head 子集。
- AG 组：第一条状态链上为 `[0,2]`、`[4,6]`；第二条为 `[1,3]`、`[5,7]`。
- 各 AG 组最后一个 rank 是 owner；前向 owner 链为 `2→6`、`3→7`，反向相反。
- rank4、6的组入口由owner6广播，rank5、7的组入口由owner7广播；每个rank再应用自己的组内前缀。
- ShortConv 的邻居仍为原始序列的 `0→1→2→…→7`，不能改成 head 分工后的状态链邻居。

`ranks == tuple(sorted(set(ranks)))` 要求唯一且升序；并不要求连续。
例如 `[0,2,4,6]` 合法，`[2,0,4,6]` 不合法。通信组内部按自然 rank 顺序收集，当前实现要求它
与 token 时间顺序一致，因此不能任意置换 rank。若在 DP2×CP8 中使用交错 CP 组，仍对 CP 局部编号
套用相同公式，再经 root rank map 映射成全局 rank。

根 mesh 检查与“CP必须等于WORLD”不同。底层 `DeviceMesh._unflatten` 用根 rank map 枚举兄弟组，
`collectives.cc.split_group` 按统一顺序调用 WORLD 范围的 `dist.new_group`。
保留覆盖 WORLD 的最终根，才能让不同 DP 副本创建同一组列表。
只用当前副本的局部 rank 列表建一个独立 root，或在这个局部 root 下再建子 mesh，都不足以满足该约定。
因此用于混合拆分的最终 root 必须覆盖 WORLD；已创建好的纯 AG 子通信组不需要重新拆分。

## 状态通信函数逐项说明

| 函数 | 输入、输出与作用 |
|------|----------------|
| `split_kda_mesh` | 接收一维 CP mesh、内轴宽度和两个轴名；校验整除、时间顺序和最终根范围，然后返回保留根布局的二维子 mesh。仅初始化时调用。 |
| `build_kda_state_cp` | 根据 `state_cp_method` 和 `allgather_group_size` 选择执行器。P2P返回None；AG覆盖整个状态组；分组g=1退化为P2P、g=R退化为AG。 |
| `_validate_summary` | 在通信前验证S/G与M为非空 `[B,H,128,128]`、FP32、同设备；只接受CPU协议验证或NPU执行。只读取元数据。 |
| `merge_affine_summaries` | 输入按rank堆叠的S/G和M；校验rank及kernel所需stride，计算本rank的exclusive前缀/后缀。零长度前缀/后缀直接返回零。 |
| `_gather_maps` | 把已经校验的本地S/G与M沿最后一维拼为 `[B,H,128,256]`，AllGather后返回 `[P,B,H,128,256]`；缓冲区只属于当前调用。 |
| `AllGatherStateCP.__init__` | 记录预建group、本地rank和组大小，没有张量工作区。 |
| `AllGatherStateCP.forward` | 校验、收集S/M、求本rank之前的前缀状态；交给本地FLA h使用。 |
| `AllGatherStateCP.backward` | 校验、收集新G和本次缓存M、逆序求本rank之后的梯度后缀；交给本地FLA dhu使用。 |
| `_scan_maps` | 从传入的实际组入口开始，按indices逐个计算M乘状态再加S/G；反向使用M转置。保持现有P2P的顺序反馈，不预合并整组M。 |
| `GroupedAllGatherP2PStateCP.__init__` | 记录状态大组、组内AG组、rank列表和组宽，计算组编号、组内位置与owner；无共享张量缓存。 |
| `GroupedAllGatherP2PStateCP._receive` | 为owner预投递来自上一组/下一组的接收，返回输入缓冲区及work；第一/最后边界使用零。 |
| `GroupedAllGatherP2PStateCP.forward` | 先预投递接收，再组内AG；owner等入口、顺序应用整组摘要并发送；组内广播入口，各rank计算自己的前缀，返回前等待本次send。 |
| `GroupedAllGatherP2PStateCP.backward` | 收集G/M，沿反向owner链传梯度、广播组末端梯度，再算本rank之后的组内后缀；保持M复用和发送缓冲区生命周期。 |
| `kda_cached_affine_merge_kernel` | 一个程序处理一个batch/head的一块value列，FP32执行所需的顺序扫描；反向通过加载索引转置M，避免单独物化转置，不输出组合M。 |

前后向中 owner 先向下一组发送，再广播本组入口，允许后续组转发与当前组恢复发生重叠。
发送包保持引用直到 `send.wait()`；接收在实际消费前wait。组内AG可能与其他组的准备并行，
但owner需要顺序应用g个摘要，组数减少并没有把状态依赖变成对数复杂度。

## 模型入口函数逐项说明

| 函数 | 作用 |
|------|------|
| `_build_kda_cp_rewrite` | 检查CP启用、TP×CP限制及重复包装，构造执行器和forward闭包；返回改写请求，由框架提交，保留参数和原模块归属。 |
| 嵌套 `cp_forward` | 将模型原forward的参数交给选定执行器，使用wraps保留原签名元信息。 |
| `kimi_delta_attention_cp_wrapper` | 新统一配置入口；选择P2P或Hybrid整层类，将状态方式、U和AG组宽传入；内部固定chunk64。 |
| `kimi_delta_attention_p2p_cp_wrapper` | 旧P2P专用入口，保留eager默认及显式chunk_size，兼容已有调用。 |
| `kimi_delta_attention_ulysses_cp_wrapper` | 旧纯Ulysses整层入口，保留已有配置接口和五路打包路径。 |

`match` 由通用planner处理，不由这些wrapper判断类型。它匹配模块FQN的glob，`*.self_attn`的星号可跨点，
但不能自动区分KDA和其他同名attention。统一wrapper还会校验被匹配模块是否具有所需KDA结构。

## 整层执行器与辅助函数逐项说明

下表包含调用链中原有的整层辅助函数，以便同时区分既有逻辑和新增状态通信职责。

| 函数 | 作用 |
|------|------|
| `_run_local_kda` | 无状态CP通信时调用所选eager或Triton本地chunk KDA，启用Q/K归一化、gate及beta处理，返回token输出。 |
| `_is_official_kimi_layer` | 判断层是否采用官方Kimi的投影属性约定。 |
| `_num_value_heads` | 从两种受支持模块接口读取value head数量。 |
| `_value_head_dim` | 读取value每head宽度。 |
| `_key_projection_dim` | 计算完整Q/K投影通道数。 |
| `_value_projection_dim` | 计算完整value投影通道数。 |
| `_gate_lower_bound` | 读取lower-bounded gate的下界。 |
| `_uses_safe_gate` | 判断该层是否采用safe gate。 |
| `_uses_short_conv` | 判断Q/K/V是否具有需要执行的ShortConv。 |
| `_forget_gate_projection` | 选择官方低秩或库内标准投影，生成递推衰减gate logits。 |
| `_output_gate_projection` | 生成输出归一化/门控所用的gate，不是状态衰减gate。 |
| `_format_layer_output` | 保留库内层的Tensor输出或官方层的三元组输出约定。 |
| `_global_peer_rank` | 将CP局部编号映射到实际全局rank，避免把局部索引误作通信peer。 |
| `_local_tensor_at_cp_boundary` | 从已分片DTensor取本地张量；普通Tensor直接使用，不做额外数据重分布。 |
| `_all_to_all_previous_rank_halo` | 将ShortConv所需尾部只发给下一个原始CP rank；使用可微AllToAll，首rank返回零halo并保留反向连接。 |
| `_run_causal_short_conv` | 将前段halo与本地投影拼接，调用一条因果depthwise卷积并裁回本地token。 |
| `_causal_short_convs_with_cp_halo` | 合并Q/K/V尾部halo进行一次通信，再分别卷积；校验核大小和本地长度。 |
| `_slice_local_heads` | 按U位置取A_log、dt_bias等共享参数的对应head视图，保留梯度传播。 |
| `_differentiable_all_to_all_shard` | 用reshape/permute、可微AllToAll及恢复布局，将一个维度的分片转换成另一个维度的分片。 |
| `_RecvKDAInitialState.forward` | eager参考路径中从前rank接收初态。 |
| `_RecvKDAInitialState.backward` | 将本rank的初态梯度发回前rank。 |
| `_SendKDAFinalState.forward` | eager路径向后rank发送末态，返回供输出连接计算图的token。 |
| `_SendKDAFinalState.backward` | 从后rank接收末态梯度，交回eager局部计算图。 |
| `_eager_kda_state_p2p` | Torch参考路径：生成摘要、连接状态收发及本地chunk KDA；用于参考和非融合路径。 |
| `_validate_layer_short_conv` | 验证两种整层wrapper共同依赖的depthwise、stride、dilation及激活约定。 |

| 类及方法 | 作用 |
|----------|------|
| `KimiDeltaAttentionUlyssesCP.__init__` | 记录一维U组、rank、后端、chunk和gate配置。 |
| `KimiDeltaAttentionUlyssesCP._validate_inputs` | 检查投影张量形状和head整除；这是置换前的契约，不由Triton补验。 |
| `KimiDeltaAttentionUlyssesCP._seq_to_head` | 将本地序列/完整head转换成组内完整序列/局部head。 |
| `KimiDeltaAttentionUlyssesCP._head_to_seq` | 将计算输出转换回本地序列/完整head。 |
| `KimiDeltaAttentionUlyssesCP.forward` | 对投影输入完成上述两次置换和中间本地KDA，并切分gate参数。 |
| `KimiDeltaAttentionP2PCP.__init__` | 记录状态组和邻居，根据state_cp_method建立AG执行器或保留P2P；类名保留已有接口。 |
| `KimiDeltaAttentionP2PCP.forward` | CP1走本地KDA，eager走Torch状态链，Triton走融合autograd并传入state_cp对象。 |
| `KimiDeltaAttentionLayerUlyssesCP.__init__` | 绑定原模块和纯U组，复用模型参数；forward改写保留原模型的参数路径。 |
| `KimiDeltaAttentionLayerUlyssesCP._validate_module` | 检查投影、head整除、gate和ShortConv是否满足纯U整层路径。 |
| `KimiDeltaAttentionLayerUlyssesCP._pack_projected_inputs` | 将Q/K/V/gate/beta按目标head组打包成一次AllToAll输入，再拆出五路张量。 |
| `KimiDeltaAttentionLayerUlyssesCP._local_short_conv` | 在已收齐序列的局部head上，切相应卷积权重执行ShortConv。 |
| `KimiDeltaAttentionLayerUlyssesCP.forward` | 投影、打包置换、局部head卷积/KDA、恢复序列、输出门控归一化和投影。 |
| `KimiDeltaAttentionLayerP2PCP.__init__` | 绑定原模块与原始CP组，保留用于halo的完整序列邻居。 |
| `KimiDeltaAttentionLayerP2PCP._validate_module` | 检查state路径支持的投影、gate、head维度和卷积属性。 |
| `KimiDeltaAttentionLayerP2PCP.forward` | 在原始序列分片上投影和交换halo，交给可覆盖的本地core，再运行输出层。 |
| `KimiDeltaAttentionLayerP2PCP._run_local_kda` | 默认把投影张量交给state/P2P core；混合类覆盖这一方法。 |
| `KimiDeltaAttentionHybridCP.__init__` | 校验U与AG组宽，拆分U/状态mesh，再建立状态CP core；不在每次forward中建组。 |
| `KimiDeltaAttentionHybridCP.forward` | 五路投影张量分别做sequence-to-head，切gate参数，执行状态core，再恢复输出布局。 |
| `KimiDeltaAttentionLayerHybridCP.__init__` | 复用整层state类的投影/halo，建立Hybrid core；halo仍使用原始CP邻居。 |
| `KimiDeltaAttentionLayerHybridCP._run_local_kda` | 将投影结果转交Hybrid core，复用父类的整层前后处理。 |

## 融合本地核心函数逐项说明

| 函数 | 作用 |
|------|------|
| `_validate_local_shapes` | 检查Q/K/V/gate/beta、A_log/dt_bias的维度关系并返回尺寸。 |
| `_validate_local_inputs` | 验证固定NPU后端、dtype、完整chunk、安全gate和局部state契约，避免进入不支持的kernel。 |
| `_validate_p2p_inputs` | 在状态通信前补验CP大小/rank和本地输入；不能依赖通信库发现模型语义错误。 |
| `fused_chunk_kda` | 无CP状态通信的融合KDA接口，加载既有FLA本地实现。 |
| `_KDAStateP2PFunction.forward` | 归一化Q/K，处理gate/beta，FLA prepare生成prepared张量；生成S/M并求输入状态；用FLA h/output计算token输出；只保存本次反向需要的张量和本地M。 |
| `_KDAStateP2PFunction.backward` | 重算prepared张量和局部状态，生成G并求末端梯度；执行FLA dhu、WY、intra、gate/beta及归一化反向，释放已无消费者的临时张量。 |
| `fused_chunk_kda_p2p` | 已有名字下的state-CP入口：参数校验、连续化、默认scale处理，调用自定义autograd；state_cp=None保持P2P。 |

摘要的通信由这一自定义autograd显式组织前后向，所以内部使用普通dist收发/AllGather是有意的。
Ulysses和halo处于普通模型autograd路径，使用可微AllToAll；不能将两处collective随意互换。

## 校验应放在哪里

| 校验内容 | 所在位置与原因 |
|----------|----------------|
| 模型结构、U与g整除、mesh根和rank顺序 | 初始化/模型适配层；进入kernel前就需要决定通信组与调用路径。 |
| 摘要shape/dtype/device | 状态通信入口，在首次irecv/AllGather之前检查；kernel发现错误时，其他rank可能已进入通信。 |
| merge的stride、rank及堆叠维度 | Python kernel wrapper；这些是专用融合扫描的寻址契约。 |
| 每个算子的局部计算条件 | 对应本地算子接口；与通信层责任不同。 |

这些元数据检查不读取张量值，不触发GPU/NPU到CPU的数据拷贝或同步。
不能把通信前的检查全部丢给Triton，也不需要在每个私有辅助函数重复校验。
`_gather_maps` 现在消费已验证的摘要，避免分组路径重复检查同一shape/dtype。
本地校验不能解决所有跨rank配置不一致；所有rank仍必须使用一致的配置、输入形状和调用顺序。

## CPU 分支的作用

`states.device.type == "cpu"` 使用Torch FP32顺序递推，用于真实Gloo通信的协议和mesh验证。
它只在输入原本就在CPU时执行，不把NPU张量搬回CPU，也不代表融合KDA整层支持CPU。
NPU输入进入Triton扫描；不支持的设备在通信入口报错。
测试另外构造FP64顺序参考，验证CPU/NPU结果，而非把CPU分支自身当作唯一真值。

## 审查发现与后续优化

本次修复了嵌套局部root绕过WORLD检查的缺口，并在首次通信前拒绝空batch/head及未支持设备。
配置和内部对象的命名与职责对齐；已有S/M/G、矩阵合成顺序和FLA数值路径保持不变。

| 观察 | 可以继续做的优化及边界 |
|------|----------------------|
| 分组 `_scan_maps` 每个摘要需要host下发矩阵乘和加法 | 研究支持非零initial_state和子区间的融合scan，复用owner完整扫描和rank局部扫描；需要保持现有浮点顺序并验证收益。 |
| Hybrid五路输入各一次AllToAll，旧纯U路径已有打包实现 | 在相同本地形状下比较打包复制与减少通信启动的净收益，不能直接假定拼包更快。 |
| 纯AG前反向都重新收集M | 这是只缓存本地M的显存/带宽取舍；保留完整M_all会重新引入跨前反向O(R)缓存，不作为默认优化。 |
| AG及分组AG每次分配打包/收集缓冲区 | 只有在明确多层、多次在途和checkpoint生命周期之后才能复用workspace，不能在共享执行器上放单份可变缓存。 |
| owner链需要组内广播及顺序状态apply | 可以减少启动和布局转换；它仍是串行组链，不能承诺对数依赖。 |
| 层接口检查attention_mask时有 `.all().item()` | 继承自已有整层路径；若dense训练不需要mask，应避免传冗余全一mask，进一步移到输入契约检查需另行设计。 |

没有发现当前AG必须额外计算完整组合A的需求。优先优化实际状态扫描的启动、布局和临时缓冲区，
比新增无人消费的组合矩阵更直接。
