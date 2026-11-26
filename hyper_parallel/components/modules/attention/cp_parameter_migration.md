# Attention CP 参数迁移记录

## 公共配置

注意力 family 的 CP 配置统一使用以下参数名。参数值按网络拓扑和算法能力选择；本表不规定所有模型使用同一组因子。

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `ulysses_cp_size` | `1` | Ulysses 域大小 |
| `ulysses_cp_strategy` | `alltoall` | Ulysses 通信策略 |
| `kv_cp_size` | `1` | KV 域大小 |
| `kv_cp_strategy` | `allgather` | KV 通信策略 |
| `kv_cp_load_balance` | `false` | KV 域负载均衡；可选 `zigzag` |
| `ring_cp_size` | `1` | Ring 域大小 |
| `ring_cp_strategy` | `p2p` | Ring 通信策略 |
| `ring_cp_load_balance` | `false` | Ring 域负载均衡；可选 `zigzag` |
| `cp_comm_overlap` | `false` | 通信计算重叠开关 |

当 CP 大小大于 1 时，三个域大小的乘积必须等于 `accelerator.cp_size`。大小为 1 表示该域关闭。
合法参数名不代表每个 attention family 都实现了对应算法；不支持的域或选项会在 family 校验时明确报错。

## 当前 family 支持范围

| Family | 支持的域 | 策略和其他限制 |
| --- | --- | --- |
| GQA、Gated GQA | Ulysses × KV；Ring 必须为 1 | Ulysses `alltoall`，KV `allgather`；不支持 zigzag；支持 `cp_comm_overlap`。TP×U 还须整除本地 Q/KV 头数。 |
| MLA、Gated MLA、Kimi MLA | Ulysses × KV；Ring 必须为 1 | Ulysses `alltoall`、`expanded_ulysses`、`latent_kv_head`；KV 仅 `allgather`；不支持 zigzag；支持 `cp_comm_overlap`。现有 `kv_exchange=all_gather/all_to_all_v` 是 MLA 的文档级 KV 交换选项，不等同于公共 `kv_cp_strategy`。 |
| DeepSeek Sparse Attention | 仅 KV；Ulysses、Ring 必须为 1 | KV 仅 `allgather`；不支持 zigzag；支持 `cp_comm_overlap`，用于异步提交 KV bank 收集。 |
| CSA2 | 仅 KV；Ulysses、Ring 必须为 1 | KV 仅 `allgather`；`kv_cp_load_balance=zigzag` 映射到现有 Indexer task 负载均衡；不支持通用 `cp_comm_overlap`。`main_kv_comm=all_gather/all_to_all_v` 是 CSA2 自己的 KV bank 传输选项。 |
| KDA | Ulysses × Ring；KV 必须为 1 | Ulysses 仅 `alltoall`；Ring 当前支持 `p2p`、`allgather`，用于循环状态通信，不是注意力 KV Ring；不支持 zigzag 或通用 `cp_comm_overlap`。Eager 后端不支持激活 Ring 状态并行。 |
| GDN | Ulysses 或 Ring 状态通信；KV 必须为 1 | Ulysses 仅 `alltoall`，Ring 仅 `p2p`；当前只支持纯 Ulysses 或纯状态 P2P，不支持 Ulysses/Ring 混合、不支持 zigzag 或通用 `cp_comm_overlap`。Ring 状态并行需要 Triton 后端。 |

CSA2 的 `kv_cp_load_balance=zigzag` 复用的是 Indexer 选择任务均衡，不重排主注意力的 token KV 分片。
KDA/GDN 的 Ring 域承载递归状态通信，与基于注意力 KV 的 Ring 算法含义不同。它们接收统一字段，
但不会因此自动获得 attention Ring、zigzag 或 overlap 实现。

## 配方迁移结果

已迁移仓库中的 YAML 配方使用公共参数名，并按迁移前行为写出明确域大小：

- GQA 的 CP2 配方显式写为 U1×KV2；CP1 写为 U1×KV1。
- GDN 的纯 Ulysses 配方显式写为 U2×Ring1；状态 P2P 配方写为 U1×Ring2。
- Gated GQA 的 CP2 配方显式写为 U1×KV2。
- DeepSeek V3 MLA 的纯 Ulysses 配方现显式使用 U4×KV1，保留 `expanded_ulysses` 策略。

旧参数名已经从 family 工厂、CP 执行器、Python 调用点和 YAML 配方中删除。调用方必须直接提供公共字段，
并在 CP 大于 1 时显式给出 Ulysses、KV、Ring 三个域的乘积。工厂不会再推导缺失域，也不会接受兼容别名。

KDA 还保留特定算法扩展 `state_cp_method` 和 `allgather_group_size`。它们不属于统一字段：旧的
`state_cp_method` 可以选择 `grouped_allgather_p2p`、`recursive_doubling` 等额外状态协议；公共
`ring_cp_strategy` 当前仅公开 `p2p`/`allgather`。需要移除这项扩展时，应先为这些协议确定标准化策略名和实现映射。

## 尚未对齐的参数语义

1. 公共接口将 Ring 描述为通用 attention 域，但 KDA/GDN 的现有 Ring 实际传递循环状态；当前以 family 校验限制语义，不尝试伪装成注意力 Ring。
2. 公共接口允许 `kv_cp_strategy=p2p`，现有 MLA/GQA/DSA/CSA2 只实现 AllGather。CSA2 的 `main_kv_comm=all_to_all_v` 与它不等价。
3. 公共接口为 KV/Ring 域定义 `zigzag`，目前 CSA2 仅将 KV zigzag 映射到 Indexer task；其他上述 family 都不支持该负载均衡。
4. `cp_comm_overlap` 是统一开关名，但各算法现有异步窗口不同。DSA、GQA、MLA 有对应实现；CSA2 使用既有 stage schedule，KDA/GDN 没有通用开关。
5. Ulysses 策略枚举包含 `latent_kv_head`，仅 MLA 系列实现；GQA/Gated GQA、DSA、CSA2、KDA、GDN 不接受此策略。

改配方时请根据实际拓扑设置因子，并查阅对应 family 的支持范围；不能仅凭参数被公共工厂接收，就推断底层通信算法可用。
