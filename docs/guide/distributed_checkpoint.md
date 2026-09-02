# DCP 分布式检查点使用指南

HyperParallel 提供 DCP（Distributed Checkpoint）能力：每个 rank 只保存自己持有的分片，配合一份描述"谁存了哪一块"的全局元数据；加载时按目标切分策略重新计算需要读哪些片段，因此**并行策略变化后无需离线转换**。此外支持异步落盘、副本张量读一次 + 广播、Plan 缓存，以及与 Hugging Face safetensors 格式的离线互转。

## 核心概念

| 概念 | 说明 |
|------|------|
| **Planner（规划器）** | 决定"写什么 / 读什么"：把 state_dict 翻译成带全局坐标的 `WriteItem` / `ReadItem` 列表 |
| **Storage（存储后端）** | 决定"怎么写 / 怎么读"：把规划落到具体文件格式（默认 safetensors） |
| **Metadata（元数据）** | `fqn → 全局 shape + chunk 列表`，是跨切分策略读取的唯一依据；`.metadata` 由 coordinator 最后写出 |
| **Chunk** | 一个张量分片在全局张量中的 `offsets` + `sizes`，即一个 N 维半开区间盒 |
| **重切分（reshard）** | "我要的区域" ∩ "磁盘上存过的区域"，交集即需要读取的片段 |

## 模块一览

DCP 模块位于 `hyper_parallel/core/distributed_checkpoint/`：

| 文件 | 说明 |
|------|------|
| `api.py` | 对外主接口：`save` / `async_save` / `load` |
| `planner.py` | Planner 抽象接口与 `SavePlan` / `LoadPlan` / `WriteItem` / `ReadItem` 数据结构 |
| `standard_planner.py` | 默认实现 `StandardSavePlanner` / `StandardLoadPlanner` |
| `storage.py` | 存储后端抽象接口 `StorageWriter` / `StorageReader` |
| `filesystem_storage.py` | 文件系统实现 `FileSystemWriter` / `FileSystemReader`（safetensors） |
| `metadata.py` | `Metadata` / `ChunkStorageMetadata` / `ChunkInfo` / `BroadcastInfo` 等元数据结构 |
| `async_persist.py` | 异步保存：staging（`DataCopier`）与子进程持久化 |
| `reshard.py` | 区间求交与 `ReshardHandler`（离线重切分工具） |
| `ragged_utils.py` | `RaggedShard`（非均匀切分）的几何适配 |
| `offline_transform.py` | Hugging Face safetensors ⇄ DCP 离线互转 |
| `layout.py` | Layout 的采集、落盘与跨 rank 汇总 |
| `util.py` | 去冗余、同分片组推导、广播、分阶段计时等公共能力 |
| `saver.py` / `loader.py` | 单文件 safetensors 存取（`save_checkpoint` / `load_checkpoint`），与分布式流程无关 |

---

## 快速上手

```python
from hyper_parallel.core.distributed_checkpoint import save, load

# 保存：每个 rank 写自己的分片，rank0 写 .metadata
save(state_dict, checkpoint_id="/ckpt/step_1000")

# 加载：原地写回 state_dict，返回 None
load(state_dict, checkpoint_id="/ckpt/step_1000")
```

> **注意**：`load` 是**原地修改**传入的 `state_dict`，不是返回一个新字典。传入的 state_dict 必须预先构造好目标形状的张量（DCP 只会往**已存在的 key** 里填数据），否则会静默读不到内容——优化器状态尤其需要先做一次"预热"让 `optimizer.state` 非空。

---

## 接口详解

### `save`

```python
save(
    state_dict: dict[str, Any],
    *,
    checkpoint_id: Optional[Union[Path, str]] = None,
    storage_writer: Optional[StorageWriter] = None,
    planner: Optional[SavePlanner] = None,
    no_dist: bool = False,
    use_collectives: bool = True,
) -> Metadata
```

| 参数 | 说明 |
|------|------|
| `checkpoint_id` | checkpoint 目录。未传 `storage_writer` 时必填，会据此构造 `FileSystemWriter` |
| `storage_writer` | 自定义存储后端；与 `checkpoint_id` 同时传入时以 `checkpoint_id` 初始化 |
| `planner` | 自定义规划器，默认 `StandardSavePlanner()` |
| `no_dist` | `True` 表示单进程保存（强制关闭 `use_collectives`） |
| `use_collectives` | `True`（默认）：跨 rank 交换 plan，写出全局 `.metadata`；`False`：各 rank 互不通信，各写各的 `{rank}.metadata` |

返回全局 `Metadata`；函数末尾会做一次 `barrier`，返回时全部 rank 均已落盘。

### `async_save`

```python
async_save(
    state_dict, *, checkpoint_id=None, storage_writer=None, planner=None,
    no_dist=False, use_collectives=True, use_gloo=False, callback=None,
) -> AsyncSaveResponse
```

**staging 是同步的，持久化是异步的**：函数内先在训练进程中把张量拷贝到 Host 内存，然后把规划、通信、写盘全部交给一个后台子进程。因此：

- `async_save` 一返回，原 `state_dict` 就可以被训练继续改写；
- 但**磁盘上还没有 `.metadata`**，要确认落盘必须等 `AsyncSaveResponse`：

```python
resp = async_save(state_dict, checkpoint_id="/ckpt/step_1000", callback=lambda: print("landed"))
# ... 训练继续 ...
metadata = resp.get_result(timeout=600)   # 等价于 resp.persist_completion.result(timeout=600)
```

子进程与 join 线程都**刻意不是 daemon**：训练脚本退出时会等在途的 checkpoint 落完，而不是丢掉它。

### `load`

```python
load(
    state_dict, *, checkpoint_id=None, storage_reader=None, planner=None,
    no_dist=False, use_collectives=True,
    broadcast_replicated_tensors=True, broadcast_groups=None,
    broadcast_batch_bytes=6 * 1024 * 1024,
) -> None
```

| 参数 | 说明 |
|------|------|
| `use_collectives` | **默认 `True`**（与 `save` 一致）：跨 rank 交换 plan，据此推导副本组。设 `False` 则各 rank 独立读自己需要的数据，不做跨 rank 协同 |
| `broadcast_replicated_tensors` | **默认 `True`**：加载内容完全相同的若干 rank 只由其中一个读盘，其余通过组内广播拿数据，读盘量按副本数成比例下降；读盘卡按已分配字节数均衡挑选，不固定在最小 rank |
| `broadcast_groups` | 预建的广播通信组 `{rank_tuple: group}`；不传则按需创建。**必须与 `broadcast_replicated_tensors=True` 同时使用**，否则抛 `ValueError` |
| `broadcast_batch_bytes` | 小于该阈值的分片**攒成一次广播**发送，大于等于的各发各的。默认 6 MiB，设 0 关闭。仅在开启广播时生效。**各 rank 必须传相同的值**——攒批位置是各卡各自算出来的，不一致会死锁 |

> `use_collectives=True`（默认）要求**默认通信组里每个 rank 都调用 `load`**——plan 交换跑在该组上，缺席的 rank 会让其余 rank 卡在一个永远凑不齐的 gather 上。没有数据要加载的 rank 也要参与，传空 state_dict 即可。只想让部分 rank 加载时请显式传 `use_collectives=False`。`save` 一直是同样的要求。
>
> 广播只在 `use_collectives=True` 时生效：副本组是从各 rank allgather 上来的 load plan 中「读请求完全相同」推导出来的，不做这次 allgather 就无从判断谁和谁读的是同一份数据。

读取元数据时若找不到 `.metadata`，会自动回退到 `{rank}.metadata`（即 `use_collectives=False` 保存出来的 checkpoint），并同步把 `use_collectives` 关掉。

---

## 落盘产物

```text
/ckpt/step_1000/
├── .metadata               # use_collectives=True：coordinator 写出的全局元数据
├── 0.metadata              # use_collectives=False：每 rank 各写一份 rank 局部元数据
├── _rank0_.safetensors     # 张量分片
├── _rank0_.bytes           # 非张量数据（标量、pickle 对象等）
├── _rank1_.safetensors
└── ...
```

`.metadata` 由 `finalize_checkpoint` 最后写出，可作为"这个目录写完整了"的判据——写到一半被打断的目录没有这个文件。

---

## 异步保存的三种协同模式

子进程无法直接复用训练侧的 HCCL/NCCL 通信域（`fork` 之后其 C++ 全局状态已失效），因此提供三档协同方式：

| 组合 | 行为 | 适用场景 |
|------|------|----------|
| `use_collectives=True, use_gloo=True` | 子进程重建一个独立的 **CPU gloo 通信域**交换 plan 与写结果 | 有可用端口，希望协同开销最低 |
| `use_collectives=True, use_gloo=False` | 通过**存储介质**交换：各 rank 写 `LOCAL_PLAN_{rank}.pkl` / `STORAGE_DATA_{rank}.pkl`，带写完标记，多线程轮询并行读 | 不便再开通信域时的默认选择 |
| `use_collectives=False` | 完全不跨 rank，各写各的 `{rank}.metadata` | 单机调试、或上层自行保证一致性 |

使用 `use_gloo=True` 需要环境变量 `MASTER_ADDR` / `MASTER_PORT`，否则抛 `AssertionError`；使用存储介质协同则必须提供 `checkpoint_id`（需要一个目录当"信箱"）。

```python
# gloo 协同
async_save(sd, checkpoint_id=path, use_collectives=True, use_gloo=True)
# 存储介质协同
async_save(sd, checkpoint_id=path, use_collectives=True, use_gloo=False)
# 不协同
async_save(sd, checkpoint_id=path, use_collectives=False)
```

---

## 加载时重切分（跨并行策略续训）

**不需要任何额外接口**：save 端忠实记录每个分片的全局坐标，load 端按目标 layout 计算自己需要的区域，与元数据中的 chunk 求交后只读重叠部分。

```python
# 用 TP=4 保存
save(state_dict_tp4, checkpoint_id="/ckpt/step_1000")

# 换成 TP=2 直接加载，state_dict 按新切分构造好即可
load(state_dict_tp2, checkpoint_id="/ckpt/step_1000")
```

已覆盖的典型倒换：`TP4 → TP2`、`TP4 → DP2×TP2`、`DP2×TP2 → TP2`，以及 `fully_shard` 分片之间的重切分。

`reshard.ReshardHandler` 是同一套区间求交逻辑的**离线工具类**，用于在训练流程之外手工搬运分片：

```python
from hyper_parallel.core.distributed_checkpoint import ReshardHandler

handler = ReshardHandler(param_name="w1", full_shape=(8, 8),
                         from_layout=src_layout, to_layout=dst_layout, to_rank_id=0)
offsets = handler.infer_all_tensor_offset()      # {源 rank: 需要从它那里取的局部区间}
tensor = handler.get_real_tensor(collected)      # 把收集到的切片拼成目标 rank 的分片
```

---

## 副本张量读一次 + 广播

在有复制维度的场景（如 DP 复制、TP 上的 `Replicate()` 参数），多张卡持有同一份数据。打开广播后，每份数据在组内只有一张卡真正读盘，其余 rank 通过一次组内 broadcast 拿到，**把磁盘读放大从 N× 降到 1×**：

```python
load(state_dict, checkpoint_id=path)   # 两者默认都是开的

# 多次加载时可预建通信组，省掉加载路径上的建组开销
groups = {ranks: platform.create_group(ranks) for ranks in my_group_ranks if rank in ranks}
load(state_dict, checkpoint_id=path, use_collectives=True,
     broadcast_replicated_tensors=True, broadcast_groups=groups)
```

**谁读哪一份**由 `StandardLoadPlanner.build_global_plan` 决定，各 rank 用同一份 allgather 结果跑同一个算法，因此算出的结果一致，不需要额外通信来对齐：按字节数从大到小依次分配，每份交给组内当前已分配字节数最少的 rank。分配结果写在 plan 的 `ReadItem.source` 上，`FileSystemReader` 照着执行。

**分组粒度是分片，不是整个张量。** 一个分片就是 `to_local()` 拿到的那块本地缓冲区，由 `dest_index`（`fqn` + `offset` + `index`）唯一标识；持有同一个 `dest_index` 的 rank 组成一个组，一次 broadcast 发整块缓冲区（`dist.broadcast` 要求连续内存，而 narrow 出来的子区域通常带 stride）。同一个张量的不同分片各广播各的，互不牵扯——0/2 卡持有分片 1、1/3 卡持有分片 2，就是 0→2 和 1→3 两次独立广播。`BYTE_IO`（存的是 pickle 出来的 Python 对象而非张量存储）永不参与。

**读取与广播是流水化的。** `execute_read` 按 `(fqn, offset, index)` 的全序遍历分片，轮到自己读的读完立刻发起异步广播，不等它完成就继续下一片，因此一次广播总是和下一片的读取重叠；最多同时有 8 个广播在飞。各卡跑的是同一份 `all_plans`、走的是同一个全序，所以入队顺序天然一致——同一个通信组上的集合通信按入队顺序在流上串行执行，这既是数据不错乱的保证，也是不死锁的前提。没有别的卡等着的分片放在最后读：早读只会挡住广播，放最后还能填满尾部广播仍在飞的时间。

**小分片会合批。** 一次 broadcast 的固定开销约 145 µs，与它搬多少数据几乎无关（64 KiB 和 1 MiB 都是这个数），要到几 MiB 才跑满 33 GiB/s。也就是说小于交叉点（实测约 4.6 MiB）的分片，时间全花在「发起广播」而不是「搬数据」上——而 checkpoint 里恰恰有大量小张量：各种 norm、bias、scalar。所以 `broadcast_batch_bytes` 以下的分片会先拷进一块连续 staging、合成一次广播发出，落地后再散回各自 buffer；以上的仍然各发各的。

只有**通信组、源卡、dtype 三者都相同**的分片才能同批——一次广播只面向一个组、一个源、一块连续 buffer。各卡遍历分片的顺序相同，所以攒批和发批的位置天然一致，这和单发时不需要额外通信对齐是同一个道理。

实测数据（4 卡、512 × 64 KiB）：单发 81.5 ms，合成一次 1.1 ms。散回用 `torch._foreach_copy_` 融合成一次 kernel，1.4 ms（逐片 copy 要 11.8 ms）。

> 代价是显存：一个批要留到它的广播落地为止，所以最多占 `broadcast_batch_bytes` × 在飞上限（8）。默认配置下约 48 MiB。另外 `torch.distributed._coalescing_manager` 在 torch_npu 上实测无效（81.0 vs 81.5 ms），所以只能走 staging 这条路。

读盘本身还会在一个后台线程上**提前读一两片**（仅 torch 后端）。读的那一半只碰 checkpoint 文件，不碰 planner 也不碰设备，所以能和主线程的 H2D 拷贝、广播下发并行；批次按交出去的顺序回来，发送顺序不受任何影响。收益来自把 `safe_open` 的开销挪出主线程，外加建 view 时顺带触发的内核预读（见下）——一张卡要在多于 `_TORCH_FILES_KEPT`（8）个 checkpoint 文件之间来回读时（常见于重切分加载），实测快 14~17%；文件局部性好、句柄缓存不失效时基本持平。

> **读序和广播序保持一致是有意的，别拆。** 试过让读线程按文件序走、主线程仍按全局序消费（中间用完成表代替 FIFO），想借此修掉跨文件跳的开销 —— 30B 实测反而**慢 38%**（15.3 s → 21.1 s，三个变体交错跑两轮，轮间偏差 0.2 s 内）。两个原因：一是 checkpoint 文件本就按 key 字典序首尾相接排布，和全局序 `(fqn, offset, index)` 完全吻合（实测 9218/9218、零空洞），单文件内的读**已经是顺序访问**，没有可修的；二是拆成两个序之后，读线程按文件序碰页、主线程按全局序 `copy_`，两个访问序在同一个文件上交错，打乱内核的顺序检测，预读失效。另外光是把 FIFO 换成完成表、顺序一个字不动，也要付 **7.6%** —— 18867 次字典插入加查找不是白来的。
>
> 推论：跨文件跳在当前架构下没有便宜解法。要让读盘按文件走，`copy_` 就得跟着按文件走，而 `copy_` 的顺序就是广播顺序，动它会把各个读者串行化（总时间从 `max(各读者)` 退化成 `sum(各读者)`）。

### 读盘的时间花在哪：mmap 与页缓存

torch 后端读 checkpoint 走的是 **mmap，不是 `read()`**——`safe_open` 内部是 `MmapOptions::map_copy_read_only`，整条读路径上没有一次读系统调用。数据进入进程靠的是访问映射区时的**缺页中断**，所以「什么时候读盘」等价于「什么时候第一次碰那块内存」。

`_fetch_torch_tensor_file` 里的 `tensor_file.get_slice(key)[tensor_slices]` **拿到的是直接落在 mmap 上的零拷贝 view**：它的 `data_ptr()` 落在 `/proc/self/maps` 里那块文件映射区内，同一个张量取两次地址相同。所以取数据这一步几乎不花时间，**大张量的字节是后面 `_apply_fetched` 里的 `copy_` 读进来的**（实测 155.6 MiB 的 embedding，事先 `posix_fadvise(DONTNEED)` 清掉页缓存）：

| 语句 | 耗时 | 磁盘读 |
|---|---|---|
| `get_slice(key)` | 0.03 ms | 0 |
| `[tensor_slices]` | 0.74 ms | 0 |
| `target_tensor.copy_(tensor)` | 127.60 ms | 147.7 MiB（179 次 major fault） |
| 同样的 `copy_` 再来一次 | 25.29 ms | 0 |

最后两行差 5 倍，差的就是**页缓存**：内核把读过的文件页留在空闲内存里（`free` 的 `buff/cache` 列，可随时回收），再访问同样的字节直接从内存拿，磁盘计数器纹丝不动。

> `[tensor_slices]` 那一行是 0，不代表建 view 不缺页——embedding 排在文件最前面，它的首页在 `safe_open` 解析 header 时就被预读带进了页缓存，所以那次是 minor fault，磁盘计数器不动。建 view 总会碰一页，只是这一页未必要走盘。

**但这张表只是大张量的情形。** 建 view 本身会碰一下映射区（一页），触发**内核预读**（本机 `read_ahead_kb=4096`，一次最多带进 4 MiB）。张量小于预读窗口时，建 view 就把它整个带进来了，`copy_` 一个扇区都不用读；远大于窗口时，建 view 只带进开头一段，剩下的由 `copy_` 边读边缺页。从文件中段取 500 个张量（共 381 MiB，MoE 专家权重每个约 0.76 MiB，避开头部被 header 预读覆盖的影响）：

| 路线 | `get_tensor(k)` / `get_slice(k)` | `[slice]` / `[spec]` | `copy_` |
|---|---|---|---|
| `get_tensor(k)[slice]` | 170.54 ms，**387.5 MiB** | 3.41 ms，0 | 0 |
| `get_slice(k)[spec]` | 0.27 ms，0 | 181.22 ms，**387.5 MiB** | ~0 |

两条路线读盘量完全一致，只是落在不同语句上：`get_tensor(k)` 自己就把 view 建好了（碰映射区在这一步），`get_slice(k)` 只取元数据、要等 `[spec]` 才建。而 `[slice]` 作用在已经建好的 torch 张量上，只改 sizes / strides / storage_offset，**永远不会缺页**。

**所以后台预读线程确实在预读，只是靠的是内核。** 它不拷贝任何字节，只是建 view 时碰一下映射区，剩下的是内核预读顺手做的。checkpoint 里张量普遍小于预读窗口时（MoE 尤其如此），这等于把整批数据提前拉进了页缓存，主线程的 `copy_` 只走内存：2486 个张量共 2.0 GiB 的场景，建 view 阶段就拉进 1903 MiB，`copy_` 只需再读 155 MiB。反过来，单个远大于窗口的张量（比如 embedding），`copy_` 仍要自己边读边缺页。

> 这份预读依赖两个条件：预读窗口够大、且张量在文件里排布紧密。要做成确定性的，读线程里得显式 `madvise(MADV_WILLNEED)`，或者真把字节拷出来（后者要多付一份主机内存）。

**`get_slice(k)[spec]` 与 `get_tensor(k)[spec]` 实测等价**（safetensors 0.8.0）：两者都返回 mmap 上的 view，地址相同，建 view 都是零点几毫秒、0 磁盘，后续 `copy_` 的耗时和读盘量也一致。差别只在 `get_slice` 可以先 `get_shape()` 而完全不碰数据——校验 key 和形状时用得上。

**但怎么切很要紧。** dim 0 的前缀切（行并行）只读要的那部分：半个 embedding 是 74.98 MiB，整个是 148.88 MiB。dim 1 的切（列并行）拿到的是 strided view，`copy_` 要跨过每一页，**整个张量都会被读上来**（148.88 MiB，和不切一样）。重切分加载时列切方向省不到 I/O。

这一点也直接决定了广播能省多少。**同一台机器上的 dp 副本读的是同一个文件的同一段字节**，关掉广播时第二个 rank 的读全部命中页缓存，磁盘并不会读第二遍：8 卡实测无论广播开关，`/proc/diskstats` 都是 **56.9 GiB**，完全相同。所以单机布局下广播省不到磁盘 I/O，端到端只有 1 秒量级的出入（读播流水化之前甚至是负收益）。真正让广播值钱的是页缓存兜不住的场景——**跨节点 dp、共享存储 / NFS，或者 checkpoint 大到装不进内存**。

> 想自己数字节数只能用 `/proc/diskstats`，别拿 `majflt` 乘 4 KiB 反推：一次阻塞式缺页会顺带把后面一整个预读窗口拉进页缓存，上面 1.866 GiB 只报了 951 次 major fault。另外统计设备时正则要锚定 `^nvme[0-9]+n[0-9]+$`，否则整盘和分区各算一遍，数字直接翻倍。还有 `posix_fadvise(DONTNEED)` 会跳过仍被映射在页表里的页——测冷读必须先丢掉 `safe_open` 的句柄，最稳妥是一个配置起一个进程。

---

## 去冗余与 Plan 缓存

`StandardSavePlanner` 的两个默认优化，均只在 `use_collectives=True` 时生效：

```python
StandardSavePlanner(
    enable_plan_caching=True,    # 缓存最终 plan 与 metadata
    remove_redundancy=True,      # 跨 rank 去重
    save_to_minimum_rank=False,  # 去重后归属策略
)
```

- **去冗余**：同一分片被多个 rank 持有时只写一份。归属默认按**已规划字节数最小**的 plan 分配（负载均衡，避免副本全堆到 rank0）；置 `save_to_minimum_rank=True` 则统一归给最小 rank。
- **Plan 缓存**：以 state_dict 的 key 集合为缓存键，缓存在类级别。模型结构不变时，第二次及以后的 `save` 会跳过 `build_local_plan` / `all_gather` / `build_global_plan` 整个规划阶段——这一步是唯一随 world_size 线性增长的通信。异步保存路径下，子进程算出的缓存会随结果回传给父进程复用。

**两种自动降级**（无需干预，但需知情）：

- `use_collectives=False` 时，去冗余与 Plan 缓存都会自动关闭（没有全局视图）；
- state_dict 中含 `ragged_shard` 的 DTensor 时，Plan 缓存自动关闭（几何结构可能逐步变化）。

---

## 与 FSDP / HSDP 配合

DCP 与 `fully_shard` 天然配合：分片后的参数已经是 DTensor，带着完整的 layout 信息。

```python
from hyper_parallel import fully_shard, init_device_mesh
from hyper_parallel.core.distributed_checkpoint import save

mesh = init_device_mesh("npu", (dp_size,), mesh_dim_names=("dp",))
model = fully_shard(model, mesh=mesh)

save(model.state_dict(), checkpoint_id="/ckpt/step_1000")
```

---

## 离线格式转换（Hugging Face ⇄ DCP）

`offline_transform` 提供单进程的离线互转，用于"HF 权重初始化训练"和"训练产物导出成 HF 格式"两个方向：

```python
from hyper_parallel.core.distributed_checkpoint.offline_transform import (
    convert_full_checkpoint_to_dcp,        # 完整权重文件/HF 目录 → DCP
    full_state_dict_to_dcp_format,         # 内存中的完整 state_dict → DCP
    dcp_to_full_state_dict,                # DCP 分片 → 合并后的完整 state_dict
    parse_checkpoint_from_huggingface,     # 读 HF safetensors（单文件或分片 + index）
    save_state_dict_as_huggingface_format, # 写成 HF 风格 safetensors（可分片）
)

# HF → DCP
convert_full_checkpoint_to_dcp("/hf/Qwen3-30B", "/ckpt/dcp_init", src_platform="huggingface")

# DCP → HF
full_sd = dcp_to_full_state_dict("/ckpt/step_1000")
save_state_dict_as_huggingface_format("/export/hf", full_sd, max_shard_size="5GB")
```

`src_platform` 可取 `"huggingface"`（HF 目录）、`"torch"` / `"mindspore"`（完整 checkpoint 文件），后两者要求当前运行时与之匹配。

---

## Layout 工具

用于采集与持久化参数的切分信息，供离线重切分等场景使用：

```python
from hyper_parallel.core.distributed_checkpoint import (
    get_current_layout, get_global_layout, save_layout, load_layout, combine_layout,
)

save_layout(get_current_layout(model), f"/ckpt/rank{rank}.layout")  # 每 rank 各存一份
all_layout = combine_layout("/ckpt")                                 # 离线合并目录下所有 .layout
all_layout = get_global_layout(model)                                # 或在线 all_gather 汇总
```

---

## 自定义扩展

- **自定义存储后端**：继承 `StorageWriter` / `StorageReader`，实现 `execute_write` / `execute_read` / `load_metadata` 等方法，通过 `storage_writer=` / `storage_reader=` 传入。
- **自定义规划器**：继承 `SavePlanner` / `LoadPlanner`，或从 `StandardSavePlanner` / `StandardLoadPlanner` 派生后覆写。
- **自定义对象的异步 staging**：`async_save` 通过 `DataCopier` 按类型分派拷贝方法，未注册的类型会走通用深拷贝并打 warning。自定义类型请显式注册：

```python
from hyper_parallel.core.distributed_checkpoint.async_persist import DataCopier

@DataCopier.register(MyState)
def _copy_my_state(obj):
    return MyState(...)   # 返回 Host 内存中的副本
```

> 注意 `DTensor` 是 `platform.Tensor` 的子类，分派按"精确类型 → 最派生的注册基类"顺序解析。自定义类型如果也存在继承关系，请确认命中的是预期的 handler。

---

## 调试与性能

**打开分阶段耗时日志**（定位存盘瓶颈的第一手段）：

```bash
export HP_LOG_CONFIG=DCP:INFO
# 输出形如：
# [INFO] [HP-DCP]: [rank=0] >>> func build_local_plan cost 0.0003 seconds
# [INFO] [HP-DCP]: [rank=0] >>> func execute_write cost 0.0061 seconds
```

覆盖 `save` / `_save_impl` / `build_local_plan` / `all_gather_object` / `build_global_plan` / `execute_write` / `execute_read` / `build_staged_state_dict` 等关键阶段，每个 rank 单独计时。命中 Plan 缓存时会打印 `Hit final plan and metadata cache.`。

**性能建议**：

1. 用 `async_save` 让落盘与训练计算重叠，训练进程只付 staging 的开销；
2. 保持 `enable_plan_caching=True`，让规划阶段的通信只在第一次存盘时付出；
3. 有复制维度时保持 `broadcast_replicated_tensors` 开启（默认），用通信换 IO；
4. 变更并行策略优先走 load 时重切分；确需离线合并再切分时才用 `offline_transform`。

---

## 注意事项

1. **`load` 只填已存在的 key**：目标 state_dict 必须先构造出正确形状的张量；优化器状态需要先预热（让 `optimizer.state` 非空）再加载。
2. **相同 FQN 会被跨 rank 去重**：权重、优化器状态天然一致，没问题；但 **RNG state 这类每卡不同的数据**不能以相同 key 内嵌保存，否则只会留下一份。这类状态请每 rank 单独存，或给 key 加 rank 后缀。
3. **`async_save` 返回不代表落盘完成**：staging 已完成（原 state_dict 可继续改写），但需等 `get_result()` 才确认磁盘写完。
4. **`save` 与 `load` 的 `use_collectives` 默认值相反**（`True` / `False`），跨函数复制参数时容易搞错。
5. **`no_dist=True` 会强制关闭 `use_collectives`**。

---

## 流水并行（PP）下的广播加载

PP 打破了「各卡 state_dict 键相同」这个前提：一个 stage 只持有分给它的层，`embed` 只在首
stage、`head` 只在末 stage，两个 stage 几乎没有共同的键。这套流程能扛住，靠的是三点：

- `build_local_plan` 是**纯本地**的，`all_gather_object` 本来就允许各卡内容不同；
- `build_global_plan` 按 `dest_index`（`fqn` + `offset` + `index`）聚合副本组，所以 stage0 的
  参数天然只组成 stage0 内部的组，stage1 的只组成 stage1 内部的组，**两组不相交**；
- 顺序不变量只需要**在组内**成立，而一个组的成员必然在同一个 stage、遍历同一个子序列，
  所以**跨 stage 不需要对齐**，一个 stage 可以远远跑在另一个前面。

### 通信组：建出来、用一次、销毁

一个 stage 内部的副本组（stage 的 4 张卡、或某个 tp 列的 2 张卡）通常不在任何已有通信域里，
得现建。`ensure_broadcast_groups` 用的是 **`platform.new_group(ranks)`**——它直接调
`dist.new_group`，**建的就是给它的那几个 rank**，建完就还给调用方。

这里**不用 `platform.create_group`**，它做的事超出所需：把 rank 列表当**模板**展开成对全世界
的一个划分，并把展开出的每个组都塞进进程级缓存。两个后果对 PP 都不合适——缓存会让这些只用
一次的通信域活到任务结束（一个 communicator 一直占着显存），而模板展开会**拒绝** PP 会产生的
rank 列表，比如绑定在两个不相邻 stage 上的参数（`[0,1,6,7]` → `ValueError: Template must have
consistent intra-group step`）。走 `new_group` 这两个问题都不存在。

组在 `load()` 里用 `broadcast_groups_for_load(...)` 圈住：读完就按 rank 元组顺序销毁（这个顺序
各卡算出来一致，而销毁只由组内成员发起，它们是一起走到读取末尾的）。调用方通过
`broadcast_groups=` 预建传进来的组**不属于**这次加载，原样留着不动。

**建组是全局集合调用。** `ensure_broadcast_groups` 的进入条件是
`broadcast_replicated_tensors and use_collectives and world_size > 1`——三个全卡一致的量，
不是「我有没有要广播的分片」。一个全部参数都被切满、一个副本分片都没有的 stage，仍然必须
进来陪着建组，否则另一个 stage 会一直等它。这些 rank 从 `new_group` 拿到的是后端的非成员标记
（torch 上是 `-100`），既不会被存进组字典，也不参与销毁。

---

## 相关 ST

```bash
# 无 NPU 也可验证（CPU + gloo 后端）
export HYPER_PARALLEL_PLATFORM=torch HYPER_PARALLEL_TEST_DEVICE_TYPE=cpu
python -m torch.distributed.run --nproc-per-node=4 -m pytest -s \
    tests/torch/distributed_checkpoint/dcp_async_save.py::test_dcp_async_save_twice_reuses_the_plan_cache
```

| 用例文件 | 覆盖内容 |
|------|------|
| `tests/torch/distributed_checkpoint/dcp_save_and_load.py` | DTensor + 普通 Tensor + 标量混合存取、跨切分加载 |
| `tests/torch/distributed_checkpoint/dcp_async_save.py` | 三种异步协同模式、Plan 缓存复用 |
| `tests/torch/distributed_checkpoint/dcp_broadcast_load.py` | 副本张量广播加载（预建 / 懒建通信组） |
| `tests/torch/distributed_checkpoint/dcp_pipeline_load.py` | 读播流水：超出在飞上限、通信组交错、非张量状态、重复加载 |
| `tests/torch/distributed_checkpoint/dcp_pp_stage_load.py` | PP 下的广播加载（8 卡 `pp=2 × dp=2 × tp=2`）：各 stage 参数不同、某个 stage 无副本分片、两 stage 在分片序中交错、跨 stage 绑定参数 |
| `tests/torch/distributed_checkpoint/dcp_resharding_api.py` | 多种 mesh 组合下的重切分读取 |
| `tests/torch/distributed_checkpoint/_test_dcp_tp_dp.py` | `fully_shard` + 优化器状态 + flatten_state_dict |
| `tests/torch/distributed_checkpoint/dcp_plan_cache_minimal_api.py` | Plan 缓存命中与 model / optimizer 缓存隔离 |

用例文件本身不带 `test_` 前缀（或以 `_` 开头），由同名的 `test_*.py` 包装器通过 `parallel_run` 拉起多进程执行；直接用 `torch.distributed.run` 跑上表中的文件即可。
