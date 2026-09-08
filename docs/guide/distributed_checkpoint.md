# DCP 分布式检查点使用指南

HyperParallel 提供 DCP（Distributed Checkpoint）能力：每个 rank 只保存自己持有的分片，配合一份描述"谁存了哪一块"的全局元数据；加载时按目标切分策略重新计算需要读哪些片段，因此**并行策略变化后无需离线转换**。此外支持异步落盘、最小 rank 读 + 广播、Plan 缓存，以及与 Hugging Face safetensors 格式的离线互转。

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
    no_dist=False, use_collectives=False,
    broadcast_from_minimum_rank=False, broadcast_groups=None,
) -> None
```

| 参数 | 说明 |
|------|------|
| `use_collectives` | **默认 `False`**（与 `save` 相反）：各 rank 独立读自己需要的数据，不做跨 rank 协同 |
| `broadcast_from_minimum_rank` | `True` 时同一分片组内只由最小 rank 读盘，其余 rank 通过组内广播拿数据 |
| `broadcast_groups` | 预建的广播通信组 `{rank_tuple: group}`；不传则按需创建。**必须与 `broadcast_from_minimum_rank=True` 同时使用**，否则抛 `ValueError` |

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

## 最小 rank 读 + 广播

在有复制维度的场景（如 DP 复制、TP 上的 `Replicate()` 参数），多张卡持有同一份数据。打开广播后，同分片组内只有 `min(rank)` 真正读盘，其余 rank 通过一次组内 broadcast 拿到数据，**把磁盘读放大从 N× 降到 1×**：

```python
load(state_dict, checkpoint_id=path,
     use_collectives=True,
     broadcast_from_minimum_rank=True)

# 多次加载时可预建通信组，省掉加载路径上的建组开销
groups = {ranks: platform.create_group(ranks) for ranks in my_group_ranks if rank in ranks}
load(state_dict, checkpoint_id=path, use_collectives=True,
     broadcast_from_minimum_rank=True, broadcast_groups=groups)
```

> **不要只在 planner 上打开这个开关**。`StandardLoadPlanner(broadcast_from_minimum_rank=True)` 单独使用会让非最小 rank 跳过读盘，而广播根本不会发生（广播由 `FileSystemReader.execute_read` 触发），这些 rank 最终拿到的是未被写入的张量。开关请加在 `load()` 上，它会同时配置 planner 和 storage reader。

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
3. 有复制维度时打开 `broadcast_from_minimum_rank`，用通信换 IO；
4. 变更并行策略优先走 load 时重切分；确需离线合并再切分时才用 `offline_transform`。

---

## 注意事项

1. **`load` 只填已存在的 key**：目标 state_dict 必须先构造出正确形状的张量；优化器状态需要先预热（让 `optimizer.state` 非空）再加载。
2. **相同 FQN 会被跨 rank 去重**：权重、优化器状态天然一致，没问题；但 **RNG state 这类每卡不同的数据**不能以相同 key 内嵌保存，否则只会留下一份。这类状态请每 rank 单独存，或给 key 加 rank 后缀。
3. **`async_save` 返回不代表落盘完成**：staging 已完成（原 state_dict 可继续改写），但需等 `get_result()` 才确认磁盘写完。
4. **`save` 与 `load` 的 `use_collectives` 默认值相反**（`True` / `False`），跨函数复制参数时容易搞错。
5. **`no_dist=True` 会强制关闭 `use_collectives`**。

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
| `tests/torch/distributed_checkpoint/dcp_broadcast_load.py` | 最小 rank 广播加载（预建 / 懒建通信组） |
| `tests/torch/distributed_checkpoint/dcp_resharding_api.py` | 多种 mesh 组合下的重切分读取 |
| `tests/torch/distributed_checkpoint/_test_dcp_tp_dp.py` | `fully_shard` + 优化器状态 + flatten_state_dict |
| `tests/torch/distributed_checkpoint/dcp_plan_cache_minimal_api.py` | Plan 缓存命中与 model / optimizer 缓存隔离 |

用例文件本身不带 `test_` 前缀（或以 `_` 开头），由同名的 `test_*.py` 包装器通过 `parallel_run` 拉起多进程执行；直接用 `torch.distributed.run` 跑上表中的文件即可。
