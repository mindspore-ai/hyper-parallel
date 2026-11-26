# 显存快照格式与实测要点

基于 `torch_npu.npu.memory`，在 **torch_npu 2.9.0.post2 + CANN 9.0.0 /
Ascend 910** 上实采核对。字段名与行为均为实测，不是推测。

## API

| 调用 | 作用 |
|---|---|
| `_record_memory_history(enabled="all", context="all", stacks="all", max_entries=N)` | 开始记录分配历史。`stacks="all"` 才有 Python 帧，归属全靠它 |
| `_snapshot(device=None)` | 返回快照 dict（进程内） |
| `_dump_snapshot(filename=...)` | 落盘为 pickle |
| `_record_memory_history(enabled=None)` | 停止记录 |

实测签名：`_record_memory_history_impl(enabled='all', context='all',
stacks='all', max_entries=sys.maxsize, device=None)`。

### 两个落盘坑

1. **`_dump_snapshot` 写出的文件权限是 `--w-------`（0200，只写）**，连属主自己都
   读不回来。`mem_hook.dump()` 落盘后补 `chmod 0600`；手工 dump 的要自己
   `chmod 600 <file>`，否则解析会报 PermissionError。
2. `_dump_snapshot` 还会在当前工作目录生成 `PROF_00000N_*` 与 `FRAMEWORK/`
   目录，属正常副产物。
3. 记录期间 torch_npu 会打 `AclrtGetMemUsageInfo failed, ret:207000` 的
   **ERROR 日志**，但快照正常产出，属已知噪音，不要据此判定采集失败。

## 快照结构

```python
{"segments": [...], "device_traces": [...]}
```

### segment 字段（实测）

`active_size` · `address` · `allocated_size` · `blocks` · `device` · `frames`
· `is_expandable` · `requested_size` · `segment_pool_id` · `segment_type`
· `stream` · `total_size`

| 字段 | 含义 |
|---|---|
| `total_size` | 该段向设备保留的字节数 |
| `allocated_size` | 段内在用字节数 |
| `active_size` | 含 pending-free 的在用字节数；与 `allocated_size` 相等表示无待释放块 |
| `segment_type` | `large` / `small` |
| `frames` | 实测为空 `[]`，归属只能取 block 的 |

### block 字段（实测）

`address` · `frames` · `requested_size` · `size` · `state`

`state` 实测取值：`active_allocated` · `inactive`（`active_pending_free` 亦属
活跃，按 `active` 前缀判定）。

### 实测恒等式

**离线可查（只用快照本身）：**

```
逐段：Σ(所有 block.size) == segment.total_size
逐段：Σ(活跃 block.size) == segment.allocated_size
```

**需要进程内计数器（`mem_hook` 写进 `<snapshot>.stats.json` 旁车文件）：**

```
Σ(segment.total_size)     == memory_reserved()
Σ(segment.allocated_size) == memory_allocated()
```

实测四条全部精确成立（零偏差）。`breakdown.py` 把它们作为校验项输出，不通过
则退出码 2。

## frames 的真实形态（归属的关键）

记录栈是**最内层优先**，且最内几帧是**快照机制自身**：

```
[0] torch_npu::unwind::unwind()                  ← 取 frames[0] 等于每块都归到这里
[1] torch_npu::CapturedTraceback::gather(...)
[2..] ??                                          ← 无符号
...
[26] 第一个 .py 帧（共 62 帧时）                   ← 位置不固定，必须扫不能索引
...
[61] python3 / __libc_start_main
```

- **第一个 `.py` 帧的下标不固定**（实测 62 帧里在第 26 位），必须遍历查找。
- Python 帧内部同样最内层优先：`torch_npu/utils/_module.py:76 convert` →
  `torch/nn/modules/module.py:957 _apply` → `train.py:31 main`。
- **归属取最内层「不在 site-packages/dist-packages 内」的 `.py` 帧**，得到调用方
  自己的代码位置；没有则退到最内层框架 `.py` 帧；再没有则退到 native 符号。
- native 退路要先跳过**分配器管道**（`at::` / `c10::` 前缀），否则拿到的是
  `at::_ops::empty_strided::redispatch` 这类对归属无用的帧。跳过后得到
  `torch::autograd::AccumulateGrad::apply`，才是有意义的。

### 梯度没有 Python 帧

实测：梯度块的 `frames` 里**一个 `.py` 帧都没有**——autograd 引擎在 C++ 侧分配。
唯一可用信号是 native 帧里的 `torch::autograd::AccumulateGrad::apply`。
按 Python 帧做分类的实现会把梯度整类漏掉。

## device_traces

每设备一条列表，元素形如
`{"action": "segment_alloc"|"alloc"|..., "addr":, "size":, "stream":, "frames": [...]}`，
`frames` 同样带 Python 帧。可用于回放峰值时刻的构成（当前 `breakdown.py` 未用，
标 `[待对齐]`：回放口径）。

## 快照不等于峰值（最重要的一条）

实测同一次运行：

| 取样时刻 | 活跃量 |
|---|---|
| `optimizer.step()` 之后 | 100.01 MiB（**激活为 0**） |
| backward 中途（autograd hook 内） | 112.01 MiB（激活在场） |
| `max_memory_allocated()` | **130.01 MiB** |

- 步后快照里**没有任何激活**，拿它当峰值构成会严重低估。
- 即便 backward 中途取样，仍低于真实峰值 18 MiB。
- 所以：`mem_hook.dump(tag)` 支持**任意时刻**调用（含从 autograd hook 内），
  并把 `max_memory_allocated` 写入旁车；`breakdown.py` 把「快照是否为峰值」
  作为一项校验，差距超 2% 即报不通过并在报告里写明。
