# Memory snapshot 配置

Trainer 可以在指定的 session-relative step 区间记录 CUDA 或 NPU allocator history，并将 snapshot 写到文件：

```yaml
memory:
  enable: true
  start_step: 1
  end_step: 3
  save_path: ./memory_snapshot
  dump_ranks: [0]
  stacks: all
  max_entries: null
  mem_info: false
```

配置字段：

- `enable`：是否记录 allocator history，默认 `false`。
- `start_step`：开始记录的控制点，默认 `1`。
- `end_step`：dump 并停止记录的控制点，默认 `2`。记录区间是 `[start_step, end_step)`；两个值相等时，在同一控制点启动、dump 并停止。
- `save_path`：snapshot 目录，默认 `./memory_snapshot`。
- `dump_ranks`：写文件的 global rank 列表，默认 `[0]`。所有 rank 都独立记录，但只有选中的 rank 写文件。
- `stacks`：`python` 或 `all`，默认 `all`。
- `max_entries`：allocator history 的最大条目数；`null` 表示不设置实际条目上限。
- `mem_info`：是否在每个训练 step 前记录 peak reserved/allocated memory 并重置 peak 统计，默认 `false`。

step `0` 位于分布式 setup 完成后、模型构建前；之后每次 `train_step` 在读取 batch 前推进一次。计数从本次 Trainer session 开始，不使用 checkpoint 恢复的 `global_step`。

正常训练在普通 `on_train_end` callbacks 之前结束 memory profiler；如果训练在 `end_step` 前正常结束，仍会 dump 已经启动的部分窗口。模型构建或训练抛出异常时只停止 allocator history，不 dump、也不执行跨 rank barrier，从而避免异常路径继续分配内存或等待其他 rank。

snapshot 文件名格式是 `snapshot_<timestamp>_<global_rank>.pickle`。不同 rank 的文件彼此独立，因此写文件前不做全局 barrier。
