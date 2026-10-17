# Training profiler

Trainer 的 `profiler` 节点控制 CPU、CUDA 或 Ascend NPU 性能采集。配置默认关闭：

```yaml
profiler:
  enable: true
  start_step: 2
  stop_step: 3
  warmup_steps: 0
  memory: true
  rank_ids: [0]
  output_path: ./output
  level: 1
  with_stack: false
  data_simplification: false
  analysis:
    enable: true
    async_mode: false
```

`start_step` 和 `stop_step` 是本次训练 session 的相对步数，均从 1 开始，记录区间包含两端。
`warmup_steps` 默认 `0`，必须是非负整数且小于 `start_step`。它指定正式采集前的预热步数，
预热期间正常训练，但不导出这些步的 trace；正式采集区间仍为 `[start_step, stop_step]`。
例如 `start_step: 3, stop_step: 4, warmup_steps: 2` 会预热第 1、2 步，只导出第 3、4 步。
也可通过 CLI 设置 `--profiler.warmup_steps=2`，同时设置 `--profiler.start_step=3` 或更大值。
性能采集从训练 step 开始，不包含模型构建。
训练提前结束时会导出已采集的部分窗口；训练异常时释放 profiler，不覆盖原始异常。
在 Ascend 提前结束的 trace 中，可能出现一个没有模型算子的末尾 `ProfilerStep` 标记；
它是后端在最后一次 `step()` 后开启、在关闭时结束的区间，不代表额外执行了一次训练。

`rank_ids` 指定采集的 global rank，为 `null` 或空列表时选择所有 rank。
输出位于 `<output_path>/profile/rank_<global_rank>/`。
Trace 内包含 TP、PP、DP、EP、world size 和 sequence parallel 元数据。
`level`（0、1、2）和 `data_simplification` 是 Ascend 选项。
CPU/CUDA 使用 `torch.profiler`，Ascend 使用
`torch_npu.profiler`。`memory` 只控制性能 trace 中的内存事件；allocator snapshot 使用独立的
[`memory` 节点](memory_profiler.md)。

`profiler.analysis` 仅控制 Ascend NPU 的采集后解析，CPU/CUDA 忽略这组配置：

- `enable` 默认 `true`，映射到 `tensorboard_trace_handler` 的 `analyse_flag`。
  数据量较大时设为 `false`，保留原始采集数据，后续离线解析，避免训练期间自动解析的阻塞。
- `async_mode` 默认 `false`，映射到同名参数。开启后异步解析，仅在 `analysis.enable: true` 时生效。
  它不控制采集是否异步，解析仍会消耗 CPU、内存和 I/O。

例如关闭自动解析可使用 `--profiler.analysis.enable=false`；开启异步解析可使用
`--profiler.analysis.enable=true --profiler.analysis.async_mode=true`。
关闭解析不会消除采集和原始数据落盘的开销。

## 从旧配置迁移

| 旧字段 | 新字段 |
| --- | --- |
| `profiling.enabled` | `profiler.enable` |
| `profiling.start_step` | `profiler.start_step` |
| `profiling.end_step`（不包含） | `profiler.stop_step`（包含，旧值减 1） |
| `profiling.trace_dir` | `profiler.output_path`（输出增加 `profile/rank_<rank>`） |
| `profiling.profile_memory` | `profiler.memory` |
| `profiling.rank` | `profiler.rank_ids: [rank]` |
| `profiling.with_stack` | `profiler.with_stack` |

旧节点 `profiling`、`record_shapes` 和 `with_modules` 不再接受。

## DeepSeek v4.1 training demo

先按 [training demo](../../../examples/training_demo/README.md) 准备本地模型配置、tokenizer、
Engram assets 和 Online 数据，并激活支持 DeepSeek-V4 的 Torch/NPU 环境。
下面使用标准 TextTrainer，保留 demo 的 40 层裁剪模型、4K 序列、Muon 和 FP32 main parameters。
8 卡主机显式选择 EP8/FSDP8/global batch 8，并关闭 checkpoint 保存和自动恢复，以便每次从头采集。

```bash
python -m torch.distributed.run --standalone --nproc_per_node=8 \
  --module examples.training_demo.train_text \
  examples/training_demo/deepseek_v41/train_deepseek_v41_online.yaml \
  --model.config_path="$MODEL_PATH" \
  --model.engram_assets_path="$ENGRAM_ASSETS" \
  --dataset.model_assets.tokenizer.pretrained_model_name_or_path="$MODEL_PATH" \
  --dataset.data_path="$DATA_PATH" \
  --accelerator.ep_size=8 --fsdp_config.dp_shard_size=8 \
  --training.global_batch_size=8 --training.train_iters=3 \
  --checkpoint.save_ckpt=false --checkpoint.restore_from=null \
  --profiler.enable=true --profiler.start_step=2 --profiler.stop_step=2 \
  --profiler.rank_ids='[0]' --profiler.output_path=output/profiler-validation \
  --profiler.memory=true \
  --memory.enable=true --memory.start_step=1 --memory.end_step=3 \
  --memory.dump_ranks='[0]' --memory.stacks=python \
  --memory.save_path=output/profiler-validation/snapshots --memory.mem_info=true
```

验收时确认完成 3 个 optimizer steps、loss/gradient norm 有限，只有 rank 0 导出性能 trace 和
snapshot；trace 包含 CPU/NPU kernel、`ProfilerStep#2` 和分布式元数据。
Snapshot 应包含非空 allocator segments、device traces 和 allocation/free events。
此命令是裁剪模型的特性 smoke，不代表完整模型精度或性能验收。
