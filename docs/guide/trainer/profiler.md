# Training profiler

Trainer 的 `profiler` 节点控制 CPU、CUDA 或 Ascend NPU 性能采集。配置默认关闭：

```yaml
profiler:
  enabled: true
  start_step: 2
  stop_step: 3
  start_on_init: false
  memory: true
  rank_ids: [0]
  pipeline_stage_leaders: false
  output_path: ./output
  level: 1
  with_stack: false
  data_simplification: false
  mstx: false
```

`start_step` 和 `stop_step` 是本次训练 session 的相对步数，均从 1 开始，记录区间包含两端。
`start_on_init` 在 `on_train_begin` 开始采集，要求 `start_step: 1`；它不包含模型构建。
训练提前结束时会导出已采集的部分窗口；训练异常时释放 profiler，不覆盖原始异常。

`rank_ids` 为 `null` 或空列表时选择所有 rank。`pipeline_stage_leaders: true` 额外选择每个
pipeline stage 的首个 rank。输出位于 `<output_path>/profile/rank_<global_rank>/`。
Trace 内包含 TP、PP、DP、EP、world size 和 sequence parallel 元数据。
`level`（0、1、2）、`data_simplification` 和 `mstx` 是 Ascend 选项，`mstx: true`
额外记录 session-relative step ranges。CPU/CUDA 使用 `torch.profiler`，Ascend 使用
`torch_npu.profiler`。`memory` 只控制性能 trace 中的内存事件；allocator snapshot 使用独立的
[`memory` 节点](memory_profiler.md)。

## 从旧配置迁移

| 旧字段 | 新字段 |
| --- | --- |
| `profiling.enabled` | `profiler.enabled` |
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
  --profiler.enabled=true --profiler.start_step=2 --profiler.stop_step=2 \
  --profiler.rank_ids='[0]' --profiler.output_path=output/profiler-validation \
  --profiler.memory=true --profiler.mstx=true \
  --memory.enable=true --memory.start_step=1 --memory.end_step=3 \
  --memory.dump_ranks='[0]' --memory.stacks=python --memory.max_entries=50000 \
  --memory.save_path=output/profiler-validation/snapshots --memory.mem_info=true
```

验收时确认完成 3 个 optimizer steps、loss/gradient norm 有限，只有 rank 0 导出性能 trace 和
snapshot；trace 包含 CPU/NPU kernel、`ProfilerStep#2`、MSTX `step 2` 和分布式元数据。
Snapshot 应包含非空 allocator segments、device traces 和 allocation/free events。
此命令是裁剪模型的特性 smoke，不代表完整模型精度或性能验收。
