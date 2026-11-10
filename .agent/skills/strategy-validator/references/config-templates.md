# Runnable validation configuration

Start from the repository's [Qwen3 MoE training template](../../../../examples/training_demo/qwen3_moe/train_parallel_online.yaml).
Preserve its `model._target_`, `optimizer._target_`, dataset, dataloader and model asset paths;
adjust those paths to the target host before benchmarking. A YAML containing only a model name is not runnable.

The following is a **patch fragment**, not a complete training configuration, for eight ranks:

```yaml
accelerator:
  tp_size: 2
  cp_size: 1
  ep_size: 2
  pp_size: 1
fsdp_config:
  dp_shard_size: 4
  edp_shard_size: 4
training:
  train_iters: 50
  train_samples: null
```

Trainer derives `DP = world_size / (TP * CP * PP)`. EP shares the stage's ranks;
it is not an extra factor in the world-size product. Each stage must be divisible by EP.
Dense FSDP shards divide `DP * CP`; expert FSDP shards divide `(world_size / PP) / EP`.
Batch settings must also obey `micro_batch_num * micro_batch_size * DP == global_batch_size`.
Recheck this invariant for each neighborhood candidate that changes DP.

Use `accelerator` and `fsdp_config` for the parallel degrees. Top-level `parallel_strategy`
and `parallelism` are search/description formats, not Trainer configuration aliases.
The optimizer preserves the template and uses the official config adapter writer for the PSA result.
It limits benchmark runs through `training.train_iters` and clears `training.train_samples`.
The exported production config retains the template's training length.

The scheduler launches a **single node** through `python -m torch.distributed.run --module
examples.training_demo.train_text CONFIG.yaml --training.train_iters=50 --training.train_samples=null`.
A custom entrypoint must accept that positional YAML and these dotted overrides.
`--devices` counts local ranks. Multi-node launch needs a separate launcher.

A successful process exit is insufficient: the log must contain distinct step records with finite,
positive `tokens/s=`, `step_time=...s` and `memory/device_max_allocated_gb=` values.
The requested number of distinct steps is required. Warmup is excluded from throughput and step-time means.
