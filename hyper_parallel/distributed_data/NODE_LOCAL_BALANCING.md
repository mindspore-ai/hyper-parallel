# Buffered node-local balancing

This opt-in path wraps the application's existing rank-local DataLoader. It
does not change the PR's `build_distributed_dataloader` or external-step path.
The source yields one step as `list[list[raw_sample]]`: one list per microbatch.
Only those samples are balanced; no future-step samples are added to the plan.

## Reference configuration

- Automatic (`ori`) double buffering; no delayed trigger or model hooks.
- Final H2D runs in the same background producer, after exchange and collation.
- `v1` shared-backbone FLOPs with capacity-constrained LPT. Keep the original
  layout if the candidate does not improve the selected objective. The current
  GLM managed-env setting is `makespan`; `balance` remains available for comparison.
- Node-local Gloo metadata and payload exchange. The payload stays on CPU;
  the H2D device is configured independently of the communication device.
- One `[dp-balance] iteration N/total` message from **global rank 0** (node 0's
  training worker 0), not from DataLoader subprocesses or other node leaders.
  The message reports node 0's ranks only, with before/after microbatches,
  send/recv samples and predicted cost. No global per-step logging gather.

```python
from hyper_parallel.distributed_data import (
    DefaultCostModel,
    DeviceStepPrefetcher,
    DistributedDatasetConfig,
    build_local_balancing_dataloader,
)

# Keep the application's existing sequence limits, microbatch count and budgets.
config = DistributedDatasetConfig(
    seq_len=seq_len,
    local_batch_size=microbatches_per_step,
    packing_budgets=packing_budgets,
    dp_dim_names=("dp_replicate", "dp_shard"),  # Match the actual root mesh.
    double_buffer=True,
    cpu_backend="gloo",
    payload_backend="gloo",
)
cost = DefaultCostModel(model_config)  # v1, using the effective backbone config.
loader = build_local_balancing_dataloader(
    native_loader,
    mesh,
    config,
    metadata_fn=metadata_fn,
    pack_fn=pack_fn,
    bin_stats_fn=summarize_bin,
    cost_model=cost,
    cost_model_id=cost.model_id,
    balancing_scope="node",
    balancing_objective="makespan",
    enable_balancing=True,
    communication_device=None,  # Gloo CPU payload; do not pass the NPU here.
    max_steps=train_steps,
    device_prefetch=DeviceStepPrefetcher(
        device,  # e.g. torch.device("npu", local_rank)
        move_fn=move_microbatch_to_device,
        pin_memory=True,
        pin_fn=pin_microbatch,
    ),
)

try:
    for cpu_microbatches in loader:
        for index, cpu_microbatch in enumerate(cpu_microbatches):
            # Retain CPU views for metering; consume each device view once.
            device_microbatch = loader.take_device_microbatch(index)
            train_microbatch(device_microbatch)
        optimizer_step()
finally:
    loader.close()
```

`metadata_fn`, `pack_fn`, `summarize_bin`, and move/pin callbacks are application
adapters, not framework-side model dependencies. `move_fn` receives
`(microbatch, device)`; `pin_fn` receives one microbatch. Preserve CPU-only
fields such as `cu_seqlens` in these callbacks. Configure Python logging at
INFO level to display the built-in message; the framework does not replace
the application's logging configuration.

For the existing GLM managed-env client, the matching options are:

```bash
export GLM_DATALOADER_TYPE=hyper_parallel
export GLM_ENABLE_DP_BALANCE=1
export GLM_ENABLE_DOUBLE_BUFFER=1
export GLM_HP_DOUBLE_BUFFER_MODE=ori
export GLM_HP_DOUBLE_BUFFER_H2D=1
export GLM_HP_COST_MODEL=v1
export GLM_HP_BALANCING_BACKEND=gloo
export GLM_HP_BALANCING_SCOPE=node
export GLM_HP_BALANCING_OBJECTIVE=makespan
```

The current lightweight GLM adapter can keep its field mappings and log
callback. No GLM model configuration, launcher, private environment file or
local test artifact is included in this framework change.

## Metadata and cost

`SampleMetadata.pack_tokens` is the physical packed length; `packing_costs`
contains per-sample additive footprints corresponding to `packing_budgets`.
The budgets constrain placement independently of predicted cost. The local
builder evaluates the cost model once per raw sample on each node's planner.

For v1, `features` contains `P` and `D` with `P + D == pack_tokens`.
For `attn_block_mode="blockmask"`, also provide `cond_image_token_lengths`,
including an empty sequence when there are no conditional images. Use the
effective model configuration, including any layer-count override.

The default training multiplier is 3 times forward FLOPs. The estimate covers
shared backbone projections, attention, dense/MoE MLPs and routers; it excludes
ViT/VAE, diffusion input/output stacks, communication and hardware efficiency.
It is a relative arithmetic estimate, **not a measured time prediction**.
Per-microbatch `cost` sums the planner's per-sample estimates; it does not
recompute attention from packed-total P/D. Each DP's cost sums its microbatches.

The optional `summarize_bin(samples)` callback returns a CPU-only dictionary.
The built-in formatter recognizes `vae_gen`, `vae_cond`, `vit`, `P`, `D`, and
`pixel_numel` for the existing multimodal log. The loader always supplies
`samples`, `seq_len`, and `cost`. Before/after costs use the same planner values,
and the after layout is the accepted plan, including no-movement steps.

## Execution and limits

The first step is prepared on demand. After delivering a ready step, the
producer immediately prepares the next source read, metadata exchange,
planning, optional sample A2A, collation and H2D. At most one future step is
buffered. This does not change the native DataLoader workers/prefetch factor.
`max_steps` prevents an extra speculative step after the final requested one.

H2D uses its own copy stream. The producer waits for its completion event
before publishing the step and retains the CPU source buffers. The consumer
waits on that event and records tensor lifetimes on its current stream. There
is no device-wide synchronization or added foreground H2D. Closing the loader
drains its producer before dropping staged views.

This local path requires a WORLD-covering named mesh with model-parallel
dimensions equal to one, matching step counts across ranks, and no dataloader
checkpoint/resume. Trusted per-step transport skips redundant validation and
error-handshake collectives; rank-local errors can leave peers waiting for the
process-group timeout. Startup topology checks and packing capacities remain.
The PR's existing transport defaults and native data pipeline stay unchanged.
