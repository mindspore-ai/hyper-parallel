# FSDP / HSDP — Detailed Guide

Companion to `fsdp-dev-expert.md`. Implementation structure, runtime
invariants, and debug tips. Load when debugging or reviewing FSDP/HSDP
changes. FSDP is torch-only and lives entirely under `core/fully_shard/`.

Hard stream/memory rules: `.agent/rules/distributed.md`.

## Implementation Structure

1. API entry and module transformation
- `fully_shard()` validates module input, chooses root modules with
  `_get_root_modules()`, dynamically extends each root with `HSDPModule`,
  creates or normalizes the mesh, and initializes one shared scheduler.
- When `fully_shard()` receives `list[module]`, the roots are treated as one
  FSDP unit and all roots share the same `hsdp_scheduler`.
- `HSDPModule.hsdp_init()` constructs a single `HSDPSchedulerV2`.

2. Mesh mode selection
- `mesh.ndim == 1` means FSDP mode via `FSDPMeshInfo`.
- `mesh.ndim == 2` means HSDP mode via `HSDPMeshInfo`, with shard dimension `1`
  and replicate dimension `0`.
- `replicate_params` are wrapped with `DDPMeshInfo` so they stay replicated and
  use DDP-style gradient reduction.

3. Scheduler and state construction
- `HSDPSchedulerV2` (single concrete class — no platform subclass) owns
  config, prefetch lists, scheduler context, and phase transitions, and
  registers the forward/backward hooks itself.
- Its `_new_cell_state()` lazily imports and constructs `TorchHSDPStateV2`.
- `HSDPState` (base) owns the shared shard/unshard/wait/prefetch lifecycle;
  `TorchHSDPStateV2` adds torch-specific param discovery, mixed precision,
  and the reduce-scatter / all-reduce staging.

4. Parameter discovery and shared-parameter tracking
- `TorchHSDPStateV2._init_hsdp_params()` deduplicates parameters across all
  managed modules and wraps each in `TorchHSDPParamV2`.
- `_get_param_module_infos()` builds `ParamModuleInfo` for every parameter,
  including shared modules and shared parameter names, so parameter
  replacement updates every shared reference, not just the first owner.

5. Forward path
- Forward pre-hook calls `_hsdp_forward_pre_hook()`, which does mixed-precision
  input casting, `lazy_init()`, `unshard()`, and optional forward prefetch.
- `_forward_pre_hook()` wraps forward inputs through
  `PostBackwardFunction.apply(...)` so a scheduler callback exists in backward.
- Forward hook calls `_hsdp_forward_hook()`, which shards when
  `reshard_after_forward=True` and may cast outputs to `mp_policy.output_dtype`.

6. Backward path
- Output hooks trigger `_hsdp_backward_pre_hook()` exactly when backward starts.
- Backward pre-hook moves the scheduler into `PRE_BACKWARD`, unshards again
  when needed, and launches backward prefetch.
- Backward hook calls `hsdp_state.post_backward()`.
- A root-level callback (`Variable._execution_engine.queue_callback`) plus
  `root_bp_state` and `_root_backward_hook()` finish remaining staged gradient
  reductions through `launch_tp_replicate_reduce_and_apply()`.

7. Gradient reduction and application
- `post_backward()` first calls `accumulate_unsharded_grad_if_needed()`.
- `replicate_params` use flattened-mesh DDP-style all-reduce over the full
  data-parallel mesh.
- Sharded params use reduce-scatter across the shard dimension, followed by
  all-reduce across the replicate dimension only in HSDP mode when required.
- Async outputs are staged in `pre_reduce_scatter_params` / `pre_all_reduce_groups`
  / `pending_all_reduce_groups` and applied by `launch_tp_replicate_reduce_and_apply()`.

8. Reshard and user-facing parameter control
- `HSDPState.shard()` swaps parameters back to sharded form and frees
  unsharded storage.
- `HSDPModule.unshard(async_op=True)` returns `_UnshardHandle`, which waits
  through `wait_for_unshard()`.

## Runtime Invariants

### Parameter lifecycle
- Parameter lifecycle depends on `reshard_after_forward`, set by
  `fully_shard(..., reshard_after_forward=...)` and later changeable through
  `HSDPModule.set_reshard_after_forward(...)`.
- `reshard_after_forward=True` (default): forward is
  `SHARDED -> UNSHARDED -> SHARDED`, backward is
  `SHARDED -> UNSHARDED -> SHARDED`.
- `reshard_after_forward=False`: forward is `SHARDED -> UNSHARDED`, backward is
  `UNSHARDED -> SHARDED`.
- `HSDPState.is_shard` and each parameter wrapper's `sharded_state` must stay
  aligned with actual storage ownership.
- `wait_for_unshard()` is where async all-gather becomes a usable unsharded
  parameter.

### Memory & Gradient Lifecycle

General rules (async handle waits, `resize_(0)`, `param.grad = None`, buffer
clearing) are in `.agent/rules/distributed.md` — **Stream Synchronization** and
**Memory Management** sections.

FSDP-specific lifecycle details:

- `init_all_gather_outputs()` allocates reusable communication buffers.
- `alloc_all_gather_outputs()` restores storage capacity before communication.
- `free_unsharded_param()` releases storage by resizing buffer storage to zero
  instead of dropping the Python object.
- A correct reshard path must leave no stale unsharded storage attached to
  active parameters.
- `unsharded_param.grad` is the source for reduce-scatter and replicate
  all-reduce unless `unsharded_accumulated_grad` is active.
- `replicate_params` do not use the shard reduction path; they use DDP-style
  all-reduce over the flattened data-parallel mesh.
- When `reduce_grads` is disabled, gradients may remain in unsharded
  accumulated form until later synchronization.

### Mixed precision and offload
- `MixedPrecisionPolicy` controls forward input casting, parameter dtype,
  reduction dtype, output dtype, and optional FP32 main-grad application.
- `CPUOffloadPolicy` changes where sharded params and reduced grads live, so
  non-blocking transfers may require explicit device synchronization after
  gradient application.
- Torch validates that CPU offload starts from CPU materialized params; meta
  parameters must be materialized before training.

## Torch-specific behavior
- `HSDPSchedulerV2` tracks root-backward state and defers final gradient
  application to `_root_backward_hook()`.
- `TorchHSDPStateV2` stages async reduce-scatter and all-reduce outputs in
  `pre_reduce_scatter_params` / `pre_all_reduce_groups` / `pending_all_reduce_groups`.
- `TorchHSDPParamV2` supports `apply_grad_on_fp32_main_grad`, explicit
  `clear_reduce_scatter_output()` / `clear_all_reduce_output()`, and
  torch-specific `to_empty()` / `DTensor` restoration edge cases.
- Unsharded execution relies on module parameters pointing at the existing
  `TorchHSDPParamV2._unsharded_param` object; hook logic may inspect or use it,
  but must not swap it out for a new `nn.Parameter`.

## Process groups and stream handles
- Process-group helpers (`create_group`, `split_group`, `get_world_size`,
  `get_group_local_rank`, `create_sub_groups`) live in
  `core/fully_shard/utils.py` and share the process-wide cache
  `EXISTING_COMM_GROUPS` from `platform/platform.py` — the only remaining
  platform dependency. They are local to fully_shard on purpose.
- The grad-reduce handle (`set_grad_reduce_handle`, `wait_grad_handle`,
  `grad_ready_stream`) is module-scope state in `utils.py`; `hsdp_sync_stream()`
  delegates to `wait_grad_handle()`.

## Debugging And Review Guide

### First places to inspect
- Entry and control toggles: `core/fully_shard/api.py`
- Hook ordering and phase transitions: `core/fully_shard/hsdp_scheduler.py`
- Backward staging and final reduction: `core/fully_shard/state.py`
- Parameter replacement and memory release: `core/fully_shard/param.py`
- Comm-fusion staging: `core/fully_shard/param_group.py`

### Common failure modes
- Unshard/reshard bugs: check `HSDPState.unshard()`, `wait_for_unshard()`,
  `shard()`, and `TorchHSDPParamV2.to_unsharded()` / `to_sharded()`.
- Shared parameter pointer desync: inspect `ParamModuleInfo`,
  `_get_param_module_infos()`, and `ParameterHookMigrator`.
- Hook-time parameter replacement: if a hook replaces
  `TorchHSDPParamV2._unsharded_param` or rebinds the module field to another
  `nn.Parameter`, expect broken graph ownership, wrong gradient collection, or
  reshard inconsistencies.
- Meta parameter failures: inspect `lazy_init()`, `_validate_no_meta_params()`,
  and `reset_sharded_param()`.
- Gradient leaks or stale storage: inspect `free_unsharded_param()`,
  `clear_reduce_scatter_output()`, `clear_all_reduce_output()`, and where
  unsharded grad references are set to `None`.
- Process-group cache collisions: `EXISTING_COMM_GROUPS` is keyed by the sorted
  rank tuple; a rank list that differs only by ordering must resolve to the
  same cached group.

### Review priorities
- Parameter replacement must preserve shared-weight consistency across every
  owning module.
- Hooks must not replace the active `_unsharded_param` object during graph
  construction.
- Gradient reduction order must stay logically correct: replicate-param
  all-reduce vs sharded-param reduce-scatter and optional HSDP all-reduce.
- Async communication outputs must be waited on before use and cleared after.
- CPU offload paths must not rely on unsynchronized non-blocking transfers.
- FSDP code must not reintroduce a platform abstraction; it should call torch
  directly (or `core/fully_shard/utils.py` helpers).
