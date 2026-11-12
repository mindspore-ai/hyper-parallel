# VeOmni adapter

The implementation follows three responsibilities:

- `adapter.py` owns activation, framework bindings, parameter loading and FSDP options.
- `optimizer.py` owns optimizer selection, shared-storage views and optimizer checkpoint state.
- `__init__.py` only exports the public activation entry points.

Reusable tensor-layout and model-checkpoint conversions remain in `integration/torch`.

Activate once before building the model:

```python
from hyper_parallel.integration.veomni import apply_hyper_parallel_patch

apply_hyper_parallel_patch()
# Construct and run the original VeOmni trainer here.
```

With VeOmni's adapter startup entry, enable it in the training configuration:

```yaml
model:
  accelerator:
    fsdp_config:
      fsdp_mode: fsdp2
      fsdp_backend: hyper_parallel
      fsdp_scope: module
      reshard_after_forward: true
      reshard_after_backward: true
      forward_prefetch: true
      offload: false
      offload_pin_memory: true
      low_precision_reduce_scatter_comm: false
      max_load_broadcast_size: 20.0
      mixed_precision:
        enable: true
        param_dtype: bfloat16
        reduce_dtype: float32
        output_dtype: null
        cast_forward_inputs: true
      hyper_parallel:
        comm_fusion: false
        comm_fusion_zero_copy: null
        wrap_policy: block
        forward_prefetch_depth: 2
        backward_prefetch_depth: 1
        reduce_comm_interval: 1
  optimizer:
    type: muon
    use_hyper_optimizer: true
```

Omni activation reads resolved per-module settings. Module-scoped Hyper FSDP
requires one shared FSDP backend and configuration across FSDP2 modules;
composed wrapping uses the top-level FSDP configuration. Muon remains selected
per module. Incompatible FSDP settings are rejected before binding.
Inference passes `for_inference=True` with the resolved configuration to activate
only FSDP bindings, without enabling Muon from inherited training settings.

Then run its normal task:

```bash
torchrun --standalone --nproc_per_node=2 \
  tasks/train_text.py config.yaml
```

FSDP and Muon are selected independently. `fsdp_backend: hyper_parallel`
selects Hyper FSDP/HSDP; its default `torch` retains native FSDP2.
`use_hyper_optimizer: true` selects Hyper Muon and requires `type: muon`;
its default `false` retains VeOmni Muon. There is no training-wide enable flag.
AdamW reuses VeOmni's parameter-name and weight-decay grouping rules. With
ExtraParallel, the adapter reads the live Hyper mesh and constructs native AdamW
on the original parameters, including shared EP LoRA. It does not create temporary
Torch parameter views or replace training parameters.

Hyper Muon accepts `muon_adjust_lr_fn: match_rms_adamw` or `original`.
With `original`, an unspecified `muon_lr` defaults to 25 times the AdamW LR,
matching VeOmni. Explicit Muon LR and decoupled weight decay are preserved.
Head splitting uses `muon_head_group_size` together with
`muon_head_split_modules`, for example `[q_proj, k_proj, v_proj]`. VeOmni
resolves head layouts; Hyper's existing reshape callback orthogonalizes each
row block and scales its update using the block dimensions. Restoring a
checkpoint with a different head-block layout or LR policy is rejected.
These options use Hyper's existing optimizer interfaces without modifying
its FSDP or Muon core algorithms.

Direct factory calls default to `gram_quack`, matching VeOmni; unsupported
devices fall back to Gram through VeOmni's existing dispatcher. Parameters
shaped `[E, 1, H]` are presented as E independent `[1, H]` matrices to NS.
Checkpoint restore requires matching NS coefficients, epsilon, iteration count,
variant, implementation and Gram reset indices. Mismatches are rejected before
loading that Muon optimizer, because NS callbacks retain construction-time
settings. During real VeOmni DCP loads, required option templates contain
unloaded markers rather than usable current defaults. Partial loading cannot
make an absent checkpoint option appear to have been saved.

Old checkpoints missing NS metadata are rejected by default. If the historical
settings are known, explicitly confirm the missing values before loading:

```python
optimizer.confirm_legacy_muon_checkpoint_options(
    ns_implementation="gram",
    gram_ns_reset_iterations=[2],
)
# Run the normal VeOmni DCP restore after this explicit confirmation.
```

Confirmation fills only missing checkpoint fields and must match the active
optimizer. It never overrides saved values or infers historical settings from
the current configuration. Supply any other missing NS options explicitly too.

The original public FSDP configuration is reused. Fusion values are forwarded
to native Hyper, with `null` retaining its zero-copy default. `block` uses
VeOmni's original wrap targets, initialization and loading. Prefetch depth
counts following blocks and `forward_prefetch: false` disables forward prefetch.
Backward reshard uses the native Hyper setter. A non-default reduction interval
requires `set_reduce_comm_interval`; unsupported intervals fail before model
construction. The Torch-only low-precision RS transport hook is not supported.

With Hyper FSDP and native Muon, the optimizer consumes shared-storage Torch
parameter views; the live training parameters stay Hyper DTensors. Native Muon
algorithms and state handling are reused. This combination currently excludes
ExtraParallel layouts. With Torch FSDP and Hyper Muon, sharding, loading and
clipping stay native; only the optimizer factory is adapted.

All bindings are confined to VeOmni module attributes. Loading readers,
broadcasting, model forward, runtime, trainer, schedulers and native DCP remain
owned by VeOmni. Hyper owns real training DTensors, sharding, clipping and Muon
execution. Checkpoint-only Torch DTensor views expose existing local storage to
native DCP; they never replace live training parameters.

Importing this package does not import VeOmni or activate bindings. Torch exports
and DTensor types are not patched. `remove_hyper_parallel_patch()` restores owned VeOmni bindings;
`VeOmniAdapter()` also supports a context manager. Activation and deactivation are
startup/shutdown operations, not safe to run concurrently with model execution.
External application-local references captured before activation are not rebound.

An adapter instance uses `patch()`, `unpatch()` and `is_patched` for its lifecycle.
Its `bound_targets` lists the module attributes replaced at application time.

## Reusable Torch-facing interfaces

Other frameworks can use the same interface boundary without importing VeOmni:

```python
from hyper_parallel.integration.torch import DTensor, fully_shard, clip_grad_norm_

fully_shard(model, mesh=mesh)
clip_grad_norm_(model.parameters(), 1.0)
```

Here `DTensor` is Hyper's original type. `fully_shard` translates Torch mesh and
policy descriptions, then calls native Hyper FSDP. Gradient clipping is the
original Hyper function. The checkpoint helpers expose shared-storage Torch views
only for transport. Native Torch type checks are unchanged; framework call sites
that explicitly require Torch DTensor still need opt-in bindings.

### Interface scope

`fully_shard(model, **kwargs)` accepts native Hyper arguments: `mesh`, boolean
`reshard_after_forward`, `shard_placement_fn`, `mp_policy`, `offload_policy`,
`ignored_params`, `replicate_params`, `comm_fusion`, `comm_fusion_zero_copy` and
`source_shard_infos`. Torch mesh, mixed-precision and ordinary/CPU offload
descriptors are translated; execution and remaining validation stay in Hyper.
Unknown keywords and integer reshard modes fail before mesh translation.
The compatibility entry takes one module; pre-sharded Torch parameters are unsupported.

`get_model_state_dict(model, *, options=None)` supports native Hyper full/sharded
export, `cpu_offload` and `ignore_frozen_params`. `set_model_state_dict(model,
model_state_dict, *, options=None)` restores sharded state in place and honors
`strict`. Full-state restore, rank-zero broadcast and non-default
`dsd_fqn_modifiers` are rejected before resharding. Checkpoint views support only
ordinary Shard/Replicate placements; special layouts fail rather than lose semantics.

VeOmni dependencies are maintained in two places: `adapter.py` owns
the weight dispatcher's `_dispatch_parameter`/`_find_submodule` (under
`models.checkpoint.weights`, or `models.module_utils` in older versions) and FSDP/clipping bindings;
`optimizer.py` owns `build_optimizer`, `_collect_muon_kwargs`,
`_make_param_groups_for_subset`, and native Muon grouping/NS calls.
Optimizer binding requires named, keyword-compatible `model`, `optimizer_type`
and `param_groups` parameters. Additional parameters are allowed; incompatible
signatures fail at activation without installing partial bindings.

The validated scope includes reduced Qwen3/Qwen3-MoE, Qwen3.5 and Flash-Next models,
FSDP/HSDP, BF16 compute with FP32 parameters/reduction, non-reentrant checkpointing,
microbatch accumulation, rank0-broadcast and EP-sharded loading, clipping,
AdamW/Hyper Muon and sharded native DCP on NPU. Reduced-model boundary checks also
cover active MTP, image/predecoded-video inputs, persistent PLE tables and shared
EP LoRA. Same-backend continuation checks compare at least ten resumed updates;
small reported-loss and PLE-moment differences remain documented in validation
records rather than being described as bitwise precision acceptance.
Full-model convergence/performance, multi-node training, CUDA/QuACK, real CPU
offload and asynchronous DCP remain outside this validation scope.

Pre-sharded Torch TP/EP parameters and full-state/broadcast DCP restore are
explicitly unsupported. Muon reuses native parameter selection instead of custom
groups; supported head splitting is described above. AdamW retains native
ExtraParallel group names and decay rules while using live Hyper parameters. Hyper's existing HSDP Muon
checkpoint limitation for an owner parameter that has never received gradients
is not fixed by this adapter; unused/frozen-owner checkpoint scenarios remain
outside the validated scope. No Hyper core algorithm is modified here.
