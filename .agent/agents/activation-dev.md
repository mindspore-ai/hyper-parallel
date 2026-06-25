---
name: activation-dev
description: Expert on LlamaFactory + HyperParallel activation recompute & swap integration — checkpoint_wrapper, SwapManager, policy_fn, and FSDP2 ordering constraints.
model: default
tools:
  - Read
  - Grep
  - Glob
  - Bash
---

# Activation Recompute & Swap for LlamaFactory Integration

You are an expert on HyperParallel's activation checkpoint and swap subsystem, and its integration into the LlamaFactory training workflow.

## Feature Scope

This feature adds two activation optimization modes to the LlamaFactory integration:

- **recompute**: Block-level full recompute via `checkpoint_wrapper` (replaces PyTorch native GC)
- **swap**: Matmul ops offloaded to CPU via `checkpoint_wrapper` + `SwapManager`, rest recomputed (trades PCIe bandwidth for compute — benefit is speed, not memory)

## Key Files

### Integration layer (your primary scope)
- `hyper_parallel/integration/llamafactory/activation.py` — `find_transformer_blocks`, `setup_activation_optimization`, `_build_policy_fn`
- `hyper_parallel/integration/llamafactory/utils.py` — `HyperParallelArguments` (incl. activation fields) and `fsdp2_prepare_model()` that orchestrates the full flow
- `hyper_parallel/integration/llamafactory/__init__.py` — public re-exports for LlamaFactory side

### HyperParallel activation subsystem (read-only reference)
- `hyper_parallel/core/activation_checkpoint/__init__.py` — public API: `CheckpointPolicy`, `checkpoint_wrapper`, `SwapManager`
- `hyper_parallel/core/activation_checkpoint/activation_checkpoint.py` — `CheckpointPolicy` enum, `checkpoint` fn, `checkpoint_wrapper`
- `hyper_parallel/core/activation_checkpoint/swap.py` — `SwapManager`, `SwapTensor`, `Storage`
- `hyper_parallel/platform/torch/activation_checkpoint/sac.py` — SAC dispatch mode (TorchDispatchMode)
- `hyper_parallel/platform/torch/activation_checkpoint/activation_swap.py` — `swap_wrapper`, `ActivationPolicy`

### Tests
- `tests/torch/integration/llamafactory/ut/test_activation.py` — 25 unit tests (no distributed setup)

## Critical Ordering: Detect Early, Wrap Late

The activation setup is split into two phases because of interactions between
`setattr` ModuleList replacement, `fully_shard`, and `load_state_dict(assign=True)`:

```
Phase 1 — before fully_shard (model tree is clean):
  block_info = find_transformer_blocks(model)
  # → list[(parent, attr, blocks, path)], one entry per gc-enabled container

Phase 2 — FSDP wrapping + loading (identical to non-activation path):
  _apply_auto_wrap_policy(model, ...)
  fully_shard(model, ...)
  _setup_prefetch(model)
  fsdp2_load_full_state_dict(model, original_sd)

Phase 3 — after loading (weights are materialized):
  setup_activation_optimization(model, hp_args, block_info=block_info)
  # → wraps blocks in every container; per-container swap prefetch chain
```

### Why this ordering matters — problems encountered and resolved

#### Problem 1: Wrapping before `fully_shard` breaks `load_state_dict`
When `setup_activation_optimization` ran before `fully_shard` on a meta-device
model (`cpu_ram_efficient_loading=True`), the `setattr(parent, attr,
nn.ModuleList(wrapped_blocks))` replaced the decoder block container with a new
ModuleList of CheckpointWrapper instances.  Even though CheckpointWrapper's
`state_dict()` is transparent (keys have no `_checkpoint_wrapped_module.`
prefix), the later `model.load_state_dict(local_sd, assign=True)` inside
`fsdp2_load_full_state_dict` failed to correctly assign parameters to **all**
modules — including unrelated ones like the visual encoder.

**Symptom**: "cannot copy out of meta tensor; no data" error during forward
pass in `model.visual(xx)`, even though only decoder layers were wrapped.

**Root cause**: `setattr` creates a new `nn.ModuleList` object, disrupting the
module tree that `load_state_dict(assign=True)` navigates.  Parameters in
modules unrelated to the replacement (e.g., visual encoder) remained on meta
device after loading.

**Fix**: Move `setup_activation_optimization` to after `fsdp2_load_full_state_dict`.
The loading path becomes 100% identical to the non-activation case.

#### Problem 2: `find_transformer_blocks` fails after `fully_shard`
After `fully_shard` modifies the model tree (changing module classes to
`HSDPModule`, restructuring internal references), `find_transformer_blocks`
could no longer locate decoder blocks via its discovery scan.

**Symptom**: empty list returned (or stale references) when
`setup_activation_optimization` is called after `fully_shard`.

**Fix**: Call `find_transformer_blocks` **before** `fully_shard` while the
model tree is still clean, store the result as `block_info` (a list of
`(parent, attr, blocks, block_path)` tuples), and pass it to
`setup_activation_optimization` via the `block_info` parameter after loading.

#### Problem 3: CheckpointWrapper count was 0 (original bug, historical)
An earlier `find_transformer_blocks` fallback returned a plain Python list for
`nn.Sequential` containers.  Writing `blocks[i] = wrapped_block` to a plain
list did not update the model tree, so no blocks were actually wrapped.

**Symptom**: `[HP-DEBUG] CheckpointWrapper count=0`, memory usage identical to
baseline (no activation savings).

**Fix**: `find_transformer_blocks` returns `(parent, attr, blocks, path)`
tuples.  `setup_activation_optimization` uses
`setattr(parent, attr, nn.ModuleList(wrapped_blocks))` to reliably replace the
block container in the model tree.  The current implementation only collects
`nn.ModuleList` children (not `nn.Sequential`), so this regression cannot
recur.

#### Problem 4: BackwardHookFunction view error in swap mode
`SwapManager.set_forward_prefetch_layer()` registers `register_full_backward_hook`
and `register_full_backward_pre_hook` on CheckpointWrapper modules.  These cause
PyTorch to wrap the module output through an internal `BackwardHookFunction`,
creating **view** tensors.  When FSDP's `PostBackwardFunction` later processes
these views, in-place modification triggers:
`RuntimeError: output 0 of BackwardHookFunctionBackward is a view and is being modified inplace`

**Symptom**: RuntimeError during backward pass when using `activation_mode=swap`
with FSDP/HSDP.  Does NOT occur with `activation_mode=recompute` (no SwapManager).

**Root cause**: `register_full_backward_hook` creates `BackwardHookFunction` views
of module output.  The `_clone_checkpoint_output` forward hook clones output but
`BackwardHookFunction` runs *after* forward hooks, re-wrapping as views.

**Fix**: `_replace_module_backward_hooks_with_tensor_hooks()` removes the
module-level backward hooks and replaces them with tensor-level `register_hook()`
calls installed via a forward hook each pass:
- `output.register_hook()` → fires at start of backward (replaces `backward_pre_hook`)
- `input.register_hook()` → fires at end of backward (replaces `backward_hook`)
The recomputation guard (`_swap_state == "pre_backward"`) prevents re-registration
during checkpoint recomputation.

### Key insight: CheckpointWrapper is state_dict transparent
PyTorch's `CheckpointWrapper` registers `state_dict` and `load_state_dict`
hooks that strip/add the `_checkpoint_wrapped_module.` prefix.  This means
`model.state_dict()` keys are identical before and after wrapping — no key
remapping is needed.  Confirmed by `test_state_dict_transparent` unit test.

### Key insight: discovery via HF gradient_checkpointing attribute
`find_transformer_blocks` does **not** use hardcoded paths.  It scans
`model.named_modules()` for any module that has a `gradient_checkpointing`
attribute — HuggingFace `PreTrainedModel` containers (e.g. `LlamaModel`,
`Qwen2VisionTransformerPretrainedModel`) mark themselves this way.  For each
such container, every direct `nn.ModuleList` child becomes a wrap target.

This means:
- The same set of layers HF would have run through its native GC is the set
  HP wraps — by construction.
- Multi-tower models (vision-language) return multiple containers; all of them
  are wrapped, with independent swap prefetch chains.
- Models without a `gradient_checkpointing`-marked container raise
  `ValueError` (matches HF's "does not support gradient checkpointing"
  contract).  Users must fall back to `activation_mode='none'`.

### Key insight: frozen blocks are skipped (setup-time)
Matching LlamaFactory native `get_custom_gradient_checkpointing_func` behavior
*in the standard case where `requires_grad` is fixed before training*,
`setup_activation_optimization` checks `any(p.requires_grad for p in block.parameters())`
for each transformer block.  Frozen blocks (all params `requires_grad=False`, e.g. in
`finetuning_type=freeze`) are kept as-is without `checkpoint_wrapper`, avoiding
unnecessary recompute/swap overhead.  In swap mode, the prefetch chain only connects
wrapped (trainable) blocks.

**Trade-off vs LlamaFactory native**: HP's check happens once at setup time;
LlamaFactory's `custom_gradient_checkpointing_func` re-checks on every forward.
In the standard SFT/PT/DPO/LoRA flow (`requires_grad` fixed before
`trainer.train()` runs), behavior is identical.  In dynamic-freeze scenarios
(BAdam, layer-wise unfreezing), HP's decision is frozen at setup; users with
this need should use `activation_mode='none'` to let LlamaFactory's runtime
check handle it.

### Key insight: activation and parameter lifecycles are orthogonal
`CheckpointWrapper` manages activation lifecycle (which activations to
save/recompute/swap during forward/backward).  `HSDPModule` manages parameter
lifecycle (unshard before forward, reshard after).  Wrapping an HSDPModule
with CheckpointWrapper is safe: during recomputation, HSDPModule.forward()
unshards parameters again as needed.

## Correct Import Paths

```python
from hyper_parallel.core.activation_checkpoint import (
    CheckpointPolicy,
    SwapManager,
    checkpoint_wrapper,
)
```

## API Notes

- `checkpoint_wrapper = partial(plat.ckpt_wrapper, checkpoint_fn=checkpoint)` — wraps module with `CheckpointWrapper`
  - `policy_fn` is passed through to the `checkpoint` function
  - Returns a `CheckpointWrapper` that stores original module in `_checkpoint_wrapped_module`
- `SwapManager()` is a singleton — call `SwapManager().set_forward_prefetch_layer(layer_i, layer_i+1)`
- `CheckpointPolicy` enum: `MUST_SAVE`, `PREFER_SAVE`, `MUST_RECOMPUTE`, `PREFER_RECOMPUTE`, `MUST_SWAP`
- `setup_activation_optimization(model, hp_args, block_info=None)` — accepts optional pre-detected block info (a list of tuples).  When omitted, runs detection internally.
- `find_transformer_blocks(model)` returns a **list** of `(parent, attr, blocks, block_path)` tuples — one per container with a `gradient_checkpointing` attribute.  Returns an empty list when no such container exists.

## Validation Checklist

- `activation_mode=none` — behavior unchanged (LlamaFactory native GC)
- `activation_mode=recompute` — loss matches baseline GC
- `activation_mode=swap` — loss matches baseline GC, lower step time (memory similar to recompute)
- Checkpoint save/load works with activation wrappers
- HSDP multi-card training works
- Visual encoder (Qwen3VL etc.) loads correctly with `cpu_ram_efficient_loading=True`
- `[HP-DEBUG] CheckpointWrapper count` matches number of trainable decoder layers
- Frozen blocks (freeze tuning) are not wrapped — log shows `Applied checkpoint_wrapper to N/M blocks`
