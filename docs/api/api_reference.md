# HyperParallel API Reference

**English** | [简体中文](api_reference.zh.md)

This document is organized by feature, with an overview and detailed interface descriptions for each module.

---

## HSDP / FSDP Data Parallelism

HyperParallel provides FSDP/HSDP data parallelism, distributing parameter, gradient, and optimizer state shards to significantly reduce per-device memory usage.

### `fully_shard`

Apply FSDP/HSDP parameter sharding to a model.

```python
fully_shard(
    module: Module,
    mesh: DeviceMesh,
    *,
    reshard_after_forward: bool = True,
    mixed_precision: Optional[MixedPrecisionPolicy] = None,
    offload_policy: Optional[OffloadPolicy] = None,
    comm_fusion: bool = True,
    comm_fusion_zero_copy: Optional[bool] = None,
) -> Module
```

**Parameters:**

| Parameter | Type | Default | Description |
|------|------|--------|------|
| `module` | `Module` | — | Module to shard |
| `mesh` | `DeviceMesh` | — | DeviceMesh for the DP dimension |
| `reshard_after_forward` | `bool` | `True` | Reshard after forward to save memory |
| `mixed_precision` | `MixedPrecisionPolicy` | `None` | Mixed-precision policy |
| `offload_policy` | `OffloadPolicy` | `None` | Offload policy |
| `comm_fusion` | `bool` | `True` | Communication fusion |
| `comm_fusion_zero_copy` | `bool` | `None` | Zero-copy communication fusion (defaults to True) |

**Returns:** The sharded module, modified in place; the same object is returned.

---

### `HSDPModule`

HSDP module wrapper.

```python
class HSDPModule
```

HSDPModule encapsulates core HSDP parameter sharding logic, including parameter unshard/reshard and gradient synchronization.

---

### `hsdp_sync_stream`

HSDP gradient synchronization stream management.

```python
hsdp_sync_stream()
```

Return the gradient synchronization stream used by HSDP, ensuring gradient all-reduce is synchronized correctly with the compute stream.

---

## DTensor Distributed Tensors

DTensor is HyperParallel's core distributed tensor abstraction, encapsulating a local shard, DeviceMesh, and Placements.

### `DTensor`

Distributed tensor class.

```python
class DTensor
```

**Core methods:**

| Method | Description |
|------|------|
| `from_local(local_tensor, layout)` | Create a DTensor from a local tensor |
| `to_local()` | Obtain the local shard tensor |
| `redistribute(device_mesh, placements)` | Redistribute to the target layout |
| `full_tensor()` | Obtain the complete gathered tensor |
| `is_partial()` | Check whether the state is partial |
| `reduce_partial()` | Reduce a partial DTensor |

---

### `Layout`

Tensor-to-mesh layout mapping.

```python
Layout(mesh_shape, mesh_dim_names)
```

**Parameters:**

| Parameter | Type | Description |
|------|------|------|
| `mesh_shape` | `tuple` | Mesh shape, such as `(4, 8)` |
| `mesh_dim_names` | `tuple` | Mesh dimension names, such as `("dp", "tp")` |

**Usage:**

```python
layout = Layout((dp, tp), ("dp", "tp"))
x_layout = layout("dp", "tp")      # Specify each dimension's layout.
w_layout = layout("tp", "None")    # "None" means no sharding.
out_layout = layout()               # An empty layout means Replicate.
```

---

### `DeviceMesh`

Multidimensional device topology management.

```python
class DeviceMesh
```

**Core methods:**

| Method | Description |
|------|------|
| `__getitem__(dim_name)` | Slice a submesh, such as `mesh["dp"]` |
| `get_device_num_along_axis(dim_name)` | Get the device count along a dimension |

---

### `init_device_mesh`

Create a DeviceMesh.

```python
init_device_mesh(
    device_type: str,
    mesh_shape: tuple,
    mesh_dim_names: tuple,
) -> DeviceMesh
```

**Parameters:**

| Parameter | Type | Description |
|------|------|------|
| `device_type` | `str` | Device type, such as `"npu"` or `"cuda"` |
| `mesh_shape` | `tuple` | Mesh shape |
| `mesh_dim_names` | `tuple` | Mesh dimension names |

---

### `get_current_mesh`

Get the currently active DeviceMesh.

```python
get_current_mesh() -> DeviceMesh
```

---

### `distribute_module`

Distribute a module's parameters according to the specified layout.

```python
distribute_module(module, device_mesh, ...)
```

---

### `init_parameters`

Sharded parameter initialization.

```python
init_parameters(model) -> Module
```

Initialize parameters as shards according to their DTensor layouts, avoiding the peak memory cost of initializing full parameters before sharding.

---

### `init_empty_weights`

Empty-weight initialization context.

```python
init_empty_weights()
```

Deferred initialization context that creates parameters without allocating memory; initialize shards later through `init_parameters`.

---

### `init_on_device`

Initialize parameters on the device.

```python
init_on_device()
```

---

### `manual_seed`

Distributed random seed control with PyTorch parity.

```python
manual_seed(seed: int)
```

Set the distributed random seed to keep random number generation consistent across ranks.

---

## Shard / Custom Sharding

### `custom_shard`

Integrate custom parallelism into the DTensor parallel workflow.

```python
custom_shard(...)
```

---

### `DFunction`

Base class for custom distributed Torch autograd functions.

```python
class DFunction(torch.autograd.Function):
    _op_name: str = None

    @staticmethod
    def forward(ctx, *args, **kwargs) -> Tensor: ...

    @staticmethod
    def backward(ctx, *grad_outputs) -> ...: ...

    @classmethod
    def apply(cls, *args, **kwargs) -> Tensor | DTensor: ...
```

**Detailed documentation:** See [DFunction](./dfunction.md).

---

### `SkipDTensorDispatch`

Marker for bypassing DTensor dispatch in gradient hooks.

```python
class SkipDTensorDispatch
```

Use in FSDP/HSDP gradient hooks to operate directly on local tensors without passing through DTensor dispatch.

---

## TP Styles: Declarative Tensor Parallelism

### `ColwiseParallel`

Column-sharded parallel strategy.

```python
class ColwiseParallel(ParallelStyle):
    input_layouts: Placement = Replicate()
    output_layouts: Placement = Shard(-1)
    use_local_output: bool = True
```

---

### `RowwiseParallel`

Row-sharded parallel strategy.

```python
class RowwiseParallel(ParallelStyle):
    input_layouts: Placement = Shard(-1)
    output_layouts: Placement = Replicate()
    use_local_output: bool = True
```

---

### `SequenceParallel`

Sequence parallel strategy.

```python
class SequenceParallel(ParallelStyle)
```

---

### `PrepareModuleInput`

Module input preparation hook.

```python
class PrepareModuleInput(ParallelStyle)
```

---

### `PrepareModuleOutput`

Module output preparation hook.

```python
class PrepareModuleOutput(ParallelStyle)
```

---

### `PrepareModuleInputOutput`

Module input/output preparation hook.

```python
class PrepareModuleInputOutput(ParallelStyle)
```

---

### `ParallelStyle`

Base class for parallel strategies.

```python
class ParallelStyle
```

Abstract base class for all parallel strategies, defining `apply(module, device_mesh)`.

---

### `parallelize_module`

Declarative TP application interface.

```python
parallelize_module(
    module: Module,
    device_mesh: DeviceMesh,
    parallelize_plan: dict | ParallelStyle,
    *,
    src_data_rank: int = 0,
) -> Module
```

**Parameters:**

| Parameter | Type | Description |
|------|------|------|
| `module` | `Module` | Module to parallelize |
| `device_mesh` | `DeviceMesh` | 1-D TP DeviceMesh |
| `parallelize_plan` | `dict` or `ParallelStyle` | Parallel configuration supporting fnmatch glob patterns |
| `src_data_rank` | `int` | Source data rank (defaults to 0) |

---

## PP Pipeline Parallelism

### `PipelineStage`

Pipeline stage wrapper.

```python
class PipelineStage:
    def __init__(
        self,
        module: Module,
        stage_index: int,
        stage_num: int,
    )
```

**Parameters:**

| Parameter | Type | Description |
|------|------|------|
| `module` | `Module` | Submodule for this stage |
| `stage_index` | `int` | Current stage index |
| `stage_num` | `int` | Total number of stages |

---

### `Schedule1F1B`

1F1B scheduling strategy.

```python
class Schedule1F1B:
    def __init__(
        self,
        stage: PipelineStage,
        micro_batch_num: int,
    )
```

---

### `ScheduleGPipe`

GPipe scheduling strategy.

```python
class ScheduleGPipe:
    def __init__(
        self,
        stage: PipelineStage,
        micro_batch_num: int,
    )
```

---

### `ScheduleInterleaved1F1B`

VPP (interleaved 1F1B) scheduling strategy.

```python
class ScheduleInterleaved1F1B:
    def __init__(
        self,
        stages: list,
        micro_batch_num: int,
        *,
        overlap_p2p: bool = False,
        overlap_b_f: bool = False,
    )
```

**Parameters:**

| Parameter | Type | Default | Description |
|------|------|--------|------|
| `stages` | `list` | — | List of PipelineStage objects |
| `micro_batch_num` | `int` | — | Number of micro-batches |
| `overlap_p2p` | `bool` | `False` | Enable P2P prefetch |
| `overlap_b_f` | `bool` | `False` | Enable B/F compute-communication overlap |

---

### `MetaStep`

Scheduling unit abstraction.

```python
class MetaStep
```

Describe a scheduling unit with `step_type` and optional `sub_steps`.

---

### `MetaStepType`

Scheduling unit type enum.

```python
class MetaStepType(enum.Enum):
    FORWARD = "F"
    BACKWARD = "B"
    OVERLAP_F_B = "OFB"
    OVERLAP_B_F = "OBF"
    FWD_RECV = "FR"
    BWD_RECV = "BR"
    FWD_SEND = "FS"
    BWD_SEND = "BS"
```

---

### `BatchDimSpec`

Batch dimension specification.

```python
class BatchDimSpec
```

---

### `CommComputeOverlap`

Two-thread compute-communication overlap coordinator.

```python
class CommComputeOverlap:
    def wrap_dispatch(self, dispatch_fn, ...) -> Callable
    def wrap_combine(self, combine_fn, is_last_layer: bool = False) -> Callable
    def run(self, fwd_fn: Callable, bwd_fn: Callable) -> None
```

---

### `HookCoordinator` / `HookRole`

COMM-first rendezvous primitives.

```python
class HookCoordinator:
    def enable(self) -> None
    def disable(self) -> None
    def rendezvous(self, hook_name: str, role: HookRole) -> None

class HookRole(enum.Enum):
    COMM = "COMM"
    COMPUTE = "COMPUTE"
```

---

## CP Context Parallelism

### `ContextParallel`

Basic context parallelism.

```python
class ContextParallel(ParallelStyle)
```

---

### `AsyncContextParallel`

Asynchronous context parallelism.

```python
class AsyncContextParallel(ParallelStyle)
```

---

### DSA Variants

| Class | Description |
|----|------|
| `DSAIndexerContextParallel` | DSA indexer CP |
| `AsyncDSAIndexerContextParallel` | Asynchronous DSA indexer CP |
| `DSAIndexerLossContextParallel` | DSA indexer + loss CP |
| `AsyncDSAIndexerLossContextParallel` | Asynchronous DSA indexer + loss CP |
| `DSASparseAttentionContextParallel` | DSA Sparse Attention CP |
| `AsyncDSASparseAttentionContextParallel` | Asynchronous DSA sparse attention CP |

All DSA classes inherit `ParallelStyle` and support `apply(module, device_mesh)`.

---

## EP Expert Parallelism

### `ExpertParallel`

Standard all-to-all EP.

```python
class ExpertParallel(BaseExpertParallel)
```

Each rank owns `num_experts / ep_degree` local experts and dispatches/combines tokens through differentiable all-to-all.

**Mesh requirement:** A 1-D mesh, such as `mesh_dim_names=("ep",)`.

---

### `TensorParallel` (within EP)

TP-only weight sharding, used when EP degree = 1.

```python
class TensorParallel(BaseExpertParallel)
```

---

### `ExpertTensorParallel`

Two-dimensional EP+TP.

```python
class ExpertTensorParallel(BaseExpertParallel)
```

**Mesh requirement:** A 2-D mesh, such as `mesh_dim_names=("ep", "tp")`.

---

## Process Group Management

### `init_process_group`

Initialize a process group.

```python
init_process_group(backend: str, ...)
```

---

### `destroy_process_group`

Destroy a process group.

```python
destroy_process_group()
```

---

### `get_process_group_ranks`

Get all ranks in a process group.

```python
get_process_group_ranks(group) -> list
```

---

### `get_backend`

Get the current process group's backend type.

```python
get_backend() -> str
```

---

### `split_group`

Split a subgroup from a parent process group.

```python
split_group(group, split_size, ...)
```

---

### `get_group_local_rank`

Get the local rank within a process group.

```python
get_group_local_rank(group) -> int
```

---

### `mark_created_groups`

Mark a created process group.

```python
mark_created_groups(groups)
```

---

## Optimizers

### `AdamW`

Standard AdamW optimizer.

```python
class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params: list,
        lr: float = 1e-3,
        weight_decay: float = 0.01,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        ...
    )
```

---

### `Muon`

Muon momentum-based optimizer.

```python
class Muon(torch.optim.Optimizer):
    def __init__(
        self,
        params: list,
        lr: float = 0.02,
        momentum: float = 0.95,
        ns_steps: int = 5,
        nesterov: bool = True,
        weight_decay: float = 0.1,
        ...
    )
```

---

### `ChainedOptimizer`

Chained optimizer combining Muon and AdamW.

```python
class ChainedOptimizer:
    def __init__(
        self,
        model: nn.Module,
        optimizers: Dict[str, Optimizer],
        flatten: bool = False,
    )

    def step(self) -> None
    def zero_grad(self, set_to_none: bool = True) -> None
    def __iter__(self)
```

---

### `get_hyper_optimizer`

Optimizer factory function.

```python
get_hyper_optimizer(
    model: nn.Module,
    muon_params: List[Dict[str, Any]],
    adamw_params: List[Dict[str, Any]],
    muon_kwargs: Optional[Dict[str, Any]] = None,
    adamw_kwargs: Optional[Dict[str, Any]] = None,
) -> ChainedOptimizer
```

**Parameters:**

| Parameter | Type | Description |
|------|------|------|
| `model` | `nn.Module` | Model |
| `muon_params` | `List[Dict]` | Muon parameter groups; an empty list disables Muon |
| `adamw_params` | `List[Dict]` | AdamW parameter groups; an empty list disables AdamW |
| `muon_kwargs` | `Dict` | Muon configuration with automatic stripping of the `muon_` prefix |
| `adamw_kwargs` | `Dict` | AdamW configuration with automatic stripping of the `adamw_` prefix |

---

## Activation Checkpoint / Swap

> Public APIs are in `hyper_parallel.core.activation_memory`.

### `checkpoint`

Functional activation checkpointing; defaults to `use_reentrant=False`, with `use_reentrant=True` available in Torch eager mode.

```python
checkpoint(
    function,
    *args,
    swap_inputs: bool = False,
    policy_fn: Optional[Callable] = None,
    context_fn: Optional[Callable[[], Tuple[object, object]]] = None,
    group_swap: bool = False,
    cpu_pool=None,
    early_stop: bool = True,
    use_reentrant: bool = False,
    **kwargs,
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|------|------|--------|------|
| `function` | `callable` | — | Function or module to checkpoint |
| `swap_inputs` | `bool` | `False` | Offload saved checkpoint inputs to CPU |
| `policy_fn` | `callable` | `None` | Per-operator SAC policy: `(ctx, op, *args, **kwargs) -> CheckpointPolicy` |
| `context_fn` | `callable` | `None` | Zero-argument factory returning `(forward_ctx, recompute_ctx)`; can be combined with `policy_fn` in non-reentrant mode |
| `group_swap` | `bool` | `False` | Enable grouped copy fusion for `MUST_SWAP` tensors |
| `cpu_pool` | Optional object | `None` | Pinned memory pool for non-reentrant input swapping |
| `early_stop` | `bool` | `True` | Stop recomputation once all tensors needed by backward are available |
| `use_reentrant` | `bool` | `False` | Whether to enable Torch eager reentrant checkpointing |

The default eager path uses HyperParallel's non-reentrant implementation. `early_stop` accepts only a per-call keyword or
`checkpoint_kwargs` configuration; it does not inherit the outer `torch.utils.checkpoint.set_checkpoint_early_stop()` setting.
`preserve_rng_state` is configurable in both modes; `determinism_check` and `debug` apply only to non-reentrant mode.
Reentrant mode supports `context_fn` and ordinary `backward()`, but not `policy_fn`, `swap_inputs`,
`group_swap`, `cpu_pool`, nondefault `early_stop`, `torch.compile`, or higher-order gradients.
Outside the explicit phased scope of a PP stage, reentrant internals may accumulate gradients for parameters captured by closures
even when `autograd.grad` requests only input gradients. Use non-reentrant mode when a general differentiation API restricted to specified targets is required.

---

### `swap`

Functional activation swapping without recomputation. Call `SwapManager.set_forward_prefetch_layer` to establish inter-layer prefetch relationships.

```python
swap(
    function,
    *args,
    policy_fn: Optional[Callable] = None,
    **kwargs,
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|------|------|--------|------|
| `function` | `callable` | — | Function or module to swap |
| `policy_fn` | `callable` | `None` | Per-tensor policy: `(tensor) -> CheckpointPolicy`; only `MUST_SAVE` / `MUST_SWAP` are valid |

---

### `checkpoint_wrapper`

Module-level checkpoint wrapper (`CheckpointWrapper`, inheriting `ActivationWrapper`).

```python
checkpoint_wrapper(module, **checkpoint_kwargs) -> CheckpointWrapper
```

`checkpoint_kwargs` matches `checkpoint`, such as `policy_fn`, `swap_inputs`, and `early_stop` in the default mode,
or `use_reentrant=True` in Torch eager mode.

---

### `checkpoint_exclude_wrapper`

Marks a function or Cell inside a checkpoint as a region excluded from recomputation. The original forward caches its outputs and tensors needed for
backward; recomputation reuses those cached outputs. Optional activation swapping offloads eligible outputs and internal activations to CPU.

```python
checkpoint_exclude_wrapper(
    module,
    *,
    swap: bool = False,
    group_swap: bool = False,
    cpu_pool: Optional[PinnedMemoryPool] = None,
) -> CheckpointExcludeWrapper
```

Supported in PyTorch eager mode inside HyperParallel's `checkpoint` or `checkpoint_wrapper`,
for both non-reentrant and reentrant modes. Supports Modules and ordinary callables.

The default is `swap=False`: outputs remain on the device and enter the replay cache. With `swap=True`, explicit outputs and internal saved tensors that cannot be
reconstructed from replay inputs attempt to register with the current `SwapManager` group. Parameters, Parameter views, empty
storage, tensors without an active group, and tensors in the last group automatically remain on the device. `group_swap` and
`cpu_pool` take effect only with `swap=True` inside an active checkpoint, and apply to both outputs and internal activations.

`swap=True` does not automatically create a prefetch chain. Configure the offload/load order through
`SwapManager().set_forward_prefetch_layer(layer, next_layer)`. Without an enclosing active
checkpoint, the wrapper passes through completely: it does not cache outputs, install saved-tensor hooks, or register swap storage.

For PP stages using reentrant checkpointing, ordinary backward needs no additional configuration. With dx/dw separation enabled in
`ScheduleInterleaved1F1B`, the stage automatically manages phased differentiation and replay state per micro-batch, without manual binding.
Reentrant + PP split + FSDP/HSDP is unsupported and raises an explicit error; ordinary FSDP backward is unaffected.

---

### `swap_wrapper`

Module-level pure swap wrapper, without recomputation.

```python
swap_wrapper(
    module,
    policy_fn: Optional[Callable] = None,
) -> SwapWrapper
```

---

### `swap_tensor_wrapper`

Register a tensor in the current swap group inside `forward`.

```python
swap_tensor_wrapper(target, tag: Optional[str] = None)
```

---

### `CheckpointPolicy`

Recomputation / swap policy enum.

```python
class CheckpointPolicy(enum.Enum):
    MUST_SAVE = 0        # Must save on the device.
    PREFER_SAVE = 1      # Prefer saving.
    MUST_RECOMPUTE = 2   # Must recompute.
    PREFER_RECOMPUTE = 3 # Prefer recomputation.
    MUST_SWAP = 4        # Offload to host and reload during backward; requires SwapManager.
```

---

### `SwapManager`

Swap group manager (singleton).

```python
class SwapManager:
    def set_forward_prefetch_layer(self, first_layer, second_layer) -> None
    def launch_offload(self, group_name: str, copy_stream=None) -> None
    def wait_offload(self, group_name: str) -> None
    def launch_load(self, group_name: str, copy_stream=None) -> None
    def wait_load(self, group_name: str) -> None
```

---

## Other Exported Interfaces

### Main Exports (`hyper_parallel/__init__.py`)

```python
__all__ = ["DFunction", "fully_shard", "hsdp_sync_stream", "HSDPModule", "DTensor",
           "Layout", "DeviceMesh", "init_device_mesh", "get_current_mesh", "distribute_module",
           "distribute_tensor", "ones", "zeros", "empty", "full", "rand", "randn",
           "Shard", "RaggedShard", "Replicate", "Partial", "Placement",
           "init_parameters", "init_empty_weights", "init_on_device",
           "shard_module", "custom_shard", "SkipDTensorDispatch",
           "MetaStep", "MetaStepType", "BatchDimSpec", "PipelineStage", "ScheduleInterleaved1F1B",
           "ScheduleMPipeTranspose",
           "init_process_group", "destroy_process_group", "get_process_group_ranks", "get_backend", "split_group",
           "get_group_local_rank", "mark_created_groups",
           "ContextParallel", "AsyncContextParallel",
           "AsyncDSAIndexerContextParallel", "AsyncDSAIndexerLossContextParallel",
           "AsyncDSASparseAttentionContextParallel",
           "DSAIndexerContextParallel", "DSAIndexerLossContextParallel", "DSASparseAttentionContextParallel",
           "ColwiseParallel", "MC2ColwiseParallel", "MC2RowwiseParallel", "MC2Linear",
           "NoParallel", "RowwiseParallel", "SequenceParallel",
           "PrepareModuleInput", "PrepareModuleInputOutput", "PrepareModuleOutput",
           "ParallelStyle", "parallelize_module", "manual_seed"]
```
