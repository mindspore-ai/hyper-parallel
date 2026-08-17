# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""MindSpore fully_shard parameter lifecycle and gradient communication."""
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple, cast
import itertools
import mindspore as ms
from mindspore import nn
from mindspore.common.api import _no_grad
from mindspore import Parameter
import mindspore.mint.distributed as dist
from hyper_parallel.core.fully_shard.utils import (
    MixedPrecisionPolicy,
    CPUOffloadPolicy,
    OffloadPolicy,
    DataParallelMeshInfo,
    DDPMeshInfo,
    FSDPMeshInfo,
    HSDPMeshInfo,
    TPShardMetaInfo,
)
from hyper_parallel.core.dtensor.device_mesh import DeviceMesh
from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.dtensor.layout import Layout
from hyper_parallel.core.fully_shard.hsdp_param import HSDPParamV2
from hyper_parallel.core.fully_shard.hsdp_utils import (
    ShardedState,
    apply_gradient_scaling_factor,
    unwrap_dtensor_param,
)
from hyper_parallel.core.dtensor.placement_types import Replicate, Shard, StridedShard
from hyper_parallel.core.fully_shard.hsdp_utils import ParamModuleInfo
from hyper_parallel.platform.mindspore.fully_shard._version_utils import copy_without_bumping_version
from hyper_parallel.platform.mindspore.utils import normalize_runtime_device
from hyper_parallel.platform.mindspore.fully_shard.pack_utils import (
    build_rs_plan,
    pack_for_reduce_scatter,
    unpack_from_all_gather,
)


def _pack_for_reduce_scatter(local_tensor: ms.Tensor, shard_dim: int, world_size: int) -> ms.Tensor:
    """Pack one local gradient into the row-major reduce-scatter layout.

    MindSpore currently aligns with the torch non-comm-fusion V1 path:

    - shard on dim 0: identity flatten
    - shard on non-dim0: chunk on shard dim, then concatenate on dim 0
    """
    if world_size <= 1 or shard_dim == 0:
        return local_tensor
    chunks = ms.mint.chunk(local_tensor, world_size, dim=shard_dim)
    return ms.mint.cat(chunks, dim=0).contiguous()


def _to_dtype_if_needed(
    tensor: ms.Tensor, dtype: Optional[ms.Type]
) -> ms.Tensor:
    """Cast tensor to the given dtype if it differs from current dtype."""
    if isinstance(dtype, ms.Type) and tensor.dtype != dtype:
        return tensor.to(dtype)
    return tensor


def make_contiguous_strides_for(shape, row_major=True):
    """
    Compute strides for a contiguous tensor of the given shape.

    Args:
        shape (tuple of int): The shape of the tensor. Each dimension must be a non-negative integer.
        row_major (bool):
            - If True (default), returns C-style (row-major) strides: last dimension changes fastest.
            - If False, returns strides where the last two dimensions are Fortran-style
              (i.e., for batched matrix operations in BLAS/LAPACK): second-to-last dim changes fastest.

    Returns:
        tuple of int: The computed strides.

    Examples:
        >>> make_contiguous_strides_for((2, 3, 4))
        (12, 4, 1)
        >>> make_contiguous_strides_for((2, 3, 4), row_major=False)
        (12, 1, 3)
        >>> make_contiguous_strides_for((5,))
        (1,)
        >>> make_contiguous_strides_for((5,), row_major=False)
        (1,)
        >>> make_contiguous_strides_for(())
        ()
    """
    if not isinstance(shape, (tuple, list)):
        raise TypeError("shape must be a tuple or list of non-negative integers")

    # Validate shape elements
    for dim in shape:
        if not isinstance(dim, int) or dim < 0:
            raise ValueError("All dimensions in shape must be non-negative integers")

    if not shape:
        return ()

    # Compute C-style (row-major) strides: stride[i] = product(shape[i+1:])
    strides = []
    multiplier = 1
    # Traverse shape in reverse order
    for size in reversed(shape):
        strides.append(multiplier)
        multiplier *= max(size, 1)  # handle size=0 gracefully (treat as 1 for stride calc)

    # Reverse to get correct order
    c_strides = tuple(reversed(strides))

    if row_major:
        return c_strides
    # For column-major: only affect last two dimensions
    if len(shape) < 2:
        return c_strides
    # In Fortran-style for matrices:
    #   stride of last dim = 1
    #   stride of second-to-last dim = shape[-1]
    # But note: in batched case (..., M, N), we want strides (..., N, 1) → wait!
    # However, the original PyTorch logic returns: result[:-2] + (1, max(shape[-2], 1))
    # Let's follow that exactly:
    # Example: shape=(B, M, N) → c_strides=(M*N, N, 1)
    #           col-major → (M*N, 1, M)
    # So: keep all but last two, then (1, shape[-2])
    return c_strides[:-2] + (1, max(shape[-2], 1))


@dataclass
class ReduceScatterCommCtx:
    """Per-parameter reduce-scatter output and asynchronous work."""

    reduce_scatter_output: Optional[ms.Tensor] = None
    reduce_scatter_handle: Optional[Any] = None


@dataclass
class AllReduceCommCtx:
    """Per-parameter all-reduce output and asynchronous work."""

    all_reduce_output: Optional[ms.Tensor] = None
    all_reduce_handle: Optional[Any] = None


@dataclass
class AllGatherCommCtx:
    """Per-parameter all-gather buffers and asynchronous work."""

    allgather_input: Optional[ms.Tensor] = None
    allgather_output: Optional[ms.Tensor] = None
    allgather_handle: Optional[Any] = None


class MindSporeHSDPParamV2(HSDPParamV2):
    """
    MindSpore HSDP parameter.
    """

    def __init__(
        self,
        param: Parameter,
        module_info: ParamModuleInfo,
        mesh_info: DataParallelMeshInfo,
        shard_placement_fn: Optional[Callable[[Parameter], Optional[Shard]]] = None,
        mp_policy: Optional[MixedPrecisionPolicy] = None,
        offload_policy: Optional[OffloadPolicy] = None,
        device: Optional[str] = None,
        tp_grad_info: Optional[TPShardMetaInfo] = None,
    ):
        self._module_info: ParamModuleInfo = module_info
        self.mesh_info = mesh_info
        self.mp_policy = mp_policy
        self.device = device
        self.orig_dtype = None
        self.param_dtype = None
        self.reduce_dtype = None
        self.offload_to_cpu: bool = isinstance(offload_policy, CPUOffloadPolicy)
        self.pin_memory = (
            self.offload_to_cpu and cast(CPUOffloadPolicy, offload_policy).pin_memory
        )
        self._orig_param_hooks: List[Callable] = []
        self.grad_offload_event: Optional[ms.runtime.Event] = None
        dtensor_payload = unwrap_dtensor_param(param)
        if (dtensor_payload is not None) != (
            tp_grad_info is not None and tp_grad_info.origin_is_dtensor
        ):
            raise ValueError(
                "tp_grad_info.origin_is_dtensor must be True exactly for native DTensor parameters, "
                f"got parameter type {type(param).__name__} and tp_grad_info={tp_grad_info}"
            )
        self.tp_grad_info = tp_grad_info
        self._orig_param_is_dtensor = (
            tp_grad_info is not None and tp_grad_info.origin_is_dtensor
        )
        self._orig_dtensor_mesh = tp_grad_info.mesh if self._orig_param_is_dtensor else None
        self._orig_dtensor_placements = (
            tuple(tp_grad_info.placements) if self._orig_param_is_dtensor else None
        )
        self._spmd_shard_mesh_dim = self.mesh_info.shard_mesh_dim
        self._spmd_replicate_mesh_dim = self.mesh_info.replicate_mesh_dim
        self._init_sharded_param(param, shard_placement_fn)
        self._save_backward_hooks(param)
        self.unsharded_param_buffers: List[ms.Tensor] = []
        self.unsharded_accumulated_grad = None
        self._unsharded_param: Optional[Parameter] = None
        self._param_fqn: Optional[str] = None
        # Communication attributes for prefetch pattern
        self.allgather_comm_ctx = AllGatherCommCtx()
        self.reduce_scatter_comm_ctx = ReduceScatterCommCtx()
        self.all_reduce_comm_ctx = AllReduceCommCtx()
        self._accumulated_allreduced_grad = True
        self._reduce_partial_output = None
        self._post_load_hook_handle = (
            module_info.module.register_load_state_dict_post_hook(
                lambda *args, **kwargs: self.reset_sharded_param()
            )
        )
        self.gradient_scaling_factor = None

    @property
    def accumulated_allreduced_grad(self) -> bool:
        return self._accumulated_allreduced_grad

    @accumulated_allreduced_grad.setter
    def accumulated_allreduced_grad(self, value: bool) -> None:
        self._accumulated_allreduced_grad = value

    @property
    def reduce_partial_output(self) -> Optional[ms.Tensor]:
        """Return reduce-scatter results accumulated before the final micro-step."""
        return self._reduce_partial_output

    @reduce_partial_output.setter
    def reduce_partial_output(self, value: Optional[ms.Tensor]) -> None:
        self._reduce_partial_output = value

    def reduce_comm_dtype(self, grad: Optional[ms.Tensor] = None):
        """Resolve the communication dtype owned by this parameter."""
        if self.reduce_dtype is not None:
            return self.reduce_dtype
        if grad is not None:
            return grad.dtype
        if self.unsharded_accumulated_grad is not None:
            return self.unsharded_accumulated_grad_data.dtype
        if self.unsharded_param is not None and self.unsharded_param.grad is not None:
            return self.unsharded_grad_data.dtype
        return self.orig_dtype

    def _get_base_spmd_placements(self) -> tuple:
        """Return source-layout placements prefixed by explicit data-parallel axes."""
        if self.tp_grad_info is not None:
            self._spmd_mesh = DeviceMesh.concatenate(
                [self.mesh_info.mesh, self.tp_grad_info.mesh]
            )
            dp_prefix = tuple(Replicate() for _ in range(self.mesh_info.mesh.ndim))
            return dp_prefix + tuple(self.tp_grad_info.placements)
        self._spmd_mesh = self.mesh_info.mesh
        return tuple(Replicate() for _ in range(self._spmd_mesh.ndim))

    def _apply_data_parallel_placements(
        self, placements: list, shard_placement: Shard
    ) -> tuple:
        """Apply the parameter-specific DDP/FSDP layout to source placements."""
        if len(placements) != self._spmd_mesh.ndim:
            raise AssertionError(
                f"Expected {self._spmd_mesh.ndim} unified placements, got "
                f"{len(placements)}: {placements}"
            )
        if (
            isinstance(self.mesh_info, DDPMeshInfo)
            and self._spmd_replicate_mesh_dim is not None
            and not self._orig_param_is_dtensor
        ):
            placements[self._spmd_replicate_mesh_dim] = Replicate()
        if isinstance(self.mesh_info, FSDPMeshInfo) and self._spmd_shard_mesh_dim is not None:
            placements[self._spmd_shard_mesh_dim] = self._get_data_parallel_shard_placement(
                placements, shard_placement
            )
        return tuple(placements)

    def _get_data_parallel_shard_placement(self, placements: list, shard_placement: Shard):
        """Return the explicit fully_shard placement on the unified SPMD mesh."""
        split_factor = 1
        shard_mesh_dim = getattr(self, "_spmd_shard_mesh_dim", None)
        for mesh_idx, placement in enumerate(placements):
            if mesh_idx == shard_mesh_dim:
                continue
            if placement.is_shard(shard_placement.dim):
                split_factor *= self._spmd_mesh.mesh_shape[mesh_idx]
        if split_factor > 1:
            return StridedShard(shard_placement.dim, split_factor=split_factor)
        return shard_placement

    def _release_full_param_storage_if_safe(self, param_data: ms.Tensor) -> None:
        """Release the temporary full-parameter storage once the sharded param is installed.

        Skip storage reclamation only for meta tensors. Both plain Tensor inputs and DTensor local
        tensors should drop their original storage after the sharded Parameter has been installed
        onto the owning modules.
        """
        if param_data.is_meta:
            return
        storage = param_data.untyped_storage()
        if storage.size() != 0:
            storage.resize_(0)

    def _iter_backward_hooks(self, param: Parameter) -> List[Callable]:
        """Return backward hooks registered on a MindSpore Tensor/Parameter."""
        hooks_getter = getattr(param, "hooks", None)
        if callable(hooks_getter):
            try:
                return list(hooks_getter())
            except (AttributeError, RuntimeError, TypeError, ValueError):
                pass

        backward_hooks = getattr(param, "_backward_hooks", None)
        if backward_hooks is None:
            return []
        if hasattr(backward_hooks, "values"):
            return list(backward_hooks.values())
        return list(backward_hooks)

    def _save_backward_hooks(self, param: Parameter) -> None:
        """Save user-registered parameter backward hooks for later parameter swaps."""
        if not hasattr(self, "_orig_param_hooks"):
            self._orig_param_hooks = []
        if not hasattr(self, "_saved_hook_ids"):
            self._saved_hook_ids = set()

        for hook_func in self._iter_backward_hooks(param):
            hook_func_id = id(hook_func)
            if hook_func_id not in self._saved_hook_ids:
                self._orig_param_hooks.append(hook_func)
                self._saved_hook_ids.add(hook_func_id)

    def _migrate_backward_hooks(self, new_param: Parameter) -> None:
        """Migrate saved user backward hooks to the active sharded/unsharded parameter."""
        if not getattr(self, "_orig_param_hooks", None):
            return
        if hasattr(new_param, "migrate_backward_hooks_run_once"):
            return
        register_hook = getattr(new_param, "register_hook", None)
        if not callable(register_hook):
            return

        for hook_func in self._orig_param_hooks:
            try:
                if getattr(new_param, "requires_grad", False):
                    register_hook(hook_func)
            except (RuntimeError, TypeError, ValueError):
                pass
        new_param.migrate_backward_hooks_run_once = True

    @_no_grad()
    def _init_sharded_param(
        self,
        param: Parameter,
        shard_placement_fn: Optional[Callable],
    ) -> None:
        param_device = normalize_runtime_device(param.device)
        if param_device not in ("meta", self.device):
            raise AssertionError(
                f"Expects the parameter to already be moved to device {self.device} but got {param.device}"
            )
        hsdp_placement = shard_placement_fn(param) if shard_placement_fn else None
        if hsdp_placement is None:
            hsdp_placement = Shard(0)
        elif hsdp_placement.dim < 0:
            # if dim is negative, add the number of dimensions of the parameter
            hsdp_placement = Shard(hsdp_placement.dim + param.ndim)

        if not isinstance(hsdp_placement, Shard):
            raise AssertionError(
                f"Expected Shard, got {type(hsdp_placement)}: {hsdp_placement}"
            )

        self.hsdp_placement = hsdp_placement
        base_placements = list(self._get_base_spmd_placements())
        self._spmd_placements = self._apply_data_parallel_placements(base_placements, hsdp_placement)
        param_data = unwrap_dtensor_param(param).to_local() if self._orig_param_is_dtensor else param

        shard_dim = hsdp_placement.dim
        self._orig_size = param_data.shape
        self._contiguous_orig_stride = make_contiguous_strides_for(self._orig_size)

        if isinstance(self.mesh_info, FSDPMeshInfo):
            self.shard_rank = self.mesh_info.shard_mesh_rank
            self.shard_world_size = self.mesh_info.shard_mesh_size
        else:
            self.shard_rank = 0
            self.shard_world_size = 1
        if isinstance(self.mesh_info, DDPMeshInfo):
            self.replicate_world_size = self.mesh_info.replicate_mesh_size
        else:
            self.replicate_world_size = 1
        self.is_replicate_param = (
            isinstance(self.mesh_info, DDPMeshInfo)
            and not isinstance(self.mesh_info, HSDPMeshInfo)
        )
        self.is_sharded = self.shard_world_size > 1

        if param_data.shape[shard_dim] % self.shard_world_size != 0:
            raise NotImplementedError(
                f"Uneven sharding on dim {shard_dim} not supported: "
                f"shape={param_data.shape}, world_size={self.shard_world_size}"
            )
        chunks = ms.mint.chunk(param_data, self.shard_world_size, dim=shard_dim)
        sharded_param = chunks[self.shard_rank].clone().contiguous()
        self.sharded_size = sharded_param.shape
        self.contiguous_sharded_stride = make_contiguous_strides_for(self.sharded_size)
        self._sharded_param_data = sharded_param.view(-1)

        self._sharding_spec = Layout.from_device_mesh(self._spmd_mesh)
        self._sharding_spec.set_placements(self._spmd_placements)
        self._sharding_spec.placement_to_tensor_map(param.ndim)

        shard_dtensor = DTensor.from_local(sharded_param, self._spmd_mesh, self._spmd_placements)
        self.sharded_param = Parameter(shard_dtensor, name=param.name)
        set_requires_grad_if_needed(param, self.sharded_param)
        self.sharded_param.grad = None

        self._setattr_on_modules(self.sharded_param)
        self._release_full_param_storage_if_safe(param_data)
        self.sharded_param._hsdp_param_initialized = True
        self.sharded_state = ShardedState.SHARDED
        self.param_dtype = None

    def init_dtype_attrs(self, mp_policy: MixedPrecisionPolicy):
        param_dtype, reduce_dtype = (mp_policy.param_dtype, mp_policy.reduce_dtype)
        self.orig_dtype = self.sharded_param.dtype
        if reduce_dtype == param_dtype:
            reduce_dtype = None
        if param_dtype == self.orig_dtype:
            param_dtype = None
        self.param_dtype = param_dtype
        self.reduce_dtype = reduce_dtype

    def init_unsharded_param_buffers(
        self,
        all_gather_input_numels: list[int],
        all_gather_input_dtypes: list[ms.Type],
        world_size: int,
        device: str,
        force_recreate: bool = False,
    ):
        if not force_recreate and len(self.unsharded_param_buffers) > 0:
            return  # already initialized
        self.unsharded_param_buffers = [
            ms.mint.empty([numel * world_size], dtype=dtype, device=device.split(':')[0])
            for numel, dtype in zip(all_gather_input_numels, all_gather_input_dtypes)
        ]

    def init_unsharded_param(self):
        """
        Initialize unsharded parameter from all-gather outputs.

        This reconstructs the full parameter after all-gather by unpacking the
        gathered flat buffer back to the original tensor layout.
        """
        unsharded_param = self._get_unsharded_param_from_all_gather_output()
        if self._unsharded_param is not None:
            # Keep the Parameter identity stable across forward-reshard-backward
            # cycles so backward hooks continue to read gradients from the same
            # object that participated in the forward graph.
            if self._orig_param_is_dtensor:
                self._unsharded_param.set_data(unsharded_param)
            else:
                self._unsharded_param.data = unsharded_param
            set_requires_grad_if_needed(self.sharded_param, self._unsharded_param)
            self._unsharded_param.grad = None
            return
        if self._orig_param_is_dtensor:
            self._unsharded_param = Parameter(
                unsharded_param,
                name=self.sharded_param.name,
                requires_grad=self.sharded_param.requires_grad,
            )
            return
        # For MindSpore, if use `Parameter(tensor)`, Parameter will create a new Tensor instead of a view.
        # Here we need to share storage, so we use the `.data = tensor` approach to create shared storage.
        self._unsharded_param = Parameter(
            [],
            name=self.sharded_param.name,
            requires_grad=False,
        )
        self._unsharded_param.data = unsharded_param
        if self.sharded_param.requires_grad:
            self._unsharded_param.requires_grad = True

    def _get_unsharded_param_from_all_gather_output(self):
        """Reconstruct the full local parameter view from the packed all-gather output."""
        if len(self.unsharded_param_buffers) != 1:
            raise AssertionError(
                f"Expected 1 unsharded_param_buffer, got {len(self.unsharded_param_buffers)}"
            )
        unsharded_tensor = self.unsharded_param_buffers[0]
        plan = build_rs_plan(
            self,
            self._sharded_local_tensor,
            self.shard_world_size if self.is_sharded else 1,
        )
        unsharded_param = unpack_from_all_gather(unsharded_tensor, plan)
        if getattr(self, "_orig_param_is_dtensor", False):
            unsharded_param = DTensor.from_local(
                unsharded_param,
                self._orig_dtensor_mesh,
                self._orig_dtensor_placements,
            )
        return unsharded_param

    def to_sharded(self) -> None:
        self._setattr_on_modules(self.sharded_param)
        self.free_unsharded_param()
        self.allgather_comm_ctx.allgather_input = None
        self.allgather_comm_ctx.allgather_output = None
        self.allgather_comm_ctx.allgather_handle = None
        self.sharded_state = ShardedState.SHARDED

    def to_unsharded(self) -> None:
        set_requires_grad_if_needed(self.sharded_param, self._unsharded_param)
        self._setattr_on_modules(self._unsharded_param)
        self.sharded_state = ShardedState.UNSHARDED

    def _setattr_on_modules(self, param: Parameter) -> None:
        if getattr(self._module_info.module.__setattr__, "__func__", None) is nn.Cell.__setattr__:
            # fast path
            self._module_info.module._params[self._module_info.param_name] = param
        else:
            # slow path
            setattr(self._module_info.module, self._module_info.param_name, param)
        if hasattr(self, "sharded_param"):
            self._save_backward_hooks(self.sharded_param)
        self._migrate_backward_hooks(param)

        # Iterate through all modules that share this parameter to prevent pointer desync.
        for shared_module, shared_param_name in zip(
            self._module_info.shared_modules, self._module_info.shared_param_names
        ):
            if getattr(shared_module.__setattr__, "__func__", None) is nn.Cell.__setattr__:
                shared_module._params[shared_param_name] = param
            else:
                setattr(shared_module, shared_param_name, param)

    def to_sharded_dtensor(self, tensor: ms.Tensor) -> DTensor:
        """
        Converts a local tensor representing either the sharded parameter or
        sharded gradient to DTensor.
        """
        return DTensor.from_local(
            tensor,
            self._sharding_spec.mesh,
            self._sharding_spec.placements
        )

    def _to_local_unsharded_grad(self, grad):
        """Normalize a pending gradient to the local tensor expected by fully_shard collectives."""
        return self._normalize_unsharded_grad_to_local(grad, reduce_partial_dtensor=False)

    def to_accumulated_grad_if_needed(self) -> None:
        if self._unsharded_param.grad is None:
            return
        unsharded_grad = self._unsharded_param.grad
        self._unsharded_param.grad = None
        if self.reduce_dtype is not None and unsharded_grad.dtype != self.reduce_dtype:
            unsharded_grad = unsharded_grad.to(self.reduce_dtype)
        if self.unsharded_accumulated_grad is None:
            self.unsharded_accumulated_grad = unsharded_grad
        else:
            self.unsharded_accumulated_grad = ms.mint.add(
                self.unsharded_accumulated_grad,
                unsharded_grad,
            )

    def accumulate_unsharded_grad_if_needed(self) -> None:
        if (
            self.unsharded_accumulated_grad is not None
            and self.unsharded_param.grad is not None
        ):
            # need to handle the gradient
            self.unsharded_accumulated_grad = ms.mint.add(
                self.unsharded_accumulated_grad,
                self._to_local_unsharded_grad(self.unsharded_param.grad),
            )
            self.unsharded_param.grad = None

    def alloc_unsharded_param_buffers(self) -> None:
        for tensor in self.unsharded_param_buffers:
            expected_size = tensor.numel() * tensor.itemsize

            storage = tensor.untyped_storage()
            if storage.size() != expected_size:
                storage.resize_(expected_size)

    def free_unsharded_param(self) -> None:
        for tensor in itertools.chain(
            self.unsharded_param_buffers
        ):
            storage = tensor.untyped_storage()
            if storage.size() != 0:
                storage.resize_(0)

    @property
    def all_gather_inputs(self) -> list[ms.Tensor]:
        self._assert_in_states(ShardedState.SHARDED)
        sharded_param_data = self._sharded_param_data
        if self.offload_to_cpu:
            sharded_param_data = sharded_param_data.to(
                self.device, non_blocking=True
            )
        if self.param_dtype is not None and self.param_dtype != sharded_param_data.dtype:
            return [sharded_param_data.to(self.param_dtype)]
        return [sharded_param_data]

    @property
    def unsharded_param(self) -> Parameter:
        """Return the full unsharded parameter after all-gather."""
        return self._unsharded_param

    @property
    def unsharded_grad_data(self) -> ms.Tensor:
        """
        Get the unsharded gradient data as a local tensor.
        """
        grad = self.unsharded_param.grad
        if grad is None:
            raise AssertionError("Expects unsharded_param.grad to not be None")
        return self._to_local_unsharded_grad(grad)

    @property
    def unsharded_accumulated_grad_data(self) -> ms.Tensor:
        """
        Get the unsharded accumulated gradient data as a local tensor.
        """
        grad = self.unsharded_accumulated_grad
        return grad

    @property
    def _sharded_local_tensor(self) -> ms.Tensor:
        """Return the underlying local tensor of the sharded DTensor parameter."""
        return cast(DTensor, self.sharded_param)._local_tensor

    def _sharded_param_storage_dtype(self) -> Optional[ms.Type]:
        """Return the dtype of the sharded parameter's on-device storage."""
        if not hasattr(self.sharded_param, "dtype"):
            return None
        dtype = self.sharded_param.dtype
        if isinstance(dtype, ms.Type):
            return dtype
        return None

    def _assert_in_states(self, *states: ShardedState) -> None:
        """Assert current state is one of expected states."""
        if self.sharded_state not in states:
            raise AssertionError(
                f"Expected sharded_state in {states}, got {self.sharded_state}"
            )

    def _is_same_sharded_local_tensor(self, local_tensor: ms.Tensor) -> bool:
        """Whether the cached flat shard view already points to the ``local_tensor`` storage."""
        if not isinstance(self._sharded_param_data, ms.Tensor):
            return False
        cached_storage = self._sharded_param_data.untyped_storage()
        local_storage = local_tensor.untyped_storage()
        # when sharding param with shape (1, ...) over 2 ranks
        # local_tensor on rank 1 can be size 0, data_ptr() can be 0
        return (
            cached_storage.data_ptr() > 0
            and cached_storage.data_ptr() == local_storage.data_ptr()
        )

    def _validate_sharded_local_tensor_shape(self, local_tensor: ms.Tensor) -> None:
        """Validate that a replaced local tensor still matches the expected shard shape."""
        if local_tensor.shape != self.sharded_size:
            raise AssertionError(
                f"Expected sharded_size to be {self.sharded_size}, got {local_tensor.shape}"
            )

    def _pin_sharded_local_tensor_if_needed(self, local_tensor: ms.Tensor) -> Tuple[ms.Tensor, bool]:
        """Pin the local tensor memory when CPU offload requires it."""
        if self.pin_memory and not local_tensor.is_pinned():
            return local_tensor.to("cpu").pin_memory(), True
        return local_tensor, False

    def _assert_sharded_param_is_dtensor(self) -> None:
        """Assert that ``self.sharded_param`` is backed by a DTensor."""
        if not isinstance(self.sharded_param, DTensor):
            raise AssertionError(f"Expected DTensor, got {type(self.sharded_param)}")

    def _refresh_sharded_local_tensor_view(
        self,
        local_tensor: ms.Tensor,
        shard_dim: int,
        length: int,
    ) -> None:
        """Refresh ``self.sharded_param`` to point to a local tensor view."""
        # Only change the local tensor object if needed
        with _no_grad():
            local_view = local_tensor.narrow(dim=shard_dim, start=0, length=length)
        set_requires_grad_if_needed(self.sharded_param, local_view)
        self.sharded_param._local_tensor = local_view
        if not self.sharded_param._local_tensor.is_contiguous():
            raise AssertionError(
                "Expected sharded_param._local_tensor to be contiguous"
            )

    def reset_sharded_param(self) -> None:
        """Reset the sharded param after ``load_state_dict``."""
        module_info = self._module_info
        new_param = getattr(module_info.module, module_info.param_name)
        if new_param is not self.sharded_param:
            if isinstance(new_param, DTensor):
                self.sharded_param = new_param
                if not getattr(self.sharded_param, "_hsdp_param_initialized", None):
                    # reset _hsdp_param_initialized flag.
                    self.sharded_param._hsdp_param_initialized = True
            elif isinstance(new_param, ms.Tensor):
                # if new_param is Tensor, don't re-ref 'self.sharded_param'
                # just update self.sharded_param._local_tensor and self.sharded_param_data.
                pass

        local_tensor = new_param._local_tensor if isinstance(new_param, DTensor) else new_param
        if local_tensor.is_meta:
            return
        # local_tensor can be padded twice
        # 1st time in fully_shard(model)
        # 2nd time in model(input) lazy_init
        # 2nd time should be no-op if parameters remain unchanged
        # 2nd time shouldn't be no-op if people call model.load_state_dict(...) before lazy_init
        # this makes it possible for trainer to call `sd = model.state_dict()` before the training loop
        # and use `sd` without calling .state_dict() per iteration
        same_local_tensor = self._is_same_sharded_local_tensor(local_tensor)
        shard_dim = self.hsdp_placement.dim
        length = local_tensor.shape[shard_dim] if local_tensor.numel() > 0 else 0
        if not same_local_tensor:
            self._validate_sharded_local_tensor_shape(local_tensor)
        local_tensor, pinned_local_tensor = self._pin_sharded_local_tensor_if_needed(local_tensor)
        updated_local_tensor = not same_local_tensor or pinned_local_tensor
        if not same_local_tensor:
            self._sharded_param_data = local_tensor.view(-1)
        self._assert_sharded_param_is_dtensor()
        if updated_local_tensor:
            self._refresh_sharded_local_tensor_view(local_tensor, shard_dim, length)
        self._sharding_spec = cast(DTensor, self.sharded_param).layout

    @_no_grad()
    def _get_unsharded_param_data(
        self,
        async_op: bool = False,
    ) -> Tuple[ms.Tensor, ms.Tensor, Optional[Any]]:
        """
        Perform all-gather to get unsharded parameter data.

        Args:
            async_op: Whether to execute asynchronously.

        Returns:
            (all_gather_input, unsharded_param, handle): Communication input,
            unsharded parameter data, and communication handle.
        """
        # Optimizer steps may refresh the underlying local tensor storage. Re-sync
        # the cached flat shard view before reading all_gather_inputs for the next
        # unshard cycle.
        self.reset_sharded_param()
        all_gather_input = self.all_gather_inputs[0]

        # If parameter is not sharded (below threshold), no communication needed
        if not self.is_sharded:
            self.init_unsharded_param_buffers(
                all_gather_input_numels=[all_gather_input.numel()],
                all_gather_input_dtypes=[all_gather_input.dtype],
                world_size=1,
                device=all_gather_input.device.split(':')[0],
            )
            self.alloc_unsharded_param_buffers()
            copy_without_bumping_version(self.unsharded_param_buffers[0], all_gather_input)
            return all_gather_input, self.unsharded_param_buffers[0], None

        # Initialize output buffer
        self.init_unsharded_param_buffers(
            all_gather_input_numels=[all_gather_input.numel()],
            all_gather_input_dtypes=[all_gather_input.dtype],
            world_size=self.shard_world_size,
            device=self._sharded_param_data.device.split(':')[0],
        )
        self.alloc_unsharded_param_buffers()

        # Get communication group
        shard_group = self.mesh_info.shard_process_group if isinstance(self.mesh_info, FSDPMeshInfo) else None

        if shard_group is None or self.shard_world_size <= 1:
            # No communication needed, just copy
            copy_without_bumping_version(self.unsharded_param_buffers[0], all_gather_input)
            return all_gather_input, self.unsharded_param_buffers[0], None

        # Execute all_gather_into_tensor
        handle = dist.all_gather_into_tensor(
            self.unsharded_param_buffers[0],
            all_gather_input,
            group=shard_group,
            async_op=async_op,
        )

        return all_gather_input, self.unsharded_param_buffers[0], handle

    def unshard(self, async_op: bool = False) -> None:
        if self.allgather_comm_ctx.allgather_output is not None:
            # Already triggered by HSDPState.prefetch(), so return directly.
            return  # no-op

        all_gather_input, output, handle = self._get_unsharded_param_data(async_op=async_op)
        self.allgather_comm_ctx.allgather_input = all_gather_input
        self.allgather_comm_ctx.allgather_output = output
        self.allgather_comm_ctx.allgather_handle = handle

    def wait_for_unshard(self) -> None:
        self._assert_in_states(ShardedState.SHARDED)

        if self.allgather_comm_ctx.allgather_handle is not None:
            self.allgather_comm_ctx.allgather_handle.wait()
            self.allgather_comm_ctx.allgather_handle = None
        self.allgather_comm_ctx.allgather_input = None

        self.init_unsharded_param()
        self.to_unsharded()

    def shard(self) -> None:
        """
        Transition parameter from unsharded back to sharded state.
        """
        self._assert_in_states(ShardedState.UNSHARDED)
        self.to_sharded()

    def reduce_scatter_output(self):
        """Return cached reduce-scatter output after waiting pending async work."""
        if self.reduce_scatter_comm_ctx.reduce_scatter_handle is not None:
            self.reduce_scatter_comm_ctx.reduce_scatter_handle.wait()
            self.reduce_scatter_comm_ctx.reduce_scatter_handle = None
        return self.reduce_scatter_comm_ctx.reduce_scatter_output

    def clear_reduce_scatter_output(self):
        """Clear cached reduce-scatter output."""
        self.reduce_scatter_comm_ctx.reduce_scatter_output = None

    def reduce_scatter_grad(
        self,
        async_op: bool = True,
        dtype: Optional[ms.Type] = None,
        reduce_op: str = "avg",
        output_buffer: Optional[ms.Tensor] = None,
    ) -> None:
        """
        Perform reduce-scatter on gradient to reduce and shard the full gradient.

        Args:
            async_op: Whether to execute asynchronously.
            dtype: reduce dtype.
            reduce_op: do reduce-scatter avg or sum.
            output_buffer: Optional pre-allocated output for fused all-reduce groups.

        The output and optional asynchronous handle are stored in
        ``reduce_scatter_comm_ctx``.
        """
        # Choose gradient source based on use_accumulated_grad flag
        if self.unsharded_accumulated_grad is not None:
            grad = self.unsharded_accumulated_grad_data
        else:
            grad = self.unsharded_grad_data
        reduce_dtype = dtype or self.reduce_comm_dtype(grad)
        grad = grad.to(reduce_dtype)
        grad = grad.contiguous()
        shard_group_info = getattr(self, "sharded_group_info", None)
        shard_group = shard_group_info.group if shard_group_info is not None else None
        shard_group_size = shard_group_info.rank_size if shard_group_info is not None else 1
        if shard_group is None and isinstance(self.mesh_info, FSDPMeshInfo):
            shard_group = self.mesh_info.shard_process_group
            shard_group_size = self.shard_world_size
        plan_world_size = (
            shard_group_size
            if self.is_sharded and shard_group is not None and shard_group_size > 1
            else 1
        )
        plan = build_rs_plan(self, grad, plan_world_size)
        grad_flat = pack_for_reduce_scatter(grad, plan).reshape(-1)
        # apply gradient_scaling_factor (reduce-scatter leg)
        apply_gradient_scaling_factor(grad_flat, self.gradient_scaling_factor)
        # If parameter is not sharded (below threshold), no reduce-scatter needed
        if not self.is_sharded:
            if output_buffer is not None:
                copy_without_bumping_version(output_buffer, grad_flat)
                self.reduce_scatter_comm_ctx.reduce_scatter_output = output_buffer
            else:
                self.reduce_scatter_comm_ctx.reduce_scatter_output = grad_flat
            self.reduce_scatter_comm_ctx.reduce_scatter_handle = None
            return

        if shard_group is None or shard_group_size <= 1:
            if output_buffer is not None:
                copy_without_bumping_version(output_buffer, grad_flat)
                self.reduce_scatter_comm_ctx.reduce_scatter_output = output_buffer
            else:
                self.reduce_scatter_comm_ctx.reduce_scatter_output = grad_flat
            self.reduce_scatter_comm_ctx.reduce_scatter_handle = None
            return

        # Calculate output size
        output_numel = grad_flat.numel() // shard_group_size
        if output_buffer is not None:
            if output_buffer.numel() != output_numel:
                raise ValueError(
                    f"output_buffer size mismatch: expected {output_numel}, got {output_buffer.numel()}"
                )
            if output_buffer.dtype != reduce_dtype:
                raise ValueError(
                    f"output_buffer dtype mismatch: expected {reduce_dtype}, got {output_buffer.dtype}"
                )
            self.reduce_scatter_comm_ctx.reduce_scatter_output = output_buffer
        else:
            self.reduce_scatter_comm_ctx.reduce_scatter_output = ms.mint.empty(
                output_numel, dtype=reduce_dtype, device=grad.device.split(":")[0]
            )

        # Ascend HCCL DistCommReduceScatter rejects non-contiguous tensors.
        # ``pack_for_reduce_scatter`` on a shard-dim-0 path returns the input
        # tensor as-is (potentially a view from to_local() / redistribute()),
        # and the trailing ``.reshape(-1)`` may yield a view. Force contiguous
        # storage here (no-op when already contig).
        grad_flat = grad_flat.contiguous()

        # Execute reduce_scatter_tensor
        self.reduce_scatter_comm_ctx.reduce_scatter_handle = dist.reduce_scatter_tensor(
            self.reduce_scatter_comm_ctx.reduce_scatter_output,
            grad_flat,
            op=reduce_op,
            group=shard_group,
            async_op=async_op,
        )

    def zero_grad(self):
        """Reset the sharded parameter's gradient buffers to None."""
        self.sharded_param.grad = None
        if hasattr(self.sharded_param, "main_grad"):
            self.sharded_param.main_grad = None

    def all_reduce_grad(
        self,
        async_op: bool = True,
        reduce_op: str = "avg",
    ) -> None:
        """
        Perform all-reduce on gradient (across replicate dimension in HSDP mode).

        Args:
            async_op: Whether to execute asynchronously.
            reduce_op: Reduction operation accepted by ``mint.distributed``.

        The output and optional asynchronous handle are stored in
        ``all_reduce_comm_ctx``.
        """
        grad = self.reduce_scatter_comm_ctx.reduce_scatter_output
        if grad is None:
            raise RuntimeError("all_reduce_grad requires a completed reduce-scatter output.")
        if self.reduce_dtype is not None and self.reduce_dtype != grad.dtype:
            grad = grad.to(self.reduce_dtype)
        reduce_group = (
            self.mesh_info.replicate_process_group
            if isinstance(self.mesh_info, DDPMeshInfo)
            else None
        )
        if reduce_group is None or self.replicate_world_size <= 1:
            self.all_reduce_comm_ctx.all_reduce_output = grad
            self.all_reduce_comm_ctx.all_reduce_handle = None
            return

        # Ascend HCCL DistCommAllReduce rejects non-contiguous tensors.
        # ``grad`` here may be a view returned by ``_to_local_unsharded_grad``
        # (DTensor.to_local() / redistribute().to_local()) or by autograd.
        # ``Tensor.contiguous()`` is itself a no-op when storage is already
        # contiguous, so the unconditional call is safe and avoids the
        # ``is_contiguous()`` query (which has been observed to under-detect
        # non-contig views from DTensor on this MS version).
        grad = grad.contiguous()

        self.all_reduce_comm_ctx.all_reduce_output = grad
        self.all_reduce_comm_ctx.all_reduce_handle = dist.all_reduce(
            grad,
            op=reduce_op,
            group=reduce_group,
            async_op=async_op,
        )

    def all_reduce_output(self):
        """Return cached all-reduce output after waiting pending async work."""
        if self.all_reduce_comm_ctx.all_reduce_handle is not None:
            self.all_reduce_comm_ctx.all_reduce_handle.wait()
            self.all_reduce_comm_ctx.all_reduce_handle = None
        return self.all_reduce_comm_ctx.all_reduce_output

    def clear_all_reduce_output(self):
        """Clear cached all-reduce output."""
        self.all_reduce_comm_ctx.all_reduce_output = None

    def clear_unsharded_source_grad(self) -> None:
        """Release the unsharded gradient after its communication input is safe."""
        if self.unsharded_accumulated_grad is not None:
            self.unsharded_accumulated_grad = None
        if self.unsharded_param is not None and self.unsharded_param.grad is not None:
            self.unsharded_param.grad = None

    def apply_reduced_grad(self, reduced_grad):
        """
        Apply reduced gradient to the sharded parameter.

        Reshapes ``reduced_grad`` to match the local shard, optionally
        offloads to CPU, then accumulates or assigns onto ``grad`` or
        ``main_grad`` depending on the mixed-precision policy.
        Args:
            reduced_grad (ms.Tensor): Gradient after reduce-scatter
                and/or all-reduce.
        """
        if self.mp_policy.apply_grad_on_fp32_main_grad:
            if not hasattr(self.sharded_param, "main_grad"):
                self.sharded_param.main_grad = None
            sharded_grad = self.sharded_param.main_grad
        else:
            sharded_grad = self.sharded_param.grad

        reduced_grad = reduced_grad.reshape(-1).narrow(
            0, 0, self._sharded_local_tensor.numel()
        ).view(self.sharded_size)
        if not self.mp_policy.apply_grad_on_fp32_main_grad:
            reduced_grad = _to_dtype_if_needed(reduced_grad, self.orig_dtype)
            reduced_grad = _to_dtype_if_needed(
                reduced_grad, self._sharded_param_storage_dtype()
            )
        to_accumulate_grad = sharded_grad is not None
        need_synchronize = False
        if self.offload_to_cpu:
            non_blocking = self.pin_memory and not to_accumulate_grad
            reduced_grad = reduced_grad.to(
                "cpu", non_blocking=non_blocking
            )
            need_synchronize = True
        if sharded_grad is None:
            if self.mp_policy.apply_grad_on_fp32_main_grad:
                self.sharded_param.main_grad = self.to_sharded_dtensor(reduced_grad)
                self.sharded_param.grad = None
            else:
                self.sharded_param.grad = self.to_sharded_dtensor(reduced_grad)
        else:
            if self.mp_policy.apply_grad_on_fp32_main_grad:
                accumulated_grad = ms.mint.add(
                    self.sharded_param.main_grad._local_tensor,
                    reduced_grad,
                )
                self.sharded_param.main_grad = self.to_sharded_dtensor(accumulated_grad)
                self.sharded_param.grad = None
            else:
                accumulated_grad = ms.mint.add(
                    self.sharded_param.grad._local_tensor,
                    reduced_grad,
                )
                self.sharded_param.grad = self.to_sharded_dtensor(accumulated_grad)

        self.clear_unsharded_source_grad()
        return need_synchronize

    def all_reduce_tp_replicate_grad_inplace(
        self,
        reduced_grad: ms.Tensor,
        reduce_op: str,
    ) -> None:
        """All-reduce a final gradient over replicated source-layout axes."""
        if self.tp_grad_info is None or not self.tp_grad_info.placements:
            return
        source_mesh = self.tp_grad_info.mesh
        replicate_mesh_dims = tuple(
            mesh_dim
            for mesh_dim, placement in enumerate(self.tp_grad_info.placements)
            if placement.is_replicate()
        )
        if not replicate_mesh_dims:
            return
        if source_mesh.mesh_dim_names is None:
            raise ValueError(
                "TP shard mesh must define mesh_dim_names to all-reduce replicated gradients."
            )
        replicate_dim_names = tuple(
            source_mesh.mesh_dim_names[mesh_dim]
            for mesh_dim in replicate_mesh_dims
        )
        replicate_mesh = source_mesh[replicate_dim_names].flatten()
        if replicate_mesh.size() <= 1:
            return
        dist.all_reduce(
            reduced_grad,
            op=reduce_op,
            group=replicate_mesh.get_group(),
            async_op=False,
        )


def set_requires_grad_if_needed(
    src_tensor: ms.Tensor, dst_tensor: ms.Tensor
) -> None:
    """Synchronize the requires_grad flag from src_tensor to dst_tensor if they differ."""
    if src_tensor.requires_grad != dst_tensor.requires_grad:
        dst_tensor.requires_grad_(src_tensor.requires_grad)
