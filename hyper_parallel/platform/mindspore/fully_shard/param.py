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
import itertools
import math
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple, cast

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
    SourceShardMetaInfo,
)
from hyper_parallel.core.dtensor.device_mesh import DeviceMesh
from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.dtensor.layout import Layout
from hyper_parallel.core.fully_shard.hsdp_param import HSDPParamV2
from hyper_parallel.core.fully_shard.hsdp_utils import (
    ParamModuleInfo,
    ShardedState,
    apply_gradient_scaling_factor,
)
from hyper_parallel.core.dtensor.placement_types import Replicate, Shard, StridedShard
from hyper_parallel.platform.mindspore.fully_shard._version_utils import copy_without_bumping_version
from hyper_parallel.platform.mindspore.utils import normalize_runtime_device


def _to_dtype_if_needed(
    tensor: ms.Tensor, dtype: Optional[ms.Type]
) -> ms.Tensor:
    """Cast tensor to the given dtype if it differs from current dtype."""
    if isinstance(dtype, ms.Type) and tensor.dtype != dtype:
        return tensor.to(dtype)
    return tensor


def _pad_dim0_for_communication(tensor: ms.Tensor, padded_dim0: int) -> ms.Tensor:
    """Pad a dim-0 shard with mint operators for fixed-size collectives."""
    actual_dim0 = tensor.shape[0]
    if actual_dim0 == padded_dim0:
        return tensor
    padding_shape = (padded_dim0 - actual_dim0, *tensor.shape[1:])
    padding = ms.mint.zeros(padding_shape, dtype=tensor.dtype)
    if normalize_runtime_device(padding.device) != normalize_runtime_device(tensor.device):
        padding = padding.to(normalize_runtime_device(tensor.device))
    return ms.mint.cat((tensor, padding), dim=0)


def _split_sharded_param(
    tensor: ms.Tensor,
    shard_world_size: int,
    shard_dim: int,
) -> Tuple[ms.Tensor, ...]:
    """Split a tensor into exactly ``shard_world_size`` contiguous shards.

    Shard sizes differ by at most one, and the first ``dim_size % shard_world_size``
    shards receive one additional element. For example, splitting a dimension of
    size 6 over 4 ranks produces shard sizes ``(2, 2, 1, 1)``.
    """
    dim_size = tensor.shape[shard_dim]
    base_chunk_size, remainder = divmod(dim_size, shard_world_size)
    split_sizes = [base_chunk_size + 1] * remainder
    split_sizes.extend([base_chunk_size] * (shard_world_size - remainder))
    return tuple(ms.mint.split(tensor, split_sizes, dim=shard_dim))


def _pack_dim0_reduce_scatter_input(
    tensor: ms.Tensor,
    shard_world_size: int,
) -> ms.Tensor:
    """Pack dim-0 shards into equal-sized reduce-scatter input slots.

    For example, six rows over four ranks produce logical chunks of sizes
    ``(2, 2, 1, 1)``. After padding each rank slot to two rows, the collective
    input is ``[r0, r1] [r2, r3] [r4, pad] [r5, pad]``. Concatenating these
    slots preserves the balanced rank assignment; padding only at the tensor
    tail would incorrectly place both ``r4`` and ``r5`` in rank 2's slot.
    """
    chunks = _split_sharded_param(tensor, shard_world_size, shard_dim=0)
    padded_dim0 = chunks[0].shape[0]
    padded_chunks = tuple(
        _pad_dim0_for_communication(chunk, padded_dim0)
        for chunk in chunks
    )
    return ms.mint.cat(padded_chunks, dim=0)


def _unpack_dim0_all_gather_output(
    packed_data: ms.Tensor,
    logical_dim0: int,
    shard_world_size: int,
    padded_sharded_shape: Tuple[int, ...],
) -> ms.Tensor:
    """Remove per-rank padding from gathered balanced dim-0 chunks.

    For the ``6 / 4`` example, all-gather returns
    ``[r0, r1] [r2, r3] [r4, pad] [r5, pad]``. Taking the first six flattened
    rows would yield ``[r0, r1, r2, r3, r4, pad]`` and lose ``r5``. The valid
    prefix of each rank slot must therefore be concatenated to restore
    ``[r0, r1, r2, r3, r4, r5]``.
    """
    base_chunk_size, remainder = divmod(logical_dim0, shard_world_size)
    chunk_sizes = [base_chunk_size + 1] * remainder
    chunk_sizes.extend([base_chunk_size] * (shard_world_size - remainder))
    packed_chunks = packed_data.view(
        (shard_world_size, *padded_sharded_shape)
    )
    logical_chunks = tuple(
        packed_chunks[rank].narrow(0, 0, chunk_size)
        for rank, chunk_size in enumerate(chunk_sizes)
    )
    return ms.mint.cat(logical_chunks, dim=0)


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
    """Per-parameter all-gather output and asynchronous work."""

    allgather_output: Optional[ms.Tensor] = None
    allgather_handle: Optional[Any] = None


class ParameterHookMigrator:
    """Preserve parameter backward hooks across HSDP parameter replacement."""

    def __init__(self) -> None:
        """Initialize the ordered hook cache and identity index."""
        self._orig_param_hooks: List[Callable] = []
        self._saved_hook_ids: set[int] = set()

    def _save_backward_hooks(self, param: Parameter) -> None:
        """Save backward hooks from a parameter, deduplicated by hook identity."""
        for hook_func in param.hooks():
            hook_func_id = id(hook_func)
            if hook_func_id not in self._saved_hook_ids:
                self._orig_param_hooks.append(hook_func)
                self._saved_hook_ids.add(hook_func_id)

    def _migrate_backward_hooks(self, new_param: Parameter) -> None:
        """Register saved backward hooks on a replacement parameter once."""
        if not self._orig_param_hooks or hasattr(new_param, "migrate_backward_hooks_run_once"):
            return

        for hook_func in self._orig_param_hooks:
            try:
                if new_param.requires_grad:
                    new_param.register_hook(hook_func)
            except RuntimeError:
                # Skip hook registration if the parameter does not require gradients.
                pass
        new_param.migrate_backward_hooks_run_once = True


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
        source_shard_info: Optional[SourceShardMetaInfo] = None,
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
        self._parameter_hook_migrator = ParameterHookMigrator()
        if isinstance(param, DTensor) != (
            source_shard_info is not None and source_shard_info.origin_is_dtensor
        ):
            raise ValueError(
                "source_shard_info.origin_is_dtensor must be True exactly for native DTensor parameters, "
                f"got parameter type {type(param).__name__} and source_shard_info={source_shard_info}"
            )
        self.source_shard_info = source_shard_info
        self._orig_param_is_dtensor = (
            source_shard_info is not None and source_shard_info.origin_is_dtensor
        )
        self._orig_dtensor_mesh = source_shard_info.mesh if self._orig_param_is_dtensor else None
        self._orig_dtensor_placements = (
            tuple(source_shard_info.placements) if self._orig_param_is_dtensor else None
        )
        self._spmd_shard_mesh_dim = self.mesh_info.shard_mesh_dim
        self._spmd_replicate_mesh_dim = self.mesh_info.replicate_mesh_dim
        self._init_sharded_param(param, shard_placement_fn)
        self._parameter_hook_migrator._save_backward_hooks(param)
        self.unsharded_param_buffers: List[ms.Tensor] = []
        self.unsharded_accumulated_grad = None
        self._param_fqn: Optional[str] = None
        # Communication attributes for prefetch pattern
        self.allgather_comm_ctx = AllGatherCommCtx()
        self.reduce_scatter_comm_ctx = ReduceScatterCommCtx()
        self.all_reduce_comm_ctx = AllReduceCommCtx()
        self._grad = None
        self._reduce_partial_output = None
        self._post_load_hook_handle = (
            module_info.module.register_load_state_dict_post_hook(
                lambda *args, **kwargs: self.reset_sharded_param()
            )
        )
        self.gradient_scaling_factor = None

    def _get_base_spmd_placements(self) -> tuple:
        """Return source-layout placements prefixed by explicit data-parallel axes."""
        if self.source_shard_info is not None:
            self._spmd_mesh = DeviceMesh.concatenate(
                [self.mesh_info.mesh, self.source_shard_info.mesh]
            )
            dp_prefix = tuple(Replicate() for _ in range(self.mesh_info.mesh.ndim))
            return dp_prefix + tuple(self.source_shard_info.placements)
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
            split_factor = 1
            for mesh_idx, placement in enumerate(placements):
                if mesh_idx == self._spmd_shard_mesh_dim:
                    continue
                if placement.is_shard(shard_placement.dim):
                    split_factor *= self._spmd_mesh.mesh_shape[mesh_idx]
            placements[self._spmd_shard_mesh_dim] = (
                StridedShard(shard_placement.dim, split_factor=split_factor)
                if split_factor > 1
                else shard_placement
            )
        return tuple(placements)

    def _build_sharding_spec(
        self,
        source_param: Parameter,
        source_local_tensor: ms.Tensor,
    ) -> Layout:
        """Build the final layout after data and model parallel sharding."""
        logical_global_stride = None
        if isinstance(source_param, DTensor):
            logical_global_size = source_param.shape
            logical_global_stride = source_param.layout.tensor_stride
        elif self.source_shard_info is not None:
            source_sharding_spec = Layout.from_device_mesh(self.source_shard_info.mesh)
            source_sharding_spec.set_placements(self.source_shard_info.placements)
            source_sharding_spec.placement_to_tensor_map(source_local_tensor.ndim)
            logical_global_size = source_sharding_spec.get_global_shape(source_local_tensor.shape)
        else:
            logical_global_size = source_local_tensor.shape

        if logical_global_stride is None:
            logical_global_stride = make_contiguous_strides_for(logical_global_size)

        sharding_spec = Layout.from_device_mesh(self._spmd_mesh)
        sharding_spec.set_placements(self._spmd_placements)
        sharding_spec.placement_to_tensor_map(source_local_tensor.ndim)
        sharding_spec.set_tensor_meta(
            logical_global_size,
            logical_global_stride,
            source_local_tensor.dtype,
        )
        return sharding_spec

    @property
    def reduce_partial_output(self) -> Optional[ms.Tensor]:
        """Return reduce-scatter results accumulated before the final micro-step."""
        return self._reduce_partial_output

    @reduce_partial_output.setter
    def reduce_partial_output(self, value: Optional[ms.Tensor]) -> None:
        """Store reduce-scatter results accumulated before the final micro-step."""
        self._reduce_partial_output = value

    def reduce_comm_dtype(self, grad: Optional[ms.Tensor] = None) -> Optional[ms.Type]:
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

    def reduce_scatter_output(self) -> Optional[ms.Tensor]:
        """Return cached reduce-scatter output after waiting asynchronous work."""
        if self.reduce_scatter_comm_ctx.reduce_scatter_handle is not None:
            self.reduce_scatter_comm_ctx.reduce_scatter_handle.wait()
            self._grad.untyped_storage().resize_(0)
            self._grad = None
            self.reduce_scatter_comm_ctx.reduce_scatter_handle = None
        return self.reduce_scatter_comm_ctx.reduce_scatter_output

    def clear_reduce_scatter_output(self) -> None:
        """Clear the cached reduce-scatter output."""
        self.reduce_scatter_comm_ctx.reduce_scatter_output = None
        self._grad = None

    def all_reduce_output(self) -> Optional[ms.Tensor]:
        """Return cached all-reduce output after waiting asynchronous work."""
        if self.all_reduce_comm_ctx.all_reduce_handle is not None:
            self.all_reduce_comm_ctx.all_reduce_handle.wait()
            self.all_reduce_comm_ctx.all_reduce_handle = None
        return self.all_reduce_comm_ctx.all_reduce_output

    def clear_all_reduce_output(self) -> None:
        """Clear the cached all-reduce output."""
        self.all_reduce_comm_ctx.all_reduce_output = None

    def clear_unsharded_source_grad(self) -> None:
        """Release the unsharded gradient after its communication input is safe."""
        if self.unsharded_accumulated_grad is not None:
            self.unsharded_accumulated_grad = None
        elif self.unsharded_param is not None and self.unsharded_param.grad is not None:
            self.unsharded_param.grad = None

    def apply_reduced_grad(self, reduced_grad: ms.Tensor) -> bool:
        """Apply a reduced gradient to the persistent sharded parameter.

        Args:
            reduced_grad: Gradient after reduce-scatter and optional all-reduce.

        Returns:
            Whether the caller must synchronize after a CPU offload.
        """
        if self.mp_policy.apply_grad_on_fp32_main_grad:
            if not hasattr(self.sharded_param, "main_grad"):
                self.sharded_param.main_grad = None
            sharded_grad = self.sharded_param.main_grad
        else:
            sharded_grad = self.sharded_param.grad

        reduced_grad = reduced_grad.reshape(-1)
        reduced_grad = reduced_grad.narrow(0, 0, self._sharded_local_tensor.numel())
        reduced_grad = reduced_grad.reshape(self.sharded_size)
        if not self.mp_policy.apply_grad_on_fp32_main_grad:
            reduced_grad = _to_dtype_if_needed(reduced_grad, self.orig_dtype)
            reduced_grad = _to_dtype_if_needed(
                reduced_grad, self._sharded_param_storage_dtype()
            )
        to_accumulate_grad = sharded_grad is not None
        need_synchronize = False
        if self.offload_to_cpu:
            non_blocking = self.pin_memory and not to_accumulate_grad
            reduced_grad = reduced_grad.to("cpu", non_blocking=non_blocking)
            need_synchronize = True
        if sharded_grad is None:
            if self.mp_policy.apply_grad_on_fp32_main_grad:
                self.sharded_param.main_grad = self.to_sharded_dtensor(reduced_grad)
                self.sharded_param.grad = None
            else:
                self.sharded_param.grad = self.to_sharded_dtensor(reduced_grad)
        else:
            if self.mp_policy.apply_grad_on_fp32_main_grad:
                self.sharded_param.main_grad._local_tensor.add_(reduced_grad)
                self.sharded_param.grad = None
            else:
                self.sharded_param.grad._local_tensor.add_(reduced_grad)

        self.clear_unsharded_source_grad()
        return need_synchronize

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

    def _resolve_hsdp_placement(
        self,
        param: Parameter,
        shard_placement_fn: Optional[Callable],
    ) -> Shard:
        """Validate and normalize the fully_shard placement for one parameter."""
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
        return hsdp_placement

    def _init_shard_placements(
        self,
        param_data: ms.Tensor,
        shard_dim: int,
        base_placements: list,
    ) -> None:
        """Build data-parallel placements for the local shard."""
        if shard_dim != 0 and param_data.shape[shard_dim] % self.shard_world_size != 0:
            raise NotImplementedError(
                f"fully_shard only supports uneven sharding on dim=0, but parameter "
                f"{self._module_info.param_name} has shape {tuple(param_data.shape)}, "
                f"shard dim {shard_dim}, and world size {self.shard_world_size}"
            )
        spmd_placements = list(
            self._apply_data_parallel_placements(base_placements, self.hsdp_placement)
        )
        if param_data.shape[shard_dim] % self.shard_world_size != 0:
            if self._spmd_shard_mesh_dim is None:
                raise AssertionError("Uneven FSDP sharding requires a shard mesh dimension")
            fsdp_placement = spmd_placements[self._spmd_shard_mesh_dim]
            if isinstance(fsdp_placement, StridedShard):
                fsdp_placement = StridedShard(
                    fsdp_placement.dim,
                    fsdp_placement.split_factor,
                    uneven_shard=True,
                )
            else:
                fsdp_placement = Shard(fsdp_placement.dim, uneven_shard=True)
            spmd_placements[self._spmd_shard_mesh_dim] = fsdp_placement
        self._spmd_placements = tuple(spmd_placements)

    @_no_grad()
    def _init_sharded_param(
        self,
        param: Parameter,
        shard_placement_fn: Optional[Callable],
    ) -> None:
        """Initialize the persistent sharded parameter and communication storage."""
        hsdp_placement = self._resolve_hsdp_placement(param, shard_placement_fn)
        self.hsdp_placement = hsdp_placement
        base_placements = list(self._get_base_spmd_placements())
        param_data = param.to_local() if self._orig_param_is_dtensor else param
        shard_dim = hsdp_placement.dim
        if param_data.ndim == 0:
            raise ValueError("fully_shard does not support scalar parameters")
        if shard_dim < 0 or shard_dim >= param_data.ndim:
            raise ValueError(
                f"Invalid fully_shard dim {shard_dim} for parameter "
                f"{self._module_info.param_name} with shape {tuple(param_data.shape)}"
            )
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
        if param_data.shape[shard_dim] < self.shard_world_size:
            raise ValueError(
                f"MindSpore fully_shard balanced chunking requires sharded dimension size to be greater than or "
                f"equal to the shard world size, but parameter {self._module_info.param_name} has shape "
                f"{tuple(param_data.shape)}, shard dim {shard_dim}, and shard world size {self.shard_world_size}"
            )

        self._init_shard_placements(param_data, shard_dim, base_placements)

        chunks = _split_sharded_param(
            param_data,
            self.shard_world_size,
            shard_dim,
        )
        local_shard = chunks[self.shard_rank]
        sharded_param = local_shard.clone().contiguous()
        self.sharded_size = sharded_param.shape
        self.contiguous_sharded_stride = make_contiguous_strides_for(self.sharded_size)
        self.padded_sharded_param_size = chunks[0].shape
        if self.offload_to_cpu and not sharded_param.is_meta:
            sharded_param = sharded_param.to("cpu")
            if self.pin_memory:
                sharded_param = sharded_param.pin_memory()

        if self.sharded_size == self.padded_sharded_param_size:
            padded_sharded_param = sharded_param
        else:
            padded_sharded_param = _pad_dim0_for_communication(
                sharded_param,
                self.padded_sharded_param_size[0],
            )
            if self.pin_memory and not padded_sharded_param.is_meta:
                padded_sharded_param = padded_sharded_param.pin_memory()
        # Keep the fixed-size communication storage outside autograd from its
        # initial construction as well as after a DelayInit refresh. The
        # logical shard remains the optimizer-owned tensor.
        self._sharded_param_data = padded_sharded_param.detach().view(-1)

        self._sharding_spec = self._build_sharding_spec(param, param_data)

        shard_dtensor = DTensor.from_local(
            sharded_param,
            self._spmd_mesh,
            self._spmd_placements,
            shape=self._sharding_spec.tensor_shape,
            stride=self._sharding_spec.tensor_stride,
        )
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
        if force_recreate and hasattr(self, "_unsharded_param"):
            raise RuntimeError(
                "Cannot recreate unsharded_param_buffers after initializing the stable "
                "unsharded parameter."
            )
        self.unsharded_param_buffers = [
            ms.mint.empty([numel * world_size], dtype=dtype, device=device.split(':')[0])
            for numel, dtype in zip(all_gather_input_numels, all_gather_input_dtypes)
        ]

    def init_unsharded_param(self) -> None:
        """Initialize the stable unsharded parameter from its final output storage."""
        if len(self.unsharded_param_buffers) != 1:
            raise AssertionError(
                f"Expected 1 unsharded_param_buffer, got {len(self.unsharded_param_buffers)}"
            )

        all_gather_output = self.allgather_comm_ctx.allgather_output
        if all_gather_output is not None:
            if self.hsdp_placement.dim == 0:
                if self._orig_size[0] % self.shard_world_size == 0:
                    output_kind = (
                        "the stable unsharded buffer"
                        if all_gather_output is self.unsharded_param_buffers[0]
                        else "a separate temporary buffer"
                    )
                    raise AssertionError(
                        "Internal fully_shard invariant violated for parameter "
                        f"'{self._module_info.param_name}': even dim-0 all-gather with logical shape "
                        f"{tuple(self._orig_size)} and shard world size {self.shard_world_size} must write "
                        f"directly into unsharded_param_buffers[0] and clear "
                        f"allgather_comm_ctx.allgather_output before init_unsharded_param(), but "
                        f"{output_kind} remains. Check output-buffer routing and cleanup in "
                        f"_get_unsharded_param_data()."
                    )
                unsharded_param = _unpack_dim0_all_gather_output(
                    all_gather_output,
                    self._orig_size[0],
                    self.shard_world_size,
                    self.padded_sharded_param_size,
                )
            else:
                packed_shape = list(self.sharded_size)
                packed_shape[0] *= self.shard_world_size
                packed_param = all_gather_output.view(packed_shape)
                param_chunks = packed_param.chunk(self.shard_world_size, dim=0)
                unsharded_param = ms.mint.cat(param_chunks, dim=self.hsdp_placement.dim)
            unsharded_param = _pad_dim0_for_communication(
                unsharded_param.view(-1),
                self.unsharded_param_buffers[0].numel(),
            )
            copy_without_bumping_version(
                self.unsharded_param_buffers[0],
                unsharded_param,
            )
            all_gather_output.untyped_storage().resize_(0)
            self.allgather_comm_ctx.allgather_output = None

        if hasattr(self, "_unsharded_param"):
            # The stable Parameter already views ``unsharded_param_buffers[0]``.
            # Refreshing that buffer above is sufficient for every later cycle.
            return

        unsharded_numel = math.prod(self._orig_size)
        # The buffer may be a view chain into `_sharded_param_data` (non-leaf when
        # rebuilt outside _no_grad, e.g. the DelayInit refresh path). Narrow first,
        # then detach the final view so the unsharded logical parameter stays a leaf.
        unsharded_param = self.unsharded_param_buffers[0].narrow(0, 0, unsharded_numel).detach()
        unsharded_param = unsharded_param.view(self._orig_size)
        if self._orig_param_is_dtensor:
            unsharded_param = DTensor.from_local(
                unsharded_param,
                self._orig_dtensor_mesh,
                self._orig_dtensor_placements,
            )
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
        # reset self._unsharded_param tensor_impl
        self._unsharded_param.data = unsharded_param
        if self.sharded_param.requires_grad:
            self._unsharded_param.requires_grad = True

    def to_sharded(self) -> None:
        self._setattr_on_modules(self.sharded_param)
        if self.unsharded_param_buffers[0] is not self._sharded_param_data:
            self.free_unsharded_param()
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
        self._parameter_hook_migrator._save_backward_hooks(self.sharded_param)
        self._parameter_hook_migrator._migrate_backward_hooks(param)

        # Iterate through all modules that share this parameter to prevent pointer desync.
        for shared_module, shared_param_name in zip(
            self._module_info.shared_modules, self._module_info.shared_param_names
        ):
            if getattr(shared_module.__setattr__, "__func__", None) is nn.Cell.__setattr__:
                shared_module._params[shared_param_name] = param
            else:
                setattr(shared_module, shared_param_name, param)

    def to_sharded_dtensor(self, tensor: ms.Tensor) -> DTensor:
        """Convert a logical local parameter or gradient shard to DTensor.

        Args:
            tensor: Logical local shard without communication padding.

        Returns:
            A DTensor using the parameter's global logical shape and layout.
        """
        return DTensor.from_local(
            tensor,
            self._sharding_spec.mesh,
            self._sharding_spec.placements,
            shape=self._sharding_spec.tensor_shape,
            stride=self._sharding_spec.tensor_stride,
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
            self.unsharded_accumulated_grad.add_(unsharded_grad)

    def accumulate_unsharded_grad_if_needed(self) -> None:
        if (
            self.unsharded_accumulated_grad is not None
            and self.unsharded_param.grad is not None
        ):
            grad = self._to_local_unsharded_grad(self.unsharded_param.grad)
            if self.reduce_dtype is not None and grad.dtype != self.reduce_dtype:
                grad = grad.to(self.reduce_dtype)
            self.unsharded_param.grad = None
            self.unsharded_accumulated_grad.add_(grad)

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

    def _resolve_reset_param(self):
        """Resolve the possibly replaced module parameter before resetting storage."""
        module_info = self._module_info
        new_param = getattr(module_info.module, module_info.param_name)
        if new_param is self.sharded_param:
            return new_param
        if isinstance(new_param, DTensor):
            self.sharded_param = new_param
            if not getattr(self.sharded_param, "_hsdp_param_initialized", None):
                self.sharded_param._hsdp_param_initialized = True
        return new_param

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

    def _validate_reset_local_tensor(self, local_tensor: ms.Tensor) -> ms.Tensor:
        """Validate that a replaced local tensor still matches the expected shard shape."""
        if local_tensor.shape != self.sharded_size:
            raise AssertionError(
                f"Expected sharded_size to be {self.sharded_size}, got {local_tensor.shape}"
            )
        return local_tensor

    def _pin_reset_local_tensor_if_needed(self, local_tensor: ms.Tensor) -> Tuple[ms.Tensor, bool]:
        """Pin the local tensor memory when CPU offload requires it."""
        if self.pin_memory and not local_tensor.is_pinned():
            return local_tensor.to("cpu").pin_memory(), True
        return local_tensor, False

    def _refresh_sharded_local_tensor(
        self,
        local_tensor: ms.Tensor,
    ) -> None:
        """Rebuild padded communication storage and refresh the DTensor local view."""
        if self.sharded_size == self.padded_sharded_param_size:
            padded_local_tensor = local_tensor
        else:
            padded_local_tensor = _pad_dim0_for_communication(
                local_tensor,
                self.padded_sharded_param_size[0],
            )
            if self.pin_memory and not padded_local_tensor.is_meta:
                padded_local_tensor = padded_local_tensor.pin_memory()
        # Communication storage must stay outside autograd when DelayInit refreshes with grad enabled.
        self._sharded_param_data = padded_local_tensor.detach().view(-1)
        set_requires_grad_if_needed(self.sharded_param, local_tensor)
        self.sharded_param._local_tensor = local_tensor
        if not self.sharded_param._local_tensor.is_contiguous():
            raise AssertionError(
                "Expected sharded_param._local_tensor to be contiguous"
            )

    def reset_sharded_param(self) -> None:
        """Reset the sharded param after ``load_state_dict``."""
        new_param = self._resolve_reset_param()
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
        if not same_local_tensor:
            local_tensor = self._validate_reset_local_tensor(local_tensor)
        local_tensor, pinned_local_tensor = self._pin_reset_local_tensor_if_needed(local_tensor)
        if not isinstance(self.sharded_param, DTensor):
            raise AssertionError(f"Expected DTensor, got {type(self.sharded_param)}")
        if not same_local_tensor or pinned_local_tensor:
            self._refresh_sharded_local_tensor(local_tensor)
        self._sharding_spec.set_tensor_meta(
            self._sharding_spec.tensor_shape,
            self._sharding_spec.tensor_stride,
            local_tensor.dtype,
        )
        self.sharded_param._layout = self._sharding_spec
        self.sharded_param._placements = tuple(self._sharding_spec.placements)
        self._setattr_on_modules(self.sharded_param)

    @_no_grad()
    def _get_unsharded_param_data(
        self,
        async_op: bool = False,
    ) -> None:
        """
        Perform all-gather to get unsharded parameter data.

        Args:
            async_op: Whether to execute asynchronously.

        The output buffer and optional asynchronous handle are stored in the
        parameter communication context.
        """
        # Optimizer steps may refresh the underlying local tensor storage. Re-sync
        # the cached flat shard view before reading all_gather_inputs for the next
        # unshard cycle.
        self.reset_sharded_param()
        all_gather_input = self.all_gather_inputs[0]

        shard_group = self.mesh_info.shard_process_group if isinstance(self.mesh_info, FSDPMeshInfo) else None
        if not self.is_sharded or shard_group is None or self.shard_world_size <= 1:
            if not self.unsharded_param_buffers:
                self.unsharded_param_buffers = [all_gather_input]
            elif self.unsharded_param_buffers[0] is not all_gather_input:
                self.alloc_unsharded_param_buffers()
                copy_without_bumping_version(self.unsharded_param_buffers[0], all_gather_input)
            self.allgather_comm_ctx.allgather_output = None
            self.allgather_comm_ctx.allgather_handle = None
            return

        self.init_unsharded_param_buffers(
            all_gather_input_numels=[all_gather_input.numel()],
            all_gather_input_dtypes=[all_gather_input.dtype],
            world_size=self.shard_world_size,
            device=self._sharded_param_data.device.split(":")[0],
        )
        self.alloc_unsharded_param_buffers()

        self.allgather_comm_ctx.allgather_output = self.unsharded_param_buffers[0]
        if (
            self.hsdp_placement.dim != 0
            or self._orig_size[0] % self.shard_world_size != 0
        ):
            # Non-dim-0 shards and balanced uneven dim-0 shards require an
            # unpack step. Preserve the stable logical full-parameter buffer.
            self.allgather_comm_ctx.allgather_output = ms.mint.empty_like(
                self.unsharded_param_buffers[0]
            )
        self.allgather_comm_ctx.allgather_handle = dist.all_gather_into_tensor(
            self.allgather_comm_ctx.allgather_output,
            all_gather_input,
            group=shard_group,
            async_op=async_op,
        )
        if self.allgather_comm_ctx.allgather_output is self.unsharded_param_buffers[0]:
            self.allgather_comm_ctx.allgather_output = None

    def unshard(self, async_op: bool = False) -> None:
        if self.allgather_comm_ctx.allgather_handle is not None:
            # Already triggered by HSDPState.prefetch(), so return directly.
            return  # no-op

        self._get_unsharded_param_data(async_op=async_op)

    def wait_for_unshard(self) -> None:
        self._assert_in_states(ShardedState.SHARDED)

        if self.allgather_comm_ctx.allgather_handle is not None:
            self.allgather_comm_ctx.allgather_handle.wait()
            self.allgather_comm_ctx.allgather_handle = None

        self.init_unsharded_param()
        self.to_unsharded()

    def shard(self) -> None:
        """
        Transition parameter from unsharded back to sharded state.
        """
        self._assert_in_states(ShardedState.UNSHARDED)
        self.to_sharded()

    def reduce_scatter_grad(
        self,
        async_op: bool = True,
        reduce_op: str = "avg",
        output_buffer: Optional[ms.Tensor] = None,
    ) -> None:
        """
        Perform reduce-scatter on gradient to reduce and shard the full gradient.

        Args:
            async_op: Whether to execute asynchronously.
            reduce_op: do reduce-scatter avg or sum.
            output_buffer: Optional pre-allocated output for fused all-reduce groups.

        The output and optional asynchronous handle are stored in
        ``reduce_scatter_comm_ctx``.
        """
        if self.unsharded_accumulated_grad is not None:
            grad = self.unsharded_accumulated_grad_data
        else:
            grad = self.unsharded_grad_data
        self._grad = grad.to(self.reduce_comm_dtype(grad))
        shard_dim = self.hsdp_placement.dim
        if self.shard_world_size <= 1:
            self._grad = self._grad.view(-1)
        elif shard_dim != 0:
            grad_chunks = self._grad.chunk(self.shard_world_size, dim=shard_dim)
            self._grad = ms.mint.cat(grad_chunks, dim=0).view(-1)
        else:
            if self._grad.shape[0] % self.shard_world_size != 0:
                self._grad = _pack_dim0_reduce_scatter_input(
                    self._grad,
                    self.shard_world_size,
                )
            self._grad = self._grad.view(-1)

        apply_gradient_scaling_factor(self._grad, self.gradient_scaling_factor)

        shard_group = self.mesh_info.shard_process_group if isinstance(self.mesh_info, FSDPMeshInfo) else None
        if shard_group is None or self.shard_world_size <= 1:
            if output_buffer is not None:
                copy_without_bumping_version(output_buffer, self._grad)
                self.reduce_scatter_comm_ctx.reduce_scatter_output = output_buffer
            else:
                self.reduce_scatter_comm_ctx.reduce_scatter_output = self._grad
            self.reduce_scatter_comm_ctx.reduce_scatter_handle = None
            return

        # Calculate output size
        output_numel = self._grad.numel() // self.shard_world_size
        if output_buffer is not None:
            if output_buffer.numel() != output_numel:
                raise ValueError(
                    f"output_buffer size mismatch: expected {output_numel}, got {output_buffer.numel()}"
                )
            if output_buffer.dtype != self._grad.dtype:
                raise ValueError(
                    f"output_buffer dtype mismatch: expected {self._grad.dtype}, got {output_buffer.dtype}"
                )
            self.reduce_scatter_comm_ctx.reduce_scatter_output = output_buffer
        else:
            self.reduce_scatter_comm_ctx.reduce_scatter_output = ms.mint.empty(
                output_numel,
                dtype=self._grad.dtype,
                device=self._grad.device.split(":")[0],
            )

        # Execute reduce_scatter_tensor
        self.reduce_scatter_comm_ctx.reduce_scatter_handle = dist.reduce_scatter_tensor(
            self.reduce_scatter_comm_ctx.reduce_scatter_output,
            self._grad,
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
        reduce_dtype = self.reduce_comm_dtype(grad)
        if grad.dtype != reduce_dtype:
            grad = grad.to(reduce_dtype)
        reduce_group = (
            self.mesh_info.replicate_process_group
            if isinstance(self.mesh_info, DDPMeshInfo)
            else None
        )
        if reduce_group is None or self.replicate_world_size <= 1:
            self.all_reduce_comm_ctx.all_reduce_output = grad
            self.all_reduce_comm_ctx.all_reduce_handle = None
            return

        # Ascend HCCL accepts contiguous views but rejects non-contiguous input.
        if not grad.is_contiguous():
            grad = ms.mint.cat((grad,), dim=0)

        self.all_reduce_comm_ctx.all_reduce_output = grad
        self.all_reduce_comm_ctx.all_reduce_handle = dist.all_reduce(
            grad,
            op=reduce_op,
            group=reduce_group,
            async_op=async_op,
        )

    def all_reduce_source_replicate_grad_inplace(
        self,
        reduced_grad: ms.Tensor,
        reduce_op: str,
    ) -> None:
        """All-reduce a final gradient over replicated source-layout axes."""
        if self.source_shard_info is None or not self.source_shard_info.placements:
            return
        source_mesh = self.source_shard_info.mesh
        replicate_mesh_dims = tuple(
            mesh_dim
            for mesh_dim, placement in enumerate(self.source_shard_info.placements)
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
