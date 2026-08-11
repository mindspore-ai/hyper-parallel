# Copyright 2025-2026 Huawei Technologies Co., Ltd
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

# Adapted from https://github.com/pytorch/pytorch/blob/release/2.6/torch/distributed/fsdp/_fully_shard/_fsdp_param.py
# enhanced with fully_shard parameter management
# ============================================================================
"""HSDP parameter"""
# pylint: disable=W0212
from dataclasses import dataclass
from typing import Callable, List, Optional, cast

import torch
import torch.distributed as dist
from torch import nn
from torch._prims_common import make_contiguous_strides_for

from hyper_parallel.core.dtensor.device_mesh import DeviceMesh
from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.dtensor.layout import Layout
from hyper_parallel.core.dtensor.placement_types import Replicate, Shard, StridedShard
from hyper_parallel.core.fully_shard.hsdp_param import HSDPParamV2
from hyper_parallel.core.fully_shard.hsdp_utils import (
    ParamModuleInfo,
    ShardedState,
    apply_gradient_scaling_factor,
)
from hyper_parallel.core.fully_shard.utils import (
    CPUOffloadPolicy,
    DataParallelMeshInfo,
    DDPMeshInfo,
    FSDPMeshInfo,
    HSDPMeshInfo,
    MixedPrecisionPolicy,
    OffloadPolicy,
    TPShardMetaInfo,
)


def _copy_without_bumping_version(dst: torch.Tensor, src: torch.Tensor) -> None:
    """Copy into ``dst`` while preserving its autograd version counter."""
    # pylint: disable=W0212
    with torch.autograd._unsafe_preserve_version_counter(dst):
        dst.copy_(src)


@dataclass
class ReduceScatterCommCtx:
    """Per-parameter reduce-scatter output and asynchronous work."""

    reduce_scatter_output: Optional[torch.Tensor] = None
    reduce_scatter_handle: Optional[dist.Work] = None


@dataclass
class AllReduceCommCtx:
    """Per-parameter all-reduce output and asynchronous work."""

    all_reduce_output: Optional[torch.Tensor] = None
    all_reduce_handle: Optional[dist.Work] = None


@dataclass
class AllGatherCommCtx:
    """Per-parameter all-gather output and asynchronous work."""

    allgather_output: Optional[torch.Tensor] = None
    allgather_handle: Optional[dist.Work] = None


class ParameterHookMigrator:
    """Preserve parameter backward hooks across HSDP parameter replacement."""

    def __init__(self) -> None:
        self._orig_param_hooks: List[Callable] = []
        self._saved_hook_ids: set[int] = set()

    def _save_backward_hooks(self, param: nn.Parameter) -> None:
        """Save backward hooks from a parameter, deduplicated by hook identity."""
        if not hasattr(param, "_backward_hooks") or param._backward_hooks is None:
            return

        for _, hook_func in param._backward_hooks.items():
            hook_func_id = id(hook_func)
            if hook_func_id not in self._saved_hook_ids:
                self._orig_param_hooks.append(hook_func)
                self._saved_hook_ids.add(hook_func_id)

    def _migrate_backward_hooks(self, new_param: nn.Parameter) -> None:
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


class TorchHSDPParamV2(HSDPParamV2):
    """
    Torch HSDP parameter.
    """

    def __init__(
        self,
        param: nn.Parameter,
        module_info: ParamModuleInfo,
        mesh_info: DataParallelMeshInfo,
        shard_placement_fn: Optional[Callable[[nn.Parameter], Optional[Shard]]] = None,
        mp_policy: Optional[MixedPrecisionPolicy] = None,
        offload_policy: Optional[OffloadPolicy] = None,
        device: Optional[torch.device] = None,
        tp_grad_info: Optional[TPShardMetaInfo] = None,
    ):
        """
        Initialize TorchHSDPParamV2 and shard the parameter.

        Args:
            param (nn.Parameter): The original full parameter to shard.
            module_info (ParamModuleInfo): Ownership and shared-weight metadata.
            mesh_info (DataParallelMeshInfo): Mesh topology for shard/replicate dimensions.
            shard_placement_fn (Callable, optional): Returns a Shard placement for the parameter,
                or None to use default (Shard(0)).
            mp_policy (MixedPrecisionPolicy, optional): Mixed precision dtype policy.
            offload_policy (OffloadPolicy, optional): CPU offload policy.
            device (torch.device, optional): Target device for the sharded parameter.
            tp_grad_info (TPShardMetaInfo, optional): Source TP/EP layout metadata, built by
                the owning ``HSDPState``. Must be supplied (with ``origin_is_dtensor=True``)
                whenever ``param`` is a native DTensor, and omitted otherwise.

        Raises:
            ValueError: If ``tp_grad_info.origin_is_dtensor`` disagrees with whether
                ``param`` is a native DTensor.
        """
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
        # ``tp_grad_info`` is built and validated by the owning state
        # (``_build_param_tp_grad_info``): for a native DTensor parameter it always
        # describes that parameter's own mesh/placements. Only the agreement
        # between the two is re-checked here, because it is the invariant the
        # sharding math below depends on.
        if isinstance(param, DTensor) != (
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
        self.unsharded_accumulated_grad = None
        self._param_fqn: Optional[str] = None
        self.unsharded_param_buffers: List[torch.Tensor] = []
        self.allgather_comm_ctx = AllGatherCommCtx()
        self._post_load_hook_handle = (
            module_info.module.register_load_state_dict_post_hook(
                lambda *args, **kwargs: self.reset_sharded_param()
            )
        )
        self.reduce_scatter_comm_ctx = ReduceScatterCommCtx()
        self.all_reduce_comm_ctx = AllReduceCommCtx()
        self._parameter_hook_migrator._save_backward_hooks(param)
        self._grad = None
        # Keep reduce-scatter accumulation in reduce_dtype until the final
        # micro-step performs the optional replicate all-reduce.
        self._reduce_partial_output = None
        self.gradient_scaling_factor = None

    def _get_base_spmd_placements(self) -> tuple:
        if self.tp_grad_info is not None:
            # Preserve the source distributed layout and prefix the explicit
            # DP/FSDP mesh dimensions on the unified mesh.
            self._spmd_mesh = DeviceMesh.concatenate([self.mesh_info.mesh, self.tp_grad_info.mesh])
            dp_prefix_placements = tuple(Replicate() for _ in range(self.mesh_info.mesh.ndim))
            return dp_prefix_placements + tuple(self.tp_grad_info.placements)

        self._spmd_mesh = self.mesh_info.mesh
        return tuple(Replicate() for _ in range(self._spmd_mesh.ndim))

    def _apply_data_parallel_placements(self, placements: list, shard_placement: Shard) -> tuple:
        if len(placements) != self._spmd_mesh.ndim:
            raise AssertionError(
                f"Expected {self._spmd_mesh.ndim} unified placements, got {len(placements)}: {placements}"
            )
        if (
            isinstance(self.mesh_info, DDPMeshInfo)
            and self._spmd_replicate_mesh_dim is not None
            and not self._orig_param_is_dtensor
        ):
            placements[self._spmd_replicate_mesh_dim] = Replicate()
        if (
            isinstance(self.mesh_info, FSDPMeshInfo)
            and self._spmd_shard_mesh_dim is not None
        ):
            # If TP/EP already shards the same tensor dimension, fully_shard must
            # use StridedShard so the unified placement preserves the intended
            # shard order on the concatenated mesh.
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

    @property
    def reduce_partial_output(self) -> Optional[torch.Tensor]:
        """Return reduce-scatter results accumulated before the final micro-step."""
        return self._reduce_partial_output

    @reduce_partial_output.setter
    def reduce_partial_output(self, value: Optional[torch.Tensor]) -> None:
        self._reduce_partial_output = value

    def reduce_comm_dtype(self, grad: Optional[torch.Tensor] = None) -> torch.dtype:
        """Resolve the communication dtype owned by this parameter.

        Args:
            grad: Optional gradient used when no mixed-precision reduction dtype
                is configured.

        Returns:
            The dtype used by reduce-scatter and all-reduce buffers.
        """
        if self.reduce_dtype is not None:
            return self.reduce_dtype
        if grad is not None:
            return grad.dtype
        if self.unsharded_accumulated_grad is not None:
            return self.unsharded_accumulated_grad_data.dtype
        if self.unsharded_param.grad is not None:
            return self.unsharded_grad_data.dtype
        return self.orig_dtype

    def reduce_scatter_output(self) -> Optional[torch.Tensor]:
        """
        Get the reduce-scatter output tensor and wait for asynchronous operation to complete.

        Returns:
            torch.Tensor: The sharded gradient tensor after reduce-scatter operation.
        """
        if self.reduce_scatter_comm_ctx.reduce_scatter_handle is not None:
            self.reduce_scatter_comm_ctx.reduce_scatter_handle.wait()
            self._grad.untyped_storage().resize_(0)
            self._grad = None
            self.reduce_scatter_comm_ctx.reduce_scatter_handle = None
        return self.reduce_scatter_comm_ctx.reduce_scatter_output

    def clear_reduce_scatter_output(self) -> None:
        """Clear the reduce-scatter output tensor to free memory."""
        self.reduce_scatter_comm_ctx.reduce_scatter_output = None
        self._grad = None

    def all_reduce_output(self) -> Optional[torch.Tensor]:
        """
        Get the all-reduce output tensor and wait for asynchronous operation to complete.

        Returns:
            torch.Tensor: The reduced gradient tensor after all-reduce operation.
        """
        if self.all_reduce_comm_ctx.all_reduce_handle is not None:
            self.all_reduce_comm_ctx.all_reduce_handle.wait()
            self.all_reduce_comm_ctx.all_reduce_handle = None
        return self.all_reduce_comm_ctx.all_reduce_output

    def clear_all_reduce_output(self) -> None:
        """Clear the all-reduce output tensor to free memory."""
        self.all_reduce_comm_ctx.all_reduce_output = None

    def apply_reduced_grad(self, reduced_grad: torch.Tensor) -> bool:
        """
        Apply reduced gradient to the sharded parameter.

        Reshapes ``reduced_grad`` to match the local shard, optionally
        offloads to CPU, then accumulates or assigns onto
        ``hsdp_param.sharded_param.grad``.

        Args:
            reduced_grad (torch.Tensor): Gradient after reduce-scatter
                and/or all-reduce.

        Returns:
            Whether the current stream must synchronize for CPU offload.

        Note:
            Gradient scaling (``gradient_scaling_factor``) is applied earlier on
            the reduce input (see ``reduce_scatter_grad`` /
            ``_build_reduce_scatter_buckets``), never here, so accumulation stays
            ``sum_i(g_i * factor)`` rather than re-scaling an already-accumulated
            gradient.
        """
        sharded_grad = None
        if not self.mp_policy.apply_grad_on_fp32_main_grad:
            sharded_grad = self.sharded_param.grad
        else:
            if not hasattr(self.sharded_param, "main_grad"):
                self.sharded_param.main_grad = None
            sharded_grad = self.sharded_param.main_grad
        sharded_param_local_shape = (
            self.sharded_param.local_shape
            if isinstance(self.sharded_param, DTensor)
            else self.sharded_param.shape
        )
        reduced_grad = (
            reduced_grad.reshape(-1)
            .narrow(0, 0, self.sharded_size.numel())
            .view(sharded_param_local_shape)
        )
        if (
            not self.mp_policy.apply_grad_on_fp32_main_grad
            and reduced_grad.dtype != self.orig_dtype
        ):
            reduced_grad = reduced_grad.to(self.orig_dtype)
        to_accumulate_grad = sharded_grad is not None
        need_synchronize = False
        if self.offload_to_cpu:
            non_blocking = self.pin_memory and not to_accumulate_grad
            reduced_grad = reduced_grad.to(
                torch.device("cpu"), non_blocking=non_blocking
            )
            need_synchronize = True
        if sharded_grad is None:
            if not self.mp_policy.apply_grad_on_fp32_main_grad:
                self.sharded_param.grad = self.to_sharded_dtensor(reduced_grad)
            else:
                self.sharded_param.main_grad = self.to_sharded_dtensor(reduced_grad)
                self.sharded_param.grad = None
        else:
            if not self.mp_policy.apply_grad_on_fp32_main_grad:
                self.sharded_param.grad._local_tensor += reduced_grad
            else:
                self.sharded_param.main_grad._local_tensor += reduced_grad
                self.sharded_param.grad = None
        if self.unsharded_accumulated_grad_data is not None:
            self.unsharded_accumulated_grad = None
        elif self.unsharded_param.grad is not None:
            self.unsharded_param.grad = None
        return need_synchronize

    @torch.no_grad()
    def _init_sharded_param(
        self,
        param: nn.Parameter,
        shard_placement_fn: Optional[Callable],
    ) -> None:
        if param.device != self.device and param.device.type != "meta":
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
        param_data = param.to_local() if self._orig_param_is_dtensor else param

        shard_dim = hsdp_placement.dim
        if param_data.ndim == 0:
            raise ValueError("fully_shard does not support scalar parameters")
        if shard_dim < 0 or shard_dim >= param_data.ndim:
            raise ValueError(
                f"Invalid fully_shard dim {shard_dim} for parameter "
                f"{self._module_info.param_name} with shape {tuple(param_data.shape)}"
            )
        self._orig_size = param_data.size()
        self._contiguous_orig_stride = make_contiguous_strides_for(self._orig_size)
        self._logical_global_size = param.size()
        if isinstance(param, DTensor) and param.layout.tensor_stride is not None:
            self._logical_global_stride = param.layout.tensor_stride
        else:
            self._logical_global_stride = make_contiguous_strides_for(self._logical_global_size)

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

        if shard_dim != 0 and param_data.size(shard_dim) % self.shard_world_size != 0:
            raise NotImplementedError(
                f"fully_shard only supports uneven sharding on dim=0, but parameter "
                f"{self._module_info.param_name} has shape {tuple(param_data.shape)}, "
                f"shard dim {shard_dim}, and world size {self.shard_world_size}"
            )
        dim_shard_size = (param_data.size(shard_dim) + self.shard_world_size - 1) // self.shard_world_size
        actual_shard_offset = min(self.shard_rank * dim_shard_size, param_data.size(shard_dim))
        actual_shard_length = min(dim_shard_size, param_data.size(shard_dim) - actual_shard_offset)
        sharded_param = param_data.narrow(
            shard_dim,
            actual_shard_offset,
            actual_shard_length,
        ).clone().contiguous()
        self.sharded_size = sharded_param.size()
        self.contiguous_sharded_stride = make_contiguous_strides_for(self.sharded_size)
        padded_sharded_size = list(param_data.size())
        padded_sharded_size[shard_dim] = dim_shard_size
        self.padded_sharded_param_size = torch.Size(padded_sharded_size)
        if self.offload_to_cpu and not sharded_param.is_meta:
            sharded_param = sharded_param.cpu()
            if self.pin_memory:
                sharded_param = sharded_param.pin_memory()

        if self.sharded_size == self.padded_sharded_param_size:
            self._sharded_param_data = sharded_param.view(-1)
        else:
            padded_sharded_param = sharded_param.new_zeros(self.padded_sharded_param_size)
            if self.pin_memory and not padded_sharded_param.is_meta:
                padded_sharded_param = padded_sharded_param.pin_memory()
            if sharded_param.numel() > 0:
                padded_sharded_param.narrow(
                    shard_dim,
                    0,
                    actual_shard_length,
                ).copy_(sharded_param)
            self._sharded_param_data = padded_sharded_param.view(-1)
            sharded_param = padded_sharded_param.narrow(
                shard_dim,
                0,
                actual_shard_length,
            )

        self._sharding_spec = Layout.from_device_mesh(self._spmd_mesh)
        self._sharding_spec.set_placements(self._spmd_placements)
        self._sharding_spec.placement_to_tensor_map(param.ndim)
        self._sharding_spec.set_tensor_meta(
            self._logical_global_size,
            self._logical_global_stride,
            param_data.dtype,
        )

        self.sharded_param = nn.Parameter(self.to_sharded_dtensor(sharded_param))
        self.sharded_param._layout = self._sharding_spec
        self.sharded_param._placements = tuple(self._sharding_spec.placements)
        self.sharded_param.requires_grad_(param.requires_grad)
        self._setattr_on_modules(self.sharded_param)
        # after init, self.sharded_param replaces original param, gradients must accumulate to this Parameter's grad
        self.sharded_param._hsdp_param_initialized = True
        self.sharded_state = ShardedState.SHARDED
        self.param_dtype = None
        self.reduce_dtype = None

    def init_dtype_attrs(self, mp_policy: MixedPrecisionPolicy) -> None:
        """Initialize param_dtype and reduce_dtype from the mixed precision policy."""
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
        all_gather_input_dtypes: list[torch.dtype],
        world_size: int,
        device: torch.device,
        force_recreate: bool = False,
    ):
        """
        Allocate buffers that hold unsharded parameter data.

        Args:
            all_gather_input_numels: Number of elements per input shard.
            all_gather_input_dtypes: Dtype of each input shard.
            world_size: Number of ranks in the shard process group.
            device: Device on which to allocate the output buffers.
            force_recreate: If True, always recreate buffers even if already initialized.
        """
        if not force_recreate and len(self.unsharded_param_buffers) > 0:
            return  # already initialized
        if force_recreate and hasattr(self, "_unsharded_param"):
            raise RuntimeError(
                "Cannot recreate unsharded_param_buffers after initializing the stable "
                "unsharded parameter."
            )
        self.unsharded_param_buffers = [
            torch.empty(torch.Size([numel * world_size]), dtype=dtype, device=device)
            for numel, dtype in zip(all_gather_input_numels, all_gather_input_dtypes)
        ]

    def init_unsharded_param(self) -> None:
        """Initialize the stable unsharded parameter from its final output storage."""
        if len(self.unsharded_param_buffers) != 1:
            raise AssertionError(
                f"Expected 1 unsharded_param_buffer, got {len(self.unsharded_param_buffers)}"
            )

        if self.allgather_comm_ctx.allgather_output is not None:
            packed_shape = list(self.sharded_size)
            packed_shape[0] *= self.shard_world_size
            chunks = torch.chunk(
                self.allgather_comm_ctx.allgather_output.view(packed_shape),
                self.shard_world_size,
                dim=0,
            )
            # pylint: disable=W0212
            with torch.autograd._unsafe_preserve_version_counter(
                self.unsharded_param_buffers[0]
            ):
                torch.cat(
                    chunks,
                    dim=self.hsdp_placement.dim,
                    out=self.unsharded_param_buffers[0].view(self._orig_size),
                )
            self.allgather_comm_ctx.allgather_output.untyped_storage().resize_(0)
            self.allgather_comm_ctx.allgather_output = None

        if hasattr(self, "_unsharded_param"):
            # Keep one stable ``_unsharded_param`` object across unshard cycles:
            # autograd-facing module state captured during forward must still be
            # the same object in backward. Only its storage is refreshed above,
            # which also avoids reusing stale weights after ``optimizer.step()``
            # mutates the sharded local shard alone (the non-dim-0 unpack path
            # materializes a contiguous copy, so a stale ``.data`` would not see
            # the update).
            return

        unsharded_param = torch.as_strided(
            self.unsharded_param_buffers[0],
            size=self._orig_size,
            stride=self._contiguous_orig_stride,
            storage_offset=0,
        )
        if self.tp_grad_info is not None and self.tp_grad_info.origin_is_dtensor:
            unsharded_param = DTensor.from_local(
                unsharded_param,
                self.tp_grad_info.mesh,
                self.tp_grad_info.placements,
            )
        self._unsharded_param = nn.Parameter(
            unsharded_param,
            requires_grad=self.sharded_param.requires_grad,
        )

    def to_sharded(self) -> None:
        self._setattr_on_modules(self.sharded_param)
        if self.unsharded_param_buffers[0] is not self._sharded_param_data:
            self.free_unsharded_param()
        self.sharded_state = ShardedState.SHARDED

    def to_unsharded(self) -> None:
        set_requires_grad_if_needed(self.sharded_param, self._unsharded_param)
        self._setattr_on_modules(self._unsharded_param)
        self.sharded_state = ShardedState.UNSHARDED

    def _setattr_on_modules(self, param: nn.Parameter) -> None:
        """Set parameter on module and shared modules, preserving pointer consistency."""
        if getattr(self._module_info.module.__setattr__, "__func__", None) is nn.Module.__setattr__:
            # fast path
            self._module_info.module._parameters[self._module_info.param_name] = param
        else:
            # slow path
            setattr(self._module_info.module, self._module_info.param_name, param)
        self._parameter_hook_migrator._save_backward_hooks(self.sharded_param)
        self._parameter_hook_migrator._migrate_backward_hooks(param)
        # Iterate through all modules that share this parameter to prevent pointer desync.
        for shared_module, shared_param_name in zip(
            self._module_info.shared_modules, self._module_info.shared_param_names
        ):
            if getattr(shared_module.__setattr__, "__func__", None) is nn.Module.__setattr__:
                shared_module._parameters[shared_param_name] = param
            else:
                setattr(shared_module, shared_param_name, param)

    def to_sharded_dtensor(self, tensor: torch.Tensor) -> DTensor:
        """
        Converts a local tensor representing either the sharded parameter or
        sharded gradient to DTensor.
        """
        sharded_dtensor = DTensor.from_local(
            tensor,
            self._sharding_spec.mesh,
            self._sharding_spec.placements,
        )
        sharded_dtensor._layout = self._sharding_spec
        sharded_dtensor._placements = tuple(self._sharding_spec.placements)
        return sharded_dtensor

    def to_accumulated_grad_if_needed(self) -> None:
        if self._unsharded_param.grad is None:
            return
        # Keep local gradients alive across no-sync / delayed-sync steps even
        # after the parameter transitions back to the sharded view.
        unsharded_grad = self._unsharded_param.grad
        self._unsharded_param.grad = None
        if self.reduce_dtype is not None and unsharded_grad.dtype != self.reduce_dtype:
            unsharded_grad = unsharded_grad.to(self.reduce_dtype)
        if self.unsharded_accumulated_grad is None:
            self.unsharded_accumulated_grad = unsharded_grad
        else:
            self.unsharded_accumulated_grad += unsharded_grad

    def accumulate_unsharded_grad_if_needed(self) -> None:
        if (
            self.unsharded_accumulated_grad is not None
            and self.unsharded_param.grad is not None
        ):
            grad = self.unsharded_param.grad
            if self.reduce_dtype is not None and grad.dtype != self.reduce_dtype:
                grad = grad.to(self.reduce_dtype)
            self.unsharded_param.grad = None
            self.unsharded_accumulated_grad += grad

    def alloc_unsharded_param_buffers(self) -> None:
        """
        Restore unsharded parameter buffers to their full capacity.
        unsharded_param_buffer is the final storage which should be reffereced by self._unsharded_param
        """
        for tensor in self.unsharded_param_buffers:
            expected_size = tensor.numel() * tensor.itemsize
            storage = tensor.untyped_storage()
            if storage.size() != expected_size:
                storage.resize_(expected_size)

    def free_unsharded_param(self) -> None:
        """Release storage of the unsharded parameter buffers."""
        for tensor in self.unsharded_param_buffers:
            storage = tensor.untyped_storage()
            if storage.size() != 0:
                storage.resize_(0)

    @property
    def all_gather_inputs(self) -> list[torch.Tensor]:
        """Return the local sharded tensor to use as input for all-gather, applying dtype cast if needed."""
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
    def unsharded_param(self) -> nn.Parameter:
        """Return the full unsharded parameter after all-gather."""
        return self._unsharded_param

    @property
    def unsharded_grad_data(self) -> torch.Tensor:
        """
        Get the unsharded gradient data as a local tensor.
        """
        return self._unsharded_param.grad

    @property
    def unsharded_accumulated_grad_data(self) -> torch.Tensor:
        """
        Get the unsharded accumulated gradient data as a local tensor.
        """
        return self.unsharded_accumulated_grad

    def _assert_in_states(self, *states: ShardedState) -> None:
        """Assert current state is one of expected states."""
        if self.sharded_state not in states:
            raise AssertionError(
                f"Expected sharded_state in {states}, got {self.sharded_state}"
            )

    def _resolve_reset_param(self):
        """Resolve the (possibly swapped) module param for ``reset_sharded_param``.

        Refreshes ``self.sharded_param`` for the DTensor case and returns the
        current module parameter for the caller to re-shard.
        """
        module_info = self._module_info
        new_param = getattr(module_info.module, module_info.param_name)
        if new_param is self.sharded_param:
            return new_param
        # Ensure object identity is preserved after parameter conversion.
        if torch.__future__.get_swap_module_params_on_conversion():
            raise AssertionError(
                f"Expects swap_tensors to preserve object but got {new_param} "
                f"instead of {self.sharded_param}"
            )
        if isinstance(new_param, DTensor):
            self.sharded_param = new_param
            if not getattr(self.sharded_param, "_hsdp_param_initialized", None):
                # reset _hsdp_param_initialized flag.
                self.sharded_param._hsdp_param_initialized = True
        # If new_param is a plain Tensor, keep the existing 'self.sharded_param' ref;
        # only its _local_tensor / _sharded_param_data are refreshed below.
        return new_param

    def reset_sharded_param(self) -> None:
        """Reset sharded param after load_state_dict."""
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
        same_local_tensor = False
        if isinstance(self._sharded_param_data, torch.Tensor):
            same_local_tensor = (
                # when sharding param with shape (1, ...) over 2 ranks
                # local_tensor on rank 1 can be size 0, data_ptr() can be 0
                self._sharded_param_data.untyped_storage().data_ptr() > 0
                and self._sharded_param_data.untyped_storage().data_ptr()
                == local_tensor.untyped_storage().data_ptr()
            )
        shard_dim = self.hsdp_placement.dim
        if not same_local_tensor:
            if local_tensor.size() != self.sharded_size:
                raise AssertionError(
                    f"Expected sharded_size to be {self.sharded_size}, got {local_tensor.size()}"
                )
        pinned_local_tensor = False
        if self.pin_memory and not local_tensor.is_pinned():
            local_tensor = local_tensor.cpu().pin_memory()
            pinned_local_tensor = True
        if not isinstance(self.sharded_param, DTensor):
            raise AssertionError(f"Expected DTensor, got {type(self.sharded_param)}")
        if not same_local_tensor or pinned_local_tensor:
            actual_shard_length = self.sharded_size[shard_dim]
            if self.sharded_size == self.padded_sharded_param_size:
                local_tensor = local_tensor.contiguous()
                self._sharded_param_data = local_tensor.view(-1)
                local_view = local_tensor.detach()
            else:
                padded_local_tensor = local_tensor.new_zeros(self.padded_sharded_param_size)
                if self.pin_memory:
                    padded_local_tensor = padded_local_tensor.pin_memory()
                if local_tensor.numel() > 0:
                    padded_local_tensor.narrow(
                        shard_dim,
                        0,
                        actual_shard_length,
                    ).copy_(local_tensor)
                self._sharded_param_data = padded_local_tensor.view(-1)
                local_view = padded_local_tensor.narrow(
                    shard_dim,
                    0,
                    actual_shard_length,
                ).detach()
            set_requires_grad_if_needed(self.sharded_param, local_view)
            self.sharded_param._local_tensor = local_view
            if not self.sharded_param._local_tensor.is_contiguous():
                raise AssertionError(
                    "Expected sharded_param._local_tensor to be contiguous"
                )
        self._sharding_spec.set_tensor_meta(
            self._logical_global_size,
            self._logical_global_stride,
            local_tensor.dtype,
        )
        self.sharded_param._layout = self._sharding_spec
        self.sharded_param._placements = tuple(self._sharding_spec.placements)
        # After ``to_empty`` replaces the module parameter with a plain tensor,
        # re-install the DTensor ``nn.Parameter`` so the optimizer and forward
        # hooks see the correct object.  Idempotent when the module already
        # holds ``self.sharded_param`` (same data_ptr → no-op in practice).
        self._setattr_on_modules(self.sharded_param)

    @torch.no_grad()
    def _get_unsharded_param_data(self, async_op: bool = False) -> None:
        """
        Perform all-gather to get unsharded parameter data.

        Args:
            async_op: Whether to execute asynchronously.

        The output buffer and optional asynchronous handle are stored in the
        parameter communication context.
        """
        all_gather_input = self.all_gather_inputs[0]

        if self.shard_world_size <= 1 or self.mesh_info.shard_process_group is None:
            if len(self.unsharded_param_buffers) == 0:
                self.unsharded_param_buffers = [all_gather_input]
            elif self.unsharded_param_buffers[0] is not all_gather_input:
                # if param_dtype cast or tensor.to caused by cpu_offload
                self.alloc_unsharded_param_buffers()
                _copy_without_bumping_version(self.unsharded_param_buffers[0], all_gather_input)
            self.allgather_comm_ctx.allgather_output = None
            self.allgather_comm_ctx.allgather_handle = None
            return

        self.init_unsharded_param_buffers(
            all_gather_input_numels=[all_gather_input.numel()],
            all_gather_input_dtypes=[all_gather_input.dtype],
            world_size=self.shard_world_size,
            device=self.device,
        )
        self.alloc_unsharded_param_buffers()

        self.allgather_comm_ctx.allgather_output = self.unsharded_param_buffers[0]
        if self.hsdp_placement.dim != 0:
            # Non-dim-0 sharding uses an extra all-gather buffer before
            # restoring the original dimension with chunk + cat.
            self.allgather_comm_ctx.allgather_output = torch.empty_like(
                self.unsharded_param_buffers[0]
            )
        # pylint: disable=W0212
        with torch.autograd._unsafe_preserve_version_counter(
            self.allgather_comm_ctx.allgather_output
        ):
            self.allgather_comm_ctx.allgather_handle = dist.all_gather_into_tensor(
                self.allgather_comm_ctx.allgather_output,
                all_gather_input,
                group=self.mesh_info.shard_process_group,
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
        reduce_op: Optional[dist.ReduceOp] = dist.ReduceOp.AVG,
        output_buffer: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Perform reduce-scatter on gradient to reduce and shard the full gradient.

        Args:
            async_op: Whether to execute asynchronously.
            reduce_op: do reduce-scatter avg or sum.
            output_buffer: Optional pre-allocated output buffer for fused all-reduce.
                          When provided, reduce_scatter writes directly into this buffer,
                          enabling zero-copy fusion with subsequent all_reduce operations.
                          The buffer must have ``padded_sharded_param_size.numel()`` elements
                          and the reduction dtype.

        The output and optional asynchronous work are stored in
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
            grad_chunks = torch.chunk(self._grad, self.shard_world_size, dim=shard_dim)
            self._grad = torch.cat(grad_chunks, dim=0).contiguous().view(-1)
        else:
            padded_unsharded_dim0 = self.padded_sharded_param_size[0] * self.shard_world_size
            if self._grad.size(0) != padded_unsharded_dim0:
                padded_unsharded_size = torch.Size((padded_unsharded_dim0, *self._grad.shape[1:]))
                padded_grad = self._grad.new_zeros(padded_unsharded_size)
                padded_grad.narrow(0, 0, self._grad.size(0)).copy_(self._grad)
                self._grad = padded_grad
            self._grad = self._grad.view(-1)
        apply_gradient_scaling_factor(self._grad, self.gradient_scaling_factor)

        shard_process_group = self.mesh_info.shard_process_group if isinstance(self.mesh_info, FSDPMeshInfo) else None
        if shard_process_group is None or self.shard_world_size <= 1:
            if output_buffer is not None:
                output_buffer.copy_(self._grad)
                self.reduce_scatter_comm_ctx.reduce_scatter_output = output_buffer
            else:
                self.reduce_scatter_comm_ctx.reduce_scatter_output = self._grad
            self.reduce_scatter_comm_ctx.reduce_scatter_handle = None
            return

        # Calculate output size
        output_numel = self._grad.numel() // self.shard_world_size
        # Use provided output buffer or allocate a new one
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
            self.reduce_scatter_comm_ctx.reduce_scatter_output = torch.empty(
                output_numel,
                dtype=self._grad.dtype,
                device=self._grad.device,
            )
        # Execute reduce_scatter_tensor
        self.reduce_scatter_comm_ctx.reduce_scatter_handle = dist.reduce_scatter_tensor(
            self.reduce_scatter_comm_ctx.reduce_scatter_output,
            self._grad,
            op=reduce_op,
            group=shard_process_group,
            async_op=async_op,
        )

    def all_reduce_grad(
        self,
        async_op: bool = True,
        reduce_op: Optional[dist.ReduceOp] = dist.ReduceOp.AVG,
    ) -> None:
        """
        All-reduce the current reduce-scatter output across the replicate mesh.

        Args:
            async_op: Whether to execute asynchronously.
            reduce_op: Reduction operation for the replicate dimension.

        The output and optional asynchronous work are stored in
        ``all_reduce_comm_ctx``.
        """
        grad = self.reduce_scatter_comm_ctx.reduce_scatter_output
        if grad is None:
            raise RuntimeError("all_reduce_grad requires a completed reduce-scatter output.")
        reduce_dtype = self.reduce_comm_dtype(grad)
        if grad.dtype != reduce_dtype:
            grad = grad.to(reduce_dtype)

        replicate_process_group = (
            self.mesh_info.replicate_process_group if isinstance(self.mesh_info, DDPMeshInfo) else None
        )
        if replicate_process_group is None or self.replicate_world_size <= 1:
            self.all_reduce_comm_ctx.all_reduce_output = grad
            self.all_reduce_comm_ctx.all_reduce_handle = None
            return

        self.all_reduce_comm_ctx.all_reduce_handle = dist.all_reduce(
            grad,
            op=reduce_op,
            group=replicate_process_group,
            async_op=async_op,
        )
        self.all_reduce_comm_ctx.all_reduce_output = grad

    def all_reduce_tp_replicate_grad_inplace(
        self,
        reduced_grad: torch.Tensor,
        reduce_op: dist.ReduceOp,
    ) -> None:
        """All-reduce a final gradient over replicated source-layout axes, in place.

        ``reduced_grad`` is modified in place; the caller keeps using its own
        reference and must not expect a returned tensor. No-op when the source
        layout has no replicated mesh axis, so callers may invoke it
        unconditionally.

        Args:
            reduced_grad: Final local gradient after FSDP/HSDP reduction.
            reduce_op: Reduction operation shared with the DP communication path.
        """
        if self.tp_grad_info is None or not self.tp_grad_info.placements:
            return
        source_mesh = self.tp_grad_info.mesh
        source_placements = self.tp_grad_info.placements
        replicate_mesh_dims = tuple(
            mesh_dim
            for mesh_dim, placement in enumerate(source_placements)
            if placement.is_replicate()
        )
        if not replicate_mesh_dims:
            return
        mesh_dim_names = source_mesh.mesh_dim_names
        if mesh_dim_names is None:
            raise ValueError(
                "TP shard mesh must define mesh_dim_names to all-reduce replicated gradients."
            )
        replicate_mesh_dim_names = tuple(mesh_dim_names[mesh_dim] for mesh_dim in replicate_mesh_dims)
        replicate_mesh = source_mesh[replicate_mesh_dim_names].flatten()
        if replicate_mesh.size() <= 1:
            return
        dist.all_reduce(
            reduced_grad,
            op=reduce_op,
            group=replicate_mesh.get_group(),
            async_op=False,
        )


def set_requires_grad_if_needed(
    src_tensor: torch.Tensor, dst_tensor: torch.Tensor
) -> None:
    """set dst_tensor requires_grads from src_tensor if needed."""
    if src_tensor.requires_grad != dst_tensor.requires_grad:
        dst_tensor.requires_grad_(src_tensor.requires_grad)
