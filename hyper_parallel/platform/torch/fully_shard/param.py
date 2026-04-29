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
import itertools
from typing import Callable, List, Optional, Tuple, Union, cast

import torch
import torch.distributed as dist
from torch import nn
from torch._prims_common import make_contiguous_strides_for

from hyper_parallel.core.dtensor.device_mesh import DeviceMesh
from hyper_parallel.core.dtensor.dtensor import DTensor, SkipDTensorDispatch
from hyper_parallel.core.dtensor.layout import Layout
from hyper_parallel.core.dtensor.placement_types import Replicate, Shard, StridedShard
from hyper_parallel.core.fully_shard.hsdp_param import HSDPParamV2
from hyper_parallel.core.fully_shard.hsdp_utils import (
    FullyShardParamMode,
    GroupInfo,
    ParamModuleInfo,
    ShardedState,
    get_rank_list_for_axes,
    get_split_rank_lists_for_axes,
)
from hyper_parallel.core.fully_shard.utils import (
    CPUOffloadPolicy,
    DDPMeshInfo,
    FSDPMeshInfo,
    MixedPrecisionPolicy,
    OffloadPolicy,
)
from hyper_parallel.platform import get_platform
from hyper_parallel.platform.torch.fully_shard.pack_utils import (
    build_rs_plan,
    pack_for_reduce_scatter,
    unpack_from_all_gather,
)

_GROUP_INFO_CACHE = {}
platform = get_platform()


def _copy_without_bumping_version(dst: torch.Tensor, src: torch.Tensor) -> None:
    """Copy into ``dst`` while preserving its autograd version counter."""
    # pylint: disable=W0212
    with torch.autograd._unsafe_preserve_version_counter(dst):
        dst.copy_(src)


def _build_group_info_from_rank_list(
    group_name: str,
    rank_list,
) -> GroupInfo:
    """Create group metadata from an explicit rank list."""
    normalized_rank_list = tuple(sorted(int(rank) for rank in rank_list))
    if len(normalized_rank_list) <= 1:
        return GroupInfo(f"{group_name}_invalid", None, 1)
    if normalized_rank_list in _GROUP_INFO_CACHE:
        cached_group = _GROUP_INFO_CACHE[normalized_rank_list]
        return GroupInfo(str(normalized_rank_list), cached_group, len(normalized_rank_list))
    try:
        group = platform.create_group(list(normalized_rank_list))
    except (RuntimeError, ValueError):  # pragma: no cover - UT may run without dist init
        group = None
    _GROUP_INFO_CACHE[normalized_rank_list] = group
    return GroupInfo(str(normalized_rank_list), group, len(normalized_rank_list))


def _build_group_info_from_process_group(
    group_name: str,
    process_group,
    rank_size: int,
) -> GroupInfo:
    """Create group metadata from an existing process group."""
    if process_group is None or rank_size <= 1:
        return GroupInfo(f"{group_name}_invalid", None, 1)
    try:
        rank_list = dist.get_process_group_ranks(process_group)
        resolved_group_name = str(tuple(sorted(rank_list)))
    except (AssertionError, AttributeError, KeyError, RuntimeError, TypeError, ValueError):
        # pragma: no cover - best-effort naming / mocked process groups in UT
        resolved_group_name = group_name
    return GroupInfo(resolved_group_name, process_group, rank_size)


class TorchHSDPParamV2(HSDPParamV2):
    """
    Torch HSDP parameter.
    """

    def __init__(
        self,
        param: nn.Parameter,
        module_info: ParamModuleInfo,
        mesh_info: FSDPMeshInfo,
        shard_placement_fn: Optional[Callable[[nn.Parameter], Optional[Shard]]] = None,
        mp_policy: Optional[MixedPrecisionPolicy] = None,
        offload_policy: Optional[OffloadPolicy] = None,
        device: Optional[torch.device] = None,
        param_mode: Optional[FullyShardParamMode] = None,
        enable_fsdp_shard: bool = True,
    ):
        """
        Initialize TorchHSDPParamV2 and shard the parameter.

        Args:
            param (nn.Parameter): The original full parameter to shard.
            module_info (ParamModuleInfo): Ownership and shared-weight metadata.
            mesh_info (FSDPMeshInfo): Mesh topology for shard/replicate dimensions.
            shard_placement_fn (Callable, optional): Returns a Shard placement for the parameter,
                or None to use default (Shard(0)).
            mp_policy (MixedPrecisionPolicy, optional): Mixed precision dtype policy.
            offload_policy (OffloadPolicy, optional): CPU offload policy.
            device (torch.device, optional): Target device for the sharded parameter.
        """
        self._module_info: ParamModuleInfo = module_info
        self.mesh_info = mesh_info
        self.mp_policy = mp_policy
        self.device = device
        if param_mode is None:
            raise AssertionError("param_mode must be resolved before TorchHSDPParamV2 initialization.")
        self.param_mode = param_mode
        self.enable_fsdp_shard = enable_fsdp_shard
        self.orig_dtype = None
        self.param_dtype = None
        self.reduce_dtype = None
        self.offload_to_cpu: bool = isinstance(offload_policy, CPUOffloadPolicy)
        self.pin_memory = (
            self.offload_to_cpu and cast(CPUOffloadPolicy, offload_policy).pin_memory
        )
        self.grad_offload_event: Optional[torch.Event] = None
        self._orig_param_is_dtensor = isinstance(param, DTensor)
        self._orig_dtensor_mesh = param.device_mesh if self._orig_param_is_dtensor else None
        self._orig_dtensor_placements = tuple(param.placements) if self._orig_param_is_dtensor else None
        self._spmd_shard_mesh_dim = self.mesh_info.shard_mesh_dim
        self._spmd_replicate_mesh_dim = self.mesh_info.replicate_mesh_dim
        self._init_sharded_param(param, shard_placement_fn)
        self._init_group_infos()
        self.all_gather_outputs: List[torch.Tensor] = []
        self.unsharded_accumulated_grad = None
        self._param_fqn: Optional[str] = None
        # Communication attributes for prefetch pattern
        self.prefetch_handle: Optional[dist.Work] = None
        self._post_load_hook_handle = (
            module_info.module.register_load_state_dict_post_hook(
                lambda *args, **kwargs: self.reset_sharded_param()
            )
        )
        self._reduce_scatter_output = None
        self.reduce_scatter_handle = None
        self._all_reduce_output = None
        self.all_reduce_handle = None

    @property
    def uses_param_shard(self) -> bool:
        """Whether fully_shard should physically shard parameter storage for this param."""
        return self.enable_fsdp_shard

    @property
    def is_dtensor_compat_mode(self) -> bool:
        """Whether the parameter is managed through the DTensor compatibility path only."""
        return self.param_mode == FullyShardParamMode.DTENSOR_COMPAT

    def _get_base_spmd_placements(self) -> tuple:
        if self.param_mode == FullyShardParamMode.DTENSOR_UNIFIED and self._orig_param_is_dtensor:
            # DTENSOR_UNIFIED keeps the original distributed layout and prefixes
            # explicit DP/FSDP mesh dimensions ahead of it on the unified mesh.
            self._spmd_mesh = DeviceMesh.concatenate([self.mesh_info.mesh, self._orig_dtensor_mesh])
            dp_prefix_placements = tuple(Replicate() for _ in range(self.mesh_info.mesh.ndim))
            return dp_prefix_placements + tuple(self._orig_dtensor_placements)

        if self.is_dtensor_compat_mode and self._orig_param_is_dtensor:
            self._spmd_mesh = self._orig_dtensor_mesh
            return tuple(self._orig_dtensor_placements)

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
            self.uses_param_shard
            and isinstance(self.mesh_info, FSDPMeshInfo)
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

    def _init_group_infos(self) -> None:
        if self.uses_param_shard and self.is_sharded and isinstance(self.mesh_info, FSDPMeshInfo):
            self.sharded_group_info = _build_group_info_from_process_group(
                "fully_shard_sharded_group",
                self.mesh_info.shard_process_group,
                self.mesh_info.shard_mesh_size,
            )
        else:
            self.sharded_group_info = GroupInfo("fully_shard_sharded_group_invalid", None, 1)

        # The all-reduce group is always derived from the final materialized layout.
        # This keeps replicate_params, DTensor compat, and unified multi-dim layouts
        # on a single source of truth.
        self.unsharded_group_info = self._build_layout_driven_group_info()

        self.shard_size = self.sharded_group_info.rank_size
        self.dp_size = self.unsharded_group_info.rank_size
        self.rank_size = max(1, self.shard_size * self.dp_size)

    def _build_layout_driven_group_info(self):
        group_axes = [
            axis
            for axis, placement in enumerate(self._spmd_placements)
            if placement.is_replicate()
        ]
        if self.uses_param_shard and self._spmd_shard_mesh_dim is not None:
            group_axes = [axis for axis in group_axes if axis != self._spmd_shard_mesh_dim]
        if not group_axes:
            return GroupInfo("fully_shard_unsharded_group_invalid", None, 1)
        group_dim_names = getattr(self._spmd_mesh, "mesh_dim_names", None)
        if group_dim_names:
            try:
                mesh_axis_names = tuple(group_dim_names[axis] for axis in group_axes)
                if len(mesh_axis_names) == 1:
                    axis_name = mesh_axis_names[0]
                    process_group = self._spmd_mesh.get_group(axis_name)
                    if process_group is not None:
                        rank_size = self._spmd_mesh.mesh_shape[group_dim_names.index(axis_name)]
                        return _build_group_info_from_process_group(
                            "fully_shard_unsharded_group",
                            process_group,
                            rank_size,
                        )

                split_rank_lists = get_split_rank_lists_for_axes(self._spmd_mesh, group_axes)
                process_group = platform.split_group(split_ranks=split_rank_lists)
                if process_group is not None:
                    rank_size = 1
                    for axis in group_axes:
                        rank_size *= self._spmd_mesh.mesh_shape[axis]
                    return _build_group_info_from_process_group(
                        "fully_shard_unsharded_group",
                        process_group,
                        rank_size,
                    )
            except (
                AssertionError,
                AttributeError,
                KeyError,
                RuntimeError,
                TypeError,
                ValueError,
            ):
                # Fall back to the explicit rank-list path for mocked meshes in UT
                # or when a mesh implementation cannot materialize a reusable group.
                pass

        rank_list = get_rank_list_for_axes(self._spmd_mesh, group_axes)
        return _build_group_info_from_rank_list("fully_shard_unsharded_group", rank_list)

    def _to_local_unsharded_grad(self, grad):
        """Normalize a pending gradient to a local tensor expected by fully_shard collectives."""
        if not isinstance(grad, DTensor):
            return grad

        if any(placement.is_partial() for placement in grad.placements):
            grad = grad.reduce_partial()

        if (
            self._orig_dtensor_mesh is not None
            and grad.device_mesh.to_hash() != self._orig_dtensor_mesh.to_hash()
        ) or (
            self._orig_dtensor_placements is not None
            and tuple(grad.placements) != tuple(self._orig_dtensor_placements)
        ):
            grad = grad.redistribute(self._orig_dtensor_mesh, self._orig_dtensor_placements)
        return grad.to_local()

    def reduce_scatter_output(self):
        """
        Get the reduce-scatter output tensor and wait for asynchronous operation to complete.

        Returns:
            torch.Tensor: The sharded gradient tensor after reduce-scatter operation.
        """
        if self.reduce_scatter_handle is not None:
            self.reduce_scatter_handle.wait()
            self.reduce_scatter_handle = None
        return self._reduce_scatter_output

    def clear_reduce_scatter_output(self):
        """Clear the reduce-scatter output tensor to free memory."""
        self._reduce_scatter_output = None

    def all_reduce_output(self):
        """
        Get the all-reduce output tensor and wait for asynchronous operation to complete.

        Returns:
            torch.Tensor: The reduced gradient tensor after all-reduce operation.
        """
        if self.all_reduce_handle is not None:
            self.all_reduce_handle.wait()
            self.all_reduce_handle = None
        return self._all_reduce_output

    def clear_all_reduce_output(self):
        """Clear the all-reduce output tensor to free memory."""
        self._all_reduce_output = None

    def apply_reduced_grad(self, reduced_grad, param_type):
        """
        Apply reduced gradient to the sharded parameter.

        Reshapes ``reduced_grad`` to match the local shard, optionally
        offloads to CPU, then accumulates or assigns onto
        ``hsdp_param.sharded_param.grad``.

        Args:
            reduced_grad (torch.Tensor): Gradient after reduce-scatter
                and/or all-reduce.
            param_type (Optional[torch.dtype]): Target dtype for the gradient (if conversion is needed).
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
        reduced_grad = reduced_grad.view(sharded_param_local_shape)
        if (not self.mp_policy.apply_grad_on_fp32_main_grad and param_type is not None
                and reduced_grad.dtype != param_type):
            reduced_grad = reduced_grad.to(param_type)
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
            with SkipDTensorDispatch():
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
        self._orig_size = param_data.size()
        self._contiguous_orig_stride = make_contiguous_strides_for(self._orig_size)

        if self.uses_param_shard and isinstance(self.mesh_info, FSDPMeshInfo):
            shard_rank = self.mesh_info.shard_mesh_rank
            shard_world_size = self.mesh_info.shard_mesh_size
        else:
            shard_rank = 0
            shard_world_size = 1

        if isinstance(param_data, DTensor) and isinstance(self.mesh_info, DDPMeshInfo):
            param_data.data = param_data.full_tensor()

        self.is_sharded = bool(self.uses_param_shard and shard_world_size > 1)

        if param_data.size(shard_dim) % shard_world_size != 0:
            raise NotImplementedError(
                f"Uneven sharding on dim {shard_dim} not supported: "
                f"shape={param_data.shape}, world_size={shard_world_size}"
            )
        chunks = torch.chunk(param_data, shard_world_size, dim=shard_dim)
        sharded_param = chunks[shard_rank].clone().contiguous()
        self.sharded_size = sharded_param.size()
        self.contiguous_sharded_stride = make_contiguous_strides_for(self.sharded_size)
        if self.offload_to_cpu and not sharded_param.is_meta:
            sharded_param = sharded_param.cpu()
            if self.pin_memory:
                sharded_param = sharded_param.pin_memory()
        self._sharded_param_data = sharded_param.view(-1)

        self._sharding_spec = Layout.from_device_mesh(self._spmd_mesh)
        self._sharding_spec.set_placements(self._spmd_placements)
        self._sharding_spec.placement_to_tensor_map(param.ndim)

        self.sharded_param = nn.Parameter(DTensor.from_local(sharded_param, self._spmd_mesh, self._spmd_placements))
        self.sharded_param.requires_grad_(param.requires_grad)
        self._setattr_on_modules(self.sharded_param)
        # after init, self.sharded_param replaces original param, gradients must accumulate to this Parameter's grad
        self.sharded_param._hsdp_param_initialized = True
        self.sharded_state = ShardedState.SHARDED
        self.param_dtype = None

    def init_dtype_attrs(self, mp_policy: MixedPrecisionPolicy):
        """Initialize param_dtype and reduce_dtype from the mixed precision policy."""
        param_dtype, reduce_dtype = (mp_policy.param_dtype, mp_policy.reduce_dtype)
        self.orig_dtype = self.sharded_param.dtype
        if reduce_dtype == param_dtype:
            reduce_dtype = None
        if param_dtype == self.orig_dtype:
            param_dtype = None
        self.param_dtype = param_dtype
        self.reduce_dtype = reduce_dtype

    def init_all_gather_outputs(
        self,
        all_gather_input_numels: list[int],
        all_gather_input_dtypes: list[torch.dtype],
        world_size: int,
        device: torch.device,
        force_recreate: bool = False,
    ):
        """
        Allocate output buffers for all-gather communication.

        Args:
            all_gather_input_numels: Number of elements per input shard.
            all_gather_input_dtypes: Dtype of each input shard.
            world_size: Number of ranks in the shard process group.
            device: Device on which to allocate the output buffers.
            force_recreate: If True, always recreate buffers even if already initialized.
        """
        if not force_recreate and len(self.all_gather_outputs) > 0:
            return  # already initialized
        self.all_gather_outputs = [
            torch.empty(torch.Size([numel * world_size]), dtype=dtype, device=device)
            for numel, dtype in zip(all_gather_input_numels, all_gather_input_dtypes)
        ]

    def init_unsharded_param(self):
        """
        Initialize unsharded parameter from all-gather outputs.

        This reconstructs the full parameter after all-gather by unpacking the
        gathered flat buffer back to the original tensor layout.
        """
        unsharded_param = self._get_unsharded_param_from_all_gather_output()
        # Always refresh the unsharded Parameter from the latest all-gather output.
        # Non-dim0 unpack currently materializes a contiguous tensor copy, so
        # keeping stale .data would otherwise reuse old weights after optimizer.step()
        # mutates only the sharded local shard. Preserve the Parameter object identity
        # so autograd-facing module state stays stable across unshard cycles.
        if hasattr(self, "_unsharded_param"):
            # pylint: disable=access-member-before-definition
            self._unsharded_param.data = unsharded_param
            self._unsharded_param.requires_grad_(self.sharded_param.requires_grad)
            self._unsharded_param.grad = None
            return
        self._unsharded_param = nn.Parameter(
            unsharded_param,
            requires_grad=self.sharded_param.requires_grad,
        )

    def _get_unsharded_param_from_all_gather_output(self) -> torch.Tensor:
        """Reconstruct the full local parameter view from the packed all-gather output."""
        if len(self.all_gather_outputs) != 1:
            raise AssertionError(
                f"Expected 1 all_gather_output, got {len(self.all_gather_outputs)}"
            )
        unsharded_tensor = self.all_gather_outputs[0]
        plan = build_rs_plan(
            self,
            self._sharded_local_tensor,
            self.shard_world_size if self.is_sharded else 1,
        )
        unsharded_param = unpack_from_all_gather(unsharded_tensor, plan)
        if self._orig_param_is_dtensor:
            # Rebuild the original DTensor view after all-gather so gradient
            # consumers keep seeing the source DTensor layout.
            unsharded_param = DTensor.from_local(
                unsharded_param,
                self._orig_dtensor_mesh,
                self._orig_dtensor_placements,
            )
        return unsharded_param

    def to_sharded(self) -> None:
        if not self.uses_param_shard and self._unsharded_param is not None:
            # Replicate params keep the same local shape across shard/unshard,
            # so persist forward-time state updates before switching objects.
            src = self._unsharded_param.to_local() if isinstance(self._unsharded_param, DTensor) \
                else self._unsharded_param
            dst = self.sharded_param.to_local() if isinstance(self.sharded_param, DTensor) else self.sharded_param
            _copy_without_bumping_version(dst, src)
        self._setattr_on_modules(self.sharded_param)
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
        return DTensor.from_local(
            tensor,
            self._sharding_spec.mesh,
            self._sharding_spec.placements
        )

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
            self.unsharded_accumulated_grad += grad
            self.unsharded_param.grad = None

    def alloc_all_gather_outputs(self) -> None:
        """Resize all-gather output buffers to their full capacity for communication."""
        for tensor in self.all_gather_outputs:
            expected_size = tensor.numel() * tensor.itemsize
            storage = tensor.untyped_storage()
            if storage.size() != expected_size:
                storage.resize_(expected_size)

    def free_unsharded_param(self) -> None:
        """Release storage of all-gather outputs to free device memory."""
        for tensor in self.all_gather_outputs:
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
        grad = self.unsharded_param.grad
        if grad is None:
            raise AssertionError("Expects unsharded_param.grad to not be None")
        return self._to_local_unsharded_grad(grad)

    @property
    def unsharded_accumulated_grad_data(self) -> torch.Tensor:
        """
        Get the unsharded accumulated gradient data as a local tensor.
        """
        grad = self.unsharded_accumulated_grad
        return self._to_local_unsharded_grad(grad)

    @property
    def _sharded_local_tensor(self) -> torch.Tensor:
        """Return the underlying local tensor of the sharded DTensor parameter."""
        return cast(DTensor, self.sharded_param)._local_tensor

    @property
    def shard_world_size(self) -> int:
        """Get the world size for shard dimension."""
        return self.shard_size

    @property
    def replicate_world_size(self) -> int:
        """Get the world size for replicate dimension (HSDP only)."""
        return self.dp_size

    def _assert_in_states(self, *states: ShardedState) -> None:
        """Assert current state is one of expected states."""
        if self.sharded_state not in states:
            raise AssertionError(
                f"Expected sharded_state in {states}, got {self.sharded_state}"
            )

    def reset_sharded_param(self) -> None:
        """Reset sharded param after load_state_dict."""
        module_info = self._module_info
        new_param = getattr(module_info.module, module_info.param_name)
        if new_param is not self.sharded_param:
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
            elif isinstance(new_param, torch.Tensor):
                # if new_param is Tensor, don't change 'self.sharded_param' ref
                # just update self.sharded_param._local_tensor and self.sharded_param_data.
                pass

        local_tensor = new_param._local_tensor if isinstance(new_param, DTensor) else new_param
        if local_tensor.is_meta:
            return
        updated_local_tensor = False
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
        sharded_size = self.sharded_size
        shard_dim = self.hsdp_placement.dim
        length = local_tensor.size(shard_dim) if local_tensor.numel() > 0 else 0
        if not same_local_tensor:
            if local_tensor.size() != sharded_size:
                raise AssertionError(
                    f"Expected sharded_size to be {sharded_size}, got {local_tensor.size()}"
                )
            updated_local_tensor = True
        if self.pin_memory and not local_tensor.is_pinned():
            local_tensor = local_tensor.cpu().pin_memory()
            updated_local_tensor = True
        if not same_local_tensor:
            self._sharded_param_data = local_tensor.view(-1)
        if not isinstance(self.sharded_param, DTensor):
            raise AssertionError(f"Expected DTensor, got {type(self.sharded_param)}")
        if updated_local_tensor:
            # Only change the local tensor object if needed
            self.sharded_param._local_tensor = local_tensor.narrow(
                dim=shard_dim, start=0, length=length
            )
            if not self.sharded_param._local_tensor.is_contiguous():
                raise AssertionError(
                    "Expected sharded_param._local_tensor to be contiguous"
                )
        self._sharding_spec = cast(DTensor, self.sharded_param).layout

    def _get_unsharded_param_data(self, async_op: bool = False) -> Tuple[torch.Tensor, Optional[dist.Work]]:
        """
        Perform all-gather to get unsharded parameter data.

        Args:
            async_op: Whether to execute asynchronously.

        Returns:
            (unsharded_param, handle): Unsharded parameter data and communication handle.
        """
        # If parameter is not sharded (below threshold), no communication needed
        if not self.is_sharded:
            all_gather_input = self.all_gather_inputs[0]
            self.init_all_gather_outputs(
                all_gather_input_numels=[all_gather_input.numel()],
                all_gather_input_dtypes=[all_gather_input.dtype],
                world_size=1,
                device=self.device,
            )
            self.alloc_all_gather_outputs()
            _copy_without_bumping_version(self.all_gather_outputs[0], all_gather_input)
            return self.all_gather_outputs[0], None

        # Get input data
        all_gather_input = self.all_gather_inputs[0]

        # Initialize output buffer
        self.init_all_gather_outputs(
            all_gather_input_numels=[all_gather_input.numel()],
            all_gather_input_dtypes=[all_gather_input.dtype],
            world_size=self.shard_world_size,
            device=self.device,
        )
        self.alloc_all_gather_outputs()

        if self.sharded_group_info.group is None or self.shard_world_size <= 1:
            # No communication needed, just copy
            _copy_without_bumping_version(self.all_gather_outputs[0], all_gather_input)
            return self.all_gather_outputs[0], None

        # Execute all_gather_into_tensor
        handle = dist.all_gather_into_tensor(
            self.all_gather_outputs[0],
            all_gather_input,
            group=self.sharded_group_info.group,
            async_op=async_op,
        )

        return self.all_gather_outputs[0], handle

    def unshard(self, async_op: bool = False) -> None:
        if self.prefetch_handle is not None:
            # Already triggered by HSDPState.prefetch(), so return directly.
            return  # no-op

        _, handle = self._get_unsharded_param_data(async_op=async_op)
        self.prefetch_handle = handle

    def wait_for_unshard(self) -> None:
        self._assert_in_states(ShardedState.SHARDED)

        if self.prefetch_handle is not None:
            self.prefetch_handle.wait()
            self.prefetch_handle = None

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
        dtype: Optional[torch.dtype] = None,
        reduce_op: Optional[dist.ReduceOp] = dist.ReduceOp.AVG
    ) -> Union[None, Tuple[torch.Tensor, Optional[dist.Work]]]:
        """
        Perform reduce-scatter on gradient to reduce and shard the full gradient.

        Args:
            async_op: Whether to execute asynchronously.
            dtype: reduce dtype.
            reduce_op: do reduce-scatter avg or sum.

        Returns:
            (sharded_grad, handle): Sharded gradient and communication handle.
        """
        self._assert_in_states(ShardedState.UNSHARDED)

        # Choose gradient source based on use_accumulated_grad flag
        if self.unsharded_accumulated_grad is not None:
            grad = self.unsharded_accumulated_grad_data
        else:
            grad = self.unsharded_grad_data
        reduce_dtype = dtype or grad.dtype
        grad = grad.to(reduce_dtype)
        plan_world_size = (
            self.shard_world_size
            if self.is_sharded
            and self.sharded_group_info.group is not None
            and self.shard_world_size > 1
            else 1
        )
        plan = build_rs_plan(self, grad, plan_world_size)
        grad_flat = pack_for_reduce_scatter(grad, plan).reshape(-1)

        # If parameter is not sharded (below threshold), no reduce-scatter needed
        if not self.is_sharded:
            return grad_flat, None

        if self.sharded_group_info.group is None or self.shard_world_size <= 1:
            # No communication needed
            return grad_flat, None

        # Calculate output size
        output_numel = grad_flat.numel() // self.shard_world_size
        self._reduce_scatter_output = torch.empty(output_numel, dtype=reduce_dtype, device=grad.device)

        # Execute reduce_scatter_tensor
        self.reduce_scatter_handle = dist.reduce_scatter_tensor(
            self._reduce_scatter_output,
            grad_flat,
            op=reduce_op,
            group=self.sharded_group_info.group,
            async_op=async_op,
        )
        return self._reduce_scatter_output, self.reduce_scatter_handle

    def all_reduce_grad(
        self,
        grad: Optional[torch.Tensor] = None,
        dtype: Optional[torch.dtype] = None,
        async_op: bool = True,
        reduce_op: Optional[dist.ReduceOp] = dist.ReduceOp.AVG
    ) -> Union[None, Tuple[torch.Tensor, Optional[dist.Work]]]:
        """
        Perform all-reduce on gradient (across replicate dimension in HSDP mode).

        Args:
            grad: Gradient tensor to reduce. If None, will use unsharded_param.grad
                or unsharded_accumulated_grad based on use_accumulated_grad flag.
            async_op: Whether to execute asynchronously.
            reduce_op: Optional[dist.ReduceOp] = dist.ReduceOp.AVG.

        Returns:
            (reduced_grad, handle): Reduced gradient and communication handle.
        """
        # If grad is not provided, get from parameter
        if grad is None:
            if self.unsharded_accumulated_grad is not None:
                grad = self.unsharded_accumulated_grad_data
            else:
                grad = self.unsharded_grad_data

        if dtype is not None and dtype != grad.dtype:
            grad = grad.to(dtype)

        if self.unsharded_group_info.group is None or self.replicate_world_size <= 1:
            return grad, None

        self.all_reduce_handle = dist.all_reduce(grad, op=reduce_op,
                                                 group=self.unsharded_group_info.group, async_op=async_op)
        self._all_reduce_output = grad
        return grad, self.all_reduce_handle


def set_requires_grad_if_needed(
    src_tensor: torch.Tensor, dst_tensor: torch.Tensor
) -> None:
    """set dst_tensor requires_grads from src_tensor if needed."""
    if src_tensor.requires_grad != dst_tensor.requires_grad:
        dst_tensor.requires_grad_(src_tensor.requires_grad)
