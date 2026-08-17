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
# ============================================================================
"""HSDP parameter"""

from hyper_parallel.core.dtensor.device_mesh import DeviceMesh
from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.dtensor.placement_types import Replicate
from hyper_parallel.core.fully_shard.hsdp_utils import (
    FullyShardParamMode,
    GroupInfo,
    get_rank_list_for_axes,
    get_split_rank_lists_for_axes,
)
from hyper_parallel.core.fully_shard.utils import DDPMeshInfo, FSDPMeshInfo
from hyper_parallel.platform import get_platform

platform = get_platform()
_GROUP_INFO_CACHE = {}


def _build_group_info_from_rank_list(group_name: str, rank_list) -> GroupInfo:
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
    *,
    resolved_group_name: str | None = None,
) -> GroupInfo:
    """Create group metadata from an existing process group."""
    if process_group is None or rank_size <= 1:
        return GroupInfo(f"{group_name}_invalid", None, 1)
    return GroupInfo(resolved_group_name or group_name, process_group, rank_size)


class HSDPParamV2:
    """
    HSDP parameter.
    """

    def __repr__(self) -> str:
        """Stable debug name used in log lines.

        Prefers the parameter FQN (assigned by the root forward pre-hook); before
        that, falls back to the owning module class plus param name and object id.
        Logging's ``%s`` calls this lazily -- only when a record is emitted.
        """
        fqn = getattr(self, "_param_fqn", None)
        if fqn:
            return str(fqn)
        module_info = getattr(self, "_module_info", None)
        if module_info is not None:
            return f"{module_info.module.__class__.__name__}.{module_info.param_name}@{id(self):x}"
        return f"{self.__class__.__name__}@{id(self):x}"

    def __init__(
        self,
        param,
        module_info,
        mesh_info,
        post_forward_mesh_info,
        shard_placement_fn,
        mp_policy,
        offload_policy,
        threshold,
    ):
        """
        Initialize HSDPParamV2.

        Args:
            param (nn.Parameter): The original parameter to shard.
            module_info (ParamModuleInfo): Ownership and shared-weight metadata for the parameter.
            mesh_info (FSDPMeshInfo): Mesh topology describing shard/replicate dimensions.
            post_forward_mesh_info: Mesh info used after forward (reserved for subclass use).
            shard_placement_fn (Callable, optional): Returns a Shard placement for the parameter,
                or None to use default (Shard(0)).
            mp_policy (MixedPrecisionPolicy, optional): Mixed precision dtype policy.
            offload_policy (OffloadPolicy, optional): CPU offload policy.
            threshold: Minimum parameter size to enable sharding (reserved for subclass use).
        """
        raise NotImplementedError("HSDP param subclasses must implement __init__")

    def _init_sharded_param(self, param, shard_placement_fn):
        """add and init sharded param"""
        raise NotImplementedError("HSDP param subclasses must implement _init_sharded_param")

    def init_dtype_attrs(self, mp_policy):
        """Initialize dtype attributes from mixed precision policy."""
        raise NotImplementedError("HSDP param subclasses must implement init_dtype_attrs")

    def init_unsharded_param_buffers(
        self, all_gather_input_numels, all_gather_input_dtypes, world_size, device, force_recreate=False
    ):
        """Allocate or reuse buffers that hold unsharded parameter data."""
        raise NotImplementedError("HSDP param subclasses must implement init_unsharded_param_buffers")

    def init_unsharded_param(self):
        """Reconstruct the full unsharded parameter from all-gather outputs."""
        raise NotImplementedError("HSDP param subclasses must implement init_unsharded_param")

    def to_sharded(self):
        """Transition parameter from unsharded back to sharded state and free unsharded storage."""
        raise NotImplementedError("HSDP param subclasses must implement to_sharded")

    def to_unsharded(self):
        """Transition parameter to unsharded state after all-gather completes."""
        raise NotImplementedError("HSDP param subclasses must implement to_unsharded")

    def to_sharded_dtensor(self, tensor):
        """Wrap a local sharded tensor as a DTensor with the correct mesh and placements."""
        raise NotImplementedError("HSDP param subclasses must implement to_sharded_dtensor")

    def to_accumulated_grad_if_needed(self):
        """Move unsharded grad to accumulated grad buffer if dtype conversion is required."""
        raise NotImplementedError("HSDP param subclasses must implement to_accumulated_grad_if_needed")

    def accumulate_unsharded_grad_if_needed(self):
        """Accumulate unsharded param grad into accumulated grad buffer if both exist."""
        raise NotImplementedError("HSDP param subclasses must implement accumulate_unsharded_grad_if_needed")

    def alloc_unsharded_param_buffers(self):
        """Restore unsharded parameter buffers to their full capacity."""
        raise NotImplementedError("HSDP param subclasses must implement alloc_unsharded_param_buffers")

    def free_unsharded_param(self):
        """Release unsharded parameter buffers and inner tensors."""
        raise NotImplementedError("HSDP param subclasses must implement free_unsharded_param")

    @property
    def all_gather_inputs(self):
        """Return the local sharded tensor(s) to use as input for all-gather communication."""
        raise NotImplementedError("HSDP param subclasses must implement all_gather_inputs")

    @property
    def unsharded_param(self):
        """Return the full unsharded parameter after all-gather."""
        raise NotImplementedError("HSDP param subclasses must implement unsharded_param")

    @property
    def unsharded_grad_data(self):
        """Return the unsharded_param.grad."""
        raise NotImplementedError("HSDP param subclasses must implement unsharded_grad_data")

    @property
    def unsharded_accumulated_grad_data(self):
        """Return the unsharded accumulated gradient buffer."""
        raise NotImplementedError("HSDP param subclasses must implement unsharded_accumulated_grad_data")

    @property
    def _sharded_local_tensor(self):
        """Return the underlying local tensor of the sharded DTensor parameter."""
        raise NotImplementedError("HSDP param subclasses must implement _sharded_local_tensor")

    def _get_unsharded_param_data(self, async_op=False):
        """Perform all-gather to obtain unsharded parameter data, returning (tensor, handle)."""
        raise NotImplementedError("HSDP param subclasses must implement _get_unsharded_param_data")

    def unshard(self, async_op=False):
        """Trigger all-gather to unshard the parameter, optionally asynchronously."""
        raise NotImplementedError("HSDP param subclasses must implement unshard")

    def wait_for_unshard(self):
        """Wait for all-gather to complete and transition parameter to unsharded state."""
        raise NotImplementedError("HSDP param subclasses must implement wait_for_unshard")

    def shard(self):
        """Transition parameter from unsharded back to sharded state."""
        raise NotImplementedError("HSDP param subclasses must implement shard")

    def reduce_scatter_grad(self):
        """Perform reduce-scatter on the unsharded gradient to produce a sharded gradient."""
        raise NotImplementedError("HSDP param subclasses must implement reduce_scatter_grad")

    def all_reduce_grad(self):
        """Perform all-reduce on gradient across the replicate dimension (HSDP mode only)."""
        raise NotImplementedError("HSDP param subclasses must implement all_reduce_grad")

    def _resolve_process_group_name(self, group_name: str, process_group) -> str:
        """Resolve the name recorded in GroupInfo for an existing process group."""
        del process_group
        return group_name

    def _get_base_spmd_placements(self) -> tuple:
        """Return placements before explicit data-parallel semantics are applied."""
        if (
            getattr(self, "param_mode", None) == FullyShardParamMode.DTENSOR_UNIFIED
            and getattr(self, "_orig_param_is_dtensor", False)
        ):
            self._spmd_mesh = DeviceMesh.concatenate([self.mesh_info.mesh, self._orig_dtensor_mesh])
            dp_prefix_placements = tuple(Replicate() for _ in range(self.mesh_info.mesh.ndim))
            return dp_prefix_placements + tuple(self._orig_dtensor_placements)

        if (
            getattr(self, "param_mode", None) == FullyShardParamMode.DTENSOR_COMPAT
            and getattr(self, "_orig_param_is_dtensor", False)
        ):
            self._spmd_mesh = self._orig_dtensor_mesh
            return tuple(self._orig_dtensor_placements)

        self._spmd_mesh = self.mesh_info.mesh
        return tuple(Replicate() for _ in range(self._spmd_mesh.ndim))

    def _get_data_parallel_shard_placement(self, placements: list, shard_placement):
        """Return the placement to apply on the explicit fully_shard dimension."""
        del placements
        return shard_placement

    def _apply_data_parallel_placements(self, placements: list, shard_placement) -> tuple:
        """Apply explicit DDP/FSDP placements on top of the base SPMD layout."""
        if len(placements) != self._spmd_mesh.ndim:
            raise AssertionError(
                f"Expected {self._spmd_mesh.ndim} unified placements, got {len(placements)}: {placements}"
            )

        spmd_replicate_mesh_dim = getattr(self, "_spmd_replicate_mesh_dim", None)
        if (
            isinstance(self.mesh_info, DDPMeshInfo)
            and spmd_replicate_mesh_dim is not None
            and not getattr(self, "_orig_param_is_dtensor", False)
        ):
            placements[spmd_replicate_mesh_dim] = Replicate()

        spmd_shard_mesh_dim = getattr(self, "_spmd_shard_mesh_dim", None)
        if (
            getattr(self, "uses_param_shard", False)
            and isinstance(self.mesh_info, FSDPMeshInfo)
            and spmd_shard_mesh_dim is not None
        ):
            placements[spmd_shard_mesh_dim] = self._get_data_parallel_shard_placement(
                placements, shard_placement
            )
        return tuple(placements)

    def _init_group_infos(self) -> None:
        """Initialize sharded/unsharded communication groups from the current layout."""
        if (
            getattr(self, "uses_param_shard", False)
            and getattr(self, "is_sharded", False)
            and isinstance(self.mesh_info, FSDPMeshInfo)
        ):
            resolved_group_name = self._resolve_process_group_name(
                "fully_shard_sharded_group",
                self.mesh_info.shard_process_group,
            )
            self.sharded_group_info = _build_group_info_from_process_group(
                "fully_shard_sharded_group",
                self.mesh_info.shard_process_group,
                self.mesh_info.shard_mesh_size,
                resolved_group_name=resolved_group_name,
            )
        else:
            self.sharded_group_info = GroupInfo("fully_shard_sharded_group_invalid", None, 1)

        self.unsharded_group_info = self._build_layout_driven_group_info()
        self.shard_size = self.sharded_group_info.rank_size
        self.dp_size = self.unsharded_group_info.rank_size
        self.rank_size = max(1, self.shard_size * self.dp_size)

    def _build_layout_driven_group_info(self) -> GroupInfo:
        """Build the group that should all-reduce an unsharded gradient from the final layout."""
        group_axes = [
            axis
            for axis, placement in enumerate(self._spmd_placements)
            if placement.is_replicate()
        ]
        spmd_shard_mesh_dim = getattr(self, "_spmd_shard_mesh_dim", None)
        if getattr(self, "uses_param_shard", False) and spmd_shard_mesh_dim is not None:
            group_axes = [axis for axis in group_axes if axis != spmd_shard_mesh_dim]
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
                        resolved_group_name = self._resolve_process_group_name(
                            "fully_shard_unsharded_group",
                            process_group,
                        )
                        return _build_group_info_from_process_group(
                            "fully_shard_unsharded_group",
                            process_group,
                            rank_size,
                            resolved_group_name=resolved_group_name,
                        )

                split_rank_lists = get_split_rank_lists_for_axes(self._spmd_mesh, group_axes)
                process_group = platform.split_group(split_ranks=split_rank_lists)
                if process_group is not None:
                    rank_size = 1
                    for axis in group_axes:
                        rank_size *= self._spmd_mesh.mesh_shape[axis]
                    resolved_group_name = self._resolve_process_group_name(
                        "fully_shard_unsharded_group",
                        process_group,
                    )
                    return _build_group_info_from_process_group(
                        "fully_shard_unsharded_group",
                        process_group,
                        rank_size,
                        resolved_group_name=resolved_group_name,
                    )
            except (
                AssertionError,
                AttributeError,
                KeyError,
                RuntimeError,
                TypeError,
                ValueError,
            ):
                pass

        rank_list = get_rank_list_for_axes(self._spmd_mesh, group_axes)
        return _build_group_info_from_rank_list("fully_shard_unsharded_group", rank_list)

    def _normalize_unsharded_grad_to_local(self, grad, *, reduce_partial_dtensor: bool = True):
        """Normalize a pending gradient to the local tensor expected by fully_shard collectives."""
        if not isinstance(grad, DTensor):
            return grad

        if reduce_partial_dtensor and any(placement.is_partial() for placement in grad.placements):
            grad = grad.reduce_partial()

        orig_dtensor_mesh = getattr(self, "_orig_dtensor_mesh", None)
        orig_dtensor_placements = getattr(self, "_orig_dtensor_placements", None)
        mesh_mismatch = (
            orig_dtensor_mesh is not None
            and grad.device_mesh.to_hash() != orig_dtensor_mesh.to_hash()
        )
        placement_mismatch = (
            orig_dtensor_placements is not None
            and tuple(grad.placements) != tuple(orig_dtensor_placements)
        )
        if mesh_mismatch or placement_mismatch:
            grad = grad.redistribute(orig_dtensor_mesh, orig_dtensor_placements)
        return grad.to_local()
