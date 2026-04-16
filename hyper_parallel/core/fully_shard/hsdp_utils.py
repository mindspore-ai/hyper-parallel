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
"""HSDP optimizer shared level"""
from dataclasses import dataclass, field
from enum import auto, Enum
from typing import Any, List, Optional, Sequence

import numpy as np

from hyper_parallel.core.dtensor.device_mesh import DeviceMesh
from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.platform import get_platform
from hyper_parallel.platform.platform import PlatformType

platform = get_platform()


class HSDPConfigV2:
    """HSDPConfigV2 inspect by torch fully_shard"""

    def __init__(self,
        mesh,
        reshard_after_forward,
        shard_placement_fn,
        mp_policy,
        offload_policy,
        ignored_params=None,
        replicate_params=None,
        comm_fusion=False,
        comm_fusion_zero_copy=False,
    ):
        self.mesh = mesh
        self.reshard_after_forward = reshard_after_forward
        self.shard_placement_fn = shard_placement_fn
        self.mp_policy = mp_policy
        self.offload_policy = offload_policy
        self.ignored_params = ignored_params
        self.replicate_params = replicate_params
        self.reduce_dtype = self.mp_policy.reduce_dtype if self.mp_policy else None
        self.comm_fusion = comm_fusion
        self.comm_fusion_zero_copy = comm_fusion_zero_copy


class ShardedState(Enum):
    """
    Parameter shard state
    """
    SHARDED = auto()
    UNSHARDED = auto()


class FullyShardParamMode(Enum):
    """
    Internal fully_shard execution modes derived from the original parameter layout.

    LOCAL_PARAM:
        The parameter is a regular local tensor parameter and fully_shard owns the
        full data-parallel sharding behaviour.
    DTENSOR_COMPAT:
        The parameter already carries a DTensor layout and fully_shard is only used
        as the compatibility wrapper without adding an extra FSDP shard dimension.
    DTENSOR_UNIFIED:
        The parameter already carries a DTensor layout and fully_shard additionally
        contributes a data-parallel/FSDP mesh that must be unified with the
        existing distributed layout.
    """

    LOCAL_PARAM = auto()
    DTENSOR_COMPAT = auto()
    DTENSOR_UNIFIED = auto()


@dataclass
class GroupInfo:
    """Communication group metadata used by fully_shard."""

    group_name: str
    group: Any
    rank_size: int


class FSDPSchedulerState(Enum):
    """
        Scheduler state:
                - PRE_FORWARD:
                  already run hook before forward.
                - FORWARD:
                  already run hook after forward.
                - PRE_BACKWARD:
                  already run hook before backward.
                - BACKWARD:
                  already run hook after backward.
    """
    PRE_FORWARD = auto()
    FORWARD = auto()
    PRE_BACKWARD = auto()
    BACKWARD = auto()


@dataclass
class ParamModuleInfo:
    """
    Tracks parameter ownership and supports shared weights in HSDP.

    This dataclass maintains the mapping between a parameter and its module(s),
    enabling parameter swapping during sharding/unsharding transitions. Shared
    weights are parameters referenced by multiple modules (e.g., tied embeddings).

    This class tracks all references to ensure proper parameter replacement during 
    sharding/unsharding operations.

    Attributes:
        module: The module that owns this parameter.
        param_name: Attribute name of the parameter in the module (e.g., "weight").
        shared_modules: List of other modules sharing this same parameter object.
        shared_param_names: Corresponding parameter names in shared_modules (aligned by index).
    """
    module: platform.Module
    param_name: str
    shared_modules: List[platform.Module] = field(default_factory=list)
    shared_param_names: List[str] = field(default_factory=list)


def _named_parameters_with_duplicates(
    module: platform.Module, **kwargs: Any
) -> list[tuple[str, platform.Parameter]]:
    """
    This API is required as some modules overwrite `named_parameters()` but do not support
    `remove_duplicate`.
    """
    if "remove_duplicate" in kwargs:
        raise AssertionError(
            "_named_parameters_with_duplicates cannot be used with `remove_duplicate` argument."
        )

    def get_named_parameters(module, **kwargs):
        if platform.platform_type == PlatformType.PYTORCH:
            return module.named_parameters(**kwargs)
        return module.parameters_and_names(expand=False)
    kwargs["remove_duplicate"] = False
    try:
        ret = list(get_named_parameters(module, **kwargs))
    except AssertionError:
        kwargs.pop("remove_duplicate")
        ret = list(get_named_parameters(module, **kwargs))
    return ret


def _get_param_module_infos(
    params: list[platform.Parameter], modules: tuple[platform.Module, ...]
) -> list['ParamModuleInfo']:
    """
    Shared parameter: lin1.weight = lin2.weight
    Shared module: mlp.lin1 = mlp.lin2
    We do not remove duplicates when traversing both modules and parameters to
    find shared modules' parameters and shared parameters within a module.
    """
    params_set = set(params)
    param_to_module_info: dict[platform.Parameter, ParamModuleInfo] = {}

    def get_named_modules(module):
        if platform.platform_type == PlatformType.PYTORCH:
            return module.named_modules(remove_duplicate=False)
        return module.cells_and_names()

    for module in modules:
        for _, submodule in get_named_modules(module):
            for param_name, param in _named_parameters_with_duplicates(
                submodule, recurse=False
            ):
                if param in params_set:
                    if param not in param_to_module_info:
                        param_to_module_info[param] = ParamModuleInfo(
                            submodule, param_name
                        )
                    else:
                        param_to_module_info[param].shared_modules.append(submodule)
                        param_to_module_info[param].shared_param_names.append(
                            param_name
                        )
    if len(param_to_module_info) != len(params):
        raise AssertionError(f"Some parameters are not in the module tree of {modules}")
    return [param_to_module_info[param] for param in params]


def get_managed_modules_parameters(
    modules: Sequence[platform.Module],
    ignored_params: Optional[Sequence[platform.Parameter]] = None,
) -> list[platform.Parameter]:
    """Collect deduplicated parameters from ``modules`` while skipping ignored params.

    Parameters that were already initialized by an inner ``fully_shard`` instance
    are intentionally excluded so nested ``fully_shard(mesh=None)`` resolves mesh
    mode from the parameters that the current wrapper will actually manage.
    """
    params: list[platform.Parameter] = []
    ignored_params_set = set(ignored_params or ())
    visited_params: set[platform.Parameter] = set()
    for mod in modules:
        for _, param in platform.parameters_dict(mod):
            if param in ignored_params_set or param in visited_params:
                continue
            if getattr(param, "_hsdp_param_initialized", False):
                continue
            visited_params.add(param)
            params.append(param)
    return params


def infer_fully_shard_param_mode(
    mesh: Optional[DeviceMesh],
    params: Optional[Sequence[Any]] = None,
) -> FullyShardParamMode:
    """
    Infer the internal fully_shard execution mode from parameter layout and mesh.

    The mode is intentionally phrased around whether parameters already carry a
    distributed layout instead of assuming the layout came from TP only. DTensor
    parameters may originate from TP, EP, or other distributed sharding paths.
    """
    has_dtensor_param = any(is_dtensor_managed_param(param) for param in params or ())
    if not has_dtensor_param:
        return FullyShardParamMode.LOCAL_PARAM
    if mesh is None:
        return FullyShardParamMode.DTENSOR_COMPAT
    return FullyShardParamMode.DTENSOR_UNIFIED


def unwrap_dtensor_param(param: Any) -> Optional[DTensor]:
    """Return the DTensor payload carried by ``param`` if one exists."""
    if isinstance(param, DTensor):
        return param
    param_data = getattr(param, "data", None)
    if isinstance(param_data, DTensor):
        return param_data
    if all(hasattr(param, attr) for attr in ("_device_mesh", "_placements", "_local_tensor")):
        return param
    return None


def is_dtensor_managed_param(param: Any) -> bool:
    """Return whether a parameter already carries DTensor layout metadata."""
    return unwrap_dtensor_param(param) is not None


def get_dtensor_managed_mesh(param: Any) -> Optional[DeviceMesh]:
    """Return the DTensor mesh carried by ``param`` if one exists."""
    payload = unwrap_dtensor_param(param)
    if payload is None:
        return None
    return getattr(payload, "device_mesh", getattr(payload, "_device_mesh", None))


def get_rank_list_for_axes(
    mesh: DeviceMesh,
    axes: Sequence[int],
    rank: Optional[int] = None,
) -> list[int]:
    """Return ranks that vary along ``axes`` and keep all other coordinates fixed."""
    if rank is None:
        rank = mesh.rank
    if rank not in mesh.rank_list:
        raise ValueError(f"Rank {rank} not found in mesh rank list {mesh.rank_list}.")

    normalized_axes = tuple(sorted(set(axes)))
    if len(normalized_axes) == 0:
        return [rank]

    mesh_tensor = np.array(mesh.rank_list).reshape(mesh.mesh_shape)
    rank_index = mesh.rank_list.index(rank)
    coord = [0] * len(mesh.mesh_shape)
    temp = rank_index
    for i in range(len(mesh.mesh_shape) - 1, -1, -1):
        coord[i] = temp % mesh.mesh_shape[i]
        temp //= mesh.mesh_shape[i]
    mesh_slice = []
    for axis, axis_coord in enumerate(coord):
        mesh_slice.append(slice(None) if axis in normalized_axes else axis_coord)
    selected = mesh_tensor[tuple(mesh_slice)]
    return [int(item) for item in np.array(selected).reshape(-1).tolist()]


def get_split_rank_lists_for_axes(
    mesh: DeviceMesh,
    axes: Sequence[int],
) -> list[list[int]]:
    """Return all rank lists induced by varying ``axes`` and fixing the complementary axes."""
    normalized_axes = tuple(sorted(set(axes)))
    if len(normalized_axes) == 0:
        return [[int(rank) for rank in mesh.rank_list]]

    mesh_tensor = np.array(mesh.rank_list).reshape(mesh.mesh_shape)
    complementary_axes = tuple(
        axis for axis in range(len(mesh.mesh_shape)) if axis not in normalized_axes
    )
    if len(complementary_axes) == 0:
        return [[int(item) for item in np.array(mesh_tensor).reshape(-1).tolist()]]

    complementary_shape = tuple(mesh.mesh_shape[axis] for axis in complementary_axes)
    split_rank_lists: list[list[int]] = []
    for complementary_coord in np.ndindex(*complementary_shape):
        mesh_slice = []
        coord_idx = 0
        for axis in range(len(mesh.mesh_shape)):
            if axis in normalized_axes:
                mesh_slice.append(slice(None))
            else:
                mesh_slice.append(complementary_coord[coord_idx])
                coord_idx += 1
        selected = mesh_tensor[tuple(mesh_slice)]
        split_rank_lists.append([int(item) for item in np.array(selected).reshape(-1).tolist()])
    return split_rank_lists
