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
"""device mesh"""

import copy
import os
import threading
from types import TracebackType
from typing import Any, List, Literal, Optional, Sequence, Type, Union
import numpy as np

from hyper_parallel.core.dtensor._mesh_layout import IntTuple, _MeshLayout, _contiguous_strides, _is_int
from hyper_parallel.platform import get_platform
from hyper_parallel.platform.platform import EXISTING_COMM_GROUPS, Platform, PlatformType

platform = get_platform()
Tensor = platform.Tensor


def _host_tensor_from_numpy(np_array: np.ndarray):
    """Build a host-resident int tensor from a NumPy array for rank/mesh bookkeeping.

    A real platform's ``from_numpy`` keeps the tensor off the meta device, so it stays
    ``asnumpy``-able even when a DeviceMesh is built under ``ms.DeviceCtx("meta")``
    (``fully_shard`` with ``mesh=None``). Unit tests run with a mocked ``platform``
    (no real ``from_numpy``, and never under a meta context), so fall back to the plain
    ``Tensor`` constructor there.
    """
    if isinstance(platform, Platform):
        return platform.from_numpy(np_array)
    return Tensor(np_array).int()


class _MeshEnv(threading.local):
    """Per-thread stack of active :class:`DeviceMesh` (PyTorch ``_mesh_resources`` parity)."""

    def __init__(self) -> None:
        super().__init__()
        self.mesh_stack: List["DeviceMesh"] = []

    def get_current_mesh(self) -> "DeviceMesh":
        """Return the innermost active :class:`DeviceMesh` for this thread (PyTorch parity)."""
        if len(self.mesh_stack) == 0:
            raise RuntimeError("No device mesh is currently active!")
        return self.mesh_stack[-1]


_mesh_resources = _MeshEnv()

BackendConfig = Optional[str]


def _get_sub_rank_list(mesh_shape, mesh_dim_names, rank_list, sub_mesh_dim_names, current_rank):
    """
    Get the sub rank list for a sub mesh.

    Args:
        mesh_shape (tuple[int]): The shape of the original mesh.
        mesh_dim_names (tuple[str]): The mesh dim names of the original mesh dimensions.
        rank_list (tuple[int]): A tuple of ranks that participate in this mesh.
        sub_mesh_dim_names (tuple[str]): The mesh dim names of the sub mesh to extract.
        current_rank (int): The current process rank.

    Returns:
        list: The sub rank list for the sub mesh.
    """
    mesh_tensor = np.array(rank_list).reshape(mesh_shape)

    for dim_index, dim_name in enumerate(mesh_dim_names):
        if dim_name in sub_mesh_dim_names:
            continue

        dim_size = mesh_shape[dim_index]
        sliced_tensors = np.split(mesh_tensor, dim_size, axis=dim_index)

        for sliced_tensor in sliced_tensors:
            rank_exists = np.isin(np.array([current_rank]), sliced_tensor).any()
            if rank_exists:
                mesh_tensor = sliced_tensor
                break

    sub_rank_list = mesh_tensor.reshape(-1).tolist()
    return sub_rank_list


def _normalize_backend_value(value: Any) -> BackendConfig:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, tuple) and len(value) > 0:
        backend = value[0]
        if backend is None or isinstance(backend, str):
            return backend
    return None


def _normalize_backend_override(
        backend_override: dict[Union[int, str], Any],
        ndim: int,
        mesh_dim_names: Optional[tuple[str, ...]] = None,
) -> tuple[BackendConfig, ...]:
    """Normalize backend overrides by dim index/name."""
    remaining = dict(backend_override)
    normalized: list[BackendConfig] = []
    mesh_dim_names = mesh_dim_names or ()

    for dim_idx in range(ndim):
        dim_name = mesh_dim_names[dim_idx] if dim_idx < len(mesh_dim_names) else None
        if dim_name is not None and dim_name in remaining:
            if dim_idx in remaining:
                raise RuntimeError(
                    f"Found redundant dim index {dim_idx} and name {dim_name} in backend_override"
                )
            normalized.append(_normalize_backend_value(remaining.pop(dim_name)))
        elif dim_idx in remaining:
            normalized.append(_normalize_backend_value(remaining.pop(dim_idx)))
        else:
            normalized.append(None)

    if remaining:
        raise RuntimeError(
            f"Found invalid keys in backend_override: got {list(remaining.keys())}, "
            f"expected integers in range [0, {ndim}) or one of {mesh_dim_names}"
        )
    return tuple(normalized)


def _should_defer_group_init(sub_layout: _MeshLayout, backend_override: BackendConfig) -> bool:
    """Whether this mesh dimension should skip eager process-group creation."""
    return backend_override == "fake" or sub_layout.numel() == 1


class DeviceMesh:
    """
    Topological abstraction describing cluster devices.

    Args:
        device_type (str): Device type. Valid values depend on the active platform:

            - **PyTorch** (same as ``torch.distributed.device_mesh.DeviceMesh``):
              ``"cpu"``, ``"cuda"``, ``"npu"``.
            - **MindSpore** (mapped to the corresponding communication backend):
              ``"cpu"`` → mccl, ``"gpu"`` → nccl, ``"npu"`` → hccl.
        mesh (Union[Tensor, list, tuple, np.ndarray, None]): A multi-dimensional array, list, or integer
            tensor describing the device layout. The IDs in the mesh are global IDs of the
            default process group, representing the multi-dimensional networking structure
            of devices in distributed training (e.g., [[0,1],[2,3]] represents a 2x2 device mesh).
            If a list or non-int32 tensor is provided, it will be automatically converted
            to an int32 tensor. If None, a 1D mesh containing all ranks
            (i.e., ``[0, 1, ..., world_size-1]``) will be created automatically.
        mesh_dim_names (tuple[str]): A tuple[str] of mesh dim names for each dimension of mesh.
        _init_backend (boolean): Whether initial process group.

    Attributes:
        ndim (int): Number of dimensions in the mesh.
        mesh_shape (tuple[int]): Shape of the device mesh.
        rank_list (tuple[int]): Flattened list of ranks from the mesh.
        root_mesh (DeviceMesh): The parent mesh if this is a sub mesh, None otherwise.
        sub_mesh (list[DeviceMesh]): List of child meshes created from this mesh.

    Context manager:
        Use ``with device_mesh:`` to set the **current** mesh for this thread.
    """

    device_type: Literal["cpu", "cuda", "gpu", "npu"]
    mesh: Union[Tensor, list, tuple, np.ndarray]
    mesh_dim_names: Union[tuple[str, ...], list[str], None]

    _VALID_DEVICE_TYPES = {
        PlatformType.PYTORCH: {"cpu", "cuda", "npu"},
        PlatformType.MINDSPORE: {"cpu", "gpu", "npu"},
    }

    def __init__(self,
                 device_type: Literal["cpu", "cuda", "gpu", "npu"],
                 mesh: Union[Tensor, list, tuple, np.ndarray, None] = None,
                 *,
                 mesh_dim_names: Union[tuple[str, ...], list[str], None] = None,
                 _init_backend: bool = True,
                 _layout: Optional[_MeshLayout] = None,
                 _rank_map: Optional[Tensor] = None,
                 _root_mesh: Optional['DeviceMesh'] = None,
                 ):
        self._validate_device_type(device_type)
        self.device_type = device_type

        if _init_backend:
            platform.init_process_group()

        self._layout, self._rank_map = self._resolve_layout_and_rank_map(mesh, _layout, _rank_map)
        self._rank = platform.get_rank()
        self._root_mesh = _root_mesh
        self._refresh_mesh_view()
        self._set_mesh_dim_names(mesh_dim_names)
        self._initialize_runtime_state(_init_backend)
        if os.getenv("MS_SIMULATION_LEVEL") is None:
            self._coordinate_on_dim = self._compute_coordinate_on_dim()

    @classmethod
    def _validate_device_type(cls, device_type: str) -> None:
        """Validate that the requested device type is supported on the active platform."""
        valid_device_types = cls._VALID_DEVICE_TYPES.get(platform.platform_type)
        if valid_device_types is not None and device_type not in valid_device_types:
            raise ValueError(
                f"Invalid device_type '{device_type}' for {platform.platform_type.name} platform. "
                f"Valid device types are: {sorted(valid_device_types)}"
            )

    @classmethod
    def _resolve_layout_and_rank_map(
            cls,
            mesh: Union[Tensor, list, tuple, np.ndarray, None],
            layout: Optional[_MeshLayout],
            rank_map: Optional[Tensor],
    ) -> tuple[_MeshLayout, Tensor]:
        """Build the internal layout and rank map from either public or private constructor inputs."""
        if mesh is not None and (layout is not None or rank_map is not None):
            raise TypeError("Cannot provide both explicit mesh and private _layout/_rank_map arguments.")

        if mesh is None and (layout is None or rank_map is None):
            world_size = platform.get_world_size()
            mesh = list(range(world_size))

        if mesh is not None:
            mesh_tensor = cls._convert_mesh_to_tensor(mesh)
            if mesh_tensor.ndim == 0:
                raise ValueError("mesh must be at least 1-dimensional")
            return cls._build_layout_from_mesh(mesh_tensor), cls._build_rank_map_from_mesh(mesh_tensor)

        rank_map_tensor = cls._convert_rank_map_to_tensor(rank_map)
        if layout is None or rank_map_tensor is None:
            raise TypeError("The mesh argument is required except for private _layout/_rank_map construction.")
        if not layout.check_non_overlap():
            raise ValueError(f"Invalid overlapping layout {layout}.")
        return layout, rank_map_tensor

    def _refresh_mesh_view(self) -> None:
        """Materialize the visible mesh tensor and the derived shape/rank metadata."""
        # Compute everything in numpy first so the intermediate ops don't need
        # a real device. Otherwise the call would fail (or SIGSEGV on Ascend)
        # when DeviceMesh is constructed inside a ``ms.DeviceCtx("meta")``
        # block — e.g., from ``DeviceMesh.concatenate`` invoked under
        # ``fully_shard``, which forces fresh ``Tensor()`` constructions onto
        # the meta device and any subsequent op (asnumpy, nonzero, …) crashes.
        rank_map_np = platform.tensor_to_numpy(self._rank_map).reshape(-1)
        full_mesh_np = self._layout.remap_to_numpy(rank_map_np)
        if full_mesh_np.shape[0] == 1:
            per_rank_mesh_np = full_mesh_np[0]
        else:
            coords = np.argwhere(full_mesh_np == self._rank)
            if coords.shape[0] == 0:
                raise RuntimeError(
                    "In order to get the mesh tensor of a DeviceMesh it needs to "
                    "either have all its original dimensions or contain the local rank."
                )
            per_rank_mesh_np = full_mesh_np[coords[0, 0]]
        # Cache the numpy view so ``_compute_coordinate_on_dim`` doesn't need
        # to operate on ``self.mesh`` (which may be on the meta device).
        self._per_rank_mesh_np = per_rank_mesh_np
        self.mesh = Tensor(per_rank_mesh_np.astype(np.int32)).int()
        self._mesh_shape = tuple(per_rank_mesh_np.shape)
        self._rank_list = tuple(per_rank_mesh_np.reshape(-1).tolist())
        self._flatten_rank_map = tuple(rank_map_np.tolist())
        self._dev_num = np.prod(np.array(self._mesh_shape))
        self._dev_rank = len(self._mesh_shape)

    def _set_mesh_dim_names(
            self,
            mesh_dim_names: Union[tuple[str, ...], list[str], None],
    ) -> None:
        """Validate mesh dim names and build lookup tables for named access."""
        self.mesh_dim_names = tuple(mesh_dim_names) if mesh_dim_names else None
        if self.mesh_dim_names is None:
            return

        if len(self._mesh_shape) != len(self.mesh_dim_names):
            raise ValueError(
                f'mesh dimensions ({len(self._mesh_shape)}) should be equal to '
                f'mesh_dim_names length ({len(self.mesh_dim_names)})'
            )
        if len(set(self.mesh_dim_names)) != len(self.mesh_dim_names):
            raise ValueError(f'Each element of mesh_dim_names {self.mesh_dim_names} should be different')
        inter_key = "interleaved_parallel"
        if inter_key in self.mesh_dim_names and self.mesh_dim_names.index(inter_key) != len(self.mesh_dim_names) - 1:
            raise ValueError(
                "'interleaved_parallel' should be at the last dim of mesh_dim_names, means virtual sharding."
            )
        self._dev_name_to_dev_id = {
            name: self._dev_rank - i - 1 for i, name in enumerate(self.mesh_dim_names)
        }
        self._dev_name_to_index = {name: i for i, name in enumerate(self.mesh_dim_names)}

    def _initialize_runtime_state(self, init_backend: bool) -> None:
        """Initialize caches and optional process-group state for the mesh view."""
        self._cache_rank_list_along_axis = {}
        self._global_shape_map = {}
        self._sub_mesh_cache = {}
        self._flatten_mapping: dict[str, 'DeviceMesh'] = {}
        self._ndim = len(self._mesh_shape)
        self._dim_group_backends = (None,) * self._ndim
        self._dim_group_sources = tuple((self, dim) for dim in range(self._ndim))
        self._sub_mesh: List['DeviceMesh'] = []
        if not init_backend:
            return
        self._dim_group_names = self._init_process_groups(
            self._mesh_shape,
            self.mesh_dim_names,
            self._rank_list,
        )

    @staticmethod
    def _build_layout_from_mesh(mesh: Tensor) -> _MeshLayout:
        mesh_shape = tuple(mesh.shape)
        return _MeshLayout(mesh_shape, _contiguous_strides(mesh_shape))

    @staticmethod
    def _build_rank_map_from_mesh(mesh: Tensor) -> Tensor:
        return _host_tensor_from_numpy(platform.tensor_to_numpy(mesh).reshape(-1).astype(np.int32))

    @staticmethod
    def _convert_rank_map_to_tensor(rank_map: Tensor) -> Tensor:
        """Normalize a rank-map input into the flat int32 Tensor stored on the mesh.

        Tensor input is returned as-is to preserve its original device; list /
        tuple / numpy input is built into a fresh flat int32 Tensor.
        """
        if isinstance(rank_map, Tensor):
            # Reuse the existing tensor as-is so we preserve its real device.
            # Going through ``Tensor(np_array)`` would re-create on whatever
            # device context is active (e.g. ``ms.DeviceCtx("meta")`` while
            # ``DeviceMesh.concatenate`` runs under ``fully_shard``), which then
            # breaks the immediate ``asnumpy()`` in ``_refresh_mesh_view``.
            # All in-tree callers that pass a Tensor pass an existing
            # ``DeviceMesh._rank_map`` — already a flat int32 tensor, so no
            # reshape/cast is needed.
            return rank_map
        rank_map_np = np.array(rank_map)
        return _host_tensor_from_numpy(rank_map_np.reshape(-1).astype(np.int32))

    @staticmethod
    def _get_mesh_tensor_from_full_mesh(full_mesh: Tensor, current_rank: Optional[int] = None) -> Tensor:
        """Select the per-rank mesh view from a fully materialized layout remap."""
        if full_mesh.shape[0] == 1:
            return full_mesh[0]

        if current_rank is None:
            current_rank = platform.get_rank()

        rank_coords = (full_mesh == current_rank).nonzero()
        if rank_coords.shape[0] > 0:
            return full_mesh[rank_coords[0, 0]]
        raise RuntimeError(
            "In order to get the mesh tensor of a DeviceMesh it needs to "
            "either have all its original dimensions or contain the local rank."
        )

    def _compute_coordinate_on_dim(self):
        """Compute the current rank coordinates inside this mesh view."""
        # Use the cached numpy view rather than ``self.mesh`` so this works
        # even when the mesh tensor lives on the meta device (DeviceMesh
        # constructed under ``ms.DeviceCtx("meta")`` via ``fully_shard``).
        per_rank_mesh_np = getattr(self, "_per_rank_mesh_np", None)
        if per_rank_mesh_np is not None:
            rank_coords = np.argwhere(per_rank_mesh_np == self._rank)
            if rank_coords.shape[0] not in (0, 1):
                raise AssertionError(
                    f"rank_coords.shape[0] must be 0 or 1, got {rank_coords.shape[0]}"
                )
            if rank_coords.shape[0] == 0:
                return None
            return tuple(int(x) for x in rank_coords[0])
        return self._compute_coordinates_from_mesh(self.mesh, self._rank)

    @staticmethod
    def _compute_coordinates_from_mesh(
            mesh_tensor: Tensor,
            rank: int,
    ):
        """Locate one rank inside a mesh tensor and return its coordinates."""
        rank_coords = (mesh_tensor == rank).nonzero()
        if rank_coords.shape[0] not in (0, 1):
            raise AssertionError(
                f"rank_coords.shape[0] must be 0 or 1, got {rank_coords.shape[0]}"
            )

        if rank_coords.shape[0] == 0:
            return None

        coords = rank_coords[0].tolist()
        return tuple(coords)

    def size(self, mesh_dim=None) -> int:
        if mesh_dim is not None:
            return self.mesh.shape[mesh_dim]
        return self.mesh.numel()

    def get_coordinate(self):
        return self._coordinate_on_dim if self._coordinate_on_dim else None

    def __enter__(self) -> "DeviceMesh":
        _mesh_resources.mesh_stack.append(self)
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        _mesh_resources.mesh_stack.pop()

    @staticmethod
    def _convert_mesh_to_tensor(mesh: Union[Tensor, list, tuple, np.ndarray]) -> Tensor:
        """Convert a public mesh input into an int32 platform tensor."""
        if isinstance(mesh, Tensor):
            mesh = platform.tensor_to_numpy(mesh)
        elif isinstance(mesh, (list, tuple)):
            mesh = np.array(mesh)
        elif not isinstance(mesh, np.ndarray):
            raise TypeError(
                f"mesh must be Tensor, list, tuple or numpy array, but got {type(mesh)}"
            )

        mesh = mesh.astype(np.int32)
        return _host_tensor_from_numpy(mesh)

    @staticmethod
    def _init_one_process_group(mesh_shape: tuple[int, ...], mesh_dim_names: tuple[str, ...],
                                dim_name: str, rank_list: tuple[int, ...]) -> str:
        """Create one process-group family for the named mesh dimension."""
        group_key = None
        split_ranks = set()
        if not isinstance(dim_name, tuple):
            dim_name = (dim_name,)
        for rank in rank_list:
            split_rank = _get_sub_rank_list(mesh_shape, mesh_dim_names, rank_list, dim_name, rank)
            sorted_rank = tuple(sorted(split_rank))
            split_ranks.add(sorted_rank)
            if rank == platform.get_rank():
                group_key = str(sorted_rank)
        split_ranks = sorted([list(item) for item in split_ranks])
        platform.split_group(split_ranks=split_ranks)
        return group_key

    @staticmethod
    def _build_dim_split_ranks(
            sub_layout: _MeshLayout,
            rank_map: Tensor,
    ) -> tuple[list[list[int]], Optional[str]]:
        """Build rank lists and the local cache key for one logical mesh axis."""
        pg_ranks_by_dim = sub_layout.remap_to_numpy(platform.tensor_to_numpy(rank_map))
        current_rank = platform.get_rank()
        split_ranks = []
        split_ranks_set = set()
        group_key = None
        for dim_mesh in np.array(pg_ranks_by_dim):
            subgroup_ranks = tuple(int(rank) for rank in np.array(dim_mesh).reshape(-1).tolist())
            subgroup_ranks_sorted = tuple(sorted(subgroup_ranks))
            if subgroup_ranks_sorted not in split_ranks_set:
                split_ranks_set.add(subgroup_ranks_sorted)
                split_ranks.append(list(subgroup_ranks_sorted))
            if current_rank in subgroup_ranks:
                if group_key is not None:
                    raise RuntimeError(
                        "Each device mesh dimension should get only one process group per rank."
                    )
                group_key = str(subgroup_ranks_sorted)
        split_ranks = sorted(split_ranks)
        return split_ranks, group_key

    @staticmethod
    def _cache_group_if_needed(group_key: Optional[str], group: Any) -> None:
        if group_key is not None and group is not None and group_key not in EXISTING_COMM_GROUPS:
            EXISTING_COMM_GROUPS[group_key] = group

    @staticmethod
    def _init_process_groups_for_layout(
            layout: _MeshLayout,
            rank_map: Tensor,
            mesh_dim_names: Union[tuple[str, ...], None],
            backend_override: Optional[tuple[BackendConfig, ...]] = None,
    ) -> list:
        """Initialize process groups for each top-level axis in the given layout."""
        if mesh_dim_names is None:
            mesh_dim_names = tuple(f"dim_{dim}" for dim in range(len(layout)))
        if backend_override is None:
            backend_override = (None,) * len(layout)
        if len(backend_override) != len(layout):
            raise ValueError(
                f"backend_override length {len(backend_override)} must match layout rank {len(layout)}"
            )

        dim_group_names = []
        for dim, sub_layout in enumerate(layout):
            split_ranks, group_key = DeviceMesh._build_dim_split_ranks(sub_layout, rank_map)
            if _should_defer_group_init(sub_layout, backend_override[dim]):
                dim_group_names.append(None)
                continue
            group = platform.split_group(split_ranks=split_ranks)
            DeviceMesh._cache_group_if_needed(group_key, group)
            dim_group_names.append(group_key)
        return dim_group_names

    @staticmethod
    def _init_process_groups(mesh_shape: tuple[int, ...], mesh_dim_names: Union[tuple[str, ...], None],
                             rank_list: tuple[int, ...],
                             backend_override: Optional[tuple[BackendConfig, ...]] = None) -> list:
        layout = _MeshLayout(mesh_shape, _contiguous_strides(mesh_shape))
        rank_map = DeviceMesh._convert_rank_map_to_tensor(rank_list)
        return DeviceMesh._init_process_groups_for_layout(
            layout,
            rank_map,
            mesh_dim_names,
            backend_override=backend_override,
        )

    @property
    def rank(self):
        return self._rank

    @property
    def mesh_shape(self):
        return self._mesh_shape

    @property
    def rank_list(self):
        return self._rank_list

    @property
    def ndim(self) -> int:
        return self._ndim

    @property
    def shape(self) -> tuple:
        return self._mesh_shape

    @property
    def root_mesh(self) -> Optional['DeviceMesh']:
        return self._root_mesh

    @root_mesh.setter
    def root_mesh(self, value: Optional['DeviceMesh']):
        self._root_mesh = value

    @property
    def sub_mesh(self) -> List['DeviceMesh']:
        return self._sub_mesh

    def get_flatten_mapping(self) -> dict:
        return self._flatten_mapping

    def add_flatten_mapping(self, name: str, mesh: 'DeviceMesh') -> None:
        self._flatten_mapping[name] = mesh

    def __getitem__(self, sub_mesh_dim_names: Union[str, tuple[str, ...]]) -> 'DeviceMesh':
        if not self.mesh_dim_names:
            raise RuntimeError("Cannot slice a DeviceMesh without mesh_dim_names!")

        sub_mesh_dim_names = DeviceMesh._normalize_sub_mesh_dim_names(sub_mesh_dim_names)
        flatten_mapping = self._get_root_mesh().get_flatten_mapping()

        flattened_result = self._try_get_from_flatten_mapping(sub_mesh_dim_names, flatten_mapping)
        if flattened_result is not None:
            return flattened_result

        layout = self._get_slice_mesh_layout(sub_mesh_dim_names)
        if sub_mesh_dim_names in self._sub_mesh_cache:
            return self._sub_mesh_cache[sub_mesh_dim_names]
        if layout == self._layout:
            return self
        return self._create_and_cache_sub_mesh(sub_mesh_dim_names, layout)

    @staticmethod
    def _normalize_sub_mesh_dim_names(sub_mesh_dim_names: Union[str, tuple[str, ...]]) -> tuple[str, ...]:
        """Normalize a slice selector into a non-empty tuple of mesh dim names."""
        if isinstance(sub_mesh_dim_names, str):
            sub_mesh_dim_names = (sub_mesh_dim_names,)

        if not isinstance(sub_mesh_dim_names, tuple):
            raise TypeError(
                f"sub_mesh_dim_names must be str or tuple, but got {type(sub_mesh_dim_names)}"
            )

        if len(sub_mesh_dim_names) == 0:
            raise ValueError("sub_mesh_dim_names cannot be empty")

        return sub_mesh_dim_names

    @staticmethod
    def _try_get_from_flatten_mapping(sub_mesh_dim_names: tuple[str, ...],
                                      flatten_mapping: dict) -> Optional['DeviceMesh']:
        if len(sub_mesh_dim_names) == 1 and sub_mesh_dim_names[0] in flatten_mapping:
            return flatten_mapping[sub_mesh_dim_names[0]]
        return None

    def _get_mesh_dim_by_name(self, mesh_dim_name: str) -> int:
        """Resolve a named mesh axis to its integer position."""
        mesh_dim_names = self.mesh_dim_names or ()
        if len(mesh_dim_names) == 0:
            raise KeyError("No mesh_dim_names found.")
        if mesh_dim_name not in mesh_dim_names:
            raise KeyError(
                f"Mesh dimension '{mesh_dim_name}' does not exist. "
                f"Available mesh dimensions are: {mesh_dim_names}"
            )
        return mesh_dim_names.index(mesh_dim_name)

    def _get_slice_mesh_layout(self, sub_mesh_dim_names: tuple[str, ...]) -> _MeshLayout:
        """Construct the layout corresponding to one named sub-mesh slice request."""
        root_mesh = self._get_root_mesh()
        slice_from_root = self == root_mesh
        flatten_name_to_layout = (
            {key: mesh._layout for key, mesh in root_mesh.get_flatten_mapping().items()}
            if slice_from_root else {}
        )
        valid_dim_names = [*(self.mesh_dim_names or ()), *flatten_name_to_layout]
        if not all(name in valid_dim_names for name in sub_mesh_dim_names):
            raise KeyError(
                f"Invalid mesh_dim_names {sub_mesh_dim_names} specified. "
                f"Valid mesh_dim_names are {valid_dim_names}."
            )

        if all(name in (self.mesh_dim_names or ()) for name in sub_mesh_dim_names):
            indices = [self.mesh_dim_names.index(name) for name in sub_mesh_dim_names]
            if indices != sorted(indices):
                raise ValueError(
                    f"sub_mesh_dim_names {sub_mesh_dim_names} must follow the order of "
                    f"original mesh_dim_names {self.mesh_dim_names}"
                )

        sliced_sizes: list[IntTuple] = []
        sliced_strides: list[IntTuple] = []
        for name in sub_mesh_dim_names:
            if name in (self.mesh_dim_names or ()):
                layout = self._layout[self.mesh_dim_names.index(name)]
            else:
                layout = flatten_name_to_layout[name]
            sliced_sizes.append(layout.sizes)
            sliced_strides.append(layout.strides)

        pre_stride = -1
        for stride in reversed(sliced_strides):
            if not _is_int(stride):
                raise NotImplementedError(
                    "Currently, this only allows slicing out a contiguous flattened dim."
                )
            if stride < pre_stride:
                raise ValueError(
                    f"Invalid mesh_dim_names {sub_mesh_dim_names} specified. "
                    "Mesh dim indices should be in ascending order."
                )
            pre_stride = stride

        if len(sliced_sizes) == 1:
            layout = _MeshLayout(sliced_sizes[0], sliced_strides[0])
        else:
            layout = _MeshLayout(tuple(sliced_sizes), tuple(sliced_strides))
        if not layout.check_non_overlap():
            raise RuntimeError(f"Slicing overlapping dim_names {sub_mesh_dim_names} is not allowed.")
        return layout

    def _create_and_cache_sub_mesh(self, sub_mesh_dim_names: tuple[str, ...], layout: _MeshLayout) -> 'DeviceMesh':
        """Create a sub-mesh view, copy group metadata, and cache the result."""
        root_mesh = self._get_root_mesh()
        sub_mesh = DeviceMesh(
            device_type=self.device_type,
            mesh_dim_names=sub_mesh_dim_names,
            _init_backend=False,
            _layout=layout,
            _rank_map=root_mesh._rank_map,
            _root_mesh=root_mesh,
        )

        slice_dim_group_name = []
        slice_dim_group_backends: list[BackendConfig] = []
        slice_dim_group_sources: list[tuple['DeviceMesh', int]] = []
        for name in sub_mesh_dim_names:
            if name in (self.mesh_dim_names or ()):
                dim_index = self.mesh_dim_names.index(name)
                if hasattr(self, "_dim_group_names"):
                    slice_dim_group_name.append(self._dim_group_names[dim_index])
                slice_dim_group_backends.append(self._dim_group_backends[dim_index])
                if hasattr(self, "_dim_group_sources"):
                    slice_dim_group_sources.append(self._dim_group_sources[dim_index])  # pylint: disable=W0212
                else:
                    slice_dim_group_sources.append((self, dim_index))
            elif name in root_mesh.get_flatten_mapping():
                flatten_mesh = root_mesh.get_flatten_mapping()[name]
                if hasattr(flatten_mesh, "_dim_group_names"):
                    slice_dim_group_name.append(flatten_mesh._dim_group_names[0])
                slice_dim_group_backends.append(flatten_mesh._dim_group_backends[0])
                if hasattr(flatten_mesh, "_dim_group_sources"):
                    slice_dim_group_sources.append(flatten_mesh._dim_group_sources[0])  # pylint: disable=W0212
                else:
                    slice_dim_group_sources.append((flatten_mesh, 0))
        if slice_dim_group_name:
            sub_mesh._dim_group_names = slice_dim_group_name  # pylint: disable=W0212
        if slice_dim_group_backends:
            sub_mesh._dim_group_backends = tuple(slice_dim_group_backends)  # pylint: disable=W0212
        if slice_dim_group_sources:
            sub_mesh._dim_group_sources = tuple(slice_dim_group_sources)  # pylint: disable=W0212

        self._sub_mesh_cache[sub_mesh_dim_names] = sub_mesh
        self.sub_mesh.append(sub_mesh)
        return sub_mesh

    def get_group(self, mesh_dim: Optional[Union[int, str]] = None):
        """Return the communication group for one mesh axis."""
        if not hasattr(self, "_dim_group_names"):
            raise RuntimeError("DeviceMesh process groups not initialized!")

        if self.ndim > 1 and mesh_dim is None:
            raise RuntimeError(
                f"Found the DeviceMesh have {self.ndim} dimensions. "
                "Optional kwarg `mesh_dim` needs to be specified when device_mesh.ndim > 1."
            )

        root_mesh = self._get_root_mesh()
        if isinstance(mesh_dim, str) and mesh_dim in root_mesh.get_flatten_mapping():
            flattened_mesh = root_mesh.get_flatten_mapping()[mesh_dim]
            return flattened_mesh.get_comm_group_by_axis(mesh_dim)

        return self.get_comm_group_by_axis(mesh_dim)

    def get_all_groups(self) -> list:
        if not hasattr(self, "_dim_group_names"):
            raise RuntimeError("DeviceMesh process groups not initialized!")

        return [self.get_group(i) for i in range(self.ndim)]

    @staticmethod
    def from_group(group: Union[Any, list[Any]],
                   device_type: str,
                   mesh: Union[Tensor, list, tuple, np.ndarray] = None,
                   mesh_dim_names: Union[tuple[str, ...], list[str]] = None
                   ) -> 'DeviceMesh':
        """Build a DeviceMesh from an existing process group or a list of groups."""
        if not isinstance(group, list):
            group_ranks = platform.get_process_group_ranks(group)
            group_key = str(tuple(sorted(group_ranks)))
            if not platform.get_created_group(group_ranks):
                EXISTING_COMM_GROUPS[group_key] = group
            if (
                    isinstance(mesh, Tensor) and mesh.tolist() != group_ranks
            ) or (
                    mesh is not None
                    and not isinstance(mesh, Tensor)
                    and mesh != group_ranks
            ):
                raise ValueError(
                    f"Invalid mesh_shape {str(mesh)} for 1D group with ranks {group_ranks}"
                )
            device_mesh = DeviceMesh(device_type, group_ranks, mesh_dim_names=mesh_dim_names, _init_backend=False)
            device_mesh._dim_group_names = [group_key]  # pylint: disable=W0212
            return device_mesh

        groups = list(group)
        if len(groups) == 0:
            raise ValueError("Expect at least one group be specified.")
        if mesh is None:
            raise ValueError("mesh_shape is must specified when group is a list.")
        mesh = DeviceMesh._convert_mesh_to_tensor(mesh)
        if mesh.ndim != len(groups):
            raise ValueError("mesh dimensions must match group dimensions.")
        device_mesh = DeviceMesh(device_type, mesh, mesh_dim_names=mesh_dim_names, _init_backend=False)
        device_mesh._dim_group_names = []  # pylint: disable=W0212
        for dim_group in groups:
            group_ranks = platform.get_process_group_ranks(dim_group)
            group_key = str(tuple(sorted(group_ranks)))
            if not platform.get_created_group(group_ranks):
                EXISTING_COMM_GROUPS[group_key] = dim_group
            device_mesh._dim_group_names.append(group_key)  # pylint: disable=W0212
        return device_mesh

    def get_local_rank(self, mesh_dim: Optional[Union[int, str]] = None) -> int:
        """Return the local coordinate of the current rank along one mesh dimension."""
        if self.ndim > 1 and mesh_dim is None:
            raise RuntimeError(
                f"Found the DeviceMesh have {self.ndim} dimensions. "
                "Optional kwarg `mesh_dim` needs to be specified when device_mesh.ndim > 1."
            )

        if mesh_dim is None:
            mesh_dim = 0

        if isinstance(mesh_dim, str):
            if mesh_dim not in self.mesh_dim_names:  # pylint: disable=E1135
                raise ValueError(
                    f"mesh_dim '{mesh_dim}' not found in mesh_dim_names {self.mesh_dim_names}"
                )
            dim_index = self.mesh_dim_names.index(mesh_dim)
        else:
            if not isinstance(mesh_dim, int) or mesh_dim < 0 or mesh_dim >= self.ndim:
                raise ValueError(
                    f"mesh_dim must be an integer in range [0, {self.ndim}), "
                    f"but got {mesh_dim}"
                )
            dim_index = mesh_dim

        if self._rank not in self._rank_list:
            raise ValueError(
                f"Current rank {self._rank} not found in rank_list {self._rank_list}"
            )

        idx = self._rank_list.index(self._rank)
        coord = [0] * len(self._mesh_shape)
        temp = idx
        for i in range(len(self._mesh_shape) - 1, -1, -1):
            coord[i] = temp % self._mesh_shape[i]
            temp //= self._mesh_shape[i]

        return coord[dim_index]

    def flatten(self, mesh_dim_name: Optional[str] = None) -> 'DeviceMesh':
        return self._create_flatten_mesh(mesh_dim_name)

    def _get_root_mesh(self) -> 'DeviceMesh':
        """Return the canonical root mesh for this view."""
        if self._root_mesh is None:
            return self
        return self._root_mesh._get_root_mesh()  # pylint: disable=protected-access

    @staticmethod
    def _validate_concatenate_inputs(
            meshes: Sequence['DeviceMesh'],
    ) -> tuple['DeviceMesh', tuple['DeviceMesh', ...], tuple[str, ...], tuple[int, ...]]:
        """Validate concatenate inputs and return root metadata plus canonical mesh views."""
        if len(meshes) == 0:
            raise ValueError("DeviceMesh.concatenate expects at least one mesh.")
        if len(meshes) == 1:
            return (
                meshes[0]._get_root_mesh(),
                tuple(meshes),
                tuple(meshes[0].mesh_dim_names or ()),
                meshes[0]._flatten_rank_map,
            )

        # Torch treats the flattened rank map as the common root tensor identity.
        # If a peer view lost root metadata, recover canonical views from any input that still has it.
        root_mesh = next(
            (mesh._get_root_mesh() for mesh in meshes if mesh.root_mesh is not None),  # pylint: disable=protected-access
            meshes[0]._get_root_mesh(),  # pylint: disable=protected-access
        )
        requested_dim_names: list[str] = []
        canonical_meshes: list['DeviceMesh'] = []
        flatten_rank_map = root_mesh._flatten_rank_map  # pylint: disable=protected-access
        anchor_meshes = DeviceMesh._collect_concatenate_anchor_meshes(meshes, root_mesh)
        for mesh in meshes:
            if not mesh.mesh_dim_names:
                raise ValueError("DeviceMesh.concatenate requires mesh_dim_names on every input mesh.")
            if mesh._flatten_rank_map == flatten_rank_map:  # pylint: disable=protected-access
                canonical_mesh = mesh
            else:
                canonical_mesh = DeviceMesh._recover_concatenate_mesh_from_anchors(
                    mesh,
                    anchor_meshes,
                    flatten_rank_map,
                )
                if canonical_mesh is None:
                    raise ValueError("DeviceMesh.concatenate expects all meshes to share the same root mesh.")
            canonical_meshes.append(canonical_mesh)
            requested_dim_names.extend(canonical_mesh.mesh_dim_names)
        return root_mesh, tuple(canonical_meshes), tuple(requested_dim_names), flatten_rank_map

    @staticmethod
    def _collect_concatenate_anchor_meshes(
            meshes: Sequence['DeviceMesh'],
            root_mesh: 'DeviceMesh',
    ) -> list['DeviceMesh']:
        """Collect mesh views that can recover orphaned concatenate inputs by dim name."""
        anchor_meshes: list['DeviceMesh'] = []
        seen_ids: set[int] = set()

        def add_anchor(mesh: Optional['DeviceMesh']) -> None:
            if mesh is None or id(mesh) in seen_ids:
                return
            seen_ids.add(id(mesh))
            anchor_meshes.append(mesh)

        add_anchor(root_mesh)
        for flatten_mesh in root_mesh.get_flatten_mapping().values():
            add_anchor(flatten_mesh)

        for mesh in meshes:
            if mesh.root_mesh is None:
                continue
            add_anchor(mesh)
            add_anchor(mesh._get_root_mesh())  # pylint: disable=protected-access
            for source_mesh, _ in getattr(mesh, "_dim_group_sources", ()):
                if isinstance(source_mesh, DeviceMesh):
                    add_anchor(source_mesh)
                    add_anchor(source_mesh._get_root_mesh())  # pylint: disable=protected-access

        return anchor_meshes

    @staticmethod
    def _recover_concatenate_mesh_from_anchors(
            mesh: 'DeviceMesh',
            anchor_meshes: Sequence['DeviceMesh'],
            flatten_rank_map: tuple[int, ...],
    ) -> Optional['DeviceMesh']:
        """Recover an orphan mesh as a view in the shared root coordinate system."""
        mesh_dim_names = tuple(mesh.mesh_dim_names or ())
        for anchor_mesh in anchor_meshes:
            try:
                candidate = anchor_mesh[mesh_dim_names]
            except (KeyError, ValueError, RuntimeError, NotImplementedError):
                continue
            if (
                    candidate.device_type == mesh.device_type
                    and candidate.mesh_shape == mesh.mesh_shape
                    and candidate.rank_list == mesh.rank_list
                    and candidate._flatten_rank_map == flatten_rank_map  # pylint: disable=protected-access
            ):
                return candidate
        return None

    @staticmethod
    def _validate_concatenate_root_order(root_mesh: 'DeviceMesh', requested_dim_names: tuple[str, ...]) -> None:
        """Require original root dims to stay in root order when concatenating by name."""
        root_dim_names = tuple(root_mesh.mesh_dim_names) if root_mesh.mesh_dim_names else ()
        if not root_dim_names or not all(dim_name in root_dim_names for dim_name in requested_dim_names):
            return

        requested_indices = [root_dim_names.index(dim_name) for dim_name in requested_dim_names]
        if requested_indices != sorted(requested_indices):
            raise ValueError(
                "DeviceMesh.concatenate expects meshes to follow the root mesh order. "
                f"Got root mesh dims {root_dim_names} and requested dims {requested_dim_names}."
            )

    @staticmethod
    def _collect_concatenate_metadata(
            meshes: Sequence['DeviceMesh'],
    ) -> tuple[
        list[str],
        list[IntTuple],
        list[IntTuple],
        list[Optional[str]],
        list[BackendConfig],
        list[tuple['DeviceMesh', int]],
    ]:
        """Collect layout and process-group metadata from all concatenate inputs."""
        concat_dim_names: list[str] = []
        concat_sizes: list[IntTuple] = []
        concat_strides: list[IntTuple] = []
        concat_dim_group_names: list[Optional[str]] = []
        concat_dim_group_backends: list[BackendConfig] = []
        concat_dim_group_sources: list[tuple['DeviceMesh', int]] = []

        for mesh in meshes:
            for dim, sub_layout in enumerate(mesh._layout):  # pylint: disable=protected-access
                concat_sizes.append(sub_layout.sizes)
                concat_strides.append(sub_layout.strides)
                if hasattr(mesh, "_dim_group_names"):
                    concat_dim_group_names.append(mesh._dim_group_names[dim])  # pylint: disable=protected-access
                concat_dim_group_backends.append(mesh._dim_group_backends[dim])  # pylint: disable=protected-access
                if hasattr(mesh, "_dim_group_sources"):
                    concat_dim_group_sources.append(mesh._dim_group_sources[dim])  # pylint: disable=protected-access
                else:
                    concat_dim_group_sources.append((mesh, dim))
            concat_dim_names.extend(mesh.mesh_dim_names)

        if len(set(concat_dim_names)) != len(concat_dim_names):
            raise ValueError(
                f"DeviceMesh.concatenate expects disjoint mesh dims, but got {tuple(concat_dim_names)}."
            )
        return (
            concat_dim_names,
            concat_sizes,
            concat_strides,
            concat_dim_group_names,
            concat_dim_group_backends,
            concat_dim_group_sources,
        )

    @staticmethod
    def _build_concatenate_layout(concat_sizes: list[IntTuple], concat_strides: list[IntTuple]) -> _MeshLayout:
        """Build the layout represented by concatenated top-level mesh axes."""
        if len(concat_sizes) == 1:
            return _MeshLayout(concat_sizes[0], concat_strides[0])
        return _MeshLayout(tuple(concat_sizes), tuple(concat_strides))

    @staticmethod
    def _set_concatenated_group_state(
            mesh: 'DeviceMesh',
            dim_group_names: list[Optional[str]],
            dim_group_backends: list[BackendConfig],
            dim_group_sources: list[tuple['DeviceMesh', int]],
    ) -> None:
        """Attach inherited process-group metadata to a concatenated mesh view."""
        if dim_group_names:
            mesh._dim_group_names = dim_group_names  # pylint: disable=W0212
        if dim_group_backends:
            mesh._dim_group_backends = tuple(dim_group_backends)  # pylint: disable=W0212
        if dim_group_sources:
            mesh._dim_group_sources = tuple(dim_group_sources)  # pylint: disable=W0212

    @staticmethod
    def concatenate(meshes: Sequence['DeviceMesh']) -> 'DeviceMesh':
        """Concatenate multiple sub-mesh views into one wider layout-backed mesh."""
        if len(meshes) == 1:
            return meshes[0]
        root_mesh, canonical_meshes, requested_dim_names, _ = DeviceMesh._validate_concatenate_inputs(meshes)
        DeviceMesh._validate_concatenate_root_order(root_mesh, requested_dim_names)
        (
            concat_dim_names,
            concat_sizes,
            concat_strides,
            concat_dim_group_names,
            concat_dim_group_backends,
            concat_dim_group_sources,
        ) = DeviceMesh._collect_concatenate_metadata(canonical_meshes)
        concat_layout = DeviceMesh._build_concatenate_layout(concat_sizes, concat_strides)
        if not concat_layout.check_non_overlap():
            raise ValueError(f"Cannot concatenate overlapping meshes: {meshes}")

        res_mesh = DeviceMesh(
            root_mesh.device_type,
            mesh_dim_names=tuple(concat_dim_names),
            _init_backend=False,
            _layout=concat_layout,
            _rank_map=root_mesh._rank_map,  # pylint: disable=protected-access
            _root_mesh=root_mesh,
        )
        DeviceMesh._set_concatenated_group_state(
            res_mesh,
            concat_dim_group_names,
            concat_dim_group_backends,
            concat_dim_group_sources,
        )
        return res_mesh

    _concatenate = concatenate

    def _create_flatten_mesh(
            self,
            mesh_dim_name: Optional[str] = None,
            backend_override: BackendConfig = None,
    ) -> 'DeviceMesh':
        """Create or reuse a flattened one-dimensional mesh view."""
        root_mesh = self._get_root_mesh()

        if mesh_dim_name is None:
            mesh_dim_name = "_".join(self.mesh_dim_names)

        if self.ndim == 1 and mesh_dim_name in self.mesh_dim_names:  # pylint: disable=E1135
            return self

        invalid_dim_names = root_mesh.mesh_dim_names
        if mesh_dim_name in invalid_dim_names:
            raise ValueError(
                f"'{mesh_dim_name}' already exists in the root mesh mesh_dim_names "
                f"{invalid_dim_names}. Please specify another valid mesh_dim_name."
            )

        flattened_mesh_layout = self._layout.coalesce()
        if len(flattened_mesh_layout) > 1:
            flattened_mesh_layout = flattened_mesh_layout.nest()

        flatten_mapping = root_mesh.get_flatten_mapping()
        if mesh_dim_name in flatten_mapping:
            cached_mesh = flatten_mapping[mesh_dim_name]
            if cached_mesh._layout == flattened_mesh_layout:  # pylint: disable=protected-access
                return cached_mesh
            raise ValueError(
                f"Flatten mesh with mesh_dim_name '{mesh_dim_name}' has been created "
                f"before with different layout. Please specify another valid mesh_dim_name."
            )

        res_flattened_mesh = DeviceMesh(
            device_type=root_mesh.device_type,
            mesh_dim_names=(mesh_dim_name,),
            _init_backend=False,
            _layout=flattened_mesh_layout,
            _rank_map=root_mesh._rank_map,
            _root_mesh=root_mesh,
        )
        res_flattened_mesh._dim_group_backends = (backend_override,)  # pylint: disable=W0212
        if hasattr(self, "_dim_group_names"):
            res_flattened_mesh._dim_group_names = DeviceMesh._init_process_groups_for_layout(  # pylint: disable=W0212
                res_flattened_mesh._layout,
                root_mesh._rank_map,
                res_flattened_mesh.mesh_dim_names,
                backend_override=(backend_override,),
            )

        root_mesh.add_flatten_mapping(mesh_dim_name, res_flattened_mesh)
        root_mesh._sub_mesh_cache[(mesh_dim_name,)] = res_flattened_mesh  # pylint: disable=W0212
        root_mesh.sub_mesh.append(res_flattened_mesh)

        return res_flattened_mesh

    def _create_unflatten_mesh(
            self,
            dim: int,
            mesh_sizes: tuple[int, ...],
            mesh_dim_names: tuple[str, ...],
            backend_override: tuple[BackendConfig, ...],
    ) -> 'DeviceMesh':
        """Split one logical mesh axis into multiple named axes."""
        inner_layout = _MeshLayout(mesh_sizes, _contiguous_strides(mesh_sizes))
        original_layout = self._layout[dim]
        if inner_layout.numel() != original_layout.numel():
            raise ValueError(
                f"The product of mesh_sizes={mesh_sizes} is {inner_layout.numel()}, "
                f"but the original dimension at dim={dim} has size {original_layout.numel()}."
            )

        partial_layout = original_layout.composition(inner_layout)
        unflattened_layout = self._layout.splice(dim, dim + 1, partial_layout)
        unflattened_mesh_dim_names = list(self.mesh_dim_names or ())
        unflattened_mesh_dim_names[dim: dim + 1] = list(mesh_dim_names)

        root_mesh = self._get_root_mesh()
        res_mesh = DeviceMesh(
            self.device_type,
            mesh_dim_names=tuple(unflattened_mesh_dim_names),
            _init_backend=False,
            _layout=unflattened_layout,
            _rank_map=root_mesh._rank_map,
            _root_mesh=root_mesh,
        )

        dim_group_backends = list(self._dim_group_backends)
        dim_group_backends[dim: dim + 1] = list(backend_override)
        res_mesh._dim_group_backends = tuple(dim_group_backends)  # pylint: disable=W0212

        if hasattr(self, "_dim_group_names"):
            dim_group_names = list(self._dim_group_names)
            dim_group_names[dim: dim + 1] = DeviceMesh._init_process_groups_for_layout(
                partial_layout,
                root_mesh._rank_map,
                mesh_dim_names,
                backend_override=backend_override,
            )
            res_mesh._dim_group_names = dim_group_names  # pylint: disable=W0212

        return res_mesh

    def _flatten(self, mesh_dim_name: Optional[str] = None, backend_override: Any = None) -> 'DeviceMesh':
        return self._create_flatten_mesh(
            mesh_dim_name,
            backend_override=_normalize_backend_value(backend_override),
        )

    def _unflatten(
            self,
            dim: Union[int, str],
            mesh_sizes: tuple[int, ...],
            mesh_dim_names: tuple[str, ...],
            backend_override: Optional[dict[Union[int, str], Any]] = None,
    ) -> 'DeviceMesh':
        """Torch-compatible helper that expands one mesh axis into a nested layout."""
        if isinstance(dim, int):
            if dim < 0 or dim >= self.ndim:
                raise ValueError(f"dim {dim} specified in `_unflatten` is out of range {self.ndim}")
        else:
            mesh_dim_names_tuple = self.mesh_dim_names or ()
            if dim not in mesh_dim_names_tuple:
                raise ValueError(f"dim {dim} specified in `_unflatten` is not in {mesh_dim_names_tuple}")
            dim = mesh_dim_names_tuple.index(dim)

        if len(mesh_sizes) != len(mesh_dim_names):
            raise RuntimeError("mesh_dim_names must have same length as mesh_sizes in _unflatten!")

        backend_override_tuple = (
            _normalize_backend_override(backend_override, len(mesh_sizes), mesh_dim_names)
            if backend_override is not None
            else (None,) * len(mesh_dim_names)
        )
        return self._create_unflatten_mesh(dim, mesh_sizes, mesh_dim_names, backend_override_tuple)

    def assert_axis(self, axis, operate_name):
        if not self.mesh_dim_names:
            raise RuntimeError(f"mesh_dim_names not specified, {operate_name} is not supported.")
        if axis not in self.mesh_dim_names:  # pylint: disable=E1135
            raise ValueError(
                f"The axis name must be one of mesh dim name {self.mesh_dim_names}, but got {axis}"
            )

    def axis_id(self, axis):
        if axis == "None":
            return -1
        self.assert_axis(axis, "axis_id")
        return self._dev_name_to_dev_id[axis]

    def axis_index(self, axis):
        self.assert_axis(axis, "axis_index")
        return self._dev_name_to_index[axis]

    def get_device_num_along_axis(self, axis):
        self.assert_axis(axis, "get_device_num_along_axis")
        return self.mesh_shape[self.mesh_dim_names.index(axis)]

    def get_rank_list_along_axis(self, mesh_dim):
        """Return the ranks that share every other coordinate with the current rank."""
        if mesh_dim in self._cache_rank_list_along_axis:
            return self._cache_rank_list_along_axis[mesh_dim]
        self.assert_axis(mesh_dim, "get_rank_list_along_axis")

        mesh_shape = self.mesh_shape
        mesh_dim_names = self.mesh_dim_names
        rank_list = self.rank_list
        rank = self.rank

        if rank not in rank_list:
            raise ValueError(f"Rank {rank} not found in rank_list")

        idx = rank_list.index(rank)
        coord = [0] * len(mesh_shape)
        temp = idx
        for i in range(len(mesh_shape) - 1, -1, -1):
            coord[i] = temp % mesh_shape[i]
            temp //= mesh_shape[i]

        dim_index = mesh_dim_names.index(mesh_dim)
        strides = [1] * len(mesh_shape)
        for i in range(len(mesh_shape) - 2, -1, -1):
            strides[i] = strides[i + 1] * mesh_shape[i + 1]

        result_ranks = []
        for v in range(mesh_shape[dim_index]):
            new_coord = coord.copy()
            new_coord[dim_index] = v
            new_idx = 0
            for i in range(len(mesh_shape)):
                new_idx += new_coord[i] * strides[i]

            result_ranks.append(rank_list[new_idx])

        self._cache_rank_list_along_axis[mesh_dim] = result_ranks
        return result_ranks

    def get_global_shape(self, slice_shape, tensor_map):
        """Infer the global tensor shape from a shard shape and tensor-map metadata."""
        map_key = hash((slice_shape, tensor_map))
        if map_key in self._global_shape_map:
            return self._global_shape_map[map_key]
        if tensor_map is None:
            raise ValueError(
                "tensor_map is not set. Please configure the tensor map by calling the layout."
            )
        if len(slice_shape) != len(tensor_map):
            raise ValueError(
                f"Length of slice_shape ({len(slice_shape)}) must match "
                f"the length of tensor_map ({len(tensor_map)})."
            )

        n_dims = len(self._mesh_shape)
        factors = [1] * len(slice_shape)

        for dev_idx, size in enumerate(self._mesh_shape):
            reverse_idx = n_dims - 1 - dev_idx
            for axis_idx, mapping in enumerate(tensor_map):
                if isinstance(mapping, int):
                    if mapping == -1:
                        continue
                    if mapping == reverse_idx:
                        factors[axis_idx] *= size
                        break
                elif isinstance(mapping, tuple):
                    if reverse_idx in mapping:
                        factors[axis_idx] *= size
                        break

        global_shape = []
        for i, dim in enumerate(slice_shape):
            global_shape.append(dim * factors[i])
        self._global_shape_map[map_key] = tuple(global_shape)
        return tuple(global_shape)

    def _materialize_dim_group(self, mesh_dim: int) -> Optional[str]:
        """Create a deferred process group for one mesh dimension on first use."""
        if not hasattr(self, "_dim_group_names"):
            self._dim_group_names = [None] * self.ndim  # pylint: disable=W0201

        if hasattr(self, "_dim_group_sources"):
            source_mesh, source_dim = self._dim_group_sources[mesh_dim]  # pylint: disable=W0212
            if source_mesh is not self or source_dim != mesh_dim:
                source_group_key = source_mesh._materialize_dim_group(source_dim)  # pylint: disable=W0212
                self._dim_group_names[mesh_dim] = source_group_key
                return source_group_key

        group_key = self._dim_group_names[mesh_dim]
        if group_key is not None and group_key in EXISTING_COMM_GROUPS:
            return group_key

        split_ranks, group_key = DeviceMesh._build_dim_split_ranks(self._layout[mesh_dim], self._rank_map)
        group = platform.split_group(split_ranks=split_ranks)
        DeviceMesh._cache_group_if_needed(group_key, group)
        self._dim_group_names[mesh_dim] = group_key
        return group_key

    def get_comm_group_by_axis(self, mesh_dim: Union[str, int]):
        """Return the cached or lazily materialized process group for one mesh axis."""
        if self.ndim == 1 and mesh_dim is None:
            mesh_dim = 0

        if isinstance(mesh_dim, str):
            if self.mesh_dim_names is None or len(self.mesh_dim_names) == 0:
                raise ValueError(f"DeviceMesh mesh_dim_names is not set, string mesh_dim {mesh_dim}, is not support.")
            if mesh_dim not in self.mesh_dim_names:  # pylint: disable=E1135
                raise ValueError(
                    f"mesh_dim can pass a string or integer, but string mesh_dim '{mesh_dim}' not found in "
                    f"mesh_dim_names {self.mesh_dim_names}"
                )
            mesh_dim = self.mesh_dim_names.index(mesh_dim)
        else:
            if not isinstance(mesh_dim, int) or mesh_dim < 0 or mesh_dim >= self.ndim:
                raise ValueError(
                    f"mesh_dim can pass a string or integer, if not string, mesh_dim should be a integer in range "
                    f"[0, {self.ndim}), but got {mesh_dim}"
                )

        if not hasattr(self, "_dim_group_names"):
            raise RuntimeError("DeviceMesh process groups not initialized!")

        group_key = self._dim_group_names[mesh_dim]
        if group_key is None or group_key not in EXISTING_COMM_GROUPS:
            group_key = self._materialize_dim_group(mesh_dim)
        if group_key not in EXISTING_COMM_GROUPS:
            raise ValueError(f"{group_key} not in group cache {EXISTING_COMM_GROUPS.keys()}")
        return EXISTING_COMM_GROUPS[group_key]

    def get_devices_for_axis(self, mesh_dim: Union[str, int], rank: int):
        """List peer ranks that share all coordinates except the requested axis."""
        if isinstance(mesh_dim, str):
            if not self.mesh_dim_names:
                raise ValueError("_mesh_dim_names is not set, string mesh_dim is not supported, please pass a integer.")
            mesh_dim_names = self.mesh_dim_names
            if mesh_dim not in mesh_dim_names:  # pylint: disable=E1135
                raise ValueError(f"mesh_dim '{mesh_dim}' not found in mesh_dim_names {mesh_dim_names}")
            mesh_dim = mesh_dim_names.index(mesh_dim)

        mesh_shape = self._mesh_shape
        if mesh_dim < 0 or mesh_dim >= self.ndim:
            raise ValueError(f"mesh_dim {mesh_dim} can not out of range [0, {self.ndim})")
        rank_list = self._rank_list
        if rank not in rank_list:
            raise ValueError(f"Rank {rank} not found in rank_list")

        idx = rank_list.index(rank)
        coord = [0] * len(mesh_shape)
        temp = idx
        for i in range(len(mesh_shape) - 1, -1, -1):
            coord[i] = temp % mesh_shape[i]
            temp //= mesh_shape[i]

        strides = [1] * len(mesh_shape)
        for i in range(len(mesh_shape) - 2, -1, -1):
            strides[i] = strides[i + 1] * mesh_shape[i + 1]

        result_ranks = []
        for v in range(mesh_shape[mesh_dim]):
            new_coord = coord.copy()
            new_coord[mesh_dim] = v
            new_idx = 0
            for i in range(len(mesh_shape)):
                new_idx += new_coord[i] * strides[i]

            result_ranks.append(rank_list[new_idx])

        return result_ranks

    def to_hash(self):
        map_key = (self.mesh_shape, self.mesh_dim_names, self.rank_list)
        return map_key

    def __repr__(self):
        return (
            f"DeviceMesh(device_type='{self.device_type}', mesh_shape={self._mesh_shape}, "
            f"mesh_dim_names={self.mesh_dim_names}, rank_list={self._rank_list})"
        )

    def __str__(self):
        return self.__repr__()

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k in ("_root_mesh", "_dim_group_sources"):
                setattr(result, k, v)
            else:
                setattr(result, k, copy.deepcopy(v, memo))
        return result


_DEVICE_MESH_MAP = {}


def _create_device_mesh(device_type: str,
                        mesh_shape: tuple[int, ...],
                        *,
                        mesh_dim_names: Union[tuple[str, ...], list[str], None] = None,
                        rank_list: tuple[int, ...],
                        init_backend: bool = True, ):
    """Create or reuse a cached DeviceMesh with the requested topology."""
    mesh = np.array(rank_list).reshape(mesh_shape)
    mesh_dim_names = tuple(mesh_dim_names) if mesh_dim_names else None
    map_key = hash((mesh_shape, mesh_dim_names, rank_list))
    if map_key not in _DEVICE_MESH_MAP:
        _DEVICE_MESH_MAP[map_key] = DeviceMesh(device_type, mesh,
                                               mesh_dim_names=mesh_dim_names,
                                               _init_backend=init_backend)
    return _DEVICE_MESH_MAP.get(map_key, None)


def init_device_mesh(
        device_type: str,
        mesh_shape: tuple[int, ...],
        *,
        mesh_dim_names: Union[tuple[str, ...], list[str], None] = None,
        rank_list: Optional[tuple[int, ...]] = None,
        init_backend: bool = True,
) -> DeviceMesh:
    """Initialize a cached DeviceMesh from the provided shape, names, and ranks."""
    total_devices = int(np.prod(np.array(mesh_shape)))
    if rank_list is not None:
        if len(rank_list) != total_devices:
            raise ValueError(
                f"rank_list length ({len(rank_list)}) must equal mesh size ({total_devices})"
            )
    else:
        if init_backend:
            platform.init_process_group()
        try:
            current_rank = platform.get_rank()
        except Exception as exc:
            raise RuntimeError(
                "init_device_mesh: failed to get current rank for automatic rank_list generation. "
                "Either pass rank_list explicitly, or ensure the process group is initialized before calling "
                "init_device_mesh (or set init_backend=True to let init_device_mesh initialize it)."
            ) from exc
        base = current_rank - (current_rank % total_devices)
        rank_list = tuple(range(base, base + total_devices))

    if not isinstance(mesh_shape, tuple):
        raise TypeError(f'mesh_shape must be a tuple, but got {type(mesh_shape)}')

    for size in mesh_shape:
        if not isinstance(size, int) or size <= 0:
            raise ValueError(
                f"Each element of mesh_shape must be a positive integer, but got {mesh_shape}"
            )

    if mesh_dim_names is not None:
        if not isinstance(mesh_dim_names, (tuple, list)):
            raise TypeError(
                f'mesh_dim_names must be a tuple or list, but got {type(mesh_dim_names)}'
            )
        mesh_dim_names = tuple(mesh_dim_names)
        if len(mesh_shape) != len(mesh_dim_names):
            raise ValueError(
                f'mesh_shape ({len(mesh_shape)}) and mesh_dim_names '
                f'({len(mesh_dim_names)}) should have same length'
            )
        if len(set(mesh_dim_names)) != len(mesh_dim_names):
            raise ValueError(f'Each element of mesh_dim_names {mesh_dim_names} should be different')
        if any(not isinstance(name, str) or name == "" for name in mesh_dim_names):
            raise ValueError(f'Each element of mesh_dim_names {mesh_dim_names} should be a non-empty string')

    return _create_device_mesh(
        device_type,
        mesh_shape,
        mesh_dim_names=mesh_dim_names,
        rank_list=rank_list,
        init_backend=init_backend,
    )
