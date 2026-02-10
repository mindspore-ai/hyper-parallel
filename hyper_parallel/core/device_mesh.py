# Copyright 2025 Huawei Technologies Co., Ltd
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

import os
from typing import Optional, Union, List, Any
import numpy as np
from hyper_parallel.platform import get_platform

platform = get_platform()
Tensor = platform.Tensor

_group_map = {}


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
    # Reshape rank list into mesh tensor according to mesh shape
    mesh_tensor = np.array(rank_list).reshape(mesh_shape)

    # Iterate through each dimension of the original mesh
    for dim_index, dim_name in enumerate(mesh_dim_names):

        # Skip dimensions that are included in the sub mesh
        if dim_name in sub_mesh_dim_names:
            continue

        # Split mesh tensor along current dimension
        dim_size = mesh_shape[dim_index]
        sliced_tensors = np.split(mesh_tensor, dim_size, axis=dim_index)

        # Find and keep only the slice containing the current rank
        for sliced_tensor in sliced_tensors:
            rank_exists = np.isin(np.array([current_rank]), sliced_tensor).any()
            if rank_exists:
                mesh_tensor = sliced_tensor
                break

    # Flatten the resulting tensor to get the sub rank list
    sub_rank_list = mesh_tensor.reshape(-1).tolist()
    return sub_rank_list


class DeviceMesh:
    """
    Topological abstraction describing cluster devices.

    Args:
        mesh (Union[Tensor, list, tuple, np.ndarray]): A multi-dimensional array, list, or integer
            tensor describing the device layout. The IDs in the mesh are global IDs of the
            default process group, representing the multi-dimensional networking structure
            of devices in distributed training (e.g., [[0,1],[2,3]] represents a 2x2 device mesh).
            If a list or non-int32 tensor is provided, it will be automatically converted
            to an int32 tensor.
        mesh_dim_names (tuple[str]): A tuple[str] of mesh dim names for each dimension of mesh.

    Attributes:
        ndim (int): Number of dimensions in the mesh.
        mesh_shape (tuple[int]): Shape of the device mesh.
        rank_list (tuple[int]): Flattened list of ranks from the mesh.
        root_mesh (DeviceMesh): The parent mesh if this is a sub mesh, None otherwise.
        sub_mesh (list[DeviceMesh]): List of child meshes created from this mesh.

    Examples:
        >>> # Using Tensor
        >>> mesh = Tensor([[0, 1], [2, 3]])
        >>> device_mesh = DeviceMesh("npu", mesh, nesh_dim_names=("dp", "tp"))
        >>> # Using list
        >>> device_mesh = DeviceMesh("npu", [[0, 1], [2, 3]], nesh_dim_names=("dp", "tp"))
        >>> # Get sub mesh
        >>> dp_mesh = device_mesh["dp"]
        >>> # Access ndim
        >>> print(device_mesh.ndim)  # Output: 2
        >>> print(device_mesh.mesh_shape)  # Output: (2, 2)
        >>> print(device_mesh.rank_list)  # Output: (0, 1, 2, 3)
    """

    def __init__(self,
                 device_type: str,
                 mesh: Union[Tensor, list, tuple, np.ndarray],
                 *,
                 mesh_dim_names: tuple[str, ...],
                 _init_backend: bool = True,
                 ):
        self._device_type = device_type
        # Convert mesh to Tensor with int32 dtype
        mesh = self._convert_mesh_to_tensor(mesh)

        # Validate mesh dimensions
        if mesh.ndim == 0:
            raise ValueError("mesh must be at least 1-dimensional")

        # Extract mesh_shape and rank_list from mesh
        self._mesh_shape = tuple(mesh.shape)
        self._rank_list = tuple(platform.tensor_to_numpy(mesh).flatten().astype(np.int32).tolist())
        self._mesh = mesh

        # Validate mesh_dim_names
        if not isinstance(mesh_dim_names, tuple):
            raise TypeError(f'mesh_dim_names must be tuple type, but got:{type(mesh_dim_names)}')
        if len(self._mesh_shape) != len(mesh_dim_names):
            raise ValueError(
                f'mesh dimensions ({len(self._mesh_shape)}) should be equal to '
                f'mesh_dim_names length ({len(mesh_dim_names)})'
            )
        for in_ele in mesh_dim_names:
            if not isinstance(in_ele, str):
                raise TypeError(
                    f'The element of mesh_dim_names must be str type, but got:{type(in_ele)}'
                )
            if not in_ele:
                raise ValueError("The element of mesh_dim_names can not be empty.")
            if in_ele == "None":
                raise ValueError(
                    "The element of mesh_dim_names can not set 'None', "
                    "because 'None' means no sharding."
                )
        if len(set(mesh_dim_names)) != len(mesh_dim_names):
            raise ValueError(f'Each element of mesh_dim_names {mesh_dim_names} should be different')
        inter_key = "interleaved_parallel"
        if inter_key in mesh_dim_names and mesh_dim_names.index(inter_key) != len(mesh_dim_names) - 1:
            raise ValueError(
                f"When mesh_dim_names {mesh_dim_names} contains keyword 'interleaved_parallel', "
                f"it should be at the last dim of mesh_dim_names, which means the virtual sharding."
            )

        self._mesh_dim_names = mesh_dim_names
        self._dev_num = np.prod(np.array(self._mesh_shape))
        self._rank = platform.get_rank()
        self._dev_rank = len(self._mesh_shape)
        self._dev_name_to_dev_id = {
            name: self._dev_rank - i - 1 for i, name in enumerate(self._mesh_dim_names)
        }
        self._dev_name_to_index = {name: i for i, name in enumerate(self._mesh_dim_names)}
        self._cache_rank_list_along_axis = {}
        self._global_shape_map = {}
        self._sub_mesh_cache = {}
        self._flatten_mapping: dict[str, 'DeviceMesh'] = {}
        self._ndim: int = len(self._mesh_shape)
        self._root_mesh: Optional['DeviceMesh'] = None
        self._sub_mesh: List['DeviceMesh'] = []
        if _init_backend:
            platform.init_process_group()
            self._init_process_groups(self._mesh_shape, self._mesh_dim_names, self._rank_list)
        if os.getenv("MS_SIMULATION_LEVEL") is None:
            self._coordinate_on_dim = self._compute_coordinate_on_dim()

    def _compute_coordinate_on_dim(self):
        # calculate the coordinates of the current global rank on the mesh
        return self._compute_coordinates_from_mesh(self.mesh, self._rank)

    @staticmethod
    def _compute_coordinates_from_mesh(
            mesh_tensor: Tensor,
            rank: int,
    ):
        """
        Compute the coordinates of a rank within a mesh tensor.

        Args:
            mesh_tensor (Tensor): The mesh tensor to search in
            rank (int): The rank to find coordinates for

        Returns:
            A tuple of coordinates if the rank is found in the mesh, None otherwise

        Raises:
            AssertionError: If the rank appears more than once in the mesh
        """
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
        """
        Return the relative indices of this rank relative to all
        dimensions of the mesh. If this rank is not part of the mesh, return None.
        """
        return self._coordinate_on_dim if self._coordinate_on_dim else None

    @staticmethod
    def _convert_mesh_to_tensor(mesh: Union[Tensor, list, tuple, np.ndarray]) -> Tensor:
        """Convert mesh to Tensor with int32 dtype."""
        if isinstance(mesh, Tensor):
            mesh = platform.tensor_to_numpy(mesh)
        elif isinstance(mesh, (list, tuple)):
            mesh = np.array(mesh)
        elif not isinstance(mesh, np.ndarray):
            raise TypeError(
                f"mesh must be Tensor, list, tuple or numpy array, but got {type(mesh)}"
            )

        mesh = mesh.astype(np.int32)
        return Tensor(mesh)

    @staticmethod
    def _init_process_groups(mesh_shape: tuple[int, ...], mesh_dim_names: tuple[str, ...], rank_list: tuple[int, ...]):
        """
        Init process groups. For every dim in mesh_shape, create split group for current rank.

        Args:
            mesh_shape (tuple[int, ...]): Shape of mesh.
            mesh_dim_names (tuple[str, ...]): Names of every dimension of mesh.
            rank_list (tuple[int, ...]): Rank list of current process group worked on.
        """
        for dim in range(len(mesh_shape)):
            split_ranks = set()
            axis = mesh_dim_names[dim]
            if not isinstance(axis, tuple):
                axis = (axis,)
            for rank in rank_list:
                split_rank = _get_sub_rank_list(mesh_shape, mesh_dim_names, rank_list, axis, rank)
                split_ranks.add(tuple(sorted(split_rank)))
            split_ranks = sorted([list(item) for item in split_ranks])
            _group_map[axis] = platform.split_group(split_ranks=split_ranks)

    @property
    def mesh(self) -> Tensor:
        """Get the mesh tensor."""
        return self._mesh

    def device_type(self) -> str:
        """Get the device type."""
        return self._device_type

    @property
    def rank(self):
        return self._rank

    @property
    def mesh_shape(self):
        return self._mesh_shape

    @property
    def mesh_dim_names(self):
        return self._mesh_dim_names

    @property
    def rank_list(self):
        return self._rank_list

    @property
    def ndim(self) -> int:
        return self._ndim

    @property
    def root_mesh(self) -> Optional['DeviceMesh']:
        return self._root_mesh

    @root_mesh.setter
    def root_mesh(self, value: Optional['DeviceMesh']):
        """Set the parent mesh reference."""
        self._root_mesh = value

    @property
    def sub_mesh(self) -> List['DeviceMesh']:
        return self._sub_mesh

    def get_flatten_mapping(self) -> dict:
        """Get the flatten mapping dictionary."""
        return self._flatten_mapping

    def add_flatten_mapping(self, name: str, mesh: 'DeviceMesh') -> None:
        """Add a flattened mesh to the flatten mapping."""
        self._flatten_mapping[name] = mesh

    def __getitem__(self, sub_mesh_dim_names: Union[str, tuple[str, ...]]) -> 'DeviceMesh':
        """
        Get a sub DeviceMesh based on the specified dimension names.

        This method supports both original dimension names and flattened dimension names.
        For example, if a mesh has dimensions ("dp", "cp", "tp") and a flattened mesh
        "dp_cp" was created via flatten(), both mesh["dp"] and mesh["dp_cp"] are valid.

        Args:
            sub_mesh_dim_names: A string or tuple of strings specifying the dimension names
                           for the sub mesh. Can be original dimension names or flattened
                           dimension names registered in the root mesh's flatten_mapping.

        Returns:
            DeviceMesh: A new DeviceMesh representing the sub mesh.

        Raises:
            ValueError: If sub_mesh_dim_names is invalid or not a contiguous prefix.
            KeyError: If sub_mesh_dim_names contains names not in mesh_dim_names or flatten_mapping.

        Examples:
            >>> mesh = platform.tensor([[0, 1], [2, 3]])
            >>> device_mesh = DeviceMesh("npu", mesh, nesh_dim_names=("dp", "tp"))
            >>> dp_mesh = device_mesh["dp"]
            >>> print(dp_mesh.mesh_shape)  # Output: (2,)
            >>> print(dp_mesh.mesh_dim_names)  # Output: ("dp",)
            >>> # After creating a flattened mesh:
            >>> flat_mesh = device_mesh.flatten()
            >>> # Can also access via flattened name:
            >>> same_flat_mesh = device_mesh["dp_tp"]
        """
        sub_mesh_dim_names = self._normalize_sub_mesh_dim_names(sub_mesh_dim_names)
        flatten_mapping = self._get_root_mesh().get_flatten_mapping()

        # Try to get from flatten_mapping first
        flattened_result = self._try_get_from_flatten_mapping(sub_mesh_dim_names, flatten_mapping)
        if flattened_result is not None:
            return flattened_result

        # Validate dimension names
        self._validate_getitem_dimensions(sub_mesh_dim_names, flatten_mapping)

        # Get or create sub mesh for original dimensions
        return self._get_or_create_original_sub_mesh(sub_mesh_dim_names)

    def _normalize_sub_mesh_dim_names(self, sub_mesh_dim_names: Union[str, tuple[str, ...]]) -> tuple[str, ...]:
        """Convert sub_mesh_dim_names to tuple format and validate basic type."""
        if isinstance(sub_mesh_dim_names, str):
            sub_mesh_dim_names = (sub_mesh_dim_names,)

        if not isinstance(sub_mesh_dim_names, tuple):
            raise TypeError(
                f"sub_mesh_dim_names must be str or tuple, but got {type(sub_mesh_dim_names)}"
            )

        if len(sub_mesh_dim_names) == 0:
            raise ValueError("sub_mesh_dim_names cannot be empty")

        return sub_mesh_dim_names

    def _try_get_from_flatten_mapping(self, sub_mesh_dim_names: tuple[str, ...],
                                      flatten_mapping: dict) -> Optional['DeviceMesh']:
        """Try to get mesh from flatten_mapping. Returns None if not applicable."""
        if len(sub_mesh_dim_names) == 1 and sub_mesh_dim_names[0] in flatten_mapping:
            return flatten_mapping[sub_mesh_dim_names[0]]
        return None

    def _validate_getitem_dimensions(self, sub_mesh_dim_names: tuple[str, ...], flatten_mapping: dict):
        """Validate dimension names for __getitem__ operation."""
        valid_dim_names = list(self._mesh_dim_names) + list(flatten_mapping.keys())

        # Validate all names exist
        for name in sub_mesh_dim_names:
            if name not in valid_dim_names:
                raise KeyError(
                    f"Dimension name '{name}' not found in mesh_dim_names {self._mesh_dim_names} "
                    f"or flatten_mapping keys {list(flatten_mapping.keys())}"
                )

        # Check for mixed or multiple flattened dimensions
        original_dims = [name for name in sub_mesh_dim_names if name in self._mesh_dim_names]
        flattened_dims = [name for name in sub_mesh_dim_names if name in flatten_mapping]

        if len(flattened_dims) == len(sub_mesh_dim_names) and len(flattened_dims) > 1:
            raise ValueError(
                f"Slicing multiple flattened dimensions {flattened_dims} simultaneously "
                f"is not supported. Please slice them separately."
            )

        if flattened_dims and original_dims:
            raise ValueError(
                f"Cannot mix original dimensions {original_dims} with flattened dimensions "
                f"{flattened_dims} in a single slice operation."
            )

    def _get_or_create_original_sub_mesh(self, sub_mesh_dim_names: tuple[str, ...]) -> 'DeviceMesh':
        """Get or create sub mesh for original (non-flattened) dimensions."""
        # Validate dimension order
        indices = [self._mesh_dim_names.index(name) for name in sub_mesh_dim_names]
        if indices != sorted(indices):
            raise ValueError(
                f"sub_mesh_dim_names {sub_mesh_dim_names} must follow the order of "
                f"original mesh_dim_names {self._mesh_dim_names}"
            )

        # Check cache
        if sub_mesh_dim_names in self._sub_mesh_cache:
            return self._sub_mesh_cache[sub_mesh_dim_names]

        # Return self if requesting all dimensions
        if len(sub_mesh_dim_names) == len(self._mesh_dim_names):
            return self

        # Create new sub mesh
        return self._create_and_cache_sub_mesh(sub_mesh_dim_names, indices)

    def _create_and_cache_sub_mesh(self, sub_mesh_dim_names: tuple[str, ...],
                                   indices: List[int]) -> 'DeviceMesh':
        """Create a new sub mesh and cache it."""
        sub_mesh_shape = tuple(self._mesh_shape[i] for i in indices)

        sub_rank_list = _get_sub_rank_list(
            self._mesh_shape,
            self._mesh_dim_names,
            self._rank_list,
            sub_mesh_dim_names,
            self._rank
        )
        sub_rank_list = tuple(sub_rank_list)

        # Create sub mesh tensor using Tensor()
        sub_mesh_tensor = Tensor(sub_rank_list).reshape(sub_mesh_shape)

        # Create sub mesh
        sub_mesh = DeviceMesh(
            device_type="npu",
            mesh=sub_mesh_tensor,
            mesh_dim_names=sub_mesh_dim_names,
            _init_backend=False
        )
        # Set root mesh reference
        sub_mesh.root_mesh = self._get_root_mesh()

        # Cache and track
        self._sub_mesh_cache[sub_mesh_dim_names] = sub_mesh
        # Add to sub_mesh list
        self.sub_mesh.append(sub_mesh)

        return sub_mesh

    def get_group(self, mesh_dim: Optional[Union[int, str]] = None):
        """
        Get the communication group for a specific mesh dimension.

        Args:
            mesh_dim: The dimension index or name. If None and mesh is 1D,
                     returns the only group. If None and mesh is multi-dimensional,
                     raises an error.

        Returns:
            The process group for the specified dimension.

        Raises:
            RuntimeError: If mesh_dim is None and mesh has more than 1 dimension.
            ValueError: If mesh_dim is invalid.

        Examples:
            >>> mesh = Tensor([[0, 1], [2, 3]])
            >>> device_mesh = DeviceMesh("npu", mesh, nesh_dim_names=("dp", "tp"))
            >>> dp_group = device_mesh.get_group("dp")
            >>> # or by index
            >>> dp_group = device_mesh.get_group(0)
        """
        if self.ndim > 1 and mesh_dim is None:
            raise RuntimeError(
                f"Found the DeviceMesh have {self.ndim} dimensions. "
                "Optional kwarg `mesh_dim` needs to be specified when device_mesh.ndim > 1."
            )

        # Quick return if the current device_mesh is a 1D mesh
        if self.ndim == 1 and mesh_dim is None:
            mesh_dim = 0

        # Check if mesh_dim is a flattened dimension name in root mesh's flatten_mapping
        root_mesh = self._get_root_mesh()
        if isinstance(mesh_dim, str) and mesh_dim in root_mesh.get_flatten_mapping():
            # Return the group from the flattened mesh
            flattened_mesh = root_mesh.get_flatten_mapping()[mesh_dim]
            return flattened_mesh.get_comm_group_by_axis(mesh_dim)

        # Convert string to axis name
        if isinstance(mesh_dim, str):
            if mesh_dim not in self._mesh_dim_names:
                raise ValueError(
                    f"mesh_dim '{mesh_dim}' not found in mesh_dim_names {self._mesh_dim_names}"
                )
            axis = mesh_dim
        else:
            if not isinstance(mesh_dim, int) or mesh_dim < 0 or mesh_dim >= self.ndim:
                raise ValueError(
                    f"mesh_dim must be an integer in range [0, {self.ndim}), "
                    f"but got {mesh_dim}"
                )
            axis = self._mesh_dim_names[mesh_dim]

        return self.get_comm_group_by_axis(axis)

    @staticmethod
    def from_group(group: Union[Any, list[Any]],
                   mesh: Union[Tensor, list, tuple, np.ndarray] = None,
                   mesh_dim_names: tuple[str, ...] = None
                   ) -> 'DeviceMesh':
        """
        Create device mesh from group or group list.

        Args:
            group: The group or group list to create device mesh from.
            mesh:
                For 1d group, mesh can pass None. If group is 1d and mesh is not None, the mesh must equal to
                group_ranks get from group, or must be a tensor which tolist value equal to group_ranks.
                For nd group, mesh must be passed.
            mesh_dim_names: Names of every mesh dimension.
        """
        if not isinstance(group, list):
            group_ranks = platform.get_process_group_ranks(group)
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
            return DeviceMesh("npu", group_ranks, mesh_dim_names=mesh_dim_names, _init_backend=False)

        if len(group) == 0:
            raise ValueError("Expect at least one group be specified.")
        if mesh is None:
            raise ValueError("mesh_shape is must specified when group is a list.")
        mesh = DeviceMesh._convert_mesh_to_tensor(mesh)
        if mesh.ndim != len(group):
            raise ValueError("mesh dimensions must match group dimensions.")
        return DeviceMesh("npu", mesh, mesh_dim_names=mesh_dim_names, _init_backend=False)

    def get_local_rank(self, mesh_dim: Optional[Union[int, str]] = None) -> int:
        """
        Get the local rank within a specific mesh dimension.

        Args:
            mesh_dim: The dimension index or name. If None and mesh is 1D,
                     uses dimension 0. If None and mesh is multi-dimensional,
                     raises an error.

        Returns:
            int: The local rank within the specified dimension.

        Raises:
            RuntimeError: If mesh_dim is None and mesh has more than 1 dimension.
            ValueError: If mesh_dim is invalid or current rank not in rank_list.

        Examples:
            >>> mesh = Tensor([[0, 1, 2, 3], [4, 5, 6, 7]])
            >>> device_mesh = DeviceMesh("npu", mesh, nesh_dim_names=("dp", "tp"))
            >>> # On rank 0
            >>> print(device_mesh.get_local_rank("dp"))  # Output: 0
            >>> print(device_mesh.get_local_rank("tp"))  # Output: 0
        """
        if self.ndim > 1 and mesh_dim is None:
            raise RuntimeError(
                f"Found the DeviceMesh have {self.ndim} dimensions. "
                "Optional kwarg `mesh_dim` needs to be specified when device_mesh.ndim > 1."
            )

        if mesh_dim is None:
            mesh_dim = 0

        # Convert string to index
        if isinstance(mesh_dim, str):
            if mesh_dim not in self._mesh_dim_names:
                raise ValueError(
                    f"mesh_dim '{mesh_dim}' not found in mesh_dim_names {self._mesh_dim_names}"
                )
            dim_index = self._mesh_dim_names.index(mesh_dim)
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

        # Calculate the coordinate of current rank in the mesh
        idx = self._rank_list.index(self._rank)
        coord = [0] * len(self._mesh_shape)
        temp = idx
        for i in range(len(self._mesh_shape) - 1, -1, -1):
            coord[i] = temp % self._mesh_shape[i]
            temp //= self._mesh_shape[i]

        return coord[dim_index]

    def flatten(self, mesh_dim_name: Optional[str] = None) -> 'DeviceMesh':
        """
        Returns a 1D DeviceMesh by flattening the current DeviceMesh.

        Args:
            mesh_dim_name (str, optional): The name for the flattened dimension.
                If not provided, the name will be generated by joining the original
                mesh dim names with underscore (e.g., "dp_tp" for ("dp", "tp")).
                This name will be used as the key in the root mesh's flatten_mapping.

        Returns:
            DeviceMesh: A 1D DeviceMesh with flattened dimensions.

        Raises:
            ValueError: If mesh_dim_name conflicts with existing mesh dim names.

        Examples:
            >>> mesh = Tensor([[0, 1], [2, 3]])
            >>> device_mesh = DeviceMesh("npu", mesh, nesh_dim_names=("dp", "tp"))
            >>> # Using default name
            >>> flat_mesh = device_mesh.flatten()
            >>> print(flat_mesh.mesh_dim_names)  # Output: ("dp_tp",)
            >>> # Using custom name
            >>> flat_mesh = device_mesh.flatten(mesh_dim_name="custom_name")
            >>> print(flat_mesh.mesh_dim_names)  # Output: ("custom_name",)
        """
        return self._create_flatten_mesh(mesh_dim_name)

    def _get_root_mesh(self) -> 'DeviceMesh':
        """Get the root mesh of this DeviceMesh."""
        if self._root_mesh is None:
            return self
        # pylint: disable=protected-access
        return self._root_mesh._get_root_mesh()

    def _create_flatten_mesh(self, mesh_dim_name: Optional[str] = None) -> 'DeviceMesh':
        """Create a flattened 1D mesh from the current mesh.

        Args:
            mesh_dim_name (str, optional): The name for the flattened dimension.
                If not provided, defaults to joining mesh dim names with underscore.
        """
        root_mesh = self._get_root_mesh()

        # Generate mesh_dim_name by joining mesh dim names if not provided
        if mesh_dim_name is None:
            mesh_dim_name = "_".join(self._mesh_dim_names)

        # Flatten a 1D device mesh into its original mesh_dim_names will return itself
        if self.ndim == 1 and mesh_dim_name in self._mesh_dim_names:
            return self

        # Check whether the mesh_dim_name for flattened mesh is valid
        # It should not conflict with existing mesh dim names in root mesh
        invalid_dim_names = root_mesh.mesh_dim_names
        if mesh_dim_name in invalid_dim_names:
            raise ValueError(
                f"'{mesh_dim_name}' already exists in the root mesh mesh_dim_names "
                f"{invalid_dim_names}. Please specify another valid mesh_dim_name."
            )

        # Quick return if the flatten mesh has been created before with same layout
        flatten_mapping = root_mesh.get_flatten_mapping()
        if mesh_dim_name in flatten_mapping:
            cached_mesh = flatten_mapping[mesh_dim_name]
            # Verify the cached mesh has the expected flattened size
            expected_size = int(np.prod(self._mesh_shape))
            if cached_mesh.mesh_shape == (expected_size,):
                return cached_mesh
            raise ValueError(
                f"Flatten mesh with mesh_dim_name '{mesh_dim_name}' has been created "
                f"before with different layout. Please specify another valid mesh_dim_name."
            )

        # Calculate the flattened mesh properties
        flattened_mesh_dim = (mesh_dim_name,)

        # Create flattened mesh tensor using Tensor()
        flattened_mesh_tensor = Tensor(self._rank_list)

        # Create the flattened mesh
        res_flattened_mesh = DeviceMesh(
            device_type="npu",
            mesh=flattened_mesh_tensor,
            mesh_dim_names=flattened_mesh_dim
        )
        # Set root mesh reference to the actual root mesh
        res_flattened_mesh.root_mesh = root_mesh

        # Cache the flattened mesh in root mesh's flatten_mapping
        root_mesh.add_flatten_mapping(mesh_dim_name, res_flattened_mesh)
        root_mesh.sub_mesh.append(res_flattened_mesh)

        return res_flattened_mesh

    def axis_id(self, axis):
        if axis == "None":
            return -1
        if axis not in self.mesh_dim_names:
            raise ValueError(
                f"The axis name must be one of mesh shape mesh dim name {self.mesh_dim_names}), "
                f"but got {axis}"
            )
        return self._dev_name_to_dev_id[axis]

    def axis_index(self, axis):
        if axis not in self.mesh_dim_names:
            raise ValueError(
                f"The axis name must be one of mesh shape mesh dim name {self.mesh_dim_names}), "
                f"but got {axis}"
            )
        return self._dev_name_to_index[axis]

    def get_device_num_along_axis(self, axis):
        """Return device num along specify device axis"""
        if axis not in self.mesh_dim_names:
            raise ValueError(
                f"The axis must be one of device mesh dim name: {self.mesh_dim_names}, but got {axis}"
            )
        return self.mesh_shape[self.mesh_dim_names.index(axis)]

    def get_rank_list_along_axis(self, axis):
        """
        Get the repeat rank list when the axis is not shard.

        Args:
            axis (str): Axis name.

        Returns:
            list: reduce rank list
        """
        if axis in self._cache_rank_list_along_axis:
            # short cut, get rank list from cache
            return self._cache_rank_list_along_axis[axis]

        mesh_shape = self.mesh_shape
        mesh_dim_names = self.mesh_dim_names
        rank_list = self.rank_list
        rank = self.rank

        if axis not in mesh_dim_names:
            raise ValueError(f"Axis '{axis}' not found in mesh_dim_names {mesh_dim_names}")

        if rank not in rank_list:
            raise ValueError(f"Rank {rank} not found in rank_list")

        idx = rank_list.index(rank)
        coord = [0] * len(mesh_shape)
        temp = idx
        for i in range(len(mesh_shape) - 1, -1, -1):
            coord[i] = temp % mesh_shape[i]
            temp //= mesh_shape[i]

        dim_index = mesh_dim_names.index(axis)
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

        self._cache_rank_list_along_axis[axis] = result_ranks
        return result_ranks

    def get_global_shape(self, slice_shape, tensor_map):
        """get global shape"""
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

    def get_comm_group_by_axis(self, axis: Union[str, tuple[str, ...]]):
        """
        Get common group for all ranks in same axis.

        Args:
            axis: Axis name.

        Return:
            group: group of specified axis.
        """
        if axis is None:
            raise ValueError("axis is not set. Please configure the axis name when calling get_comm_group_by_axis.")
        if not isinstance(axis, tuple):
            axis = (axis,)
        if axis not in _group_map:
            raise ValueError(f"axis {axis} not in _group_map {_group_map.keys()}")
        return _group_map[axis]

    def get_devices_for_axis(self, axis, rank):
        """
        Get the repeat rank list when the axis is not shard.

        Args:
            axis (str): Axis name.
            rank (int): Global rank

        Returns:
            list: reduce rank list
        """
        mesh_shape = self._mesh_shape
        mesh_dim_names = self._mesh_dim_names
        rank_list = self._rank_list

        if axis not in mesh_dim_names:
            raise ValueError(f"Axis '{axis}' not found in mesh_dim_names {mesh_dim_names}")

        if rank not in rank_list:
            raise ValueError(f"Rank {rank} not found in rank_list")

        idx = rank_list.index(rank)
        coord = [0] * len(mesh_shape)
        temp = idx
        for i in range(len(mesh_shape) - 1, -1, -1):
            coord[i] = temp % mesh_shape[i]
            temp //= mesh_shape[i]

        dim_index = mesh_dim_names.index(axis)
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

        return result_ranks

    def to_hash(self):
        rank_ids = (self.rank_list[0], self.rank_list[-1])
        map_key = (self.mesh_shape, self.mesh_dim_names, rank_ids)
        return map_key

    def __repr__(self):
        """__repr__"""
        return (
            f"DeviceMesh(device_type='npu', mesh_shape={self._mesh_shape}, "
            f"mesh_dim_names={self._mesh_dim_names}, rank_list={self._rank_list})"
        )

    def __str__(self):
        """__str__"""
        return self.__repr__()


_DEVICE_MESH_MAP = {}


def _create_device_mesh(device_type: str,
                        mesh_shape: tuple[int, ...],
                        *,
                        mesh_dim_names: tuple[str, ...],
                        rank_list: tuple[int, ...],
                        init_backend: bool = True, ):
    """
    Create or retrieve a cached DeviceMesh.

    Args:
        device_type (str): Device type.
        mesh_shape (Tensor): A multi dimension tensor describing the device layout.
        mesh_dim_names (tuple[str]): A tuple of mesh dim names for each dimension.
        rank_list (tuple[int]): A tuple of rank.
        init_backend (bool): Whether to initialize the device mesh.

    Returns:
        DeviceMesh: A DeviceMesh object.
    """
    mesh = np.array(rank_list).reshape(mesh_shape)
    rank_ids = (rank_list[0], rank_list[-1])
    map_key = hash((mesh_shape, mesh_dim_names, rank_ids))
    if map_key not in _DEVICE_MESH_MAP:
        _DEVICE_MESH_MAP[map_key] = DeviceMesh(device_type, mesh,
                                               mesh_dim_names=mesh_dim_names,
                                               _init_backend=init_backend)
    return _DEVICE_MESH_MAP.get(map_key, None)


def init_device_mesh(
        device_type: str,
        mesh_shape: tuple[int, ...],
        *,
        mesh_dim_names: tuple[str, ...],
        rank_list: Optional[tuple[int, ...]] = None,
        init_backend: bool = True,
) -> DeviceMesh:
    """
    Initialize a DeviceMesh based on mesh_shape and mesh_dim_names parameters.

    This function creates a DeviceMesh with an n-dimensional array layout, where n is the
    length of mesh_shape. Each dimension is labeled with the corresponding mesh_dim_names.
    When rank_list is not provided, it is generated so that the current rank is included
    (e.g. for onecard/simulation: base, base+1, ..., base+n-1 where base aligns to mesh size).

    Compared to directly constructing DeviceMesh, init_device_mesh provides:
    - Automatic mesh array generation from mesh_shape
    - Caching mechanism to reuse existing DeviceMesh objects
    - Validation of parameters

    Args:
        mesh_shape (tuple[int]): A tuple describing the dimensions of the multi-dimensional
            array that describes the layout of devices. For example, (2, 4) creates
            a 2D mesh with 2 rows and 4 columns.
        mesh_dim_names (tuple[str, ...]): A tuple of names to assign to each dimension of the mesh.
            Its length must match the length of mesh_shape. Each string must be unique.
        rank_list (tuple[int], optional): Flattened list of ranks for the mesh. When None,
            generated so that the current process rank is included (for onecard/simulation).
        device_type (str): The type of device to create.
        init_backend (bool): Whether to initialize the backend.

    Returns:
        DeviceMesh: A DeviceMesh object representing the device layout.

    Raises:
        TypeError: If mesh_shape or mesh_dim_names is not a tuple.
        ValueError: If mesh_shape and mesh_dim_names have different lengths.
        ValueError: If mesh_dim_names contains duplicate or empty strings.

    Examples:
        >>> # Create a 2D mesh with shape (2, 2)
        >>> device_mesh = init_device_mesh(
        ...     device_type="npu",
        ...     mesh_shape=(2, 2),
        ...     mesh_dim_names=("dp", "tp")
        ... )
        >>> print(device_mesh.mesh_shape)  # Output: (2, 2)
        >>> print(device_mesh.mesh_dim_names)  # Output: ("dp", "tp")
        >>> print(device_mesh.rank_list)  # Output: (0, 1, 2, 3)

        >>> # Get sub mesh
        >>> dp_mesh = device_mesh["dp"]
        >>> print(dp_mesh.mesh_shape)  # Output: (2,)

        >>> # Create a larger mesh
        >>> mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))
        >>> print(mesh.rank_list)  # Output: (0, 1, 2, 3, 4, 5, 6, 7)
    """
    # Validate mesh_shape
    if not isinstance(mesh_shape, tuple):
        raise TypeError(f"mesh_shape must be tuple type, but got: {type(mesh_shape)}")

    # Validate mesh_dim_names
    if not isinstance(mesh_dim_names, tuple):
        raise TypeError(f"mesh_dim_names must be tuple type, but got: {type(mesh_dim_names)}")

    if len(mesh_shape) != len(mesh_dim_names):
        raise ValueError(
            f"mesh_shape length ({len(mesh_shape)}) should be equal to "
            f"mesh_dim_names length ({len(mesh_dim_names)})"
        )

    # Generate rank_list: use provided or build one that includes current rank
    total_devices = int(np.prod(np.array(mesh_shape)))
    if rank_list is not None:
        if len(rank_list) != total_devices:
            raise ValueError(
                f"rank_list length ({len(rank_list)}) must equal mesh size ({total_devices})"
            )
        rank_list = tuple(rank_list)
    else:
        current_rank = platform.get_rank()
        base = current_rank - (current_rank % total_devices)
        rank_list = tuple(base + i for i in range(total_devices))

    # Use the caching mechanism
    return _create_device_mesh(device_type, mesh_shape, mesh_dim_names=mesh_dim_names, rank_list=rank_list,
                               init_backend=init_backend)
