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

from typing import Optional, Tuple, Union, List
import numpy as np
from hyper_parallel.platform import get_platform

platform = get_platform()
Tensor = platform.Tensor
dtype = platform.tensor_dtype

_group_map = {}


def _get_sub_rank_list(mesh_shape, alias_name, rank_list, sub_mesh_dim_names, current_rank):
    """
    Get the sub rank list for a sub mesh.

    Args:
        mesh_shape (tuple[int]): The shape of the original mesh.
        alias_name (tuple[str]): The alias names of the original mesh dimensions.
        rank_list (tuple[int]): A tuple of ranks that participate in this mesh.
        sub_mesh_dim_names (tuple[str]): The alias names of the sub mesh to extract.
        current_rank (int): The current process rank.

    Returns:
        list: The sub rank list for the sub mesh.
    """
    # Reshape rank list into mesh tensor according to mesh shape
    mesh_tensor = np.array(rank_list).reshape(mesh_shape)

    # Iterate through each dimension of the original mesh
    for dim_index, dim_name in enumerate(alias_name):

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
        mesh (Union[Tensor, list, np.ndarray]): A multi-dimensional array, list, or integer
            tensor describing the device layout. The IDs in the mesh are global IDs of the
            default process group, representing the multi-dimensional networking structure
            of devices in distributed training (e.g., [[0,1],[2,3]] represents a 2x2 device mesh).
            If a list or non-int32 tensor is provided, it will be automatically converted
            to an int32 tensor.
        alias_name (Tuple[str]): A tuple of alias names for each dimension of mesh.

    Attributes:
        ndim (int): Number of dimensions in the mesh.
        mesh_shape (tuple[int]): Shape of the device mesh.
        rank_list (tuple[int]): Flattened list of ranks from the mesh.
        root_mesh (DeviceMesh): The parent mesh if this is a sub mesh, None otherwise.
        sub_mesh (list[DeviceMesh]): List of child meshes created from this mesh.

    Examples:
        >>> # Using Tensor
        >>> mesh = Tensor([[0, 1], [2, 3]])
        >>> device_mesh = DeviceMesh(mesh, ("dp", "tp"))
        >>> # Using list
        >>> device_mesh = DeviceMesh([[0, 1], [2, 3]], ("dp", "tp"))
        >>> # Get sub mesh
        >>> dp_mesh = device_mesh["dp"]
        >>> # Access ndim
        >>> >>> print(device_mesh.ndim)  # Output: 2.
        >>> print(device_mesh.mesh_shape)  # Output: (2, 2)
        >>> print(device_mesh.rank_list)  # Output: (0, 1, 2, 3)
    """
    def __init__(self,
                 mesh: Union[Tensor, list, np.ndarray],
                 alias_name: Tuple[str]):
        # Convert mesh to Tensor with int32 dtype
        mesh = self._convert_mesh_to_tensor(mesh)

        # Validate mesh dimensions
        if mesh.ndim == 0:
            raise ValueError("mesh must be at least 1-dimensional")

        # Extract mesh_shape and rank_list from mesh
        self._mesh_shape = tuple(mesh.shape)
        self._rank_list = tuple(mesh.flatten().tolist())
        self._mesh = mesh

        # Validate alias_name
        if not isinstance(alias_name, tuple):
            raise TypeError(f'alias_name must be tuple type, but got:{type(alias_name)}')
        if len(self._mesh_shape) != len(alias_name):
            raise ValueError(
                f'mesh dimensions ({len(self._mesh_shape)}) should be equal to '
                f'alias_name length ({len(alias_name)})'
            )
        for in_ele in alias_name:
            if not isinstance(in_ele, str):
                raise TypeError(
                    f'The element of alias_name must be str type, but got:{type(in_ele)}'
                )
            if not in_ele:
                raise ValueError("The element of alias_name can not be empty.")
            if in_ele == "None":
                raise ValueError(
                    "The element of alias_name can not set 'None', "
                    "because 'None' means no sharding."
                )
        if len(set(alias_name)) != len(alias_name):
            raise ValueError(f'Each element of alias_name {alias_name} should be different')
        inter_key = "interleaved_parallel"
        if inter_key in alias_name and alias_name.index(inter_key) != len(alias_name) - 1:
            raise ValueError(
                f"When alias_name {alias_name} contains keyword 'interleaved_parallel', "
                f"it should be at the last dim of alias_name, which means the virtual sharding."
            )

        self._alias_name = alias_name
        self._dev_num = np.prod(np.array(self._mesh_shape))
        self._rank = platform.get_rank()
        self._dev_rank = len(self._mesh_shape)
        self._dev_name_to_dev_id = {
            name: self._dev_rank - i - 1 for i, name in enumerate(self._alias_name)
        }
        self._dev_name_to_index = {name: i for i, name in enumerate(self._alias_name)}
        self._cache_rank_list_along_axis = {}
        self._global_shape_map = {}
        self._sub_mesh_cache = {}
        self._flatten_mapping: dict[str, 'DeviceMesh'] = {}
        self._ndim: int = len(self._mesh_shape)
        self._root_mesh: Optional['DeviceMesh'] = None
        self._sub_mesh: List['DeviceMesh'] = []

    @staticmethod
    def _convert_mesh_to_tensor(mesh: Union[Tensor, list, np.ndarray]) -> Tensor:
        """Convert mesh to Tensor with int32 dtype."""
        if isinstance(mesh, (list, np.ndarray)):
            mesh = Tensor(mesh)
        elif not isinstance(mesh, Tensor):
            raise TypeError(
                f"mesh must be Tensor, list, or numpy array, but got {type(mesh)}"
            )
        # Convert to int32 if needed
        if mesh.dtype != dtype.int32:
            mesh = mesh.to(dtype.int32)
        return mesh

    @property
    def mesh(self) -> Tensor:
        """Get the mesh tensor."""
        return self._mesh

    @property
    def rank(self):
        return self._rank

    @property
    def mesh_shape(self):
        return self._mesh_shape

    @property
    def alias_name(self):
        return self._alias_name

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

    def __getitem__(self, sub_alias_name: Union[str, Tuple[str, ...]]) -> 'DeviceMesh':
        """
        Get a sub DeviceMesh based on the specified dimension names.

        Args:
            sub_alias_name: A string or tuple of strings specifying the dimension names
                           for the sub mesh.

        Returns:
            DeviceMesh A new DeviceMesh representing the sub mesh.

        Raises:
            ValueError: If sub_alias_name is invalid or not a contiguous prefix.
            KeyError: If sub_alias_name contains names not in alias_name.

        Examples:
            >>> mesh = platform.tensor([[0, 1], [2, 3]])
            >>> device_mesh = DeviceMesh(mesh, ("dp", "tp"))
            >>> dp_mesh = device_mesh["dp"]
            >>> print(dp_mesh.mesh_shape)  # Output: (2,)
            >>> print(dp_mesh.alias_name)  # Output: ("dp",)
        """
        # Convert string to tuple for uniform handling
        if isinstance(sub_alias_name, str):
            sub_alias_name = (sub_alias_name,)

        if not isinstance(sub_alias_name, tuple):
            raise TypeError(
                f"sub_alias_name must be str or tuple, but got {type(sub_alias_name)}"
            )

        if len(sub_alias_name) == 0:
            raise ValueError("sub_alias_name cannot be empty")

        # Validate all names exist in alias_name
        for name in sub_alias_name:
            if name not in self._alias_name:
                raise KeyError(
                    f"Dimension name '{name}' not found in alias_name {self._alias_name}"
                )

        # Check if sub_alias_name is a contiguous prefix of alias_name
        # Get the indices of sub_alias_name in alias_name
        indices = [self._alias_name.index(name) for name in sub_alias_name]
        # Ensure the sub_alias_name follows the same order as the original alias_name
        if indices != sorted(indices):
            raise ValueError(
                f"sub_alias_name {sub_alias_name} must follow the order of "
                f"original alias_name {self._alias_name}"
            )

        # Check cache first
        cache_key = sub_alias_name
        if cache_key in self._sub_mesh_cache:
            return self._sub_mesh_cache[cache_key]

        # If requesting all dimensions, return self
        if len(sub_alias_name) == len(self._alias_name):
            return self

        # Calculate sub mesh shape
        sub_mesh_shape = tuple(self._mesh_shape[i] for i in indices)

        # Calculate sub rank list using the provided algorithm
        sub_rank_list = _get_sub_rank_list(
            self._mesh_shape,
            self._alias_name,
            self._rank_list,
            sub_alias_name,
            self._rank
        )
        sub_rank_list = tuple(sub_rank_list)

        # Create sub mesh tensor using Tensor()
        sub_mesh_tensor = Tensor(sub_rank_list).reshape(sub_mesh_shape)

        # Create sub mesh
        sub_mesh = DeviceMesh(
            mesh=sub_mesh_tensor,
            alias_name=sub_alias_name
        )
        # Set root mesh reference
        sub_mesh.root_mesh = self

        # Cache the sub mesh
        self._sub_mesh_cache[cache_key] = sub_mesh
        # Add to sub_mesh list
        self.sub_mesh.append(sub_mesh)

        return sub_mesh

    def get_group(self, mesh_dim: Optional[Union[int, str]] = None):
        """
        Get the communication group for a specific mesh dimension.

        Args:
            mesh_dim (Optional[Union[int, str]], optional): The dimension index or name. If None and mesh is 1D,
                     returns the only group. If None and mesh is multi-dimensional,
                     raises an error. Default: None.

        Returns:
            The process group for the specified dimension.

        Raises:
            RuntimeError: If mesh_dim is None and mesh has more than 1 dimension.
            ValueError: If mesh_dim is invalid.

        Examples:
            >>> mesh = Tensor([[0, 1], [2, 3]])
            >>> device_mesh = DeviceMesh(mesh, ("dp", "tp"))
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
            return flattened_mesh.get_comm_group_by_axis(mesh_dim, self._rank)

        # Convert string to axis name
        if isinstance(mesh_dim, str):
            if mesh_dim not in self._alias_name:
                raise ValueError(
                    f"mesh_dim '{mesh_dim}' not found in alias_name {self._alias_name}"
                )
            axis = mesh_dim
        else:
            if not isinstance(mesh_dim, int) or mesh_dim < 0 or mesh_dim >= self.ndim:
                raise ValueError(
                    f"mesh_dim must be an integer in range [0, {self.ndim}), "
                    f"but got {mesh_dim}"
                )
            axis = self._alias_name[mesh_dim]

        return self.get_comm_group_by_axis(axis, self._rank)

    def get_local_rank(self, mesh_dim: Optional[Union[int, str]] = None) -> int:
        """
        Get the local rank within a specific mesh dimension.

        Args:
            mesh_dim: The dimension index or name. If None and mesh is 1D,
                     uses dimension 0. If None and mesh is multi-dimensional,
                     raises an error.

        Returns:
            int The local rank within the specified dimension.

        Raises:
            RuntimeError: If mesh_dim is None and mesh has more than 1 dimension.
            ValueError: If mesh_dim is invalid or current rank not in rank_list.

        Examples:
            >>> mesh = Tensor([[0, 1, 2, 3], [4, 5, 6, 7]])
            >>> device_mesh = DeviceMesh(mesh, ("dp", "tp"))
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
            if mesh_dim not in self._alias_name:
                raise ValueError(
                    f"mesh_dim '{mesh_dim}' not found in alias_name {self._alias_name}"
                )
            dim_index = self._alias_name.index(mesh_dim)
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

    def flatten(self) -> 'DeviceMesh':
        """
        Returns a 1D DeviceMesh by flattening the current DeviceMesh.

        Returns:
            DeviceMesh: A 1D DeviceMesh with flattened dimensions.

        Raises:
            ValueError: If generated mesh_dim_name conflicts with existing alias names.

        Examples:
            >>> mesh = Tensor([[0, 1], [2, 3]])
            >>> device_mesh = DeviceMesh(mesh, ("dp", "tp"))
            >>> flat_mesh = device_mesh.flatten()
            >>> print(flat_mesh.mesh_shape)  # Output: (4,)
            >>> print(flat_mesh.alias_name)  # Output: ("dp_tp",)
        """
        return self._create_flatten_mesh()

    def _get_root_mesh(self) -> 'DeviceMesh':
        """Get the root mesh of this DeviceMesh."""
        if self._root_mesh is None:
            return self
        # pylint: disable=protected-access
        return self._root_mesh._get_root_mesh()

    def _create_flatten_mesh(self) -> 'DeviceMesh':
        """Create a flattened 1D mesh from the current mesh."""
        root_mesh = self._get_root_mesh()

        # Generate mesh_dim_name by joining alias names
        mesh_dim_name = "_".join(self._alias_name)

        # Flatten a 1D device mesh into its original alias_name will return itself
        if self.ndim == 1 and mesh_dim_name in self._alias_name:
            return self

        # Check whether the mesh_dim_name for flattened mesh is valid
        # It should not conflict with existing alias names in root mesh
        invalid_dim_names = root_mesh.alias_name
        if mesh_dim_name in invalid_dim_names:
            raise ValueError(
                f"'{mesh_dim_name}' already exists in the root mesh alias_name "
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
        flattened_alias = (mesh_dim_name,)

        # Create flattened mesh tensor using Tensor()
        flattened_mesh_tensor = Tensor(self._rank_list)

        # Create the flattened mesh
        res_flattened_mesh = DeviceMesh(
            mesh=flattened_mesh_tensor,
            alias_name=flattened_alias
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
        if axis not in self.alias_name:
            raise ValueError(
                f"The axis name must be one of mesh shape alias name {self.alias_name}), "
                f"but got {axis}"
            )
        return self._dev_name_to_dev_id[axis]

    def axis_index(self, axis):
        if axis not in self.alias_name:
            raise ValueError(
                f"The axis name must be one of mesh shape alias name {self.alias_name}), "
                f"but got {axis}"
            )
        return self._dev_name_to_index[axis]

    def get_device_num_along_axis(self, axis):
        """Return device num along specify device axis"""
        if axis not in self.alias_name:
            raise ValueError(
                f"The axis must be one of device alias name: {self.alias_name}, but got {axis}"
            )
        return self.mesh_shape[self.alias_name.index(axis)]

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
        alias_name = self.alias_name
        rank_list = self.rank_list
        rank = self.rank

        if axis not in alias_name:
            raise ValueError(f"Axis '{axis}' not found in alias_name {alias_name}")

        if rank not in rank_list:
            raise ValueError(f"Rank {rank} not found in rank_list")

        idx = rank_list.index(rank)
        coord = [0] * len(mesh_shape)
        temp = idx
        for i in range(len(mesh_shape) - 1, -1, -1):
            coord[i] = temp % mesh_shape[i]
            temp //= mesh_shape[i]

        dim_index = alias_name.index(axis)
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

    def get_comm_group_by_axis(self, axis, rank):
        if (axis, rank) in _group_map:
            return _group_map[(axis, rank)]
        rank_list = self.get_rank_list_along_axis(axis=axis)
        group = platform.create_group(rank_list)
        _group_map[(axis, rank)] = group
        return group

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
        alias_name = self._alias_name
        rank_list = self._rank_list

        if axis not in alias_name:
            raise ValueError(f"Axis '{axis}' not found in alias_name {alias_name}")

        if rank not in rank_list:
            raise ValueError(f"Rank {rank} not found in rank_list")

        idx = rank_list.index(rank)
        coord = [0] * len(mesh_shape)
        temp = idx
        for i in range(len(mesh_shape) - 1, -1, -1):
            coord[i] = temp % mesh_shape[i]
            temp //= mesh_shape[i]

        dim_index = alias_name.index(axis)
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
        map_key = (self.mesh_shape, self.alias_name, rank_ids)
        return map_key

    def __repr__(self):
        """__repr__"""
        return (
            f"DeviceMesh(mesh_shape={self._mesh_shape}, "
            f"alias_name={self._alias_name}, rank_list={self._rank_list})"
        )

    def __str__(self):
        """__str__"""
        return self.__repr__()


_DEVICE_MESH_MAP = {}


def _create_device_mesh(mesh: Tensor, alias_name: Tuple[str]):
    """
    Create or retrieve a cached DeviceMesh.

    Args:
        mesh (Tensor): A multi-dimensional tensor describing the device layout.
        alias_name (tuple[str]): A tuple of alias names for each dimension.

    Returns:
        DeviceMesh: A DeviceMesh object.
    """
    mesh_shape = tuple(mesh.shape)
    rank_list = tuple(mesh.flatten().tolist())
    rank_ids = (rank_list[0], rank_list[-1])
    map_key = hash((mesh_shape, alias_name, rank_ids))
    if map_key not in _DEVICE_MESH_MAP:
        _DEVICE_MESH_MAP[map_key] = DeviceMesh(mesh, alias_name)
    return _DEVICE_MESH_MAP.get(map_key, None)


def init_device_mesh(
    mesh_shape: Tuple[int, ...],
    alias_name: Tuple[str, ...]
) -> DeviceMesh:
    """
    Initialize a DeviceMesh based on mesh_shape and alias_name parameters.

    This function creates a DeviceMesh with an n-dimensional array layout, where n is the
    length of mesh_shape. Each dimension is labeled with the corresponding alias_name.
    The rank_list is automatically generated as sequential ranks (0, 1, 2, ..., n-1).

    Compared to directly constructing DeviceMesh, init_device_mesh provides:
    - Automatic mesh array generation from mesh_shape
    - Caching mechanism to reuse existing DeviceMesh objects
    - Validation of parameters

    Args:
        mesh_shape (Tuple[int, ...]): A tuple describing the dimensions of the multi-dimensional
            array that describes the layout of devices. For example, (2, 4) creates
            a 2D mesh with 2 rows and 4 columns.
        alias_name (Tuple[str, ...]): A tuple of names to assign to each dimension of the mesh.
            Its length must match the length of mesh_shape. Each string must be unique.

    Returns:
        DeviceMesh: A DeviceMesh object representing the device layout.

    Raises:
        TypeError: If mesh_shape or alias_name is not a tuple.
        ValueError: If mesh_shape and alias_name have different lengths.
        ValueError: If alias_name contains duplicate or empty strings.

    Examples:
        >>> # Create a 2D mesh with shape (2, 2)
        >>> device_mesh = init_device_mesh(
        ...     mesh_shape=(2, 2),
        ...     alias_name=("dp", "tp")
        ... )
        >>> print(device_mesh.mesh_shape)  # Output: (2, 2)
        >>> print(device_mesh.alias_name)  # Output: ("dp", "tp")
        >>> print(device_mesh.rank_list)  # Output: (0, 1, 2, 3)

        >>> # Get sub mesh
        >>> dp_mesh = device_mesh["dp"]
        >>> print(dp_mesh.mesh_shape)  # Output: (2,)

        >>> # Create a larger mesh
        >>> mesh = init_device_mesh(mesh_shape=(2, 4), alias_name=("dp", "tp"))
        >>> print(mesh.rank_list)  # Output: (0, 1, 2, 3, 4, 5, 6, 7)
    """
    # Validate mesh_shape
    if not isinstance(mesh_shape, tuple):
        raise TypeError(f"mesh_shape must be tuple type, but got: {type(mesh_shape)}")

    # Validate alias_name
    if not isinstance(alias_name, tuple):
        raise TypeError(f"alias_name must be tuple type, but got: {type(alias_name)}")

    if len(mesh_shape) != len(alias_name):
        raise ValueError(
            f"mesh_shape length ({len(mesh_shape)}) should be equal to "
            f"alias_name length ({len(alias_name)})"
        )

    # Generate mesh tensor from mesh_shape with sequential ranks using Tensor()
    total_devices = int(np.prod(np.array(mesh_shape)))
    rank_list = list(range(total_devices))
    mesh = Tensor(rank_list).reshape(mesh_shape)

    # Use the caching mechanism
    return _create_device_mesh(mesh, alias_name)
