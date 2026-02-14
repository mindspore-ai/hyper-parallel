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
"""layout"""

import copy
import functools
import numpy as np

from hyper_parallel.core.placement_types import Placement, Shard, Replicate, Partial
from hyper_parallel.core.device_mesh import DeviceMesh, _create_device_mesh
from hyper_parallel.platform import get_platform

platform = get_platform()


def _infer_slice_area_by_rank(mesh_shape, tensor_map, rank_id: int, full_shape: tuple):  # -> tuple[tuple[int]]:
    """Return the range of each axis from full tensor for slice in current rank."""

    def _get_dev_num_alone_dim(mesh_shape, dim):
        """_get_dev_num_alone_dim."""
        return mesh_shape[-dim - 1] if dim != -1 else 1

    def _rank_id_to_dev_id_list(mesh_shape, rank_id):
        """Infer dev id list by rank_id and mesh_shape"""
        dims = len(mesh_shape)
        dev_id_list = [0] * dims
        for i in range(dims - 1, -1, -1):
            dev_id_list[i] = rank_id % mesh_shape[i]
            rank_id = rank_id // mesh_shape[i]
        return dev_id_list

    dev_id_list = _rank_id_to_dev_id_list(mesh_shape, rank_id)

    dims = len(full_shape)
    area = []
    for axis in range(dims):
        mapping = tensor_map[axis]
        if isinstance(mapping, int):
            mapping = (mapping,)
        split_num = 1
        for dim in mapping:
            split_num *= _get_dev_num_alone_dim(mesh_shape, dim)

        slice_id = 0
        coef = 1
        for dim in reversed(mapping):
            if dim == -1:
                continue
            slice_id += dev_id_list[-dim - 1] * coef
            coef *= _get_dev_num_alone_dim(mesh_shape, dim)
        slice_size = full_shape[axis] // split_num
        start = slice_id * slice_size
        end = start + slice_size
        area.append((start, end))
    return area


def _get_slice_tensor_by_layout(global_tensor, layout):
    """Transfer global tensor to local tensor by layout"""
    inner_rank_id = layout.rank_list.index(layout.mesh.rank)
    slice_area = _infer_slice_area_by_rank(layout.mesh_shape, layout.tensor_map, inner_rank_id, global_tensor.shape)

    def get_slice_data(full_data, offset):
        area = ()
        for begin, end in offset:
            area += (slice(begin, end),)
        return full_data[area]

    local_tensor = get_slice_data(global_tensor, slice_area)
    return local_tensor


def _infer_slice_shape_by_layout(global_shape, layout):
    """Infer slice shape from global_shape and layout"""
    slice_shape = list(global_shape)
    alias_tensor_map = layout.alias_tensor_map
    for i in range(len(global_shape)):
        axis_name = alias_tensor_map[i]
        if isinstance(axis_name, str):
            axis_name = (axis_name,)
        for sub_axis_name in axis_name:
            if sub_axis_name != "None":
                slice_shape[i] = slice_shape[i] // layout.mesh.get_device_num_along_axis(sub_axis_name)
    return slice_shape


class Layout:
    """
    Topological abstraction describing cluster devices for tensor slice placement on the cluster.

    Note:
        - It is valid only in semi auto parallel or auto parallel mode.
        - The multiplication result of the `mesh_shape` must be equal to the device count in a pipeline stage.
        - When the layout function is invoked to constructs a sharding strategy, each alias name is only allowed to be
          used once to shard a tensor.

    Args:
        mesh_shape (tuple): Describe the shape of devices arrangement, its element type is int.
        alias_name (tuple): The alias name for each axis of mesh_shape, its length shoits element type is string.
                            When using "interleaved_parallel" as an alias name, the tensor would be split into multiple
                            copies on the corresponding partition dimension on a single card.
        rank_list (tuple, optional): Data is allocated to the device according to rank_list. Default: ``None``.

    Raises:
        TypeError: `mesh_shape` is not a tuple type.
        TypeError: `alias_name` is not a tuple type.
        TypeError: 'rank_list' is not a list type.
        ValueError: `mesh_shape` length is not equal to `alias_name` length.
        TypeError: The element of `mesh_shape` is not int type.
        TypeError: The element of `alias_name` is not a str type.
        TypeError: The element of `rank_list` is not int type.
        ValueError: The element of `alias_name` is an empty str.
        ValueError: The element of `alias_name` is "None".
        ValueError: `alias_name` contains repeated element.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> from mindspore.parallel import Layout
        >>> layout = Layout((2, 2, 2), ("dp", "sp", "mp"))
        >>> layout0 = layout("dp", "mp")
        >>> print(layout0.to_dict())
        {"mesh_shape": (2, 2, 2), "tensor_map": (2, 0), "interleaved_parallel": False,
        'alias_name': {'dp', 'sp', 'mp'}, "rank_list": [0, 1, 2, 3, 4, 5, 6, 7]}
        >>> layout = Layout((2, 2, 2), ("dp", "sp", "interleaved_parallel"))
        >>> layout1 = layout(("dp", "interleaved_parallel"), "sp")
    """

    def __init__(self, mesh_shape, alias_name, rank_list=None, init_backend=True):
        self._alias_name = alias_name
        self._tensor_map = None
        if not rank_list:
            self._rank_list = tuple(range(np.prod(np.array(mesh_shape))))
        else:
            self._rank_list = tuple(rank_list)
        self._partial = [None] * len(mesh_shape)  # partial status for each dev dim
        self._support_partial_op = ['sum', 'max', 'min', 'avg', 'prod', 'all', None]
        self._alias_tensor_map = None
        self._mesh = _create_device_mesh("npu", mesh_shape, mesh_dim_names=alias_name, rank_list=self._rank_list,
                                         init_backend=init_backend)
        self._compact_str = self._to_compact_string()
        self._placements = None

    @classmethod
    def from_device_mesh(cls, device_mesh: DeviceMesh) -> 'Layout':
        """
        Create a Layout from an existing DeviceMesh.

        Args:
            device_mesh (DeviceMesh): The device mesh to create layout from.

        Returns:
            Layout: A new Layout instance initialized with the properties of the provided device mesh.

        Examples:
            >>> from hyper_parallel.core.layout import Layout, DeviceMesh
            >>> device_mesh = DeviceMesh("npu", (2, 2), nesh_dim_names=("dp", "mp"))
            >>> layout = Layout.from_device_mesh(device_mesh)
        """
        obj = cls.__new__(cls)
        obj._mesh = device_mesh
        obj._alias_name = device_mesh.mesh_dim_names
        obj._rank_list = device_mesh.rank_list
        obj._tensor_map = None
        obj._partial = [None] * len(device_mesh.mesh_shape)
        obj._support_partial_op = ['sum', 'max', 'min', 'avg', 'prod', 'all', None]
        obj._alias_tensor_map = None
        obj._placements = None
        obj._compact_str = obj._to_compact_string()
        return obj

    def __call__(self, *alias_tensor_map):
        obj = copy.deepcopy(self)

        if len(alias_tensor_map) == 1 and isinstance(alias_tensor_map[0], (list, tuple)):
            if len(alias_tensor_map[0]) > 0 and isinstance(alias_tensor_map[0][0], Placement):
                return self._process_placement_layout(obj, alias_tensor_map[0])

        if len(alias_tensor_map) > 0 and isinstance(alias_tensor_map[0], Placement):
            return self._process_placement_layout(obj, alias_tensor_map)

        return self._process_alias_layout(obj, alias_tensor_map)

    def _process_placement_layout(self, obj, placements):
        """Process layout defined by Placement types."""
        obj.set_placements(placements)
        return copy.deepcopy(obj)

    def _process_alias_layout(self, obj, alias_tensor_map):
        """Process layout defined by alias strings."""
        obj.set_alias_tensor_map(alias_tensor_map)
        tensor_map = ()
        writed_map = ()
        for ele in alias_tensor_map:
            if isinstance(ele, tuple):
                ele_map = ()
                for item in ele:
                    if item == "None":
                        ele_map += (-1,)
                        continue
                    if item not in obj.alias_name:
                        raise ValueError(f'The axis {item} is not found in {obj.alias_name}')
                    if item in writed_map:
                        raise ValueError(f'The axis {item} has been set more than one in {obj.alias_name}')
                    ele_map += (len(obj.alias_name) - 1 - obj.alias_name.index(item),)
                    writed_map += (item,)
                tensor_map += (ele_map,)
                continue
            if ele == "None":
                tensor_map += (-1,)
                continue
            if ele not in obj.alias_name:
                raise ValueError(f'The axis {ele} is not found in {obj.alias_name}')
            if ele in writed_map:
                raise ValueError(f'The axis {ele} has been set more than one in {obj.alias_name}')
            tensor_map += (len(obj.alias_name) - 1 - obj.alias_name.index(ele),)
            writed_map += (ele,)
        obj.set_tensor_map(tensor_map)
        obj.tensor_map_to_placement()
        obj.update_compact_str()
        return copy.deepcopy(obj)

    def to_dict(self):
        """
        Transform layout to a dictionary.
        """
        if self._mesh.mesh_shape is None:
            raise ValueError("The device_shape of layout is None")
        if self._tensor_map is None:
            raise ValueError("The tensor_map of layout is None")
        interleaved_parallel = "interleaved_parallel" in self._mesh.mesh_dim_names
        return {"mesh_shape": self._mesh.mesh_shape, "tensor_map": self._tensor_map,
                "interleaved_parallel": interleaved_parallel, "alias_name": self._mesh.mesh_dim_names,
                "rank_list": self._rank_list}

    def placement_to_tensor_map(self, dim):
        """
        Transform placement to tensor map.

        This method converts the `placements` configuration (consisting of Shard, Replicate, Partial)
        into a `tensor_map` representation used for distributed tensor operations.

        Args:
            dim (int): The dimension of the tensor. Must be a positive integer.

        Returns:
            tuple: A tuple representing the tensor map, where each element corresponds to a tensor dimension.
                   A value of -1 indicates the dimension is not sharded, while other values indicate
                   the mesh dimension index along which the tensor dimension is sharded.

        Raises:
            ValueError: If `dim` is negative.
            ValueError: If a shard dimension in `placements` is out of bounds for the given tensor dimension.
            ValueError: If a tensor dimension is sharded by multiple mesh axes.
        """
        if dim < 0:
            raise ValueError(f"Tensor dimension must be positive, but got {dim}")
        if dim == 0:
            return self._handle_zero_dim_placement()

        dim_map = self._build_dim_map_from_placements(dim)
        tensor_map = self._convert_dim_map_to_tensor_map(dim_map)
        self.set_tensor_map(tuple(tensor_map))
        self._alias_tensor_map = self._build_readable_tensor_map()
        self.update_compact_str()
        return tensor_map

    def _handle_zero_dim_placement(self):
        """Handle the special case of zero-dimensional tensor."""
        self.set_tensor_map(())
        self._alias_tensor_map = ()
        for mesh_idx, placement in enumerate(self.placements):
            if isinstance(placement, Partial):
                self._partial[mesh_idx] = self._extract_reduce_op(placement)
        return []

    def _build_dim_map_from_placements(self, dim):
        """Build dimension map from placements."""
        dim_map = [-1] * dim
        self.partial_ops = {}
        for mesh_idx, placement in enumerate(self.placements):
            if isinstance(placement, Shard):
                shard_dim = placement.dim
                if shard_dim < -dim or shard_dim >= dim:
                    raise ValueError(f"Shard dimension {shard_dim} is out of bounds for tensor of dimension {dim}")
                if shard_dim < 0:
                    shard_dim += dim
                if dim_map[shard_dim] != -1:
                    raise ValueError(f"Dimension {shard_dim} has been sharded by Mesh axis {dim_map[shard_dim]}")
                dim_map[shard_dim] = mesh_idx
            elif isinstance(placement, Partial):
                self._partial[mesh_idx] = self._extract_reduce_op(placement)
        return dim_map

    def _extract_reduce_op(self, placement):
        """Extract reduce operation name from Partial placement."""
        op_name = getattr(placement, "reduce_op", "sum")
        if isinstance(op_name, str):
            op_name = op_name.lower()
        return op_name

    def _convert_dim_map_to_tensor_map(self, dim_map):
        """Convert dimension map to tensor map format."""
        device_dim_count = len(self.mesh_shape)
        return [
            device_dim_count - 1 - mesh_idx if mesh_idx != -1 else -1
            for mesh_idx in dim_map
        ]

    def _build_readable_tensor_map(self):
        """Build human-readable alias tensor map from tensor_map."""
        readable_map = []
        for item in self._tensor_map:
            if self._mesh.mesh_dim_names is None:
                readable_map.append("None")
            elif isinstance(item, tuple):
                mapped_tuple = tuple(
                    self._mesh.mesh_dim_names[len(self._mesh.mesh_dim_names) - 1 - dim] if dim != -1 else "None"
                    for dim in item
                )
                readable_map.append(mapped_tuple)
            else:
                readable_map.append(
                    self._mesh.mesh_dim_names[len(self._mesh.mesh_dim_names) - 1 - item] if item != -1 else "None"
                )
        return tuple(readable_map)

    def tensor_map_to_placement(self):
        """
        Transform tensor map to placement.

        This method converts the existing `tensor_map` and `partial` status into a list of `Placement` objects
        (Shard, Replicate, Partial). This is the inverse operation of `placement_to_tensor_map`.

        Returns:
            list[Placement]: A list of Placement objects describing the distribution strategy for each
                             dimension of the device mesh.

        Raises:
            ValueError: If `tensor_map` is not configured (None).
        """
        if self._tensor_map is None:
            raise ValueError("The tensor_map is None, cannot transform to placements.")
        mesh_ndim = len(self.mesh_shape)
        placements = [Replicate()] * mesh_ndim
        for tensor_dim, mapping in enumerate(self._tensor_map):
            mapping_list = mapping if isinstance(mapping, tuple) else (mapping,)
            for map_val in mapping_list:
                if map_val != -1:
                    root_mesh_idx = mesh_ndim - 1 - map_val
                    placements[root_mesh_idx] = Shard(dim=tensor_dim)
        for mesh_idx, op in enumerate(self.partial):
            if op is not None:
                placements[mesh_idx] = Partial(reduce_op=op)
        self.set_placements(placements)
        return placements

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.update_mesh()

    @property
    def mesh(self):
        return self._mesh

    def update_mesh(self):
        self._mesh = _create_device_mesh("npu", self.mesh_shape, mesh_dim_names=self.alias_name,
                                         rank_list=self.rank_list)

    @property
    def rank_list(self):
        """rank list"""
        return self._rank_list

    @rank_list.setter
    def rank_list(self, val):
        self._rank_list = val

    @property
    def mesh_shape(self):
        """mesh shape"""
        return self._mesh.mesh_shape

    @property
    def alias_name(self):
        """alias name"""
        return self._mesh.mesh_dim_names

    @property
    def alias_tensor_map(self):
        return self._alias_tensor_map

    def set_alias_tensor_map(self, alias_tensor_map):
        """Set alias_tensor_map"""
        self._alias_tensor_map = alias_tensor_map

    @property
    def placements(self):
        """placements"""
        return self._placements

    def set_placements(self, placements):
        """Set placements."""
        self._placements = placements

    @property
    def tensor_map(self):
        """tensor map"""
        return self._tensor_map

    def set_tensor_map(self, tensor_map):
        """Set tensor_map."""
        self._tensor_map = tensor_map

    @property
    def partial(self):
        """partial status"""
        return self._partial

    def set_partial_by_dev_axis(self, axis, op):
        """Set the partial status for the specified dev ID, means pending to do reduce by op."""
        if op not in self._support_partial_op:
            raise ValueError(f"Partial op must be one of {self._support_partial_op}, but got {op}")
        if self.is_dev_axis_apply_shard(axis):
            raise ValueError("Partial dim must be replicate.")
        self._partial[self._mesh.axis_index(axis)] = op
        self.tensor_map_to_placement()
        self.update_compact_str()

    def get_partial_by_dev_id(self, axis):
        """Get the partial status for the specified dev id"""
        return self.partial[self._mesh.axis_index(axis)]

    def is_dev_axis_apply_shard(self, axis):
        """Return true if device axis is applying shard"""
        axis_id = self._mesh.axis_id(axis)

        def flatten(input_x):
            flatten_res = []
            for item in input_x:
                if isinstance(item, tuple):
                    flatten_res.extend(flatten(item))
                else:
                    flatten_res.append(item)
            return flatten_res

        flatten_tensor_map = flatten(self.tensor_map)
        return axis_id in flatten_tensor_map

    def get_dev_axis_apply_shard_axis(self, axis):
        """Return the axis which be split by axis. If axis not be apply to shard, return None."""
        for dim, dim_map in enumerate(self.alias_tensor_map):
            if (isinstance(dim_map, tuple) and axis in dim_map) or axis == dim_map:
                return dim
        return None

    def reset_partial(self):
        self._partial = [None] * len(self.mesh_shape)
        self.tensor_map_to_placement()
        self.update_compact_str()

    def is_partial(self):
        """Return true if any dim in mesh_shape is partial"""
        return any(self.partial)

    def get_global_shape(self, slice_shape):
        """get global shape"""
        return self._mesh.get_global_shape(slice_shape, self._tensor_map)

    def get_devices_for_axis(self, axis, rank):
        """
        Get the repeat rank list when the axis is not shard.

        Args:
            layout (Layout): Layout
            axis (str): Axis name.
            rank (int): Global rank

        Returns:
            list: reduce rank list
        """
        return self._mesh.get_devices_for_axis(axis, rank)

    def get_comm_group_by_axis(self, axis):
        return self._mesh.get_comm_group_by_axis(axis)

    def repeat_num(self):
        """
        Number of repeated placements.
        In pipeline parallel, only the last stage return repeat num, other stages return -1.
        For example:
        layout = Layout((2, 4), ("dp", "mp"))
        x_layout = layout("dp", "None")
        The repeat_num is equal to all device num 8 divided by device num corresponding to used axis 2, that is 4.
        """
        if self._tensor_map is None:
            raise ValueError(f"The tensor_map is None, the mesh_shape is {self._mesh.mesh_shape},"
                             f" alias_name is {self._mesh.mesh_dim_names}")

        # if it is not the last stage, return -1
        group_size = platform.get_world_size()
        if self._rank_list[-1] != (group_size - 1):
            return -1

        all_device_num = functools.reduce(lambda x, y: x * y, self._mesh.mesh_shape)
        used_dev_num = 1
        for ele in self._tensor_map:
            if isinstance(ele, tuple):
                for item in ele:
                    if item >= 0:
                        used_dev_num *= self._mesh.mesh_shape[len(self._mesh.mesh_shape) - item - 1]
                continue
            if ele >= 0:
                used_dev_num *= self._mesh.mesh_shape[len(self._mesh.mesh_shape) - ele - 1]

        return all_device_num // used_dev_num

    def _to_compact_string(self):
        """
        generate dict key

        Returns:
            str: string for compact
        """
        mesh_key = self._mesh.to_hash()
        hash_key = (self._tensor_map, self.partial)
        hash_key += mesh_key
        return str(hash_key)

    @property
    def compact_str(self):
        return self._compact_str

    def update_compact_str(self):
        self._compact_str = self._to_compact_string()

    def to_string(self):
        """
        layout dump

        Returns:
            str: layout string
        """
        device_info = f"Mesh shape: {self._mesh.mesh_shape}"
        alias_info = f"Alias Names: {self._mesh.mesh_dim_names}"
        rank_info = f"Rank List: {self._rank_list}"
        partial_info = f"Partial: {self.partial}"

        if self._tensor_map is None:
            tensor_info = "Tensor Map: Not configured"
        else:
            readable_map = []
            for item in self._tensor_map:
                if isinstance(item, tuple):
                    # 处理嵌套元组
                    mapped_tuple = tuple(
                        self._mesh.mesh_dim_names[len(self._mesh.mesh_dim_names) - 1 - dim] if dim != -1 else "None"
                        for dim in item
                    )
                    readable_map.append(mapped_tuple)
                else:
                    readable_map.append(
                        self._mesh.mesh_dim_names[len(self._mesh.mesh_dim_names) - 1 - item] if item != -1 else "None"
                    )

            tensor_info = f"Tensor Map: {tuple(readable_map)}"

        interleaved = "Yes" if "interleaved_parallel" in self._mesh.mesh_dim_names else "No"
        interleaved_info = f"Interleaved Parallel: {interleaved}"

        return (
            f"Layout Configuration:\n"
            f"  {device_info}\n"
            f"  {alias_info}\n"
            f"  {partial_info}\n"
            f"  {tensor_info}\n"
            f"  {interleaved_info}\n"
            f"  {rank_info}"
        )

    def __str__(self):
        """__str__"""
        return self.to_string()

    def __repr__(self):
        """__repr__"""
        return f"<Layout at {hex(id(self))}>"

    def __eq__(self, other):
        """
        __eq__
        """
        if not isinstance(other, Layout):
            return False

        if (self.mesh_shape != other.mesh_shape or
                self.alias_name != other.alias_name or
                self.partial != other.partial or
                self.rank_list != other.rank_list):
            return False

        if self._tensor_map is None or other.tensor_map is None:
            return self._tensor_map is other.tensor_map
        return self._tensor_map == other.tensor_map
