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
from typing import Tuple
import functools
import numpy as np

import mindspore as ms
from ._group_manager import _get_comm_group

_group_map = {}

def _infer_slice_area_by_rank(dev_matrix, tensor_map, rank_id: int, full_shape: tuple): # -> tuple[tuple[int]]:
    """Return the range of each axis from full tensor for slice in current rank."""
    def _get_dev_num_alone_dim(dev_matrix, dim):
        """_get_dev_num_alone_dim."""
        return dev_matrix[-dim - 1] if dim != -1 else 1

    def _rank_id_to_dev_id_list(dev_matrix, rank_id):
        """Infer dev id list by rank_id and dev_matrix"""
        dims = len(dev_matrix)
        dev_id_list = [0] * dims
        for i in range(dims - 1, -1, -1):
            dev_id_list[i] = rank_id % dev_matrix[i]
            rank_id = rank_id // dev_matrix[i]
        return dev_id_list

    dev_id_list = _rank_id_to_dev_id_list(dev_matrix, rank_id)

    dims = len(full_shape)
    area = []
    for axis in range(dims):
        mapping = tensor_map[axis]
        if isinstance(mapping, int):
            mapping = (mapping,)
        split_num = 1
        for dim in mapping:
            split_num *= _get_dev_num_alone_dim(dev_matrix, dim)

        slice_id = 0
        coef = 1
        for dim in reversed(mapping):
            if dim == -1:
                continue
            slice_id += dev_id_list[-dim - 1] * coef
            coef *= _get_dev_num_alone_dim(dev_matrix, dim)
        slice_size = full_shape[axis] // split_num
        start = slice_id * slice_size
        end = start + slice_size
        area.append((start, end))
    return area


def _get_slice_tensor_by_layout(global_tensor, layout):
    """Transfer global tensor to local tensor by layout"""
    inner_rank_id = layout.rank_list.index(layout.mesh.rank)
    slice_area = _infer_slice_area_by_rank(layout.device_matrix, layout.tensor_map, inner_rank_id, global_tensor.shape)

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


class _DeviceMatrix:
    """
    Topological abstraction describing cluster devices.
    Args:
        device_matrix:
        alias_name:
        rank_list:
    """
    def __init__(self,
                 device_matrix: Tuple[int],
                 alias_name: Tuple[str],
                 rank_list: Tuple[int]):
        if not isinstance(device_matrix, tuple):
            raise TypeError(f'device_matrix must be tuple type, but got:{type(device_matrix)}')
        if not isinstance(alias_name, tuple):
            raise TypeError(f'alias_name must be tuple type, but got:{type(alias_name)}')
        if len(device_matrix) != len(alias_name):
            raise ValueError('device_matrix length should be equal to alias_name length')
        for in_ele in device_matrix:
            if not isinstance(in_ele, int):
                raise TypeError(f'The element of device_matrix must be int type, but got:{type(in_ele)}')
        for in_ele in alias_name:
            if not isinstance(in_ele, str):
                raise TypeError(f'The element of alias_name must be str type, but got:{type(in_ele)}')
            if not in_ele:
                raise ValueError("The element of alias_name can not be empty.")
            if in_ele == "None":
                raise ValueError("The element of alias_name can not set 'None', because 'None' means no sharding.")
        if len(set(alias_name)) != len(alias_name):
            raise ValueError(f'Each element of alias_name {alias_name} should be different')
        inter_key = "interleaved_parallel"
        if inter_key in alias_name and alias_name.index(inter_key) != len(alias_name) - 1:
            raise ValueError(f"When alias_name {alias_name} contains keyword 'interleaved_parallel',"
                             f" it should be at the last dim of alias_name, which means the virtual sharding.")
        self._device_shape = device_matrix
        self._alias_name = alias_name
        self._dev_num = np.prod(np.array(self._device_shape))
        self._global_shape_map = {}
        self._rank = ms.communication.management.get_rank()
        self._dev_rank = len(self._device_shape)
        self._dev_name_to_dev_id = {name: self._dev_rank - i - 1 for i, name in enumerate(self.alias_name)}
        self._dev_name_to_index = {name: i for i, name in enumerate(self.alias_name)}
        self._cache_rank_list_along_axis = {}
        self._rank_list = rank_list
        self._check_rank_list()
        self._global_shape_map = {}

    def _check_rank_list(self):
        """_check_rank_list"""
        if not isinstance(self._rank_list, tuple):
            raise TypeError(f"The rank_list should be a tuple, but got {type(self._rank_list).__name__}.")
        for in_ele in self._rank_list:
            if not isinstance(in_ele, int):
                raise TypeError(f"The element of rank_list should be int, but got {type(in_ele).__name__}.")
        if len(np.array(self._rank_list).shape) != 1:
            raise ValueError(
                f"The rank_list should be a 1-D list, but got {len(np.array(self._rank_list).shape)}-D list.")
        if len(self._rank_list) != np.prod(np.array(self._device_shape)):
            raise ValueError(f"The length of rank_list should be equal to the product of device_matrix, "
                             f"but got {len(self._rank_list)} and {np.prod(np.array(self._device_shape))}.")

    @property
    def rank(self):
        return self._rank

    @property
    def device_matrix(self):
        return self._device_shape

    @property
    def alias_name(self):
        return self._alias_name

    @property
    def rank_list(self):
        return self._rank_list

    def axis_id(self, axis):
        if axis == "None":
            return -1
        if axis not in self.alias_name:
            raise ValueError(f"The axis name must be one of device matrix alias name {self.alias_name}), "
                             f"but got {axis}")
        return self._dev_name_to_dev_id[axis]

    def axis_index(self, axis):
        if axis not in self.alias_name:
            raise ValueError(f"The axis name must be one of device matrix alias name {self.alias_name}), "
                             f"but got {axis}")
        return self._dev_name_to_index[axis]

    def get_device_num_along_axis(self, axis):
        """Return device num along specify device axis"""
        if axis not in self.alias_name:
            raise ValueError(f"The axis must be one of device alias name: {self.alias_name}, but got {axis}")
        return self.device_matrix[self.alias_name.index(axis)]

    def get_rank_list_along_axis(self, axis):
        """
        Get the repeat rank list when the axis is not shard.

        Args:
            layout (Layout): Layout
            axis (str): Axis name.
            rank (int): Global rank

        Returns:
            list: reduce rank list
        """
        if axis in self._cache_rank_list_along_axis:
            # short cut, get rank list from cache
            return self._cache_rank_list_along_axis[axis]

        device_matrix = self.device_matrix
        alias_name = self.alias_name
        rank_list = self.rank_list
        rank = self.rank

        if axis not in alias_name:
            raise ValueError(f"Axis '{axis}' not found in alias_name {alias_name}")

        if rank not in rank_list:
            raise ValueError(f"Rank {rank} not found in rank_list")

        idx = rank_list.index(rank)
        coord = [0] * len(device_matrix)
        temp = idx
        for i in range(len(device_matrix)-1, -1, -1):
            coord[i] = temp % device_matrix[i]
            temp //= device_matrix[i]

        dim_index = alias_name.index(axis)
        strides = [1] * len(device_matrix)
        for i in range(len(device_matrix)-2, -1, -1):
            strides[i] = strides[i+1] * device_matrix[i+1]

        result_ranks = []
        for v in range(device_matrix[dim_index]):
            new_coord = coord.copy()
            new_coord[dim_index] = v
            new_idx = 0
            for i in range(len(device_matrix)):
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
            raise ValueError("tensor_map is not set. Please configure the tensor map by calling the layout.")
        if len(slice_shape) != len(tensor_map):
            raise ValueError(f"Length of slice_shape ({len(slice_shape)}) must match "
                             f"the length of tensor_map ({len(tensor_map)}).")

        n_dims = len(self._device_shape)
        factors = [1] * len(slice_shape)

        for dev_idx, size in enumerate(self._device_shape):
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
        group = _get_comm_group(rank_list)
        _group_map[(axis, rank)] = group
        return group

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
        device_matrix = self._device_shape
        alias_name = self._alias_name
        rank_list = self._rank_list

        if axis not in alias_name:
            raise ValueError(f"Axis '{axis}' not found in alias_name {alias_name}")

        if rank not in rank_list:
            raise ValueError(f"Rank {rank} not found in rank_list")

        if rank not in rank_list:
            raise ValueError(f"Rank {rank} not found in rank_list")

        idx = rank_list.index(rank)
        coord = [0] * len(device_matrix)
        temp = idx
        for i in range(len(device_matrix)-1, -1, -1):
            coord[i] = temp % device_matrix[i]
            temp //= device_matrix[i]

        dim_index = alias_name.index(axis)
        strides = [1] * len(device_matrix)
        for i in range(len(device_matrix)-2, -1, -1):
            strides[i] = strides[i+1] * device_matrix[i+1]

        result_ranks = []
        for v in range(device_matrix[dim_index]):
            new_coord = coord.copy()
            new_coord[dim_index] = v
            new_idx = 0
            for i in range(len(device_matrix)):
                new_idx += new_coord[i] * strides[i]

            result_ranks.append(rank_list[new_idx])

        return result_ranks

    def to_hash(self):
        rank_ids = (self.rank_list[0], self.rank_list[-1])
        map_key = (self.device_matrix, self.alias_name, rank_ids)
        return map_key


_DEVICE_MESH_MAP = {}


def _create_device_mesh(device_matrix: Tuple[int], alias_name: Tuple[str], rank_list: Tuple[int]):
    """
    create_device_mesh
    """
    rank_ids = (rank_list[0], rank_list[-1])
    map_key = hash((device_matrix, alias_name, rank_ids))
    if map_key not in _DEVICE_MESH_MAP:
        _DEVICE_MESH_MAP[map_key] = _DeviceMatrix(device_matrix, alias_name, rank_list)
    return _DEVICE_MESH_MAP.get(map_key, None)


class Layout:
    """
    Topological abstraction describing cluster devices for tensor slice placement on the cluster.

    Note:
        - It is valid only in semi auto parallel or auto parallel mode.
        - The multiplication result of the `device_matrix` must be equal to the device count in a pipeline stage.
        - When the layout function is invoked to constructs a sharding strategy, each alias name is only allowed to be
          used once to shard a tensor.

    Args:
        device_matrix (tuple): Describe the shape of devices arrangement, its element type is int.
        alias_name (tuple): The alias name for each axis of device_matrix, its length shoits element type is string.
                            When using "interleaved_parallel" as an alias name, the tensor would be split into multiple
                            copies on the corresponding partition dimension on a single card.
        rank_list (tuple, optional): Data is allocated to the device according to rank_list. Default: ``None``.

    Raises:
        TypeError: `device_matrix` is not a tuple type.
        TypeError: `alias_name` is not a tuple type.
        TypeError: 'rank_list' is not a list type.
        ValueError: `device_matrix` length is not equal to `alias_name` length.
        TypeError: The element of `device_matrix` is not int type.
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
        {"device_matrix": (2, 2, 2), "tensor_map": (2, 0), "interleaved_parallel": False,
        'alias_name': {'dp', 'sp', 'mp'}, "rank_list": [0, 1, 2, 3, 4, 5, 6, 7]}
        >>> layout = Layout((2, 2, 2), ("dp", "sp", "interleaved_parallel"))
        >>> layout1 = layout(("dp", "interleaved_parallel"), "sp")
    """

    def __init__(self, device_matrix, alias_name, rank_list=None):
        self._alias_name = alias_name
        self._tensor_map = None
        if not rank_list:
            self._rank_list = tuple(range(np.prod(np.array(device_matrix))))
        else:
            self._rank_list = tuple(rank_list)
        self._partial = [None] * len(device_matrix) # partial status for each dev dim
        self._support_partial_op = ['sum', 'max', 'min', 'avg', None]
        self._alias_tensor_map = None
        self._mesh = _create_device_mesh(device_matrix, alias_name, self._rank_list)
        self._compact_str = self._to_compact_string()

    def __call__(self, *alias_tensor_map):
        obj = copy.deepcopy(self)
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
        obj.update_compact_str()
        return copy.deepcopy(obj)

    def to_dict(self):
        """
        Transform layout to a dictionary.
        """
        if self._mesh.device_matrix is None:
            raise ValueError("The device_shape of layout is None")
        if self._tensor_map is None:
            raise ValueError("The tensor_map of layout is None")
        interleaved_parallel = "interleaved_parallel" in self._mesh.alias_name
        return {"device_matrix": self._mesh.device_matrix, "tensor_map": self._tensor_map,
                "interleaved_parallel": interleaved_parallel, "alias_name": self._mesh.alias_name,
                "rank_list": self._rank_list}

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.update_mesh()

    @property
    def mesh(self):
        return self._mesh

    def update_mesh(self):
        self._mesh = _create_device_mesh(self.device_matrix, self.alias_name, self.rank_list)

    @property
    def rank_list(self):
        """rank list"""
        return self._rank_list

    @rank_list.setter
    def rank_list(self, val):
        self._rank_list = val

    @property
    def device_matrix(self):
        """device matrix"""
        return self._mesh.device_matrix

    @property
    def alias_name(self):
        """alias name"""
        return self._mesh.alias_name

    @property
    def alias_tensor_map(self):
        return self._alias_tensor_map

    def set_alias_tensor_map(self, alias_tensor_map):
        """Set alias_tensor_map"""
        self._alias_tensor_map = alias_tensor_map

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
        self._partial = [None] * len(self.device_matrix)

    def is_partial(self):
        """Return true if any dim in dev_matrix is partial"""
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

    def get_comm_group_by_axis(self, axis, rank):
        return self._mesh.get_comm_group_by_axis(axis, rank)

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
            raise ValueError(f"The tensor_map is None, the device_matrix is {self._mesh.device_matrix},"
                             f" alias_name is {self._mesh.alias_name}")

        # if it is not the last stage, return -1
        if enable_ms:
            from mindspore.communication.management import get_group_size
            group_size = get_group_size()
        else:
            group_size = torch.distributed.get_world_size()
        if self._rank_list[-1] != (group_size - 1):
            return -1

        all_device_num = functools.reduce(lambda x, y: x * y, self._mesh.device_matrix)
        used_dev_num = 1
        for ele in self._tensor_map:
            if isinstance(ele, tuple):
                for item in ele:
                    if item >= 0:
                        used_dev_num *= self._mesh.device_matrix[len(self._mesh.device_matrix) - item - 1]
                continue
            if ele >= 0:
                used_dev_num *= self._mesh.device_matrix[len(self._mesh.device_matrix) - ele -1]

        return all_device_num // used_dev_num

    def _to_compact_string(self):
        """
        generate dict key

        Returns:
            str: string for compact
        """
        mesh_key = self._mesh.to_hash()
        hash_key = (self._tensor_map,)
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
        device_info = f"Device Matrix: {self._mesh.device_matrix}"
        alias_info = f"Alias Names: {self._mesh.alias_name}"
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
                        self._mesh.alias_name[len(self._mesh.alias_name)-1-dim] if dim != -1 else "None"
                        for dim in item
                    )
                    readable_map.append(mapped_tuple)
                else:
                    readable_map.append(
                        self._mesh.alias_name[len(self._mesh.alias_name)-1-item] if item != -1 else "None"
                    )

            tensor_info = f"Tensor Map: {tuple(readable_map)}"

        interleaved = "Yes" if "interleaved_parallel" in self._mesh.alias_name else "No"
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

        if (self.device_matrix != other.device_matrix or
                self.alias_name != other.alias_name or
                self.partial != other.partial or
                self.rank_list != other.rank_list):
            return False

        if self._tensor_map is None or other.tensor_map is None:
            return self._tensor_map is other.tensor_map
        return self._tensor_map == other.tensor_map


