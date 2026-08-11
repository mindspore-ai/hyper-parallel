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
from typing import Any, Optional, Sequence

import numpy as np


from hyper_parallel.core.dtensor.placement_types import Placement, Shard, StridedShard, Replicate, Partial
from hyper_parallel.core.dtensor.device_mesh import DeviceMesh, _create_device_mesh
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
        return full_data[area].clone()

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
        self._tensor_shape = None
        self._tensor_stride = None
        self._tensor_dtype = None
        self._mesh = _create_device_mesh("npu", mesh_shape, mesh_dim_names=alias_name, rank_list=self._rank_list,
                                         init_backend=init_backend)
        self._compact_str = self._to_compact_string()
        self._placements = None
        self.partial_ops = {}  # Initialized in _build_dim_map_from_placements()

    @classmethod
    def from_device_mesh(cls, device_mesh: DeviceMesh) -> 'Layout':
        """
        Create a Layout from an existing DeviceMesh.

        Args:
            device_mesh (DeviceMesh): The device mesh to create layout from.

        Returns:
            Layout: A new Layout instance initialized with the properties of the provided device mesh.

        Examples:
            >>> from hyper_parallel.core.dtensor.layout import Layout, DeviceMesh
            >>> device_mesh = DeviceMesh("npu", (2, 2), mesh_dim_names=("dp", "mp"))
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
        obj._tensor_shape = None
        obj._tensor_stride = None
        obj._tensor_dtype = None
        obj._placements = None
        obj._compact_str = obj._to_compact_string()
        return obj

    def __call__(self, *alias_tensor_map):
        obj = copy.deepcopy(self)

        # Clear the inherited partial status.
        # When creating a new layout mapping configuration via __call__,
        # it should not inherit the dynamic execution state (Partial) of the original layout.
        # If the user intends to create a Partial placement, it will be parsed from alias_tensor_map.
        obj._partial = [None] * len(obj.mesh_shape)

        if len(alias_tensor_map) == 1 and isinstance(alias_tensor_map[0], (list, tuple)):
            if len(alias_tensor_map[0]) > 0 and isinstance(alias_tensor_map[0][0], Placement):
                return self._process_placement_layout(obj, alias_tensor_map[0])

        if len(alias_tensor_map) > 0 and isinstance(alias_tensor_map[0], Placement):
            return self._process_placement_layout(obj, alias_tensor_map)

        return self._process_alias_layout(obj, alias_tensor_map)

    def __deepcopy__(self, memo):
        """Deep copy layout without rebuilding the underlying device mesh."""
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memo))
        return result

    @staticmethod
    def _process_placement_layout(obj, placements):
        """Process layout defined by Placement types."""
        obj.set_placements(placements)
        return copy.deepcopy(obj)

    @staticmethod
    def _process_alias_layout(obj, alias_tensor_map):
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

        This method converts the `placements` configuration (consisting of Shard, StridedShard,
        Replicate, Partial)
        into a `tensor_map` representation used for distributed tensor operations.

        Args:
            dim (int): The dimension of the tensor. Must be a positive integer.

        Returns:
            tuple: A tuple representing the tensor map, where each element corresponds to a tensor dimension.
                   A value of -1 indicates the dimension is not sharded, an integer indicates the mesh
                   dimension index along which the tensor dimension is sharded, and a tuple indicates
                   that the same tensor dimension is sharded multiple times in order.

        Raises:
            ValueError: If `dim` is negative.
            ValueError: If a shard dimension in `placements` is out of bounds for the given tensor dimension.
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
                if dim_map[shard_dim] == -1:
                    dim_map[shard_dim] = [mesh_idx]
                else:
                    dim_map[shard_dim].append(mesh_idx)
            elif isinstance(placement, Partial):
                self._partial[mesh_idx] = self._extract_reduce_op(placement)
        self._validate_strided_shard_split_factor(dim_map)
        self._reorder_dim_map_for_strided_shard(dim_map)
        return dim_map

    @staticmethod
    def _placement_split_factor(placement):
        """Return the effective split factor carried by a placement."""
        return placement.split_factor if isinstance(placement, StridedShard) else 1

    @staticmethod
    def _build_order_positions(shard_order):
        """Build a mesh axis to order position mapping."""
        return {mesh_idx: order_idx for order_idx, mesh_idx in enumerate(shard_order)}

    def _compute_expected_split_factors(self, shard_axes, shard_order):
        """Infer the split_factor each mesh axis should carry for the given sharding order."""
        order_positions = self._build_order_positions(shard_order)
        expected_split_factors = {}
        for mesh_idx in shard_axes:
            split_factor = 1
            for right_mesh_idx in shard_axes:
                if right_mesh_idx <= mesh_idx:
                    continue
                if order_positions[right_mesh_idx] < order_positions[mesh_idx]:
                    split_factor *= self.mesh_shape[right_mesh_idx]
            expected_split_factors[mesh_idx] = split_factor
        return expected_split_factors

    def _get_effective_shard_axes(self, shard_axes):
        """Return shard axes ordered by their effective sharding order."""
        return sorted(
            shard_axes,
            key=lambda mesh_idx: self._placement_split_factor(self.placements[mesh_idx]),
        )

    def _reorder_dim_map_for_strided_shard(self, dim_map):
        """Reorder dim_map entries to reflect the effective sharding order."""
        for i, shard_axes in enumerate(dim_map):
            if shard_axes == -1 or len(shard_axes) <= 1:
                continue
            dim_map[i] = self._get_effective_shard_axes(shard_axes)

    def _validate_strided_shard_split_factor(self, dim_map):
        """Validate that split factors match the effective sharding order."""
        for shard_axes in dim_map:
            if shard_axes == -1:
                continue
            shard_order = self._get_effective_shard_axes(shard_axes)
            expected_split_factors = self._compute_expected_split_factors(
                shard_axes, shard_order
            )
            for mesh_idx in shard_axes:
                placement = self.placements[mesh_idx]
                actual_split_factor = self._placement_split_factor(placement)
                expected_split_factor = expected_split_factors[mesh_idx]
                if actual_split_factor != expected_split_factor:
                    raise ValueError(
                        f"StridedShard split_factor mismatch on mesh axis {mesh_idx}: "
                        f"expected {expected_split_factor}, got {actual_split_factor}."
                    )

    @staticmethod
    def _extract_reduce_op(placement):
        """Extract reduce operation name from Partial placement."""
        op_name = getattr(placement, "reduce_op", "sum")
        if isinstance(op_name, str):
            op_name = op_name.lower()
        return op_name

    def _convert_dim_map_to_tensor_map(self, dim_map):
        """Convert dimension map to tensor map format."""
        device_dim_count = len(self.mesh_shape)
        tensor_map = []
        for mesh_idx in dim_map:
            if mesh_idx == -1:
                tensor_map.append(-1)
                continue
            mapped_axes = tuple(device_dim_count - 1 - axis for axis in mesh_idx)
            tensor_map.append(mapped_axes[0] if len(mapped_axes) == 1 else mapped_axes)
        return tensor_map

    def _build_readable_tensor_map(self):
        """Build human-readable alias tensor map from tensor_map."""
        mesh_dim_names = self._mesh.mesh_dim_names
        has_names = mesh_dim_names is not None

        def _map_dim(dim):
            """convert dimension index to dimension name."""
            if dim == -1:
                return "None"
            if not has_names:
                return f"dim_{dim}"
            return mesh_dim_names[len(mesh_dim_names) - 1 - dim]

        readable_map = []
        for item in self._tensor_map:
            if isinstance(item, tuple):
                mapped_tuple = tuple(_map_dim(dim) for dim in item)
                readable_map.append(mapped_tuple)
            else:
                readable_map.append(_map_dim(item))
        return tuple(readable_map)

    def tensor_map_to_placement(self):
        """
        Transform tensor map to placement.

        This method converts the existing `tensor_map` and `partial` status into a list of `Placement` objects
        (Shard, StridedShard, Replicate, Partial). This is the inverse operation of
        `placement_to_tensor_map`.

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
            valid_mapping = [map_val for map_val in mapping_list if map_val != -1]
            mesh_indices = [mesh_ndim - 1 - map_val for map_val in valid_mapping]
            shard_axes = sorted(mesh_indices)
            expected_split_factors = self._compute_expected_split_factors(
                shard_axes, mesh_indices
            )
            for mesh_idx in shard_axes:
                split_factor = expected_split_factors[mesh_idx]
                placement = (
                    StridedShard(dim=tensor_dim, split_factor=split_factor)
                    if split_factor > 1
                    else Shard(dim=tensor_dim)
                )
                placements[mesh_idx] = placement
        for mesh_idx, op in enumerate(self.partial):
            if op is not None:
                placements[mesh_idx] = Partial(reduce_op=op)
        self.set_placements(placements)
        return placements

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.update_mesh(init_backend=False)

    @property
    def mesh(self):
        """
        Get the device mesh associated with this layout.

        Returns:
            DeviceMesh: The device mesh describing the device topology.
        """
        return self._mesh

    def update_mesh(self, init_backend: bool = True):
        """Recreate the internal DeviceMesh from current layout properties.

        Args:
            init_backend (bool): Whether to initialize communication backend
                (process groups). Set to ``False`` during deserialization to
                avoid creating process groups with a stale rank_list from the
                sender side. Default ``True``.
        """
        self._mesh = _create_device_mesh("npu", self.mesh_shape, mesh_dim_names=self.alias_name,
                                         rank_list=self.rank_list, init_backend=init_backend)

    @property
    def rank_list(self):
        """
        Get the list of ranks participating in this layout.

        Returns:
            tuple[int]: The rank list.
        """
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
        """Return the human-readable alias tensor map for this layout."""
        return self._alias_tensor_map

    @property
    def alias_placements(self):
        """Return alias_tensor_map when it contains multi-axis tuples, otherwise placements.

        alias_tensor_map preserves multi-axis ordering information
        (e.g., (("dp", "tp"), "None") vs (("tp", "dp"), "None"))
        that Placement objects cannot represent, since both map to
        [Shard(0), Shard(0)].

        For single-axis layouts, Placement objects are preferred because they
        also carry Partial status which alias_tensor_map cannot encode.

        Use this property when constructing DTensors from an existing Layout
        to avoid the lossy Placement round-trip for multi-axis cases.
        """
        if self._alias_tensor_map is not None and any(
            isinstance(item, tuple) for item in self._alias_tensor_map
        ):
            return self._alias_tensor_map
        return self._placements

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
    def tensor_shape(self) -> Optional[tuple[int, ...]]:
        """Return the explicit logical tensor shape, if one was provided."""
        return self._tensor_shape

    @property
    def tensor_stride(self) -> Optional[tuple[int, ...]]:
        """Return the explicit logical tensor stride, if one was provided."""
        return self._tensor_stride

    @property
    def tensor_dtype(self) -> Optional[Any]:
        """Return the explicit logical tensor dtype, if one was provided."""
        return self._tensor_dtype

    def set_tensor_meta(
        self,
        shape: Sequence[int],
        stride: Sequence[int],
        dtype: Any,
    ) -> None:
        """Set logical tensor metadata independently from local shard storage.

        Args:
            shape: Logical global tensor shape.
            stride: Logical global tensor stride.
            dtype: Logical tensor dtype.
        """
        self._tensor_shape = tuple(shape)
        self._tensor_stride = tuple(stride)
        self._tensor_dtype = dtype

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
        """Clear all partial statuses and regenerate placements from the tensor map."""
        self._partial = [None] * len(self.mesh_shape)
        self.tensor_map_to_placement()
        self.update_compact_str()

    def is_partial(self):
        """Return true if any dim in mesh_shape is partial"""
        return any(self.partial)

    def get_dim_split_num(self, tensor_dim: int) -> int:
        """Return the total shard count for ``tensor_dim`` via alias_tensor_map.

        Args:
            tensor_dim: Tensor dimension index to check.

        Returns:
            Number of shards (1 if not sharded or no alias_tensor_map set).
        """
        alias_tm = self.alias_tensor_map
        if alias_tm is None or tensor_dim >= len(alias_tm):
            return 1
        dim_entry = alias_tm[tensor_dim]
        if dim_entry == 'None':
            return 1
        if isinstance(dim_entry, str):
            return self.mesh.get_device_num_along_axis(dim_entry)
        if isinstance(dim_entry, tuple):
            total = 1
            for axis in dim_entry:
                if axis != 'None':
                    total *= self.mesh.get_device_num_along_axis(axis)
            return total
        return 1

    def get_split_id(self, tensor_dim: int) -> int:
        """Return this rank's global position among all shards of ``tensor_dim``.

        For a single sharding axis, returns the rank's position within that axis group.
        For multiple sharding axes (e.g. dp+cp both sharding T1), computes the combined
        global position as a mixed-radix number ordered by the axis tuple:
            global_id = ax0_pos * ax1_size * ... + ax1_pos * ax2_size * ... + axN_pos
        This matches MindFormers' ``offset_id = dp_rank * (cp*tp) + cp_rank * tp + tp_rank``
        for combined sequence-parallel sharding across dp, cp, and tp dimensions.

        Args:
            tensor_dim: Tensor dimension index to query.

        Returns:
            Split index for this rank (0 if not sharded or rank not in rank list).
        """
        alias_tm = self.alias_tensor_map
        if alias_tm is None or tensor_dim >= len(alias_tm):
            return 0
        dim_entry = alias_tm[tensor_dim]
        if dim_entry == 'None':
            return 0
        rank = platform.get_rank()
        if isinstance(dim_entry, tuple):
            non_none = [ax for ax in dim_entry if ax != 'None']
            if not non_none:
                return 0
            global_id = 0
            for ax in non_none:
                rank_list = self.mesh.get_rank_list_along_axis(ax)
                local_id = rank_list.index(rank) if rank in rank_list else 0
                ax_size = self.mesh.get_device_num_along_axis(ax)
                global_id = global_id * ax_size + local_id
            return global_id
        if isinstance(dim_entry, str):
            rank_list = self.mesh.get_rank_list_along_axis(dim_entry)
            return rank_list.index(rank) if rank in rank_list else 0
        return 0

    def get_global_shape(self, slice_shape):
        """get global shape"""
        if self._tensor_shape is not None:
            return self._tensor_shape
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
        """Return the communication group for the specified mesh axis via the underlying DeviceMesh."""
        return self._mesh.get_comm_group_by_axis(axis)

    def repeat_num(self):
        """
        Number of repeated placements.
        For example:
        layout = Layout((2, 4), ("dp", "mp"))
        x_layout = layout("dp", "None")
        The repeat_num is equal to all device num 8 divided by device num corresponding to used axis 2, that is 4.
        """
        if self._tensor_map is None:
            raise ValueError(f"The tensor_map is None, the mesh_shape is {self._mesh.mesh_shape},"
                             f" alias_name is {self._mesh.mesh_dim_names}")

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
        """Return the cached compact string representation of this layout."""
        return self._compact_str

    def update_compact_str(self):
        """Recompute and store the compact string representation of this layout."""
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
                    # handle nested tuple
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

        same_layout_attrs = (
            self.mesh_shape,
            self.alias_name,
            self.partial,
            self.rank_list,
        ) == (
            other.mesh_shape,
            other.alias_name,
            other.partial,
            other.rank_list,
        )
        if not same_layout_attrs:
            return False

        if self._tensor_map is None or other.tensor_map is None:
            return self._tensor_map is other.tensor_map
        return self._tensor_map == other.tensor_map
