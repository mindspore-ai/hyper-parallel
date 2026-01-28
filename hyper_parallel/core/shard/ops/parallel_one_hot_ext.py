# Copyright 2026 Huawei Technologies Co., Ltd
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
"""
Distributed implementation for OneHotExt operator.
"""

# pylint: disable=import-outside-toplevel
from hyper_parallel.core.layout import Layout
from hyper_parallel.core.placement_types import Shard, Replicate
from hyper_parallel.platform import get_platform
from .parallel_ops import DistributedOp

platform = get_platform()


class OneHotExtDistributedOp(DistributedOp):
    """Distributed implementation for OneHotExt operator."""

    def infer_layout(self, layouts, extra_args):
        """
        Infer output layout for OneHotExt.

        Args:
            layouts (tuple): Tuple containing input layouts.
            extra_args (tuple): Additional arguments containing [num_classes, on_value, off_value, axis].

        Returns:
            Layout: Output layout with one-hot dimension inserted at specified axis.
        """
        if not layouts:
            return None

        indices_layout = layouts[0]
        if indices_layout is None or indices_layout.mesh_shape is None:
            raise ValueError(f"{self.op_name}: indices layout cannot be None")

        if indices_layout.is_partial():
            raise ValueError(
                f"{self.op_name}: indices cannot be in partial state. "
                f"Indices must contain complete index values for OneHot operation."
            )

        num_classes = self._get_num_classes(extra_args)
        self._validate_num_classes(num_classes)

        axis = self._get_axis(extra_args)

        in_tensor_map = indices_layout.tensor_map
        if not in_tensor_map:
            raise ValueError(f"{self.op_name}: indices tensor_map is empty")

        self._validate_multi_dim_restriction(in_tensor_map, axis, indices_layout)
        self._validate_inputs_layouts(layouts)

        out_tensor_map = self._infer_output_tensor_map(in_tensor_map, axis)
        out_layout = self._create_layout_from_tensor_map(indices_layout, out_tensor_map)

        out_placements = self._tensor_map_to_placements(indices_layout, out_tensor_map)
        out_layout.set_placements(out_placements)

        return out_layout

    def get_expand_impl(self, func, output_layout, layouts, extra_args):
        """Get expanded implementation for OneHotExt operator."""
        import mindspore as ms
        from mindspore import ops, Tensor

        del output_layout

        indices_layout = layouts[0] if layouts else None
        if indices_layout is None:
            return None

        sharded_axes = self._get_sharded_axes(indices_layout)
        if not sharded_axes:
            return None

        original_op = func
        rank = platform.get_rank()
        reduce_max = ops.ReduceMax(keep_dims=False)

        def expanded_one_hot(indices, num_classes, on_value, off_value, axis):
            self._validate_num_classes(num_classes)
            self._validate_indices_dtype(indices)

            if num_classes != -1:
                return original_op(indices, num_classes, on_value, off_value, axis)

            local_max = reduce_max(indices, ())
            if not isinstance(local_max, Tensor):
                local_max = Tensor(local_max, ms.int64)

            local_max_host = int(local_max.asnumpy())
            if local_max_host > 2147483647:
                raise ValueError(
                    f"{self.op_name}: indices max value {local_max_host} exceeds int32 range"
                )

            zero_dim = local_max.ndim == 0
            local_max_i32 = ops.cast(local_max, ms.int32)

            if zero_dim:
                local_max_i32 = ops.expand_dims(local_max_i32, 0)

            global_max_i32 = local_max_i32
            for axis_name in sharded_axes:
                group = indices_layout.get_comm_group_by_axis(axis_name, rank)
                global_max_i32 = platform.differentiable_all_reduce(
                    global_max_i32, "max", group
                )

            if zero_dim:
                global_max_i32 = ops.squeeze(global_max_i32, 0)

            depth = int(global_max_i32.asnumpy()) + 1
            return original_op(indices, depth, on_value, off_value, axis)

        return expanded_one_hot

    def _get_num_classes(self, extra_args):
        """Extract num_classes from extra arguments."""
        if isinstance(extra_args, (list, tuple)) and len(extra_args) >= 1:
            num_classes = extra_args[0]
            if isinstance(num_classes, int):
                return num_classes
        return -1

    def _validate_num_classes(self, num_classes):
        """Validate num_classes parameter."""
        if not isinstance(num_classes, int):
            raise TypeError(
                f"{self.op_name}: num_classes must be int, but got {type(num_classes).__name__}"
            )
        if num_classes < -1:
            raise ValueError(
                f"{self.op_name}: num_classes must be >= -1, but got {num_classes}"
            )

    def _validate_indices_dtype(self, indices):
        """Validate indices dtype."""
        import mindspore as ms

        if indices.dtype != ms.int64:
            raise TypeError(
                f"{self.op_name}: indices dtype must be int64, but got {indices.dtype}"
            )

    def _get_sharded_axes(self, layout):
        """Get all device axes that are used for sharding."""
        sharded_axes = set()

        if layout is None or layout.alias_tensor_map is None:
            return []

        for dim_alias in layout.alias_tensor_map:
            if dim_alias == "None":
                continue

            if isinstance(dim_alias, tuple):
                for axis_name in dim_alias:
                    if axis_name != "None":
                        sharded_axes.add(axis_name)
            else:
                sharded_axes.add(dim_alias)

        return list(sharded_axes)

    def _get_axis(self, extra_args):
        """Extract axis parameter from extra arguments."""
        if isinstance(extra_args, (list, tuple)) and len(extra_args) >= 4:
            axis = extra_args[3]
            if isinstance(axis, int):
                return self._validate_axis(axis)
        return -1

    def _validate_axis(self, axis):
        """Validate axis parameter."""
        if not isinstance(axis, int):
            raise TypeError(
                f"{self.op_name}: axis must be int, but got {type(axis).__name__}"
            )

        if axis > 1 or axis < -1:
            raise ValueError(f"{self.op_name}: axis {axis} is out of range[-1, 1]")

        return axis

    def _validate_multi_dim_restriction(self, in_tensor_map, axis, indices_layout):
        """Validate restriction for multi-dimensional inputs."""
        in_rank = len(in_tensor_map)
        if in_rank <= 1:
            return

        if axis != -1:
            raise ValueError(
                f"{self.op_name}: when input dimension is > 1, axis must be -1, but got {axis}"
            )

        alias_map = indices_layout.alias_tensor_map
        for i in range(1, len(alias_map)):
            if alias_map[i] != "None":
                raise ValueError(
                    f"{self.op_name}: when input dimension is > 1, strategy must be data parallel, "
                    f"but dimension {i} is sharded on '{alias_map[i]}'"
                )

    def _validate_inputs_layouts(self, layouts):
        """Validate that non-indices inputs are fully replicated."""
        for layout in layouts[1:]:
            if layout is None:
                continue
            alias_map = layout.alias_tensor_map
            if alias_map and any(x != "None" for x in alias_map):
                raise ValueError(
                    f"{self.op_name}: non-indices inputs must be replicated, but got {alias_map}"
                )

    def _infer_output_tensor_map(self, in_tensor_map, axis):
        """Infer output tensor map by inserting one-hot dimension at specified axis."""
        in_rank = len(in_tensor_map)

        if axis in (-1, in_rank):
            insert_pos = in_rank
        else:
            insert_pos = axis

        if insert_pos < 0 or insert_pos > in_rank:
            raise ValueError(
                f"{self.op_name}: axis {axis} is out of range for input with rank {in_rank}"
            )

        out_tensor_map = list(in_tensor_map)
        out_tensor_map.insert(insert_pos, -1)
        return tuple(out_tensor_map)

    def _create_layout_from_tensor_map(self, base_layout, out_tensor_map):
        """Create output layout from tensor map."""
        out_layout = Layout(
            mesh_shape=base_layout.mesh_shape,
            alias_name=base_layout.alias_name,
            rank_list=base_layout.rank_list,
        )

        out_layout.set_tensor_map(out_tensor_map)
        out_layout.set_alias_tensor_map(
            self._tensor_map_to_alias_tensor_map(base_layout, out_tensor_map)
        )
        out_layout.update_compact_str()
        return out_layout

    def _tensor_map_to_alias_tensor_map(self, base_layout, tensor_map):
        """Convert numeric tensor map to alias tensor map."""
        alias_name = base_layout.alias_name
        alias_tensor_map = []

        for dim in tensor_map:
            if dim == -1:
                alias_tensor_map.append("None")
                continue

            if isinstance(dim, tuple):
                names = tuple(
                    alias_name[len(alias_name) - 1 - d] for d in dim if d != -1
                )
                alias_tensor_map.append(names if names else "None")
                continue

            alias_tensor_map.append(alias_name[len(alias_name) - 1 - dim])

        return tuple(alias_tensor_map)

    def _tensor_map_to_placements(self, base_layout, tensor_map):
        """
        Convert tensor_map to placements.
        
        Args:
            base_layout: Base layout to get mesh dimension info
            tensor_map: Tensor map to convert
            
        Returns:
            tuple: Placements tuple (Shard/Replicate for each mesh dimension)
        """
        mesh_ndim = len(base_layout.mesh_shape)
        placements = []

        for mesh_dim_idx in range(mesh_ndim):
            is_sharded = False

            for tensor_dim_idx, tensor_dim_map in enumerate(tensor_map):
                if tensor_dim_map == -1:
                    continue

                if isinstance(tensor_dim_map, tuple):
                    if mesh_dim_idx in tensor_dim_map:
                        placements.append(Shard(tensor_dim_idx))
                        is_sharded = True
                        break
                elif tensor_dim_map == mesh_dim_idx:
                    placements.append(Shard(tensor_dim_idx))
                    is_sharded = True
                    break

            if not is_sharded:
                placements.append(Replicate())

        return tuple(placements)
