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

from typing import Tuple

from hyper_parallel.core.dtensor.layout import Layout
from hyper_parallel.platform import get_platform
from .parallel_ops import DistributedOp

platform = get_platform()


def _normalize_one_hot_ext_args(indices, num_classes, on_value, off_value, axis):
    return (indices, num_classes, on_value, off_value, axis), {}


class OneHotExtDistributedOp(DistributedOp):
    """Distributed implementation for OneHotExt operator."""

    def preprocess(self, args: tuple, kwargs: dict) -> tuple:
        """
        Preprocess arguments for OneHotExt operator.

        Args:
            args (tuple): Input arguments (indices, num_classes, on_value, off_value, axis).
            kwargs (dict): Keyword arguments (empty for this operator).

        Returns:
            tuple: (local_args, local_kwargs, cache_values)
        """
        args, kwargs = _normalize_one_hot_ext_args(*args, **kwargs)
        indices, num_classes, on_value, off_value, axis = args

        indices_local = indices.to_local()
        on_value_local = on_value.to_local() if hasattr(on_value, '_layout') else on_value
        off_value_local = off_value.to_local() if hasattr(off_value, '_layout') else off_value

        on_value_layout = on_value.layout if hasattr(on_value, '_layout') else None
        off_value_layout = off_value.layout if hasattr(off_value, '_layout') else None

        local_args = (indices_local, num_classes, on_value_local, off_value_local, axis)
        local_kwargs = {}
        cache_values = [indices.layout, on_value_layout, off_value_layout, num_classes, axis]
        return local_args, local_kwargs, cache_values

    # pylint: disable=W0237
    def infer_layout(self, cache_values: list) -> Tuple[tuple, None]:
        """
        Infer output layout for OneHotExt.

        Rules:
            1. Indices must not have Partial status.
            2. num_classes must be int >= -1.
            3. axis must be in [-1, 1].
            4. For multi-dimensional input (>1D), axis must be -1 and only dim0 may be sharded.
            5. Non-indices inputs must be fully replicated.
            6. Output layout inserts a replicated one-hot dimension at the specified axis.

        Args:
            cache_values (list): [indices_layout, on_value_layout, off_value_layout, num_classes, axis]

        Returns:
            tuple: ((output_layout,), None)

        Raises:
            ValueError: If any rule above is violated.
            TypeError: If num_classes or axis has invalid type.
        """
        indices_layout = cache_values[0]
        on_value_layout = cache_values[1]
        off_value_layout = cache_values[2]
        num_classes = cache_values[3]
        axis = cache_values[4]

        if indices_layout is None or indices_layout.mesh_shape is None:
            raise ValueError(
                f"For {self.op_name}, indices layout cannot be None."
            )

        if not self._allow_partial_inputs:
            self._check_partial_inputs([indices_layout])

        self._validate_num_classes(num_classes)
        axis = self._validate_axis(axis)

        in_tensor_map = indices_layout.tensor_map
        if not in_tensor_map:
            raise ValueError(
                f"For {self.op_name}, indices tensor_map is empty."
            )

        self._validate_multi_dim_restriction(in_tensor_map, axis, indices_layout)
        self._validate_inputs_layouts(
            [indices_layout, on_value_layout, off_value_layout]
        )

        out_tensor_map = self._infer_output_tensor_map(in_tensor_map, axis)
        out_layout = self._create_layout_from_tensor_map(indices_layout, out_tensor_map)
        out_layout.tensor_map_to_placement()

        return ((out_layout,), None)

    # pylint: disable=W0237
    def get_expand_impl(self, func, infer_result, cache_values):
        """
        Get expanded implementation for OneHotExt operator.

        When indices are sharded and num_classes is -1 (auto-detect), returns a
        closure that computes the global maximum index across all shards via
        AllReduce(max) before calling the original operator.

        Args:
            func: Original operator callable.
            infer_result: Result from infer_layout (unused).
            cache_values (list): [indices_layout, on_value_layout, off_value_layout, num_classes, axis]

        Returns:
            Optional[callable]: Closure or None if no expansion is needed.
        """
        # pylint: disable=C0415
        import mindspore as ms
        from mindspore import ops, Tensor

        indices_layout = cache_values[0]
        if indices_layout is None or indices_layout.mesh_shape is None:
            return None

        sharded_axes = self._get_sharded_axes(indices_layout)
        if not sharded_axes:
            return None

        original_op = func
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
                    f"For {self.op_name}, indices max value {local_max_host} "
                    f"exceeds int32 range."
                )

            zero_dim = local_max.ndim == 0
            local_max_i32 = ops.cast(local_max, ms.int32)

            if zero_dim:
                local_max_i32 = ops.expand_dims(local_max_i32, 0)

            global_max_i32 = local_max_i32
            for axis_name in sharded_axes:
                group = indices_layout.get_comm_group_by_axis(axis_name)
                global_max_i32 = platform.differentiable_all_reduce(
                    global_max_i32, "max", group
                )

            if zero_dim:
                global_max_i32 = ops.squeeze(global_max_i32, 0)

            depth = int(global_max_i32.asnumpy()) + 1
            return original_op(indices, depth, on_value, off_value, axis)

        return expanded_one_hot

    def _validate_num_classes(self, num_classes):
        """Validate num_classes parameter."""
        if not isinstance(num_classes, int):
            raise TypeError(
                f"For {self.op_name}, num_classes should be int, "
                f"but got {type(num_classes).__name__}."
            )
        if num_classes < -1:
            raise ValueError(
                f"For {self.op_name}, num_classes should be >= -1, "
                f"but got {num_classes}."
            )

    def _validate_indices_dtype(self, indices):
        """Validate indices dtype."""
        # pylint: disable=C0415
        import mindspore as ms

        if indices.dtype != ms.int64:
            raise TypeError(
                f"For {self.op_name}, indices dtype should be int64, "
                f"but got {indices.dtype}."
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

    def _validate_axis(self, axis):
        """Validate axis parameter."""
        if not isinstance(axis, int):
            raise TypeError(
                f"For {self.op_name}, axis should be int, "
                f"but got {type(axis).__name__}."
            )

        if axis > 1 or axis < -1:
            raise ValueError(
                f"For {self.op_name}, axis {axis} is out of range [-1, 1]."
            )

        return axis

    def _validate_multi_dim_restriction(self, in_tensor_map, axis, indices_layout):
        """Validate restriction for multi-dimensional inputs."""
        in_rank = len(in_tensor_map)
        if in_rank <= 1:
            return

        if axis != -1:
            raise ValueError(
                f"For {self.op_name}, when input dimension is > 1, axis should be -1, "
                f"but got {axis}."
            )

        alias_map = indices_layout.alias_tensor_map
        for i in range(1, len(alias_map)):
            if alias_map[i] != "None":
                raise ValueError(
                    f"For {self.op_name}, when input dimension is > 1, "
                    f"strategy should be data parallel, "
                    f"but dimension {i} is sharded on '{alias_map[i]}'."
                )

    def _validate_inputs_layouts(self, layouts):
        """Validate that non-indices inputs are fully replicated."""
        for layout in layouts[1:]:
            if layout is None:
                continue
            alias_map = layout.alias_tensor_map
            if alias_map and any(x != "None" for x in alias_map):
                raise ValueError(
                    f"For {self.op_name}, non-indices inputs should be replicated, "
                    f"but got {alias_map}."
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
                f"For {self.op_name}, axis {axis} is out of range "
                f"for input with rank {in_rank}."
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
        out_layout.tensor_map_to_placement()
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
    