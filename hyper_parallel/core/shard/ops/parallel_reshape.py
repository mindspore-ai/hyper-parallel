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
"""
Distributed implementation for Reshape operator.
"""

from hyper_parallel.core.dtensor.layout import Layout
from hyper_parallel.platform import get_platform
from .parallel_ops import DistributedOp
platform = get_platform()
Tensor = platform.Tensor


def _filter_none_split_tensor_map(tensor_map, mesh_shape):
    """
    Filter out the elements in tensor_map where the size of the corresponding dimension in device_matrix is 1.

    Args:
        tensor_map (list): A list of tensor mappings, which may contain integers or tuples.
        device_matrix (list): A device matrix representing the device distribution across each dimension.

    Returns:
        list: The filtered list of tensor mappings, where invalid mappings are replaced with -1 or valid mappings are
        retained.
    """
    filtered_tensor_map = []
    for item in tensor_map:
        if isinstance(item, tuple):
            filtered = []
            for i in item:
                if mesh_shape[-1 - i] != 1:
                    filtered.append(i)
            if len(filtered) == 0:
                filtered_tensor_map.append(-1)
            elif len(filtered) == 1:
                filtered_tensor_map.append(filtered[0])
            else:
                filtered_tensor_map.append(tuple(filtered))
        else:
            filtered_tensor_map.append(item if mesh_shape[-1 - item] != 1 else -1)
    return filtered_tensor_map


class ReshapeDistributedOp(DistributedOp):
    """Distributed implementation for Reshape operator."""

    def __init__(self, op_name):
        super().__init__(op_name)
        self._allow_partial_inputs = True

    def _get_dynamic_shape_info(self, shape):
        total_size = 1
        dynamic_axis = -1
        for axis, s in enumerate(shape):
            total_size *= s
            if s < 0:
                dynamic_axis = axis
        return total_size < 0, dynamic_axis, total_size

    def _handle_dynamic_shape(self, input_shape, output_shape):
        """
        Check dynamic shape. Calculate unknown axis if one of input and output shape is known. If both are unknown,
        calculate the relative multiple.
        [2, -1, 8], [4, -1, 8] -> [2, -2, 8], [4, -1, 8]
        """
        input_shape = list(input_shape)
        output_shape = list(output_shape)
        is_input_dynamic, input_dynamic_axis, input_total_size = self._get_dynamic_shape_info(input_shape)
        is_output_dynamic, output_dynamic_axis, output_total_size = self._get_dynamic_shape_info(output_shape)
        dynamic_can_shard = False
        if not is_input_dynamic and not is_output_dynamic:
            if input_total_size != output_total_size:
                raise ValueError(f"The total elements number of input shape {input_shape} and output shape "
                                 f"{output_shape} are different.")
            return input_shape, output_shape, dynamic_can_shard

        if not is_input_dynamic:
            accurate_output_shape = output_shape
            accurate_output_shape[output_dynamic_axis] = -input_total_size // output_total_size
            return input_shape, accurate_output_shape, dynamic_can_shard

        if not is_output_dynamic:
            accurate_input_shape = input_shape
            accurate_input_shape[input_dynamic_axis] = -output_total_size // input_total_size
            return accurate_input_shape, output_shape, dynamic_can_shard

        if output_total_size >= input_total_size:
            output_shape[output_dynamic_axis] = -(input_total_size // output_total_size)
            dynamic_can_shard = True
        else:
            input_shape[input_dynamic_axis] = -(output_total_size // input_total_size)
        return input_shape, output_shape, dynamic_can_shard

    def _merge_unshared_axis(self, global_shape, tensor_map):
        """
        Merge those axes that are not sharded to the high dimension which is shared.
        shape[4, 2, 6, 8], tensor map[-1, -1, 0, -1] -> merged shape[8, 48]
        """
        merged_size = 1
        merged_shape = []
        merged_tensor_map = []
        for axis in range(len(global_shape) - 1, -1, -1):
            merged_size *= global_shape[axis]
            if tensor_map[axis] != -1:
                merged_shape.insert(0, merged_size)
                merged_tensor_map.insert(0, tensor_map[axis])
                merged_size = 1
        if tensor_map[0] == -1:
            merged_shape.insert(0, merged_size)
            merged_tensor_map.insert(0, -1)
        return merged_shape, merged_tensor_map


    def _cal_output_layout_and_dst_shape(self, output_tensor_map, dst_shape, x_dict):
        """
        calculate output layout tensor map and local dst shape.
        """
        x_mesh_shape = x_dict["mesh_shape"]
        output_map = []
        local_dst_shape = []
        for idx, map_id in enumerate(output_tensor_map):
            if isinstance(map_id, tuple):
                shard_size = 1
                map_idx = []
                for shard_id in map_id:
                    map_idx.append(x_dict["alias_name"][-1 - shard_id])
                    shard_size *= x_mesh_shape[-1 - shard_id]
                output_map.append(tuple(map_idx))
                local_dst_shape.append(dst_shape[idx] // shard_size if dst_shape[idx] > 0 else -1)
                continue
            if map_id < 0:
                output_map.append("None")
                local_dst_shape.append(dst_shape[idx] if dst_shape[idx] > 0 else -1)
            else:
                output_map.append(x_dict["alias_name"][-1 - map_id])
                local_dst_shape.append(dst_shape[idx] // x_mesh_shape[-1 - map_id] if dst_shape[idx] > 0 else -1)
        return output_map, local_dst_shape

    def _parse_shape_args(self, extra_args):
        """Parse shape arguments from extra_args.

        Args:
            extra_args: Extra arguments containing shape info

        Returns:
            tuple: (dst_shape, input_shape)
        """
        if self.op_name in ["reshape", "view"]:
            return self._parse_torch_shape_args(extra_args)
        return self._parse_mindspore_shape_args(extra_args)

    def _parse_torch_shape_args(self, extra_args):
        """Parse PyTorch style shape arguments."""
        if len(extra_args) < 2:
            raise ValueError(f"{self.op_name} requires output shape and input shape.")

        input_shape = extra_args[-1]
        shape_args = extra_args[:-1]

        if len(shape_args) == 1:
            first_arg = shape_args[0]
            if isinstance(first_arg, (list, tuple)):
                dst_shape = first_arg
            elif isinstance(first_arg, Tensor):
                dst_shape = first_arg.tolist()
            else:
                dst_shape = shape_args
        else:
            dst_shape = shape_args

        return dst_shape, input_shape

    def _parse_mindspore_shape_args(self, extra_args):
        """Parse MindSpore style shape arguments."""
        if len(extra_args) != 2:
            raise ValueError("Reshape requires output shape and input shape.")

        return extra_args[0], extra_args[1]

    def _normalize_shape(self, dst_shape):
        """Normalize dst_shape to list format."""
        if isinstance(dst_shape, Tensor):
            dst_shape = dst_shape.tolist()
        if not isinstance(dst_shape, (list, tuple)):
            raise ValueError("Shape should be a tensor or a tuple or a list.")
        return dst_shape

    def _compute_output_tensor_map(self, merged_shape, merge_tensor_map, dst_shape, x_mesh_shape, dynamic_can_shard,
                                   input_shape, x_map):
        """Compute output tensor_map from merged information.

        Args:
            merged_shape: Merged shape from _merge_unshared_axis
            merge_tensor_map: Merged tensor_map from _merge_unshared_axis
            dst_shape: Target shape
            x_mesh_shape: Mesh shape
            dynamic_can_shard: Whether dynamic shape can be sharded
            input_shape: Original input shape
            x_map: Input tensor_map

        Returns:
            list: Output tensor_map
        """
        output_tensor_map = []
        cur_axis = len(merged_shape) - 1
        cur_size = merged_shape[cur_axis]

        for shape in reversed(dst_shape):
            if cur_size % shape != 0:
                raise ValueError(f"Can not reshape {input_shape} to {dst_shape} with tensor map {x_map}")
            cur_size = cur_size // shape

            if cur_size == 1:
                map_val = self._handle_sharded_axis(
                    merge_tensor_map, cur_axis, x_mesh_shape, shape, dynamic_can_shard, input_shape, x_map, dst_shape
                )
                output_tensor_map.insert(0, map_val)
                cur_axis -= 1
                cur_size = merged_shape[cur_axis]
            else:
                output_tensor_map.insert(0, -1)

        return output_tensor_map

    def _handle_sharded_axis(self, merge_tensor_map, cur_axis, x_mesh_shape, shape, dynamic_can_shard,
                              input_shape, x_map, dst_shape):
        """Handle sharded axis in tensor_map computation."""
        map_val = merge_tensor_map[cur_axis]

        if isinstance(map_val, tuple):
            shard_size = 1
            for axis in map_val:
                shard_size *= x_mesh_shape[-axis - 1]
        else:
            shard_size = x_mesh_shape[-map_val - 1]

        if shape < 0:
            if not dynamic_can_shard:
                raise ValueError(f"Can not reshape {input_shape} to {dst_shape} with tensor map {x_map}")
        elif shard_size > shape or shape % shard_size != 0:
            raise ValueError(f"Can not reshape {input_shape} to {dst_shape} with tensor map {x_map}")

        return map_val

    def _apply_partial_status(self, x_layout, out_layout):
        """Apply partial status from input to output layout."""
        if x_layout.is_partial():
            input_partial = x_layout.partial
            for i, partial_op in enumerate(input_partial):
                if partial_op is not None and i < len(out_layout.alias_name):
                    out_layout.set_partial_by_dev_axis(out_layout.alias_name[i], partial_op)

    def infer_layout(self, layouts, extra_args=None):
        """
        Infer output layout for reshape operator.

        For reshape operations, data slice on each device after reshape should be same as data slice before reshape.

        Args:
            layouts (Layout): Layout of input x
            extra_args:
                For MindSpore Reshape: (destination shape, original shape)
                For PyTorch reshape/view: (shape_arg1, shape_arg2, ..., original shape) or (shape_tuple, original shape)

        Returns:
            tuple: Layout for output tensor
        """
        x_layout = layouts[0]
        x_dict = x_layout.to_dict()

        dst_shape, input_shape = self._parse_shape_args(extra_args)
        dst_shape = self._normalize_shape(dst_shape)

        x_map = _filter_none_split_tensor_map(x_dict["tensor_map"], x_dict["mesh_shape"])
        x_mesh_shape = x_dict["mesh_shape"]

        input_shape, dst_shape, dynamic_can_shard = self._handle_dynamic_shape(input_shape, dst_shape)
        merged_shape, merge_tensor_map = self._merge_unshared_axis(input_shape, x_map)

        output_tensor_map = self._compute_output_tensor_map(
            merged_shape, merge_tensor_map, dst_shape, x_mesh_shape, dynamic_can_shard, input_shape, x_map
        )

        output_layout = Layout(
            mesh_shape=x_mesh_shape,
            alias_name=x_layout.alias_name,
            rank_list=x_layout.rank_list
        )
        output_map, local_dst_shape = self._cal_output_layout_and_dst_shape(output_tensor_map, dst_shape, x_dict)
        out_layout = output_layout(*output_map)

        self._apply_partial_status(x_layout, out_layout)

        return out_layout, local_dst_shape
