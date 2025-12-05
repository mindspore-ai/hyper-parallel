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

from ... import Layout
from mindspore.common.tensor import Tensor
from .parallel_ops import DistributedOp


class ReshapeDistributedOp(DistributedOp):
    """Distributed implementation for MatMul operator."""

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
        x_device_matrix = x_dict["device_matrix"]
        output_map = []
        local_dst_shape = []
        for idx, map_id in enumerate(output_tensor_map):
            if isinstance(map_id, tuple):
                shard_size = 1
                map_idx = []
                for shard_id in map_id:
                    map_idx.append(x_dict["alias_name"][-1 - shard_id])
                    shard_size *= x_device_matrix[-1 - shard_id]
                output_map.append(tuple(map_idx))
                local_dst_shape.append(dst_shape[idx] // shard_size if dst_shape[idx] > 0 else -1)
                continue
            if map_id < 0:
                output_map.append("None")
                local_dst_shape.append(dst_shape[idx] if dst_shape[idx] > 0 else -1)
            else:
                output_map.append(x_dict["alias_name"][-1 - map_id])
                local_dst_shape.append(dst_shape[idx] // x_device_matrix[-1 - map_id] if dst_shape[idx] > 0 else -1)
        return output_map, local_dst_shape

    def infer_layout(self, input_layouts, extra_args):
        """
        Infer output layout for reshape operator.

        For reshape operations, data slice on each device after reshape should be same as data slice before reshape.

        Args:
            input_layouts (Layout): Layout of input x
            extra_args: (destination shape, original shape)

        Returns:
            tuple: Layout for output tensor
        """
        x_layout = input_layouts[0]
        x_dict = x_layout.to_dict()

        if len(extra_args) != 2:
            raise ValueError("Reshape requires output shape and input shape.")
        # Check output shape.
        dst_shape = extra_args[0]
        if isinstance(dst_shape, Tensor):
            dst_shape = dst_shape.tolist()
        if not isinstance(dst_shape, list) and not isinstance(dst_shape, tuple):
            raise ValueError("Shape should be a tensor or a tuple or a list.")

        input_shape = extra_args[1]

        x_map = x_dict["tensor_map"]
        x_device_matrix = x_dict["device_matrix"]

        input_shape, dst_shape, dynamic_can_shard = self._handle_dynamic_shape(input_shape, dst_shape)
        merged_shape, merge_tensor_map = self._merge_unshared_axis(input_shape, x_map)

        output_tensor_map = []
        cur_axis = len(merged_shape) - 1
        cur_size = merged_shape[cur_axis]
        for shape in reversed(dst_shape):
            if cur_size % shape != 0:
                raise ValueError(f"Can not reshape {input_shape} to {dst_shape} with tensor map {x_map}")
            cur_size = cur_size // shape
            if cur_size == 1:
                if isinstance(merge_tensor_map[cur_axis], tuple):
                    shard_size = 1
                    for axis in merge_tensor_map[cur_axis]:
                        shard_size *= x_device_matrix[-axis - 1]
                else:
                    shard_size = x_device_matrix[-merge_tensor_map[cur_axis] - 1]
                if shape < 0:
                    if not dynamic_can_shard:
                        raise ValueError(f"Can not reshape {input_shape} to {dst_shape} with tensor map {x_map}")
                elif shard_size > shape or shape % shard_size != 0:
                    raise ValueError(f"Can not reshape {input_shape} to {dst_shape} with tensor map {x_map}")
                output_tensor_map.insert(0, merge_tensor_map[cur_axis])
                cur_axis -= 1
                cur_size = merged_shape[cur_axis]
            else:
                output_tensor_map.insert(0, -1)

        output_layout = Layout(
            device_matrix=x_device_matrix,
            alias_name=x_layout.alias_name,
            rank_list=x_layout.rank_list
        )
        output_map, local_dst_shape = self._cal_output_layout_and_dst_shape(output_tensor_map, dst_shape, x_dict)
        out_layout = output_layout(*output_map)
        return out_layout, local_dst_shape
