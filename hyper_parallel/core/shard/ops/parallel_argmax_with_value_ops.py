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
Distributed implementation for ArgMaxWithValue operator.
"""

from .parallel_ops import DistributedOp


class ArgMaxWithValueDistributedOp(DistributedOp):
    """Distributed implementation for ArgMaxWithValue operator."""

    def infer_layout(self, layouts, extra_args):
        """
        Infer output layout for ArgMaxWithValue operator.
        Args:
            x_layout (Layout): Layout of input x
        Returns:
            tuple: Layout for output tensor
        Raises:
            ValueError: If input layouts have partial status.
        """
        # Check partial inputs
        if not self._allow_partial_inputs:
            self._check_partial_inputs(layouts)

        # Parse input layout
        if len(layouts) != 3:
            raise ValueError(f"ArgMaxWithValue requires 3 layouts, but {len(layouts)}")
        if len(extra_args) != 2:
            raise ValueError(f"ArgMaxWithValue requires 2 extra args, but {len(extra_args)}")

        input_layout = layouts[0]
        input_mesh_shape = input_layout.mesh_shape
        input_tensor_map = input_layout.tensor_map
        input_alias_tensor_map = input_layout.alias_tensor_map
        axis, keep_dims = extra_args[0], extra_args[1]

        def is_shard(index):
            mapping = input_tensor_map[index]
            if isinstance(mapping, tuple):
                shard_flag = False
                for elem in mapping:
                    if elem != -1 and input_mesh_shape[len(input_mesh_shape) - 1 - elem] != 1:
                        shard_flag = True
                return shard_flag

            if mapping == -1 or input_mesh_shape[len(input_mesh_shape) - 1 - mapping] == 1:
                return False
            return True

        # Create output layout
        for i in range(len(input_tensor_map)):
            if axis != i and is_shard(i):
                raise ValueError(f"{self.__class__.__name__} cannot perform sharding on non axis dim")

        if not keep_dims:
            tensor_map = input_alias_tensor_map[:axis] + input_alias_tensor_map[axis + 1 :]
        else:
            tensor_map = input_alias_tensor_map[:axis] + ("None",) + input_alias_tensor_map[axis + 1 :]

        output_layout = input_layout
        output_layout = output_layout(*tensor_map)
        return output_layout, output_layout
