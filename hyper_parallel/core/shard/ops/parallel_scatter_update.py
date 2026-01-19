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
Distributed implementation for ScatterUpdate operator.
"""
from hyper_parallel.core.layout import Layout
from .parallel_ops import DistributedOp


class ScatterUpdateDistributedOp(DistributedOp):
    """Distributed implementation for ScatterUpdate operator."""

    def infer_layout(self, layouts, extra_args):
        """
        Infer output layout for ScatterUpdate.

        Args:
            layouts (tuple): Tuple containing (input_layout, indices_layout, updates_layout).
            extra_args: Additional arguments (not used).

        Returns:
            Layout: Output layout (same as input layout).
        """
        if len(layouts) != 3:
            raise ValueError(f"{self.__class__.__name__} requires exactly 3 input layouts")

        input_layout, indices_layout, updates_layout = layouts

        if input_layout.mesh_shape is None:
            raise ValueError("Input layout cannot be None")

        self._validate_strategy(input_layout, indices_layout, updates_layout)

        output_layout = Layout(
            mesh_shape=input_layout.mesh_shape,
            alias_name=input_layout.alias_name,
            rank_list=input_layout.rank_list
        )
        output_layout = output_layout(*input_layout.alias_tensor_map)

        for i, partial_op in enumerate(input_layout.partial):
            if partial_op is not None:
                dev_axis_name = input_layout.alias_name[i]
                output_layout.set_partial_by_dev_axis(dev_axis_name, partial_op)

        return output_layout

    def _validate_strategy(self, input_layout, indices_layout, updates_layout):
        """Validate sharding strategy for ScatterUpdate."""
        input_map = input_layout.alias_tensor_map
        indices_map = indices_layout.alias_tensor_map
        updates_map = updates_layout.alias_tensor_map

        if not input_map:
            raise ValueError(f"{self.op_name}: input tensor map is empty")

        if input_map[0] != "None":
            raise ValueError(
                f"{self.op_name}: first dimension of input cannot be sharded"
            )

        for i, axis in enumerate(indices_map):
            if axis != "None":
                raise ValueError(
                    f"{self.op_name}: indices cannot be sharded, "
                    f"but dimension {i} is sharded on '{axis}'"
                )

        indices_ndim = len(indices_map)
        for i in range(indices_ndim):
            if i >= len(updates_map):
                raise ValueError(
                    f"{self.op_name}: updates rank is smaller than indices rank"
                )
            if updates_map[i] != "None":
                raise ValueError(
                    f"{self.op_name}: first {indices_ndim} dimensions of updates cannot be sharded, "
                    f"but dimension {i} is sharded on '{updates_map[i]}'"
                )

        expected_updates_ndim = indices_ndim + len(input_map) - 1
        if len(updates_map) != expected_updates_ndim:
            raise ValueError(
                f"{self.op_name}: updates rank mismatch. "
                f"Expected {expected_updates_ndim}, got {len(updates_map)}"
            )

        for i in range(1, len(input_map)):
            updates_idx = indices_ndim + i - 1
            if input_map[i] != updates_map[updates_idx]:
                raise ValueError(
                    f"{self.op_name}: updates sharding must match input[1:]. "
                    f"Mismatch at input dim {i}: '{input_map[i]}' != '{updates_map[updates_idx]}'"
                )
