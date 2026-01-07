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
Distributed implementation for StackExt operator.
"""

from hyper_parallel.core.layout import Layout
from hyper_parallel.core.shard.ops.parallel_ops import DistributedOp


class StackExtDistributedOp(DistributedOp):
    """Distributed implementation for StackExt operator."""

    def infer_layout(self, layouts, extra_args):
        """
        Infer output layout for StackExt.

        Args:
            layouts: tuple/list of Layout (or None). Each input tensor has same shape.
            extra_args: [axis]  axis is the insert-dimension index on output tensor.

        Returns:
            Layout: output layout

        Raises:
            ValueError: layout mismatch or invalid axis
        """
        if not layouts or layouts[0] is None:
            raise ValueError(f"Operation {self.op_name}: base layout is None")

        axis = extra_args[0]
        base_layout = layouts[0]

        in_rank = len(base_layout.tensor_map)
        out_rank = in_rank + 1

        if axis < -out_rank or axis >= out_rank:
            raise ValueError(
                f"Operation {self.op_name}: axis value is out of valid range"
            )
        if axis < 0:
            axis += out_rank

        base_mesh_shape = base_layout.mesh_shape
        base_map = base_layout.tensor_map

        for layout in layouts[1:]:
            if not layout:
                continue

            if layout.mesh_shape != base_mesh_shape:
                raise ValueError(
                    f"Operation {self.op_name}: inputs must have same mesh_shape"
                )

            if layout.tensor_map != base_map:
                raise ValueError(
                    f"Operation {self.op_name}: inputs must have same tensor_map"
                )

        out_alias_map = (
            list(base_layout.alias_tensor_map[:axis])
            + ["None"]
            + list(base_layout.alias_tensor_map[axis:])
        )

        output_layout = Layout(
            base_layout.mesh_shape,
            base_layout.alias_name,
            base_layout.rank_list,
        )
        output_layout = output_layout(*out_alias_map)
        return output_layout
