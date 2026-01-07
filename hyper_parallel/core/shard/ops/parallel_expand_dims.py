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
Distributed implementation for ExpandDims operator.
"""
from hyper_parallel.core.layout import Layout
from .parallel_ops import DistributedOp


class ExpandDimsDistributedOp(DistributedOp):
    """Distributed implementation for ExpandDims operator."""

    def __init__(self, op_name="ExpandDims"):
        super().__init__(op_name)

    def infer_layout(self, layouts, extra_args):
        """
        Infer output layout for ExpandDims.

        Args:
            layouts (tuple): Tuple containing input layout.
            extra_args: axis parameter (int or list/tuple containing int).

        Returns:
            Layout: Output layout with inserted dimension.
        """
        if not layouts:
            raise ValueError(f"{self.__class__.__name__} requires at least one input layout")

        x_layout = layouts[0]

        if x_layout.mesh_shape is None:
            raise ValueError("Input layout cannot be None.")

        if not extra_args:
            raise ValueError("ExpandDims: axis parameter is required")

        axis = extra_args[0] if isinstance(extra_args, (tuple, list)) else extra_args

        in_rank = len(x_layout.alias_tensor_map)
        if axis < 0:
            axis = axis + in_rank + 1

        if axis < 0 or axis > in_rank:
            raise ValueError(
                f"ExpandDims: axis {axis} out of range for input rank {in_rank}. "
                f"Valid range is [{-in_rank-1}, {in_rank}]"
            )

        x_map = list(x_layout.alias_tensor_map)
        x_map.insert(axis, "None")

        output_layout = Layout(
            mesh_shape=x_layout.mesh_shape,
            alias_name=x_layout.alias_name,
            rank_list=x_layout.rank_list
        )
        output_layout = output_layout(*x_map)

        for i, partial_op in enumerate(x_layout.partial):
            if partial_op is not None:
                dev_axis_name = x_layout.alias_name[i]
                output_layout.set_partial_by_dev_axis(dev_axis_name, partial_op)

        return output_layout
