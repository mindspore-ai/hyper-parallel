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
Distributed implementation for Gather operator.
"""

from dist_parallel.core.layout import Layout
from .parallel_ops import DistributedOp


class IndexSelectDistributedOp(DistributedOp):
    """Distributed implementation for Index Select operator."""

    def infer_layout(self, layouts, extra_args):
        """
        Infer output layouts for Index Select operations.

        Args:
            layouts: Layouts of input tensors
            extra_args: extra_args of input tensors

        Returns:
            tuple: Layout for output tensor.

        Raises:
            ValueError: If input layouts are not compatible.
        """
        # Check
        if len(layouts) != 3:
            raise ValueError(f"Gather ops requires 3 layouts, but {len(layouts)}")
        if len(extra_args) != 1:
            raise ValueError(f"Gather ops requires 1 extra args, but {len(extra_args)}")

        # Parse layout info
        p_layout, i_layout = layouts[0], layouts[2]
        axis, batch_dims = extra_args[0], 0

        p_tensor_map = p_layout.alias_tensor_map
        i_tensor_map = i_layout.alias_tensor_map

        # Create output layout
        if p_tensor_map[axis] != "None":
            raise ValueError(
                f"Operation {self.op_name}: Cannot perform sharding on params along the axis"
            )

        if len(i_tensor_map) != 1:
            raise ValueError(
                f"Operation {self.op_name}: index is not a one-dimensional Tensor"
            )

        if axis < -len(p_tensor_map) or axis >= len(p_tensor_map):
            raise ValueError(
                f"Operation {self.op_name}: dim value is out of valid range"
            )

        output_tensor_map = (
            p_tensor_map[:axis] + i_tensor_map[batch_dims:] + p_tensor_map[axis + 1 :]
        )
        output_layout = i_layout
        output_layout = Layout(
            device_matrix=output_layout.device_matrix,
            alias_name=output_layout.alias_name,
            rank_list=output_layout.rank_list,
        )
        output_layout = output_layout(*output_tensor_map)
        return output_layout


class GatherDistributedOp(DistributedOp):
    """Distributed implementation for Gather operator."""

    def infer_layout(self, layouts, extra_args):
        """
        Infer output layouts for Gather operations.

        Args:
            layouts: Layouts of input tensors
            extra_args: extra_args of input tensors

        Returns:
            tuple: Layout for output tensor.

        Raises:
            ValueError: If input layouts are not compatible.
        """
        # Check
        if len(layouts) != 3:
            raise ValueError(f"Gather ops requires 3 layouts, but {len(layouts)}")
        if len(extra_args) != 1:
            raise ValueError(f"Gather ops requires 1 extra args, but {len(extra_args)}")

        # Parse layout info
        p_layout, i_layout = layouts[0], layouts[2]
        axis = extra_args[0]

        p_tensor_map = p_layout.alias_tensor_map
        i_tensor_map = i_layout.alias_tensor_map

        # Create output layout
        if p_tensor_map[axis] != "None":
            raise ValueError(
                f"Operation {self.op_name}: Cannot perform sharding on params along the axis"
            )

        if len(p_tensor_map) != len(i_tensor_map):
            raise ValueError(
                f"Operation {self.op_name}: input and index must have the same number of dimensions"
            )

        if axis < -len(p_tensor_map) or axis >= len(p_tensor_map):
            raise ValueError(
                f"Operation {self.op_name}: dim value is out of valid range"
            )

        output_tensor_map = i_tensor_map
        output_layout = i_layout
        output_layout = Layout(
            device_matrix=output_layout.device_matrix,
            alias_name=output_layout.alias_name,
            rank_list=output_layout.rank_list,
        )
        output_layout = output_layout(*output_tensor_map)
        return output_layout
