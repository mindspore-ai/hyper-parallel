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
Distributed implementation for Concat operator.
"""

from hyper_parallel.core.layout import Layout
from .parallel_ops import DistributedOp


class ConcatDistributedOp(DistributedOp):
    """Distributed implementation for Concat operator."""

    def infer_layout(self, layouts, extra_args):
        """
        Infer output layouts for Concat operations.

        Args:
            layouts: Layouts of input tensors
            extra_args: extra_args of input tensors

        Returns:
            tuple: Layout for output tensor.

        Raises:
            ValueError: If input layouts are not compatible.
        """
        # Parse input layout
        dim = extra_args[0]

        base_layout = layouts[0]
        rank = len(base_layout.tensor_map)
        base_map = base_layout.tensor_map
        base_device_matrix = base_layout.device_matrix

        if dim < -rank or dim >= rank:
            raise ValueError(
                f"Operation {self.op_name}: dim value is out of valid range"
            )

        for layout in layouts[1:]:
            if not layout:
                continue

            if layout.device_matrix != base_device_matrix:
                raise ValueError(
                    f"Operation {self.op_name}: Concat inputs must have same device_matrix"
                )

            if layout.tensor_map[:dim] + layout.tensor_map[dim + 1 :] != base_map[:dim] + base_map[dim + 1 :]:
                raise ValueError(
                    f"Operation {self.op_name}: Except for dim, the tensor map of inputs must be equal"
                )

        # Create output layout
        output_layout = Layout(
            device_matrix=base_layout.device_matrix,
            alias_name=base_layout.alias_name,
            rank_list=base_layout.rank_list,
        )
        output_layout = output_layout(*base_layout.alias_tensor_map)
        return output_layout
