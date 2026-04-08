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
Distributed implementation for InplaceScatterValue operator.
"""

from .parallel_ops import DistributedOp


class InplaceScatterValueDistributedOp(DistributedOp):
    """Distributed implementation for InplaceScatterValue operator."""

    def infer_layout(self, layouts, extra_args):
        """
        Infer output layout for InplaceScatterValue.

        Requirements:
            1. Must have exactly 4 inputs: input, dim, index, value
            2. extra_args must have exactly 2 elements: dim (int), value (scalar)
            3. Output layout = input layout (inplace)
        """
        if not layouts or len(layouts) != 4:
            raise ValueError(
                f"Operation {self.op_name}: InplaceScatterValue requires exactly 4 inputs: "
                f"input, dim, index, value. Got {len(layouts) if layouts else 0}."
            )

        input_layout = layouts[0]
        index_layout = layouts[2]
        if input_layout is None or not hasattr(input_layout, "tensor_map"):
            raise ValueError(
                f"Operation {self.op_name}: input tensor layout cannot be None."
            )
        if index_layout is None or not hasattr(index_layout, "tensor_map"):
            raise ValueError(
                f"Operation {self.op_name}: index tensor layout cannot be None."
            )
        input_map = input_layout.tensor_map
        index_map = index_layout.tensor_map
        ndim = len(input_map)
        if len(input_map) != len(index_map):
            raise ValueError(
                f"Operation {self.op_name}: input and index must have the same number of dimensions. "
                f"Got input rank={len(input_map)}, index rank={len(index_map)}"
            )

        if not extra_args or len(extra_args) != 2:
            raise ValueError(
                f"Operation {self.op_name}: extra_args must contain exactly 2 elements: "
                f"dim (int), value (scalar). Got {len(extra_args) if extra_args else 0}."
            )
        dim = extra_args[0]

        if not isinstance(dim, int):
            raise ValueError(f"Operation {self.op_name}: 'dim' must be an integer.")
        if dim < 0:
            dim += ndim
        if dim < 0 or dim >= ndim:
            raise ValueError(
                f"Operation {self.op_name}: dim {dim} is out of bounds for tensor with {ndim} dims."
            )
        for axis, (input_axis_map, index_axis_map) in enumerate(zip(input_map, index_map)):
            if input_axis_map != index_axis_map:
                raise ValueError(
                    f"Operation {self.op_name}: input and index must use the same sharding on non-dim axis {axis}. "
                    f"Got input tensor_map={input_map}, index tensor_map={index_map}, dim={dim}"
                )

        if input_map[dim] != -1:
            raise ValueError(
                f"Operation {self.op_name}: Scatter along sharded dimension {dim} is not supported. "
                f"The target dimension must be replicated (unsharded)."
            )

        return input_layout
