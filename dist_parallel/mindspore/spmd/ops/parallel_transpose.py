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
Distributed implementation for MatMul operator.
"""

from ... import Layout
from .parallel_ops import DistributedOp


class TransposeDistributedOp(DistributedOp):
    """Distributed implementation for MatMul operator."""       

    def infer_layout(self, layouts, extra_args):
        """
        Infer output layout for Transpose operator.

        Rules:
          1. Output layout is determined by input layout and permutation

        Args:
            input_layouts: Layouts of input tensor
            extra_args (tuple): permutation of transpose

        Returns:
            tuple: Layout for output tensor
        """

        layout = layouts[0]
        axis = extra_args[0]
        in_tensor_map = layout.alias_tensor_map

        if len(in_tensor_map) != len(axis):
            raise ValueError(f"Input tensor shape and permutation must have the same size."
                             f"Got {len(in_tensor_map)} and {len(axis)}")

        # check if axis is a permutation
        num = len(in_tensor_map)
        seen = set()
        for v in axis:
            if v < 0 or v >= num or v in seen:
                raise ValueError("Invalid permutation")
            seen.add(v)

        out_tensor_map = type(in_tensor_map)(in_tensor_map[i] for i in axis)

        output_layout = Layout(
            device_matrix=layout.device_matrix,
            alias_name=layout.alias_name,
            rank_list=layout.rank_list
        )

        out_layout = output_layout(*out_tensor_map)

        return out_layout
