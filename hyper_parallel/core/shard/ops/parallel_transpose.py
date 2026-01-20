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

from hyper_parallel.core.layout import Layout
from .parallel_ops import DistributedOp


class TransposeDistributedOp(DistributedOp):
    """Distributed implementation for Transpose operator."""

    def infer_layout(self, layouts, extra_args):
        """
        Infer output layout for Transpose operator.

        Based on the op_name initialized in the base class, this method switches behavior:
        1. op_name == 'Transpose' or 'permute': Implements MindSpore Transpose behavior or PyTorch permute behavior.
           - extra_args expected: (perm,) where perm is a tuple of indices.
           - Rules: Output layout is determined by input layout and permutation.
        2. op_name == 'transpose': Implements PyTorch transpose behavior.
           - extra_args expected: (dim0, dim1) where dim0 and dim1 are integers.
           - Rules: Output layout is determined by swapping the specified dimensions in input layout.

        Args:
            layouts (tuple): Layouts of input tensor.
            extra_args (tuple): Arguments for the operator.

        Returns:
            Layout: Layout for output tensor.
        """
        layout = layouts[0]
        in_tensor_map = layout.alias_tensor_map
        ndim = len(in_tensor_map)
        out_tensor_map = None

        if self.op_name in ("Transpose", "permute"):
            # MindSpore style: Transpose(input, input_perm)
            # extra_args should contain a single element: the permutation tuple
            if not extra_args or not isinstance(extra_args[0], (list, tuple)):
                raise ValueError(f"For 'Transpose', expected permutation tuple in extra_args, got {extra_args}")

            axis = extra_args[0]

            if len(in_tensor_map) != len(axis):
                raise ValueError(f"Input tensor shape and permutation must have the same size. "
                                 f"Got {len(in_tensor_map)} and {len(axis)}")

            # check if axis is a permutation
            seen = set()
            for v in axis:
                if v < 0 or v >= ndim or v in seen:
                    raise ValueError(f"Invalid permutation {axis} for rank {ndim}")
                seen.add(v)

            out_tensor_map = tuple(in_tensor_map[i] for i in axis)

        elif self.op_name == "transpose":
            # PyTorch style: transpose(input, dim0, dim1)
            # extra_args should contain two elements: dim0 and dim1
            if len(extra_args) != 2:
                raise ValueError(f"For 'transpose', expected (dim0, dim1), got {extra_args}")

            dim0, dim1 = extra_args

            if not isinstance(dim0, int) or not isinstance(dim1, int):
                raise ValueError(f"Dimensions must be integers, got {dim0}, {dim1}")

            # Handle negative indices
            if dim0 < 0:
                dim0 += ndim
            if dim1 < 0:
                dim1 += ndim

            # Validate dimensions
            if not (0 <= dim0 < ndim and 0 <= dim1 < ndim):
                raise ValueError(f"Transpose dimensions out of bounds: ({dim0}, {dim1}) for rank {ndim}")

            # Swap the dimensions in the tensor map
            out_tensor_map_list = list(in_tensor_map)
            out_tensor_map_list[dim0], out_tensor_map_list[dim1] = out_tensor_map_list[dim1], out_tensor_map_list[dim0]
            out_tensor_map = tuple(out_tensor_map_list)

        else:
            raise ValueError(f"Unsupported op_name: {self.op_name}. Expected 'Transpose' , 'transpose' or 'permute'.")

        output_layout = Layout(
            mesh_shape=layout.mesh_shape,
            alias_name=layout.alias_name,
            rank_list=layout.rank_list
        )

        return output_layout(*out_tensor_map)
