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
Distributed implementation for new_ones operator.
"""


from hyper_parallel.core.layout import Layout
from .parallel_ops import DistributedOp


class NewOnesDistributedOp(DistributedOp):
    """Distributed implementation for new_ones operator."""

    def infer_layout(self, layouts, extra_args):
        """
        Infer output layout for new_ones operator.

        The new_ones operator creates a new tensor of ones with the specified size.
        In a distributed context, without explicit layout information for the new shape,
        the safest default is to create a Replicated tensor (tensor_map all -1) on the
        same Device Mesh as the input tensor.

        Args:
            layouts (tuple): Layouts of input tensor. layouts[0] is the layout of 'self'.
            extra_args (tuple): Arguments for the operator.
                                extra_args[0] is expected to be 'size'.

        Returns:
            Layout: Layout for output tensor.
        """
        # self input layout
        layout = layouts[0]

        # In OpDispatcher, non-tensor arguments are collected into extra_args.
        # For new_ones(size, dtype=...), size is the first argument.
        if not extra_args:
            raise ValueError(f"For '{self.op_name}', expected 'size' in extra_args, but got empty args.")

        size = extra_args[0]

        # Determine the number of dimensions for the new tensor
        if isinstance(size, int):
            ndim = 1
        elif isinstance(size, (tuple, list)):
            ndim = len(size)
        else:
            raise TypeError(f"new_ones 'size' argument must be int, tuple or list, but got {type(size)}")

        # Construct a Replicated tensor map using "None" string instead of -1 integer.
        # Layout.__call__ expects alias names (strings) or "None".
        out_tensor_map = tuple(["None"] * ndim)

        # Create a new layout reusing the device mesh configuration from the input
        output_layout = Layout(
            mesh_shape=layout.mesh_shape,
            alias_name=layout.alias_name,
            rank_list=layout.rank_list
        )

        # Apply the Replicated tensor map
        return output_layout(*out_tensor_map)
