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
Distributed implementation for Unbind operator.
"""

from hyper_parallel.core.layout import Layout
from .parallel_ops import DistributedOp


class UnbindDistributedOp(DistributedOp):
    """Distributed implementation for Unbind operator."""

    def infer_layout(self, layouts, extra_args):
        """
        Infer output layouts for Unbind operator.

        Args:
            layouts (tuple): Layouts of input tensor.
            extra_args (list):
                - If configured with 'WithShape' suffix, the last element is input_shapes (list of tuples).
                - Preceding elements are scalar arguments (e.g., dim).

        Returns:
            tuple: A tuple of Layouts for the output tensors.
        """
        # Parse arguments provided by _with_layout_infer_with_shape
        input_shapes = extra_args[-1]
        args = extra_args[:-1]

        # Pytorch unbind(input, dim=0)
        dim = args[0] if args else 0

        layout = layouts[0]
        shape = input_shapes[0]
        tensor_map = layout.tensor_map
        alias_tensor_map = layout.alias_tensor_map
        ndim = len(shape)

        # Handle negative dimension
        if dim < 0:
            dim += ndim

        if not 0 <= dim < ndim:
            raise ValueError(f"Dimension out of range (expected to be in range of [0, {ndim-1}], but got {dim})")

        # Check if the dimension to unbind is sharded.
        # tensor_map values != -1 indicate the dimension is split across a mesh axis.
        # We cannot unbind a sharded dimension without explicit redistribution (Gather),
        # as it would result in different tensor lists on different ranks.
        if tensor_map[dim] != -1:
            raise ValueError(
                f"For 'unbind', the dimension {dim} is sharded (mapped to mesh axis {tensor_map[dim]}). "
                f"Unbinding a sharded dimension is not supported. "
                f"Please redistribute the tensor to replicate this dimension first."
            )

        # Construct output layout: remove the mapping for the unbound dimension
        # We use alias_tensor_map to utilize Layout's robust initialization
        out_alias_map = alias_tensor_map[:dim] + alias_tensor_map[dim+1:]

        # Create a base layout object to generate the new layout
        base_layout = Layout(
            mesh_shape=layout.mesh_shape,
            alias_name=layout.alias_name,
            rank_list=layout.rank_list
        )

        # Generate the specific output layout
        out_layout = base_layout(*out_alias_map)

        # Unbind returns a tuple of tensors, the number of which is the size of the unbound dimension
        num_outputs = shape[dim]

        return (out_layout,) * num_outputs
