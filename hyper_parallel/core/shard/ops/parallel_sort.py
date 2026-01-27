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
Distributed implementation for Sort operator.
"""

from .parallel_ops import DistributedOp


class SortDistributedOp(DistributedOp):
    """Distributed implementation for Sort operator."""

    def infer_layout(self, layouts, extra_args):
        """
        Infer output layout for Sort operator.

        The sort operator expects the sorting dimension to be fully available on each device
        (i.e., not sharded). If the dimension is sharded, a global sort cannot be performed
        locally without redistribution.

        Args:
            layouts (tuple): Layouts of input tensor.
            extra_args (tuple): Arguments for the operator. Expected: (dim, descending, stable).
                                If empty, dim defaults to -1.

        Returns:
            tuple: (Layout, Layout) representing the layouts for (values, indices).
        """
        layout = layouts[0]

        # Parse dim from extra_args if available, otherwise default to -1
        dim = -1
        if extra_args:
            # extra_args[0] corresponds to 'dim' in torch.sort(input, dim, ...)
            dim = extra_args[0]

        if not isinstance(dim, int):
            raise TypeError(f"For 'sort', dimension must be int, but got {type(dim)}")

        # Get tensor map to check sharding status
        in_tensor_map = layout.tensor_map
        ndim = len(in_tensor_map)

        # Handle negative dimension index
        if dim < -ndim or dim >= ndim:
            raise ValueError(f"Dimension out of range (expected to be in range of [{-ndim}, {ndim-1}], but got {dim})")

        if dim < 0:
            dim += ndim

        # Check if the sorting dimension is sharded
        # In tensor_map, -1 means Replicate (not sharded). Any other value implies sharding.
        mapping = in_tensor_map[dim]
        is_sharded = False

        if isinstance(mapping, (list, tuple)):
            # If mapped to multiple mesh axes, check if any is not -1
            if any(m != -1 for m in mapping):
                is_sharded = True
        elif mapping != -1:
            is_sharded = True

        if is_sharded:
            raise ValueError(
                f"For 'sort', sorting along a sharded dimension (dim {dim} mapped to {mapping}) is not supported. "
                f"Please redistribute the tensor to Replicate status on this dimension before sorting."
            )

        # The output layouts for 'values' and 'indices' are the same as the input layout
        return (layout, layout)
