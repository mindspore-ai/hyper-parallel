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
SwiGLU distributed operator implementation.

SwiGLU splits the input along an axis into gate and up halves, applies silu to the
gate, and multiplies element-wise with up. The split is performed on the local shard.
When the split axis is sharded, the producer layout strategy is expected to make each
local shard contain paired gate/up halves.
"""

import copy
from typing import Tuple

from .parallel_ops import DistributedOp


def _normalize_swiglu_args(x, axis=-1, dim=None):
    """Normalize SwiGLU arguments across platform call sites.

    Args:
        x: Input tensor.
        axis: The split axis (default -1).
        dim: Alias for axis; when provided, overrides axis.

    Returns:
        tuple: ((x, axis), {})
    """
    if dim is not None:
        axis = dim
    return (x, axis), {}


class SwiGLUDistributedOp(DistributedOp):
    """
    Distributed implementation for the SwiGLU operator.

    Rules:
        1. Input must not have Partial status.
        2. axis must be an int within the valid range [-ndim, ndim-1].
        3. The split axis may be sharded. The split is performed on the local shard;
           when the split axis is sharded, the producer layout strategy is expected
           to make each local shard contain paired gate/up halves.
        4. The split axis must have an even global size so it can be halved.
        5. Output keeps the same sharding mapping as input.
    """

    def preprocess(self, args: tuple, kwargs: dict) -> tuple:
        """
        Preprocess arguments for SwiGLU.

        Args:
            args (tuple): Positional arguments (x,), with optional axis or dim in kwargs.
            kwargs (dict): Keyword arguments, optionally containing 'axis' or 'dim'.

        Returns:
            tuple: (local_args, local_kwargs, cache_values)
        """
        args, _ = _normalize_swiglu_args(*args, **kwargs)
        input_tensor, axis = args

        local_args = (input_tensor.to_local(), axis)
        local_kwargs = {}
        # Cache the global shape to validate the split axis is even in infer_layout.
        cache_values = [input_tensor.layout, axis, input_tensor.shape]
        return local_args, local_kwargs, cache_values

    def infer_layout(self, cache_values: list) -> Tuple[tuple, None]:  # pylint: disable=W0221
        """
        Infer output layout for SwiGLU.

        Args:
            cache_values (list): [input_layout, axis, input_shape]

        Returns:
            tuple: ((output_layout,), None)

        Raises:
            ValueError: If input has Partial status, axis is invalid, or the split
                axis length is not divisible by 2.
        """
        layout = cache_values[0]
        axis = cache_values[1]
        input_shape = cache_values[2]

        if not self._allow_partial_inputs:
            self._check_partial_inputs([layout])

        if not isinstance(axis, int):
            raise ValueError(
                f"For {self.op_name}, axis should be int, but got {type(axis)}"
            )

        # Use alias_tensor_map to support StridedShard multi-axis mappings.
        alias_map = layout.alias_tensor_map
        ndim = len(alias_map)

        if axis < 0:
            axis += ndim
        if axis < 0 or axis >= ndim:
            raise ValueError(
                f"For {self.op_name}, axis out of range "
                f"(expected to be in range of [{-ndim}, {ndim - 1}], but got {cache_values[1]})"
            )

        # The split axis may be sharded — the split is performed on the local
        # shard. When the split axis is sharded, the producer layout strategy is
        # expected to make each local shard contain paired gate/up halves.
        #
        # The split axis must have an even global length so it can be halved into
        # gate and up within each shard.
        axis_size = input_shape[axis]
        if axis_size % 2 != 0:
            raise ValueError(
                f"For {self.op_name}, the split axis (dim {axis}) must have an even size, "
                f"but got {axis_size}. SwiGLU splits the input along this axis "
                f"into two equal halves for gate and up."
            )

        return (copy.deepcopy(layout),), None
