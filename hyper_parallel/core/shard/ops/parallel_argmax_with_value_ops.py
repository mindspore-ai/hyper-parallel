# Copyright 2025-2026 Huawei Technologies Co., Ltd
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
Distributed implementation for ArgMaxWithValue operator.
"""

from typing import Tuple

from .parallel_ops import DistributedOp


def _normalize_argmax_with_value_args(x, axis, keep_dims=False):
    return (x, axis, keep_dims), {}


class ArgMaxWithValueDistributedOp(DistributedOp):
    """Distributed implementation for ArgMaxWithValue operator."""

    def preprocess(self, args: tuple, kwargs: dict) -> tuple:
        """
        Preprocess arguments for ArgMaxWithValue operator.

        Args:
            args (tuple): Input arguments, first element is the input tensor.
            kwargs (dict): Keyword arguments (axis, keep_dims).

        Returns:
            tuple: (local_args, local_kwargs, cache_values)
        """
        args, kwargs = _normalize_argmax_with_value_args(*args, **kwargs)
        input_tensor = args[0]
        axis = args[1]
        keep_dims = args[2]

        local_args = (input_tensor.to_local(), axis, keep_dims)
        local_kwargs = {}

        cache_values = [input_tensor.layout, axis, keep_dims]
        return local_args, local_kwargs, cache_values

    def infer_layout(self, cache_values: list) -> Tuple[tuple, None]:
        """
        Infer output layouts for ArgMaxWithValue operator.

        Rules:
            1. Input must not have Partial status.
            2. axis must be an integer within the valid range [-ndim, ndim-1].
            3. The axis dimension must not be sharded (including StridedShard multi-axis mappings).
            4. If keep_dims is False, the reduced dimension is removed from the output layout.
            5. Both output layouts (values, indices) are identical.

        Args:
            cache_values (list): [input_layout, axis, keep_dims]

        Returns:
            tuple: ((values_layout, indices_layout), None)

        Raises:
            ValueError: If any rule above is violated.
        """
        input_layout = cache_values[0]
        axis = cache_values[1]
        keep_dims = cache_values[2]

        # Check partial inputs
        if not self._allow_partial_inputs:
            self._check_partial_inputs([input_layout])

        if not isinstance(axis, int):
            raise ValueError(
                f"For {self.op_name}, axis should be int, but got {type(axis)}"
            )

        rank = len(input_layout.tensor_map)

        if axis < 0:
            axis += rank
        if axis < 0 or axis >= rank:
            raise ValueError(
                f"For {self.op_name}, axis out of range "
                f"(expected to be in range of [{-rank}, {rank - 1}], but got {cache_values[1]})"
            )

        # Check if the axis dimension is sharded.
        # Use alias_tensor_map to support StridedShard multi-axis mappings.
        alias_map = input_layout.alias_tensor_map
        mapping = alias_map[axis]
        if isinstance(mapping, tuple):
            is_sharded = any(m != "None" for m in mapping)
        else:
            is_sharded = mapping != "None"

        if is_sharded:
            raise ValueError(
                f"For {self.op_name}, cannot perform sharding on axis dim "
                f"(dim {axis} mapped to {mapping}). "
                f"Please redistribute the tensor to Replicate on this dimension."
            )

        # Build output tensor map
        if not keep_dims:
            tensor_map = alias_map[:axis] + alias_map[axis + 1:]
        else:
            tensor_map = alias_map[:axis] + ("None",) + alias_map[axis + 1:]

        output_layout = input_layout(*tensor_map)
        return ((output_layout, output_layout), None)
    