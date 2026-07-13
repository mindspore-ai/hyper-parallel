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
Distributed implementation for TopK operator.
"""

from copy import deepcopy
from typing import Tuple

from .parallel_ops import DistributedOp


def _normalize_topk_args(input_tensor, k, dim=-1, largest=True, sorted_output=True):
    return (input_tensor, k, dim, largest, sorted_output), {}


class TopKDistributedOp(DistributedOp):
    """Distributed implementation for TopK operator."""

    def preprocess(self, args: tuple, kwargs: dict) -> tuple:
        """
        Preprocess arguments for TopK operator.

        Args:
            args (tuple): Input arguments, first element is the input tensor.
            kwargs (dict): Keyword arguments.

        Returns:
            tuple: (local_args, local_kwargs, cache_values)
        """
        args, kwargs = _normalize_topk_args(*args, **kwargs)
        input_tensor = args[0]
        k = args[1]
        dim = args[2]
        largest = args[3]
        sorted_flag = args[4]

        local_args = (input_tensor.to_local(), k, dim, largest, sorted_flag)
        local_kwargs = {}
        cache_values = [input_tensor.layout, dim]
        return local_args, local_kwargs, cache_values

    def infer_layout(self, cache_values: list) -> Tuple[tuple, None]:
        """
        Infer output layouts for TopK operator.

        TopK: values, indices = topk(input, k, dim)

        Rules:
            1. Input must not have Partial status.
            2. dim must be an integer within the valid range [-ndim, ndim-1].
            3. The topk dimension must not be sharded (including StridedShard multi-axis mappings).
            4. Both values and indices output layouts are identical to the input layout.

        Args:
            cache_values (list): [input_layout, dim] where dim is the topk dimension.

        Returns:
            tuple: ((values_layout, indices_layout), None)

        Raises:
            ValueError: If input has Partial status, dim is out of range, or the topk dimension
                is sharded.
        """
        layout = cache_values[0]
        dim = cache_values[1]

        if not self._allow_partial_inputs:
            self._check_partial_inputs([layout])

        if dim is None:
            dim = -1
        if not isinstance(dim, int):
            raise ValueError(
                f"For {self.op_name}, dimension should be int, but got {type(dim)}"
            )

        alias_map = layout.alias_tensor_map
        ndim = len(alias_map)

        original_dim = dim
        if dim < 0:
            dim += ndim
        if not 0 <= dim < ndim:
            raise ValueError(
                f"For {self.op_name}, dimension out of range "
                f"(expected to be in range of [{-ndim}, {ndim - 1}], but got {original_dim})"
            )

        # Check if the topk dimension is sharded.
        # In alias_tensor_map, "None" means Replicate (not sharded); any other value implies sharding.
        mapping = alias_map[dim]
        if isinstance(mapping, (list, tuple)):
            is_sharded = any(m != "None" for m in mapping)
        else:
            is_sharded = mapping != "None"

        if is_sharded:
            raise ValueError(
                f"For {self.op_name}, topk along a sharded dimension "
                f"(dim {dim} mapped to {mapping}) is not supported. "
                f"Please redistribute the tensor to Replicate on this dimension before topk."
            )

        return ((deepcopy(layout), deepcopy(layout)), None)
    