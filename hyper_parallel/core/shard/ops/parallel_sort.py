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

from typing import Tuple

from .parallel_ops import DistributedOp


def _normalize_sort_args(x, dim=-1, descending=False, stable=False):
    return (x,), {'dim': dim, 'descending': descending, 'stable': stable}


class SortDistributedOp(DistributedOp):
    """Distributed implementation for Sort operator."""
    _MS_PRIMITIVE_OP_NAMES = frozenset({'SortExt'})

    def preprocess(self, args: tuple, kwargs: dict) -> tuple:
        """
        Preprocess arguments for Sort operator.

        Args:
            args (tuple): Input arguments, first element is the input tensor.
            kwargs (dict): Keyword arguments (dim, descending, stable).

        Returns:
            tuple: (local_args, local_kwargs, cache_values)
        """
        args, kwargs = _normalize_sort_args(*args, **kwargs)
        input_tensor = args[0]
        dim = kwargs['dim']
        descending = kwargs['descending']
        stable = kwargs['stable']

        if self.op_name in self._MS_PRIMITIVE_OP_NAMES:
            local_args = (input_tensor.to_local(), dim, descending, stable)
            local_kwargs = {}
        else:
            local_args = (input_tensor.to_local(),)
            local_kwargs = {'dim': dim, 'descending': descending, 'stable': stable}

        cache_values = [input_tensor.layout, dim]
        return local_args, local_kwargs, cache_values

    def infer_layout(self, cache_values: list) -> Tuple[tuple, None]:
        """
        Infer output layouts for Sort operator.

        Rules:
            1. Input must not have Partial status.
            2. dim must be an integer within the valid range [-ndim, ndim-1].
            3. The sort dimension must not be sharded (including StridedShard multi-axis mappings).
            4. Output values and indices layouts are identical to the input layout.

        Args:
            cache_values (list): [input_layout, dim] where dim is the sort dimension.

        Returns:
            tuple: ((values_layout, indices_layout), None)

        Raises:
            ValueError: If input has Partial status, dim is out of range, or the sort dimension
                is sharded.
        """
        layout = cache_values[0]
        dim = cache_values[1]

        self._check_partial_inputs([layout])

        if not isinstance(dim, int):
            raise ValueError(
                f"For {self.op_name}, dimension should be int, but got {type(dim)}"
            )

        # Get tensor map to check sharding status
        in_tensor_map = layout.tensor_map
        ndim = len(in_tensor_map)

        if dim < -ndim or dim >= ndim:
            raise ValueError(
                f"For {self.op_name}, dimension out of range "
                f"(expected to be in range of [{-ndim}, {ndim - 1}], but got {dim})"
            )

        if dim < 0:
            dim += ndim

        # Check if the sorting dimension is sharded.
        # In tensor_map, -1 means Replicate (not sharded); any other value implies sharding.
        mapping = in_tensor_map[dim]
        if isinstance(mapping, (list, tuple)):
            is_sharded = any(m != -1 for m in mapping)
        else:
            is_sharded = mapping != -1

        if is_sharded:
            raise ValueError(
                f"For {self.op_name}, sorting along a sharded dimension "
                f"(dim {dim} mapped to {mapping}) is not supported. "
                f"Please redistribute the tensor to Replicate on this dimension before sorting."
            )

        return ((layout, layout), None)
