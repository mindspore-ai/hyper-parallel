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
Distributed implementation for Cumsum operator.
"""

import copy
from typing import Tuple

from .parallel_ops import DistributedOp


def _normalize_cumsum_args(x, dim, dtype=None):
    return (x, dim), {'dtype': dtype}


class CumsumDistributedOp(DistributedOp):
    """Distributed implementation for torch.cumsum."""
    _MS_PRIMITIVE_OP_NAMES = frozenset({'CumsumExt'})

    def preprocess(self, args: tuple, kwargs: dict) -> tuple:
        """
        Preprocess arguments for Cumsum operator.

        Args:
            args (tuple): Input arguments, first element is the input tensor.
            kwargs (dict): Keyword arguments (dim, dtype).

        Returns:
            tuple: (local_args, local_kwargs, cache_values)
        """
        args, kwargs = _normalize_cumsum_args(*args, **kwargs)
        input_tensor = args[0]
        dim = args[1]
        dtype = kwargs['dtype']

        local_input = input_tensor.to_local()
        if self.op_name in self._MS_PRIMITIVE_OP_NAMES:
            local_args = (local_input, dim, dtype)
            local_kwargs = {}
        else:
            local_args = (local_input, dim)
            local_kwargs = {}
            if dtype is not None:
                local_kwargs['dtype'] = dtype

        cache_values = [input_tensor.layout, dim]
        return local_args, local_kwargs, cache_values

    def infer_layout(self, cache_values: list) -> Tuple[tuple, None]:
        """
        Infer output layout for Cumsum operator.

        Rules:
            1. Input must not have Partial status.
            2. dim must be an integer within the valid range [-ndim, ndim-1].
            3. The cumsum dimension must not be sharded, including StridedShard mappings.
            4. Output layout is identical to the input layout.

        Args:
            cache_values (list): [input_layout, dim]

        Returns:
            tuple: ((output_layout,), None)

        Raises:
            ValueError: If input has Partial status, dim is not an int, dim is out of range,
                or the cumsum dimension is sharded.
        """
        layout = cache_values[0]
        dim = cache_values[1]

        if not self._allow_partial_inputs:
            self._check_partial_inputs([layout])

        if not isinstance(dim, int):
            raise ValueError(
                f"For {self.op_name}, dimension should be int, but got {type(dim)}"
            )

        alias_map = layout.alias_tensor_map
        ndim = len(alias_map)
        if dim < -ndim or dim >= ndim:
            raise ValueError(
                f"For {self.op_name}, dimension out of range "
                f"(expected to be in range of [{-ndim}, {ndim - 1}], but got {dim})"
            )
        if dim < 0:
            dim += ndim

        mapping = alias_map[dim]
        if mapping != "None":
            raise ValueError(
                f"For {self.op_name}, cumsum along a sharded dimension "
                f"(dim {dim} mapped to {mapping}) is not supported."
            )

        return ((copy.deepcopy(layout),), None)
