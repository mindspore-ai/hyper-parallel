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
Distributed implementation for InplaceScatterValue operator.
"""

from copy import deepcopy
from typing import Tuple

from .parallel_ops import DistributedOp


def _normalize_inplace_scatter_value_args(input_tensor, dim, index, value):
    """Normalize InplaceScatterValue arguments to positional args + empty kwargs."""
    return (input_tensor, dim, index, value), {}


class InplaceScatterValueDistributedOp(DistributedOp):
    """Distributed implementation for InplaceScatterValue operator."""

    def preprocess(self, args: tuple, kwargs: dict) -> tuple:
        """
        Preprocess arguments for InplaceScatterValue operator.

        Args:
            args (tuple): Input arguments (input_tensor, dim, index, value).
            kwargs (dict): Keyword arguments (none expected).

        Returns:
            tuple: (local_args, local_kwargs, cache_values)
        """
        args, kwargs = _normalize_inplace_scatter_value_args(*args, **kwargs)
        input_tensor, dim, index, value = args

        local_args = (
            input_tensor.to_local() if hasattr(input_tensor, '_layout') else input_tensor,
            dim,
            index.to_local() if hasattr(index, '_layout') else index,
            value,
        )

        cache_values = [
            input_tensor.layout if hasattr(input_tensor, '_layout') else None,
            index.layout if hasattr(index, '_layout') else None,
            dim,
        ]
        local_kwargs = kwargs
        return local_args, local_kwargs, cache_values

    def infer_layout(self, cache_values: list) -> Tuple[tuple, None]:
        """
        Infer output layout for InplaceScatterValue operator.

        Rules:
            1. Input and index must be DTensors (layout not None).
            2. Input must not have Partial status.
            3. dim must be an integer and within bounds after normalization.
            4. Input and index must have the same number of dimensions.
            5. Input and index must use the same sharding on all non-dim axes.
            6. The target dim axis must be replicated (unsharded).
            7. Output layout is identical to input layout (inplace operation).

        Args:
            cache_values (list): [input_layout, index_layout, dim]

        Returns:
            tuple: ((output_layout,), None)

        Raises:
            ValueError: If any validation rule above is violated.
        """
        input_layout, index_layout, dim = cache_values

        if not self._allow_partial_inputs:
            self._check_partial_inputs([input_layout, index_layout])

        if input_layout is None or not hasattr(input_layout, "tensor_map"):
            raise ValueError(
                f"For {self.op_name}, input layout should not be None"
            )

        if index_layout is None or not hasattr(index_layout, "tensor_map"):
            raise ValueError(
                f"For {self.op_name}, index must be a DTensor when input is a DTensor"
            )

        input_map = input_layout.alias_tensor_map
        index_map = index_layout.alias_tensor_map

        if not isinstance(dim, int):
            raise ValueError(
                f"For {self.op_name}, dim should be an integer, but got {type(dim).__name__}"
            )

        ndim = len(input_map)
        original_dim = dim
        if dim < 0:
            dim += ndim
        if dim < 0 or dim >= ndim:
            raise ValueError(
                f"For {self.op_name}, dim {original_dim} is out of bounds for tensor with {ndim} dims"
            )

        if len(input_map) != len(index_map):
            raise ValueError(
                f"For {self.op_name}, input and index must have the same number of dimensions, "
                f"but got input rank={len(input_map)}, index rank={len(index_map)}"
            )

        for axis, (input_axis_map, index_axis_map) in enumerate(zip(input_map, index_map)):
            if axis == dim:
                continue
            if input_axis_map != index_axis_map:
                raise ValueError(
                    f"For {self.op_name}, input and index must use the same sharding "
                    f"on non-dim axes, "
                    f"but got mismatch at axis {axis}: "
                    f"input='{input_axis_map}', index='{index_axis_map}'"
                )

        if input_map[dim] != "None":
            raise ValueError(
                f"For {self.op_name}, scatter along sharded dimension {dim} "
                f"is not supported, "
                f"dim {dim} is sharded on '{input_map[dim]}'"
            )

        return ((deepcopy(input_layout),), None)
