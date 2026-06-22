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
Element-wise distributed operator implementation.
"""

import copy
from typing import Tuple

from .parallel_ops import DistributedOp


def _unwrap_local_value(value):
    """Convert DTensor-like values to local tensors while preserving scalar slots."""
    return value.to_local() if hasattr(value, "_layout") else value


class TupleElementWiseDistributedOp(DistributedOp):
    """
    Distributed implementation for tuple element-wise operators.

    Inherits from DistributedOp and provides element-wise specific implementations.
    """

    def preprocess(self, args: tuple, kwargs: dict) -> tuple:
        """
        Preprocess arguments for tuple element-wise operators.

        Args:
            args (tuple): Positional arguments passed to the operator.
            kwargs (dict): Keyword arguments passed to the operator.

        Returns:
            tuple: (local_args, local_kwargs, cache_values)
        """
        expanded_args = []
        local_args = []
        for arg in args:
            if isinstance(arg, (tuple, list)):
                expanded_args.extend(arg)
                local_args.append(tuple(_unwrap_local_value(item) for item in arg))
            else:
                expanded_args.append(arg)
                local_args.append(_unwrap_local_value(arg))

        local_kwargs = {key: _unwrap_local_value(value) for key, value in kwargs.items()}
        cache_values = [getattr(arg, "layout", None) for arg in expanded_args]
        cache_values.extend(getattr(value, "layout", None) for value in kwargs.values())

        return tuple(local_args), local_kwargs, cache_values

    def infer_layout(self, cache_values: list) -> Tuple[tuple, None]:
        """
        Infer output layouts for element-wise operations.

        Rules:
            1. Inputs must not have Partial status.
            2. Tuple/list positional arguments are inferred from their expanded elements.
            3. Output layouts are identical to the expanded input layouts.

        Args:
            cache_values (list): Expanded input layouts, using None for non-DTensor slots.

        Returns:
            tuple: (output_layouts, None)

        Raises:
            ValueError: If input has Partial status.
        """
        if not cache_values:
            return None

        self._check_partial_inputs(cache_values)

        output_layouts = tuple(
            copy.deepcopy(layout) if layout is not None else None
            for layout in cache_values
        )
        return output_layouts, None
