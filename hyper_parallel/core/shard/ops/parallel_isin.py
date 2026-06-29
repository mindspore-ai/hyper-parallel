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
Distributed implementation for Isin operator.
"""

import copy
from typing import Tuple

from .parallel_ops import DistributedOp


def _normalize_isin_args(elements, test_elements, assume_unique=False, invert=False):
    return (elements, test_elements), {'assume_unique': assume_unique, 'invert': invert}


class IsinDistributedOp(DistributedOp):
    """Distributed implementation for torch.isin."""

    def preprocess(self, args: tuple, kwargs: dict) -> tuple:
        """
        Preprocess arguments for Isin operator.

        Args:
            args (tuple): Input arguments (elements, test_elements).
            kwargs (dict): Keyword arguments (assume_unique, invert).

        Returns:
            tuple: (local_args, local_kwargs, cache_values)
        """
        args, kwargs = _normalize_isin_args(*args, **kwargs)
        elements, test_elements = args[0], args[1]
        assume_unique = kwargs['assume_unique']
        invert = kwargs['invert']

        local_args = (
            elements.to_local(),
            test_elements.to_local() if hasattr(test_elements, '_layout') else test_elements,
        )
        local_kwargs = {'assume_unique': assume_unique, 'invert': invert}

        cache_values = [
            elements.layout,
            test_elements.layout if hasattr(test_elements, '_layout') else None,
        ]
        return local_args, local_kwargs, cache_values

    def infer_layout(self, cache_values: list) -> Tuple[tuple, None]:
        """
        Infer output layout for torch.isin(elements, test_elements, ...)

        PyTorch semantics:
          - Returns boolean tensor with SAME SHAPE as `elements`
          - Each element is tested against ALL values in `test_elements`, so requires GLOBAL view of `test_elements`

        Rules:
            1. elements must have a valid DTensor layout.
            2. elements must not have Partial status.
            3. If test_elements is a DTensor, it must be fully unsharded (replicated across all dimensions)
               and must not have Partial status.
            4. If test_elements is a plain Tensor or scalar, no sharding validation is needed.
            5. Output layout is identical to elements layout.

        Args:
            cache_values (list): [elements_layout, test_elements_layout_or_None]

        Returns:
            tuple: ((output_layout,), None)

        Raises:
            ValueError: If any rule above is violated.
        """
        if not cache_values or cache_values[0] is None:
            raise ValueError(
                f"For {self.op_name}, 'elements' requires a valid tensor layout, "
                f"but got {cache_values[0] if cache_values else None}."
            )

        elements_layout = cache_values[0]
        test_elements_layout = cache_values[1] if len(cache_values) >= 2 else None

        if not self._allow_partial_inputs:
            check_layouts = [elements_layout]
            if test_elements_layout is not None:
                check_layouts.append(test_elements_layout)
            self._check_partial_inputs(check_layouts)

        # test_elements must be unsharded if it is a DTensor
        if test_elements_layout is not None:
            alias_map = test_elements_layout.alias_tensor_map
            if not all(entry == "None" for entry in alias_map):
                raise ValueError(
                    f"For {self.op_name}, 'test_elements' must be unsharded, "
                    f"but got alias_tensor_map: {alias_map}."
                )

        return ((copy.deepcopy(elements_layout),), None)
