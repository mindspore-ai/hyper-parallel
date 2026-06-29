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
Distributed implementation for SliceExt operator.
"""

import copy
from typing import Tuple

from .parallel_ops import DistributedOp


def _normalize_slice_ext_args(x, axis, begin, end, step):
    return (x, axis, begin, end, step), {}


class SliceExtDistributedOp(DistributedOp):
    """Distributed implementation for SliceExt operator."""

    def preprocess(self, args: tuple, kwargs: dict) -> tuple:
        """
        Preprocess arguments for SliceExt operator.

        Args:
            args (tuple): Input arguments (input, axis, begin, end, step).
            kwargs (dict): Keyword arguments (empty for this operator).

        Returns:
            tuple: (local_args, local_kwargs, cache_values)
        """
        args, kwargs = _normalize_slice_ext_args(*args, **kwargs)
        input_tensor, axis, begin, end, step = args
        local_args = (input_tensor.to_local(), axis, begin, end, step)
        local_kwargs = {}
        cache_values = [input_tensor.layout, axis]
        return local_args, local_kwargs, cache_values

    # pylint: disable=W0237
    def infer_layout(self, cache_values: list) -> Tuple[tuple, None]:
        """
        Infer output layouts for SliceExt operator.

        Rules:
            1. Input must not have Partial status.
            2. The sliced axis must not be sharded.
            3. Output layout is identical to the input layout.

        Args:
            cache_values (list): [input_layout, axis]

        Returns:
            tuple: ((output_layout,), None)

        Raises:
            ValueError: If input has Partial status or the sliced axis is sharded.
        """
        input_layout = cache_values[0]
        axis = cache_values[1]

        if not self._allow_partial_inputs:
            self._check_partial_inputs([input_layout])

        alias_map = input_layout.alias_tensor_map
        ndim = len(alias_map)

        if not isinstance(axis, int):
            raise ValueError(
                f"For {self.op_name}, axis should be int, but got {type(axis)}."
            )

        if axis < -ndim or axis >= ndim:
            raise ValueError(
                f"For {self.op_name}, axis out of range "
                f"(expected to be in range of [{-ndim}, {ndim - 1}], but got {axis})."
            )

        if axis < 0:
            axis += ndim

        if alias_map[axis] != "None":
            raise ValueError(
                f"For {self.op_name}, can not slice tensor at sharded axis[{axis}], "
                f"layout: {input_layout}."
            )

        return ((copy.deepcopy(input_layout),), None)
