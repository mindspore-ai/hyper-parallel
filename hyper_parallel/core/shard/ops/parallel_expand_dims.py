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
Distributed implementation for ExpandDims operator.
"""
from typing import Tuple

from hyper_parallel.core.dtensor.layout import Layout
from .parallel_ops import DistributedOp


def _normalize_expand_dims_args(x, axis=None, dim=None):
    if axis is None:
        axis = dim
    return (x, axis), {}


class ExpandDimsDistributedOp(DistributedOp):
    """Distributed implementation for ExpandDims operator."""

    def preprocess(self, args: tuple, kwargs: dict) -> tuple:
        """
        Preprocess arguments for ExpandDims operator.

        Args:
            args (tuple): Input arguments, first element is the input tensor.
            kwargs (dict): Keyword arguments (axis or dim).

        Returns:
            tuple: (local_args, local_kwargs, cache_values)
        """
        args, _ = _normalize_expand_dims_args(*args, **kwargs)
        input_tensor, axis = args[0], args[1]
        local_args = (input_tensor.to_local(), axis)
        cache_values = [input_tensor.layout, axis]
        return local_args, {}, cache_values

    def infer_layout(self, cache_values: list) -> Tuple[tuple, None]:
        """
        Infer output layout for ExpandDims operator.

        Rules:
            1. Input must not have Partial status.
            2. axis must be an integer within the valid range [-(rank + 1), rank].
            3. The inserted dimension is replicated.
            4. Existing input dimension mappings are shifted and otherwise preserved.

        Args:
            cache_values (list): [input_layout, axis]

        Returns:
            tuple: ((output_layout,), None)

        Raises:
            ValueError: If input has Partial status, input layout is missing,
                axis is missing or invalid, or axis is out of range.
        """
        if not cache_values:
            raise ValueError(
                f"For {self.op_name}, cache_values should contain input layout, "
                f"but got empty cache_values."
            )

        x_layout = cache_values[0]
        if not self._allow_partial_inputs:
            self._check_partial_inputs([x_layout])

        if x_layout.mesh_shape is None:
            raise ValueError(
                f"For {self.op_name}, input layout mesh_shape should not be None, "
                f"but got None."
            )

        axis = cache_values[1] if len(cache_values) > 1 else None

        if axis is None:
            raise ValueError(f"For {self.op_name}, axis parameter is required.")
        if not isinstance(axis, int):
            raise ValueError(
                f"For {self.op_name}, axis should be int, but got {type(axis)}."
            )

        in_rank = len(x_layout.alias_tensor_map)
        original_axis = axis
        if axis < 0:
            axis = axis + in_rank + 1

        if axis < 0 or axis > in_rank:
            raise ValueError(
                f"For {self.op_name}, axis {original_axis} out of range for input rank {in_rank}. "
                f"Valid range is [{-in_rank - 1}, {in_rank}]."
            )

        x_map = list(x_layout.alias_tensor_map)
        x_map.insert(axis, "None")

        output_layout = Layout(
            mesh_shape=x_layout.mesh_shape,
            alias_name=x_layout.alias_name,
            rank_list=x_layout.rank_list
        )
        output_layout = output_layout(*x_map)

        if self._allow_partial_inputs:
            for i, partial_op in enumerate(x_layout.partial):
                if partial_op is not None:
                    dev_axis_name = x_layout.alias_name[i]
                    output_layout.set_partial_by_dev_axis(dev_axis_name, partial_op)

        return ((output_layout,), None)
