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
Activation with axis distributed operator implementation.
"""

from typing import Tuple

from .parallel_ops import DistributedOp


def _normalize_activation_with_axis_args(x, axis=-1, dim=None):
    if dim is not None:
        axis = dim
    return (x, axis), {}


class ActivationWithAxisDistributedOp(DistributedOp):
    """
    Distributed implementation for activation-with-axis operators (e.g., softmax).

    Inherits from DistributedOp and provides activation-with-axis specific implementations.
    """

    def preprocess(self, args: tuple, kwargs: dict) -> tuple:
        """
        Preprocess arguments for activation-with-axis operators.

        Args:
            args (tuple): Input arguments, first element is the input tensor.
            kwargs (dict): Keyword arguments, optionally containing axis/dim.

        Returns:
            tuple: (local_args, local_kwargs, cache_values)
        """
        args, _ = _normalize_activation_with_axis_args(*args, **kwargs)
        input_tensor = args[0]
        axis = args[1]

        local_args = (input_tensor.to_local(), axis)
        local_kwargs = {}
        cache_values = [input_tensor.layout, axis]
        return local_args, local_kwargs, cache_values

    def infer_layout(self, cache_values: list) -> Tuple[tuple, None]:  # pylint: disable=W0221
        """
        Infer output layouts for activation-with-axis operations.

        Rules:
            1. Input must not have Partial status.
            2. axis must be an int or tuple.
            3. Activation axes must not be sharded.
            4. If multiple input layouts are provided, all tensor inputs must share the same layout.
            5. Output layout is identical to the input layout.

        Args:
            cache_values (list): [input_layout, axis], or [input_layouts..., axis]

        Returns:
            tuple: ((output_layout,), None)

        Raises:
            ValueError: If input has Partial status, axis is invalid, or an activation
                axis is sharded.
        """
        axis = cache_values[-1]
        layouts = cache_values[:-1]
        if not layouts:
            return None

        if not self._allow_partial_inputs:
            self._check_partial_inputs(layouts)

        self.check_layout(layouts, axis)

        first_layout = None
        for layout in layouts:
            if first_layout is None and layout is not None:
                first_layout = layout
            if layout is not None and first_layout is not None and layout != first_layout:
                raise ValueError(
                    f"For {self.op_name}, requires all tensor inputs to have the same layout. "
                    f"Input a: {first_layout}, Input b: {layout}"
                )

        return (first_layout,), None

    def check_layout(self, layouts, axis):
        """
        check_layout
        """
        min_slice_num = 1
        x_dict = layouts[0].to_dict()
        x_dev = x_dict["tensor_map"]

        if not isinstance(axis, (int, tuple)):
            raise ValueError(
                f"For {self.op_name}, axis should be int or tuple, but got {type(axis)}"
            )

        axes = (axis,) if isinstance(axis, int) else axis
        for axis_index in axes:
            tensor_map = x_dev[axis_index]
            if tensor_map == -1:
                continue
            axis_strategy = x_dict["mesh_shape"][len(x_dict["mesh_shape"]) - tensor_map - 1]
            if axis_strategy != min_slice_num:
                raise ValueError(
                    f"For {self.op_name}, the axis dimension (in dim {axis_index}) is sharded "
                    f"(strategy is {axis_strategy}). This operation requires the reduction axis to be un-sharded."
                )
