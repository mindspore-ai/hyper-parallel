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
Distributed implementation for TopK operator.
"""

import math
from .parallel_ops import DistributedOp


class SplitWithSizeDistributedOp(DistributedOp):
    """Distributed implementation for SplitWithSize operator."""

    def infer_layout(self, layouts, extra_args):
        """
        Infer output layouts for Split operator.

        Rules:
        1. Shared axis can not be split.

        Args:
            layouts (Layout): Layout of input tensor
            extra_args (list): split size or sections, axis, input shape

        Returns:
            tuple: Layouts for output tensors
        """

        input_layout = layouts[0]
        axis = extra_args[1]
        # Check shared axis can not be split.
        tensor_map = input_layout.tensor_map
        if tensor_map[axis] != -1:
            raise ValueError(f"Can not split tensor at sharded axis[{axis}], layout: {input_layout}")

        split_sections = extra_args[0]
        output_num = len(split_sections)
        output_layouts = (input_layout,) * output_num
        return output_layouts


class SplitWithSizeViewDistributedOp(DistributedOp):
    """Distributed implementation for SplitWithSizeView operator."""

    def infer_layout(self, layouts, extra_args):
        """
        Infer output layouts for SplitWithSizeView operator.

        Rules:
        1. Shared axis can not be split.

        Args:
            layouts (Layout): Layout of input tensor
            extra_args (list): split size or sections, axis, input shape

        Returns:
            tuple: Layouts for output tensors
        """

        input_layout = layouts[0]
        axis = extra_args[1]
        # Check shared axis can not be split.
        tensor_map = input_layout.tensor_map
        if tensor_map[axis] != -1:
            raise ValueError(f"Can not split tensor at sharded axis[{axis}], layout: {input_layout}")

        split_sections = extra_args[0]
        output_num = len(split_sections)
        output_layouts = (input_layout,) * output_num
        return output_layouts


class SplitDistributedOp(DistributedOp):
    """Distributed implementation for Split operator."""

    def infer_layout(self, layouts, extra_args):
        """
        Infer output layouts for Split operator.

        Rules:
        1. Shared axis can not be split.
        2. Default: dim = 0 if not specified.

        Args:
            layouts (Layout): Layout of input tensor
            extra_args (list): split size or sections, axis, input shape. Expected:
                extra_args[0]: split_size (required)
                extra_args[1]: axis (optional)
                extra_args[2][0]: input_shape

        Returns:
            tuple: Layouts for output tensors
        """

        if not layouts or layouts[0] is None:
            raise ValueError("split requires a valid input tensor layout.")
        input_layout = layouts[0]

        if len(extra_args) == 2:
            split_size = extra_args[0]
            axis = 0 # default
            input_shape = extra_args[1][0]
        elif len(extra_args) == 3:
            split_size = extra_args[0]
            axis = extra_args[1]
            input_shape = extra_args[2][0]
        else:
            raise ValueError("Split ops extra_args requires 'axis' and contains 'output_num' optionally.")

        tensor_map = input_layout.tensor_map
        input_dim = len(tensor_map)
        if axis < 0:
            axis = input_dim + axis
        if not 0 <= axis < input_dim:
            raise ValueError(f"Dimension out of range (expected [0, {input_dim}), got {axis}).")

        # Check shared axis can not be split.
        if tensor_map[axis] != -1:
            raise ValueError(f"Can not split tensor at sharded axis[{axis}], layout: {input_layout}")

        output_num = 1
        if isinstance(split_size, int):
            output_num = math.ceil(input_shape[axis] / split_size)
        elif isinstance(split_size, (list, tuple)):
            output_num = len(split_size)

        output_layouts = (input_layout,) * output_num
        return output_layouts


class SplitTensorDistributedOp(DistributedOp):
    """Distributed implementation for SplitTensor operator."""

    def infer_layout(self, layouts, extra_args):
        """
        Infer output layouts for Split operator.

        Rules:
        1. Shared axis can not be split.

        Args:
            layouts (Layout): Layout of input tensor
            extra_args (list): split size or sections, axis, input shape

        Returns:
            tuple: Layouts for output tensors
        """

        input_layout = layouts[0]
        axis = extra_args[1]
        # Check shared axis can not be split.
        tensor_map = input_layout.tensor_map
        if tensor_map[axis] != -1:
            raise ValueError(f"Can not split tensor at sharded axis[{axis}], layout: {input_layout}")

        split_size = extra_args[0]
        input_shape = extra_args[2][0]
        output_num = input_shape[axis] // split_size
        if input_shape[axis] % split_size != 0:
            output_num += 1

        output_layouts = (input_layout,) * output_num
        return output_layouts


class SplitTensorViewDistributedOp(DistributedOp):
    """Distributed implementation for SplitTensorView operator."""

    def infer_layout(self, layouts, extra_args):
        """
        Infer output layouts for SplitTensorView operator.

        Rules:
        1. Shared axis can not be split.

        Args:
            layouts (Layout): Layout of input tensor
            extra_args (list): split size or sections, axis, input shape

        Returns:
            tuple: Layouts for output tensors
        """

        input_layout = layouts[0]
        axis = extra_args[1]
        # Check shared axis can not be split.
        tensor_map = input_layout.tensor_map
        if tensor_map[axis] != -1:
            raise ValueError(f"Can not split tensor at sharded axis[{axis}], layout: {input_layout}")

        split_size = extra_args[0]
        input_shape = extra_args[2][0]
        output_num = input_shape[axis] // split_size
        if input_shape[axis] % split_size != 0:
            output_num += 1

        output_layouts = (input_layout,) * output_num
        return output_layouts
