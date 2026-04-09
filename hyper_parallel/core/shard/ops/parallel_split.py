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
Distributed implementation for Split operator.
"""

import math
from .parallel_ops import DistributedOp


class SplitWithSizeDistributedOp(DistributedOp):
    """Distributed implementation for SplitWithSize operator."""

    def infer_layout(self, layouts, extra_args=None):
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

    def infer_layout(self, layouts, extra_args=None):
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

    def infer_layout(self, layouts, extra_args=None):
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

    def infer_layout(self, layouts, extra_args=None):
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

    def infer_layout(self, layouts, extra_args=None):
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


class TensorSplitDistributedOp(DistributedOp):
    """Distributed implementation for tensor_split operator."""

    def infer_layout(self, layouts, extra_args=None):
        """
        Infer output layouts for tensor_split operator.

        Rules:
        1. Shared (sharded) axis cannot be split.
        2. Default: dim = 0 if not specified.

        Args:
            layouts (list): Layout of the input tensor.
            extra_args (list): Extracted non-tensor arguments and input shapes. Expected:
                extra_args[0]: indices_or_sections (int, tuple, list, or 1D tensor)
                extra_args[1]: dim (optional, default is 0)
                extra_args[-1]: input_shapes (list containing the shape of the input tensor)

        Returns:
            tuple: Layouts for the output tensors
        """
        if not layouts or layouts[0] is None:
            raise ValueError("tensor_split requires a valid input tensor layout.")

        input_layout = layouts[0]

        # Parse extra_args based on the dispatcher's WithShape suffix rules
        if len(extra_args) == 1:
            indices_or_sections = extra_args[0]
            dim = 0  # default
        elif len(extra_args) == 2:
            indices_or_sections = extra_args[0]
            dim = extra_args[1]
        else:
            raise ValueError("tensor_split ops extra_args requires 'indices_or_sections' and optionally 'dim'.")

        tensor_map = input_layout.tensor_map
        input_dim = len(tensor_map)

        # Handle negative dimensions
        if dim < 0:
            dim = input_dim + dim

        if not 0 <= dim < input_dim:
            raise ValueError(f"Dimension out of range (expected [0, {input_dim}), got {dim}).")

        # Check: shared (sharded) axis cannot be split
        if tensor_map[dim] != -1:
            raise ValueError(f"Cannot perform tensor_split on sharded axis[{dim}], layout: {input_layout}")

        # Calculate the number of output tensors based on PyTorch's tensor_split rules
        if isinstance(indices_or_sections, int):
            output_num = indices_or_sections
        elif isinstance(indices_or_sections, (list, tuple)):
            output_num = len(indices_or_sections) + 1
        elif hasattr(indices_or_sections, "shape") and len(indices_or_sections.shape) == 1:
            # Handle 1D Tensor case
            output_num = indices_or_sections.shape[0] + 1
        else:
            raise TypeError("tensor_split: indices_or_sections must be an integer, list, tuple, or 1D tensor.")

        output_layouts = (input_layout,) * output_num
        return output_layouts
