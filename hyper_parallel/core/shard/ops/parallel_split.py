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
Distributed implementation for Split operator.
"""

import copy
import math
from typing import Tuple

from .parallel_ops import DistributedOp


def _normalize_split_with_size_args(x, split_sections, dim):
    return (x, split_sections, dim), {}


def _normalize_split_args(x, split_size_or_sections, dim=0):
    return (x, split_size_or_sections, dim), {}


def _normalize_split_tensor_args(x, split_size, dim):
    return (x, split_size, dim), {}


def _normalize_tensor_split_args(x, indices_or_sections, dim=0):
    return (x, indices_or_sections, dim), {}


class SplitWithSizeDistributedOp(DistributedOp):
    """Distributed implementation for SplitWithSize operator."""

    def preprocess(self, args: tuple, kwargs: dict) -> tuple:
        """
        Preprocess arguments for SplitWithSize operator.

        Args:
            args (tuple): Input arguments (input, split_sections, dim).
            kwargs (dict): Keyword arguments (empty for this operator).

        Returns:
            tuple: (local_args, local_kwargs, cache_values)
        """
        args, kwargs = _normalize_split_with_size_args(*args, **kwargs)
        input_tensor, split_sections, dim = args
        output_num = len(split_sections)
        local_args = (input_tensor.to_local(), split_sections, dim)
        local_kwargs = {}
        cache_values = [input_tensor.layout, dim, output_num]
        return local_args, local_kwargs, cache_values

    # pylint: disable=W0237
    def infer_layout(self, cache_values: list) -> Tuple[tuple, None]:
        """
        Infer output layouts for SplitWithSize operator.

        Rules:
            1. Input must not have Partial status.
            2. The split dimension must not be sharded.

        Args:
            cache_values (list): [input_layout, dim, output_num]

        Returns:
            tuple: ((output_layouts,), None)

        Raises:
            ValueError: If any rule above is violated.
        """
        layout = cache_values[0]
        dim = cache_values[1]
        output_num = cache_values[2]

        if not self._allow_partial_inputs:
            self._check_partial_inputs([layout])

        in_tensor_map = layout.alias_tensor_map
        ndim = len(in_tensor_map)

        if dim < 0:
            dim = ndim + dim
        if not 0 <= dim < ndim:
            raise ValueError(
                f"For {self.op_name}, dimension should be in range [0, {ndim}), "
                f"but got {dim}."
            )

        if in_tensor_map[dim] != "None":
            raise ValueError(
                f"For {self.op_name}, can not split tensor at sharded axis[{dim}], "
                f"but got layout: {layout}."
            )

        return (tuple(copy.deepcopy(layout) for _ in range(output_num)), None)


class SplitWithSizeViewDistributedOp(DistributedOp):
    """Distributed implementation for SplitWithSizeView operator."""

    def preprocess(self, args: tuple, kwargs: dict) -> tuple:
        """
        Preprocess arguments for SplitWithSizeView operator.

        Args:
            args (tuple): Input arguments (input, split_sections, dim).
            kwargs (dict): Keyword arguments (empty for this operator).

        Returns:
            tuple: (local_args, local_kwargs, cache_values)
        """
        args, kwargs = _normalize_split_with_size_args(*args, **kwargs)
        input_tensor, split_sections, dim = args
        output_num = len(split_sections)
        local_args = (input_tensor.to_local(), split_sections, dim)
        local_kwargs = {}
        cache_values = [input_tensor.layout, dim, output_num]
        return local_args, local_kwargs, cache_values

    # pylint: disable=W0237
    def infer_layout(self, cache_values: list) -> Tuple[tuple, None]:
        """
        Infer output layouts for SplitWithSizeView operator.

        Rules:
            1. Input must not have Partial status.
            2. The split dimension must not be sharded.

        Args:
            cache_values (list): [input_layout, dim, output_num]

        Returns:
            tuple: ((output_layouts,), None)

        Raises:
            ValueError: If any rule above is violated.
        """
        layout = cache_values[0]
        dim = cache_values[1]
        output_num = cache_values[2]

        if not self._allow_partial_inputs:
            self._check_partial_inputs([layout])

        in_tensor_map = layout.alias_tensor_map
        ndim = len(in_tensor_map)

        if dim < 0:
            dim = ndim + dim
        if not 0 <= dim < ndim:
            raise ValueError(
                f"For {self.op_name}, dimension should be in range [0, {ndim}), "
                f"but got {dim}."
            )

        if in_tensor_map[dim] != "None":
            raise ValueError(
                f"For {self.op_name}, can not split tensor at sharded axis[{dim}], "
                f"but got layout: {layout}."
            )

        return (tuple(copy.deepcopy(layout) for _ in range(output_num)), None)


class SplitDistributedOp(DistributedOp):
    """Distributed implementation for Split operator (MindSpore Split and torch.split)."""

    def preprocess(self, args: tuple, kwargs: dict) -> tuple:
        """
        Preprocess arguments for Split operator.

        Args:
            args (tuple): Input arguments (input, split_size_or_sections[, dim]).
            kwargs (dict): Keyword arguments (may contain dim).

        Returns:
            tuple: (local_args, local_kwargs, cache_values)
        """
        args, kwargs = _normalize_split_args(*args, **kwargs)
        input_tensor, split_size_or_sections, dim = args

        if isinstance(split_size_or_sections, int):
            output_num = math.ceil(input_tensor.shape[dim] / split_size_or_sections)
        else:
            output_num = len(split_size_or_sections)

        local_args = (input_tensor.to_local(), split_size_or_sections, dim)
        local_kwargs = {}
        cache_values = [input_tensor.layout, dim, output_num]
        return local_args, local_kwargs, cache_values

    # pylint: disable=W0237
    def infer_layout(self, cache_values: list) -> Tuple[tuple, None]:
        """
        Infer output layouts for Split operator.

        Rules:
            1. Input must not have Partial status.
            2. The split dimension must not be sharded.
            3. dim must be within the valid range of input dimensions.

        Args:
            cache_values (list): [input_layout, dim, output_num]

        Returns:
            tuple: ((output_layouts,), None)

        Raises:
            ValueError: If any rule above is violated.
        """
        layout = cache_values[0]
        dim = cache_values[1]
        output_num = cache_values[2]

        if not self._allow_partial_inputs:
            self._check_partial_inputs([layout])

        in_tensor_map = layout.alias_tensor_map
        ndim = len(in_tensor_map)

        if dim < 0:
            dim = ndim + dim
        if not 0 <= dim < ndim:
            raise ValueError(
                f"For {self.op_name}, dimension should be in range [0, {ndim}), "
                f"but got {dim}."
            )

        if in_tensor_map[dim] != "None":
            raise ValueError(
                f"For {self.op_name}, can not split tensor at sharded axis[{dim}], "
                f"but got layout: {layout}."
            )

        return (tuple(copy.deepcopy(layout) for _ in range(output_num)), None)


class SplitTensorDistributedOp(DistributedOp):
    """Distributed implementation for SplitTensor operator."""

    def preprocess(self, args: tuple, kwargs: dict) -> tuple:
        """
        Preprocess arguments for SplitTensor operator.

        Args:
            args (tuple): Input arguments (input, split_size, dim).
            kwargs (dict): Keyword arguments (empty for this operator).

        Returns:
            tuple: (local_args, local_kwargs, cache_values)
        """
        args, kwargs = _normalize_split_tensor_args(*args, **kwargs)
        input_tensor, split_size, dim = args
        output_num = math.ceil(input_tensor.shape[dim] / split_size)
        local_args = (input_tensor.to_local(), split_size, dim)
        local_kwargs = {}
        cache_values = [input_tensor.layout, dim, output_num]
        return local_args, local_kwargs, cache_values

    # pylint: disable=W0237
    def infer_layout(self, cache_values: list) -> Tuple[tuple, None]:
        """
        Infer output layouts for SplitTensor operator.

        Rules:
            1. Input must not have Partial status.
            2. The split dimension must not be sharded.

        Args:
            cache_values (list): [input_layout, dim, output_num]

        Returns:
            tuple: ((output_layouts,), None)

        Raises:
            ValueError: If any rule above is violated.
        """
        layout = cache_values[0]
        dim = cache_values[1]
        output_num = cache_values[2]

        if not self._allow_partial_inputs:
            self._check_partial_inputs([layout])

        in_tensor_map = layout.alias_tensor_map
        ndim = len(in_tensor_map)

        if dim < 0:
            dim = ndim + dim
        if not 0 <= dim < ndim:
            raise ValueError(
                f"For {self.op_name}, dimension should be in range [0, {ndim}), "
                f"but got {dim}."
            )

        if in_tensor_map[dim] != "None":
            raise ValueError(
                f"For {self.op_name}, can not split tensor at sharded axis[{dim}], "
                f"but got layout: {layout}."
            )

        return (tuple(copy.deepcopy(layout) for _ in range(output_num)), None)


class SplitTensorViewDistributedOp(DistributedOp):
    """Distributed implementation for SplitTensorView operator."""

    def preprocess(self, args: tuple, kwargs: dict) -> tuple:
        """
        Preprocess arguments for SplitTensorView operator.

        Args:
            args (tuple): Input arguments (input, split_size, dim).
            kwargs (dict): Keyword arguments (empty for this operator).

        Returns:
            tuple: (local_args, local_kwargs, cache_values)
        """
        args, kwargs = _normalize_split_tensor_args(*args, **kwargs)
        input_tensor, split_size, dim = args
        output_num = math.ceil(input_tensor.shape[dim] / split_size)
        local_args = (input_tensor.to_local(), split_size, dim)
        local_kwargs = {}
        cache_values = [input_tensor.layout, dim, output_num]
        return local_args, local_kwargs, cache_values

    # pylint: disable=W0237
    def infer_layout(self, cache_values: list) -> Tuple[tuple, None]:
        """
        Infer output layouts for SplitTensorView operator.

        Rules:
            1. Input must not have Partial status.
            2. The split dimension must not be sharded.

        Args:
            cache_values (list): [input_layout, dim, output_num]

        Returns:
            tuple: ((output_layouts,), None)

        Raises:
            ValueError: If any rule above is violated.
        """
        layout = cache_values[0]
        dim = cache_values[1]
        output_num = cache_values[2]

        if not self._allow_partial_inputs:
            self._check_partial_inputs([layout])

        in_tensor_map = layout.alias_tensor_map
        ndim = len(in_tensor_map)

        if dim < 0:
            dim = ndim + dim
        if not 0 <= dim < ndim:
            raise ValueError(
                f"For {self.op_name}, dimension should be in range [0, {ndim}), "
                f"but got {dim}."
            )

        if in_tensor_map[dim] != "None":
            raise ValueError(
                f"For {self.op_name}, can not split tensor at sharded axis[{dim}], "
                f"but got layout: {layout}."
            )

        return (tuple(copy.deepcopy(layout) for _ in range(output_num)), None)


class TensorSplitDistributedOp(DistributedOp):
    """Distributed implementation for tensor_split operator."""

    def preprocess(self, args: tuple, kwargs: dict) -> tuple:
        """
        Preprocess arguments for tensor_split operator.

        Args:
            args (tuple): Input arguments (input, indices_or_sections[, dim]).
            kwargs (dict): Keyword arguments (may contain dim).

        Returns:
            tuple: (local_args, local_kwargs, cache_values)
        """
        args, kwargs = _normalize_tensor_split_args(*args, **kwargs)
        input_tensor, indices_or_sections, dim = args

        if isinstance(indices_or_sections, int):
            output_num = indices_or_sections
        elif isinstance(indices_or_sections, (list, tuple)):
            output_num = len(indices_or_sections) + 1
        elif hasattr(indices_or_sections, "shape") and len(indices_or_sections.shape) == 1:
            output_num = indices_or_sections.shape[0] + 1
        else:
            raise TypeError(
                f"For {self.op_name}, indices_or_sections must be an integer, "
                f"list, tuple, or 1D tensor."
            )

        local_indices = indices_or_sections
        if hasattr(indices_or_sections, "_layout"):
            local_indices = indices_or_sections.to_local()

        local_args = (input_tensor.to_local(), local_indices, dim)
        local_kwargs = {}
        cache_values = [input_tensor.layout, dim, output_num]
        return local_args, local_kwargs, cache_values

    # pylint: disable=W0237
    def infer_layout(self, cache_values: list) -> Tuple[tuple, None]:
        """
        Infer output layouts for tensor_split operator.

        Rules:
            1. Input must not have Partial status.
            2. The split dimension must not be sharded.
            3. dim must be within the valid range of input dimensions.

        Args:
            cache_values (list): [input_layout, dim, output_num]

        Returns:
            tuple: ((output_layouts,), None)

        Raises:
            ValueError: If any rule above is violated.
        """
        layout = cache_values[0]
        dim = cache_values[1]
        output_num = cache_values[2]

        if not self._allow_partial_inputs:
            self._check_partial_inputs([layout])

        in_tensor_map = layout.alias_tensor_map
        ndim = len(in_tensor_map)

        if dim < 0:
            dim = ndim + dim
        if not 0 <= dim < ndim:
            raise ValueError(
                f"For {self.op_name}, dimension should be in range [0, {ndim}), "
                f"but got {dim}."
            )

        if in_tensor_map[dim] != "None":
            raise ValueError(
                f"For {self.op_name}, can not split tensor at sharded axis[{dim}], "
                f"but got layout: {layout}."
            )

        return (tuple(copy.deepcopy(layout) for _ in range(output_num)), None)
    