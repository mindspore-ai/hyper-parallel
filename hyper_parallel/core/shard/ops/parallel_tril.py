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
"""Distributed implementation for the Tril operator."""
import copy
from typing import Callable, Optional, Tuple

from .parallel_ops import DistributedOp


def _normalize_tril_args(input_tensor=None, diagonal=0, *, out=None):
    """Normalize Tril arguments while preserving PyTorch's keyword-only out.

    Args:
        input_tensor: Input tensor.
        diagonal: Diagonal offset. Defaults to 0.
        out: Optional PyTorch output tensor. Defaults to None.

    Returns:
        tuple: Canonical positional arguments and keyword arguments.
    """
    return (input_tensor, diagonal), {"out": out}


class TrilDistributedOp(DistributedOp):
    """Distributed implementation for lower-triangular masking.

    Batch dimensions can be freely sharded. Row and column sharding are
    supported for uniformly divisible continuous shards by adjusting the
    local diagonal with each shard's global row and column offsets.
    """

    def preprocess(self, args: tuple, kwargs: dict) -> tuple:
        """Extract the local tensor and build the layout cache.

        Args:
            args: Positional operator arguments.
            kwargs: Keyword operator arguments.

        Returns:
            tuple: ``(local_args, local_kwargs, cache_values)`` where
                cache_values is ``[input_layout, input_shape]``.

        Raises:
            ValueError: If a non-None ``out`` argument is provided.
        """
        normalized_kwargs = dict(kwargs)
        if "input" in normalized_kwargs:
            normalized_kwargs["input_tensor"] = normalized_kwargs.pop("input")
        normalized_args, normalized_kwargs = _normalize_tril_args(*args, **normalized_kwargs)
        input_tensor, diagonal = normalized_args
        out = normalized_kwargs["out"]

        if out is not None:
            raise ValueError(
                f"For {self.op_name}, out should be None, but got {type(out)}; "
                f"the out keyword is not supported."
            )
        local_args = (input_tensor.to_local(), diagonal)
        cache_values = [input_tensor.layout, tuple(input_tensor.shape)]
        return local_args, {}, cache_values

    @staticmethod
    def _validate_input_layouts(input_layout, input_shape: tuple, op_name: str) -> None:
        """Validate Tril input layout constraints.

        Rules:
            1. Input layout must be present.
            2. Input must have at least two dimensions.
            3. Shape rank must match layout rank.
            4. Row and column dimensions may each map to at most one mesh axis.
            5. Sharded row and column dimensions must be uniformly divisible.

        Args:
            input_layout: Input tensor Layout.
            input_shape: Input tensor global shape under the uniform-shard assumption.
            op_name: Operator name used in error messages.

        Raises:
            ValueError: If any layout constraint is violated.
        """
        if input_layout is None:
            raise ValueError(
                f"For {op_name}, input layout should not be None, but got None."
            )

        input_map = input_layout.alias_tensor_map
        ndim = len(input_map)
        if ndim < 2:
            raise ValueError(
                f"For {op_name}, input should have at least 2 dimensions, but got {ndim}."
            )
        if len(input_shape) != ndim:
            raise ValueError(
                f"For {op_name}, shape rank must match layout rank, "
                f"but got shape rank {len(input_shape)} and layout rank {ndim}."
            )

        row_map = input_map[-2]
        col_map = input_map[-1]
        if isinstance(row_map, (tuple, list)):
            raise ValueError(
                f"For {op_name}, row dimension should map to at most one mesh axis, "
                f"but got multiple mesh axes {row_map}."
            )
        if isinstance(col_map, (tuple, list)):
            raise ValueError(
                f"For {op_name}, column dimension should map to at most one mesh axis, "
                f"but got multiple mesh axes {col_map}."
            )

        mesh = input_layout.mesh
        if row_map != "None":
            row_size = mesh.get_device_num_along_axis(row_map)
            if input_shape[-2] % row_size != 0:
                raise ValueError(
                    f"For {op_name}, row dimension size {input_shape[-2]} should be divisible "
                    f"by the device count {row_size} along mesh axis {row_map}."
                )
        if col_map != "None":
            col_size = mesh.get_device_num_along_axis(col_map)
            if input_shape[-1] % col_size != 0:
                raise ValueError(
                    f"For {op_name}, column dimension size {input_shape[-1]} should be divisible "
                    f"by the device count {col_size} along mesh axis {col_map}."
                )

    def infer_layout(self, cache_values: list) -> Tuple[tuple, None]:
        """Infer the output layout for Tril.

        Rules:
            1. Input must not have Partial status.
            2. Input rank must be at least two and match the cached shape rank.
            3. Row and column dimensions may each map to at most one mesh axis.
            4. Sharded row and column dimensions must be uniformly divisible.
            5. Output layout is a deep copy of the input layout.
            6. Row and column offsets are applied by ``get_expand_impl``.

        Args:
            cache_values: ``[input_layout, input_shape]``.

        Returns:
            tuple: ``((output_layout,), None)``.

        Raises:
            ValueError: If the input layout violates any Tril constraint.
        """
        input_layout, input_shape = cache_values
        self._check_partial_inputs([input_layout])
        self._validate_input_layouts(input_layout, input_shape, self.op_name)
        return ((copy.deepcopy(input_layout),), None)

    def get_expand_impl(
            self,
            func: Optional[Callable],
            infer_result: tuple,
            cache_values: list,
    ) -> Optional[Callable]:
        """Return a local Tril callable with a rank-adjusted diagonal.

        Args:
            func: Underlying Tril callable.
            infer_result: Result returned by ``infer_layout``.
            cache_values: ``[input_layout, input_shape]``.

        Returns:
            callable | None: Adjustment closure for row or column sharding;
                otherwise None.
        """
        input_layout, input_shape = cache_values
        input_map = input_layout.alias_tensor_map
        row_axis = input_map[-2]
        col_axis = input_map[-1]

        if row_axis == "None" and col_axis == "None":
            return None

        mesh = input_layout.mesh
        row_offset = 0
        if row_axis != "None":
            row_size = mesh.get_device_num_along_axis(row_axis)
            row_offset = mesh.get_local_rank(row_axis) * (input_shape[-2] // row_size)

        col_offset = 0
        if col_axis != "None":
            col_size = mesh.get_device_num_along_axis(col_axis)
            col_offset = mesh.get_local_rank(col_axis) * (input_shape[-1] // col_size)

        diagonal_offset = row_offset - col_offset

        def _tril_expand_impl(input_tensor, diagonal=0):
            return func(input_tensor, diagonal + diagonal_offset)

        return _tril_expand_impl
