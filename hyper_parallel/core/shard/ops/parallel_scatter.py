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
Distributed implementation for Scatter operator.
"""

from typing import Tuple

from .parallel_ops import DistributedOp


def _normalize_scatter_args(input_tensor, dim, index, src):
    return (input_tensor, dim, index, src), {}


class ScatterDistributedOp(DistributedOp):
    """Distributed implementation for torch.scatter."""

    def preprocess(self, args: tuple, kwargs: dict) -> tuple:
        """
        Preprocess arguments for Scatter operator.

        Args:
            args (tuple): Input arguments (input, dim, index, src).
            kwargs (dict): Keyword arguments (unused for scatter).

        Returns:
            tuple: (local_args, local_kwargs, cache_values) where local_args contains
                local tensors and cache_values contains layouts plus dim.
        """
        args, kwargs = _normalize_scatter_args(*args, **kwargs)
        input_tensor, dim, index_tensor, src = args

        input_local = input_tensor.to_local()
        index_local = index_tensor.to_local() if hasattr(index_tensor, '_layout') else index_tensor
        src_local = src.to_local() if hasattr(src, '_layout') else src
        local_args = (input_local, dim, index_local, src_local)
        local_kwargs = {}

        cache_values = [
            input_tensor.layout,
            dim,
            index_tensor.layout if hasattr(index_tensor, '_layout') else None,
            src.layout if hasattr(src, '_layout') else None,
        ]
        return local_args, local_kwargs, cache_values

    def infer_layout(self, cache_values: list) -> Tuple[tuple, None]:
        """
        Infer output layout for Scatter operator.

        Rules:
            1. Input must not have Partial status.
            2. Input must be a DTensor with a valid layout.
            3. dim must be an integer within the valid range [-ndim, ndim-1].
            4. The scatter dimension must be replicated (not sharded).
            5. Index layout must match input layout (if index is a DTensor).
            6. Src layout must match input layout (if src is a DTensor).
            7. Output layout is identical to input layout.

        Args:
            cache_values (list): [input_layout, dim, index_layout, src_layout]
                where index_layout and src_layout may be None.

        Returns:
            tuple: ((output_layout,), None)

        Raises:
            ValueError: If any rule above is violated.
        """
        if not self._allow_partial_inputs:
            self._check_partial_inputs([cache_values[0]])

        input_layout = cache_values[0]
        if input_layout is None:
            raise ValueError(
                f"For {self.op_name}, input should be a DTensor with a valid layout, "
                f"but got None."
            )

        dim = cache_values[1]
        if not isinstance(dim, int):
            raise ValueError(
                f"For {self.op_name}, dim should be an integer, "
                f"but got {type(dim)}."
            )

        alias_map = input_layout.alias_tensor_map
        ndim = len(alias_map)

        if dim < 0:
            dim += ndim

        if dim < 0 or dim >= ndim:
            raise ValueError(
                f"For {self.op_name}, dim should be in range [{-ndim}, {ndim - 1}], "
                f"but got {cache_values[1]}."
            )

        # Scatter dimension must be replicated
        dim_alias = alias_map[dim]
        if dim_alias != "None":
            raise ValueError(
                f"For {self.op_name}, scatter dim should be replicated, "
                f"but dim {cache_values[1]} is mapped to {dim_alias}."
            )

        # Index layout must match input layout
        index_layout = cache_values[2]
        if index_layout is not None:
            index_alias = index_layout.alias_tensor_map
            if index_alias != alias_map:
                raise ValueError(
                    f"For {self.op_name}, index layout should match input layout, "
                    f"but got {index_alias} vs {alias_map}."
                )

        # Src layout must match input layout
        src_layout = cache_values[3]
        if src_layout is not None:
            src_alias = src_layout.alias_tensor_map
            if src_alias != alias_map:
                raise ValueError(
                    f"For {self.op_name}, src layout should match input layout, "
                    f"but got {src_alias} vs {alias_map}."
                )

        return ((input_layout,), None)
    