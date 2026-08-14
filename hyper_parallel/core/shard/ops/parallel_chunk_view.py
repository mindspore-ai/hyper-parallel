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
Distributed implementation for ChunkView operator.
"""

import copy
from typing import Tuple

from .parallel_ops import DistributedOp


def _normalize_chunk_view_args(input_tensor, chunks, dim=0):
    return (input_tensor, chunks, dim), {}


class ChunkViewDistributedOp(DistributedOp):
    """Distributed implementation for ChunkView operator."""

    @staticmethod
    def _calculate_output_count(dim_size, chunks):
        """Calculate the number of output chunks based on dimension size."""
        if dim_size == 0:
            return chunks
        split_size = (dim_size + chunks - 1) // chunks
        output_num = max((dim_size + split_size - 1) // split_size, 1)
        return min(output_num, chunks)

    def preprocess(self, args: tuple, kwargs: dict) -> tuple:
        """
        Preprocess arguments for ChunkView operator.

        Args:
            args (tuple): Input arguments containing the input tensor, chunks, and dim.
            kwargs (dict): Keyword arguments (none expected).

        Returns:
            tuple: (local_args, local_kwargs, cache_values)
        """
        args, kwargs = _normalize_chunk_view_args(*args, **kwargs)
        input_tensor, chunks, dim = args
        input_shape = input_tensor.shape

        local_args = (input_tensor.to_local(), chunks, dim)
        local_kwargs = {}

        cache_values = [input_tensor.layout, chunks, dim, input_shape]
        return local_args, local_kwargs, cache_values

    def infer_layout(self, cache_values: list) -> Tuple[tuple, None]:
        """
        Infer output layouts for ChunkView operator.

        Rules:
            1. Input must not have Partial status.
            2. Split dimension cannot be sharded (including StridedShard multi-axis mappings).
            3. dim must be an integer within the valid range [-ndim, ndim-1].
            4. Default: dim = 0 if not specified.
            5. Output count may be less than chunks if dimension size < chunks.
            6. All output layouts are identical to the input layout.

        Args:
            cache_values (list): [input_layout, chunks, dim, input_shape]

        Returns:
            tuple: ((output_layout_1, output_layout_2, ...), None)

        Raises:
            ValueError: If any rule above is violated.
            TypeError: If chunks or dim is not an integer.
        """
        input_layout = cache_values[0]
        chunks = cache_values[1]
        dim = cache_values[2]
        input_shape = cache_values[3]

        if input_layout is None:
            raise ValueError(
                f"For {self.op_name}, input layout should not be None"
            )

        if not self._allow_partial_inputs:
            self._check_partial_inputs([input_layout])

        if not isinstance(chunks, int):
            raise TypeError(
                f"For {self.op_name}, chunks must be an integer, but got {type(chunks)}"
            )
        if chunks < 1:
            raise ValueError(
                f"For {self.op_name}, chunks must be greater than 0, but got {chunks}"
            )
        if not isinstance(dim, int):
            raise TypeError(
                f"For {self.op_name}, dim must be an integer, but got {type(dim)}"
            )

        alias_map = input_layout.alias_tensor_map
        ndim = len(alias_map)

        original_dim = dim
        if dim < 0:
            dim = ndim + dim

        if not 0 <= dim < ndim:
            raise ValueError(
                f"For {self.op_name}, dimension out of range "
                f"(expected to be in range of [{-ndim}, {ndim - 1}], but got {original_dim})"
            )

        mapping = alias_map[dim]
        if isinstance(mapping, (list, tuple)):
            is_sharded = any(m != "None" for m in mapping)
        else:
            is_sharded = mapping != "None"

        if is_sharded:
            raise ValueError(
                f"For {self.op_name}, cannot split tensor at sharded axis[{dim}], "
                f"layout: {input_layout}"
            )

        output_num = self._calculate_output_count(input_shape[dim], chunks)

        output_layouts = tuple(copy.deepcopy(input_layout) for _ in range(output_num))
        return (output_layouts, None)
