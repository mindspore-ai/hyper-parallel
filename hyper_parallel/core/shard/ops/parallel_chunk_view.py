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

from .parallel_ops import DistributedOp


class ChunkViewDistributedOp(DistributedOp):
    """Distributed implementation for ChunkView operator."""

    @staticmethod
    def _parse_extra_args(extra_args):
        """Parse and validate extra_args, returning (chunks, dim, input_shape)."""
        if len(extra_args) < 1:
            raise ValueError("chunk_view requires 'chunks' in extra_args.")

        chunks = extra_args[0]
        dim = extra_args[1] if len(extra_args) > 2 else 0
        input_shapes = extra_args[-1] if len(extra_args) > 1 else None

        if not isinstance(chunks, int):
            raise TypeError(f"chunks must be an integer, got {type(chunks)}")
        if chunks < 1:
            raise ValueError(f"chunks must be greater than 0, got {chunks}")
        if not isinstance(dim, int):
            raise TypeError(f"dim must be an integer, got {type(dim)}")

        if input_shapes:
            input_shape = input_shapes[0] if isinstance(input_shapes[0], (list, tuple)) else input_shapes
        else:
            input_shape = None

        return chunks, dim, input_shape

    @staticmethod
    def _calculate_output_count(dim_size, chunks):
        """Calculate the number of output chunks based on dimension size."""
        if dim_size == 0:
            return chunks
        split_size = (dim_size + chunks - 1) // chunks
        output_num = max((dim_size + split_size - 1) // split_size, 1)
        return min(output_num, chunks)

    def infer_layout(self, layouts, extra_args=None):
        """
        Infer output layouts for ChunkView operator.

        Rules:
        1. Split dimension cannot be sharded.
        2. Default: dim = 0 if not specified.
        3. Output count may be less than chunks if dimension size < chunks.

        Args:
            layouts (Layout): Layout of input tensor
            extra_args (list): chunks, dim, input_shape. Expected:
                extra_args[0]: chunks (required) - number of chunks to split into
                extra_args[1]: dim (optional, default=0) - dimension along which to split
                extra_args[2][0]: input_shapes (optional) - shape of input tensor

        Returns:
            tuple: Layouts for output tensors
        """

        if not layouts or layouts[0] is None:
            raise ValueError("chunk_view requires a valid input tensor layout.")

        input_layout = layouts[0]
        chunks, dim, input_shape = self._parse_extra_args(extra_args)

        tensor_map = input_layout.tensor_map
        input_dim = len(tensor_map)

        if dim < 0:
            dim = input_dim + dim

        if not 0 <= dim < input_dim:
            raise ValueError(f"Dimension out of range (expected [0, {input_dim}), got {dim}).")

        if tensor_map[dim] != -1:
            raise ValueError(f"Cannot split tensor at sharded axis[{dim}], layout: {input_layout}")

        if input_shape is not None:
            output_num = self._calculate_output_count(input_shape[dim], chunks)
        else:
            output_num = chunks

        output_layouts = (input_layout,) * output_num
        return output_layouts
