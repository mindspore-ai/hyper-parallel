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
Distributed implementation for Transpose operator.
"""

from typing import Tuple

from hyper_parallel.core.dtensor.layout import Layout
from .parallel_ops import DistributedOp


def _normalize_transpose_args(*args):
    """Normalize transpose arguments to consistent positional args.

    All transpose / permute / TransposeView / TransposeExtView interfaces pass parameters
    positionally, so args are returned as-is with empty kwargs.

    Args:
        *args: Positional arguments from the op call.

    Returns:
        tuple: (args, {}) — all parameters as positional args, kwargs empty.
    """
    return args, {}


class TransposeDistributedOp(DistributedOp):
    """Distributed implementation for Transpose operator."""

    def preprocess(self, args: tuple, kwargs: dict) -> tuple:
        """
        Preprocess arguments for Transpose operator.

        Args:
            args (tuple): Input arguments, first element is the input tensor.
            kwargs (dict): Keyword arguments (always empty for transpose ops).

        Returns:
            tuple: (local_args, local_kwargs, cache_values)
        """
        args, _ = _normalize_transpose_args(*args)
        input_tensor = args[0]

        if self.op_name in ("Transpose", "permute", "TransposeView"):
            axis = args[1]
            local_args = (input_tensor.to_local(), axis)
            local_kwargs = {}
            cache_values = [input_tensor.layout, axis]
        elif self.op_name in ("transpose", "TransposeExtView"):
            dim0, dim1 = args[1], args[2]
            local_args = (input_tensor.to_local(), dim0, dim1)
            local_kwargs = {}
            cache_values = [input_tensor.layout, dim0, dim1]
        else:
            raise ValueError(
                f"For TransposeDistributedOp, unsupported op_name: {self.op_name}. "
                f"Expected 'Transpose', 'transpose', 'permute', "
                f"'TransposeView', or 'TransposeExtView'."
            )

        return local_args, local_kwargs, cache_values

    def infer_layout(self, cache_values: list) -> Tuple[tuple, None]:
        """
        Infer output layout for Transpose operator.

        Based on the op_name initialized in the base class, this method switches behavior:
        1. op_name == 'Transpose', 'permute' or 'TransposeView': axis-based permutation.
           - cache_values: [input_layout, axis] where axis is a tuple of indices.
           - Rules: Output tensor_map is permuted according to axis.
        2. op_name == 'transpose' or 'TransposeExtView': dim-based swap.
           - cache_values: [input_layout, dim0, dim1] where dim0 and dim1 are integers.
           - Rules: Output tensor_map has the two dimensions swapped.

        Rules:
            1. Input must not have Partial status.
            2. For axis-based: axis must be a valid permutation of [0, ndim-1].
            3. For dim-based: dim0 and dim1 must be integers within [-ndim, ndim-1].
            4. Output layout inherits mesh info from input, with tensor_map permuted accordingly.

        Args:
            cache_values (list): [input_layout, ...] where the remaining elements depend on op_name.

        Returns:
            tuple: ((output_layout,), None)

        Raises:
            ValueError: If any rule above is violated.
        """
        layout = cache_values[0]
        if not self._allow_partial_inputs:
            self._check_partial_inputs([layout])

        in_tensor_map = layout.alias_tensor_map
        ndim = len(in_tensor_map)

        if self.op_name in ("Transpose", "permute", "TransposeView"):
            axis = cache_values[1]

            if not isinstance(axis, (list, tuple)):
                raise ValueError(
                    f"For {self.op_name}, axis should be a list or tuple, "
                    f"but got {type(axis)}."
                )

            if len(in_tensor_map) != len(axis):
                raise ValueError(
                    f"For {self.op_name}, input tensor shape and permutation "
                    f"must have the same size. "
                    f"Got {len(in_tensor_map)} and {len(axis)}."
                )

            # check if axis is a permutation
            seen = set()
            for v in axis:
                if not isinstance(v, int):
                    raise ValueError(
                        f"For {self.op_name}, axis elements must be integers, "
                        f"but got {type(v)}."
                    )
                if v < 0 or v >= ndim or v in seen:
                    raise ValueError(
                        f"For {self.op_name}, invalid permutation {axis} for rank {ndim}."
                    )
                seen.add(v)

            out_tensor_map = tuple(in_tensor_map[i] for i in axis)

        else:
            dim0, dim1 = cache_values[1], cache_values[2]

            if not isinstance(dim0, int) or not isinstance(dim1, int):
                raise ValueError(
                    f"For {self.op_name}, dimensions must be integers, "
                    f"but got {type(dim0)} and {type(dim1)}."
                )

            if dim0 < 0:
                dim0 += ndim
            if dim1 < 0:
                dim1 += ndim

            if not (0 <= dim0 < ndim and 0 <= dim1 < ndim):
                raise ValueError(
                    f"For {self.op_name}, transpose dimensions out of bounds: "
                    f"({dim0}, {dim1}) for rank {ndim}."
                )

            out_tensor_map_list = list(in_tensor_map)
            out_tensor_map_list[dim0], out_tensor_map_list[dim1] = (
                out_tensor_map_list[dim1], out_tensor_map_list[dim0]
            )
            out_tensor_map = tuple(out_tensor_map_list)

        output_layout = Layout(
            mesh_shape=layout.mesh_shape,
            alias_name=layout.alias_name,
            rank_list=layout.rank_list
        )

        return ((output_layout(*out_tensor_map),), None)
    