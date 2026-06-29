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
Distributed implementation for Unbind operator.
"""

from typing import Tuple

from hyper_parallel.core.dtensor.layout import Layout
from .parallel_ops import DistributedOp


def _normalize_unbind_args(input_tensor, dim=0):
    return (input_tensor, dim), {}


class UnbindDistributedOp(DistributedOp):
    """Distributed implementation for Unbind operator."""

    def preprocess(self, args: tuple, kwargs: dict) -> tuple:
        """
        Preprocess arguments for Unbind operator.

        Args:
            args (tuple): Input arguments (input_tensor, dim).
            kwargs (dict): Keyword arguments (empty for this operator).

        Returns:
            tuple: (local_args, local_kwargs, cache_values)
        """
        args, kwargs = _normalize_unbind_args(*args, **kwargs)
        input_tensor, dim = args

        local_args = (input_tensor.to_local(), dim)
        local_kwargs = {}
        cache_values = [input_tensor.layout, tuple(input_tensor.shape), dim]
        return local_args, local_kwargs, cache_values

    # pylint: disable=W0237
    def infer_layout(self, cache_values: list) -> Tuple[tuple, None]:
        """
        Infer output layouts for Unbind operator.

        Rules:
            1. Input must not have Partial status.
            2. dim must be an integer within the valid range [-ndim, ndim-1].
            3. The dimension to unbind must not be sharded.
            4. Output layout removes the mapping for the unbound dimension;
               all output tensors share the same layout.

        Args:
            cache_values (list): [input_layout, input_shape, dim]

        Returns:
            tuple: ((output_layouts_tuple,), None)

        Raises:
            ValueError: If any rule above is violated.
        """
        layout = cache_values[0]
        shape = cache_values[1]
        dim = cache_values[2]

        if not self._allow_partial_inputs:
            self._check_partial_inputs([layout])

        alias_tensor_map = layout.alias_tensor_map
        ndim = len(shape)

        if not isinstance(dim, int):
            raise ValueError(
                f"For {self.op_name}, dimension should be int, but got {type(dim)}"
            )

        if dim < -ndim or dim >= ndim:
            raise ValueError(
                f"For {self.op_name}, dimension out of range "
                f"(expected to be in range of [{-ndim}, {ndim - 1}], but got {dim})"
            )

        if dim < 0:
            dim += ndim

        # Check if the dimension to unbind is sharded.
        # alias_tensor_map returns "None" for replicated dimensions.
        if alias_tensor_map[dim] != "None":
            raise ValueError(
                f"For {self.op_name}, the dimension {dim} is sharded "
                f"(mapped to {alias_tensor_map[dim]}). "
                f"Unbinding a sharded dimension is not supported. "
                f"Please redistribute the tensor to replicate this dimension first."
            )

        # Construct output layout: remove the mapping for the unbound dimension
        out_alias_map = alias_tensor_map[:dim] + alias_tensor_map[dim + 1:]

        base_layout = Layout(
            mesh_shape=layout.mesh_shape,
            alias_name=layout.alias_name,
            rank_list=layout.rank_list
        )
        out_layout = base_layout(*out_alias_map)

        num_outputs = shape[dim]
        return ((out_layout,) * num_outputs, None)
