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
Distributed implementation for Outer operator.
"""

from typing import Tuple

from hyper_parallel.core.dtensor.layout import Layout
from .parallel_ops import DistributedOp


def _normalize_outer_args(vec1, vec2):
    return (vec1, vec2), {}


def _get_alias_shard_set(dim_alias):
    if isinstance(dim_alias, str):
        return {dim_alias} if dim_alias != "None" else set()
    return set(dim_alias)


class OuterDistributedOp(DistributedOp):
    """Distributed implementation for torch.outer."""

    def preprocess(self, args: tuple, kwargs: dict) -> tuple:
        """
        Preprocess arguments for Outer operator.

        Args:
            args (tuple): Input arguments (input, vec2).
            kwargs (dict): Keyword arguments (unused for outer).

        Returns:
            tuple: (local_args, local_kwargs, cache_values) where local_args contains
                local tensors for input and vec2, and cache_values contains their layouts.
        """
        args, kwargs = _normalize_outer_args(*args, **kwargs)
        input_tensor, vec2_tensor = args[0], args[1]
        local_args = (input_tensor.to_local(), vec2_tensor.to_local())
        local_kwargs = {}
        cache_values = [input_tensor.layout, vec2_tensor.layout]
        return local_args, local_kwargs, cache_values

    def infer_layout(self, cache_values: list) -> Tuple[tuple, None]:
        """
        Infer output layout for Outer operator.

        PyTorch semantics:
          - Computes the outer product of two 1-D tensors.
          - If input is of size N and vec2 is of size M, the output is of size (N, M).
          - Input tensors must be 1-D.

        Rules:
            1. Inputs must not have Partial status.
            2. Exactly two input layouts are required, both must be non-None.
            3. Both inputs must be exactly 1-D.
            4. The two inputs cannot be sharded along the same device mesh dimension.
            5. Output dim 0 inherits the sharding of input; output dim 1 inherits the sharding of vec2.

        Args:
            cache_values (list): [input_layout, vec2_layout]

        Returns:
            tuple: ((output_layout,), None)

        Raises:
            ValueError: If any rule above is violated.
        """
        layout1, layout2 = cache_values[0], cache_values[1]

        if layout1 is None or layout2 is None:
            raise ValueError(
                f"For {self.op_name}, both inputs should be DTensors with valid layouts, "
                f"but got layout1={layout1}, layout2={layout2}."
            )

        if not self._allow_partial_inputs:
            self._check_partial_inputs([layout1, layout2])

        alias_map1 = layout1.alias_tensor_map
        alias_map2 = layout2.alias_tensor_map

        if len(alias_map1) != 1 or len(alias_map2) != 1:
            raise ValueError(
                f"For {self.op_name}, both inputs should be exactly 1-D tensors, "
                f"but got {len(alias_map1)}-D and {len(alias_map2)}-D."
            )

        dim0_alias = alias_map1[0]
        dim1_alias = alias_map2[0]

        set1 = _get_alias_shard_set(dim0_alias)
        set2 = _get_alias_shard_set(dim1_alias)

        if set1.intersection(set2):
            raise ValueError(
                f"For {self.op_name}, the two inputs should not be sharded on the "
                f"same device mesh dimension, "
                f"but got conflict on mesh dimension(s): {set1.intersection(set2)}."
            )

        output_alias_map = (dim0_alias, dim1_alias)

        output_layout = Layout(
            mesh_shape=layout1.mesh_shape,
            alias_name=layout1.alias_name,
            rank_list=layout1.rank_list,
        )
        output_layout = output_layout(*output_alias_map)

        return ((output_layout,), None)
    