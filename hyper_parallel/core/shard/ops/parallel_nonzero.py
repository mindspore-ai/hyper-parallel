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
Distributed implementation for Nonzero operator.
"""

from typing import Tuple

from hyper_parallel.core.dtensor.layout import Layout
from .parallel_ops import DistributedOp


def _normalize_nonzero_args(x, as_tuple=False):
    return (x,), {'as_tuple': as_tuple}


class NonzeroDistributedOp(DistributedOp):
    """Distributed implementation for torch.nonzero."""

    def preprocess(self, args: tuple, kwargs: dict) -> tuple:
        """
        Preprocess arguments for Nonzero operator.

        Args:
            args (tuple): Input arguments, first element is the input tensor.
            kwargs (dict): Keyword arguments (as_tuple).

        Returns:
            tuple: (local_args, local_kwargs, cache_values)
        """
        args, kwargs = _normalize_nonzero_args(*args, **kwargs)
        input_tensor = args[0]
        as_tuple = kwargs['as_tuple']

        local_args = (input_tensor.to_local(),)
        local_kwargs = {'as_tuple': as_tuple}

        cache_values = [input_tensor.layout, as_tuple]
        return local_args, local_kwargs, cache_values

    def infer_layout(self, cache_values: list) -> Tuple[tuple, None]:
        """
        Infer output layouts for Nonzero operator.

        Rules:
            1. Input must not have Partial status.
            2. Input must be fully replicated (all dimensions mapped to "None").
               nonzero produces data-dependent output shapes, which would differ
               across sharded ranks.
            3. If as_tuple=True: returns a tuple of 1D replicated layouts, one per
               input dimension.
            4. If as_tuple=False: returns a single 2D replicated layout.

        Args:
            cache_values (list): [input_layout, as_tuple]

        Returns:
            tuple: ((output_layout(s),), None)

        Raises:
            ValueError: If input has Partial status or is sharded.
        """
        input_layout = cache_values[0]
        as_tuple = cache_values[1]

        if input_layout is None:
            raise ValueError(
                f"For {self.op_name}, input_layout should be a valid Layout, but got None."
            )

        # Rule 1: Input must not have Partial status
        if not self._allow_partial_inputs:
            self._check_partial_inputs([input_layout])

        if not isinstance(as_tuple, bool):
            raise ValueError(
                f"For {self.op_name}, as_tuple should be bool, but got {type(as_tuple)}."
            )

        alias_map = input_layout.alias_tensor_map
        input_ndim = len(alias_map)

        # Rule 2: Input must be fully replicated due to data-dependent dynamic shapes
        for dim, dim_sharding in enumerate(alias_map):
            if dim_sharding != "None":
                raise ValueError(
                    f"For {self.op_name}, input tensor should be fully replicated, "
                    f"but got dim {dim} mapped to {dim_sharding}. "
                    f"nonzero produces dynamic shapes that depend on data values, "
                    f"which causes shape mismatches across ranks if the tensor is sharded."
                )

        mesh_shape = input_layout.mesh_shape
        alias_name = input_layout.alias_name
        rank_list = input_layout.rank_list

        def _create_replicated_layout(ndim):
            """Helper to create a fully replicated layout for a given dimension."""
            layout = Layout(
                mesh_shape=mesh_shape,
                alias_name=alias_name,
                rank_list=rank_list
            )
            alias_map = tuple("None" for _ in range(ndim))
            return layout(*alias_map)

        # Rule 3 & 4: Construct the return layout based on as_tuple flag
        if as_tuple:
            out_layout = _create_replicated_layout(1)
            return (tuple(out_layout for _ in range(input_ndim)), None)

        return ((_create_replicated_layout(2),), None)
