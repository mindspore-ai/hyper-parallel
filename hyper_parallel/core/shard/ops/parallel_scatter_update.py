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
Distributed implementation for ScatterUpdate operator.
"""

from typing import Tuple

from hyper_parallel.core.dtensor.layout import Layout
from .parallel_ops import DistributedOp


def _normalize_scatter_update_args(input_tensor, indices, updates):
    return (input_tensor, indices, updates), {}


class ScatterUpdateDistributedOp(DistributedOp):
    """Distributed implementation for ScatterUpdate operator."""

    def __init__(self, op_name: str):
        super().__init__(op_name)
        self._allow_partial_inputs = True

    def preprocess(self, args: tuple, kwargs: dict) -> tuple:
        """
        Preprocess arguments for ScatterUpdate operator.

        Args:
            args (tuple): Input arguments containing input_tensor, indices, and updates.
            kwargs (dict): Keyword arguments (none expected).

        Returns:
            tuple: (local_args, local_kwargs, cache_values)
        """
        args, kwargs = _normalize_scatter_update_args(*args, **kwargs)
        x, indices, updates = args

        local_args = (
            x.to_local() if hasattr(x, '_layout') else x,
            indices.to_local() if hasattr(indices, '_layout') else indices,
            updates.to_local() if hasattr(updates, '_layout') else updates,
        )

        cache_values = [
            x.layout if hasattr(x, '_layout') else None,
            indices.layout if hasattr(indices, '_layout') else None,
            updates.layout if hasattr(updates, '_layout') else None,
        ]
        local_kwargs = kwargs
        return local_args, local_kwargs, cache_values

    def infer_layout(self, cache_values: list) -> Tuple[tuple, None]:
        """
        Infer output layout for ScatterUpdate operator.

        Rules:
            1. All three inputs must be DTensors (layout not None).
            2. Input first dimension (indexed/scatter dim) cannot be sharded.
            3. Indices must be all-Replicate on every dimension.
            4. updates[0:indices_ndim] prefix dimensions cannot be sharded.
            5. updates rank must equal indices_ndim + input_ndim - 1.
            6. updates[indices_ndim:] sharding must match input[1:] sharding.
            7. Output layout inherits input layout (including Partial status).

        Args:
            cache_values (list): [input_layout, indices_layout, updates_layout]

        Returns:
            tuple: ((output_layout,), None)

        Raises:
            ValueError: If any validation rule above is violated.
        """
        input_layout, indices_layout, updates_layout = cache_values

        if input_layout is None:
            raise ValueError(
                f"For {self.op_name}, input layout should not be None"
            )

        if indices_layout is None:
            raise ValueError(
                f"For {self.op_name}, indices must be a DTensor when input is a DTensor"
            )

        if updates_layout is None:
            raise ValueError(
                f"For {self.op_name}, updates must be a DTensor when input is a DTensor"
            )

        # Partial inputs are intentionally allowed. The output inherits Partial status
        # from the input layout (see lines below), making this a Partial-preserving op.
        if not self._allow_partial_inputs:
            self._check_partial_inputs([input_layout])

        self._validate_strategy(input_layout, indices_layout, updates_layout)

        output_layout = Layout(
            mesh_shape=input_layout.mesh_shape,
            alias_name=input_layout.alias_name,
            rank_list=input_layout.rank_list,
        )
        output_layout = output_layout(*input_layout.alias_tensor_map)

        for i, partial_op in enumerate(input_layout.partial):
            if partial_op is not None:
                dev_axis_name = input_layout.alias_name[i]
                output_layout.set_partial_by_dev_axis(dev_axis_name, partial_op)

        return ((output_layout,), None)

    def _validate_strategy(self, input_layout, indices_layout, updates_layout):
        """Validate sharding strategy for ScatterUpdate."""
        input_map = input_layout.alias_tensor_map
        indices_map = indices_layout.alias_tensor_map
        updates_map = updates_layout.alias_tensor_map

        if not input_map:
            raise ValueError(
                f"For {self.op_name}, input tensor map should not be empty"
            )

        if input_map[0] != "None":
            raise ValueError(
                f"For {self.op_name}, first dimension of input cannot be sharded, "
                f"but it is sharded on '{input_map[0]}'"
            )

        for i, axis in enumerate(indices_map):
            if axis != "None":
                raise ValueError(
                    f"For {self.op_name}, indices cannot be sharded, "
                    f"but dimension {i} is sharded on '{axis}'"
                )

        indices_ndim = len(indices_map)
        for i in range(indices_ndim):
            if i >= len(updates_map):
                raise ValueError(
                    f"For {self.op_name}, updates rank is smaller than indices rank"
                )
            if updates_map[i] != "None":
                raise ValueError(
                    f"For {self.op_name}, first {indices_ndim} dimensions of updates "
                    f"cannot be sharded, but dimension {i} is sharded on '{updates_map[i]}'"
                )

        expected_updates_ndim = indices_ndim + len(input_map) - 1
        if len(updates_map) != expected_updates_ndim:
            raise ValueError(
                f"For {self.op_name}, updates rank mismatch. "
                f"Expected {expected_updates_ndim}, got {len(updates_map)}"
            )

        for i in range(1, len(input_map)):
            updates_idx = indices_ndim + i - 1
            if input_map[i] != updates_map[updates_idx]:
                raise ValueError(
                    f"For {self.op_name}, updates sharding must match input[1:]. "
                    f"Mismatch at input dim {i}: '{input_map[i]}' != '{updates_map[updates_idx]}'"
                )
            