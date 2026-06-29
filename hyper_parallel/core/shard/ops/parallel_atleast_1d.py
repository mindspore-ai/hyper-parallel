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
Distributed implementation for atleast_1d operator.
"""

from typing import Tuple

from hyper_parallel.core.dtensor.layout import Layout
from .parallel_ops import DistributedOp


def _normalize_atleast_1d_args(*tensors):
    return tensors, {}


class Atleast1DDistributedOp(DistributedOp):
    """Distributed implementation for torch.atleast_1d."""

    def preprocess(self, args: tuple, kwargs: dict) -> tuple:
        """
        Preprocess arguments for atleast_1d operator.

        torch.atleast_1d(*tensors) takes a variable number of tensor inputs.
        All arguments are positional tensors with no keyword-only parameters.

        Args:
            args (tuple): Positional arguments (tensors).
            kwargs (dict): Keyword arguments (none expected).

        Returns:
            tuple: (local_args, local_kwargs, cache_values)
        """
        args, kwargs = _normalize_atleast_1d_args(*args, **kwargs)
        tensors = args

        local_args = tuple(
            t.to_local() if hasattr(t, 'to_local') else t
            for t in tensors
        )
        local_kwargs = {}

        cache_values = [
            t.layout if hasattr(t, 'layout') else None
            for t in tensors
        ]

        return local_args, local_kwargs, cache_values

    def infer_layout(self, cache_values: list) -> Tuple[tuple, None]:
        """
        Infer output layouts for atleast_1d operator.

        Rules:
            1. Inputs must not have Partial status.
            2. For 0D → 1D: the newly created dimension is unsharded (-1).
            3. For 1D or higher: the layout is preserved unchanged.
            4. If a single tensor is provided, a single Layout is returned.
               If multiple tensors are provided, a tuple of Layouts is returned.

        Args:
            cache_values (list): List of Layout objects (one per input tensor).
                None entries represent non-DTensor inputs and produce None outputs.

        Returns:
            tuple: ((output_layout_or_tuple,), None)

        Raises:
            ValueError: If no inputs are provided or any input has Partial status.
        """
        layouts = cache_values

        if not layouts:
            raise ValueError(
                f"For {self.op_name}, at least one input tensor is required, "
                f"but got an empty input list."
            )

        # Check partial inputs (atleast_1d does not support partial)
        if not self._allow_partial_inputs:
            self._check_partial_inputs(layouts)

        output_layouts = []

        # Process each layout for the case of multiple input tensors
        for input_layout in layouts:
            if input_layout is None:
                output_layouts.append(None)
                continue

            in_tensor_map = input_layout.tensor_map
            input_ndim = len(in_tensor_map)

            # Build output tensor map
            if input_ndim == 0:
                # 0D -> 1D: the newly created dimension is unsharded
                output_map = (-1,)
            else:
                # 1D or higher: preserve original layout
                output_map = in_tensor_map

            # Construct output layout using the same mesh properties
            mesh_shape = input_layout.mesh_shape
            alias_name = input_layout.alias_name
            rank_list = input_layout.rank_list

            def idx_to_alias(idx, aliases):
                if idx == -1:
                    return "None"
                # Map index back to alias name string
                return aliases[len(aliases) - idx - 1]

            output_alias_map = tuple(idx_to_alias(idx, alias_name) for idx in output_map)

            out_layout = Layout(
                mesh_shape=mesh_shape,
                alias_name=alias_name,
                rank_list=rank_list
            )
            # Re-apply the alias mapping to generate the full internal layout
            out_layout = out_layout(*output_alias_map)

            output_layouts.append(out_layout)

        # If there's only one input, return a single Layout.
        # If there are multiple inputs, return a tuple of Layouts.
        if len(output_layouts) == 1:
            return ((output_layouts[0],), None)

        return (tuple(output_layouts), None)
