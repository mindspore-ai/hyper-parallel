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
Distributed implementation for Squeeze operator.
"""
from copy import deepcopy
from typing import Tuple

from hyper_parallel.core.dtensor.layout import Layout
from .parallel_ops import DistributedOp


def _normalize_squeeze_args(x, axis=None):
    return (x, axis), {}


class SqueezeDistributedOp(DistributedOp):
    """Distributed implementation for Squeeze operator."""

    def preprocess(self, args: tuple, kwargs: dict) -> tuple:
        args, _ = _normalize_squeeze_args(*args, **kwargs)
        input_tensor, axis = args[0], args[1]

        if axis is None:
            local_args = (input_tensor.to_local(),)
        else:
            local_args = (input_tensor.to_local(), axis)

        input_shape = getattr(input_tensor, "shape", None)
        cache_values = [input_tensor.layout, axis, input_shape]
        return local_args, {}, cache_values

    def infer_layout(self, cache_values: list) -> Tuple[tuple, None]:
        """
        Infer output layout for Squeeze operator.

        Rules:
            1. Input must not have Partial status.
            2. input_shape rank must match input layout rank.
            3. If axis is None, only size-1 replicated dimensions are removed.
            4. If axis is specified, every axis must be in range, have size 1, and
               must not be sharded.
            5. Output layout removes the squeezed dimensions and preserves the
               remaining input dimension mappings.

        Args:
            cache_values (list): [input_layout, axis, input_shape]

        Returns:
            tuple: ((output_layout,), None)

        Raises:
            ValueError: If input has Partial status, input_shape is missing, axis
                is invalid, or a requested squeeze dimension is sharded.
        """
        if not cache_values:
            raise ValueError(
                f"For {self.op_name}, cache_values should contain input layout, "
                f"but got empty cache_values."
            )

        x_layout = cache_values[0]
        if not self._allow_partial_inputs:
            self._check_partial_inputs([x_layout])

        if x_layout.mesh_shape is None:
            raise ValueError(
                f"For {self.op_name}, input layout mesh_shape should not be None, "
                f"but got None."
            )

        axis = cache_values[1] if len(cache_values) > 1 else None
        input_shape = cache_values[2] if len(cache_values) > 2 else None
        if input_shape is None:
            raise ValueError(
                f"For {self.op_name}, input_shape should be provided in cache_values, "
                f"but got None."
            )
        if not isinstance(input_shape, (list, tuple)):
            raise ValueError(
                f"For {self.op_name}, input_shape should be list or tuple, "
                f"but got {type(input_shape)}."
            )

        output_layout = self._compute_squeeze_layout(x_layout, axis, input_shape)
        return ((output_layout,), None)

    def _compute_squeeze_layout(self, x_layout, axis, input_shape):
        """Compute the squeezed layout."""
        # Handle scalar case
        if not input_shape:
            return self._handle_scalar_case(x_layout, axis)

        # Validate input_shape matches layout rank
        self._validate_input_shape(x_layout, input_shape)

        # Find dimensions to squeeze
        dims_to_squeeze = self._get_dims_to_squeeze(x_layout, axis, input_shape)

        # Create output layout
        return self._create_output_layout(x_layout, dims_to_squeeze)

    def _handle_scalar_case(self, x_layout, axis):
        """Handle scalar input case."""
        if axis is not None and axis != [] and axis != ():
            raise ValueError(
                f"For {self.op_name}, axis should be None for scalar input, "
                f"but got {axis}."
            )

        return deepcopy(x_layout)

    def _validate_input_shape(self, x_layout, input_shape):
        """Validate that input shape matches layout rank."""
        x_map = list(x_layout.alias_tensor_map)
        in_rank = len(x_map)

        if len(input_shape) != in_rank:
            raise ValueError(
                f"For {self.op_name}, input shape rank should match layout rank, "
                f"but got {len(input_shape)} and {in_rank}."
            )

    def _get_dims_to_squeeze(self, x_layout, axis, input_shape):
        """Get list of dimensions to squeeze."""
        x_map = list(x_layout.alias_tensor_map)
        in_rank = len(x_map)

        if axis is None:
            return self._get_all_squeezable_dims(x_map, input_shape)
        return self._get_specified_dims_to_squeeze(x_map, axis, input_shape, in_rank)

    def _get_all_squeezable_dims(self, x_map, input_shape):
        """Get all squeezable dimensions when axis is None."""
        dims_to_squeeze = []
        for i, shape in enumerate(input_shape):
            if shape == 1 and x_map[i] == "None":
                dims_to_squeeze.append(i)
        return dims_to_squeeze

    def _get_specified_dims_to_squeeze(self, x_map, axis, input_shape, in_rank):
        """Get dimensions to squeeze when axis is specified."""
        # Convert axis to list if it's a single integer
        if isinstance(axis, int):
            axis = [axis]
        elif isinstance(axis, tuple):
            axis = list(axis)
        elif not isinstance(axis, list):
            raise ValueError(
                f"For {self.op_name}, axis should be int, list or tuple, "
                f"but got {type(axis)}."
            )

        if not all(isinstance(ax, int) for ax in axis):
            raise ValueError(
                f"For {self.op_name}, every axis value should be int, "
                f"but got {axis}."
            )

        # Convert negative indices to positive
        axis = [ax if ax >= 0 else ax + in_rank for ax in axis]

        # Validate axis range
        self._validate_axis_range(axis, in_rank)

        # Check all specified axes
        for ax in axis:
            self._validate_axis_for_squeeze(x_map, input_shape, ax)

        # Return sorted unique axes
        return sorted(set(axis))

    def _validate_axis_range(self, axis, in_rank):
        """Validate axis values are within range."""
        for ax in axis:
            if ax < 0 or ax >= in_rank:
                raise ValueError(
                    f"For {self.op_name}, axis should be in range [{-in_rank}, {in_rank-1}], "
                    f"but got {ax}."
                )

    def _validate_axis_for_squeeze(self, x_map, input_shape, ax):
        """Validate a specific axis can be squeezed."""
        # Check shape == 1
        if input_shape[ax] != 1:
            raise ValueError(
                f"For {self.op_name}, dimension should have size 1, "
                f"but got shape {input_shape[ax]} at dimension {ax}."
            )

        # Check mapping is "None" (not distributed)
        if x_map[ax] != "None":
            raise ValueError(
                f"For {self.op_name}, dimension should not be distributed, "
                f"but got dimension {ax} mapped to device axis {x_map[ax]}."
            )

    def _create_output_layout(self, x_layout, dims_to_squeeze):
        """Create output layout after squeezing dimensions."""
        if not dims_to_squeeze:
            return deepcopy(x_layout)

        # Get current alias tensor map
        x_map = list(x_layout.alias_tensor_map)

        # Sort in descending order for safe removal
        dims_to_squeeze = sorted(set(dims_to_squeeze), reverse=True)

        # Remove specified dimensions
        for dim in dims_to_squeeze:
            del x_map[dim]

        new_map = x_map

        # Create output layout with new mapping
        output_layout = Layout(
            mesh_shape=x_layout.mesh_shape,
            alias_name=x_layout.alias_name,
            rank_list=x_layout.rank_list
        )

        if new_map:
            output_layout = output_layout(*new_map)
        else:
            # For scalar result
            output_layout = output_layout()

        # Copy partial operations from input layout
        self._copy_partial_operations(x_layout, output_layout, new_map)

        return output_layout

    def _copy_partial_operations(self, x_layout, output_layout, new_map):
        """Copy partial operations from input to output layout."""
        for i, partial_op in enumerate(x_layout.partial):
            if partial_op is not None:
                dev_axis_name = x_layout.alias_name[i]
                # Check if this device axis is still used in the output
                if dev_axis_name in new_map:
                    output_layout.set_partial_by_dev_axis(dev_axis_name, partial_op)
