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
Distributed implementation for Conv3d operator.
"""

from typing import Callable, Optional, Tuple

from hyper_parallel.core.dtensor.layout import Layout
from .parallel_ops import DistributedOp


def _normalize_conv3d_args(input_tensor, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    return (input_tensor, weight, bias, stride, padding, dilation, groups), {}


class Conv3dDistributedOp(DistributedOp):
    """
    Distributed implementation for torch.nn.functional.conv3d.
    Supports Data Parallel, Tensor Parallel (Column/Row), and Spatial Parallel.
    """

    def __init__(self, op_name):
        super().__init__(op_name)
        self._allow_partial_inputs = False

    def _validate_row_parallelism(self, in_map, w_map, groups):
        """
        Validate constraints for Row Parallelism.
        """
        # 1. Handle Groups Constraint for Row Parallelism
        if groups > 1:
            if in_map[1] != "None" or w_map[1] != "None":
                # Row Parallelism with groups > 1 requires advanced group-wise communication
                raise ValueError(f"For {self.op_name}, Sharding on C_in with groups > 1 is not supported.")

        # 2. Check Row Parallelism (Sharding on Channel In)
        # Input: (N, C_in, D, H, W), Weight: (C_out, C_in/groups, kD, kH, kW)
        if in_map[1] != "None":
            if in_map[1] != w_map[1]:
                raise ValueError(f"For {self.op_name}, Input C_in and Weight C_in must be sharded on the same axis.")

    def _validate_column_parallelism(self, w_layout, b_layout, groups):
        """
        Validate constraints for Column Parallelism.
        """
        w_map = w_layout.alias_tensor_map
        w_map_0 = w_map[0]

        if w_map_0 != "None":
            # Check bias alignment
            if b_layout is not None:
                b_map = b_layout.alias_tensor_map
                b_map_0 = b_map[0]
                if w_map_0 != b_map_0:
                    raise ValueError(
                        f"For {self.op_name}, Weight C_out and Bias C_out must be sharded on the same axis."
                    )

            # Check groups divisibility for Column Parallelism
            if groups > 1:
                dev_num = 1
                axes = w_map_0 if isinstance(w_map_0, tuple) else (w_map_0,)
                for axis_name in axes:
                    dev_num *= w_layout.mesh.get_device_num_along_axis(axis_name)

                if groups % dev_num != 0:
                    raise ValueError(
                        f"For {self.op_name}, groups ({groups}) "
                        f"must be divisible by tp_size ({dev_num})."
                    )

    def preprocess(self, args: tuple, kwargs: dict) -> tuple:
        """
        Preprocess arguments for Conv3d operator.

        Args:
            args (tuple): Conv3d positional arguments.
            kwargs (dict): Conv3d keyword arguments.

        Returns:
            tuple: (local_args, local_kwargs, cache_values)
        """
        args, _ = _normalize_conv3d_args(*args, **kwargs)
        input_tensor, weight, bias, stride, padding, dilation, groups = args
        local_args = (
            input_tensor.to_local(),
            weight.to_local(),
            bias.to_local() if hasattr(bias, '_layout') else bias,
            stride,
            padding,
            dilation,
            groups,
        )
        local_kwargs = {}
        cache_values = [
            input_tensor.layout,
            weight.layout,
            bias.layout if hasattr(bias, '_layout') else None,
            stride,
            padding,
            dilation,
            groups,
        ]
        return local_args, local_kwargs, cache_values

    def infer_layout(self, cache_values: list) -> Tuple[tuple, None]:
        """
        Infer output layout for Conv3d operator.

        Rules:
            1. Input and weight must not have Partial status.
            2. Input and weight must both be 5D.
            3. Input C_in and weight C_in sharding must match for row parallelism.
            4. Sharding C_in with groups > 1 is not supported.
            5. Bias C_out sharding must match weight C_out sharding.
            6. Output layout inherits N/D/H/W sharding from input and C_out sharding from weight.
            7. Row parallelism marks output as Partial('sum') on the C_in mesh axis.

        Args:
            cache_values (list): [input_layout, weight_layout, bias_layout_or_None,
                stride, padding, dilation, groups]

        Returns:
            tuple: ((output_layout,), None)

        Raises:
            ValueError: If layouts are missing, partial, malformed, or violate Conv3d
                sharding constraints.
        """

        in_layout, w_layout, b_layout = cache_values[0], cache_values[1], cache_values[2]
        groups = cache_values[6]

        if not in_layout or not w_layout:
            raise ValueError(f"For {self.op_name}, Requires at least input and weight layouts.")

        self._check_partial_inputs([in_layout, w_layout])

        if b_layout is not None:
            self._check_partial_inputs([b_layout])

        if in_layout.mesh_shape != w_layout.mesh_shape:
            raise ValueError(
                f"For {self.op_name}, input and weight must have the same mesh_shape, "
                f"but got input: {in_layout.mesh_shape} and weight: {w_layout.mesh_shape}"
            )
        if b_layout is not None and b_layout.mesh_shape != in_layout.mesh_shape:
            raise ValueError(
                f"For {self.op_name}, bias and input must have the same mesh_shape, "
                f"but got bias: {b_layout.mesh_shape} and input: {in_layout.mesh_shape}"
            )

        in_map = in_layout.alias_tensor_map
        w_map = w_layout.alias_tensor_map

        # Validate dimensions
        if len(in_map) != 5 or len(w_map) != 5:
            raise ValueError(f"For {self.op_name}, Input and weight must be 5D.")

        # Delegate validation to helper methods to reduce cyclomatic complexity
        self._validate_row_parallelism(in_map, w_map, groups)
        self._validate_column_parallelism(w_layout, b_layout, groups)

        # Construct Output Map (N, C_out, D_out, H_out, W_out)
        out_map = [
            in_map[0],  # N
            w_map[0],   # C_out
            in_map[2],  # D
            in_map[3],  # H
            in_map[4]   # W
        ]

        # Build Layout
        output_layout = Layout(
            mesh_shape=in_layout.mesh_shape,
            alias_name=in_layout.alias_name,
            rank_list=in_layout.rank_list,
        )
        output_layout = output_layout(*tuple(out_map))

        # Set Partial status for Row Parallelism
        if in_map[1] != "None":
            axes = in_map[1] if isinstance(in_map[1], tuple) else (in_map[1],)
            for axis in axes:
                output_layout.set_partial_by_dev_axis(axis, "sum")

        return (output_layout,), None

    def get_expand_impl(self, func: Optional[Callable], infer_result: tuple,
                        cache_values: list) -> Optional[Callable]:
        """
        Get expand implementation for the operator.
        Intercepts the execution to handle Grouped Convolution with Column Parallelism.
        """
        w_layout = cache_values[1]
        w_map = w_layout.alias_tensor_map
        w_map_0 = w_map[0]

        # If Weight is NOT sharded on C_out (dim=0), native conv3d works fine.
        if w_map_0 == "None":
            return None

        parsed_groups = cache_values[6]
        if parsed_groups == 1:
            return None

        mesh = w_layout.mesh
        axes = w_map_0 if isinstance(w_map_0, tuple) else (w_map_0,)
        dev_num = 1
        local_rank = 0
        for axis_name in axes:
            axis_size = mesh.get_device_num_along_axis(axis_name)
            dev_num *= axis_size
            local_rank = local_rank * axis_size + mesh.get_local_rank(axis_name)

        # Pre-calculate local groups and group boundaries for the current device ahead of time.
        # This hoisting optimization avoids redundant calculations during every forward pass.
        local_groups = parsed_groups // dev_num
        start_group = local_rank * local_groups
        end_group = start_group + local_groups

        def distributed_conv3d_impl(input_tensor, weight_tensor, bias=None, stride=1, padding=0, dilation=1, groups=1):
            # --- Handling Groups > 1 with Column Parallelism ---
            # Calculate the input channel chunk size
            c_in = input_tensor.shape[1]
            c_in_per_group = c_in // groups

            # Map the pre-calculated groups to the actual input channels
            # Uses start_group and end_group captured from the outer scope
            start_channel = start_group * c_in_per_group
            end_channel = end_group * c_in_per_group

            # Slice the replicated input to match the local groups
            sliced_input = input_tensor[:, start_channel:end_channel, ...]

            # Execute native conv3d with the sliced input and adjusted local groups
            return func(sliced_input, weight_tensor, bias, stride, padding, dilation, local_groups)

        return distributed_conv3d_impl
