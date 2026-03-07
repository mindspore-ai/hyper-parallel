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

from hyper_parallel.core.dtensor.layout import Layout
from .parallel_ops import DistributedOp

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
            if in_map[1] != -1 or w_map[1] != -1:
                # Row Parallelism with groups > 1 requires advanced group-wise communication
                raise ValueError(f"{self.op_name}: Sharding on C_in with groups > 1 is not supported.")

        # 2. Check Row Parallelism (Sharding on Channel In)
        # Input: (N, C_in, D, H, W), Weight: (C_out, C_in/groups, kD, kH, kW)
        if in_map[1] != -1:
            if in_map[1] != w_map[1]:
                raise ValueError(f"{self.op_name}: Input C_in and Weight C_in must be sharded on the same axis.")

    def _validate_column_parallelism(self, w_layout, b_layout, groups):
        """
        Validate constraints for Column Parallelism.
        """
        w_map = w_layout.tensor_map
        w_map_0 = w_map[0][0] if isinstance(w_map[0], tuple) else w_map[0]

        if w_map_0 != -1:
            # Check bias alignment
            if b_layout is not None:
                b_map = b_layout.tensor_map
                b_map_0 = b_map[0][0] if isinstance(b_map[0], tuple) else b_map[0]
                if w_map_0 != b_map_0:
                    raise ValueError(f"{self.op_name}: Weight C_out and Bias C_out must be sharded on the same axis.")

            # Check groups divisibility for Column Parallelism
            if groups > 1:
                axis_name = w_layout.alias_name[len(w_layout.alias_name) - 1 - w_map_0]
                dev_num = w_layout.mesh.get_device_num_along_axis(axis_name)

                if groups % dev_num != 0:
                    raise ValueError(
                        f"{self.op_name}: For Column Parallelism, groups ({groups}) "
                        f"must be divisible by tp_size ({dev_num})."
                    )

    def infer_layout(self, layouts, extra_args=None):
        """
        Infer output layout for Conv3d based on PyTorch functional.conv3d signature:
        (input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1)

        Args:
            layouts (tuple): (input_layout, weight_layout, bias_layout)
            extra_args (tuple): (stride, padding, dilation, groups)
        """
        self._check_partial_inputs(layouts)

        if not layouts or len(layouts) < 2:
            raise ValueError(f"{self.op_name}: Requires at least input and weight layouts.")

        in_layout, w_layout = layouts[0], layouts[1]
        b_layout = layouts[2] if len(layouts) > 2 else None

        # Extract groups from extra_args (index 3 based on functional.conv3d signature)
        # stride=0, padding=1, dilation=2, groups=3
        groups = extra_args[3] if extra_args and len(extra_args) > 3 else 1

        in_map = in_layout.tensor_map
        w_map = w_layout.tensor_map

        # Validate dimensions
        if len(in_map) != 5 or len(w_map) != 5:
            raise ValueError(f"{self.op_name}: Input and weight must be 5D.")

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
        mesh_shape = in_layout.mesh_shape
        alias_name = in_layout.alias_name
        rank_list = in_layout.rank_list

        def idx_to_alias(idx):
            if idx == -1: return "None"
            return alias_name[len(alias_name) - idx - 1]

        output_alias_map = tuple(idx_to_alias(idx) for idx in out_map)
        output_layout = Layout(mesh_shape, alias_name, rank_list)
        output_layout = output_layout(*output_alias_map)

        # Set Partial status for Row Parallelism
        if in_map[1] != -1:
            partial_axis = idx_to_alias(in_map[1])
            output_layout.set_partial_by_dev_axis(partial_axis, "sum")

        return output_layout

    def get_expand_impl(self, func, output_layout, layouts, extra_args):
        """
        Get expand implementation for the operator.
        Intercepts the execution to handle Grouped Convolution with Column Parallelism.
        """
        _, w_layout = layouts[0], layouts[1]
        w_map = w_layout.tensor_map

        # Extract the exact mesh mapping for C_out
        w_map_0 = w_map[0][0] if isinstance(w_map[0], tuple) else w_map[0]

        # If Weight is NOT sharded on C_out (dim=0), native conv3d works fine.
        if w_map_0 == -1:
            return None

        parsed_groups = extra_args[3] if extra_args and len(extra_args) > 3 else 1

        mesh = w_layout.mesh
        # Find the mesh axis name where C_out is sharded
        axis_name = w_layout.alias_name[len(w_layout.alias_name) - 1 - w_map_0]
        dev_num = mesh.get_device_num_along_axis(axis_name)
        local_rank = mesh.get_local_rank(axis_name)

        # Pre-calculate local groups and group boundaries for the current device ahead of time.
        # This hoisting optimization avoids redundant calculations during every forward pass.
        local_groups = parsed_groups // dev_num if parsed_groups > 1 else 1
        start_group = local_rank * local_groups
        end_group = start_group + local_groups


        def distributed_conv3d_impl(input_tensor, weight_tensor, bias=None, stride=1, padding=0, dilation=1, groups=1):
            # If standard convolution, fallback to native PyTorch function
            if groups == 1:
                return func(input_tensor, weight_tensor, bias, stride, padding, dilation, groups)

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
