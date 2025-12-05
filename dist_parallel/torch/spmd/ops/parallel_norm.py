# Copyright 2025 Huawei Technologies Co., Ltd
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
Distributed implementation for TopK operator.
"""

from ... import Layout
from .parallel_ops import DistributedOp

class NormDistributedOp(DistributedOp):
    """Distributed implementation for Norm operator."""

    def infer_layout(self, input_layouts, extra_args=None):
        """
        Infer output layouts for normalization operator (e.g., RmsNorm).

        This method determines the proper output layout for normalization operations
        based on the input layouts, ensuring that the normalization operation is
        compatible with the distributed training setup.

        Args:
            input_layouts (tuple): A tuple of Layout objects representing the input tensor layouts.
                Expected to contain at least three layouts: input tensor, gamma parameter, and beta parameter.
            extra_args (dict, optional): Additional arguments that might be needed for layout inference.
                Defaults to None.

        Returns:
            tuple: A tuple containing two Layout objects:
                - First layout: Layout for the input gradient tensor
                - Second layout: Layout for the output tensor

        Raises:
            ValueError: If the number of input layouts is less than 3.
            ValueError: If input layouts are inconsistent.
            ValueError: If device matrices of input layouts don't match.
            ValueError: If normalization axis is sharded, which is not supported.
            ValueError: If gamma parameter layout doesn't match the input layout in normalization dimensions.
        """
        if len(input_layouts) < 3:
            raise ValueError(f"RmsNorm input layouts size {len(input_layouts)} is less than 3.")
        x_layout = input_layouts[0]
        gamma_layout = input_layouts[-2]
        x_device_matrix = x_layout.device_matrix
        for i, layout in enumerate(input_layouts[:-2]):
            if layout != x_layout:
                raise ValueError(f"RmsNorm inputs must have same layout, but input 0 layout is: {x_layout},"
                                 f"input {i} layout is: {layout}.")
        gamma_device_matrix = gamma_layout.device_matrix
        if x_device_matrix != gamma_device_matrix:
            raise ValueError("RmsNorm inputs must have same device_matrix")
        x_tensor_map = x_layout.tensor_map
        gamma_tensor_map = gamma_layout.tensor_map
        begin_norm_axis = len(x_tensor_map) - len(gamma_tensor_map)
        for axis in x_tensor_map[begin_norm_axis:]:
            if axis == -1:
                continue
            if isinstance(axis, tuple):
                for iaxis in axis:
                    if iaxis == -1:
                        continue
                    if x_device_matrix[len(x_device_matrix) - 1 - iaxis] > 1:
                        raise ValueError(f"RmsNorm is disabled to support the splitting after "
                                         f"begin_norm_axis {begin_norm_axis} for input 0.")
            if x_device_matrix[len(x_device_matrix) - 1 - axis] > 1:
                raise ValueError(f"RmsNorm is disabled to support the splitting after "
                                 f"begin_norm_axis {begin_norm_axis} for input 0.")
        if x_tensor_map[begin_norm_axis:] != gamma_tensor_map:
            raise ValueError(f"The input sharding in the first {begin_norm_axis} dimensions "
                             f"{x_layout.alias_tensor_map[begin_norm_axis:]} should equal to"
                             f" the gamma sharding {gamma_layout.alias_tensor_map}")
        output_layout = Layout(
            device_matrix=x_layout.device_matrix,
            alias_name=x_layout.alias_name,
            rank_list=x_layout.rank_list
        )
        output_map = x_layout.alias_tensor_map[:begin_norm_axis] + ("None",) * len(gamma_tensor_map)
        out_layout = output_layout(*output_map)
        return x_layout, out_layout
