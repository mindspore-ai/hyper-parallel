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

from hyper_parallel.core.layout import Layout
from .parallel_ops import DistributedOp

class NormDistributedOp(DistributedOp):
    """Distributed implementation for Norm operator."""

    def infer_layout(self, layouts, extra_args=None):
        """
        Infer output layouts for normalization operator (e.g., RmsNorm).

        This method determines the proper output layout for normalization operations
        based on the input layouts, ensuring that the normalization operation is
        compatible with the distributed training setup.

        Args:
            layouts (tuple): A tuple of Layout objects representing the input tensor layouts.
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
            ValueError: If input layouts have partial status.
        """
        if len(layouts) < 3:
            raise ValueError(f"RmsNorm input layouts size {len(layouts)} is less than 3.")
        # Check partial inputs
        if not self._allow_partial_inputs:
            self._check_partial_inputs(layouts)
        x_layout = layouts[0]
        gamma_layout = layouts[-2]
        x_mesh_shape = x_layout.mesh_shape
        for i, layout in enumerate(layouts[:-2]):
            if layout != x_layout:
                raise ValueError(f"RmsNorm inputs must have same layout, but input 0 layout is: {x_layout},"
                                 f"input {i} layout is: {layout}.")
        gamma_mesh_shape = gamma_layout.mesh_shape
        if x_mesh_shape != gamma_mesh_shape:
            raise ValueError("RmsNorm inputs must have same mesh_shape")
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
                    if x_mesh_shape[len(x_mesh_shape) - 1 - iaxis] > 1:
                        raise ValueError(f"RmsNorm is disabled to support the splitting after "
                                         f"begin_norm_axis {begin_norm_axis} for input 0.")
            if x_mesh_shape[len(x_mesh_shape) - 1 - axis] > 1:
                raise ValueError(f"RmsNorm is disabled to support the splitting after "
                                 f"begin_norm_axis {begin_norm_axis} for input 0.")
        if x_tensor_map[begin_norm_axis:] != gamma_tensor_map:
            raise ValueError(f"The input sharding in the first {begin_norm_axis} dimensions "
                             f"{x_layout.alias_tensor_map[begin_norm_axis:]} should equal to"
                             f" the gamma sharding {gamma_layout.alias_tensor_map}")
        output_layout = Layout(
            mesh_shape=x_layout.mesh_shape,
            alias_name=x_layout.alias_name,
            rank_list=x_layout.rank_list
        )
        output_map = x_layout.alias_tensor_map[:begin_norm_axis] + ("None",) * len(gamma_tensor_map)
        out_layout = output_layout(*output_map)
        return x_layout, out_layout

class LayerNormDistributedOp(DistributedOp):
    """Distributed implementation for torch.nn.functional.layer_norm."""

    def infer_layout(self, layouts, extra_args=None):
        """
        Infer output layout for layer_norm.

        PyTorch rules:
          - normalized_shape specifies the last N dimensions to normalize over.
          - All dimensions in normalized_shape MUST be unsharded for correctness.
          - Output layout is identical to input layout (shape unchanged).

        Args:
        layouts (tuple): Layouts of inputs. Expected:
            layouts[0] (Layout): Input tensor layout (required).
        extra_args (tuple): Should contain 'normalized_shape'. Expected:
            extra_args[0] (int | list | tuple): Normalized shape to be unsharded.

        Returns:
            Layout object representing output tensor layout (same as input if valid).
        """
        if not layouts or layouts[0] is None:
            raise ValueError("layer_norm requires a valid input tensor layout.")
        input_layout = layouts[0]
        in_tensor_map = input_layout.tensor_map  # e.g., (-1, 0, -1) for 3D tensor

        if not extra_args or extra_args[0] is None:
            raise ValueError("layer_norm requires normalized_shape in extra_args.")
        normalized_shape = extra_args[0]

        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        elif isinstance(normalized_shape, (list, tuple)):
            normalized_shape = tuple(normalized_shape)
        else:
            raise ValueError(f"normalized_shape must be int, list, or tuple, got {type(normalized_shape)}")

        input_ndim = len(in_tensor_map)
        norm_ndim = len(normalized_shape)

        if norm_ndim > input_ndim:
            raise ValueError(
                f"normalized_shape {normalized_shape} (dims={norm_ndim}) is larger than input ndim={input_ndim}."
            )

        # The last `norm_ndim` dimensions are going to be normalized
        dims_to_normalize = list(range(input_ndim - norm_ndim, input_ndim))

        # All normalized dims must be unsharded
        for dim in dims_to_normalize:
            if in_tensor_map[dim] != -1:
                raise ValueError(
                    f"Operation {self.op_name}: Cannot perform sharding on normalized dimension {dim}, "
                    f"but found sharding assignment: {in_tensor_map[dim]}"
                )

        mesh_shape = input_layout.mesh_shape
        alias_name = input_layout.alias_name
        rank_list = input_layout.rank_list

        # Create output layout
        def idx_to_alias(idx, aliases):
            if idx == -1:
                return "None"
            return aliases[len(aliases) - idx - 1]
        output_map = tuple(idx_to_alias(idx, alias_name) for idx in in_tensor_map)

        output_layout = Layout(
            mesh_shape=mesh_shape,
            alias_name=alias_name,
            rank_list=rank_list
        )
        output_layout = output_layout(*output_map)
        return output_layout
