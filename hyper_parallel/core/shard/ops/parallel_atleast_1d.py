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

from hyper_parallel.core.layout import Layout
from .parallel_ops import DistributedOp

class Atleast1DDistributedOp(DistributedOp):
    """Distributed implementation for torch.atleast_1d."""

    def infer_layout(self, layouts, extra_args=None):
        """
        Infer output layout for torch.atleast_1d.

        PyTorch semantics:
          - If a single tensor is provided, a single tensor is returned.
          - If multiple tensors are provided, a tuple of tensors is returned.
          - 0-dimensional tensors are converted to 1-dimensional tensors.
          - 1-dimensional or higher tensors are preserved.

        Distributed Rule:
          - If 0D -> 1D: The new dimension is unsharded (-1).
          - If ND -> ND (N >= 1): The layout remains unchanged.

        Args:
            layouts (list or tuple): Layouts of input tensors.
            extra_args (tuple): Additional arguments (usually empty for atleast_1d).

        Returns:
            Layout or tuple[Layout]: Output tensor layout(s).
        """
        if not layouts:
            raise ValueError(
                f"Operation {self.op_name}: atleast_1d requires at least one input tensor layout."
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
            return output_layouts[0]

        return tuple(output_layouts)
