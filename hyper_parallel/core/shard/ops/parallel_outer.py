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

from hyper_parallel.core.layout import Layout
from .parallel_ops import DistributedOp


class OuterDistributedOp(DistributedOp):
    """Distributed implementation for torch.outer."""

    def infer_layout(self, layouts, extra_args=None):
        """
        Infer output layout for torch.outer.

        PyTorch semantics:
          - Computes the outer product of two 1-D tensors.
          - If input is of size N and vec2 is of size M, the output is of size (N, M).
          - Input tensors must be 1-D.

        Distributed semantics:
          - The 0-th dimension of the output inherits the layout of input.
          - The 1-st dimension of the output inherits the layout of vec2.
          - The two inputs cannot be sharded along the same device mesh dimension.

        Args:
            layouts (tuple): Layouts of inputs. Expected:
                layouts[0] (Layout): Layout of the first 1-D tensor (input).
                layouts[1] (Layout): Layout of the second 1-D tensor (vec2).
            extra_args (tuple, optional): Unused for outer.

        Returns:
            Layout: The 2-D output tensor layout.
        """
        if not layouts or len(layouts) != 2:
            raise ValueError(
                f"Operation {self.op_name}: requires exactly 2 input layouts."
            )

        layout1, layout2 = layouts[0], layouts[1]

        if layout1 is None or layout2 is None:
            raise ValueError(
                f"Operation {self.op_name}: requires both inputs to have valid layouts."
            )

        map1 = layout1.tensor_map
        map2 = layout2.tensor_map

        if len(map1) != 1 or len(map2) != 1:
            raise ValueError(
                f"Operation {self.op_name}: requires exactly 1-D tensors as inputs, "
                f"but got {len(map1)}-D and {len(map2)}-D."
            )

        dim0_map = map1[0]
        dim1_map = map2[0]

        # Helper to extract all sharded mesh dimensions for a tensor dimension
        def _get_flattened_map(dim_map):
            if isinstance(dim_map, int):
                return {dim_map} if dim_map != -1 else set()
            return set(dim_map)

        set1 = _get_flattened_map(dim0_map)
        set2 = _get_flattened_map(dim1_map)

        # Ensure the two 1D tensors are not sharded on the same mesh dimension
        if set1.intersection(set2):
            raise ValueError(
                f"Operation {self.op_name}: the two inputs cannot be sharded on the "
                f"same device mesh dimension. Conflict on mesh index: {set1.intersection(set2)}"
            )

        # Build output tensor map: (input_dim, vec2_dim)
        output_map = [dim0_map, dim1_map]

        # Construct output layout
        mesh_shape = layout1.mesh_shape
        alias_name = layout1.alias_name
        rank_list = layout1.rank_list

        def idx_to_alias(idx_item, aliases):
            # Handles both single int and nested tuple mapping
            if isinstance(idx_item, int):
                if idx_item == -1:
                    return "None"
                return aliases[len(aliases) - idx_item - 1]
            # Handle multi-axis sharding (tuple)
            return tuple(
                "None" if sub_idx == -1 else aliases[len(aliases) - sub_idx - 1]
                for sub_idx in idx_item
            )

        output_alias_map = tuple(idx_to_alias(idx, alias_name) for idx in output_map)

        output_layout = Layout(
            mesh_shape=mesh_shape,
            alias_name=alias_name,
            rank_list=rank_list
        )

        output_layout = output_layout(*output_alias_map)
        return output_layout
