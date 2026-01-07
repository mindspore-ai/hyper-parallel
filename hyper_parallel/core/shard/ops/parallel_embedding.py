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
Distributed implementation for Embedding operator.
"""

from hyper_parallel.core.layout import Layout
from .parallel_ops import DistributedOp


class EmbeddingDistributedOp(DistributedOp):
    """Distributed implementation for Embedding operator."""

    def infer_layout(self, layouts, extra_args):
        """
        Infer output layout for Embedding operator.
        Args:
            x_layout (Layout): Layout of input x
        Returns:
            tuple: Layout for output tensor
        """
        # Check
        if len(layouts) != 6:
            raise ValueError(f"Embedding requires 6 layouts, but {len(layouts)}")
        if len(extra_args) != 2:
            raise ValueError(f"Embedding requires 2 extra args, but {len(extra_args)}")

        # Parse input layout info
        w_layout = layouts[1]

        w_dict = w_layout.to_dict()
        w_tensor_map, w_aliases = w_dict["tensor_map"], w_dict["alias_name"]

        mesh_shape = w_dict["mesh_shape"]
        rank_list = w_dict["rank_list"]

        # Create output layout
        def idx_to_alias(idx, aliases):
            if idx == -1:
                return "None"
            return aliases[len(aliases) - idx - 1]

        output_map = ()

        out_aliases = w_aliases
        if not (w_tensor_map[0] == -1 or mesh_shape[len(mesh_shape) - 1 - w_tensor_map[0]] == 1):
            raise ValueError(
                f"Operation {self.op_name}: Cannot perform sharding on params along the axis"
            )
        output_map += (-1, -1,)
        for i in range(1, len(w_tensor_map)):
            output_map += (len(w_tensor_map) - 1 - i,)

        output_map = tuple(idx_to_alias(idx, out_aliases) for idx in output_map)
        output_layout = Layout(
            mesh_shape=mesh_shape,
            alias_name=out_aliases,
            rank_list=rank_list,
        )
        output_layout = output_layout(*output_map)
        return output_layout
