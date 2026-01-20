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
            layouts (tuple): Layouts of input tensors.
            extra_args (dict): Additional arguments.

        Returns:
            Layout: Layout for output tensor
        """
        # Step 1: Validate input length  FIRST
        if len(layouts) < 2:
            raise ValueError(f"Embedding requires at least 2 layouts (input, weight), but got {len(layouts)}")


        # Step 2: Now it is safe to extract common weight layout info
        # MS: layouts[1] is weight
        # Torch: embedding(input, weight), so layouts[1] is weight
        w_layout = layouts[1]
        w_dict = w_layout.to_dict()
        w_tensor_map = w_dict["tensor_map"]
        w_aliases = w_dict["alias_name"]
        mesh_shape = w_dict["mesh_shape"]
        rank_list = w_dict["rank_list"]

        def idx_to_alias(idx, aliases):
            if idx == -1:
                return "None"
            return aliases[len(aliases) - idx - 1]

        output_map = ()

        # Step 3: Specific Logic Calculation
        # Inputs: input (indices), weight (table), [padding_idx, max_norm, ...]
        input_layout = layouts[0]
        inp_tensor_map = input_layout.tensor_map

        # Output Layout Logic:
        # 1. Inherit the distribution of the input indices (Batch/Seq dimensions)
        output_map += inp_tensor_map

        # 2. Inherit the distribution of the embedding dimension from the weight table.
        if len(w_tensor_map) > 0:
            embed_dim_map = w_tensor_map[-1]
            output_map += (embed_dim_map,)

        # Reconstruct final Layout object
        output_aliases_tuple = tuple(idx_to_alias(idx, w_aliases) for idx in output_map)

        output_layout = Layout(
            mesh_shape=mesh_shape,
            alias_name=w_aliases,
            rank_list=rank_list,
        )

        return output_layout(*output_aliases_tuple)
