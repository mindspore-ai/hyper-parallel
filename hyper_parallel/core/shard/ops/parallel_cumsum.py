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
Distributed implementation for Cumsum operator.
"""

from hyper_parallel.core.layout import Layout
from .parallel_ops import DistributedOp

class CumsumDistributedOp(DistributedOp):
    """Distributed implementation for torch.cumsum."""

    def infer_layout(self, layouts, extra_args=None):
        """
        Infer output layout for torch.cumsum

        PyTorch semantics:
          - Computes cumulative sum along dimension `dim`
          - Output shape is identical to input shape
          - Operation is sequential along `dim`: each element depends on all preceding elements in that dimension

        Critical sharding constraint:
          - The dimension `dim` MUST be unsharded (-1 in tensor_map)

        Args:
            layouts (tuple): Layouts of inputs. Expected:
                layouts[0] (Layout): Input tensor layout (required).
            extra_args (tuple): Should contain 'dim'. Expected:
                extra_args[0] (int): Dimension to perform cumsum over (required).

        Returns:
            Layout: Output tensor layout (identical to input layout after validation).
        """
        if not layouts or layouts[0] is None:
            raise ValueError(
                f"Operation {self.op_name}: cumsum requires a valid input tensor layout."
            )
        input_layout = layouts[0]
        in_tensor_map = input_layout.tensor_map
        input_ndim = len(in_tensor_map)

        if not extra_args or extra_args[0] is None:
            raise ValueError(
                f"Operation {self.op_name}: cumsum requires 'dim' parameter in extra_args."
            )
        dim = extra_args[0]
        if not isinstance(dim, int):
            raise ValueError(
                f"Operation {self.op_name}: 'dim' must be an integer, got {type(dim)}."
            )

        # Normalize negative dimensions
        normalized_dim = dim if dim >= 0 else dim + input_ndim
        if normalized_dim < 0 or normalized_dim >= input_ndim:
            raise ValueError(
                f"Operation {self.op_name}: Dimension {dim} out of range for "
                f"{input_ndim}-dimensional input tensor."
            )

        # cumsum dimension must be unsharded
        if in_tensor_map[normalized_dim] != -1:
            raise ValueError(
                    f"Operation {self.op_name}: Cannot perform sharding on normalized dimension {normalized_dim}, "
                    f"but found sharding assignment: {in_tensor_map[normalized_dim]}"
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
