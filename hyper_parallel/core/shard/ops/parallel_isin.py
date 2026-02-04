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
Distributed implementation for Isin operator.
"""

from hyper_parallel.core.layout import Layout
from .parallel_ops import DistributedOp

class IsinDistributedOp(DistributedOp):
    """Distributed implementation for torch.isin."""

    def infer_layout(self, layouts, extra_args=None):
        """
        Infer output layout for torch.isin(elements, test_elements, ...)

        PyTorch semantics:
          - Returns boolean tensor with SAME SHAPE as `elements`
          - Each element is tested against ALL values in `test_elements`, so requires GLOBAL view of `test_elements`

        Args:
            layouts (tuple): Layouts of inputs. Expected:
                layouts[0] (Layout): Layout of `elements` tensor (required).
                layouts[1] (Layout): Layout of `test_elements` tensor (required).
            extra_args: No need.

        Returns:
            Layout: Output layout identical to `elements` layout (boolean tensor with same sharding).
        """
        # Validate elements layout
        if not layouts or layouts[0] is None:
            raise ValueError(
                f"Operation {self.op_name}: 'elements' requires a valid tensor layout."
            )
        elements_layout = layouts[0]

        # Validate test_elements layout
        if len(layouts) < 2 or layouts[1] is None:
            raise ValueError(
                f"Operation {self.op_name}: 'test_elements' requires a valid tensor layout."
            )
        test_elements_layout = layouts[1]

        # test_elements must be unsharded
        if not all(shard_way == -1 for shard_way in test_elements_layout.tensor_map):
            raise ValueError(
                f"Operation {self.op_name}: 'test_elements' must be unsharded. "
                f"Current tensor_map: {test_elements_layout.tensor_map}."
            )

        mesh_shape = elements_layout.mesh_shape
        alias_name = elements_layout.alias_name
        rank_list = elements_layout.rank_list
        tensor_map = elements_layout.tensor_map

        def idx_to_alias(idx, aliases):
            if idx == -1:
                return "None"
            return aliases[len(aliases) - idx - 1]

        output_map_aliases = tuple(
            idx_to_alias(idx, alias_name) for idx in tensor_map
        )

        output_layout = Layout(
            mesh_shape=mesh_shape,
            alias_name=alias_name,
            rank_list=rank_list
        )
        output_layout = output_layout(*output_map_aliases)

        return output_layout
