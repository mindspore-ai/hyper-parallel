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
Auto Overlap Pass - Communication-Compute Overlap Optimization

Responsibilities:
1. Find wait_tensor nodes for communication operations
2. Analyze dependencies to find overlap opportunities
3. Move wait_tensor to later position to maximize overlap

Optimization Strategies:
- Move wait_tensor as late as possible (just before first use)
- Allow communication to overlap with independent computation
"""

from typing import Any, List
from torch import fx

from ...parallel_config import PassConfig
from ..base import GraphPass


class AutoOverlapPass(GraphPass):
    """
    Auto Overlap Pass

    Automatically identifies opportunities to overlap communication with computation
    by reordering operations and issuing async communication early.
    """

    name = "auto_overlap"

    def run(
        self,
        graph_module: fx.GraphModule,
        pass_config: PassConfig,
        **kwargs: Any,
    ) -> fx.GraphModule:
        """Optimize communication-compute overlap by moving wait_tensor nodes.

        FSDPPass inserts wait_tensor immediately after communication (for correctness).
        This pass moves wait_tensor later to allow overlap with independent computation.
        """
        if not pass_config.enable_overlap:
            return graph_module

        graph = graph_module.graph
        wait_nodes = self._find_wait_nodes(graph)
        for wait_node in wait_nodes:
            graph_module = self._move_wait_later(graph_module, wait_node)

        graph_module.recompile()
        return graph_module

    def _find_wait_nodes(self, graph: fx.Graph) -> List[fx.Node]:
        """Find all wait_tensor nodes in the graph."""
        wait_nodes = []
        for node in graph.nodes:
            if node.op == "call_function":
                target_str = str(node.target)
                if (
                    "wait_tensor" in target_str
                    or "torch.distributed._functional_collectives.wait_tensor"
                    == target_str
                ):
                    wait_nodes.append(node)
        return wait_nodes

    def _move_wait_later(
        self,
        graph_module: fx.GraphModule,
        wait_node: fx.Node,
    ) -> fx.GraphModule:
        """Move wait_tensor later so it overlaps with independent compute.

        Placeholder: keeps the wait where FSDPPass placed it (immediate
        wait). A future implementation will sink each wait past independent
        nodes up to its first dependent user.
        """
        _ = wait_node  # intentional no-op until the scheduler lands
        return graph_module


__all__ = ["AutoOverlapPass"]
