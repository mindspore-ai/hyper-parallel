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
        parallel_config: Any,
        **kwargs: Any,
    ) -> fx.GraphModule:
        """
        Optimize communication-compute overlap by moving wait_tensor nodes.

        FSDPPass inserts wait_tensor immediately after communication (for correctness).
        This pass moves wait_tensor later to allow overlap with independent computation.
        """
        if not getattr(parallel_config, "enable_overlap", False):
            return graph_module

        graph = graph_module.graph

        # Find all wait_tensor nodes
        wait_nodes = self._find_wait_nodes(graph)

        # For each wait, try to move it later in the graph
        for wait_node in wait_nodes:
            graph_module = self._move_wait_later(graph_module, wait_node)

        graph_module.recompile()
        return graph_module

    def _find_wait_nodes(self, graph: fx.Graph) -> List[fx.Node]:
        """Find all wait_tensor nodes in the graph."""
        wait_nodes = []
        for node in graph.nodes:
            # Match both string target and function object target
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
        """
        Move wait_tensor to a later position to maximize overlap.

        Strategy:
        1. Find all users of wait_node
        2. Find independent computation that can run during communication
        3. Move wait_tensor to just before the first dependent user

        Note: This is a placeholder for future implementation.
        Current implementation keeps wait_tensor in place.
        """
        # Reserved for the future overlap-scheduling implementation below.
        _ = wait_node

        # TODO: Implement sophisticated wait_tensor movement
        # For now, keep the wait where FSDPPass placed it (immediate wait)
        # This ensures correctness while we develop the optimization

        # Future implementation:
        # 1. Find the comm_node that this wait_node is waiting for
        # 2. Find all nodes between comm_node and first user of wait_node
        # 3. Check which of these nodes don't depend on comm_node
        # 4. Move wait_node after those independent nodes

        return graph_module


__all__ = ["AutoOverlapPass"]
