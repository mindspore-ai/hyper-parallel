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
"""Schedule FSDP parameter communication in the joint graph."""

__all__ = ["AutoOverlapPass"]

from typing import Any, List

from torch import fx

from ...pass_config import PassConfig
from ..base import GraphPass
from .overlap_param_gather import overlap_param_gather


class AutoOverlapPass(GraphPass):
    """Overlap tagged FSDP gathers with forward and backward computation."""

    name = "auto_overlap"

    def run(
        self,
        graph_module: fx.GraphModule,
        pass_config: PassConfig,
        **kwargs: Any,
    ) -> fx.GraphModule:
        """Prefetch FSDP launches and sink their complete wait/view DAGs.

        Args:
            graph_module: Graph transformed by the parallel passes.
            pass_config: Overlap switch, prefetch depths, and byte budget.
            **kwargs: Pipeline context, unused by this graph-only scheduler.

        Returns:
            The transformed graph module.
        """
        return overlap_param_gather(graph_module, pass_config)

    @staticmethod
    def _find_wait_nodes(graph: fx.Graph) -> List[fx.Node]:
        """Find functional wait nodes for graph inspection."""
        return [node for node in graph.nodes
                if node.op == "call_function" and "wait_tensor" in str(node.target)]
