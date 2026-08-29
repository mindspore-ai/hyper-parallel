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
Pass Pipeline - Pass Orchestrator

Automatically builds Pass execution order based on parallel configuration.

Execution Order:
1. Basic optimization: DeadCodeElimination, CanonicalizeGraph
2. Execution layer:
   - FSDPPass (FSDP)
3. Communication-compute overlap: AutoOverlapPass

The backend compilation slot (InductorPass) is intentionally not wired yet;
add it here when an inductor backend integration lands.
"""

from typing import TYPE_CHECKING, Any, List, Optional

from ..parallel_config import ParallelConfig
from .base import GraphPass
from .overlap.schedule import AutoOverlapPass
from .parallel.fsdp_pass import FSDPPass

if TYPE_CHECKING:
    from torch import fx
    from ..sharding_config import ShardingPlan


class PassPipeline:
    """Pass Pipeline Orchestrator.

    Automatically builds Pass execution order based on parallel configuration.
    """

    def __init__(
        self,
        parallel_config: ParallelConfig,
        sharding_plan: Optional["ShardingPlan"] = None,
    ) -> None:
        """Initialize the pipeline with a parallel config and optional plan.

        Args:
            parallel_config: Parallel configuration driving pass selection.
            sharding_plan: Optional sharding plan forwarded to partitioning
                passes (e.g. ``FSDPPass``).
        """
        self.config: ParallelConfig = parallel_config
        self.sharding_plan = sharding_plan
        self.passes: List[GraphPass] = []

    def build(self) -> "PassPipeline":
        """Build Pass Pipeline."""
        # 1. Basic optimization
        self.passes.append(DeadCodeEliminationPass())
        self.passes.append(CanonicalizeGraphPass())

        # 2. Execution layer: Parallel dimension partitioning
        if getattr(self.config, "fsdp_enabled", False):
            self.passes.append(FSDPPass(sharding_plan=self.sharding_plan))

        # 3. Communication-compute overlap optimization
        if getattr(self.config, "enable_overlap", False):
            self.passes.append(AutoOverlapPass())

        return self

    def run(
        self,
        graph_module: "fx.GraphModule",
        **kwargs: Any,
    ) -> "fx.GraphModule":
        """Execute all Passes against ``graph_module`` using the pipeline's config.

        The pipeline is built from a single ``ParallelConfig``
        (``from_config`` / ``__init__``); that same config drives every
        pass. If a caller needs to run the same pipeline against a
        different config, build a new pipeline.

        Args:
            graph_module: The FX GraphModule to transform.
            **kwargs: Extra keyword arguments forwarded to every pass
                (e.g. ``model`` / ``fsdp_group_name``). ``sharding_plan``
                is auto-filled from the pipeline's plan when omitted.

        Returns:
            The transformed graph module.
        """
        # Ensure the sharding_plan is always available to passes that need
        # it (FSDPPass reads it from kwargs), even if the caller did not
        # pass it explicitly.
        if "sharding_plan" not in kwargs and self.sharding_plan is not None:
            kwargs["sharding_plan"] = self.sharding_plan

        config = self.config
        for graph_pass in self.passes:
            graph_module = graph_pass.run(graph_module, config, **kwargs)
        return graph_module

    @classmethod
    def from_config(
        cls,
        config: ParallelConfig,
        sharding_plan: Optional["ShardingPlan"] = None,
    ) -> "PassPipeline":
        """Create Pipeline from configuration."""
        return cls(config, sharding_plan).build()


class DeadCodeEliminationPass(GraphPass):
    """Dead code elimination Pass."""

    name = "dead_code_elimination"

    def run(
        self,
        graph_module: "fx.GraphModule",
        parallel_config: ParallelConfig,
        **kwargs: Any,
    ) -> "fx.GraphModule":
        """Eliminate dead nodes and recompile the graph."""
        graph_module.graph.eliminate_dead_code()
        graph_module.recompile()
        return graph_module


class CanonicalizeGraphPass(GraphPass):
    """Graph canonicalization Pass."""

    name = "canonicalize_graph"

    def run(
        self,
        graph_module: "fx.GraphModule",
        parallel_config: ParallelConfig,
        **kwargs: Any,
    ) -> "fx.GraphModule":
        """Lint and recompile the graph to a canonical form."""
        graph_module.graph.lint()
        graph_module.recompile()
        return graph_module


__all__ = ["PassPipeline"]
