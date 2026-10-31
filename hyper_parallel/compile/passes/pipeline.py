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
   - FsdpBucketingPass (opt-in AG/RS fusion; supports FSDP with PP disabled).
   - PpPass (pipeline-parallel stage split) — runs AFTER FSDPPass: the FSDP
     collectives are value-level FX nodes that follow the data flow, so the
     stage slices inherit this stage's all_gather/reduce_scatter untouched
     while foreign stages' collectives are simply never copied. Splitting
     first would hand FSDPPass two subgraphs and a loss-less output list,
     breaking its grad-index contract.
3. Communication-compute overlap: AutoOverlapPass, then FsdpOverlapPass
   (independent FSDP all-gather prefetch and gradient overlap).
4. Pipeline split: PpPass runs last, after every graph-level pass. It
   replaces the joint graph with a schedule stub, moving the collectives
   into the schedule's per-stage submodules where a graph-level pass can
   no longer reach them. Running it last keeps the non-PP pass order
   untouched and still lets FsdpOverlapPass reorder the FSDP collectives
   before the split.

The backend compilation slot (InductorPass) is intentionally not wired yet;
add it here when an inductor backend integration lands.
"""

__all__ = ["PassPipeline"]

from typing import TYPE_CHECKING, Any, List, Optional

from ..pass_config import PassConfig
from .base import GraphPass
from .bucketing.bucketing_pass import FsdpBucketingPass
from .overlap.schedule import AutoOverlapPass, FsdpOverlapPass
from .parallel.fsdp_pass import FSDPPass
from .parallel.pp_pass import PpPass

if TYPE_CHECKING:
    from torch import fx
    from ..graph_parallel_plan import GraphParallelPlan


class PassPipeline:
    """Pass Pipeline Orchestrator.

    Automatically builds Pass execution order based on parallel configuration.
    """

    def __init__(
        self,
        pass_config: PassConfig,
        parallel_plan: Optional["GraphParallelPlan"] = None,
    ) -> None:
        """Initialize the pipeline with a parallel config and optional plan.

        Args:
            pass_config: Parallel configuration driving pass selection.
            parallel_plan: Optional sharding plan forwarded to partitioning
                passes (e.g. ``FSDPPass``).
        """
        self.config: PassConfig = pass_config
        self.parallel_plan = parallel_plan
        self.passes: List[GraphPass] = []

    def build(self) -> "PassPipeline":
        """Build Pass Pipeline."""
        # 1. Basic optimization
        self.passes.append(DeadCodeEliminationPass())
        self.passes.append(CanonicalizeGraphPass())

        # 2. Execution layer: Parallel dimension partitioning
        if getattr(self.config, "fsdp_enabled", False):
            self.passes.append(FSDPPass(parallel_plan=self.parallel_plan))
        if self.config.fsdp_bucketing:
            self.passes.append(FsdpBucketingPass())
        if getattr(self.config, "pp_enabled", False):
            self.passes.append(PpPass(parallel_plan=self.parallel_plan))

        # 3. Communication-compute overlap optimization
        if getattr(self.config, "enable_overlap", False):
            self.passes.append(AutoOverlapPass())
        if self.config.fsdp_allgather_prefetch or self.config.fsdp_overlap_grad_reduce:
            self.passes.append(FsdpOverlapPass())

        # 4. Pipeline split must run last: it replaces the joint graph with a
        # schedule stub, moving the collectives into the schedule's per-stage
        # submodules. Every graph-level pass (notably FsdpOverlapPass) must
        # therefore see the collectives before the split.
        if getattr(self.config, "pp_enabled", False):
            self.passes.append(PpPass(parallel_plan=self.parallel_plan))

        return self

    def run(
        self,
        graph_module: "fx.GraphModule",
        **kwargs: Any,
    ) -> "fx.GraphModule":
        """Execute all Passes against ``graph_module`` using the pipeline's config.

        The pipeline is built from a single ``PassConfig``
        (``from_config`` / ``__init__``); that same config drives every
        pass. If a caller needs to run the same pipeline against a
        different config, build a new pipeline.

        Args:
            graph_module: The FX GraphModule to transform.
            **kwargs: Extra keyword arguments forwarded to every pass
                (e.g. ``model`` / ``fsdp_group_name``). ``parallel_plan``
                is auto-filled from the pipeline's plan when omitted.

        Returns:
            The transformed graph module.
        """
        # Ensure the parallel_plan is always available to passes that need
        # it (FSDPPass reads it from kwargs), even if the caller did not
        # pass it explicitly.
        if "parallel_plan" not in kwargs and self.parallel_plan is not None:
            kwargs["parallel_plan"] = self.parallel_plan

        config = self.config
        for graph_pass in self.passes:
            graph_module = graph_pass.run(graph_module, config, **kwargs)
        return graph_module

    @classmethod
    def from_config(
        cls,
        config: PassConfig,
        parallel_plan: Optional["GraphParallelPlan"] = None,
    ) -> "PassPipeline":
        """Create Pipeline from configuration."""
        return cls(config, parallel_plan).build()


class DeadCodeEliminationPass(GraphPass):
    """Dead code elimination Pass."""

    name = "dead_code_elimination"

    def run(
        self,
        graph_module: "fx.GraphModule",
        pass_config: PassConfig,
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
        pass_config: PassConfig,
        **kwargs: Any,
    ) -> "fx.GraphModule":
        """Lint and recompile the graph to a canonical form."""
        graph_module.graph.lint()
        graph_module.recompile()
        return graph_module
