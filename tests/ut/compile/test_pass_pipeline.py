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
"""Unit tests for ``hyper_parallel.compile.passes.pipeline.PassPipeline``.

Covers the API contract asserted by the refactor:

1. ``PassPipeline.build`` selects passes from the config flags
   (``fsdp_enabled`` / ``enable_overlap``), always prefixed by the basic
   optimization passes.
2. ``PassPipeline.run`` drives every pass with the pipeline's own config
   (``self.config``); the redundant ``parallel_config`` parameter that
   shadowed it is gone — passing it raises a clean TypeError.
3. ``PassPipeline.run`` auto-fills ``sharding_plan`` from the pipeline's
   plan when the caller omits it.
4. ``from_config`` builds + returns a ready pipeline.
"""

import os
import unittest
from typing import Any, List, Tuple

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

import torch
from torch import fx

from hyper_parallel.compile.parallel_config import ParallelConfig
from hyper_parallel.compile.passes.base import GraphPass
from hyper_parallel.compile.passes.overlap.schedule import AutoOverlapPass
from hyper_parallel.compile.passes.parallel.fsdp_pass import FSDPPass
from hyper_parallel.compile.passes.pipeline import (
    CanonicalizeGraphPass,
    DeadCodeEliminationPass,
    PassPipeline,
)
from hyper_parallel.compile.sharding_config import ShardingPlan


class _RecordingPass(GraphPass):
    """A pass that records every call (graph_module, config, kwargs) and returns gm unchanged."""

    name = "recording"

    def __init__(self) -> None:
        """Initialize the recording pass with an empty call log."""
        super().__init__()
        self.calls: List[Tuple[Any, Any, dict]] = []

    def run(
        self,
        graph_module: fx.GraphModule,
        parallel_config: ParallelConfig,
        **kwargs: Any,
    ) -> fx.GraphModule:
        """Record the call arguments and return the graph_module unchanged."""
        self.calls.append((graph_module, parallel_config, dict(kwargs)))
        return graph_module


def _simple_graph() -> fx.GraphModule:
    """Return a tiny traced GraphModule (``f(x) = x + 1``) for pass tests."""

    def f(x: torch.Tensor) -> torch.Tensor:
        """Return the input plus one (a tiny traced graph for pass tests)."""
        return x + 1

    return torch.fx.symbolic_trace(f)


class TestPassPipelineBuild(unittest.TestCase):
    """``PassPipeline.build`` picks passes from the config flags."""

    def test_default_config_adds_fsdp_and_overlap(self):
        """Test default config (fsdp+overlap on) adds FSDPPass + AutoOverlapPass."""
        pipeline = PassPipeline(ParallelConfig()).build()
        names = [type(p).__name__ for p in pipeline.passes]
        self.assertIn("DeadCodeEliminationPass", names)
        self.assertIn("CanonicalizeGraphPass", names)
        self.assertIn("FSDPPass", names)
        self.assertIn("AutoOverlapPass", names)

    def test_fsdp_disabled_skips_fsdp_pass(self):
        """Test ``fsdp_enabled=False`` skips FSDPPass but keeps overlap."""
        cfg = ParallelConfig(fsdp_enabled=False, enable_overlap=True)
        pipeline = PassPipeline(cfg).build()
        names = [type(p).__name__ for p in pipeline.passes]
        self.assertNotIn(
            "FSDPPass",
            names,
            (f"FSDPPass should be skipped when fsdp_enabled=False, got passes {names}"),
        )
        self.assertIn("AutoOverlapPass", names)

    def test_overlap_disabled_skips_auto_overlap_pass(self):
        """Test ``enable_overlap=False`` skips AutoOverlapPass but keeps FSDP."""
        cfg = ParallelConfig(fsdp_enabled=True, enable_overlap=False)
        pipeline = PassPipeline(cfg).build()
        names = [type(p).__name__ for p in pipeline.passes]
        self.assertIn("FSDPPass", names)
        self.assertNotIn(
            "AutoOverlapPass",
            names,
            (
                f"AutoOverlapPass should be skipped when enable_overlap=False, "
                f"got passes {names}"
            ),
        )

    def test_basic_optimization_passes_always_present(self):
        """Test DeadCode + Canonicalize are always added first."""
        cfg = ParallelConfig(fsdp_enabled=False, enable_overlap=False)
        pipeline = PassPipeline(cfg).build()
        names = [type(p).__name__ for p in pipeline.passes]
        self.assertEqual(names[0], "DeadCodeEliminationPass")
        self.assertEqual(names[1], "CanonicalizeGraphPass")

    def test_sharding_plan_forwarded_to_fsdp_pass(self):
        """Test the pipeline's sharding_plan is forwarded to FSDPPass at build time."""
        plan = ShardingPlan().fsdp_wrap("foo")
        pipeline = PassPipeline(ParallelConfig(), plan).build()
        fsdp_pass = next(p for p in pipeline.passes if isinstance(p, FSDPPass))
        self.assertIs(
            fsdp_pass._sharding_plan,
            plan,  # pylint: disable=protected-access
            "FSDPPass should receive the pipeline's sharding_plan",
        )


class TestPassPipelineRun(unittest.TestCase):
    """``PassPipeline.run`` uses ``self.config`` and forwards kwargs."""

    def test_run_uses_pipeline_config_not_caller_config(self):
        """Test ``run`` drives passes with the pipeline's config, not a caller-supplied one.

        The previous ``run(gm, parallel_config=None, **kwargs)`` accepted a
        config override that no caller used; the refactor removed it so the
        pipeline is the single source of truth.
        """
        cfg = ParallelConfig(fsdp_enabled=False, enable_overlap=False)
        pipeline = PassPipeline(cfg)
        pipeline.passes = [_RecordingPass()]  # bypass build()
        gm = _simple_graph()
        pipeline.run(gm, model="dummy-model")

        rec_pass = pipeline.passes[0]
        self.assertEqual(len(rec_pass.calls), 1)
        _, used_cfg, kwargs = rec_pass.calls[0]
        self.assertIs(
            used_cfg,
            cfg,
            "run() should pass self.config to each pass, not a caller override",
        )
        self.assertEqual(kwargs.get("model"), "dummy-model")

    def test_run_no_parallel_config_kwarg(self):
        """Test passing ``parallel_config=`` raises (API removed).

        This documents the migration: a caller still passing the old
        ``parallel_config`` kwarg gets a clear failure instead of silent
        double-config behaviour.
        """
        cfg = ParallelConfig(fsdp_enabled=False, enable_overlap=False)
        pipeline = PassPipeline(cfg)
        pipeline.passes = [_RecordingPass()]
        gm = _simple_graph()
        with self.assertRaises(TypeError):
            pipeline.run(gm, parallel_config=ParallelConfig())

    def test_run_auto_fills_sharding_plan(self):
        """Test ``sharding_plan`` is auto-filled from the pipeline when omitted."""
        plan = ShardingPlan().fsdp_wrap("auto")
        cfg = ParallelConfig(fsdp_enabled=False, enable_overlap=False)
        pipeline = PassPipeline(cfg, plan)
        rec_pass = _RecordingPass()
        pipeline.passes = [rec_pass]
        gm = _simple_graph()

        pipeline.run(gm)  # caller does NOT pass sharding_plan

        _, _, kwargs = rec_pass.calls[0]
        self.assertIs(
            kwargs.get("sharding_plan"),
            plan,
            "run() should auto-fill sharding_plan from self.sharding_plan",
        )

    def test_run_does_not_override_explicit_sharding_plan(self):
        """Test an explicitly-passed sharding_plan is NOT clobbered by the pipeline's."""
        own_plan = ShardingPlan().fsdp_wrap("pipeline-plan")
        caller_plan = ShardingPlan().fsdp_wrap("caller-plan")
        cfg = ParallelConfig(fsdp_enabled=False, enable_overlap=False)
        pipeline = PassPipeline(cfg, own_plan)
        rec_pass = _RecordingPass()
        pipeline.passes = [rec_pass]
        gm = _simple_graph()

        pipeline.run(gm, sharding_plan=caller_plan)

        _, _, kwargs = rec_pass.calls[0]
        self.assertIs(
            kwargs.get("sharding_plan"),
            caller_plan,
            "explicit caller sharding_plan should win over the pipeline's own",
        )

    def test_run_returns_transformed_graph(self):
        """Test ``run`` returns the (mutated) graph_module each pass returns."""
        cfg = ParallelConfig(fsdp_enabled=False, enable_overlap=False)
        pipeline = PassPipeline(cfg)
        pipeline.passes = [_RecordingPass()]
        gm_in = _simple_graph()

        gm_out = pipeline.run(gm_in, model="dummy")
        self.assertIs(gm_out, gm_in, "run() should return the graph_module")


class TestPassPipelineFromConfig(unittest.TestCase):
    """``from_config`` builds a ready pipeline."""

    def test_from_config_builds(self):
        """Test ``from_config`` returns a built pipeline (passes non-empty)."""
        pipeline = PassPipeline.from_config(ParallelConfig())
        self.assertIsInstance(pipeline, PassPipeline)
        self.assertGreater(len(pipeline.passes), 0)

    def test_from_config_with_sharding_plan(self):
        """Test ``from_config`` forwards the sharding plan to the pipeline."""
        plan = ShardingPlan().fsdp_wrap_pattern("*")
        pipeline = PassPipeline.from_config(ParallelConfig(), plan)
        self.assertIs(pipeline.sharding_plan, plan)


if __name__ == "__main__":
    unittest.main()
