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
"""Megatron-style fused SwiGLU leaves must not hijack the mlp boundary.

Regression guard for the combination-test finding that a whole-module
``replace_qwen3_swiglu_mlp`` swap (HF ``gate_proj``/``up_proj``/``down_proj``
→ fused ``linear_fc1``/``linear_fc2``) failed the HP-PLAN-004 coverage
check: a colwise-only ``linear_fc1`` leaf was promoted to its own ``mlp``
boundary (parent-path pattern match) and consumed the group, leaving the
rowwise-only ``linear_fc2`` to a template-less boundary.
"""

import types
import unittest
from unittest import mock

from torch import nn

from hyper_parallel.codegen.plan.offline_mesh import build_offline_mesh
from hyper_parallel.codegen.spec.types import ParallelDimsSpec
from hyper_parallel.distributed._builder.planner import (
    _LEAF_SEGMENT_GUARD,
    ShardingPlanner,
    _get_architecture,
)
from hyper_parallel.distributed.tensor_parallel.param_role import (
    ParamRole,
)


class TestLeafSegmentGuard(unittest.TestCase):
    """The guard treats fused SwiGLU leaves as non-boundary containers."""

    def test_linear_fc_segments_guarded(self):
        self.assertIn("linear_fc1", _LEAF_SEGMENT_GUARD)
        self.assertIn("linear_fc2", _LEAF_SEGMENT_GUARD)

    def test_infer_type_of_guarded_leaves_is_unknown(self):
        planner = ShardingPlanner()
        fc1 = [("model.layers.0.mlp.linear_fc1.weight", ParamRole.FUSED_GATE_UP)]
        fc2 = [("model.layers.0.mlp.linear_fc2.weight", ParamRole.ROWWISE)]
        self.assertEqual(planner._infer_boundary_type(
            "model.layers.0.mlp.linear_fc1", fc1), "unknown")
        self.assertEqual(planner._infer_boundary_type(
            "model.layers.0.mlp.linear_fc2", fc2), "unknown")

    def test_merged_mlp_group_infers_mlp(self):
        planner = ShardingPlanner()
        group = [
            ("model.layers.0.mlp.linear_fc1.weight", ParamRole.FUSED_GATE_UP),
            ("model.layers.0.mlp.linear_fc2.weight", ParamRole.ROWWISE),
        ]
        self.assertEqual(
            planner._infer_boundary_type("model.layers.0.mlp", group), "mlp")


class _SwigluMlp(nn.Module):
    """Stand-in for the generic fused SwiGLU replacement module."""

    def __init__(self, hidden=8, ffn=16):
        super().__init__()
        self.linear_fc1 = nn.Linear(hidden, 2 * ffn, bias=False)
        self.linear_fc2 = nn.Linear(ffn, hidden, bias=False)


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = _SwigluMlp()


class _SwigluModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([_Layer(), _Layer()])
        self.config = types.SimpleNamespace(
            architectures=["SwigluModel"], model_type="swiglu_model"
        )


def _fake_adapter(rules):
    return types.SimpleNamespace(
        sharding_rules=lambda: rules, encoder_parallel=None
    )


class TestSwigluPlanCoverage(unittest.TestCase):
    """With the qwen3-style family rules, the replaced MLP plans cleanly."""

    def _plan(self, model, rules):
        planner = ShardingPlanner()
        mesh = build_offline_mesh(ParallelDimsSpec(tp_size=2))
        with mock.patch(
            "hyper_parallel.models.registry.get_model_adapter",
            return_value=_fake_adapter(rules),
        ):
            return planner.plan(
                model, mesh, tp_size=2, cp_size=1, ep_size=1,
                sequence_parallel=False,
            )

    def test_swiglu_mlp_fully_covered(self):
        model = _SwigluModel()
        rules = [
            ("linear_fc1.weight", ParamRole.FUSED_GATE_UP),
            ("linear_fc2.weight", ParamRole.ROWWISE),
        ]
        plan = self._plan(model, rules)
        spec = plan.modules["layers.0.mlp"]
        self.assertIn("linear_fc1.weight", spec.params)
        self.assertIn("linear_fc2.weight", spec.params)

    def test_without_rules_both_params_fail_coverage(self):
        # No family rules: neither fused name classifies, so the F4b
        # coverage check fires — the guard alone is not a classifier.
        model = _SwigluModel()
        planner = ShardingPlanner()
        mesh = build_offline_mesh(ParallelDimsSpec(tp_size=2))
        with mock.patch(
            "hyper_parallel.models.registry.get_model_adapter",
            return_value=_fake_adapter([]),
        ):
            with self.assertRaises(ValueError) as ctx:
                planner.plan(model, mesh, tp_size=2, cp_size=1, ep_size=1,
                             sequence_parallel=False)
        self.assertIn("plan-time coverage check failed", str(ctx.exception))
        self.assertIn("linear_fc1.weight", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
