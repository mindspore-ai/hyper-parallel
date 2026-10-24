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
"""Pure-replicable leftovers get all-Replicate specs, scattered per module.

Regression guards for the qwen3_vl_moe vision-tower fix:

* a subtree whose family sharding_rules classify every parameter as
  replicable matches no communication template — it now receives a
  non-boundary all-Replicate spec (no communication inserted) instead of
  failing the HP-PLAN-004 coverage check;
* root-backtracked parameters scatter back to each parameter's own module
  (attributing the whole merged group to the first parameter's module gave
  every other parameter a wrong relative path);
* frozen-only leftovers keep the legacy skip so existing frozen-param plans
  stay byte-identical.
"""

import types
import unittest
from unittest import mock

from torch import nn

from hyper_parallel.codegen.plan.offline_mesh import build_offline_mesh
from hyper_parallel.codegen.spec.types import ParallelDimsSpec
from hyper_parallel.distributed._builder.planner import (
    ShardingPlanner,
    _get_architecture,
)
from hyper_parallel.distributed.tensor_parallel.param_role import ParamRole


class _Tower(nn.Module):
    """Isolated subtree whose names no default naming rule classifies."""

    def __init__(self, hidden=8):
        super().__init__()
        self.stem = nn.Linear(hidden, hidden)
        self.neck = nn.Linear(hidden, hidden)


class _DenseMlp(nn.Module):
    def __init__(self, hidden=8, ffn=16):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, ffn, bias=False)
        self.down_proj = nn.Linear(ffn, hidden, bias=False)


class _TinyVLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = _Tower()
        self.mlp = _DenseMlp()
        self.config = types.SimpleNamespace(
            architectures=["TinyVLM"], model_type="tiny_vl"
        )


def _fake_adapter(rules):
    return types.SimpleNamespace(
        sharding_rules=lambda: rules, encoder_parallel=None
    )


class TestReplicatedLeftoverCoverage(unittest.TestCase):
    """Visual-tower-style subtrees pass the coverage check via fallback specs."""

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

    def test_trainable_leftover_gets_replicate_spec(self):
        model = _TinyVLM()
        rules = [("visual.", ParamRole.REPLICATED)]
        plan = self._plan(model, rules)
        for module_fqn in ("visual.stem", "visual.neck"):
            spec = plan.modules.get(module_fqn)
            self.assertIsNotNone(spec, f"{module_fqn} missing from plan")
            self.assertFalse(spec.is_boundary)
            for placement in spec.params.values():
                self.assertTrue(
                    all(p.is_replicated() for p in placement.values()),
                    f"{module_fqn} params must be all-Replicate",
                )

    def test_no_coverage_error_for_replicated_subtree(self):
        model = _TinyVLM()
        rules = [("visual.", ParamRole.REPLICATED)]
        plan = self._plan(model, rules)
        covered = ShardingPlanner._declared_parameter_names(plan)
        for name, _ in model.named_parameters():
            self.assertIn(name, covered)

    def test_frozen_leftover_keeps_legacy_skip(self):
        model = _TinyVLM()
        for parameter in model.visual.parameters():
            parameter.requires_grad_(False)
        rules = [("visual.", ParamRole.REPLICATED)]
        plan = self._plan(model, rules)
        self.assertNotIn("visual.stem", plan.modules)
        self.assertNotIn("visual.neck", plan.modules)


class TestRootBacktrackScatter(unittest.TestCase):
    """Root-backtracked params land at their own modules, not the first's."""

    def test_scatter_per_parameter(self):
        planner = ShardingPlanner()
        roles = {
            "visual.stem.weight": ParamRole.REPLICATED,
            "other.neck.weight": ParamRole.REPLICATED,
            "mlp.gate_proj.weight": ParamRole.COLWISE,
            "mlp.down_proj.weight": ParamRole.ROWWISE,
        }
        groups = planner._group_by_boundary(roles)
        self.assertIn("visual.stem", groups)
        self.assertIn("other.neck", groups)
        self.assertEqual(
            {f for f, _ in groups["visual.stem"]}, {"visual.stem.weight"})
        self.assertEqual(
            {f for f, _ in groups["other.neck"]}, {"other.neck.weight"})
        # the dense part still groups into its mlp boundary
        self.assertIn("mlp", groups)

    def test_architecture_detection_ignores_task_suffix(self):
        model = _TinyVLM()
        self.assertEqual(_get_architecture(model), "tinyvlm")


if __name__ == "__main__":
    unittest.main()
