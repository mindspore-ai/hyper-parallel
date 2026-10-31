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
"""Regression tests for codegen parallel-strategy combination boundaries."""

import os
from types import SimpleNamespace
import unittest
from unittest import mock

from torch import nn

from tests.common.mark_utils import arg_mark

from hyper_parallel.codegen.planner_overlay import (
    CodegenShardingPlanner,
    use_codegen_planner_overlay,
)
from hyper_parallel.codegen.spec.project import _project_dp_size
from hyper_parallel.distributed._builder.planner import _LEAF_SEGMENT_GUARD, ShardingPlanner
from hyper_parallel.distributed.tensor_parallel.param_role import ParamRole
from hyper_parallel.models.kimi_k3.adapter.policies.sharding import (
    build_parameter_sharding_rules as kimi_rules,
)
from hyper_parallel.models._transformers import auto_model
from hyper_parallel.models.qwen3_vl_moe.adapter.policies.sharding import (
    build_parameter_sharding_rules as qwen3_vl_rules,
)


class TestParallelDegreeValidation(unittest.TestCase):
    """Invalid topology degrees fail inside codegen projection."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_windows"], level_mark="level0",
              card_mark="onecard", essential_mark="unessential")
    def test_dp_projection_does_not_clamp(self):
        """Project DP exactly and reject an invalid explicit degree.

        Feature: codegen topology projection
        Description: Project an implicit DP degree and then provide zero explicitly.
        Expectation: The implicit degree is derived while zero raises ``ValueError``.
        """
        accelerator = SimpleNamespace(
            dp_size=None, tp_size=2, cp_size=1, ep_size=1, pp_size=1,
        )
        with mock.patch.dict(os.environ, {"WORLD_SIZE": "6"}):
            self.assertEqual(_project_dp_size(accelerator), 3)
        with self.assertRaises(ValueError):
            _project_dp_size(SimpleNamespace(dp_size=0))


class TestPlannerCombinationBoundaries(unittest.TestCase):
    """Codegen additions remain outside the native planner."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_windows"], level_mark="level0",
              card_mark="onecard", essential_mark="unessential")
    def test_root_backtrack_scatters_to_each_parameter_owner(self):
        """Separate unknown owners only in the codegen overlay.

        Feature: codegen planner isolation
        Description: Group replicated parameters owned by two unknown visual modules.
        Expectation: Codegen preserves both owners without changing native grouping.
        """
        roles = {
            "visual.stem.weight": ParamRole.REPLICATED,
            "visual.neck.weight": ParamRole.REPLICATED,
        }
        native_groups = ShardingPlanner()._group_by_boundary(roles)
        self.assertNotEqual(set(native_groups), {"visual.stem", "visual.neck"})
        groups = CodegenShardingPlanner()._group_by_boundary(roles)
        self.assertEqual(groups["visual.stem"], [("visual.stem.weight", ParamRole.REPLICATED)])
        self.assertEqual(groups["visual.neck"], [("visual.neck.weight", ParamRole.REPLICATED)])

    @arg_mark(plat_marks=["cpu_linux", "cpu_windows"], level_mark="level0",
              card_mark="onecard", essential_mark="unessential")
    def test_replicated_fallback_uses_relative_paths_and_yields_to_override(self):
        """Keep replicated fallback codegen-only and respect overrides.

        Feature: codegen replicated metadata
        Description: Build fallback metadata for a replicated weight and overridden bias.
        Expectation: Only the relative weight path appears in the generated specification.
        """
        model = nn.Module()
        model.visual = nn.Linear(4, 4)
        parameters = dict(model.named_parameters())
        group = [
            ("visual.weight", ParamRole.REPLICATED),
            ("visual.bias", ParamRole.BIAS),
        ]
        self.assertFalse(hasattr(ShardingPlanner, "_build_replicated_fallback_spec"))
        spec = CodegenShardingPlanner._build_replicated_fallback_spec(
            "visual", group, parameters, {"visual.bias"},
        )
        self.assertEqual(set(spec.params), {"weight"})
        self.assertFalse(spec.is_boundary)
        self.assertTrue(all(value.is_replicate() for value in spec.params["weight"].values()))

    @arg_mark(plat_marks=["cpu_linux", "cpu_windows"], level_mark="level0",
              card_mark="onecard", essential_mark="unessential")
    def test_megatron_swiglu_leaves_merge_upward(self):
        """Interpret Megatron SwiGLU leaves only in codegen.

        Feature: codegen boundary discovery
        Description: Inspect linear_fc1 and linear_fc2 boundary inference.
        Expectation: Native guards stay unchanged and the overlay merges both upward.
        """
        self.assertNotIn("linear_fc1", _LEAF_SEGMENT_GUARD)
        self.assertNotIn("linear_fc2", _LEAF_SEGMENT_GUARD)
        planner = CodegenShardingPlanner()
        fc1 = [("model.mlp.linear_fc1.weight", ParamRole.FUSED_GATE_UP)]
        fc2 = [("model.mlp.linear_fc2.weight", ParamRole.ROWWISE)]
        self.assertEqual(planner._infer_boundary_type("model.mlp.linear_fc1", fc1), "unknown")
        self.assertEqual(planner._infer_boundary_type("model.mlp.linear_fc2", fc2), "unknown")

    @arg_mark(plat_marks=["cpu_linux", "cpu_windows"], level_mark="level0",
              card_mark="onecard", essential_mark="unessential")
    def test_overlay_upgrades_and_restores_only_the_codegen_build(self):
        """Scope the planner factory override to codegen.

        Feature: codegen planner isolation
        Description: Create a planner inside the overlay and inspect the factory afterward.
        Expectation: The planner is upgraded temporarily and the original factory is restored.
        """
        native_planner = ShardingPlanner()
        factory = mock.Mock(return_value=(native_planner, None))
        with mock.patch.object(auto_model, "instantiate_infrastructure", factory):
            with use_codegen_planner_overlay():
                planner, _ = auto_model.instantiate_infrastructure()
                self.assertIsInstance(planner, CodegenShardingPlanner)
            self.assertIs(auto_model.instantiate_infrastructure, factory)


class TestModelShardingRules(unittest.TestCase):
    """Model-owned policies cover multimodal and Kimi-specific parameters."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_windows"], level_mark="level0",
              card_mark="onecard", essential_mark="unessential")
    def test_qwen3_vl_visual_is_replicated(self):
        """Keep Qwen3-VL visual replication in its model policy.

        Feature: model-owned sharding policy
        Description: Inspect the Qwen3-VL parameter rules.
        Expectation: The visual module is explicitly classified as replicated.
        """
        self.assertIn(("model.visual.", ParamRole.REPLICATED), qwen3_vl_rules())

    @arg_mark(plat_marks=["cpu_linux", "cpu_windows"], level_mark="level0",
              card_mark="onecard", essential_mark="unessential")
    def test_kimi_special_parameters_are_replicated(self):
        """Cover Kimi special parameters in its model adapter.

        Feature: model-owned sharding policy
        Description: Collect the replicated patterns declared by the Kimi adapter.
        Expectation: Delta-attention special parameters are present in that policy.
        """
        rules = kimi_rules()
        replicated = {
            pattern
            for patterns, role in rules
            if role == ParamRole.REPLICATED
            for pattern in ([patterns] if isinstance(patterns, str) else patterns)
        }
        self.assertTrue({
            "A_log", "dt_bias", "q_a_proj", "output_attn_res_proj",
        }.issubset(replicated))

    @arg_mark(plat_marks=["cpu_linux", "cpu_windows"], level_mark="level0",
              card_mark="onecard", essential_mark="unessential")
    def test_kimi_sparse_gate_classifies_as_router(self):
        """Assign Kimi sparse gates their codegen-only router role.

        Feature: codegen parameter classification
        Description: Classify a minimal block_sparse_moe gate with the overlay planner.
        Expectation: Its weight is classified as ``MOE_GATE``.
        """
        model = nn.Module()
        model.block_sparse_moe = nn.Module()
        model.block_sparse_moe.gate = nn.Linear(4, 4, bias=False)
        roles = CodegenShardingPlanner()._classify_all_params(model, "kimi_k3")
        self.assertEqual(
            roles["block_sparse_moe.gate.weight"], ParamRole.MOE_GATE,
        )


if __name__ == "__main__":
    unittest.main()
