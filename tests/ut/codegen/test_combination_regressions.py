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
import sys
import types
from types import SimpleNamespace
import unittest
from unittest import mock

from torch import nn

from hyper_parallel.codegen.spec.project import _project_dp_size
from hyper_parallel.distributed._builder.planner import (
    _LEAF_SEGMENT_GUARD,
    ShardingPlanner,
)
from hyper_parallel.distributed.tensor_parallel.param_role import (
    ParameterClassifier,
    ParamRole,
)
from hyper_parallel.models.kimi_k3.adapter.policies.sharding import (
    build_parameter_sharding_rules as kimi_rules,
)
from hyper_parallel.models.qwen3_vl_moe.adapter.policies.sharding import (
    build_parameter_sharding_rules as qwen3_vl_rules,
)


class TestParallelDegreeValidation(unittest.TestCase):
    """Invalid topology degrees fail before codegen projection."""

    @staticmethod
    def _config_class():
        """Import trainer config with its optional Hub dependency stubbed."""
        huggingface_hub = types.ModuleType("huggingface_hub")
        huggingface_hub.snapshot_download = mock.Mock()
        with mock.patch.dict(sys.modules, {"huggingface_hub": huggingface_hub}):
            from hyper_parallel.trainer.config.parallelism import (  # pylint: disable=C0415
                AcceleratorConfig,
            )
        return AcceleratorConfig

    def test_non_positive_and_bool_degrees_rejected(self):
        accelerator_config = self._config_class()
        for name in ("tp_size", "cp_size", "ep_size", "pp_size"):
            for value in (0, -1, True):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    accelerator_config(**{name: value})

    def test_dp_projection_does_not_clamp(self):
        accelerator = SimpleNamespace(
            dp_size=None, tp_size=2, cp_size=1, ep_size=1, pp_size=1,
        )
        with mock.patch.dict(os.environ, {"WORLD_SIZE": "6"}):
            self.assertEqual(_project_dp_size(accelerator), 3)
        with self.assertRaises(ValueError):
            _project_dp_size(SimpleNamespace(dp_size=0))


class TestPlannerCombinationBoundaries(unittest.TestCase):
    """Replicated leftovers and fused leaves retain correct ownership."""

    def test_root_backtrack_scatters_to_each_parameter_owner(self):
        roles = {
            "visual.stem.weight": ParamRole.REPLICATED,
            "other.neck.weight": ParamRole.REPLICATED,
        }
        groups = ShardingPlanner()._group_by_boundary(roles)
        self.assertEqual(groups["visual.stem"], [("visual.stem.weight", ParamRole.REPLICATED)])
        self.assertEqual(groups["other.neck"], [("other.neck.weight", ParamRole.REPLICATED)])

    def test_replicated_fallback_uses_relative_paths_and_yields_to_override(self):
        model = nn.Module()
        model.visual = nn.Linear(4, 4)
        parameters = dict(model.named_parameters())
        group = [
            ("visual.weight", ParamRole.REPLICATED),
            ("visual.bias", ParamRole.BIAS),
        ]
        spec = ShardingPlanner._build_replicated_fallback_spec(
            "visual", group, parameters, {"visual.bias"},
        )
        self.assertEqual(set(spec.params), {"weight"})
        self.assertFalse(spec.is_boundary)
        self.assertTrue(all(value.is_replicate() for value in spec.params["weight"].values()))

    def test_megatron_swiglu_leaves_merge_upward(self):
        self.assertIn("linear_fc1", _LEAF_SEGMENT_GUARD)
        self.assertIn("linear_fc2", _LEAF_SEGMENT_GUARD)
        planner = ShardingPlanner()
        fc1 = [("model.mlp.linear_fc1.weight", ParamRole.FUSED_GATE_UP)]
        fc2 = [("model.mlp.linear_fc2.weight", ParamRole.ROWWISE)]
        self.assertEqual(planner._infer_boundary_type("model.mlp.linear_fc1", fc1), "unknown")
        self.assertEqual(planner._infer_boundary_type("model.mlp.linear_fc2", fc2), "unknown")


class TestModelShardingRules(unittest.TestCase):
    """Model-owned policies cover multimodal and Kimi-specific parameters."""

    def test_qwen3_vl_visual_is_replicated(self):
        self.assertIn(("model.visual.", ParamRole.REPLICATED), qwen3_vl_rules())

    def test_kimi_special_parameters_are_replicated(self):
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

    def test_kimi_sparse_gate_classifies_as_router(self):
        model = nn.Module()
        model.block_sparse_moe = nn.Module()
        model.block_sparse_moe.gate = nn.Linear(4, 4, bias=False)
        roles = ParameterClassifier().classify(model, "kimi_k3")
        self.assertEqual(
            roles["block_sparse_moe.gate.weight"], ParamRole.MOE_GATE,
        )


if __name__ == "__main__":
    unittest.main()
