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
"""Vision subtree isolation in an E0-shaped sharding plan."""

from types import SimpleNamespace
import unittest

from torch import nn

from hyper_parallel.core.dtensor.device_mesh import _DEVICE_MESH_MAP, init_device_mesh
from hyper_parallel.core.dtensor.placement_types import Replicate
from hyper_parallel.distributed._builder.forward_rewriter import _descendant_boundary_fqns
from hyper_parallel.distributed._builder.planner import ShardingPlanner
from hyper_parallel.distributed.recipe_spec import TP


class _Model(nn.Module):
    """Small family-identifiable visual model for planner tests."""

    def __init__(self) -> None:
        """Build a small family-identifiable visual model."""
        super().__init__()
        self.config = SimpleNamespace(model_type="deepseek_v41", architectures=["DeepseekV41ForCausalLM"])
        self.model = nn.Module()
        self.model.vision = nn.Module()
        self.model.vision.blocks = nn.ModuleList([nn.Module()])
        self.model.vision.blocks[0].attn = nn.Linear(4, 4)
        self.model.aligner = nn.Linear(4, 4)
        self.lm_head = nn.Linear(4, 4, bias=False)


class TestEncoderPlanner(unittest.TestCase):
    """Verify source-only specs preserve placement and terminal semantics."""

    def tearDown(self) -> None:
        """Evict backend-less cached sub-meshes so later suites rebuild real groups."""
        _DEVICE_MESH_MAP.clear()

    def test_vision_source_only_specs_do_not_become_boundaries(self):
        """Feature: isolated visual source placement in the planner.

        Description: Derive an E0-shaped plan with ViT and aligner modules.
        Expectation: Visual weights replicate on TP without new boundaries.
        """
        model = _Model()
        mesh = init_device_mesh("cpu", (1, 2, 4), mesh_dim_names=("dp", "cp", "tp"),
                                rank_list=tuple(range(8)), init_backend=False)
        plan = ShardingPlanner().plan(model, mesh, tp_size=4, cp_size=2, ep_size=1,
                                      sequence_parallel=True)
        self.assertFalse(plan.modules["model.vision"].is_boundary)
        self.assertFalse(plan.modules["model.aligner"].is_boundary)
        self.assertTrue(plan.modules["lm_head"]._is_terminal)  # pylint: disable=protected-access
        self.assertEqual(_descendant_boundary_fqns(plan, "model"), [])
        self.assertEqual(set(plan.modules["model.vision"].params),
                         {"blocks.0.attn.weight", "blocks.0.attn.bias"})
        for source in (plan.modules["model.vision"], plan.modules["model.aligner"]):
            self.assertTrue(all(isinstance(placement[TP], Replicate)
                                for placement in source.params.values()))
