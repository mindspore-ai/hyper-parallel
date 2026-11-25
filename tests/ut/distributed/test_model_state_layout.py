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
"""Explicit layouts for architecture-owned bare state without a compute-boundary wrapper."""
import unittest
from types import SimpleNamespace

import torch
from torch import nn

from hyper_parallel.core.dtensor.placement_types import Replicate
from hyper_parallel.distributed import ShardingPlanner
from hyper_parallel.distributed.mesh import LogicalDeviceMesh
from hyper_parallel.distributed.tensor_parallel.param_role import ParamRole
from hyper_parallel.models.adapter_spec import ModelAdapterSpec


class TestModelStateLayout(unittest.TestCase):
    """Preserve explicit replication while keeping unidentified parameters fail-closed."""

    @staticmethod
    def _model() -> nn.Module:
        model = nn.Module()
        model.config = SimpleNamespace(model_type="state_layout_probe")
        model.anchor = nn.Parameter(torch.ones(3))
        model.norm = nn.LayerNorm(3)
        model._hp_model_adapter_spec = ModelAdapterSpec(
            architecture="StateLayoutProbe", model_type="state_layout_probe",
            sharding_rules=lambda: [("anchor", ParamRole.REPLICATED)],
        )
        return model

    def test_explicit_bare_state_replication_has_no_forward_boundary(self) -> None:
        """A declared bare vector receives a layout without inventing an attention/MLP protocol."""
        model = self._model()
        plan = ShardingPlanner().plan(model, LogicalDeviceMesh(("tp", "cp"), (2, 2)),
                                      tp_size=2, cp_size=2, sequence_parallel=False)
        owners = [spec for spec in plan.modules.values() if "anchor" in (spec.params or {})]
        self.assertEqual(len(owners), 1)
        self.assertFalse(owners[0].is_boundary)
        self.assertTrue(all(value == Replicate() for value in owners[0].params["anchor"].values()))

    def test_other_bare_state_remains_uncovered(self) -> None:
        """A known replicated vector never authorizes unidentified sibling parameters."""
        model = self._model()
        model.unknown_state = nn.Parameter(torch.ones(3))
        with self.assertRaisesRegex(ValueError, "not covered"):
            ShardingPlanner().plan(model, LogicalDeviceMesh(("dp",), (1,)))
