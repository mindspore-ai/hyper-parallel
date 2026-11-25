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
"""Contract tests for Kimi K3 registration and KDA CP injection."""
# pylint: disable=wrong-import-position

import inspect
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import torch
from torch import nn

from hyper_parallel.components.modules import KimiDeltaAttention
from hyper_parallel.distributed._builder.forward_rewriter import (
    _ForwardRewriteRequest,
)
from hyper_parallel.distributed._builder.planner import ShardingPlanner
from hyper_parallel.components.modules.attention.kimi_delta_attention import cp_kimi_delta_attention as state_cp
from hyper_parallel.distributed.recipe_spec import (
    CP,
    INNER_WRAPPER,
    ModuleShardingSpec,
)
from hyper_parallel.core.dtensor.placement_types import Replicate
from hyper_parallel.distributed.tensor_parallel.param_role import (
    ParamRole,
    ParameterClassifier,
)
from hyper_parallel.models.registry import get_model_adapter
from hyper_parallel.trainer.config import entries_to_plan_overrides
from hyper_parallel.trainer.config.parser import parse_training_args
from tests.ut.auto_models.distributed.conftest import FakeDeviceMesh
from tests.common.mark_utils import arg_mark

_WRAPPER_NAMES = (
    "kimi_delta_attention_ulysses_cp_wrapper",
    "kimi_delta_attention_p2p_cp_wrapper",
)
_MESH_FAMILY = frozenset({"mesh", "tp_mesh", "cp_mesh", "ep_mesh"})








class TestKimiK3AdapterRegistration(unittest.TestCase):
    """Pin model-family discovery and KDA-specific parameter roles."""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_top_level_and_text_specs_share_kda_providers(self):
        """Feature: Kimi model family registration.

        Description: Inspect text and full-model adapter providers.
        Expectation: Both variants expose the same canonical KDA replacement and sharding policies.
        """
        top_spec = get_model_adapter("KimiK3ForConditionalGeneration")
        text_spec = get_model_adapter("KimiLinearForCausalLM")
        self.assertIsNotNone(top_spec)
        self.assertIsNotNone(text_spec)
        self.assertIsNone(top_spec.context_parallel)
        self.assertIsNone(text_spec.context_parallel)
        self.assertIsNotNone(top_spec.replacements)
        self.assertIsNotNone(text_spec.replacements)
        self.assertIsNot(top_spec.replacements, text_spec.replacements)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_kda_specific_parameter_roles_override_generic_rules(self):
        """Feature: Kimi KDA parameter roles.

        Description: Inspect the model-specific role rules before generic matching.
        Expectation: ShortConv, state and low-rank gate parameters receive their explicit roles.
        """
        rules = get_model_adapter("kimi_k3").sharding_rules()
        classifier = ParameterClassifier()
        expected = {
            "self_attn.q_conv1d.weight": ParamRole.COLWISE,
            "self_attn.A_log": ParamRole.REPLICATED,
            "self_attn.dt_bias": ParamRole.REPLICATED,
            "self_attn.f_a_proj.weight": ParamRole.REPLICATED,
            "self_attn.g_a_proj.weight": ParamRole.REPLICATED,
            "self_attn.f_b_proj.weight": ParamRole.COLWISE,
            "self_attn.g_b_proj.weight": ParamRole.COLWISE,
        }
        for name, role in expected.items():
            with self.subTest(parameter=name):
                self.assertIs(classifier.classify_param(name, rules), role)






class TestKDACombinedAdapter(unittest.TestCase):
    """Pin the recipe options for all pairwise and three-way combinations."""



    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_invalid_contracts_rejected_before_transport(self):
        """Feature: KDA state CP validation.

        Description: Pass incompatible shapes, dtypes, rank orders and options.
        Expectation: Invalid calls fail before transport is invoked.
        """
        state = torch.zeros(1, 2, 128, 128)
        protocols = [state_cp.KDAStateAllGather(None, 0, 4),
                     state_cp.KDAStateAllGatherP2P(None, None, 0, (0, 2, 4, 6), 2)]
        with patch.object(state_cp, "dist") as transport:
            for protocol in protocols:
                with self.assertRaises(ValueError):
                    protocol.forward(state, state.double())
                with self.assertRaises(ValueError):
                    protocol.backward(state, (state[..., :64],))
                for invalid in (state[:0], torch.empty_like(state, device="meta")):
                    with self.assertRaises(ValueError):
                        protocol.forward(invalid, invalid)
            transport.all_gather_into_tensor.assert_not_called()
            transport.irecv.assert_not_called()
        with self.assertRaisesRegex(ValueError, "unique"):
            state_cp.KDAStateAllGatherP2P(None, None, 0, (0, 0, 4, 6), 2)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_mesh_split_preserves_world_root_and_noncontiguous_peers(self):
        """Feature: Root-preserving KDA subgroup creation.

        Description: Split an interleaved CP axis and reject a partial root or reordered peers.
        Expectation: Valid siblings share WORLD metadata; invalid meshes fail before group creation.
        """
        mesh = MagicMock()
        mesh.ndim = 1
        mesh.size.return_value = 4
        mesh.rank_list = (0, 2, 4, 6)
        mesh._get_root_mesh.return_value.size.return_value = 8
        with patch.object(state_cp.dist, "get_world_size", return_value=8), \
                patch.object(state_cp.dist, "get_process_group_ranks", return_value=[0, 2, 4, 6]):
            state_cp.split_kda_mesh(mesh, 2, ("state", "ulysses"))
            mesh._unflatten.assert_called_once_with(0, (2, 2), ("state", "ulysses"))
            mesh._unflatten.reset_mock()
            mesh._get_root_mesh.return_value.size.return_value = 4
            with self.assertRaisesRegex(ValueError, "WORLD root"):
                state_cp.split_kda_mesh(mesh, 2, ("state", "ulysses"))
            mesh.rank_list = (2, 0, 4, 6)
            with self.assertRaisesRegex(ValueError, "communication group order"):
                state_cp.split_kda_mesh(mesh, 2, ("state", "ulysses"))
            mesh._unflatten.assert_not_called()


if __name__ == "__main__":
    unittest.main()
