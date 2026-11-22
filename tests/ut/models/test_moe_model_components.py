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
"""Shared-expert initialization and explicit MoE model configuration contracts."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from hyper_parallel.models._transformers.auto_model import _load_pretrained_config
from tests.common.mark_utils import arg_mark


class TestMoEModelComponents(unittest.TestCase):
    """Cover shared experts and model configuration independently of router migration."""


    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_explicit_moe_config_survives_hf_loading(self) -> None:
        """Feature: MoE family lifecycle and source arithmetic.
        Description: Unknown explicit metadata survives while typed nested HF configs remain intact.
        Expectation: Explicit router fields survive without replacing typed nested configs.
        """
        nested = SimpleNamespace(hidden_size=16)
        config = SimpleNamespace(text_config=nested, hidden_size=16)
        overrides = {"moe_router_force_load_balancing": True, "moe_aux_loss_coeff": .001,
                     "text_config": {"hidden_size": 16}}
        with patch("hyper_parallel.models._transformers.auto_model.get_hf_config", return_value=config) as loader:
            actual = _load_pretrained_config("local-model", "eager", torch.float32, overrides, {})
        self.assertIs(actual.text_config, nested)
        self.assertTrue(actual.moe_router_force_load_balancing)
        self.assertEqual(actual.moe_aux_loss_coeff, .001)
        self.assertEqual(loader.call_args.kwargs["moe_aux_loss_coeff"], .001)
