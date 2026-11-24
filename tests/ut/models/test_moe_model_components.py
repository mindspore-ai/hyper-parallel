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

from functools import partial
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn
from transformers import DeepseekV3Config
from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3MLP

from examples.torch.module_family.config import SharedExpertConfig
from hyper_parallel.components.modules import SharedExpert
from hyper_parallel.models._transformers.auto_model import _load_pretrained_config
from hyper_parallel.models.replacement import (
    ModuleReplacementSpec, apply_module_replacements, compile_module_replacements, module_replacement,
)
from tests.common.mark_utils import arg_mark


class TestMoEModelComponents(unittest.TestCase):
    """Cover shared experts and model configuration independently of router migration."""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_shared_preserves_native_activation_child(self) -> None:
        """Feature: MoE family lifecycle and source arithmetic.
        Description: The Kimi native SiLU module remains registered when projection layout is preserved.
        Expectation: Replacement keeps the activation child and native shared outputs.
        """
        source = DeepseekV3MLP(DeepseekV3Config.from_dict({"hidden_size": 16, "intermediate_size": 24}))
        model = nn.Module()
        model.shared = source
        rules = [ModuleReplacementSpec(match=("shared",),
                                       factory=module_replacement(partial(SharedExpert, preserve_source_layout=True)))]
        model, _ = apply_module_replacements(model, compile_module_replacements(model, rules), weights_mapping=[])
        self.assertIs(model.shared.act_fn, source.act_fn)
        hidden = torch.randn(2, 3, 16)
        torch.testing.assert_close(model.shared(hidden), source(hidden), rtol=0, atol=0)

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

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_shared_reset_uses_megatron_initializers(self) -> None:
        """Feature: Shared expert initialization ownership.
        Description: FC1 uses Megatron init_method and FC2 uses output_layer_init_method.
        Expectation: All three projection tensors match independently generated normal samples exactly.
        """
        config = DeepseekV3Config.from_dict({
            "hidden_size": 16, "intermediate_size": 24, "num_hidden_layers": 8, "initializer_range": .11,
        })
        config.init_method_std = .03
        source = DeepseekV3MLP(config)
        shared = SharedExpert(module=source, module_fqn="shared", context={"num_layers": 2},
                              preserve_source_layout=True, activation_dtype="float32", swiglu_limit=5.)
        projections = (shared.gate_proj, shared.up_proj, shared.down_proj)
        torch.manual_seed(249)
        expected = [torch.empty_like(part.weight).normal_(mean=0., std=std)
                    for part, std in zip(projections, (.03, .03, .015))]
        torch.manual_seed(249)
        shared.reset_parameters()
        for projection, value in zip(projections, expected):
            torch.testing.assert_close(projection.weight, value, rtol=0, atol=0)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_shared_default_initializers_match_megatron(self) -> None:
        """Feature: Default shared-expert initialization.
        Description: Packed and separate projections use normal FC1 and depth-scaled FC2.
        Expectation: FC1 std=.02, FC2 std=.005 at global depth eight, and biases are zero.
        """
        for separate in (False, True):
            with self.subTest(separate=separate):
                shared = SharedExpert(config=SharedExpertConfig(16, 24, bias=True, num_layers=8),
                                      preserve_source_layout=separate)
                projections = ((shared.gate_proj, shared.up_proj, shared.down_proj) if separate
                               else (shared.linear_fc1, shared.linear_fc2))
                deviations = (.02, .02, .005) if separate else (.02, .005)
                torch.manual_seed(317)
                expected = [torch.empty_like(layer.weight).normal_(std=std)
                            for layer, std in zip(projections, deviations)]
                torch.manual_seed(317)
                shared.reset_parameters()
                for projection, value in zip(projections, expected):
                    torch.testing.assert_close(projection.weight, value, rtol=0, atol=0)
                    torch.testing.assert_close(projection.bias, torch.zeros_like(projection.bias), rtol=0, atol=0)
