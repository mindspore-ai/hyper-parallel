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
"""Trainer replacement, current-weight and checkpoint contracts on CPU."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from typing import Any

import torch
from torch import nn

from hyper_parallel.components.checkpoint.weight_conversion import revert_weight_conversion
from hyper_parallel.models._transformers.model_builder import _initialize_model_weights
from hyper_parallel.models.deepseek_v41.adapter.megamoe_training import DeepseekV41TrainingExperts
from hyper_parallel.models.deepseek_v41.modeling_deepseek_v41 import DeepseekV41CroppedForCausalLM
from hyper_parallel.models.replacement import apply_module_replacements, compile_module_replacements
from hyper_parallel.trainer.config import entries_to_module_replacements
from hyper_parallel.trainer.config.manager import parse_training_args
from tests.common.mark_utils import arg_mark
from tests.ut.auto_models.models.deepseek_v41.test_deepseek_v41_crop import _tiny_config, _write_engram_assets
from tests.ut.auto_models.models.deepseek_v41.test_megamoe import _make_source


class _WeightProbe(nn.Module):
    """Emulate only the execution boundary, retaining differentiable weights."""

    def __init__(self, **kwargs: Any) -> None:
        """Verify execution has no separately owned expert parameters."""
        super().__init__()
        if kwargs["create_parameters"]:
            raise AssertionError("Trainer must own the only expert parameters")
        self.expert_capacity_factor = kwargs["expert_capacity_factor"]
        self.calls = []

    def forward(self, hidden: torch.Tensor, _indices: torch.Tensor, _weights: torch.Tensor,
                *, expert_weights: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Record the current objects and preserve their autograd connections."""
        self.calls.append(expert_weights)
        return hidden * sum(weight.sum() for weight in expert_weights)


class TestMegaMoeTraining(unittest.TestCase):
    """Exercise boundaries that standalone expert precision cannot cover."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_current_weights_and_nested_module_hooks(self):
        """
        Feature: MegaMoe Trainer parameter ownership.
        Description: Replace parameters between forwards as FSDP unshard can do.
        Expectation: New parameters receive gradients with no duplicate executor weights.
        """
        experts = DeepseekV41TrainingExperts(module=_make_source().experts)
        hidden = torch.ones(128, 8)
        indices, weights = torch.zeros(128, 2, dtype=torch.int64), torch.ones(128, 2)
        hooks = []
        experts.register_forward_pre_hook(lambda *_args: hooks.append(True))
        with patch("hyper_parallel.models.deepseek_v41.adapter.megamoe_training.MegaMoeExperts", _WeightProbe):
            experts.configure(None, 1, 128, "push", 2, 1.5)
            self.assertEqual(experts._kernel.expert_capacity_factor, 1.5)
            experts(hidden, indices, weights).sum().backward()
            old = experts.gate_up_proj
            experts.gate_up_proj = nn.Parameter(torch.ones_like(old))
            experts(hidden, indices, weights).sum().backward()
        self.assertIs(experts._kernel.calls[-1][0], experts.gate_up_proj)
        self.assertIsNotNone(experts.gate_up_proj.grad)
        self.assertEqual(set(dict(experts.named_parameters())), {"gate_up_proj", "down_proj"})
        self.assertEqual(len(hooks), 2)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_recipe_checkpoint_conversion_and_meta_initialization(self):
        """
        Feature: MegaMoe Trainer checkpoint layout.
        Description: Apply the actual recipe and reverse its scoped expert conversions.
        Expectation: HF checkpoint values round-trip exactly and meta weights initialize.
        """
        recipe = parse_training_args([str(Path(__file__).resolve().parents[5]
                                         / "examples/training_demo/train_deepseek_v41_megamoe.yaml")])
        rules = entries_to_module_replacements(recipe.plan_overrides)
        with tempfile.TemporaryDirectory() as directory:
            config = _tiny_config(_write_engram_assets(directory))
            config.swiglu_limit = 0.0
            model = DeepseekV41CroppedForCausalLM(config)
            original = {name: value.clone() for name, value in model.state_dict().items()
                        if ".mlp.experts." in name}
            conversions = []
            apply_module_replacements(model, compile_module_replacements(model, rules), weights_mapping=conversions)
            model._weight_conversions = conversions
            restored = revert_weight_conversion(model, dict(model.state_dict()))
            for name, value in original.items():
                torch.testing.assert_close(restored[name], value, rtol=0, atol=0)
            experts = model.model.layers[0].mlp.experts
            self.assertIsInstance(experts, DeepseekV41TrainingExperts)
            self.assertEqual(tuple(experts.gate_up_proj.shape), (4, 32, 32))
            model.to("meta").to_empty(device="cpu")
            _initialize_model_weights(model)
            for name, parameter in model.named_parameters():
                if ".mlp.experts." in name:
                    self.assertTrue(torch.isfinite(parameter).all())
                    self.assertGreater(parameter.std().item(), 0.01)
                    self.assertLess(parameter.std().item(), 0.03)
