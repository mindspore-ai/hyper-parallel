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
"""Acceptance of the documented user API without fixture registration calls."""

import importlib
import unittest
from unittest.mock import patch

import torch

from examples.torch.module_family import CausalAttention, DecoderConfig, FamilyDecoder, GroupedExperts
from hyper_parallel.components.modules.family import get_module_family
from hyper_parallel.models import HyperAutoModelForCausalLM
from tests.common.mark_utils import arg_mark


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
class TestPublicModuleFamilyExample(unittest.TestCase):
    """Exercise final public construction, real math and ordinary optimization."""

    def setUp(self) -> None:
        """Keep model initialization from changing other tests' RNG streams."""
        self.addCleanup(torch.set_rng_state, torch.get_rng_state())

    def test_import_and_repeated_public_construction(self) -> None:
        """Feature: User imports install model-owned family declarations.
        Description: Build twice through the public facade without register/prepare calls.
        Expectation: Both models train and every owned weight receives finite gradients.
        """
        self.assertIsNotNone(get_module_family(CausalAttention))
        self.assertIsNotNone(get_module_family(GroupedExperts))
        auto = importlib.import_module("hyper_parallel.models._transformers.auto_model")
        with patch.object(auto, "_current_device", return_value=torch.device("cpu")):
            for _ in range(2):
                model = HyperAutoModelForCausalLM.from_config(
                    DecoderConfig(top_k=4), model_factory=FamilyDecoder, torch_dtype=torch.float32)
                optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, foreach=False)
                inputs = torch.randn(2, 4, 8, requires_grad=True)
                before = {name: value.detach().clone() for name, value in model.named_parameters()}
                outputs = model(inputs)
                self.assertEqual(tuple(outputs.shape), (2, 4, 8))
                outputs.square().mean().backward()
                self.assertTrue(torch.isfinite(inputs.grad).all())
                for param in model.parameters():
                    self.assertIsNotNone(param.grad)
                    self.assertTrue(torch.isfinite(param.grad).all())
                optimizer.step()
                self.assertTrue(any(not torch.equal(value, before[name])
                                    for name, value in model.named_parameters()))

    def test_local_stages_and_causal_mask(self) -> None:
        """Feature: Ordinary members compute real algorithms with identity transport.
        Description: Alter future tokens and compare the unchanged causal prefix.
        Expectation: No collective is entered and the first token output is unchanged.
        """
        model = FamilyDecoder(DecoderConfig())
        inputs = torch.randn(2, 4, 8)
        changed = inputs.clone()
        changed[:, 1:] += 100
        with patch("torch.distributed.all_to_all_single", side_effect=AssertionError("collective")), \
                patch("torch.distributed.all_gather", side_effect=AssertionError("collective")):
            torch.testing.assert_close(model(inputs)[:, :1], model(changed)[:, :1])

    def test_invalid_dimensions_fail_before_construction(self) -> None:
        """Feature: Public model config validates the structural contract.
        Description: Supply invalid head division and expert selection counts.
        Expectation: Invalid structures fail with useful ValueError messages.
        """
        for kwargs in ({"hidden_size": 7}, {"top_k": 5}, {"num_experts": 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                DecoderConfig(**kwargs)
