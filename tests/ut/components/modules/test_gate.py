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
"""Gate base contract and local/Mega variants against an actual HF router."""
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from transformers.models.qwen2_moe.modeling_qwen2_moe import Qwen2MoeTopKRouter
from hyper_multicore.modules.mega_gate import MegaGate as SourceMegaGate

from examples.torch.module_family.config import GateConfig
from hyper_parallel.components.modules.gate_family import Gate, GateModule
from hyper_parallel.components.modules.module_family import get_family_build_info
from hyper_parallel.components.modules.gate_family import MegaGateModule


class TestGate(unittest.TestCase):
    """Exercise a shared base without swallowing model auxiliary state."""

    def test_local_hf_fp32_outputs_and_gradients(self):
        """HF softmax gate agrees with the declared FP32 local gate contract."""
        config = GateConfig(8, 6, 2, scoring_func="softmax", sorted_indices=True)
        member = Gate(config=config)
        source = Qwen2MoeTopKRouter(SimpleNamespace(
            hidden_size=8, num_experts=6, num_experts_per_tok=2, norm_topk_prob=True))
        source.weight.data.copy_(member.weight)
        left = torch.randn(2, 3, 8, requires_grad=True)
        right = left.detach().clone().requires_grad_()
        actual, expected = member(left), source(right)
        for value, reference in zip(actual, expected):
            torch.testing.assert_close(value, reference)
        (actual[0].square().sum() + actual[1].square().sum()).backward()
        (expected[0].square().sum() + expected[1].square().sum()).backward()
        torch.testing.assert_close(left.grad, right.grad, atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(member.weight.grad, source.weight.grad, atol=2e-6, rtol=2e-5)
        self.assertIs(member.__class__, GateModule)
        self.assertEqual(get_family_build_info(member).resolved.variant, "local")

    def test_mega_uses_base_and_same_local_math(self):
        """MegaGate is a Gate variant, preserving the base output protocol and scoring."""
        config = GateConfig(8, 6, 2, scoring_func="sigmoid")
        local = Gate(config=config)
        mega = Gate(config=config, implementation="mega")
        self.assertIsInstance(mega, GateModule)
        self.assertEqual(MegaGateModule.__bases__, (GateModule,))
        self.assertNotIsInstance(mega, SourceMegaGate)
        self.assertFalse(getattr(type(mega), "_hp_module_replacement", False))
        mega.weight.data.copy_(local.weight)
        values = torch.randn(2, 3, 8)
        for value, reference in zip(mega(values), local(values)):
            torch.testing.assert_close(value, reference, atol=0, rtol=0)
        self.assertEqual(get_family_build_info(mega).resolved.variant, "mega")
        with self.assertRaises(ValueError):
            Gate(config=replace(config, normalize_topk=False), implementation="mega")

    def test_base_owns_projection_state_and_initialization(self):
        """The standalone Gate base constructs and resets the common state used by both variants."""
        config = GateConfig(8, 6, 2, scoring_func="sigmoid")
        torch.manual_seed(12)
        base = GateModule(config=config)
        torch.manual_seed(12)
        mega = MegaGateModule(config=config)
        torch.testing.assert_close(base.weight, mega.weight, atol=0, rtol=0)
        torch.manual_seed(34)
        base.reset_parameters()
        torch.manual_seed(34)
        mega.reset_parameters()
        torch.testing.assert_close(base.weight, mega.weight, atol=0, rtol=0)
        self.assertEqual(tuple(base.state_dict()), ("weight",))
        self.assertEqual(tuple(mega.state_dict()), ("weight", "bias"))
        values = torch.randn(2, 3, 8)
        for actual, expected in zip(mega(values), base(values)):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_mega_preserves_fp32_bias_values_and_gradients(self):
        """Dtype conversion changes the projection while retaining full-precision correction state."""
        member = MegaGateModule(config=GateConfig(8, 6, 2, vision_enabled=True))
        with torch.no_grad():
            member.bias.copy_(torch.linspace(-0.12345, 0.98765, 6))
            member.bias_vl.copy_(-member.bias)
        member.bias.grad = torch.linspace(0.12345, 0.98765, 6)
        before = {name: member.get_parameter(name).detach().clone() for name in ("bias", "bias_vl")}
        gradient = member.bias.grad.clone()
        member.to(dtype=torch.bfloat16)
        self.assertEqual(member.weight.dtype, torch.bfloat16)
        for name, value in before.items():
            self.assertEqual(member.get_parameter(name).dtype, torch.float32)
            torch.testing.assert_close(member.get_parameter(name), value, atol=0, rtol=0)
        torch.testing.assert_close(member.bias.grad, gradient, atol=0, rtol=0)

    def test_native_bridge_receives_gate_owned_state(self):
        """The optional fused function receives the same Parameters without a nested router module."""
        member = MegaGateModule(config=GateConfig(8, 6, 2, vision_enabled=True))
        values = torch.randn(2, 3, 8)
        mask = torch.tensor([[True, False, True], [False, True, False]])
        expected = member(values, mask)
        with patch("hyper_parallel.components.modules.gate_family._plan_for_device") as plan:
            with patch("hyper_multicore.modules.mega_gate.function.mega_gate", return_value=expected) as native:
                actual = member._native_forward(values, mask)  # pylint: disable=protected-access
        self.assertIs(actual, expected)
        args, options = native.call_args
        self.assertIs(args[0], values)
        self.assertIs(args[1], member.weight)
        self.assertIs(args[2], member.bias)
        self.assertIs(args[3], plan.return_value)
        self.assertIs(options["vision_bias"], member.bias_vl)
        torch.testing.assert_close(options["image_mask"], mask.flatten())
        self.assertEqual(options["top_k"], member.top_k)
        self.assertEqual(options["routed_scaling_factor"], member.routed_scaling_factor)
        self.assertEqual(tuple(member.children()), ())

    def test_mega_rejects_invalid_visual_mask(self):
        """Visual correction requires the declared modality state and a matching boolean mask."""
        values = torch.randn(2, 3, 8)
        plain = MegaGateModule(config=GateConfig(8, 6, 2))
        with self.assertRaises(ValueError):
            plain(values, torch.ones(2, 3, dtype=torch.bool))
        visual = MegaGateModule(config=GateConfig(8, 6, 2, vision_enabled=True))
        for mask in (torch.ones(2, 3), torch.ones(6, dtype=torch.bool)):
            with self.subTest(shape=mask.shape, dtype=mask.dtype), self.assertRaises(ValueError):
                visual(values, mask)
