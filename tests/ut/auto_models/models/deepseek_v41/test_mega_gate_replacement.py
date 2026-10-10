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
"""CPU contracts for the DeepSeek V4.1 MegaGate replacement."""

from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from hyper_parallel.core.multicore.modules.mega_gate import function as gate_function
from hyper_parallel.models.deepseek_v41.adapter.distributed.mega_gate import (
    DeepseekV41MegaGate,
)
from hyper_parallel.models.deepseek_v41.modeling_deepseek_v41 import (
    DeepseekV41TopKRouter,
    _initialize_v41_owned_module,
)
from tests.common.mark_utils import arg_mark


def _config(vision: bool, scoring: str = "sqrtsoftplus") -> SimpleNamespace:
    return SimpleNamespace(
        hidden_size=128, num_local_experts=16, num_experts_per_tok=6,
        scoring_func=scoring, routed_scaling_factor=1.5, v41_vision_enabled=vision,
        initializer_range=0.02,
    )


def _replace(router: DeepseekV41TopKRouter) -> DeepseekV41MegaGate:
    return DeepseekV41MegaGate(
        module=router, module_fqn="model.layers.0.mlp.gate", context={},
    )


@arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
          card_mark="allcards", essential_mark="essential")
class TestMegaGateReplacement(unittest.TestCase):
    """Preserve model state and construction semantics across replacement."""

    def test_parameter_identity_and_lazy_construction(self):
        """Preserve parameters, RNG and training state without allocating a native plan."""
        for vision in (False, True):
            for dtype in (torch.float32, torch.bfloat16):
                with self.subTest(vision=vision, dtype=dtype):
                    router = DeepseekV41TopKRouter(_config(vision)).to(dtype=dtype).eval()
                    router.weight.requires_grad_(False)
                    rng = torch.get_rng_state().clone()
                    with patch.object(gate_function, "_native_ops", side_effect=AssertionError("native load")):
                        gate = _replace(router)
                    self.assertIs(gate.weight, router.weight)
                    self.assertIs(gate.bias, router.bias)
                    self.assertIs(gate.bias_vl, router.bias_vl)
                    self.assertFalse(gate.training)
                    self.assertFalse(gate.weight.requires_grad)
                    self.assertEqual(set(gate.state_dict()), set(router.state_dict()))
                    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)

    def test_meta_materialization_and_initialization(self):
        """Handle both initialized and checkpoint-loaded meta routers."""
        for vision in (False, True):
            with self.subTest(vision=vision):
                with torch.device("meta"):
                    gate = _replace(DeepseekV41TopKRouter(_config(vision)))
                self.assertTrue(all(parameter.is_meta for parameter in gate.parameters()))
                gate.to_empty(device="cpu")
                with torch.no_grad():
                    _initialize_v41_owned_module(gate, 0.035)
                self.assertTrue(torch.isfinite(gate.weight).all())
                self.assertGreater(gate.weight.std().item(), 0.02)
                self.assertEqual(torch.count_nonzero(gate.bias).item(), 0)
                if vision:
                    self.assertEqual(torch.count_nonzero(gate.bias_vl).item(), 0)
                source = deepcopy(gate)
                with torch.device("meta"):
                    restored = _replace(DeepseekV41TopKRouter(_config(vision)))
                restored.to_empty(device="cpu")
                restored.load_state_dict(source.state_dict(), strict=True)
                for name, parameter in restored.named_parameters():
                    torch.testing.assert_close(parameter, source.state_dict()[name], rtol=0, atol=0)
                restored(torch.randn(2, 7, 128))[1].square().sum().backward()
                self.assertIsNotNone(restored.weight.grad)
                self.assertIsNone(restored.bias.grad)

    def test_cpu_fallback_matches_actual_router(self):
        """Keep all scoring functions and visual token selection compatible."""
        for vision in (False, True):
            for scoring in ("sqrtsoftplus", "softmax", "sigmoid"):
                with self.subTest(vision=vision, scoring=scoring):
                    golden = DeepseekV41TopKRouter(_config(vision, scoring))
                    with torch.no_grad():
                        _initialize_v41_owned_module(golden, 0.02)
                        golden.bias.uniform_(-0.5, 0.5)
                        if vision:
                            golden.bias_vl.uniform_(-0.5, 0.5)
                    gate = _replace(deepcopy(golden))
                    mask = torch.arange(14).reshape(2, 7).remainder(4) == 0 if vision else None
                    hidden = torch.randn(2, 7, 128, requires_grad=True)
                    reference = hidden.detach().clone().requires_grad_(True)
                    expected = golden(reference, image_mask=mask)
                    actual = gate(hidden, image_mask=mask)
                    for result, baseline in zip(actual, expected):
                        torch.testing.assert_close(result, baseline, rtol=1e-5, atol=1e-6)
                    expected[1].square().sum().backward()
                    actual[1].square().sum().backward()
                    torch.testing.assert_close(hidden.grad, reference.grad, rtol=1e-5, atol=1e-6)
                    torch.testing.assert_close(gate.weight.grad, golden.weight.grad, rtol=1e-5, atol=1e-6)

    def test_cpu_fallback_matches_router_image_mask_contract(self):
        """Use the actual router as the shape-validation baseline with or without visual bias."""
        hidden = torch.randn(2, 7, 128)
        valid_mask = torch.arange(14).reshape(2, 7).remainder(3) == 0
        invalid_mask = torch.zeros(14, dtype=torch.bool)
        for vision in (False, True):
            for scoring in ("sqrtsoftplus", "softmax", "sigmoid"):
                with self.subTest(vision=vision, scoring=scoring):
                    golden = DeepseekV41TopKRouter(_config(vision, scoring))
                    gate = _replace(deepcopy(golden))
                    expected = golden(hidden, image_mask=valid_mask)
                    actual = gate(hidden, image_mask=valid_mask)
                    for result, baseline in zip(actual, expected):
                        torch.testing.assert_close(result, baseline, rtol=1e-5, atol=1e-6)
                    with self.assertRaisesRegex(ValueError, "image_mask must have shape \\[batch, sequence\\]"):
                        golden(hidden, image_mask=invalid_mask)
                    with self.assertRaisesRegex(ValueError, "image_mask must have shape \\[batch, sequence\\]"):
                        gate(hidden, image_mask=invalid_mask)

    def test_native_bias_conversion_uses_current_parameter_values(self):
        """Promote FSDP bias values without modifying or caching the parameters."""
        hidden = torch.randn(1, 7, 128, dtype=torch.bfloat16)
        weight = torch.randn(16, 128, dtype=torch.bfloat16)
        bias = torch.nn.Parameter(torch.randn(16, dtype=torch.bfloat16))
        vision_bias = torch.nn.Parameter(torch.randn(16, dtype=torch.bfloat16))
        mask = torch.zeros(7, dtype=torch.bool)
        plan = SimpleNamespace(vision_mask_placeholder=mask[:1])
        captured = []

        def capture(
            logits: torch.Tensor,
            text: torch.Tensor,
            visual: torch.Tensor,
            *_args: object,
        ) -> tuple[torch.Tensor, ...]:
            """Capture native boundary dtypes and return deterministic outputs."""
            self.assertEqual(logits.dtype, torch.float32)
            self.assertEqual(text.dtype, torch.float32)
            self.assertEqual(visual.dtype, torch.float32)
            self.assertFalse(text.requires_grad)
            captured.append((text.clone(), visual.clone()))
            return torch.empty(7, 6), torch.empty(7, 6, dtype=torch.int64)

        with torch.no_grad(), patch.object(gate_function, "_launch_route", side_effect=capture):
            for _ in range(2):
                gate_function.mega_gate(
                    hidden, weight, bias, plan, vision_bias=vision_bias, image_mask=mask,
                    top_k=6, routed_scaling_factor=1.5,
                )
                torch.testing.assert_close(captured[-1][0], bias.float(), rtol=0, atol=0)
                torch.testing.assert_close(captured[-1][1], vision_bias.float(), rtol=0, atol=0)
                bias.add_(1)
                vision_bias.sub_(1)
        self.assertFalse(torch.equal(captured[0][0], captured[1][0]))
        self.assertEqual(bias.dtype, torch.bfloat16)
