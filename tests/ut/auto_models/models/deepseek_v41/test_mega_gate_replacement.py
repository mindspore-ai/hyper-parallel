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

from hyper_multicore.modules.mega_gate import function as gate_function

from hyper_parallel.models._transformers.model_builder import _initialize_model_weights
from hyper_parallel.models.deepseek_v41.adapter.distributed.mega_gate import (
    DeepseekV41MegaGate,
)
from hyper_parallel.models.deepseek_v41.modeling_deepseek_v41 import (
    DeepseekV41TopKRouter,
    _initialize_v41_owned_module,
)
from hyper_parallel.models.materialization import MaterializationContext, rebuild_materialized_state
from hyper_parallel.models.replacement import (
    ModuleReplacementSpec,
    apply_module_replacements,
    compile_module_replacements,
)
from tests.common.mark_utils import arg_mark


def _config(vision: bool, scoring: str = "sqrtsoftplus", initializer_range: float = 0.02) -> SimpleNamespace:
    return SimpleNamespace(
        hidden_size=128, num_local_experts=16, num_experts_per_tok=6,
        scoring_func=scoring, routed_scaling_factor=1.5, v41_vision_enabled=vision,
        initializer_range=initializer_range, router_aux_loss_coef=0.1, router_bias_update_rate=0.01,
    )


def _replace(router: DeepseekV41TopKRouter) -> DeepseekV41MegaGate:
    model = torch.nn.Module()
    model.gate = router
    plan = compile_module_replacements(model, [ModuleReplacementSpec(
        match=("gate",), factory=DeepseekV41MegaGate,
    )])
    apply_module_replacements(model, plan)
    return model.gate


class TestMegaGateReplacement(unittest.TestCase):
    """Preserve model state and construction semantics across replacement."""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_replacement_preserves_router_state(self) -> None:
        """Feature: MegaGate replacement state adoption.

        Description: Replace text and visual routers through the module replacement pipeline.
        Expectation: Parameters, buffers, training state and RNG remain unchanged without native loading.
        """
        for vision in (False, True):
            with self.subTest(vision=vision):
                router = DeepseekV41TopKRouter(_config(vision)).eval()
                router.weight.requires_grad_(False)
                rng = torch.get_rng_state().clone()
                with patch.object(gate_function, "_native_ops", side_effect=AssertionError("native load")):
                    gate = _replace(router)
                self.assertIs(gate.weight, router.weight)
                self.assertIs(gate.bias, router.bias)
                self.assertIs(gate.bias_vl, router.bias_vl)
                self.assertIs(gate.tokens_per_expert, router.tokens_per_expert)
                self.assertEqual(gate.aux_loss_coeff, router.aux_loss_coeff)
                self.assertEqual(gate.expert_bias_update_rate, router.expert_bias_update_rate)
                self.assertEqual(gate.enable_expert_bias, router.enable_expert_bias)
                self.assertFalse(gate.training)
                self.assertFalse(gate.weight.requires_grad)
                self.assertEqual(set(gate.state_dict()), set(router.state_dict()))
                torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_replacement_supports_meta_initialization(self) -> None:
        """Feature: MegaGate meta initialization.

        Description: Replace and materialize text and visual routers with a configured initialization scale.
        Expectation: Weights use the requested scale, biases are zero and transient statistics are rebuilt.
        """
        std = 0.035
        for vision in (False, True):
            with self.subTest(vision=vision):
                with torch.device("meta"):
                    gate = _replace(DeepseekV41TopKRouter(_config(vision, initializer_range=std)))
                self.assertTrue(all(parameter.is_meta for parameter in gate.parameters()))
                self.assertTrue(getattr(gate, "_hp_reset_after_materialization", False))
                self.assertEqual(gate.initializer_range, std)

                gate.to_empty(device="cpu")
                # Leave sentinel values for the replacement's own reset to overwrite.
                with torch.no_grad(), patch.object(gate, "initialize_weights", return_value=None, create=True):
                    for parameter in gate.parameters():
                        parameter.fill_(float("nan"))
                    gate.tokens_per_expert.fill_(7)
                    _initialize_model_weights(gate)
                rebuild_materialized_state(gate, MaterializationContext(
                    reason="random_init", device=torch.device("cpu"),
                ))

                self.assertTrue(torch.isfinite(gate.weight).all())
                self.assertAlmostEqual(gate.weight.std().item(), std, delta=std * 0.1)
                self.assertEqual(torch.count_nonzero(gate.bias).item(), 0)
                if vision:
                    self.assertEqual(torch.count_nonzero(gate.bias_vl).item(), 0)
                else:
                    self.assertIsNone(gate.bias_vl)
                torch.testing.assert_close(gate.tokens_per_expert, torch.zeros(2, 16, dtype=torch.long))
                self.assertNotIn("tokens_per_expert", gate.state_dict())

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_cpu_fallback_matches_router(self) -> None:
        """Feature: MegaGate CPU fallback parity.

        Description: Compare routing scores, masks, gradients and bias updates against the source router.
        Expectation: All supported score functions preserve outputs, gradients and routing statistics.
        """
        mask = torch.arange(14).reshape(2, 7).remainder(3) == 0
        token_mask = torch.arange(14).reshape(2, 7).remainder(5) != 0
        sequence_ids = torch.tensor([
            [0, 0, 0, 1, 1, 1, 1],
            [2, 2, 2, 2, 3, 3, 3],
        ])
        invalid_mask = torch.zeros(14, dtype=torch.bool)
        for scoring in ("sqrtsoftplus", "softmax", "sigmoid"):
            with self.subTest(scoring=scoring):
                golden = DeepseekV41TopKRouter(_config(vision=True, scoring=scoring))
                with torch.no_grad():
                    _initialize_v41_owned_module(golden, 0.02)
                    golden.bias.uniform_(-0.5, 0.5)
                    golden.bias_vl.uniform_(-0.5, 0.5)
                gate = _replace(deepcopy(golden))
                hidden = torch.randn(2, 7, 128, requires_grad=True)
                reference = hidden.detach().clone().requires_grad_(True)
                expected = golden(
                    reference, image_mask=mask, token_mask=token_mask,
                    sequence_ids=sequence_ids, num_sequences=4,
                )
                actual = gate(
                    hidden, image_mask=mask, token_mask=token_mask,
                    sequence_ids=sequence_ids, num_sequences=4,
                )
                for result, baseline in zip(actual, expected):
                    torch.testing.assert_close(result, baseline, rtol=1e-5, atol=1e-6)
                expected[1].square().sum().backward()
                actual[1].square().sum().backward()
                torch.testing.assert_close(hidden.grad, reference.grad, rtol=1e-5, atol=1e-6)
                torch.testing.assert_close(gate.weight.grad, golden.weight.grad, rtol=1e-5, atol=1e-6)
                torch.testing.assert_close(gate.last_aux_loss, golden.last_aux_loss, rtol=1e-5, atol=1e-6)
                torch.testing.assert_close(gate.tokens_per_expert, golden.tokens_per_expert)
                golden.update_expert_bias()
                gate.update_expert_bias()
                torch.testing.assert_close(gate.bias, golden.bias)
                torch.testing.assert_close(gate.bias_vl, golden.bias_vl)
                torch.testing.assert_close(gate.tokens_per_expert, torch.zeros_like(gate.tokens_per_expert))

        for router in (golden, gate):
            with self.assertRaisesRegex(ValueError, "image_mask must be boolean and match"):
                router(torch.randn(2, 7, 128), image_mask=invalid_mask)

        for router in (golden, gate):
            router.scoring_func = "unsupported"
            with self.assertRaisesRegex(ValueError, "Unsupported V4.1 router scoring function"):
                router(torch.randn(2, 7, 128))
