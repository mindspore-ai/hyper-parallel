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
"""Training semantics and parameter compatibility of common MoE chunks."""
import copy
import unittest
from unittest.mock import patch

import torch

from hyper_parallel.components.modules.moe import FeedForward, MoE, MoEAuxLossAutoScaler


class TestChunkedMoE(unittest.TestCase):
    """Check complete first-order gradients with real CPU reference operators."""

    def setUp(self) -> None:
        """Fix initialization and preserve process-local auxiliary scaling."""
        torch.manual_seed(17)
        self.old_scale = MoEAuxLossAutoScaler.main_loss_backward_scale
        MoEAuxLossAutoScaler.main_loss_backward_scale = torch.tensor(0.25)

    def tearDown(self) -> None:
        """Restore auxiliary scaling for unrelated tests."""
        MoEAuxLossAutoScaler.main_loss_backward_scale = self.old_scale

    @staticmethod
    def _model(score_func="sigmoid"):
        return MoE(16, 24, 8, top_k=2, score_before_experts=False,
                   shared_expert=FeedForward(16, 12), load_balance_coeff=0.07,
                   router_kwargs={"score_func": score_func, "num_expert_groups": 4,
                                  "num_limited_groups": 2, "route_scale": 1.7})

    def test_full_gradients_statistics_and_original_routes(self):
        """Short tails preserve all gradients and do not rerun discrete selection."""
        for score_func in ("sigmoid", "softmax"):
            with self.subTest(score_func=score_func):
                expected = self._model(score_func)
                actual = copy.deepcopy(expected)
                actual.chunk_size = 7
                expected.experts.w1.requires_grad_(False)
                actual.experts.w1.requires_grad_(False)
                x = torch.randn(2, 11, 16, requires_grad=True)
                ref_x = x.detach().clone().requires_grad_()
                cotangent = torch.randn_like(x)
                ref_output = expected(ref_x)
                with patch.object(actual.router, "forward", wraps=actual.router.forward) as route:
                    output = actual(x)
                    output.backward(cotangent)
                ref_output.backward(cotangent)
                torch.testing.assert_close(output, ref_output, atol=1e-6, rtol=1e-5)
                torch.testing.assert_close(x.grad, ref_x.grad, atol=1e-6, rtol=1e-5)
                torch.testing.assert_close(actual.tokens_per_expert, expected.tokens_per_expert)
                torch.testing.assert_close(actual.last_aux_loss, expected.last_aux_loss)
                self.assertEqual(route.call_count, 4, f"Expected 4 forward route calls, got {route.call_count}")
                for (name, parameter), (_, reference) in zip(actual.named_parameters(), expected.named_parameters()):
                    if parameter.requires_grad:
                        torch.testing.assert_close(parameter.grad, reference.grad, atol=2e-6, rtol=2e-5, msg=name)
                    else:
                        self.assertIsNone(parameter.grad, f"Frozen {name} acquired a gradient")

    def test_parameters_checkpoint_optimizer_and_microbatch_accumulation(self):
        """Enabling chunks preserves parameter identities and an existing optimizer."""
        expected = self._model()
        actual = copy.deepcopy(expected)
        actual.load_state_dict(expected.state_dict(), strict=True)
        identities = {name: id(parameter) for name, parameter in actual.named_parameters()}
        optimizer = torch.optim.AdamW(actual.parameters(), lr=1e-3)
        reference_optimizer = torch.optim.AdamW(expected.parameters(), lr=1e-3)
        actual.chunk_size = 5
        outputs, reference_outputs = [], []
        for _ in range(2):
            x = torch.randn(1, 13, 16)
            outputs.append(actual(x))
            reference_outputs.append(expected(x))
        for output, reference in zip(reversed(outputs), reversed(reference_outputs)):
            output.square().sum().backward()
            reference.square().sum().backward()
        optimizer.step()
        reference_optimizer.step()
        self.assertEqual(set(actual.state_dict()), set(expected.state_dict()))
        for name, parameter in actual.named_parameters():
            self.assertEqual(id(parameter), identities[name], f"Parameter identity changed for {name}")
        for name, value in actual.state_dict().items():
            torch.testing.assert_close(value, expected.state_dict()[name], atol=1e-6, rtol=2e-5, msg=name)
        restored = self._model()
        restored.chunk_size = actual.chunk_size
        restored.load_state_dict(actual.state_dict(), strict=True)
        x = torch.randn(1, 9, 16)
        torch.testing.assert_close(restored(x), actual(x))

    def test_multilayer_and_one_backward_contract(self):
        """Two layers retain complete chain gradients and reject consumed plans."""
        expected = torch.nn.Sequential(self._model(), self._model())
        actual = copy.deepcopy(expected)
        for layer in actual:
            layer.chunk_size = 5
        x = torch.randn(1, 12, 16, requires_grad=True)
        ref_x = x.detach().clone().requires_grad_()
        output, reference = actual(x), expected(ref_x)
        output.sum().backward(retain_graph=True)
        reference.sum().backward()
        torch.testing.assert_close(x.grad, ref_x.grad, atol=2e-6, rtol=2e-5)
        for name, parameter in actual.named_parameters():
            torch.testing.assert_close(parameter.grad, expected.get_parameter(name).grad, atol=2e-6, rtol=2e-5)
        with self.assertRaisesRegex(RuntimeError, "one backward"):
            output.sum().backward()

    def test_invalid_configuration_and_eval(self):
        """Unsupported training modes fail before executing a partial forward."""
        for size in (0, -1, True, 1.5):
            with self.subTest(size=size), self.assertRaisesRegex(ValueError, "chunk_size"):
                MoE(16, 24, 8, score_before_experts=False, chunk_size=size)
        with self.assertRaisesRegex(ValueError, "score_before"):
            MoE(16, 24, 8, chunk_size=4)
        model = self._model()
        model.chunk_size = 4
        with torch.no_grad(), self.assertRaisesRegex(ValueError, "grad mode"):
            model(torch.randn(1, 9, 16))
        model.eval()
        with torch.no_grad():
            output = model(torch.randn(1, 9, 16))
        self.assertEqual(output.shape, (1, 9, 16), f"Unexpected eval shape {output.shape}")
