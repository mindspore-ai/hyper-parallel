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
"""Transport-independent EP execution and expert computation contracts."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from hyper_parallel.distributed.expert_parallel.routing import MOE_ROUTER_ADAPTERS
from hyper_parallel.compile.ep_capture import _compute_swiglu_expert


class TestSwiGLUExpert(unittest.TestCase):
    """The shared primitive preserves layouts, activations, and autograd."""

    def test_fused_and_split_weights_match_forward_and_gradients(self):
        """Weight packing must not change input gradients or projection math."""
        generator = torch.Generator().manual_seed(42)
        states = torch.randn(3, 4, generator=generator, dtype=torch.float64, requires_grad=True)
        gate = torch.randn(2, 6, 4, generator=generator, dtype=torch.float64, requires_grad=True)
        up = torch.randn(2, 6, 4, generator=generator, dtype=torch.float64, requires_grad=True)
        down = torch.randn(2, 4, 6, generator=generator, dtype=torch.float64, requires_grad=True)
        for activation in (torch.nn.functional.silu, torch.nn.functional.gelu):
            with self.subTest(activation=activation.__name__):
                expected = (activation(states @ gate[1].T) * (states @ up[1].T)) @ down[1].T
                expected_grads = torch.autograd.grad(expected.sum(), (states, gate, up, down))
                for fused in (False, True):
                    weights = (torch.cat((gate, up), dim=1), None, down) if fused else (gate, up, down)
                    actual = _compute_swiglu_expert(states, weights, 1, activation)
                    actual_grads = torch.autograd.grad(actual.sum(), (states, gate, up, down))
                    torch.testing.assert_close(actual, expected)
                    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
                        torch.testing.assert_close(actual_grad, expected_grad)

    def test_empty_expert_retains_zero_parameter_gradients(self):
        """Experts receiving no tokens still participate in autograd."""
        states = torch.empty(0, 4, requires_grad=True)
        gate_up = torch.randn(2, 6, 4, requires_grad=True)
        down = torch.randn(2, 4, 3, requires_grad=True)
        output = _compute_swiglu_expert(
            states, (gate_up, None, down), 0, torch.nn.functional.silu
        )
        self.assertEqual(output.shape, (0, 4))
        gradients = torch.autograd.grad(output.sum(), (states, gate_up, down))
        for gradient in gradients:
            torch.testing.assert_close(gradient, torch.zeros_like(gradient))


class TestNativeRouterOutput(unittest.TestCase):
    """Preselected routes retain the model's weighting and normalization."""

    def test_default_logits_router_honors_normalization_switch(self):
        """Logits routing remains the explicitly selected default adapter."""
        logits = torch.tensor([[[1.0, 2.0, -1.0, 0.5]]])
        for normalize in (True, False):
            with self.subTest(normalize=normalize):
                module = SimpleNamespace(
                    gate=Mock(return_value=logits),
                    config=SimpleNamespace(num_experts_per_tok=2, norm_topk_prob=normalize),
                )
                expected_weights, expected_indices = logits.reshape(1, 4).softmax(-1).topk(2, dim=-1)
                if normalize:
                    expected_weights = expected_weights / expected_weights.sum(-1, keepdim=True)
                indices, weights = MOE_ROUTER_ADAPTERS["default"](module, torch.ones(1, 1, 4))
                torch.testing.assert_close(indices, expected_indices)
                torch.testing.assert_close(weights, expected_weights)
                module.gate.assert_called_once()

    def test_native_routes_are_returned_without_reweighting(self):
        """The top-k adapter extracts scores and indices from the router triple."""
        indices = torch.tensor([[2, 0]])
        weights = torch.tensor([[1.5, 0.75]], requires_grad=True)
        for output_type in (tuple, list):
            with self.subTest(output_type=output_type):
                router_output = output_type((torch.zeros(1, 4), weights, indices))
                module = SimpleNamespace(gate=Mock(return_value=router_output))
                actual_indices, actual_weights = MOE_ROUTER_ADAPTERS["qwen3moe"](
                    module, torch.ones(1, 1, 4)
                )
                self.assertIs(actual_indices, indices)
                self.assertIs(actual_weights, weights)
                module.gate.assert_called_once()

    def test_topk_adapter_rejects_logits_and_non_triples(self):
        """Unsupported outputs fail instead of silently selecting another router policy."""
        outputs = (torch.ones(1, 4), (), (torch.ones(1, 2), torch.zeros(1, 2)), (None,) * 4)
        for router_output in outputs:
            with self.subTest(router_output=router_output):
                module = SimpleNamespace(gate=Mock(return_value=router_output))
                with self.assertRaisesRegex(TypeError, "TopKRouter should return"):
                    MOE_ROUTER_ADAPTERS["qwen3moe"](module, torch.ones(1, 1, 4))
                module.gate.assert_called_once()
