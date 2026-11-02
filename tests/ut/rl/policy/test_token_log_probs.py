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
"""The RL probability kernel preserves arbitrary vector gradients and bounds head work."""

from copy import deepcopy
import unittest

import torch
from torch.utils.checkpoint import DefaultDeviceType

from rl.roles.policy.token_log_probs import chunked_token_log_probs


class TestChunkedTokenLogProbs(unittest.TestCase):
    """Check mathematical invariants independently of any model-family adapter."""

    def setUp(self) -> None:
        """Keep recomputation on CPU and restore the caller's defaults."""
        previous = DefaultDeviceType.get_device_type()
        DefaultDeviceType.set_device_type("cpu")
        self.addCleanup(DefaultDeviceType.set_device_type, previous)

    def test_sparse_long_context_preserves_vector_gradients(self) -> None:
        """Only selected head rows execute, while hidden/head derivatives match dense loss."""
        for dtype in (torch.float32, torch.bfloat16):
            for zero_actions in (False, True):
                with self.subTest(dtype=dtype, zero_actions=zero_actions):
                    hidden = torch.randn(2, 6400, 8, dtype=dtype, requires_grad=True)
                    candidate = hidden.detach().clone().requires_grad_()
                    head = torch.nn.Linear(8, 64, bias=False, dtype=dtype)
                    selected_head = deepcopy(head)
                    labels = torch.randint(0, 64, (2, 6400))
                    mask = torch.zeros_like(labels, dtype=torch.bool)
                    if not zero_actions:
                        mask[:, 10:13] = True
                        mask[:, -5:] = True
                    weights = torch.randn(2, 6400) * mask
                    expected = head(hidden).float().log_softmax(-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
                    calls = []
                    selected_head.register_forward_pre_hook(
                        lambda module, inputs, calls=calls: calls.append(inputs[0].shape[0]),
                    )
                    actual = chunked_token_log_probs(candidate, labels, selected_head, position_mask=mask, chunk_size=3)
                    tolerance = 2e-2 if dtype == torch.bfloat16 else 1e-5
                    torch.testing.assert_close(actual, expected * mask, rtol=tolerance, atol=tolerance)
                    expected_grads = torch.autograd.grad(expected, (hidden, head.weight), grad_outputs=weights)
                    actual_grads = torch.autograd.grad(actual, (candidate, selected_head.weight), grad_outputs=weights)
                    for left, right in zip(actual_grads, expected_grads):
                        torch.testing.assert_close(left, right, rtol=tolerance, atol=tolerance)
                    self.assertLessEqual(max(calls), 3)
                    self.assertLess(sum(calls), 100)

    def test_no_grad_and_full_position_modes(self) -> None:
        """Reference computation can return every position without a backward graph."""
        head = torch.nn.Linear(8, 16)
        hidden = torch.randn(2, 7, 8)
        labels = torch.randint(0, 16, (2, 7))
        with torch.no_grad():
            expected = head(hidden).float().log_softmax(-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
            actual = chunked_token_log_probs(hidden, labels, head, chunk_size=3)
        torch.testing.assert_close(actual, expected)
        self.assertFalse(actual.requires_grad)

    def test_rejects_invalid_alignment(self) -> None:
        """Fail at the boundary instead of silently projecting the wrong token positions."""
        hidden = torch.randn(1, 5, 8)
        labels = torch.zeros(1, 5, dtype=torch.long)
        head = torch.nn.Linear(8, 16)
        with self.assertRaises(ValueError):
            chunked_token_log_probs(hidden, labels[:, :-1], head)
        with self.assertRaises(ValueError):
            chunked_token_log_probs(hidden, labels, head, position_mask=torch.ones(1, 5))
        with self.assertRaises(ValueError):
            chunked_token_log_probs(hidden, labels, head, chunk_size=True)
