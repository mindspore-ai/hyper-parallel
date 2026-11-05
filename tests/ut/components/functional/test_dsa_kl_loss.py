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
"""CPU coverage for the fused CANN sparse indexer KL adapter."""

import sys
import unittest
from types import ModuleType
from unittest.mock import Mock, patch

import torch

from hyper_parallel.components.functional.dsa_kl_loss import dsa_kl_loss


class TestDSAKLLoss(unittest.TestCase):
    """Check fused sparse KL gradient routing."""

    def test_sparse_kl_preserves_mean_coefficient_and_upstream_seed(self):
        """
        Feature: Sparse DSA KL gradient scaling.
        Description: Backpropagate with a loss coefficient and non-unit seed.
        Expectation: Preserve coefficient, seed, and token-mean gradient scaling.
        """
        iq = torch.randn(4, 2, 3, requires_grad=True)
        ik = torch.randn(7, 1, 3, requires_grad=True)
        weights = torch.randn(4, 2, requires_grad=True)
        query = torch.randn(4, 3, 4, requires_grad=True)
        key = torch.randn(7, 1, 4, requires_grad=True)
        qr = torch.randn(4, 3, 2, requires_grad=True)
        kr = torch.randn(7, 1, 2, requires_grad=True)
        gradients = tuple(torch.full_like(x, value) for x, value in zip((iq, ik, weights), (2, 3, 5)))
        backend = ModuleType("torch_npu")
        backend.npu_sparse_lightning_indexer_grad_kl_loss = Mock(
            return_value=(*gradients, torch.tensor([8.0])),
        )
        with patch.dict(sys.modules, {"torch_npu": backend}):
            loss = dsa_kl_loss(
                iq, ik, weights, query, key, torch.zeros(4, 1, 2, dtype=torch.int32),
                torch.zeros(1), torch.ones(1), qr, kr,
                torch.tensor([2, 4]), torch.tensor([3, 7]), 0.5, 0.25,
            )
            (loss * 3).backward()
        torch.testing.assert_close(loss, torch.tensor(0.5))
        for tensor, expected in zip((iq, ik, weights), gradients):
            torch.testing.assert_close(tensor.grad, expected * (0.25 * 3 / 4))
        for tensor in (query, key, qr, kr):
            self.assertIsNone(tensor.grad)
        options = backend.npu_sparse_lightning_indexer_grad_kl_loss.call_args.kwargs
        self.assertEqual(options["actual_seq_qlen"], [2, 4])
        self.assertEqual(options["actual_seq_klen"], [3, 7])
