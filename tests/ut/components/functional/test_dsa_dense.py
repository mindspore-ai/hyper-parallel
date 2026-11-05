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
"""CPU coverage for the fused CANN dense DSA autograd contract."""

import sys
import unittest
from types import ModuleType
from unittest.mock import Mock, patch

import torch

from hyper_parallel.components.functional.dsa_dense import dsa_dense_indexer_kl_loss, dsa_dense_indexer_lse


class TestDenseDSA(unittest.TestCase):
    """Check native max/sum and token-mean gradient scaling."""

    def test_lse_returns_native_max_sum_without_autograd(self):
        """
        Feature: Dense DSA softmax statistics.
        Description: Call the native LSE operator with packed sequence metadata.
        Expectation: Return the native maximum and sum without autograd wrapping.
        """
        query = torch.randn(4, 2, 3, requires_grad=True)
        key = torch.randn(7, 1, 3)
        weights = torch.randn(4, 2)
        maximum, total = torch.zeros(1, 4), torch.ones(1, 4)
        backend = ModuleType("torch_npu")
        backend.npu_dense_lightning_indexer_softmax_lse = Mock(return_value=(maximum, total))
        with patch.dict(sys.modules, {"torch_npu": backend}):
            actual = dsa_dense_indexer_lse(
                query, key, weights, query_lengths=torch.tensor([2, 4]), key_lengths=torch.tensor([3, 7]),
            )
        self.assertEqual(actual, (maximum, total))
        options = backend.npu_dense_lightning_indexer_softmax_lse.call_args.kwargs
        self.assertEqual(options["actual_seq_qlen"], [2, 4])
        self.assertEqual(options["actual_seq_klen"], [3, 7])

    def test_fused_dense_kl_scales_only_indexer_gradients(self):
        """
        Feature: Fused dense DSA KL gradients.
        Description: Backpropagate through the fused adapter with a non-unit seed.
        Expectation: Scale only indexer gradients by coefficient and token mean.
        """
        iq = torch.randn(4, 2, 3, requires_grad=True)
        ik = torch.randn(7, 1, 3, requires_grad=True)
        weights = torch.randn(4, 2, requires_grad=True)
        query = torch.randn(4, 3, 4, requires_grad=True)
        key = torch.randn(7, 3, 4, requires_grad=True)
        stats = (torch.zeros(1), torch.ones(1))
        native_grads = tuple(torch.full_like(x, value) for x, value in zip((iq, ik, weights), (2, 3, 5)))
        backend = ModuleType("torch_npu")
        backend.npu_dense_lightning_indexer_grad_kl_loss = Mock(
            return_value=(*native_grads, torch.tensor([8.0])),
        )
        with patch.dict(sys.modules, {"torch_npu": backend}):
            loss = dsa_dense_indexer_kl_loss(
                iq, ik, weights, query, key, *stats, stats, scale=0.5,
                query_lengths=torch.tensor([2, 4]), key_lengths=torch.tensor([3, 7]),
                loss_coefficient=0.25,
            )
            (loss * 3).backward()

        torch.testing.assert_close(loss, torch.tensor(0.5))
        for tensor, expected in zip((iq, ik, weights), native_grads):
            torch.testing.assert_close(tensor.grad, expected * (0.25 * 3 / 4))
        self.assertIsNone(query.grad)
        self.assertIsNone(key.grad)
