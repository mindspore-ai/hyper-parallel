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
"""CPU coverage for the CANN DSA sparse-attention autograd adapter."""

import unittest
from unittest.mock import Mock

import torch

from hyper_parallel.components.functional.dsa_sparse_attention import _cann_sparse_attention


class TestSparseAttentionAutograd(unittest.TestCase):
    """Check lazy backward loading and the shared K/V gradient contract."""

    def test_backward_uses_explicit_shared_kv_and_sums_gradients(self):
        """
        Feature: Sparse-attention autograd adapter.
        Description: Run backward with shared K/V passed explicitly to the gradient operator.
        Expectation: Sum separate key and value gradients for the shared input.
        """
        query = torch.randn(2, 1, 3, requires_grad=True)
        key = torch.randn(2, 1, 3, requires_grad=True)
        indices = torch.zeros(2, 1, 1, dtype=torch.int32)
        query_lengths = torch.tensor([2], dtype=torch.int32)
        key_lengths = torch.tensor([2], dtype=torch.int32)
        attention_op = Mock(side_effect=lambda q, *_args, **_kwargs: (
            q.clone(), torch.zeros(1), torch.ones(1),
        ))
        expected_dq = torch.full_like(query, 2)
        expected_dk = torch.full_like(key, 3)
        expected_dv = torch.full_like(key, 5)
        grad_op = Mock(return_value=(expected_dq, expected_dk, expected_dv, None, None, torch.empty(0)))
        grad_loader = Mock(return_value=grad_op)

        output, _, _ = _cann_sparse_attention(
            query, key, indices, 0.5,
            query_rope=None,
            key_rope=None,
            query_lengths=query_lengths,
            key_lengths=key_lengths,
            layout="TND",
            attention_op=attention_op,
            grad_op=grad_loader,
        )

        self.assertIs(attention_op.call_args.kwargs["actual_seq_lengths_query"], query_lengths)
        grad_loader.assert_not_called()
        output.sum().backward()

        grad_loader.assert_called_once_with()
        self.assertIs(grad_op.call_args.args[1], grad_op.call_args.args[2])
        torch.testing.assert_close(query.grad, expected_dq)
        torch.testing.assert_close(key.grad, expected_dk + expected_dv)
