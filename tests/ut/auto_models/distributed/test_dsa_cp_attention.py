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
"""DSA indexer and teacher reductions must retain all heads under CP."""

import unittest
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import torch

from hyper_parallel.distributed.context_parallel import attention


def _unused(*args, **kwargs):
    """Provide unwrapped adapter symbols for isolated boundary tests."""
    return args, kwargs


class TestDSACPAttention(unittest.TestCase):
    """Validate head-reduced boundaries without real collectives or kernels."""

    def test_indexer_reduces_all_heads_before_topk(self):
        """A per-head subgroup selects a different key from the full indexer."""
        query = torch.tensor([[[[2., 0.], [0., 3.]], [[2., 0.], [0., 3.]]]])
        key = torch.tensor([[[[1., 0.]], [[0., 1.]]]])
        weights = torch.ones(1, 2, 2)

        def _indexer(_module, iq, ik, weight, _q_length, _k_length):
            """Literal full-head weighted ReLU selection reference."""
            scores = (torch.einsum("bqhd,bkd->bqhk", iq, ik.squeeze(2)).relu()
                      * weight.unsqueeze(-1)).sum(2)
            causal = torch.triu(torch.ones(2, 2, dtype=torch.bool), diagonal=1)
            return scores.masked_fill(causal, -torch.inf).argmax(-1), iq, ik, weight

        for rank in (0, 1):
            with self.subTest(rank=rank):
                module = SimpleNamespace(dsa_lightning_indexer_forward=_indexer,
                                         SparseLightningIndexerKLLossTrainFunction=type("KL", (), {"apply": _unused}))
                registry = {"dsa_sparse_attention": _unused}
                context = SimpleNamespace(rank=rank, size=2, group=object())
                local = [value[:, rank:rank + 1] for value in (query, key, weights)]
                full_by_id = {id(shard): value for shard, value in zip(local, (query, key, weights))}

                def _exchange(value, _context):
                    """Model the old head-sharding behavior for a failing regression."""
                    full = query if value.shape[-1] == 2 else weights.unsqueeze(-1)
                    return full[:, :, rank:rank + 1]

                with patch.object(attention, "_gather_sequence", side_effect=lambda value, _: full_by_id[id(value)]), \
                        patch.object(attention, "_sequence_to_head", side_effect=_exchange):
                    attention._dsa_cp_alltoall(module, registry, context)
                    actual = module.dsa_lightning_indexer_forward(None, *local, [2], [2])
                torch.testing.assert_close(actual[0], torch.tensor([[0, 1]]))
                torch.testing.assert_close(actual[1], query)
                torch.testing.assert_close(actual[3], weights)

    def test_kl_teacher_restores_all_heads_without_gradients(self):
        """KL must see the same full-head teacher and softmax stats as CP1."""
        query = torch.randn(1, 4, 1, 3, requires_grad=True)
        key = torch.randn(1, 4, 1, 3)
        query_rope = torch.randn(1, 4, 1, 2, requires_grad=True)
        key_rope = torch.randn(1, 4, 1, 2)
        maximum, total = torch.randn(1, 4, 1), torch.ones(1, 4, 1)
        index_query = torch.randn(4, 2, 3, requires_grad=True)
        index_key = torch.randn(4, 1, 3, requires_grad=True)
        weights = torch.randn(4, 2, requires_grad=True)
        observed = {}
        gathered_requires_grad = []

        class KL:
            """Record the exact teacher arguments consumed by the loss."""

            @staticmethod
            def apply(*args: Any) -> torch.Tensor:
                """Only indexer inputs participate in the auxiliary backward."""
                observed.update(zip(attention._DSA_KL_ARGUMENTS, args))
                return sum(value.sum() for value in args[:3])

        def _gather(value, group):
            """Return distinct head groups while recording teacher detach."""
            del group
            gathered_requires_grad.append(value.requires_grad)
            return value, value + 1

        module = SimpleNamespace(dsa_lightning_indexer_forward=_unused,
                                 SparseLightningIndexerKLLossTrainFunction=KL)
        context = SimpleNamespace(rank=0, size=2, group=object())
        attention._dsa_cp_alltoall(module, {"dsa_sparse_attention": _unused}, context)
        token = attention._dsa_tensor_context.set(attention._DSATensorContext(query, key, query_rope, key_rope))
        try:
            with patch.object(attention, "differentiable_all_gather", side_effect=_gather):
                loss = module.SparseLightningIndexerKLLossTrainFunction.apply(
                    index_query, index_key, weights, query.flatten(0, 1), key.flatten(0, 1),
                    torch.zeros(4, 1, 1, dtype=torch.int32), maximum, total,
                    query_rope.flatten(0, 1), key_rope.flatten(0, 1), [4], [4], 1., 0.1,
                )
            torch.testing.assert_close(observed["query"], torch.cat((query, query + 1), dim=2).flatten(0, 1))
            torch.testing.assert_close(observed["query_rope"],
                                       torch.cat((query_rope, query_rope + 1), dim=2).flatten(0, 1))
            torch.testing.assert_close(observed["softmax_max"], torch.cat((maximum, maximum + 1), dim=-1))
            torch.testing.assert_close(observed["softmax_sum"], torch.cat((total, total + 1), dim=-1))
            self.assertEqual(gathered_requires_grad, [False] * 4)
            self.assertIsNone(attention._dsa_tensor_context.get())
            loss.backward()
            self.assertIsNone(query.grad)
            self.assertIsNone(query_rope.grad)
            for tensor in (index_query, index_key, weights):
                torch.testing.assert_close(tensor.grad, torch.ones_like(tensor))
        finally:
            attention._dsa_tensor_context.reset(token)
