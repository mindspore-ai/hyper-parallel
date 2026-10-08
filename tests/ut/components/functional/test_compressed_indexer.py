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
"""Compare hand-written Indexer gradients with the derivative of the scalar KL."""

import unittest
from unittest.mock import patch

import torch

from hyper_parallel.components.functional import compressed_indexer
from hyper_parallel.components.functional.compressed_indexer import (
    compressed_attention_teacher, gather_selected_keys, gather_selected_keys_fp32, sort_key_indices,
)
from hyper_parallel.components.functional.compressed_cann import _stable_prediction_log
from hyper_parallel.components.modules.shared_compressed_dsa_attention import shared_compressed_indexer_kl_loss
from hyper_parallel.components.modules.shared_compressed_dsa_attention import (
    compressed_causal_topk,
)


def _scalar_reference(query, key, weights, attention_query, main_key, indices, sinks):
    """Independent differentiable student and detached dense selected teacher."""
    batch = torch.arange(query.shape[0]).view(-1, 1, 1)
    valid = indices >= 0
    selected = indices.clamp_min(0).long()
    dots = torch.einsum("bqhd,bqkd->bqhk", query, key[batch, selected]).relu()
    logits = (dots * weights.unsqueeze(-1)).sum(2).masked_fill(~valid, -1e9)
    with torch.no_grad():
        scores = torch.einsum("bhqd,bqkd->bhqk", attention_query, main_key[batch, selected]) * .5
        scores.masked_fill_(~valid.unsqueeze(1), -1e9)
        sink = sinks.view(1, -1, 1, 1).expand(*scores.shape[:-1], 1)
        teacher = torch.cat((scores, sink), -1).softmax(-1)[..., :-1]
        teacher = teacher.masked_fill(~valid.unsqueeze(1), 0).sum(1)
        teacher /= teacher.sum(-1, keepdim=True).clamp_min(torch.finfo(torch.float32).tiny)
    return (teacher * (teacher.clamp_min(torch.finfo(torch.float32).tiny).log()
                       - logits.log_softmax(-1))).sum() * (.7 / (query.shape[0]*query.shape[1]))


class TestCompressedIndexerKL(unittest.TestCase):
    """Cover padding, repeated keys, head weights, autocast and teacher underflow."""

    def test_fused_loss_recovers_log_softmax_when_prediction_underflows(self):
        """A positive teacher on a zero student probability must not be clamped away."""
        query = torch.tensor([[[[1., 1.]], [[1., 1.]]]])
        key = torch.tensor([[[1., 1.], [0., 0.]]])
        weights = torch.tensor([[[1000.], [1.]]])
        indices = torch.tensor([[[0, 1], [0, 1]]])
        logits = torch.tensor([[[2000., 0.], [2., 0.]]])
        prediction = logits.softmax(-1)
        actual = _stable_prediction_log(prediction, query, key, weights, indices, 1)
        torch.testing.assert_close(actual, logits.log_softmax(-1))

    def test_loss_and_gradient_match_scalar_objective(self):
        """Forward precomputation honors the scalar derivative and outer seed."""
        for autocast in (False, True):
            for underflow in (False, True):
                with self.subTest(autocast=autocast, underflow=underflow):
                    torch.manual_seed(1418)
                    inputs = [torch.randn(2, 3, 2, 4)*.2, torch.randn(2, 5, 4)*.2,
                              torch.randn(2, 3, 2)*.1]
                    actual_inputs = [value.clone().requires_grad_() for value in inputs]
                    oracle_inputs = [value.clone().requires_grad_() for value in inputs]
                    aq = (torch.randn(2, 3, 3, 4)*.3).requires_grad_()
                    ak = (torch.randn(2, 5, 4)*.3).requires_grad_()
                    sinks = torch.full((3,), 1000. if underflow else .7, requires_grad=True)
                    indices = torch.tensor([[[0, 0, 3], [-1, -1, -1], [4, -1, 2]],
                                            [[4, 2, -1], [0, 1, 3], [-1, 2, -1]]])
                    expected = _scalar_reference(*oracle_inputs, aq, ak, indices, sinks)
                    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
                        actual = shared_compressed_indexer_kl_loss(
                            *actual_inputs, aq, ak, indices, sinks,
                            attention_scale=.5, loss_coeff=.7, query_chunk_size=2,
                        )
                    seed = torch.tensor(.37)
                    actual.backward(seed)
                    expected.backward(seed)
                    torch.testing.assert_close(actual, expected)
                    for value, reference in zip(actual_inputs, oracle_inputs):
                        torch.testing.assert_close(value.grad, reference.grad)
                    self.assertIsNone(aq.grad)
                    self.assertIsNone(ak.grad)
                    self.assertIsNone(sinks.grad)

    def test_flat_gather_keeps_batch_and_duplicate_gradient(self):
        """Noncontiguous banks retain independent batch offsets and repeated-key SUM."""
        bank = torch.arange(48.).reshape(2, 4, 6).transpose(1, 2).requires_grad_()
        indices = torch.tensor([[[0, 0, 5], [3, 1, 4]], [[5, 2, 5], [4, 0, 2]]])
        actual = gather_selected_keys(bank, indices)
        expected = bank[torch.arange(2).view(-1, 1, 1), indices]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        grad = torch.randn_like(actual)
        left = torch.autograd.grad(actual, bank, grad, retain_graph=True)[0]
        right = torch.autograd.grad(expected, bank, grad)[0]
        torch.testing.assert_close(left, right)


class TestBoundedIndexerTemporaries(unittest.TestCase):
    """Preserve discrete selections and teacher math while bounding large tensors."""

    def test_index_sort_preserves_values_at_float32_integer_boundary(self):
        """Adjacent IDs above 2**24 must never collapse into one float value."""
        for maximum in (2**24, 2**24+1, 2**25):
            with self.subTest(maximum=maximum):
                values = torch.tensor([[maximum, maximum-1, -1, 0, maximum-2]], dtype=torch.int64)
                torch.testing.assert_close(sort_key_indices(values, maximum), values.sort(-1).values,
                                           rtol=0, atol=0)

    def test_gather_cast_chooses_small_bank_and_keeps_large_bank_unconverted(self):
        """The long-bank fallback returns the same FP32 values without a full cast."""
        for keys in (4, 128):
            bank = torch.arange(keys*8).reshape(1, keys, 8).bfloat16()
            indices = torch.tensor([[[0, 1, 1, 2]*8]])
            with patch.object(compressed_indexer, "_FP32_TEMPORARY_BYTES", 512):
                prepared = compressed_indexer._teacher_bank(bank, indices.numel())
                self.assertEqual(prepared.dtype, torch.float32 if keys == 4 else torch.bfloat16)
                actual = gather_selected_keys_fp32(bank, indices)
            torch.testing.assert_close(actual, gather_selected_keys(bank, indices).float(), rtol=0, atol=0)

    def test_streamed_teacher_keeps_per_head_sink_and_final_normalization(self):
        """Force both head and selected-key blocks, including empty and underflow rows."""
        torch.manual_seed(975)
        query = torch.randn(2, 5, 3, 8)
        bank = torch.randn(2, 13, 8).bfloat16()
        indices = torch.tensor([[[0, 0, 4, 12, -1], [-1]*5, [2, 3, 8, -1, -1]],
                                [[1, 9, 3, 2, 0], [4, -1, 5, -1, 6], [2, 2, 3, 5, 9]]])
        valid = indices >= 0
        selected = bank[torch.arange(2).view(-1, 1, 1), indices.clamp_min(0)].float()
        for underflow in (False, True):
            sinks = torch.full((5,), 1000.) if underflow else torch.linspace(-2, 3, 5)
            logits = (torch.einsum("bhqd,bqkd->bhqk", query, selected)*.5).masked_fill(~valid[:, None], -1e9)
            sink = sinks.view(1, -1, 1, 1).expand(2, 5, 3, 1)
            expected = torch.cat((logits, sink), -1).softmax(-1)[..., :-1]
            expected = expected.masked_fill(~valid[:, None], 0).sum(1)
            expected /= expected.sum(-1, keepdim=True).clamp_min(torch.finfo(torch.float32).tiny)
            with patch.object(compressed_indexer, "_FP32_TEMPORARY_BYTES", 256):
                actual = compressed_attention_teacher(query, bank, indices, sinks, .5)
            torch.testing.assert_close(actual, expected)

    def test_discrete_selection_keeps_global_topk_ties_without_autograd(self):
        """Dense scalar-score TopK remains the oracle even for all-equal scores."""
        torch.manual_seed(1435)
        for tied in (False, True):
            query = torch.randn(1, 7, 3, 8).requires_grad_()
            key = torch.randn(1, 19, 8).requires_grad_()
            weights = (torch.zeros(1, 7, 3) if tied else torch.randn(1, 7, 3)).requires_grad_()
            with torch.no_grad():
                scores = torch.matmul(query, key.transpose(1, 2).unsqueeze(1)).relu()
                scores = (scores * weights.unsqueeze(-1)).sum(2)
                visible = (torch.arange(7)+8)//2
                scores.masked_fill_(torch.arange(19).view(1, 1, -1) >= visible.view(1, -1, 1), -float("inf"))
                top = scores.topk(5, sorted=False)
                expected = top.indices.masked_fill(~top.values.isfinite(), 19).sort(-1).values
                expected = expected.masked_fill(expected == 19, -1).int()
            saved = []
            with torch.autograd.graph.saved_tensors_hooks(lambda tensor: saved.append(tensor) or tensor, lambda x: x):
                actual = compressed_causal_topk(
                    query, key, weights, compress_ratio=2, sparse_count=5,
                    query_offset=7,
                )
            self.assertEqual(saved, [])
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
