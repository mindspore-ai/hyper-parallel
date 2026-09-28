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
from types import SimpleNamespace
from unittest.mock import patch

import torch

from hyper_parallel.components.functional import compressed_attention_utils, compressed_smla
from hyper_parallel.components.functional.compressed_attention_utils import (
    compressed_attention_teacher, gather_selected_keys, gather_selected_keys_fp32, sort_key_indices, _teacher_bank,
)
from hyper_parallel.components.functional.compressed_indexer_ops import (
    _stable_prediction_log, split_compressed_segments,
)
from hyper_parallel.components.functional.compressed_attention_utils import causal_attention_teacher
from hyper_parallel.components.functional.compressed_smla import _geometry, _SparseMla
from hyper_parallel.components.modules.shared_compressed_dsa_attention import SharedCompressedPackedSequence
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

    def test_attention_cache_supports_inference_then_training(self):
        """A validation forward must not poison residuals saved by later training.

        Feature: Attention metadata cache.
        Description: A validation forward must not poison residuals saved by later training.
        Expectation: Training gradients remain valid after inference populated the same cache.
        """
        metadata_modes = []

        def _metadata(*_args, **_kwargs):
            metadata_modes.append(torch.is_inference_mode_enabled())
            return torch.zeros(1, dtype=torch.int32)

        def _forward(query, **_kwargs):
            return query.clone(), torch.zeros(query.shape[:-1])

        def _backward(query, grad, *_args, **kwargs):
            del query
            return (grad, torch.zeros_like(kwargs["ori_kv"]), torch.zeros_like(kwargs["cmp_kv"]),
                    torch.zeros_like(kwargs["sinks"]), None, None)

        operators = SimpleNamespace(sparse_flash_mla_metadata=_metadata, sparse_flash_mla=_forward,
                                    sparse_flash_mla_grad=_backward)
        _geometry.cache_clear()
        self.addCleanup(_geometry.cache_clear)
        with patch.object(compressed_smla, "_operators", return_value=operators):
            for ratio in (1, 2):
                query = torch.ones(1, 2, 1, 3, requires_grad=True)
                raw = torch.ones_like(query)
                main = torch.ones(1, 1, 1, 3)
                indices = torch.zeros(1, 2, 1, 1, dtype=torch.int32)
                sinks = torch.zeros(1)
                with torch.inference_mode():
                    _SparseMla.apply(query, raw, main, indices, sinks, 1., ratio, 0, 128)
                actual = _SparseMla.apply(query, raw, main, indices, sinks, 1., ratio, 0, 128)
                actual.sum().backward()
                torch.testing.assert_close(query.grad, torch.ones_like(query))
        self.assertEqual(metadata_modes, [False, False])

    def test_fused_loss_recovers_log_softmax_when_prediction_underflows(self):
        """A positive teacher on a zero student probability must not be clamped away.

        Feature: Scalar KL recovery.
        Description: A positive teacher on a zero student probability must not be clamped away.
        Expectation: Recovered values match stable log-softmax without probability clamping.
        """
        query = torch.tensor([[[[1., 1.]], [[1., 1.]]]])
        key = torch.tensor([[[1., 1.], [0., 0.]]])
        weights = torch.tensor([[[1000.], [1.]]])
        indices = torch.tensor([[[0, 1], [0, 1]]])
        logits = torch.tensor([[[2000., 0.], [2., 0.]]])
        prediction = logits.softmax(-1)
        actual = _stable_prediction_log(prediction, query, key, weights, indices, 1)
        torch.testing.assert_close(actual, logits.log_softmax(-1))

    def test_underflow_recovery_ignores_invalid_slots_and_empty_rows(self):
        """Only valid zero probabilities need stable recomputation.

        Feature: Scalar KL padding.
        Description: Only valid zero probabilities need stable recomputation.
        Expectation: Only valid zero probabilities trigger recovery; padding contributes zero.
        """
        query = torch.ones(1, 3, 1, 2)
        key = torch.tensor([[[1., 1.], [0., 0.]]])
        weights = torch.tensor([[[1000.], [1.], [1.]]])
        indices = torch.tensor([[[0, 1, -1], [0, -1, -1], [-1, -1, -1]]])
        prediction = torch.tensor([[[1., 0., 0.], [1., 0., 0.], [0., 0., 0.]]])
        actual = _stable_prediction_log(prediction, query, key, weights, indices, 1)
        torch.testing.assert_close(actual, torch.tensor([[[0., -2000., 0.], [0., 0., 0.], [0., 0., 0.]]]))

    def test_prefix_teacher_matches_selected_teacher(self):
        """Cover partial/full transition, a zero row, and odd CP offsets.

        Feature: Causal KL teacher.
        Description: Cover partial/full transition, a zero row, and odd CP offsets.
        Expectation: Prefix and selected-key teacher implementations agree.
        """
        torch.manual_seed(975)
        for ratio, offset, length in ((1, 0, 19), (2, 0, 33), (2, 7, 19)):
            width = 8
            key_length = (offset+length)//ratio
            query = torch.randn(1, 3, length, 8)
            key = torch.randn(1, key_length, 8)
            sinks = torch.tensor([-2., 1., 4.])
            indices = torch.full((1, length, width), -1, dtype=torch.int64)
            for row in range(length):
                selected = torch.randperm((offset+row+1)//ratio)[:width].sort().values
                indices[0, row, :selected.numel()] = selected
            expected = compressed_attention_teacher(query, key, indices, sinks, .5)
            actual = causal_attention_teacher(query, key, indices, sinks, .5, ratio, (offset+length)%ratio, 4)
            torch.testing.assert_close(actual, expected)

    def test_segment_splitting_preserves_causal_coordinates(self):
        """A short native call must not inherit its parent's future K prefix.

        Feature: Compressed causal geometry.
        Description: A short native call must not inherit its parent's future K prefix.
        Expectation: Every subcall keeps the correct K prefix and residual.
        """
        segments = ((0, 9, 6, 14, 1), (9, 17, 16, 20, 0))
        parts = split_compressed_segments(segments, 2, maximum=4)
        self.assertEqual(parts, ((0, 4, 6, 12, 0), (4, 8, 6, 14, 0), (8, 9, 6, 14, 1),
                                 (9, 13, 16, 18, 0), (13, 17, 16, 20, 0)))

    def test_teacher_bank_reuse_respects_memory_limit_and_probability(self):
        """Reuse a small bank across Q chunks, but never cast an oversized bank.

        Feature: Teacher memory budget.
        Description: Reuse a small bank across Q chunks, but never cast an oversized bank.
        Expectation: Small banks are reused and oversized banks keep bounded conversion.
        """
        torch.manual_seed(926)
        query = torch.randn(1, 3, 13, 8).bfloat16()
        key = torch.randn(1, 29, 8).bfloat16()
        indices = torch.arange(7).view(1, 1, 7).expand(1, 13, 7)
        sinks = torch.tensor([-4., 0., 90.])
        expected = compressed_attention_teacher(query, key, indices, sinks, .5)
        seen = []
        original = compressed_attention_utils.compressed_attention_teacher

        def _observe(q, bank, *args, **kwargs):
            """Record the bank reused by each teacher Q chunk."""
            seen.append(bank)
            return original(q, bank, *args, **kwargs)

        for budget in (29 * 8 * 4, 29 * 8 * 4 - 1):
            seen.clear()
            with patch.object(compressed_attention_utils, "_FP32_TEMPORARY_BYTES", budget), \
                 patch.object(compressed_attention_utils, "compressed_attention_teacher", _observe):
                actual = causal_attention_teacher(query, key, indices, sinks, .5, 1, 0, 4)
            torch.testing.assert_close(actual, expected)
            self.assertEqual(len(seen), 4)
            self.assertTrue(all(bank is seen[0] for bank in seen))
            self.assertEqual(seen[0].dtype, torch.float32 if budget == 29 * 8 * 4 else key.dtype)

    def test_packed_snapshot_keeps_old_geometry_after_input_mutation(self):
        """Saved activations retain their own document boundaries.

        Feature: Packed snapshot ownership.
        Description: Saved activations retain their own document boundaries.
        Expectation: Old geometry remains unchanged and the next snapshot reflects input updates.
        """
        boundaries = torch.tensor([0, 12, 32, 64])
        source = SharedCompressedPackedSequence(boundaries, 7, 26, 64)
        old = source.prepare(torch.device("cpu"), (1, 2))
        expected = old.indexer_segments(2)
        boundaries[1] = 16
        new = source.prepare(torch.device("cpu"), (1, 2))
        self.assertIsNot(old, new)
        self.assertEqual(old.indexer_segments(2), expected)
        self.assertNotEqual(old.indexer_segments(2), new.indexer_segments(2))

    def test_loss_and_gradient_match_scalar_objective(self):
        """Forward precomputation honors the scalar derivative and outer seed.

        Feature: Indexer KL derivatives.
        Description: Forward precomputation honors the scalar derivative and outer seed.
        Expectation: Scalar loss and student gradients match an independent autograd objective.
        """
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
        """Noncontiguous banks retain independent batch offsets and repeated-key SUM.

        Feature: Selected-key gather.
        Description: Noncontiguous banks retain independent batch offsets and repeated-key SUM.
        Expectation: Batch coordinates and repeated-key gradients match direct indexing.
        """
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
        """Adjacent IDs above 2**24 must never collapse into one float value.

        Feature: Exact integer sorting.
        Description: Adjacent IDs above 2**24 must never collapse into one float value.
        Expectation: Sorted indices remain exact on both sides of the FP32 integer limit.
        """
        for maximum in (2**24, 2**24+1, 2**25):
            with self.subTest(maximum=maximum):
                values = torch.tensor([[maximum, maximum-1, -1, 0, maximum-2]], dtype=torch.int64)
                torch.testing.assert_close(sort_key_indices(values, maximum), values.sort(-1).values,
                                           rtol=0, atol=0)

    def test_gather_cast_chooses_small_bank_and_keeps_large_bank_unconverted(self):
        """The long-bank fallback returns the same FP32 values without a full cast.

        Feature: Selected-key FP32 conversion.
        Description: The long-bank fallback returns the same FP32 values without a full cast.
        Expectation: The temporary budget selects the expected cast strategy with equal results.
        """
        for keys in (4, 128):
            bank = torch.arange(keys*8).reshape(1, keys, 8).bfloat16()
            indices = torch.tensor([[[0, 1, 1, 2]*8]])
            with patch.object(compressed_attention_utils, "_FP32_TEMPORARY_BYTES", 512):
                prepared = _teacher_bank(bank, indices.numel())
                self.assertEqual(prepared.dtype, torch.float32 if keys == 4 else torch.bfloat16)
                actual = gather_selected_keys_fp32(bank, indices)
            torch.testing.assert_close(actual, gather_selected_keys(bank, indices).float(), rtol=0, atol=0)

    def test_streamed_teacher_keeps_per_head_sink_and_final_normalization(self):
        """Force both head and selected-key blocks, including empty and underflow rows.

        Feature: Chunked KL teacher.
        Description: Force both head and selected-key blocks, including empty and underflow rows.
        Expectation: Chunking preserves each head sink and the normalized teacher distribution.
        """
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
            with patch.object(compressed_attention_utils, "_FP32_TEMPORARY_BYTES", 256):
                actual = compressed_attention_teacher(query, bank, indices, sinks, .5)
            torch.testing.assert_close(actual, expected)

    def test_discrete_selection_keeps_global_topk_ties_without_autograd(self):
        """Dense scalar-score TopK remains the oracle even for all-equal scores.

        Feature: Indexer TopK selection.
        Description: Dense scalar-score TopK remains the oracle even for all-equal scores.
        Expectation: Selected IDs match dense TopK and do not retain an autograd graph.
        """
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
            with torch.autograd.graph.saved_tensors_hooks(lambda tensor, saved=saved: saved.append(tensor) or tensor, lambda x: x):
                actual = compressed_causal_topk(
                    query, key, weights, compress_ratio=2, sparse_count=5,
                    query_offset=7,
                )
            self.assertEqual(saved, [])
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
