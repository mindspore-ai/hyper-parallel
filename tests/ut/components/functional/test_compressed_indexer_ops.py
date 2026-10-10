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
"""Numerical and interface regressions for compressed Indexer/KL fusion."""

import sys
import unittest
from types import ModuleType
from unittest.mock import patch

import torch
from torch.utils.checkpoint import DefaultDeviceType

from hyper_parallel.components.functional import compressed_attention_utils, compressed_indexer_ops, compressed_smla
from hyper_parallel.components.functional.compressed_attention_utils import (
    compressed_attention_teacher, gather_selected_keys, gather_selected_keys_fp32, sort_key_indices, _teacher_bank,
)
from hyper_parallel.components.functional.compressed_indexer_ops import (
    _load_indexer_op, _load_indexer_metadata_op, _load_kl_ops, _stable_prediction_log, fused_compressed_topk,
)
from hyper_parallel.components.functional.compressed_attention_utils import causal_attention_teacher
from hyper_parallel.components.functional.compressed_smla import _attention_metadata, _SparseMla, _load_attention_ops
from hyper_parallel.core.activation_memory import checkpoint
from hyper_parallel.distributed.activation_checkpoint import make_selective_checkpoint_context_fn
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
        """A validation forward must not poison residuals saved by later training."""
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

        operators = (_forward, _backward, _metadata)
        _attention_metadata.cache_clear()
        self.addCleanup(_attention_metadata.cache_clear)
        with patch.object(compressed_smla, "_load_attention_ops", return_value=operators):
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

    def test_underflow_recovery_ignores_invalid_slots_and_empty_rows(self):
        """Only valid zero probabilities need stable recomputation."""
        query = torch.ones(1, 3, 1, 2)
        key = torch.tensor([[[1., 1.], [0., 0.]]])
        weights = torch.tensor([[[1000.], [1.], [1.]]])
        indices = torch.tensor([[[0, 1, -1], [0, -1, -1], [-1, -1, -1]]])
        prediction = torch.tensor([[[1., 0., 0.], [1., 0., 0.], [0., 0., 0.]]])
        actual = _stable_prediction_log(prediction, query, key, weights, indices, 1)
        torch.testing.assert_close(actual, torch.tensor([[[0., -2000., 0.], [0., 0., 0.], [0., 0., 0.]]]))

    def test_prefix_teacher_matches_selected_teacher(self):
        """Cover partial/full transition, a zero row, and odd CP offsets."""
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

    def test_indexer_keeps_whole_document_query_segment(self):
        """A document longer than 8192 Q rows remains one native selection call."""
        segments = ((0, 8193, 0, 8193, 0), (8193, 8196, 8193, 8196, 0))
        query = torch.ones(1, 8196, 1, 128)
        key = torch.ones(1, 8196, 128)
        weights = torch.ones(1, 8196, 1)

        def _indexer(queries, keys, merge_weights, topk, **kwargs):
            del keys, merge_weights, kwargs
            return torch.zeros(1, queries.shape[1], 1, topk, dtype=torch.int32), None

        with patch.object(compressed_indexer_ops, "_requires_metadata", return_value=False), \
                patch.object(compressed_indexer_ops, "_load_indexer_op") as loader:
            loader.return_value.side_effect = _indexer
            actual = fused_compressed_topk(query, key, weights, 1, 4, segments)
        self.assertEqual(loader.return_value.call_count, 2)
        self.assertEqual(loader.return_value.call_args_list[0].args[0].shape[1], 8193)
        torch.testing.assert_close(actual[:, :8193], torch.zeros_like(actual[:, :8193]))
        torch.testing.assert_close(actual[:, 8193:], torch.full_like(actual[:, 8193:], 8193))

    def test_packed_snapshot_keeps_old_geometry_after_input_mutation(self):
        """Saved activations retain their own document boundaries."""
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
            with patch.object(compressed_attention_utils, "_FP32_TEMPORARY_BYTES", 512):
                prepared = _teacher_bank(bank, indices.numel())
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
            with patch.object(compressed_attention_utils, "_FP32_TEMPORARY_BYTES", 256):
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
            with torch.autograd.graph.saved_tensors_hooks(
                    lambda tensor, saved=saved: saved.append(tensor) or tensor, lambda x: x,
            ):
                actual = compressed_causal_topk(
                    query, key, weights, compress_ratio=2, sparse_count=5,
                    query_offset=7,
                )
            self.assertEqual(saved, [])
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


class TestCompressedIndexerOps(unittest.TestCase):
    """Keep probability recovery compatible with strict selective recomputation."""

    def test_underflow_recovery_preserves_selective_checkpoint_cache(self):
        """Recover underflowed probabilities during strict SAC replay."""
        self.addCleanup(DefaultDeviceType.set_device_type, DefaultDeviceType.get_device_type())
        # CPU inputs otherwise inherit the installed accelerator's checkpoint default.
        DefaultDeviceType.set_device_type("cpu")
        query = torch.tensor([[[[1., 1.]]]], requires_grad=True)
        key = torch.tensor([[[1., 1.], [-1., -1.]]])
        weights = torch.tensor([[[1000.]]])
        indices = torch.tensor([[[0, 1]]])
        prediction = torch.tensor([[[1., 0.]]])

        def _contraction(equation, queries, keys):
            """Expose the same cache/version aliasing on a CPU matmul backend."""
            self.assertEqual(equation, "bqhd,bqkd->bqhk")
            return torch.mm(queries[0, 0], keys[0, 0].T).unsqueeze(0).unsqueeze(0)

        def _region(value):
            return _stable_prediction_log(prediction, value, key, weights, indices, 1).sum()

        # CPU einsum can hide mutation behind an unsafe view. Regular views
        # preserve the cached matmul version counter, as in the NPU failure.
        with patch("torch.einsum", side_effect=_contraction):
            loss = checkpoint(
                _region, query, use_reentrant=False,
                context_fn=make_selective_checkpoint_context_fn(),
            )
            loss.backward()
        torch.testing.assert_close(loss, torch.tensor(-2000.), rtol=0, atol=0)
        torch.testing.assert_close(query.grad, torch.full_like(query, -1000.), rtol=0, atol=0)

    def test_missing_extension_reports_installation_requirement(self):
        """Each direct loader reports its required entry and preserves the import cause."""
        loaders = (_load_indexer_op, _load_indexer_metadata_op,
                   _load_kl_ops, _load_attention_ops)
        for loader in loaders:
            loader.cache_clear()
            self.addCleanup(loader.cache_clear)
            with self.subTest(loader=loader.__name__), patch.dict(sys.modules, {"cann_ops_transformer": None}):
                with self.assertRaisesRegex(RuntimeError, "use_fused_ops=True requires cann_ops_transformer") as error:
                    loader()
                self.assertIsInstance(error.exception.__cause__, ModuleNotFoundError)

    def test_installed_package_must_export_the_requested_entries(self):
        """Package presence alone is insufficient; LI does not require unrelated entries."""
        module = ModuleType("cann_ops_transformer")
        module.lightning_indexer = object()
        loaders = (_load_indexer_op, _load_indexer_metadata_op,
                   _load_kl_ops, _load_attention_ops)
        for loader in loaders:
            loader.cache_clear()
            self.addCleanup(loader.cache_clear)
        with patch.dict(sys.modules, {"cann_ops_transformer": module}):
            self.assertIs(_load_indexer_op(), module.lightning_indexer)
            for loader in loaders[1:]:
                with self.subTest(loader=loader.__name__):
                    with self.assertRaisesRegex(RuntimeError, "use_fused_ops=True requires") as error:
                        loader()
                    self.assertIsInstance(error.exception.__cause__, ImportError)
