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
"""Input-layout pairing: natural prefixes, kernel metadata and saved backward plans."""
from typing import Any
import unittest

import torch

from hyper_parallel.core.shard.ops import dsa_cp_fold as fold
from hyper_parallel.core.shard.ops.tnd_kv_plan import plan_tnd_kv


def attention(query: Any, key: Any, value: Any, indices: Any, scale: Any, **kwargs: Any) -> Any:
    """Independent document-local dense reference with right-aligned causal masks."""
    del indices
    qends = kwargs['actual_seq_lengths_query'].tolist()
    kends = kwargs['actual_seq_lengths_kv'].tolist()
    outputs = []
    qstart = kstart = 0
    for qend, kend in zip(qends, kends):
        if qend > qstart:
            q, k, v = query[qstart:qend], key[kstart:kend, 0], value[kstart:kend, 0]
            scores = torch.einsum('qhd,kd->hqk', q, k) * scale
            mask = torch.arange(kend-kstart)[None, :] > torch.arange(qend-qstart)[:, None] + kend-kstart-(qend-qstart)
            weights = scores.masked_fill(mask[None], float('-inf')).softmax(-1)
            outputs.append(torch.einsum('hqk,kd->qhd', weights, v))
        qstart, kstart = qend, kend
    output = torch.cat(outputs)
    stats = output.sum(-1).unsqueeze(0)
    return output, stats, stats


class TestInputPairing(unittest.TestCase):
    """Validate pairing metadata, natural prefixes and saved backward layouts."""

    def test_empty_document_padding_does_not_change_compact_plan(self):
        """Empty documents and repeated ends must leave consumer interval plans unchanged."""
        short = (52, 110, 206, 612, 4096, 8192)
        padded = (0, 0) + short + (8192,) * 1024
        queries = ((2048, 3072), (7168, 8192))
        self.assertEqual(plan_tnd_kv(short, 8192, queries), plan_tnd_kv(padded, 8192, queries))

    def test_forward_backward_and_plan_lifetime(self):
        """Interleaved noncompact calls retain their own backward permutation."""
        self._check_forward_backward(False)

    def test_compact_forward_backward_and_plan_lifetime(self):
        """Interleaved compact calls retain their own backward intervals."""
        self._check_forward_backward(True)

    def _check_forward_backward(self, compact):
        """Interleave two layouts, then backward after both scopes have exited."""
        torch.manual_seed(7)
        layouts = [((0, 2), (7, 4), (1, 5), (6, 3)),
                   ((6, 1), (0, 7), (5, 2), (3, 4))]
        total, width = 24, 3
        actual_losses, reference_losses, tensors = [], [], []
        for pairs in layouts:
            ends = torch.tensor([0, 5, 5, 9, 24, 24], dtype=torch.int32)
            if compact:
                setattr(ends, "_hp_compact_ends", (0, 5, 5, 9, 24, 24))
            setattr(ends, "_hp_fold_pairs", pairs)
            q, k, v = [torch.randn(total, heads, 4, requires_grad=True) for heads in (2, 1, 1)]
            rq, rk, rv = [x.detach().clone().requires_grad_() for x in (q, k, v)]
            order = [t for row in pairs for chunk in row for t in range(chunk*width, (chunk+1)*width)]
            folded_k, folded_v = k[order], v[order]
            reference = attention(rq, rk, rv, None, 0.5,
                                  actual_seq_lengths_query=ends, actual_seq_lengths_kv=ends)[0]
            losses = []
            for rank, pair in enumerate(pairs):
                positions = [t for chunk in pair for t in range(chunk*width, (chunk+1)*width)]
                actual = fold.fold_sparse_flash_attention(attention, rank, 4,
                    q[positions], folded_k, folded_v, torch.zeros(6, 1, 1, dtype=torch.int32), 0.5,
                    fold_layout='TND', actual_seq_lengths_query=ends, actual_seq_lengths_kv=ends)
                torch.testing.assert_close(actual[0], reference[positions])
                torch.testing.assert_close(actual[1], reference[positions].sum(-1).unsqueeze(0))
                losses.append(actual[0].square().sum())
            actual_losses.append(sum(losses))
            reference_losses.append(reference.square().sum())
            tensors.append(((q, k, v), (rq, rk, rv)))
        # A third active plan must not overwrite either saved autograd route.
        with fold.fold_pairing(((0, 7), (2, 5), (4, 3), (6, 1))):
            sum(actual_losses).backward()
        sum(reference_losses).backward()
        for actual, reference in tensors:
            for a, b in zip(actual, reference):
                torch.testing.assert_close(a.grad, b.grad, rtol=2e-5, atol=2e-5)

    def test_mismatched_metadata_is_rejected(self):
        """Reject conflicting Q/K pairing and document plans without leaking scopes."""
        first = torch.tensor([5, 9, 24], dtype=torch.int32)
        second = first.clone()
        pairs = ((0, 7), (2, 5), (4, 3), (6, 1))
        setattr(first, "_hp_fold_pairs", pairs)
        setattr(second, "_hp_fold_pairs", ((0, 2), (7, 4), (1, 5), (6, 3)))
        with self.assertRaisesRegex(ValueError, "different fold plans"):
            fold.fold_sparse_flash_attention(
                None, 0, 4, actual_seq_lengths_query=first, actual_seq_lengths_kv=second)
        setattr(second, "_hp_fold_pairs", pairs)
        setattr(first, "_hp_compact_ends", (5, 9, 24))
        setattr(second, "_hp_compact_ends", (4, 9, 24))
        with self.assertRaisesRegex(ValueError, "different compact document plans"):
            fold.fold_sparse_flash_attention(
                None, 0, 4, actual_seq_lengths_query=first, actual_seq_lengths_kv=second)
        self.assertEqual(fold.build_fold_order(4), [0, 7, 2, 5, 4, 3, 6, 1])

    def test_permutation_and_prefix_inverse(self):
        """Restore every natural prefix for an arbitrary chunk assignment."""
        pairs = ((0, 2), (7, 4), (1, 5), (6, 3))
        x = torch.arange(24., requires_grad=True).reshape(24, 1)
        ids = [t for pair in pairs for c in pair for t in range(c*3, c*3+3)]
        with fold.fold_pairing(pairs):
            inverse = fold.build_fold_order(4)
            self.assertEqual([c for pair in pairs for c in pair][inverse[3]], 3)
            for rank in range(4):
                for block in (0, 1):
                    prefix = fold.unfold_prefix(x[ids], rank, 4, block, 0)
                    torch.testing.assert_close(prefix, x[:(pairs[rank][block]+1)*3])
        self.assertEqual(fold.build_fold_order(4), [0, 7, 2, 5, 4, 3, 6, 1])
        with self.assertRaises(ValueError):
            fold.validate_fold_pairs(((0, 1), (1, 3)))


class TestCompactIntervals(unittest.TestCase):
    """Validate compact document coordinates and their inverse gradients."""

    def test_interval_merge_and_missing_gradients(self):
        """Disjoint, touching, overlapping and nested intervals invert the gather."""
        order = (2, 5, 1, 0, 7, 3, 6, 4)
        for intervals in (((1, 6), (17, 24)), ((0, 6), (6, 12)),
                          ((1, 12), (5, 24)), ((1, 24), (8, 12))):
            for missing in (None, 0, 1):
                grads = [torch.randn(stop-start, 3) for start, stop in intervals]
                if missing is not None:
                    grads[missing] = None
                expected = torch.zeros(24, 3)
                for (start, stop), grad in zip(intervals, grads):
                    if grad is not None:
                        expected[start:stop] += grad
                folded = torch.empty_like(expected)
                for natural, slot in enumerate(order):
                    folded[slot*3:(slot+1)*3] = expected[natural*3:(natural+1)*3]
                actual = fold.compact_pair_grad(*grads, intervals, order, 24)
                torch.testing.assert_close(actual, folded)
        self.assertIsNone(fold.compact_pair_grad(None, None, ((0, 3), (3, 6)), order, 24))

    def test_compact_plan_removes_empty_documents(self):
        """The public indexer path drops empty documents and validates total length."""
        query = torch.ones(6, 1, 1)
        key = torch.ones(24, 1, 1)
        weights = torch.ones(6, 1)
        seen = []

        def indexer(q: Any, key_tensor: Any, selector_weights: Any, **kwargs: Any) -> Any:
            """Record the normalized metadata delivered to the actual kernel."""
            del key_tensor, selector_weights
            seen.append((kwargs['actual_seq_lengths_query'].tolist(),
                         kwargs['actual_seq_lengths_key'].tolist()))
            return torch.zeros(len(q), 1, 1, dtype=torch.int32)

        ends = torch.tensor([0, 5, 5, 9, 24, 24], dtype=torch.int32)
        setattr(ends, "_hp_compact_ends", tuple(ends.tolist()))
        fold.fold_lightning_indexer(indexer, 1, 4, query, key, weights,
            fold_layout='TND', actual_seq_lengths_query=ends, actual_seq_lengths_key=ends)
        self.assertEqual(seen, [([3], [4]), ([3], [9])])
        invalid = torch.tensor([5, 23], dtype=torch.int32)
        setattr(invalid, "_hp_compact_ends", (5, 23))
        with self.assertRaises(ValueError):
            fold.fold_lightning_indexer(indexer, 1, 4, query, key, weights,
                fold_layout='TND', actual_seq_lengths_query=invalid, actual_seq_lengths_key=invalid)

    def test_three_kernel_paths_share_compact_coordinates(self):
        """Indexer, attention and KL must use the same compact document coordinates."""
        ends = torch.tensor([5, 9, 24], dtype=torch.int32)
        setattr(ends, "_hp_compact_ends", (5, 9, 24))
        order = fold.build_fold_order(4)
        natural = torch.arange(24.).reshape(24, 1, 1)
        key = natural.reshape(8, 3, 1, 1)[order].reshape(24, 1, 1)
        query = torch.ones(6, 1, 1)
        weights = torch.ones(6, 1)
        seen = []

        def indexer(q: Any, k: Any, selector_weights: Any, **kwargs: Any) -> Any:
            """Record the compact coordinates passed to the selector."""
            del selector_weights
            seen.append((k[:, 0, 0].tolist(), kwargs['actual_seq_lengths_key'].tolist()))
            return torch.zeros(len(q), 1, 1, dtype=torch.int32)

        indices = fold.fold_lightning_indexer(indexer, 1, 4, query, key, weights,
            fold_layout='TND', actual_seq_lengths_query=ends, actual_seq_lengths_key=ends)
        self.assertEqual(seen, [(list(range(5, 9)), [4]), (list(range(9, 18)), [9])])
        calls = []

        def kl(*args: Any) -> Any:
            """Return a known KV gradient for each compact interval."""
            calls.append(args[1][:, 0, 0].tolist())
            self.assertEqual(args[12].tolist(), [len(args[1])])
            return args[2], torch.ones_like(args[3]) * len(calls), args[4], torch.tensor(1.)

        stats = torch.ones(1, 6, 1)
        result = fold.fold_sparse_indexer_kl_loss(kl, 1, 4, query, key, query, key, weights,
            indices, stats, stats, 1., query, key, ends, ends, 'TND', 3, 2147483647, 2147483647,
            fold_layout='TND')
        expected = torch.zeros_like(natural)
        expected[5:9] = 1
        expected[9:18] = 2
        expected = expected.reshape(8, 3, 1, 1)[order].reshape(24, 1, 1)
        torch.testing.assert_close(result[1], expected)
        self.assertEqual(calls, [list(range(5, 9)), list(range(9, 18))])
