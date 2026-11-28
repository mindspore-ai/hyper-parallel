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
"""Pooled routing boundaries and separately differentiable indexer training."""
import unittest
from types import SimpleNamespace

import torch

from hyper_parallel.components.modules import KPoolIndexer, QSAIndexer
from hyper_parallel.components.modules.attention.pooled_indexer import block_indexer_kl, block_scores
from hyper_parallel.components.ops import ops, block_sparse_attention, pooled_indexer_kl
from hyper_parallel.components.modules.attention.pooled_indexer import sparse_block_teacher
from hyper_parallel.components.ops.impl.pooled_indexer_kl import cann_rejection
from tests.common.op_precision import check_case
from tests.common.op_precision_suite import precision_cases


def _config():
    return SimpleNamespace(hidden_size=16, q_lora_rank=8, index_n_heads=4, index_head_dim=8,
                           index_topk=4, index_kpool=4, qk_rope_head_dim=0, index_kpool_always_select_tail=True,
                           indexer_n_heads=4, indexer_kv_heads=1, indexer_head_dim=8, indexer_budget=4,
                           indexer_compress_ratio=4, rms_norm_eps=1e-6)


class TestPooledIndexer(unittest.TestCase):
    """No cross-document route; forced tails; independent score gradients and zero rows."""

    def setUp(self) -> None:
        """Use native kernels for portable module checks."""
        self.previous = ops.get_backends()
        ops.set_backends({"rope": "native", "rms_norm": "native"})

    def tearDown(self) -> None:
        """Restore backend choices for other tests."""
        ops.set_backends(self.previous)

    def test_packed_matches_separate_documents(self):
        generator = torch.Generator().manual_seed(81)
        x = torch.randn(2, 13, 16, generator=generator)
        latent = torch.randn(2, 13, 8, generator=generator)
        angle = torch.randn(2, 13, 2, generator=generator).repeat(1, 1, 2)
        positions = (angle.cos(), angle.sin())
        for module in (KPoolIndexer(config=_config()), QSAIndexer(config=_config())):
            is_glm = isinstance(module, KPoolIndexer)
            extra = latent if is_glm else positions
            route = module(x, extra, actual_seq_len=(3, 8, 13, 17, 26), query_chunk_size=3)
            for b, start, end in ((0, 0, 3), (0, 3, 8), (0, 8, 13), (1, 0, 4), (1, 4, 13)):
                subextra = latent[b:b+1, start:end] if is_glm else tuple(v[b:b+1, start:end] for v in positions)
                reference = module(x[b:b+1, start:end], subextra)[0]
                reference = torch.where(reference >= 0, reference + start, reference)
                torch.testing.assert_close(route[b, start:end], reference)
                for row in range(start, end):
                    selected = route[b, row][route[b, row] >= 0].tolist()
                    self.assertEqual(len(selected), len(set(selected)))
                    self.assertTrue(all(start <= token <= row for token in selected))
                    tail_start = start + ((row - start + 1) // 4) * 4
                    self.assertTrue(set(range(tail_start, row + 1)).issubset(selected))

    def test_indexer_kl_trains_parameters_not_backbone(self):
        generator = torch.Generator().manual_seed(82)
        for module in (KPoolIndexer(config=_config()), QSAIndexer(config=_config())):
            x = torch.randn(1, 13, 16, generator=generator, requires_grad=True)
            latent = torch.randn(1, 13, 8, generator=generator, requires_grad=True)
            if isinstance(module, KPoolIndexer):
                q, k, gate, weight = module.project(x, latent)
                pooled = module.pool(k[0], gate[0])
                weights = weight[0]
            else:
                angles = torch.randn(1, 13, 4, generator=generator)
                positions = (angles.cos(), angles.sin())
                q, k = module.project(x, positions)
                pooled = module.pool(k[0], tuple(v[0] for v in positions))
                weights = None
            logits = block_scores(q[0], pooled, weights)
            valid = (torch.arange(3)[None, :] + 1) * 4 <= torch.arange(1, 14)[:, None]
            teacher = torch.rand(logits.shape, generator=generator, requires_grad=True)
            loss = block_indexer_kl(logits, teacher, valid)
            self.assertTrue(torch.isfinite(loss).all())
            torch.testing.assert_close(loss[:3], torch.zeros(3))
            loss.sum().backward()
            self.assertIsNone(x.grad)
            self.assertIsNone(latent.grad)
            self.assertIsNone(teacher.grad)
            for name, parameter in module.named_parameters():
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_kl_gradient_and_teacher_isolation(self):
        logits = torch.tensor([[0.2, -0.1, 1.1], [0.2, 1.2, 4.0]], requires_grad=True)
        teacher = torch.tensor([[1., 3., 2.], [0., 0., 0.]], requires_grad=True)
        valid = torch.tensor([[True, True, False], [False, False, False]])
        block_indexer_kl(logits, teacher, valid).sum().backward()
        expected = torch.zeros_like(logits)
        expected[0, :2] = logits.detach()[0, :2].softmax(-1) - torch.tensor([0.25, 0.75])
        torch.testing.assert_close(logits.grad, expected)
        self.assertIsNone(teacher.grad)


class TestPooledSparseTraining(unittest.TestCase):
    """Exercise actual sparse routing, forced tails and teacher gradient isolation."""

    def test_native_operator_contracts(self) -> None:
        """Cover routing, independent/shared KV and KL with the common formula oracles."""
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            for family in ("pooled_indexer", "block_sparse_attention", "pooled_indexer_kl"):
                for case in precision_cases(family, dtype):
                    with self.subTest(family=family, variant=case.variant, dtype=dtype):
                        check_case(case)

    def test_sparse_teacher_and_gradients_match_independent_formula(self):
        torch.manual_seed(194)
        batch, length, heads, kv_heads, dim, pool = 2, 13, 4, 2, 8, 4
        ids = torch.full((batch, length, 2), -1, dtype=torch.int32)
        allowed = torch.zeros(batch, length, length, dtype=torch.bool)
        for b in range(batch):
            for row in range(length):
                complete = (row + 1) // pool
                if complete:
                    for slot in range(min(complete, 2)):
                        selected = (complete-1+b-slot) % complete
                        ids[b, row, slot] = selected
                        allowed[b, row, selected*pool:(selected+1)*pool] = True
                allowed[b, row, complete*pool:row+1] = True
        values = [torch.randn(batch, length, h, dim) for h in (heads, kv_heads, kv_heads)]
        actual_inputs = [x.clone().requires_grad_() for x in values]
        reference_inputs = [x.clone().requires_grad_() for x in values]
        previous = ops.get_backends()
        try:
            ops.set_backends({"block_sparse_attention": "native", "pooled_indexer_kl": "native"})
            output, lse = block_sparse_attention(*actual_inputs, ids, block_size=pool, scale=dim**-0.5,
                                                 query_chunk_size=3)
            q, k, v = reference_inputs
            k, v = (x.repeat_interleave(heads//kv_heads, dim=2).transpose(1, 2) for x in (k, v))
            scores = (q.transpose(1, 2) @ k.transpose(-1, -2)) * dim**-0.5
            scores = scores.masked_fill(~allowed[:, None], float("-inf"))
            probs = scores.softmax(-1)
            expected = (probs @ v).transpose(1, 2)
            upstream = torch.randn_like(output)
            output.backward(upstream)
            expected.backward(upstream)
            torch.testing.assert_close(output, expected)
            for actual, reference in zip(actual_inputs, reference_inputs):
                torch.testing.assert_close(actual.grad, reference.grad)
            teacher = sparse_block_teacher(q, reference_inputs[1], lse, ids,
                                            pool_size=pool, scale=dim**-0.5, query_chunk_size=3)
            self.assertFalse(teacher.requires_grad)
            self.assertFalse(lse.requires_grad)
            self.assertTrue(torch.equal(teacher[:, :pool-1], torch.zeros_like(teacher[:, :pool-1])))
            torch.testing.assert_close(teacher[:, pool-1:].sum(-1), torch.ones(batch, length-pool+1))
            mass = probs.detach().mean(1)[..., :length//pool*pool].unflatten(-1, (-1, pool)).sum(-1)
            expected_teacher = mass.gather(-1, ids.clamp_min(0).long()).masked_fill(ids < 0, 0)
            expected_teacher /= expected_teacher.sum(-1, keepdim=True).clamp_min(1e-20)
            torch.testing.assert_close(teacher, expected_teacher)
            # Independent row-wise student oracle also covers empty and one-class rows.
            iq = torch.randn(batch, length, 4, 8, requires_grad=True)
            ik = torch.randn(batch, length//pool, 8, requires_grad=True)
            iw = torch.randn(batch, length, 4, requires_grad=True)
            target = teacher.clone().requires_grad_()
            loss = pooled_indexer_kl(iq, ik, iw, ids, target, pool_size=pool, query_chunk_size=3)
            references = [t.detach().clone().requires_grad_() for t in (iq, ik, iw)]
            expected_loss = references[0].sum()*0 + references[1].sum()*0 + references[2].sum()*0
            for b in range(batch):
                for row in range(length):
                    valid = ids[b, row] >= 0
                    if valid.any():
                        selected = references[1][b, ids[b, row, valid].long()]
                        scores = (references[0][b, row] @ selected.T * 8**-0.5).relu()
                        scores = (scores * references[2][b, row, :, None]).sum(0)
                        target_row = expected_teacher[b, row, valid]
                        expected_loss += (target_row * (target_row.log() - scores.log_softmax(-1))).sum()
            loss.backward()
            expected_loss.backward()
            torch.testing.assert_close(loss, expected_loss)
            self.assertIsNone(target.grad)
            for actual, reference in zip((iq, ik, iw), references):
                torch.testing.assert_close(actual.grad, reference.grad)
            # Master preserves the exact sparse capacity instead of padding the budget.
            qi = torch.empty(batch, length, 4, 128, dtype=torch.bfloat16)
            ki = torch.empty(batch, length//pool, 128, dtype=torch.bfloat16)
            self.assertIsNone(cann_rejection(qi, ki, ids, pool))
            oversized = torch.empty(batch, length, 8193, dtype=torch.int32)
            self.assertIn("at most 8192", cann_rejection(qi, ki, oversized, pool))
            dense_ids = torch.empty(batch, length, length//pool, dtype=torch.int32)
            self.assertIsNone(cann_rejection(qi, ki, dense_ids, pool))
        finally:
            ops.set_backends(previous)
