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
"""CPU regression coverage for the additional operator-family contracts."""

import unittest
from types import SimpleNamespace

import torch
from torch.nn import functional as F

from hyper_parallel.components.modules.dsa_attention import apply_mome
from hyper_parallel.components.modules.moe import GroupedExperts
from hyper_parallel.components.ops import (
    dsa_indexer_kl_dense,
    dsa_indexer_kl_packed,
    dsa_indexer_packed,
    dsa_sparse_attention_absorbed,
    dsa_sparse_attention_with_sink,
    gqa_attention,
    mla_attention_with_stats,
    ops,
    rope_interleaved,
    rope_qk,
)
from hyper_parallel.components.ops.catalog import builtin_registry


def _native_runtime(**selections):
    """Create a CPU runtime with explicit per-family choices."""
    return ops.configure(registry=builtin_registry, target_device="cpu", selections=selections)


class TestBackendMigration(unittest.TestCase):
    """Verify reference math and selection at the ordinary forward entry points."""

    def test_rotary_variants_preserve_the_unrotated_tail(self) -> None:
        """Half and interleaved conventions retain different pair layouts and the same tail."""
        x = torch.tensor([[[[1.0, 2.0, 3.0, 4.0, 5.0]]]], requires_grad=True)
        cos, sin = torch.zeros(1, 4), torch.ones(1, 4)
        runtime = _native_runtime(rope="native")
        with runtime.activate():
            half, _ = rope_qk(x, x, cos, sin)
            interleave, _ = rope_interleaved(x, x, cos, sin)
        torch.testing.assert_close(half.flatten(), torch.tensor([-3.0, -4.0, 1.0, 2.0, 5.0]))
        torch.testing.assert_close(interleave.flatten(), torch.tensor([-2.0, -4.0, 1.0, 3.0, 5.0]))
        torch.testing.assert_close(torch.autograd.grad(half.sum() + interleave.sum(), x)[0][..., -1],
                                   torch.full_like(x[..., -1], 2))
        self.assertEqual(set(runtime.used_backends()), {("rope", "half", "native"), ("rope", "interleaved", "native")})

    def test_mome_conv_masks_updates_and_preserves_bias(self) -> None:
        """Interior holes mask only the residual update and remain valid convolution history."""
        x = torch.arange(1.0, 5.0).reshape(1, 4, 1).requires_grad_()
        conv = torch.nn.Conv1d(1, 1, 2, groups=1)
        with torch.no_grad():
            conv.weight.fill_(1.0)
            conv.bias.fill_(2.0)
        mask = torch.tensor([[True, False, True, True]])
        runtime = _native_runtime(mome="native")
        with runtime.activate():
            output = apply_mome(x, mask, conv, fused=None)
        torch.testing.assert_close(output.flatten(), torch.tensor([4.0, 2.0, 10.0, 13.0]))
        output.sum().backward()
        torch.testing.assert_close(conv.bias.grad, torch.tensor([3.0]))
        self.assertIn(("mome", "standard", "native"), runtime.used_backends())

    def test_grouped_expert_forward_with_empty_tokens_keeps_parameter_gradients(self) -> None:
        """Empty expert dispatch preserves a zero gradient on all original parameters."""
        experts = GroupedExperts(4, 8, 2)
        runtime = _native_runtime(grouped_swiglu="native")
        x = torch.empty(0, 4, requires_grad=True)
        with runtime.activate():
            output = experts(x, torch.zeros(2, dtype=torch.int64))
        output.sum().backward()
        for parameter in experts.parameters():
            torch.testing.assert_close(parameter.grad, torch.zeros_like(parameter))
        self.assertIn(("grouped_swiglu", "standard", "native"), runtime.used_backends())

    def test_packed_gqa_matches_independent_documents(self) -> None:
        """Multiple query heads share K/V; the packed window never crosses a document."""
        generator = torch.Generator().manual_seed(19)
        query = torch.randn(1, 4, 5, 4, generator=generator)
        key = torch.randn(1, 2, 5, 4, generator=generator)
        value = torch.randn(1, 2, 5, 3, generator=generator)
        module = SimpleNamespace(is_causal=True)
        runtime = _native_runtime(attention="native")
        with runtime.activate():
            output, weights = gqa_attention(module, query, key, value, None, actual_seq_len=[2, 5], sliding_window=2)
        expected = []
        for start, end in ((0, 2), (2, 5)):
            mask = torch.ones(end - start, end - start, dtype=torch.bool).tril().triu(-1)
            expected.append(F.scaled_dot_product_attention(
                query[:, :, start:end], key[:, :, start:end], value[:, :, start:end], attn_mask=mask, enable_gqa=True,
            ).transpose(1, 2))
        torch.testing.assert_close(output, torch.cat(expected, dim=1))
        self.assertIsNone(weights)
        self.assertIn(("attention", "gqa", "native"), runtime.used_backends())

    def test_packed_gqa_padding_keeps_batch_axis(self) -> None:
        """Packed reference respects each physical batch's key and query padding mask."""
        query = torch.ones(2, 1, 3, 2)
        value = torch.arange(6.0).reshape(2, 1, 3, 1)
        mask = torch.tensor([[True, False, True], [True, True, False]])
        module = SimpleNamespace(is_causal=True)
        runtime = _native_runtime(attention="native")
        with runtime.activate():
            output, _ = gqa_attention(module, query, query, value, mask, actual_seq_len=[3, 6])
        torch.testing.assert_close(output.flatten(), torch.tensor([0.0, 0.0, 1.0, 3.0, 3.5, 0.0]))

    def test_packed_dsa_uses_document_local_indices_and_detaches_teacher(self) -> None:
        """Sparse MLA statistics and KL share document ordering with the CANN contract."""
        generator = torch.Generator().manual_seed(5)
        student_query = torch.rand(1, 5, 2, 2, generator=generator, requires_grad=True)
        student_key = torch.rand(1, 5, 1, 2, generator=generator, requires_grad=True)
        student_weights = torch.rand(1, 5, 2, generator=generator, requires_grad=True)
        query = torch.rand(1, 5, 2, 2, generator=generator, requires_grad=True)
        key = torch.rand(1, 5, 1, 2, generator=generator, requires_grad=True)
        q_rope = torch.zeros_like(query, requires_grad=True)
        k_rope = torch.zeros_like(key, requires_grad=True)
        ends = torch.tensor([2, 5], dtype=torch.int32)
        runtime = _native_runtime(dsa_indexer="native", dsa_sparse_attention="native", dsa_indexer_kl="native")
        with runtime.activate():
            indices, *index_states = dsa_indexer_packed(student_query, student_key, student_weights, ends, ends, 3)
            output, maximum, total = dsa_sparse_attention_absorbed(query, key, q_rope, k_rope, indices, 0.5, ends, ends)
            loss = dsa_indexer_kl_packed(
                *index_states, query.flatten(0, 1), key.flatten(0, 1), indices, maximum, total,
                q_rope.flatten(0, 1), k_rope.flatten(0, 1), ends, ends, 0.5, 0.7,
            )
        self.assertEqual(indices[2].flatten().tolist(), [0, -1, -1])
        torch.testing.assert_close(output[0, 2, :, :2], key[0, 2].expand(2, -1))
        torch.testing.assert_close(total[0, 2], torch.ones(2))
        self.assertEqual(maximum.shape, (1, 5, 2))
        loss.backward()
        for student in (student_query, student_key, student_weights):
            self.assertIsNotNone(student.grad)
        for teacher in (query, key, q_rope, k_rope):
            self.assertIsNone(teacher.grad)

    def test_parameter_sink_and_dense_teacher_use_reference_paths(self) -> None:
        """Sink logits enter the same softmax and dense MLA reports detached max/sum."""
        query = torch.tensor([[[[1.0]], [[2.0]]]], requires_grad=True)
        key = torch.tensor([[[[2.0]], [[3.0]]]], requires_grad=True)
        rope = torch.zeros_like(query)
        indices = torch.tensor([[[0, -1]], [[0, 1]]], dtype=torch.int32)
        ends = torch.tensor([2], dtype=torch.int32)
        sink_key = torch.tensor([[[[4.0, 0.0]]]], requires_grad=True)
        sink_value = torch.tensor([[[[7.0, 0.0]]]], requires_grad=True)
        runtime = _native_runtime(attention="native", dsa_sparse_attention="native", dsa_indexer_kl="native")
        with runtime.activate():
            sparse, _, _ = dsa_sparse_attention_with_sink(
                query, key, rope, rope, indices, 1.0, ends, ends, sink_key=sink_key, sink_value=sink_value,
            )
            dense, maximum, total = mla_attention_with_stats(query, key, key, ends, ends, scale=1.0, need_stats=True)
            student = query.detach().clone().requires_grad_().flatten(0, 1)
            loss = dsa_indexer_kl_dense(
                student, key.detach().flatten(0, 1), torch.ones(2, 1),
                query.flatten(0, 1), key.flatten(0, 1), maximum, total, rope.flatten(0, 1), rope.flatten(0, 1),
                ends, ends, scale=1.0, loss_coeff=1.0,
            )
        expected_sparse = torch.tensor([2.0, 4.0]).softmax(-1) @ torch.tensor([2.0, 7.0])
        torch.testing.assert_close(sparse[0, 0, 0, 0], expected_sparse)
        torch.testing.assert_close(maximum.flatten(), torch.tensor([2.0, 6.0]))
        torch.testing.assert_close(dense[0, 0, 0, 0], torch.tensor(2.0))
        torch.testing.assert_close(loss, torch.tensor(0.0))
        self.assertFalse(maximum.requires_grad)
        sparse.sum().backward()
        self.assertIsNotNone(sink_key.grad)
        self.assertIsNotNone(sink_value.grad)
