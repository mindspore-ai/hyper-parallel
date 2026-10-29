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
"""Hardware-independent DSA allocation, argument and gradient contracts."""

import importlib
import unittest
from typing import Any
from unittest.mock import patch

import torch

from hyper_parallel.components.functional import dsa_ops
from hyper_parallel.components.functional import dsa_dense as dense_api
from hyper_parallel.components.functional.dsa_indexer import dsa_indexer
from hyper_parallel.components.functional.dsa_sparse_attention import dsa_sparse_attention

kl_ops = importlib.import_module("hyper_parallel.components.functional.dsa_kl_loss")


class TestDSAOps(unittest.TestCase):
    """Verify package adapter contracts without importing NPU extensions."""

    def test_reject_cpu_before_extension_loading(self):
        """A CPU call fails at its boundary, without trying a compiler or CANN."""
        with patch.object(dsa_ops, "lightning_indexer") as kernel:
            with self.assertRaisesRegex(ValueError, "Ascend NPU"):
                dsa_ops._lightning_indexer(torch.zeros(1, 4, 8, 128), None, None)
            kernel.assert_not_called()

    def test_indexer_adapts_tnd_lengths_and_value_dtype(self):
        """The package receives cu-seqlens and preserves the public value dtype."""
        query = torch.zeros(8, 2, 4, dtype=torch.bfloat16)
        key = torch.zeros(8, 1, 4, dtype=torch.bfloat16)
        weights = torch.zeros(8, 2, dtype=torch.bfloat16)
        lengths = torch.tensor([4, 8], dtype=torch.int32)
        expected_indices = torch.zeros(8, 1, 16, dtype=torch.int32)
        expected_values = torch.zeros(8, 1, 16, dtype=torch.float32)

        def package_indexer(*_args: Any, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
            """Check packed lengths passed to the package API."""
            torch.testing.assert_close(kwargs["cu_seqlens_q"], torch.tensor([0, 4, 8], dtype=torch.int32))
            torch.testing.assert_close(kwargs["cu_seqlens_k"], torch.tensor([0, 4, 8], dtype=torch.int32))
            return expected_indices, expected_values

        with patch.object(dsa_ops, "_validate"), \
                patch.object(dsa_ops, "lightning_indexer", side_effect=package_indexer):
            indices, values = dsa_ops._lightning_indexer(
                query, key, weights, actual_seq_qlen=lengths, actual_seq_klen=lengths,
                layout="TND", sparse_count=16,
            )
        self.assertIs(indices, expected_indices)
        self.assertEqual(values.dtype, query.dtype)

    def test_package_kl_target_uses_document_local_indices(self):
        """Sparse KL target reconstruction resets key offsets at each document."""
        query = torch.tensor([[[1., 0.]], [[2., 0.]], [[3., 0.]], [[4., 0.]]])
        key = torch.tensor([[[1., 0.]], [[2., 0.]], [[10., 0.]], [[20., 0.]]])
        indices = torch.tensor([[[0, -1]], [[0, 1]], [[0, -1]], [[0, 1]]], dtype=torch.int32)
        maximum = torch.zeros(1, 4, 1)
        total = torch.ones(1, 4, 1)
        target = dsa_ops._package_sparse_kl_target(
            query, key, indices, maximum, total, 1.0, None, None,
            torch.tensor([2, 4], dtype=torch.int32), torch.tensor([2, 4], dtype=torch.int32),
        )
        self.assertEqual(target.shape, (4, 1, 2))
        self.assertGreater(target[3, 0, 1], target[3, 0, 0])

    def test_tnd_requires_both_sequence_boundaries(self):
        """Incomplete packed metadata must fail before launching a kernel."""
        with patch.object(dsa_ops, "_validate"):
            with self.assertRaisesRegex(ValueError, "cumulative"):
                dsa_ops._dense_indexer_lse_v2(
                    torch.zeros(4, 8, 128), torch.zeros(4, 1, 128), torch.zeros(4, 8), layout="TND")

    def test_invalid_metadata_fails_before_launch(self):
        """Mismatched documents, non-integer lengths and unpaired metadata are rejected."""
        query, key, weights = torch.zeros(8, 8, 128), torch.zeros(8, 1, 128), torch.zeros(8, 8)
        cases = ((torch.tensor([4, 8]), torch.tensor([8])),
                 (torch.tensor([4., 8.]), torch.tensor([4, 8])),
                 (torch.tensor([4, 8]), None))
        with patch.object(dsa_ops, "_validate"):
            for qlen, klen in cases:
                with self.subTest(qlen=qlen, klen=klen), self.assertRaises(ValueError):
                    dsa_ops._dense_indexer_lse_v2(
                        query, key, weights, actual_seq_qlen=qlen, actual_seq_klen=klen, layout="TND")

    def test_attention_rejects_unsupported_key_value_and_rope(self):
        """Independent V, malformed indices and unpaired RoPE cannot reach CANN."""
        query, key = torch.zeros(1, 3, 2, 4), torch.zeros(1, 3, 1, 4)
        indices = torch.zeros(1, 3, 1, 2048, dtype=torch.int32)
        with patch.object(dsa_ops, "_validate"):
            with self.assertRaisesRegex(ValueError, "alias"):
                dsa_ops._sparse_attention(query, key, key.clone(), indices, .5)
            with self.assertRaisesRegex(ValueError, "together"):
                dsa_ops._sparse_attention(
                    query, key, key, indices, .5, query_rope=torch.zeros(1, 3, 2, 2)
                )
            with self.assertRaisesRegex(ValueError, "indices"):
                dsa_ops._sparse_attention(query, key, key, indices.long(), .5)

    def test_empty_kl_query_fails_before_launch(self):
        """Empty batches cannot reach CANN or divide a precomputed loss by zero."""
        inputs = (
            torch.empty(0, 1, 4), torch.empty(0, 1, 4), torch.empty(0, 1),
            torch.empty(0, 1, 1), torch.empty(1, 0),
        )
        with patch.object(dense_api, "_cann_dsa_dense_indexer_kl_loss_grad") as kernel:
            with self.assertRaisesRegex(ValueError, "at least one valid query"):
                dense_api.dsa_dense_indexer_kl_loss(*inputs[:3], *inputs[3:4], inputs[4])
            kernel.assert_not_called()

    def test_dense_indexer_lse_uses_v2_metadata_and_lse_entrypoint(self):
        """Dense warm-up calls the V2 metadata and LSE package APIs."""
        query = torch.zeros(8, 2, 128, dtype=torch.bfloat16)
        key = torch.zeros(8, 1, 128, dtype=torch.bfloat16)
        weights = torch.zeros(8, 2, dtype=torch.bfloat16)
        lengths = torch.tensor([4, 8], dtype=torch.int32)
        lse = torch.zeros(1, 8)

        def metadata_op(*_args: Any, **kwargs: Any) -> dict[str, Any]:
            """Return metadata arguments for verification by the wrapped op."""
            return kwargs

        with patch.object(dsa_ops, "_validate"), \
                patch.object(dsa_ops, "dense_lightning_indexer_softmax_lse_metadata", side_effect=metadata_op), \
                patch.object(dsa_ops, "dense_lightning_indexer_softmax_lse", return_value=lse):
            result = dsa_ops._dense_indexer_lse_v2(
                query, key, weights, actual_seq_qlen=lengths, actual_seq_klen=lengths, layout="TND",
            )
        self.assertIs(result, lse)

    def test_dense_kl_passes_target_lse_and_packed_metadata(self):
        """Dense KL uses the package target contract and masks causal padding."""
        query = torch.zeros(5, 2, 128, dtype=torch.bfloat16)
        key = torch.zeros(8, 1, 128, dtype=torch.bfloat16)
        weights = torch.zeros(5, 2, dtype=torch.bfloat16)
        lengths_q = torch.tensor([2, 5], dtype=torch.int32)
        lengths_k = torch.tensor([3, 8], dtype=torch.int32)
        target = torch.tensor([
            [[1.0, 0, 0, 0, 0]],
            [[0.5, 0.5, 0, 0, 0]],
            [[1 / 3, 1 / 3, 1 / 3, 0, 0]],
            [[0.25, 0.25, 0.25, 0.25, 0]],
            [[0.2, 0.2, 0.2, 0.2, 0.2]],
        ])
        gradients = (torch.ones_like(query), torch.ones_like(key), torch.ones_like(weights))
        calls = {}

        def metadata(*args: Any, **kwargs: Any) -> torch.Tensor:
            """Capture generated dense-KL metadata parameters."""
            calls["metadata"] = (args, kwargs)
            return torch.ones(64, dtype=torch.int32)

        def dense_kl(*args: Any, **kwargs: Any) -> tuple[torch.Tensor, ...]:
            """Return the supplied target as the student probability."""
            calls["dense_kl"] = (args, kwargs)
            return (*gradients, target.clone())

        with patch.object(dsa_ops, "_validate"), \
                patch.object(dsa_ops, "dense_lightning_indexer_kl_loss_grad_metadata", side_effect=metadata), \
                patch.object(dsa_ops, "dense_lightning_indexer_kl_loss_grad", side_effect=dense_kl):
            result = dsa_ops._dense_indexer_kl_loss_grad(
                query, key, weights, target, torch.zeros(1, 5),
                actual_seq_qlen=lengths_q, actual_seq_klen=lengths_k, layout="TND",
            )

        self.assertEqual(calls["metadata"][0], (2, 1, 128))
        torch.testing.assert_close(
            calls["metadata"][1]["cu_seqlens_q"], torch.tensor([0, 2, 5], dtype=torch.int32),
        )
        torch.testing.assert_close(
            calls["metadata"][1]["cu_seqlens_k"], torch.tensor([0, 3, 8], dtype=torch.int32),
        )
        torch.testing.assert_close(calls["dense_kl"][0][3], target)
        self.assertEqual(result[0].shape, query.shape)
        torch.testing.assert_close(result[3], torch.zeros(1))

    def test_attention_accumulates_shared_key_value_and_rope_gradients(self):
        """Aliased K/V receive both derivatives; softmax statistics stay detached."""
        query = torch.zeros(1, 3, 2, 4, requires_grad=True)
        key = torch.zeros(1, 3, 1, 4, requires_grad=True)
        qrope = torch.zeros(1, 3, 2, 2, requires_grad=True)
        krope = torch.zeros(1, 3, 1, 2, requires_grad=True)
        indices = torch.zeros(1, 3, 1, 4, dtype=torch.int32)

        forward = (torch.zeros(1, 3, 2, 4), torch.zeros(1, 3, 2, 1), torch.zeros(1, 3, 2, 1))
        backward = tuple(torch.full_like(value, factor) for value, factor in zip(
            (query, key, key, qrope, krope), (2, 3, 5, 7, 11)))
        with patch.object(dsa_ops, "_validate"), \
                patch.object(dsa_ops, "_package_sparse_attention", return_value=forward), \
                patch.object(dsa_ops, "_package_sparse_attention_grad", return_value=backward):
            output, maximum, total = dsa_ops._sparse_attention(
                query, key, key, indices, .5, query_rope=qrope, key_rope=krope)
            (3 * output.sum()).backward()
        torch.testing.assert_close(query.grad, torch.full_like(query, 2))
        torch.testing.assert_close(key.grad, torch.full_like(key, 8))
        torch.testing.assert_close(qrope.grad, torch.full_like(qrope, 7))
        torch.testing.assert_close(krope.grad, torch.full_like(krope, 11))
        self.assertFalse(maximum.requires_grad)
        self.assertFalse(total.requires_grad)

    def test_dense_kl_routes_explicit_indexer_gradients_once(self):
        """Dense KL normalizes only indexer gradients and ignores teacher inputs."""
        inputs = [torch.ones(4, 2, 3, requires_grad=True), torch.ones(4, 1, 3, requires_grad=True),
                  torch.ones(4, 2, requires_grad=True)]
        target = torch.zeros(4, 1, 4)
        lse = torch.ones(1, 4)
        lengths = torch.tensor([4], dtype=torch.int32)
        gradients = [torch.full_like(value, factor) for value, factor in zip(inputs, (2, 3, 5))]
        with patch.object(dense_api, "_cann_dsa_dense_indexer_kl_loss_grad") as kernel:
            kernel.return_value = (*gradients, torch.tensor([8.]))
            loss = dense_api.dsa_dense_indexer_kl_loss(
                *inputs, target, lse, query_lengths=lengths, key_lengths=lengths,
                layout="TND", loss_coefficient=.7,
            )
            (loss * 3).backward()
        torch.testing.assert_close(loss, torch.tensor(1.4))
        for value, gradient in zip(inputs, gradients):
            torch.testing.assert_close(value.grad, gradient * (.7 * 3 / 4))
        kernel.assert_called_once()

    def test_sparse_kl_routes_explicit_indexer_gradients_once(self):
        """Sparse KL keeps its own CANN contract and loss normalization."""
        inputs = [torch.ones(4, 2, 3, requires_grad=True), torch.ones(4, 1, 3, requires_grad=True),
                  torch.ones(4, 2, requires_grad=True)]
        teacher_query = torch.ones(4, 2, 3, requires_grad=True)
        teacher_key = torch.ones(4, 1, 3, requires_grad=True)
        stats = torch.ones(2, 4)
        lengths = torch.tensor([4], dtype=torch.int32)
        gradients = [torch.full_like(value, factor) for value, factor in zip(inputs, (2, 3, 5))]
        with patch.object(kl_ops, "_cann_dsa_sparse_indexer_kl_loss_grad") as kernel:
            kernel.return_value = (*gradients, torch.tensor([8.]))
            loss = kl_ops.dsa_sparse_indexer_kl_loss(
                *inputs, teacher_query, teacher_key, torch.zeros(4, 1, 4, dtype=torch.int32),
                stats, stats, None, None, lengths, lengths, .5, .7,
            )
            (loss * 3).backward()
        torch.testing.assert_close(loss, torch.tensor(1.4))
        for value, gradient in zip(inputs, gradients):
            torch.testing.assert_close(value.grad, gradient * (.7 * 3 / 4))
        self.assertIsNone(teacher_query.grad)
        self.assertIsNone(teacher_key.grad)
        kernel.assert_called_once()

    def test_existing_entrypoints_keep_tnd_and_padding_contracts(self):
        """Existing DSA modules can switch kernels without changing their inputs."""
        query = torch.zeros(2, 3, 2, 4, requires_grad=True)
        key = torch.zeros(2, 3, 1, 4, requires_grad=True)
        weights = torch.zeros(2, 3, 2, requires_grad=True)
        lengths = torch.tensor([3, 6])
        with patch(
                "hyper_parallel.components.functional.dsa_indexer._cann_dsa_indexer",
                return_value=(torch.zeros(6, 1, 2048, dtype=torch.int32), torch.zeros(6, 1, 2048)),
        ) as indexer:
            indices, iq, ik, iw = dsa_indexer(query, key, weights, lengths, lengths, 2048)
            self.assertEqual(iq.shape, (6, 2, 4))
            self.assertEqual(ik.shape, (6, 1, 4))
            self.assertEqual(iw.shape, (6, 2))
            self.assertTrue(iq.requires_grad)
            self.assertEqual(indexer.call_args.kwargs["layout"], "TND")
        qrope = torch.zeros(2, 3, 2, 2)
        krope = torch.zeros(2, 3, 1, 2)
        with patch(
                "hyper_parallel.components.functional.dsa_sparse_attention._cann_dsa_sparse_attention",
        ) as attention:
            attention.return_value = (torch.ones(6, 2, 4), torch.ones(1, 6, 2), torch.ones(1, 6, 2))
            output, _, _ = dsa_sparse_attention(query, key, qrope, krope, indices, .5, lengths, lengths)
        self.assertEqual(output.shape, (2, 3, 2, 6))
        torch.testing.assert_close(output[..., :4], torch.ones_like(output[..., :4]))
        torch.testing.assert_close(output[..., 4:], torch.zeros_like(output[..., 4:]))
