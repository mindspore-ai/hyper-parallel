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
"""Numerical parity checks for compressed-key selection in CSA2."""

from __future__ import annotations

import unittest
from unittest import mock

import torch

from hyper_parallel.components.modules import shared_compressed_dsa_attention as csa


def _advanced_gather(key: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Evaluate the original per-batch advanced indexing as an independent oracle."""
    batch_indices = torch.arange(key.shape[0], device=key.device).view(-1, 1, 1)
    return key[batch_indices, indices]


class TestCompressedKeyGather(unittest.TestCase):
    """Check both CSA2 consumers against their original advanced-indexing path."""

    def test_gather_matches_values_and_gradients(self) -> None:
        """Batch offsets, repeated ids, dtypes and noncontiguous keys retain exact values."""
        for batch_size in (1, 2):
            for dtype in (torch.float32, torch.bfloat16):
                for transposed in (False, True):
                    with self.subTest(batch_size=batch_size, dtype=dtype, transposed=transposed):
                        generator = torch.Generator().manual_seed(47)
                        shape = (batch_size, 5, 7) if transposed else (batch_size, 7, 5)
                        key = torch.randn(shape, generator=generator).to(dtype)
                        if transposed:
                            key = key.transpose(1, 2)
                        key.requires_grad_()
                        indices = torch.tensor(
                            [[[0, 6, 0], [3, 3, 1]], [[6, 2, 0], [1, 6, 6]]][:batch_size],
                            dtype=torch.int64,
                        )

                        actual = csa._gather_compressed_keys(key, indices)
                        expected = _advanced_gather(key, indices)
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                        actual_grad = torch.autograd.grad(actual.float().square().sum(), key, retain_graph=True)[0]
                        expected_grad = torch.autograd.grad(expected.float().square().sum(), key)[0]
                        torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)

    def test_candidate_topk_matches_advanced_indexing(self) -> None:
        """Causal masks, missing blocks, duplicate blocks and query chunks retain selected ids."""
        generator = torch.Generator().manual_seed(53)
        query = torch.randn((2, 9, 2, 4), generator=generator)
        key = torch.randn((2, 7, 4), generator=generator)
        merge_weight = torch.randn((2, 9, 2), generator=generator)
        blocks = torch.tensor(
            [[[0, 1, -1], [0, 0, 2], [1, 2, 3]] * 3, [[-1, 3, 0], [1, 1, 2], [3, 0, -1]] * 3]
        )
        minimum = torch.tensor([[0, 0, 1, 0, 2, 1, 2, 0, 1], [0, 1, 0, 2, 1, 0, 0, 2, 1]])
        kwargs = {
            "compress_ratio": 2,
            "sparse_count": 3,
            "block_size": 2,
            "query_chunk_size": 4,
            "minimum_key_indices": minimum,
        }

        with mock.patch.object(csa, "_gather_compressed_keys", _advanced_gather):
            expected = csa.compressed_candidate_topk(query, key, merge_weight, blocks, **kwargs)
        actual = csa.compressed_candidate_topk(query, key, merge_weight, blocks, **kwargs)

        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertTrue((actual == -1).any())

    def test_indexer_kl_matches_loss_and_all_indexer_gradients(self) -> None:
        """Both key gathers preserve the KL objective and its three precomputed gradients."""
        selected = torch.tensor(
            [
                [[0, 1, 1], [2, -1, 3], [-1, -1, -1], [5, 0, 2], [3, 3, 4]],
                [[5, 5, 1], [0, -1, 2], [4, 3, 1], [-1, -1, -1], [2, 2, 0]],
            ],
            dtype=torch.int32,
        )
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                generator = torch.Generator().manual_seed(59)
                inputs = [
                    torch.randn(shape, generator=generator).to(dtype).requires_grad_()
                    for shape in ((2, 5, 2, 4), (2, 6, 4), (2, 5, 2))
                ]
                attention_query = torch.randn((2, 3, 5, 6), generator=generator).to(dtype)
                compressed_key = torch.randn((2, 6, 6), generator=generator).to(dtype)
                sinks = torch.randn((3,), generator=generator).to(dtype)
                args = (*inputs, attention_query, compressed_key, selected, sinks)
                kwargs = {"attention_scale": 0.4, "loss_coeff": 0.07, "query_chunk_size": 2}

                with mock.patch.object(csa, "_gather_compressed_keys", _advanced_gather):
                    expected = csa.shared_compressed_indexer_kl_loss(*args, **kwargs)
                actual = csa.shared_compressed_indexer_kl_loss(*args, **kwargs)
                expected_grads = torch.autograd.grad(expected, inputs, retain_graph=True)
                actual_grads = torch.autograd.grad(actual, inputs)

                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                for actual_grad, expected_grad in zip(actual_grads, expected_grads):
                    torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)
