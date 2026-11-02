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
"""CPU checks for target-peer token deduplication metadata."""

import unittest
from unittest.mock import patch

import numpy as np
import torch

from hyper_parallel.distributed.expert_parallel.moonep import token_dedup
from hyper_parallel.distributed.expert_parallel.moonep.token_dedup import (
    failed_local_dedup, finish_dedup, prepare_local_dedup,
)


class TestLocalTokenDedup(unittest.TestCase):
    """The bounded count table must preserve sorted unique and inverse semantics."""

    def _check_reference(self, ranks: int, top_k: int, tokens: int, reorder: bool) -> None:
        rng = np.random.default_rng(71)
        rows = tokens * top_k
        order = rng.permutation(rows).astype(np.int64)
        send = np.bincount(rng.integers(ranks, size=rows), minlength=ranks)
        source_rows = rng.permutation(rows).astype(np.int64) if reorder else None
        choices = order if source_rows is None else order[source_rows]
        peers = np.repeat(np.arange(ranks, dtype=np.int64), send)
        stride = rows + 1
        keys = peers * stride + choices // top_k
        unique, inverse = np.unique(keys, return_inverse=True)
        counts = np.bincount(unique // stride, minlength=ranks)
        offsets = np.cumsum(counts) - counts

        local = prepare_local_dedup(torch.from_numpy(order), send, send, top_k,
                                    source_rows=source_rows, forward_ratio=1.0)

        self.assertEqual(local.unique_send, tuple(counts.tolist()))
        np.testing.assert_array_equal(local.unique_tokens.numpy(), unique % stride)
        np.testing.assert_array_equal(local.source_choices.numpy(), choices)
        np.testing.assert_array_equal(local.packet.numpy()[np.arange(rows) + peers + 1],
                                      inverse - offsets[peers])
        np.testing.assert_array_equal(local.packet.numpy()[np.cumsum(send + 1) - send - 1], counts)

    def test_bounded_table_matches_sort_at_ep16_topk8(self) -> None:
        """Exercise the dense-key fast path with realistic route dimensions."""
        self._check_reference(16, 8, 4096, False)
        self._check_reference(16, 8, 128, True)

    def test_sparse_key_domain_and_empty_route_match_sort(self) -> None:
        """Large key domains use the sorting fallback; empty routes stay valid."""
        self._check_reference(16, 1, 32, False)
        self._check_reference(1, 1, 0, False)


    def test_mixed_peer_modes_share_one_metadata_packet(self) -> None:
        """The peer threshold preserves compact and dense source rows in one route."""
        order = torch.tensor([0, 1, 2, 4, 3, 5, 6, 7])
        local = prepare_local_dedup(order, (2, 2, 4), (2, 2, 4),
                                    2, forward_ratio=0.5, backward_ratio=1.0)
        self.assertEqual(local.unique_send, (1, 2, 4))
        self.assertEqual(local.backward_send, (1, 2, 3))
        np.testing.assert_array_equal(local.unique_tokens.numpy(), [0, 1, 2, 1, 2, 3, 3])
        np.testing.assert_array_equal(local.backward_tokens.numpy(), [0, 1, 2, 1, 2, 3])
        np.testing.assert_array_equal(local.packet.numpy()[[0, 3, 6]], [1, -3, -4])
        np.testing.assert_array_equal(local.packet.numpy()[[1, 2, 4, 5, 7, 8, 9, 10]],
                                      [0, 0, 0, 1, 0, 1, 2, 2])
        forward_dense = prepare_local_dedup(order, (2, 2, 4), (2, 2, 4),
                                            2, forward_ratio=0.0, backward_ratio=1.0)
        backward_dense = prepare_local_dedup(order, (2, 2, 4), (2, 2, 4),
                                             2, forward_ratio=1.0, backward_ratio=0.0)
        self.assertEqual(forward_dense.unique_send, (2, 2, 4))
        self.assertEqual(forward_dense.backward_send, (1, 2, 3))
        self.assertEqual(backward_dense.unique_send, (1, 2, 3))
        self.assertEqual(backward_dense.backward_send, (2, 2, 4))

    def test_failed_local_packet_reaches_every_peer(self) -> None:
        """An invalid header travels through the normal fixed-size metadata exchange."""
        local = failed_local_dedup((2, 0, 3), (1, 2, 2))
        self.assertEqual(local.packet_send, (3, 1, 4))
        np.testing.assert_array_equal(local.packet.numpy()[[0, 3, 4]], [-1, -1, -1])

    def test_fp32_scores_preserve_bits_and_gradients_in_bf16_payload(self) -> None:
        """Feature: FP32 router-score transport with BF16 activations.

        Description: Carry each score through two BF16 payload lanes and differentiate the result.
        Expectation: Both received score bits and score gradients exactly match the source values.
        """
        order = torch.arange(4, dtype=torch.int64)
        local = prepare_local_dedup(order, (4,), (4,), 2, forward_ratio=1.0)
        plan = finish_dedup(local, None, device=torch.device("cpu"), width=16, score_lanes=2)
        hidden = torch.randn(2, 16, dtype=torch.bfloat16, requires_grad=True)
        scores = torch.tensor([[0.1234567, 0.8765433], [0.3333333, 0.6666667]],
                              dtype=torch.float32, requires_grad=True)
        with patch.object(token_dedup, "ep_all_to_all", side_effect=lambda payload, *_args: payload.clone()):
            received, received_scores = plan.dispatch_with_scores(hidden, scores)
            (received.float() * received_scores[:, None]).sum().backward()
        torch.testing.assert_close(received_scores, scores.flatten(), rtol=0, atol=0)
        torch.testing.assert_close(scores.grad, received.float().sum(-1).reshape_as(scores), rtol=0, atol=0)
