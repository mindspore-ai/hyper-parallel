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

"""Packet bit-preservation, autograd and token/destination dedup invariants."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from hyper_parallel.distributed.expert_parallel.moonep_collective_dedup import (
    _unique_sends,
    collective_token_forward,
)
from hyper_parallel.distributed.expert_parallel.moonep_collective_packet import (
    PackedDispatch,
    _layouts,
    _pack,
    _unpack,
    packet_layout,
)


class TestDedupPacket(unittest.TestCase):
    """Test mixed-dtype packets without initializing a process group."""

    def test_mixed_dtype_alignment_and_exact_roundtrip(self):
        for hidden_dtype in (torch.bfloat16, torch.float16, torch.float32, torch.float64):
            for hidden_size in (1, 3, 8):
                with self.subTest(dtype=hidden_dtype, hidden_size=hidden_size):
                    unique, assignments = [0, 2, 1], [0, 5, 2]
                    hidden = torch.arange(3 * hidden_size).reshape(3, hidden_size).to(hidden_dtype)
                    weights = torch.tensor([.123456789, 1e-9, 1e-30, -.012345678, 5., 0., 9.], dtype=torch.float32)
                    mapping = torch.tensor([0, 0, 1, 1, 1, 0, 0])
                    layouts = _layouts(unique, assignments, hidden_size, hidden_dtype, weights.dtype)
                    packed = _pack(hidden, weights, mapping, unique, assignments, layouts)
                    received, route, expansion = _unpack(packed, unique, assignments, layouts,
                                                         hidden_size, hidden_dtype, weights.dtype)
                    self.assertTrue(torch.equal(received, hidden))
                    self.assertTrue(torch.equal(route.view(torch.int32), weights.view(torch.int32)))
                    self.assertTrue(torch.equal(expansion, torch.tensor([0, 0, 1, 1, 1, 2, 2])))
                    for layout in layouts:
                        self.assertEqual(layout.weight_offset % weights.element_size(), 0)
                        self.assertEqual(layout.index_offset % 8, 0)
                        self.assertEqual(layout.total_bytes % 8, 0)

    def test_packed_exchange_gradcheck(self):
        def exchange(output, tensor, *_args, **_kwargs):
            output.copy_(tensor)

        hidden = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
        weights = torch.randn(4, dtype=torch.float64, requires_grad=True)
        mapping = torch.tensor([0, 1, 0, 1])
        with patch("hyper_parallel.distributed.expert_parallel.moonep_collective_packet.dist.all_to_all_single",
                   side_effect=exchange):
            self.assertTrue(torch.autograd.gradcheck(
                lambda x, w: PackedDispatch.apply(x, w, mapping, [2], [2], [4], [4], None)[:2],
                (hidden, weights)))

    def test_deduplicate_only_within_destination(self):
        order = torch.tensor([0, 2, 4, 6, 1, 3, 5, 7])
        tokens, inverse, unique = _unique_sends(order, [4, 4], 4)
        self.assertEqual(unique, [2, 2])
        self.assertEqual(tokens.tolist(), [0, 1, 0, 1])
        self.assertEqual(inverse.tolist(), [0, 0, 1, 1, 0, 0, 1, 1])
        self.assertEqual(_unique_sends(torch.empty(0, dtype=torch.int64), [0, 0], 4)[2], [0, 0])

    def test_full_dedup_gradient_and_single_packed_dispatch(self):
        experts = SimpleNamespace(
            local_expert_count=2,
            gate_up_proj=torch.randn(2, 4, 2, dtype=torch.float64, requires_grad=True),
            down_proj=torch.randn(2, 2, 2, dtype=torch.float64, requires_grad=True))
        calls = []

        def gather(outputs, tensor, **_kwargs):
            outputs[0].copy_(tensor)

        def exchange(output, tensor, *_args, **_kwargs):
            calls.append(tensor.dtype)
            output.copy_(tensor)

        states = torch.randn(2, 2, dtype=torch.float64, requires_grad=True)
        weights = torch.randn(2, 2, dtype=torch.float64, requires_grad=True)
        ids = torch.tensor([[0, 1], [1, 0]])
        diagnostics = {}
        with patch("hyper_parallel.distributed.expert_parallel.moonep_collective.dist",
                   get_world_size=lambda _g: 1, get_rank=lambda _g: 0,
                   all_gather=gather, all_to_all_single=exchange) as mock_dist, \
                patch("hyper_parallel.distributed.expert_parallel.moonep_collective_dedup.dist", mock_dist), \
                patch("hyper_parallel.distributed.expert_parallel.moonep_collective_packet.dist", mock_dist):
            collective_token_forward(experts, states, ids, weights, group=None, diagnostics=diagnostics)
            self.assertEqual(calls.count(torch.uint8), 1)
            self.assertEqual(diagnostics["assignment_rows"], 4)
            self.assertEqual(diagnostics["unique_token_rank_rows"], 2)
            self.assertTrue(torch.autograd.gradcheck(
                lambda x, w: collective_token_forward(experts, x, ids, w, group=None), (states, weights)))

    def test_payload_schema_mismatch_is_rejected_collectively(self):
        experts = SimpleNamespace(local_expert_count=1)
        ids = torch.tensor([[0]])

        def gather(outputs, metadata, **_kwargs):
            for output in outputs:
                output.copy_(metadata)
            outputs[1][-1] += 1

        with patch("hyper_parallel.distributed.expert_parallel.moonep_collective.dist",
                   get_world_size=lambda _g: 2, get_rank=lambda _g: 0, all_gather=gather), \
                self.assertRaisesRegex(ValueError, "matching hidden sizes and payload dtypes"):
            collective_token_forward(experts, torch.ones(1, 3), ids, torch.ones(1, 1), group=None)

    def test_packet_rejects_negative_dimensions(self):
        with self.assertRaises(ValueError):
            packet_layout(-1, 1, 3, torch.bfloat16, torch.float32)

    def test_packet_rejects_noninteger_dimensions_and_peer_mismatch(self):
        for value in (True, 1.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                packet_layout(value, 1, 3, torch.float32, torch.float32)
        with self.assertRaisesRegex(ValueError, 'matching peer counts'):
            _layouts([1], [1, 2], 3, torch.float32, torch.float32)
        with self.assertRaisesRegex(ValueError, 'floating fields'):
            packet_layout(1, 1, 3, torch.int64, torch.float32)
