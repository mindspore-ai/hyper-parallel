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
"""CPU-only boundary and P2P lifetime tests for Ring Attention."""

from types import SimpleNamespace
import importlib
import unittest
import weakref
from unittest.mock import Mock, patch

import torch

from hyper_parallel.distributed.context_parallel.ring_attention import (
    _merge_blocks, _RingContext, _RingExchange, _ring_context, ring_attention, ulysses_ring_attention,
)

_RING_MODULE = importlib.import_module("hyper_parallel.distributed.context_parallel.ring_attention")


class TestRingArguments(unittest.TestCase):
    """Reject unsupported contracts before reaching a kernel or collective."""

    def setUp(self) -> None:
        """Allocate only CPU tensors; valid calls are exercised by the NPU ST."""
        self.query = torch.zeros(1, 4, 16, 64, dtype=torch.float16)
        self.key = torch.zeros(1, 2, 16, 64, dtype=torch.float16)
        self.value = torch.zeros_like(self.key)

    def test_unsupported_options(self) -> None:
        cases = (
            ({"input_layout": "TNHD"}, "layouts"),
            ({"causal": 1}, "causal"),
            ({"overlap": 1}, "overlap"),
            ({"dropout_p": 0.1}, "dropout_p=0"),
            ({"scale": 0}, "scale"),
            ({"scale": -0.25}, "scale"),
            ({"scale": float("nan")}, "scale"),
            ({"scale": float("inf")}, "scale"),
            ({"scale": True}, "scale"),
            ({"scale": "0.25"}, "scale"),
        )
        for options, message in cases:
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, message):
                ring_attention(self.query, self.key, self.value, **options)

    def test_invalid_tensor_shapes_and_dtypes(self) -> None:
        cases = (
            ((None, self.key, self.value), "four-dimensional"),
            ((self.query.squeeze(0), self.key, self.value), "four-dimensional"),
            ((self.query[:, :, :0], self.key, self.value), "nonempty"),
            ((self.query[:, :3], self.key, self.value), "divisible"),
            ((self.query[:, :, :8], self.key, self.value), "local sequence"),
            ((self.query, self.key, self.value[..., :32]), "equal shapes"),
            ((self.query.float(), self.key.float(), self.value.float()), "FP16 or BF16"),
            ((self.query, self.key.bfloat16(), self.value), "share"),
        )
        for tensors, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                ring_attention(*tensors)

    def test_cpu_backend_is_explicitly_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "same NPU"):
            ring_attention(self.query, self.key, self.value)


class TestRingCommunication(unittest.TestCase):
    """Check group-relative peers and receive readiness without initializing distributed state."""

    @patch.object(_RING_MODULE, "dist")
    def test_noncontiguous_group_peers(self, mock_dist: Mock) -> None:
        mesh = Mock(ndim=1)
        mesh.size.return_value = 4
        mesh.get_local_rank.return_value = 2
        mock_dist.get_rank.return_value = 2
        mock_dist.get_world_size.return_value = 4
        mock_dist.get_global_rank.side_effect = lambda _, rank: (1, 3, 5, 7)[rank]
        ring = _ring_context(mesh)
        self.assertEqual((ring.rank, ring.next_rank, ring.previous_rank), (2, 7, 3))
        self.assertIs(ring.group, mesh.get_group.return_value)
        mock_dist.new_group.assert_not_called()

    @patch.object(_RING_MODULE, "dist")
    def test_mismatched_mesh_order_is_rejected(self, mock_dist: Mock) -> None:
        mesh = Mock(ndim=1)
        mesh.size.return_value = 4
        mesh.get_local_rank.return_value = 1
        mock_dist.get_world_size.return_value = 4
        mock_dist.get_rank.return_value = 2
        with self.assertRaisesRegex(ValueError, "rank order"):
            _ring_context(mesh)
        with self.assertRaisesRegex(ValueError, "one-dimensional"):
            _ring_context(Mock(ndim=2))

    @patch.object(_RING_MODULE, "dist")
    def test_single_rank_requires_no_communication(self, mock_dist: Mock) -> None:
        payload = (torch.ones(4),)
        exchange = _RingExchange(payload, _ring_context(None))
        self.assertIs(exchange.rotate(payload), payload)
        mock_dist.isend.assert_not_called()
        mock_dist.irecv.assert_not_called()

    @patch.object(_RING_MODULE, "dist")
    def test_wait_before_receive_and_bounded_storage(self, mock_dist: Mock) -> None:
        process_group = object()
        for rank, next_rank, previous_rank in ((2, 7, 3), (3, 1, 5)):
            with self.subTest(rank=rank):
                order, waits, sent, received = [], [], [], []

                def _send(tensor, dst, group):
                    self.assertEqual(dst, next_rank)
                    self.assertIs(group, process_group)
                    sent.append(tensor)
                    order.append("send")
                    return SimpleNamespace(wait=lambda: waits.append("send"))

                def _receive(buffer, src, group):
                    self.assertEqual(src, previous_rank)
                    self.assertIs(group, process_group)
                    order.append("receive")
                    index = len(received)
                    received.append(buffer)

                    def _finish():
                        self.assertNotEqual(sent[index].data_ptr(), buffer.data_ptr())
                        buffer.copy_(sent[index] + 1)
                        waits.append("receive")

                    return SimpleNamespace(wait=_finish)

                mock_dist.isend.side_effect = _send
                mock_dist.irecv.side_effect = _receive
                current = (torch.zeros(8, 2)[:, 0], torch.full((4,), 10.0))
                exchange = _RingExchange(current, _RingContext(process_group, rank, 4, next_rank, previous_rank))
                pointers = (set(), set())
                for step in range(12):
                    current = exchange.rotate(current)
                    for tensor, addresses in zip(current, pointers):
                        addresses.add(tensor.data_ptr())
                    torch.testing.assert_close(current[0], torch.full((8,), step + 1.0))
                    torch.testing.assert_close(current[1], torch.full((4,), step + 11.0))
                expected_order = ["send", "receive"] if rank % 2 == 0 else ["receive", "send"]
                self.assertEqual(order, expected_order * 12)
                self.assertEqual(len(waits), 24)
                self.assertEqual(tuple(len(addresses) for addresses in pointers), (2, 2))


    def test_deferred_wait_retains_payload_and_guards_reuse(self) -> None:
        for count in (1, 2):
            waits = []

            def _send(*_args, **_kwargs):
                return SimpleNamespace(wait=lambda: waits.append("send"))

            def _receive(buffer, **_kwargs):
                def _finish():
                    buffer.fill_(3)
                    waits.append("receive")

                return SimpleNamespace(wait=_finish)

            tensor = torch.ones(8)
            tensors = (tensor,) if count == 1 else (tensor, torch.ones(4))
            reference = weakref.ref(tensor)
            exchange = _RingExchange(tensors, _RingContext(object(), 0, 2, 1, 1))
            with (patch.object(_RING_MODULE.dist, "isend", new=_send),
                  patch.object(_RING_MODULE.dist, "irecv", new=_receive)):
                exchange.launch(tensors)
                del tensor, tensors
                self.assertEqual(waits, [], "launch must not wait before independent compute can run")
                self.assertIsNotNone(reference(), "pending sends must keep caller storage alive")
                with self.assertRaisesRegex(RuntimeError, "pending Ring exchange"):
                    exchange.launch((torch.zeros(8),))
                received = exchange.wait()
                self.assertEqual(len(received), count)
                torch.testing.assert_close(received[0], torch.full((8,), 3.0))
                self.assertEqual(waits, ["send", "receive"])
                self.assertIsNone(reference(), "completed exchange must release the send payload")
                with self.assertRaisesRegex(RuntimeError, "no pending"):
                    exchange.wait()


class TestRingSoftmaxMerge(unittest.TestCase):
    """Exercise stable normalization with logits that overflow a naive exponential merge."""

    def test_large_statistics_use_stable_weights(self) -> None:
        maximum = torch.full((1, 2, 3, 8), 1000.0)
        denominator = torch.ones_like(maximum)
        for layout in ("BNSD", "BSND"):
            with self.subTest(layout=layout):
                shape = (1, 2, 3, 4) if layout == "BNSD" else (1, 3, 2, 4)
                first, second = torch.ones(shape), torch.full(shape, 3.0)
                output, merged_max, merged_sum = _merge_blocks(
                    (first, maximum, denominator), (second, maximum + 1, denominator), layout,
                )
                weights = torch.tensor([1000.0, 1001.0], dtype=torch.float64).softmax(0)
                expected = torch.full(shape, float(weights[0] + 3 * weights[1]))
                torch.testing.assert_close(output, expected)
                torch.testing.assert_close(merged_max, maximum + 1)
                torch.testing.assert_close(merged_sum, denominator * (1 + torch.exp(torch.tensor(-1.0))))


class TestUlyssesRingArguments(unittest.TestCase):
    """Invalid decomposition must fail before mesh creation or communication."""

    def test_invalid_degrees_before_collectives(self) -> None:
        mesh = Mock(ndim=1)
        mesh.size.return_value = 4
        tensor = torch.zeros(1, 4, 16, 64, dtype=torch.float16)
        cases = ((None, "integer"), (True, "integer"), (2.0, "integer"), ("2", "integer"),
                 (0, "positive divisor"), (-1, "positive divisor"), (3, "positive divisor"),
                 (8, "positive divisor"))
        with (patch.object(_RING_MODULE, "_build_hybrid_cp_submeshes") as build,
              patch.object(_RING_MODULE, "ulysses_seq_to_head") as exchange):
            for degree, message in cases:
                with self.subTest(degree=degree), self.assertRaisesRegex(ValueError, message):
                    ulysses_ring_attention(tensor, tensor, tensor, mesh, ulysses_degree=degree)
            with self.assertRaisesRegex(ValueError, r"CP size \(1\)"):
                ulysses_ring_attention(tensor, tensor, tensor, ulysses_degree=2)
            build.assert_not_called()
            exchange.assert_not_called()

    def test_multidimensional_mesh_is_rejected(self) -> None:
        tensor = torch.zeros(1, 4, 16, 64, dtype=torch.float16)
        with self.assertRaisesRegex(ValueError, "one-dimensional CP mesh"):
            ulysses_ring_attention(tensor, tensor, tensor, Mock(ndim=2), ulysses_degree=2)
