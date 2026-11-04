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
"""CPU coverage for prepared sparse transport and cross-stream ownership."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from threading import RLock
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from hyper_parallel.core.multicore.modules.routed_experts import readiness, transport
from hyper_parallel.core.multicore.modules.routed_experts.descriptor import RouteDescriptor, RouteEdge
from hyper_parallel.core.multicore.modules.routed_experts.workspace import RingSpec, RingWorkspace
from tests.common.mark_utils import arg_mark


class _Event:
    """Record stream identities without touching an accelerator runtime."""

    def __init__(self) -> None:
        """Create an empty event-recording history."""
        self.records = []

    def record(self, stream: _Stream) -> None:
        """Capture the event's enqueue stream."""
        self.records.append(stream)
        stream.operations.append(("record", self))


class _Stream:
    """Expose only the stream ordering used by transport and workspace."""

    def __init__(self, name: str) -> None:
        """Create a named stream with an independent operation history."""
        self.name = name
        self.operations = []
        self.wait_records = []

    def wait_event(self, event: _Event) -> None:
        """Capture an event dependency without synchronizing the CPU."""
        self.operations.append(("wait", event))
        self.wait_records.append((event, len(event.records)))


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
class TestPreparedExchange(unittest.TestCase):
    """Check real preparation/sequence logic with only hardware boundaries mocked."""

    def setUp(self) -> None:
        """Provide CPU storage and explicit fake streams for each independent test."""
        self.compute_stream = _Stream("compute")
        self.current = self.compute_stream
        npu = SimpleNamespace(Event=_Event, current_stream=lambda _device: self.current,
                              stream=self._stream, device=lambda _device: nullcontext())
        self.enterContext(patch.object(torch, "npu", npu, create=True))
        self.enterContext(patch.object(torch.Tensor, "record_stream"))
        self.native = self.enterContext(patch.object(
            torch.ops.hyper_parallel, "routed_experts_transport", create=True))
        self.enterContext(patch.object(transport, "_load_native"))
        self.enterContext(patch.object(transport, "vector_core_count", return_value=48))
        self.enterContext(patch.object(transport, "upload_metadata", side_effect=self._upload))
        self.enterContext(patch.object(readiness, "upload_metadata", side_effect=self._upload))
        self.workspace = self._workspace()
        self.route = RouteDescriptor(
            rank=0, ranks=2, source_rows=3, received_rows=4, active_experts=(0, 1), boundaries=(2, 4),
            outgoing=(RouteEdge(0, 0, 0, 0, 0, 1), RouteEdge(0, 1, 1, 1, 0, 2)),
            incoming=(RouteEdge(0, 0, 0, 0, 0, 1), RouteEdge(1, 0, 0, 0, 1, 1),
                      RouteEdge(1, 0, 1, 1, 2, 2)))
        self.source = torch.empty((3, 16), dtype=torch.bfloat16)

    @contextmanager
    def _stream(self, stream):
        previous, self.current = self.current, stream
        try:
            yield
        finally:
            self.current = previous

    @staticmethod
    def _upload(value, device):
        return value.to(device=device)

    @staticmethod
    def _workspace(ranks: int = 2, hidden: int = 16):
        workspace = RingWorkspace.__new__(RingWorkspace)
        spec = RingSpec(ranks=ranks, hidden=hidden, block_rows=2)
        workspace.__dict__.update(
            spec=spec, rank=0, device=torch.device("cpu"), closed=False, poisoned=False, closing=False,
            metadata={}, send_sequence=[[0] * ranks for _ in range(2)],
            receive_sequence=[[0] * ranks for _ in range(2)],
            data=torch.empty(spec.data_shape, dtype=torch.bfloat16),
            signals=torch.zeros(spec.signal_shape, dtype=torch.int32),
            headers=torch.empty(spec.header_shape, dtype=torch.int64), transport_stream=_Stream("transport"),
            last_progress=None, _lock=RLock(), _completion=None)

        def _preflight(error):
            if error is not None:
                workspace.poisoned = True
                raise RuntimeError("preflight failed") from error

        workspace.check_preflight = Mock(side_effect=_preflight)
        return workspace


    def test_counted_dispatch_and_reduced_return_share_token_metadata(self) -> None:
        """Forward token ownership is retained for a prepared reverse reduction."""
        workspace = self._workspace(hidden=32)
        source = torch.empty((self.route.source_rows, 32), dtype=torch.bfloat16)
        mapping = torch.arange(self.route.source_rows, dtype=torch.int64)
        group_metadata = torch.empty((self.route.received_rows, 8), dtype=torch.int64)

        incoming = transport.prepare_exchange(source, self.route, workspace, mapping=mapping,
                                              dedup=True, group_metadata=group_metadata)
        returned = transport.prepare_exchange(incoming.value, self.route, workspace, reverse=True,
                                              reduce=True, group_metadata=group_metadata)

        self.assertIs(incoming.group_metadata, group_metadata)
        self.assertEqual(incoming.dedup_state.shape, (2, 2, 8))
        self.assertFalse(incoming.reduce)
        self.assertIs(returned.group_metadata, group_metadata)
        self.assertIsNone(returned.dedup_state)
        self.assertTrue(returned.reduce)
        self.native.assert_not_called()


    def test_metadata_matches_explicit_blocks_in_both_directions(self):
        """Vectorized expansion preserves peer order, odd tails, and sequence bases."""
        self.workspace.send_sequence = [[3, 19], [7, 11]]
        self.workspace.receive_sequence = [[13, 29], [17, 23]]
        for reverse in (False, True):
            with self.subTest(reverse=reverse):
                outgoing, incoming = self.route.outgoing, self.route.incoming
                if reverse:
                    outgoing, incoming = tuple(RouteEdge(e.target, e.source, e.expert, e.target_row,
                                                        e.source_row, e.rows) for e in incoming), tuple(
                        RouteEdge(e.target, e.source, e.expert, e.target_row, e.source_row, e.rows) for e in outgoing)
                sent, received, expected_offsets = [], [], []
                for peer in range(self.route.ranks):
                    send_begin, receive_begin = len(sent), len(received)
                    sent.extend(block for edge in outgoing if edge.target == peer for block in edge.blocks(2))
                    received.extend(block for edge in incoming if edge.source == peer for block in edge.blocks(2))
                    expected_offsets.append((send_begin, len(sent), receive_begin, len(received),
                                             self.workspace.send_sequence[int(reverse)][peer],
                                             self.workspace.receive_sequence[int(reverse)][peer]))
                send, receive, offsets = transport.compile_transport_metadata(self.route, self.workspace, reverse)
                for expected, actual in ((sent, send), (received, receive)):
                    packed = torch.tensor([(e.source, e.target, e.expert, e.source_row, e.target_row, e.rows)
                                           for e in expected], dtype=torch.int64).reshape(-1, 6)
                    torch.testing.assert_close(actual, packed)
                torch.testing.assert_close(offsets, torch.tensor(expected_offsets, dtype=torch.int64))


    def test_both_directions_prepare_before_incoming_submission(self):
        """An opposite-direction submission cannot stale a prepared reverse phase."""
        incoming = transport.prepare_exchange(self.source, self.route, self.workspace)
        producer = readiness.ExpertReadiness.create(torch.ones((1, 2), dtype=torch.int32), torch.device("cpu"))
        returned = transport.prepare_exchange(incoming.value, self.route, self.workspace, reverse=True,
                                              output=self.source, dependency=producer)

        first = transport.submit_prepared(incoming, self.workspace)
        second = transport.submit_prepared(returned, self.workspace)

        self.assertIs(first.value, incoming.value)
        self.assertIs(second.value, self.source)
        self.assertIsNone(second.ready)
        self.assertEqual(self.workspace.send_sequence, [[0, 1], [0, 2]])
        self.assertEqual(self.workspace.receive_sequence, [[0, 2], [0, 1]])
        self.assertEqual(self.native.call_count, 2)
        self.workspace.check_preflight.assert_not_called()
        self.assertEqual(incoming.complete.records, [self.workspace.transport_stream])
        self.assertEqual(returned.complete.records, [self.workspace.transport_stream])
        self.assertEqual(returned.source_admitted.records, [])
        self.assertIn(("wait", producer.admitted), self.workspace.transport_stream.operations)
        self.assertIn(("wait", producer.poll_after.event), self.workspace.transport_stream.operations)


    def test_stale_sequence_is_rejected(self):
        """Even external directional sequence changes invalidate a preparation."""
        prepared = transport.prepare_exchange(self.source, self.route, self.workspace)
        self.workspace.send_sequence[0][1] += 1

        with self.assertRaisesRegex(RuntimeError, "stale directional sequences"):
            transport.submit_prepared(prepared, self.workspace)

        self.native.assert_not_called()


    def test_prepared_reverse_joins_latest_producer_poll_record_before_launch(self):
        """Preparation stores the event; submission waits its post-producer-setup generation."""
        source = torch.empty((4, 16), dtype=torch.bfloat16)
        producer = readiness.ExpertReadiness.create(torch.ones((1, 2), dtype=torch.int32), torch.device("cpu"))
        prepared = transport.prepare_exchange(source, self.route, self.workspace, reverse=True, dependency=producer)
        self.assertEqual(self.workspace.transport_stream.operations, [])
        producer_stream = _Stream("producer")
        producer.poll_after.event.record(producer_stream)
        self.native.side_effect = lambda *_args: self.current.operations.append(("transport", None))

        transport.submit_prepared(prepared, self.workspace)

        stream = self.workspace.transport_stream
        self.assertIn((producer.admitted, 1), stream.wait_records)
        self.assertIn((producer.poll_after.event, 2), stream.wait_records)
        self.assertLess(stream.operations.index(("wait", producer.poll_after.event)),
                        stream.operations.index(("transport", None)))
        self.assertEqual(prepared.source_admitted.records, [])
        self.assertNotIn(("wait", prepared.complete), producer_stream.operations)

    def test_unsubmitted_linear_producer_is_rejected_before_payload_mutation(self):
        """A wait cannot accidentally bind the initial record of a future producer."""
        producer = readiness.ExpertReadiness.create(
            torch.ones((1, 2), dtype=torch.int32), torch.device("cpu"), deferred_poll=True)
        source = torch.empty((4, 16), dtype=torch.bfloat16)
        prepared = transport.prepare_exchange(source, self.route, self.workspace, reverse=True, dependency=producer)
        self.assertFalse(producer.poll_after.queued)

        with self.assertRaisesRegex(ValueError, "producer to be submitted first"):
            transport.submit_prepared(prepared, self.workspace)

        self.native.assert_not_called()
        self.assertFalse(prepared.submitted)
        self.assertEqual(self.workspace.send_sequence, [[0, 0], [0, 0]])
        self.assertEqual(self.workspace.receive_sequence, [[0, 0], [0, 0]])
        self.assertEqual(prepared.complete.records, [])
        self.assertIsNone(self.workspace.last_progress)
        self.assertNotIn(("wait", producer.poll_after.event), self.workspace.transport_stream.operations)
        self.assertEqual(self.workspace.transport_stream.operations, [])
        self.assertFalse(self.workspace.poisoned)


    def test_cached_host_metadata_uses_current_directional_sequences(self):
        """Neither direction captures mutable sequence bases in its block template."""
        transport.compile_transport_metadata(self.route, self.workspace, True)
        self.workspace.send_sequence = [[11, 13], [17, 19]]
        self.workspace.receive_sequence = [[23, 29], [31, 37]]
        for reverse in (False, True):
            with self.subTest(reverse=reverse):
                _, _, offsets = transport.compile_transport_metadata(self.route, self.workspace, reverse)
                expected = torch.tensor(list(zip(self.workspace.send_sequence[int(reverse)],
                                                 self.workspace.receive_sequence[int(reverse)])))
                torch.testing.assert_close(offsets[:, 4:], expected)
