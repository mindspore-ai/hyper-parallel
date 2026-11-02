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
from dataclasses import replace
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

    def test_prepare_reuses_one_stream_lookup_without_publication(self) -> None:
        """Metadata and frame fences share the caller stream within one preparation."""
        with patch.object(torch.npu, "current_stream", return_value=self.current) as query:
            prepared = transport.prepare_exchange(self.source, self.route, self.workspace, publish=False)

        query.assert_called_once_with(self.source.device)
        self.assertIsNone(prepared.ready)
        self.assertEqual(prepared.initialized.records, [self.compute_stream])
        self.assertEqual(self.workspace.metadata[(self.route, False)][3].records, [self.compute_stream])
        self.native.assert_not_called()
        self.assertEqual(self.workspace.transport_stream.operations, [])

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

    def test_counted_transport_rejects_missing_route_metadata(self) -> None:
        """A compressed direction must have the matching ownership inputs."""
        workspace = self._workspace(hidden=32)
        source = torch.empty((self.route.source_rows, 32), dtype=torch.bfloat16)
        group_metadata = torch.empty((self.route.received_rows, 8), dtype=torch.int64)

        with self.assertRaisesRegex(ValueError, "forward mapping"):
            transport.prepare_exchange(source, self.route, workspace, dedup=True, group_metadata=group_metadata)
        with self.assertRaisesRegex(ValueError, "forward group metadata"):
            transport.prepare_exchange(source, self.route, workspace, reverse=True, reduce=True)
        self.native.assert_not_called()

    def test_invalid_source_is_rejected_before_current_stream_lookup(self) -> None:
        """Local input validation still precedes any accelerator stream query."""
        with patch.object(torch.npu, "current_stream") as query:
            with self.assertRaises(ValueError):
                transport.prepare_exchange(self.source.float(), self.route, self.workspace)

        query.assert_not_called()
        self.native.assert_not_called()
        self.assertEqual(self.workspace.metadata, {})
        self.assertEqual(self.workspace.transport_stream.operations, [])

    def test_payload_uses_explicit_transport_stream_after_caller_queries(self) -> None:
        """T submission preserves caller and lease checks without another T lookup."""
        prepared = transport.prepare_exchange(self.source, self.route, self.workspace, publish=False)
        queried_streams = []
        submitted_streams = []

        def current_stream(_device):
            queried_streams.append(self.current)
            return self.current

        self.native.side_effect = lambda *_args: submitted_streams.append(self.current)
        with patch.object(torch.npu, "current_stream", side_effect=current_stream) as query:
            transport.submit_prepared(prepared, self.workspace)

        self.assertEqual(query.call_count, 2)
        self.assertEqual(queried_streams, [self.compute_stream, self.workspace.transport_stream])
        self.assertEqual(submitted_streams, [self.workspace.transport_stream])
        self.assertEqual(prepared.complete.records, [self.workspace.transport_stream])
        self.assertIs(self.current, self.compute_stream)

    def test_prepare_allocates_without_submitting(self):
        """All payload/control storage and admission fences exist before any T work."""
        prepared = transport.prepare_exchange(self.source, self.route, self.workspace)

        self.native.assert_not_called()
        self.workspace.check_preflight.assert_not_called()
        self.assertEqual(self.workspace.send_sequence, [[0, 0], [0, 0]])
        self.assertEqual(self.workspace.receive_sequence, [[0, 0], [0, 0]])
        self.assertEqual(prepared.value.shape, (4, 16))
        self.assertEqual(prepared.progress.shape, (4, 8))
        self.assertEqual(prepared.indices.numel(), 0)
        self.assertEqual(prepared.offsets.shape, (2, 6))
        torch.testing.assert_close(prepared.progress, torch.zeros_like(prepared.progress))
        torch.testing.assert_close(prepared.ready.expected, torch.tensor([[1, 0], [1, 1]], dtype=torch.int32))
        self.assertEqual(prepared.initialized.records, [self.compute_stream])
        self.assertEqual(prepared.ready.admitted.records, [self.compute_stream])
        self.assertEqual(prepared.ready.poll_after.event.records, [self.compute_stream])
        self.assertEqual(prepared.complete.records, [])
        self.assertEqual(self.workspace.transport_stream.operations, [])

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

    def test_metadata_sequence_overflow_rejects_without_publication(self):
        """A boundary crossing requires a drained reset before native submission."""
        self.workspace.send_sequence[0][1] = 2**30 - 1
        with self.assertRaisesRegex(RuntimeError, "collective drained reset"):
            transport.compile_transport_metadata(self.route, self.workspace, False)
        self.native.assert_not_called()

    def test_progress_allocates_one_cache_line_per_actual_worker(self):
        """Small roots split directions, while larger roots retain at most 48 workers."""
        for ranks, workers in ((1, 2), (16, 32), (24, 48), (25, 25), (48, 48)):
            with self.subTest(ranks=ranks, workers=workers):
                workspace = self._workspace(ranks)
                route = RouteDescriptor(0, ranks, 0, 0, (), (), (), ())

                prepared = transport.prepare_exchange(self.source[:0], route, workspace)

                self.assertEqual(prepared.progress.shape, (workers, 8))
                self.assertEqual(prepared.progress.dtype, torch.int64)
                self.assertEqual(prepared.progress.data_ptr() % 64, 0)
                self.assertEqual(prepared.progress.stride(0) * prepared.progress.element_size(), 64)
                torch.testing.assert_close(prepared.progress, torch.zeros((workers, 8), dtype=torch.int64))
                self.assertEqual(prepared.offsets.shape, (ranks, 6))
                self.native.assert_not_called()

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

    def test_submit_is_single_use(self):
        """Reusing the same descriptor fails before a second native launch."""
        prepared = transport.prepare_exchange(self.source, self.route, self.workspace)
        transport.submit_prepared(prepared, self.workspace)

        with self.assertRaisesRegex(RuntimeError, "only be submitted once"):
            transport.submit_prepared(prepared, self.workspace)

        self.native.assert_called_once()
        self.assertTrue(self.workspace.poisoned)

    def test_stale_sequence_is_rejected(self):
        """Even external directional sequence changes invalidate a preparation."""
        prepared = transport.prepare_exchange(self.source, self.route, self.workspace)
        self.workspace.send_sequence[0][1] += 1

        with self.assertRaisesRegex(RuntimeError, "stale directional sequences"):
            transport.submit_prepared(prepared, self.workspace)

        self.native.assert_not_called()

    def test_empty_phase_invalidates_same_direction_preparations(self):
        """A no-traffic phase still consumes its logical directional generation."""
        route = RouteDescriptor(0, 2, 0, 0, (), (), (), ())
        source = self.source[:0]
        first = transport.prepare_exchange(source, route, self.workspace)
        stale = transport.prepare_exchange(source, route, self.workspace)
        transport.submit_prepared(first, self.workspace)

        with self.assertRaisesRegex(RuntimeError, "stale directional generation"):
            transport.submit_prepared(stale, self.workspace)

        self.native.assert_called_once()
        self.assertEqual(self.workspace.send_sequence, [[0, 0], [0, 0]])
        self.assertEqual(self.workspace.receive_sequence, [[0, 0], [0, 0]])

    def test_reverse_dependency_can_bind_after_preparation(self):
        """Late producer binding uses its early event without recording the C tail."""
        source = torch.empty((4, 16), dtype=torch.bfloat16)
        prepared = transport.prepare_exchange(source, self.route, self.workspace, reverse=True)
        producer = readiness.ExpertReadiness.create(torch.ones((1, 2), dtype=torch.int32), torch.device("cpu"))

        transport.submit_prepared(prepared, self.workspace, dependency=producer)

        self.assertIsNotNone(prepared.slots)
        self.assertEqual(prepared.source_admitted.records, [])
        self.assertIs(self.native.call_args.args[16], producer.flags)
        self.assertIs(self.native.call_args.args[17], producer.expected)
        self.assertIn(("wait", prepared.initialized), self.workspace.transport_stream.operations)
        self.assertIn(("wait", producer.admitted), self.workspace.transport_stream.operations)
        self.assertIn(("wait", producer.poll_after.event), self.workspace.transport_stream.operations)

    def test_dependency_override_replaces_original_fence(self):
        """The submit override is the producer that native readiness actually polls."""
        original = readiness.ExpertReadiness.create(torch.ones((1, 2), dtype=torch.int32), torch.device("cpu"))
        replacement = readiness.ExpertReadiness.create(torch.ones((2, 2), dtype=torch.int32), torch.device("cpu"))
        source = torch.empty((4, 16), dtype=torch.bfloat16)
        prepared = transport.prepare_exchange(source, self.route, self.workspace, reverse=True, dependency=original)

        transport.submit_prepared(prepared, self.workspace, dependency=replacement)

        self.assertIs(self.native.call_args.args[16], replacement.flags)
        self.assertIn(("wait", replacement.admitted), self.workspace.transport_stream.operations)
        self.assertNotIn(("wait", original.admitted), self.workspace.transport_stream.operations)
        self.assertIn(("wait", replacement.poll_after.event), self.workspace.transport_stream.operations)
        self.assertNotIn(("wait", original.poll_after.event), self.workspace.transport_stream.operations)

    def test_cached_metadata_initialization_is_joined_on_other_stream(self):
        """Cache hits carry producer events for both edge and expert-slot uploads."""
        first = transport.prepare_exchange(self.source, self.route, self.workspace)
        edge_initialized = self.workspace.metadata[(self.route, False)][3]
        slots_initialized = self.workspace.metadata[(self.route, "readiness")][2]
        another = _Stream("another_compute")

        with self._stream(another):
            second = transport.prepare_exchange(self.source, self.route, self.workspace)

        self.assertIs(first.send, second.send)
        self.assertIs(first.slots, second.slots)
        self.assertIn(("wait", edge_initialized), another.operations)
        self.assertIn(("wait", slots_initialized), another.operations)
        self.assertEqual(second.initialized.records, [another])
        self.assertEqual(edge_initialized.records, [self.compute_stream])
        self.assertEqual(slots_initialized.records, [self.compute_stream])

    def test_prepare_rejects_bad_dependency_before_preflight(self):
        """Incorrect producer expert extents are local preparation failures."""
        producer = readiness.ExpertReadiness.create(torch.ones((1, 3), dtype=torch.int32), torch.device("cpu"))

        with self.assertRaisesRegex(ValueError, "compact active experts"):
            transport.prepare_exchange(self.source, self.route, self.workspace, dependency=producer)

        self.native.assert_not_called()
        self.workspace.check_preflight.assert_not_called()

    def test_begin_exchange_keeps_preflight_and_transport_stream(self):
        """The compatibility wrapper performs one collective preparation decision."""
        pending = transport.begin_exchange(self.source, self.route, self.workspace)

        self.workspace.check_preflight.assert_called_once_with(None)
        self.native.assert_called_once()
        self.assertEqual(pending.complete.records, [self.workspace.transport_stream])
        self.assertIs(pending.wait(), pending.value)
        self.assertIn(("wait", pending.complete), self.compute_stream.operations)

    def test_begin_exchange_converges_local_preparation_failure(self):
        """A malformed source still reaches the compatibility preflight once."""
        with self.assertRaisesRegex(RuntimeError, "preflight failed"):
            transport.begin_exchange(self.source.float(), self.route, self.workspace)

        self.workspace.check_preflight.assert_called_once()
        self.assertIsInstance(self.workspace.check_preflight.call_args.args[0], ValueError)
        self.native.assert_not_called()

    def test_exchange_rows_keeps_current_stream_behavior(self):
        """The original direct API submits on its caller's stream with preflight."""
        output = torch.empty((4, 16), dtype=torch.bfloat16)
        actual = transport.exchange_rows(self.source, self.route, self.workspace, output=output)

        self.assertIs(actual, output)
        self.workspace.check_preflight.assert_called_once_with(None)
        self.native.assert_called_once()
        self.assertEqual(self.workspace.transport_stream.operations, [])
        self.assertIsNone(self.native.call_args.args[18])

    def test_preparation_cannot_submit_to_another_workspace(self):
        """Window identity protects same-valued sequences in independent leases."""
        prepared = transport.prepare_exchange(self.source, self.route, self.workspace)

        with self.assertRaisesRegex(ValueError, "another workspace"):
            transport.submit_prepared(prepared, self._workspace())

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

    def test_begin_exchange_converges_pending_producer_before_payload(self):
        """A local producer-order mistake reaches the existing collective preflight."""
        producer = readiness.ExpertReadiness.create(
            torch.ones((1, 2), dtype=torch.int32), torch.device("cpu"), deferred_poll=True)
        source = torch.empty((4, 16), dtype=torch.bfloat16)

        with self.assertRaisesRegex(RuntimeError, "preflight failed"):
            transport.begin_exchange(source, self.route, self.workspace, reverse=True, dependency=producer)

        self.workspace.check_preflight.assert_called_once()
        error = self.workspace.check_preflight.call_args.args[0]
        self.assertIsInstance(error, ValueError)
        self.assertIn("producer to be submitted first", str(error))
        self.native.assert_not_called()
        self.assertEqual(self.workspace.transport_stream.operations, [])
        self.assertEqual(self.workspace.send_sequence, [[0, 0], [0, 0]])
        self.assertEqual(self.workspace.receive_sequence, [[0, 0], [0, 0]])

    def test_host_metadata_compiles_once_for_either_first_direction(self):
        """Forward and reverse share validated blocks regardless of first access."""
        for directions in ((False, True), (True, False)):
            with self.subTest(directions=directions):
                self.workspace.metadata.clear()
                with patch.object(transport, "_pack_blocks", wraps=transport._pack_blocks) as pack:
                    first = transport.compile_transport_metadata(self.route, self.workspace, directions[0])
                    second = transport.compile_transport_metadata(self.route, self.workspace, directions[1])
                self.assertEqual(pack.call_count, 2)
                self.assertIs(self.workspace.metadata[(self.route, "host_transport", 2)][0], self.route)
                self.assertEqual(first[0].shape, second[1].shape)
                self.assertEqual(first[1].shape, second[0].shape)
                torch.testing.assert_close(first[2][:, :2], second[2][:, 2:4])
                torch.testing.assert_close(first[2][:, 2:4], second[2][:, :2])

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

    def test_cached_host_metadata_rejects_current_sequence_overflow_and_types(self):
        """Warm blocks cannot bypass fresh int32 credit limits or integer checks."""
        transport.compile_transport_metadata(self.route, self.workspace, False)
        for direction in (0, 1):
            for attribute in ("send_sequence", "receive_sequence"):
                for value in (2**30 - 1, 2**30, -1, False, 0.0):
                    with self.subTest(direction=direction, attribute=attribute, value=value):
                        sequences = getattr(self.workspace, attribute)
                        previous = sequences[direction][1]
                        sequences[direction][1] = value
                        with self.assertRaisesRegex(RuntimeError, "collective drained reset"):
                            transport.compile_transport_metadata(self.route, self.workspace, bool(direction))
                        sequences[direction][1] = previous
        self.native.assert_not_called()

    def test_host_metadata_returns_independent_mutable_tables(self):
        """Caller mutations of send, receive and offsets cannot poison either direction."""
        expected = {reverse: tuple(value.clone() for value in transport.compile_transport_metadata(
            self.route, self.workspace, reverse)) for reverse in (False, True)}
        for reverse in (False, True):
            result = transport.compile_transport_metadata(self.route, self.workspace, reverse)
            for table in result:
                table.fill_(-73)
            for direction in (False, True):
                actual = transport.compile_transport_metadata(self.route, self.workspace, direction)
                for value, reference in zip(actual, expected[direction]):
                    torch.testing.assert_close(value, reference)

    def test_host_cache_revalidates_equal_bool_and_float_route_instances(self):
        """Structural equality does not establish that a new object's ABI fields are valid."""
        for bad_source in (False, 0.0):
            for warm in (False, True):
                for reverse in (False, True):
                    with self.subTest(bad_source=bad_source, warm=warm, reverse=reverse):
                        self.workspace.metadata.clear()
                        if warm:
                            transport.compile_transport_metadata(self.route, self.workspace, not reverse)
                        malformed = replace(self.route, outgoing=(
                            replace(self.route.outgoing[0], source=bad_source), *self.route.outgoing[1:]))
                        self.assertEqual(malformed, self.route)
                        self.assertEqual(hash(malformed), hash(self.route))
                        with self.assertRaisesRegex(ValueError, "nonnegative int64 integers"):
                            transport.compile_transport_metadata(malformed, self.workspace, reverse)
        self.native.assert_not_called()

    def test_device_cache_revalidates_equal_bool_and_float_route_instances(self):
        """The device descriptor cache must not hide a new invalid host owner."""
        for reverse in (False, True):
            source = self.source if not reverse else torch.empty((4, 16), dtype=torch.bfloat16)
            transport.prepare_exchange(source, self.route, self.workspace, reverse=reverse)
            for bad_source in (False, 0.0):
                with self.subTest(reverse=reverse, bad_source=bad_source):
                    malformed = replace(self.route, outgoing=(
                        replace(self.route.outgoing[0], source=bad_source), *self.route.outgoing[1:]))
                    with self.assertRaisesRegex(ValueError, "nonnegative int64 integers"):
                        transport.prepare_exchange(source, malformed, self.workspace, reverse=reverse)
                    self.assertIs(self.workspace.metadata[(self.route, reverse)][4], self.route)
        self.native.assert_not_called()

    def test_equal_valid_new_owner_recompiles_before_device_cache_reuse(self):
        """Only the same immutable route instance can bypass packing validation."""
        first = transport.prepare_exchange(self.source, self.route, self.workspace)
        other = replace(self.route)
        with patch.object(transport, "_pack_blocks", wraps=transport._pack_blocks) as pack:
            second = transport.prepare_exchange(self.source, other, self.workspace)
            third = transport.prepare_exchange(self.source, other, self.workspace)
        self.assertEqual(pack.call_count, 2)
        self.assertIs(self.workspace.metadata[(other, False)][4], other)
        self.assertIs(self.workspace.metadata[(other, "host_transport", 2)][0], other)
        self.assertIs(second.send, third.send)
        self.assertIsNot(first.send, second.send)
        torch.testing.assert_close(first.send, second.send)
        torch.testing.assert_close(first.receive, second.receive)

    def test_empty_host_metadata_preserves_direction_and_positive_range_table(self):
        """A zero-work route still has a per-peer sequence table with no block rows."""
        route = RouteDescriptor(0, 2, 0, 0, (), (), (), ())
        self.workspace.send_sequence = [[11, 13], [17, 19]]
        self.workspace.receive_sequence = [[23, 29], [31, 37]]
        for reverse in (True, False):
            send, receive, offsets = transport.compile_transport_metadata(route, self.workspace, reverse)
            self.assertEqual(send.shape, (0, 6))
            self.assertEqual(receive.shape, (0, 6))
            self.assertEqual(offsets.shape, (2, 6))
            torch.testing.assert_close(offsets[:, :4], torch.zeros((2, 4), dtype=torch.int64))
            expected = torch.tensor(list(zip(self.workspace.send_sequence[int(reverse)],
                                             self.workspace.receive_sequence[int(reverse)])))
            torch.testing.assert_close(offsets[:, 4:], expected)

    def test_host_metadata_cache_keys_include_block_rows(self):
        """Compiling another block size cannot return a prior block partition."""
        first = transport.compile_transport_metadata(self.route, self.workspace, False)
        self.workspace.spec = replace(self.workspace.spec, block_rows=1)
        second = transport.compile_transport_metadata(self.route, self.workspace, False)
        self.assertIn((self.route, "host_transport", 2), self.workspace.metadata)
        self.assertIn((self.route, "host_transport", 1), self.workspace.metadata)
        self.assertEqual(first[0].shape[0], 2)
        self.assertEqual(second[0].shape[0], 3)
        self.assertEqual(first[1].shape[0], 3)
        self.assertEqual(second[1].shape[0], 4)
        self.assertTrue(bool((second[0][:, 5] == 1).all()))
        self.assertTrue(bool((second[1][:, 5] == 1).all()))

    def test_device_cache_rechecks_mutable_sequence_types_and_bounds(self):
        """A warm device table never makes malformed current credit bases admissible."""
        for reverse in (False, True):
            source = self.source if not reverse else torch.empty((4, 16), dtype=torch.bfloat16)
            transport.prepare_exchange(source, self.route, self.workspace, reverse=reverse)
            for attribute in ("send_sequence", "receive_sequence"):
                for value in (2**30 - 1, 2**30, -1, False, 0.0):
                    with self.subTest(reverse=reverse, attribute=attribute, value=value):
                        sequences = getattr(self.workspace, attribute)
                        previous = sequences[int(reverse)][1]
                        sequences[int(reverse)][1] = value
                        with self.assertRaisesRegex(RuntimeError, "collective drained reset"):
                            transport.prepare_exchange(source, self.route, self.workspace, reverse=reverse)
                        sequences[int(reverse)][1] = previous
        self.native.assert_not_called()

    def test_device_cache_rejects_sequence_length_change(self):
        """Missing peer bases fail in preparation before an offsets table can be submitted."""
        transport.prepare_exchange(self.source, self.route, self.workspace)
        self.workspace.receive_sequence[0].pop()
        with self.assertRaisesRegex(RuntimeError, "collective drained reset"):
            transport.prepare_exchange(self.source, self.route, self.workspace)
        self.native.assert_not_called()
