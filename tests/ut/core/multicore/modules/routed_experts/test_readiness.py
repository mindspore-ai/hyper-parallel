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
"""CPU checks for immutable readiness templates and fresh frame counters."""

from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from hyper_parallel.core.multicore.modules.routed_experts import compute, forward, readiness, transport
from hyper_parallel.core.multicore.modules.routed_experts.descriptor import RouteDescriptor, RouteEdge
from tests.common.mark_utils import arg_mark


class _Stream:
    """Capture event waits submitted by a prepared metadata consumer."""

    def __init__(self, name: str) -> None:
        """Create an independently tracked stream."""
        self.name, self.calls = name, []

    def wait_event(self, event: _Event) -> None:
        """Reject unrecorded dependencies and capture the producer stream."""
        if event.stream is None:
            raise AssertionError("cached metadata event was never recorded")
        self.calls.append(('wait', event, event.stream))


class _Event:
    """Capture the stream owning the latest queued event record."""

    def __init__(self) -> None:
        """Create an event that is not yet recorded."""
        self.stream = None

    def record(self, stream: _Stream) -> None:
        """Record which stream initializes this event version."""
        self.stream = stream
        stream.calls.append(('record', self))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='dryrun', essential_mark='essential')
class TestReadinessTemplate(unittest.TestCase):
    """Keep immutable expectations ordered while every frame has independent cells."""

    def setUp(self) -> None:
        """Replace only accelerator allocation, submission and event boundaries."""
        self.current = _Stream('first')
        self.uploads = []
        self.enterContext(patch.object(torch, 'npu', SimpleNamespace(
            Event=_Event, current_stream=lambda _device: self.current,
            Stream=lambda **_: _Stream('vector')), create=True))
        for module in (readiness, transport, forward, compute):
            self.enterContext(patch.object(module, 'upload_metadata', side_effect=self._upload))
            if module is not readiness:
                self.enterContext(patch.object(module, 'upload_metadata_batch', side_effect=self._upload_batch))
        for module in (forward, compute):
            self.enterContext(patch.object(module, '_load_native', return_value=None))
        self.enterContext(patch.object(forward, '_forward_config', return_value=torch.empty(256, dtype=torch.uint8)))
        self.enterContext(patch.object(compute, '_linear_config', return_value=torch.empty(256, dtype=torch.uint8)))
        ops = SimpleNamespace(
            routed_experts_weight_table=lambda values: torch.tensor([v.data_ptr() for v in values]),
            routed_experts_forward_validate=lambda *_: None)
        self.enterContext(patch.object(torch.ops, 'hyper_parallel', ops))

    def _upload(self, value: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Model an ordered metadata copy with distinct CPU storage."""
        target = value.clone().to(device)
        self.uploads.append((value.dtype, tuple(value.shape), self.current, target))
        self.current.calls.append(('upload', target))
        return target

    def _upload_batch(self, values: tuple, device: torch.device) -> tuple:
        """Model each typed view in one ordered control-table batch."""
        return tuple(self._upload(value, device) for value in values)

    def _expectation_uploads(self) -> list:
        """Select readiness matrices independently of task and configuration copies."""
        return [item for item in self.uploads if item[0] == torch.int32 and len(item[1]) == 2]

    def _new_stream(self, name: str) -> _Stream:
        """Switch preparation to another tracked stream."""
        self.current = _Stream(name)
        return self.current

    def test_control_batch_preserves_typed_disjoint_views_and_source_independence(self) -> None:
        """Feature: Batched immutable control metadata.
        Description: Pack mixed dtypes, large int64 IDs and empty tails.
        Expectation: Typed views are exact, aligned, disjoint and independent of inputs.
        """
        values = (torch.tensor([2**60 + 3, 2**60 + 5], dtype=torch.int64),
                  torch.tensor([1, 3, 9], dtype=torch.uint8),
                  torch.tensor([[5, -2]], dtype=torch.int32), torch.empty((0, 3), dtype=torch.int32))
        references = tuple(value.clone() for value in values)
        packed = readiness.upload_metadata_batch(values, torch.device('cpu'))
        self.assertEqual(len(packed), len(values), f"expected={len(values)}, got={len(packed)}")
        storage = packed[0].untyped_storage().data_ptr()
        previous_end = 0
        for value, expected in zip(packed, references):
            torch.testing.assert_close(value, expected, rtol=0, atol=0)
            self.assertEqual(value.untyped_storage().data_ptr(), storage,
                             f"expected common storage={storage}, got={value.untyped_storage().data_ptr()}")
            begin = value.storage_offset() * value.element_size()
            self.assertEqual(begin % 64, 0, f"expected aligned offset=0, got={begin % 64}")
            self.assertGreaterEqual(begin, previous_end, f"expected begin>={previous_end}, got={begin}")
            previous_end = begin + value.numel() * value.element_size()
        for value in values:
            value.fill_(0)
        for value, expected in zip(packed, references):
            torch.testing.assert_close(value, expected, rtol=0, atol=0)
        self.assertEqual(readiness.upload_metadata_batch((), torch.device('cpu')), ())
        with self.assertRaisesRegex(ValueError, "contiguous"):
            readiness.upload_metadata_batch((torch.ones(2, 3).t(),), torch.device('cpu'))

    def test_receive_tile_ordinals_follow_interleaved_peer_expert_blocks(self) -> None:
        """Feature: Block-level receive readiness.
        Description: Mix peer/expert edge order and blocks crossing tile boundaries.
        Expectation: Every tile waits for its last required block; missing coverage fails.
        """
        incoming = (RouteEdge(0, 2, 1, 0, 129, 128), RouteEdge(0, 2, 3, 128, 257, 43),
                    RouteEdge(0, 2, 1, 171, 0, 65), RouteEdge(1, 2, 1, 0, 65, 64),
                    RouteEdge(1, 2, 3, 64, 300, 213))
        route = RouteDescriptor(2, 3, 0, 513, (1, 3), (257, 513), (), incoming)
        route.validate()
        for block_rows, counts, tasks in (
                (256, [[2, 1], [1, 1], [0, 0]], [[2, 1, 1, 1, 0], [1, 1, 0, 1, 1], [0] * 5]),
                (127, [[3, 1], [1, 2], [0, 0]], [[3, 1, 2, 1, 0], [1, 1, 0, 1, 2], [0] * 5])):
            with self.subTest(block_rows=block_rows):
                expected, task_expected = transport._task_readiness_metadata(route, block_rows)
                torch.testing.assert_close(expected, torch.tensor(counts, dtype=torch.int32), rtol=0, atol=0)
                torch.testing.assert_close(task_expected, torch.tensor(tasks, dtype=torch.int32), rtol=0, atol=0)
        missing = RouteDescriptor(0, 1, 0, 1, (0,), (1,), (), ())
        with self.assertRaisesRegex(ValueError, "do not cover"):
            transport._task_readiness_metadata(missing, 256)

    def test_output_progress_covers_every_required_column_and_tail(self) -> None:
        """Feature: Producer task-prefix readiness.
        Description: Compare uneven column/core grids and tails with explicit task overlap.
        Expectation: Every block threshold covers exactly its required producer tasks.
        """
        boundaries = (127, 257, 1026)
        for columns in (1, 4, 25):
            tasks = compute.linear_tasks(boundaries, columns * 16, 128, 16)
            for cores in (1, 7, min(24, tasks.shape[0])):
                with self.subTest(columns=columns, cores=cores):
                    initial = readiness.producer_progress(tasks, boundaries, cores, 128, columns)
                    self.assertEqual(initial[..., 0].count_nonzero().item(), 0)
                    for expert, (begin, end) in enumerate(zip((0, *boundaries[:-1]), boundaries)):
                        for core in range(cores):
                            owned = tasks[core::cores]
                            owned = owned[owned[:, 0] == expert]
                            cell = initial[core, expert].tolist()
                            self.assertEqual(cell[6], len(owned))
                            if owned.shape[0] == 0:
                                self.assertEqual(cell[1], 0)
                                continue
                            for first_row in (begin, min(begin + 129, end - 1), end - 1):
                                last_row = min(first_row + 255, end - 1)
                                first_task = (first_row - cell[2]) // cell[3] * cell[4]
                                last_task = (last_row - cell[2]) // cell[3] * cell[4] + cell[4] - 1
                                first_local = max(0, (first_task - cell[5] + cell[8] - 1) // cell[8])
                                last_local = min(cell[6], max(0, (last_task - cell[5]) // cell[8] + 1))
                                needed = last_local if first_local < last_local else 0
                                overlaps = ((owned[:, 1] <= last_row)
                                            & (owned[:, 1] + owned[:, 2] > first_row))
                                positions = overlaps.nonzero().flatten()
                                reference = int(positions[-1]) + 1 if len(positions) else 0
                                self.assertEqual(needed, reference)


    @staticmethod
    def _route() -> RouteDescriptor:
        """Create a tail-bearing route shared by original, replay and backward."""
        edges = (RouteEdge(0, 0, 0, 0, 0, 257),)
        return RouteDescriptor(0, 1, 257, 257, (0,), (257,), edges, edges)

    def test_transport_three_phases_upload_once_and_wait_cross_stream(self) -> None:
        """Every phase waits the one template event before admitting fresh counters."""
        route = self._route()
        workspace = SimpleNamespace(metadata={}, device=torch.device('cpu'), spec=SimpleNamespace(block_rows=256))
        first, _ = transport._prepare_readiness(route, workspace, False, None, True)
        template_event = workspace.metadata[(route, 'readiness')][2]
        first.flags.fill_(7)
        for name in ('replay', 'backward'):
            stream = self._new_stream(name)
            ready, _ = transport._prepare_readiness(route, workspace, False, None, True)
            self.assertIn(('wait', template_event, template_event.stream), stream.calls)
            self.assertIs(ready.expected, first.expected)
            self.assertNotEqual(ready.flags.data_ptr(), first.flags.data_ptr())
            self.assertEqual(ready.flags.count_nonzero().item(), 0)
            self.assertIs(ready.admitted.stream, stream)
            self.assertLess(stream.calls.index(('wait', template_event, template_event.stream)),
                            stream.calls.index(('record', ready.admitted)))
        self.assertEqual(len(self._expectation_uploads()), 1)


    def test_forward_replay_expected_once_flags_and_fences_independent(self) -> None:
        """Feature: Forward replay readiness ownership.
        Description: Reuse expectations across streams after updating original counters.
        Expectation: Metadata is ordered while replay counters and fences remain private.
        """
        pipeline = forward.ForwardPipeline((127, 257), torch.device('cpu'), vector_stream=_Stream('vector'))
        x = torch.empty((257, 2048), dtype=torch.bfloat16)
        gate = [torch.empty((2048, 1536), dtype=x.dtype) for _ in range(2)]
        down = [torch.empty((768, 2048), dtype=x.dtype) for _ in range(2)]
        first = pipeline.prepare(x, gate, down, 768)
        template_event = pipeline.metadata[(2048, False)][-1]
        first.ready.flags.fill_(7)
        stream = self._new_stream('replay')
        replay = pipeline.prepare(x, gate, down, 768)
        self.assertIn(('wait', template_event, template_event.stream), stream.calls)
        self.assertEqual(len(self._expectation_uploads()), 1)
        self.assertIs(first.ready.expected, replay.ready.expected)
        self.assertNotEqual(first.ready.flags.data_ptr(), replay.ready.flags.data_ptr())
        self.assertIsNot(first.ready.admitted, replay.ready.admitted)
        self.assertIsNot(first.ready.poll_after.event, replay.ready.poll_after.event)
        self.assertFalse(first.ready.poll_after.queued)
        self.assertFalse(replay.ready.poll_after.queued)
        self.assertIs(replay.initialized.stream, stream)
        torch.testing.assert_close(replay.ready.flags, pipeline.metadata[(2048, False)][-2], rtol=0, atol=0)
