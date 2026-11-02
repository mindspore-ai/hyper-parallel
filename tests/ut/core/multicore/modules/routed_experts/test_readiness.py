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

    def _expectation_uploads(self) -> list:
        """Select readiness matrices independently of task and configuration copies."""
        return [item for item in self.uploads if item[0] == torch.int32 and len(item[1]) == 2]

    def _new_stream(self, name: str) -> _Stream:
        """Switch preparation to another tracked stream."""
        self.current = _Stream(name)
        return self.current


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
        """F and replay share immutable expectations with independent producer fences."""
        pipeline = forward.ForwardPipeline((127, 257), torch.device('cpu'), vector_stream=_Stream('vector'))
        x = torch.empty((257, 2048), dtype=torch.bfloat16)
        gate = [torch.empty((2048, 1536), dtype=x.dtype) for _ in range(2)]
        down = [torch.empty((768, 2048), dtype=x.dtype) for _ in range(2)]
        first = pipeline.prepare(x, gate, down, 768)
        template_event = pipeline.metadata[(2048, False)][-1]
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
