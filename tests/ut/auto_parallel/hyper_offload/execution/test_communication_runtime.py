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
"""Runtime contracts for delayed D2H release and asynchronous H2D prefetch."""

from __future__ import annotations

import unittest
from typing import Any
from unittest.mock import MagicMock, patch

import torch

from hyper_parallel.auto_parallel.hyper_offload.execution.replay import ReplayExecutor
from hyper_parallel.auto_parallel.hyper_offload.ir.replay import OpGuide
from hyper_parallel.auto_parallel.hyper_offload.ir.trace import (
    AccessKind,
    ActivationTrace,
    StorageAccess,
    TraceOp,
)
from hyper_parallel.auto_parallel.hyper_offload.planning import PeakResidencyPlanner
from hyper_parallel.auto_parallel.hyper_offload.runtime.pinned_memory import (
    PinnedMemoryPool,
)
from hyper_parallel.auto_parallel.hyper_offload.runtime.residency import (
    ResidencyManager,
)


class TestCommunicationRuntime(unittest.TestCase):
    """Use CPU byte copies with explicit stream/event mocks for ordering."""

    def setUp(self) -> None:
        """Mock device queues while preserving real CPU tensor storage."""
        self.manager = ResidencyManager(max_host_bytes=4096)
        self.compute = MagicMock(name="compute_stream")
        self.copy = MagicMock(name="copy_stream")
        self.copy.device = torch.device("cpu")
        self.copy.__enter__.return_value = self.copy
        self.pool = MagicMock(spec=PinnedMemoryPool)
        self.pool.acquire.side_effect = lambda size: torch.empty(
            size, dtype=torch.uint8
        )
        self.manager._host_pool = self.pool
        for patcher in (
            patch.object(self.manager, "_get_copy_stream", return_value=self.copy),
            patch("torch.accelerator.current_stream", return_value=self.compute),
            patch("torch.Tensor.record_stream"),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.event_patch = patch("torch.Event")
        self.events = self.event_patch.start()
        self.addCleanup(self.event_patch.stop)
        self.d2h_event = MagicMock(name="d2h_event")
        self.h2d_event = MagicMock(name="h2d_event")
        self.events.side_effect = [self.d2h_event, self.h2d_event]

    def test_forward_copy_waits_only_at_delayed_release(self):
        tensor = torch.tensor([1.0, 2.0, 3.0])
        buffer = self.manager.bind(1, tensor)
        self.manager.copy_d2h(1)
        self.assertIsNotNone(buffer.device_buffer)
        self.d2h_event.wait.assert_not_called()
        self.d2h_event.synchronize.assert_not_called()
        torch.testing.assert_close(buffer.host_buffer.view(torch.float32), tensor)
        self.copy.wait_stream.assert_called_once_with(self.compute)
        self.d2h_event.record.assert_called_once_with(self.copy)

        self.manager.release_device(1)
        self.assertIsNone(buffer.device_buffer)
        self.d2h_event.wait.assert_called_once_with(self.compute)
        self.d2h_event.synchronize.assert_not_called()
        self.pool.release.assert_not_called()

    def test_backward_prefetch_waits_at_consumer_and_fences_host_reuse(self):
        tensor = torch.tensor([2.0, 4.0])
        buffer = self.manager.bind(1, tensor)
        self.manager.copy_d2h(1)
        self.manager.release_device(1)
        self.manager.copy_h2d(1)
        self.h2d_event.wait.assert_not_called()
        self.h2d_event.synchronize.assert_not_called()
        host = buffer.host_buffer
        self.manager.release_host(1)
        self.assertIs(self.pool.release.call_args.args[0], host)
        self.assertIs(self.pool.release.call_args.kwargs["event"], self.h2d_event)
        self.assertIsNone(buffer.host_buffer)
        self.h2d_event.wait.assert_not_called()

        restored = torch.empty(0, dtype=torch.float32)
        restored.set_(buffer.device_storage(), 0, tensor.shape, (1,))
        self.h2d_event.wait.assert_called_once_with(self.compute)
        self.h2d_event.synchronize.assert_not_called()
        torch.testing.assert_close(restored, tensor, rtol=0, atol=0)

    def test_incomplete_copy_keeps_source_without_blocking_compute(self):
        buffer = self.manager.bind(1, torch.arange(8, dtype=torch.float32))
        self.manager.copy_d2h(1)
        source = buffer.device_buffer
        self.d2h_event.query.return_value = False

        self.assertFalse(self.manager.try_release_device(1))
        self.assertIs(buffer.device_buffer, source)
        self.d2h_event.wait.assert_not_called()
        self.d2h_event.synchronize.assert_not_called()

        self.manager.release_device(1)
        self.assertIsNone(buffer.device_buffer)
        self.d2h_event.wait.assert_called_once_with(self.compute)
        self.d2h_event.synchronize.assert_not_called()

    def test_completed_source_retires_without_wait_and_restores_exactly(self):
        tensor = torch.arange(8, dtype=torch.float32)
        buffer = self.manager.bind(1, tensor)
        self.manager.copy_d2h(1)
        self.d2h_event.query.return_value = True

        self.assertTrue(self.manager.try_release_device(1))
        self.assertTrue(self.manager.try_release_device(1))
        self.manager.release_device(1)
        self.assertIsNone(buffer.device_buffer)
        self.d2h_event.wait.assert_not_called()
        self.d2h_event.synchronize.assert_not_called()
        self.pool.release.assert_not_called()

        self.manager.copy_h2d(1)
        self.manager.release_host(1)
        restored = torch.empty(0, dtype=torch.float32)
        restored.set_(buffer.device_storage(), 0, tensor.shape, (1,))
        torch.testing.assert_close(restored, tensor, rtol=0, atol=0)
        self.h2d_event.wait.assert_called_once_with(self.compute)

    def test_optional_release_does_not_retire_unsaved_or_prefetched_data(self):
        buffer = self.manager.bind(1, torch.ones(8))
        self.assertFalse(self.manager.try_release_device(1))
        self.assertTrue(self.manager.try_release_device(99))
        self.manager.copy_d2h(1)
        self.manager.release_device(1)
        self.manager.copy_h2d(1)
        self.assertFalse(self.manager.try_release_device(1))
        self.assertIsNotNone(buffer.device_buffer)
        self.h2d_event.synchronize.assert_not_called()

    def test_completion_query_errors_preserve_source_and_propagate(self):
        buffer = self.manager.bind(1, torch.ones(8))
        self.manager.copy_d2h(1)
        self.d2h_event.query.side_effect = RuntimeError("query failed")
        with self.assertRaisesRegex(RuntimeError, "query failed"):
            self.manager.try_release_device(1)
        self.assertIsNotNone(buffer.device_buffer)
        self.d2h_event.wait.assert_not_called()

    def test_pending_host_allocation_is_not_reused_before_h2d_completes(self):
        pool = PinnedMemoryPool(max_host_bytes=2048)
        original_empty = torch.empty

        def cpu_empty(*args: Any, **kwargs: Any) -> torch.Tensor:
            """Allocate host test memory without requiring an accelerator."""
            kwargs.pop("pin_memory", None)
            return original_empty(*args, **kwargs)

        with (
            patch("torch.empty", side_effect=cpu_empty),
            patch("torch.Tensor.is_pinned", return_value=True),
        ):
            first = pool.acquire(32)
            event = MagicMock()
            event.query.return_value = False
            pool.release(first, event=event)
            second = pool.acquire(32)
            self.assertNotEqual(first.data_ptr(), second.data_ptr())
            event.query.return_value = True
            recycled = pool.acquire(32)
            self.assertEqual(first.data_ptr(), recycled.data_ptr())
            event.synchronize.assert_not_called()

    def test_generated_schedule_runs_on_baseline_replay_and_real_buffers(self):
        self.events.side_effect = MagicMock
        rows = [
            [(1, AccessKind.WRITE)],
            [(2, AccessKind.WRITE)],
            [],
            [],
            [],
            [(3, AccessKind.WRITE)],
            [(3, AccessKind.READ)],
            [],
            [(1, AccessKind.READ)],
            [(2, AccessKind.READ)],
        ]
        trace = ActivationTrace(
            ops=[
                TraceOp(
                    f"op{op}",
                    accesses=[StorageAccess(op, sid, kind) for sid, kind in row],
                )
                for op, row in enumerate(rows)
            ],
            storage_sizes={1: 240, 2: 240, 3: 520},
            memory_limit_bytes=720,
        )
        for op in trace.ops:
            op.duration_ms = 1.0
        trace.d2h_bandwidth_gbps = 240 * 1000 / (2 * 1024**3)
        bindings = {0: {0: 1}, 1: {0: 2}, 5: {0: 3}}
        guide = [OpGuide(f"op{op}", 1, bindings.get(op, {})) for op in range(10)]
        schedule = PeakResidencyPlanner().build(trace)
        executor = ReplayExecutor(self.manager, schedule, guide)
        first = executor.dispatch(torch.arange, (60,), {"dtype": torch.float32})
        second = executor.dispatch(torch.ones, (60,), {})
        for _ in range(3):
            executor.dispatch(torch.zeros, (1,), {})
        peak = executor.dispatch(torch.ones, (130,), {})
        self.assertIsNone(first._buffer.device_buffer)
        self.assertIsNone(second._buffer.device_buffer)
        executor.dispatch(torch.sum, (peak,), {})
        executor.dispatch(torch.zeros, (1,), {})
        self.assertIsNotNone(first._buffer.device_buffer)
        self.assertIsNotNone(second._buffer.device_buffer)
        self.assertIsNone(first._buffer.host_buffer)
        first_sum = executor.dispatch(torch.sum, (first,), {})
        second_sum = executor.dispatch(torch.sum, (second,), {})
        torch.testing.assert_close(
            first_sum, torch.arange(60, dtype=torch.float32).sum(), rtol=0, atol=0
        )
        torch.testing.assert_close(second_sum, torch.tensor(60.0), rtol=0, atol=0)
        self.assertEqual(self.manager.resident_bytes, 0)
        self.assertEqual(self.pool.release.call_count, 2)
