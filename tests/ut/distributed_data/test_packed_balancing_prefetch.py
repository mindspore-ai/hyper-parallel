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
"""Tests for caller-owned collectives and background batch completion."""

import threading
import unittest
from contextlib import nullcontext
from itertools import product
from typing import Any
from unittest.mock import Mock, patch

from hyper_parallel.distributed_data import DistributedDatasetConfig
from hyper_parallel.distributed_data.packed_balancing import _LocalBalancingIterator, _LocalBatch


class _SynchronousCollectiveLoader:
    """Record collective and worker ownership without accelerator hardware."""

    local_dataloader = (("step-0",), ("step-1",))
    _uses_synchronous_collectives = True

    def __init__(self, broadcast_plan: bool = True) -> None:
        """Initialize fake transport state and the source event log."""
        self.last_balance_stats = None
        self.config = DistributedDatasetConfig(seq_len=10, local_batch_size=1, broadcast_plan=broadcast_plan)
        self.events = []
        self._transport = Mock()
        self._transport.begin_all_gather_object.side_effect = self._gather
        self._transport.broadcast_from_planner.side_effect = self._broadcast

    def _event(self, name, step):
        self.events.append((name, step, threading.current_thread().name))

    def _data_stream_context(self):
        return nullcontext()

    def _read_step(self, raw_bins, step):
        self._event("source", step)
        return step, {"raw": raw_bins}

    def _gather(self, step):
        self._event("gather", step)
        return Mock(wait=lambda: self._wait_metadata(step))

    def _wait_metadata(self, step):
        self._event("metadata_wait", step)
        return step

    def _make_plan(self, gathered, step):
        self._event("plan", step)
        return gathered, {"step": step}

    def _broadcast(self, planned):
        self._event("broadcast", planned[0])
        return planned

    def _begin_batch(self, payloads, plan, stats):
        del payloads
        self._event("exchange", plan)
        return plan, stats

    def _finish_batch(self, pending):
        step, stats = pending
        self._event("collate", step)
        return _LocalBatch([step], stats)

    def _deliver_batch(self, result, step):
        del step
        return result.data


class TestPackedBalancingPrefetch(unittest.TestCase):
    """Verify staged prefetch preserves data, ownership, source exhaustion and failures."""

    def test_collectives_use_caller_with_or_without_explicit_hooks(self) -> None:
        """Feature: Caller-owned accelerator collectives.
        Description: Consume steps with and without explicit prefetch hooks.
        Expectation: Both paths preserve order and collective thread ownership.
        """
        for explicit, broadcast_plan in product((False, True), repeat=2):
            with self.subTest(explicit=explicit, broadcast_plan=broadcast_plan):
                loader = _SynchronousCollectiveLoader(broadcast_plan=broadcast_plan)
                iterator = _LocalBalancingIterator(loader)
                try:
                    self.assertEqual(next(iterator), [0])
                    if explicit:
                        iterator.prefetch_plan()
                        iterator.prefetch_plan()
                        iterator.prefetch()
                        iterator.prefetch()
                        iterator.wait_for_prefetch()
                        self.assertEqual(loader.last_balance_stats, {"step": 0})
                    self.assertEqual(next(iterator), [1])
                    iterator.prefetch_plan()
                    iterator.prefetch()
                    with self.assertRaises(StopIteration):
                        next(iterator)
                finally:
                    iterator.close()
                for step in (0, 1):
                    events = [event for event in loader.events if event[1] == step]
                    expected_events = ["source", "gather", "metadata_wait", "plan"]
                    if broadcast_plan:
                        expected_events.append("broadcast")
                    expected_events.extend(("exchange", "collate"))
                    self.assertEqual([event[0] for event in events], expected_events)
                    for name, _, thread in events:
                        expected = "MainThread" if name in ("gather", "broadcast", "exchange") else \
                            "hp-local-balance-prefetch"
                        self.assertEqual(thread, expected)

    def test_metadata_completion_does_not_block_launch(self) -> None:
        """Feature: Background local-step metadata completion.
        Description: Delay the gathered result after the foreground launches metadata exchange.
        Expectation: Planning hooks return while completion is pending and iteration later delivers the batch.
        """
        loader = _SynchronousCollectiveLoader()
        iterator = _LocalBalancingIterator(loader)
        started, release = threading.Event(), threading.Event()
        original = loader._wait_metadata

        def delayed(step: int) -> int:
            """Hold metadata completion until the foreground confirms launch returned."""
            started.set()
            if not release.wait(5):
                raise RuntimeError("metadata completion timed out")
            return original(step)

        try:
            self.assertEqual(next(iterator), [0])
            with patch.object(loader, "_wait_metadata", side_effect=delayed):
                iterator.prefetch_plan()
                iterator.prefetch_plan()
                self.assertTrue(started.wait(5))
                self.assertEqual(iterator._phase, "plan")
                release.set()
                self.assertEqual(next(iterator), [1])
        finally:
            release.set()
            iterator.close()

    def test_payload_completion_does_not_block_launch(self) -> None:
        """Feature: Background payload completion.
        Description: Delay collation after the foreground launches the next step.
        Expectation: The iterator remains usable while completion is pending.
        """
        loader = _SynchronousCollectiveLoader()
        iterator = _LocalBalancingIterator(loader)
        started, release = threading.Event(), threading.Event()
        original = loader._finish_batch

        def delayed(pending: Any) -> _LocalBatch:
            """Hold batch completion until the test releases the worker."""
            started.set()
            if not release.wait(5):
                raise RuntimeError("test completion timed out")
            return original(pending)

        try:
            self.assertEqual(next(iterator), [0])
            with patch.object(loader, "_finish_batch", side_effect=delayed):
                iterator.prefetch()
                self.assertTrue(started.wait(5))
                self.assertEqual(iterator._step, 1)
                self.assertTrue(iterator._thread.is_alive())
                release.set()
                self.assertEqual(next(iterator), [1])
        finally:
            release.set()
            iterator.close()

    def test_background_errors_reach_consumer(self) -> None:
        """Feature: Prefetch error propagation.
        Description: Fail planning and collation in the background worker.
        Expectation: The consumer receives the original worker exception.
        """
        for method in ("_wait_metadata", "_make_plan", "_finish_batch"):
            with self.subTest(method=method):
                loader = _SynchronousCollectiveLoader()
                iterator = _LocalBalancingIterator(loader)
                try:
                    with patch.object(loader, method, side_effect=ValueError("worker failed")):
                        with self.assertRaisesRegex(ValueError, "worker failed"):
                            next(iterator)
                finally:
                    iterator.close()

    def test_source_exhaustion_and_early_close(self) -> None:
        """Feature: Prefetch lifecycle.
        Description: Exercise source EOF and close during each pipeline phase.
        Expectation: Speculative work drains and later iteration stops cleanly.
        """
        loader = _SynchronousCollectiveLoader()
        iterator = _LocalBalancingIterator(loader)
        self.assertEqual(next(iterator), [0])
        self.assertEqual(next(iterator), [1])
        iterator.prefetch_plan()
        iterator.prefetch()
        with self.assertRaises(StopIteration):
            next(iterator)
        iterator.close()
        for phase in ("source", "plan", "batch"):
            loader = _SynchronousCollectiveLoader()
            iterator = _LocalBalancingIterator(loader)
            next(iterator)
            if phase in ("plan", "batch"):
                iterator.prefetch_plan()
            if phase == "batch":
                iterator.prefetch()
            iterator.close()
            with self.assertRaises(StopIteration):
                next(iterator)
