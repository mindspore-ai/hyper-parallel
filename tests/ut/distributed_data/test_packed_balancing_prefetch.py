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
from typing import Any
from unittest.mock import Mock, patch

from hyper_parallel.distributed_data.packed_balancing import _LocalBalancingIterator, _LocalBatch
from tests.common.mark_utils import arg_mark


class _SynchronousCollectiveLoader:
    """Record collective and worker ownership without accelerator hardware."""

    local_dataloader = (("step-0",), ("step-1",))
    max_steps = 2
    _uses_synchronous_collectives = True

    def __init__(self) -> None:
        """Initialize fake transport state and the source event log."""
        self.last_balance_stats = None
        self.events = []
        self._transport = Mock()
        self._transport.all_gather_object.side_effect = self._gather
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
        return step

    def _make_plan(self, gathered, step):
        self._event("plan", step)
        return gathered, {"step": step}

    def _broadcast(self, planned):
        self._event("broadcast", planned[0])
        return planned

    def _begin_batch(self, _payloads, plan, stats):
        self._event("exchange", plan)
        return plan, stats

    def _finish_batch(self, pending):
        step, stats = pending
        self._event("collate", step)
        return _LocalBatch([step], stats)

    def _deliver_batch(self, result, _step):
        return result.data


class TestPackedBalancingPrefetch(unittest.TestCase):
    """Verify staged prefetch preserves data, ownership, limits and failures."""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
    def test_collectives_use_caller_with_or_without_explicit_hooks(self) -> None:
        """Feature: Caller-owned accelerator collectives.
        Description: Consume steps with and without explicit prefetch hooks.
        Expectation: Both paths preserve order and collective thread ownership.
        """
        for explicit in (False, True):
            with self.subTest(explicit=explicit):
                loader = _SynchronousCollectiveLoader()
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
                    self.assertEqual([event[0] for event in events],
                                     ["source", "gather", "plan", "broadcast", "exchange", "collate"])
                    for name, _, thread in events:
                        expected = "MainThread" if name in ("gather", "broadcast", "exchange") else \
                            "hp-local-balance-prefetch"
                        self.assertEqual(thread, expected)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
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

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
    def test_background_errors_reach_consumer(self) -> None:
        """Feature: Prefetch error propagation.
        Description: Fail planning and collation in the background worker.
        Expectation: The consumer receives the original worker exception.
        """
        for method in ("_make_plan", "_finish_batch"):
            with self.subTest(method=method):
                loader = _SynchronousCollectiveLoader()
                iterator = _LocalBalancingIterator(loader)
                try:
                    with patch.object(loader, method, side_effect=ValueError("worker failed")):
                        with self.assertRaisesRegex(ValueError, "worker failed"):
                            next(iterator)
                finally:
                    iterator.close()

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
    def test_source_exhaustion_and_early_close(self) -> None:
        """Feature: Prefetch lifecycle.
        Description: Exercise source EOF and close during each pipeline phase.
        Expectation: Speculative work drains and later iteration stops cleanly.
        """
        loader = _SynchronousCollectiveLoader()
        loader.max_steps = None
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
