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
"""Tests for CPU-only source preparation in accelerator collective mode."""

import threading
import unittest

from hyper_parallel.distributed_data.packed_balancing import _LocalBalancingIterator, _LocalBatch


class _SynchronousCollectiveLoader:
    """Minimal loader exposing the iterator's HCCL scheduling contract."""

    local_dataloader = (("step-0",), ("step-1",))
    max_steps = 2
    _uses_synchronous_collectives = True
    _transport = type("Transport", (), {"communication_device": None})()

    def __init__(self) -> None:
        """Record source and collective phase ownership."""
        self.last_balance_stats = None
        self.events = []

    def _read_step(self, raw_bins, step):
        """Prepare source metadata without entering a distributed collective."""
        self.events.append(("read", raw_bins, step, threading.current_thread().name))
        return (tuple(raw_bins), ()), {}

    def _construct_batch(self, raw_bins, step, *, prepared=None):
        """Record the foreground collective phase and consume prepared metadata."""
        self.events.append(("collect", raw_bins, step, threading.current_thread().name, prepared is not None))
        if prepared is None:
            raise AssertionError("Synchronous mode must consume a prepared source step.")
        return _LocalBatch([step])

    def _deliver_batch(self, result, step):
        """Return the fake batch to the iterator consumer."""
        return result.data


class TestPackedBalancingPrefetch(unittest.TestCase):
    """Verify that HCCL mode prefetches only CPU work."""

    def test_synchronous_collectives_keep_foreground_ownership(self) -> None:
        """Source preparation runs ahead while collection remains on MainThread."""
        loader = _SynchronousCollectiveLoader()
        iterator = _LocalBalancingIterator(loader)

        self.assertEqual(next(iterator), [0])
        self.assertEqual(next(iterator), [1])
        iterator.close()

        self.assertEqual([event[0] for event in loader.events], ["read", "collect", "read", "collect"])
        self.assertTrue(all(event[3] == "hp-local-balance-prefetch" for event in loader.events if event[0] == "read"))
        self.assertTrue(all(event[3] == "MainThread" for event in loader.events if event[0] == "collect"))
        self.assertTrue(all(event[4] for event in loader.events if event[0] == "collect"))

