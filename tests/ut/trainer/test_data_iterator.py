# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for trainer-side dataloader iteration helpers."""

import inspect
import threading
import unittest

from hyper_parallel.trainer.runtime.data_iterator import BackgroundPrefetcher, HyperIter


class _BlockingIterator:
    """Return one item after the test releases a blocked ``next`` call."""

    def __init__(self) -> None:
        self.next_started = threading.Event()
        self.release_next = threading.Event()

    def __iter__(self) -> "_BlockingIterator":
        return self

    def __next__(self) -> object:
        self.next_started.set()
        self.release_next.wait()
        return object()


class _BlockingDataLoader:
    """Expose a blocking iterator and record state snapshot requests."""

    def __init__(self) -> None:
        self.iterator = _BlockingIterator()
        self.state_dict_calls = 0

    def __iter__(self) -> _BlockingIterator:
        return self.iterator

    def state_dict(self) -> dict[str, int]:
        self.state_dict_calls += 1
        return {"calls": self.state_dict_calls}


class _SingleItemStatefulDataLoader:
    """Produce one item with a checkpointable iteration position."""

    def __init__(self) -> None:
        self.index = 0

    def __iter__(self) -> "_SingleItemStatefulDataLoader":
        return self

    def __next__(self) -> int:
        if self.index > 0:
            raise StopIteration
        self.index += 1
        return self.index

    def state_dict(self) -> dict[str, int]:
        """Return the current iteration position."""
        return {"index": self.index}


class _TwoItemIterator:
    """Produce two items so the second queue write waits behind the first."""

    def __init__(self) -> None:
        self.index = 0
        self.second_item_ready = threading.Event()

    def __iter__(self) -> "_TwoItemIterator":
        return self

    def __next__(self) -> int:
        self.index += 1
        if self.index == 2:
            self.second_item_ready.set()
        if self.index <= 2:
            return self.index
        raise StopIteration


class TestBackgroundPrefetcher(unittest.TestCase):
    """Tests for cooperative prefetch worker shutdown and reference release."""

    def test_stop_uses_finite_default_timeout(self) -> None:
        """The public stop methods should retain a bounded default wait."""
        prefetcher_timeout = inspect.signature(BackgroundPrefetcher.stop).parameters["timeout"].default
        hyper_iter_timeout = inspect.signature(HyperIter.stop).parameters["timeout"].default

        self.assertEqual(prefetcher_timeout, 5.0)
        self.assertEqual(hyper_iter_timeout, 5.0)

    def test_stop_waits_for_next_then_discards_result_and_releases_references(self) -> None:
        """The default stop should wait for ``next`` without retaining its result."""
        dataloader = _BlockingDataLoader()
        prefetcher = BackgroundPrefetcher(dataloader)
        self.addCleanup(prefetcher.stop, 1.0)
        self.addCleanup(dataloader.iterator.release_next.set)
        self.assertTrue(dataloader.iterator.next_started.wait(timeout=1.0))

        stop_result = []
        stop_thread = threading.Thread(target=lambda: stop_result.append(prefetcher.stop()))
        stop_thread.start()
        self.assertTrue(prefetcher.stop_event.wait(timeout=1.0))
        self.assertTrue(stop_thread.is_alive())

        dataloader.iterator.release_next.set()
        stop_thread.join(timeout=1.0)

        self.assertFalse(stop_thread.is_alive())
        self.assertEqual(stop_result, [True])
        self.assertFalse(prefetcher.thread.is_alive())
        self.assertTrue(prefetcher.queue.empty())
        self.assertEqual(dataloader.state_dict_calls, 0)
        self.assertIsNone(prefetcher.iterator)
        self.assertIsNone(prefetcher.dataloader)
        self.assertIsNone(prefetcher.original_state_dict)
        self.assertIsNone(prefetcher.current_state)

    def test_stop_preserves_last_consumed_state_for_checkpoint(self) -> None:
        """Stopping should retain the consumed-batch state for a final checkpoint."""
        data_iterator = HyperIter(_SingleItemStatefulDataLoader(), use_background_prefetcher=True)
        self.addCleanup(data_iterator.stop, 1.0)

        self.assertEqual(next(data_iterator), 1)
        state_before_stop = data_iterator.state_dict()
        self.assertTrue(data_iterator.stop(timeout=1.0))

        self.assertEqual(state_before_stop, {"index": 1})
        self.assertEqual(data_iterator.state_dict(), state_before_stop)

    def test_timed_out_stop_cleans_up_after_worker_exits(self) -> None:
        """A worker should release its references after a timed-out stop returns."""
        dataloader = _BlockingDataLoader()
        prefetcher = BackgroundPrefetcher(dataloader)
        self.addCleanup(prefetcher.stop, 1.0)
        self.addCleanup(dataloader.iterator.release_next.set)
        self.assertTrue(dataloader.iterator.next_started.wait(timeout=1.0))

        with self.assertLogs("hyper_parallel.trainer.runtime.data_iterator", level="WARNING"):
            stopped = prefetcher.stop(timeout=0.01)

        self.assertFalse(stopped)
        self.assertTrue(prefetcher.thread.is_alive())
        self.assertIs(prefetcher.iterator, dataloader.iterator)
        self.assertIs(prefetcher.dataloader, dataloader)

        dataloader.iterator.release_next.set()
        prefetcher.thread.join(timeout=1.0)

        self.assertFalse(prefetcher.thread.is_alive())
        self.assertTrue(prefetcher.queue.empty())
        self.assertIsNone(prefetcher.iterator)
        self.assertIsNone(prefetcher.dataloader)
        self.assertIsNone(prefetcher.original_state_dict)
        self.assertTrue(prefetcher.stop(timeout=1.0))

    def test_stop_unblocks_worker_waiting_to_write_to_full_queue(self) -> None:
        """A stop request should cancel a producer waiting on a full queue."""
        iterator = _TwoItemIterator()
        prefetcher = BackgroundPrefetcher(iterator, maxsize=1)
        self.addCleanup(prefetcher.stop, 1.0)
        self.assertTrue(iterator.second_item_ready.wait(timeout=1.0))

        self.assertTrue(prefetcher.stop())

        self.assertFalse(prefetcher.thread.is_alive())
        self.assertTrue(prefetcher.queue.empty())

    def test_hyper_iter_stop_is_successful_without_background_prefetching(self) -> None:
        """Stopping direct iteration should be an immediate successful no-op."""
        iterator = HyperIter([1, 2], use_background_prefetcher=False)

        self.assertTrue(iterator.stop())


if __name__ == "__main__":
    unittest.main()
