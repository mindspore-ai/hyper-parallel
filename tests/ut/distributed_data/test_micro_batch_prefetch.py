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
"""Bound device inputs independently of Host planning windows."""

from __future__ import annotations

import unittest
import weakref
from contextlib import contextmanager
from itertools import product
from threading import Event, current_thread, local
from unittest.mock import Mock, patch

import torch

from hyper_parallel.distributed_data import DistributedDatasetConfig
from tests.ut.distributed_data.test_accumulation_window import _GatedDataset, _ids, _loader


class _DeviceBatch:
    """Track device input ownership without accelerator hardware."""

    def __init__(self, host: object) -> None:
        """Retain the test payload and its consumer stream identities."""
        self.host = host
        self.streams = []

    def record_stream(self, stream: object) -> None:
        """Record the producer and training consumers."""
        self.streams.append(stream)


class _Accelerator:
    """Provide thread-local streams and controllable consumption events."""

    def __init__(self) -> None:
        """Initialize allocation accounting and the simulated training gate."""
        self.context = local()
        self.training_stream = Mock()
        self.module = Mock()
        self.module.stream.side_effect = self.stream
        self.module.current_stream.side_effect = lambda _: getattr(self.context, "stream", self.training_stream)
        self.module.Event.side_effect = self.event
        self.module.Stream.side_effect = lambda **_: Mock()
        self.live_batches = weakref.WeakSet()
        self.copies = []
        self.copy_threads = []
        self.pin_threads = []
        self.events = []
        self.peak_batches = 0
        self.broadcasts = 0
        self.broadcast_streams = []
        self.retire_started = Event()
        self.retire_allowed = Event()
        self.retire_allowed.set()

    @contextmanager
    def stream(self, stream: object) -> object:
        """Select the stream for the current thread only."""
        previous = getattr(self.context, "stream", self.training_stream)
        self.context.stream = stream
        try:
            yield
        finally:
            self.context.stream = previous

    def event(self) -> object:
        """Create an event whose training completion can be delayed."""
        event = Mock()
        event.query.side_effect = lambda: (
            event.record.call_args is None or event.record.call_args.args[0] is not self.training_stream
            or self.retire_allowed.is_set()
        )

        def synchronize() -> None:
            """Wait only when this event marks consumption on the training stream."""
            if event.record.call_args.args[0] is self.training_stream and not self.retire_allowed.is_set():
                self.retire_started.set()
                if not self.retire_allowed.wait(10):
                    raise RuntimeError("training completion was not released")

        event.synchronize.side_effect = synchronize
        self.events.append(event)
        return event

    def move(self, host: object, device: object) -> _DeviceBatch:
        """Count live inputs at the point where H2D allocates device storage."""
        del device
        batch = _DeviceBatch(host)
        self.live_batches.add(batch)
        self.copies.append(_ids(host))
        self.copy_threads.append(current_thread())
        self.peak_batches = max(self.peak_batches, len(self.live_batches))
        return batch

    @contextmanager
    def installed(self) -> object:
        """Replace only device mechanics, keeping actual loading and planning."""
        with patch.object(torch, "cuda", self.module), patch(
                "hyper_parallel.distributed_data.device_prefetch._pin_memory", side_effect=self.pin,
        ), patch("hyper_parallel.distributed_data.device_prefetch._move_to_device", side_effect=self.move):
            yield

    def pin(self, batch: object) -> object:
        """Track the owner of pinned Host allocation."""
        self.pin_threads.append(current_thread())
        return batch

    def broadcast(self, batch: _DeviceBatch | None) -> _DeviceBatch | None:
        """Count broadcasts without retaining arguments as Mock call history would."""
        self.broadcasts += 1
        self.broadcast_streams.append(self.module.current_stream(None))
        return batch


class TestMicroBatchPrefetch(unittest.TestCase):
    """Verify Host lookahead, device backpressure and delivery checkpoints."""

    def test_lookahead_copies_without_broadcasting_or_waiting_on_future_copies(self) -> None:
        """Feature: Micro-batch prefetch.
        Description: Future H2D must not add communication or dependencies to current delivery.
        Expectation: Only the consumed batch is broadcast and waited on.
        """
        accelerator = _Accelerator()
        with accelerator.installed(), _loader(
                accumulation=3, device="cuda:0", device_prefetch_depth=2,
        ) as loader, patch.object(loader, "_broadcast_batch", new=accelerator.broadcast), patch(
                "hyper_parallel.distributed_data.transport.dist.all_reduce",
                side_effect=AssertionError("H2D prefetch must not coordinate model peers"),
        ):
            loader.prefetch()
            loader.wait_for_prefetch()
            self.assertEqual(accelerator.copies, [[0, 1], [2, 3]])
            self.assertEqual(accelerator.broadcasts, 0)
            accelerator.training_stream.wait_event.assert_not_called()
            current_copy_event = loader._device_batches[0].ready_event
            batch = next(loader)
            self.assertEqual(_ids(batch.host), [0, 1])
            self.assertEqual(accelerator.broadcasts, 1)
            accelerator.training_stream.wait_event.assert_called_once_with(current_copy_event)
            self.assertEqual(accelerator.broadcast_streams, [accelerator.training_stream])
            self.assertTrue(all(stream is accelerator.training_stream for stream in batch.streams))
            self.assertEqual(accelerator.module.Stream.call_count, 2)
            loader.wait_for_prefetch()
            self.assertEqual(accelerator.broadcasts, 1)
            self.assertEqual(accelerator.copies, [[0, 1], [2, 3], [4, 5]])
            del batch

    def test_broadcast_uses_callers_non_default_training_stream(self) -> None:
        """Feature: Current-stream model broadcast.
        Description: Deliver input on a caller-selected stream while future H2D stays independent.
        Expectation: Only the current copy is waited on, with no extra stream or delivery wait.
        """
        accelerator = _Accelerator()
        training_stream = Mock()
        with accelerator.installed(), _loader(
                accumulation=3, device="cuda:0", device_prefetch_depth=2,
        ) as loader, patch.object(loader, "_broadcast_batch", new=accelerator.broadcast):
            loader.prefetch()
            loader.wait_for_prefetch()
            copy_event = loader._device_batches[0].ready_event
            with accelerator.stream(training_stream):
                batch = next(loader)
                self.assertIs(accelerator.module.current_stream(None), training_stream)
            training_stream.wait_event.assert_called_once_with(copy_event)
            accelerator.training_stream.wait_event.assert_not_called()
            self.assertEqual(accelerator.broadcast_streams, [training_stream])
            self.assertTrue(all(stream is training_stream for stream in batch.streams))
            self.assertEqual(accelerator.module.Stream.call_count, 2)
            marker = loader._last_ready_event
            marker.record.assert_called_once_with(training_stream)
            marker.synchronize.assert_not_called()
            loader.wait_for_prefetch()
            marker.synchronize.assert_called_once_with()
            training_stream.synchronize.assert_not_called()
            del batch

    def test_close_drains_partial_broadcast_after_failure(self) -> None:
        """Feature: Current-stream broadcast cleanup.
        Description: Fail a broadcast after submission and close the loader.
        Expectation: Cleanup waits for the failed delivery marker without synchronizing later training.
        """
        accelerator = _Accelerator()
        failure = RuntimeError("broadcast failed after submission")
        with accelerator.installed(), _loader(device="cuda:0") as loader, patch.object(
                loader, "_broadcast_batch", side_effect=failure,
        ):
            with self.assertRaises(RuntimeError) as caught:
                next(loader)
            self.assertIs(caught.exception, failure)
            marker = loader._last_ready_event
            marker.record.assert_called_once_with(accelerator.training_stream)
            marker.synchronize.assert_not_called()
            loader.close()
            marker.synchronize.assert_called_once_with()
            accelerator.training_stream.synchronize.assert_not_called()

    def test_speculative_copy_failure_is_delivered_after_earlier_batches(self) -> None:
        """Feature: Micro-batch prefetch.
        Description: A future H2D failure is forwarded only when its micro-batch is consumed.
        Expectation: Earlier batches remain deliverable before the original copy error is raised.
        """
        accelerator = _Accelerator()
        failure = RuntimeError("H2D failed")

        def stage(host: object, device: object) -> _DeviceBatch:
            """Fail one future copy after preserving the earlier staged inputs."""
            if _ids(host) == [4, 5]:
                raise failure
            return accelerator.move(host, device)

        with accelerator.installed(), _loader(
                accumulation=3, device="cuda:0", device_prefetch_depth=4,
        ) as loader, patch(
                "hyper_parallel.distributed_data.device_prefetch._move_to_device", side_effect=stage,
        ), patch.object(loader, "_broadcast_batch", new=accelerator.broadcast):
            loader.prefetch()
            loader.wait_for_prefetch()
            self.assertEqual(accelerator.broadcasts, 0)
            for expected in ([0, 1], [2, 3]):
                batch = next(loader)
                self.assertEqual(_ids(batch.host), expected)
                del batch
            self.assertEqual(accelerator.broadcasts, 2)
            with self.assertRaises(RuntimeError) as caught:
                next(loader)
            self.assertIs(caught.exception, failure)
            self.assertEqual(loader.state_dict()["_index_sampler_state"]["consumed_samples"], 0)

    def test_configurable_depth_bounds_allocations_and_preserves_order(self) -> None:
        """Feature: Micro-batch prefetch.
        Description: Include zero lookahead and a queue deeper than an entire short epoch.
        Expectation: Input order is preserved and live device batches stay within depth plus one.
        """
        for depth in (0, 1, 2, 4, 12):
            for metadata_mode in (False, True):
                with self.subTest(depth=depth, metadata=metadata_mode):
                    accelerator = _Accelerator()
                    with accelerator.installed(), _loader(
                            samples=[{"id": i} for i in range(18)], accumulation=3, device="cuda:0",
                            metadata_mode=metadata_mode, device_prefetch_depth=depth,
                    ) as loader, patch.object(loader, "_broadcast_batch", new=accelerator.broadcast):
                        loader.prefetch()
                        loader.prefetch()
                        loader.wait_for_prefetch()
                        self.assertEqual(len(accelerator.copies), min(depth, 9))
                        batch = next(loader)
                        loader.wait_for_prefetch()
                        self.assertEqual(len(accelerator.copies), min(depth + 1, 9))
                        self.assertEqual(loader.state_dict()["_index_sampler_state"]["consumed_samples"], 0)
                        outputs = [_ids(batch.host)]
                        del batch
                        while (batch := next(loader, None)) is not None:
                            outputs.append(_ids(batch.host))
                            del batch
                        loader.wait_for_prefetch()
                        self.assertEqual(outputs, [list(range(i, i + 2)) for i in range(0, 18, 2)])
                        self.assertEqual(accelerator.peak_batches, min(depth + 1, 9))
                        self.assertEqual(len(accelerator.live_batches), 0)
                        self.assertEqual(accelerator.broadcasts, 10)
                        self.assertEqual(loader.state_dict()["_index_sampler_state"]["consumed_samples"], 18)

    def test_depth_validation_and_cpu_behavior(self) -> None:
        """Feature: Micro-batch prefetch.
        Description: Reject invalid depths locally and leave CPU Host buffering unchanged.
        Expectation: Invalid depths fail and CPU iteration never creates a device stream.
        """
        for depth in (-1, True, False, 1.5, "2", None):
            with self.subTest(depth=depth), self.assertRaisesRegex(ValueError, "device_prefetch_depth"):
                DistributedDatasetConfig(seq_len=8, local_batch_size=2, device_prefetch_depth=depth)
        for depth in (0, 2):
            with self.subTest(depth=depth), _loader(device_prefetch_depth=depth) as loader:
                self.assertEqual([_ids(batch) for batch in loader], [list(range(i, i + 2)) for i in range(0, 44, 2)])
                self.assertIsNone(loader._consumer_stream)

    def test_prefetch_after_close_does_not_restart_loading(self) -> None:
        """Feature: Prefetch lifecycle.
        Description: Request and drain prefetch after early close on CPU and simulated accelerators.
        Expectation: No reads, copies or Host workers restart, and the delivered checkpoint stays unchanged.
        """
        for device, depth, metadata_mode in product(("cpu", "cuda:0"), (0, 2), (False, True)):
            with self.subTest(device=device, depth=depth, metadata=metadata_mode):
                accelerator = _Accelerator()
                dataset = _GatedDataset()
                dataset.release.set()
                with accelerator.installed(), _loader(
                        samples=dataset, accumulation=3, device=device, device_prefetch_depth=depth,
                        metadata_mode=metadata_mode,
                ) as loader:
                    batch = next(loader)
                    del batch
                    worker = loader._prefetch_thread
                    loader.close()
                    reads, copies, state = list(dataset.reads), list(accelerator.copies), loader.state_dict()
                    loader.prefetch()
                    loader.wait_for_prefetch()
                    loader.close()
                    self.assertIsNone(next(loader, None))
                    self.assertFalse(worker.is_alive())
                    self.assertIsNone(loader._prefetch_thread)
                    self.assertEqual(dataset.reads, reads)
                    self.assertEqual(accelerator.copies, copies)
                    self.assertEqual(loader.state_dict(), state)

    def test_deeper_prefetch_waits_before_reusing_each_slot(self) -> None:
        """Feature: Micro-batch prefetch.
        Description: Increasing depth still requires retirement before each replacement allocation.
        Expectation: A busy slot blocks additional copies without exceeding the allocation bound.
        """
        for depth in (2, 4):
            with self.subTest(depth=depth):
                accelerator = _Accelerator()
                with accelerator.installed(), _loader(device="cuda:0", device_prefetch_depth=depth) as loader:
                    try:
                        batch = next(loader)
                        loader.wait_for_prefetch()
                        accelerator.retire_allowed.clear()
                        del batch
                        batch = next(loader)
                        self.assertFalse(accelerator.retire_started.is_set())
                        self.assertEqual(len(accelerator.copies), depth + 1)
                        accelerator.retire_allowed.set()
                        loader.wait_for_prefetch()
                        self.assertEqual(len(accelerator.copies), depth + 2)
                        self.assertEqual(accelerator.peak_batches, depth + 1)
                        del batch
                    finally:
                        accelerator.retire_allowed.set()

    def test_host_windows_do_not_expand_device_prefetch(self) -> None:
        """Feature: Micro-batch prefetch.
        Description: Prepare two three-round Host windows but allocate only current and next device inputs.
        Expectation: Host reads span two windows while device input allocation stays bounded.
        """
        for metadata_mode in (False, True):
            with self.subTest(metadata=metadata_mode):
                accelerator = _Accelerator()
                dataset = _GatedDataset()
                dataset.release.set()
                with accelerator.installed(), _loader(
                        samples=dataset, accumulation=3, metadata_mode=metadata_mode, device="cuda:0",
                ) as loader:
                    batch = next(loader)
                    loader.wait_for_prefetch()
                    self.assertEqual(dataset.reads, list(range(12)))
                    self.assertEqual(accelerator.copies, [[0, 1], [2, 3]])
                    self.assertEqual(loader.state_dict()["_index_sampler_state"]["consumed_samples"], 0)
                    self.assertEqual(_ids(batch.host), [0, 1])
                    self.assertFalse(hasattr(loader, "_batch_thread"))
                    self.assertTrue(all(thread is current_thread() for thread in accelerator.copy_threads))
                    self.assertTrue(all(thread is loader._prefetch_thread for thread in accelerator.pin_threads))
                    outputs = [_ids(batch.host)]
                    del batch
                    for _ in range(8):
                        batch = next(loader)
                        outputs.append(_ids(batch.host))
                        del batch
                    self.assertIsNone(next(loader, None))
                    self.assertEqual(outputs, [list(range(i, i + 2)) for i in range(0, 18, 2)])
                    self.assertEqual(accelerator.peak_batches, 2)
                    self.assertEqual(len(accelerator.live_batches), 0)
                    self.assertEqual(loader.state_dict()["_index_sampler_state"]["consumed_samples"], 18)

    def test_retired_training_event_blocks_third_allocation(self) -> None:
        """Feature: Micro-batch prefetch.
        Description: Keep the previous backward in flight and forbid allocating a third device batch.
        Expectation: The third copy starts only after the retired training event completes.
        """
        accelerator = _Accelerator()
        with accelerator.installed(), _loader(accumulation=3, device="cuda:0") as loader:
            try:
                batch = next(loader)
                loader.wait_for_prefetch()
                accelerator.retire_allowed.clear()
                del batch
                batch = next(loader)
                self.assertFalse(accelerator.retire_started.is_set())
                self.assertEqual(_ids(batch.host), [2, 3])
                self.assertEqual(len(accelerator.copies), 2)
                self.assertEqual(loader.state_dict()["_index_sampler_state"]["consumed_samples"], 0)
                accelerator.retire_allowed.set()
                loader.wait_for_prefetch()
                self.assertEqual(len(accelerator.copies), 3)
                self.assertEqual(accelerator.peak_batches, 2)
                del batch
            finally:
                accelerator.retire_allowed.set()

    def test_blocked_host_lookahead_does_not_block_current_device_batches(self) -> None:
        """Feature: Micro-batch prefetch.
        Description: Consume the current window while the Host producer is blocked on the next window.
        Expectation: Current-window inputs remain available while future Host reads are blocked.
        """
        dataset = _GatedDataset()
        accelerator = _Accelerator()
        with accelerator.installed(), _loader(samples=dataset, accumulation=3, device="cuda:0") as loader:
            try:
                batch = next(loader)
                self.assertTrue(dataset.started.wait(5))
                self.assertEqual(_ids(batch.host), [0, 1])
                del batch
                for expected in ([2, 3], [4, 5]):
                    batch = next(loader)
                    self.assertEqual(_ids(batch.host), expected)
                    del batch
                self.assertFalse(dataset.release.is_set())
                self.assertEqual(loader.state_dict()["_index_sampler_state"]["consumed_samples"], 6)
            finally:
                dataset.release.set()

    def test_device_failure_preserves_last_delivered_checkpoint(self) -> None:
        """Feature: Micro-batch prefetch.
        Description: Propagate a broadcast failure despite the Host producer having committed later reads.
        Expectation: Repeated delivery raises the same error without advancing the checkpoint.
        """
        for depth in (0, 1, 4):
            with self.subTest(depth=depth):
                self._check_device_failure(depth)

    def _check_device_failure(self, depth: int) -> None:
        """Check failure replay and checkpoint safety at the requested device prefetch depth."""
        accelerator = _Accelerator()
        with accelerator.installed(), _loader(
                accumulation=3, device="cuda:0", device_prefetch_depth=depth,
        ) as loader:
            failure = RuntimeError("device broadcast failed")

            def broadcast(batch: _DeviceBatch) -> _DeviceBatch:
                """Fail the last micro-batch before its checkpoint can be delivered."""
                if _ids(batch.host) == [4, 5]:
                    raise failure
                return batch

            with patch.object(loader, "_broadcast_batch", side_effect=broadcast):
                for _ in range(2):
                    batch = next(loader)
                    del batch
                with self.assertRaises(RuntimeError) as caught:
                    next(loader)
            self.assertIs(caught.exception, failure)
            self.assertEqual(loader.state_dict()["_index_sampler_state"]["consumed_samples"], 0)
            copies = list(accelerator.copies)
            loader.prefetch()
            loader.wait_for_prefetch()
            self.assertEqual(accelerator.copies, copies)
            with self.assertRaises(RuntimeError) as repeated:
                next(loader)
            self.assertIs(repeated.exception, failure)

    def test_checkpoint_tail_and_epoch_reuse(self) -> None:
        """Feature: Micro-batch prefetch.
        Description: Replay incomplete steps and reuse the Host thread and streams across an uneven tail.
        Expectation: Resume replays the tail and epoch changes retain the existing thread and streams.
        """
        for depth in (0, 1, 2, 12):
            with self.subTest(depth=depth):
                self._check_checkpoint_tail_and_epoch_reuse(depth)

    def _check_checkpoint_tail_and_epoch_reuse(self, depth: int) -> None:
        """Check checkpoint replay, partial windows and producer reuse at a given depth."""
        accelerator = _Accelerator()
        options = {
            "samples": [{"id": i} for i in range(14)], "accumulation": 5,
            "device": "cuda:0", "device_prefetch_depth": depth,
        }
        with accelerator.installed():
            with _loader(**options) as loader:
                for _ in range(5):
                    batch = next(loader)
                    del batch
                state = loader.state_dict()
                self.assertEqual(state["_index_sampler_state"]["consumed_samples"], 10)
                expected = []
                while (batch := next(loader, None)) is not None:
                    expected.append(_ids(batch.host))
                    del batch
                worker, stream = loader._prefetch_thread, loader._device_prefetch._copy_stream
                loader.set_epoch(1)
                actual = []
                while (batch := next(loader, None)) is not None:
                    actual.append(_ids(batch.host))
                    del batch
                self.assertEqual(actual, [list(range(i, i + 2)) for i in range(0, 14, 2)])
                self.assertIs(loader._prefetch_thread, worker)
                self.assertIs(loader._device_prefetch._copy_stream, stream)
            self.assertFalse(worker.is_alive())
            with _loader(**options) as resumed:
                resumed.load_state_dict(state)
                actual = []
                while (batch := next(resumed, None)) is not None:
                    actual.append(_ids(batch.host))
                    del batch
                self.assertEqual(actual, expected)

    def test_reset_before_first_device_delivery_and_drain_after_eof(self) -> None:
        """Feature: Micro-batch prefetch.
        Description: Discard a speculative first batch on epoch reset and drain the final device event after EOF.
        Expectation: Reset releases speculative inputs and EOF draining waits for final delivery.
        """
        for depth in (0, 1, 4):
            with self.subTest(depth=depth):
                self._check_reset_before_first_device_delivery(depth)

    def _check_reset_before_first_device_delivery(self, depth: int) -> None:
        """Check that epoch reset drains speculative device batches before first delivery."""
        accelerator = _Accelerator()
        with accelerator.installed(), _loader(
                samples=[{"id": i} for i in range(2)], accumulation=3, device="cuda:0",
                device_prefetch_depth=depth,
        ) as loader:
            loader.prefetch()
            loader.wait_for_prefetch()
            self.assertEqual(len(accelerator.live_batches), min(depth, 1))
            worker = loader._prefetch_thread
            loader.set_epoch(1)
            self.assertEqual(len(accelerator.live_batches), 0)
            self.assertEqual(loader.state_dict()["_index_sampler_state"]["epoch"], 1)
            batch = next(loader)
            self.assertEqual(_ids(batch.host), [0, 1])
            del batch
            self.assertIsNone(next(loader, None))
            last_event = loader._last_ready_event
            last_event.synchronize.reset_mock()
            copies = list(accelerator.copies)
            loader.prefetch()
            loader.wait_for_prefetch()
            self.assertEqual(accelerator.copies, copies)
            last_event.synchronize.assert_called_once_with()
            self.assertIs(loader._prefetch_thread, worker)
            last_event.record.assert_called_once_with(accelerator.training_stream)
