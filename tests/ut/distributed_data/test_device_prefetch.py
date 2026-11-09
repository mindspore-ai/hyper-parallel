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
"""Tests for shared producer-owned asynchronous H2D batch prefetch."""

import unittest
import weakref
from contextlib import nullcontext
from unittest.mock import Mock, patch

import torch

from hyper_parallel.distributed_data.device_prefetch import DeviceStepPrefetcher, _move_to_device, _resolve_device


class _DeviceBatch:
    """Expose the stream-lifetime protocol used by the prefetcher."""

    def __init__(self, events: list[tuple[str, object]]) -> None:
        """Store the event sink."""
        self._events = events

    def record_stream(self, stream: object) -> None:
        """Record the consumer stream."""
        self._events.append(("record_stream", stream))


class TestDeviceStepPrefetcher(unittest.TestCase):
    """Verify copy-stream launch, event ordering, and slot lifecycle."""

    def test_recursive_transfer_moves_all_tensor_fields(self) -> None:
        """Feature: Uniform device placement.
        Description: Transfer nested inputs, offsets and labels to a hardware-free meta device.
        Expectation: All tensor leaves move without dtype changes; host inputs and scalar metadata survive.
        """
        batch = {
            "input_ids": torch.tensor([1, 2], dtype=torch.int64),
            "offsets": torch.tensor([0, 2], dtype=torch.int32),
            "nested": [{"labels": (torch.tensor([1.0, 2.0]),)}],
            "metadata": {"name": "sample", "length": 2, "optional": None},
        }
        moved = _move_to_device(batch, torch.device("meta"))
        for host, device in (
                (batch["input_ids"], moved["input_ids"]),
                (batch["offsets"], moved["offsets"]),
                (batch["nested"][0]["labels"][0], moved["nested"][0]["labels"][0]),
        ):
            self.assertEqual(device.device.type, "meta")
            self.assertEqual(device.dtype, host.dtype)
            self.assertEqual(device.shape, host.shape)
            self.assertEqual(host.device.type, "cpu")
        self.assertIsInstance(moved["nested"], list)
        self.assertIsInstance(moved["nested"][0]["labels"], tuple)
        self.assertEqual(moved["metadata"], batch["metadata"])

    def test_gloo_device_selection_does_not_probe_accelerators(self) -> None:
        """Feature: Gloo device selection.
        Description: Resolve implicit CPU and explicit accelerator training devices.
        Expectation: Neither operation probes available accelerators.
        """
        with patch.object(torch, "npu", create=True) as npu, patch.object(torch, "cuda") as cuda:
            self.assertEqual(_resolve_device(communication_backend="gloo"), torch.device("cpu"))
            self.assertEqual(_resolve_device("cuda:1", communication_backend="gloo"), torch.device("cuda:1"))
            npu.is_available.assert_not_called()
            cuda.is_available.assert_not_called()

    def test_failed_copy_drains_only_copy_stream(self) -> None:
        """Feature: Failed H2D staging lifecycle.
        Description: Fail a batch item after launching the first copy.
        Expectation: Drain only the copy stream before releasing staging storage.
        """
        device_module = Mock()
        device_module.stream.side_effect = lambda _stream: nullcontext()
        first, second = Mock(), Mock()
        first.to.return_value = object()
        second.to.side_effect = RuntimeError("copy failed")
        with patch.object(torch, "cuda", device_module), patch(
                "hyper_parallel.distributed_data.device_prefetch._pin_memory", side_effect=lambda value: value,
        ):
            prefetcher = DeviceStepPrefetcher("cuda:0")
            with self.assertRaisesRegex(RuntimeError, "copy failed"):
                prefetcher([first, second])
            first.to.assert_called_once_with(torch.device("cuda:0"), non_blocking=True)
            second.to.assert_called_once_with(torch.device("cuda:0"), non_blocking=True)
            device_module.Stream.return_value.synchronize.assert_called_once()
            device_module.synchronize.assert_not_called()
            self.assertEqual(prefetcher._pending_staging, [])

    def test_discarded_step_retains_staging_until_close(self) -> None:
        """Feature: Pending H2D staging lifecycle.
        Description: Close immediately after enqueueing an unconsumed device step.
        Expectation: Copies drain before pinned staging storage is released.
        """
        device_module = Mock()
        device_module.stream.side_effect = lambda _stream: nullcontext()
        device_module.Event.return_value.query.return_value = False
        pinned = {"pinned": object()}
        with patch.object(torch, "cuda", device_module), patch(
                "hyper_parallel.distributed_data.device_prefetch._pin_memory", return_value=pinned,
        ):
            prefetcher = DeviceStepPrefetcher("cuda:0")
            prefetcher({"host": 1})
            self.assertIs(prefetcher._pending_staging[0][1], pinned)
            device_module.Event.return_value.synchronize.assert_not_called()
            prefetcher.close()
            device_module.Event.return_value.synchronize.assert_called_once()
            self.assertEqual(prefetcher._pending_staging, [])

    def test_producer_prefetch_binds_device_and_hands_off_storage(self) -> None:
        """Feature: Producer-owned H2D.
        Description: Stage two steps on one copy stream and consume the first.
        Expectation: Device binding, copy completion and consumer lifetime are explicit.
        """
        events = []
        device_module = Mock()
        device_module.stream.side_effect = lambda _stream: nullcontext()
        device_batch = _DeviceBatch(events)
        with patch.object(torch, "cuda", device_module), patch(
                "hyper_parallel.distributed_data.device_prefetch._pin_memory", side_effect=lambda value: value,
        ), patch(
                "hyper_parallel.distributed_data.device_prefetch._move_to_device", return_value=device_batch,
        ):
            prefetcher = DeviceStepPrefetcher("cuda:0")
            first = prefetcher({"host": 1})
            prefetcher({"host": 2})
            self.assertEqual(device_module.set_device.call_count, 2)
            device_module.Stream.assert_called_once_with(device=torch.device("cuda:0"))
            # H2D is intentionally not host-synchronized here.  The consumer
            # stream waits on each step's ready event in ``take_batch``.
            self.assertEqual(device_module.Event.return_value.synchronize.call_count, 0)
            self.assertIs(first.take_batch(), device_batch)
            device_module.current_stream.return_value.wait_event.assert_called_once_with(first.ready_event)
            self.assertEqual(events, [("record_stream", device_module.current_stream.return_value)])
            self.assertIsNone(first.device_batch)

    def test_complete_batch_preserves_containers_and_records_all_storage(self) -> None:
        """Feature: Complete-batch device handoff.
        Description: Prefetch multiple nested items in list, tuple and mapping batches.
        Expectation: One take preserves the batch structure and registers every device item after one event wait.
        """
        for container in (list, tuple, lambda items: {"items": items, "name": "batch"}):
            with self.subTest(container=container):
                events = []
                device_module = Mock()
                device_module.stream.side_effect = lambda _stream: nullcontext()
                device_items = [_DeviceBatch(events), _DeviceBatch(events)]
                host_items = [Mock(), Mock()]
                for host, device in zip(host_items, device_items):
                    host.to.return_value = device
                host_batch = container([{"data": item} for item in host_items])
                expected = container([{"data": item} for item in device_items])
                with patch.object(torch, "cuda", device_module), patch(
                        "hyper_parallel.distributed_data.device_prefetch._pin_memory", side_effect=lambda value: value,
                ):
                    prefetcher = DeviceStepPrefetcher("cuda:0")
                    staged = prefetcher(host_batch)
                    self.assertEqual(staged.take_batch(), expected)
                    self.assertIsNone(staged.device_batch)
                    device_module.current_stream.return_value.wait_event.assert_called_once_with(staged.ready_event)
                    self.assertEqual(events, [("record_stream", device_module.current_stream.return_value)] * 2)
                    prefetcher.close()

    def test_completed_copy_releases_source_before_device_delivery(self) -> None:
        """Feature: Pinned source ownership.
        Description: Keep a device result alive while retiring its copy by query or explicit close.
        Expectation: Pending copies retain their source, but completed copies do not retain it through the result.
        """
        for drain in (False, True):
            with self.subTest(drain=drain):
                device_module = Mock()
                device_module.stream.side_effect = lambda _stream: nullcontext()
                events = [Mock(), Mock(), Mock()]
                for event in events:
                    event.query.return_value = False
                device_module.Event.side_effect = events
                with patch.object(torch, "cuda", device_module), patch(
                        "hyper_parallel.distributed_data.device_prefetch._move_to_device",
                        new=lambda value, _device: value.to("meta"),
                ):
                    prefetcher = DeviceStepPrefetcher("cuda:0")
                    source = torch.tensor([1])
                    source_ref = weakref.ref(source)
                    staged = prefetcher.stage_prepared(source)
                    del source
                    prefetcher.stage_prepared(torch.tensor([2]))
                    self.assertIsNotNone(source_ref())
                    events[0].synchronize.assert_not_called()
                    if drain:
                        events[0].synchronize.side_effect = lambda ref=source_ref: self.assertIsNotNone(ref())
                        prefetcher.close()
                        events[0].synchronize.assert_called_once()
                    else:
                        events[0].query.return_value = True
                        prefetcher.stage_prepared(torch.tensor([3]))
                        events[0].synchronize.assert_not_called()
                    self.assertIsNone(source_ref())
                    self.assertIsNotNone(staged.device_batch)
                    prefetcher.close()
