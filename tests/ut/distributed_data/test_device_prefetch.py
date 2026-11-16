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
from contextlib import nullcontext
from unittest.mock import Mock, patch

import torch

from hyper_parallel.distributed_data.device_prefetch import DeviceStepPrefetcher, _move_to_device, _resolve_device
from tests.common.mark_utils import arg_mark


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

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
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

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
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

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
    def test_failed_copy_drains_only_copy_stream(self) -> None:
        """Feature: Failed H2D staging lifecycle.
        Description: Fail a mapping after launching the first copy.
        Expectation: Drain only the copy stream before releasing staging storage.
        """
        accelerator = Mock()
        accelerator.stream.side_effect = lambda _stream: nullcontext()
        move = Mock(side_effect=[object(), RuntimeError("copy failed")])
        with patch.object(torch, "cuda", accelerator), patch(
                "hyper_parallel.distributed_data.device_prefetch._pin_memory", side_effect=lambda value: value,
        ), patch(
                "hyper_parallel.distributed_data.device_prefetch._move_to_device", move,
        ):
            prefetcher = DeviceStepPrefetcher("cuda:0")
            with self.assertRaisesRegex(RuntimeError, "copy failed"):
                prefetcher([{"host": 1}, {"host": 2}])
            accelerator.Stream.return_value.synchronize.assert_called_once()
            accelerator.synchronize.assert_not_called()
            self.assertEqual(prefetcher._pending_staging, [])

    def test_custom_placement_preserves_host_fields_and_consumer_order(self) -> None:
        """Allow application placement while retaining copy-event lifetime guarantees."""
        accelerator = Mock()
        accelerator.stream.side_effect = lambda _stream: nullcontext()
        events = []
        lengths = torch.tensor([2, 3])
        host = [{"lengths": lengths, "values": torch.arange(5)}]
        placed = {"lengths": lengths, "values": _DeviceBatch(events)}
        placement = Mock(return_value=[placed])
        with patch.object(torch, "cuda", accelerator), patch(
                "hyper_parallel.distributed_data.device_prefetch._pin_memory", side_effect=lambda value: value,
        ):
            prefetcher = DeviceStepPrefetcher("cuda:0", placement)
            batch = prefetcher(host)
            placement.assert_called_once_with(host, torch.device("cuda:0"))
            self.assertIs(batch.take_microbatch(0), placed)
            self.assertIs(placed["lengths"], lengths)
            self.assertEqual(placed["lengths"].device.type, "cpu")
            accelerator.current_stream.return_value.wait_event.assert_called_once_with(batch.ready_event)
            self.assertEqual(events, [("record_stream", accelerator.current_stream.return_value)])
            prefetcher.close()

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
    def test_discarded_step_retains_staging_until_close(self) -> None:
        """Feature: Pending H2D staging lifecycle.
        Description: Close immediately after enqueueing an unconsumed device step.
        Expectation: Copies drain before pinned staging storage is released.
        """
        accelerator = Mock()
        accelerator.stream.side_effect = lambda _stream: nullcontext()
        accelerator.Event.return_value.query.return_value = False
        pinned = {"pinned": object()}
        with patch.object(torch, "cuda", accelerator), patch(
                "hyper_parallel.distributed_data.device_prefetch._pin_memory", return_value=pinned,
        ):
            prefetcher = DeviceStepPrefetcher("cuda:0")
            prefetcher([{"host": 1}])
            self.assertIs(prefetcher._pending_staging[0][1][0], pinned)
            accelerator.Event.return_value.synchronize.assert_not_called()
            prefetcher.close()
            accelerator.Event.return_value.synchronize.assert_called_once()
            self.assertEqual(prefetcher._pending_staging, [])

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
    def test_producer_prefetch_binds_device_and_hands_off_storage(self) -> None:
        """Feature: Producer-owned H2D.
        Description: Stage two steps on one copy stream and consume the first.
        Expectation: Device binding, copy completion and consumer lifetime are explicit.
        """
        events = []
        accelerator = Mock()
        accelerator.stream.side_effect = lambda _stream: nullcontext()
        device_batch = _DeviceBatch(events)
        with patch.object(torch, "cuda", accelerator), patch(
                "hyper_parallel.distributed_data.device_prefetch._pin_memory", side_effect=lambda value: value,
        ), patch(
                "hyper_parallel.distributed_data.device_prefetch._move_to_device", return_value=device_batch,
        ):
            prefetcher = DeviceStepPrefetcher("cuda:0")
            first = prefetcher([{"host": 1}])
            prefetcher([{"host": 2}])
            self.assertEqual(accelerator.set_device.call_count, 2)
            accelerator.Stream.assert_called_once_with(device=torch.device("cuda:0"))
            # H2D is intentionally not host-synchronized here.  The consumer
            # stream waits on each step's ready event in ``take_microbatch``.
            self.assertEqual(accelerator.Event.return_value.synchronize.call_count, 0)
            self.assertIs(first.take_microbatch(0), device_batch)
            accelerator.current_stream.return_value.wait_event.assert_called_once_with(first.ready_event)
            self.assertEqual(events, [("record_stream", accelerator.current_stream.return_value)])
            self.assertEqual(first.device_micro_batches, [None])
            self.assertEqual(first.cpu_micro_batches, [{"host": 1}])
