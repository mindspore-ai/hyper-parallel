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
"""Tests for Trainer-side asynchronous H2D batch prefetch."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from hyper_parallel.distributed_data import (
    DeviceBatchPrefetcher,
    DistributedDatasetConfig,
    SampleMetadata,
    build_distributed_dataloader,
)
from tests.common.mark_utils import arg_mark


class _StandaloneMesh:
    """Represent one DP rank without initializing torch.distributed."""

    mesh_shape = (1,)
    mesh_dim_names = ("dp",)
    rank_list = (0,)


class _StreamContext:
    """Record copy-stream context entry and exit."""

    def __init__(self, events: list[tuple[str, object]], stream: object) -> None:
        """Store the event sink and selected stream."""
        self._events = events
        self._stream = stream

    def __enter__(self) -> None:
        """Record context entry."""
        self._events.append(("enter", self._stream))

    def __exit__(self, exception_type: object, exception: object, traceback: object) -> None:
        """Record context exit."""
        self._events.append(("exit", self._stream))


class _DeviceBatch:
    """Expose the stream-lifetime protocol used by the prefetcher."""

    def __init__(self, events: list[tuple[str, object]]) -> None:
        """Store the event sink."""
        self._events = events

    def record_stream(self, stream: object) -> None:
        """Record the consumer stream."""
        self._events.append(("record_stream", stream))


def _fake_accelerator(events: list[tuple[str, object]]) -> SimpleNamespace:
    """Build deterministic fake Stream/Event APIs without accelerator hardware."""
    copy_stream = object()
    current_stream = object()

    def create_stream(*, device: torch.device) -> object:
        """Return the fixed fake copy stream."""
        events.append(("create_stream", device))
        return copy_stream

    def stream_context(stream: object) -> _StreamContext:
        """Return a recording fake stream context."""
        return _StreamContext(events, stream)

    def create_event() -> Mock:
        """Return one fake ready Event."""
        event = Mock()
        event.record.side_effect = lambda stream: events.append(("event_record", stream))
        event.wait.side_effect = lambda stream: events.append(("event_wait", stream))
        return event

    def get_current_stream(device: torch.device) -> object:
        """Return the fixed fake consumer stream."""
        events.append(("current_stream", device))
        return current_stream

    return SimpleNamespace(
        Event=create_event,
        Stream=create_stream,
        current_stream=get_current_stream,
        stream=stream_context,
    )


class TestDeviceBatchPrefetcher(unittest.TestCase):
    """Verify copy-stream launch, event ordering, and slot lifecycle."""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
    def test_prepares_copies_and_waits_before_returning_batch(self) -> None:
        """Feature: Asynchronous device batch prefetch.
        Description: Prepare and copy a Host batch on the copy stream.
        Expectation: Preparation and H2D precede the consumer-stream dependency.
        """
        events: list[tuple[str, object]] = []
        accelerator = _fake_accelerator(events)
        device_batch = _DeviceBatch(events)

        def prepare(host_batch: str) -> str:
            """Record CPU preparation and return its transformed value."""
            events.append(("prepare", host_batch))
            return f"prepared-{host_batch}"

        def move(prepared_batch: str, device: torch.device) -> _DeviceBatch:
            """Record H2D launch and return the fake device batch."""
            events.append(("move", (prepared_batch, device)))
            return device_batch

        with patch(
                "hyper_parallel.distributed_data.device_prefetch._accelerator_module",
                return_value=accelerator,
        ):
            prefetcher = DeviceBatchPrefetcher("cuda:3", prepare_fn=prepare, move_fn=move)
            prefetcher.prefetch("host")
            self.assertTrue(prefetcher.has_pending)
            result = prefetcher.wait()

        self.assertIs(result, device_batch)
        self.assertFalse(prefetcher.has_pending)
        self.assertEqual(
            [name for name, _ in events],
            [
                "prepare",
                "create_stream",
                "enter",
                "move",
                "event_record",
                "exit",
                "current_stream",
                "event_wait",
                "record_stream",
            ],
        )

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
    def test_reuses_one_copy_stream_across_batches(self) -> None:
        """Feature: Device copy-stream lifecycle.
        Description: Prefetch and consume two consecutive batches.
        Expectation: One lazy copy stream is reused while Events remain per-batch.
        """
        events: list[tuple[str, object]] = []
        accelerator = _fake_accelerator(events)
        with patch(
                "hyper_parallel.distributed_data.device_prefetch._accelerator_module",
                return_value=accelerator,
        ):
            prefetcher = DeviceBatchPrefetcher(
                "cuda:0",
                move_fn=lambda batch, _device: _DeviceBatch(events),
            )
            prefetcher.prefetch("first")
            prefetcher.wait()
            prefetcher.prefetch("second")
            prefetcher.wait()

        self.assertEqual([name for name, _ in events].count("create_stream"), 1)
        self.assertEqual([name for name, _ in events].count("event_record"), 2)
        self.assertEqual([name for name, _ in events].count("event_wait"), 2)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
    def test_default_move_uses_non_blocking_batch_to(self) -> None:
        """Feature: Default device movement.
        Description: Prefetch a model-specific batch object with a ``to`` method.
        Expectation: The move targets the configured device and is non-blocking.
        """
        events: list[tuple[str, object]] = []
        accelerator = _fake_accelerator(events)
        host_batch = Mock()
        device_batch = _DeviceBatch(events)
        host_batch.to.return_value = device_batch

        with patch(
                "hyper_parallel.distributed_data.device_prefetch._accelerator_module",
                return_value=accelerator,
        ):
            prefetcher = DeviceBatchPrefetcher("cuda:1")
            prefetcher.prefetch(host_batch)
            prefetcher.wait()

        host_batch.to.assert_called_once_with(torch.device("cuda:1"), non_blocking=True)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
    def test_rejects_slot_overwrite_and_empty_wait(self) -> None:
        """Feature: Device prefetch slot validation.
        Description: Wait on an empty slot and prefetch over an occupied slot.
        Expectation: Both invalid lifecycle operations are rejected.
        """
        events: list[tuple[str, object]] = []
        with patch(
                "hyper_parallel.distributed_data.device_prefetch._accelerator_module",
                return_value=_fake_accelerator(events),
        ):
            prefetcher = DeviceBatchPrefetcher(
                "cuda:0",
                move_fn=lambda batch, _device: _DeviceBatch(events),
            )
            with self.assertRaisesRegex(ValueError, "No prefetched device batch"):
                prefetcher.wait()
            prefetcher.prefetch("first")
            with self.assertRaisesRegex(ValueError, "Cannot prefetch a second device batch"):
                prefetcher.prefetch("second")

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
    def test_composes_with_distributed_host_double_buffer(self) -> None:
        """Feature: Host and device double buffering.
        Description: Feed Host-prefetched distributed batches into device prefetch.
        Expectation: Device prefetch preserves both selected batch payloads.
        """
        samples = [{"id": 0, "tokens": 8}, {"id": 1, "tokens": 8}]

        def metadata_fn(sample: dict[str, int]) -> SampleMetadata:
            """Expose one full packing bin per sample."""
            return SampleMetadata(pack_tokens=sample["tokens"], sample_id=sample["id"])

        loader = build_distributed_dataloader(
            samples,
            _StandaloneMesh(),
            DistributedDatasetConfig(
                seq_len=8,
                local_batch_size=1,
                buffer_size_multiplier=1.0,
                max_buffered_samples=1,
                double_buffer=True,
            ),
            metadata_fn=metadata_fn,
        )
        events: list[tuple[str, object]] = []
        with patch(
                "hyper_parallel.distributed_data.device_prefetch._accelerator_module",
                return_value=_fake_accelerator(events),
        ):
            prefetcher = DeviceBatchPrefetcher(
                "cuda:0",
                move_fn=lambda batch, _device: {"host_batch": batch, "storage": _DeviceBatch(events)},
            )
            prefetcher.prefetch(next(loader))
            first = prefetcher.wait()
            prefetcher.prefetch(next(loader))
            second = prefetcher.wait()

        self.assertEqual(first["host_batch"], ((samples[0],),))
        self.assertEqual(second["host_batch"], ((samples[1],),))
        with self.assertRaises(StopIteration):
            next(loader)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
    def test_rejects_cpu_and_invalid_callbacks(self) -> None:
        """Feature: Device prefetch input validation.
        Description: Configure a CPU device and non-callable hooks.
        Expectation: Validation fails before accelerator resources are allocated.
        """
        with self.assertRaisesRegex(ValueError, "requires an accelerator device"):
            DeviceBatchPrefetcher("cpu")
        with self.assertRaisesRegex(ValueError, "prepare_fn must be callable"):
            DeviceBatchPrefetcher("cuda:0", prepare_fn=object())
        with self.assertRaisesRegex(ValueError, "move_fn must be callable"):
            DeviceBatchPrefetcher("cuda:0", move_fn=object())


if __name__ == "__main__":
    unittest.main()
