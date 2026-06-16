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
"""Unit tests for PhysicalBuffer and ResidencyManager."""

from __future__ import annotations

import contextlib
from typing import Any

import unittest
from unittest.mock import patch

import torch

from hyper_parallel.auto_parallel.hyper_offload.runtime.residency import (
    PhysicalBuffer,
    ResidencyManager,
)

# ---------------------------------------------------------------------------
# Mock helpers for CPU-only testing
# ---------------------------------------------------------------------------


def _mock_accelerator_if_needed() -> Any:
    """Return a ``patch`` context that mocks accelerator APIs for CPU.

    When CUDA is not available, this patches ``torch.accelerator``,
    ``torch.Stream``, ``torch.Event``, and ``pin_memory`` so that the
    offload runtime can function on CPU-only hosts.
    """
    if torch.cuda.is_available() or (
        hasattr(torch, "npu") and torch.npu.is_available()
    ):
        return _NullPatchContext()

    patchers = []

    mock_stream = _MockStream()

    def _current_accelerator() -> torch.device:
        return torch.device("cpu")

    def _current_stream() -> _MockStream:
        return mock_stream

    def _empty_cache() -> None:
        pass

    def _reset_peak_stats() -> None:
        pass

    def _max_memory_allocated() -> int:
        return 0

    patchers.append(
        patch.multiple(
            "torch.accelerator",
            is_available=lambda: True,
            current_accelerator=_current_accelerator,
            current_stream=_current_stream,
            empty_cache=_empty_cache,
            reset_peak_memory_stats=_reset_peak_stats,
            max_memory_allocated=_max_memory_allocated,
        )
    )

    def _pin_memory(tensor: torch.Tensor) -> torch.Tensor:
        return tensor

    def _is_pinned(*args: Any, **kwargs: Any) -> bool:
        return True

    patchers.append(patch("torch.Tensor.pin_memory", _pin_memory))
    patchers.append(patch("torch.Tensor.is_pinned", _is_pinned))

    # Patch torch.Tensor.record_stream so mocked streams work on CPU.
    def _record_stream(tensor: torch.Tensor, stream: object) -> None:
        pass

    patchers.append(patch("torch.Tensor.record_stream", _record_stream))

    # Patch torch.empty at the pinned_memory import site so pin_memory=True works on CPU.
    orig_empty = torch.empty

    def _empty_no_pin(*args: object, **kwargs: object) -> torch.Tensor:
        kwargs.pop("pin_memory", None)
        return orig_empty(*args, **kwargs)

    patchers.append(
        patch(
            "hyper_parallel.auto_parallel.hyper_offload.runtime.pinned_memory.torch.empty",
            _empty_no_pin,
        )
    )

    return _CompositePatchContext(patchers)


class _MockStream:
    """Minimal stream mock for CPU-only testing."""

    def __init__(self) -> None:
        self.device = torch.device("cpu")

    def wait_stream(self, stream: Any) -> None:
        pass

    def synchronize(self) -> None:
        pass

    def record_event(self, event: Any = None) -> Any:
        pass

    def __enter__(self) -> _MockStream:
        return self

    def __exit__(self, *args: Any) -> None:
        pass


class _NullPatchContext:
    """No-op patch context used when a real accelerator is available."""

    def __enter__(self) -> _NullPatchContext:
        return self

    def __exit__(self, *args: Any) -> None:
        pass


class _CompositePatchContext:
    """Compose multiple ``unittest.mock.patch`` contexts."""

    def __init__(self, patchers: list[Any]) -> None:
        self._patchers = patchers

    def __enter__(self) -> list[Any]:
        return [p.__enter__() for p in self._patchers]

    def __exit__(self, *args: Any) -> None:
        for p in reversed(self._patchers):
            p.__exit__(*args)


class TestPhysicalBuffer(unittest.TestCase):
    """PhysicalBuffer lifecycle and device_storage()."""

    def setUp(self) -> None:
        self._exit_stack = contextlib.ExitStack()
        self._exit_stack.enter_context(_mock_accelerator_if_needed())

    def tearDown(self) -> None:
        self._exit_stack.close()

    def test_creation(self) -> None:
        buf = PhysicalBuffer()
        self.assertIsNone(buf.device)
        self.assertIsNone(buf.host_buffer)
        self.assertIsNone(buf.host_event)
        self.assertIsNone(buf.device_buffer)
        self.assertIsNone(buf.device_event)

    def test_device_storage_raises_when_no_data(self) -> None:
        buf = PhysicalBuffer()
        with self.assertRaisesRegex(RuntimeError, "No device buffer available"):
            buf.device_storage()

    def test_device_storage_returns_storage_when_device_buffer_present(self) -> None:
        """device_storage() returns a storage when a device buffer is present."""
        device = torch.device("cpu")
        dev_data = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, device=device)
        dev_buf = dev_data.view(dtype=torch.uint8).clone()

        buf = PhysicalBuffer(
            device=device,
            device_buffer=dev_buf,
        )
        storage = buf.device_storage()

        reconstructed = torch.empty(0, dtype=torch.float32, device=device)
        reconstructed.set_(storage, 0, (3,), (1,))
        torch.testing.assert_close(reconstructed, dev_data)

    def test_device_storage_demand_pages_from_host(self) -> None:
        """When device_buffer is None but host_buffer is present, demand-page it."""
        device = torch.device("cpu")
        host_data = torch.tensor([10.0, 20.0, 30.0], dtype=torch.float32)
        host_buf = host_data.view(dtype=torch.uint8).clone()

        buf = PhysicalBuffer(
            device=device,
            host_buffer=host_buf,
            device_buffer=None,
        )

        storage = buf.device_storage()
        self.assertIsNotNone(buf.device_buffer)

        reconstructed = torch.empty(0, dtype=torch.float32, device=device)
        reconstructed.set_(storage, 0, (3,), (1,))
        torch.testing.assert_close(reconstructed, host_data.to(device))

    def test_device_storage_synchronises_host_event(self) -> None:
        """If host_event is pending, device_storage should synchronise it before demand-paging."""
        device = torch.device("cpu")
        host_data = torch.tensor([1.0, 2.0], dtype=torch.float32)
        host_buf = host_data.view(dtype=torch.uint8).clone()

        buf = PhysicalBuffer(
            device=device,
            host_buffer=host_buf,
            host_event=torch.Event(enable_timing=False),
            device_buffer=None,
        )
        buf.device_storage()
        self.assertIsNotNone(buf.device_buffer)
        self.assertIsNone(buf.host_event)  # should be cleared after sync

    def test_device_storage_without_event_succeeds(self) -> None:
        """device_storage should work when there's no device_event."""
        device = torch.device("cpu")
        dev_data = torch.tensor([5.0], dtype=torch.float32, device=device)
        dev_buf = dev_data.view(dtype=torch.uint8).clone()

        buf = PhysicalBuffer(
            device=device,
            device_buffer=dev_buf,
            device_event=None,
        )
        storage = buf.device_storage()
        self.assertIsNotNone(storage)
        reconstructed = torch.empty(0, dtype=torch.float32, device=device)
        reconstructed.set_(storage, 0, (1,), (1,))
        torch.testing.assert_close(reconstructed, dev_data)


class TestResidencyManagerBind(unittest.TestCase):
    """ResidencyManager.bind() behaviour."""

    def setUp(self) -> None:
        self._exit_stack = contextlib.ExitStack()
        self._exit_stack.enter_context(_mock_accelerator_if_needed())
        self.manager = ResidencyManager(max_host_bytes=1024 * 1024)

    def tearDown(self) -> None:
        self._exit_stack.close()

    def test_bind_creates_buffer(self) -> None:
        t = torch.randn(4, 4)
        buf = self.manager.bind(1, t)
        self.assertIsNotNone(buf.device_buffer)
        self.assertEqual(buf.device.type, "cpu")

    def test_bind_returns_same_buffer_for_same_sid(self) -> None:
        t1 = torch.randn(4, 4)
        buf1 = self.manager.bind(1, t1)
        t2 = torch.randn(4, 4)
        buf2 = self.manager.bind(1, t2)
        self.assertIs(buf1, buf2)

    def test_bind_different_sids_return_different_buffers(self) -> None:
        buf1 = self.manager.bind(1, torch.randn(2, 2))
        buf2 = self.manager.bind(2, torch.randn(2, 2))
        self.assertIsNot(buf1, buf2)

    def test_resident_bytes_after_bind(self) -> None:
        t = torch.randn(8)  # 32 bytes (8 * 4)
        self.manager.bind(1, t)
        self.assertEqual(self.manager.resident_bytes, t.untyped_storage().size())

    def test_resident_bytes_multiple_binds(self) -> None:
        t1 = torch.randn(8)  # 32 bytes
        t2 = torch.randn(16)  # 64 bytes
        self.manager.bind(1, t1)
        self.manager.bind(2, t2)
        expected = t1.untyped_storage().size() + t2.untyped_storage().size()
        self.assertEqual(self.manager.resident_bytes, expected)


class TestResidencyManagerTransitions(unittest.TestCase):
    """State transitions: copy_d2h, copy_h2d, release_device, release_host."""

    def setUp(self) -> None:
        self._exit_stack = contextlib.ExitStack()
        self._exit_stack.enter_context(_mock_accelerator_if_needed())
        self.manager = ResidencyManager(max_host_bytes=1024 * 1024)
        # When no real accelerator is available, patch _get_copy_stream to
        # return a mock with device != "cpu", forcing the synchronous copy path
        # (which avoids torch.Event.record).  With a real accelerator we keep
        # the real _get_copy_stream so wait_for_transfers / sync_all_transfers
        # can interoperate with real CUDA streams.
        self._copy_stream_patch = None
        if not torch.cuda.is_available() and not (
            hasattr(torch, "npu") and torch.npu.is_available()
        ):
            self._copy_stream_mock = unittest.mock.MagicMock()
            self._copy_stream_mock.device = torch.device("meta")
            self._copy_stream_mock.__enter__.return_value = self._copy_stream_mock
            self._copy_stream_patch = patch.object(
                self.manager,
                "_get_copy_stream",
                return_value=self._copy_stream_mock,
            )
            self._copy_stream_patch.start()

    def tearDown(self) -> None:
        if self._copy_stream_patch is not None:
            self._copy_stream_patch.stop()
        self._exit_stack.close()

    def test_copy_d2h_populates_host_buffer(self) -> None:
        t = torch.randn(4, 4)
        self.manager.bind(1, t)
        self.manager.copy_d2h(1)
        buf = self.manager._residency[1]  # pylint: disable=protected-access
        self.assertIsNotNone(buf.host_buffer)

    def test_copy_d2h_noop_when_already_on_host(self) -> None:
        t = torch.randn(4, 4)
        self.manager.bind(1, t)
        self.manager.copy_d2h(1)
        host_ptr = self.manager._residency[
            1
        ].host_buffer.data_ptr()  # pylint: disable=protected-access
        self.manager.copy_d2h(1)  # should be no-op
        self.assertEqual(
            self.manager._residency[
                1
            ].host_buffer.data_ptr(),  # pylint: disable=protected-access
            host_ptr,
        )

    def test_copy_d2h_raises_on_unregistered_sid(self) -> None:
        """copy_d2h on unregistered sid should raise RuntimeError."""
        with self.assertRaises(RuntimeError):
            self.manager.copy_d2h(999)

    def test_copy_d2h_raises_when_device_buffer_none(self) -> None:
        """copy_d2h on sid with no device_buffer should raise RuntimeError."""
        self.manager._residency[1] = PhysicalBuffer(
            device=torch.device("cpu")
        )  # pylint: disable=protected-access
        with self.assertRaises(RuntimeError):
            self.manager.copy_d2h(1)

    def test_release_device_clears_device_buffer(self) -> None:
        t = torch.randn(4, 4)
        self.manager.bind(1, t)
        self.manager.copy_d2h(1)
        self.manager.release_device(1)
        buf = self.manager._residency[1]  # pylint: disable=protected-access
        self.assertIsNone(buf.device_buffer)
        self.assertIsNotNone(buf.host_buffer)  # host preserved

    def test_release_device_noop_when_not_resident(self) -> None:
        self.manager.release_device(999)  # no-op

    def test_copy_h2d_after_release_device(self) -> None:
        t = torch.randn(4, 4)
        self.manager.bind(1, t)
        self.manager.copy_d2h(1)
        self.manager.release_device(1)

        self.manager.copy_h2d(1)
        buf = self.manager._residency[1]  # pylint: disable=protected-access
        self.assertIsNotNone(buf.device_buffer)

    def test_copy_h2d_raises_on_unregistered_sid(self) -> None:
        with self.assertRaises(RuntimeError):
            self.manager.copy_h2d(999)

    def test_copy_h2d_noop_when_already_on_device(self) -> None:
        t = torch.randn(4, 4)
        self.manager.bind(1, t)
        self.manager.copy_h2d(1)  # already on device → no-op
        # Should remain
        buf = self.manager._residency[1]  # pylint: disable=protected-access
        self.assertIsNotNone(buf.device_buffer)

    def test_copy_h2d_raises_when_device_unknown(self) -> None:
        self.manager._residency[1] = PhysicalBuffer(  # pylint: disable=protected-access
            host_buffer=torch.empty(16, dtype=torch.uint8),
            device=None,
        )
        with self.assertRaises(RuntimeError):
            self.manager.copy_h2d(1)

    def test_release_host_frees_host_buffer(self) -> None:
        t = torch.randn(4, 4)
        self.manager.bind(1, t)
        self.manager.copy_d2h(1)
        self.manager.release_host(1)
        buf = self.manager._residency[1]  # pylint: disable=protected-access
        self.assertIsNone(buf.host_buffer)

    def test_release_host_noop_when_no_host(self) -> None:
        self.manager.release_host(999)  # no-op

    def test_full_offload_cycle(self) -> None:
        """D2H → release_device → H2D → verify data correctness."""
        source = torch.arange(8, dtype=torch.float32)
        self.manager.bind(1, source)
        self.manager.copy_d2h(1)
        self.manager.release_device(1)
        self.manager.copy_h2d(1)

        buf = self.manager._residency[1]  # pylint: disable=protected-access
        restored = torch.empty(0, dtype=torch.float32)
        restored.set_(
            buf.device_buffer.untyped_storage(), 0, source.size(), source.stride()
        )
        torch.testing.assert_close(restored, source)

    def test_release_device_removes_entry_if_no_host(self) -> None:
        """If release_device is called without a host copy, the entry is removed."""
        t = torch.randn(4, 4)
        self.manager.bind(1, t)
        self.manager.release_device(1)
        # The buffer entry should be deleted (no host data)
        self.assertNotIn(1, self.manager._residency)  # pylint: disable=protected-access

    def test_release_host_removes_entry_if_no_device(self) -> None:
        """If release_host is called after device was already released, entry is removed."""
        t = torch.randn(4, 4)
        self.manager.bind(1, t)
        self.manager.copy_d2h(1)
        self.manager.release_device(1)
        self.manager.release_host(1)
        self.assertNotIn(1, self.manager._residency)  # pylint: disable=protected-access

    def test_device_resident_size(self) -> None:
        t = torch.randn(8)
        self.manager.bind(1, t)
        size = self.manager.device_resident_size(1)
        self.assertEqual(size, t.untyped_storage().size())
        # Non-existent sid
        self.assertIsNone(self.manager.device_resident_size(999))
        # After release
        self.manager.release_device(1)
        self.assertIsNone(self.manager.device_resident_size(1))

    def test_copy_h2d_with_host_event_dependency(self) -> None:
        """If host_event is set, copy_h2d should wait for it before copying."""
        t = torch.randn(8)
        self.manager.bind(1, t)
        self.manager.copy_d2h(1)
        # Set a dummy host_event
        self.manager._residency[1].host_event = torch.Event(
            enable_timing=False
        )  # pylint: disable=protected-access
        # release device so copy_h2d has something to do
        self.manager.release_device(1)
        # Should not raise
        self.manager.copy_h2d(1)

    def test_multiple_d2h_cycles_no_memory_leak(self) -> None:
        """Multiple D2H → release_device → H2D cycles should work."""
        for _ in range(3):
            t = torch.randn(16)
            self.manager.bind(1, t)
            self.manager.copy_d2h(1)
            self.manager.release_device(1)
            self.manager.copy_h2d(1)
            # Verify data is intact
            buf = self.manager._residency[1]  # pylint: disable=protected-access
            self.assertIsNotNone(buf.device_buffer)
            self.manager.release_device(1)
            self.manager.release_host(1)

    def test_release_device_with_pending_event(self) -> None:
        """release_device should sync any pending device_event."""
        t = torch.randn(8)
        self.manager.bind(1, t)
        self.manager.copy_d2h(1)
        self.manager.release_device(1)
        # Simulate pending H2D event
        self.manager.copy_h2d(1)
        # Now release device while H2D event is pending
        self.manager.release_device(1)
        self.assertIsNone(
            self.manager._residency[1].device_buffer
        )  # pylint: disable=protected-access

    def test_release_device_with_device_event_synchronizes(self) -> None:
        """release_device should synchronize a pending device_event before dropping."""
        t = torch.randn(8)
        self.manager.bind(1, t)
        self.manager.copy_d2h(1)
        self.manager.release_device(1)
        self.manager.copy_h2d(1)
        # Force a device_event onto the buffer
        self.manager._residency[1].device_event = torch.Event(
            enable_timing=False
        )  # pylint: disable=protected-access
        # Should not raise — synchronize is a no-op for an unrecorded event
        self.manager.release_device(1)
        self.assertIsNone(
            self.manager._residency[1].device_buffer
        )  # pylint: disable=protected-access

    def test_release_device_removes_entry_when_no_host(self) -> None:
        """release_device should delete the residency entry if no host copy exists."""
        t = torch.randn(8)
        self.manager.bind(1, t)
        # No copy_d2h — no host_buffer
        self.manager.release_device(1)
        self.assertNotIn(1, self.manager._residency)  # pylint: disable=protected-access

    def test_release_host_with_device_event_passes_event_to_pool(self) -> None:
        """release_host should pass device_event to the pool when device_event is set."""
        t = torch.randn(8)
        self.manager.bind(1, t)
        self.manager.copy_d2h(1)
        self.manager.release_device(1)
        self.manager.copy_h2d(1)
        # Add a device_event to exercise the 'device_event is not None' branch
        self.manager._residency[1].device_event = torch.Event(
            enable_timing=False
        )  # pylint: disable=protected-access
        # Should not raise
        self.manager.release_host(1)
        buf = self.manager._residency.get(1)  # pylint: disable=protected-access
        self.assertIsNone(buf.host_buffer if buf else None)

    def test_copy_h2d_waits_for_host_event(self) -> None:
        """copy_h2d should wait for a pending host_event before launching the copy."""
        t = torch.randn(8)
        self.manager.bind(1, t)
        self.manager.copy_d2h(1)
        # Set a dummy host_event
        self.manager._residency[1].host_event = torch.Event(
            enable_timing=False
        )  # pylint: disable=protected-access
        self.manager.release_device(1)
        # Should not raise (host_event.wait is a no-op for unrecorded event)
        self.manager.copy_h2d(1)
        self.assertIsNotNone(
            self.manager._residency[1].device_buffer
        )  # pylint: disable=protected-access


class TestResidencyManagerClearAndSync(unittest.TestCase):
    """ResidencyManager.clear_runtime and sync methods."""

    def setUp(self) -> None:
        self._exit_stack = contextlib.ExitStack()
        self._exit_stack.enter_context(_mock_accelerator_if_needed())
        self.manager = ResidencyManager(max_host_bytes=1024 * 1024)
        self._copy_stream_patch = None
        if not torch.cuda.is_available() and not (
            hasattr(torch, "npu") and torch.npu.is_available()
        ):
            self._copy_stream_mock = unittest.mock.MagicMock()
            self._copy_stream_mock.device = torch.device("meta")
            self._copy_stream_mock.__enter__.return_value = self._copy_stream_mock
            self._copy_stream_patch = patch.object(
                self.manager,
                "_get_copy_stream",
                return_value=self._copy_stream_mock,
            )
            self._copy_stream_patch.start()

    def tearDown(self) -> None:
        if self._copy_stream_patch is not None:
            self._copy_stream_patch.stop()
        self._exit_stack.close()

    def test_clear_runtime_removes_all(self) -> None:
        self.manager.bind(1, torch.randn(4, 4))
        self.manager.bind(2, torch.randn(4, 4))
        self.manager.clear_runtime()
        self.assertEqual(
            len(self.manager._residency), 0
        )  # pylint: disable=protected-access
        self.assertEqual(self.manager.resident_bytes, 0)

    def test_wait_for_transfers_no_error(self) -> None:
        self.manager.wait_for_transfers()  # should not raise

    def test_sync_all_transfers_no_error(self) -> None:
        self.manager.sync_all_transfers()  # should not raise

    def test_clear_runtime_with_pinned_memory_release(self) -> None:
        """clear_runtime should release pinned host buffers back to the pool."""
        t = torch.randn(8)
        self.manager.bind(1, t)
        self.manager.copy_d2h(1)
        self.manager.clear_runtime()
        # After clear, host memory should be returned (pool might have it in pending)
        self.assertEqual(self.manager.resident_bytes, 0)
        self.assertNotIn(1, self.manager._residency)  # pylint: disable=protected-access

    def test_device_storage_demand_page_with_host_event_sync(self) -> None:
        """device_storage should sync host_event before demand-paging from host."""
        device = torch.device("cpu")
        host_data = torch.tensor([10.0, 20.0, 30.0], dtype=torch.float32)
        host_buf = host_data.view(dtype=torch.uint8).clone()

        ev = torch.Event(enable_timing=False)
        buf = PhysicalBuffer(
            device=device,
            host_buffer=host_buf,
            host_event=ev,
            device_buffer=None,
        )
        storage = buf.device_storage()
        # host_event should have been cleared after sync
        self.assertIsNone(buf.host_event)
        # device_buffer should now be populated
        self.assertIsNotNone(buf.device_buffer)
        # Data should be intact
        reconstructed = torch.empty(0, dtype=torch.float32, device=device)
        reconstructed.set_(storage, 0, (3,), (1,))
        torch.testing.assert_close(reconstructed, host_data.to(device))

    def test_device_storage_raises_when_neither_device_nor_host(self) -> None:
        """device_storage should raise when both device_buffer and host_buffer are None."""
        buf = PhysicalBuffer(device=torch.device("cpu"))
        with self.assertRaisesRegex(RuntimeError, "No device buffer available"):
            buf.device_storage()
