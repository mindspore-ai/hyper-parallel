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
"""Pure physical control plane for byte-level residency.

This module provides the data-plane (:class:`PhysicalBuffer`) and
the control-plane (:class:`ResidencyManager`) for raw byte buffers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch

from hyper_parallel.auto_parallel.hyper_offload.runtime.pinned_memory import PinnedMemoryPool

logger = logging.getLogger(__name__)


@dataclass
class PhysicalBuffer:
    """Pure physical memory block tracking host/device byte buffers.

    This dataclass is intentionally minimal — it carries **no** knowledge
    of logical tensors or ShadowTensor objects.  Every field is
    ``None`` when the corresponding resource is not held.

    The **control plane** (:class:`ResidencyManager`) orchestrates
    allocations and copies.
    """

    device: torch.device | None = None
    """Target accelerator device (set on first registration)."""

    host_buffer: torch.Tensor | None = None
    """1-D uint8 pinned CPU buffer, or ``None`` when not resident on host."""

    host_event: torch.Event | None = None
    """:class:`torch.Event` recorded after the latest D2H copy completes."""

    device_buffer: torch.Tensor | None = None
    """1-D uint8 device buffer, or ``None`` when not resident on device."""

    device_event: torch.Event | None = None
    """:class:`torch.Event` recorded after the latest H2D copy completes."""

    # ------------------------------------------------------------------
    # Device-storage access (called by ShadowTensor)
    # ------------------------------------------------------------------

    def device_storage(self) -> torch.UntypedStorage:
        """Return the device storage, waiting for any in-flight H2D event.

        If the device buffer is not resident but host data is available,
        this will synchronously demand-page the data back to the device.

        Returns:
            The underlying :class:`torch.UntypedStorage` of ``device_buffer``.

        """
        if self.device_buffer is None:
            if self.host_buffer is not None and self.device is not None:
                # Synchronous demand-paging fallback.
                # Wait for any in-flight D2H copy to complete before reading
                # from host_buffer (race-condition avoidance).
                if self.host_event is not None:
                    self.host_event.synchronize()
                    self.host_event = None

                size_bytes = self.host_buffer.numel()
                dev_bytes = torch.empty(size_bytes, dtype=torch.uint8, device=self.device)
                dev_bytes.copy_(self.host_buffer, non_blocking=False)
                self.device_buffer = dev_bytes
            else:
                raise RuntimeError(
                    "No device buffer available and cannot demand-page. "
                    "Ensure host data is available if device data is evicted."
                )
        if self.device_event is not None:
            current_stream = torch.accelerator.current_stream()
            self.device_event.wait(current_stream)
            self.device_buffer.record_stream(current_stream)
            self.device_event = None
        return self.device_buffer.untyped_storage()


class ResidencyManager:
    """Pure physical controller for byte-level tensor residency.

    Owns
    ----
    * Physical residency table (``storage_id → PhysicalBuffer``).
    * High-level state transitions (``copy_d2h``, ``copy_h2d``).

    All public methods accept ``storage_id: int``.
    """

    def __init__(
        self,
        max_host_bytes: int,
    ) -> None:
        self._copy_stream = None
        self._host_pool = PinnedMemoryPool(max_host_bytes)
        self._residency: dict[int, PhysicalBuffer] = {}

    def _get_copy_stream(self) -> torch.Stream:
        """Return the internal copy stream, creating it lazily on first access.

        Delaying stream creation avoids requiring an accelerator device context
        at construction time (e.g. when running on CPU-only hosts or before
        NPU/CUDA is initialised).
        """
        if self._copy_stream is None:
            self._copy_stream = torch.Stream()
        return self._copy_stream

    # ------------------------------------------------------------------
    # Device-side memory query
    # ------------------------------------------------------------------

    @property
    def resident_bytes(self) -> int:
        """Total bytes currently resident on the device side across all storage IDs."""
        total = 0
        for buf in self._residency.values():
            if buf.device_buffer is not None:
                total += buf.device_buffer.numel()
        return total

    def device_resident_size(self, sid: int) -> int | None:
        """Return the device buffer size in bytes, or ``None`` if not device-resident."""
        buffer = self._residency.get(sid)
        if buffer is None or buffer.device_buffer is None:
            return None
        return buffer.device_buffer.numel()

    # ------------------------------------------------------------------
    # Stream synchronisation
    # ------------------------------------------------------------------

    def wait_for_transfers(self) -> None:
        """Make the current accelerator stream wait for pending async transfers on the copy stream."""
        torch.accelerator.current_stream().wait_stream(self._get_copy_stream())

    def sync_all_transfers(self) -> None:
        """Synchronise streams."""
        self._get_copy_stream().synchronize()

    # ------------------------------------------------------------------
    # Registration: bind a storage ID to a tensor's device storage
    # ------------------------------------------------------------------

    def bind(self, sid: int, tensor: torch.Tensor) -> PhysicalBuffer:
        """Point the physical buffer for *sid* at *tensor*'s device storage.

        Returns the :class:`PhysicalBuffer` so that the caller can pass
        it to a new :class:`~offload.execution.tensor.ShadowTensor`.
        """
        if sid not in self._residency:
            self._residency[sid] = PhysicalBuffer()
        buffer = self._residency[sid]
        buffer.device = tensor.device

        storage = tensor.untyped_storage()
        dev_view = torch.empty(0, dtype=torch.uint8, device=tensor.device)
        dev_view.set_(storage, 0, (storage.size(),), (1,))
        buffer.device_buffer = dev_view
        return buffer

    # ------------------------------------------------------------------
    # State transition: copy D2H
    # ------------------------------------------------------------------

    def copy_d2h(self, sid: int) -> None:
        """Copy the physical storage for *sid* from device to host.

        1. Look up the physical buffer for ``sid``.
        2. If ``host_buffer`` is already present → no-op.
        3. Launch an async D2H copy.
        4. Keep the device buffer resident until ``release_device``.
        """
        buffer = self._residency.get(sid)
        if buffer is None:
            raise RuntimeError(
                f"copy_d2h sid={sid}: no physical buffer registered"
            )
        if buffer.host_buffer is not None:
            logger.debug("copy_d2h sid=%d: already on host, skip", sid)
            return
        if buffer.device_buffer is None:
            raise RuntimeError(
                f"copy_d2h sid={sid}: no device data to copy"
            )

        dev_src = buffer.device_buffer
        size_bytes = dev_src.numel()
        logger.debug(
            "copy_d2h sid=%d: copying %d bytes (%.2f MiB) D2H",
            sid,
            size_bytes,
            size_bytes / 1024**2,
        )

        # 1. Allocate pinned host buffer.
        host_buf = self._host_pool.acquire(size_bytes)

        # 2. Prevent the caching allocator from recycling the source
        #    memory while the copy stream reads it.
        copy_stream = self._get_copy_stream()
        if dev_src.device == copy_stream.device:
            dev_src.record_stream(copy_stream)

        # 3. Launch asynchronous D2H copy.
        event = None
        if dev_src.device != copy_stream.device:
            host_buf.copy_(dev_src, non_blocking=False)
        else:
            producer_stream = torch.accelerator.current_stream()
            event = torch.Event()
            with copy_stream:
                copy_stream.wait_stream(producer_stream)
                host_buf.copy_(dev_src, non_blocking=True)
                event.record(copy_stream)

        # 4. Update physical buffer.
        buffer.host_buffer = host_buf
        buffer.host_event = event

        logger.debug("copy_d2h sid=%d: done", sid)

    # ------------------------------------------------------------------
    # State transition: copy H2D
    # ------------------------------------------------------------------

    def copy_h2d(self, sid: int) -> None:
        """Asynchronously copy (H2D) the physical storage for *sid* to device.

        Allocates fresh device memory, launches an async H2D copy on the
        copy stream, and updates the physical buffer with the
        new ``device_buffer`` and ``device_event``.  Returns immediately
        without waiting for the copy to complete.
        """
        buffer = self._residency.get(sid)
        if buffer is None:
            raise RuntimeError(
                f"copy_h2d sid={sid}: no physical buffer registered"
            )

        if buffer.device_buffer is not None:
            logger.debug("copy_h2d sid=%d: already on device, skip", sid)
            return

        if buffer.host_buffer is None:
            raise RuntimeError(
                f"copy_h2d sid={sid}: no host data to copy"
            )

        if buffer.device is None:
            raise RuntimeError(
                f"copy_h2d sid={sid}: target device unknown"
            )

        size_bytes = buffer.host_buffer.numel()
        logger.debug(
            "copy_h2d sid=%d: copying %d bytes (%.2f MiB) H2D to %s",
            sid,
            size_bytes,
            size_bytes / 1024**2,
            buffer.device,
        )

        # 1. Allocate device memory.
        dev_bytes = torch.empty(size_bytes, dtype=torch.uint8, device=buffer.device)

        # 2. Launch async H2D copy (wait for prior D2H event if present).
        copy_stream = self._get_copy_stream()
        event = None
        if dev_bytes.device != copy_stream.device:
            dev_bytes.copy_(buffer.host_buffer, non_blocking=False)
        else:
            producer_stream = torch.accelerator.current_stream()
            event = torch.Event()
            with copy_stream:
                copy_stream.wait_stream(producer_stream)
                if buffer.host_event is not None:
                    buffer.host_event.wait(copy_stream)
                dev_bytes.copy_(buffer.host_buffer, non_blocking=True)
                event.record(copy_stream)

        # 3. Update physical buffer.
        if dev_bytes.device == copy_stream.device:
            dev_bytes.record_stream(copy_stream)
        buffer.device_buffer = dev_bytes
        buffer.device_event = event

    # ------------------------------------------------------------------
    # Release helpers
    # ------------------------------------------------------------------

    def release_device(self, sid: int) -> None:
        """Release device-resident bytes for a storage ID.

        Frees the device buffer. If an H2D prefetch is still in flight,
        waits for it before dropping the destination buffer reference.
        Pending D2H copies are protected by ``record_stream`` during
        ``copy_d2h``.
        """
        buffer = self._residency.get(sid)
        if buffer is None or buffer.device_buffer is None:
            return

        if buffer.device_event is not None:
            buffer.device_event.synchronize()
            buffer.device_event = None

        buffer.device_buffer = None
        if buffer.host_buffer is None:
            del self._residency[sid]

    def release_host(self, sid: int) -> None:
        """Release host-resident bytes for a storage ID.

        Waits for any in-flight H2D copy that may be reading from the
        host buffer before returning it to the pool.
        """
        buffer = self._residency.get(sid)
        if buffer is None or buffer.host_buffer is None:
            return

        # Ensure any in-flight H2D copy that reads this host buffer
        # has completed before we return it to the pool.
        event_to_wait = buffer.device_event if buffer.device_event is not None else buffer.host_event
        self._host_pool.release(buffer.host_buffer, event=event_to_wait)
        buffer.host_buffer = None
        buffer.host_event = None
        if buffer.device_buffer is None:
            del self._residency[sid]

    # ------------------------------------------------------------------
    # Runtime clear
    # ------------------------------------------------------------------

    def clear_runtime(self) -> None:
        """Release all physical resources and reset tracking."""
        for buffer in self._residency.values():
            if buffer.host_buffer is not None:
                self._host_pool.release(buffer.host_buffer, event=buffer.host_event)
        self._residency.clear()
