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
"""Process-wide device-runtime simulation switch with no-op stream primitives.

Logical simulation (the dry-run tool) executes unmodified trainer and core
code over FakeTensors without ever binding an accelerator or initializing its
runtime. Components that schedule asynchronous transfers — activation swap,
optimizer-state swap, FSDP CPU offload — fetch their stream and event
primitives through seams that consult this switch. Under simulation every
transfer degenerates to its host-visible program order: the same storages are
allocated, copied, resized, and released; only device-side overlap disappears,
which is exactly the part a logical memory tracker does not account anyway.
"""

import contextlib
from typing import Any, Optional

__all__ = [
    "SimulatedEvent",
    "SimulatedStream",
    "hold_for_simulated_stream",
    "is_device_runtime_simulated",
    "release_simulated_stream_holds",
    "simulate_device_runtime",
    "simulated_accelerator",
    "simulated_device_handle",
]

_SIMULATED_DEVICE_RUNTIME = False
_SIMULATED_ACCELERATOR: Optional[str] = None
_PENDING_STREAM_HOLDS: list = []


def simulate_device_runtime(enabled: bool = True, accelerator: Optional[str] = None) -> None:
    """Toggle no-op device effects for logical (FakeTensor) simulation.

    Args:
        enabled: Whether device-side effects become no-ops.
        accelerator: Device type the simulation lies as (``npu``/``cuda``), or
            ``None`` for plain CPU simulation. Components that pick a
            collective staging device consult it so simulated runs allocate
            communication buffers on the same (fake) device real runs do.
    """
    global _SIMULATED_DEVICE_RUNTIME, _SIMULATED_ACCELERATOR  # pylint: disable=global-statement
    _SIMULATED_DEVICE_RUNTIME = bool(enabled)
    _SIMULATED_ACCELERATOR = accelerator if enabled else None
    if not enabled:
        _PENDING_STREAM_HOLDS.clear()


def is_device_runtime_simulated() -> bool:
    """Return whether device-side effects are currently simulated."""
    return _SIMULATED_DEVICE_RUNTIME


def simulated_accelerator() -> Optional[str]:
    """Return the accelerator type the simulation lies as, or ``None``."""
    return _SIMULATED_ACCELERATOR


def hold_for_simulated_stream(tensor: Any) -> None:
    """Keep ``tensor`` alive until the next simulated synchronization point.

    This is the simulation analogue of ``Tensor.record_stream``: a real
    caching allocator retains a staging buffer until the recorded stream's
    pending work completes, even after the last Python reference dies. The
    stand-in holds the reference until an event wait or stream synchronize is
    observed, so the logical lifetime is conservative (never shorter than the
    real one) without modeling stream timing.
    """
    _PENDING_STREAM_HOLDS.append(tensor)


def release_simulated_stream_holds() -> None:
    """Drop stream-held references; called from simulated waits."""
    _PENDING_STREAM_HOLDS.clear()


class SimulatedEvent:
    """Event stand-in whose completion is always immediate.

    Host code that records, waits on, or queries an event observes the same
    control flow as a synchronous device: ``query`` answers ``True`` so
    event-gated reuse (for example pinned-buffer pools) proceeds eagerly.
    Waits release any stream-held staging references (see
    :func:`hold_for_simulated_stream`) — the same point a real allocator would
    be allowed to reclaim them.
    """

    def record(self, stream: Optional[Any] = None) -> "SimulatedEvent":
        """Accept any stream; completion is immediate."""
        del stream
        return self

    def wait(self, stream: Optional[Any] = None) -> None:
        """Accept any stream; pending stream holds are now reclaimable."""
        del stream
        release_simulated_stream_holds()

    def query(self) -> bool:
        """Report the event as elapsed."""
        return True

    def synchronize(self) -> None:
        """Pending stream holds are now reclaimable."""
        release_simulated_stream_holds()


class SimulatedStream:
    """Stream stand-in on which every submission completes in program order."""

    def synchronize(self) -> None:
        """Pending stream holds are now reclaimable."""
        release_simulated_stream_holds()

    def query(self) -> bool:
        """Report the stream as drained."""
        return True

    def wait_event(self, event: Any) -> None:
        """Accept any event; pending stream holds are now reclaimable."""
        del event
        release_simulated_stream_holds()

    def wait_stream(self, stream: Any) -> None:
        """Accept any stream; pending stream holds are now reclaimable."""
        del stream
        release_simulated_stream_holds()

    def record_event(self, event: Optional[Any] = None) -> Any:
        """Return an elapsed event."""
        return event if event is not None else SimulatedEvent()


class _SimulatedDeviceHandle:
    """``torch.<device>``-shaped namespace whose effects are no-ops.

    Only the stream/event/synchronize surface is provided on purpose: a
    simulated process must never reach device state queries that have no
    meaningful logical answer, and an ``AttributeError`` names the gap.
    """

    Stream = SimulatedStream
    Event = SimulatedEvent

    def __init__(self) -> None:
        self._current_stream = SimulatedStream()

    def current_stream(self, device: Optional[Any] = None) -> SimulatedStream:
        """Return the process-wide simulated current stream."""
        del device
        return self._current_stream

    @staticmethod
    def stream(stream: Optional[Any] = None) -> Any:
        """Return a context manager that activates ``stream`` logically."""
        del stream
        return contextlib.nullcontext()

    @staticmethod
    def synchronize(device: Optional[Any] = None) -> None:
        """Pending stream holds are now reclaimable."""
        del device
        release_simulated_stream_holds()


_SIMULATED_DEVICE_HANDLE = _SimulatedDeviceHandle()


def simulated_device_handle() -> _SimulatedDeviceHandle:
    """Return the process-wide simulated device handle."""
    return _SIMULATED_DEVICE_HANDLE
