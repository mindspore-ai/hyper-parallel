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
"""Device timer using asynchronous events.

:class:`DeviceTimer` encapsulates the ``torch.Event`` lifecycle so
that callers do not need to deal with API details directly.
"""

from __future__ import annotations

import torch


class DeviceTimer:
    """Asynchronous op timer backed by events.

    Usage::

        timer = DeviceTimer()
        timer.start()
        ...  # Device work
        elapsed_ms = timer.stop()

    Only the **end** event is synchronised (light-weight); the start
    and end events are recorded on ``torch.accelerator.current_stream()``.
    """

    def __init__(self) -> None:
        self._start_event: torch.Event | None = None

    def start(self) -> None:
        """Record the start event on the current accelerator stream.

        Raises:
            RuntimeError: If no accelerator is available.
        """
        if not torch.accelerator.is_available():
            raise RuntimeError(
                "Cannot start DeviceTimer: no accelerator available"
            )
        ev = torch.Event(enable_timing=True)
        ev.record(torch.accelerator.current_stream())
        self._start_event = ev

    def stop(self) -> float:
        """Record the end event, synchronise, and return elapsed milliseconds.

        Raises:
            RuntimeError: If ``start()`` was not called or no accelerator is
                available.
        """
        if self._start_event is None:
            raise RuntimeError(
                "Cannot stop DeviceTimer: start() was not called"
            )
        if not torch.accelerator.is_available():
            raise RuntimeError(
                "Cannot stop DeviceTimer: no accelerator available"
            )
        end_event = torch.Event(enable_timing=True)
        end_event.record(torch.accelerator.current_stream())
        end_event.synchronize()
        duration_ms = self._start_event.elapsed_time(end_event)
        self._start_event = None
        return duration_ms
