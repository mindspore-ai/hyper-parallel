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
"""Local producer fences for actual expert ranges, without stream-wide payload waits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


def upload_metadata(value: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Use pinned host storage so tiny control copies do not drain queued NPU work."""
    if value.device.type != "cpu" or not value.is_contiguous():
        raise ValueError("native control metadata must be contiguous on the host")
    return value.pin_memory(device=device).to(device, non_blocking=True)


@dataclass
class PollFence:
    """Retain a preallocated event and whether its producer record has been queued."""

    event: Any
    queued: bool


@dataclass(frozen=True)
class ExpertReadiness:
    """Own initialized cache lines and the event admitting their asynchronous users.

    A producer owns one 64-byte cell per active expert. The admission event only
    covers allocation/initialization. The polling fence covers work preceding
    the producer kernel, while native consumers poll its actual progress.
    A prepared linear producer must be submitted before a polling consumer;
    an event wait cannot observe a record submitted later on the host.
    """

    flags: torch.Tensor
    expected: torch.Tensor
    admitted: Any
    poll_after: PollFence
    task_expected: torch.Tensor | None = None

    def validate_polling(self) -> None:
        """Reject consumers submitted before their prepared producer is queued."""
        if not self.poll_after.queued:
            raise ValueError("polling requires its linear producer to be submitted first")

    @classmethod
    def create(cls, expected: torch.Tensor, device: torch.device, *,
               deferred_poll: bool = False,
               expected_device: torch.Tensor | None = None,
               task_expected: torch.Tensor | None = None) -> "ExpertReadiness":
        """Initialize fresh counters, optionally borrowing an ordered immutable template.

        A caller supplying expected_device must first wait for its cached upload
        event on the current stream. The template is read-only across frames.
        """
        if (expected.device.type != "cpu" or expected.dtype != torch.int32 or expected.ndim != 2
                or not expected.is_contiguous()):
            raise ValueError("readiness expectations must be a contiguous CPU int32 producer/expert matrix")
        if expected_device is not None:
            if (expected_device.device != device or expected_device.dtype != torch.int32
                    or expected_device.shape != expected.shape or not expected_device.is_contiguous()):
                raise ValueError("cached readiness expectations must match the producer/expert device matrix")
            if expected_device.numel() and (expected_device.data_ptr() == 0
                    or expected_device.untyped_storage().nbytes()
                    < (expected_device.storage_offset() + expected_device.numel()) * expected_device.element_size()):
                raise ValueError("cached readiness expectation storage is invalid or undersized")
        flags = torch.zeros((*expected.shape, 16), dtype=torch.int32, device=device)
        target = upload_metadata(expected, device) if expected_device is None else expected_device
        if task_expected is not None:
            if (task_expected.device != device or task_expected.dtype != torch.int32
                    or task_expected.ndim != 2 or task_expected.shape[0] != expected.shape[0]
                    or not task_expected.is_contiguous()):
                raise ValueError("task readiness expectations must match the producer device matrix")
        current = torch.npu.current_stream(device)
        event = torch.npu.Event()
        event.record(current)
        poll_after = torch.npu.Event()
        poll_after.record(current)
        return cls(flags, target, event, PollFence(poll_after, not deferred_poll), task_expected)
