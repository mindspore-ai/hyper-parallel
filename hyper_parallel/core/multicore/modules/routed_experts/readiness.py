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
from typing import Any, Sequence

import numpy as np
import torch


def producer_progress(tasks: torch.Tensor, boundaries: tuple[int, ...], cores: int,
                      tile_rows: int, column_tiles: int) -> torch.Tensor:
    """Describe each producer's ordered output tasks in existing readiness padding.

    Args:
        tasks: CPU int64 descriptors sorted by expert, row and output column.
        boundaries: Exclusive row ends of the packed experts.
        cores: Number of producers; task i belongs to producer i modulo cores.
        tile_rows: Rows in a full task, with a possibly shorter final task.
        column_tiles: Number of tasks covering all output columns of one row tile.

    Returns:
        CPU int32 [cores, experts, 16] initial cells. Word zero is the mutable
        completed-task count; words 1..9 describe its immutable task sequence.
        For example, 24 producers and four column tiles give producer zero
        tasks 0,24,...; a return block ending in row tile one needs task 0 only.
    """
    if (tasks.device.type != "cpu" or tasks.dtype != torch.int64 or tasks.ndim != 2
            or tasks.shape[1] < 3 or cores < 1 or tile_rows < 1 or column_tiles < 1):
        raise ValueError("producer progress requires CPU task descriptors and positive tile dimensions")
    result = torch.zeros((cores, len(boundaries), 16), dtype=torch.int32)
    table, cells = tasks.numpy(), result.numpy()
    indices = np.arange(table.shape[0], dtype=np.int32)
    experts, owners = table[:, 0], indices % cores
    first = np.full((cores, len(boundaries)), table.shape[0], dtype=np.int32)
    counts = np.zeros_like(first)
    np.minimum.at(first, (owners, experts), indices)
    np.add.at(counts, (owners, experts), 1)
    expert_first = np.full(len(boundaries), table.shape[0], dtype=np.int32)
    np.minimum.at(expert_first, experts, indices)
    active = counts > 0
    cells[..., 1] = active
    cells[..., 2] = np.asarray((0, *boundaries[:-1]))
    cells[..., 3] = tile_rows
    cells[..., 4] = column_tiles
    cells[..., 5] = np.where(active, first - expert_first, 0)
    cells[..., 6] = counts
    cells[..., 7] = np.where(active, first, 0)
    cells[..., 8] = cores
    cells[..., 9] = np.diff((0, *boundaries))
    return result


def upload_metadata(value: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Use pinned host storage so tiny control copies do not drain queued NPU work."""
    if value.device.type != "cpu" or not value.is_contiguous():
        raise ValueError("native control metadata must be contiguous on the host")
    return value.pin_memory(device=device).to(device, non_blocking=True)


def upload_metadata_batch(values: Sequence[torch.Tensor], device: torch.device) -> tuple[torch.Tensor, ...]:
    """Upload immutable control tables through one aligned pinned allocation.

    Args:
        values: Contiguous CPU tables; dtypes and shapes are retained without casting.
        device: Device on whose current stream the asynchronous copy is submitted.

    Returns:
        Disjoint, 64-byte-aligned typed views of one device allocation, in input order.
        An empty input produces an empty tuple. Mutable readiness counters must
        keep their own per-frame allocation instead of sharing these templates.
    """
    offsets = []
    size = 0
    for value in values:
        if value.device.type != "cpu" or not value.is_contiguous():
            raise ValueError("native control metadata must be contiguous on the host")
        size = (size + 63) // 64 * 64
        offsets.append(size)
        size += value.numel() * value.element_size()
    host = torch.empty(size, dtype=torch.uint8, device="cpu", pin_memory=device.type != "cpu")
    host_bytes = host.numpy()
    for value, offset in zip(values, offsets):
        source = value.reshape(-1).view(torch.uint8).numpy()
        host_bytes[offset:offset + source.size] = source
    uploaded = host.to(device, non_blocking=True)
    return tuple(uploaded.narrow(0, offset, value.numel() * value.element_size()).view(value.dtype).view(value.shape)
                 for value, offset in zip(values, offsets))


@dataclass
class PollFence:
    """Retain a preallocated event and whether its producer record has been queued."""

    event: Any
    queued: bool


@dataclass(frozen=True)
class ExpertReadiness:
    """Own initialized cache lines and the event admitting their asynchronous users.

    A producer owns one 64-byte cell per active expert. Optional task geometry
    lets consumers admit output blocks from its monotone completed-task prefix.
    The admission event only
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
               task_expected: torch.Tensor | None = None,
               progress: torch.Tensor | None = None) -> "ExpertReadiness":
        """Initialize fresh counters, optionally borrowing an ordered immutable template.

        A caller supplying expected_device must first wait for its cached upload
        event on the current stream. The template is read-only across frames.
        Optional CPU progress geometry occupies existing cache-line padding;
        its zero counters are private to this frame.

        Args:
            expected: CPU int32 producer/expert expectation matrix.
            device: Device owning counters and the admission events.
            deferred_poll: Require the producer to be queued before polling.
            expected_device: Optional previously uploaded immutable expectation matrix.
            task_expected: Optional device matrix of per-tile receive thresholds.
            progress: Optional CPU geometry with fresh zero counters in word zero.

        Returns:
            Fresh phase counters and fences, retaining the immutable templates.
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
        if progress is not None:
            if (progress.device.type != "cpu" or progress.dtype != torch.int32
                    or progress.shape != (*expected.shape, 16) or not progress.is_contiguous()
                    or bool(progress[..., 0].any())):
                raise ValueError("producer progress must initialize zero counters in matching CPU cache lines")
            flags = upload_metadata(progress, device)
        else:
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
