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
"""Fixed replica storage admitted before any weight or gradient collective."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from math import prod
from typing import Any

import torch


def check_dense(value: torch.Tensor, shape: tuple[int, ...], dtype: torch.dtype, device: torch.device,
                *, transposed: bool = False) -> None:
    """Validate a complete contiguous physical matrix storage, including views."""
    if value.shape != shape or value.dtype != dtype or value.device != device:
        raise ValueError("replica tensor does not match the admitted shape, dtype, or device")
    physical = value.transpose(-2, -1) if transposed else value
    if not physical.is_contiguous():
        raise ValueError("replica tensor does not match the admitted physical layout")
    capacity = value.untyped_storage().nbytes() // value.element_size()
    if value.storage_offset() < 0 or value.storage_offset() + value.numel() > capacity:
        raise ValueError("replica tensor exceeds its allocated storage")


def overlaps(left: torch.Tensor, right: torch.Tensor) -> bool:
    """Return whether two validated dense tensors address overlapping bytes."""
    if not left.numel() or not right.numel():
        return False
    left_start, right_start = left.data_ptr(), right.data_ptr()
    return (left_start < right_start + right.numel() * right.element_size()
            and right_start < left_start + left.numel() * left.element_size())


def prepare_gradient_storage(allocator: Callable, shape: tuple[int, int],
                             home_shapes: tuple[tuple[int, ...], tuple[int, ...]], *, staging_elements: int
                             ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """Reserve incoming dV and later FP32 accumulator/send/receive views.

    The caller admits this allocation before submitting T. G may overwrite the
    region only after C's last dV reader, and C must join G before arena release.
    G's three writable views are disjoint and aligned to 32 bytes. Empty token
    ranks reserve the complete accumulator and both fixed staging quotas.
    """
    if len(shape) != 2 or shape[0] < 0 or shape[1] < 1:
        raise ValueError("incoming gradient shape requires nonnegative rows and positive columns")
    if isinstance(staging_elements, bool) or not isinstance(staging_elements, int) or staging_elements < 1:
        raise ValueError("gradient staging requires a positive integer element quota")
    elements = max(map(prod, home_shapes))
    send_start = (elements + 7) // 8 * 8
    receive_start = (send_start + staging_elements + 7) // 8 * 8
    capacity = (max(prod(shape), (receive_start + staging_elements) * 2) + 1) // 2 * 2
    storage = allocator("incoming", (capacity,), torch.bfloat16)
    check_dense(storage, (capacity,), torch.bfloat16, storage.device)
    if storage.data_ptr() % 32 or storage.storage_offset() % 2:
        raise ValueError("incoming gradient storage must support aligned BF16/FP32 reinterpretation")
    incoming = storage[:prod(shape)].view(shape)
    floating = storage.view(torch.float32)
    accumulator = floating[:elements]
    staging = (floating[send_start:send_start + staging_elements],
               floating[receive_start:receive_start + staging_elements])
    return incoming, accumulator, staging


@dataclass
class ReplicaSlot:
    """One physical gate/down allocation with an explicit last-reader fence."""

    matrices: dict[str, torch.Tensor] = field(default_factory=dict)
    generation: int = 0
    in_use: bool = False
    last_read: Any = None


class ReplicaPool:
    """Own two BF16 replica slots or one reusable FP32 home accumulator.

    For role element counts ``g`` and ``d``, the W allocation is
    ``2 * replica_capacity * (g + d) * 2`` bytes. The G allocation is
    ``home_count * max(g, d) * 4`` bytes unless the caller supplies an accumulator.
    Staging belongs to TransferDomain.
    """

    def __init__(self, device: torch.device, dtype: torch.dtype) -> None:
        """Create an empty pool bound to one device and transfer dtype."""
        self.device, self.dtype = device, dtype
        self.home_count = 0
        self.replica_capacity = 0
        self.shapes: tuple[tuple[int, ...], ...] = ()
        self.transposed: tuple[bool, ...] = ()
        self.slots: list[ReplicaSlot] = []
        self.accumulator: torch.Tensor | None = None
        self.caller_accumulator = False
        self.home: dict[str, torch.Tensor] = {}

    def allocate(self, home_count: int, replica_capacity: int, shapes: tuple[tuple[int, ...], tuple[int, ...]],
                 transposed: tuple[bool, bool], *, caller_accumulator: bool = False) -> None:
        """Allocate once; the caller converges admission before submitting work."""
        self._check_geometry(home_count, replica_capacity, shapes, transposed)
        if not isinstance(caller_accumulator, bool) or caller_accumulator and self.dtype != torch.float32:
            raise ValueError("only an FP32 gradient pool can borrow a caller accumulator")
        self.home_count, self.replica_capacity = home_count, replica_capacity
        self.shapes, self.transposed = shapes, transposed
        self.caller_accumulator = caller_accumulator
        if self.dtype == torch.bfloat16:
            self._allocate_slots()
        elif not caller_accumulator:
            self.accumulator = torch.empty(home_count * max(map(prod, shapes)), dtype=self.dtype, device=self.device)
            self.home = {role: self.accumulator[:home_count * prod(shape)].view(home_count, *shape)
                         for role, shape in zip(("gate", "down"), shapes)}

    @staticmethod
    def _check_geometry(home_count: int, replica_capacity: int, shapes: tuple[tuple[int, ...], ...],
                        transposed: tuple[bool, ...]) -> None:
        if isinstance(home_count, bool) or not isinstance(home_count, int) or home_count < 1:
            raise ValueError("replica admission requires a positive home expert count")
        if isinstance(replica_capacity, bool) or not isinstance(replica_capacity, int) or replica_capacity < 0:
            raise ValueError("replica capacity must be a nonnegative integer")
        ReplicaPool._check_matrix_geometry(shapes, transposed)

    @staticmethod
    def _check_matrix_geometry(shapes: tuple[tuple[int, ...], ...], transposed: tuple[bool, ...]) -> None:
        """Check role geometry after validating the expert and replica capacities."""
        if len(shapes) != 2 or len(transposed) != 2 or any(not isinstance(value, bool) for value in transposed):
            raise ValueError("replica admission requires gate/down shapes and physical layouts")
        if any(len(shape) != 2 or any(isinstance(size, bool) or not isinstance(size, int) or size < 1
                                      for size in shape) for shape in shapes):
            raise ValueError("each replica matrix shape must contain two positive integers")

    def _allocate_slots(self) -> None:
        for _ in range(2):
            slot = ReplicaSlot()
            self.slots.append(slot)
            self._allocate_slot(slot)

    def _allocate_slot(self, slot: ReplicaSlot) -> None:
        for role, shape, transpose in zip(("gate", "down"), self.shapes, self.transposed):
            physical = tuple(reversed(shape)) if transpose else shape
            slot.matrices[role] = torch.empty((self.replica_capacity, *physical), dtype=self.dtype,
                                              device=self.device)
            if transpose:
                slot.matrices[role] = slot.matrices[role].transpose(-2, -1)

    def retire_slots(self) -> None:
        """Drop idle W storage references while preserving generations and fences."""
        if self.dtype != torch.bfloat16 or len(self.slots) != 2 or any(slot.in_use for slot in self.slots):
            raise RuntimeError("weight retirement requires two admitted idle slots")
        for slot in self.slots:
            slot.matrices.clear()

    def restore_slots(self) -> None:
        """Rebuild retired W storage locally; the caller must admit it collectively."""
        if (self.dtype != torch.bfloat16 or len(self.slots) != 2
                or any(slot.in_use or slot.matrices for slot in self.slots)):
            raise RuntimeError("weight restoration requires two empty idle slots")
        try:
            for slot in self.slots:
                self._allocate_slot(slot)
        except BaseException:
            self.retire_slots()
            raise

    @property
    def reserved_bytes(self) -> int:
        """Return physical bytes without counting logical views twice."""
        if self.accumulator is not None:
            return self.accumulator.numel() * self.accumulator.element_size()
        return sum(value.numel() * value.element_size() for slot in self.slots for value in slot.matrices.values())

    def clear_unsubmitted(self) -> None:
        """Drop partial initialization allocations before any work is submitted."""
        self.home.clear()
        self.accumulator = None
        for slot in self.slots:
            # Allocation failures retain frame locals in their traceback.
            slot.matrices.clear()
        self.slots.clear()
