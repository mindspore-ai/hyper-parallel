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
"""Root-owned temporary regions reused by one compute phase at a time."""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from math import prod
from threading import RLock
from typing import Any

import torch


_QUANTUM_BYTES = 2 * 1024 * 1024
_MAX_PENDING_PHASES = 2


class PhaseArena:
    """Keep independent named storage at its rounded high-water capacity.

    Only temporary tensors belong here. Saved activations and tensors returned to
    autograd must retain their own storage. All begin/allocate/end calls use one
    compute stream. Before end, that stream must wait for every T/G reader and
    writer. End records a completion event without waiting on the host. Begin
    limits host submission to two in-flight phases by waiting for the oldest
    event when necessary. Independent weight prefetch is outside this queue.

    Each name has one fixed dtype and independent storage. A phase may allocate a
    name only once; intentional within-phase reuse must use views of that result.
    Growth allocates a replacement before dropping the old region, so allocation
    failure preserves the prior capacity for the caller's admission protocol.
    """

    def __init__(self, device: torch.device) -> None:
        """Create an empty arena; no device allocation occurs before allocate."""
        self.device = torch.device(device)
        self._regions: dict[str, torch.Tensor] = {}
        self._capacities: dict[str, tuple[torch.dtype, int]] = {}
        self._used: set[str] = set()
        self._active: object | None = None
        self._bound_stream: Any = None
        self._stream_bound = False
        self._pending: deque[Any] = deque()
        self._closed = False
        self._lock = RLock()

    @property
    def reserved_bytes(self) -> int:
        """Return retained region bytes, excluding replaced in-flight storage."""
        with self._lock:
            return sum(value.numel() * value.element_size() for value in self._regions.values())

    def begin(self) -> object:
        """Wait for an available in-flight phase slot, then return its release token."""
        with self._lock:
            self._check_open()
            if self._active is not None:
                raise RuntimeError("temporary arena already has an active phase")
            self._check_stream(bind=True)
            if len(self._pending) >= _MAX_PENDING_PHASES:
                self._wait_oldest()
            self._active = object()
            self._used.clear()
            return self._active

    @torch.no_grad()
    def allocate(self, name: str, shape: Sequence[int], dtype: torch.dtype) -> torch.Tensor:
        """Borrow a dense view, growing only this name during phase preparation.

        The caller must converge allocation failures before launching the phase.
        A failed allocation does not consume the name or alter its old storage.
        Returned views expire at end even if Python references remain alive.
        """
        with self._lock:
            self._check_open()
            if self._active is None:
                raise RuntimeError("temporary allocation requires an active phase")
            self._check_stream()
            shape = self._check_request(name, shape, dtype)
            if name in self._used:
                raise ValueError("a temporary region may be requested only once per phase")
            previous = self._regions.get(name)
            admitted_dtype, capacity = self._capacities.get(name, (dtype, 0))
            if admitted_dtype != dtype:
                raise ValueError("a named temporary region cannot change dtype")
            elements = prod(shape)
            if previous is None or previous.numel() < elements:
                rounded_bytes = ((max(elements, capacity) * dtype.itemsize + _QUANTUM_BYTES - 1)
                                 // _QUANTUM_BYTES * _QUANTUM_BYTES)
                replacement = torch.empty(rounded_bytes // dtype.itemsize, dtype=dtype, device=self.device)
                self._regions[name] = replacement
                self._capacities[name] = dtype, replacement.numel()
            result = self._regions[name][:elements].view(shape)
            self._used.add(name)
            return result

    def end(self, token: object) -> None:
        """Record completion after C joins external uses; never wait on the host."""
        with self._lock:
            self._check_open()
            if self._active is None or token is not self._active:
                raise ValueError("temporary arena release token is foreign, expired, or already released")
            self._check_stream()
            if self.device.type != "cpu":
                completion = getattr(torch, self.device.type).Event()
                completion.record(self._bound_stream)
                self._pending.append(completion)
            self._active = None
            self._used.clear()

    def release_regions(self, names: Sequence[str]) -> int:
        """Drop selected idle regions while remembering their dtype and capacity.

        Queued commands and allocator stream records protect outstanding device
        users. This removes only the arena's references: it neither invalidates
        storage nor drains or discards completion events. Rebuilding a region
        uses its previous capacity during the caller's next phase admission.

        Returns:
            Bytes no longer retained by the arena, excluding other live owners.
        """
        with self._lock:
            self._check_open()
            if self._active is not None:
                raise RuntimeError("cannot release temporary regions with an active phase")
            if self._stream_bound:
                self._check_stream()
            if (not isinstance(names, Sequence) or isinstance(names, str)
                    or any(not isinstance(name, str) or not name for name in names)):
                raise ValueError("temporary region release requires a sequence of nonempty names")
            released = 0
            for name in names:
                region = self._regions.pop(name, None)
                if region is not None:
                    released += region.numel() * region.element_size()
            return released

    def close(self) -> None:
        """Drain submitted phases before dropping regions; repeated close is safe."""
        with self._lock:
            if self._closed:
                return
            if self._active is not None:
                raise RuntimeError("cannot close a temporary arena with an active phase")
            if self._stream_bound:
                self._check_stream()
            while self._pending:
                self._wait_oldest()
            self._regions.clear()
            self._capacities.clear()
            self._used.clear()
            self._closed = True

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("temporary arena is closed")

    def _wait_oldest(self) -> None:
        completion = self._pending[0]
        if not completion.query():
            completion.synchronize()
        self._pending.popleft()

    def _check_stream(self, *, bind: bool = False) -> None:
        current = None if self.device.type == "cpu" else getattr(torch, self.device.type).current_stream(self.device)
        if not self._stream_bound and bind:
            self._bound_stream = current
            self._stream_bound = True
        elif self._stream_bound and current != self._bound_stream:
            raise RuntimeError("temporary arena operations must use its original compute stream")

    @staticmethod
    def _check_request(name: str, shape: Sequence[int], dtype: torch.dtype) -> tuple[int, ...]:
        if not isinstance(name, str) or not name:
            raise ValueError("temporary region name must be a nonempty string")
        if not isinstance(shape, Sequence):
            raise ValueError("temporary region shape must be a sequence of nonnegative integers")
        if any(isinstance(size, bool) or not isinstance(size, int) or size < 0 for size in shape):
            raise ValueError("temporary region dimensions must be nonnegative integers")
        if not isinstance(dtype, torch.dtype) or dtype.itemsize < 1:
            raise ValueError("temporary region dtype must have a positive whole-byte element size")
        return tuple(shape)
