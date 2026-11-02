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
"""Positive-capacity symmetric windows for the independent sparse transport."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from threading import RLock
from typing import Any, Iterator

import torch
import torch.distributed as dist

from hyper_parallel.core.multicore import shmem
from hyper_parallel.core.multicore.shmem._runtime import _load_native


@dataclass(frozen=True)
class RingSpec:
    """A uniform root-wide manifest, independent of local expert/token counts."""

    ranks: int
    hidden: int
    block_rows: int = 256
    lanes: int = 2

    def __post_init__(self) -> None:
        """Reject unsupported sizes before any collective allocation."""
        values = (self.ranks, self.hidden, self.block_rows, self.lanes)
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in values):
            raise ValueError("ring dimensions must be positive integers")
        if self.ranks > 48 or self.lanes > 2 or self.hidden > 32768 or self.hidden % 16 or self.block_rows > 1024:
            raise ValueError("ring dimensions exceed the independent transport kernel's limits")

    @property
    def data_shape(self) -> tuple[int, ...]:
        """Return two disjoint directional windows in BF16 elements."""
        return (2, self.ranks, self.lanes, self.block_rows, self.hidden)

    @property
    def signal_shape(self) -> tuple[int, ...]:
        """Return [direction, ready/credit, peer, lane, cacheline] INT32 words."""
        return (2, 2, self.ranks, self.lanes, 16)

    @property
    def header_shape(self) -> tuple[int, ...]:
        """Return fixed-size per-block metadata slots for device-counted rows."""
        words = ((self.block_rows + 1 + 7) // 8) * 8
        return (2, self.ranks, self.lanes, words)

    @property
    def requested_bytes(self) -> int:
        """Return payload plus control bytes before heap alignment and reservation."""
        payload = 2 * self.ranks * self.lanes * self.block_rows * self.hidden * 2
        control = 2 * 2 * self.ranks * self.lanes * 64
        headers = 2 * self.ranks * self.lanes * self.header_shape[-1] * 8
        return payload + control + headers

    @property
    def heap_bytes(self) -> int:
        """Include both allocation alignments and round the heap to two MiB."""
        quantum = 2 * 1024 * 1024
        return ((self.requested_bytes + 3 * 512 + quantum - 1) // quantum) * quantum


class RingWorkspace:
    """Own one root reference and three allocations until all native users quiesce."""

    def __init__(self, spec: RingSpec, group: Any, *, control_group: Any = None) -> None:
        """Collectively create one uniform manifest, then publish initial credits."""
        self.spec, self.group = spec, group
        self.control_group = group if control_group is None else control_group
        self.closed = False
        self.poisoned = False
        self.closing = False
        self._lock = RLock()
        self._completion = None
        self.last_progress = None
        self.metadata = {}
        self.data = self.signals = self.headers = None
        self.rank = dist.get_rank(group) if dist.is_initialized() else 0
        ranks = dist.get_world_size(group) if dist.is_initialized() else 1
        if ranks != spec.ranks:
            raise ValueError("ring manifest does not match the communication root")
        self.send_sequence = [[0] * ranks for _ in range(2)]
        self.receive_sequence = [[0] * ranks for _ in range(2)]
        self.device = torch.device("npu", torch.npu.current_device())
        self.transport_stream = torch.npu.Stream(device=self.device)
        control_device = "cpu" if control_group is not None else self.device
        self._status = torch.zeros(1, dtype=torch.int32, device=control_device)
        manifest = torch.tensor([1, spec.ranks, spec.hidden, spec.block_rows, spec.lanes,
                                 spec.requested_bytes], dtype=torch.int64, device=self.device)
        if ranks > 1:
            manifests = torch.empty((ranks * manifest.numel(),), dtype=manifest.dtype, device=self.device)
            dist.all_gather_into_tensor(manifests, manifest, group=group)
            if not bool((manifests.reshape(ranks, -1) == manifest).all().item()):
                raise ValueError("routed-expert ring manifests differ across the communication root")
        shmem.acquire(group, heap_size_bytes=spec.heap_bytes)
        try:
            state = _load_native()._debug_state()  # pylint: disable=protected-access
            error = (MemoryError("symmetric heap has insufficient remaining capacity")
                     if state["remaining_bytes"] < spec.requested_bytes + 3 * 512 else None)
            self.check_preflight(error)
            for name, shape, dtype in (("data", spec.data_shape, torch.bfloat16),
                                       ("signals", spec.signal_shape, torch.int32),
                                       ("headers", spec.header_shape, torch.int64)):
                error = None
                try:
                    setattr(self, name, shmem.empty(shape, dtype=dtype, alignment=512))
                except Exception as exc:  # All peers converge before the next allocation.
                    error = exc
                self.check_preflight(error)
            self.signals.zero_()
            for lane in range(spec.lanes):
                self.signals.select(1, 1).select(2, lane).select(2, 0).fill_(lane + 1)
            shmem.host_barrier()
        except BaseException:
            # An asymmetric collective allocation failure cannot safely be unwound with free.
            self.poisoned = True
            raise

    def check_preflight(self, error: Exception | None) -> None:
        """Converge local preparation failures before any rank publishes payload."""
        self._status.fill_(int(error is not None))
        if self.spec.ranks > 1:
            dist.all_reduce(self._status, op=dist.ReduceOp.MAX, group=self.control_group)
        # This explicit host confirmation is part of the measured protocol cost.
        if int(self._status.item()):
            self.poisoned = True
            if error is not None:
                raise RuntimeError("routed-expert preflight failed on this rank") from error
            raise RuntimeError("routed-expert preflight failed on another rank")

    @contextmanager
    def submission(self, device: torch.device) -> Iterator[None]:
        """Serialize sequence reservations and join the previous phase across streams."""
        with self._lock:
            if self.closed or self.poisoned or self.closing:
                raise RuntimeError("routed-expert workspace is not usable")
            if device != self.data.device:
                raise ValueError("transport device does not match the window allocation")
            with torch.npu.device(self.device):
                stream = torch.npu.current_stream(device)
                if self._completion is not None:
                    stream.wait_event(self._completion)
                try:
                    yield
                    self._completion = torch.npu.Event()
                    self._completion.record(stream)
                except BaseException:
                    self.poisoned = True
                    raise

    def close(self) -> None:
        """Collectively free complete allocations after all local streams quiesce."""
        with self._lock:
            if self.poisoned:
                raise RuntimeError("poisoned routed-expert windows require whole-job termination")
            if self.closed:
                return
            self.closing = True
            try:
                with torch.npu.device(self.device):
                    torch.npu.synchronize(self.device)
                    shmem.host_barrier()
                    shmem.free(self.signals)
                    shmem.free(self.headers)
                    shmem.free(self.data)
                    shmem.host_barrier()
                    shmem.release()
            except BaseException:
                self.poisoned = True
                raise
            self.signals = self.data = self.headers = None
            self.closed = True
