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
"""Prepared single-slot cube/vector forward with bounded activation service."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Optional

import numpy as np
import torch

from hyper_parallel.core.multicore.modules.routed_experts.frame import ExpertState
from hyper_parallel.core.multicore.modules.routed_experts.readiness import ExpertReadiness, upload_metadata
from hyper_parallel.core.multicore.modules.routed_experts.transport import _load_native


_TILE_ROWS = 128
_MAX_EXPERTS = 128
_WATCHDOG = 60_000_000_000
Allocator = Callable[[str, tuple[int, ...], torch.dtype], Optional[torch.Tensor]]


def forward_vector_budget(transport_workers: int) -> int:
    """Reserve eight service workers only when persistent transport leaves them free.

    The caller supplies its actual transport launch budget. This is a necessary
    occupancy bound; successful scheduling still requires device validation.
    Standalone compute uses zero transport workers and may reserve eight.
    """
    if (isinstance(transport_workers, bool) or not isinstance(transport_workers, int)
            or not 0 <= transport_workers <= 48):
        raise ValueError("persistent transport workers must be an integer between zero and 48")
    return 8 if transport_workers <= 40 else 0


def forward_tasks(boundaries: tuple[int, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    """Build actual full-column row chunks and each participating C/expert cell."""
    if (len(boundaries) > _MAX_EXPERTS or any(isinstance(end, bool) or not isinstance(end, int)
            or end <= begin for begin, end in zip((0, *boundaries[:-1]), boundaries))
            or boundaries and boundaries[-1] > 2**20):
        raise ValueError("forward boundaries must contain at most 128 positive expert ranges within 2**20 rows")
    if not boundaries:
        return torch.tensor([], dtype=torch.int64).reshape(-1, 3), torch.from_numpy(np.zeros((0, 0), dtype=np.int32))
    ends = np.asarray(boundaries, dtype=np.int64)
    lengths = np.diff(ends, prepend=0)
    counts = (lengths + _TILE_ROWS - 1) // _TILE_ROWS
    experts = np.repeat(np.arange(len(boundaries), dtype=np.int64), counts)
    ordinal = np.arange(len(experts), dtype=np.int64) - np.repeat(np.cumsum(counts) - counts, counts)
    table = np.empty((len(experts), 3), dtype=np.int64)
    table[:, 0] = experts
    table[:, 1] = (ends - lengths)[experts] + ordinal * _TILE_ROWS
    table[:, 2] = np.minimum(_TILE_ROWS, ends[experts] - table[:, 1])
    cores = min(24, len(table))
    expected = np.zeros((cores, len(boundaries)), dtype=np.int32)
    expected[np.arange(len(table)) % cores, experts] = 1
    return torch.from_numpy(table), torch.from_numpy(expected)


def _check_tensor(value: torch.Tensor, shape: tuple[int, ...], dtype: torch.dtype,
                  device: torch.device, name: str, *, dense: bool = False) -> None:
    if value.shape != shape or value.dtype != dtype or value.device != device:
        raise ValueError(f"{name} must have shape {shape}, dtype {dtype}, and device {device}")
    if not (value.is_contiguous() or dense and value.ndim == 2 and value.t().is_contiguous()):
        raise ValueError(f"{name} must have a supported dense layout")
    if value.numel() and (value.data_ptr() == 0 or value.data_ptr() % 32
                          or value.untyped_storage().nbytes()
                          < (value.storage_offset() + value.numel()) * value.element_size()):
        raise ValueError(f"{name} storage is invalid, unaligned, or undersized")


def _check_writers(writers: Sequence[torch.Tensor], readers: Sequence[torch.Tensor]) -> None:
    writers, readers = tuple(writers), tuple(readers)
    sizes = tuple(value.numel() for value in writers)
    if not any(sizes):
        return
    sizes += tuple(value.numel() for value in readers)
    extents = tuple(None if not size else (value.untyped_storage().data_ptr(), value.data_ptr(),
                                          size * value.element_size())
                    for value, size in zip(writers + readers, sizes))
    for index, writer in enumerate(extents[:len(writers)]):
        if writer is None:
            continue
        for other_index, other in enumerate(extents):
            if index == other_index or other is None:
                continue
            if writer[0] == other[0]:
                raise ValueError("forward writable storage must not alias another operand")
            if writer[1] < other[1] + other[2] and other[1] < writer[1] + writer[2]:
                raise ValueError("forward writable storage must not overlap another operand")


@lru_cache(maxsize=16)
def _forward_config(hidden: int, transposed: bool) -> torch.Tensor:
    _load_native()
    config, workspace = torch.ops.hyper_parallel.routed_experts_forward_config(_TILE_ROWS, hidden, transposed)
    if workspace:
        raise RuntimeError("forward cube must not require a vector Matmul service workspace")
    size = torch.ops.hyper_parallel.routed_experts_forward_config_bytes()
    _check_tensor(config, (size,), torch.uint8, torch.device("cpu"), "forward configuration")
    return config


@dataclass
class PipelineForwardWork:
    """Retain prepared operands until C has joined the independent V service."""

    owner: object
    inputx: torch.Tensor
    z: torch.Tensor
    activation: torch.Tensor
    output: torch.Tensor
    ready: ExpertReadiness | None
    state: ExpertState
    gate: tuple[torch.Tensor, ...]
    down: tuple[torch.Tensor, ...]
    gate_table: torch.Tensor | None
    down_table: torch.Tensor | None
    tasks: torch.Tensor
    config: torch.Tensor | None
    mailboxes: torch.Tensor
    initialized: Any
    vector_complete: Any
    vector_stream: Any
    transposed: bool
    cores: int
    submitted: bool = False


class ForwardPipeline:
    """Prepare one full-N row pipeline using a shared bounded vector service stream."""

    def __init__(self, boundaries: tuple[int, ...], device: torch.device, vector_stream: Any = None,
                 vector_workers: int = 8) -> None:
        """Retain immutable host boundaries; defer device resources to phase admission."""
        if isinstance(vector_workers, bool) or vector_workers not in (0, 8):
            raise ValueError("forward service must be disabled or have eight available vector workers")
        self.enabled = bool(vector_workers)
        self.boundaries = boundaries
        self.device = device
        self.vector_stream = vector_stream
        self.metadata: dict[tuple, tuple] = {}

    @staticmethod
    def supports(hidden: int, intermediate: int, gate: Sequence[torch.Tensor],
                 down: Sequence[torch.Tensor], expert_count: int) -> bool:
        """Enable only dimensions and physical layouts covered by device parity probes."""
        if (hidden, intermediate) != (2048, 768) or expert_count > _MAX_EXPERTS:
            return False
        if not gate or not down:
            return not gate and not down and not expert_count
        if any(weight.ndim != 2 for weight in (*gate, *down)):
            return False
        transpose = gate[0].stride(-2) == 1
        return all((weight.stride(-2) == 1) == transpose for weight in (*gate, *down))

    def _allocate(self, allocator: Allocator | None, name: str, shape: tuple[int, ...]) -> torch.Tensor:
        value = None if allocator is None else allocator(name, shape, torch.bfloat16)
        if value is None:
            value = torch.empty(shape, dtype=torch.bfloat16, device=self.device)
        _check_tensor(value, shape, torch.bfloat16, self.device, name)
        return value

    def _check_forward_operands(self, x: torch.Tensor, gate: Sequence[torch.Tensor],
                                down: Sequence[torch.Tensor], intermediate: int, rows: int) -> tuple:
        """Check pipeline support and expert matrices before stream or allocation work."""
        if (not self.enabled or x.ndim != 2
                or not self.supports(x.shape[1], intermediate, gate, down, len(self.boundaries))):
            raise ValueError("forward pipeline dimensions or physical layouts have not been admitted")
        hidden, gate, down = x.shape[1], tuple(gate), tuple(down)
        if len(gate) != len(self.boundaries) or len(down) != len(self.boundaries):
            raise ValueError("forward weights must cover every compact active expert")
        _check_tensor(x, (rows, hidden), torch.bfloat16, self.device, "forward input")
        for matrices, shape in ((gate, (hidden, 2 * intermediate)), (down, (intermediate, hidden))):
            for matrix in matrices:
                _check_tensor(matrix, shape, x.dtype, self.device, "forward weight", dense=True)
        return hidden, gate, down

    def _prepare_metadata(self, hidden: int, transposed: bool, gate: tuple[torch.Tensor, ...],
                          current: Any) -> tuple:
        """Create or join immutable tasks and expectations before frame allocations."""
        key = hidden, transposed
        if key not in self.metadata:
            host_tasks, expected = forward_tasks(self.boundaries)
            tasks = upload_metadata(host_tasks, self.device)
            config = upload_metadata(_forward_config(hidden, transposed), self.device) if gate else None
            expected_device = upload_metadata(expected, self.device) if expected.shape[0] else None
            event = torch.npu.Event()
            event.record(current)
            self.metadata[key] = tasks, config, expected, expected_device, event
        tasks, config, expected, expected_device, metadata_ready = self.metadata[key]
        current.wait_event(metadata_ready)
        return tasks, config, expected, expected_device

    @torch.no_grad()
    def prepare(self, x: torch.Tensor, gate: Sequence[torch.Tensor], down: Sequence[torch.Tensor],
                intermediate: int, allocator: Allocator | None = None) -> PipelineForwardWork:
        """Allocate both kernels' storage and events before collective phase admission."""
        _load_native()
        rows = self.boundaries[-1] if self.boundaries else 0
        hidden, gate, down = self._check_forward_operands(x, gate, down, intermediate, rows)
        current = torch.npu.current_stream(self.device)
        transposed = bool(gate and gate[0].stride(-2) == 1)
        tasks, config, expected, expected_device = self._prepare_metadata(hidden, transposed, gate, current)
        cores = expected.shape[0]
        z = torch.empty((rows, 2 * intermediate), dtype=x.dtype, device=self.device)
        activation = self._allocate(allocator, "forward_activation", (cores, _TILE_ROWS, intermediate))
        output = self._allocate(allocator, "project", tuple(x.shape))
        mailboxes = torch.zeros((2, cores, 16), dtype=torch.int32, device=self.device)
        gate_table = torch.ops.hyper_parallel.routed_experts_weight_table(gate) if gate else None
        down_table = torch.ops.hyper_parallel.routed_experts_weight_table(down) if down else None
        readers = [x, tasks, *gate, *down]
        for table in (gate_table, down_table):
            if table is not None:
                _check_tensor(table, (len(gate),), torch.int64, self.device, "forward weight table")
                readers.append(table)
        if config is not None:
            readers.append(config)
        ready = (ExpertReadiness.create(expected, self.device, deferred_poll=True, expected_device=expected_device)
                 if cores else None)
        writers = [z, activation, output, mailboxes]
        if ready is not None:
            writers.append(ready.flags)
            readers.append(ready.expected)
        _check_writers(writers, readers)
        if cores:
            torch.ops.hyper_parallel.routed_experts_forward_validate(
                x, gate, down, gate_table, down_table, tasks, z, activation, output, config,
                mailboxes, _TILE_ROWS, transposed, cores, _WATCHDOG)
        if cores and self.vector_stream is None:
            self.vector_stream = torch.npu.Stream(device=self.device)
        initialized, vector_complete = torch.npu.Event(), torch.npu.Event()
        initialized.record(current)
        vector_complete.record(current)
        return PipelineForwardWork(self, x, z, activation, output, ready, ExpertState(x, z), gate, down,
                                   gate_table, down_table, tasks, config, mailboxes, initialized,
                                   vector_complete, self.vector_stream, transposed, cores)

    def _check_dependency(self, work: PipelineForwardWork, dependency: ExpertReadiness | None) -> None:
        if dependency is None:
            return
        dependency.validate_polling()
        flags, expected = dependency.flags, dependency.expected
        if flags.ndim != 3 or not 1 <= flags.shape[0] <= 48:
            raise ValueError("forward dependency requires between one and 48 producers")
        _check_tensor(flags, (flags.shape[0], len(self.boundaries), 16), torch.int32,
                      self.device, "forward input flags")
        _check_tensor(expected, tuple(flags.shape[:2]), torch.int32, self.device, "forward input expectations")
        if dependency.task_expected is not None:
            _check_tensor(dependency.task_expected, (flags.shape[0], work.tasks.shape[0]),
                          torch.int32, self.device, "forward task expectations")
        if flags.numel() and flags.data_ptr() % 64:
            raise ValueError("forward readiness cells must be aligned to 64 bytes")
        writers = [work.z, work.activation, work.output, work.mailboxes]
        if work.ready is not None:
            writers.append(work.ready.flags)
        readers = (flags, expected) if dependency.task_expected is None else (flags, expected,
                                                                             dependency.task_expected)
        _check_writers(writers, readers)

    @torch.no_grad()
    def run(self, work: PipelineForwardWork, dependency: ExpertReadiness | None = None) -> tuple:
        """Submit V/C after all C predecessors; join V before returning to the caller."""
        if work.owner is not self or work.submitted:
            raise ValueError("forward work belongs to another pipeline or was already submitted")
        self._check_dependency(work, dependency)
        work.submitted = True
        current = torch.npu.current_stream(self.device)
        current.wait_event(work.initialized)
        if dependency is not None:
            current.wait_event(dependency.admitted)
            current.wait_event(dependency.poll_after.event)
        if not work.cores:
            return work.output, work.state, None
        # V must also wait for preceding C-stream producers, including the W lease fence.
        work.ready.poll_after.event.record(current)
        with torch.npu.stream(work.vector_stream):
            work.vector_stream.wait_event(work.ready.poll_after.event)
            torch.ops.hyper_parallel.routed_experts_forward_vector(
                work.z, work.activation, work.tasks, work.mailboxes, len(self.boundaries),
                _TILE_ROWS, work.cores, min(8, work.cores), _WATCHDOG)
            work.vector_complete.record(work.vector_stream)
        torch.ops.hyper_parallel.routed_experts_forward_cube(
            work.inputx, work.gate, work.down, work.gate_table, work.down_table, work.tasks, work.z,
            work.activation, work.output, work.config, work.mailboxes, _TILE_ROWS, work.transposed,
            work.cores, _WATCHDOG, None if dependency is None else dependency.flags,
            None if dependency is None else (dependency.task_expected
                                             if dependency.task_expected is not None else dependency.expected),
            work.ready.flags)
        work.ready.poll_after.queued = True
        current.wait_event(work.vector_complete)
        return work.output, work.state, work.ready
