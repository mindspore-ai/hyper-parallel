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
"""Owned asynchronous Ulysses tickets for Head Chunk's manual autograd boundary.

These transports intentionally do not create their own autograd edges. The
enclosing Head Chunk Function supplies the inverse operations in its explicit
backward. They must not be used as differentiable standalone tensor operations.
Layouts are the same as the shared synchronous Ulysses primitives.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist

from hyper_parallel.distributed.context_parallel.collectives import (
    _prepare_ulysses_send,
    _reconstruct_all_to_all,
)


logger = logging.getLogger(__name__)


def _record_consumer_stream(*tensors: torch.Tensor | None) -> None:
    """Keep launched buffers alive through the stream's collective wait."""
    for tensor in tensors:
        if tensor is not None and tensor.device.type in ("npu", "cuda"):
            device_module = getattr(torch, tensor.device.type)
            tensor.record_stream(device_module.current_stream(tensor.device))


@dataclass(eq=False)
class _AllToAllTicket:
    """Own communication storage and any source that still needs protection."""

    source: torch.Tensor | None
    send: torch.Tensor | None
    received: torch.Tensor | None
    work: Any
    gather_dim: int
    identity: bool = False
    communication_waited: bool = False
    closed: bool = False
    launch_stream: Any | None = None
    trim_stream_records: bool = False
    stream_records_trimmed: bool = False
    cross_stream_fallback: bool = False

    def _finish_communication(self) -> None:
        if not self.communication_waited:
            if self.work is not None:
                self.work.wait()
            if self.trim_stream_records and self.launch_stream is not None:
                current = torch.npu.current_stream(self.send.device)
                self.stream_records_trimmed = current == self.launch_stream
                self.cross_stream_fallback = not self.stream_records_trimmed
            # HCCL wait already orders this stream after communication. Under
            # the private same-allocation-stream contract, recording it again
            # can defer reuse until the NPU allocator's long queued-work tail.
            if not self.stream_records_trimmed:
                _record_consumer_stream(self.source, self.send, self.received)
            self.communication_waited = True

    def _release(self) -> None:
        self.source = None
        self.send = None
        self.received = None
        self.work = None
        self.launch_stream = None
        self.closed = True

    def wait(self) -> torch.Tensor:
        """Wait before reconstruction and consume the ticket exactly once."""
        if self.closed:
            raise RuntimeError("Head Chunk communication ticket has already been consumed")
        self._finish_communication()
        result = self.source if self.identity else _reconstruct_all_to_all(self.received, self.gather_dim)
        self._release()
        return result

    def drain(self) -> None:
        """Wait for an already-launched operation without materializing its layout."""
        if not self.closed:
            self._finish_communication()
            self._release()


class HeadChunkTransport:
    """Invocation-local registry of at most two outstanding collective tickets.

    Launches run on the caller's compute stream and existing CP process group.
    The backend orders its communication stream after that producer stream;
    ``work.wait`` orders the consuming stream before any output is accessed.
    Explicit source/send/receive references and consumer-stream records protect
    allocation lifetime without relying on a temporary packed-send variable.

    Call ``close`` on every normal or exceptional exit. A process-group failure
    remains fatal to the distributed invocation: draining does not repair a
    rank-divergent collective schedule or submit missing peer collectives.

    Optional NPU stream-record trimming relies on this private call-site
    contract: Q, FA output and dQ storage originate on the launch stream;
    packing and receive allocation also use that stream. Reconstruction and
    all downstream reads stay on the waiting stream. Only launch/wait stream
    equality is checked at runtime; Tensor has no allocation-stream query.
    A cross-stream wait retains the original explicit record protection.
    This is not a general-purpose external-tensor transport contract.

    Optional source release uses that same private allocation-stream contract.
    After an independent packed send has been produced on the NPU compute
    stream, communication reads send rather than source. Dropping the source
    reference does not cancel pack or its stream ordering. CPU/CUDA, identity
    transports and aliased packed storage keep the conservative ownership.
    Release counters describe references relinquished by this transport, not
    guaranteed allocator savings: graphs and offload can still own the source.
    """

    def __init__(
        self, cp_mesh: Any | None, *, trim_stream_records: bool = False, release_source: bool = False,
    ) -> None:
        if not isinstance(trim_stream_records, bool):
            raise ValueError("trim_stream_records must be a boolean")
        if not isinstance(release_source, bool):
            raise ValueError("release_source must be a boolean")
        self.cp_mesh = cp_mesh
        self.trim_stream_records = trim_stream_records
        self.release_source = release_source
        self._pending: list[_AllToAllTicket] = []
        self._launches = 0
        self._peak_pending = 0
        self._trimmed_waits = 0
        self._recorded_waits = 0
        self._cross_stream_fallbacks = 0
        self._released_sources = 0
        self._released_source_bytes = 0
        self._release_source_aliases = 0
        self._release_source_fallbacks = 0

    def __enter__(self) -> HeadChunkTransport:
        return self

    def __exit__(self, exception_type: Any, exception: Any, traceback: Any) -> bool:
        try:
            self.close()
        except Exception:
            if exception is None:
                raise
            logger.exception("Failed to drain Head Chunk communication while handling an earlier error")
        return False

    def launch(self, tensor: torch.Tensor, *, inverse: bool = False) -> _AllToAllTicket:
        """Submit one fixed-layout transport from inside the manual Function.

        Args:
            tensor: Local BNSD stage payload, not an ordinary autograd input.
            inverse: Gather heads/scatter sequence instead of the forward map.

        Raises:
            RuntimeError: If used with grad recording or more than two tickets.
        """
        if torch.is_grad_enabled():
            raise RuntimeError("Head Chunk private transport requires its explicit autograd Function boundary")
        if len(self._pending) >= 2:
            raise RuntimeError("Head Chunk permits at most two outstanding communication tickets")
        size = 1 if self.cp_mesh is None else self.cp_mesh.size()
        gather_dim, scatter_dim = (1, 2) if inverse else (2, 1)
        if size <= 1:
            ticket = _AllToAllTicket(tensor, None, None, None, gather_dim, identity=True)
            self._release_source_fallbacks += int(self.release_source)
        else:
            launch_stream = None
            if self.trim_stream_records and tensor.device.type == "npu":
                launch_stream = torch.npu.current_stream(tensor.device)
            send = _prepare_ulysses_send(tensor.detach(), scatter_dim=scatter_dim, world_size=size)
            received = torch.empty_like(send)
            # Autograd is supplied by the enclosing Function, including inverse
            # Q/O/dQ transports. No general-purpose differentiable API is changed.
            work = dist.all_to_all_single(received, send, group=self.cp_mesh.get_group(), async_op=True)
            source = tensor
            if self.release_source:
                if tensor.device.type != "npu":
                    self._release_source_fallbacks += 1
                elif send.untyped_storage().data_ptr() == tensor.untyped_storage().data_ptr():
                    self._release_source_aliases += 1
                else:
                    # Only the same-stream pack reads this independent source;
                    # the pending HCCL operation holds its separate send storage.
                    source = None
                    self._released_sources += 1
                    self._released_source_bytes += tensor.untyped_storage().nbytes()
            ticket = _AllToAllTicket(
                source, send, received, work, gather_dim,
                launch_stream=launch_stream, trim_stream_records=launch_stream is not None,
            )
        self._pending.append(ticket)
        self._launches += 1
        self._peak_pending = max(self._peak_pending, len(self._pending))
        return ticket

    def wait(self, ticket: _AllToAllTicket) -> torch.Tensor:
        """Consume a registered ticket and forget its completed operation."""
        if ticket not in self._pending:
            raise RuntimeError("Head Chunk communication ticket does not belong to this invocation")
        result = ticket.wait()
        self._record_completion(ticket)
        self._pending.remove(ticket)
        return result

    def _record_completion(self, ticket: _AllToAllTicket) -> None:
        if not ticket.identity:
            self._trimmed_waits += int(ticket.stream_records_trimmed)
            self._recorded_waits += int(not ticket.stream_records_trimmed)
            self._cross_stream_fallbacks += int(ticket.cross_stream_fallback)

    def statistics(self) -> dict[str, int]:
        """Return invocation-local counters without tensors or synchronization."""
        return {
            "launches": self._launches,
            "pending": len(self._pending),
            "peak_pending": self._peak_pending,
            "trimmed_waits": self._trimmed_waits,
            "recorded_waits": self._recorded_waits,
            "cross_stream_fallbacks": self._cross_stream_fallbacks,
            "released_sources": self._released_sources,
            "released_source_bytes": self._released_source_bytes,
            "release_source_aliases": self._release_source_aliases,
            "release_source_fallbacks": self._release_source_fallbacks,
        }

    def close(self) -> None:
        """Drain every submitted operation, retaining the first cleanup failure."""
        first_error = None
        for ticket in tuple(self._pending):
            try:
                ticket.drain()
            except Exception as error:
                if first_error is None:
                    first_error = error
            else:
                self._record_completion(ticket)
                self._pending.remove(ticket)
        if first_error is not None:
            raise first_error


__all__ = ["HeadChunkTransport"]
