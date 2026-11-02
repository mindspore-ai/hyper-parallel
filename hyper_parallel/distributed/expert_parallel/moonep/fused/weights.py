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
"""Bounded replica payloads; home weights always remain owned by FSDP."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import wraps
from math import prod
from threading import RLock
from typing import Any

import torch
import torch.distributed as dist

from hyper_parallel.distributed.expert_parallel.moonep.fused.pool import ReplicaPool, check_dense, overlaps
from hyper_parallel.distributed.expert_parallel.moonep.fused.transfers import (
    Fragment, TransferRound, copy_matrix_fragment,
)


def _submission(method: Callable) -> Callable:
    """Serialize a complete domain submission and poison it on any failure."""
    @wraps(method)
    def _wrapped(self, *args, **kwargs):
        with self._lock:  # pylint: disable=protected-access
            if self.poisoned:
                raise RuntimeError("replica communication domain is poisoned")
            try:
                return method(self, *args, **kwargs)
            except BaseException:
                self.poisoned = True
                raise
    return _wrapped


@dataclass
class WeightLease:
    """Remote views borrowed until release; home weights remain owned by FSDP."""

    remote: dict[tuple[int, str], torch.Tensor]
    ready: Any
    owner: Any
    slot: int | None
    generation: int
    released: bool = False


@dataclass
class GradientLease:
    """Complete owner gradients and the event after remote accumulation."""

    value: torch.Tensor
    ready: Any


@dataclass(frozen=True)
class _LocalRound:
    sent: tuple[Fragment, ...]
    received: tuple[Fragment, ...]
    send_sizes: tuple[int, ...]
    receive_sizes: tuple[int, ...]


class TransferDomain:
    """One communicator and stream with storage allocated at admission.

    The caller converges W/G admission and phase preparation across ranks. This
    domain performs no status collective during materialization or gradient return. Explicit
    restoration of retired W storage adds one caller-ordered admission.
    """

    def __init__(self, group: Any, rank: int, device: torch.device, *, dtype: torch.dtype = torch.bfloat16,
                 quota_bytes: int = 64 * 1024 * 1024, control_group: Any = None) -> None:
        """Bind a pre-created communication domain; call admit before submission."""
        if dtype not in (torch.bfloat16, torch.float32):
            raise ValueError("replica domains require BF16 weights or FP32 gradients")
        element_bytes = 2 if dtype == torch.bfloat16 else 4
        if (isinstance(quota_bytes, bool) or not isinstance(quota_bytes, int)
                or quota_bytes < element_bytes or quota_bytes % element_bytes):
            raise ValueError("replica communication quota must contain a positive integral element count")
        self.group, self.rank, self.device = group, rank, device
        self.control_group = group if control_group is None else control_group
        self.dtype = dtype
        self.quota_elements = quota_bytes // element_bytes
        self.stream = torch.npu.Stream(device=device)
        self.buffers: dict[str, torch.Tensor] = {}
        self.status = torch.zeros(1, dtype=torch.int32, device="cpu" if control_group is not None else device)
        self.admitted = False
        self.parked = False
        self.caller_staging = False
        self.poisoned = False
        self._lock = RLock()
        self._ranks = dist.get_world_size(group)
        self._pool = ReplicaPool(device, dtype)
        self._lease_owner = object()

    @torch.no_grad()
    @_submission
    def admit(self, home_count: int, replica_capacity: int, shapes: tuple[tuple[int, ...], tuple[int, ...]],
              transposed: tuple[bool, bool], *, caller_accumulator: bool = False,
              caller_staging: bool = False) -> None:
        """Reserve fixed staging and replica storage during one caller preflight.

        This is local allocation only. The caller must converge success before any
        rank submits payloads. Failed partial allocations are dropped locally.
        A G domain with caller_accumulator=True borrows phase storage on each
        return_gradient call; it never retains or allocates its own accumulator.
        caller_staging similarly borrows both fixed-quota buffers for each G call.
        """
        if self.admitted:
            raise RuntimeError("replica communication domain has already been admitted")
        try:
            if not isinstance(caller_staging, bool) or caller_staging and self.dtype != torch.float32:
                raise ValueError("only an FP32 gradient domain can borrow caller staging")
            if not 0 <= self.rank < self._ranks:
                raise ValueError("replica rank is outside its communication group")
            self._pool.allocate(home_count, replica_capacity, shapes, transposed,
                                caller_accumulator=caller_accumulator)
            if not caller_staging:
                for direction in ("send", "receive"):
                    self.buffers[direction] = torch.empty(self.quota_elements, dtype=self.dtype, device=self.device)
            # The first submission may arrive on a different compute stream.
            self.stream.wait_stream(torch.npu.current_stream(self.device))
        except BaseException:
            self.buffers.clear()
            self._pool.clear_unsubmitted()
            raise
        self.caller_staging = caller_staging
        self.admitted = True

    @property
    def reserved_bytes(self) -> int:
        """Return persistent W/G storage bytes, excluding caller-owned gradients."""
        return self._pool.reserved_bytes + sum(value.numel() * value.element_size()
                                             for value in self.buffers.values())

    @_submission
    def retire_storage(self) -> int:
        """Drop idle W references without draining streams or invalidating views.

        Existing Tensor stream records and queued command references protect
        prior users. Slot generations and last-reader events remain unchanged.
        Returns bytes no longer owned here, not bytes already returned to HBM.
        """
        if not self.admitted or self.dtype != torch.bfloat16:
            raise RuntimeError("only an admitted weight domain can retire storage")
        if self.parked:
            return 0
        released = self.reserved_bytes
        self._pool.retire_slots()
        self.buffers.clear()
        self.parked = True
        return released

    @torch.no_grad()
    @_submission
    def restore_storage(self, error: Exception | None = None) -> None:
        """Rebuild retired W storage and converge once before any W payload.

        Every rank must call at the same admitted first sparse demand, including
        empty receivers. A local allocation or caller validation error is carried
        by the existing control group. Remote failure drops local new references.
        """
        if error is None:
            try:
                if not self.admitted or not self.parked or self.dtype != torch.bfloat16:
                    raise RuntimeError("weight restoration requires an admitted parked domain")
                self._pool.restore_slots()
                for direction in ("send", "receive"):
                    self.buffers[direction] = torch.empty(self.quota_elements, dtype=self.dtype, device=self.device)
                self.stream.wait_stream(torch.npu.current_stream(self.device))
            except Exception as exc:
                error = exc
        try:
            self.check_preflight(error)
        except BaseException:
            if self.parked:
                self.buffers.clear()
                self._pool.retire_slots()
            raise
        self.parked = False

    @property
    def has_free_slot(self) -> bool:
        """Whether W can claim a slot; its last-reader event may still be pending."""
        with self._lock:
            return (self.admitted and not self.parked and not self.poisoned and self.dtype == torch.bfloat16
                    and any(not slot.in_use for slot in self._pool.slots))

    def check_preflight(self, error: Exception | None) -> None:
        """Converge preparation before the fixed-order replica collectives."""
        self.status.fill_(int(error is not None or self.poisoned))
        if self._ranks > 1:
            dist.all_reduce(self.status, op=dist.ReduceOp.MAX, group=self.control_group)
        if int(self.status.item()):
            self.poisoned = True
            raise RuntimeError("replica preparation failed; terminate this distributed execution") from error

    @torch.no_grad()
    def _exchange(self, rounds: tuple[_LocalRound, ...], source: Callable, target: Callable, *,
                  accumulate: bool = False, staging: tuple[torch.Tensor, torch.Tensor] | None = None) -> None:
        """Pack, exchange, and consume each bounded round on this domain's stream.

        The caller has reserved all destination matrices and both buffers before
        entry. Empty ranks participate in the same rounds with zero splits.
        """
        send_buffer, receive_buffer = ((self.buffers["send"], self.buffers["receive"])
                                       if staging is None else staging)
        for item in rounds:
            sent, received, send_sizes, receive_sizes = item.sent, item.received, item.send_sizes, item.receive_sizes
            packed = send_buffer[:sum(send_sizes)]
            returned = receive_buffer[:sum(receive_sizes)]
            packed.record_stream(self.stream)
            returned.record_stream(self.stream)
            offset = 0
            for fragment in sent:
                copy_matrix_fragment(packed[offset:offset + fragment.elements],
                                     source(fragment.expert, fragment.role), fragment.offset)
                offset += fragment.elements
            work = dist.all_to_all_single(returned, packed, list(receive_sizes), list(send_sizes),
                                          group=self.group, async_op=True)
            work.wait()
            offset = 0
            for fragment in received:
                copy_matrix_fragment(returned[offset:offset + fragment.elements],
                                     target(fragment.expert, fragment.role), fragment.offset,
                                     unpack=True, accumulate=accumulate)
                offset += fragment.elements

    @torch.no_grad()
    @_submission
    def materialize(self, active: tuple[int, ...], home_count: int, read_weights: Callable,
                    rounds: tuple[TransferRound, ...], shapes: tuple[tuple[int, ...], tuple[int, ...]],
                    transposed: tuple[bool, bool]) -> WeightLease:
        """Borrow one admitted remote slot, then read current FSDP views on W."""
        self._check_call(active, home_count, torch.bfloat16)
        if shapes != self._pool.shapes or transposed != self._pool.transposed:
            raise ValueError("weight layout differs from replica admission")
        prepared = self._prepare_rounds(rounds, active)
        slot_index, generation, remote = self._claim_slot(active)
        self.stream.wait_stream(torch.npu.current_stream(self.device))
        with torch.npu.stream(self.stream):
            gate, down = read_weights()
            for value, shape, transpose in zip((gate, down), shapes, transposed):
                check_dense(value, (home_count, *shape), torch.bfloat16, self.device, transposed=transpose)
            for value in (gate, down, *remote.values()):
                value.record_stream(self.stream)
            weights = {"gate": gate, "down": down}
            self._exchange(prepared, lambda expert, role: weights[role][expert % home_count],
                           lambda expert, role: remote[expert, role])
            ready = torch.npu.Event()
            ready.record(self.stream)
        return WeightLease(remote, ready, self._lease_owner, slot_index, generation)

    @_submission
    def release_lease(self, lease: WeightLease) -> None:
        """Expire a W borrow after its last use has been enqueued on current C.

        C waits for readiness even if a prefetch was cancelled without a reader.
        Reuse on W waits for the recorded C event. No later reads through this
        lease are valid after release, including reads on unrelated streams.
        """
        if lease.owner is not self._lease_owner or lease.released:
            raise ValueError("weight lease belongs to another domain or has already been released")
        current = torch.npu.current_stream(self.device)
        current.wait_event(lease.ready)
        if lease.slot is not None:
            slot = self._pool.slots[lease.slot]
            if not slot.in_use or slot.generation != lease.generation:
                raise ValueError("weight lease no longer owns its replica slot")
            slot.last_read = torch.npu.Event()
            slot.last_read.record(current)
            slot.in_use = False
        lease.released = True

    @torch.no_grad()
    @_submission
    def return_gradient(self, role: str, active: tuple[int, ...], home_count: int, gradient: torch.Tensor,
                        rounds: tuple[TransferRound, ...], *, output: torch.Tensor,
                        accumulator: torch.Tensor | None = None,
                        staging: tuple[torch.Tensor, torch.Tensor] | None = None) -> GradientLease:
        """Merge in reusable FP32 storage and cast once into caller-owned BF16.

        The output must remain owned by the caller/autograd and must not be reused
        with this domain's staging or accumulator. G serializes zero/merge/cast.
        A borrowed accumulator's earlier C readers must already be enqueued;
        the existing G wait joins those readers before the first overwrite.
        The caller must join the returned fence before reusing borrowed storage.
        Borrowed staging shares that lifetime and is never retained by this domain.
        """
        self._check_call(active, home_count, torch.float32)
        home = self._gradient_storage(role, accumulator)
        buffers = self._gradient_staging(staging)
        check_dense(gradient, (len(active), *home.shape[1:]), torch.float32, self.device)
        check_dense(output, tuple(home.shape), torch.bfloat16, self.device)
        full_accumulator = accumulator if accumulator is not None else self._pool.accumulator
        if any(overlaps(full_accumulator, value) for value in buffers):
            raise ValueError("gradient accumulator must not alias domain staging")
        owned = (*buffers, full_accumulator)
        aliases_pool = any(overlaps(value, tensor) for value in (gradient, output) for tensor in owned)
        if overlaps(output, gradient) or aliases_pool:
            raise ValueError("gradient payload/output must not alias each other, staging, or accumulator storage")
        prepared = self._prepare_rounds(rounds, active, role=role)
        current = torch.npu.current_stream(self.device)
        self.stream.wait_stream(current)
        with torch.npu.stream(self.stream):
            gradient.record_stream(self.stream)
            home.record_stream(self.stream)
            output.record_stream(self.stream)
            home.zero_()
            by_expert = {expert: gradient[index] for index, expert in enumerate(active)}
            for expert, value in by_expert.items():
                if expert // home_count == self.rank:
                    home[expert % home_count].copy_(value)
            self._exchange(prepared, lambda expert, unused_role: by_expert[expert],
                           lambda expert, unused_role: home[expert % home_count], accumulate=True, staging=buffers)
            # Every replica contributes before the single parameter-dtype cast.
            output.copy_(home)
            ready = torch.npu.Event()
            ready.record(self.stream)
        return GradientLease(output, ready)

    def _gradient_staging(self, staging: tuple[torch.Tensor, torch.Tensor] | None
                          ) -> tuple[torch.Tensor, torch.Tensor]:
        """Validate two independent fixed-quota views without retaining caller storage."""
        if self.caller_staging != (staging is not None):
            raise ValueError("gradient staging ownership differs from domain admission")
        if staging is None:
            return self.buffers["send"], self.buffers["receive"]
        if not isinstance(staging, tuple) or len(staging) != 2:
            raise ValueError("gradient staging requires a send/receive tensor pair")
        for value in staging:
            check_dense(value, (self.quota_elements,), self.dtype, self.device)
            if value.data_ptr() % 32:
                raise ValueError("gradient staging must have 32-byte aligned addresses")
        if overlaps(*staging):
            raise ValueError("gradient send and receive staging must not alias")
        return staging

    def _gradient_storage(self, role: str, accumulator: torch.Tensor | None) -> torch.Tensor:
        """Validate a role view without retaining caller-owned phase storage."""
        if role not in ("gate", "down"):
            raise ValueError("gradient role must be gate or down")
        if self._pool.caller_accumulator != (accumulator is not None):
            raise ValueError("gradient accumulator ownership differs from domain admission")
        if accumulator is None:
            return self._pool.home[role]
        elements = self._pool.home_count * max(map(prod, self._pool.shapes))
        check_dense(accumulator, (elements,), torch.float32, self.device)
        if accumulator.data_ptr() % 32:
            raise ValueError("gradient accumulator must have a 32-byte aligned address")
        if any(overlaps(accumulator, value) for value in self.buffers.values()):
            raise ValueError("gradient accumulator must not alias domain staging")
        shape = self._pool.shapes[0 if role == "gate" else 1]
        return accumulator[:self._pool.home_count * prod(shape)].view(self._pool.home_count, *shape)

    def _check_call(self, active: tuple[int, ...], home_count: int, dtype: torch.dtype) -> None:
        if not self.admitted or self.parked or self.dtype != dtype or home_count != self._pool.home_count:
            raise ValueError("replica domain admission does not match this operation")
        if (len(set(active)) != len(active)
                or any(isinstance(expert, bool) or not isinstance(expert, int)
                       or not 0 <= expert < home_count * self._ranks for expert in active)):
            raise ValueError("active expert ids must be unique and within the admitted ownership range")

    def _claim_slot(self, active: tuple[int, ...]) -> tuple[int | None, int, dict[tuple[int, str], torch.Tensor]]:
        experts = tuple(expert for expert in active if expert // self._pool.home_count != self.rank)
        if len(experts) > self._pool.replica_capacity:
            raise ValueError("active remote experts exceed the admitted replica capacity")
        # Empty receivers still consume a ticket so slot availability follows
        # the same collective submission schedule on every rank.
        for index, slot in enumerate(self._pool.slots):
            if not slot.in_use:
                if slot.last_read is not None:
                    self.stream.wait_event(slot.last_read)
                slot.generation += 1
                slot.in_use = True
                remote = {(expert, role): value[row] for row, expert in enumerate(experts)
                          for role, value in slot.matrices.items()}
                return index, slot.generation, remote
        raise RuntimeError("both replica slots are leased; skip prefetch until a lease is released")

    def _prepare_rounds(self, rounds: tuple[TransferRound, ...], active: tuple[int, ...], *,
                        role: str | None = None) -> tuple[_LocalRound, ...]:
        prepared, coverage, selected = [], {}, set()
        for item in rounds:
            if item.ranks != self._ranks:
                raise ValueError("transfer round does not match the admitted communication group")
            local = _LocalRound(*item.local(self.rank))
            if max(sum(local.send_sizes), sum(local.receive_sizes)) > self.quota_elements:
                raise ValueError("replica transfer exceeded the fixed communication quota")
            self._check_peer_order(local)
            for fragment in (*local.sent, *local.received):
                self._check_fragment(fragment, role)
                key = (fragment.source, fragment.target, fragment.expert, fragment.role)
                if fragment.offset != coverage.get(key, 0):
                    raise ValueError("replica fragments must cover each matrix once in logical order")
                coverage[key] = fragment.offset + fragment.elements
            selected.update((fragment.expert, fragment.role) for fragment in
                            (local.received if role is None else local.sent))
            prepared.append(local)
        roles = ("gate", "down") if role is None else (role,)
        expected = {(expert, name) for expert in active
                    if expert // self._pool.home_count != self.rank for name in roles}
        if selected != expected or any(end != self._role_elements(key[3]) for key, end in coverage.items()):
            raise ValueError("replica rounds do not completely cover the local remote experts")
        return tuple(prepared)

    @staticmethod
    def _check_peer_order(local: _LocalRound) -> None:
        sent = tuple(fragment.target for fragment in local.sent)
        received = tuple(fragment.source for fragment in local.received)
        if sent != tuple(sorted(sent)) or received != tuple(sorted(received)):
            raise ValueError("replica fragments must follow the all-to-all peer order")

    def _role_elements(self, role: str) -> int:
        if role not in ("gate", "down"):
            raise ValueError("replica transfer role must be gate or down")
        return prod(self._pool.shapes[0 if role == "gate" else 1])

    def _check_fragment(self, fragment: Fragment, role: str | None) -> None:
        indices = (fragment.source, fragment.target, fragment.expert, fragment.offset, fragment.elements)
        if any(isinstance(value, bool) or not isinstance(value, int) for value in indices):
            raise ValueError("replica fragment indices and counts must be integers")
        if (not 0 <= fragment.source < self._ranks or not 0 <= fragment.target < self._ranks
                or not 0 <= fragment.expert < self._pool.home_count * self._ranks):
            raise ValueError("replica fragment exceeds the admitted peer/expert range")
        owner = fragment.expert // self._pool.home_count
        direction_owner = fragment.source if role is None else fragment.target
        if direction_owner != owner or fragment.source == fragment.target or role is not None and fragment.role != role:
            raise ValueError("replica fragment does not match ownership or gradient role")
        if (fragment.offset < 0 or fragment.elements < 1
                or fragment.offset + fragment.elements > self._role_elements(fragment.role)):
            raise ValueError("replica fragment exceeds its admitted matrix bounds")

    def drain(self) -> None:
        """Wait for all local readers before a normal shutdown or owner reshard."""
        torch.npu.current_stream(self.device).wait_stream(self.stream)
