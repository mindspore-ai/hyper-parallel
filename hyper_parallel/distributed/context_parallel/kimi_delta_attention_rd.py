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
"""Cached eager recursive doubling using public batched point-to-point transport."""
from __future__ import annotations

# pylint: disable=forbidden-backend-import

from dataclasses import dataclass
from functools import lru_cache
import importlib
from typing import Any

import torch
import torch.distributed as dist

from hyper_parallel.distributed.context_parallel.kimi_delta_attention_rd_plan import (
    RDResetPlan, build_reset_plan, build_scan_rounds,
)


_RD_EPILOGUE_BLOCK_ROWS = 64


@lru_cache(maxsize=1)
def _rd_kernels() -> Any:
    """Load optional Ascend kernels only when an NPU epilogue is executed."""
    return importlib.import_module(
        "hyper_parallel.components.functional._triton.kimi_delta_attention.rd_pointwise"
    )


def _finish_round(product: torch.Tensor, previous: torch.Tensor, save_m: bool = True) -> torch.Tensor | None:
    """Add the previous S and copy the next M in one launch, without fusing GEMM.

    Inputs must be distinct contiguous FP32 [B,H,128,256] buffers. Only the
    newly computed product's S half is modified; input summaries are immutable.
    The validated scan inputs and fresh matrix product establish this layout.

    Args:
        product: Fresh contiguous FP32 packed matrix product [B, H, 128, 256].
        previous: Previous immutable packed S/M tensor with the same layout.
        save_m: Whether a future round consumes the new contiguous M.
    """
    state_width, packed_width = product.shape[-2:]
    matrix = (torch.empty((*product.shape[:-1], state_width), dtype=product.dtype, device=product.device)
              if save_m else None)
    kernels = _rd_kernels()

    rows = product.numel() // packed_width
    block_rows = _RD_EPILOGUE_BLOCK_ROWS
    kernels.kda_rd_finish_round_kernel[((rows + block_rows - 1) // block_rows,)](
        product, previous, matrix if save_m else product,
        row_count=rows, state_width=state_width, packed_width=packed_width,
        block_rows=block_rows, save_matrix=save_m, num_warps=4)
    return matrix


def _finish_to_state(
    product: torch.Tensor, previous: torch.Tensor, save_m: bool,
) -> tuple[torch.Tensor | None, torch.Tensor]:
    """Emit contiguous final-stage S alongside the unchanged packed epilogue.

    Args:
        product: Fresh contiguous FP32 packed matrix product [B, H, 128, 256].
        previous: Previous immutable packed S/M tensor with the same layout.
        save_m: Whether a future round consumes the new contiguous M.
    """
    state_width, packed_width = product.shape[-2:]
    state = torch.empty((*product.shape[:-1], state_width), dtype=product.dtype, device=product.device)
    matrix = torch.empty_like(state) if save_m else None
    kernels = _rd_kernels()

    rows = product.numel() // packed_width
    block_rows = _RD_EPILOGUE_BLOCK_ROWS
    kernels.kda_rd_finish_to_state_kernel[((rows + block_rows - 1) // block_rows,)](
        product, previous, matrix if save_m else product, state,
        row_count=rows, state_width=state_width, packed_width=packed_width,
        block_rows=block_rows, save_matrix=save_m, num_warps=4)
    return matrix, state


class RoundExchange:
    """Keep immutable payload alive until send completion; outputs belong to the call."""

    def __init__(self, group: Any, rank: int, ranks: tuple[int, ...]) -> None:
        """Resolve ordered peers and the process group's protocol device."""
        if not ranks or len(set(ranks)) != len(ranks) or not 0 <= rank < len(ranks):
            raise ValueError("RD requires unique global peers and a valid local rank")
        if tuple(dist.get_process_group_ranks(group)) != ranks:
            raise ValueError("RD mesh sequence order must match its communication group order")
        self.group, self.rank, self.size, self.peers = group, rank, len(ranks), ranks
        if dist.get_backend(group) != "hccl":
            raise ValueError("KDA recursive doubling requires an HCCL process group")
        # Keep the optional NPU extension out of unrelated package imports.
        torch_npu = importlib.import_module("torch_npu")
        self.device = torch.device("npu", torch_npu.npu.current_device())

    def __call__(
        self, payload: torch.Tensor | None, destination: int | None, source: int | None,
        receive_shape: tuple[int, ...] | None = None,
    ) -> torch.Tensor | None:
        """Exchange peers, allocating only a live receive independently of the send.

        Args:
            payload: Immutable send tensor; may be absent when there is no destination.
            destination: Group-local send peer, or None.
            source: Group-local receive peer, or None.
            receive_shape: Explicit receive shape, required for a receive without a payload.
        """
        output = self._prepare_receive(payload, destination, source, receive_shape)
        if source is None and destination is None:
            return output
        operations = []
        if source is not None:
            operations.append(dist.P2POp(dist.irecv, output, self.peers[source], self.group))
        if destination is not None:
            operations.append(dist.P2POp(dist.isend, payload, self.peers[destination], self.group))
        for handle in dist.batch_isend_irecv(operations):
            handle.wait()
        return output

    def _prepare_receive(
        self, payload: torch.Tensor | None, destination: int | None, source: int | None,
        receive_shape: tuple[int, ...] | None,
    ) -> torch.Tensor | None:
        """Validate one exchange and allocate only its live receive buffer."""
        if payload is None:
            if destination is not None:
                raise ValueError('RD send requires a payload')
            if source is not None and receive_shape is None:
                raise ValueError('RD receive without a payload requires a shape')
        elif payload.device != self.device or not payload.is_contiguous():
            raise ValueError('RD payload must be contiguous on the protocol device')
        if any(peer is not None and not 0 <= peer < self.size for peer in (source, destination)):
            raise ValueError('RD peer is outside the process group')
        output = None
        if source is not None:
            output = (torch.empty_like(payload) if receive_shape is None else torch.empty(
                receive_shape, dtype=torch.float32 if payload is None else payload.dtype, device=self.device))
        return output


def _validate_summary(
    tensor: torch.Tensor, device: torch.device, state_shape: tuple[int, ...] | None = None,
) -> None:
    """Validate caller-owned S/M/G and cached transfers before any peer exchange."""
    if (not isinstance(tensor, torch.Tensor) or tensor.ndim != 4
            or tensor.shape[-2:] != (128, 128) or tensor.dtype != torch.float32
            or tensor.device != device or not tensor.is_contiguous()):
        raise ValueError("RD requires contiguous FP32 B/H/K128/V128 summaries on the protocol device")
    if state_shape is not None and tensor.shape != state_shape:
        raise ValueError("RD summary shape differs from the invocation")


def _validate_backward_summary(
    gradient: torch.Tensor | None, cache: tuple, expected: int,
    device: torch.device, state_shape: tuple[int, ...] | None = None,
) -> None:
    """Reject incompatible invocation cache layouts before any peer exchange."""
    if gradient is None:
        if state_shape is None:
            raise ValueError("RD gradient may be absent only with invocation shape metadata")
    else:
        _validate_summary(gradient, device, state_shape)
        state_shape = gradient.shape
    if not isinstance(cache, tuple) or len(cache) != expected:
        raise ValueError("RD received an incompatible invocation cache")
    for matrix in cache:
        _validate_summary(matrix, device, state_shape)


def _compact_transfer(matrix: torch.Tensor) -> torch.Tensor:
    """Keep a contiguous M view from retaining a larger producer allocation."""
    if matrix.untyped_storage().nbytes() != matrix.numel() * matrix.element_size():
        return matrix.clone()
    return matrix


class KDAStateRecursiveDoubling:
    """Exclusive forward scan and its cached adjoint, preserving the established rounding.

    Caller creates/warm-ups the HCCL group collectively. Inputs are immutable,
    contiguous FP32 B/H/K128/V128 summaries. Keep the returned invocation cache
    with its own autograd context. No shared graph buffers or mutable layer state.
    """

    requires_contiguous_summaries = True

    def __init__(self, group: Any, rank: int, ranks: tuple[int, ...]) -> None:
        """Prepare communication metadata and logarithmic scan distances."""
        self.exchange = RoundExchange(group, rank, ranks)
        self.rank, self.size = self.exchange.rank, self.exchange.size
        self.device = self.exchange.device
        self.distances = tuple(1 << level for level in range((self.size - 1).bit_length()))
        self.rounds = build_scan_rounds(self.rank, self.size, self.distances)

    def bind_sequence(
        self, sequence_info: Any, state_shape: tuple[int, ...] | None = None,
    ) -> _ResetRDInvocation:
        """Bind immutable packed metadata, optionally allowing absent dead summaries.

        Args:
            sequence_info: This call's packed document offsets and local interval.
            state_shape: B/H/K/V dimensions, without retaining a tensor allocation.
        """
        if sequence_info.start != self.rank * sequence_info.length:
            raise ValueError("RD packed interval must follow the state process-group order")
        if state_shape is not None:
            state_shape = tuple(state_shape)
            if (len(state_shape) != 4 or state_shape[-2:] != (128, 128)
                    or any(not isinstance(dim, int) or dim < 1 for dim in state_shape)):
                raise ValueError("RD invocation shape requires positive B/H and K=V128")
        plan = build_reset_plan(sequence_info.documents, sequence_info.length, self.size, self.rank)
        return _ResetRDInvocation(self, plan, state_shape)

    def _shift_peers(self, plan: RDResetPlan | None) -> tuple[int | None, int | None]:
        """Only continued documents consume the preceding rank's exit."""
        if plan is not None:
            return plan.shift_peers(self.rank)
        return (self.rank + 1 if self.rank + 1 < self.size else None,
                self.rank - 1 if self.rank else None)

    def _compose_packed_round(self, step, matrix, received, previous):
        """Keep packed arithmetic, materializing a transfer only for a later receive."""
        aggregate = matrix @ received
        if 2 * self.distances[step.stage] == self.distances[-1]:
            next_matrix, state = _finish_to_state(aggregate, previous, step.save_matrix)
        else:
            next_matrix = _finish_round(aggregate, previous, save_m=step.save_matrix)
            state = None
        return aggregate, next_matrix, state

    def forward(
        self, state: torch.Tensor | None, matrix: torch.Tensor | None,
        plan: RDResetPlan | None = None, state_shape: tuple[int, ...] | None = None,
    ) -> tuple[torch.Tensor | None, tuple[torch.Tensor, ...]]:
        """Return the exclusive incoming state and immutable pre-round transfer cache.

        Args:
            state: Local summary S tensor or kernel output.
            matrix: Contiguous FP32 local transfer M.
            plan: Internal immutable plan supplied by the packed invocation wrapper.
            state_shape: Optional per-call dimensions enabling absent dead summaries.
        """
        rounds = self.rounds if plan is None else plan.execution_rounds(self.rank)
        destination, source = self._shift_peers(plan)
        if state_shape is not None and plan is None:
            raise ValueError("RD optional summaries require a packed invocation plan")
        if state is None:
            if state_shape is None or matrix is not None or rounds or destination is not None:
                raise ValueError("RD state may be absent only when no exit-state consumer exists")
            return self.exchange(None, None, source, receive_shape=state_shape), ()
        _validate_summary(state, self.device, state_shape)
        _validate_summary(matrix, self.device, state.shape)
        narrow_state, cache = self._scan_forward(state, matrix, rounds)
        received = self.exchange(narrow_state, destination, source)
        if received is None and state_shape is None:
            received = torch.zeros_like(state)
        return received, cache

    def _scan_forward(
        self, state: torch.Tensor, matrix: torch.Tensor, rounds: tuple,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        """Compose live inclusive-scan rounds, preserving per-round adjoint inputs."""
        matrix = _compact_transfer(matrix) if rounds else matrix
        # Reset-only shifts never consume a packed S/M payload.
        packed = (torch.cat((state, matrix), dim=-1)
                  if any(step.stage < len(self.distances) - 1 for step in rounds) else None)
        before_m, narrow_state = matrix, state
        cache = []
        for step in rounds:
            final = step.stage == len(self.distances) - 1
            if final and narrow_state is None:
                narrow_state = packed[..., :128].contiguous()
            destination, source = step.destination, step.source
            received = self.exchange(narrow_state if final else packed, destination, source)
            if source is not None:
                cache.append(before_m)
                if final:
                    narrow_state = before_m @ received + narrow_state
                else:
                    packed, before_m, narrow_state = self._compose_packed_round(
                        step, before_m, received, packed)
        if narrow_state is None:
            narrow_state = packed[..., :128].contiguous()
        return narrow_state, tuple(cache)

    def backward(
        self, gradient: torch.Tensor | None, cache: tuple[torch.Tensor, ...],
        plan: RDResetPlan | None = None, state_shape: tuple[int, ...] | None = None,
    ) -> torch.Tensor | None:
        """Transpose the exact recorded forward composition order.

        Args:
            gradient: Contiguous FP32 local state-gradient summary G.
            cache: Invocation-owned transfers returned by this protocol forward.
            plan: The same internal plan used by this invocation's forward.
            state_shape: The same optional per-call shape used by forward.
        """
        expected = (sum(self.rank >= distance for distance in self.distances)
                    if plan is None else plan.cache_count(self.rank))
        source, destination = self._shift_peers(plan)
        if state_shape is not None and plan is None:
            raise ValueError("RD optional summaries require a packed invocation plan")
        if gradient is None and destination is not None:
            raise ValueError("RD gradient may be absent only when no head-gradient consumer exists")
        _validate_backward_summary(gradient, cache, expected, self.device, state_shape)
        received = (self.exchange(gradient, destination, source) if gradient is not None
                    else self.exchange(None, destination, source, receive_shape=state_shape))
        if received is None and state_shape is not None:
            # A tail without a consumer has no live scan rounds in a reset plan.
            return None
        adjoint = torch.zeros_like(gradient) if received is None else received
        index = len(cache)
        rounds = self.rounds if plan is None else plan.execution_rounds(self.rank)
        for step in reversed(rounds):
            source, destination = step.destination, step.source
            payload = adjoint
            if destination is not None:
                index -= 1
                payload = cache[index].transpose(-2, -1) @ adjoint
            received = self.exchange(payload, destination, source)
            if received is not None:
                # Adjoint is invocation-owned and every send has completed.
                adjoint.add_(received)
        return adjoint


@dataclass(frozen=True)
class _ResetRDInvocation:
    """Share static transport while keeping the exact plan with one autograd call."""

    requires_contiguous_summaries = True

    protocol: KDAStateRecursiveDoubling
    plan: RDResetPlan
    state_shape: tuple[int, ...] | None = None

    @property
    def allows_absent_summaries(self) -> bool:
        """Whether shape metadata replaces unused full-size placeholders."""
        return self.state_shape is not None

    @property
    def needs_state_summary(self) -> bool:
        """Only a continuing tail document has an exit-state consumer."""
        return self.plan.shift_peers(self.protocol.rank)[0] is not None

    @property
    def needs_transition_summary(self) -> bool:
        """Produce M only when a consumed exit depends on this rank's entry."""
        return self.needs_state_summary and self.plan.is_transfer_active(self.protocol.rank)

    def forward(
        self, state: torch.Tensor | None, matrix: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, tuple[torch.Tensor, ...]]:
        """Execute the pruned graph and return its tensor-only cache."""
        return self.protocol.forward(state, matrix, self.plan, self.state_shape)

    def backward(self, gradient: torch.Tensor | None, cache: tuple[torch.Tensor, ...]) -> torch.Tensor | None:
        """Transpose this invocation's graph, independent of later forwards."""
        return self.protocol.backward(gradient, cache, self.plan, self.state_shape)
