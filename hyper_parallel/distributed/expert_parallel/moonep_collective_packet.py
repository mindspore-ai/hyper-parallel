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

"""Bit-preserving packed dispatch for deduplicated token/rank payloads.

Hidden rows, per-assignment route weights and expansion indices share one
byte packet per peer. Packing is a collective transport layout, not a fused
NPU communication kernel. No numerical conversion is applied to packet fields.
"""

from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class PeerPacket:
    """Aligned byte offsets of one peer's hidden, weight and index fields."""

    hidden_bytes: int
    weight_offset: int
    weight_bytes: int
    index_offset: int
    index_bytes: int
    total_bytes: int


def packet_layout(
    unique_rows: int,
    assignments: int,
    hidden: int,
    hidden_dtype: torch.dtype,
    weight_dtype: torch.dtype,
    *,
    include_mapping: bool = True,
) -> PeerPacket:
    """Describe an aligned packet without converting hidden or weight dtypes.

    Args:
        unique_rows: Unique token payload rows sent to the peer.
        assignments: Number of routed expert choices on the peer.
        hidden: Hidden dimension.
        hidden_dtype: Hidden row dtype.
        weight_dtype: Route-weight dtype, which can differ from hidden dtype.
        include_mapping: Forward packets carry expansion IDs; backward omits them.

    Returns:
        Byte offsets aligned for dtype views, including int64 expansion IDs.
    """
    if any(type(value) is not int or value < 0 for value in (unique_rows, assignments, hidden)):
        raise ValueError("packet dimensions must be nonnegative")
    supported = (torch.float16, torch.bfloat16, torch.float32, torch.float64)
    if hidden_dtype not in supported or weight_dtype not in supported:
        raise ValueError("packet floating fields require float16, bfloat16, float32 or float64")
    hidden_bytes = unique_rows * hidden * hidden_dtype.itemsize
    weight_offset = (hidden_bytes + weight_dtype.itemsize - 1) // weight_dtype.itemsize * weight_dtype.itemsize
    weight_bytes = assignments * weight_dtype.itemsize
    index_offset = (weight_offset + weight_bytes + 7) // 8 * 8
    index_bytes = assignments * 8 if include_mapping else 0
    total_bytes = (index_offset + index_bytes + 7) // 8 * 8
    return PeerPacket(
        hidden_bytes,
        weight_offset,
        weight_bytes,
        index_offset,
        index_bytes,
        total_bytes,
    )


def _layouts(
    unique: list[int],
    assignments: list[int],
    hidden: int,
    hidden_dtype: torch.dtype,
    weight_dtype: torch.dtype,
    *,
    include_mapping: bool = True,
) -> list[PeerPacket]:
    if len(unique) != len(assignments):
        raise ValueError("unique and assignment counts must have matching peer counts")
    return [
        packet_layout(u, a, hidden, hidden_dtype, weight_dtype, include_mapping=include_mapping)
        for u, a in zip(unique, assignments)
    ]


def _pack(
    hidden: torch.Tensor,
    weights: torch.Tensor,
    mapping: torch.Tensor | None,
    unique_counts: list[int],
    assignment_counts: list[int],
    layouts: list[PeerPacket],
) -> torch.Tensor:
    """Copy floating fields bit-for-bit and initialize alignment padding."""
    buffer = torch.zeros(
        sum(layout.total_bytes for layout in layouts),
        dtype=torch.uint8,
        device=hidden.device,
    )
    offset = unique_start = assignment_start = 0
    for unique, assignments, layout in zip(unique_counts, assignment_counts, layouts):
        if unique == 0 and assignments == 0:
            continue
        packet = buffer[offset : offset + layout.total_bytes]
        hidden_bytes = hidden[unique_start : unique_start + unique].contiguous().view(torch.uint8).flatten()
        packet[: layout.hidden_bytes].copy_(hidden_bytes)
        packet[layout.weight_offset : layout.weight_offset + layout.weight_bytes].copy_(
            weights[assignment_start : assignment_start + assignments].contiguous().view(torch.uint8)
        )
        if layout.index_bytes:
            packet[layout.index_offset : layout.index_offset + layout.index_bytes].copy_(
                mapping[assignment_start : assignment_start + assignments].contiguous().view(torch.uint8)
            )
        offset += layout.total_bytes
        unique_start += unique
        assignment_start += assignments
    return buffer


def _unpack(
    buffer: torch.Tensor,
    unique_counts: list[int],
    assignment_counts: list[int],
    layouts: list[PeerPacket],
    hidden_size: int,
    hidden_dtype: torch.dtype,
    weight_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Restore peer fields and translate expansion IDs to receive-row offsets."""
    hidden = torch.empty((sum(unique_counts), hidden_size), dtype=hidden_dtype, device=buffer.device)
    weights = torch.empty(sum(assignment_counts), dtype=weight_dtype, device=buffer.device)
    mapping = torch.empty(
        sum(assignment_counts) if any(p.index_bytes for p in layouts) else 0,
        dtype=torch.int64,
        device=buffer.device,
    )
    offset = unique_start = assignment_start = 0
    for unique, assignments, layout in zip(unique_counts, assignment_counts, layouts):
        if unique == 0 and assignments == 0:
            continue
        packet = buffer[offset : offset + layout.total_bytes]
        hidden_view = packet[: layout.hidden_bytes].view(hidden_dtype).reshape(unique, hidden_size)
        hidden[unique_start : unique_start + unique].copy_(hidden_view)
        weights[assignment_start : assignment_start + assignments].copy_(
            packet[layout.weight_offset : layout.weight_offset + layout.weight_bytes].view(weight_dtype)
        )
        if layout.index_bytes:
            mapping[assignment_start : assignment_start + assignments].copy_(
                packet[layout.index_offset : layout.index_offset + layout.index_bytes].view(torch.int64) + unique_start
            )
        offset += layout.total_bytes
        unique_start += unique
        assignment_start += assignments
    return hidden, weights, mapping


class PackedDispatch(torch.autograd.Function):
    """Exchange mixed-dtype payloads once and differentiate both floating fields."""

    @staticmethod
    def forward(
        ctx: Any,
        hidden: torch.Tensor,
        weights: torch.Tensor,
        mapping: torch.Tensor,
        send_unique: list[int],
        receive_unique: list[int],
        send: list[int],
        receive: list[int],
        group: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Dispatch unique token rows and every expert choice's route weight."""
        ctx.group = group
        ctx.send_unique, ctx.receive_unique = send_unique, receive_unique
        ctx.send, ctx.receive = send, receive
        ctx.hidden_size, ctx.hidden_dtype, ctx.weight_dtype = (
            hidden.shape[1],
            hidden.dtype,
            weights.dtype,
        )
        ctx.send_layout = _layouts(send_unique, send, hidden.shape[1], hidden.dtype, weights.dtype)
        ctx.receive_layout = _layouts(receive_unique, receive, hidden.shape[1], hidden.dtype, weights.dtype)
        payload = _pack(hidden, weights, mapping, send_unique, send, ctx.send_layout)
        result = payload.new_empty(sum(layout.total_bytes for layout in ctx.receive_layout))
        dist.all_to_all_single(
            result,
            payload,
            [layout.total_bytes for layout in ctx.receive_layout],
            [layout.total_bytes for layout in ctx.send_layout],
            group=group,
        )
        received_hidden, received_weights, received_mapping = _unpack(
            result,
            receive_unique,
            receive,
            ctx.receive_layout,
            hidden.shape[1],
            hidden.dtype,
            weights.dtype,
        )
        ctx.mark_non_differentiable(received_mapping)
        ctx.save_for_backward(hidden.new_empty(0))
        return received_hidden, received_weights, received_mapping

    @staticmethod
    def backward(
        ctx: Any,
        grad_hidden: torch.Tensor | None,
        grad_weights: torch.Tensor | None,
        _grad_mapping: torch.Tensor | None,
    ) -> tuple:
        """Return unique hidden gradients and distinct per-choice weight gradients."""
        (sentinel,) = ctx.saved_tensors
        if grad_hidden is None:
            grad_hidden = sentinel.new_zeros((sum(ctx.receive_unique), ctx.hidden_size), dtype=ctx.hidden_dtype)
        if grad_weights is None:
            grad_weights = sentinel.new_zeros(sum(ctx.receive), dtype=ctx.weight_dtype)
        reverse_send = _layouts(
            ctx.receive_unique,
            ctx.receive,
            ctx.hidden_size,
            ctx.hidden_dtype,
            ctx.weight_dtype,
            include_mapping=False,
        )
        reverse_receive = _layouts(
            ctx.send_unique,
            ctx.send,
            ctx.hidden_size,
            ctx.hidden_dtype,
            ctx.weight_dtype,
            include_mapping=False,
        )
        payload = _pack(
            grad_hidden,
            grad_weights,
            None,
            ctx.receive_unique,
            ctx.receive,
            reverse_send,
        )
        result = payload.new_empty(sum(layout.total_bytes for layout in reverse_receive))
        dist.all_to_all_single(
            result,
            payload,
            [layout.total_bytes for layout in reverse_receive],
            [layout.total_bytes for layout in reverse_send],
            group=ctx.group,
        )
        hidden, weights, _ = _unpack(
            result,
            ctx.send_unique,
            ctx.send,
            reverse_receive,
            ctx.hidden_size,
            ctx.hidden_dtype,
            ctx.weight_dtype,
        )
        return hidden, weights, None, None, None, None, None, None
