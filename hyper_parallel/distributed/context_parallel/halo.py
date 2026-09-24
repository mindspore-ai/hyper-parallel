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
"""Differentiable left-window halo exchange for contiguous Torch CP shards."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

# This transport serves the Torch local-tensor CP adapter.
import torch  # pylint: disable=forbidden-backend-import
import torch.distributed as dist  # pylint: disable=forbidden-backend-import


@dataclass(frozen=True)
class SequenceHaloRoute:
    """Immutable token-row routing; no tensors or process groups are cached."""

    global_start: int
    send_offsets: tuple[int, ...]
    input_splits: tuple[int, ...]
    output_splits: tuple[int, ...]


@lru_cache(maxsize=128)
def build_sequence_halo_route(
        local_length: int, window_size: int, world_size: int, rank: int,
) -> SequenceHaloRoute:
    """Route intersections of owner shards with each consumer's left halo.

    The causal window includes the current token, so at most window_size-1
    remote rows are needed. Equal, contiguous CP shards are required.

    Args:
        local_length: Number of tokens owned by each contiguous CP shard.
        window_size: Causal window width, including the current token.
        world_size: Number of ranks in the CP group.
        rank: Local rank within the CP group.
    """
    if local_length <= 0 or window_size <= 0 or world_size <= 0 or not 0 <= rank < world_size:
        raise ValueError("halo requires positive lengths/world size and a valid CP rank")
    local_start = rank * local_length
    halo_start = max(0, local_start - window_size + 1)
    offsets, sends, receives = [], [], []
    for peer in range(world_size):
        peer_start = peer * local_length
        peer_halo_start = max(0, peer_start - window_size + 1)
        send_start = max(local_start, peer_halo_start)
        send_count = max(0, min(local_start + local_length, peer_start) - send_start)
        recv_count = max(0, min(peer_start + local_length, local_start) - max(peer_start, halo_start))
        offsets.append(send_start - local_start if send_count else 0)
        sends.append(send_count)
        receives.append(recv_count)
    return SequenceHaloRoute(halo_start, tuple(offsets), tuple(sends), tuple(receives))


class _SequenceHaloWait(torch.autograd.Function):
    """Return boundary gradients directly, without full-shard slice backwards."""

    @staticmethod
    def forward(
            ctx: Any, local_front: torch.Tensor, received: torch.Tensor,
            work: Any, route: SequenceHaloRoute, group: Any,
    ) -> torch.Tensor:
        """Finish the async exchange and attach a single local autograd edge.

        Args:
            ctx: Autograd context for the inverse route.
            local_front: Local sequence shard with sequence moved to axis zero.
            received: Pending left-boundary receive buffer.
            work: Forward exchange handle, waited before reading the buffer.
            route: Per-peer split sizes and local source offsets.
            group: Context-parallel process group.
        """
        ctx.route = route
        ctx.group = group
        work.wait()
        return torch.cat((received, local_front), dim=0)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None, None, None, None]:
        """Reverse A2AV and sum each consumer's boundary into its owner rows.

        Args:
            ctx: Context containing the forward route and process group.
            grad_output: Gradient for the concatenated halo and local shard.
        """
        route = ctx.route
        halo_length = sum(route.output_splits)
        send = grad_output[:halo_length].contiguous()
        received = grad_output.new_empty((sum(route.input_splits), *grad_output.shape[1:]))
        work = dist.all_to_all_single(
            received, send, output_split_sizes=list(route.input_splits),
            input_split_sizes=list(route.output_splits), group=ctx.group, async_op=True,
        )
        # Retain the local consumer's contribution while remote gradients travel.
        grad_local = grad_output[halo_length:].contiguous().clone()
        work.wait()
        cursor = 0
        for offset, count in zip(route.send_offsets, route.input_splits):
            if count:
                grad_local.narrow(0, offset, count).add_(received.narrow(0, cursor, count))
                cursor += count
        return grad_local, None, None, None, None


@dataclass
class AsyncSequenceHalo:
    """Keep send storage alive and materialize [left halo, local] at the consumer."""

    local_front: torch.Tensor
    send_buffer: torch.Tensor | None
    received: torch.Tensor | None
    sequence_dim: int
    global_start: int
    work: Any = None
    route: SequenceHaloRoute | None = None
    group: Any = None
    waited: bool = False

    def wait(self) -> torch.Tensor:
        """Wait once and attach the inverse boundary exchange to autograd."""
        if self.waited:
            raise RuntimeError("an async halo handle cannot be waited twice")
        self.waited = True
        if self.received is None:
            result = self.local_front
        else:
            # Keep this autograd edge even for rank zero's empty receive. That
            # rank must participate in the reverse exchange to receive gradients.
            result = _SequenceHaloWait.apply(
                self.local_front, self.received, self.work, self.route, self.group,
            )
        return result.movedim(0, self.sequence_dim).contiguous()


def async_cp_halo_launch(
        tensor: torch.Tensor, sequence_dim: int, window_size: int, cp_mesh: Any,
) -> AsyncSequenceHalo:
    """Launch differentiable all-to-all-v using token-row splits on the CP group.

    Args:
        tensor: Local raw KV, with an equal contiguous shard of the sequence.
        sequence_dim: Sequence axis, moved to dimension zero for communication.
        window_size: Causal window width, including the current token.
        cp_mesh: Framework-owned CP mesh; no process groups are created here.

    Returns:
        A single-use handle with the received buffer's global starting position.
    """
    if not -tensor.ndim <= sequence_dim < tensor.ndim:
        raise ValueError("halo sequence_dim is outside the input tensor dimensions")
    sequence_dim %= tensor.ndim
    size, rank = cp_mesh.size(), cp_mesh.get_local_rank()
    route = build_sequence_halo_route(tensor.shape[sequence_dim], window_size, size, rank)
    local_front = tensor.movedim(sequence_dim, 0).contiguous()
    if size == 1 or window_size == 1:
        return AsyncSequenceHalo(local_front, None, None, sequence_dim, route.global_start)
    send_parts = [
        local_front.detach().narrow(0, offset, count)
        for offset, count in zip(route.send_offsets, route.input_splits)
        if count
    ]
    # Autograd is attached at wait, including on ranks with no outgoing rows.
    if len(send_parts) == 1:
        send = send_parts[0]
    else:
        send = torch.cat(send_parts, dim=0) if send_parts else local_front.detach()[:0]
    received = local_front.new_empty((sum(route.output_splits), *local_front.shape[1:]))
    group = cp_mesh.get_group()
    work = dist.all_to_all_single(
        received, send, output_split_sizes=list(route.output_splits),
        input_split_sizes=list(route.input_splits), group=group, async_op=True,
    )
    return AsyncSequenceHalo(
        local_front, send, received, sequence_dim, route.global_start, work, route, group,
    )
