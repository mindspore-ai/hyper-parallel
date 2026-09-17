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
"""KDA summary collectives with invocation-local transfer storage."""
# This is the Torch-only functional KDA path, matching its staged execution module.
# pylint: disable=forbidden-backend-import
from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist


def _validate_summary(state: torch.Tensor, matrix: torch.Tensor) -> None:
    """Validate the dense affine contract before entering communication."""
    if state.ndim != 4 or state.shape[-2:] != (128, 128) or matrix.shape != state.shape:
        raise ValueError("KDA boundaries require matching [B,H,128,128] summaries.")
    if state.dtype != torch.float32 or matrix.dtype != torch.float32 or state.device != matrix.device:
        raise ValueError("KDA boundaries require FP32 summaries on the same device.")


def merge_affine_summaries(
    states: torch.Tensor, matrices: torch.Tensor, rank: int, forward: bool,
) -> torch.Tensor:
    """Merge the exclusive prefix or transposed suffix in chronological order.

    Args:
        states: Affine offsets shaped [P,B,H,128,128].
        matrices: Matching transfer matrices, possibly views of packed storage.
        rank: Chronological rank whose boundary is requested.
        forward: Select forward prefix instead of backward suffix.

    Returns:
        Contiguous FP32 boundary state shaped [B,H,128,128].
    """
    if states.ndim != 5 or states.shape != matrices.shape or states.shape[0] == 0:
        raise ValueError("KDA merge requires matching rank-stacked summaries.")
    size, batch, heads = states.shape[:3]
    _validate_summary(states[0], matrices[0])
    if not 0 <= rank < size:
        raise ValueError("KDA merge rank is outside the summary group.")
    if any(t.stride(1) != heads * t.stride(2) or t.stride(-1) != 1 for t in (states, matrices)):
        raise ValueError("KDA merge requires consecutive batch/head slabs and columns.")
    count = rank if forward else size - rank - 1
    output = states.new_empty((batch, heads, 128, 128))
    if count == 0:
        return output.zero_()
    if states.device.type == "cpu":
        output.zero_()
        indices = range(rank) if forward else range(size - 1, rank, -1)
        for index in indices:
            matrix = matrices[index] if forward else matrices[index].transpose(-1, -2)
            output = matrix @ output + states[index]
        return output
    # Triton is an optional dependency; CPU protocol checks do not need it.
    from ._kda_triton.cp_merge import kda_cached_affine_merge_kernel  # pylint: disable=C0415

    kda_cached_affine_merge_kernel[(2, batch * heads)](
        output, states, matrices, count, rank, S_RANK=states.stride(0), S_HEAD=states.stride(2),
        S_ROW=states.stride(3), M_RANK=matrices.stride(0), M_HEAD=matrices.stride(2),
        M_ROW=matrices.stride(3), FORWARD=forward, BV=64, num_warps=2, num_stages=2)
    return output


def _gather_maps(state: torch.Tensor, matrix: torch.Tensor, group: Any, size: int) -> torch.Tensor:
    """Collect offsets and transfers without retaining a gather in autograd."""
    _validate_summary(state, matrix)
    packed = torch.cat((state, matrix), dim=-1)
    output = packed.new_empty((size * packed.shape[0], *packed.shape[1:]))
    dist.all_gather_into_tensor(output, packed, group=group)
    return output.view(size, *packed.shape)


class AllGatherBoundary:
    """Gather S/M forward and G/M backward; only local M survives forward."""

    def __init__(self, group: Any, rank: int, size: int) -> None:
        """Bind precreated communication metadata, without tensor workspaces."""
        if not isinstance(size, int) or isinstance(size, bool) or size < 1 or not 0 <= rank < size:
            raise ValueError("AllGather requires a positive size and valid local rank.")
        self.group, self.rank, self.size = group, rank, size

    def forward(self, state: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
        """Return the incoming state; the caller saves its original local M.

        Args:
            state: Local FP32 offset S [B,H,128,128].
            matrix: Local forward transfer M of the same shape.

        Returns:
            This rank's incoming state or outgoing state gradient.
        """
        maps = _gather_maps(state, matrix, self.group, self.size)
        return merge_affine_summaries(maps[..., :128], maps[..., 128:], self.rank, True)

    def backward(self, gradient: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
        """Collect new G with the saved local M and recover the outgoing gradient.

        Args:
            gradient: Local FP32 offset G [B,H,128,128].
            matrix: This invocation's cached local forward M.

        Returns:
            This rank's incoming state or outgoing state gradient.
        """
        maps = _gather_maps(gradient, matrix, self.group, self.size)
        return merge_affine_summaries(maps[..., :128], maps[..., 128:], self.rank, False)


def _scan_maps(maps: torch.Tensor, incoming: torch.Tensor,
               indices: range, forward: bool) -> torch.Tensor:
    """Preserve P2P's state feedback instead of reassociating group transfers."""
    result = incoming
    for index in indices:
        matrix = maps[index, ..., 128:]
        if not forward:
            matrix = matrix.transpose(-1, -2)
        result = matrix @ result + maps[index, ..., :128]
    return result


class GroupedAllGatherBoundary:
    """Use group-local gathers and an owner chain, retaining only each local M."""

    def __init__(self, group: Any, intra_group: Any, rank: int,
                 ranks: tuple[int, ...], width: int) -> None:
        """Bind chronological CP ranks and their precreated consecutive subgroup."""
        size = len(ranks)
        if not isinstance(width, int) or isinstance(width, bool) or width < 2 or size % width:
            raise ValueError("Grouped AllGather width must be >= 2 and divide the CP size.")
        if not 0 <= rank < size or tuple(sorted(set(ranks))) != tuple(ranks):
            raise ValueError("Grouped AllGather requires ordered unique global ranks and a valid local rank.")
        self.group, self.intra, self.rank, self.width = group, intra_group, rank, width
        self.ranks = tuple(ranks)
        self.block, self.local = divmod(rank, width)
        self.blocks = size // width
        self.owner = self.block * width + width - 1

    def _receive(self, state: torch.Tensor, forward: bool) -> tuple[torch.Tensor, Any]:
        """Post the owner receive before gathering local summaries."""
        incoming = torch.zeros_like(state)
        work = None
        peer = self.owner - self.width if forward else self.owner + self.width
        if self.rank == self.owner and 0 <= peer < len(self.ranks):
            work = dist.irecv(incoming, src=self.ranks[peer], group=self.group)
        return incoming, work

    def forward(self, state: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
        """Relay ordered group scans and restore this rank's incoming state.

        Args:
            state: Local FP32 offset S [B,H,128,128].
            matrix: Local forward transfer M of the same shape.

        Returns:
            This rank's incoming state or outgoing state gradient.
        """
        _validate_summary(state, matrix)
        incoming, receive = self._receive(state, True)
        maps = _gather_maps(state, matrix, self.intra, self.width)
        send, packet = None, None
        if self.rank == self.owner:
            if receive is not None:
                receive.wait()
            if self.block + 1 < self.blocks:
                packet = _scan_maps(maps, incoming, range(self.width), True)
                send = dist.isend(packet, dst=self.ranks[self.owner + self.width], group=self.group)
        if self.block > 0:
            dist.broadcast(incoming, src=self.ranks[self.owner], group=self.intra)
        result = _scan_maps(maps, incoming, range(self.local), True)
        if send is not None:
            send.wait()
        return result

    def backward(self, gradient: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
        """Regather G/M and reverse the owner chain with ordered state feedback.

        Args:
            gradient: Local FP32 offset G [B,H,128,128].
            matrix: This invocation's cached local forward M.

        Returns:
            This rank's incoming state or outgoing state gradient.
        """
        _validate_summary(gradient, matrix)
        incoming, receive = self._receive(gradient, False)
        maps = _gather_maps(gradient, matrix, self.intra, self.width)
        send, packet = None, None
        if self.rank == self.owner:
            if receive is not None:
                receive.wait()
            if self.block > 0:
                packet = _scan_maps(maps, incoming, range(self.width - 1, -1, -1), False)
                send = dist.isend(packet, dst=self.ranks[self.owner - self.width], group=self.group)
        if self.block + 1 < self.blocks:
            dist.broadcast(incoming, src=self.ranks[self.owner], group=self.intra)
        result = _scan_maps(maps, incoming, range(self.width - 1, self.local, -1), False)
        if send is not None:
            send.wait()
        return result
