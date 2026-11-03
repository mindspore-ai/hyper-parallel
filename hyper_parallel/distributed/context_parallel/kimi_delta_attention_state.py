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
"""KDA state exchange and mesh construction for gathered or recursive scans."""
# pylint: disable=forbidden-backend-import
from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist

from hyper_parallel.core.dtensor.device_mesh import DeviceMesh
from hyper_parallel.distributed.context_parallel.kimi_delta_attention_rd import KDAStateRecursiveDoubling


def _validate_mesh_order(mesh: DeviceMesh) -> None:
    """Prevent collective slots from being mistaken for chronological positions."""
    group_ranks = tuple(dist.get_process_group_ranks(mesh.get_group()))
    if tuple(mesh.rank_list) != group_ranks:
        raise ValueError("KDA mesh sequence order must match its communication group order.")


def split_kda_mesh(mesh: DeviceMesh, degree: int, names: tuple[str, str]) -> DeviceMesh:
    """Split a CP axis without losing the root rank map or sibling DP groups.

    Args:
        mesh: One-dimensional chronological CP mesh.
        degree: Size of the inner split axis.
        names: Names of the outer and inner axes.

    Returns:
        Two-dimensional submesh attached to the original root.
    """
    if mesh.ndim != 1 or not isinstance(degree, int) or isinstance(degree, bool):
        raise ValueError("KDA mesh splitting requires a 1-D mesh and integer degree.")
    if degree < 1 or mesh.size() % degree:
        raise ValueError("KDA split degree must be positive and divide the CP size.")
    _validate_mesh_order(mesh)
    # Group creation uses WORLD; every rank must derive the same sibling groups.
    root = mesh._get_root_mesh()  # pylint: disable=protected-access
    if root.size() != dist.get_world_size():
        raise ValueError("Hybrid KDA requires a CP submesh retaining its WORLD root mesh.")
    # Like the existing hybrid CP helper, preserve all sibling groups through the root layout.
    return mesh._unflatten(0, (mesh.size() // degree, degree), names)  # pylint: disable=protected-access


def build_kda_state_cp(mesh: DeviceMesh, protocol: str, allgather_group_size: int) -> object | None:
    """Construct a state-CP executor holding only communication metadata.

    Args:
        mesh: One-dimensional mesh ordered along the state sequence.
        protocol: p2p, allgather, grouped_allgather_p2p, or recursive_doubling.
        allgather_group_size: Consecutive ranks per local gather, only for the grouped protocol.

    Returns:
        State executor with an invocation-owned cache, or None for the P2P/local path.
    """
    if protocol not in ("p2p", "allgather", "grouped_allgather_p2p", "recursive_doubling"):
        raise ValueError(f"Unknown KDA state_cp_method: {protocol!r}.")
    if not isinstance(allgather_group_size, int) or isinstance(allgather_group_size, bool) or allgather_group_size < 1:
        raise ValueError("KDA allgather_group_size must be a positive integer.")
    if protocol != "grouped_allgather_p2p" and allgather_group_size != 1:
        raise ValueError("KDA allgather_group_size is only valid for grouped_allgather_p2p.")
    if protocol == "p2p":
        return None
    ranks = tuple(mesh.rank_list)
    rank, size = mesh.get_local_rank(), mesh.size()
    if protocol == "recursive_doubling":
        return KDAStateRecursiveDoubling(mesh.get_group(), rank, ranks) if size > 1 else None
    if protocol == "allgather":
        _validate_mesh_order(mesh)
        return KDAStateAllGather(mesh.get_group(), rank, size)
    if size % allgather_group_size:
        raise ValueError("KDA allgather_group_size must divide the state CP size.")
    if allgather_group_size == 1:
        return None
    if allgather_group_size == size:
        _validate_mesh_order(mesh)
        return KDAStateAllGather(mesh.get_group(), rank, size)
    split = split_kda_mesh(mesh, allgather_group_size, ("kda_outer", "kda_group"))
    return KDAStateAllGatherP2P(
        mesh.get_group(), split["kda_group"].get_group(), rank, ranks, allgather_group_size)


def _validate_summary(state: torch.Tensor, matrix: torch.Tensor) -> None:
    """Validate the dense affine contract before entering communication."""
    if (state.ndim != 4 or state.shape[-2:] != (128, 128) or matrix.shape != state.shape
            or state.shape[0] == 0 or state.shape[1] == 0):
        raise ValueError("KDA state CP requires nonempty matching [B,H,128,128] summaries.")
    if state.dtype != torch.float32 or matrix.dtype != torch.float32 or state.device != matrix.device:
        raise ValueError("KDA state CP requires FP32 summaries on the same device.")
    if state.device.type not in ("cpu", "npu"):
        raise ValueError("KDA state CP supports CPU protocol checks and Ascend NPU execution only.")


def merge_affine_summaries(
    states: torch.Tensor, matrices: torch.Tensor, rank: int, forward: bool,
) -> torch.Tensor:
    """Merge the exclusive prefix or transposed suffix in chronological order.

    Args:
        states: Affine offsets shaped [P,B,H,128,128].
        matrices: Matching transfer matrices, possibly views of packed storage.
        rank: Chronological rank whose input state or final-state gradient is requested.
        forward: Select forward prefix instead of backward suffix.

    Returns:
        Contiguous FP32 recurrent state or state gradient shaped [B,H,128,128].
    """
    if states.ndim != 5 or states.shape != matrices.shape or states.shape[0] == 0:
        raise ValueError("KDA merge requires matching rank-stacked summaries.")
    size, batch, heads = states.shape[:3]
    _validate_summary(states[0], matrices[0])
    if not 0 <= rank < size:
        raise ValueError("KDA merge rank is outside the summary group.")
    if any(t.stride(1) != heads * t.stride(2) or t.stride(-1) != 1 for t in (states, matrices)):
        raise ValueError("KDA merge requires consecutive batch/head slabs and columns.")
    if states.device.type != "npu":
        raise ValueError("KDA fused state merge requires Ascend NPU tensors.")
    count = rank if forward else size - rank - 1
    output = states.new_empty((batch, heads, 128, 128))
    if count == 0:
        return output.zero_()
    # Triton is optional for imports; production merging requires Ascend.
    # pylint: disable=import-outside-toplevel
    from hyper_parallel.components.functional._triton.kimi_delta_attention.cp_merge import (
        kda_cached_affine_merge_kernel,
    )
    # pylint: enable=import-outside-toplevel

    kda_cached_affine_merge_kernel[(2, batch * heads)](
        output, states, matrices, count, rank,
        state_rank_stride=states.stride(0), state_head_stride=states.stride(2), state_row_stride=states.stride(3),
        matrix_rank_stride=matrices.stride(0), matrix_head_stride=matrices.stride(2),
        matrix_row_stride=matrices.stride(3),
        forward=forward, block_v=64, num_warps=2, num_stages=2)
    return output


def _gather_maps(state: torch.Tensor, matrix: torch.Tensor, group: Any, size: int) -> torch.Tensor:
    """Pack validated local summaries for one invocation; the caller owns the gathered storage."""
    packed = torch.cat((state, matrix), dim=-1)
    output = packed.new_empty((size * packed.shape[0], *packed.shape[1:]))
    dist.all_gather_into_tensor(output, packed, group=group)
    return output.view(size, *packed.shape)


class KDAStateAllGather:
    """Gather S/M forward and G/M backward; only local M survives forward."""

    def __init__(self, group: Any, rank: int, size: int) -> None:
        """Bind precreated communication metadata, without tensor workspaces."""
        if not isinstance(size, int) or isinstance(size, bool) or size < 1 or not 0 <= rank < size:
            raise ValueError("AllGather requires a positive size and valid local rank.")
        self.group, self.rank, self.size = group, rank, size

    def forward(self, state: torch.Tensor, matrix: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        """Return the incoming state and this invocation's original local M.

        Args:
            state: Local FP32 offset S [B,H,128,128].
            matrix: Local forward transfer M of the same shape.

        Returns:
            This rank's incoming recurrent state and a one-tensor local M cache.
        """
        _validate_summary(state, matrix)
        maps = _gather_maps(state, matrix, self.group, self.size)
        return merge_affine_summaries(maps[..., :128], maps[..., 128:], self.rank, True), (matrix,)

    def backward(self, gradient: torch.Tensor, cache: tuple[torch.Tensor, ...]) -> torch.Tensor:
        """Collect new G with the saved local M and recover the outgoing gradient.

        Args:
            gradient: Local FP32 offset G [B,H,128,128].
            cache: One-tensor tuple containing this invocation's local forward M.

        Returns:
            This rank's outgoing recurrent-state gradient.
        """
        matrix, = cache
        _validate_summary(gradient, matrix)
        maps = _gather_maps(gradient, matrix, self.group, self.size)
        return merge_affine_summaries(maps[..., :128], maps[..., 128:], self.rank, False)


def _validate_scan_range(indices: range, size: int, forward: bool, prefix_count: int | None) -> None:
    """Check the requested chronological interval and optional saved prefix."""
    count = len(indices)
    if prefix_count is not None and not 0 <= prefix_count <= count:
        raise ValueError("The saved prefix must lie within the scan interval.")
    if count:
        endpoints_valid = all(0 <= index < size for index in (indices[0], indices[-1]))
        if indices.step != (1 if forward else -1) or not endpoints_valid:
            raise ValueError("KDA scans require an ordered interval inside the group.")


def _scan_maps(maps: torch.Tensor, incoming: torch.Tensor,
               indices: range, forward: bool, prefix_count: int | None = None,
               ) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Scan validated group maps once, optionally returning an intermediate prefix."""
    count = len(indices)
    _validate_scan_range(indices, maps.shape[0], forward, prefix_count)
    if not count:
        return incoming, incoming if prefix_count == 0 else None
    if incoming.device.type != "npu" or not incoming.is_contiguous():
        raise ValueError("KDA fused scans require a contiguous Ascend input state.")
    output = torch.empty_like(incoming)
    prefix = None
    if prefix_count is not None:
        prefix = incoming if prefix_count == 0 else output if prefix_count == count else torch.empty_like(incoming)
    save_prefix = prefix is not None and prefix_count not in (0, count)
    # Keep the optional Triton dependency out of CPU-only model imports.
    # pylint: disable=import-outside-toplevel
    from hyper_parallel.components.functional._triton.kimi_delta_attention.cp_merge import kda_affine_scan_kernel
    # pylint: enable=import-outside-toplevel

    batch, heads = incoming.shape[:2]
    kda_affine_scan_kernel[(2, batch * heads)](
        output, prefix, maps, incoming, count, indices.start, prefix_count or 0,
        rank_stride=maps.stride(0), head_stride=maps.stride(2), row_stride=maps.stride(3),
        forward=forward, save_prefix=save_prefix, block_v=64, num_warps=2, num_stages=2)
    return output, prefix


class KDAStateAllGatherP2P:
    """Use group-local gathers and an owner chain, retaining only each local M."""

    def __init__(self, group: Any, intra_group: Any, rank: int,
                 ranks: tuple[int, ...], width: int) -> None:
        """Bind ranks already validated against the process-group order by the factory."""
        size = len(ranks)
        if not isinstance(width, int) or isinstance(width, bool) or width < 2 or size % width:
            raise ValueError("Grouped AllGather width must be >= 2 and divide the CP size.")
        if not 0 <= rank < size or len(set(ranks)) != size:
            raise ValueError("Grouped AllGather requires unique global ranks and a valid local rank.")
        self.group, self.intra, self.rank, self.width = group, intra_group, rank, width
        self.ranks = tuple(ranks)
        self.block, self.local = divmod(rank, width)
        self.blocks = size // width
        self.owner = self.block * width + width - 1

    def _receive(self, state: torch.Tensor, forward: bool) -> tuple[torch.Tensor, Any]:
        """Post the owner receive before gathering local summaries."""
        incoming = torch.zeros_like(state)
        recv_handle = None
        peer = self.owner - self.width if forward else self.owner + self.width
        if self.rank == self.owner and 0 <= peer < len(self.ranks):
            recv_handle = dist.irecv(incoming, src=self.ranks[peer], group=self.group)
        return incoming, recv_handle

    def forward(self, state: torch.Tensor, matrix: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        """Relay ordered group scans and restore this rank's incoming state.

        Args:
            state: Local FP32 offset S [B,H,128,128].
            matrix: Local forward transfer M of the same shape.

        Returns:
            This rank's incoming recurrent state and a one-tensor local M cache.
        """
        _validate_summary(state, matrix)
        incoming, recv_handle = self._receive(state, True)
        maps = _gather_maps(state, matrix, self.intra, self.width)
        send_handle, send_state = None, None
        result = None
        if self.rank == self.owner:
            if recv_handle is not None:
                recv_handle.wait()
            if self.block + 1 < self.blocks:
                send_state, result = _scan_maps(maps, incoming, range(self.width), True,
                                                prefix_count=self.local)
                send_handle = dist.isend(send_state, dst=self.ranks[self.owner + self.width], group=self.group)
        if self.block > 0:
            dist.broadcast(incoming, src=self.ranks[self.owner], group=self.intra)
        if result is None:
            result, _ = _scan_maps(maps, incoming, range(self.local), True)
        if send_handle is not None:
            send_handle.wait()
        return result, (matrix,)

    def backward(self, gradient: torch.Tensor, cache: tuple[torch.Tensor, ...]) -> torch.Tensor:
        """Regather G/M and reverse the owner chain with ordered state feedback.

        Args:
            gradient: Local FP32 offset G [B,H,128,128].
            cache: One-tensor tuple containing this invocation's local forward M.

        Returns:
            This rank's outgoing recurrent-state gradient.
        """
        matrix, = cache
        _validate_summary(gradient, matrix)
        incoming, recv_handle = self._receive(gradient, False)
        maps = _gather_maps(gradient, matrix, self.intra, self.width)
        send_handle, send_state = None, None
        if self.rank == self.owner:
            if recv_handle is not None:
                recv_handle.wait()
            if self.block > 0:
                send_state, _ = _scan_maps(maps, incoming, range(self.width - 1, -1, -1), False)
                send_handle = dist.isend(send_state, dst=self.ranks[self.owner - self.width], group=self.group)
        if self.block + 1 < self.blocks:
            dist.broadcast(incoming, src=self.ranks[self.owner], group=self.intra)
        result, _ = _scan_maps(maps, incoming, range(self.width - 1, self.local, -1), False)
        if send_handle is not None:
            send_handle.wait()
        return result
