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
"""Deterministic, bounded replica transfers, independent of expert capacity."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Fragment:
    """A contiguous logical matrix range in one replica transfer round."""

    source: int
    target: int
    expert: int
    role: str
    offset: int
    elements: int


@dataclass(frozen=True)
class TransferRound:
    """All ranks share this ordered round, including ranks with zero splits."""

    fragments: tuple[Fragment, ...]
    ranks: int

    def local(self, rank: int) -> tuple[tuple[Fragment, ...], tuple[Fragment, ...], tuple[int, ...], tuple[int, ...]]:
        """Return source-major peer splits and matching logical fragments."""
        sent = tuple(fragment for fragment in self.fragments if fragment.source == rank)
        received = tuple(fragment for fragment in self.fragments if fragment.target == rank)
        send_sizes = tuple(sum(item.elements for item in sent if item.target == peer) for peer in range(self.ranks))
        receive_sizes = tuple(sum(item.elements for item in received if item.source == peer)
                              for peer in range(self.ranks))
        return sent, received, send_sizes, receive_sizes


def _check_transfer_allocation(allocation: torch.Tensor, quota_elements: int) -> tuple[int, int]:
    """Check the allocation table and quota before inspecting transfer roles."""
    if (allocation.device.type != "cpu" or allocation.dtype != torch.int64 or allocation.ndim != 2
            or min(allocation.shape) < 1 or bool((allocation < 0).any())):
        raise ValueError("allocation must be a nonnegative CPU int64 [expert, rank] table")
    experts, ranks = allocation.shape
    if experts % ranks or isinstance(quota_elements, bool) or not isinstance(quota_elements, int) or quota_elements < 1:
        raise ValueError("equal home ownership and a positive integral transfer quota are required")
    return experts, ranks


def _check_transfer_roles(role_elements: tuple[tuple[str, int], ...]) -> None:
    """Validate role identity and element counts without reordering evaluation."""
    if (not role_elements or len({name for name, _ in role_elements}) != len(role_elements)
            or any(name not in ("gate", "down") or isinstance(size, bool) or not isinstance(size, int) or size < 1
                   for name, size in role_elements)):
        raise ValueError("each transfer role needs one positive integral matrix size")


def compile_transfer_rounds(allocation: torch.Tensor, role_elements: tuple[tuple[str, int], ...], *,
                            reverse: bool = False, quota_elements: int = 32 * 1024 * 1024
                            ) -> tuple[TransferRound, ...]:
    """Cover actual replica ranges once with bounded per-rank send/receive sizes.

    The default quota is 64 MiB of BF16 per direction. Gradient roles are compiled
    separately so down gradients can enter their communicator before gate dW.
    """
    experts, ranks = _check_transfer_allocation(allocation, quota_elements)
    _check_transfer_roles(role_elements)
    transfers = []
    home_count = experts // ranks
    for expert, target in allocation.nonzero().tolist():
        owner = expert // home_count
        if target != owner:
            source, destination = (target, owner) if reverse else (owner, target)
            transfers.extend((source, destination, expert, role, elements) for role, elements in role_elements)
    rounds, fragments = [], []
    send_used, receive_used = [0] * ranks, [0] * ranks
    for source, target, expert, role, total in sorted(transfers):
        offset = 0
        while offset < total:
            count = min(total - offset, quota_elements - send_used[source], quota_elements - receive_used[target])
            if count == 0:
                rounds.append(TransferRound(tuple(fragments), ranks))
                fragments, send_used, receive_used = [], [0] * ranks, [0] * ranks
                continue
            fragments.append(Fragment(source, target, expert, role, offset, count))
            send_used[source] += count
            receive_used[target] += count
            offset += count
    if fragments:
        rounds.append(TransferRound(tuple(fragments), ranks))
    return tuple(rounds)


def copy_matrix_fragment(flat: torch.Tensor, matrix: torch.Tensor, offset: int, *, unpack: bool = False,
                         accumulate: bool = False) -> None:
    """Copy a logical matrix interval without materializing a transposed weight.

    Full rows copy through a matrix view; only the first and last partial rows
    require separate copies. Accumulation is used solely for owner gradients.
    """
    if (matrix.ndim != 2 or flat.ndim != 1 or not flat.is_contiguous() or offset < 0
            or offset + flat.numel() > matrix.numel() or accumulate and not unpack):
        raise ValueError("invalid logical matrix fragment")
    width, remaining, cursor = matrix.shape[1], flat.numel(), 0
    while remaining:
        row, column = divmod(offset, width)
        count = min(remaining, width - column) if column else min(remaining // width * width, remaining)
        if count == 0:
            count = remaining
        if column or count < width:
            view = matrix[row, column:column + count]
            segment = flat[cursor:cursor + count]
        else:
            view = matrix[row:row + count // width]
            segment = flat[cursor:cursor + count].view(count // width, width)
        if accumulate:
            view.add_(segment)
        elif unpack:
            view.copy_(segment)
        else:
            segment.copy_(view)
        offset, cursor, remaining = offset + count, cursor + count, remaining - count
