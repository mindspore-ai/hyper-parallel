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
"""Document-local compression blocks and compact bank ownership for contiguous CP."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from functools import lru_cache
from itertools import accumulate
from typing import Any

import torch  # pylint: disable=forbidden-backend-import
from torch.nn import functional  # pylint: disable=forbidden-backend-import


@dataclass(frozen=True)
class CompressionBoundaryRoute:
    """Projected rows needed to complete a block owned by its final-token rank."""

    global_start: int
    send_offsets: tuple[int, ...]
    input_splits: tuple[int, ...]
    output_splits: tuple[int, ...]
    needs_exchange: bool


def _boundary_route(local_length: int, rank: int, intervals: tuple) -> CompressionBoundaryRoute:
    """Derive left-boundary exchanges; all ranks enter the same collective."""
    local_start = rank * local_length
    global_start, global_end = intervals[rank]
    offsets, sends, receives = [], [], []
    for peer, (start, end) in enumerate(intervals):
        first = max(local_start, start)
        count = max(0, min(local_start + local_length, end) - first) if peer != rank else 0
        offsets.append(first - local_start if count else 0)
        sends.append(count)
        peer_start = peer * local_length
        receives.append(max(0, min(peer_start + local_length, global_end) - max(peer_start, global_start))
                        if peer != rank else 0)
    needs_exchange = any(start < peer * local_length for peer, (start, _) in enumerate(intervals))
    return CompressionBoundaryRoute(global_start, tuple(offsets), tuple(sends), tuple(receives), needs_exchange)


@lru_cache(maxsize=128)
def compressed_document_offsets(boundaries: tuple[int, ...], ratio: int) -> tuple[int, ...]:
    """Return canonical K boundaries, dropping only incomplete document tails."""
    if ratio < 1 or len(boundaries) < 2 or boundaries[0] != 0 or any(
            end <= start for start, end in zip(boundaries, boundaries[1:])):
        raise ValueError("Compression requires a positive ratio and strictly increasing document boundaries")
    return (0, *accumulate((end-start)//ratio for start, end in zip(boundaries, boundaries[1:])))


@lru_cache(maxsize=128)
def compressed_owner_offsets(boundaries: tuple[int, ...], ratio: int, local_length: int) -> tuple[int, ...]:
    """Assign each complete block to the raw rank containing its last token."""
    keys = compressed_document_offsets(boundaries, ratio)
    if local_length <= 0 or boundaries[-1] % local_length:
        raise ValueError("Compression requires equal positive raw CP shard lengths")
    offsets = []
    for position in range(0, boundaries[-1], local_length):
        document = bisect_right(boundaries, position) - 1
        offsets.append(keys[document] + (position-boundaries[document])//ratio)
    return (*offsets, keys[-1])


@dataclass(frozen=True)
class DocumentCompressionPlan:
    """Immutable block ranges; never enumerate the full sequence on the CPU."""

    ratio: int
    local_length: int
    rank: int
    counts: tuple[int, ...]
    local_blocks: tuple[tuple[int, int], ...]
    route: CompressionBoundaryRoute

    def indices(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Reuse device indices for local blocks and their original rotary positions."""
        return _compression_indices(self, device)

    def exchange(self, tensor: torch.Tensor, context: Any) -> torch.Tensor:
        """Complete boundary blocks with the existing differentiable A2AV protocol."""
        if not self.route.needs_exchange:
            return tensor
        if context is None or context.exchange_boundary is None:
            raise ValueError("natural document compression across CP requires boundary dispatch")
        return context.exchange_boundary(tensor, self.route)

    def pool(self, key: torch.Tensor, gate: torch.Tensor, context: Any) -> torch.Tensor:
        """Pool complete same-document groups using the model's original FP32 softmax."""
        rows = self.indices(key.device)[0]
        projected = self.exchange(torch.cat((key, gate), dim=-1), context)
        blocks = projected.index_select(1, rows).reshape(
            key.shape[0], self.counts[self.rank], self.ratio, projected.shape[-1],
        )
        values, scores = blocks.chunk(2, dim=-1)
        return (values * scores.float().softmax(dim=2).to(values.dtype)).sum(dim=2)

    def positions(self, cos: torch.Tensor, sin: torch.Tensor, context: Any) -> tuple:
        """Take original block-start RoPE; retain FP32 boundary metadata precision."""
        positions = self.indices(cos.device)[1]
        embeddings = self.exchange(torch.cat((cos, sin), dim=-1), context)
        return embeddings.index_select(1, positions).chunk(2, dim=-1)

    def launch_bank(self, tensor: torch.Tensor, context: Any) -> CompactBankGather:
        """All-gather equal-capacity containers and exclude every padding slot."""
        capacity = max(self.counts)
        if not capacity:
            return CompactBankGather(None, tensor, None)
        padded = tensor if tensor.shape[1] == capacity else functional.pad(
            tensor, (0, 0, 0, capacity - tensor.shape[1]),
        )
        slots = _bank_slots(self.counts, tensor.device)
        return CompactBankGather(context.launch(padded, 1), tensor, slots)


@dataclass
class CompactBankGather:
    """Complete one bank gather with at most one compacting index selection."""

    handle: Any
    tensor: torch.Tensor
    slots: torch.Tensor | None

    def wait(self) -> torch.Tensor:
        """Keep compaction in autograd so all consumers return gradients to owners."""
        output = self.tensor if self.handle is None else self.handle.wait()
        return output if self.slots is None else output.index_select(1, self.slots)


@lru_cache(maxsize=128)
def document_compression_plan(boundaries: tuple[int, ...], ratio: int, local_length: int,
                              size: int, rank: int) -> DocumentCompressionPlan:
    """Assign complete blocks to the rank owning their final token.

    Work on document/rank intersections rather than all global token rows.
    The packed-sequence boundary validator owns interval validation.
    """
    if ratio <= 1 or local_length <= 0 or not 0 <= rank < size:
        raise ValueError("document block planning requires ratio > 1 and a valid contiguous CP shard")
    if boundaries[0] != 0 or boundaries[-1] != size * local_length:
        raise ValueError("document boundaries must cover the complete contiguous CP sequence")
    owners = compressed_owner_offsets(boundaries, ratio, local_length)
    counts = tuple(end-start for start, end in zip(owners, owners[1:]))
    first_rows = [peer * local_length for peer in range(size)]
    local_blocks = []
    for start, end in zip(boundaries, boundaries[1:]):
        blocks = (end - start) // ratio
        if not blocks:
            continue
        first_owner = (start + ratio - 1) // local_length
        last_owner = (start + blocks * ratio - 1) // local_length
        for peer in range(first_owner, last_owner + 1):
            first = max(0, (peer * local_length - start) // ratio)
            last = min(blocks, ((peer + 1) * local_length - start) // ratio)
            if last > first:
                first_rows[peer] = min(first_rows[peer], start + first * ratio)
                if peer == rank:
                    local_blocks.append((start + first * ratio, last - first))
    intervals = tuple((first, (peer + 1) * local_length) for peer, first in enumerate(first_rows))
    return DocumentCompressionPlan(ratio, local_length, rank, tuple(counts), tuple(local_blocks),
                                   _boundary_route(local_length, rank, intervals))


@lru_cache(maxsize=8)
@torch.inference_mode(False)
def _compression_indices(plan: DocumentCompressionPlan, device: torch.device) -> tuple:
    """Reuse block rows; bound the cache since compact AG maps scale with global K."""
    starts = [torch.arange(start, start + count * plan.ratio, plan.ratio, device=device)
              for start, count in plan.local_blocks]
    starts = torch.cat(starts) if starts else torch.empty(0, device=device, dtype=torch.long)
    starts = starts - plan.route.global_start
    rows = (starts.unsqueeze(1) + torch.arange(plan.ratio, device=device)).flatten()
    return rows.long(), starts.long()


@lru_cache(maxsize=8)
@torch.inference_mode(False)
def _bank_slots(counts: tuple[int, ...], device: torch.device) -> torch.Tensor | None:
    """Materialize global padding maps only for the all-gather reference path."""
    if len(set(counts)) == 1:
        return None
    capacity = max(counts)
    return torch.cat([torch.arange(peer*capacity, peer*capacity+count, device=device)
                      for peer, count in enumerate(counts)])
