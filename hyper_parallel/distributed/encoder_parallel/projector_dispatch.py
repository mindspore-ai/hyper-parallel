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
"""Route spatial projector inputs and outputs between their unique owners.

Within an Encoder-CP bucket, each output row owns one spatial patch block.
The input plan gathers local patches and exchanges only missing remote patches;
negative bank indices represent spatial padding, not communication requests.

After projection, a separate parent CP-times-TP exchange sends output rows to
their decoder token intervals. This routing may replicate rows for decoder TP
without sequence parallelism, but never repeats their projector computation.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

import torch

from hyper_parallel.core.utils.communication import differentiable_all_to_all_single
from hyper_parallel.distributed.encoder_parallel.contracts import EncoderFeature, EncoderWorkItem
from hyper_parallel.distributed.encoder_parallel.exchange import cat_or_empty
from hyper_parallel.distributed.encoder_parallel.planning import DecoderSequenceLayout


def _exchange_rows(value: torch.Tensor, input_splits: Sequence[int],
                   output_splits: Sequence[int], group: Any) -> torch.Tensor:
    """Use the common variable-split exchange and its inverse gradient routing."""
    return differentiable_all_to_all_single(
        value.contiguous(), input_splits=list(input_splits), output_splits=list(output_splits), group=group,
    )


@dataclass(frozen=True)
class ProjectorShard:
    """One image's output-row interval and its offset in this producer's output.

    ``start``/``stop`` are image-local; ``offset`` is rank-local across images.
    """

    item: EncoderWorkItem
    start: int
    stop: int
    offset: int


def projector_shards(items: Sequence[EncoderWorkItem], rank: int, size: int) -> tuple[ProjectorShard, ...]:
    """Assign equal-sized ranges of concatenated projector rows to ranks.

    Args:
        items: Images in this Encoder-DP bucket.
        rank: Local producer coordinate.
        size: Number of producers in the bucket.
    """
    if size < 1 or not 0 <= rank < size:
        raise ValueError("projector rank must lie in a nonempty producer group")
    width = (sum(item.output_rows for item in items) + size - 1) // size
    left, right = rank * width, (rank + 1) * width
    shards = []
    begin = offset = 0
    for item in items:
        start, stop = max(left, begin), min(right, begin + item.output_rows)
        if start < stop:
            shards.append(ProjectorShard(item, start - begin, stop - begin, offset))
            offset += stop - start
        begin += item.output_rows
    return tuple(shards)


def _cat_indices(parts: Sequence[torch.Tensor]) -> torch.Tensor:
    return torch.cat(tuple(parts)) if parts else torch.empty(0, dtype=torch.int64)


def _input_requests(items: Sequence[EncoderWorkItem], shards: Sequence[ProjectorShard],
                    size: int, packed: bool, bridge: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Map each block input to its full-bank row and unique sequence owner."""
    total = sum(item.sequence_length for item in items)
    packed_width = max(1, (total + size - 1) // size)
    offsets = {}
    global_offset = local_offset = 0
    for item in items:
        local_width = (item.sequence_length + size - 1) // size
        offsets[item.item_id] = (global_offset, local_offset, local_width)
        global_offset += item.sequence_length
        local_offset += local_width
    bank_parts, owner_parts, local_parts = [], [], []
    for shard in shards:
        indices = bridge.projector_input_indices(shard.item, shard.start, shard.stop).reshape(-1)
        global_start, local_start, local_width = offsets[shard.item.item_id]
        bank = indices + global_start
        if packed:
            owners = bank.clamp_min(0).div(packed_width, rounding_mode="floor")
            local = bank.remainder(packed_width)
        else:
            owners = indices.clamp_min(0).div(local_width, rounding_mode="floor")
            local = indices.remainder(local_width) + local_start
        bank_parts.append(bank.masked_fill(indices < 0, -1))
        owner_parts.append(owners.masked_fill(indices < 0, -1))
        local_parts.append(local)
    return _cat_indices(bank_parts), _cat_indices(owner_parts), _cat_indices(local_parts)


@dataclass(frozen=True)
class EncoderProjectorPlan:
    """CPU-only spatial input routing, fixed for one microbatch.

    ``bank_indices`` maps flattened block slots to valid full-bucket patch rows
    or -1 for spatial padding. Send indices address this rank's ViT output.
    Local and receive positions address flattened block slots, in peer order.
    """

    shards: tuple[ProjectorShard, ...]
    patch_count: int
    bank_indices: torch.Tensor
    send_indices: torch.Tensor
    input_splits: tuple[int, ...]
    output_splits: tuple[int, ...]
    local_indices: torch.Tensor
    local_positions: torch.Tensor
    receive_positions: torch.Tensor
    has_images: bool

    def dispatch_blocks(self, hidden: torch.Tensor, mesh: Any) -> torch.Tensor:
        """Move each remote patch once and bypass patches already held locally.

        Args:
            hidden: This rank's packed ViT output, including local padding.
            mesh: Encoder-CP group that owns the bucket's patch shards.
        """
        width = hidden.shape[-1]
        result = hidden.new_zeros((self.bank_indices.numel(), width))
        local = hidden.index_select(0, self.local_indices.to(hidden.device))
        result = result.index_copy(0, self.local_positions.to(hidden.device), local)
        if self.has_images and mesh.size() > 1:
            send = hidden.index_select(0, self.send_indices.to(hidden.device))
            received = _exchange_rows(
                send, input_splits=list(self.input_splits), output_splits=list(self.output_splits),
                group=mesh.get_group(),
            )
            if self.receive_positions.numel():
                result = result.index_copy(0, self.receive_positions.to(hidden.device), received)
            else:
                # Empty index_copy is not an identity on every NPU backend.
                # Keep the exchange edge: send-only ranks still receive gradients.
                result = result + received.sum() * 0.0
        return result.reshape(-1, self.patch_count, width)


def build_projector_plan(items: Sequence[EncoderWorkItem], rank: int, size: int,
                         *, packed: bool, bridge: Any) -> EncoderProjectorPlan:
    """Plan block inputs without inspecting any activation or model parameter.

    Args:
        items: Images assigned to this Encoder-DP bucket.
        rank: Local Encoder-CP coordinate.
        size: Encoder-CP degree.
        packed: Whether the ViT shards the concatenated bucket sequence.
        bridge: Model-owned spatial block mapping.
    """
    shards = [projector_shards(items, peer, size) for peer in range(size)]
    requests = [_input_requests(items, value, size, packed, bridge) for value in shards]
    bank, owners, local = requests[rank]
    sends = [indices[peer_owners == rank] if peer != rank else indices[:0]
             for peer, (_, peer_owners, indices) in enumerate(requests)]
    positions = [torch.nonzero(owners == peer, as_tuple=False).flatten() if peer != rank else owners[:0]
                 for peer in range(size)]
    own_positions = torch.nonzero(owners == rank, as_tuple=False).flatten()
    return EncoderProjectorPlan(
        shards=shards[rank], patch_count=bridge.projector_patch_count, bank_indices=bank,
        send_indices=_cat_indices(sends), input_splits=tuple(part.numel() for part in sends),
        output_splits=tuple(part.numel() for part in positions), local_indices=local[own_positions],
        local_positions=own_positions, receive_positions=_cat_indices(positions), has_images=bool(items),
    )


def project_spatial_blocks(plan: EncoderProjectorPlan, blocks: torch.Tensor,
                           hidden: torch.Tensor, bridge: Any) -> tuple[torch.Tensor, torch.Tensor]:
    """Call the same FSDP projector once on every rank, including empty owners.

    Args:
        plan: Spatial block ownership for this rank.
        blocks: Local block inputs, including zero spatial padding.
        hidden: Graph-connected local ViT output for empty work.
        bridge: Model-owned projector entry point.
    """
    anchor = blocks.sum() * 0.0 + hidden.sum() * 0.0
    count = blocks.shape[0]
    if not count:
        blocks = hidden[:1].unsqueeze(1).expand(1, plan.patch_count, -1) * 0.0 + anchor
    projected = bridge.project_blocks(blocks)
    return projected[:count], anchor + projected.sum() * 0.0


def _needed_rows(bridge: Any, item: EncoderWorkItem, layout: DecoderSequenceLayout) -> tuple[int, int]:
    return bridge.feature_interval(item, layout.global_start, layout.global_stop)


def route_projected_rows(features: torch.Tensor, *, buckets: Sequence[Sequence[EncoderWorkItem]],
                         layouts: Mapping[int, DecoderSequenceLayout], parent_rank: int,
                         encoder_cp_size: int, mesh: Any, bridge: Any) -> tuple[list[EncoderFeature], torch.Tensor]:
    """Send contiguous feature intersections directly to their decoder owners.

    Args:
        features: This producer's packed projector output rows.
        buckets: Images assigned to each Encoder-DP bucket.
        layouts: Decoder token intervals for every parent rank.
        parent_rank: Local coordinate in the CP-times-TP parent.
        encoder_cp_size: Number of projector producers in each bucket.
        mesh: Parent mesh, including TP replicas when decoder SP is disabled.
        bridge: Model-owned mapping from token intervals to feature rows.

    Returns:
        One partial record per locally relevant image and a collective anchor.
    """
    size = len(layouts)
    source_shards = [projector_shards(buckets[peer // encoder_cp_size], peer % encoder_cp_size,
                                      encoder_cp_size) for peer in range(size)]
    send_parts, input_splits = [], []
    for destination in range(size):
        parts = []
        for shard in source_shards[parent_rank]:
            first, last = _needed_rows(bridge, shard.item, layouts[destination])
            start, stop = max(first, shard.start), min(last, shard.stop)
            if start < stop:
                begin = shard.offset + start - shard.start
                parts.append(features[begin:begin + stop - start])
        input_splits.append(sum(part.shape[0] for part in parts))
        send_parts.extend(parts)
    send = cat_or_empty(send_parts, anchor=features, feature_width=features.shape[-1])
    output_splits, receipts = [], []
    offset = 0
    layout = layouts[parent_rank]
    for shards in source_shards:
        count = 0
        for shard in shards:
            first, last = _needed_rows(bridge, shard.item, layout)
            start, stop = max(first, shard.start), min(last, shard.stop)
            if start < stop:
                receipts.append((shard.item.item_id, start, stop, offset + count))
                count += stop - start
        output_splits.append(count)
        offset += count
    rows = _exchange_rows(
        send, input_splits=input_splits, output_splits=output_splits, group=mesh.get_group(),
    ) if size > 1 else send
    items = tuple(item for bucket in buckets for item in bucket)
    records = _received_projector_records(rows, receipts, items, layout, bridge)
    return records, rows.sum() * 0.0


def _received_projector_records(rows: torch.Tensor, receipts: Sequence[tuple[int, int, int, int]],
                                items: Sequence[EncoderWorkItem], layout: DecoderSequenceLayout,
                                bridge: Any) -> list[EncoderFeature]:
    """Restore each locally relevant image from its ordered producer intervals."""
    records = []
    for item in sorted(items,
                       key=lambda value: (value.batch_index, value.decoder_token_start, value.item_id)):
        if max(item.decoder_token_start, layout.global_start) >= min(
                item.decoder_token_start + item.decoder_token_length, layout.global_stop):
            continue
        first, last = _needed_rows(bridge, item, layout)
        spans = sorted((start, stop, offset) for item_id, start, stop, offset in receipts if item_id == item.item_id)
        if sum(stop - start for start, stop, _ in spans) != last - first:
            raise RuntimeError(f"projector route does not cover image {item.item_id} rows [{first}, {last})")
        pieces = [rows[offset:offset + stop - start] for start, stop, offset in spans]
        feature = cat_or_empty(pieces, anchor=rows, feature_width=rows.shape[-1])
        records.append((replace(item, feature_row_start=first), feature))
    return records
