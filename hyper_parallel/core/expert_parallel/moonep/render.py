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
"""Render virtual experts with AscendC metadata and existing NPU indexing ops."""
import torch

from .allocator import _device_ub_bytes, load_allocator


def _native_counts(
    gathered: torch.Tensor, cumulative: torch.Tensor, copies: torch.Tensor,
    rank: int, extra: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ranks, experts = gathered.shape
    local = experts // ranks
    width = local + extra
    home = torch.arange(experts, device=gathered.device, dtype=torch.int32).reshape(ranks, local)
    table = torch.cat((home, copies[:, :extra]), dim=1)
    valid = table >= 0
    indices = table.clamp_min(0).long()
    destination_starts = torch.cat((torch.zeros_like(cumulative[:, :1]), cumulative[:, :-1]), dim=1)
    ends = cumulative.T.gather(1, indices)
    starts = destination_starts.T.gather(1, indices)
    source_ends = gathered.cumsum(0, dtype=torch.int32)
    source_starts = source_ends - gathered
    end_rows = source_ends.index_select(1, indices.flatten()).reshape(ranks, ranks, width)
    start_rows = source_starts.index_select(1, indices.flatten()).reshape(ranks, ranks, width)
    counts = (torch.minimum(end_rows, ends) - torch.maximum(start_rows, starts)).clamp_min_(0)
    counts.masked_fill_(~valid, 0)
    lookup = torch.zeros((ranks, experts + 1), dtype=torch.int32, device=gathered.device)
    slots = torch.arange(width, device=gathered.device, dtype=torch.int32).expand(ranks, width)
    lookup.scatter_(1, table.masked_fill(~valid, experts).long(), slots)
    repeats = (
        torch.minimum(source_ends[rank, :, None], cumulative)
        - torch.maximum(source_starts[rank, :, None], destination_starts)
    ).clamp_min_(0)
    labels = lookup[:, :experts].T + torch.arange(ranks, device=gathered.device, dtype=torch.int32) * width
    return counts.reshape(ranks, ranks * width), repeats.flatten(), labels.flatten()


def render_route(
    ids: torch.Tensor,
    inverse: torch.Tensor,
    gathered: torch.Tensor,
    cumulative: torch.Tensor,
    copies: torch.Tensor,
    rank: int,
    extra: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Expand expert-major assignments and restore the native token order."""
    ranks, experts = gathered.shape
    width = experts // ranks + extra
    def aligned(value: int) -> int:
        """Align metadata lengths for the native vector scratch layout."""
        return (value + 7) // 8 * 8

    scratch_bytes = (5 * aligned(ranks * experts) + aligned(experts) + aligned(ranks * width)) * 4
    if scratch_bytes <= _device_ub_bytes(ids.device.index) * 3 // 4:
        counts, repeats, labels = load_allocator().render(gathered, cumulative, copies, rank, width)
    else:
        counts, repeats, labels = _native_counts(gathered, cumulative, copies, rank, extra)
    sorted_ids = torch.repeat_interleave(labels, repeats, output_size=ids.numel())
    virtual = sorted_ids.index_select(0, inverse.flatten()).reshape_as(ids)
    return virtual, counts
