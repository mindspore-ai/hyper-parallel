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
"""FP32 teacher geometry and selected-key access shared by CSA Indexer paths."""

from __future__ import annotations

from collections.abc import Callable

import torch  # pylint: disable=forbidden-backend-import


_FP32_TEMPORARY_BYTES = 64 * 1024 * 1024


def sort_key_indices(indices: torch.Tensor, maximum: int) -> torch.Tensor:
    """Sort key IDs without losing integer precision or changing their dtype.

    Args:
        indices: Integer key IDs in [-1, maximum], including any sentinel.
        maximum: A shape-derived upper bound; does not synchronize the device.
    """
    if maximum <= 2**24:
        return indices.float().sort(dim=-1).values.to(indices.dtype)
    return indices.sort(dim=-1).values


def gather_selected_keys(bank: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gather batch-local nonnegative indices without multidimensional indexing.

    Args:
        bank: Key bank with shape [batch, keys, dimension].
        indices: Safe, nonnegative indices with shape [batch, queries, selected].
    """
    offsets = torch.arange(bank.shape[0], device=bank.device).view(-1, 1, 1) * bank.shape[1]
    flat_indices = (indices + offsets).reshape(-1)
    selected = bank.reshape(-1, bank.shape[-1]).index_select(0, flat_indices)
    return selected.reshape(*indices.shape, bank.shape[-1])


def _teacher_bank(bank: torch.Tensor, selected_count: int) -> torch.Tensor:
    """Cast a bank only if it saves a larger gathered low-precision temporary."""
    cast_bytes = bank.numel() * 4
    gathered_bytes = selected_count * bank.shape[-1] * bank.element_size()
    if bank.dtype != torch.float32 and cast_bytes < min(gathered_bytes, _FP32_TEMPORARY_BYTES):
        return bank.float()
    return bank


def gather_selected_keys_fp32(bank: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Choose the smaller temporary without converting an unbounded global bank.

    Args:
        bank: Key bank with shape [batch, keys, dimension].
        indices: Nonnegative key IDs with shape [batch, queries, selected].
    """
    return gather_selected_keys(_teacher_bank(bank, indices.numel()), indices).float()


def compressed_attention_teacher(
        query: torch.Tensor, key: torch.Tensor, indices: torch.Tensor, sinks: torch.Tensor,
        scale: float, reduce_sum: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> torch.Tensor:
    """Compute the detached selected-main-plus-sink teacher for one Q chunk.

    Args:
        query: Main attention query [batch, heads, queries, dimension].
        key: Compressed main key bank [batch, keys, dimension].
        indices: Batch-local main-key indices; negative slots are invalid.
        sinks: Scalar attention sink per head.
        scale: Main attention logit scale.
        reduce_sum: Optional TP SUM of head contributions, before normalization.
    """
    with torch.no_grad(), torch.autocast(device_type=query.device.type, enabled=False):
        batch, heads, length, dim = query.shape
        width = indices.shape[-1]
        if width == 0:
            return query.new_zeros(indices.shape, dtype=torch.float32)
        valid = indices >= 0
        safe = indices.clamp_min(0).long()
        selected_chunk = min(width, max(1, _FP32_TEMPORARY_BYTES // max(1, batch * length * dim * 4)))
        head_chunk = min(heads, max(1, _FP32_TEMPORARY_BYTES // max(1, batch * length * width * 4)))
        bank = _teacher_bank(key, batch * length * selected_chunk)
        target = query.new_zeros(indices.shape, dtype=torch.float32)
        for head_start in range(0, heads, head_chunk):
            head_end = min(head_start + head_chunk, heads)
            queries = query[:, head_start:head_end].float()
            scores = query.new_empty((batch, head_end-head_start, length, width), dtype=torch.float32)
            for start in range(0, width, selected_chunk):
                end = min(start + selected_chunk, width)
                selected = gather_selected_keys(bank, safe[:, :, start:end]).float()
                scores[..., start:end] = torch.einsum("bhqd,bqkd->bhqk", queries, selected) * scale
                del selected
            scores.masked_fill_(~valid.unsqueeze(1), -1.0e9)
            sink_logits = sinks[head_start:head_end].float().view(1, -1, 1, 1).expand(*scores.shape[:-1], 1)
            probabilities = torch.cat((scores, sink_logits), dim=-1).softmax(dim=-1)
            target.add_(probabilities[..., :-1].masked_fill(~valid.unsqueeze(1), 0).sum(dim=1))
            del scores, probabilities, queries
        if reduce_sum is not None:
            target = reduce_sum(target)
        return target / target.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(torch.float32).tiny)
