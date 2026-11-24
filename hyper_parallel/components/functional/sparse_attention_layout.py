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
"""Shared token packing and native sparse-attention sequence layouts."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import accumulate

import torch


def pack_ranges(tensor: torch.Tensor, ranges: tuple[tuple[int, int], ...]) -> torch.Tensor:
    """Pack token ranges without copying an already contiguous partition.

    Args:
        tensor: Source tensor with tokens on the first dimension.
        ranges: Nonempty, ordered half-open token intervals.

    Returns:
        Contiguous packed tokens; autograd restores gradients to the source.
    """
    if all(left[1] == right[0] for left, right in zip(ranges, ranges[1:])):
        start, end = ranges[0][0], ranges[-1][1]
        if start == 0 and end == tensor.shape[0]:
            return tensor.contiguous()
        return tensor[start:end].contiguous()
    return torch.cat([tensor[start:end] for start, end in ranges], dim=0).contiguous()


@dataclass(frozen=True)
class SparseTNDPlan:
    """Document-local lengths and canonical K offsets; no training activations."""

    segments: tuple[tuple[int, int, int, int, int], ...]
    q_ranges: tuple[tuple[int, int], ...]
    k_ranges: tuple[tuple[int, int], ...]
    q_lengths: tuple[int, ...]
    k_lengths: tuple[int, ...]
    residuals: tuple[int, ...]

    def query(self, tensor: torch.Tensor) -> torch.Tensor:
        """Pack a [1,Q,...] tensor in the plan's document order.

        Args:
            tensor: Input tensor to pack or map using this immutable plan.
        """
        return pack_ranges(tensor.squeeze(0), self.q_ranges)

    def key(self, tensor: torch.Tensor, key_offset: int = 0) -> torch.Tensor:
        """Pack global document prefixes using the physical origin of a [1,K,...] bank.

        Args:
            tensor: Input tensor to pack or map using this immutable plan.
            key_offset: Global token origin of the supplied physical key bank.
        """
        ranges = self.k_ranges if not key_offset else tuple((start-key_offset, end-key_offset)
                                                           for start, end in self.k_ranges)
        return pack_ranges(tensor.squeeze(0), ranges)

    def lengths(self, device: torch.device) -> tuple:
        """Reuse device length tensors, including across inference and recomputation.

        Args:
            device: Device on which immutable metadata will be consumed.
        """
        stream = None if device.type == "cpu" else torch.get_device_module(device).current_stream(device)
        return _device_lengths(self, device, stream)

    def indexer_options(self, device: torch.device, ratio: int) -> dict:
        """Share the document partition and causal mask between Indexer and KL.

        Args:
            device: Device on which immutable metadata will be consumed.
            ratio: Compression ratio determining causal block visibility.
        """
        cu_q, cu_k, residuals, _ = self.lengths(device)
        return {"cu_seqlens_q": cu_q, "cu_seqlens_k": cu_k,
                "cmp_residual_k": residuals if ratio > 1 else None,
                "layout_q": "TND", "layout_k": "TND", "mask_mode": 3, "cmp_ratio": ratio}

    def local_indices(self, indices: torch.Tensor, padded_width: int = 0) -> torch.Tensor:
        """Translate bank IDs; pad only when the consumer explicitly requires a width.

        Args:
            indices: Sparse key indices with negative invalid slots.
            padded_width: Minimum sparse-index width required by the consumer, or zero.
        """
        selected = self.query(indices)
        offsets = self.lengths(indices.device)[3]
        selected = torch.where(selected >= 0, selected - offsets, -1).int()
        if selected.shape[-1] < padded_width:
            selected = torch.nn.functional.pad(selected, (0, padded_width-selected.shape[-1]), value=-1)
        return selected.unsqueeze(1).contiguous()

    def covers_queries(self, length: int) -> bool:
        """Return whether packed rows already follow the complete original Q order.

        Args:
            length: Number of original local query rows.
        """
        return (self.q_ranges[0][0] == 0 and self.q_ranges[-1][1] == length
                and all(left[1] == right[0] for left, right in zip(self.q_ranges, self.q_ranges[1:])))

    def restore(self, part: torch.Tensor, output: torch.Tensor) -> None:
        """Write packed query rows into their original positions, preserving autograd.

        Args:
            part: Packed query rows to restore.
            output: Packed output to restore to original query order.
        """
        cursor = 0
        for start, end in self.q_ranges:
            output[0, start:end] = part[cursor:cursor+end-start]
            cursor += end-start


@lru_cache(maxsize=128)
def sparse_tnd_plan(segments: tuple, ratio: int, has_keys: bool = True) -> SparseTNDPlan | None:
    """Separate true SWA-only/empty-prefix queries without inventing valid keys.

    Args:
        segments: Query ranges and corresponding document key prefixes.
        ratio: Compression ratio determining causal block visibility.
        has_keys: Select nonempty compressed prefixes when true, or empty-prefix queries otherwise.
    """
    active = tuple(item for item in segments
                   if item[1] > item[0] and (bool(ratio and item[3] > item[2]) == has_keys))
    if not active:
        return None
    return SparseTNDPlan(
        active, tuple((s, e) for s, e, _, _, _ in active), tuple((s, e) for _, _, s, e, _ in active),
        tuple(e-s for s, e, _, _, _ in active), tuple(e-s for _, _, s, e, _ in active),
        tuple(r for _, _, _, _, r in active),
    )


@lru_cache(maxsize=128)
@torch.inference_mode(False)
def _device_lengths(plan: SparseTNDPlan, device: torch.device, stream: object = None) -> tuple:
    """Cache by the complete document partition, not merely by total Q/K sizes."""
    del stream
    cu_q = torch.tensor((0, *accumulate(plan.q_lengths)), device=device, dtype=torch.int32)
    cu_k = torch.tensor((0, *accumulate(plan.k_lengths)), device=device, dtype=torch.int32)
    residuals = torch.tensor(plan.residuals, device=device, dtype=torch.int32)
    offsets = torch.tensor([s for s, _ in plan.k_ranges], device=device, dtype=torch.int32)
    lengths = torch.tensor(plan.q_lengths, device=device, dtype=torch.int64)
    offsets = torch.repeat_interleave(offsets, lengths, output_size=sum(plan.q_lengths)).unsqueeze(1)
    return cu_q, cu_k, residuals, offsets


def cumulative_length_tensors(query_ends: tuple[int, ...], key_ends: tuple[int, ...],
                              device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Reuse immutable TND endpoints on their creating stream.

    Args:
        query_ends: Cumulative query endpoints without a leading zero.
        key_ends: Matching cumulative key endpoints without a leading zero.
        device: Consumer device; tensors are never copied back to CPU.
    """
    stream = None if device.type == "cpu" else torch.get_device_module(device).current_stream(device)
    return _cumulative_length_tensors(query_ends, key_ends, device, stream)


@lru_cache(maxsize=128)
@torch.inference_mode(False)
def _cumulative_length_tensors(query_ends, key_ends, device, stream):
    """Keep lengths independent of activations and safe for checkpoint replay."""
    del stream
    return (torch.tensor(query_ends, device=device, dtype=torch.int32),
            torch.tensor(key_ends, device=device, dtype=torch.int32))
