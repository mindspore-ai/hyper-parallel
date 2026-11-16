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
"""Attention over raw and compressed KV banks in DeepSeek V4.1."""

from __future__ import annotations

from functools import partial
from typing import Protocol

import torch

from hyper_parallel.components.functional.sparse_flash_mla import sparse_flash_mla, supports_fused_attention_inputs

from .._dependency import load_cann
from .._dsa_layout import DsaQueryLayout
from ..registry import OpRegistry
from ..spec import OpSpec


class DsaCompressedBanksFn(Protocol):
    """Attend raw and compressed KV banks in the same document coordinates."""

    def __call__(
        self, x: torch.Tensor, raw_key: torch.Tensor, compressed_key: torch.Tensor | None,
        topk_indices: torch.Tensor | None, sinks: torch.Tensor, *, scale: float, window: int,
        layout: DsaQueryLayout, global_sequence_length: int, raw_key_offset: int = 0,
        main_key_offset: int = 0, global_key_length: int | None = None,
    ) -> torch.Tensor:
        """Return attention output in BSND layout."""


DSA_COMPRESSED_BANKS = OpSpec[DsaCompressedBanksFn](config_key="dsa_sparse_attention", variant="compressed_banks")


def build_sliding_window_indices(
        batch_size: int,
        sequence_length: int,
        window_size: int,
        device: torch.device,
        *,
        query_offset: int = 0,
        key_length: int | None = None,
) -> torch.Tensor:
    """Build fixed-width causal sliding-window indices."""
    if key_length is None:
        key_length = sequence_length
    width = min(key_length, window_size)
    query = torch.arange(
        query_offset,
        query_offset + sequence_length,
        device=device,
    ).unsqueeze(-1)
    offsets = torch.arange(width - 1, -1, -1, device=device)
    indices = query - offsets
    indices = indices.masked_fill(indices < 0, -1)
    return indices.unsqueeze(0).expand(batch_size, -1, -1)


def reference_sparse_attention(
        query: torch.Tensor,
        key_value: torch.Tensor,
        sparse_indices: torch.Tensor,
        sinks: torch.Tensor,
        scale: float,
) -> torch.Tensor:
    """Compute sink attention with eager Torch operations on any device.

    Args:
        query: Main queries [batch, heads, local queries, head dimension].
        key_value: Shared K=V bank [batch, 1, keys, head dimension].
        sparse_indices: Visible bank IDs [batch, local queries, selections];
            negative entries are padding and repeated IDs count only once.
        sinks: Learnable scalar sink logits [heads], with zero-valued output.
        scale: Main attention logit scale.

    Returns:
        Attention values [batch, local queries, heads, head dimension].
        Softmax uses FP32; matrix multiplications retain the input precision.

    Note:
        This reference materializes a dense query/key mask and scores. It is
        intended for explicit unfused comparisons, not long-sequence efficiency.
    """
    batch_size, num_heads, sequence_length, _ = query.shape
    key_length = key_value.shape[2]
    valid = sparse_indices >= 0
    safe_indices = sparse_indices.clamp_min(0).long()
    allowed_counts = torch.zeros(
        batch_size,
        sequence_length,
        key_length,
        dtype=torch.int32,
        device=query.device,
    )
    allowed_counts.scatter_add_(2, safe_indices, valid.to(torch.int32))
    allowed = allowed_counts > 0
    logits = torch.matmul(query, key_value.transpose(-1, -2)) * scale
    logits = logits.masked_fill(~allowed.unsqueeze(1), float("-inf"))
    sink_logits = sinks.view(1, num_heads, 1, 1).expand(batch_size, -1, sequence_length, -1)
    probabilities = torch.cat((logits, sink_logits), dim=-1).softmax(dim=-1, dtype=torch.float32)
    return torch.matmul(probabilities[..., :-1].to(key_value.dtype), key_value).transpose(1, 2)


def native_dsa_sparse_attention(
    x: torch.Tensor, raw_key: torch.Tensor, compressed_key: torch.Tensor | None,
    topk_indices: torch.Tensor | None, sinks: torch.Tensor, *, scale: float, window: int,
    layout: DsaQueryLayout, global_sequence_length: int, raw_key_offset: int = 0,
    main_key_offset: int = 0, global_key_length: int | None = None,
) -> torch.Tensor:
    """Assemble the reference sliding-window and compressed key indices."""
    raw_indices = build_sliding_window_indices(
        raw_key.shape[0], x.shape[2], window, raw_key.device,
        query_offset=layout.query_offset, key_length=global_sequence_length,
    )
    starts = layout.segment_starts
    if starts is None and layout.segments is not None:
        starts = torch.zeros(x.shape[2], dtype=torch.long, device=x.device)
        for (begin, end, *_), (_, origin) in zip(layout.segments, layout.raw_document_starts()):
            starts[begin:end] = origin
        starts = starts.unsqueeze(0)
    if starts is not None:
        raw_indices = raw_indices.masked_fill(raw_indices < starts.unsqueeze(-1), -1)
    raw_indices = torch.where(raw_indices >= 0, raw_indices - raw_key_offset, -1)
    if compressed_key is None:
        return reference_sparse_attention(x, raw_key, raw_indices, sinks, scale)
    indices = topk_indices
    if global_key_length is not None:
        indices = indices.masked_fill(indices >= global_key_length, -1)
    indices = torch.where(indices >= 0, indices.long() - main_key_offset + raw_key.shape[2], -1)
    return reference_sparse_attention(
        x, torch.cat((raw_key, compressed_key.unsqueeze(1)), dim=2),
        torch.cat((raw_indices, indices), dim=-1), sinks, scale,
    )


def attention_native_only(
    x: torch.Tensor, layout: DsaQueryLayout, window: int, topk_indices: torch.Tensor | None,
    global_sequence_length: int, global_key_length: int | None,
) -> str | None:
    """Explain incompatible layouts before attempting to load Sparse Flash MLA."""
    ratio = layout.compress_ratio
    topk = 0 if topk_indices is None else topk_indices.shape[-1]
    expected = global_sequence_length // max(1, ratio) if global_key_length is None else global_key_length
    if (layout.segments is None or (topk_indices is not None and topk != min(512, expected))
            or not supports_fused_attention_inputs(x, ratio, window, topk)):
        return "Sparse Flash MLA requires supported BF16 NPU inputs, document segments and ordinary TopK"
    return None


def _cann_attention(
    x: torch.Tensor, raw_key: torch.Tensor, compressed_key: torch.Tensor | None,
    topk_indices: torch.Tensor | None, sinks: torch.Tensor, *, scale: float, window: int,
    layout: DsaQueryLayout, global_sequence_length: int, raw_key_offset: int = 0,
    main_key_offset: int = 0, global_key_length: int | None = None,
) -> torch.Tensor:
    del global_sequence_length, global_key_length
    output = sparse_flash_mla(
        x, raw_key, compressed_key, topk_indices, sinks, scale, layout.compress_ratio, window,
        layout.query_offset, layout.segments, raw_key_offset, main_key_offset, layout.raw_document_starts(),
    )
    if compressed_key is not None and all(end == start for _, _, start, end, _ in layout.segments):
        # Keep the shared bank's collective backward active on empty-prefix ranks.
        output = output + compressed_key[:, :0].sum().to(output.dtype)
    return output


def register_dsa_sparse_attention(registry: OpRegistry) -> None:
    """Register the compressed-banks attention contract and its implementations."""
    registry.register_family("dsa_sparse_attention", specs=(DSA_COMPRESSED_BANKS,))
    registry.register_impl(
        DSA_COMPRESSED_BANKS, name="native", devices=("cpu", "cuda", "npu"), load=lambda: native_dsa_sparse_attention,
    )
    registry.register_impl(
        DSA_COMPRESSED_BANKS, name="cann", devices=("npu",),
        load=partial(load_cann, _cann_attention, "sparse_flash_mla", "sparse_flash_mla_grad",
                     "sparse_flash_mla_metadata"),
    )
