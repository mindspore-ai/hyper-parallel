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
"""LightningIndexer V2 adapter for document-local causal TopK selection."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Callable

import torch  # pylint: disable=forbidden-backend-import

from hyper_parallel.components.functional.sparse_flash_mla import csa_tnd_plan


# Float32 represents every integer up to this inclusive bound exactly.
_FP32_EXACT_INT_LIMIT = 1 << 24


def validate_indexer_tensors(query: torch.Tensor, key: torch.Tensor, weights: torch.Tensor) -> None:
    """Check logical tensor relationships before packing Indexer or KL inputs.

    Native operators validate their hardware-specific dtype and shape limits.

    Args:
        query: Indexer query [batch, queries, heads, dimension].
        key: Indexer key bank [batch, keys, dimension].
        weights: Per-query head weights [batch, queries, heads].

    Raises:
        ValueError: Tensor layouts, devices or query/key dtypes disagree.
    """
    if query.ndim != 4:
        raise ValueError(f"Indexer query must have shape [B, Q, H, D], got {tuple(query.shape)}")
    if key.ndim != 3 or key.shape[::2] != (query.shape[0], query.shape[-1]):
        raise ValueError(f"Indexer key must have shape [B, K, D] matching Q, got Q={tuple(query.shape)}, "
                         f"K={tuple(key.shape)}")
    if weights.shape != query.shape[:3]:
        raise ValueError(f"Indexer weights must have shape {tuple(query.shape[:3])}, got {tuple(weights.shape)}")
    if key.device != query.device or weights.device != query.device:
        raise ValueError(f"Indexer Q/K/weights must share a device, got {query.device}, {key.device}, {weights.device}")
    if key.dtype != query.dtype:
        raise ValueError(f"Indexer Q/K must share a dtype, got {query.dtype} and {key.dtype}")


def sort_key_indices(indices: torch.Tensor, maximum: int) -> torch.Tensor:
    """Sort key IDs without losing integer precision or changing their dtype.

    Args:
        indices: Integer key IDs in [-1, maximum], including any sentinel.
        maximum: A shape-derived upper bound; does not synchronize the device.
    """
    if maximum <= _FP32_EXACT_INT_LIMIT:
        return indices.float().sort(dim=-1).values.to(indices.dtype)
    return indices.sort(dim=-1).values


@lru_cache(maxsize=1)
def _load_indexer_op() -> Callable[..., Any]:
    """Import the LI V2 entry only when fused selection is requested."""
    # Keep the optional NPU extension out of CPU and eager-only imports.
    try:
        from cann_ops_transformer import (  # pylint: disable=import-outside-toplevel
            lightning_indexer as lightning_indexer_op,
        )
    except ImportError as err:
        raise RuntimeError(
            "use_fused_ops=True requires cann_ops_transformer.lightning_indexer (LI V2). "
            "See hyper_parallel/components/functional/README.md for ops-transformer installation, "
            "environment setup and tested versions."
        ) from err
    return lightning_indexer_op


@lru_cache(maxsize=1)
def _load_indexer_metadata_op() -> Callable[..., Any]:
    """Import the optional LI scheduling entry only when it will be used."""
    # The verified 910B kernel computes its own split and does not read metadata.
    try:
        from cann_ops_transformer import lightning_indexer_metadata  # pylint: disable=import-outside-toplevel
    except ImportError as err:
        raise RuntimeError(
            "use_fused_ops=True requires cann_ops_transformer.lightning_indexer_metadata on this device. "
            "See hyper_parallel/components/functional/README.md for ops-transformer installation, "
            "environment setup and tested versions."
        ) from err
    return lightning_indexer_metadata


@lru_cache(maxsize=16)
def _uses_indexer_metadata(device: torch.device) -> bool:
    """Omit unused LI scheduling metadata on the verified 910B path.

    LI V2's arch22 kernel computes its split internally; arch35 requires
    metadata. Keep the standard metadata call on other devices rather than
    assuming they share the verified 910B behavior. TND still needs cu_seqlens.
    """
    return "910B" not in torch.npu.get_device_name(device)


@lru_cache(maxsize=128)
@torch.inference_mode(False)
def _bsnd_indexer_metadata(device, batch, length, heads, dim, key_length, topk, ratio, residual):
    """Reuse shape-only BSND metadata without caching inference-mode tensors."""
    residue = torch.full((batch,), residual, device=device, dtype=torch.int32)
    metadata = None
    if _uses_indexer_metadata(device):
        metadata = _load_indexer_metadata_op()(
            heads, 1, dim, topk, cmp_residual_k=residue,
            batch_size=batch, max_seqlen_q=length, max_seqlen_k=key_length,
            layout_q="BSND", layout_k="BSND", mask_mode=3, cmp_ratio=ratio,
        )
    return residue, metadata


@torch.no_grad()
def _bsnd_lightning_indexer(
        query: torch.Tensor, key: torch.Tensor, weights: torch.Tensor, ratio: int, topk: int,
        segment: tuple[int, int, int, int, int],
) -> torch.Tensor:
    """Select one document segment per batch item in one BSND call.

    Query tiling is owned by the operator, independently of the external
    teacher chunk size. Multiple packed documents use the TND entry.

    Args:
        query: Local Indexer queries [batch, queries, heads, 128].
        key: Global Indexer key bank [batch, compressed keys, 128].
        weights: Local per-head merge weights, including model scaling.
        ratio: Model compression ratio.
        topk: Output selection width, including invalid tail slots.
        segment: Local Q start/end, global compressed K start/end, and raw
            prefix length modulo ratio, shared by all physical batch items.
    """
    batch, length, heads, dim = query.shape
    result = torch.full((batch, length, topk), -1, device=query.device, dtype=torch.int32)
    start, end, key_start, key_end, residual = segment
    if key_end <= key_start or start == end:
        return result
    indexer = _load_indexer_op()
    residue, metadata = _bsnd_indexer_metadata(
        query.device, batch, end-start, heads, dim, key_end-key_start, topk, ratio, residual,
    )
    with torch.autocast(device_type=query.device.type, enabled=False):
        indices, _ = indexer(
            query[:, start:end].contiguous(), key[:, key_start:key_end].unsqueeze(2).contiguous(),
            weights[:, start:end].float().contiguous(), topk,
            cmp_residual_k=residue, metadata=metadata,
            layout_q="BSND", layout_k="BSND", mask_mode=3, cmp_ratio=ratio,
        )
    indices = indices[:, :, 0]
    indices = torch.where(indices >= 0, indices + key_start, key.shape[1])
    indices = sort_key_indices(indices, key.shape[1])
    result[:, start:end] = indices.masked_fill(indices == key.shape[1], -1)
    return result


@lru_cache(maxsize=128)
@torch.inference_mode(False)
def _tnd_indexer_metadata(plan, device, heads, dim, topk, ratio):
    """Cache native selection metadata for the complete packed document partition."""
    options = plan.indexer_options(device, ratio)
    metadata = None
    if _uses_indexer_metadata(device):
        metadata = _load_indexer_metadata_op()(
            heads, 1, dim, topk, batch_size=len(plan.segments),
            max_seqlen_q=max(plan.q_lengths), max_seqlen_k=max(plan.k_lengths), **options,
        )
    return options, metadata


@torch.no_grad()
def lightning_indexer(
        query: torch.Tensor, key: torch.Tensor, weights: torch.Tensor, ratio: int, topk: int, segments: tuple,
) -> torch.Tensor:
    """Select packed document prefixes in one TND call; single documents use BSND.

    Args:
        query: Local Indexer queries [batch, queries, heads, 128].
        key: Global Indexer key bank [batch, compressed keys, 128].
        weights: Per-query head weights including model scaling.
        ratio: Model compression ratio.
        topk: Selection width, including invalid tail slots.
        segments: Local Q start/end, global document K start/prefix end and
            raw-prefix residual. Packed documents require batch size one.

    Returns:
        Sorted global key IDs [batch, queries, topk], with invalid slots -1.
    """
    if len(segments) == 1:
        return _bsnd_lightning_indexer(query, key, weights, ratio, topk, segments[0])
    if query.shape[0] != 1:
        raise ValueError("Packed Lightning Indexer requires batch size one")
    result = torch.full((*query.shape[:2], topk), -1, device=query.device, dtype=torch.int32)
    plan = csa_tnd_plan(segments, ratio)
    if plan is None:
        return result
    options, metadata = _tnd_indexer_metadata(plan, query.device, query.shape[2], query.shape[3], topk, ratio)
    with torch.autocast(device_type=query.device.type, enabled=False):
        indices, _ = _load_indexer_op()(
            plan.query(query), plan.key(key).unsqueeze(1), plan.query(weights).float(), topk,
            metadata=metadata, **options,
        )
    offsets = plan.lengths(query.device)[3]
    indices = torch.where(indices[:, 0] >= 0, indices[:, 0] + offsets, key.shape[1])
    indices = sort_key_indices(indices, key.shape[1])
    indices = indices.masked_fill(indices == key.shape[1], -1)
    plan.restore(indices, result)
    return result
