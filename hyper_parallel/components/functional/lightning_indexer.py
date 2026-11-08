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

from collections import OrderedDict
from dataclasses import dataclass, field
from functools import lru_cache
from itertools import accumulate
from threading import RLock
from typing import Any, Callable

import torch  # pylint: disable=forbidden-backend-import

from hyper_parallel.components.functional.sparse_flash_mla import CSATNDPlan, csa_tnd_plan


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
        # Singleton batch strides may survive contiguous(); view canonicalizes the native BSND layout.
        native_key = key[:, key_start:key_end].contiguous().view(batch, key_end-key_start, 1, dim)
        indices, _ = indexer(
            query[:, start:end].contiguous(), native_key,
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


def _cached_tensor_bytes(value: Any) -> int:
    """Conservatively count retained tensor storage, including storage behind views."""
    if isinstance(value, torch.Tensor):
        return value.untyped_storage().nbytes()
    if isinstance(value, dict):
        return sum(_cached_tensor_bytes(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return sum(_cached_tensor_bytes(item) for item in value)
    return 0


@dataclass(frozen=True)
class _PagedCacheKey:
    """Hash a batch plan once, even when LRU lookup and promotion both need its key."""

    fields: tuple
    cached_hash: int = field(init=False, compare=False)

    def __post_init__(self) -> None:
        """Avoid repeatedly walking thousands of document ranges on a cache hit."""
        object.__setattr__(self, "cached_hash", hash(self.fields))

    def __hash__(self) -> int:
        """Return the precomputed hash; equality still compares the complete fields."""
        return self.cached_hash


class _PagedTensorCache:
    """Bound retained page-address and scheduling tensors by bytes, not batch count."""

    def __init__(self, maximum_bytes: int) -> None:
        """Set a process-local budget for immutable paging tensors."""
        self.maximum_bytes = maximum_bytes
        self.retained_bytes = 0
        self.entries = OrderedDict()
        self.lock = RLock()

    def get(self, key: Any, create: Callable[[], Any]) -> Any:
        """Reuse immutable tensors; an oversized result remains valid but is not cached."""
        with self.lock:
            entry = self.entries.get(key)
            if entry is not None:
                self.entries.move_to_end(key)
                return entry[0]
            value = create()
            size = _cached_tensor_bytes(value)
            if size <= self.maximum_bytes:
                while self.entries and self.retained_bytes + size > self.maximum_bytes:
                    _, (_, removed) = self.entries.popitem(last=False)
                    self.retained_bytes -= removed
                self.entries[key] = (value, size)
                self.retained_bytes += size
            return value

    def clear(self) -> None:
        """Release only cached addresses and metadata; never model activations."""
        with self.lock:
            self.entries.clear()
            self.retained_bytes = 0


# This budget covers only cached paging tensors, not active bank values or native workspace.
_PAGED_CACHE = _PagedTensorCache(64 * 1024 * 1024)


def _paged_cache_key(name: str, device: torch.device, *fields: Any) -> _PagedCacheKey:
    """Scope device tensors to their producer stream to avoid cross-stream read hazards."""
    stream = None if device.type == "cpu" else torch.get_device_module(device).current_stream(device)
    return _PagedCacheKey((name, device, stream, *fields))


@dataclass(frozen=True)
class _PagedKeyPlan:
    """Compact page ranges per logical prefix; no rectangular Python page table."""

    regions: tuple[tuple[int, int], ...]
    page_starts: tuple[int, ...]
    page_counts: tuple[int, ...]
    page_size: int
    zero_copy: bool

    @property
    def maximum_pages(self) -> int:
        """Number of columns required by the native rectangular block table."""
        return max(self.page_counts)

    def key(self, tensor: torch.Tensor, rows: torch.Tensor | None = None) -> torch.Tensor:
        """Materialize each required document prefix at most once, with page padding."""
        source = tensor.squeeze(0)
        if not self.zero_copy:
            if rows is None:
                rows = _paged_key_rows(self.regions, self.page_size, tensor.device)
            source = source.index_select(0, rows)
        # The verified kernel requires physically contiguous pages, including dimension zero.
        return source.contiguous().view(-1, self.page_size, 1, tensor.shape[-1])


@lru_cache(maxsize=128)
def _paged_key_plan(plan: CSATNDPlan, bank_length: int, page_size: int = 128) -> _PagedKeyPlan:
    """Share physical pages between repeated prefixes with the same document origin."""
    required = {}
    for start, end in plan.k_ranges:
        if not 0 <= start < end <= bank_length:
            raise ValueError("Paged Indexer segments must address valid key-bank prefixes")
        required[start] = max(required.get(start, start), end)
    regions = tuple(sorted(required.items()))
    zero_copy = bank_length % page_size == 0 and all(start % page_size == 0 for start, _ in regions)
    bases = {}
    cursor = 0
    for start, end in regions:
        bases[start] = start // page_size if zero_copy else cursor
        cursor += (end - start + page_size - 1) // page_size
    starts = tuple(bases[start] for start, _ in plan.k_ranges)
    counts = tuple((length + page_size - 1) // page_size for length in plan.k_lengths)
    return _PagedKeyPlan(regions, starts, counts, page_size, zero_copy)


@torch.inference_mode(False)
def _paged_block_table(pages: _PagedKeyPlan, device: torch.device) -> torch.Tensor:
    """Expand compact page ranges on device and bound retained tables by bytes."""
    def create() -> torch.Tensor:
        """Expand native table columns from page ranges."""
        columns = torch.arange(pages.maximum_pages, device=device, dtype=torch.int32).unsqueeze(0)
        starts = torch.tensor(pages.page_starts, device=device, dtype=torch.int32).unsqueeze(1)
        counts = torch.tensor(pages.page_counts, device=device, dtype=torch.int32).unsqueeze(1)
        return (starts + columns).masked_fill_(columns >= counts, 0)
    return _PAGED_CACHE.get(_paged_cache_key("table", device, pages), create)


@torch.inference_mode(False)
def _paged_key_rows(regions: tuple, page_size: int, device: torch.device) -> torch.Tensor:
    """Cache only row addresses; unused page-tail values are masked by seqused_k."""
    def create() -> torch.Tensor:
        """Expand per-page source rows, repeating only the unused tail."""
        starts, limits = [], []
        for start, end in regions:
            positions = range(start, end, page_size)
            starts.extend(positions)
            limits.extend([end - 1] * len(positions))
        starts = torch.tensor(starts, device=device, dtype=torch.long)
        limits = torch.tensor(limits, device=device, dtype=torch.long)
        rows = starts.unsqueeze(1) + torch.arange(page_size, device=device)
        return torch.minimum(rows, limits.unsqueeze(1)).reshape(-1)
    return _PAGED_CACHE.get(_paged_cache_key("rows", device, regions, page_size), create)


@torch.inference_mode(False)
def _paged_indexer_metadata(plan, device, heads, dim, topk, ratio):
    """Cache scheduling inputs separately, so they cannot retain an evicted block table."""
    def create() -> tuple:
        """Build native scheduling inputs without a persistent block table reference."""
        cu_q, residuals, _ = _paged_indexer_lengths(plan, device)
        options = {
            "cu_seqlens_q": cu_q,
            "seqused_k": torch.tensor(plan.k_lengths, device=device, dtype=torch.int32),
            "cmp_residual_k": residuals if ratio > 1 else None,
            "layout_q": "TND", "layout_k": "PA_BBND", "mask_mode": 3, "cmp_ratio": ratio,
        }
        metadata = None
        if _uses_indexer_metadata(device):
            metadata = _load_indexer_metadata_op()(
                heads, 1, dim, topk, batch_size=len(plan.segments),
                max_seqlen_q=max(plan.q_lengths), max_seqlen_k=max(plan.k_lengths), **options,
            )
        return options, metadata
    return _PAGED_CACHE.get(_paged_cache_key("metadata", device, plan, heads, dim, topk, ratio), create)


@torch.inference_mode(False)
def _paged_indexer_lengths(plan: CSATNDPlan, device: torch.device) -> tuple:
    """Build only paged inputs; repeated K prefixes have no cumulative length.

    PA_BBND uses individual ``seqused_k`` lengths. Constructing an unused TND
    ``cu_seqlens_k`` can overflow int32 even when the unique K bank is valid.
    """
    def create() -> tuple:
        """Create Q lengths and global output offsets without cumulative K."""
        cu_q = torch.tensor((0, *accumulate(plan.q_lengths)), device=device, dtype=torch.int32)
        residuals = torch.tensor(plan.residuals, device=device, dtype=torch.int32)
        offsets = torch.tensor([start for start, _ in plan.k_ranges], device=device, dtype=torch.int32)
        lengths = torch.tensor(plan.q_lengths, device=device, dtype=torch.int64)
        offsets = torch.repeat_interleave(offsets, lengths, output_size=sum(plan.q_lengths)).unsqueeze(1)
        return cu_q, residuals, offsets
    return _PAGED_CACHE.get(_paged_cache_key("lengths", device, plan), create)


@torch.inference_mode(False)
def _paged_indexer_inputs(plan, pages, device, heads, dim, topk, ratio):
    """Reuse one complete call plan, accounting for every referenced tensor in its budget."""
    def create() -> tuple:
        """Combine page addressing and metadata without caching any K values."""
        options, metadata = _paged_indexer_metadata(plan, device, heads, dim, topk, ratio)
        options = {**options, "block_table": _paged_block_table(pages, device)}
        offsets = _paged_indexer_lengths(plan, device)[2]
        rows = None if pages.zero_copy else _paged_key_rows(pages.regions, pages.page_size, device)
        return options, metadata, offsets, rows
    return _PAGED_CACHE.get(_paged_cache_key("inputs", device, plan, pages, heads, dim, topk, ratio), create)


@torch.no_grad()
def lightning_indexer(
        query: torch.Tensor, key: torch.Tensor, weights: torch.Tensor, ratio: int, topk: int, segments: tuple,
        *, paged_key: bool = False,
) -> torch.Tensor:
    """Select document prefixes with ordinary or shared-page key storage.

    Args:
        query: Local Indexer queries [batch, queries, heads, 128].
        key: Global Indexer key bank [batch, compressed keys, 128].
        weights: Per-query head weights including model scaling.
        ratio: Model compression ratio.
        topk: Selection width, including invalid tail slots.
        segments: Local Q start/end, global document K start/prefix end and
            raw-prefix residual. Packed documents require batch size one.
        paged_key: Share LI V2 physical pages between repeated document prefixes.
            Requires batch size one; affects selection storage only, not owner KL.
            Uses TND queries and PA_BBND keys with 128 rows per physical page.

    Returns:
        Sorted IDs in the supplied key bank [batch, queries, topk], with invalid slots -1.
    """
    if len(segments) == 1 and not paged_key:
        return _bsnd_lightning_indexer(query, key, weights, ratio, topk, segments[0])
    if query.shape[0] != 1:
        raise ValueError("Packed Lightning Indexer requires batch size one")
    plan = csa_tnd_plan(segments, ratio)
    if plan is None:
        return torch.full((*query.shape[:2], topk), -1, device=query.device, dtype=torch.int32)
    if paged_key:
        pages = _paged_key_plan(plan, key.shape[1])
        options, metadata, offsets, rows = _paged_indexer_inputs(
            plan, pages, query.device, query.shape[2], query.shape[3], topk, ratio,
        )
        native_key = pages.key(key, rows)
    else:
        options, metadata = _tnd_indexer_metadata(plan, query.device, query.shape[2], query.shape[3], topk, ratio)
        native_key = plan.key(key).unsqueeze(1)
        offsets = plan.lengths(query.device)[3]
    with torch.autocast(device_type=query.device.type, enabled=False):
        indices, _ = _load_indexer_op()(
            plan.query(query), native_key, plan.query(weights).float(), topk,
            metadata=metadata, **options,
        )
    indices = torch.where(indices[:, 0] >= 0, indices[:, 0] + offsets, key.shape[1])
    indices = sort_key_indices(indices, key.shape[1])
    indices = indices.masked_fill(indices == key.shape[1], -1)
    if plan.covers_queries(query.shape[1]):
        return indices.unsqueeze(0)
    result = torch.full((*query.shape[:2], topk), -1, device=query.device, dtype=torch.int32)
    plan.restore(indices, result)
    return result
