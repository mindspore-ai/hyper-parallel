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
"""Bounded host-only preparation of packed context-parallel batch plans."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import accumulate

from hyper_parallel.distributed.context_parallel.compressed_document import (
    DocumentCompressionPlan, document_compression_plan,
)
from hyper_parallel.distributed.context_parallel.index_key_dispatch import IndexKeyPlan, index_key_plan
from hyper_parallel.distributed.context_parallel.indexer_balance import (
    IndexerRoute, indexer_route, resolve_indexer_balance_block_size, validate_indexer_hybrid_threshold,
)
from hyper_parallel.distributed.context_parallel.sequence_dispatch import (
    SequenceDispatchRoute, document_prefix_route, sliding_window_route,
)
from hyper_parallel.distributed.context_parallel.sequence_pack import prepare_sequence_pack


def _host_integers(values: tuple[int, ...], name: str, minimum: int = 1) -> tuple[int, ...]:
    """Snapshot host input without accepting tensors or mutable nested values."""
    if not isinstance(values, (tuple, list)) or not values:
        raise ValueError(f"{name} must be a nonempty sequence of host integers")
    if any(not isinstance(value, int) or isinstance(value, bool) or value < minimum for value in values):
        raise ValueError(f"{name} must contain host integers >= {minimum}")
    return tuple(values)


@dataclass(frozen=True)
class CPBatchPlanConfig:
    """Static host planning configuration; contains no module, device or group.

    Args:
        size: Sequence-ordered CP group size.
        rank: Rank within that group.
        ratios: Compression ratios used by attention layers, including zero for SWA.
        windows: Raw attention windows used by those layers.
        mode: Indexer selection allocation, independent of owner-side KL.
        block_size: Greedy/hybrid query-block option, defaulting to 2048.
        zigzag_min_length: Hybrid-only original-document threshold.
        index_k_comm: Whether selection requires a compact Index K dispatch plan.
        main_kv_comm: Whether main attention uses a document-prefix dispatch plan.
    """

    size: int
    rank: int
    ratios: tuple[int, ...] = (1, 2)
    windows: tuple[int, ...] = (128,)
    mode: str = "none"
    block_size: int | None = None
    zigzag_min_length: int | None = None
    index_k_comm: str = "all_gather"
    main_kv_comm: str = "all_gather"

    def __post_init__(self) -> None:
        """Normalize static options while rejecting tensors and Boolean dimensions."""
        _host_integers((self.size,), "CP size")
        _host_integers((self.rank,), "CP rank", 0)
        if self.rank >= self.size:
            raise ValueError("Batch planning requires a positive CP size and a valid rank")
        object.__setattr__(self, "ratios", tuple(sorted(set(_host_integers(self.ratios, "ratios", 0)))))
        if any(ratio not in (0, 1, 2) for ratio in self.ratios):
            raise ValueError("V4.1 batch planning supports compression ratios 0, 1 and 2")
        object.__setattr__(self, "windows", tuple(sorted(set(_host_integers(self.windows, "windows")))))
        object.__setattr__(self, "block_size", resolve_indexer_balance_block_size(self.mode, self.block_size))
        validate_indexer_hybrid_threshold(self.mode, self.zigzag_min_length)
        if self.main_kv_comm not in ("all_gather", "all_to_all_v"):
            raise ValueError("main_kv_comm must be all_gather or all_to_all_v")
        if self.index_k_comm not in ("all_gather", "all_to_all_v"):
            raise ValueError("index_k_comm must be 'all_gather' or 'all_to_all_v'")


@dataclass(frozen=True)
class CPRatioPlan:
    """Actual immutable host objects consumed by one compression-ratio family."""

    ratio: int
    main_route: SequenceDispatchRoute | None
    index_key: IndexKeyPlan | None
    worker_route: IndexerRoute | None
    compression: DocumentCompressionPlan | None
    owner_segments: tuple


@dataclass(frozen=True)
class CPBatchPlan:
    """Owned batch snapshot, safe to retain across backward and checkpoint replay."""

    lengths: tuple[int, ...]
    config: CPBatchPlanConfig
    raw_routes: tuple[tuple[int, SequenceDispatchRoute], ...]
    ratio_plans: tuple[CPRatioPlan, ...]

    def validate(self, lengths: tuple[int, ...], size: int, rank: int) -> None:
        """Reject stale document boundaries or a plan for another CP shard.

        Args:
            lengths: Global natural document lengths of the active batch.
            size: Active CP group size.
            rank: Active rank within the CP group.
        """
        if tuple(lengths) != self.lengths or (size, rank) != (self.config.size, self.config.rank):
            raise ValueError("Prepared CP batch does not match document lengths, CP size or rank")

    def for_ratio(self, ratio: int) -> CPRatioPlan:
        """Return prepared objects without invoking a plan builder or an LRU lookup.

        Args:
            ratio: Compression ratio used by the consumer layer.
        """
        for plan in self.ratio_plans:
            if plan.ratio == ratio:
                return plan
        raise ValueError(f"Compression ratio {ratio} was not included in this prepared CP batch")

    def raw_route(self, window: int) -> SequenceDispatchRoute:
        """Return the same-document SWA route for a configured window.

        Args:
            window: Consumer layer's raw attention window in tokens.
        """
        for prepared_window, route in self.raw_routes:
            if prepared_window == window:
                return route
        raise ValueError(f"SWA window {window} was not included in this prepared CP batch")


def _owner_segments(boundaries: tuple[int, ...], local: int, rank: int, ratio: int) -> tuple:
    """Describe original-owner queries using natural-document compressed prefixes."""
    query_start, query_end = rank * local, (rank + 1) * local
    segments, key_start = [], 0
    for start, end in zip(boundaries, boundaries[1:]):
        first, last = max(start, query_start), min(end, query_end)
        if first < last:
            segments.append((first - query_start, last - query_start, key_start,
                             key_start + (last - start) // ratio, (last - start) % ratio))
        key_start += (end - start) // ratio
    return tuple(segments)


def _prepare_ratio(boundaries: tuple[int, ...], lengths: tuple[int, ...],
                   config: CPBatchPlanConfig, ratio: int) -> CPRatioPlan:
    local = boundaries[-1] // config.size
    main = None
    if ratio and config.main_kv_comm == "all_to_all_v":
        main = document_prefix_route(boundaries, local, config.size, config.rank, ratio)
    compression = None
    if ratio > 1 and (local % ratio or any(boundary % ratio for boundary in boundaries)):
        compression = document_compression_plan(boundaries, ratio, local, config.size, config.rank)
    selection = (lengths, config.size, config.rank, ratio, config.mode, config.block_size,
                 config.zigzag_min_length)
    keys = index_key_plan(*selection) if ratio and config.index_k_comm == "all_to_all_v" else None
    worker = indexer_route(*selection) if ratio and config.mode != "none" else None
    return CPRatioPlan(ratio, main, keys, worker, compression,
                       _owner_segments(boundaries, local, config.rank, max(1, ratio)))


def prepare_cp_batch(lengths: tuple[int, ...], config: CPBatchPlanConfig) -> CPBatchPlan:
    """Build concrete host routes for a batch without touching a device or collective.

    Args:
        lengths: Natural global document lengths supplied by packing/collation.
        config: Immutable static context-parallel planning configuration.

    Returns:
        A batch-owned plan with concrete worker and Index K routes. Other bank
        consumers reuse the host route and pack caches warmed during preparation.
    """
    lengths = _host_integers(lengths, "lengths")
    if not isinstance(config, CPBatchPlanConfig):
        raise TypeError("config must be CPBatchPlanConfig")
    if sum(lengths) % config.size:
        raise ValueError("Global packed length must divide evenly into contiguous CP query shards")
    return _prepared_cp_batch(lengths, config)


@lru_cache(maxsize=32)
def _prepared_cp_batch(lengths: tuple[int, ...], config: CPBatchPlanConfig) -> CPBatchPlan:
    """Reuse complete immutable host plans without retaining payloads or device storage."""
    boundaries = (0, *accumulate(lengths))
    local = boundaries[-1] // config.size
    raw = tuple((window, sliding_window_route(boundaries, local, config.size, config.rank, window))
                for window in config.windows)
    ratios = tuple(_prepare_ratio(boundaries, lengths, config, ratio) for ratio in config.ratios)
    for _, route in raw:
        prepare_sequence_pack(route)
    for plan in ratios:
        for route in (plan.main_route, None if plan.index_key is None else plan.index_key.dispatch):
            if route is not None:
                prepare_sequence_pack(route)
    return CPBatchPlan(lengths, config, raw, ratios)
