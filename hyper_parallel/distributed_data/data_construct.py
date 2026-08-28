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
"""Single-card local-batch sources used by distributed orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol, Sequence

from hyper_parallel.distributed_data.schema import BatchPlan, LocalBatchMeta


class LocalBatchSource(Protocol):
    """Common lifecycle and size contract for a single-card data source."""

    @property
    def has_sidecar(self) -> bool:
        """Return whether metadata is available before payload materialization."""

    def __len__(self) -> int:
        """Return globally addressable local-batch entries."""

    def close(self) -> None:
        """Release resources owned by the single-card data source."""


@dataclass(frozen=True)
class LoadedLocalBatch:
    """One materialized local batch and metadata derived after processing."""

    metadata: LocalBatchMeta
    data: Any


class SidecarLocalBatchSource:
    """Expose sidecar metadata and fetch planned local batches by ID."""

    def __init__(
        self,
        metadata: Sequence[LocalBatchMeta],
        fetch_fn: Callable[[int | str], Any],
        *,
        close_fn: Callable[[], None] | None = None,
    ) -> None:
        """Initialize a shared, randomly addressable sidecar source.

        Args:
            metadata: Lightweight descriptors for complete local batches.
            fetch_fn: Single-card callback that materializes one local batch ID.
            close_fn: Optional callback that releases single-card loader resources.
        """
        if any(not isinstance(item, LocalBatchMeta) for item in metadata):
            raise ValueError("metadata must contain only LocalBatchMeta entries.")
        if not callable(fetch_fn):
            raise ValueError("fetch_fn must be callable.")
        if close_fn is not None and not callable(close_fn):
            raise ValueError("close_fn must be callable or None.")
        self._metadata = metadata
        self._fetch_fn = fetch_fn
        self._close_fn = close_fn
        self._closed = False

    @property
    def has_sidecar(self) -> bool:
        """Return that metadata is available before local-batch reads."""
        return True

    def __len__(self) -> int:
        """Return global sidecar entry count."""
        return len(self._metadata)

    def get_metadata(self, index: int) -> LocalBatchMeta:
        """Return one sidecar entry by global source index."""
        self._raise_if_closed()
        if index < 0 or index >= len(self):
            raise ValueError(f"Metadata index must be in [0, {len(self)}), but got {index}.")
        return self._metadata[index]

    def fetch(self, metadata: LocalBatchMeta) -> Any:
        """Materialize one planned local batch through the single-card source."""
        self._raise_if_closed()
        return self._fetch_fn(metadata.local_batch_id)

    def close(self) -> None:
        """Release optional single-card source resources."""
        if self._closed:
            return
        if self._close_fn is not None:
            self._close_fn()
        self._closed = True

    def _raise_if_closed(self) -> None:
        if self._closed:
            raise ValueError("Cannot use a closed SidecarLocalBatchSource.")


class OnlineLocalBatchSource:
    """Read and process map-style local batches before deriving metadata."""

    def __init__(
        self,
        local_batches: Any,
        metadata_fn: Callable[[Any, int], LocalBatchMeta],
        *,
        close_fn: Callable[[], None] | None = None,
    ) -> None:
        """Initialize an online single-card source.

        Args:
            local_batches: Map-style source whose item is one complete local batch.
            metadata_fn: Callback run after the local batch has been materialized.
            close_fn: Optional callback that releases single-card loader resources.
        """
        if not hasattr(local_batches, "__len__") or not hasattr(local_batches, "__getitem__"):
            raise ValueError("local_batches must implement __len__ and __getitem__.")
        if not callable(metadata_fn):
            raise ValueError("metadata_fn must be callable.")
        if close_fn is not None and not callable(close_fn):
            raise ValueError("close_fn must be callable or None.")
        self._local_batches = local_batches
        self._metadata_fn = metadata_fn
        self._close_fn = close_fn
        self._closed = False

    @property
    def has_sidecar(self) -> bool:
        """Return that metadata is derived only after online processing."""
        return False

    def __len__(self) -> int:
        """Return global local-batch entry count."""
        return len(self._local_batches)

    def load(self, source_index: int) -> LoadedLocalBatch:
        """Materialize one local batch and derive its validated metadata."""
        self._raise_if_closed()
        if source_index < 0 or source_index >= len(self):
            raise ValueError(f"Source index must be in [0, {len(self)}), but got {source_index}.")
        data = self._local_batches[source_index]
        metadata = self._metadata_fn(data, source_index)
        if not isinstance(metadata, LocalBatchMeta):
            raise ValueError(f"metadata_fn must return LocalBatchMeta, but got {type(metadata)}.")
        return LoadedLocalBatch(metadata, data)

    def close(self) -> None:
        """Release optional single-card source resources."""
        if self._closed:
            return
        if self._close_fn is not None:
            self._close_fn()
        self._closed = True

    def _raise_if_closed(self) -> None:
        if self._closed:
            raise ValueError("Cannot use a closed OnlineLocalBatchSource.")


class LocalBatchMetadataView:
    """Expose one deterministic DP shard of a shared sidecar source."""

    def __init__(
        self,
        source: SidecarLocalBatchSource,
        shard_rank: int,
        num_shards: int,
        *,
        max_entries: int,
    ) -> None:
        """Initialize a strided sidecar metadata view."""
        _validate_shard(shard_rank, num_shards, max_entries)
        self._source = source
        self._shard_rank = shard_rank
        self._num_shards = num_shards
        self._max_entries = max_entries

    def __len__(self) -> int:
        """Return complete-step entries owned by this data rank."""
        return self._max_entries

    def get(self, index: int) -> LocalBatchMeta:
        """Return metadata at one per-owner local-batch offset."""
        if index < 0 or index >= len(self):
            raise ValueError(f"Metadata index must be in [0, {len(self)}), but got {index}.")
        global_index = self._shard_rank + index * self._num_shards
        return self._source.get_metadata(global_index)


class OnlineLocalBatchView:
    """Materialize one deterministic DP shard of an online source."""

    def __init__(
        self,
        source: OnlineLocalBatchSource,
        shard_rank: int,
        num_shards: int,
        *,
        max_entries: int,
    ) -> None:
        """Initialize a strided online local-batch view."""
        _validate_shard(shard_rank, num_shards, max_entries)
        self._source = source
        self._shard_rank = shard_rank
        self._num_shards = num_shards
        self._max_entries = max_entries

    def __len__(self) -> int:
        """Return complete-step entries owned by this data rank."""
        return self._max_entries

    def get_range(self, start: int, end: int) -> tuple[LoadedLocalBatch, ...]:
        """Load an ordered range of per-owner local-batch offsets."""
        if start < 0 or end < start or end > len(self):
            raise ValueError(f"Online local-batch range must satisfy 0 <= start <= end <= {len(self)}.")
        return tuple(
            self._source.load(self._shard_rank + index * self._num_shards)
            for index in range(start, end)
        )


class SidecarLocalBatchFetcher:
    """Fetch one complete local batch assigned by a sidecar plan."""

    def __init__(self, source: SidecarLocalBatchSource) -> None:
        """Initialize target-rank direct fetching from the single-card source."""
        self._source = source

    def fetch(self, plan: BatchPlan, data_rank: int, micro_batch_index: int) -> Any:
        """Fetch the local batch assigned to one execution slot."""
        planned = plan.local_batch_for(data_rank, micro_batch_index)
        return self._source.fetch(planned.meta)


def _validate_shard(shard_rank: int, num_shards: int, max_entries: int) -> None:
    if num_shards < 1:
        raise ValueError(f"num_shards must be positive, but got {num_shards}.")
    if shard_rank < 0 or shard_rank >= num_shards:
        raise ValueError(f"shard_rank must be in [0, {num_shards}), but got {shard_rank}.")
    if not isinstance(max_entries, int) or isinstance(max_entries, bool) or max_entries < 0:
        raise ValueError(f"max_entries must be a non-negative integer, but got {max_entries!r}.")
