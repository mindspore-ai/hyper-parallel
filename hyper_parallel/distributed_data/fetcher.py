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
"""Map-style metadata access and sample fetching."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol, Sequence

from hyper_parallel.distributed_data.schema import BatchPlan, SampleMeta


class MetadataSource(Protocol):
    """Random-access lightweight metadata source for one data rank."""

    def __len__(self) -> int:
        """Return local metadata entry count."""

    def get(self, index: int) -> SampleMeta:
        """Return metadata at one local cursor position."""


class SampleFetcher(Protocol):
    """Fetch one heavyweight sample from lightweight metadata."""

    def fetch(self, metadata: SampleMeta) -> Any:
        """Read and transform one sample."""


@dataclass(frozen=True)
class LoadedSample:
    """One owner-loaded raw sample and its online metadata."""

    metadata: SampleMeta
    data: Any


class OnlineSampleSource(Protocol):
    """Load raw samples and derive metadata on one data owner."""

    def __len__(self) -> int:
        """Return the local candidate count."""

    def get(self, index: int) -> LoadedSample:
        """Load one local raw sample and derive its metadata."""


class StridedMetadataSource:
    """Expose one deterministic shard of a shared metadata sidecar."""

    def __init__(
        self,
        metadata: Sequence[SampleMeta],
        shard_rank: int,
        num_shards: int,
        *,
        max_entries: int | None = None,
    ) -> None:
        """Initialize a strided view over a shared metadata sequence."""
        if num_shards < 1:
            raise ValueError(f"num_shards must be positive, but got {num_shards}.")
        if shard_rank < 0 or shard_rank >= num_shards:
            raise ValueError(f"shard_rank must be in [0, {num_shards}), but got {shard_rank}.")
        if max_entries is not None and (
            not isinstance(max_entries, int) or isinstance(max_entries, bool) or max_entries < 0
        ):
            raise ValueError(f"max_entries must be non-negative or None, but got {max_entries}.")
        self._metadata = metadata
        self._shard_rank = shard_rank
        self._num_shards = num_shards
        self._max_entries = max_entries

    def __len__(self) -> int:
        """Return metadata entries owned by this deterministic shard."""
        remaining = len(self._metadata) - self._shard_rank
        if remaining <= 0:
            return 0
        local_entries = (remaining + self._num_shards - 1) // self._num_shards
        return min(local_entries, self._max_entries) if self._max_entries is not None else local_entries

    def get(self, index: int) -> SampleMeta:
        """Return metadata at one local cursor position."""
        if index < 0 or index >= len(self):
            raise ValueError(f"Metadata index must be in [0, {len(self)}), but got {index}.")
        return self._metadata[self._shard_rank + index * self._num_shards]


class StridedOnlineSampleSource:
    """Load one deterministic DP shard and derive metadata online."""

    def __init__(
        self,
        dataset: Any,
        metadata_fn: Callable[[Any, int], SampleMeta],
        shard_rank: int,
        num_shards: int,
        *,
        max_entries: int | None = None,
    ) -> None:
        """Initialize online loading over a shared map-style dataset."""
        if not hasattr(dataset, "__len__") or not hasattr(dataset, "__getitem__"):
            raise ValueError("Online metadata requires a map-style dataset implementing __len__ and __getitem__.")
        if not callable(metadata_fn):
            raise ValueError("metadata_fn must be callable.")
        if num_shards < 1:
            raise ValueError(f"num_shards must be positive, but got {num_shards}.")
        if shard_rank < 0 or shard_rank >= num_shards:
            raise ValueError(f"shard_rank must be in [0, {num_shards}), but got {shard_rank}.")
        if max_entries is not None and (
            not isinstance(max_entries, int) or isinstance(max_entries, bool) or max_entries < 0
        ):
            raise ValueError(f"max_entries must be non-negative or None, but got {max_entries}.")
        self._dataset = dataset
        self._metadata_fn = metadata_fn
        self._shard_rank = shard_rank
        self._num_shards = num_shards
        self._max_entries = max_entries

    def __len__(self) -> int:
        """Return raw samples owned by this deterministic DP shard."""
        remaining = len(self._dataset) - self._shard_rank
        if remaining <= 0:
            return 0
        local_entries = (remaining + self._num_shards - 1) // self._num_shards
        return min(local_entries, self._max_entries) if self._max_entries is not None else local_entries

    def get(self, index: int) -> LoadedSample:
        """Load one raw sample and derive metadata without rereading it."""
        if index < 0 or index >= len(self):
            raise ValueError(f"Online sample index must be in [0, {len(self)}), but got {index}.")
        sample_id = self._shard_rank + index * self._num_shards
        sample = self._dataset[sample_id]
        metadata = self._metadata_fn(sample, sample_id)
        if not isinstance(metadata, SampleMeta):
            raise ValueError(f"metadata_fn must return SampleMeta, but got {type(metadata)}.")
        if metadata.sample_id != sample_id:
            raise ValueError(
                f"metadata_fn must preserve sample_id {sample_id!r}, but returned {metadata.sample_id!r}."
            )
        return LoadedSample(metadata, sample)


class MapDatasetFetcher:
    """Fetch samples by indexing a map-style dataset with ``sample_id``."""

    def __init__(self, dataset: Any) -> None:
        """Initialize the fetcher with a map-style dataset."""
        if not hasattr(dataset, "__getitem__"):
            raise ValueError("MapDatasetFetcher requires a dataset implementing __getitem__.")
        self._dataset = dataset

    def fetch(self, metadata: SampleMeta) -> Any:
        """Read one map-style dataset entry."""
        return self._dataset[metadata.sample_id]


def _identity_collate(samples: list[Any]) -> list[Any]:
    return samples


def _pin_memory_batch(batch: Any) -> Any:
    """Recursively copy tensor-like batch leaves into pinned Host memory."""
    if isinstance(batch, dict):
        return {key: _pin_memory_batch(value) for key, value in batch.items()}
    if isinstance(batch, list):
        return [_pin_memory_batch(value) for value in batch]
    if isinstance(batch, tuple):
        return tuple(_pin_memory_batch(value) for value in batch)
    pin_memory = getattr(batch, "pin_memory", None)
    return pin_memory() if callable(pin_memory) else batch


class MicroBatchFetcher:
    """Fetch and collate one planned data-rank microbatch."""

    def __init__(
        self,
        sample_fetcher: SampleFetcher,
        collate_fn: Callable[[list[Any]], Any] | None = None,
    ) -> None:
        """Initialize sample fetching and owner-side collation."""
        self._sample_fetcher = sample_fetcher
        self._collate_fn = collate_fn or _identity_collate

    def fetch(self, plan: BatchPlan, data_rank: int, micro_batch_index: int) -> Any:
        """Fetch only samples assigned to the requested execution slot."""
        planned_samples = plan.samples_for(data_rank, micro_batch_index)
        if len(planned_samples) != plan.micro_batch_size:
            raise ValueError(
                f"Plan slot ({data_rank}, {micro_batch_index}) expected {plan.micro_batch_size} samples, "
                f"but got {len(planned_samples)}."
            )
        samples = [self._sample_fetcher.fetch(planned.meta) for planned in planned_samples]
        return self._collate_fn(samples)

    def collate(self, samples: list[Any]) -> Any:
        """Collate already-loaded samples after online redistribution."""
        return self._collate_fn(samples)
