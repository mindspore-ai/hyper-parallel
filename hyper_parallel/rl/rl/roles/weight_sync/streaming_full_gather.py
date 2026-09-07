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
"""Bounded full-gather planning and transport-independent execution."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from hashlib import sha256
import json
from math import prod
from typing import Any, Optional, Union

from rl.roles.weight_sync.layout import (
    DestinationTensorLayout,
    SourceTensorLayout,
    TensorRegion,
    bucketize_entries,
    tile_region as _tile_region,
    intersect_regions,
)
from rl.roles.weight_sync.tensor_ops import local_tensor


@dataclass(frozen=True)
class StreamingGatherContribution:
    """One source-rank slice contributing to a canonical gathered fragment."""

    source_rank: int
    source_name: str
    source_starts: tuple[int, ...]
    fragment_starts: tuple[int, ...]
    lengths: tuple[int, ...]

    @property
    def numel(self) -> int:
        """Return the number of values supplied by this source rank."""
        return prod(self.lengths)


@dataclass(frozen=True)
class StreamingGatherFragment:
    """One bounded destination fragment assembled through a gather barrier."""

    name: str
    dtype_name: str
    element_size: int
    global_shape: tuple[int, ...]
    target_tp_rank: int
    destination_name: str
    canonical_starts: tuple[int, ...]
    lengths: tuple[int, ...]
    destination_starts: tuple[int, ...]
    destination_permutation: tuple[int, ...]
    destination_dtype_name: str
    destination_element_size: int
    contributions: tuple[StreamingGatherContribution, ...]
    buffer_offset: int = 0

    @property
    def numel(self) -> int:
        """Return the number of canonical values in this fragment."""
        return prod(self.lengths)

    @property
    def num_bytes(self) -> int:
        """Return the source-format byte count of this fragment."""
        return self.numel * self.element_size

    @property
    def destination_lengths(self) -> tuple[int, ...]:
        """Return fragment lengths in physical destination dimension order."""
        return tuple(self.lengths[axis] for axis in self.destination_permutation)

    def with_buffer_offset(self, offset: int) -> "StreamingGatherFragment":
        """Return the fragment assigned to one packed-buffer offset."""
        return replace(self, buffer_offset=int(offset))

    def worker_metadata(self) -> dict[str, Any]:
        """Serialize the destination slice without source-rank details."""
        return {
            "name": self.destination_name,
            "canonical_name": self.name,
            "dtype_name": self.dtype_name,
            "element_size": self.element_size,
            "destination_dtype_name": self.destination_dtype_name,
            "destination_element_size": self.destination_element_size,
            "canonical_starts": list(self.canonical_starts),
            "destination_starts": list(self.destination_starts),
            "lengths": list(self.lengths),
            "destination_lengths": list(self.destination_lengths),
            "destination_permutation": list(self.destination_permutation),
            "buffer_offset": self.buffer_offset,
            "num_bytes": self.num_bytes,
        }


@dataclass(frozen=True)
class StreamingGatherBucket:
    """One bounded canonical gather and destination update unit."""

    target_tp_rank: int
    entries: tuple[StreamingGatherFragment, ...]
    total_bytes: int

    def worker_metadata(self) -> dict[str, Any]:
        """Serialize one streamed receive-and-scatter operation."""
        return {
            "target_tp_rank": self.target_tp_rank,
            "total_bytes": self.total_bytes,
            "entries": [entry.worker_metadata() for entry in self.entries],
        }


@dataclass(frozen=True)
class StreamingTensorAlias:
    """Declare a tied contract name whose target storage is sent once."""

    alias_name: str
    target_name: str


@dataclass(frozen=True)
class StreamingFullGatherPlan:
    """A target-TP indexed bounded full-gather plan."""

    destination_tp_size: int
    bucket_size_bytes: int
    buckets: Mapping[int, tuple[StreamingGatherBucket, ...]]
    aliases: tuple[StreamingTensorAlias, ...] = ()

    def for_target(self, tp_rank: int) -> tuple[StreamingGatherBucket, ...]:
        """Return all ordered buckets for one destination TP rank."""
        return self.buckets.get(int(tp_rank), ())

    @property
    def bucket_count(self) -> int:
        """Return the total number of streamed buckets."""
        return sum(len(buckets) for buckets in self.buckets.values())

    @property
    def fragment_count(self) -> int:
        """Return the total number of canonical fragments."""
        return sum(
            len(bucket.entries)
            for buckets in self.buckets.values()
            for bucket in buckets
        )

    @property
    def total_bytes(self) -> int:
        """Return source bytes sent across all TP destinations."""
        return sum(
            entry.num_bytes
            for buckets in self.buckets.values()
            for bucket in buckets
            for entry in bucket.entries
        )





def _bucketize(
    target_tp_rank: int, entries: Sequence[StreamingGatherFragment], bucket_size_bytes: int,
) -> tuple[StreamingGatherBucket, ...]:
    return tuple(
        StreamingGatherBucket(target_tp_rank, items, size)
        for items, size in bucketize_entries(entries, bucket_size_bytes)
    )


def _layout_signature(layout: Union[SourceTensorLayout, DestinationTensorLayout]) -> tuple[Any, ...]:
    return (
        layout.dtype_name,
        layout.element_size,
        layout.global_shape,
    )


def _source_alias_signature(
    layouts: Sequence[SourceTensorLayout],
) -> tuple[tuple[Any, ...], ...]:
    return tuple(
        sorted(
            (
                layout.source_rank,
                layout.dtype_name,
                layout.element_size,
                layout.global_shape,
                layout.region.starts,
                layout.region.lengths,
                layout.local_starts,
            )
            for layout in layouts
        )
    )


def _destination_alias_signature(
    layouts: Sequence[DestinationTensorLayout],
) -> tuple[tuple[Any, ...], ...]:
    return tuple(
        sorted(
            (
                layout.tp_rank,
                layout.tp_size,
                layout.dtype_name,
                layout.element_size,
                layout.global_shape,
                layout.placement,
                layout.shard_dim,
                layout.region.starts,
                layout.region.lengths,
                layout.local_starts,
                layout.physical_permutation,
                layout.accepted_source_dtypes,
            )
            for layout in layouts
        )
    )


def _normalize_aliases(
    sources_by_name: dict[str, list[SourceTensorLayout]],
    destinations_by_name: dict[str, list[DestinationTensorLayout]],
    aliases: Mapping[str, str],
) -> tuple[StreamingTensorAlias, ...]:
    normalized = []
    for alias_name, target_name in sorted(aliases.items()):
        alias_name = str(alias_name)
        target_name = str(target_name)
        if alias_name == target_name:
            raise ValueError(f"Streaming full-gather alias {alias_name!r} targets itself")
        if target_name in aliases:
            raise ValueError(
                "Streaming full-gather aliases must point directly to a transferred "
                f"tensor, got {alias_name!r} -> {target_name!r}"
            )
        target_sources = sources_by_name.get(target_name)
        target_destinations = destinations_by_name.get(target_name)
        if not target_sources or not target_destinations:
            raise ValueError(
                f"Streaming full-gather alias target {target_name!r} is not complete"
            )
        alias_sources = sources_by_name.get(alias_name, ())
        alias_destinations = destinations_by_name.get(alias_name, ())
        if alias_sources and (
            _source_alias_signature(alias_sources)
            != _source_alias_signature(target_sources)
        ):
            raise ValueError(
                f"Streaming full-gather source alias {alias_name!r} differs from {target_name!r}"
            )
        if alias_destinations and (
            _destination_alias_signature(alias_destinations)
            != _destination_alias_signature(target_destinations)
        ):
            raise ValueError(
                f"Streaming full-gather destination alias {alias_name!r} differs from {target_name!r}"
            )
        sources_by_name.pop(alias_name, None)
        destinations_by_name.pop(alias_name, None)
        normalized.append(StreamingTensorAlias(alias_name, target_name))
    return tuple(normalized)


def _build_fragment(
    sources: Sequence[SourceTensorLayout],
    destination: DestinationTensorLayout,
    region: TensorRegion,
) -> StreamingGatherFragment:
    contributions = []
    for source in sources:
        intersection = intersect_regions(source.region, region)
        if intersection is None:
            continue
        contributions.append(
            StreamingGatherContribution(
                source_rank=source.source_rank,
                source_name=source.source_key,
                source_starts=tuple(
                    local_start + start - source_start
                    for local_start, start, source_start in zip(
                        source.local_starts,
                        intersection.starts,
                        source.region.starts,
                    )
                ),
                fragment_starts=tuple(
                    start - fragment_start
                    for start, fragment_start in zip(
                        intersection.starts,
                        region.starts,
                    )
                ),
                lengths=intersection.lengths,
            )
        )
    covered = sum(contribution.numel for contribution in contributions)
    if covered != region.numel:
        raise ValueError(
            f"Streaming full-gather sources cover {covered} values for {destination.name!r}, "
            f"expected {region.numel} at starts={region.starts}"
        )
    contribution_regions = [
        TensorRegion(contribution.fragment_starts, contribution.lengths)
        for contribution in contributions
    ]
    for index, left in enumerate(contribution_regions):
        for right in contribution_regions[index + 1:]:
            if intersect_regions(left, right) is not None:
                raise ValueError(
                    f"Streaming full-gather sources overlap for {destination.name!r} "
                    f"at starts={region.starts}"
                )
    canonical_offsets = tuple(
        start - base for start, base in zip(region.starts, destination.region.starts)
    )
    destination_starts = tuple(
        local_start + canonical_offsets[axis]
        for local_start, axis in zip(
            destination.local_starts,
            destination.physical_permutation,
        )
    )
    return StreamingGatherFragment(
        name=destination.name,
        dtype_name=sources[0].dtype_name,
        element_size=sources[0].element_size,
        global_shape=destination.global_shape,
        target_tp_rank=destination.tp_rank,
        destination_name=destination.target_name,
        canonical_starts=region.starts,
        lengths=region.lengths,
        destination_starts=destination_starts,
        destination_permutation=destination.physical_permutation,
        destination_dtype_name=destination.dtype_name,
        destination_element_size=destination.element_size,
        contributions=tuple(
            sorted(
                contributions,
                key=lambda contribution: (
                    contribution.source_rank,
                    contribution.source_name,
                    contribution.fragment_starts,
                ),
            )
        ),
    )


def _group_layouts(
    sources: Sequence[SourceTensorLayout],
    destinations: Sequence[DestinationTensorLayout],
    aliases: Mapping[str, str],
) -> tuple[
    dict[str, list[SourceTensorLayout]],
    dict[str, list[DestinationTensorLayout]],
    tuple[StreamingTensorAlias, ...],
    int,
]:
    sources_by_name: dict[str, list[SourceTensorLayout]] = {}
    destinations_by_name: dict[str, list[DestinationTensorLayout]] = {}
    for source in sources:
        sources_by_name.setdefault(source.name, []).append(source)
    for destination in destinations:
        destinations_by_name.setdefault(destination.name, []).append(destination)
    normalized_aliases = _normalize_aliases(
        sources_by_name,
        destinations_by_name,
        aliases,
    )
    if set(sources_by_name) != set(destinations_by_name):
        raise ValueError(
            "Streaming full-gather source/destination parameter mismatch: "
            f"source_only={sorted(set(sources_by_name) - set(destinations_by_name))}, "
            f"destination_only={sorted(set(destinations_by_name) - set(sources_by_name))}"
        )
    tp_sizes = {
        destination.tp_size
        for named_destinations in destinations_by_name.values()
        for destination in named_destinations
    }
    if len(tp_sizes) != 1:
        raise ValueError(
            f"Streaming full-gather destination TP sizes differ: {sorted(tp_sizes)}"
        )
    destination_tp_size = tp_sizes.pop()
    if destination_tp_size <= 0:
        raise ValueError("Streaming full-gather destination TP size must be positive")
    return (
        sources_by_name,
        destinations_by_name,
        normalized_aliases,
        destination_tp_size,
    )


def _build_named_fragments(
    name: str,
    sources: Sequence[SourceTensorLayout],
    destinations: Sequence[DestinationTensorLayout],
    *,
    destination_tp_size: int,
    bucket_size_bytes: int,
) -> tuple[StreamingGatherFragment, ...]:
    named_sources = sorted(
        sources,
        key=lambda source: (source.source_rank, source.region.starts),
    )
    source_signature = _layout_signature(named_sources[0])
    if any(_layout_signature(source) != source_signature for source in named_sources):
        raise ValueError(f"Streaming full-gather source metadata differs for {name!r}")
    destination_ranks = [destination.tp_rank for destination in destinations]
    if sorted(destination_ranks) != list(range(destination_tp_size)):
        raise ValueError(
            f"Streaming full-gather destination TP ranks differ for {name!r}: "
            f"actual={sorted(destination_ranks)}, expected={list(range(destination_tp_size))}"
        )
    fragments = []
    for destination in sorted(destinations, key=lambda value: value.tp_rank):
        dtype_compatible = (
            named_sources[0].dtype_name == destination.dtype_name
            and named_sources[0].element_size == destination.element_size
        ) or named_sources[0].dtype_name in destination.accepted_source_dtypes
        if (
            named_sources[0].global_shape != destination.global_shape
            or not dtype_compatible
        ):
            raise ValueError(
                f"Streaming full-gather tensor contract mismatch for {name!r}: "
                f"source={(named_sources[0].global_shape, named_sources[0].dtype_name)}, "
                f"destination={(destination.global_shape, destination.dtype_name)}"
            )
        fragments.extend(
            _build_fragment(named_sources, destination, region)
            for region in _tile_region(
                destination.region,
                element_size=named_sources[0].element_size,
                bucket_size_bytes=bucket_size_bytes,
            )
        )
    return tuple(fragments)


def build_streaming_full_gather_plan(
    sources: Sequence[SourceTensorLayout],
    destinations: Sequence[DestinationTensorLayout],
    *,
    source_world_size: int,
    bucket_size_bytes: int,
    aliases: Optional[Mapping[str, str]] = None,
) -> StreamingFullGatherPlan:
    """Build an independent gather-first plan with bounded destination fragments.

    Args:
        sources: Canonical FSDP source regions from every Trainer rank.
        destinations: Canonical rollout TP destination regions.
        source_world_size: Number of Trainer ranks participating in each gather.
        bucket_size_bytes: Hard maximum for each gathered and packed buffer.
        aliases: Optional tied contract names mapped to the one transferred owner.

    Returns:
        A deterministic target-TP indexed streaming plan.

    Raises:
        ValueError: If layouts, aliases, coverage, or sizes are invalid.
    """
    if source_world_size <= 0:
        raise ValueError("Streaming full-gather source_world_size must be positive")
    if bucket_size_bytes <= 0:
        raise ValueError("Streaming full-gather bucket_size_bytes must be positive")
    if not sources or not destinations:
        raise ValueError("Streaming full-gather requires source and destination layouts")
    if any(
        source.source_rank < 0 or source.source_rank >= source_world_size
        for source in sources
    ):
        raise ValueError("Streaming full-gather source rank is outside source_world_size")
    (
        sources_by_name,
        destinations_by_name,
        normalized_aliases,
        destination_tp_size,
    ) = _group_layouts(sources, destinations, aliases or {})
    entries_by_target: dict[int, list[StreamingGatherFragment]] = {
        tp_rank: [] for tp_rank in range(destination_tp_size)
    }
    for name in sorted(sources_by_name):
        for fragment in _build_named_fragments(
            name,
            sources_by_name[name],
            destinations_by_name[name],
            destination_tp_size=destination_tp_size,
            bucket_size_bytes=bucket_size_bytes,
        ):
            entries_by_target[fragment.target_tp_rank].append(fragment)
    expected_targets = set(range(destination_tp_size))
    if set(entries_by_target) != expected_targets or any(
        not entries_by_target[tp_rank] for tp_rank in expected_targets
    ):
        raise ValueError("Streaming full-gather destination TP coverage is incomplete")
    buckets = {
        tp_rank: _bucketize(tp_rank, entries, bucket_size_bytes)
        for tp_rank, entries in entries_by_target.items()
    }
    return StreamingFullGatherPlan(
        destination_tp_size=destination_tp_size,
        bucket_size_bytes=bucket_size_bytes,
        buckets=buckets,
        aliases=normalized_aliases,
    )


@dataclass(frozen=True)
class MaterializedStreamingContribution:
    """Pair one planned source slice with its materialized local tensor."""

    layout: StreamingGatherContribution
    tensor: Any


def extract_streaming_contributions(
    fragment: StreamingGatherFragment,
    source_rank: int,
    state_dict: Mapping[str, Any],
) -> tuple[MaterializedStreamingContribution, ...]:
    """Extract this rank's local tensors for one canonical fragment.

    Args:
        fragment: Planned canonical destination fragment.
        source_rank: Rank owning ``state_dict``.
        state_dict: FSDP-local values keyed by physical Trainer names.

    Returns:
        Materialized local contributions owned by ``source_rank``.

    Raises:
        ValueError: If a planned source tensor or slice is unavailable.
    """
    materialized = []
    for contribution in fragment.contributions:
        if contribution.source_rank != source_rank:
            continue
        value = state_dict.get(contribution.source_name)
        if value is None:
            raise ValueError(
                f"Streaming full-gather source {contribution.source_name!r} is missing "
                f"on rank {source_rank}"
            )
        source_tensor = local_tensor(value)
        source_slice = tuple(
            slice(start, start + length)
            for start, length in zip(
                contribution.source_starts,
                contribution.lengths,
            )
        )
        tensor = source_tensor[source_slice].detach().contiguous()
        if tuple(int(size) for size in tensor.shape) != contribution.lengths:
            raise ValueError(
                f"Streaming full-gather source slice {contribution.source_name!r} has "
                f"shape={tuple(tensor.shape)}, expected={contribution.lengths}"
            )
        materialized.append(MaterializedStreamingContribution(contribution, tensor))
    return tuple(materialized)


def assemble_streaming_fragment(
    fragment: StreamingGatherFragment,
    contributions: Sequence[MaterializedStreamingContribution],
) -> Any:
    """Assemble gathered source contributions in canonical dimension order.

    Args:
        fragment: Fragment contract shared by all ranks.
        contributions: Gathered local slices from its declared source ranks.

    Returns:
        One contiguous tensor in canonical dimension order.

    Raises:
        ValueError: If contribution identity, shape, dtype, or coverage differs.
    """
    # Torch is optional outside the Torch RL runtime.
    import torch  # pylint: disable=C0415,forbidden-backend-import

    by_layout: dict[StreamingGatherContribution, Any] = {}
    for contribution in contributions:
        if contribution.layout in by_layout:
            raise ValueError(
                f"Streaming full-gather contribution is duplicated: {contribution.layout}"
            )
        by_layout[contribution.layout] = contribution.tensor
    expected = set(fragment.contributions)
    if set(by_layout) != expected:
        raise ValueError(
            "Streaming full-gather contributions differ from the fragment plan: "
            f"missing={sorted(expected - set(by_layout), key=repr)}, "
            f"unexpected={sorted(set(by_layout) - expected, key=repr)}"
        )
    first = by_layout[fragment.contributions[0]]
    assembled = torch.empty(fragment.lengths, dtype=first.dtype, device=first.device)
    copied = 0
    with torch.no_grad():
        for layout in fragment.contributions:
            tensor = by_layout[layout]
            if (
                str(tensor.dtype).rsplit(".", maxsplit=1)[-1] != fragment.dtype_name
                or int(tensor.element_size()) != fragment.element_size
                or tuple(int(size) for size in tensor.shape) != layout.lengths
            ):
                raise ValueError(
                    f"Streaming full-gather contribution differs from plan: {layout}"
                )
            target_slice = tuple(
                slice(start, start + length)
                for start, length in zip(layout.fragment_starts, layout.lengths)
            )
            assembled[target_slice].copy_(tensor)
            copied += layout.numel
    if copied != fragment.numel:
        raise ValueError(
            f"Streaming full-gather copied {copied} values, expected {fragment.numel}"
        )
    return assembled


@dataclass(frozen=True)
class StreamingMaterializedBucket:
    """One bounded packed payload and its independently measured allocations."""

    value: Any
    gathered_bytes: int
    packed_bytes: int


def pack_streaming_bucket(
    bucket: StreamingGatherBucket,
    fragment_tensors: Sequence[Any],
) -> StreamingMaterializedBucket:
    """Pack ordered assembled fragments into one bounded byte tensor.

    Args:
        bucket: Planned bucket including aligned offsets.
        fragment_tensors: Materialized tensors aligned with ``bucket.entries``.

    Returns:
        A packed payload with gathered and packed allocation counters.

    Raises:
        ValueError: If tensor count, shape, dtype, or bytes differ from the plan.
    """
    # Torch is optional outside the Torch RL runtime.
    import torch  # pylint: disable=C0415,forbidden-backend-import

    if len(fragment_tensors) != len(bucket.entries):
        raise ValueError(
            "Streaming full-gather materialized fragment count differs from bucket: "
            f"actual={len(fragment_tensors)}, expected={len(bucket.entries)}"
        )
    if not fragment_tensors:
        raise ValueError("Streaming full-gather cannot pack an empty bucket")
    packed = torch.empty(
        bucket.total_bytes,
        dtype=torch.uint8,
        device=fragment_tensors[0].device,
    )
    gathered_bytes = 0
    for entry, tensor in zip(bucket.entries, fragment_tensors):
        if (
            tuple(int(size) for size in tensor.shape) != entry.lengths
            or str(tensor.dtype).rsplit(".", maxsplit=1)[-1] != entry.dtype_name
        ):
            raise ValueError(
                f"Streaming full-gather fragment {entry.name!r} differs from its plan"
            )
        identity = tuple(range(len(entry.lengths)))
        if entry.destination_permutation != identity:
            tensor = tensor.permute(entry.destination_permutation)
        raw = tensor.contiguous().view(torch.uint8).view(-1)
        if int(raw.numel()) != entry.num_bytes:
            raise ValueError(
                f"Streaming full-gather fragment {entry.name!r} has {raw.numel()} "
                f"bytes, expected {entry.num_bytes}"
            )
        packed.narrow(0, entry.buffer_offset, entry.num_bytes).copy_(raw)
        gathered_bytes += entry.num_bytes
    return StreamingMaterializedBucket(
        value=packed,
        gathered_bytes=gathered_bytes,
        packed_bytes=int(packed.numel()),
    )


@dataclass(frozen=True)
class StreamingBucketAck:
    """Confirm that all intended workers consumed one exact bucket."""

    target_tp_rank: int
    bucket_index: int
    total_bytes: int
    worker_count: int


@dataclass(frozen=True)
class StreamingBucketStats:
    """Record one fully acknowledged and released streaming bucket."""

    target_tp_rank: int
    bucket_index: int
    fragment_count: int
    gathered_bytes: int
    packed_bytes: int
    worker_count: int
    sent_bytes: int
    acked_bytes: int
    released_bytes: int


@dataclass(frozen=True)
class StreamingFullGatherStats:
    """Report bounded allocations and completed streaming progress."""

    bucket_count: int
    fragment_count: int
    total_bytes: int
    max_gathered_bytes: int
    max_packed_bytes: int
    max_ipc_shared_bytes: int
    max_hccl_bytes: int
    max_transport_bytes: int
    transport_buffer_count: int
    max_inflight_buckets: int
    acked_buckets: int
    released_buckets: int
    buckets: tuple[StreamingBucketStats, ...]


class StreamingContentIdentityAccumulator:
    """Hash canonical tensor bytes independently of fragment and bucket boundaries."""

    def __init__(self, plan: StreamingFullGatherPlan, target_tp_rank: int) -> None:
        """Bind the accumulator to one plan's deterministic TP-local order."""
        self._expected = tuple(
            entry
            for bucket in plan.for_target(target_tp_rank)
            for entry in bucket.entries
        )
        if not self._expected:
            raise ValueError(
                f"Streaming full-gather target TP rank {target_tp_rank} has no fragments"
            )
        self._target_tp_rank = int(target_tp_rank)
        self._next_index = 0
        self._digest = sha256()
        self._current_tensor: Optional[tuple[Any, ...]] = None
        self._tensor_count = 0
        self._total_bytes = 0

    @property
    def fragment_count(self) -> int:
        """Return the number of unique canonical fragments already hashed."""
        return self._next_index

    def add(self, fragment: StreamingGatherFragment, values: bytes) -> None:
        """Hash one fragment and immediately discard its raw byte ownership.

        Args:
            fragment: Canonical fragment whose content was transferred.
            values: Contiguous source-format bytes in canonical dimension order.

        Raises:
            ValueError: If the byte count differs or the fragment is duplicated.
        """
        if len(values) != fragment.num_bytes:
            raise ValueError(
                f"Streaming full-gather digest bytes differ for {fragment.name!r}: "
                f"actual={len(values)}, expected={fragment.num_bytes}"
            )
        if self._next_index >= len(self._expected):
            raise ValueError(
                f"Streaming full-gather digest received an extra fragment: {fragment}"
            )
        expected = self._expected[self._next_index]
        if fragment != expected:
            raise ValueError(
                "Streaming full-gather digest order differs from its plan: "
                f"actual={fragment}, expected={expected}"
            )
        tensor_identity = (
            fragment.name,
            fragment.dtype_name,
            fragment.global_shape,
            fragment.target_tp_rank,
        )
        if tensor_identity != self._current_tensor:
            self._digest.update(
                json.dumps(
                    [
                        fragment.name,
                        fragment.dtype_name,
                        list(fragment.global_shape),
                        fragment.target_tp_rank,
                    ],
                    separators=(",", ":"),
                ).encode("utf-8")
            )
            self._current_tensor = tensor_identity
            self._tensor_count += 1
        self._digest.update(values)
        self._next_index += 1
        self._total_bytes += len(values)

    def finalize(self) -> dict[str, Any]:
        """Return a fragment-boundary-independent TP-local content identity."""
        if self._next_index != len(self._expected):
            raise RuntimeError(
                "Streaming full-gather digest is incomplete: "
                f"actual={self._next_index}, expected={len(self._expected)}"
            )
        return {
            "algorithm": "sha256-canonical-tensor-stream-v1",
            "target_tp_rank": self._target_tp_rank,
            "tensor_count": self._tensor_count,
            "fragment_count": self._next_index,
            "total_bytes": self._total_bytes,
            "digest": self._digest.hexdigest(),
        }


class StreamingFullGatherExecutor:
    """Execute one transaction while retaining failed asynchronous payloads."""

    def __init__(self) -> None:
        """Initialize transaction state and failed-payload ownership."""
        self._failed_payloads: list[StreamingMaterializedBucket] = []

    @property
    def failed_payload_count(self) -> int:
        """Return payloads retained until fallback or shutdown makes release safe."""
        return len(self._failed_payloads)

    @staticmethod
    def _validate_payload(
        plan: StreamingFullGatherPlan,
        bucket: StreamingGatherBucket,
        payload: StreamingMaterializedBucket,
    ) -> None:
        expected_gathered_bytes = sum(entry.num_bytes for entry in bucket.entries)
        if payload.gathered_bytes != expected_gathered_bytes:
            raise ValueError(
                "Streaming full-gather gathered bytes differ from the bucket plan: "
                f"gathered={payload.gathered_bytes}, expected={expected_gathered_bytes}"
            )
        if payload.packed_bytes != bucket.total_bytes:
            raise ValueError(
                "Streaming full-gather packed bytes differ from the bucket plan: "
                f"packed={payload.packed_bytes}, expected={bucket.total_bytes}"
            )
        if payload.packed_bytes > plan.bucket_size_bytes:
            raise ValueError(
                "Streaming full-gather packed bytes exceed the configured bucket: "
                f"packed={payload.packed_bytes}, bucket={plan.bucket_size_bytes}"
            )
        if payload.gathered_bytes > plan.bucket_size_bytes:
            raise ValueError(
                "Streaming full-gather gathered bytes exceed the configured bucket: "
                f"gathered={payload.gathered_bytes}, bucket={plan.bucket_size_bytes}"
            )
        value_numel = getattr(payload.value, "numel", None)
        if callable(value_numel) and int(value_numel()) != payload.packed_bytes:
            raise ValueError(
                "Streaming full-gather payload storage differs from its packed bytes: "
                f"storage={value_numel()}, packed={payload.packed_bytes}"
            )

    @staticmethod
    def _validate_ack(
        target_tp_rank: int,
        bucket_index: int,
        bucket: StreamingGatherBucket,
        ack: StreamingBucketAck,
    ) -> None:
        expected = (target_tp_rank, bucket_index, bucket.total_bytes)
        actual = (ack.target_tp_rank, ack.bucket_index, ack.total_bytes)
        if actual != expected or ack.worker_count <= 0:
            raise RuntimeError(
                f"Streaming full-gather bucket ACK is invalid: expected={expected}, "
                f"actual={actual}, workers={ack.worker_count}"
            )

    def execute(
        self,
        plan: StreamingFullGatherPlan,
        *,
        start: Callable[[], None],
        materialize_bucket: Callable[[StreamingGatherBucket], StreamingMaterializedBucket],
        send_bucket: Callable[
            [int, int, StreamingGatherBucket, StreamingMaterializedBucket],
            StreamingBucketAck,
        ],
        release_payload: Callable[[Any], None],
        finish: Callable[[], None],
        abort: Callable[[Exception], None],
        transport: str = "ipc",
    ) -> StreamingFullGatherStats:
        """Execute one ordered, ACK-gated streaming transaction.

        Failed payloads remain strongly referenced because abort only restores
        transaction identity; callers release them after fallback or shutdown.

        Args:
            plan: Immutable streaming plan.
            start: Begin one unpublished policy transaction.
            materialize_bucket: Gather and pack one bounded bucket.
            send_bucket: Deliver the payload and return an exact worker ACK.
            release_payload: Release one acknowledged payload.
            finish: Finalize the transaction after every bucket ACK.
            abort: Abort transaction identity after any failure.

        Returns:
            Allocation and progress counters for the completed transaction.

        Raises:
            RuntimeError: If a previous failed payload is still retained or ACK is invalid.
            Exception: Re-raises gather, send, release, finish, or abort failures.
        """
        if transport not in ("ipc", "hccl"):
            raise ValueError(
                f"Streaming full-gather transport must be 'ipc' or 'hccl', got {transport!r}"
            )
        if self._failed_payloads:
            raise RuntimeError(
                "Streaming full-gather cannot start while failed payloads are retained"
            )
        current_payload: Optional[StreamingMaterializedBucket] = None
        bucket_stats = []
        try:
            start()
            for target_tp_rank in range(plan.destination_tp_size):
                for bucket_index, bucket in enumerate(plan.for_target(target_tp_rank)):
                    if bucket.target_tp_rank != target_tp_rank:
                        raise RuntimeError(
                            "Streaming full-gather bucket target differs from plan index: "
                            f"bucket={bucket.target_tp_rank}, index={target_tp_rank}"
                        )
                    current_payload = materialize_bucket(bucket)
                    self._validate_payload(plan, bucket, current_payload)
                    ack = send_bucket(
                        target_tp_rank,
                        bucket_index,
                        bucket,
                        current_payload,
                    )
                    self._validate_ack(target_tp_rank, bucket_index, bucket, ack)
                    release_payload(current_payload.value)
                    bucket_stats.append(
                        StreamingBucketStats(
                            target_tp_rank=target_tp_rank,
                            bucket_index=bucket_index,
                            fragment_count=len(bucket.entries),
                            gathered_bytes=current_payload.gathered_bytes,
                            packed_bytes=current_payload.packed_bytes,
                            worker_count=ack.worker_count,
                            sent_bytes=ack.total_bytes * ack.worker_count,
                            acked_bytes=ack.total_bytes * ack.worker_count,
                            released_bytes=current_payload.packed_bytes,
                        )
                    )
                    current_payload = None
            finish()
        except Exception as error:
            if current_payload is not None:
                self._failed_payloads.append(current_payload)
            try:
                abort(error)
            except Exception as abort_error:
                raise RuntimeError(
                    "Streaming full-gather transaction and abort both failed: "
                    f"transaction={error!r}, abort={abort_error!r}"
                ) from abort_error
            raise
        # Records are appended only after a validated ACK and completed release.
        # Derive successful totals from that single source of truth.
        bucket_count = len(bucket_stats)
        max_packed_bytes = max((item.packed_bytes for item in bucket_stats), default=0)
        return StreamingFullGatherStats(
            bucket_count=bucket_count,
            fragment_count=sum(item.fragment_count for item in bucket_stats),
            total_bytes=sum(item.gathered_bytes for item in bucket_stats),
            max_gathered_bytes=max((item.gathered_bytes for item in bucket_stats), default=0),
            max_packed_bytes=max_packed_bytes,
            max_ipc_shared_bytes=max_packed_bytes if transport == "ipc" else 0,
            max_hccl_bytes=max_packed_bytes if transport == "hccl" else 0,
            max_transport_bytes=max((item.gathered_bytes + item.packed_bytes for item in bucket_stats), default=0),
            transport_buffer_count=2 if bucket_count else 0,
            max_inflight_buckets=1 if bucket_count else 0,
            acked_buckets=bucket_count,
            released_buckets=bucket_count,
            buckets=tuple(bucket_stats),
        )

    def release_failed_payloads(self, release_payload: Callable[[Any], None]) -> None:
        """Release retained buffers after fallback or shutdown completes safely."""
        while self._failed_payloads:
            payload = self._failed_payloads[-1]
            release_payload(payload.value)
            self._failed_payloads.pop()


__all__ = [
    "MaterializedStreamingContribution",
    "StreamingBucketAck",
    "StreamingBucketStats",
    "StreamingContentIdentityAccumulator",
    "StreamingFullGatherExecutor",
    "StreamingFullGatherPlan",
    "StreamingFullGatherStats",
    "StreamingGatherBucket",
    "StreamingGatherContribution",
    "StreamingGatherFragment",
    "StreamingMaterializedBucket",
    "StreamingTensorAlias",
    "assemble_streaming_fragment",
    "build_streaming_full_gather_plan",
    "extract_streaming_contributions",
    "pack_streaming_bucket",
]
