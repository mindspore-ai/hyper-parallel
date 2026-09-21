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
"""Tensor layouts, bounded direct plans, and plan-driven payload packing."""


from dataclasses import dataclass, replace
from itertools import product
from math import prod
from typing import Any, Iterable, Mapping, Optional, Sequence


@dataclass(frozen=True)
class TensorRegion:
    """One axis-aligned region in a tensor's global coordinate space."""

    starts: tuple[int, ...]
    lengths: tuple[int, ...]

    @property
    def ends(self) -> tuple[int, ...]:
        """Return the exclusive global end coordinate for every dimension."""
        return tuple(start + length for start, length in zip(self.starts, self.lengths))

    @property
    def numel(self) -> int:
        """Return the number of elements covered by this region."""
        return prod(self.lengths)


@dataclass(frozen=True)
class SourceTensorLayout:
    """Describe one rank-local FSDP shard in global tensor coordinates."""

    name: str
    dtype_name: str
    element_size: int
    global_shape: tuple[int, ...]
    source_rank: int
    region: TensorRegion
    source_name: Optional[str] = None
    source_starts: Optional[tuple[int, ...]] = None

    @property
    def source_key(self) -> str:
        """Return the physical Trainer state entry containing this tensor."""
        return self.source_name or self.name

    @property
    def local_starts(self) -> tuple[int, ...]:
        """Return the source-local offset corresponding to ``region``."""
        return self.source_starts or (0,) * len(self.global_shape)


@dataclass(frozen=True)
class DestinationTensorLayout:
    """Describe one Actor tensor region inside a rollout TP parameter."""

    name: str
    dtype_name: str
    element_size: int
    global_shape: tuple[int, ...]
    tp_rank: int
    tp_size: int
    region: TensorRegion
    destination_name: Optional[str] = None
    destination_starts: Optional[tuple[int, ...]] = None
    destination_permutation: Optional[tuple[int, ...]] = None
    accepted_source_dtypes: tuple[str, ...] = ()

    @property
    def target_name(self) -> str:
        """Return the physical rollout parameter receiving this region."""
        return self.destination_name or self.name

    @property
    def local_starts(self) -> tuple[int, ...]:
        """Return the physical parameter offset corresponding to ``region``."""
        return self.destination_starts or (0,) * len(self.global_shape)

    @property
    def physical_permutation(self) -> tuple[int, ...]:
        """Return the canonical-to-physical axis order."""
        return self.destination_permutation or tuple(range(len(self.global_shape)))


@dataclass(frozen=True)
class TransferEntry:
    """Copy one source-local slice into one destination-local slice."""

    name: str
    dtype_name: str
    element_size: int
    source_starts: tuple[int, ...]
    destination_starts: tuple[int, ...]
    lengths: tuple[int, ...]
    destination_name: Optional[str] = None
    source_name: Optional[str] = None
    destination_permutation: Optional[tuple[int, ...]] = None
    buffer_offset: int = 0
    destination_dtype_name: Optional[str] = None
    destination_element_size: Optional[int] = None

    @property
    def source_key(self) -> str:
        """Return the physical Trainer state entry supplying this fragment."""
        return self.source_name or self.name

    @property
    def physical_permutation(self) -> tuple[int, ...]:
        """Return the canonical-to-destination axis order."""
        return self.destination_permutation or tuple(range(len(self.lengths)))

    @property
    def destination_lengths(self) -> tuple[int, ...]:
        """Return fragment lengths in destination physical axis order."""
        return tuple(self.lengths[axis] for axis in self.physical_permutation)

    @property
    def target_name(self) -> str:
        """Return the physical rollout parameter receiving this fragment."""
        return self.destination_name or self.name

    @property
    def numel(self) -> int:
        """Return this fragment's number of values."""
        return prod(self.lengths)

    @property
    def num_bytes(self) -> int:
        """Return this fragment's serialized byte count."""
        return self.numel * self.element_size

    def with_buffer_offset(self, offset: int) -> "TransferEntry":
        """Return a copy assigned to one packed-buffer offset."""
        return replace(self, buffer_offset=offset)

    def worker_metadata(self) -> dict[str, Any]:
        """Serialize this fragment's worker write contract."""
        return {
            "name": self.target_name,
            "dtype_name": self.dtype_name,
            "element_size": self.element_size,
            "destination_dtype_name": (
                self.destination_dtype_name or self.dtype_name
            ),
            "destination_element_size": (
                self.destination_element_size or self.element_size
            ),
            "destination_starts": list(self.destination_starts),
            "destination_lengths": list(self.destination_lengths),
            "lengths": list(self.lengths),
            "buffer_offset": self.buffer_offset,
            "num_bytes": self.num_bytes,
        }


@dataclass(frozen=True)
class TransferBucket:
    """One bounded packed buffer broadcast from a source rank to a TP rank."""

    entries: tuple[TransferEntry, ...]
    total_bytes: int

    def worker_metadata(self) -> dict[str, Any]:
        """Serialize one receive-and-scatter operation."""
        return {
            "total_bytes": self.total_bytes,
            "entries": [entry.worker_metadata() for entry in self.entries],
        }


@dataclass(frozen=True)
class DirectReshardPlan:
    """A cached source-rank and destination-TP indexed transfer plan."""

    source_world_size: int
    destination_tp_size: int
    bucket_size_bytes: int
    buckets: Mapping[tuple[int, int], tuple[TransferBucket, ...]]

    def for_route(self, source_rank: int, tp_rank: int) -> tuple[TransferBucket, ...]:
        """Return all ordered buckets for one source-to-TP route."""
        return self.buckets.get((source_rank, tp_rank), ())

    @property
    def route_count(self) -> int:
        """Return the number of routes carrying at least one fragment."""
        return len(self.buckets)

    @property
    def fragment_count(self) -> int:
        """Return the number of planned rectangular copies."""
        return sum(
            len(bucket.entries)
            for route_buckets in self.buckets.values()
            for bucket in route_buckets
        )


def local_tensor(value: Any) -> Any:
    """Return a DTensor's local shard or the original plain tensor."""
    to_local = getattr(value, "to_local", None)
    return to_local() if callable(to_local) else value


def pack_direct_bucket(
    state_dict: Mapping[str, Any],
    bucket: TransferBucket,
    device: Any,
) -> Any:
    """Pack one direct route into a bounded byte tensor on ``device``."""
    import torch  # pylint: disable=C0415,forbidden-backend-import

    packed = torch.empty(bucket.total_bytes, dtype=torch.uint8, device=device)
    for entry in bucket.entries:
        value = state_dict.get(entry.source_key)
        if value is None:
            raise ValueError(
                f"Direct reshard source parameter {entry.source_key!r} is missing"
            )
        source_slice = tuple(
            slice(start, start + length)
            for start, length in zip(entry.source_starts, entry.lengths)
        )
        fragment = local_tensor(value)[source_slice].detach()
        if entry.physical_permutation != tuple(range(len(entry.lengths))):
            fragment = fragment.permute(entry.physical_permutation)
        fragment = fragment.contiguous()
        if str(fragment.device) != str(device):
            fragment = fragment.to(device)
        raw = fragment.view(torch.uint8).view(-1)
        if int(raw.numel()) != entry.num_bytes:
            raise ValueError(
                f"Direct reshard source fragment {entry.source_key!r} has "
                f"{raw.numel()} bytes, expected {entry.num_bytes}"
            )
        packed.narrow(0, entry.buffer_offset, entry.num_bytes).copy_(raw)
    return packed


def describe_source_tensor(name: str, tensor: Any, source_rank: int) -> dict[str, Any]:
    """Describe a local state-dict value without materializing its full tensor."""
    local_value = local_tensor(tensor)
    placements = tuple(getattr(tensor, "placements", ()) or ())
    global_shape = tuple(int(size) for size in tensor.shape)
    local_shape = tuple(int(size) for size in local_value.shape)
    if len(global_shape) != len(local_shape):
        raise ValueError(
            f"Direct reshard source tensor {name!r} rank mismatch: "
            f"global={global_shape}, local={local_shape}"
        )
    description = {
        "name": name,
        "dtype_name": str(local_value.dtype).rsplit(".", maxsplit=1)[-1],
        "element_size": int(local_value.element_size()),
        "global_shape": list(global_shape),
        "local_shape": list(local_shape),
        "source_rank": int(source_rank),
    }
    shard_placements = [
        placement
        for placement in placements
        if callable(getattr(placement, "is_shard", None)) and placement.is_shard()
    ]
    device_mesh = getattr(tensor, "device_mesh", None)
    if not shard_placements:
        description["shard_dim"] = None
        description["region_starts"] = [0] * len(global_shape)
        return description
    if device_mesh is None or len(placements) != int(device_mesh.ndim):
        if len(shard_placements) == 1:
            description["shard_dim"] = int(shard_placements[0].dim)
            return description
        raise ValueError(
            f"Direct reshard source tensor {name!r} requires mesh metadata for "
            f"multi-axis placements={placements}"
        )
    coordinate = device_mesh.get_coordinate()
    if coordinate is None or len(coordinate) != len(placements):
        raise ValueError(
            f"Direct reshard source tensor {name!r} has no complete mesh coordinate"
        )
    starts = [0] * len(global_shape)
    lengths = list(global_shape)
    # TP is applied before the outer FSDP wrapper.  When both placements shard
    # the same tensor dimension, recover global offsets from inner to outer.
    for mesh_dim in reversed(range(len(placements))):
        placement = placements[mesh_dim]
        if not callable(getattr(placement, "is_shard", None)) or not placement.is_shard():
            continue
        tensor_dim = int(placement.dim)
        mesh_size = int(device_mesh.size(mesh_dim))
        mesh_rank = int(coordinate[mesh_dim])
        current_length = lengths[tensor_dim]
        base, remainder = divmod(current_length, mesh_size)
        shard_length = base + int(mesh_rank < remainder)
        relative_start = mesh_rank * base + min(mesh_rank, remainder)
        starts[tensor_dim] += relative_start
        lengths[tensor_dim] = shard_length
    if tuple(lengths) != local_shape:
        raise ValueError(
            f"Direct reshard source tensor {name!r} mesh-derived shape differs from local: "
            f"derived={tuple(lengths)}, local={local_shape}, placements={placements}"
        )
    description["shard_dim"] = None
    description["region_starts"] = starts
    return description


def resolve_source_layouts(
    rank_descriptions: Sequence[Sequence[Mapping[str, Any]]],
) -> tuple[SourceTensorLayout, ...]:
    """Resolve rank-local source descriptions into global rectangular regions."""
    if not rank_descriptions:
        raise ValueError("Direct reshard requires at least one source rank")
    by_name: dict[str, list[Mapping[str, Any]]] = {}
    for rank, descriptions in enumerate(rank_descriptions):
        for description in descriptions:
            if int(description["source_rank"]) != rank:
                raise ValueError(
                    "Direct reshard source metadata rank mismatch: "
                    f"list_rank={rank}, metadata_rank={description['source_rank']}"
                )
            by_name.setdefault(str(description["name"]), []).append(description)
    layouts = []
    for name, descriptions in sorted(by_name.items()):
        descriptions = sorted(descriptions, key=lambda value: int(value["source_rank"]))
        if not all("region_starts" in description for description in descriptions):
            raise ValueError(
                f"Direct reshard source tensor {name!r} requires explicit regions"
            )
        first = descriptions[0]
        global_shape = tuple(int(size) for size in first["global_shape"])
        dtype_name = str(first["dtype_name"])
        element_size = int(first["element_size"])
        signatures = {
            (
                tuple(int(size) for size in description["global_shape"]),
                str(description["dtype_name"]),
                int(description["element_size"]),
            )
            for description in descriptions
        }
        if signatures != {(global_shape, dtype_name, element_size)}:
            raise ValueError(
                f"Direct reshard source tensor {name!r} metadata differs across ranks"
            )
        unique_regions = _unique_source_regions(name, descriptions, global_shape)
        layouts.extend(
            SourceTensorLayout(
                name,
                dtype_name,
                element_size,
                global_shape,
                int(description["source_rank"]),
                TensorRegion(starts, lengths),
                str(description.get("source_name", name)),
                tuple(
                    int(value)
                    for value in description.get(
                        "source_starts",
                        [0] * len(global_shape),
                    )
                ),
            )
            for (starts, lengths), description in unique_regions.items()
        )
        continue
    return tuple(layouts)


def _validate_source_coverage(name, regions, global_shape):
    """Reject overlapping or incomplete source regions before planning transfers."""
    for index, left in enumerate(regions):
        for right in regions[index + 1:]:
            if all(
                max(left_start, right_start) < min(left_end, right_end)
                for left_start, left_end, right_start, right_end in zip(
                    left.starts, left.ends, right.starts, right.ends,
                )
            ):
                raise ValueError(f"Direct reshard source tensor {name!r} regions overlap")
    if sum(region.numel for region in regions) != prod(global_shape):
        raise ValueError(f"Direct reshard source tensor {name!r} regions do not cover global shape")


def _destination_shard_offsets(
    name: str,
    tensors: list[Mapping[str, Any]],
    global_shape: tuple[int, ...],
    shard_dim: int,
) -> list[int]:
    """Resolve dense TP offsets and reject retired expert-group coordinates."""
    if not 0 <= shard_dim < len(global_shape):
        raise ValueError(f"Invalid destination shard dimension for {name!r}")
    offsets = []
    offset = 0
    for tensor in tensors:
        if "shard_rank" in tensor or "shard_group_size" in tensor:
            raise ValueError(f"Qwen3 destination {name!r} requires ordinary TP coordinates")
        shape = tuple(int(size) for size in tensor["local_shape"])
        if len(shape) != len(global_shape) or shape[shard_dim] <= 0:
            raise ValueError(f"Invalid destination shard shape for {name!r}: {shape}")
        offsets.append(offset)
        offset += shape[shard_dim]
    if offset != global_shape[shard_dim]:
        raise ValueError(f"Rollout parameter {name!r} TP shards cover {offset} values, expected {global_shape}")
    return offsets


def _destination_signature(
    tensor: Mapping[str, Any],
    name: str,
    rank: int,
) -> tuple[Any, ...]:
    """Return the TP-invariant part of one destination description."""
    permutation = tuple(
        int(value)
        for value in tensor.get("destination_permutation", range(rank))
    )
    if sorted(permutation) != list(range(rank)):
        raise ValueError(
            f"Rollout parameter {name!r} has invalid destination permutation "
            f"{permutation}"
        )
    return (
        str(tensor["placement"]),
        tensor.get("shard_dim"),
        str(tensor.get("destination_name", name)),
        str(tensor["dtype_name"]),
        int(tensor["element_size"]),
        permutation,
        tuple(str(value) for value in tensor.get("accepted_source_dtypes", ())),
    )


def resolve_destination_layouts(
    worker_descriptions: Sequence[Mapping[str, Any]],
    global_shapes: Mapping[str, tuple[int, ...]],
) -> tuple[DestinationTensorLayout, ...]:
    """Resolve per-worker TP metadata into Actor-coordinate destination regions."""
    if not worker_descriptions:
        raise ValueError("Direct reshard rollout layout returned no TP workers")
    workers = sorted(worker_descriptions, key=lambda value: int(value["tp_rank"]))
    tp_size = int(workers[0]["tp_size"])
    if len(workers) != tp_size or [int(worker["tp_rank"]) for worker in workers] != list(range(tp_size)):
        raise ValueError(
            "Direct reshard rollout TP ranks must be dense and unique: "
            f"ranks={[worker['tp_rank'] for worker in workers]}, tp_size={tp_size}"
        )
    tensors_by_worker = {
        int(worker["tp_rank"]): {
            str(tensor["name"]): tensor for tensor in worker["tensors"]
        }
        for worker in workers
    }
    parameter_names = set(tensors_by_worker[0])
    if any(set(tensors) != parameter_names for tensors in tensors_by_worker.values()):
        raise ValueError("Direct reshard rollout parameter names differ across TP workers")
    layouts = []
    for name in sorted(parameter_names):
        if name not in global_shapes:
            raise ValueError(f"Rollout parameter {name!r} is absent from the Actor policy")
        global_shape = global_shapes[name]
        tensors = [tensors_by_worker[rank][name] for rank in range(tp_size)]
        layouts.extend(_parameter_destination_layouts(name, tensors, global_shape, tp_size))
    return tuple(layouts)


def _intersect_regions(
    source: TensorRegion,
    destination: TensorRegion,
) -> Optional[TensorRegion]:
    """Return the overlap between two tensor regions, if any."""
    starts = tuple(max(left, right) for left, right in zip(source.starts, destination.starts))
    ends = tuple(min(left, right) for left, right in zip(source.ends, destination.ends))
    lengths = tuple(end - start for start, end in zip(starts, ends))
    if any(length <= 0 for length in lengths):
        return None
    return TensorRegion(starts, lengths)


def _aligned_offset(offset: int, alignment: int) -> int:
    """Round one byte offset up to the requested alignment."""
    return ((offset + alignment - 1) // alignment) * alignment


def _tile_region(
    region: TensorRegion,
    *,
    element_size: int,
    bucket_size_bytes: int,
) -> tuple[TensorRegion, ...]:
    """Tile a rectangle in canonical order, independently of gather/direct routing."""
    max_numel = bucket_size_bytes // element_size
    if max_numel <= 0:
        raise ValueError("Weight-sync bucket is smaller than one tensor element")
    if region.numel <= max_numel:
        return (region,)
    chunks = [1] * len(region.lengths)
    remaining = max_numel
    for dim in reversed(range(len(chunks))):
        chunks[dim] = min(region.lengths[dim], max(1, remaining))
        remaining = max(1, remaining // chunks[dim])
    ranges = [range(0, length, chunk) for length, chunk in zip(region.lengths, chunks)]
    return tuple(
        TensorRegion(
            tuple(start + offset for start, offset in zip(region.starts, offsets)),
            tuple(min(chunk, length - offset) for offset, chunk, length in zip(offsets, chunks, region.lengths)),
        )
        for offsets in product(*ranges)
    )


def _bucketize(
    entries: Iterable[TransferEntry],
    bucket_size_bytes: int,
) -> tuple[TransferBucket, ...]:
    """Assign aligned offsets and group bounded direct entries."""
    buckets, current = [], []
    size = 0
    for entry in entries:
        if entry.num_bytes > bucket_size_bytes:
            raise ValueError(f"Weight-sync fragment {entry.name!r} exceeds its bucket")
        offset = _aligned_offset(size, entry.element_size)
        if current and offset + entry.num_bytes > bucket_size_bytes:
            buckets.append(TransferBucket(tuple(current), size))
            current, offset = [], 0
        current.append(entry.with_buffer_offset(offset))
        size = offset + entry.num_bytes
    if current:
        buckets.append(TransferBucket(tuple(current), size))
    return tuple(buckets)


def _split_entry(entry: TransferEntry, bucket_size_bytes: int) -> tuple[TransferEntry, ...]:
    """Apply shared canonical tiles to the source and permuted destination."""
    region = TensorRegion((0,) * len(entry.lengths), entry.lengths)
    tiles = _tile_region(
        region,
        element_size=entry.element_size,
        bucket_size_bytes=bucket_size_bytes,
    )
    if tiles == (region,):
        return (entry,)
    return tuple(
        replace(
            entry,
            source_starts=tuple(start + offset for start, offset in zip(entry.source_starts, tile.starts)),
            destination_starts=tuple(
                start + tile.starts[axis] for start, axis in zip(entry.destination_starts, entry.physical_permutation)
            ),
            lengths=tile.lengths,
            buffer_offset=0,
        )
        for tile in tiles
    )


def _transfer_entry(
    source: SourceTensorLayout,
    destination: DestinationTensorLayout,
    intersection: TensorRegion,
) -> TransferEntry:
    """Translate one global intersection into source and destination offsets."""
    source_starts = tuple(
        local + start - base
        for local, start, base in zip(
            source.local_starts,
            intersection.starts,
            source.region.starts,
        )
    )
    destination_offsets = tuple(
        start - base
        for start, base in zip(
            intersection.starts,
            destination.region.starts,
        )
    )
    destination_starts = tuple(
        local + destination_offsets[axis]
        for local, axis in zip(
            destination.local_starts,
            destination.physical_permutation,
        )
    )
    return TransferEntry(
        name=source.name,
        dtype_name=source.dtype_name,
        element_size=source.element_size,
        source_starts=source_starts,
        destination_starts=destination_starts,
        lengths=intersection.lengths,
        destination_name=destination.target_name,
        source_name=source.source_key,
        destination_permutation=destination.physical_permutation,
        destination_dtype_name=destination.dtype_name,
        destination_element_size=destination.element_size,
    )


def _validate_coverage(
    name: str,
    destinations: Sequence[DestinationTensorLayout],
    coverage: Mapping[tuple[str, int], int],
) -> None:
    """Require every destination region to receive each value exactly once."""
    for destination in destinations:
        actual = coverage.get((name, destination.tp_rank), 0)
        if actual != destination.region.numel:
            raise ValueError(
                f"Direct reshard plan covers {actual} values for {name!r} TP rank "
                f"{destination.tp_rank}, expected {destination.region.numel}"
            )


def build_direct_reshard_plan(
    sources: Sequence[SourceTensorLayout],
    destinations: Sequence[DestinationTensorLayout],
    *,
    source_world_size: int,
    bucket_size_bytes: int,
) -> DirectReshardPlan:
    """Compile global source/destination regions into bounded broadcast routes."""
    if bucket_size_bytes <= 0:
        raise ValueError("Direct reshard bucket_size_bytes must be positive")
    sources_by_name: dict[str, list[SourceTensorLayout]] = {}
    destinations_by_name: dict[str, list[DestinationTensorLayout]] = {}
    for source in sources:
        sources_by_name.setdefault(source.name, []).append(source)
    for destination in destinations:
        destinations_by_name.setdefault(destination.name, []).append(destination)
    if set(sources_by_name) != set(destinations_by_name):
        raise ValueError(
            "Direct reshard source/destination parameter mismatch: "
            f"source_only={sorted(set(sources_by_name) - set(destinations_by_name))}, "
            f"destination_only={sorted(set(destinations_by_name) - set(sources_by_name))}"
        )
    route_entries: dict[tuple[int, int], list[TransferEntry]] = {}
    coverage: dict[tuple[str, int], int] = {}
    for name in sorted(sources_by_name):
        for source in sources_by_name[name]:
            for destination in destinations_by_name[name]:
                dtype_compatible = (
                    source.dtype_name == destination.dtype_name
                    and source.element_size == destination.element_size
                ) or source.dtype_name in destination.accepted_source_dtypes
                if source.global_shape != destination.global_shape or not dtype_compatible:
                    raise ValueError(
                        f"Direct reshard tensor contract mismatch for {name!r}: "
                        f"source={(source.global_shape, source.dtype_name)}, "
                        f"destination={(destination.global_shape, destination.dtype_name)}"
                    )
                intersection = _intersect_regions(
                    source.region,
                    destination.region,
                )
                if intersection is None:
                    continue
                entry = _transfer_entry(source, destination, intersection)
                route_entries.setdefault((source.source_rank, destination.tp_rank), []).append(entry)
                coverage[(name, destination.tp_rank)] = (
                    coverage.get((name, destination.tp_rank), 0) + entry.numel
                )
        _validate_coverage(name, destinations_by_name[name], coverage)
    tp_sizes = {destination.tp_size for destination in destinations}
    if len(tp_sizes) != 1:
        raise ValueError(f"Direct reshard destination TP sizes differ: {sorted(tp_sizes)}")
    buckets = {
        route: _bucketize(
            (
                fragment
                for entry in entries
                for fragment in _split_entry(entry, bucket_size_bytes)
            ),
            bucket_size_bytes,
        )
        for route, entries in route_entries.items()
    }
    return DirectReshardPlan(
        source_world_size=source_world_size,
        destination_tp_size=tp_sizes.pop(),
        bucket_size_bytes=bucket_size_bytes,
        buckets=buckets,
    )


def _unique_source_regions(name, descriptions, global_shape):
    """Validate rectangular source coverage and retain the first replica owner."""
    unique_regions: dict[tuple[tuple[int, ...], tuple[int, ...]], Mapping[str, Any]] = {}
    for description in descriptions:
        starts = tuple(int(value) for value in description["region_starts"])
        lengths = tuple(int(value) for value in description["local_shape"])
        if len(starts) != len(global_shape) or len(lengths) != len(global_shape):
            raise ValueError(
                f"Direct reshard source tensor {name!r} region rank mismatch"
            )
        if any(
            start < 0 or length <= 0 or start + length > global_size
            for start, length, global_size in zip(starts, lengths, global_shape)
        ):
            raise ValueError(
                f"Direct reshard source tensor {name!r} has invalid region "
                f"starts={starts}, lengths={lengths}, global={global_shape}"
            )
        unique_regions.setdefault((starts, lengths), description)
    regions = [TensorRegion(starts, lengths) for starts, lengths in unique_regions]
    _validate_source_coverage(name, regions, global_shape)
    return unique_regions


def _destination_region_starts(name, placement, local_shape, global_shape, shard_dim, shard_offset):
    """Validate placement shape and calculate the destination region origin."""
    if placement == "replicate":
        if local_shape != global_shape:
            raise ValueError(
                f"Replicated rollout parameter {name!r} has local shape "
                f"{local_shape}, expected {global_shape}"
            )
        starts = (0,) * len(global_shape)
    elif placement == "shard":
        for dim, (local_size, global_size) in enumerate(zip(local_shape, global_shape)):
            if dim != shard_dim and local_size != global_size:
                raise ValueError(
                    f"Rollout parameter {name!r} changes non-sharded dim {dim}"
                )
        starts_list = [0] * len(global_shape)
        starts_list[shard_dim] = shard_offset
        starts = tuple(starts_list)
    else:
        raise ValueError(
            f"Unsupported rollout placement {placement!r} for parameter {name!r}"
        )
    return starts


def _parameter_destination_layouts(name, tensors, global_shape, tp_size):
    """Resolve a single parameter after validating common TP worker metadata."""
    layouts = []
    signature = _destination_signature(tensors[0], name, len(global_shape))
    if any(
        _destination_signature(tensor, name, len(global_shape)) != signature
        for tensor in tensors[1:]
    ):
        raise ValueError(
            f"Rollout parameter {name!r} layout differs across TP workers"
        )
    (
        placement,
        shard_dim,
        destination_name,
        dtype_name,
        element_size,
        destination_permutation,
        accepted_source_dtypes,
    ) = signature
    if placement == "shard":
        if shard_dim is None:
            raise ValueError(f"Sharded rollout parameter {name!r} has no shard_dim")
        shard_dim = int(shard_dim)
        shard_offsets = _destination_shard_offsets(
            name,
            tensors,
            global_shape,
            shard_dim,
        )
    else:
        shard_offsets = ()
    for tp_rank, tensor in enumerate(tensors):
        local_shape = tuple(int(size) for size in tensor["local_shape"])
        starts = _destination_region_starts(
            name, placement, local_shape, global_shape, shard_dim,
            shard_offsets[tp_rank] if placement == "shard" else None,
        )
        destination_starts = tuple(
            int(value)
            for value in tensor.get(
                "destination_starts",
                [0] * len(global_shape),
            )
        )
        if len(destination_starts) != len(global_shape):
            raise ValueError(
                f"Rollout parameter {name!r} destination offset rank mismatch: "
                f"offset={destination_starts}, global_shape={global_shape}"
            )
        layouts.append(
            DestinationTensorLayout(
                name,
                dtype_name,
                element_size,
                global_shape,
                tp_rank,
                tp_size,
                TensorRegion(starts, local_shape),
                destination_name,
                destination_starts,
                destination_permutation,
                accepted_source_dtypes,
            )
        )
    return layouts
