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
"""Layout metadata and cached plans for FSDP-to-TP direct resharding."""

from dataclasses import dataclass, replace
from itertools import product
from math import prod
from typing import Any, Iterable, Mapping, Optional, Sequence


@dataclass(frozen=True)
class PhysicalRolloutWorker:
    """Bind one rollout DP x TP worker to a colocated physical device."""

    dp_rank: int
    tp_rank: int
    physical_device_id: Any


def resolve_physical_worker_topology(
    physical_device_ids: Sequence[Any],
    *,
    data_parallel_size: int,
    tensor_parallel_size: int,
) -> tuple[PhysicalRolloutWorker, ...]:
    """Resolve vLLM's DP-major worker order into an explicit device mapping."""
    if data_parallel_size <= 0 or tensor_parallel_size <= 0:
        raise ValueError("Rollout data_parallel_size and tensor_parallel_size must be positive")
    expected = data_parallel_size * tensor_parallel_size
    if len(physical_device_ids) != expected:
        raise ValueError(
            "Rollout physical devices must match DP x TP: "
            f"expected={expected}, got={len(physical_device_ids)}"
        )
    if len(set(physical_device_ids)) != expected:
        raise ValueError("Rollout physical device identities must be unique")
    return tuple(
        PhysicalRolloutWorker(
            dp_rank=worker_index // tensor_parallel_size,
            tp_rank=worker_index % tensor_parallel_size,
            physical_device_id=physical_device_id,
        )
        for worker_index, physical_device_id in enumerate(physical_device_ids)
    )


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
    placement: str
    shard_dim: Optional[int]
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
    canonical_starts: Optional[tuple[int, ...]] = None
    destination_permutation: Optional[tuple[int, ...]] = None
    buffer_offset: int = 0
    destination_dtype_name: Optional[str] = None
    destination_element_size: Optional[int] = None

    @property
    def source_key(self) -> str:
        """Return the physical Trainer state entry supplying this fragment."""
        return self.source_name or self.name

    @property
    def logical_starts(self) -> tuple[int, ...]:
        """Return this fragment's offset in canonical tensor coordinates."""
        return self.canonical_starts or self.source_starts

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
        """Serialize the destination half of this copy for a worker RPC."""
        return {
            "name": self.target_name,
            "canonical_name": self.name,
            "canonical_starts": list(self.logical_starts),
            "dtype_name": self.dtype_name,
            "element_size": self.element_size,
            "destination_dtype_name": self.destination_dtype_name or self.dtype_name,
            "destination_element_size": (
                self.destination_element_size or self.element_size
            ),
            "destination_starts": list(self.destination_starts),
            "destination_lengths": list(self.destination_lengths),
            "destination_permutation": list(self.physical_permutation),
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


def describe_source_tensor(name: str, tensor: Any, source_rank: int) -> dict[str, Any]:
    """Describe a local state-dict value without materializing its full tensor."""
    local_tensor = tensor.to_local() if callable(getattr(tensor, "to_local", None)) else tensor
    placements = tuple(getattr(tensor, "placements", ()) or ())
    global_shape = tuple(int(size) for size in tensor.shape)
    local_shape = tuple(int(size) for size in local_tensor.shape)
    if len(global_shape) != len(local_shape):
        raise ValueError(
            f"Direct reshard source tensor {name!r} rank mismatch: "
            f"global={global_shape}, local={local_shape}"
        )
    description = {
        "name": name,
        "dtype_name": str(local_tensor.dtype).rsplit(".", maxsplit=1)[-1],
        "element_size": int(local_tensor.element_size()),
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
        has_explicit_regions = all(
            "region_starts" in description for description in descriptions
        )
        if len(descriptions) != len(rank_descriptions) and not has_explicit_regions:
            raise ValueError(
                f"Direct reshard source tensor {name!r} is missing on an FSDP rank"
            )
        first = descriptions[0]
        global_shape = tuple(int(size) for size in first["global_shape"])
        dtype_name = str(first["dtype_name"])
        element_size = int(first["element_size"])
        explicit_regions = ["region_starts" in description for description in descriptions]
        if any(explicit_regions):
            if not all(explicit_regions):
                raise ValueError(
                    f"Direct reshard source tensor {name!r} mixes explicit and legacy regions"
                )
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
            for index, left in enumerate(regions):
                for right in regions[index + 1:]:
                    if all(
                        max(left_start, right_start) < min(left_end, right_end)
                        for left_start, left_end, right_start, right_end in zip(
                            left.starts,
                            left.ends,
                            right.starts,
                            right.ends,
                        )
                    ):
                        raise ValueError(
                            f"Direct reshard source tensor {name!r} regions overlap"
                        )
            if sum(region.numel for region in regions) != prod(global_shape):
                raise ValueError(
                    f"Direct reshard source tensor {name!r} regions do not cover global shape"
                )
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
        shard_dim = first.get("shard_dim")
        for description in descriptions[1:]:
            signature = (
                tuple(int(size) for size in description["global_shape"]),
                str(description["dtype_name"]),
                int(description["element_size"]),
                description.get("shard_dim"),
            )
            if signature != (global_shape, dtype_name, element_size, shard_dim):
                raise ValueError(
                    f"Direct reshard source tensor {name!r} metadata differs across ranks"
                )
        if shard_dim is None:
            local_shape = tuple(int(size) for size in first["local_shape"])
            if local_shape != global_shape:
                raise ValueError(
                    f"Replicated source tensor {name!r} must be globally shaped, "
                    f"got local={local_shape}, global={global_shape}"
                )
            layouts.append(
                SourceTensorLayout(
                    name,
                    dtype_name,
                    element_size,
                    global_shape,
                    int(first["source_rank"]),
                    TensorRegion((0,) * len(global_shape), global_shape),
                )
            )
            continue
        shard_dim = int(shard_dim)
        offset = 0
        for description in descriptions:
            local_shape = tuple(int(size) for size in description["local_shape"])
            for dim, (local_size, global_size) in enumerate(zip(local_shape, global_shape)):
                if dim != shard_dim and local_size != global_size:
                    raise ValueError(
                        f"Source tensor {name!r} unexpectedly changes non-sharded dim {dim}: "
                        f"local={local_shape}, global={global_shape}"
                    )
            starts = [0] * len(global_shape)
            starts[shard_dim] = offset
            layouts.append(
                SourceTensorLayout(
                    name,
                    dtype_name,
                    element_size,
                    global_shape,
                    int(description["source_rank"]),
                    TensorRegion(tuple(starts), local_shape),
                )
            )
            offset += local_shape[shard_dim]
        if offset != global_shape[shard_dim]:
            raise ValueError(
                f"Source tensor {name!r} shards cover {offset} values on dim {shard_dim}, "
                f"expected {global_shape[shard_dim]}"
            )
    return tuple(layouts)


def _destination_shard_offsets(
    name: str, tensors: list[Mapping[str, Any]], global_shape: tuple[int, ...],
) -> list[int]:
    """Resolve parameter-specific shard groups within flattened physical targets."""
    size = int(tensors[0].get("shard_group_size", len(tensors)))
    if size <= 0 or len(tensors) % size:
        raise ValueError(f"Invalid destination shard group size for {name!r}: {size}")
    widths = {}
    replicas = {}
    coordinates = []
    for target, tensor in enumerate(tensors):
        if tensor["placement"] != "shard" or tensor.get("shard_dim") != tensors[0].get("shard_dim"):
            raise ValueError(f"Rollout parameter {name!r} layout differs across TP workers")
        if ("shard_rank" in tensor) != ("shard_group_size" in tensor):
            raise ValueError(f"Incomplete destination shard coordinates for {name!r}")
        rank = int(tensor.get("shard_rank", target))
        if int(tensor.get("shard_group_size", len(tensors))) != size or not 0 <= rank < size:
            raise ValueError(f"Invalid destination shard rank/group for {name!r}")
        shape = tuple(int(value) for value in tensor["local_shape"])
        dim = tensor.get("shard_dim")
        if dim is None or not 0 <= int(dim) < len(global_shape) or len(shape) != len(global_shape):
            raise ValueError(f"Invalid destination shard dimension/shape for {name!r}")
        width = shape[int(dim)]
        if width <= 0 or (rank in widths and widths[rank] != width):
            raise ValueError(f"Destination shard replicas disagree on shape for {name!r}")
        widths[rank] = width
        replicas[rank] = replicas.get(rank, 0) + 1
        coordinates.append(rank)
    if set(widths) != set(range(size)) or set(replicas.values()) != {len(tensors) // size}:
        raise ValueError(f"Destination shards do not cover every replica for {name!r}")
    offsets = {}
    offset = 0
    for rank in range(size):
        offsets[rank] = offset
        offset += widths[rank]
    if offset != global_shape[int(tensors[0]["shard_dim"])]:
        raise ValueError(f"Rollout parameter {name!r} TP shards cover {offset} values, expected {global_shape}")
    return [offsets[rank] for rank in coordinates]


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
        first = tensors_by_worker[0][name]
        placement = str(first["placement"])
        shard_dim = first.get("shard_dim")
        destination_name = str(first.get("destination_name", name))
        destination_permutation = tuple(
            int(value)
            for value in first.get(
                "destination_permutation",
                range(len(global_shape)),
            )
        )
        if sorted(destination_permutation) != list(range(len(global_shape))):
            raise ValueError(
                f"Rollout parameter {name!r} has invalid destination permutation "
                f"{destination_permutation}"
            )
        dtype_name = str(first["dtype_name"])
        element_size = int(first["element_size"])
        accepted_source_dtypes = tuple(
            str(value) for value in first.get("accepted_source_dtypes", ())
        )
        shard_offsets = (
            _destination_shard_offsets(name, [tensors_by_worker[rank][name] for rank in range(tp_size)], global_shape)
            if placement == "shard" else ()
        )
        for tp_rank in range(tp_size):
            tensor = tensors_by_worker[tp_rank][name]
            signature = (
                str(tensor["placement"]),
                tensor.get("shard_dim"),
                str(tensor.get("destination_name", name)),
                str(tensor["dtype_name"]),
                int(tensor["element_size"]),
                tuple(
                    int(value)
                    for value in tensor.get(
                        "destination_permutation",
                        range(len(global_shape)),
                    )
                ),
                tuple(str(value) for value in tensor.get("accepted_source_dtypes", ())),
            )
            if signature != (
                placement,
                shard_dim,
                destination_name,
                dtype_name,
                element_size,
                destination_permutation,
                accepted_source_dtypes,
            ):
                raise ValueError(
                    f"Rollout parameter {name!r} layout differs across TP workers"
                )
            local_shape = tuple(int(size) for size in tensor["local_shape"])
            if placement == "replicate":
                if local_shape != global_shape:
                    raise ValueError(
                        f"Replicated rollout parameter {name!r} has local shape "
                        f"{local_shape}, expected {global_shape}"
                    )
                starts = (0,) * len(global_shape)
            elif placement == "shard":
                if shard_dim is None:
                    raise ValueError(f"Sharded rollout parameter {name!r} has no shard_dim")
                shard_dim = int(shard_dim)
                for dim, (local_size, global_size) in enumerate(zip(local_shape, global_shape)):
                    if dim != shard_dim and local_size != global_size:
                        raise ValueError(
                            f"Rollout parameter {name!r} changes non-sharded dim {dim}"
                        )
                starts_list = [0] * len(global_shape)
                starts_list[shard_dim] = shard_offsets[tp_rank]
                starts = tuple(starts_list)
            else:
                raise ValueError(
                    f"Unsupported rollout placement {placement!r} for parameter {name!r}"
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
                    placement,
                    None if shard_dim is None else int(shard_dim),
                    TensorRegion(starts, local_shape),
                    destination_name,
                    destination_starts,
                    destination_permutation,
                    accepted_source_dtypes,
                )
            )
    return tuple(layouts)


def intersect_regions(
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


def aligned_offset(offset: int, alignment: int) -> int:
    """Round one byte offset up to the requested alignment."""
    return ((offset + alignment - 1) // alignment) * alignment


def tile_region(region: TensorRegion, *, element_size: int, bucket_size_bytes: int) -> tuple[TensorRegion, ...]:
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


def bucketize_entries(entries: Iterable[Any], bucket_size_bytes: int) -> tuple[tuple[tuple[Any, ...], int], ...]:
    """Assign aligned byte offsets to already bounded direct or gather entries."""
    buckets, current = [], []
    size = 0
    for entry in entries:
        if entry.num_bytes > bucket_size_bytes:
            raise ValueError(f"Weight-sync fragment {entry.name!r} exceeds its bucket")
        offset = aligned_offset(size, entry.element_size)
        if current and offset + entry.num_bytes > bucket_size_bytes:
            buckets.append((tuple(current), size))
            current, offset = [], 0
        current.append(entry.with_buffer_offset(offset))
        size = offset + entry.num_bytes
    if current:
        buckets.append((tuple(current), size))
    return tuple(buckets)


def _bucketize(entries: Iterable[TransferEntry], bucket_size_bytes: int) -> tuple[TransferBucket, ...]:
    return tuple(TransferBucket(items, size) for items, size in bucketize_entries(entries, bucket_size_bytes))


def _split_entry(entry: TransferEntry, bucket_size_bytes: int) -> tuple[TransferEntry, ...]:
    """Apply shared canonical tiles to the source and permuted destination."""
    region = TensorRegion((0,) * len(entry.lengths), entry.lengths)
    tiles = tile_region(region, element_size=entry.element_size, bucket_size_bytes=bucket_size_bytes)
    if tiles == (region,):
        return (entry,)
    return tuple(
        replace(
            entry,
            source_starts=tuple(start + offset for start, offset in zip(entry.source_starts, tile.starts)),
            destination_starts=tuple(
                start + tile.starts[axis] for start, axis in zip(entry.destination_starts, entry.physical_permutation)
            ),
            canonical_starts=tuple(start + offset for start, offset in zip(entry.logical_starts, tile.starts)),
            lengths=tile.lengths,
            buffer_offset=0,
        )
        for tile in tiles
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
                intersection = intersect_regions(source.region, destination.region)
                if intersection is None:
                    continue
                source_starts = tuple(
                    local_start + start - base
                    for local_start, start, base in zip(
                        source.local_starts,
                        intersection.starts,
                        source.region.starts,
                    )
                )
                canonical_destination_offsets = tuple(
                    start - base
                    for start, base in zip(
                        intersection.starts,
                        destination.region.starts,
                    )
                )
                destination_starts = tuple(
                    local_start + canonical_destination_offsets[axis]
                    for local_start, axis in zip(
                        destination.local_starts,
                        destination.physical_permutation,
                    )
                )
                entry = TransferEntry(
                    name=name,
                    dtype_name=source.dtype_name,
                    element_size=source.element_size,
                    source_starts=source_starts,
                    destination_starts=destination_starts,
                    lengths=intersection.lengths,
                    destination_name=destination.target_name,
                    source_name=source.source_key,
                    canonical_starts=intersection.starts,
                    destination_permutation=destination.physical_permutation,
                    destination_dtype_name=destination.dtype_name,
                    destination_element_size=destination.element_size,
                )
                route_entries.setdefault((source.source_rank, destination.tp_rank), []).append(entry)
                coverage[(name, destination.tp_rank)] = (
                    coverage.get((name, destination.tp_rank), 0) + entry.numel
                )
        for destination in destinations_by_name[name]:
            actual = coverage.get((name, destination.tp_rank), 0)
            if actual != destination.region.numel:
                raise ValueError(
                    f"Direct reshard plan covers {actual} values for {name!r} TP rank "
                    f"{destination.tp_rank}, expected {destination.region.numel}"
                )
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


__all__ = [
    "DestinationTensorLayout",
    "DirectReshardPlan",
    "PhysicalRolloutWorker",
    "SourceTensorLayout",
    "TensorRegion",
    "TransferBucket",
    "TransferEntry",
    "build_direct_reshard_plan",
    "aligned_offset",
    "describe_source_tensor",
    "intersect_regions",
    "resolve_destination_layouts",
    "resolve_physical_worker_topology",
    "resolve_source_layouts",
]
