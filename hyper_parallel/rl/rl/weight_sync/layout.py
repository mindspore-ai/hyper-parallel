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
"""Source, destination, and transfer layout contracts."""


from dataclasses import dataclass, replace
from math import prod
from typing import Any, Mapping, NamedTuple, Optional, Sequence


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
    worker_rank: Optional[int] = None

    @property
    def route_rank(self) -> int:
        """Return the physical worker route, or the shared dense TP route."""
        return self.tp_rank if self.worker_rank is None else self.worker_rank

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


class _DestinationSignature(NamedTuple):
    """TP-invariant destination metadata used to compare worker descriptions."""

    placement: str
    shard_dim: Any
    destination_name: str
    dtype_name: str
    element_size: int
    permutation: tuple[int, ...]
    accepted_source_dtypes: tuple[str, ...]


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
    """A cached source-rank plan indexed by shared TP or physical worker route."""

    source_world_size: int
    destination_tp_size: int
    bucket_size_bytes: int
    buckets: Mapping[tuple[int, int], tuple[TransferBucket, ...]]
    destination_worker_size: Optional[int] = None

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


def _mesh_source_region(
    name: str, placements: tuple[Any, ...], device_mesh: Any,
    global_shape: tuple[int, ...], local_shape: tuple[int, ...],
) -> list[int]:
    """Recover global offsets from inner TP placement to outer FSDP placement."""
    coordinate = device_mesh.get_coordinate()
    if coordinate is None or len(coordinate) != len(placements):
        raise ValueError(f"Direct reshard source tensor {name!r} has no complete mesh coordinate")
    starts = [0] * len(global_shape)
    lengths = list(global_shape)
    for mesh_dim in reversed(range(len(placements))):
        placement = placements[mesh_dim]
        if not callable(getattr(placement, "is_shard", None)) or not placement.is_shard():
            continue
        tensor_dim = int(placement.dim)
        mesh_rank = int(coordinate[mesh_dim])
        base, remainder = divmod(lengths[tensor_dim], int(device_mesh.size(mesh_dim)))
        starts[tensor_dim] += mesh_rank * base + min(mesh_rank, remainder)
        lengths[tensor_dim] = base + int(mesh_rank < remainder)
    if tuple(lengths) != local_shape:
        raise ValueError(
            f"Direct reshard source tensor {name!r} mesh-derived shape differs from local: "
            f"derived={tuple(lengths)}, local={local_shape}, placements={placements}"
        )
    return starts



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
        placement for placement in placements
        if callable(getattr(placement, "is_shard", None)) and placement.is_shard()
    ]
    if not shard_placements:
        description["shard_dim"] = None
        description["region_starts"] = [0] * len(global_shape)
        return description
    device_mesh = getattr(tensor, "device_mesh", None)
    if device_mesh is None or len(placements) != int(device_mesh.ndim):
        if len(shard_placements) == 1:
            description["shard_dim"] = int(shard_placements[0].dim)
            return description
        raise ValueError(
            f"Direct reshard source tensor {name!r} requires mesh metadata for "
            f"multi-axis placements={placements}"
        )
    description["shard_dim"] = None
    description["region_starts"] = _mesh_source_region(name, placements, device_mesh, global_shape, local_shape)
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
        for (starts, lengths), description in unique_regions.items():
            layouts.append(SourceTensorLayout(
                name,
                dtype_name,
                element_size,
                global_shape,
                int(description["source_rank"]),
                TensorRegion(starts, lengths),
                str(description.get("source_name", name)),
                tuple(int(value) for value in description.get("source_starts", [0] * len(global_shape))),
            ))
    return tuple(layouts)


def _validate_source_coverage(name, regions, global_shape):
    """Reject overlapping or incomplete source regions before planning transfers."""
    for index, left in enumerate(regions):
        for right in regions[index + 1:]:
            if _regions_overlap(left, right):
                raise ValueError(f"Direct reshard source tensor {name!r} regions overlap")
    if sum(region.numel for region in regions) != prod(global_shape):
        raise ValueError(f"Direct reshard source tensor {name!r} regions do not cover global shape")


def _regions_overlap(left: TensorRegion, right: TensorRegion) -> bool:
    """Return whether two axis-aligned regions intersect on every axis."""
    for left_start, left_end, right_start, right_end in zip(left.starts, left.ends, right.starts, right.ends):
        if max(left_start, right_start) >= min(left_end, right_end):
            return False
    return True


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
) -> _DestinationSignature:
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
    return _DestinationSignature(
        placement=str(tensor["placement"]),
        shard_dim=tensor.get("shard_dim"),
        destination_name=str(tensor.get("destination_name", name)),
        dtype_name=str(tensor["dtype_name"]),
        element_size=int(tensor["element_size"]),
        permutation=permutation,
        accepted_source_dtypes=tuple(str(value) for value in tensor.get("accepted_source_dtypes", ())),
    )


def resolve_destination_layouts(
    worker_descriptions: Sequence[Mapping[str, Any]],
    global_shapes: Mapping[str, tuple[int, ...]],
) -> tuple[DestinationTensorLayout, ...]:
    """Resolve per-worker TP metadata into Actor-coordinate destination regions."""
    if not worker_descriptions:
        raise ValueError("Direct reshard rollout layout returned no TP workers")
    if any("worker_rank" in worker for worker in worker_descriptions):
        return _resolve_worker_destinations(worker_descriptions, global_shapes)
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


def _validate_worker_coordinates(workers: Sequence[Mapping[str, Any]]) -> None:
    """Require a complete DP-major physical worker numbering."""
    ranks = [int(worker["worker_rank"]) for worker in workers]
    if sorted(ranks) != list(range(len(workers))):
        raise ValueError("Direct reshard physical worker ranks must be dense and unique")
    tp_sizes = {int(worker["tp_size"]) for worker in workers}
    if len(tp_sizes) != 1 or min(tp_sizes) <= 0:
        raise ValueError("Direct reshard physical workers require one positive TP size")
    for worker in workers:
        tp_rank, tp_size = int(worker["tp_rank"]), int(worker["tp_size"])
        dp_rank = int(worker["dp_rank"])
        if not 0 <= tp_rank < tp_size or dp_rank < 0 or int(worker["worker_rank"]) != dp_rank * tp_size + tp_rank:
            raise ValueError("Direct reshard physical worker coordinates must use DP-major order")


def _explicit_worker_destination(tensor, shape, worker):
    """Resolve one expert tensor region in its physical worker storage."""
    name = str(tensor["name"])
    starts = tuple(int(value) for value in tensor["region_starts"])
    lengths = tuple(int(value) for value in tensor["local_shape"])
    if (len(starts) != len(shape) or len(lengths) != len(shape) or any(
            start < 0 or length <= 0 or start + length > size
            for start, length, size in zip(starts, lengths, shape))):
        raise ValueError(f"Rollout parameter {name!r} has invalid explicit region")
    signature = _destination_signature(tensor, name, len(shape))
    return DestinationTensorLayout(
        name, signature[3], signature[4], shape, int(worker["tp_rank"]), int(worker["tp_size"]),
        TensorRegion(starts, lengths), signature[2],
        tuple(tensor.get("destination_starts", [0] * len(shape))), signature[5], signature[6],
    )


def _worker_tp_destination(name, shape, worker, workers):
    """Retain ordinary attention TP layout within each physical DP group."""
    peers = sorted(
        (peer for peer in workers if int(peer["dp_rank"]) == int(worker["dp_rank"])),
        key=lambda peer: int(peer["tp_rank"]),
    )
    tensors = [next((item for item in peer["tensors"] if item["name"] == name), None) for peer in peers]
    tp_size = int(worker["tp_size"])
    if len(peers) != tp_size or any(item is None for item in tensors):
        raise ValueError(f"Rollout parameter {name!r} has incomplete TP coverage")
    return _parameter_destination_layouts(name, tensors, shape, tp_size)[int(worker["tp_rank"])]


def _resolve_worker_destinations(workers, global_shapes):
    """Resolve physical worker routes, allowing expert subsets to differ."""
    _validate_worker_coordinates(workers)
    layouts = []
    for worker in workers:
        tensors = worker["tensors"]
        if len({tensor["name"] for tensor in tensors}) != len(tensors):
            raise ValueError("Direct reshard worker has duplicate tensor descriptions")
        for tensor in tensors:
            name = str(tensor["name"])
            if name not in global_shapes:
                raise ValueError(f"Rollout parameter {name!r} is absent from the Actor policy")
            shape = global_shapes[name]
            if "region_starts" in tensor:
                layout = _explicit_worker_destination(tensor, shape, worker)
            else:
                layout = _worker_tp_destination(name, shape, worker, workers)
            layouts.append(replace(layout, worker_rank=int(worker["worker_rank"])))
    return tuple(layouts)


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
    if signature.placement == "shard":
        if signature.shard_dim is None:
            raise ValueError(f"Sharded rollout parameter {name!r} has no shard_dim")
        resolved_shard_dim = int(signature.shard_dim)
        shard_offsets = _destination_shard_offsets(
            name,
            tensors,
            global_shape,
            resolved_shard_dim,
        )
    else:
        resolved_shard_dim = signature.shard_dim
        shard_offsets = ()
    for tp_rank, tensor in enumerate(tensors):
        local_shape = tuple(int(size) for size in tensor["local_shape"])
        starts = _destination_region_starts(
            name, signature.placement, local_shape, global_shape, resolved_shard_dim,
            shard_offsets[tp_rank] if signature.placement == "shard" else None,
        )
        offset_values = tensor.get("destination_starts", [0] * len(global_shape))
        destination_starts = tuple(int(value) for value in offset_values)
        if len(destination_starts) != len(global_shape):
            raise ValueError(
                f"Rollout parameter {name!r} destination offset rank mismatch: "
                f"offset={destination_starts}, global_shape={global_shape}"
            )
        layouts.append(
            DestinationTensorLayout(
                name,
                signature.dtype_name,
                signature.element_size,
                global_shape,
                tp_rank,
                tp_size,
                TensorRegion(starts, local_shape),
                signature.destination_name,
                destination_starts,
                signature.permutation,
                signature.accepted_source_dtypes,
            )
        )
    return layouts
