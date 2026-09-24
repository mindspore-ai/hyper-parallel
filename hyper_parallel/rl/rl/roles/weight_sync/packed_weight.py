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
"""Whole-parameter buckets for vLLM load_weights publication."""

__all__ = [
    "PackedWeight",
    "PackedWeightAck",
    "PackedWeightBucket",
    "build_packed_weight_buckets",
    "materialize_packed_weight_bucket",
    "unpack_packed_weights",
]


from dataclasses import dataclass, replace
from math import prod
from typing import Any, Mapping, Optional

import torch
import torch.distributed as dist

from rl.roles.weight_sync.layout import local_tensor


@dataclass(frozen=True)
class PackedWeight:
    """Describe one complete HF parameter inside a packed byte buffer."""

    name: str
    dtype_name: str
    shape: tuple[int, ...]
    element_size: int
    buffer_offset: int = 0

    @property
    def num_bytes(self) -> int:
        """Return the serialized size of the complete parameter."""
        return prod(self.shape) * self.element_size

    def at_offset(self, offset: int) -> "PackedWeight":
        """Return this parameter assigned to one packed-buffer offset."""
        return replace(self, buffer_offset=int(offset))

    def worker_metadata(self) -> dict[str, Any]:
        """Serialize the contract needed to reconstruct the parameter view."""
        return {
            "name": self.name,
            "dtype_name": self.dtype_name,
            "shape": list(self.shape),
            "buffer_offset": self.buffer_offset,
            "num_bytes": self.num_bytes,
        }


@dataclass(frozen=True)
class PackedWeightBucket:
    """A batch of complete parameters bounded when each parameter fits."""

    entries: tuple[PackedWeight, ...]
    total_bytes: int

    def worker_metadata(self) -> list[dict[str, Any]]:
        """Return ordered metadata for vLLM worker reconstruction."""
        return [entry.worker_metadata() for entry in self.entries]


@dataclass(frozen=True)
class PackedWeightAck:
    """Confirm that every intended worker loaded one packed bucket."""

    bucket_index: int
    total_bytes: int
    worker_count: int


def _aligned_offset(offset: int, alignment: int) -> int:
    return ((offset + alignment - 1) // alignment) * alignment


def build_packed_weight_buckets(
    state_dict: Mapping[str, Any],
    bucket_size_bytes: int,
    *,
    skip_names: frozenset[str] = frozenset(),
) -> tuple[PackedWeightBucket, ...]:
    """Group complete parameters without splitting a vLLM load unit."""
    if bucket_size_bytes <= 0:
        raise ValueError("Packed weight bucket_size_bytes must be positive")
    entries = []
    for name, value in sorted(state_dict.items()):
        if name in skip_names:
            continue
        local_value = local_tensor(value)
        is_floating_point = getattr(local_value, "is_floating_point", None)
        if callable(is_floating_point) and not is_floating_point():
            continue
        shape = tuple(int(size) for size in value.shape)
        if not shape or any(size <= 0 for size in shape):
            raise ValueError(f"Packed weight {name!r} has invalid shape {shape}")
        entries.append(
            PackedWeight(
                name=name,
                dtype_name=str(value.dtype).rsplit(".", maxsplit=1)[-1],
                shape=shape,
                element_size=int(local_value.element_size()),
            )
        )
    if not entries:
        raise ValueError("Packed weight publication found no parameters")

    buckets = []
    current = []
    current_bytes = 0
    for entry in entries:
        offset = _aligned_offset(current_bytes, entry.element_size)
        if current and offset + entry.num_bytes > bucket_size_bytes:
            buckets.append(PackedWeightBucket(tuple(current), current_bytes))
            current = []
            current_bytes = 0
            offset = 0
        current.append(entry.at_offset(offset))
        current_bytes = offset + entry.num_bytes
        if entry.num_bytes >= bucket_size_bytes:
            buckets.append(PackedWeightBucket(tuple(current), current_bytes))
            current = []
            current_bytes = 0
    if current:
        buckets.append(PackedWeightBucket(tuple(current), current_bytes))
    return tuple(buckets)


def materialize_packed_weight_bucket(
    state_dict: Mapping[str, Any],
    bucket: PackedWeightBucket,
    *,
    producer_rank: int = 0,
) -> Optional[Any]:
    """Gather every complete parameter; only producer_rank packs it."""

    rank = dist.get_rank()
    tensors = []
    for entry in bucket.entries:
        value = state_dict.get(entry.name)
        if value is None:
            raise ValueError(f"Packed weight source {entry.name!r} is missing")
        full_tensor = getattr(value, "full_tensor", None)
        tensor = full_tensor() if callable(full_tensor) else value
        tensor = tensor.detach()
        if (
            tuple(int(size) for size in tensor.shape) != entry.shape
            or str(tensor.dtype).rsplit(".", maxsplit=1)[-1] != entry.dtype_name
            or int(tensor.element_size()) != entry.element_size
        ):
            raise ValueError(
                f"Packed weight {entry.name!r} differs from its bucket contract"
            )
        if rank == producer_rank:
            tensors.append(tensor.contiguous())
        else:
            del tensor
    if rank != producer_rank:
        return None
    packed = torch.empty(
        bucket.total_bytes,
        dtype=torch.uint8,
        device=tensors[0].device,
    )
    for entry, tensor in zip(bucket.entries, tensors):
        raw = tensor.view(torch.uint8).view(-1)
        if int(raw.numel()) != entry.num_bytes:
            raise ValueError(
                f"Packed weight {entry.name!r} has {raw.numel()} bytes, "
                f"expected {entry.num_bytes}"
            )
        packed.narrow(0, entry.buffer_offset, entry.num_bytes).copy_(raw)
    return packed


def unpack_packed_weights(
    packed: Any,
    metadata: list[Mapping[str, Any]],
) -> list[tuple[str, Any]]:
    """Reconstruct complete parameter views for model.load_weights()."""

    weights = []
    for entry in metadata:
        dtype = getattr(torch, str(entry["dtype_name"]))
        shape = tuple(int(size) for size in entry["shape"])
        num_bytes = int(entry["num_bytes"])
        offset = int(entry["buffer_offset"])
        if num_bytes != prod(shape) * dtype.itemsize:
            raise ValueError(f"Packed weight metadata is invalid for {entry['name']!r}")
        if offset < 0 or offset + num_bytes > int(packed.numel()):
            raise ValueError(f"Packed weight {entry['name']!r} exceeds its buffer")
        tensor = packed.narrow(0, offset, num_bytes).view(dtype).view(shape)
        rows = entry.get("canonical_rows")
        if rows is None:
            weights.append((str(entry["name"]), tensor))
        else:
            weights.extend(_unpack_canonical_rows(tensor, rows))
    return weights


def _unpack_canonical_rows(tensor: Any, rows: list[Mapping[str, Any]]) -> list[tuple[str, Any]]:
    """Restore canonical matrices from explicit disjoint physical row ranges."""
    if tensor.ndim != 2 or not rows:
        raise ValueError("Canonical row conversion requires a matrix and non-empty ranges")
    grouped = {}
    source_end = 0
    for row in sorted(rows, key=lambda item: int(item["source_start"])):
        start, length = int(row["source_start"]), int(row["length"])
        if start != source_end or length <= 0 or start + length > tensor.shape[0]:
            raise ValueError("Canonical row ranges must cover the fused weight exactly once")
        grouped.setdefault(str(row["name"]), []).append(row)
        source_end += length
    if source_end != tensor.shape[0]:
        raise ValueError("Canonical row ranges do not cover the fused weight")
    weights = []
    for name, parts in grouped.items():
        tensors = []
        target_end = 0
        target_rows = int(parts[0]["target_rows"])
        for part in sorted(parts, key=lambda item: int(item["target_start"])):
            if int(part["target_start"]) != target_end or int(part["target_rows"]) != target_rows:
                raise ValueError(f"Canonical row ranges for {name!r} overlap or have gaps")
            length = int(part["length"])
            tensors.append(tensor.narrow(0, int(part["source_start"]), length))
            target_end += length
        if target_end != target_rows:
            raise ValueError(f"Canonical row ranges for {name!r} are incomplete")
        weights.append((name, torch.cat(tensors, dim=0)))
    return weights
