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
"""Stable contracts for planned distributed data loading."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class WorkloadCost:
    """Normalized cost components used by the batch planner.

    The components intentionally remain separate because multimodal samples
    can stress different stages. Values are expected to be normalized by the
    selected cost model before the planner compares them.
    """

    io: float = 0.0
    transform: float = 0.0
    encoder: float = 0.0
    llm: float = 0.0
    memory: float = 0.0
    communication: float = 0.0

    def __post_init__(self) -> None:
        for name, value in self.as_dict().items():
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"WorkloadCost.{name} must be finite and non-negative, but got {value!r}.")

    def __add__(self, other: "WorkloadCost") -> "WorkloadCost":
        if not isinstance(other, WorkloadCost):
            return NotImplemented
        return WorkloadCost(
            io=self.io + other.io,
            transform=self.transform + other.transform,
            encoder=self.encoder + other.encoder,
            llm=self.llm + other.llm,
            memory=self.memory + other.memory,
            communication=self.communication + other.communication,
        )

    @property
    def dominant(self) -> float:
        """Return the largest normalized stage cost."""
        return max(self.as_tuple())

    @property
    def total(self) -> float:
        """Return the sum of all normalized stage costs."""
        return sum(self.as_tuple())

    def as_tuple(self) -> tuple[float, ...]:
        """Return components in their stable planner order."""
        return (self.io, self.transform, self.encoder, self.llm, self.memory, self.communication)

    def as_dict(self) -> dict[str, float]:
        """Return a serializable component mapping."""
        return {
            "io": self.io,
            "transform": self.transform,
            "encoder": self.encoder,
            "llm": self.llm,
            "memory": self.memory,
            "communication": self.communication,
        }


@dataclass(frozen=True)
class SampleMeta:
    """Lightweight sample information visible to the planner.

    ``data_ref`` identifies the map-style dataset entry. It must not contain
    decoded images, token tensors, or other heavyweight payloads.
    """

    sample_id: str
    source_id: str
    data_ref: int | str
    modality: str = "text"
    text_tokens: int = 0
    vision_tokens: int = 0
    audio_tokens: int = 0
    io_bytes: int = 0
    cost_hint: WorkloadCost | None = None

    def __post_init__(self) -> None:
        for name in ("sample_id", "source_id", "modality"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"SampleMeta.{name} must be a non-empty string, but got {value!r}.")
        if not isinstance(self.data_ref, (int, str)) or isinstance(self.data_ref, bool):
            raise ValueError(
                "SampleMeta.data_ref must be an integer index or string key, "
                f"but got {type(self.data_ref)}."
            )
        if isinstance(self.data_ref, int) and self.data_ref < 0:
            raise ValueError(f"SampleMeta.data_ref integer index must be non-negative, but got {self.data_ref}.")
        if isinstance(self.data_ref, str) and not self.data_ref:
            raise ValueError("SampleMeta.data_ref string key must not be empty.")
        for name in ("text_tokens", "vision_tokens", "audio_tokens", "io_bytes"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"SampleMeta.{name} must be a non-negative integer, but got {value!r}.")


@dataclass(frozen=True)
class TensorShardSpec:
    """Describe one tensor field that context parallelism shards.

    ``path`` addresses a leaf in a nested dict/list/tuple payload. For
    example, ``("input_ids",)`` targets a top-level field and
    ``("images", 0)`` targets the first element of ``images``.
    """

    path: tuple[str | int, ...]
    dim: int

    def __post_init__(self) -> None:
        if not self.path:
            raise ValueError("TensorShardSpec.path must not be empty.")
        if any(not isinstance(part, (str, int)) or isinstance(part, bool) for part in self.path):
            raise ValueError(f"TensorShardSpec.path must contain only strings or integers, but got {self.path}.")
        if not isinstance(self.dim, int) or isinstance(self.dim, bool):
            raise ValueError(f"TensorShardSpec.dim must be an integer, but got {self.dim!r}.")


@dataclass(frozen=True)
class PlannedSample:
    """Placement of one sample within an optimizer-step plan."""

    meta: SampleMeta
    source_position: int
    target_data_rank: int
    micro_batch_index: int
    position_in_micro_batch: int
    cost: WorkloadCost


@dataclass(frozen=True)
class BatchPlan:
    """Deterministic plan for every microbatch in one optimizer step."""

    replay_id: str
    step: int
    cursor_start: int
    cursor_end: int
    data_world_size: int
    micro_batch_size: int
    micro_batch_count: int
    samples: tuple[PlannedSample, ...]
    cp_shards: tuple[TensorShardSpec, ...] = ()

    def __post_init__(self) -> None:
        if not self.replay_id:
            raise ValueError("BatchPlan.replay_id must not be empty.")
        for name in ("step", "cursor_start", "cursor_end"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"BatchPlan.{name} must be a non-negative integer, but got {value!r}.")
        if self.cursor_end < self.cursor_start:
            raise ValueError(
                f"BatchPlan.cursor_end={self.cursor_end} must not precede cursor_start={self.cursor_start}."
            )
        for name in ("data_world_size", "micro_batch_size", "micro_batch_count"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"BatchPlan.{name} must be a positive integer, but got {value!r}.")
        expected = self.data_world_size * self.micro_batch_size * self.micro_batch_count
        if len(self.samples) != expected:
            raise ValueError(f"BatchPlan expected {expected} planned samples, but got {len(self.samples)}.")
        slot_positions: dict[tuple[int, int], set[int]] = {}
        for sample in self.samples:
            if sample.target_data_rank < 0 or sample.target_data_rank >= self.data_world_size:
                raise ValueError(f"Planned sample has invalid target_data_rank={sample.target_data_rank}.")
            if sample.micro_batch_index < 0 or sample.micro_batch_index >= self.micro_batch_count:
                raise ValueError(f"Planned sample has invalid micro_batch_index={sample.micro_batch_index}.")
            slot = (sample.target_data_rank, sample.micro_batch_index)
            slot_positions.setdefault(slot, set()).add(sample.position_in_micro_batch)
        expected_positions = set(range(self.micro_batch_size))
        if any(positions != expected_positions for positions in slot_positions.values()) or len(slot_positions) != (
            self.data_world_size * self.micro_batch_count
        ):
            raise ValueError(
                "Every BatchPlan (data_rank, microbatch) slot must contain each microbatch position exactly once."
            )

    def samples_for(self, data_rank: int, micro_batch_index: int) -> tuple[PlannedSample, ...]:
        """Return samples assigned to one data rank and microbatch."""
        if data_rank < 0 or data_rank >= self.data_world_size:
            raise ValueError(f"data_rank must be in [0, {self.data_world_size}), but got {data_rank}.")
        if micro_batch_index < 0 or micro_batch_index >= self.micro_batch_count:
            raise ValueError(
                f"micro_batch_index must be in [0, {self.micro_batch_count}), but got {micro_batch_index}."
            )
        selected = (
            sample
            for sample in self.samples
            if sample.target_data_rank == data_rank and sample.micro_batch_index == micro_batch_index
        )
        return tuple(sorted(selected, key=lambda sample: sample.position_in_micro_batch))


@dataclass(frozen=True)
class RankPayload:
    """Materialized payload consumed by one rank for one microbatch."""

    replay_id: str
    global_rank: int
    data_rank: int
    cp_rank: int
    micro_batch_index: int
    sample_ids: tuple[str, ...]
    data: Any


@dataclass(frozen=True)
class DistributedDataStep:
    """One optimizer step containing all local microbatch payloads."""

    plan: BatchPlan
    payloads: tuple[RankPayload, ...]

    def micro_batches(self) -> list[Any]:
        """Return payload data in microbatch execution order."""
        return [payload.data for payload in self.payloads]
