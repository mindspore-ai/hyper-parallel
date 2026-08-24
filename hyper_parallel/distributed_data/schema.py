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

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Callable, Iterator


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
    decoded images, token tensors, or other heavyweight sample data.
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

    ``path`` addresses a leaf in a nested dict/list/tuple microbatch. For
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
    """Deterministic plan for a contiguous optimizer-step microbatch window."""

    replay_id: str
    step: int
    cursor_start: int
    cursor_end: int
    data_world_size: int
    micro_batch_size: int
    micro_batch_count: int
    samples: tuple[PlannedSample, ...]
    cp_shards: tuple[TensorShardSpec, ...] = ()
    micro_batch_start: int = 0

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
        if (
            not isinstance(self.micro_batch_start, int)
            or isinstance(self.micro_batch_start, bool)
            or self.micro_batch_start < 0
        ):
            raise ValueError(
                f"BatchPlan.micro_batch_start must be a non-negative integer, but got {self.micro_batch_start!r}."
            )
        expected = self.data_world_size * self.micro_batch_size * self.micro_batch_count
        if len(self.samples) != expected:
            raise ValueError(f"BatchPlan expected {expected} planned samples, but got {len(self.samples)}.")
        slot_positions: dict[tuple[int, int], set[int]] = {}
        micro_batch_end = self.micro_batch_start + self.micro_batch_count
        for sample in self.samples:
            if sample.target_data_rank < 0 or sample.target_data_rank >= self.data_world_size:
                raise ValueError(f"Planned sample has invalid target_data_rank={sample.target_data_rank}.")
            if sample.micro_batch_index < self.micro_batch_start or sample.micro_batch_index >= micro_batch_end:
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
        micro_batch_end = self.micro_batch_start + self.micro_batch_count
        if micro_batch_index < self.micro_batch_start or micro_batch_index >= micro_batch_end:
            raise ValueError(
                f"micro_batch_index must be in [{self.micro_batch_start}, {micro_batch_end}), "
                f"but got {micro_batch_index}."
            )
        selected = (
            sample
            for sample in self.samples
            if sample.target_data_rank == data_rank and sample.micro_batch_index == micro_batch_index
        )
        return tuple(sorted(selected, key=lambda sample: sample.position_in_micro_batch))


@dataclass(frozen=True)
class RankMicroBatch:
    """Materialized microbatch and its planning-window replay ID for one rank."""

    replay_id: str
    global_rank: int
    data_rank: int
    cp_rank: int
    micro_batch_index: int
    sample_ids: tuple[str, ...]
    data: Any


class DistributedDataStep(Iterator[RankMicroBatch]):
    """One optimizer step that materializes local microbatches lazily."""

    def __init__(
        self,
        *,
        step: int,
        cursor_start: int,
        cursor_end: int,
        micro_batch_count: int,
        load_micro_batch: Callable[[int], tuple[BatchPlan, RankMicroBatch]],
        on_complete: Callable[["DistributedDataStep", str], None],
    ) -> None:
        """Initialize a single-use optimizer-step iterator.

        Args:
            step: Logical optimizer-step index.
            cursor_start: Per-owner candidate cursor before this step.
            cursor_end: Per-owner candidate cursor after this step.
            micro_batch_count: Number of local microbatches in the step.
            load_micro_batch: Runtime callback that produces one planned microbatch.
            on_complete: Callback invoked after the final microbatch is produced.
        """
        for name, value in (
            ("step", step),
            ("cursor_start", cursor_start),
            ("cursor_end", cursor_end),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer, but got {value!r}.")
        if cursor_end < cursor_start:
            raise ValueError(f"cursor_end={cursor_end} must not precede cursor_start={cursor_start}.")
        if not isinstance(micro_batch_count, int) or isinstance(micro_batch_count, bool) or micro_batch_count < 1:
            raise ValueError(f"micro_batch_count must be a positive integer, but got {micro_batch_count!r}.")
        if not callable(load_micro_batch) or not callable(on_complete):
            raise ValueError("load_micro_batch and on_complete must be callable.")
        self.step = step
        self.cursor_start = cursor_start
        self.cursor_end = cursor_end
        self.micro_batch_count = micro_batch_count
        self._load_micro_batch: Callable[[int], tuple[BatchPlan, RankMicroBatch]] | None = load_micro_batch
        self._on_complete: Callable[["DistributedDataStep", str], None] | None = on_complete
        self._micro_batch_index = 0
        self._plan_replay_ids: list[str] = []
        self._replay_id: str | None = None

    def __iter__(self) -> "DistributedDataStep":
        """Return this single-use microbatch iterator."""
        return self

    def __next__(self) -> RankMicroBatch:
        """Materialize and return the next local microbatch."""
        if self._micro_batch_index >= self.micro_batch_count:
            raise StopIteration
        if self._load_micro_batch is None:
            raise ValueError("DistributedDataStep is no longer attached to its dataset runtime.")
        plan, micro_batch = self._load_micro_batch(self._micro_batch_index)
        if micro_batch.micro_batch_index != self._micro_batch_index:
            raise ValueError(
                f"Expected microbatch {self._micro_batch_index}, "
                f"but runtime returned {micro_batch.micro_batch_index}."
            )
        self._plan_replay_ids.append(plan.replay_id)
        self._micro_batch_index += 1
        if self._micro_batch_index == self.micro_batch_count:
            self._replay_id = self._build_replay_id()
            on_complete = self._on_complete
            self._load_micro_batch = None
            self._on_complete = None
            if on_complete is None:
                raise ValueError("DistributedDataStep completion callback is unavailable.")
            on_complete(self, self._replay_id)
        return micro_batch

    @property
    def replay_id(self) -> str:
        """Return the optimizer-step replay ID after all microbatches are produced."""
        if self._replay_id is None:
            raise ValueError("Consume every microbatch before requesting the optimizer-step replay ID.")
        return self._replay_id

    @property
    def is_complete(self) -> bool:
        """Return whether every local microbatch has been produced."""
        return self._micro_batch_index == self.micro_batch_count

    def micro_batches(self) -> Iterator[RankMicroBatch]:
        """Return the single-use lazy microbatch iterator."""
        return self

    def _build_replay_id(self) -> str:
        stable_step = {
            "step": self.step,
            "cursor_start": self.cursor_start,
            "cursor_end": self.cursor_end,
            "plan_replay_ids": self._plan_replay_ids,
        }
        encoded = json.dumps(stable_step, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:24]
