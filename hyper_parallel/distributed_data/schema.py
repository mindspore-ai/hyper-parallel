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
from typing import Any, Callable, Iterator


@dataclass(frozen=True)
class WorkloadCost:
    """Normalized cost components used by the batch planner.

    The components intentionally remain separate because multimodal local batches
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
class LocalBatchMeta:
    """Lightweight local-batch information visible to the planner.

    ``local_batch_id`` identifies one complete input consumed by one data rank
    for one forward/backward pass. Metadata must not contain the heavyweight
    local-batch payload.
    """

    local_batch_id: int | str
    text_tokens: int = 0
    vision_tokens: int = 0
    audio_tokens: int = 0
    io_bytes: int = 0
    cost_hint: WorkloadCost | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.local_batch_id, (int, str)) or isinstance(self.local_batch_id, bool):
            raise ValueError(
                "LocalBatchMeta.local_batch_id must be an integer index or string key, "
                f"but got {type(self.local_batch_id)}."
            )
        if isinstance(self.local_batch_id, int) and self.local_batch_id < 0:
            raise ValueError(
                "LocalBatchMeta.local_batch_id integer index must be non-negative, "
                f"but got {self.local_batch_id}."
            )
        if isinstance(self.local_batch_id, str) and not self.local_batch_id:
            raise ValueError("LocalBatchMeta.local_batch_id string key must not be empty.")
        for name in ("text_tokens", "vision_tokens", "audio_tokens", "io_bytes"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"LocalBatchMeta.{name} must be a non-negative integer, but got {value!r}.")


@dataclass(frozen=True)
class TensorLocalBatchSpec:
    """Tensor layout metadata synchronized with online local-batch metadata."""

    shape: tuple[int, ...]
    dtype: str
    numel: int

    def __post_init__(self) -> None:
        if not isinstance(self.shape, tuple) or any(
            not isinstance(size, int) or isinstance(size, bool) or size < 0
            for size in self.shape
        ):
            raise ValueError(
                f"TensorLocalBatchSpec.shape must contain non-negative integers, but got {self.shape!r}."
            )
        if not isinstance(self.dtype, str) or not self.dtype:
            raise ValueError(f"TensorLocalBatchSpec.dtype must be a non-empty string, but got {self.dtype!r}.")
        expected_numel = math.prod(self.shape)
        if (
            not isinstance(self.numel, int)
            or isinstance(self.numel, bool)
            or self.numel != expected_numel
        ):
            raise ValueError(
                f"TensorLocalBatchSpec.numel must equal shape product {expected_numel}, but got {self.numel!r}."
            )


@dataclass(frozen=True)
class OnlineLocalBatchMetadata:
    """Online local-batch metadata with an optional tensor transport descriptor."""

    local_batch_meta: LocalBatchMeta
    tensor_spec: TensorLocalBatchSpec | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.local_batch_meta, LocalBatchMeta):
            raise ValueError(
                "OnlineLocalBatchMetadata.local_batch_meta must be LocalBatchMeta, "
                f"but got {type(self.local_batch_meta)}."
            )
        if self.tensor_spec is not None and not isinstance(self.tensor_spec, TensorLocalBatchSpec):
            raise ValueError(
                "OnlineLocalBatchMetadata.tensor_spec must be TensorLocalBatchSpec or None, "
                f"but got {type(self.tensor_spec)}."
            )


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
class PlannedLocalBatch:
    """Placement of one local batch within an optimizer-step plan."""

    meta: LocalBatchMeta
    source_position: int
    target_data_rank: int
    micro_batch_index: int
    cost: WorkloadCost


@dataclass(frozen=True)
class BatchPlan:
    """Deterministic local-batch placement for one optimizer step."""

    plan_id: str
    step: int
    local_batch_offset_start: int
    local_batch_offset_end: int
    data_parallel_size: int
    micro_batch_num: int
    local_batches: tuple[PlannedLocalBatch, ...]
    cp_shards: tuple[TensorShardSpec, ...] = ()
    micro_batch_start: int = 0

    def __post_init__(self) -> None:
        if not self.plan_id:
            raise ValueError("BatchPlan.plan_id must not be empty.")
        for name in ("step", "local_batch_offset_start", "local_batch_offset_end"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"BatchPlan.{name} must be a non-negative integer, but got {value!r}.")
        if self.local_batch_offset_end < self.local_batch_offset_start:
            raise ValueError(
                f"BatchPlan.local_batch_offset_end={self.local_batch_offset_end} must not precede "
                f"local_batch_offset_start={self.local_batch_offset_start}."
            )
        for name in ("data_parallel_size", "micro_batch_num"):
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
        expected = self.data_parallel_size * self.micro_batch_num
        if len(self.local_batches) != expected:
            raise ValueError(
                f"BatchPlan expected {expected} planned local batches, but got {len(self.local_batches)}."
            )
        slots = set()
        micro_batch_end = self.micro_batch_start + self.micro_batch_num
        for local_batch in self.local_batches:
            if local_batch.target_data_rank < 0 or local_batch.target_data_rank >= self.data_parallel_size:
                raise ValueError(
                    f"Planned local batch has invalid target_data_rank={local_batch.target_data_rank}."
                )
            if (
                local_batch.micro_batch_index < self.micro_batch_start
                or local_batch.micro_batch_index >= micro_batch_end
            ):
                raise ValueError(
                    f"Planned local batch has invalid micro_batch_index={local_batch.micro_batch_index}."
                )
            slots.add((local_batch.target_data_rank, local_batch.micro_batch_index))
        if len(slots) != expected:
            raise ValueError(
                "Every BatchPlan (data_rank, microbatch) slot must contain exactly one local batch."
            )

    def local_batch_for(self, data_rank: int, micro_batch_index: int) -> PlannedLocalBatch:
        """Return the local batch assigned to one data-rank execution slot."""
        if data_rank < 0 or data_rank >= self.data_parallel_size:
            raise ValueError(f"data_rank must be in [0, {self.data_parallel_size}), but got {data_rank}.")
        micro_batch_end = self.micro_batch_start + self.micro_batch_num
        if micro_batch_index < self.micro_batch_start or micro_batch_index >= micro_batch_end:
            raise ValueError(
                f"micro_batch_index must be in [{self.micro_batch_start}, {micro_batch_end}), "
                f"but got {micro_batch_index}."
            )
        selected = [
            local_batch
            for local_batch in self.local_batches
            if local_batch.target_data_rank == data_rank and local_batch.micro_batch_index == micro_batch_index
        ]
        if len(selected) != 1:
            raise ValueError(
                f"BatchPlan slot ({data_rank}, {micro_batch_index}) expected one local batch, got {len(selected)}."
            )
        return selected[0]


@dataclass(frozen=True)
class LocalBatch:
    """One rank-local forward/backward input and its plan identity."""

    plan_id: str
    global_rank: int
    data_rank: int
    cp_rank: int
    micro_batch_index: int
    local_batch_id: int | str
    data: Any


class DistributedDataStep(Iterator[LocalBatch]):
    """One optimizer step that fetches local batches lazily."""

    def __init__(
        self,
        *,
        step: int,
        local_batch_offset_start: int,
        local_batch_offset_end: int,
        micro_batch_num: int,
        load_local_batch: Callable[[int], tuple[BatchPlan, LocalBatch]],
        on_complete: Callable[["DistributedDataStep", str], None],
    ) -> None:
        """Initialize a single-use optimizer-step iterator.

        Args:
            step: Logical optimizer-step index.
            local_batch_offset_start: Inclusive per-owner local-batch offset.
            local_batch_offset_end: Exclusive per-owner local-batch offset.
            micro_batch_num: Number of local batches consumed by each data rank.
            load_local_batch: Runtime callback that produces one planned local batch.
            on_complete: Callback invoked after the final microbatch is produced.
        """
        for name, value in (
            ("step", step),
            ("local_batch_offset_start", local_batch_offset_start),
            ("local_batch_offset_end", local_batch_offset_end),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer, but got {value!r}.")
        if local_batch_offset_end < local_batch_offset_start:
            raise ValueError(
                f"local_batch_offset_end={local_batch_offset_end} must not precede "
                f"local_batch_offset_start={local_batch_offset_start}."
            )
        if not isinstance(micro_batch_num, int) or isinstance(micro_batch_num, bool) or micro_batch_num < 1:
            raise ValueError(f"micro_batch_num must be a positive integer, but got {micro_batch_num!r}.")
        if not callable(load_local_batch) or not callable(on_complete):
            raise ValueError("load_local_batch and on_complete must be callable.")
        self.step = step
        self.local_batch_offset_start = local_batch_offset_start
        self.local_batch_offset_end = local_batch_offset_end
        self.micro_batch_num = micro_batch_num
        self._load_local_batch: Callable[[int], tuple[BatchPlan, LocalBatch]] | None = load_local_batch
        self._on_complete: Callable[["DistributedDataStep", str], None] | None = on_complete
        self._micro_batch_index = 0
        self._micro_batch_plan_ids: list[str] = []
        self._plan_id: str | None = None

    def __iter__(self) -> "DistributedDataStep":
        """Return this single-use microbatch iterator."""
        return self

    def __next__(self) -> LocalBatch:
        """Fetch and return the next local batch."""
        if self._micro_batch_index >= self.micro_batch_num:
            raise StopIteration
        if self._load_local_batch is None:
            raise ValueError("DistributedDataStep is no longer attached to its dataset runtime.")
        plan, local_batch = self._load_local_batch(self._micro_batch_index)
        if local_batch.micro_batch_index != self._micro_batch_index:
            raise ValueError(
                f"Expected microbatch {self._micro_batch_index}, "
                f"but runtime returned {local_batch.micro_batch_index}."
            )
        self._micro_batch_plan_ids.append(plan.plan_id)
        self._micro_batch_index += 1
        if self._micro_batch_index == self.micro_batch_num:
            self._plan_id = self._build_plan_id()
            on_complete = self._on_complete
            self._load_local_batch = None
            self._on_complete = None
            if on_complete is None:
                raise ValueError("DistributedDataStep completion callback is unavailable.")
            on_complete(self, self._plan_id)
        return local_batch

    @property
    def plan_id(self) -> str:
        """Return the optimizer-step plan ID after all microbatches are produced."""
        if self._plan_id is None:
            raise ValueError("Consume every microbatch before requesting the optimizer-step plan ID.")
        return self._plan_id

    @property
    def is_complete(self) -> bool:
        """Return whether every local microbatch has been produced."""
        return self._micro_batch_index == self.micro_batch_num

    def local_batches(self) -> Iterator[LocalBatch]:
        """Return the single-use lazy local-batch iterator."""
        return self

    def _build_plan_id(self) -> str:
        plan_ids = set(self._micro_batch_plan_ids)
        if len(plan_ids) != 1:
            raise ValueError(f"One optimizer step must use one whole-step plan, but got {sorted(plan_ids)}.")
        return self._micro_batch_plan_ids[0]
