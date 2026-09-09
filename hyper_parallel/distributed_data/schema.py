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
"""Stable contracts for sample-level distributed dynamic packing."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal


OversizedPolicy = Literal["error", "single"]


@dataclass(frozen=True)
class WorkloadCost:
    """Normalized sample costs used to balance multimodal work across DP ranks.

    The planner keeps stage costs separate so a user-supplied metadata callback
    can represent samples that stress different parts of a multimodal model.
    """

    io: float = 0.0
    transform: float = 0.0
    encoder: float = 0.0
    llm: float = 0.0
    memory: float = 0.0
    communication: float = 0.0

    def __post_init__(self) -> None:
        """Validate every cost component."""
        for name, value in self.as_dict().items():
            if (
                    not isinstance(value, (int, float))
                    or isinstance(value, bool)
                    or not math.isfinite(value)
                    or value < 0
            ):
                raise ValueError(f"WorkloadCost.{name} must be finite and non-negative, but got {value!r}.")

    def __add__(self, other: "WorkloadCost") -> "WorkloadCost":
        """Add corresponding stage costs."""
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
        """Return the largest stage cost."""
        return max(self.as_tuple())

    @property
    def total(self) -> float:
        """Return the sum of stage costs."""
        return sum(self.as_tuple())

    def as_tuple(self) -> tuple[float, ...]:
        """Return components in deterministic planner order."""
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
class SampleMetadata:
    """Lightweight metadata used to plan one Dataset sample.

    Args:
        pack_tokens: Tokens occupied by the sample in a packed sequence. This
            must use the same accounting as ``pack_fn``.
        cost: Optional normalized multimodal workload estimate.
        sample_id: Optional user-facing identifier used for diagnostics. The
            runtime uses :class:`SampleKey` for routing and uniqueness.
    """

    pack_tokens: int
    cost: WorkloadCost = WorkloadCost()
    sample_id: int | str | None = None

    def __post_init__(self) -> None:
        """Validate token count and optional identifier."""
        if not isinstance(self.pack_tokens, int) or isinstance(self.pack_tokens, bool) or self.pack_tokens < 1:
            raise ValueError(f"SampleMetadata.pack_tokens must be a positive integer, but got {self.pack_tokens!r}.")
        if not isinstance(self.cost, WorkloadCost):
            raise ValueError(f"SampleMetadata.cost must be WorkloadCost, but got {type(self.cost)}.")
        if self.sample_id is not None:
            valid_id = isinstance(self.sample_id, (int, str)) and not isinstance(self.sample_id, bool)
            if not valid_id or (isinstance(self.sample_id, int) and self.sample_id < 0):
                raise ValueError("SampleMetadata.sample_id must be a non-negative integer, non-empty string, or None.")
            if isinstance(self.sample_id, str) and not self.sample_id:
                raise ValueError("SampleMetadata.sample_id string must not be empty.")


@dataclass(frozen=True, order=True)
class SampleKey:
    """Stable identity for one Dataset sample occurrence owned by a Reader.

    ``global_sample_position`` distinguishes repeated indices in native
    BatchSampler mode. The ordinary unique-index stream keeps the default zero.
    """

    reader_rank: int
    dataset_index: int
    global_sample_position: int = 0

    def __post_init__(self) -> None:
        """Validate non-negative routing coordinates."""
        for name in ("reader_rank", "dataset_index", "global_sample_position"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"SampleKey.{name} must be a non-negative integer, but got {value!r}.")


@dataclass(frozen=True)
class BufferedSampleMetadata:
    """Planner-visible metadata for one sample owned by a Dataset Reader."""

    key: SampleKey
    metadata: SampleMetadata
    global_sample_position: int

    def __post_init__(self) -> None:
        """Validate the sample's position in the canonical epoch stream."""
        if (
                not isinstance(self.global_sample_position, int)
                or isinstance(self.global_sample_position, bool)
                or self.global_sample_position < 0
        ):
            raise ValueError(
                "BufferedSampleMetadata.global_sample_position must be a non-negative integer, "
                f"but got {self.global_sample_position!r}."
            )


@dataclass(frozen=True)
class StepSampleSelection:
    """Frozen sample membership and a known-feasible reference packing.

    ``samples`` contains exactly the samples admitted to one distributed step.
    ``reference_bins`` records the canonical streaming packing or native
    BatchSampler singleton grouping. Balanced placement may change those bins, but it must
    conserve every selected key exactly once.
    """

    samples: tuple[BufferedSampleMetadata, ...]
    reference_bins: tuple[tuple[SampleKey, ...], ...]

    def __post_init__(self) -> None:
        """Validate sample identity, stream order, and reference conservation."""
        if not self.samples:
            raise ValueError("StepSampleSelection.samples must not be empty.")
        if any(not isinstance(item, BufferedSampleMetadata) for item in self.samples):
            raise ValueError("StepSampleSelection.samples must contain BufferedSampleMetadata entries.")
        if not self.reference_bins or any(not packing_bin for packing_bin in self.reference_bins):
            raise ValueError("StepSampleSelection.reference_bins must contain non-empty bins.")
        keys = tuple(item.key for item in self.samples)
        if len(keys) != len(set(keys)):
            raise ValueError("StepSampleSelection samples must have unique SampleKey values.")
        positions = tuple(item.global_sample_position for item in self.samples)
        if positions != tuple(range(positions[0], positions[0] + len(positions))):
            raise ValueError("StepSampleSelection samples must be contiguous in canonical stream order.")
        reference_keys = tuple(key for packing_bin in self.reference_bins for key in packing_bin)
        if reference_keys != keys:
            raise ValueError(
                "StepSampleSelection reference bins must contain every selected sample exactly once "
                "in canonical stream order."
            )


@dataclass(frozen=True)
class PlannedSample:
    """Placement of one raw sample inside a target packed-sequence bin."""

    key: SampleKey
    metadata: SampleMetadata
    target_data_rank: int
    pack_index: int
    order: int


@dataclass(frozen=True)
class PackingBinPlan:
    """Ordered raw samples that one Data Constructor passes to ``pack_fn``."""

    pack_index: int
    samples: tuple[PlannedSample, ...]
    pack_tokens: int
    oversized: bool = False

    def __post_init__(self) -> None:
        """Validate one non-empty sequence-packing bin."""
        if not isinstance(self.pack_index, int) or isinstance(self.pack_index, bool) or self.pack_index < 0:
            raise ValueError(f"PackingBinPlan.pack_index must be non-negative, but got {self.pack_index!r}.")
        if not self.samples:
            raise ValueError("PackingBinPlan.samples must not be empty.")
        if not isinstance(self.pack_tokens, int) or isinstance(self.pack_tokens, bool) or self.pack_tokens < 1:
            raise ValueError(f"PackingBinPlan.pack_tokens must be positive, but got {self.pack_tokens!r}.")
        if not isinstance(self.oversized, bool):
            raise ValueError(f"PackingBinPlan.oversized must be boolean, but got {self.oversized!r}.")
        if tuple(sample.order for sample in self.samples) != tuple(range(len(self.samples))):
            raise ValueError("PackingBinPlan sample order must be contiguous from zero.")
        if any(sample.pack_index != self.pack_index for sample in self.samples):
            raise ValueError("PackingBinPlan samples must reference their containing pack_index.")


@dataclass(frozen=True)
class DataConstructorPlan:
    """All sequence bins assembled by one DP Data Constructor for one yield."""

    target_data_rank: int
    bins: tuple[PackingBinPlan, ...]
    cost: WorkloadCost

    @property
    def sample_keys(self) -> tuple[SampleKey, ...]:
        """Return planned sample keys in constructor order."""
        return tuple(sample.key for packing_bin in self.bins for sample in packing_bin.samples)


@dataclass(frozen=True)
class DistributedPackingPlan:
    """Deterministic sample-to-constructor plan for one distributed yield."""

    plan_id: str
    step: int
    seq_len: int
    local_batch_size: int
    data_parallel_size: int
    constructors: tuple[DataConstructorPlan, ...]

    def __post_init__(self) -> None:
        """Validate plan dimensions, slots, and unique sample assignments."""
        self._validate_dimensions()
        keys = self._validate_constructors()
        if len(keys) != len(set(keys)):
            raise ValueError("A sample key may appear only once in a distributed packing plan.")

    def _validate_dimensions(self) -> None:
        if not isinstance(self.plan_id, str) or not self.plan_id:
            raise ValueError("DistributedPackingPlan.plan_id must be a non-empty string.")
        if not isinstance(self.step, int) or isinstance(self.step, bool) or self.step < 0:
            raise ValueError(f"DistributedPackingPlan.step must be non-negative, but got {self.step!r}.")
        for name in ("seq_len", "local_batch_size", "data_parallel_size"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"DistributedPackingPlan.{name} must be positive, but got {value!r}.")
        if len(self.constructors) != self.data_parallel_size:
            raise ValueError(
                f"DistributedPackingPlan expected {self.data_parallel_size} constructors, "
                f"but got {len(self.constructors)}."
            )
        data_ranks = tuple(constructor.target_data_rank for constructor in self.constructors)
        if data_ranks != tuple(range(self.data_parallel_size)):
            raise ValueError("DistributedPackingPlan constructors must be ordered by contiguous data rank.")

    def _validate_constructors(self) -> list[SampleKey]:
        keys = []
        for constructor in self.constructors:
            self._validate_constructor(constructor)
            keys.extend(constructor.sample_keys)
        return keys

    def _validate_constructor(self, constructor: DataConstructorPlan) -> None:
        if len(constructor.bins) != self.local_batch_size:
            raise ValueError(
                f"Data rank {constructor.target_data_rank} expected {self.local_batch_size} bins, "
                f"but got {len(constructor.bins)}."
            )
        pack_indices = tuple(packing_bin.pack_index for packing_bin in constructor.bins)
        if pack_indices != tuple(range(self.local_batch_size)):
            raise ValueError("Data Constructor bins must be ordered by contiguous pack_index.")
        for packing_bin in constructor.bins:
            if packing_bin.pack_tokens > self.seq_len and not packing_bin.oversized:
                raise ValueError("A non-oversized packing bin exceeds seq_len.")
            if packing_bin.oversized and len(packing_bin.samples) != 1:
                raise ValueError("An oversized sample must occupy its packing bin alone.")

    @property
    def selected_keys(self) -> tuple[SampleKey, ...]:
        """Return all selected keys in deterministic constructor order."""
        return tuple(key for constructor in self.constructors for key in constructor.sample_keys)

    def constructor_for(self, data_rank: int) -> DataConstructorPlan:
        """Return the plan for one data-parallel rank.

        Args:
            data_rank: Rank within the data-parallel domain.

        Returns:
            Target Data Constructor plan.
        """
        valid_rank = isinstance(data_rank, int) and not isinstance(data_rank, bool)
        if not valid_rank or not 0 <= data_rank < self.data_parallel_size:
            raise ValueError(f"data_rank must be in [0, {self.data_parallel_size}), but got {data_rank!r}.")
        return self.constructors[data_rank]


@dataclass(frozen=True)
class ConstructedBatch:
    """Internal delivery envelope broadcast from a Data Constructor to MP peers."""

    step: int
    plan_id: str | None
    data: Any = None
    stopped: bool = False
    error: str | None = None
