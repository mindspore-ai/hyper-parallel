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
from typing import Literal


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
class PackingBinPlan:
    """One ordered packed sequence assigned to a Data Constructor."""

    sample_keys: tuple[SampleKey, ...]
    pack_tokens: int
    oversized: bool = False

    def __post_init__(self) -> None:
        """Validate one non-empty sequence-packing bin."""
        if not self.sample_keys:
            raise ValueError("PackingBinPlan.sample_keys must not be empty.")
        if any(not isinstance(key, SampleKey) for key in self.sample_keys):
            raise ValueError("PackingBinPlan.sample_keys must contain SampleKey values.")
        if len(self.sample_keys) != len(set(self.sample_keys)):
            raise ValueError("PackingBinPlan.sample_keys must not contain duplicates.")
        if not isinstance(self.pack_tokens, int) or isinstance(self.pack_tokens, bool) or self.pack_tokens < 1:
            raise ValueError(f"PackingBinPlan.pack_tokens must be positive, but got {self.pack_tokens!r}.")
        if not isinstance(self.oversized, bool):
            raise ValueError(f"PackingBinPlan.oversized must be boolean, but got {self.oversized!r}.")


@dataclass(frozen=True)
class DistributedPackingPlan:
    """Deterministic sample-to-constructor plan for one distributed yield."""

    plan_id: str
    step: int
    seq_len: int
    local_batches: tuple[tuple[PackingBinPlan, ...], ...]
    rank_costs: tuple[WorkloadCost, ...]

    def __post_init__(self) -> None:
        """Validate plan dimensions, slots, and unique sample assignments."""
        self._validate_dimensions()
        keys = self._validate_local_batches()
        if len(keys) != len(set(keys)):
            raise ValueError("A sample key may appear only once in a distributed packing plan.")

    def _validate_dimensions(self) -> None:
        if not isinstance(self.plan_id, str) or not self.plan_id:
            raise ValueError("DistributedPackingPlan.plan_id must be a non-empty string.")
        if not isinstance(self.step, int) or isinstance(self.step, bool) or self.step < 0:
            raise ValueError(f"DistributedPackingPlan.step must be non-negative, but got {self.step!r}.")
        if not isinstance(self.seq_len, int) or isinstance(self.seq_len, bool) or self.seq_len < 1:
            raise ValueError(f"DistributedPackingPlan.seq_len must be positive, but got {self.seq_len!r}.")
        if not self.local_batches:
            raise ValueError("DistributedPackingPlan.local_batches must not be empty.")
        if len(self.rank_costs) != len(self.local_batches):
            raise ValueError("DistributedPackingPlan.rank_costs must match local_batches.")
        if any(not isinstance(cost, WorkloadCost) for cost in self.rank_costs):
            raise ValueError("DistributedPackingPlan.rank_costs must contain WorkloadCost values.")
        batch_sizes = {len(local_batch) for local_batch in self.local_batches}
        if len(batch_sizes) != 1 or 0 in batch_sizes:
            raise ValueError("DistributedPackingPlan local batches must have equal non-zero sizes.")

    @property
    def data_parallel_size(self) -> int:
        """Return the number of target Data Constructor ranks."""
        return len(self.local_batches)

    @property
    def local_batch_size(self) -> int:
        """Return the number of packed sequences per target rank."""
        size = len(self.local_batches[0])
        if any(len(local_batch) != size for local_batch in self.local_batches):
            raise ValueError("DistributedPackingPlan local batches must have equal sizes.")
        return size

    def _validate_local_batches(self) -> list[SampleKey]:
        keys = []
        for local_batch in self.local_batches:
            self._validate_local_batch(local_batch)
            keys.extend(key for packing_bin in local_batch for key in packing_bin.sample_keys)
        return keys

    def _validate_local_batch(self, local_batch: tuple[PackingBinPlan, ...]) -> None:
        if any(not isinstance(packing_bin, PackingBinPlan) for packing_bin in local_batch):
            raise ValueError("DistributedPackingPlan local batches must contain PackingBinPlan values.")
        if len(local_batch) < 1:
            raise ValueError(
                "DistributedPackingPlan local batches must contain at least one packing bin."
            )
        for packing_bin in local_batch:
            if packing_bin.pack_tokens > self.seq_len and not packing_bin.oversized:
                raise ValueError("A non-oversized packing bin exceeds seq_len.")
            if packing_bin.oversized and len(packing_bin.sample_keys) != 1:
                raise ValueError("An oversized sample must occupy its packing bin alone.")

    @property
    def selected_keys(self) -> tuple[SampleKey, ...]:
        """Return all selected keys in deterministic constructor order."""
        return tuple(
            key
            for local_batch in self.local_batches
            for packing_bin in local_batch
            for key in packing_bin.sample_keys
        )

    def local_batch_for(self, data_rank: int) -> tuple[PackingBinPlan, ...]:
        """Return the ordered packed sequences for one data-parallel rank.

        Args:
            data_rank: Rank within the data-parallel domain.

        Returns:
            Target rank's local packing bins.
        """
        valid_rank = isinstance(data_rank, int) and not isinstance(data_rank, bool)
        if not valid_rank or not 0 <= data_rank < self.data_parallel_size:
            raise ValueError(f"data_rank must be in [0, {self.data_parallel_size}), but got {data_rank!r}.")
        return self.local_batches[data_rank]

    def local_sample_keys(self, data_rank: int) -> tuple[SampleKey, ...]:
        """Return the sample keys assigned to one target data rank.

        Args:
            data_rank: Rank within the data-parallel domain.

        Returns:
            Sample keys in local pack order.
        """
        return tuple(
            key
            for packing_bin in self.local_batch_for(data_rank)
            for key in packing_bin.sample_keys
        )
