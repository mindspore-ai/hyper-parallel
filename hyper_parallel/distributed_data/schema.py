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
from collections.abc import Mapping, Sequence
from dataclasses import InitVar, dataclass, field
from typing import Any, Literal


OversizedPolicy = Literal["error", "single"]


def _validate_nonnegative_costs(values: Mapping[str, float], name: str) -> None:
    """Validate named finite, non-negative scalar costs or budgets."""
    if not isinstance(values, Mapping):
        raise ValueError(f"{name} must be a mapping of non-empty strings to finite, non-negative numbers.")
    for key, value in values.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"{name} keys must be non-empty strings, but got {key!r}.")
        if (
                type(value) not in (int, float)
                or (isinstance(value, float) and not math.isfinite(value))
                or value < 0
        ):
            raise ValueError(f"{name}[{key!r}] must be finite and non-negative, but got {value!r}.")


def _validate_feature(value: Any, name: str, ancestors: set[int]) -> None:
    """Reject tensors, arbitrary objects, cycles, and non-finite feature values."""
    if value is None:
        return
    if type(value) not in (bool, int, float, str, dict, list, tuple):
        raise ValueError(f"{name} must contain only CPU-serializable basic types, but got {type(value)}.")
    if isinstance(value, (bool, int, str)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite, but got {value!r}.")
        return
    if id(value) in ancestors:
        raise ValueError(f"{name} must not contain reference cycles.")
    ancestors.add(id(value))
    if isinstance(value, dict):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{name} keys must be strings, but got {key!r}.")
            _validate_feature(child, f"{name}[{key!r}]", ancestors)
    else:
        for index, child in enumerate(value):
            _validate_feature(child, f"{name}[{index}]", ancestors)
    ancestors.remove(id(value))


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
        features: CPU metadata for workload estimation. Values may recursively
            contain ``None``, booleans, integers, finite floats, strings, lists,
            tuples, and dictionaries with string keys. Tensors are rejected.
        packing_costs: Named finite, non-negative physical packing footprints.
            These are independent of ``cost`` and enforce per-bin hard budgets.
    """

    pack_tokens: int
    cost: WorkloadCost = WorkloadCost()
    sample_id: int | str | None = None
    features: dict[str, Any] = field(default_factory=dict)
    packing_costs: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate identity, workload, and lightweight packing metadata."""
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
        if not isinstance(self.features, dict):
            raise ValueError("SampleMetadata.features must be a dictionary.")
        _validate_feature(self.features, "SampleMetadata.features", set())
        if not isinstance(self.packing_costs, dict):
            raise ValueError("SampleMetadata.packing_costs must be a dictionary.")
        _validate_nonnegative_costs(self.packing_costs, "SampleMetadata.packing_costs")


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
class PackingConstraints:
    """Shared packing feasibility contract for selection and balanced placement.

    Args:
        seq_len: Token capacity of a regular bin.
        oversized_policy: Permit token overflow only for a singleton when
            ``single``. Named stage budgets never permit singleton overflow.
        packing_budgets: Per-bin additive hard caps. Every configured name must
            be explicitly present in each sample's ``packing_costs``, including
            zero-cost stages. ``None`` retains token-only packing.
    """

    seq_len: int
    oversized_policy: OversizedPolicy = "error"
    packing_budgets: Mapping[str, float] | None = None

    def __post_init__(self) -> None:
        """Validate token and stage capacities and copy the caller's mapping."""
        if not isinstance(self.seq_len, int) or isinstance(self.seq_len, bool) or self.seq_len < 1:
            raise ValueError(f"seq_len must be a positive integer, but got {self.seq_len!r}.")
        if self.oversized_policy not in ("error", "single"):
            raise ValueError("oversized_policy must be 'error' or 'single'.")
        budgets = {} if self.packing_budgets is None else self.packing_budgets
        _validate_nonnegative_costs(budgets, "packing_budgets")
        object.__setattr__(self, "packing_budgets", dict(budgets))

    def validate_sample(self, item: BufferedSampleMetadata) -> None:
        """Reject samples that cannot occupy even an otherwise empty bin."""
        for name, budget in self.packing_budgets.items():
            if name not in item.metadata.packing_costs:
                raise ValueError(f"Sample {item.key} is missing packing_costs[{name!r}] required by packing_budgets.")
            value = item.metadata.packing_costs[name]
            if value > budget:
                raise ValueError(
                    f"Sample {item.key} requires packing_costs[{name!r}]={value}, "
                    f"exceeding packing_budgets[{name!r}]={budget}. Stage budgets do not permit singleton overflow."
                )
        if item.metadata.pack_tokens > self.seq_len and self.oversized_policy == "error":
            raise ValueError(
                f"Sample {item.key} requires {item.metadata.pack_tokens} tokens, exceeding seq_len={self.seq_len}. "
                "Set oversized_policy='single' only when the packer supports singleton overflow."
            )

    def fits(
            self,
            pack_tokens: int,
            packing_costs: Mapping[str, float],
            item: BufferedSampleMetadata,
    ) -> bool:
        """Return whether an individually valid sample fits the current bin."""
        if item.metadata.pack_tokens > self.seq_len:
            token_fit = pack_tokens == 0 and self.oversized_policy == "single"
        else:
            token_fit = pack_tokens + item.metadata.pack_tokens <= self.seq_len
        return token_fit and all(
            packing_costs.get(name, 0) + item.metadata.packing_costs[name] <= budget
            for name, budget in self.packing_budgets.items()
        )

    def add_costs(
            self,
            packing_costs: Mapping[str, float],
            item: BufferedSampleMetadata,
    ) -> dict[str, float]:
        """Return accumulated physical costs for the constrained stages."""
        return {
            name: packing_costs.get(name, 0) + item.metadata.packing_costs[name]
            for name in self.packing_budgets
        }

    def validate_bin(self, items: Sequence[BufferedSampleMetadata]) -> None:
        """Validate a reference or final bin against the same placement rules."""
        pack_tokens = 0
        packing_costs: dict[str, float] = {}
        for item in items:
            self.validate_sample(item)
            if not self.fits(pack_tokens, packing_costs, item):
                raise ValueError(
                    f"Packing bin cannot admit sample {item.key} within seq_len={self.seq_len} "
                    f"and packing_budgets={dict(self.packing_budgets)}."
                )
            pack_tokens += item.metadata.pack_tokens
            packing_costs = self.add_costs(packing_costs, item)


@dataclass(frozen=True)
class PackingBinPlan:
    """Ordered raw samples that one Data Constructor passes to ``pack_fn``.

    ``validate=False`` skips repeated scans for internally constructed, trusted
    bins. The option is construction-only and is not serialized.
    """

    sample_keys: tuple[SampleKey, ...]
    pack_tokens: int
    oversized: bool = False
    validate: InitVar[bool] = True

    def __post_init__(self, validate: bool) -> None:
        """Validate one non-empty sequence-packing bin."""
        if not validate:
            return
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
    """Deterministic sample-to-constructor plan for one distributed yield.

    ``validate=False`` skips repeated scans for internally constructed, trusted
    plans. The option is construction-only and is not serialized.
    """

    plan_id: str
    step: int
    seq_len: int
    local_batches: tuple[tuple[PackingBinPlan, ...], ...]
    rank_costs: tuple[WorkloadCost, ...]
    validate: InitVar[bool] = True

    def __post_init__(self, validate: bool) -> None:
        """Validate plan dimensions, slots, and unique sample assignments."""
        if not validate:
            return
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
        return len(self.local_batches[0])

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
