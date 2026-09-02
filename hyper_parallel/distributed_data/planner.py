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
"""Deterministic sample-level balancing with capacity-aware sequence bins."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Sequence

from hyper_parallel.distributed_data.schema import (
    BufferedSampleMetadata,
    DataConstructorPlan,
    DistributedPackingPlan,
    OversizedPolicy,
    PackingBinPlan,
    PlannedSample,
    StepSampleSelection,
    WorkloadCost,
)

@dataclass
class _MutableBin:
    data_rank: int
    pack_index: int
    samples: list[BufferedSampleMetadata] = field(default_factory=list)
    pack_tokens: int = 0
    cost: WorkloadCost = WorkloadCost()
    oversized: bool = False


class DynamicPackingPlanner:
    """Place a frozen step sample set across DP packing bins."""

    def __init__(
            self,
            *,
            data_parallel_size: int,
            seq_len: int,
            local_batch_size: int,
            oversized_policy: OversizedPolicy = "error",
    ) -> None:
        """Initialize fixed constructor dimensions.

        Args:
            data_parallel_size: Number of independent DP Data Constructors.
            seq_len: Capacity of one packed sequence.
            local_batch_size: Packed sequences produced by each constructor.
            oversized_policy: ``error`` or explicit singleton overflow.
        """
        for name, value in (
                ("data_parallel_size", data_parallel_size),
                ("seq_len", seq_len),
                ("local_batch_size", local_batch_size),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer, but got {value!r}.")
        if oversized_policy not in ("error", "single"):
            raise ValueError("oversized_policy must be 'error' or 'single'.")
        self.data_parallel_size = data_parallel_size
        self.seq_len = seq_len
        self.local_batch_size = local_batch_size
        self.oversized_policy = oversized_policy

    @property
    def distributed_bin_count(self) -> int:
        """Return sequence bins required for one distributed yield."""
        return self.data_parallel_size * self.local_batch_size

    @property
    def distributed_token_budget(self) -> int:
        """Return maximum non-oversized tokens in one distributed yield."""
        return self.distributed_bin_count * self.seq_len

    def plan(
            self,
            selection: StepSampleSelection,
            *,
            step: int,
    ) -> DistributedPackingPlan:
        """Balance every selected sample exactly once.

        Args:
            selection: Frozen current-step membership plus a known-feasible
                canonical reference packing.
            step: Zero-based distributed-yield index.

        Returns:
            Full plan containing exactly the selected sample keys.
        """
        ordered = self._validate_plan_request(selection, step)
        bins = [
            _MutableBin(data_rank=data_rank, pack_index=pack_index)
            for data_rank in range(self.data_parallel_size)
            for pack_index in range(self.local_batch_size)
        ]
        rank_costs = [WorkloadCost() for _ in range(self.data_parallel_size)]
        bins, rank_costs = self._place_samples(selection, ordered, bins, rank_costs)
        constructors = self._freeze_bins(bins, rank_costs)
        self._validate_conservation(constructors, selection)
        plan_id = self._plan_id(step, constructors)
        return DistributedPackingPlan(
            plan_id=plan_id,
            step=step,
            seq_len=self.seq_len,
            local_batch_size=self.local_batch_size,
            data_parallel_size=self.data_parallel_size,
            constructors=constructors,
        )

    def _validate_plan_request(
            self,
            selection: StepSampleSelection,
            step: int,
    ) -> tuple[BufferedSampleMetadata, ...]:
        if not isinstance(step, int) or isinstance(step, bool) or step < 0:
            raise ValueError(f"step must be a non-negative integer, but got {step!r}.")
        if not isinstance(selection, StepSampleSelection):
            raise ValueError(f"selection must be StepSampleSelection, but got {type(selection)}.")
        if len(selection.reference_bins) != self.distributed_bin_count:
            raise ValueError(
                f"Step selection expected {self.distributed_bin_count} reference bins, "
                f"but got {len(selection.reference_bins)}."
            )
        ordered = self._validate_and_order(selection.samples)
        if len(ordered) < self.distributed_bin_count:
            raise ValueError(
                f"Step selection has {len(ordered)} samples for {self.distributed_bin_count} non-empty bins."
            )
        return ordered

    def _place_samples(
            self,
            selection: StepSampleSelection,
            ordered: Sequence[BufferedSampleMetadata],
            bins: list[_MutableBin],
            rank_costs: list[WorkloadCost],
    ) -> tuple[list[_MutableBin], list[WorkloadCost]]:
        rank_tokens = [0 for _ in range(self.data_parallel_size)]
        seed_items = ordered[:self.distributed_bin_count]
        remaining_items = ordered[self.distributed_bin_count:]
        for item in seed_items:
            selected = min(
                (packing_bin for packing_bin in bins if not packing_bin.samples),
                key=lambda packing_bin: self._placement_score(
                    packing_bin, item, rank_costs, rank_tokens, seeding=True
                ),
            )
            self._place(selected, item, rank_costs, rank_tokens)

        for item in remaining_items:
            feasible = [packing_bin for packing_bin in bins if self._fits(packing_bin, item)]
            if not feasible:
                bins, rank_costs = self._place_reference_bins(selection)
                break
            selected = min(
                feasible,
                key=lambda packing_bin: self._placement_score(
                    packing_bin, item, rank_costs, rank_tokens, seeding=False
                ),
            )
            self._place(selected, item, rank_costs, rank_tokens)
        return bins, rank_costs

    def _place_reference_bins(
            self,
            selection: StepSampleSelection,
    ) -> tuple[list[_MutableBin], list[WorkloadCost]]:
        """Balance known-feasible reference packs when sample-level packing fails."""
        samples_by_key = {item.key: item for item in selection.samples}
        reference_bins = []
        for original_index, key_bin in enumerate(selection.reference_bins):
            items = tuple(samples_by_key[key] for key in key_bin)
            cost = sum((item.metadata.cost for item in items), WorkloadCost())
            tokens = sum(item.metadata.pack_tokens for item in items)
            reference_bins.append((original_index, items, cost, tokens))
        reference_bins.sort(key=lambda item: (-item[2].dominant, -item[2].total, -item[3], item[0]))

        rank_costs = [WorkloadCost() for _ in range(self.data_parallel_size)]
        rank_tokens = [0 for _ in range(self.data_parallel_size)]
        rank_bins: list[list[_MutableBin]] = [[] for _ in range(self.data_parallel_size)]
        for _, items, cost, tokens in reference_bins:
            eligible_ranks = [
                data_rank
                for data_rank in range(self.data_parallel_size)
                if len(rank_bins[data_rank]) < self.local_batch_size
            ]
            data_rank = min(
                eligible_ranks,
                key=lambda rank: (
                    (rank_costs[rank] + cost).dominant,
                    (rank_costs[rank] + cost).total,
                    rank_tokens[rank] + min(tokens, self.seq_len),
                    rank,
                ),
            )
            packing_bin = _MutableBin(data_rank=data_rank, pack_index=len(rank_bins[data_rank]))
            for item in items:
                self._place(packing_bin, item, rank_costs, rank_tokens)
            rank_bins[data_rank].append(packing_bin)
        return [packing_bin for bins in rank_bins for packing_bin in bins], rank_costs

    @staticmethod
    def _validate_conservation(
            constructors: Sequence[DataConstructorPlan],
            selection: StepSampleSelection,
    ) -> None:
        """Reject any balanced plan that drops or duplicates a selected key."""
        selected_keys = tuple(item.key for item in selection.samples)
        planned_keys = tuple(key for constructor in constructors for key in constructor.sample_keys)
        if len(planned_keys) != len(set(planned_keys)) or set(planned_keys) != set(selected_keys):
            missing = sorted(set(selected_keys) - set(planned_keys))
            unexpected = sorted(set(planned_keys) - set(selected_keys))
            raise ValueError(
                "Balanced placement must conserve the frozen step sample set exactly; "
                f"missing={missing}, unexpected={unexpected}."
            )

    def _validate_and_order(
            self,
            candidates: Sequence[BufferedSampleMetadata],
    ) -> tuple[BufferedSampleMetadata, ...]:
        if any(not isinstance(item, BufferedSampleMetadata) for item in candidates):
            raise ValueError("Every planner candidate must be BufferedSampleMetadata.")
        keys = [item.key for item in candidates]
        if len(keys) != len(set(keys)):
            raise ValueError("Planner candidates must have unique SampleKey values.")
        oversized = [item for item in candidates if item.metadata.pack_tokens > self.seq_len]
        if oversized and self.oversized_policy == "error":
            first = min(oversized, key=lambda item: item.key)
            raise ValueError(
                f"Sample {first.key} requires {first.metadata.pack_tokens} tokens, exceeding seq_len={self.seq_len}. "
                "Set oversized_policy='single' only when the packer supports singleton overflow."
            )
        return tuple(sorted(
            candidates,
            key=lambda item: (
                -item.metadata.pack_tokens,
                -item.metadata.cost.dominant,
                -item.metadata.cost.total,
                item.key,
            ),
        ))

    def _fits(self, packing_bin: _MutableBin, item: BufferedSampleMetadata) -> bool:
        if packing_bin.oversized or item.metadata.pack_tokens > self.seq_len:
            return False
        return packing_bin.pack_tokens + item.metadata.pack_tokens <= self.seq_len

    def _placement_score(
            self,
            packing_bin: _MutableBin,
            item: BufferedSampleMetadata,
            rank_costs: Sequence[WorkloadCost],
            rank_tokens: Sequence[int],
            *,
            seeding: bool,
    ) -> tuple[float, ...]:
        data_rank = packing_bin.data_rank
        projected_cost = rank_costs[data_rank] + item.metadata.cost
        projected_rank_tokens = rank_tokens[data_rank] + min(item.metadata.pack_tokens, self.seq_len)
        remaining_capacity = self.seq_len - min(
            self.seq_len,
            packing_bin.pack_tokens + item.metadata.pack_tokens,
        )
        placement_phase = 0.0 if seeding else 1.0
        return (
            projected_cost.dominant,
            projected_cost.total,
            float(projected_rank_tokens),
            float(remaining_capacity),
            placement_phase,
            float(data_rank),
            float(packing_bin.pack_index),
        )

    def _place(
            self,
            packing_bin: _MutableBin,
            item: BufferedSampleMetadata,
            rank_costs: list[WorkloadCost],
            rank_tokens: list[int],
    ) -> None:
        packing_bin.samples.append(item)
        packing_bin.pack_tokens += item.metadata.pack_tokens
        packing_bin.cost = packing_bin.cost + item.metadata.cost
        packing_bin.oversized = packing_bin.pack_tokens > self.seq_len
        data_rank = packing_bin.data_rank
        rank_costs[data_rank] = rank_costs[data_rank] + item.metadata.cost
        rank_tokens[data_rank] += min(item.metadata.pack_tokens, self.seq_len)

    def _freeze_bins(
            self,
            bins: Sequence[_MutableBin],
            rank_costs: Sequence[WorkloadCost],
    ) -> tuple[DataConstructorPlan, ...]:
        constructors = []
        for data_rank in range(self.data_parallel_size):
            rank_bins = []
            for packing_bin in bins:
                if packing_bin.data_rank != data_rank:
                    continue
                planned_samples = tuple(
                    PlannedSample(
                        key=item.key,
                        metadata=item.metadata,
                        target_data_rank=data_rank,
                        pack_index=packing_bin.pack_index,
                        order=order,
                    )
                    for order, item in enumerate(packing_bin.samples)
                )
                rank_bins.append(PackingBinPlan(
                    pack_index=packing_bin.pack_index,
                    samples=planned_samples,
                    pack_tokens=packing_bin.pack_tokens,
                    oversized=packing_bin.oversized,
                ))
            constructors.append(DataConstructorPlan(
                target_data_rank=data_rank,
                bins=tuple(rank_bins),
                cost=rank_costs[data_rank],
            ))
        return tuple(constructors)

    def _plan_id(
            self,
            step: int,
            constructors: tuple[DataConstructorPlan, ...],
    ) -> str:
        stable_plan = {
            "step": step,
            "seq_len": self.seq_len,
            "local_batch_size": self.local_batch_size,
            "data_parallel_size": self.data_parallel_size,
            "bins": [
                {
                    "data_rank": constructor.target_data_rank,
                    "packs": [
                        {
                            "pack_index": packing_bin.pack_index,
                            "samples": [
                                [sample.key.reader_rank, sample.key.dataset_index]
                                for sample in packing_bin.samples
                            ],
                        }
                        for packing_bin in constructor.bins
                    ],
                }
                for constructor in constructors
            ],
        }
        encoded = json.dumps(stable_plan, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:24]


__all__ = ["DynamicPackingPlanner", "OversizedPolicy"]
