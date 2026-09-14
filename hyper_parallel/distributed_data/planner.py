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
import logging
import math
import os
from collections.abc import Mapping, Sequence
from copy import copy
from dataclasses import dataclass, field
from typing import Literal

from hyper_parallel.distributed_data.cost_model import CostModel, DefaultCostModel
from hyper_parallel.distributed_data.schema import (
    BufferedSampleMetadata,
    DistributedPackingPlan,
    OversizedPolicy,
    PackingBinPlan,
    PackingConstraints,
    StepSampleSelection,
    WorkloadCost,
)

logger = logging.getLogger(__name__)


@dataclass
class _MutableBin:
    data_rank: int
    pack_index: int
    samples: list[BufferedSampleMetadata] = field(default_factory=list)
    pack_tokens: int = 0
    packing_costs: dict[str, float] = field(default_factory=dict)
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
            packing_budgets: Mapping[str, float] | None = None,
            cost_model: CostModel | None = None,
            validate: bool = True,
            min_balance_gain: float = 0.0,
    ) -> None:
        """Initialize fixed constructor dimensions.

        Args:
            data_parallel_size: Number of independent DP Data Constructors.
            seq_len: Capacity of one packed sequence.
            local_batch_size: Packed sequences produced by each constructor.
            oversized_policy: ``error`` or explicit singleton overflow.
            packing_budgets: Additive hard caps on each bin's named
                ``SampleMetadata.packing_costs``; independent of cost estimates.
            cost_model: Deterministic CPU callback evaluated only on the frozen
                step samples. Defaults to their metadata-provided workload cost.
            validate: Audit metadata types and generated plan membership/order.
                Disable for trusted local packing; placement capacities still apply.
            min_balance_gain: Minimum relative reduction in the maximum dominant
                rank cost before replacing the canonical reference packing.
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
        if not isinstance(min_balance_gain, (int, float)) or isinstance(min_balance_gain, bool) \
                or not 0.0 <= min_balance_gain < 1.0:
            raise ValueError("min_balance_gain must be in [0, 1).")
        self.data_parallel_size = data_parallel_size
        self.seq_len = seq_len
        self.local_batch_size = local_batch_size
        self.oversized_policy = oversized_policy
        self._validate = validate
        self._constraints = PackingConstraints(seq_len, oversized_policy, packing_budgets)
        self.cost_model = DefaultCostModel() if cost_model is None else cost_model
        if not callable(self.cost_model):
            raise ValueError("cost_model must be callable: SampleMetadata -> WorkloadCost.")
        self.min_balance_gain = float(min_balance_gain)

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
        selection = self._estimate_selection(selection)
        ordered = self._validate_plan_request(selection, step)
        bins = [
            _MutableBin(data_rank=data_rank, pack_index=pack_index)
            for data_rank in range(self.data_parallel_size)
            for pack_index in range(self.local_batch_size)
        ]
        rank_costs = [WorkloadCost() for _ in range(self.data_parallel_size)]
        bins, rank_costs = self._place_samples(selection, ordered, bins, rank_costs)
        candidate_rank_costs = list(rank_costs)
        reference_bins = None
        reference_costs = None
        if self.min_balance_gain or os.environ.get("PR1371_COST_DEBUG"):
            reference_bins, reference_costs = self._reference_bins_in_order(selection)
        reference_is_better = (
            self.min_balance_gain
            and self._reference_is_better(reference_costs, rank_costs)
        )
        if reference_is_better:
            bins, rank_costs = reference_bins, reference_costs
        local_batches = self._freeze_bins(bins)
        if self._validate:
            self._validate_conservation(local_batches, selection)
        if os.environ.get("PR1371_COST_DEBUG"):
            ref_costs = reference_costs or []
            ref_max = max((cost.dominant for cost in ref_costs), default=0.0)
            candidate_max = max((cost.dominant for cost in candidate_rank_costs), default=0.0)
            candidate_gain = ((ref_max - candidate_max) / ref_max) if ref_max else 0.0
            final_max = max((cost.dominant for cost in rank_costs), default=0.0)
            sample_costs = [item.metadata.cost.dominant for item in ordered]
            sample_tokens = [item.metadata.pack_tokens for item in ordered]
            logger.warning(
                "[HP cost] step=%d samples=%d placement=%s min_gain=%.4f "
                "reference_max=%.4f candidate_max=%.4f candidate_gain=%.4f "
                "final_max=%.4f "
                "sample_dominant(min/mean/max)=%.4f/%.4f/%.4f "
                "sample_tokens(min/max)=%d/%d candidate_ranks=%s final_ranks=%s",
                step,
                len(ordered),
                "canonical" if reference_is_better else "balanced",
                self.min_balance_gain,
                ref_max,
                candidate_max,
                candidate_gain,
                final_max,
                min(sample_costs, default=0.0),
                sum(sample_costs) / len(sample_costs) if sample_costs else 0.0,
                max(sample_costs, default=0.0),
                min(sample_tokens, default=0),
                max(sample_tokens, default=0),
                [round(cost.dominant, 3) for cost in candidate_rank_costs],
                [round(cost.dominant, 3) for cost in rank_costs],
            )
        plan_id = self._plan_id(step, local_batches)
        return DistributedPackingPlan(
            plan_id=plan_id,
            step=step,
            seq_len=self.seq_len,
            local_batches=local_batches,
            rank_costs=tuple(rank_costs),
            validate=self._validate,
        )

    def _estimate_selection(self, selection: StepSampleSelection) -> StepSampleSelection:
        """Attach balancing estimates without mutating the reader's metadata."""
        if self._validate and not isinstance(selection, StepSampleSelection):
            raise ValueError(f"selection must be StepSampleSelection, but got {type(selection)}.")
        estimated_samples = []
        for item in selection.samples:
            cost = self.cost_model(item.metadata)
            if self._validate and not isinstance(cost, WorkloadCost):
                raise ValueError(f"cost_model must return WorkloadCost for sample {item.key}, but got {type(cost)}.")
            # Physical metadata and membership are unchanged from the input.
            # Only cost changes; do not replay recursive feature/key validation.
            metadata = copy(item.metadata)
            object.__setattr__(metadata, "cost", cost)
            estimated_item = copy(item)
            object.__setattr__(estimated_item, "metadata", metadata)
            estimated_samples.append(estimated_item)
        estimated_selection = copy(selection)
        object.__setattr__(estimated_selection, "samples", tuple(estimated_samples))
        return estimated_selection

    def _reference_bins_in_order(
            self, selection: StepSampleSelection,
    ) -> tuple[list[_MutableBin], list[WorkloadCost]]:
        """Materialize the canonical bins in their original rank order."""
        samples_by_key = {item.key: item for item in selection.samples}
        bins: list[_MutableBin] = []
        rank_costs = [WorkloadCost() for _ in range(self.data_parallel_size)]
        rank_tokens = [0] * self.data_parallel_size
        for index, key_bin in enumerate(selection.reference_bins):
            data_rank = index // self.local_batch_size
            packing_bin = _MutableBin(data_rank=data_rank, pack_index=index % self.local_batch_size)
            for key in key_bin:
                self._place(packing_bin, samples_by_key[key], rank_costs, rank_tokens)
            bins.append(packing_bin)
        return bins, rank_costs

    def _reference_is_better(
            self, reference_costs: Sequence[WorkloadCost], balanced_costs: Sequence[WorkloadCost],
    ) -> bool:
        reference_max = max((cost.dominant for cost in reference_costs), default=0.0)
        balanced_max = max((cost.dominant for cost in balanced_costs), default=0.0)
        if reference_max <= 0.0:
            return True
        gain = (reference_max - balanced_max) / reference_max
        return gain < self.min_balance_gain

    def _validate_plan_request(
            self,
            selection: StepSampleSelection,
            step: int,
    ) -> tuple[BufferedSampleMetadata, ...]:
        if self._validate and (not isinstance(step, int) or isinstance(step, bool) or step < 0):
            raise ValueError(f"step must be a non-negative integer, but got {step!r}.")
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
            local_batches: Sequence[Sequence[PackingBinPlan]],
            selection: StepSampleSelection,
    ) -> None:
        """Reject any balanced plan that drops or duplicates a selected key."""
        selected_keys = tuple(item.key for item in selection.samples)
        planned_keys = tuple(
            key
            for local_batch in local_batches
            for packing_bin in local_batch
            for key in packing_bin.sample_keys
        )
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
        for item in candidates:
            self._constraints.validate_sample(item)
        return tuple(sorted(candidates, key=self._ordering_key))

    @staticmethod
    def _ordering_key(item: BufferedSampleMetadata) -> tuple:
        return (
            -item.metadata.pack_tokens,
            -item.metadata.cost.dominant,
            -item.metadata.cost.total,
            item.key,
        )

    def _fits(self, packing_bin: _MutableBin, item: BufferedSampleMetadata) -> bool:
        return self._constraints.fits(packing_bin.pack_tokens, packing_bin.packing_costs, item)

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
        if not self._fits(packing_bin, item):
            raise ValueError(f"Sample {item.key} cannot fit within this bin's token and stage budgets.")
        packing_bin.samples.append(item)
        packing_bin.pack_tokens += item.metadata.pack_tokens
        packing_bin.packing_costs = self._constraints.add_costs(packing_bin.packing_costs, item)
        packing_bin.cost = packing_bin.cost + item.metadata.cost
        packing_bin.oversized = packing_bin.pack_tokens > self.seq_len
        data_rank = packing_bin.data_rank
        rank_costs[data_rank] = rank_costs[data_rank] + item.metadata.cost
        rank_tokens[data_rank] += min(item.metadata.pack_tokens, self.seq_len)

    def _freeze_bins(
            self,
            bins: Sequence[_MutableBin],
    ) -> tuple[tuple[PackingBinPlan, ...], ...]:
        local_batches = []
        for data_rank in range(self.data_parallel_size):
            rank_bins = []
            for packing_bin in bins:
                if packing_bin.data_rank != data_rank:
                    continue
                rank_bins.append(PackingBinPlan(
                    sample_keys=tuple(item.key for item in packing_bin.samples),
                    pack_tokens=packing_bin.pack_tokens,
                    oversized=packing_bin.oversized,
                    validate=self._validate,
                ))
            local_batches.append(tuple(rank_bins))
        return tuple(local_batches)

    def _plan_id(
            self,
            step: int,
            local_batches: tuple[tuple[PackingBinPlan, ...], ...],
    ) -> str:
        stable_plan = {
            "step": step,
            "seq_len": self.seq_len,
            "local_batch_size": self.local_batch_size,
            "data_parallel_size": self.data_parallel_size,
            "bins": [
                {
                    "data_rank": data_rank,
                    "packs": [
                        {
                            "samples": [
                                [key.reader_rank, key.dataset_index]
                                for key in packing_bin.sample_keys
                            ],
                        }
                        for packing_bin in local_batch
                    ],
                }
                for data_rank, local_batch in enumerate(local_batches)
            ],
        }
        positions = [
            key.global_sample_position
            for local_batch in local_batches
            for packing_bin in local_batch
            for key in packing_bin.sample_keys
        ]
        if any(positions):
            stable_plan["global_sample_positions"] = positions
        encoded = json.dumps(stable_plan, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:24]


class LPTPackingPlanner(DynamicPackingPlanner):
    """Balance backbone FLOPs with capacity-constrained longest-processing-time.

    ``WorkloadCost.llm`` is the only balancing objective; other workload stages
    do not affect ordering or placement. Token and named packing budgets remain
    hard constraints. Reference bins must be rank-major, with exactly
    ``local_batch_size`` consecutive bins belonging to each input data rank.

    ``balance`` compares variance, load range, then maximum load (the default).
    ``makespan`` compares maximum load first, then variance and load range.
    The original rank-local packing is retained if greedy repacking fails or
    does not improve the selected objective. A mere permutation of the same
    rank loads therefore never causes an exchange. Both modes are heuristics,
    not globally optimal schedulers.
    """

    def __init__(
            self,
            *,
            data_parallel_size: int,
            seq_len: int,
            local_batch_size: int,
            oversized_policy: OversizedPolicy = "error",
            packing_budgets: Mapping[str, float] | None = None,
            cost_model: CostModel | None = None,
            validate: bool = True,
            min_balance_gain: float = 0.0,
            objective: Literal["balance", "makespan"] = "balance",
    ) -> None:
        """Initialize capacity-constrained LPT and its plan-selection objective.

        Args:
            data_parallel_size: Independent DP ranks to balance.
            seq_len: Token capacity of one packed sequence.
            local_batch_size: Packed sequences produced by each rank.
            oversized_policy: ``error`` or explicit singleton overflow.
            packing_budgets: Additive hard caps independent of workload estimates.
            cost_model: CPU callback estimating per-sample workload.
            validate: Audit metadata and output plans; capacities always apply.
            min_balance_gain: Additional inherited maximum-dominant-cost gain gate.
            objective: ``balance`` for similar loads, or ``makespan`` to prioritize
                the lowest maximum estimated backbone cost across ranks.
        """
        if objective not in ("balance", "makespan"):
            raise ValueError("objective must be 'balance' or 'makespan'.")
        super().__init__(
            data_parallel_size=data_parallel_size,
            seq_len=seq_len,
            local_batch_size=local_batch_size,
            oversized_policy=oversized_policy,
            packing_budgets=packing_budgets,
            cost_model=cost_model,
            validate=validate,
            min_balance_gain=min_balance_gain,
        )
        self.objective = objective

    @staticmethod
    def _ordering_key(item: BufferedSampleMetadata) -> tuple:
        return (-item.metadata.cost.llm, -item.metadata.pack_tokens, item.key)

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
        remaining_capacity = self.seq_len - min(
            self.seq_len, packing_bin.pack_tokens + item.metadata.pack_tokens,
        )
        # The item has the same cost on every rank. Choosing the lightest
        # feasible rank minimizes both the next peak load and variance increase.
        # The objectives differ when accepting a completed constrained plan.
        return (
            rank_costs[data_rank].llm,
            float(remaining_capacity),
            float(rank_tokens[data_rank]),
            float(data_rank),
            float(packing_bin.pack_index),
        )

    def _place_samples(
            self,
            selection: StepSampleSelection,
            ordered: Sequence[BufferedSampleMetadata],
            bins: list[_MutableBin],
            rank_costs: list[WorkloadCost],
    ) -> tuple[list[_MutableBin], list[WorkloadCost]]:
        reference_bins, reference_costs = self._place_reference_bins(selection)
        bins, rank_costs = super()._place_samples(selection, ordered, bins, rank_costs)
        # Use the same scale for both candidates so normalization cannot change
        # which raw FLOPs variance is smaller, even for very large cost units.
        scale = max((cost.llm for cost in (*reference_costs, *rank_costs)), default=0.0)
        if self._objective_score(rank_costs, scale) < self._objective_score(reference_costs, scale):
            return bins, rank_costs
        return reference_bins, reference_costs

    def _place_reference_bins(
            self,
            selection: StepSampleSelection,
    ) -> tuple[list[_MutableBin], list[WorkloadCost]]:
        """Keep the exact pre-balancing ownership, grouping, and sample order."""
        samples_by_key = {item.key: item for item in selection.samples}
        rank_costs = [WorkloadCost() for _ in range(self.data_parallel_size)]
        rank_tokens = [0 for _ in range(self.data_parallel_size)]
        bins = []
        for index, key_bin in enumerate(selection.reference_bins):
            data_rank, pack_index = divmod(index, self.local_batch_size)
            packing_bin = _MutableBin(data_rank=data_rank, pack_index=pack_index)
            for key in key_bin:
                self._place(packing_bin, samples_by_key[key], rank_costs, rank_tokens)
            bins.append(packing_bin)
        return bins, rank_costs

    @staticmethod
    def _imbalance_score(rank_costs: Sequence[WorkloadCost], scale: float) -> tuple[float, ...]:
        if scale == 0.0:
            return (0.0, 0.0, 0.0)
        loads = [cost.llm / scale for cost in rank_costs]
        mean = math.fsum(loads) / len(loads)
        variance = math.fsum((load - mean) ** 2 for load in loads) / len(loads)
        return (variance, max(loads) - min(loads), max(loads))

    def _objective_score(self, rank_costs: Sequence[WorkloadCost], scale: float) -> tuple[float, ...]:
        variance, spread, peak = self._imbalance_score(rank_costs, scale)
        if self.objective == "makespan":
            return (peak, variance, spread)
        return (variance, spread, peak)


__all__ = ["DynamicPackingPlanner", "LPTPackingPlanner", "OversizedPolicy"]
