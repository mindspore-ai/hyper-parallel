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
"""Estimate frozen samples, propose assignments, and accept beneficial plans."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from copy import copy
from itertools import chain
from typing import Any

from hyper_parallel.distributed_data.balancing_algorithm import (
    BalancingAlgorithm,
    FeasibleLPTBalancingAlgorithm,
    LPTBalancingAlgorithm,
    resolve_balancing_algorithm,
)
from hyper_parallel.distributed_data.cost_model import CostModel, resolve_cost_model
from hyper_parallel.distributed_data.schema import (
    DistributedPackingPlan,
    OversizedPolicy,
    PackingBinPlan,
    PackingConstraints,
    SampleKey,
    SampleMetadata,
    WorkloadCost,
)


class DynamicPackingPlanner:
    """Keep cost estimation and placement independently replaceable.

    Every step is evaluated using the supplied or default workload model and
    assignment algorithm. The original rank-local grouping is retained unless
    the candidate's relative objective reduction strictly exceeds the threshold.
    """

    def __init__(
            self,
            *,
            data_parallel_size: int,
            seq_len: int,
            local_batch_size: int,
            oversized_policy: OversizedPolicy = "error",
            packing_budgets: Mapping[str, float] | None = None,
            model_config: Any = None,
            cost_model: CostModel | None = None,
            balancing_algorithm: BalancingAlgorithm | None = None,
            validate: bool = True,
            min_balance_gain: float = 0.0,
    ) -> None:
        """Configure packing boundaries and independent planning policies.

        Args:
            data_parallel_size: Number of target DP ranks.
            seq_len: Token capacity of one packed sequence.
            local_batch_size: Non-empty packed sequences required per rank.
            oversized_policy: ``error`` or singleton token overflow.
            packing_budgets: Additive per-bin stage limits, independent of costs.
            model_config: Effective architecture for the default workload model.
            cost_model: Optional deterministic SampleMetadata-to-WorkloadCost callback.
            balancing_algorithm: Optional assignment and objective implementation.
            validate: Audit trusted input/output membership. Custom assignments
                always receive basic shape, conservation, and capacity checks.
            min_balance_gain: Minimum relative objective reduction in [0, 1).
                Equal, worse, or below-threshold proposals retain the reference.
        """
        for name, value in (
                ("data_parallel_size", data_parallel_size),
                ("seq_len", seq_len),
                ("local_batch_size", local_batch_size),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer, but got {value!r}.")
        if (
                not isinstance(min_balance_gain, (int, float))
                or isinstance(min_balance_gain, bool)
                or not math.isfinite(min_balance_gain)
                or not 0.0 <= min_balance_gain < 1.0
        ):
            raise ValueError("min_balance_gain must be in [0, 1).")
        self.data_parallel_size = data_parallel_size
        self.seq_len = seq_len
        self.local_batch_size = local_batch_size
        self.oversized_policy = oversized_policy
        self._validate = validate
        self._constraints = PackingConstraints(seq_len, oversized_policy, packing_budgets)
        self.cost_model = resolve_cost_model(cost_model, model_config)
        self.balancing_algorithm = resolve_balancing_algorithm(balancing_algorithm)
        if callable(getattr(self.cost_model, "padding_cost", None)) and isinstance(
                self.balancing_algorithm, LPTBalancingAlgorithm,
        ):
            self.balancing_algorithm = copy(self.balancing_algorithm)
            self.balancing_algorithm.padding_cost = self.padding_cost
        self.objective = getattr(
            self.balancing_algorithm, "objective_name", type(self.balancing_algorithm).__qualname__,
        )
        self.min_balance_gain = float(min_balance_gain)
        self.last_sample_costs: dict[SampleKey, WorkloadCost] = {}
        self.last_balance_decision: dict[str, float | bool] = {}

    @property
    def distributed_bin_count(self) -> int:
        """Return the number of bins required for one distributed yield."""
        return self.data_parallel_size * self.local_batch_size

    @property
    def distributed_token_budget(self) -> int:
        """Return the aggregate non-oversized token capacity."""
        return self.distributed_bin_count * self.seq_len

    def padding_cost(self, padding_tokens: int) -> WorkloadCost:
        """Estimate constructor padding in the same units as source-sample costs.

        Args:
            padding_tokens: Number of tokens appended to one packed row.

        Returns:
            Padding workload, or zero when the model has no padding estimate.
        """
        estimate = getattr(self.cost_model, "padding_cost", lambda _: WorkloadCost())
        if not callable(estimate):
            raise ValueError("cost_model.padding_cost must be callable")
        cost = estimate(padding_tokens)
        if not isinstance(cost, WorkloadCost):
            raise ValueError("cost_model.padding_cost must return WorkloadCost")
        return cost

    def plan(
            self,
            samples: Sequence[SampleMetadata],
            *,
            reference_bins: Sequence[Sequence[SampleKey]],
            step: int,
            target_slot_count: int | None = None,
    ) -> DistributedPackingPlan:
        """Evaluate one frozen sample set and preserve non-beneficial layouts.

        Args:
            samples: Exactly the selected occurrences, each with a bound SampleKey.
            reference_bins: Known-feasible original grouping, ordered by
                micro-batch, DP rank, and local bin. Every selected key occurs once.
            step: Zero-based index of the first distributed micro-batch.
            target_slot_count: Number of rank-by-micro-batch destinations.
                Defaults to data_parallel_size for one round. Must be a positive
                integer multiple of data_parallel_size; each slot holds
                local_batch_size bins.

        Returns:
            A canonical plan with one local batch per target slot. Window slots
            are ordered by micro-batch, then DP rank.
        """
        target_slot_count = self._resolve_target_slot_count(target_slot_count)
        samples = self._estimate_samples(samples)
        for item in samples:
            self._constraints.validate_sample(item)
        samples_by_key = {item.key: item for item in samples}
        original, reference_costs = self._build_rank_packing_plans(
            reference_bins, samples_by_key, target_slot_count=target_slot_count, audit=self._validate,
        )
        assignment_options = {}
        if (
                isinstance(self.balancing_algorithm, FeasibleLPTBalancingAlgorithm)
                and type(self.balancing_algorithm).assign is FeasibleLPTBalancingAlgorithm.assign
        ):
            # Custom assign implementations keep the existing policy contract.
            assignment_options["objective_group_size"] = self.data_parallel_size
        candidate_keys = self.balancing_algorithm.assign(
            samples,
            reference_bins=reference_bins,
            constraints=self._constraints,
            data_parallel_size=target_slot_count,
            local_batch_size=self.local_batch_size,
            **assignment_options,
        )
        trusted_assignment = (
            isinstance(self.balancing_algorithm, LPTBalancingAlgorithm)
            and type(self.balancing_algorithm).assign is LPTBalancingAlgorithm.assign
        )
        audit_candidate = self._validate or not trusted_assignment
        candidate, candidate_costs = self._build_rank_packing_plans(
            candidate_keys, samples_by_key, target_slot_count=target_slot_count, audit=audit_candidate,
        )
        if (
                self.data_parallel_size < target_slot_count and self.local_batch_size == 1
                and isinstance(self.balancing_algorithm, LPTBalancingAlgorithm)
        ):
            # Validate assignments before scheduling so a feasible reference
            # cannot hide an invalid candidate from a custom LPT subclass.
            scheduled = (
                self._schedule_window_bins(batches, costs)
                for batches, costs in ((candidate, candidate_costs), (original, reference_costs))
            )
            # Cost sorting need not improve normalized variance or subclass
            # objectives, so retain the original candidate as an option.
            candidate, candidate_costs = min(
                ((candidate, candidate_costs), *scheduled), key=lambda result: self._objective_value(result[1]),
            )
        accepted = self._accept_candidate(reference_costs, candidate_costs)
        local_batches, rank_costs = (candidate, candidate_costs) if accepted else (original, reference_costs)
        return DistributedPackingPlan(
            step=step,
            seq_len=self.seq_len,
            local_batches=local_batches,
            rank_costs=rank_costs,
            validate=False,
        )

    def plan_window(
            self,
            samples: Sequence[SampleMetadata],
            *,
            reference_bins: Sequence[Sequence[SampleKey]],
            step: int,
    ) -> tuple[DistributedPackingPlan, ...]:
        """Balance native rounds jointly over rank-by-micro-batch destinations.

        Args:
            samples: The frozen union of the window's native sample occurrences.
            reference_bins: Original bins ordered by micro-batch, DP rank, then local bin.
            step: First distributed micro-batch index in the window.

        Returns:
            One ordinary distributed plan per micro-batch. The benefit gate sums
            per-round objectives so improvements do not hide a slower window.
        """
        rounds, remainder = divmod(len(reference_bins), self.distributed_bin_count)
        if not rounds or remainder:
            raise ValueError("A balancing window must contain complete distributed micro-batches.")
        plan = self.plan(
            samples, reference_bins=reference_bins, step=step, target_slot_count=self.data_parallel_size * rounds,
        )
        return tuple(
            DistributedPackingPlan(
                step=step + offset,
                seq_len=self.seq_len,
                local_batches=plan.local_batches[start:start + self.data_parallel_size],
                rank_costs=plan.rank_costs[start:start + self.data_parallel_size],
                validate=False,
            )
            for offset, start in enumerate(range(0, len(plan.local_batches), self.data_parallel_size))
        )

    def _resolve_target_slot_count(self, target_slot_count: int | None) -> int:
        """Default to one round and reject incomplete or empty target windows."""
        if target_slot_count is None:
            return self.data_parallel_size
        if (
                not isinstance(target_slot_count, int) or isinstance(target_slot_count, bool)
                or target_slot_count < 1 or target_slot_count % self.data_parallel_size
        ):
            raise ValueError("target_slot_count must be a positive integer multiple of data_parallel_size.")
        return target_slot_count

    @staticmethod
    def _schedule_window_bins(
            local_batches: tuple[tuple[PackingBinPlan, ...], ...],
            rank_costs: tuple[WorkloadCost, ...],
    ) -> tuple[tuple[tuple[PackingBinPlan, ...], ...], tuple[WorkloadCost, ...]]:
        """Group intact, validated packs by cost within the frozen window."""
        order = sorted(
            range(len(local_batches)),
            key=lambda index: (-rank_costs[index].llm, local_batches[index][0].sample_keys),
        )
        return tuple(local_batches[index] for index in order), tuple(rank_costs[index] for index in order)

    def _estimate_samples(
            self, samples: Sequence[SampleMetadata],
    ) -> tuple[SampleMetadata, ...]:
        """Estimate keyed sample costs without mutating physical source metadata."""
        self.last_sample_costs = {}
        estimated_samples = []
        for item in samples:
            if not isinstance(item.key, SampleKey):
                raise ValueError("Planner samples must have a SampleKey.")
            cost = self.cost_model(item)
            if not isinstance(cost, WorkloadCost):
                raise ValueError(f"cost_model must return WorkloadCost for sample {item.key}, but got {type(cost)}.")
            # Preserve physical metadata without replaying recursive feature validation.
            estimated_item = copy(item)
            object.__setattr__(estimated_item, "cost", cost)
            estimated_samples.append(estimated_item)
            self.last_sample_costs[item.key] = cost
        return tuple(estimated_samples)

    def _build_rank_packing_plans(
            self,
            key_bins: Sequence[Sequence[SampleKey]],
            samples_by_key: Mapping[SampleKey, SampleMetadata],
            *,
            target_slot_count: int,
            audit: bool,
    ) -> tuple[tuple[tuple[PackingBinPlan, ...], ...], tuple[WorkloadCost, ...]]:
        """Validate assignments and reconstruct per-slot bins and costs, including new PAD."""
        key_bins = tuple(tuple(keys) for keys in key_bins)
        if len(key_bins) != target_slot_count * self.local_batch_size or any(not keys for keys in key_bins):
            raise ValueError("Assignments must contain target_slot_count * local_batch_size non-empty bins.")
        if audit:
            planned_keys = list(chain.from_iterable(key_bins))
            if len(planned_keys) != len(samples_by_key) or set(planned_keys) != set(samples_by_key):
                raise ValueError("Assignments must conserve every selected sample key exactly once.")
        local_batches = [[] for _ in range(target_slot_count)]
        rank_costs = [WorkloadCost() for _ in range(target_slot_count)]
        for index, keys in enumerate(key_bins):
            items = tuple(samples_by_key[key] for key in keys)
            self._constraints.validate_bin(items)
            tokens = sum(item.pack_tokens for item in items)
            slot = index // self.local_batch_size
            rank_costs[slot] += sum((item.cost for item in items), WorkloadCost())
            rank_costs[slot] += self.padding_cost(max(0, self.seq_len - tokens))
            local_batches[slot].append(PackingBinPlan(
                sample_keys=keys,
                pack_tokens=tokens,
                oversized=tokens > self.seq_len,
                validate=False,
            ))
        return tuple(tuple(bins) for bins in local_batches), tuple(rank_costs)

    def _accept_candidate(
            self, reference_costs: Sequence[WorkloadCost], candidate_costs: Sequence[WorkloadCost],
    ) -> bool:
        """Record the objective comparison and enforce the minimum relative-gain threshold."""
        before = self._objective_value(reference_costs)
        after = self._objective_value(candidate_costs)
        gain = (before - after) / before if before > 0.0 else 0.0
        accepted = gain > self.min_balance_gain
        self.last_balance_decision = {
            "original_objective": before,
            "candidate_objective": after,
            "accepted_objective": after if accepted else before,
            "relative_gain": gain,
            "min_balance_gain": self.min_balance_gain,
            "accepted": accepted,
        }
        return accepted

    def _objective_value(self, rank_costs: Sequence[WorkloadCost]) -> float:
        scores = (
            self._round_objective(rank_costs[start:start + self.data_parallel_size])
            for start in range(0, len(rank_costs), self.data_parallel_size)
        )
        return math.fsum(scores)

    def _round_objective(self, rank_costs: Sequence[WorkloadCost]) -> float:
        score = self.balancing_algorithm.objective(rank_costs)
        if (
                not isinstance(score, (int, float)) or isinstance(score, bool)
                or not math.isfinite(score) or score < 0.0
        ):
            raise ValueError("balancing_algorithm.objective must return a finite, non-negative number.")
        return float(score)


__all__ = ["DynamicPackingPlanner", "OversizedPolicy"]
