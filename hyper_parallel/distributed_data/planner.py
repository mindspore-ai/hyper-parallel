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

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from copy import copy
from typing import Any

from hyper_parallel.distributed_data.balancing_algorithm import (
    BalancingAlgorithm,
    LPTBalancingAlgorithm,
    resolve_balancing_algorithm,
)
from hyper_parallel.distributed_data.cost_model import CostModel, resolve_cost_model
from hyper_parallel.distributed_data.schema import (
    BufferedSampleMetadata,
    DistributedPackingPlan,
    OversizedPolicy,
    PackingBinPlan,
    PackingConstraints,
    SampleKey,
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

    def plan(
            self,
            samples: Sequence[BufferedSampleMetadata],
            *,
            reference_bins: Sequence[Sequence[SampleKey]],
            step: int,
    ) -> DistributedPackingPlan:
        """Evaluate one frozen sample set and preserve non-beneficial layouts.

        Args:
            samples: Exactly the sample occurrences selected for this step.
            reference_bins: Known-feasible original grouping, ordered by rank
                and local bin. Every selected sample key occurs exactly once.
            step: Zero-based distributed-yield index.

        Returns:
            A canonical plan whose costs are reconstructed from this step's estimates.
        """
        samples = self._estimate_samples(samples)
        for item in samples:
            self._constraints.validate_sample(item)
        samples_by_key = {item.key: item for item in samples}
        original, reference_costs = self._materialize(reference_bins, samples_by_key, audit=self._validate)
        candidate_keys = self.balancing_algorithm.assign(
            samples,
            reference_bins=reference_bins,
            constraints=self._constraints,
            data_parallel_size=self.data_parallel_size,
            local_batch_size=self.local_batch_size,
        )
        trusted_assignment = (
            isinstance(self.balancing_algorithm, LPTBalancingAlgorithm)
            and type(self.balancing_algorithm).assign is LPTBalancingAlgorithm.assign
        )
        audit_candidate = self._validate or not trusted_assignment
        candidate, candidate_costs = self._materialize(candidate_keys, samples_by_key, audit=audit_candidate)
        accepted = self._accept_candidate(reference_costs, candidate_costs)
        local_batches, rank_costs = (candidate, candidate_costs) if accepted else (original, reference_costs)
        return DistributedPackingPlan(
            plan_id=self._plan_id(step, local_batches),
            step=step,
            seq_len=self.seq_len,
            local_batches=local_batches,
            rank_costs=rank_costs,
            validate=False,
        )

    def _estimate_samples(
            self, samples: Sequence[BufferedSampleMetadata],
    ) -> tuple[BufferedSampleMetadata, ...]:
        self.last_sample_costs = {}
        estimated_samples = []
        for item in samples:
            cost = self.cost_model(item.metadata)
            if not isinstance(cost, WorkloadCost):
                raise ValueError(f"cost_model must return WorkloadCost for sample {item.key}, but got {type(cost)}.")
            # Preserve physical metadata without replaying recursive feature validation.
            metadata = copy(item.metadata)
            object.__setattr__(metadata, "cost", cost)
            estimated_item = copy(item)
            object.__setattr__(estimated_item, "metadata", metadata)
            estimated_samples.append(estimated_item)
            self.last_sample_costs[item.key] = cost
        return tuple(estimated_samples)

    def _materialize(
            self,
            key_bins: Sequence[Sequence[SampleKey]],
            samples_by_key: Mapping[SampleKey, BufferedSampleMetadata],
            *,
            audit: bool,
    ) -> tuple[tuple[tuple[PackingBinPlan, ...], ...], tuple[WorkloadCost, ...]]:
        key_bins = tuple(tuple(keys) for keys in key_bins)
        if len(key_bins) != self.distributed_bin_count or any(not keys for keys in key_bins):
            raise ValueError("Assignments must contain data_parallel_size * local_batch_size non-empty bins.")
        if audit:
            planned_keys = [key for keys in key_bins for key in keys]
            if len(planned_keys) != len(samples_by_key) or set(planned_keys) != set(samples_by_key):
                raise ValueError("Assignments must conserve every selected sample key exactly once.")
        local_batches = [[] for _ in range(self.data_parallel_size)]
        rank_costs = [WorkloadCost() for _ in range(self.data_parallel_size)]
        for index, keys in enumerate(key_bins):
            items = tuple(samples_by_key[key] for key in keys)
            self._constraints.validate_bin(items)
            tokens = sum(item.metadata.pack_tokens for item in items)
            rank = index // self.local_batch_size
            rank_costs[rank] += sum((item.metadata.cost for item in items), WorkloadCost())
            local_batches[rank].append(PackingBinPlan(
                sample_keys=keys,
                pack_tokens=tokens,
                oversized=tokens > self.seq_len,
                validate=False,
            ))
        return tuple(tuple(bins) for bins in local_batches), tuple(rank_costs)

    def _accept_candidate(
            self, reference_costs: Sequence[WorkloadCost], candidate_costs: Sequence[WorkloadCost],
    ) -> bool:
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
        score = self.balancing_algorithm.objective(rank_costs)
        if (
                not isinstance(score, (int, float)) or isinstance(score, bool)
                or not math.isfinite(score) or score < 0.0
        ):
            raise ValueError("balancing_algorithm.objective must return a finite, non-negative number.")
        return float(score)

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
                        {"samples": [[key.reader_rank, key.dataset_index] for key in packing_bin.sample_keys]}
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


__all__ = ["DynamicPackingPlanner", "OversizedPolicy"]
