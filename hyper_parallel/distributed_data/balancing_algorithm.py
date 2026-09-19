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
"""Replaceable capacity-aware assignment algorithms and their objectives."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol

from hyper_parallel.distributed_data.schema import (
    BufferedSampleMetadata,
    PackingConstraints,
    SampleKey,
    WorkloadCost,
)


class BalancingAlgorithm(Protocol):
    """Deterministic CPU assignment, independent of workload estimation.

    Samples already contain estimated costs. Implementations must not mutate
    metadata, read future samples, or issue collectives. Return every selected
    key exactly once in non-empty bins, ordered by rank then local bin index.
    The planner applies capacities and the benefit gate to the returned layout.
    """

    def assign(
            self,
            samples: Sequence[BufferedSampleMetadata],
            *,
            reference_bins: Sequence[Sequence[SampleKey]],
            constraints: PackingConstraints,
            data_parallel_size: int,
            local_batch_size: int,
    ) -> Sequence[Sequence[SampleKey]]:
        """Return candidate bins, or the original bins if no candidate is feasible."""

    def objective(self, rank_costs: Sequence[WorkloadCost]) -> float:
        """Return a finite, non-negative score; a lower score is better."""


@dataclass
class _AssignmentBin:
    data_rank: int
    pack_index: int
    keys: list[SampleKey] = field(default_factory=list)
    tokens: int = 0
    packing_costs: dict[str, float] = field(default_factory=dict)


class LPTBalancingAlgorithm:
    """Longest-processing-time placement with explicit packing constraints.

    Assignment sorts by backbone cost and picks the least-loaded feasible rank.
    ``makespan`` minimizes maximum rank cost. ``balance`` minimizes squared
    coefficient of variation, equivalent to variance for a frozen sample set.
    Other workload components remain available to user-defined algorithms.
    """

    def __init__(self, objective: Literal["makespan", "balance"] = "makespan") -> None:
        """Select the scalar objective used by the planner's relative-gain gate.

        Args:
            objective: Maximum backbone load, or relative backbone-load variance.
        """
        if objective not in ("makespan", "balance"):
            raise ValueError("objective must be 'makespan' or 'balance'.")
        self.objective_name = objective
        self.algorithm_id = f"lpt-v1:{objective}"

    def assign(
            self,
            samples: Sequence[BufferedSampleMetadata],
            *,
            reference_bins: Sequence[Sequence[SampleKey]],
            constraints: PackingConstraints,
            data_parallel_size: int,
            local_batch_size: int,
    ) -> Sequence[Sequence[SampleKey]]:
        """Propose cost-first bins without changing the selected sample set.

        Args:
            samples: Individually feasible samples with estimated workload costs.
            reference_bins: Known-feasible original rank-major bins.
            constraints: Token capacity and additive stage budgets.
            data_parallel_size: Number of target ranks in this balancing domain.
            local_batch_size: Number of non-empty bins required on each rank.

        Returns:
            Candidate rank-major bins, or the reference on a greedy dead end.
        """
        ordered = sorted(samples, key=lambda item: (-item.metadata.cost.llm, -item.metadata.pack_tokens, item.key))
        bins = [
            _AssignmentBin(data_rank, pack_index)
            for data_rank in range(data_parallel_size)
            for pack_index in range(local_batch_size)
        ]
        rank_loads = [0.0] * data_parallel_size
        rank_tokens = [0] * data_parallel_size
        for index, item in enumerate(ordered):
            feasible = [
                packing_bin for packing_bin in bins
                if (index >= len(bins) or not packing_bin.keys)
                and constraints.fits(packing_bin.tokens, packing_bin.packing_costs, item)
            ]
            if not feasible:
                return reference_bins
            selected = min(
                feasible,
                key=lambda packing_bin, sample=item: self._placement_score(
                    packing_bin, sample, constraints.seq_len, rank_loads, rank_tokens,
                ),
            )
            selected.keys.append(item.key)
            selected.tokens += item.metadata.pack_tokens
            selected.packing_costs = constraints.add_costs(selected.packing_costs, item)
            rank_loads[selected.data_rank] += item.metadata.cost.llm
            rank_tokens[selected.data_rank] += min(item.metadata.pack_tokens, constraints.seq_len)
        return tuple(tuple(packing_bin.keys) for packing_bin in bins)

    @staticmethod
    def _placement_score(
            packing_bin: _AssignmentBin,
            item: BufferedSampleMetadata,
            seq_len: int,
            rank_loads: Sequence[float],
            rank_tokens: Sequence[int],
    ) -> tuple[float | int, ...]:
        remaining_capacity = seq_len - min(seq_len, packing_bin.tokens + item.metadata.pack_tokens)
        return (
            rank_loads[packing_bin.data_rank],
            remaining_capacity,
            rank_tokens[packing_bin.data_rank],
            packing_bin.data_rank,
            packing_bin.pack_index,
        )

    def objective(self, rank_costs: Sequence[WorkloadCost]) -> float:
        """Score completed rank loads using the configured optimization target.

        Args:
            rank_costs: Additive workloads reconstructed by the planner.
        """
        maximum = max((cost.llm for cost in rank_costs), default=0.0)
        if self.objective_name == "makespan" or not maximum:
            return maximum
        # Scaling before squaring avoids overflow with FLOPs-sized estimates.
        loads = [cost.llm / maximum for cost in rank_costs]
        mean = math.fsum(loads) / len(loads)
        return math.fsum((load / mean - 1.0) ** 2 for load in loads) / len(loads)


def resolve_balancing_algorithm(algorithm: BalancingAlgorithm | None = None) -> BalancingAlgorithm:
    """Resolve the default assignment policy or validate an explicit one.

    Args:
        algorithm: Optional deterministic assignment and objective implementation.
    """
    resolved = LPTBalancingAlgorithm() if algorithm is None else algorithm
    if not callable(getattr(resolved, "assign", None)) or not callable(getattr(resolved, "objective", None)):
        raise ValueError("balancing_algorithm must provide callable assign() and objective() methods.")
    return resolved


__all__ = ["BalancingAlgorithm", "LPTBalancingAlgorithm", "resolve_balancing_algorithm"]
