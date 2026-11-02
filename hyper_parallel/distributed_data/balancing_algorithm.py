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
import sys
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol

from hyper_parallel.distributed_data.schema import (
    PackingConstraints,
    SampleKey,
    SampleMetadata,
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
            samples: Sequence[SampleMetadata],
            *,
            reference_bins: Sequence[Sequence[SampleKey]],
            constraints: PackingConstraints,
            data_parallel_size: int,
            local_batch_size: int,
    ) -> Sequence[Sequence[SampleKey]]:
        """Return candidate bins, or the original bins if no candidate is feasible.

        Args:
            samples: Metadata with bound keys and estimated workload costs.
            reference_bins: Original rank-major grouping of the selected keys.
            constraints: Per-bin token capacity and physical stage budgets.
            data_parallel_size: Number of target DP ranks.
            local_batch_size: Number of non-empty bins required per rank.
        """

    def objective(self, rank_costs: Sequence[WorkloadCost]) -> float:
        """Return a finite, non-negative score; a lower score is better.

        Args:
            rank_costs: Additive workloads for one synchronous round.
        """


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
        self.padding_cost: Callable[[int], WorkloadCost] | None = None

    def assign(
            self,
            samples: Sequence[SampleMetadata],
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
        ordered = sorted(samples, key=lambda item: (-item.cost.llm, -item.pack_tokens, item.key))
        bins = [
            _AssignmentBin(data_rank, pack_index)
            for data_rank in range(data_parallel_size)
            for pack_index in range(local_batch_size)
        ]
        rank_loads = [local_batch_size * self._padding_llm_cost(constraints.seq_len)] * data_parallel_size
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
            rank_loads[selected.data_rank] += item.cost.llm + self._padding_delta(
                selected.tokens, item.pack_tokens, constraints.seq_len,
            )
            selected.keys.append(item.key)
            selected.tokens += item.pack_tokens
            selected.packing_costs = constraints.add_costs(selected.packing_costs, item)
            rank_tokens[selected.data_rank] += min(item.pack_tokens, constraints.seq_len)
        return tuple(tuple(packing_bin.keys) for packing_bin in bins)

    def _padding_llm_cost(self, tokens: int) -> float:
        if self.padding_cost is None:
            return 0.0
        # The planner installs the callback after policy construction; static inference sees only None.
        return self.padding_cost(max(0, tokens)).llm  # pylint: disable=not-callable

    def _padding_delta(self, tokens: int, added_tokens: int, capacity: int) -> float:
        return self._padding_llm_cost(capacity - tokens - added_tokens) - self._padding_llm_cost(capacity - tokens)

    def _placement_score(
            self, packing_bin: _AssignmentBin,
            item: SampleMetadata,
            seq_len: int,
            rank_loads: Sequence[float],
            rank_tokens: Sequence[int],
    ) -> tuple[float | int, ...]:
        """Rank feasible placements by workload, remaining capacity and deterministic ties."""
        remaining_capacity = seq_len - min(seq_len, packing_bin.tokens + item.pack_tokens)
        score = rank_loads[packing_bin.data_rank]
        if self.padding_cost is not None:
            loads = list(rank_loads)
            loads[packing_bin.data_rank] += item.cost.llm + self._padding_delta(
                packing_bin.tokens, item.pack_tokens, seq_len,
            )
            score = self.objective([WorkloadCost(llm=load) for load in loads])
        return (
            score,
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


class _WindowLPT(LPTBalancingAlgorithm):
    """Score placements after scheduling complete packs into synchronous rounds."""

    def __init__(self, group_size: int) -> None:
        """Bind the physical DP size without mutating the caller's policy."""
        super().__init__()
        self.group_size = group_size

    def _placement_score(
            self, packing_bin: _AssignmentBin, item: SampleMetadata, seq_len: int,
            rank_loads: Sequence[float], rank_tokens: Sequence[int],
    ) -> tuple[float | int, ...]:
        loads = list(rank_loads)
        loads[packing_bin.data_rank] += item.cost.llm + self._padding_delta(
            packing_bin.tokens, item.pack_tokens, seq_len,
        )
        score = math.fsum(sorted(loads, reverse=True)[::self.group_size])
        return (
            score, seq_len - min(seq_len, packing_bin.tokens + item.pack_tokens),
            rank_tokens[packing_bin.data_rank], packing_bin.data_rank, packing_bin.pack_index,
        )


class FeasibleLPTBalancingAlgorithm(LPTBalancingAlgorithm):
    """Refine token-only greedy assignments within a fixed pair budget.

    Suffix subset sums preserve feasible completion for two bins. Larger
    domains prioritize round bottlenecks. A failed greedy pass starts from
    the known-feasible reference; successful assignments can also improve.
    Repairs require one bin per rank and no extra packing budgets.
    Each pair's bitset table is limited to an estimated 256 MiB, not process RSS.
    The planner still applies its window objective and minimum-gain gate.
    """

    def __init__(
            self, objective: Literal["makespan", "balance"] = "makespan", *, max_repair_pairs: int = 64,
    ) -> None:
        """Bound repair attempts independently of the number of packs.

        Args:
            objective: Maximum backbone load, or relative backbone-load variance.
            max_repair_pairs: Positive maximum number of pair checks per assignment.
                A window shares one budget across all its rounds.
        """
        super().__init__(objective)
        if not isinstance(max_repair_pairs, int) or isinstance(max_repair_pairs, bool) or max_repair_pairs < 1:
            raise ValueError("max_repair_pairs must be a positive integer.")
        self.max_repair_pairs = max_repair_pairs

    def assign(
            self,
            samples: Sequence[SampleMetadata],
            *,
            reference_bins: Sequence[Sequence[SampleKey]],
            constraints: PackingConstraints,
            data_parallel_size: int,
            local_batch_size: int,
            objective_group_size: int | None = None,
    ) -> Sequence[Sequence[SampleKey]]:
        """Propose window-aware placements and bounded capacity-feasible refinements.

        Args:
            samples: Selected occurrences with estimated workload costs.
            reference_bins: Known-feasible original grouping.
            constraints: Token capacity and optional physical stage budgets.
            data_parallel_size: Number of target ranks, including window rounds.
            local_batch_size: Required non-empty bins per target rank.
            objective_group_size: Physical DP ranks per window round. Defaults
                to data_parallel_size for an ordinary single-round assignment.
        """
        group_size = data_parallel_size if objective_group_size is None else objective_group_size
        if (
                not isinstance(group_size, int) or isinstance(group_size, bool) or group_size < 1
                or data_parallel_size % group_size
        ):
            raise ValueError("objective_group_size must be a positive divisor of data_parallel_size.")
        placement = self
        if (
                self._uses_scheduled_makespan() and group_size < data_parallel_size and local_batch_size == 1
                and type(self)._placement_score is LPTBalancingAlgorithm._placement_score
        ):
            placement = _WindowLPT(group_size)
            placement.padding_cost = self.padding_cost
        candidate = LPTBalancingAlgorithm.assign(
            placement,
            samples, reference_bins=reference_bins, constraints=constraints,
            data_parallel_size=data_parallel_size, local_batch_size=local_batch_size,
        )
        if (
                local_batch_size != 1 or constraints.packing_budgets or len(reference_bins) < 2
        ):
            return candidate
        return self._refine_candidate(samples, candidate, reference_bins, constraints.seq_len, group_size)

    def _refine_candidate(
            self,
            samples: Sequence[SampleMetadata],
            candidate: Sequence[Sequence[SampleKey]],
            reference_bins: Sequence[Sequence[SampleKey]],
            capacity: int,
            group_size: int,
    ) -> Sequence[Sequence[SampleKey]]:
        """Refine one-bin-per-slot assignments without token overflow or stage budgets."""
        lookup = {item.key: item for item in samples}
        if any(sum(lookup[key].pack_tokens for key in keys) > capacity for keys in candidate):
            return candidate
        if self.padding_cost is not None:
            candidate = min(
                (candidate, reference_bins),
                key=lambda bins: self._repair_objective(
                    [self._bin_cost(lookup, keys, capacity) for keys in bins], group_size,
                ),
            )
        if not self._uses_scheduled_makespan() and (
                candidate is not reference_bins
                or any(sum(lookup[key].pack_tokens for key in keys) != capacity for keys in candidate)
        ):
            return candidate
        return self._repair_bin_pairs(lookup, candidate, capacity, group_size)

    def _uses_scheduled_makespan(self) -> bool:
        return self.objective_name == "makespan" and type(self).objective is LPTBalancingAlgorithm.objective

    def _bin_cost(
            self, lookup: dict[SampleKey, SampleMetadata], keys: Sequence[SampleKey], capacity: int,
    ) -> WorkloadCost:
        cost = sum((lookup[key].cost for key in keys), WorkloadCost())
        if self.padding_cost is not None:
            cost += self.padding_cost(max(0, capacity - sum(lookup[key].pack_tokens for key in keys)))
        return cost

    def _repair_bin_pairs(
            self,
            lookup: dict[SampleKey, SampleMetadata],
            reference_bins: Sequence[Sequence[SampleKey]],
            capacity: int,
            group_size: int,
    ) -> Sequence[Sequence[SampleKey]]:
        """Try bounded pair repairs, accepting only improvements to the window objective."""
        bins = [tuple(keys) for keys in reference_bins]
        costs = [self._bin_cost(lookup, keys, capacity) for keys in bins]
        score = self._repair_objective(costs, group_size)
        checked: set[tuple[int, int, int, int]] = set()
        versions = [0] * len(bins)
        changed = False
        scheduled_makespan = self._uses_scheduled_makespan()
        while len(checked) < self.max_repair_pairs:
            pairs = {pair for pair in self._repair_candidates(costs, group_size)
                     if (*pair, versions[pair[0]], versions[pair[1]]) not in checked}
            if scheduled_makespan:
                def _potential(pair: tuple[int, int]) -> float:
                    """Estimate candidate gain without treating PAD cost as conserved."""
                    # PAD cost changes with the partition. Source costs alone
                    # give an optimistic lower bound for candidate ordering.
                    total = (sum(lookup[key].cost.llm for index in pair for key in bins[index])
                             if self.padding_cost is not None else None)
                    return self._pair_potential(costs, pair, group_size, total)

                pairs = sorted(pairs, key=lambda pair: (-_potential(pair), pair))
            else:
                pairs = sorted(pairs)
            for left, right in pairs:
                version = (left, right, versions[left], versions[right])
                if len(checked) == self.max_repair_pairs:
                    break
                # Failed and memory-limited attempts consume the same budget.
                checked.add(version)
                items = [lookup[key] for index in (left, right) for key in bins[index]]
                items.sort(key=lambda item: (-item.cost.llm, -item.pack_tokens, item.key))
                candidate = self._repair_bins(items, capacity, (bins[left], bins[right]), costs, (left, right),
                                              group_size)
                candidate_costs = costs.copy()
                candidate_costs[left], candidate_costs[right] = (
                    self._bin_cost(lookup, keys, capacity) for keys in candidate
                )
                candidate_score = self._repair_objective(candidate_costs, group_size)
                if candidate_score < score:
                    bins[left], bins[right] = candidate
                    costs, score = candidate_costs, candidate_score
                    changed = True
                    versions[left] += 1
                    versions[right] += 1
                    break
            else:
                break
        return tuple(bins) if changed else reference_bins

    def _repair_candidates(
            self, costs: Sequence[WorkloadCost], group_size: int,
    ) -> Iterator[tuple[int, int]]:
        """Pair round leaders with heavy, light and nearby-cost packs."""
        order = sorted(range(len(costs)), key=lambda index: (-costs[index].llm, index))
        if self._uses_scheduled_makespan():
            leaders = order[::group_size]
        else:
            leaders = [
                max(range(start, start + group_size), key=lambda index: costs[index].llm)
                for start in range(0, len(costs), group_size)
            ]
        positions = {index: position for position, index in enumerate(order)}
        # Heavy, light and nearby packs offer different capacity-feasible swaps.
        for left in leaders:
            partners = order[:4] + order[-4:]
            partners += [order[(positions[left] + offset + 1) % len(order)] for offset in range(4)]
            for right in partners:
                if left != right:
                    yield min(left, right), max(left, right)

    @staticmethod
    def _pair_potential(
            costs: Sequence[WorkloadCost], pair: tuple[int, int], group_size: int, total: float | None = None,
    ) -> float:
        """Rank sparse candidates by a continuous two-pack relaxation."""
        left, right = pair
        if total is None:
            total = costs[left].llm + costs[right].llm
        others = sorted((cost.llm for index, cost in enumerate(costs) if index not in pair), reverse=True)
        # Only ranks around a round maximum can change the scheduled objective.
        # This bounds the number of splits by the round count, even at large DP.
        breakpoints = {0, (group_size - 1) % group_size, (group_size - 2) % group_size}
        boundaries = [cost for index, cost in enumerate(others) if index % group_size in breakpoints]
        splits = {0.0, total, total / 2}
        splits.update(cost for cost in boundaries if cost <= total)
        splits.update(total - cost for cost in boundaries if cost <= total)
        best = min(math.fsum(sorted((*others, split, total - split), reverse=True)[::group_size])
                   for split in splits)
        before = math.fsum(sorted((cost.llm for cost in costs), reverse=True)[::group_size])
        return before - best

    def _repair_objective(self, costs: Sequence[WorkloadCost], group_size: int) -> float:
        if self._uses_scheduled_makespan():
            costs = sorted(costs, key=lambda cost: cost.llm, reverse=True)
        return math.fsum(
            self.objective(costs[start:start + group_size]) for start in range(0, len(costs), group_size)
        )

    def _repair_bins(
            self,
            ordered: Sequence[SampleMetadata],
            capacity: int,
            reference_bins: Sequence[Sequence[SampleKey]],
            window_costs: Sequence[WorkloadCost],
            pair: tuple[int, int],
            group_size: int,
    ) -> Sequence[Sequence[SampleKey]]:
        """Greedily repartition a pair while suffix reachability preserves capacity feasibility."""
        # Python integers store bits in digits; account for their object overhead.
        digits = (capacity + 1 + sys.int_info.bits_per_digit - 1) // sys.int_info.bits_per_digit
        table_bytes = (len(ordered) + 1) * (sys.getsizeof(0) + digits * sys.int_info.sizeof_digit)
        # Allow document-rich 1M-token packs while bounding each repair's workspace.
        if table_bytes > 256 * 1024 * 1024:
            return reference_bins
        suffix = [1] * (len(ordered) + 1)
        mask = (1 << (capacity + 1)) - 1
        for index in range(len(ordered) - 1, -1, -1):
            reachable = suffix[index + 1]
            suffix[index] = (reachable | reachable << ordered[index].pack_tokens) & mask
        tokens = [0, 0]
        costs = [0.0, 0.0]
        loads = [cost.llm for cost in window_costs]
        bins: list[list[SampleKey]] = [[], []]
        remaining = sum(item.pack_tokens for item in ordered)
        scheduled_makespan = self._uses_scheduled_makespan()
        for index, item in enumerate(ordered):
            remaining -= item.pack_tokens
            candidates = []
            for rank in range(2):
                trial_tokens = tokens.copy()
                trial_tokens[rank] += item.pack_tokens
                low = max(0, 1 - trial_tokens[0], remaining + trial_tokens[1] - capacity)
                high = min(remaining, capacity - trial_tokens[0], remaining + trial_tokens[1] - 1)
                # Reachability within this interval guarantees two non-empty bins,
                # including when their combined tokens leave room for padding.
                if low > high or not (suffix[index + 1] >> low) & ((1 << (high - low + 1)) - 1):
                    continue
                trial = loads.copy()
                trial[pair[0]], trial[pair[1]] = costs
                trial[pair[rank]] += item.cost.llm
                for side in range(2):
                    trial[pair[side]] += self._padding_llm_cost(capacity - trial_tokens[side])
                if scheduled_makespan:
                    score = math.fsum(sorted(trial, reverse=True)[::group_size])
                elif self.padding_cost is not None:
                    score = self._repair_objective([WorkloadCost(llm=value) for value in trial], group_size)
                else:
                    score = costs[rank]
                candidates.append((score, capacity - trial_tokens[rank], rank))
            if not candidates:
                return reference_bins
            rank = min(candidates)[-1]
            bins[rank].append(item.key)
            tokens[rank] += item.pack_tokens
            costs[rank] += item.cost.llm
        return tuple(tuple(keys) for keys in bins)


def resolve_balancing_algorithm(algorithm: BalancingAlgorithm | None = None) -> BalancingAlgorithm:
    """Default to LPT with bounded feasibility repair or validate an explicit policy.

    Args:
        algorithm: Optional deterministic assignment and objective implementation.
    """
    resolved = FeasibleLPTBalancingAlgorithm() if algorithm is None else algorithm
    if not callable(getattr(resolved, "assign", None)) or not callable(getattr(resolved, "objective", None)):
        raise ValueError("balancing_algorithm must provide callable assign() and objective() methods.")
    return resolved


__all__ = [
    "BalancingAlgorithm", "FeasibleLPTBalancingAlgorithm", "LPTBalancingAlgorithm", "resolve_balancing_algorithm",
]
