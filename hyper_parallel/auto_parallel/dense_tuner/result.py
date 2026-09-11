# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Result data structures for Dense LLM strategy tuner."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from hyper_parallel.auto_parallel.dense_tuner.strategy import (
    ParallelStrategy,
    StrategyStatus,
)


@dataclass
class StrategySummary:
    """Summary statistics for all candidate strategies.

    Attributes:
        total_candidates: Total number of candidate strategies generated.
        feasible_count: Number of feasible strategies.
        infeasible_count: Number of infeasible strategies.
        not_supported_count: Number of not-supported strategies.
        filter_reasons: Mapping of filter reason to count.
    """

    total_candidates: int = 0
    feasible_count: int = 0
    infeasible_count: int = 0
    not_supported_count: int = 0
    filter_reasons: Dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_strategies(cls, strategies: List[ParallelStrategy]) -> StrategySummary:
        """Create a summary from a list of strategies.

        Args:
            strategies: List of candidate strategies.

        Returns:
            StrategySummary instance.
        """
        feasible = sum(1 for s in strategies if s.status == StrategyStatus.FEASIBLE)
        infeasible = sum(1 for s in strategies if s.status == StrategyStatus.INFEASIBLE)
        not_supported = sum(1 for s in strategies if s.status == StrategyStatus.NOT_SUPPORTED)
        reasons: Dict[str, int] = {}
        for s in strategies:
            if s.filter_reason:
                reasons[s.filter_reason] = reasons.get(s.filter_reason, 0) + 1
        return cls(
            total_candidates=len(strategies),
            feasible_count=feasible,
            infeasible_count=infeasible,
            not_supported_count=not_supported,
            filter_reasons=reasons,
        )


@dataclass
class PPBResult:
    """Pipeline balancing result from SAPP-PPB for a single strategy.

    Attributes:
        vpp_degree: Virtual pipeline-parallel degree (interleave count).
        stage_offsets: Per-stage layer offsets relative to equal partition.
        recompute_policy: Per-stage recompute policy from PPB solver.
        ppb_distribution: Raw PPB solver distribution output.
    """

    vpp_degree: int = 1
    stage_offsets: Optional[List[int]] = None
    recompute_policy: Optional[Any] = None
    ppb_distribution: Optional[Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation of the PPB result.
        """
        return {
            "vpp_degree": self.vpp_degree,
            "stage_offsets": (
                list(self.stage_offsets)
                if self.stage_offsets is not None
                else None
            ),
            "recompute_policy": (
                list(self.recompute_policy)
                if self.recompute_policy is not None
                else None
            ),
            "ppb_distribution": (
                str(self.ppb_distribution)
                if self.ppb_distribution is not None
                else None
            ),
        }


@dataclass
class TunerResult:
    """Result of the Dense LLM strategy tuning process.

    Attributes:
        top_k_strategies: Sorted top-k feasible strategies.
        all_candidates: All candidate strategies with their status.
        all_candidate_summary: Summary statistics for all candidates.
        infeasible_summary: Summary of infeasible strategies.
        memory_breakdown: Per-strategy memory breakdown for top-k.
        performance_breakdown: Per-strategy performance breakdown for top-k.
        stage_summary: Per-stage summary for top-k PP strategies.
        ppb_results: Per-strategy PPB solver results for top-k PP strategies.
        search_log: Human-readable log of the search process.
    """

    top_k_strategies: List[ParallelStrategy] = field(default_factory=list)
    all_candidates: List[ParallelStrategy] = field(default_factory=list)
    all_candidate_summary: Optional[StrategySummary] = None
    infeasible_summary: Optional[StrategySummary] = None
    memory_breakdown: List[Dict[str, float]] = field(default_factory=list)
    performance_breakdown: List[Dict[str, float]] = field(default_factory=list)
    stage_summary: List[List[Dict[str, Any]]] = field(default_factory=list)
    ppb_results: List[Optional[PPBResult]] = field(default_factory=list)
    search_log: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation of the result.
        """
        return {
            "top_k_strategies": [s.to_dict() for s in self.top_k_strategies],
            "all_candidates": [s.to_dict() for s in self.all_candidates],
            "all_candidate_summary": (
                dict(self.all_candidate_summary.__dict__)
                if self.all_candidate_summary
                else None
            ),
            "infeasible_summary": (
                dict(self.infeasible_summary.__dict__)
                if self.infeasible_summary
                else None
            ),
            "memory_breakdown": self.memory_breakdown,
            "performance_breakdown": self.performance_breakdown,
            "stage_summary": self.stage_summary,
            "ppb_results": [
                pr.to_dict() if pr is not None else None
                for pr in self.ppb_results
            ],
            "search_log": self.search_log,
        }
