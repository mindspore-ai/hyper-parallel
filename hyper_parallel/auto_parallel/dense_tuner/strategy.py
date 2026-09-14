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
"""Parallel strategy data structure for Dense LLM tuning."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from dataclasses import dataclass, field


class StrategyStatus(Enum):
    """Feasibility status of a parallel strategy."""

    FEASIBLE = "feasible"
    INFEASIBLE = "infeasible"
    NOT_SUPPORTED = "not_supported"


@dataclass
class MemoryCost:
    """Memory cost breakdown in MB."""

    model_states: float = 0.0
    activations: float = 0.0
    gradients: float = 0.0
    optimizer_states: float = 0.0
    communication: float = 0.0
    total: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "model_states": self.model_states,
            "activations": self.activations,
            "gradients": self.gradients,
            "optimizer_states": self.optimizer_states,
            "communication": self.communication,
            "total": self.total,
        }


@dataclass
class PerformanceCost:
    """Performance cost breakdown (estimated step time in ms)."""

    compute: float = 0.0
    communication: float = 0.0
    pipeline_bubble: float = 0.0
    recompute: float = 0.0
    total: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "compute": self.compute,
            "communication": self.communication,
            "pipeline_bubble": self.pipeline_bubble,
            "recompute": self.recompute,
            "total": self.total,
        }


@dataclass
class StageInfo:
    """Per-stage information for pipeline parallelism."""

    stage_id: int = 0
    layer_range: Tuple[int, int] = (0, 0)
    memory_mb: float = 0.0
    time_ms: float = 0.0
    num_layers: int = 0
    recompute_layers: int = 0


@dataclass
class ParallelStrategy:
    """A candidate parallel strategy with estimation results.

    Attributes:
        dp_degree: Data parallelism degree.
        tp_degree: Tensor parallelism degree.
        pp_degree: Pipeline parallelism degree.
        cp_degree: Context parallelism degree.
        micro_batch_num: Number of micro-batches.
        stage_partition: PP stage partition result.
        layer_offset: PP stage boundary offset strategy.
        layer_recompute: Layer recompute strategy.
        memory_cost: Total memory estimation and breakdown.
        performance_cost: Total performance estimation and breakdown.
        status: Feasibility status of this strategy.
        filter_reason: Reason if infeasible or filtered.
        stage_info: Per-stage information for PP strategies.
        extra: Additional metadata.
    """

    dp_degree: int = 1
    tp_degree: int = 1
    pp_degree: int = 1
    cp_degree: int = 1
    ep_degree: int = 1
    vpp_degree: int = 1
    op_degree: int = 1
    micro_batch_num: int = 1
    micro_batch_size: int = 1
    fsdp_enabled: bool = False
    stage_partition: Optional[List[int]] = None
    layer_offset: Optional[Any] = None
    layer_recompute: Optional[Any] = None
    memory_cost: MemoryCost = field(default_factory=MemoryCost)
    performance_cost: PerformanceCost = field(default_factory=PerformanceCost)
    status: StrategyStatus = StrategyStatus.FEASIBLE
    filter_reason: str = ""
    stage_info: List[StageInfo] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_devices(self) -> int:
        """Total number of devices required."""
        return self.dp_degree * self.tp_degree * self.pp_degree * self.cp_degree

    @property
    def is_feasible(self) -> bool:
        """Whether the strategy is feasible."""
        return self.status == StrategyStatus.FEASIBLE

    @property
    def key(self) -> str:
        """Human-readable label for this strategy's parallel configuration.

        Note: Does not include stage_partition, layer_offset, or
        layer_recompute. Strategies with the same parallel dimensions but
        different stage/recompute configs will share the same key.
        """
        parts = [
            f"dp{self.dp_degree}",
            f"tp{self.tp_degree}",
            f"pp{self.pp_degree}",
            f"cp{self.cp_degree}",
            f"mbn{self.micro_batch_num}",
        ]
        if self.vpp_degree > 1:
            parts.append(f"vpp{self.vpp_degree}")
        if self.ep_degree > 1:
            parts.append(f"ep{self.ep_degree}")
        if self.op_degree > 1:
            parts.append(f"op{self.op_degree}")
        if self.fsdp_enabled:
            parts.append("fsdp")
        return "_".join(parts)

    def mark_infeasible(self, reason: str) -> None:
        """Mark this strategy as infeasible with a reason.

        Args:
            reason: The reason the strategy is infeasible.
        """
        self.status = StrategyStatus.INFEASIBLE
        self.filter_reason = reason

    def mark_not_supported(self, reason: str) -> None:
        """Mark this strategy as not supported with a reason.

        Args:
            reason: The reason the strategy is not supported.
        """
        self.status = StrategyStatus.NOT_SUPPORTED
        self.filter_reason = reason

    @classmethod
    def from_sapp_dimensions(cls, dims: Any) -> ParallelStrategy:
        """Create a ParallelStrategy from an SAPP-ND Dimensions object.

        Args:
            dims: An SAPP-ND ``Dimensions`` instance with val() method.

        Returns:
            ParallelStrategy with parallel dimensions populated.

        Example::

            from hyper_parallel.auto_parallel.sapp_nd.nd.dimensions import DP, TP, PP, CP, EP, VPP, MBN, MBS, OP
            strategy = ParallelStrategy.from_sapp_dimensions(dims)
        """
        # pylint: disable=import-outside-toplevel
        from hyper_parallel.auto_parallel.sapp_nd.nd.dimensions import (
            CP,
            DP,
            EP,
            MBN,
            MBS,
            OP,
            PP,
            TP,
            VPP,
        )

        def _val(d: Any, default: int = 1) -> int:
            try:
                return int(dims.val(d))
            except KeyError:
                return default

        return cls(
            dp_degree=_val(DP),
            tp_degree=_val(TP),
            pp_degree=_val(PP, 1),
            cp_degree=_val(CP, 1),
            ep_degree=_val(EP, 1),
            vpp_degree=_val(VPP, 1),
            op_degree=_val(OP, 1),
            micro_batch_num=_val(MBN, 1),
            micro_batch_size=_val(MBS, 1),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "dp_degree": self.dp_degree,
            "tp_degree": self.tp_degree,
            "pp_degree": self.pp_degree,
            "cp_degree": self.cp_degree,
            "ep_degree": self.ep_degree,
            "vpp_degree": self.vpp_degree,
            "op_degree": self.op_degree,
            "micro_batch_num": self.micro_batch_num,
            "micro_batch_size": self.micro_batch_size,
            "fsdp_enabled": self.fsdp_enabled,
            "stage_partition": self.stage_partition,
            "layer_offset": self.layer_offset,
            "layer_recompute": self.layer_recompute,
            "memory_cost": self.memory_cost.to_dict(),
            "performance_cost": self.performance_cost.to_dict(),
            "status": self.status.value,
            "filter_reason": self.filter_reason,
            "stage_info": [
                {"stage_id": st.stage_id,
                 "layer_range": list(st.layer_range),
                 "memory_mb": st.memory_mb,
                 "time_ms": st.time_ms,
                 "num_layers": st.num_layers,
                 "recompute_layers": st.recompute_layers}
                for st in self.stage_info
            ],
            "extra": dict(self.extra) if self.extra else {},
            "total_devices": self.total_devices,
            "key": self.key,
        }
