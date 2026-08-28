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
"""Core data structures and types for Pipeline Parallelism modeling."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple

from hyper_parallel.auto_parallel.sapp_ppb.utils.recompute import TYPE as RecomputeType


@dataclass
class PPBOutput:
    """Output from PPB load balancing algorithm.

    :class:`PPStrategyResult` inherits this class and adds
    ``pp_degree``, ``micro_batch_num``, ``vpp_less_memory``, and
    ``pipeline_bubble``.

    Args:
        stage_partition: Stage partition result.  When
            ``num_of_interleave == 1``, each entry is a physical stage
            list of ``(layer_id, RecomputeType)`` tuples.  When
            ``num_of_interleave > 1``, the list has ``vpp * pp``
            entries representing **virtual stages**: entry
            ``v * pp + s`` corresponds to VPP chunk ``v`` on physical
            stage ``s``.
        layer_offset: Per-group layer offset.  Key is the BODY group name
            from the sapp-ppb ``Layer.name_`` attribute (see
            :attr:`PPStrategyResult.layer_offset` for naming rules); value is
            the offset matrix with shape ``[vpp][pp]``.
        is_feasible: Whether the ILP solution is feasible.  This field
            reflects **only** the ILP solver outcome; it is **not**
            affected by pipeline simulator success or failure.
            Defaults to ``False`` (fail-safe) — callers must
            explicitly set it to ``True`` when the solver finds a
            feasible solution.
        infeasibility_details: Details about infeasibility.
        is_successful: Whether the ILP solver found a successful solution.
            ``True`` when the solver proved optimality or found a feasible
            incumbent; ``False`` otherwise.  Propagated to
            :attr:`PPStrategyResult.is_successful`.
        simulation_status: Pipeline simulator execution status.  One of
            ``"not_run"`` (simulator was not invoked), ``"success"``
            (simulator completed), or ``"failed"`` (simulator could not
            run, e.g. ``micro_batch_num < pp_degree``).  Independent of
            ``is_feasible``.
        simulation_error: Human-readable error message when
            ``simulation_status`` is ``"failed"``.
        simulator_end_time: Pipeline step time (ms) from the
            simulator.  0.0 when the simulator was not run or failed.
        simulator_bubbles: Per-type bubble ratios from the simulator.
            Empty dict when the simulator was not run.
        simulator_peak_memory: Per-stage peak memory (MB) from the
            simulator.  Empty list when the simulator was not run.
        num_of_interleave: VPP interleaving factor.  When > 1,
            ``stage_partition`` contains ``vpp * pp_degree`` entries
            (virtual stages) rather than ``pp_degree`` entries.

    Example:
        >>> output = PPBOutput(
        ...     stage_partition=[[(0, RecomputeType.NONE), (1, RecomputeType.NONE)],
        ...                      [(2, RecomputeType.SLCT), (3, RecomputeType.NONE)]],
        ...     is_successful=True,
        ... )
    """

    stage_partition: List[List[Tuple[int, RecomputeType]]] = field(default_factory=list)
    layer_offset: Dict[str, List[List[int]]] = field(default_factory=dict)
    is_feasible: bool = False
    infeasibility_details: Dict[str, Any] = field(default_factory=dict)
    is_successful: bool = False
    simulator_end_time: float = 0.0
    simulator_bubbles: Dict[str, float] = field(default_factory=dict)
    simulator_peak_memory: List[float] = field(default_factory=list)
    simulation_status: str = "not_run"
    simulation_error: Optional[str] = None
    num_of_interleave: int = 1


@dataclass
class PPStrategyResult(PPBOutput):
    """Pipeline parallelism strategy evaluation result.

    Inherits :class:`PPBOutput` and adds pipeline-topology fields and
    the convenience ``pipeline_bubble`` accessor.

    Inherited fields (from :class:`PPBOutput`):

    * ``stage_partition`` — per-stage ``(layer_id, RecomputeType)`` tuples
    * ``layer_offset`` — per-group offset matrix ``[vpp][pp]``
    * ``is_feasible`` — ILP feasibility flag
    * ``infeasibility_details`` — reason / solver status when infeasible
    * ``is_successful`` — ``True`` when ILP found a usable solution
    * ``simulator_end_time`` — estimated step time in ms (0.0 when not
      run or failed; replaces the former ``estimated_step_time`` field)
    * ``simulator_bubbles`` — per-type bubble ratios
    * ``simulator_peak_memory`` — per-stage peak memory (MB)
    * ``simulation_status`` — ``"not_run"`` / ``"success"`` / ``"failed"``
    * ``simulation_error`` — error message when simulation fails
    * ``num_of_interleave`` — VPP interleaving factor

    Args:
        pp_degree: Number of pipeline stages.
        micro_batch_num: Number of micro batches.
        vpp_less_memory: Whether the less-memory VPP schedule (``vpp2``)
            was used during optimization.  Downstream consumers need
            this to reconstruct the correct pipeline schedule.
        pipeline_bubble: Pipeline bubble ratio (0.0 to 1.0).  This is a
            **ratio**, not an absolute time in ms.  ``None`` when the
            pipeline simulator did not produce a usable result (e.g.
            ``simulation_status`` is not ``"success"``).

    Example:
        >>> result = PPStrategyResult(
        ...     pp_degree=2,
        ...     micro_batch_num=4,
        ...     pipeline_bubble=0.25,
        ...     is_successful=True,
        ... )
    """

    pp_degree: int = 0
    micro_batch_num: int = 1
    vpp_less_memory: bool = False
    pipeline_bubble: Optional[float] = None
