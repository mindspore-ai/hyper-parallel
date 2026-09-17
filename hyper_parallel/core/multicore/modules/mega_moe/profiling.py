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
"""MegaMoe owner semantics for the graph-driven generic profiler."""

from dataclasses import dataclass
from typing import Any

from hyper_parallel.core.multicore.profiler.profiling import (
    _ProfileSpec,
    _apply_mega_kernel_profile_graph,
)
from hyper_parallel.core.multicore.scheduler.config import (
    RuntimeConfigC,
    TaskDescC,
    TaskSplitValue,
)
from hyper_parallel.core.multicore.scheduler.graph import ComputeGraph, OperatorNode, OpType


__all__ = []


MEGA_MOE_PROFILE_OWNER_LABEL = "Expert"


@dataclass(frozen=True)
class _MegaMoeProfileContext:
    """Host values needed to resolve expert ownership after task construction."""

    topology: TaskSplitValue
    num_cube_cores: int


def _require_profile_context(context: Any) -> _MegaMoeProfileContext:
    if not isinstance(context, _MegaMoeProfileContext):
        raise TypeError(
            f"MegaMoe profile context must be _MegaMoeProfileContext, got {type(context).__name__}"
        )
    return context


def _resolve_expert_owner(
    operator: OperatorNode,
    task_desc: TaskDescC,
    raw_context: Any,
) -> int:
    """Resolve the global expert ID from a graph operator and its concrete task."""
    context = _require_profile_context(raw_context)
    topology = context.topology
    rank_owner_base = topology.rank_id * topology.single_rank_expert_num

    if operator.op_type == OpType.GMM:
        return rank_owner_base + task_desc.task_index // context.num_cube_cores

    if operator.op_type in (OpType.SWIGLU, OpType.SWIGLU_GRAD):
        if task_desc.task_split_value == 0:
            raise ValueError("SwiGLU profile owner requires a non-zero task_split_value")
        num_triggers = topology.per_expert_seq // task_desc.task_split_value
        if num_triggers == 0:
            raise ValueError("SwiGLU profile owner resolved a zero task count per expert")
        return rank_owner_base + task_desc.task_index // num_triggers

    if operator.op_type == OpType.ALLTOALL:
        task_count_per_expert = task_desc.task_split_num // topology.all_expert_num
        if task_count_per_expert == 0:
            raise ValueError("AllToAll profile owner resolved a zero task count per expert")
        expert_index = task_desc.task_index // task_count_per_expert
        if operator.name == "dispatch":
            if topology.dispatch_mode == "pull":
                return rank_owner_base + expert_index % topology.single_rank_expert_num
            return expert_index
        if operator.name == "combine":
            return rank_owner_base + expert_index % topology.single_rank_expert_num
        raise ValueError(f"MegaMoe AllToAll operator has no expert owner rule: {operator.name!r}")

    raise ValueError(f"MegaMoe operator type has no expert owner rule: {operator.op_type}")


def _configure_mega_moe_profile_metadata(
    runtime_config: RuntimeConfigC,
    graph: ComputeGraph,
    topology: TaskSplitValue,
    *,
    num_cube_cores: int,
    is_backward: bool,
) -> None:
    """Derive stage IDs from the MegaMoe graph and attach expert ownership metadata."""
    if not isinstance(topology, TaskSplitValue):
        raise TypeError(f"topology must be TaskSplitValue, got {type(topology).__name__}")
    if not isinstance(num_cube_cores, int) or isinstance(num_cube_cores, bool) or num_cube_cores <= 0:
        raise ValueError(f"num_cube_cores must be a positive integer, got {num_cube_cores!r}")
    context = _MegaMoeProfileContext(topology=topology, num_cube_cores=num_cube_cores)
    spec = _ProfileSpec(
        kernel_name="MegaMoeGrad" if is_backward else "MegaMoe",
        owner_label=MEGA_MOE_PROFILE_OWNER_LABEL,
        owner_resolver=_resolve_expert_owner,
    )
    _apply_mega_kernel_profile_graph(runtime_config, graph, spec, context=context)
