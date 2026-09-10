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
"""MegaMoe task classification and display semantics for the generic profiler."""

from dataclasses import dataclass
from typing import Any

from hyper_parallel.core.multicore.profiling import (
    _ProfileSpec,
    _ProfileStage,
    _apply_mega_kernel_profile_spec,
)
from hyper_parallel.core.multicore.scheduler.config import (
    RuntimeConfigC,
    TaskDescC,
    TaskSplitValue,
    TaskType,
)


__all__ = []


_PROFILE_DESC_TERMINATE = 0x10006
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


def _resolve_expert_owner(task_desc: TaskDescC, raw_context: Any) -> int:
    """Resolve the global expert ID from the task fields and build topology."""
    context = _require_profile_context(raw_context)
    topology = context.topology
    rank_owner_base = topology.rank_id * topology.single_rank_expert_num

    if task_desc.task_type == TaskType.TASK_GROUPED_MATMUL:
        return rank_owner_base + task_desc.task_index // context.num_cube_cores

    if task_desc.task_type in (TaskType.TASK_SWI_GLU, TaskType.TASK_SWI_GLU_GRAD):
        if task_desc.task_split_value == 0:
            raise ValueError("SwiGLU profile owner requires a non-zero task_split_value")
        num_triggers = topology.per_expert_seq // task_desc.task_split_value
        if num_triggers == 0:
            raise ValueError("SwiGLU profile owner resolved a zero task count per expert")
        return rank_owner_base + task_desc.task_index // num_triggers

    if task_desc.task_type == TaskType.TASK_SHMEM_PUT_MEM_SIGNAL:
        task_count_per_expert = task_desc.task_split_num // topology.all_expert_num
        if task_count_per_expert == 0:
            raise ValueError("AllToAll profile owner resolved a zero task count per expert")
        expert_index = task_desc.task_index // task_count_per_expert
        if task_desc.inputs[0].input_position == 1:
            return expert_index
        return rank_owner_base + expert_index % topology.single_rank_expert_num

    raise ValueError(f"MegaMoe task type has no expert owner rule: {task_desc.task_type}")


_MEGA_MOE_FORWARD_STAGES = (
    _ProfileStage(
        0x10001,
        "Dispatch",
        task_type=TaskType.TASK_SHMEM_PUT_MEM_SIGNAL,
        input_position=1,
        owner_resolver=_resolve_expert_owner,
    ),
    _ProfileStage(
        0x10002,
        "GMM1",
        task_type=TaskType.TASK_GROUPED_MATMUL,
        tiling_data_position=17,
        owner_resolver=_resolve_expert_owner,
    ),
    _ProfileStage(
        0x10003,
        "SwiGLU",
        task_type=TaskType.TASK_SWI_GLU,
        owner_resolver=_resolve_expert_owner,
    ),
    _ProfileStage(
        0x10004,
        "GMM2",
        task_type=TaskType.TASK_GROUPED_MATMUL,
        tiling_data_position=19,
        owner_resolver=_resolve_expert_owner,
    ),
    _ProfileStage(
        0x10005,
        "Combine",
        task_type=TaskType.TASK_SHMEM_PUT_MEM_SIGNAL,
        input_position=13,
        owner_resolver=_resolve_expert_owner,
    ),
    _ProfileStage(_PROFILE_DESC_TERMINATE, "TerminateTask", task_type=TaskType.TASK_TERMINATE),
)
_MEGA_MOE_GRAD_PROFILE_STAGES = (
    _ProfileStage(
        0x11001,
        "DispatchGrad",
        task_type=TaskType.TASK_SHMEM_PUT_MEM_SIGNAL,
        input_position=1,
        owner_resolver=_resolve_expert_owner,
    ),
    _ProfileStage(
        0x11002,
        "ActGrad",
        task_type=TaskType.TASK_GROUPED_MATMUL,
        tiling_data_position=20,
        owner_resolver=_resolve_expert_owner,
    ),
    _ProfileStage(
        0x11003,
        "W2Grad",
        task_type=TaskType.TASK_GROUPED_MATMUL,
        tiling_data_position=23,
        owner_resolver=_resolve_expert_owner,
    ),
    _ProfileStage(
        0x11004,
        "SwiGLUGrad",
        task_type=TaskType.TASK_SWI_GLU_GRAD,
        owner_resolver=_resolve_expert_owner,
    ),
    _ProfileStage(
        0x11005,
        "GateGrad",
        task_type=TaskType.TASK_GROUPED_MATMUL,
        tiling_data_position=21,
        owner_resolver=_resolve_expert_owner,
    ),
    _ProfileStage(
        0x11006,
        "W1Grad",
        task_type=TaskType.TASK_GROUPED_MATMUL,
        tiling_data_position=22,
        owner_resolver=_resolve_expert_owner,
    ),
    _ProfileStage(
        0x11007,
        "CombineGrad",
        task_type=TaskType.TASK_SHMEM_PUT_MEM_SIGNAL,
        input_position=14,
        owner_resolver=_resolve_expert_owner,
    ),
    _ProfileStage(_PROFILE_DESC_TERMINATE, "TerminateTask", task_type=TaskType.TASK_TERMINATE),
)

MEGA_MOE_PROFILE_STAGE_NAMES = {
    stage.desc_id: stage.name
    for stage in _MEGA_MOE_FORWARD_STAGES
    if stage.desc_id != _PROFILE_DESC_TERMINATE
}
MEGA_MOE_GRAD_PROFILE_STAGE_NAMES = {
    stage.desc_id: stage.name
    for stage in _MEGA_MOE_GRAD_PROFILE_STAGES
    if stage.desc_id != _PROFILE_DESC_TERMINATE
}
_MEGA_MOE_FORWARD_PROFILE_SPEC = _ProfileSpec(
    kernel_name="MegaMoe",
    owner_label=MEGA_MOE_PROFILE_OWNER_LABEL,
    stages=_MEGA_MOE_FORWARD_STAGES,
)
_MEGA_MOE_BACKWARD_PROFILE_SPEC = _ProfileSpec(
    kernel_name="MegaMoeGrad",
    owner_label=MEGA_MOE_PROFILE_OWNER_LABEL,
    stages=_MEGA_MOE_GRAD_PROFILE_STAGES,
)


def _configure_mega_moe_profile_metadata(
        runtime_config: RuntimeConfigC,
        topology: TaskSplitValue,
        *,
        num_cube_cores: int,
        is_backward: bool,
) -> None:
    """Resolve all MegaMoe task profile IDs and attach Host display metadata once."""
    if not isinstance(topology, TaskSplitValue):
        raise TypeError(f"topology must be TaskSplitValue, got {type(topology).__name__}")
    if not isinstance(num_cube_cores, int) or isinstance(num_cube_cores, bool) or num_cube_cores <= 0:
        raise ValueError(f"num_cube_cores must be a positive integer, got {num_cube_cores!r}")
    context = _MegaMoeProfileContext(topology=topology, num_cube_cores=num_cube_cores)
    spec = _MEGA_MOE_BACKWARD_PROFILE_SPEC if is_backward else _MEGA_MOE_FORWARD_PROFILE_SPEC
    _apply_mega_kernel_profile_spec(runtime_config, spec, context=context)
