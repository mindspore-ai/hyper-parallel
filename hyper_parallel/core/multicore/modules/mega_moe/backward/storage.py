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
"""Check the physical backward schedule before sharing transient storage."""

from collections import defaultdict

from hyper_parallel.core.multicore.scheduler.config import RuntimeConfigC, TaskDescC, TaskType


_DISPATCH = 0
_WEIGHT2_GRAD = 6
_ACTIVATION_GRAD = 8
_SWIGLU_GRAD = 10
_INPUT_GRAD = 12


def _worker_read_order(
    tasks: list[TaskDescC], num_experts: int, num_cores: int,
) -> bool:
    """Require every Cube to finish both receive readers before writing dX."""
    for worker in range(num_cores):
        locations = {}
        for index, task in enumerate(tasks[worker::num_cores]):
            output = task.outputs[0].input_position
            if output not in (_WEIGHT2_GRAD, _ACTIVATION_GRAD, _INPUT_GRAD):
                continue
            key = (output, task.task_index // num_cores)
            if key in locations or task.task_type != TaskType.TASK_GROUPED_MATMUL:
                return False
            locations[key] = index
        if len(locations) != 3 * num_experts:
            return False
        for expert in range(num_experts):
            order = [locations.get((output, expert), -1)
                     for output in (_WEIGHT2_GRAD, _ACTIVATION_GRAD, _INPUT_GRAD)]
            if not 0 <= order[0] < order[1] < order[2]:
                return False
    return True


def _expert_events_join_readers(
    cfg: RuntimeConfigC, cube_tasks: list[TaskDescC], vector_tasks: list[TaskDescC],
    num_experts: int, num_cores: int,
) -> bool:
    """Require the existing event chain to join all readers of each expert."""
    acts = defaultdict(list)
    gates = defaultdict(list)
    swiglu = defaultdict(list)
    for task in cube_tasks:
        expert = task.task_index // num_cores
        if task.outputs[0].input_position == _ACTIVATION_GRAD:
            acts[expert].append(task)
        elif task.outputs[0].input_position == _INPUT_GRAD:
            gates[expert].append(task)
    for task in vector_tasks:
        if task.task_type != TaskType.TASK_SWI_GLU_GRAD:
            continue
        if task.task_split_num < num_experts or task.task_split_num % num_experts:
            return False
        expert = task.task_index // (task.task_split_num // num_experts)
        swiglu[expert].append(task)
    producers = defaultdict(list)
    for task in cube_tasks + vector_tasks:
        producers[task.trigger_event].append(task)
    return all(_expert_dependency_chain(cfg, acts[expert], gates[expert], swiglu[expert], num_cores, producers)
               for expert in range(num_experts))


def _event_joins_tasks(cfg: RuntimeConfigC, tasks: list[TaskDescC], producers: dict) -> bool:
    """Require a completion event to include exactly these nonempty tasks."""
    event = tasks[0].trigger_event
    return (len(producers[event]) == len(tasks)
            and cfg.all_event_num_triggers[event] == len(tasks)
            and all(task.trigger_event == event for task in tasks))


def _expert_dependency_chain(
    cfg: RuntimeConfigC, acts: list[TaskDescC], gates: list[TaskDescC],
    swiglu: list[TaskDescC], num_cores: int, producers: dict,
) -> bool:
    """Join every expert reader before any task overwrites its receive rows."""
    if len(acts) != num_cores or len(gates) != num_cores or not swiglu:
        return False
    if not _event_joins_tasks(cfg, acts, producers) or not _event_joins_tasks(cfg, swiglu, producers):
        return False
    act_event = acts[0].trigger_event
    swiglu_event = swiglu[0].trigger_event
    return (all(task.dependent_event == act_event and task.outputs[0].input_position == _SWIGLU_GRAD
                for task in swiglu)
            and all(task.dependent_event == swiglu_event for task in gates))


def can_reuse_backward_dispatch(cfg: RuntimeConfigC, num_experts: int, num_cores: int) -> bool:
    """Check a validated runtime before reusing received dY storage for dX.

    Args:
        cfg: Validated, fully reordered backward runtime descriptors.
        num_experts: Number of local experts.
        num_cores: Number of active Cube workers.

    Returns:
        Whether every receive reader precedes dX through worker order and
        the existing full-expert completion events. Unknown readers or changed
        schedules disable reuse without adding dependencies to the hot path.
    """
    if cfg.task_index_num[2]:
        return False
    cube_tasks = [cfg.all_tasks[index] for index in cfg.cube_task_indices[:cfg.task_index_num[0]]]
    vector_tasks = [cfg.all_tasks[index] for index in cfg.vector_task_indices[:cfg.task_index_num[1]]]
    for task in cube_tasks + vector_tasks:
        reads_dispatch = any(task.inputs[index].input_position == _DISPATCH for index in range(task.num_inputs))
        if reads_dispatch and (task.task_type != TaskType.TASK_GROUPED_MATMUL
                               or task.outputs[0].input_position not in (_WEIGHT2_GRAD, _ACTIVATION_GRAD)):
            return False
    return (_worker_read_order(cube_tasks, num_experts, num_cores)
            and _expert_events_join_readers(cfg, cube_tasks, vector_tasks, num_experts, num_cores))
