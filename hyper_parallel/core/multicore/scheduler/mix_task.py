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
"""Register and validate all-core gang tasks for a mixed AIC/AIV kernel."""

from hyper_parallel.core.multicore.scheduler.config import (
    ATOMIC_ADD_VALUE_LEN,
    EVENT_INVALID_ID,
    NUM_WORKERS_VECTOR,
    RuntimeConfigC,
    TaskAiCoreType,
    TaskDescC,
    TaskSplitValue,
    TaskType,
)


MIX_TASK_INDEX_SLOT = 2


def get_mix_task_participant_count(runtime_config: RuntimeConfigC) -> int:
    """Return the AIC plus AIV participants in a 1:2 mixed kernel.

    Args:
        runtime_config: Runtime configuration whose num_workers is the number
            of AIV blocks. The corresponding AIC count is half of it.

    Returns:
        Total number of AIC and AIV blocks that execute every MIX task.

    Raises:
        ValueError: If num_workers cannot describe a 1:2 mixed kernel.
    """
    num_vector_workers = int(runtime_config.num_workers)
    if num_vector_workers <= 0 or num_vector_workers % 2 != 0 or num_vector_workers > NUM_WORKERS_VECTOR:
        raise ValueError(
            f"RuntimeConfig.num_workers must be a positive even number <= {NUM_WORKERS_VECTOR} for "
            f"TASK_AICORE_MIX, got {num_vector_workers}"
        )
    return num_vector_workers + num_vector_workers // 2


def append_mix_task(
    runtime_config: RuntimeConfigC,
    task_split_value: TaskSplitValue,
    task_desc: TaskDescC,
) -> int:
    """Append one all-core gang task to a runtime configuration.

    The task is stored once in mix_task_indices. Every AIC and AIV block
    executes that TaskDesc in the same epoch, so its trigger event receives one
    contribution per participating block.

    Args:
        runtime_config: Destination runtime configuration.
        task_split_value: Mutable scheduler counters used by FillConfig code.
        task_desc: Fully populated task descriptor.

    Returns:
        The global task ID assigned to task_desc.

    Raises:
        TypeError: If task_desc is not a TaskDescC.
        ValueError: If a queue/event capacity would be exceeded.
    """
    if not isinstance(task_desc, TaskDescC):
        raise TypeError(f"task_desc must be TaskDescC, got {type(task_desc).__name__}")

    task_id = int(task_split_value.pre_task_num)
    mix_schedule_index = int(task_split_value.pre_mix_task_num)
    if not 0 <= task_id < len(runtime_config.all_tasks):
        raise ValueError(f"MIX task ID is outside RuntimeConfig capacity: {task_id}")
    if not 0 <= mix_schedule_index < len(runtime_config.mix_task_indices):
        raise ValueError(
            f"MIX schedule index is outside RuntimeConfig capacity: {mix_schedule_index}"
        )
    if int(runtime_config.task_index_num[MIX_TASK_INDEX_SLOT]) != mix_schedule_index:
        raise ValueError(
            "RuntimeConfig MIX task count and TaskSplitValue.pre_mix_task_num "
            "must advance together"
        )

    _validate_task_events(runtime_config, task_desc, task_id)
    participant_count = get_mix_task_participant_count(runtime_config)
    trigger_event = int(task_desc.trigger_event)
    if task_desc.task_type != TaskType.TASK_SHMEM_PUT_MEM_SIGNAL:
        trigger_count = int(runtime_config.all_event_num_triggers[trigger_event]) + participant_count
        if trigger_count > 0x7FFFFFFF:
            raise ValueError(f"MIX task trigger count exceeds int32 capacity: {trigger_count}")

    task_desc.task_aicore_type = TaskAiCoreType.TASK_AICORE_MIX
    runtime_config.all_tasks[task_id] = task_desc
    runtime_config.mix_task_indices[mix_schedule_index] = task_id
    runtime_config.task_index_num[MIX_TASK_INDEX_SLOT] += 1

    if task_desc.task_type != TaskType.TASK_SHMEM_PUT_MEM_SIGNAL:
        runtime_config.all_event_num_triggers[trigger_event] += participant_count

    task_split_value.pre_task_num += 1
    task_split_value.pre_mix_task_num += 1
    return task_id


def validate_mix_task_schedule(runtime_config: RuntimeConfigC) -> None:
    """Validate the task-ID phase ordering required by the Device merge loop.

    MIX tasks are global phase boundaries. Cube/vector queues must remain in
    increasing global task-ID order so every worker executes its pre-MIX work
    before entering the same gang epoch.

    Args:
        runtime_config: Runtime configuration ready for serialization.

    Raises:
        ValueError: If the MIX queue or a core-specific queue violates the
            all-core gang scheduling contract.
    """
    mix_task_count = int(runtime_config.task_index_num[MIX_TASK_INDEX_SLOT])
    if mix_task_count == 0:
        return
    if not 0 <= mix_task_count <= len(runtime_config.mix_task_indices):
        raise ValueError(
            f"MIX task count is outside RuntimeConfig capacity: {mix_task_count}"
        )

    participant_count = get_mix_task_participant_count(runtime_config)
    mix_task_ids = []
    expected_trigger_counts = {}
    for schedule_index in range(mix_task_count):
        task_id = int(runtime_config.mix_task_indices[schedule_index])
        if not 0 <= task_id < len(runtime_config.all_tasks):
            raise ValueError(
                f"MIX task ID is outside RuntimeConfig capacity: {task_id}"
            )
        if mix_task_ids and task_id <= mix_task_ids[-1]:
            raise ValueError("MIX task IDs must be strictly increasing")
        task_desc = runtime_config.all_tasks[task_id]
        _validate_task_events(runtime_config, task_desc, task_id)
        if task_desc.task_aicore_type != TaskAiCoreType.TASK_AICORE_MIX:
            raise ValueError(
                f"Task {task_id} is in the MIX queue but is not TASK_AICORE_MIX"
            )
        mix_task_ids.append(task_id)

        if task_desc.task_type != TaskType.TASK_SHMEM_PUT_MEM_SIGNAL:
            trigger_event = int(task_desc.trigger_event)
            expected_trigger_counts[trigger_event] = (
                expected_trigger_counts.get(trigger_event, 0) + participant_count
            )

    mix_task_id_set = set(mix_task_ids)
    core_schedules = (
        (
            "cube",
            runtime_config.cube_task_indices,
            int(runtime_config.task_index_num[0]),
        ),
        (
            "vector",
            runtime_config.vector_task_indices,
            int(runtime_config.task_index_num[1]),
        ),
    )
    for core_name, task_indices, scheduled_task_count in core_schedules:
        if not 0 <= scheduled_task_count <= len(task_indices):
            raise ValueError(
                f"{core_name} task count is outside RuntimeConfig capacity: {scheduled_task_count}"
            )
        previous_task_id = -1
        for schedule_index in range(scheduled_task_count):
            task_id = int(task_indices[schedule_index])
            if not 0 <= task_id < len(runtime_config.all_tasks):
                raise ValueError(
                    f"{core_name} task ID is outside RuntimeConfig capacity: {task_id}"
                )
            if task_id <= previous_task_id:
                raise ValueError(
                    f"{core_name} task IDs must be strictly increasing when MIX tasks are present"
                )
            if task_id in mix_task_id_set:
                raise ValueError(
                    f"MIX task {task_id} must not also appear in the {core_name} queue"
                )
            if runtime_config.all_tasks[task_id].task_aicore_type == TaskAiCoreType.TASK_AICORE_MIX:
                raise ValueError(f"MIX task {task_id} must be scheduled only in the MIX queue")
            _validate_task_events(runtime_config, runtime_config.all_tasks[task_id], task_id)
            previous_task_id = task_id

    for trigger_event, expected_count in expected_trigger_counts.items():
        actual_count = int(runtime_config.all_event_num_triggers[trigger_event])
        if actual_count < expected_count:
            raise ValueError(
                f"MIX trigger event {trigger_event} expects at least {expected_count} participants, "
                f"got {actual_count}"
            )


def _validate_task_events(runtime_config: RuntimeConfigC, task: TaskDescC, task_id: int) -> None:
    """Bound dependencies and the complete vector written by an atomic trigger."""
    event_capacity = len(runtime_config.all_event_num_triggers)
    if task.dependent_event != EVENT_INVALID_ID and task.dependent_event >= event_capacity:
        raise ValueError(f"MIX schedule task {task_id} has an invalid dependent event: {task.dependent_event}")
    if task.trigger_event + ATOMIC_ADD_VALUE_LEN > event_capacity:
        raise ValueError(f"MIX schedule task {task_id} has an invalid trigger event: {task.trigger_event}")
