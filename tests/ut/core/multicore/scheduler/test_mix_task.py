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
"""CPU-only tests for all-core mixed AIC/AIV task scheduling."""

import unittest

from hyper_parallel.core.multicore.profiler.profiling import _calculate_profile_layout
from hyper_parallel.core.multicore.scheduler.config import (
    ATOMIC_ADD_VALUE_LEN,
    EVENT_INVALID_ID,
    TaskAiCoreType,
    TaskDescC,
    TaskSplitValue,
    TaskType,
)
from hyper_parallel.core.multicore.scheduler.mix_task import (
    append_mix_task,
    get_mix_task_participant_count,
    validate_mix_task_schedule,
)

from hyper_parallel.core.multicore.scheduler.runtime import allocate_runtime_config, serialize_runtime_config


def _task(core_type: TaskAiCoreType) -> TaskDescC:
    task_desc = TaskDescC()
    task_desc.task_type = TaskType.TASK_ADD_CUSTOM
    task_desc.task_aicore_type = core_type
    task_desc.dependent_event = EVENT_INVALID_ID
    task_desc.trigger_event = 7
    return task_desc


class TestMixTaskScheduler(unittest.TestCase):
    """Validate Host contracts required by the Device gang scheduler."""

    def test_append_mix_task_registers_all_participant_triggers(self) -> None:
        """Store one task once while accounting for every 1:2 MIX participant.

        Feature: Collective AIC/AIV MIX task scheduling.
        Description: Append one MIX descriptor to a 24-AIC/48-AIV schedule.
        Expectation: The task is queued once and its event receives 72 contributions.
        """
        runtime_config = allocate_runtime_config(16)
        runtime_config.num_workers = 48
        task_split_value = TaskSplitValue()
        mix_task_id = append_mix_task(
            runtime_config,
            task_split_value,
            _task(TaskAiCoreType.TASK_AICORE_INVALID),
        )

        self.assertEqual(mix_task_id, 0)
        self.assertEqual(runtime_config.task_index_num[2], 1)
        self.assertEqual(runtime_config.mix_task_indices[0], mix_task_id)
        self.assertEqual(
            runtime_config.all_tasks[mix_task_id].task_aicore_type,
            TaskAiCoreType.TASK_AICORE_MIX,
        )
        self.assertEqual(runtime_config.all_event_num_triggers[7], 72)
        self.assertEqual(task_split_value.pre_task_num, 1)
        self.assertEqual(task_split_value.pre_mix_task_num, 1)

    def test_validate_accepts_normal_tasks_around_a_mix_phase(self) -> None:
        """Allow increasing cube/vector task IDs on either side of a MIX boundary.

        Feature: Collective AIC/AIV MIX task scheduling.
        Description: Place Cube and Vector descriptors before and after one MIX phase.
        Expectation: The ordered queues and all 72 MIX participants are accepted.
        """
        runtime_config = allocate_runtime_config(16)
        runtime_config.num_workers = 48
        task_split_value = TaskSplitValue()

        runtime_config.all_tasks[0] = _task(TaskAiCoreType.TASK_AICORE_CUBE)
        runtime_config.cube_task_indices[0] = 0
        runtime_config.task_index_num[0] = 1
        task_split_value.pre_task_num = 1

        mix_task_id = append_mix_task(
            runtime_config, task_split_value, _task(TaskAiCoreType.TASK_AICORE_INVALID)
        )

        runtime_config.all_tasks[2] = _task(TaskAiCoreType.TASK_AICORE_VECTOR)
        runtime_config.vector_task_indices[0] = 2
        runtime_config.task_index_num[1] = 1
        runtime_config.task_num = 3

        validate_mix_task_schedule(runtime_config)

        self.assertEqual(mix_task_id, 1)
        self.assertEqual(get_mix_task_participant_count(runtime_config), 72)

    def test_validate_rejects_reordered_core_queue(self) -> None:
        """Reject schedules that cannot be merged deterministically with MIX epochs.

        Feature: Collective AIC/AIV MIX task scheduling.
        Description: Reorder a Cube queue while a MIX phase is present.
        Expectation: Validation rejects the nonmonotonic queue.
        """
        runtime_config = allocate_runtime_config(16)
        runtime_config.num_workers = 48
        task_split_value = TaskSplitValue()
        task_split_value.pre_task_num = 2
        append_mix_task(
            runtime_config, task_split_value, _task(TaskAiCoreType.TASK_AICORE_INVALID)
        )

        runtime_config.all_tasks[0] = _task(TaskAiCoreType.TASK_AICORE_CUBE)
        runtime_config.all_tasks[1] = _task(TaskAiCoreType.TASK_AICORE_CUBE)
        runtime_config.cube_task_indices[0] = 1
        runtime_config.cube_task_indices[1] = 0
        runtime_config.task_index_num[0] = 2
        runtime_config.task_num = 3

        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            validate_mix_task_schedule(runtime_config)

    def test_validate_rejects_mix_task_in_a_normal_queue(self) -> None:
        """Keep a gang task out of independently distributed worker queues.

        Feature: Collective AIC/AIV MIX task scheduling.
        Description: Reference one MIX descriptor from the Cube queue too.
        Expectation: Validation rejects duplicate scheduling across queue kinds.
        """
        runtime_config = allocate_runtime_config(16)
        runtime_config.num_workers = 48
        task_split_value = TaskSplitValue()
        mix_task_id = append_mix_task(
            runtime_config, task_split_value, _task(TaskAiCoreType.TASK_AICORE_INVALID)
        )
        runtime_config.cube_task_indices[0] = mix_task_id
        runtime_config.task_index_num[0] = 1
        runtime_config.task_num = 1

        with self.assertRaisesRegex(ValueError, "must not also appear"):
            validate_mix_task_schedule(runtime_config)

    def test_participant_count_rejects_invalid_worker_topology(self) -> None:
        """Require the 1:2 AIC/AIV topology used by the Device gang barrier.

        Feature: Collective AIC/AIV MIX task scheduling.
        Description: Use zero, odd, and unsupported worker counts.
        Expectation: Each invalid 1:2 topology raises ValueError.
        """
        runtime_config = allocate_runtime_config(16)
        for worker_count in (0, 1, 47, 50):
            with self.subTest(worker_count=worker_count):
                runtime_config.num_workers = worker_count
                with self.assertRaisesRegex(ValueError, "positive even number"):
                    get_mix_task_participant_count(runtime_config)

    def test_append_uses_dynamic_task_and_event_capacities(self) -> None:
        """Allow descriptors and events beyond the former fixed runtime limits.

        Feature: Collective AIC/AIV MIX task scheduling.
        Description: Append a task above ID 25600 with a trigger event above 1024.
        Expectation: The graph-sized configuration validates and serializes successfully.
        """
        cfg = allocate_runtime_config(25632, 2048)
        cfg.num_workers = 40
        values = TaskSplitValue()
        values.pre_task_num = 25601
        task = _task(TaskAiCoreType.TASK_AICORE_INVALID)
        task.trigger_event = 1500

        self.assertEqual(append_mix_task(cfg, values, task), 25601)
        cfg.task_num = values.pre_task_num
        self.assertEqual(cfg.all_event_num_triggers[1500], 60)
        validate_mix_task_schedule(cfg)
        self.assertTrue(serialize_runtime_config(cfg))

    def test_append_rejects_out_of_bounds_without_mutating_config(self) -> None:
        """Check queue capacity, events, counters and overflow before any write.

        Feature: Collective AIC/AIV MIX task scheduling.
        Description: Exercise capacity, counter, event, and int32 overflow errors.
        Expectation: Append raises ValueError before mutating any configuration or descriptor.
        """
        for failure in ("task", "queue", "counter", "dependency", "trigger", "overflow"):
            with self.subTest(failure=failure):
                cfg = allocate_runtime_config(16)
                cfg.num_workers = 48
                values = TaskSplitValue()
                task = _task(TaskAiCoreType.TASK_AICORE_INVALID)
                if failure == "task":
                    values.pre_task_num = len(cfg.all_tasks)
                elif failure == "queue":
                    values.pre_mix_task_num = len(cfg.mix_task_indices)
                elif failure == "counter":
                    values.pre_mix_task_num = 1
                elif failure == "dependency":
                    task.dependent_event = cfg.event_capacity
                elif failure == "trigger":
                    task.trigger_event = cfg.event_capacity - ATOMIC_ADD_VALUE_LEN + 1
                else:
                    cfg.all_event_num_triggers[7] = 0x7FFFFFFF
                original = bytes(cfg)
                with self.assertRaises(ValueError):
                    append_mix_task(cfg, values, task)
                self.assertEqual(bytes(cfg), original)
                self.assertEqual(task.task_aicore_type, TaskAiCoreType.TASK_AICORE_INVALID)

    def test_serialize_rejects_invalid_mix_schedule_without_builder(self) -> None:
        """Validate direct model builders at the common serialization boundary.

        Feature: Collective AIC/AIV MIX task scheduling.
        Description: Serialize malformed MIX queues, descriptors, events, and trigger counts.
        Expectation: Every malformed schedule is rejected at the serialization boundary.
        """
        for failure in ("count", "task_id", "type", "order", "triggers", "dependency", "trigger"):
            with self.subTest(failure=failure):
                cfg = allocate_runtime_config(16)
                cfg.num_workers = 48
                values = TaskSplitValue()
                for _ in range(2):
                    append_mix_task(cfg, values, _task(TaskAiCoreType.TASK_AICORE_INVALID))
                cfg.task_num = 2
                if failure == "count":
                    cfg.task_index_num[2] = -1
                elif failure == "task_id":
                    cfg.mix_task_indices[1] = 16
                elif failure == "type":
                    cfg.all_tasks[1].task_aicore_type = TaskAiCoreType.TASK_AICORE_CUBE
                elif failure == "order":
                    cfg.mix_task_indices[:2] = [1, 0]
                elif failure == "triggers":
                    cfg.all_event_num_triggers[7] -= 1
                elif failure == "dependency":
                    cfg.all_tasks[1].dependent_event = cfg.event_capacity
                else:
                    cfg.all_tasks[1].trigger_event = cfg.event_capacity - 1
                with self.assertRaises(ValueError):
                    serialize_runtime_config(cfg)

    def test_shared_mix_trigger_counts_accumulate(self) -> None:
        """Count each participating AIC/AIV for every producer of a shared event.

        Feature: Collective AIC/AIV MIX task scheduling.
        Description: Append three MIX producers sharing one event with 60 participants each.
        Expectation: The expected trigger count is 180 and schedule validation passes.
        """
        cfg = allocate_runtime_config(16)
        cfg.num_workers = 40
        values = TaskSplitValue()
        for _ in range(3):
            append_mix_task(cfg, values, _task(TaskAiCoreType.TASK_AICORE_INVALID))
        cfg.task_num = 3
        self.assertEqual(cfg.all_event_num_triggers[7], 180)
        validate_mix_task_schedule(cfg)

    def test_shmem_mix_does_not_add_local_triggers(self) -> None:
        """Leave SHMEM signal accounting to the concrete operation.

        Feature: Collective AIC/AIV MIX task scheduling.
        Description: Append a SHMEM MIX task whose concrete kernel owns its signal.
        Expectation: Append does not add local event contributions.
        """
        cfg = allocate_runtime_config(16)
        cfg.num_workers = 48
        task = _task(TaskAiCoreType.TASK_AICORE_INVALID)
        task.task_type = TaskType.TASK_SHMEM_PUT_MEM_SIGNAL
        append_mix_task(cfg, TaskSplitValue(), task)
        cfg.task_num = 1
        self.assertEqual(cfg.all_event_num_triggers[7], 0)
        validate_mix_task_schedule(cfg)

    def test_no_mix_does_not_require_sorted_core_queues(self) -> None:
        """Allow MegaMoe's existing queue revision when there are no MIX epochs.

        Feature: Collective AIC/AIV MIX task scheduling.
        Description: Serialize a reordered Cube queue with no MIX phases.
        Expectation: Existing independent-queue ordering remains accepted.
        """
        cfg = allocate_runtime_config(16)
        cfg.task_num = 2
        cfg.cube_task_indices[:2] = [1, 0]
        cfg.task_index_num[0] = 2
        validate_mix_task_schedule(cfg)
        self.assertTrue(serialize_runtime_config(cfg))

    def test_mix_records_are_added_to_every_participating_worker(self) -> None:
        """Reserve all MIX records in addition to independently distributed work.

        Feature: Collective AIC/AIV MIX task scheduling.
        Description: Combine distributed core tasks with five three-record MIX tasks.
        Expectation: Both AIC and AIV capacities include every MIX task.
        """
        cfg = allocate_runtime_config(32)
        cfg.num_workers = 48
        for task_id in range(32):
            cfg.all_tasks[task_id] = _task(TaskAiCoreType.TASK_AICORE_INVALID)
            cfg.all_tasks[task_id].dependent_event = 0
        cfg.cube_task_indices[:25] = list(range(25))
        cfg.vector_task_indices[:2] = [25, 26]
        cfg.mix_task_indices[:5] = list(range(27, 32))
        cfg.task_index_num[:3] = [25, 2, 5]

        layout = _calculate_profile_layout(cfg)

        self.assertEqual(layout.aic_required_records, 21)
        self.assertEqual(layout.aiv_required_records, 18)
        self.assertEqual(layout.aic_record_capacity, 32)
        self.assertEqual(layout.aiv_record_capacity, 32)

    def test_mix_only_profile_requires_workers(self) -> None:
        """Reject a zero-worker MIX profile even when ordinary queues are empty.

        Feature: Collective AIC/AIV MIX task scheduling.
        Description: Plan a profile for a MIX task with zero workers.
        Expectation: Planning raises ValueError instead of reserving an unusable profile.
        """
        cfg = allocate_runtime_config(16)
        cfg.all_tasks[0] = _task(TaskAiCoreType.TASK_AICORE_INVALID)
        cfg.task_index_num[2] = 1
        with self.assertRaisesRegex(ValueError, "at least one AIC/AIV worker"):
            _calculate_profile_layout(cfg)


if __name__ == "__main__":
    unittest.main()
