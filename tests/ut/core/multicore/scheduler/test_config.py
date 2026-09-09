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
"""Unit tests for Multicore runtime configuration limits."""

import unittest

from hyper_parallel.core.multicore.scheduler.config import (
    MAX_EXPERT_NUM_PER_RANK,
    MAX_GROUP_LIST,
    NUM_WORKERS_CUBE,
    RuntimeConfigC,
    TaskType,
    TaskSplitValue,
    validate_runtime_config,
)


class TestTaskSplitValue(unittest.TestCase):
    """Validate scheduler topology before runtime-config serialization."""

    def test_accepts_supported_topology(self) -> None:
        """Build derived values for a supported expert partition."""
        values = TaskSplitValue(
            tp=4,
            ep=4,
            seq_size=8192,
            all_expert_num=64,
            top_k=8,
        )

        self.assertEqual(values.single_rank_expert_num, MAX_EXPERT_NUM_PER_RANK)
        self.assertLessEqual(
            NUM_WORKERS_CUBE * values.single_rank_expert_num,
            MAX_GROUP_LIST,
        )

    def test_rejects_values_that_break_runtime_arithmetic(self) -> None:
        """Reject zero divisors, uneven partitions, and oversized scratch use."""
        cases = (
            ({"tp": 0}, "tp must be a positive integer"),
            ({"ep": 0}, "ep must be a positive integer"),
            ({"seq_size": 0}, "seq_size must be a positive integer"),
            ({"all_expert_num": 0}, "all_expert_num must be a positive integer"),
            ({"top_k": 0}, "top_k must be a positive integer"),
            ({"all_expert_num": 4, "top_k": 5}, "cannot exceed"),
            ({"ep": 3}, "must be divisible"),
            ({"ep": 1, "all_expert_num": 17}, "device scratch capacity"),
        )
        defaults = {
            "tp": 4,
            "ep": 4,
            "seq_size": 8192,
            "all_expert_num": 32,
            "top_k": 8,
        }

        for overrides, message in cases:
            with (
                self.subTest(overrides=overrides),
                self.assertRaisesRegex(ValueError, message),
            ):
                TaskSplitValue(**(defaults | overrides))


class TestValidateRuntimeConfig(unittest.TestCase):
    """Validate generated task descriptors before device execution."""

    def setUp(self) -> None:
        """Create a valid topology and minimal runtime configuration."""
        self.values = TaskSplitValue(
            tp=1,
            ep=2,
            seq_size=128,
            all_expert_num=4,
            top_k=2,
        )
        self.config = RuntimeConfigC()
        self.config.num_workers = 2 * NUM_WORKERS_CUBE
        self.config.dynamic_data.dynamic_group_size = self.values.single_rank_expert_num
        self.config.task_num = 1

    def test_accepts_valid_swiglu_task(self) -> None:
        """Accept a task whose split arithmetic stays within grouped-list bounds."""
        task = self.config.all_tasks[0]
        task.task_type = TaskType.TASK_SWI_GLU
        task.task_index = 3
        task.task_split_num = 4

        validate_runtime_config(self.config, self.values, NUM_WORKERS_CUBE)

    def test_rejects_unsafe_task_arithmetic(self) -> None:
        """Reject zero divisors and task indices that can exceed device buffers."""
        cases = (
            (TaskType.TASK_SWI_GLU, 0, 3, "cannot be partitioned"),
            (TaskType.TASK_GROUPED_MATMUL, 0, 47, "expected 48"),
            (TaskType.TASK_SHMEM_PUT_MEM_SIGNAL, 0, 3, "cannot be partitioned"),
            (TaskType.TASK_SHMEM_PUT_MEM_SIGNAL, 8, 8, "task_index/task_split_num"),
        )
        for task_type, task_index, task_split_num, message in cases:
            with (
                self.subTest(task_type=task_type, task_index=task_index),
                self.assertRaisesRegex(ValueError, message),
            ):
                task = self.config.all_tasks[0]
                task.task_type = task_type
                task.task_index = task_index
                task.task_split_num = task_split_num
                validate_runtime_config(self.config, self.values, NUM_WORKERS_CUBE)


if __name__ == "__main__":
    unittest.main()
