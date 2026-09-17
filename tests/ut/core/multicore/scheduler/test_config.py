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
    TaskSplitValue,
    TaskType,
    configure_ready_handshake,
    event_workspace_bytes,
    mega_moe_event_capacity,
    validate_runtime_config,
)
from hyper_parallel.core.multicore.scheduler.runtime import allocate_runtime_config

from tests.common.mark_utils import arg_mark


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
        self.config = allocate_runtime_config(1)
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


class TestReadyHandshakeConfig(unittest.TestCase):
    """Keep ready state inside the single graph-sized runtime contract."""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard",
              essential_mark="essential")
    def test_ready_event_uses_expanded_counter_capacity(self) -> None:
        """Feature: ready event uses expanded counter capacity.

        Description: Configure ready events for 1024 experts distributed over 64 ranks.
        Expectation: Reserve the ready event and its atomic lanes beyond event 1024.
        """
        values = TaskSplitValue(tp=1, ep=64, seq_size=128, all_expert_num=1024, top_k=2)
        config = allocate_runtime_config(16, mega_moe_event_capacity(1024, 64))
        configure_ready_handshake(config, values)
        rebuilt = type(config).from_buffer_copy(bytes(config))
        self.assertEqual(rebuilt.ready_event, values.all_event_num + 3)
        self.assertGreater(rebuilt.ready_event, 1024)
        self.assertEqual(rebuilt.all_event_num_triggers[rebuilt.ready_event], 1)
        self.assertEqual(event_workspace_bytes(64, 1024), 1104 * 4 + 2 * 65 * 64)
        with self.assertRaisesRegex(ValueError, "event_capacity"):
            configure_ready_handshake(allocate_runtime_config(16), values)

    def test_ready_atomic_tail_crosses_event_alignment_boundary(self) -> None:
        """Include ready's final zero lanes when termination alone fits."""
        values = TaskSplitValue(tp=1, ep=71, seq_size=128, all_expert_num=1065, top_k=2)
        capacity = mega_moe_event_capacity(values.all_expert_num, values.ep)
        config = allocate_runtime_config(16, capacity)
        configure_ready_handshake(config, values)
        self.assertEqual(capacity, 1136)
        self.assertEqual(config.ready_event, 1114)
        self.assertLessEqual(config.ready_event + 8, capacity)

    def test_single_rank_disables_handshake_and_persistent_tail(self) -> None:
        """EP1 needs neither a peer event nor persistent signal storage."""
        values = TaskSplitValue(tp=1, ep=1, seq_size=128, all_expert_num=4, top_k=2)
        config = allocate_runtime_config(16)
        configure_ready_handshake(config, values)
        self.assertEqual(config.ready_event, 0)
        self.assertEqual(event_workspace_bytes(1, 4), 4096)


if __name__ == "__main__":
    unittest.main()
