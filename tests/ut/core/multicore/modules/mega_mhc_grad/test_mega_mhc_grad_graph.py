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
"""RuntimeConfig tests for the fused HyperMegaMhcGrad graph."""

import unittest
from unittest.mock import patch

from hyper_parallel.core.multicore.modules.mega_mhc_grad.function import _resolve_core_counts
from hyper_parallel.core.multicore.modules.mega_mhc_grad.graph import (
    DEFAULT_GRAD_TOKEN_TILE,
    NATIVE_GRAD_CUBE_TILE,
    build_event_layout,
    build_mega_mhc_grad_graph,
    order_vector_tasks_by_stage,
    resolve_grad_token_tile,
)
from hyper_parallel.core.multicore.scheduler.builder import build_runtime_config
from hyper_parallel.core.multicore.scheduler.config import (
    FAST_DEPENDENCY_POLL_INTERVAL_US,
    TaskAiCoreType,
    TaskType,
)
from tests.common.mark_utils import arg_mark


NUM_CUBE_CORES = 20
NUM_VECTOR_CORES = 40


class TestMegaMhcGradGraph(unittest.TestCase):
    """Validate task types, dependencies, tiling, and AIV queue order."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    @patch("hyper_parallel.core.multicore.modules.mega_mhc_grad.function.torch.npu.get_device_limit")
    def test_resolve_core_counts_uses_device_vector_count(self, mock_get_device_limit) -> None:
        """Feature: HyperMegaMhcGrad device core discovery.

        Description: Read distinct AIC and AIV limits reported for a 910B3.
        Expectation: The execution plan receives the physical 20/40 core counts.
        """
        mock_get_device_limit.return_value = {
            "cube_core_num": NUM_CUBE_CORES,
            "vector_core_num": NUM_VECTOR_CORES,
        }

        self.assertEqual(
            _resolve_core_counts(0),
            (NUM_CUBE_CORES, NUM_VECTOR_CORES),
        )

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    @patch("hyper_parallel.core.multicore.modules.mega_mhc_grad.function.torch.npu.get_device_limit")
    def test_resolve_core_counts_rejects_incompatible_mixed_ratio(self, mock_get_device_limit) -> None:
        """Feature: HyperMegaMhcGrad mixed-kernel core ratio.

        Description: Report an AIV limit that is not twice the AIC limit.
        Expectation: Planning fails before launching the fixed 1:2 mixed kernel.
        """
        mock_get_device_limit.return_value = {
            "cube_core_num": NUM_CUBE_CORES,
            "vector_core_num": 32,
        }

        with self.assertRaisesRegex(RuntimeError, "KERNEL_TYPE_MIX_AIC_1_2"):
            _resolve_core_counts(0)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_builds_rms_and_fused_prepare_pipeline(self) -> None:
        """Feature: HyperMegaMhcGrad task graph.

        Description: Build a four-tile graph and inspect every fused stage.
        Expectation: Task types, dependencies, event counts, and queues match the pipeline.
        """
        graph, topology = build_mega_mhc_grad_graph(
            128,
            5120,
            num_vector_cores=NUM_VECTOR_CORES,
        )
        config = build_runtime_config(graph, topology, num_cube_cores=NUM_CUBE_CORES)
        order_vector_tasks_by_stage(config)
        layout = build_event_layout(4, 2)

        self.assertEqual(config.num_workers, NUM_VECTOR_CORES)
        self.assertEqual(
            [operator.name for operator in graph.topological_sort()],
            [
                "grad_output_init",
                "rms_norm_grad",
                "mhc_grad_prev_a_and_mapping",
                "mhc_grad_phi_rms",
                "mhc_grad_prev_x_and_post",
            ],
        )
        self.assertEqual(config.task_num, 54)
        self.assertEqual(tuple(config.task_index_num[:3]), (2, 52, 0))

        init_tasks = config.all_tasks[:NUM_VECTOR_CORES]
        self.assertEqual(
            [task.task_index for task in init_tasks],
            list(range(NUM_VECTOR_CORES)),
        )
        self.assertEqual(
            {task.task_type for task in init_tasks},
            {TaskType.TASK_MHC_GRAD_PREV_A},
        )
        self.assertEqual(
            {task.task_aicore_type for task in init_tasks},
            {TaskAiCoreType.TASK_AICORE_VECTOR},
        )
        self.assertEqual({task.task_split_num for task in init_tasks}, {NUM_VECTOR_CORES})
        self.assertEqual({task.task_split_value for task in init_tasks}, {0})

        rms_tasks = config.all_tasks[40:44]
        prepare_tasks = config.all_tasks[44:48]
        phi_rms_tasks = config.all_tasks[48:50]
        prev_x_tasks = config.all_tasks[50:54]
        for index, task in enumerate(rms_tasks):
            self.assertEqual(task.task_type, TaskType.TASK_RMS_NORM_GRAD)
            self.assertEqual(task.task_split_value, DEFAULT_GRAD_TOKEN_TILE)
            self.assertEqual(task.dependent_event, layout.init_done)
            self.assertEqual(task.trigger_event, layout.rms_ready_base + index)
            self.assertEqual(
                task.extra_value_2,
                FAST_DEPENDENCY_POLL_INTERVAL_US,
            )
        for index, task in enumerate(prepare_tasks):
            self.assertEqual(task.task_type, TaskType.TASK_MHC_GRAD_PREV_A_AND_MAPPING)
            self.assertEqual(task.task_split_value, DEFAULT_GRAD_TOKEN_TILE)
            self.assertEqual(
                task.extra_value_2,
                FAST_DEPENDENCY_POLL_INTERVAL_US,
            )
            self.assertEqual(task.dependent_event, layout.rms_ready_base + index)
            self.assertEqual(task.trigger_event, layout.macro_ready_base + index // 2)
        for index, task in enumerate(phi_rms_tasks):
            self.assertEqual(task.task_type, TaskType.TASK_MHC_GRAD_PHI_RMS)
            self.assertEqual(task.task_aicore_type, TaskAiCoreType.TASK_AICORE_CUBE)
            self.assertEqual(task.task_split_value, NATIVE_GRAD_CUBE_TILE)
            self.assertEqual(task.dependent_event, layout.macro_ready_base + index)
            self.assertEqual(task.trigger_event, layout.phi_rms_base + index)
        for index, task in enumerate(prev_x_tasks):
            self.assertEqual(task.task_type, TaskType.TASK_MHC_GRAD_PREV_X_AND_POST)
            self.assertEqual(task.dependent_event, layout.phi_rms_base + index // 2)
            self.assertEqual(task.trigger_event, layout.final)

        self.assertEqual(config.all_event_num_triggers[layout.init_done], NUM_VECTOR_CORES)
        self.assertEqual(config.all_event_num_triggers[layout.rms_ready_base], 1)
        self.assertEqual(config.all_event_num_triggers[layout.rms_ready_base + 3], 1)
        self.assertEqual(config.all_event_num_triggers[layout.macro_ready_base], 2)
        self.assertEqual(config.all_event_num_triggers[layout.macro_ready_base + 1], 2)
        self.assertEqual(config.all_event_num_triggers[layout.final], 4)
        self.assertEqual(
            list(config.vector_task_indices[:52]),
            list(range(48)) + list(range(50, 54)),
        )

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_order_groups_all_tiles_by_stage(self) -> None:
        """Feature: HyperMegaMhcGrad AIV scheduling.

        Description: Reorder an 80-tile graph by fused AIV stage.
        Expectation: Every stage remains contiguous in the vector task queue.
        """
        graph, topology = build_mega_mhc_grad_graph(
            2560,
            5120,
            num_vector_cores=NUM_VECTOR_CORES,
        )
        config = build_runtime_config(graph, topology, num_cube_cores=NUM_CUBE_CORES)
        order_vector_tasks_by_stage(config)
        task_types = [
            TaskType(config.all_tasks[task_id].task_type)
            for task_id in config.vector_task_indices[:280]
        ]
        self.assertEqual(
            task_types[:NUM_VECTOR_CORES],
            [TaskType.TASK_MHC_GRAD_PREV_A] * NUM_VECTOR_CORES,
        )
        self.assertEqual(task_types[40:120], [TaskType.TASK_RMS_NORM_GRAD] * 80)
        self.assertEqual(task_types[120:200], [TaskType.TASK_MHC_GRAD_PREV_A_AND_MAPPING] * 80)
        self.assertEqual(task_types[200:280], [TaskType.TASK_MHC_GRAD_PREV_X_AND_POST] * 80)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_tail_macro_tile_waits_for_two_producers(self) -> None:
        """Feature: HyperMegaMhcGrad tail dependency.

        Description: Build a three-tile graph with a partial final macro-tile.
        Expectation: Each macro event waits for exactly its contributing AIV tiles.
        """
        graph, topology = build_mega_mhc_grad_graph(
            65,
            5120,
            num_vector_cores=NUM_VECTOR_CORES,
        )
        config = build_runtime_config(graph, topology, num_cube_cores=NUM_CUBE_CORES)
        layout = build_event_layout(3, 2)
        self.assertEqual(config.all_event_num_triggers[layout.macro_ready_base], 2)
        self.assertEqual(config.all_event_num_triggers[layout.macro_ready_base + 1], 1)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_token_tile_is_tunable_and_auto_grows_for_event_capacity(self) -> None:
        """Feature: HyperMegaMhcGrad token tiling.

        Description: Request explicit and capacity-constrained token tiles.
        Expectation: Tiles stay aligned and grow only enough to fit the event table.
        """
        graph, topology = build_mega_mhc_grad_graph(
            128,
            5120,
            token_tile=64,
            num_vector_cores=NUM_VECTOR_CORES,
        )
        config = build_runtime_config(graph, topology, num_cube_cores=NUM_CUBE_CORES)
        order_vector_tasks_by_stage(config)
        layout = build_event_layout(2, 2)

        self.assertEqual(config.task_num, 48)
        self.assertEqual(tuple(config.task_index_num[:3]), (2, 46, 0))
        self.assertEqual(config.all_event_num_triggers[layout.macro_ready_base], 1)
        self.assertEqual(config.all_event_num_triggers[layout.macro_ready_base + 1], 1)
        self.assertEqual(resolve_grad_token_tile(128, 33), 64)
        self.assertEqual(resolve_grad_token_tile(20000, 32), 64)
        self.assertEqual(resolve_grad_token_tile(40000, 32), 128)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_prev_x_post_uses_the_same_token_tile_as_other_aiv_tasks(self) -> None:
        """Feature: HyperMegaMhcGrad fused final stage.

        Description: Inspect final-stage tasks for the representative 8192-token shape.
        Expectation: PrevX and MhcPost stay fused and use the common AIV tile size.
        """
        graph, topology = build_mega_mhc_grad_graph(
            8192,
            5120,
            token_tile=32,
            num_vector_cores=NUM_VECTOR_CORES,
        )
        config = build_runtime_config(graph, topology, num_cube_cores=NUM_CUBE_CORES)
        prev_x_post_tasks = [
            task
            for task in config.all_tasks[: config.task_num]
            if task.task_type == TaskType.TASK_MHC_GRAD_PREV_X_AND_POST
        ]

        self.assertEqual(len(prev_x_post_tasks), 256)
        self.assertEqual({task.task_split_num for task in prev_x_post_tasks}, {256})
        self.assertEqual({task.task_split_value for task in prev_x_post_tasks}, {32})
        self.assertFalse(
            any(
                task.task_type == TaskType.TASK_MHC_POST_GRAD
                for task in config.all_tasks[: config.task_num]
            )
        )

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_reduction_output_initialization_uses_all_vector_workers(self) -> None:
        """Feature: HyperMegaMhcGrad reduction-output initialization.

        Description: Build the representative 4096x5120 training graph.
        Expectation: Output clearing is sharded over every AIV before producers start.
        """
        graph, topology = build_mega_mhc_grad_graph(
            4096,
            5120,
            num_vector_cores=NUM_VECTOR_CORES,
        )
        config = build_runtime_config(graph, topology, num_cube_cores=NUM_CUBE_CORES)
        order_vector_tasks_by_stage(config)
        vector_tasks = [
            config.all_tasks[config.vector_task_indices[index]]
            for index in range(config.task_index_num[1])
        ]
        init_tasks = [
            task
            for task in vector_tasks
            if task.task_type == TaskType.TASK_MHC_GRAD_PREV_A
        ]

        self.assertEqual(len(init_tasks), NUM_VECTOR_CORES)
        self.assertEqual(
            [task.task_index for task in init_tasks],
            list(range(NUM_VECTOR_CORES)),
        )
        self.assertEqual(
            {task.task_split_num for task in init_tasks},
            {NUM_VECTOR_CORES},
        )
        self.assertEqual(config.all_event_num_triggers[0], NUM_VECTOR_CORES)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_rejects_non_positive_shapes(self) -> None:
        """Feature: HyperMegaMhcGrad input validation.

        Description: Build graphs with non-positive token or hidden dimensions.
        Expectation: Both invalid shapes raise descriptive errors.
        """
        with self.assertRaisesRegex(ValueError, "must be positive"):
            build_mega_mhc_grad_graph(0, 5120, num_vector_cores=NUM_VECTOR_CORES)
        with self.assertRaisesRegex(ValueError, "must be positive"):
            build_mega_mhc_grad_graph(128, 0, num_vector_cores=NUM_VECTOR_CORES)
        with self.assertRaisesRegex(ValueError, "must be positive"):
            build_mega_mhc_grad_graph(128, 5120, num_vector_cores=0)


if __name__ == "__main__":
    unittest.main()
