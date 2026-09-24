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
"""RuntimeConfig tests for the pure AIC/AIV HyperMegaMhc task graph."""

import unittest

from hyper_parallel.core.multicore.modules.mega_mhc.graph import (
    POST_DONE_EVENT,
    build_event_layout,
    build_mega_mhc_graph,
    resolve_token_tile,
)
from hyper_parallel.core.multicore.scheduler.builder import build_runtime_config
from hyper_parallel.core.multicore.scheduler.config import TaskAiCoreType, TaskType
from tests.common.mark_utils import arg_mark


class TestMegaMhcGraph(unittest.TestCase):
    """Validate token-tile ordering, ring reuse, and per-stage dependencies."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_builds_six_stage_pure_aic_aiv_pipeline(self) -> None:
        """Validate the six-stage task graph.

        Feature: HyperMegaMhc forward scheduling.
        Description: Build a multi-tile graph and inspect every task dependency.
        Expectation: AIC/AIV queues, event counts, and task ordering match the pipeline.
        """
        graph, topology = build_mega_mhc_graph(4096, 7168, 20, token_tile=128)
        config = build_runtime_config(graph, topology, num_cube_cores=20)
        tile_count = 32
        layout = build_event_layout(tile_count)

        self.assertEqual(
            [operator.name for operator in graph.topological_sort()],
            [
                "mhc_post",
                "mhc_norm_cast",
                "mhc_projection",
                "mhc_input_mix",
                "mhc_mapping",
                "mhc_rms_norm",
            ],
        )
        self.assertEqual(config.task_num, 6 * tile_count)
        self.assertEqual(tuple(config.task_index_num[:3]), (tile_count, 5 * tile_count, 0))
        self.assertEqual(config.all_event_num_triggers[POST_DONE_EVENT], tile_count)
        self.assertEqual(config.all_event_num_triggers[layout.final], 2 * tile_count)

        for tile_index in range(tile_count):
            post = config.all_tasks[tile_index]
            norm = config.all_tasks[tile_count + tile_index]
            projection = config.all_tasks[2 * tile_count + tile_index]
            input_mix = config.all_tasks[3 * tile_count + tile_index]
            mapping = config.all_tasks[4 * tile_count + tile_index]
            rms_norm = config.all_tasks[5 * tile_count + tile_index]

            self.assertEqual(post.task_type, TaskType.TASK_MHC_POST)
            self.assertEqual(post.task_aicore_type, TaskAiCoreType.TASK_AICORE_VECTOR)
            self.assertEqual(norm.task_type, TaskType.TASK_MHC_NORM_CAST)
            self.assertEqual(norm.task_aicore_type, TaskAiCoreType.TASK_AICORE_VECTOR)
            expected_norm_dependency = layout.post_done
            if tile_index >= 20:
                expected_norm_dependency = layout.projection_base + tile_index - 20
            self.assertEqual(norm.dependent_event, expected_norm_dependency)
            self.assertEqual(norm.trigger_event, layout.norm_base + tile_index)
            self.assertEqual(projection.task_type, TaskType.TASK_MHC_PROJECTION)
            self.assertEqual(projection.task_aicore_type, TaskAiCoreType.TASK_AICORE_CUBE)
            self.assertEqual(projection.dependent_event, layout.norm_base + tile_index)
            self.assertEqual(projection.trigger_event, layout.projection_base + tile_index)
            self.assertEqual(input_mix.task_type, TaskType.TASK_MHC_INPUT_MIX)
            self.assertEqual(input_mix.dependent_event, layout.norm_base + tile_index)
            self.assertEqual(input_mix.trigger_event, layout.input_mix_base + tile_index)
            self.assertEqual(rms_norm.task_type, TaskType.TASK_RMS_NORM)
            self.assertEqual(rms_norm.dependent_event, layout.input_mix_base + tile_index)
            self.assertEqual(rms_norm.trigger_event, layout.final)
            self.assertEqual(mapping.task_type, TaskType.TASK_MHC_MAPPING)
            self.assertEqual(mapping.dependent_event, layout.projection_base + tile_index)
            self.assertEqual(mapping.trigger_event, layout.final)
            self.assertEqual(config.all_event_num_triggers[layout.projection_base + tile_index], 1)
            self.assertEqual(config.all_event_num_triggers[layout.input_mix_base + tile_index], 1)

        cube_ids = tuple(config.cube_task_indices[: config.task_index_num[0]])
        vector_ids = tuple(config.vector_task_indices[: config.task_index_num[1]])
        self.assertEqual(cube_ids, tuple(range(2 * tile_count, 3 * tile_count)))
        expected_vector_ids = (
            tuple(range(0, tile_count))
            + tuple(range(tile_count, 2 * tile_count))
            + tuple(range(3 * tile_count, 4 * tile_count))
            + tuple(range(5 * tile_count, 6 * tile_count))
            + tuple(range(4 * tile_count, 5 * tile_count))
        )
        self.assertEqual(vector_ids, expected_vector_ids)
        self.assertNotIn(
            TaskAiCoreType.TASK_AICORE_MIX,
            {config.all_tasks[task_id].task_aicore_type for task_id in range(config.task_num)},
        )

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_schedules_aiv_pipeline_in_core_sized_waves(self) -> None:
        """Validate AIV wave ordering.

        Feature: HyperMegaMhc AIV scheduling.
        Description: Build an 80-tile graph on 40 vector workers.
        Expectation: NormCast, InputMix, and RMSNorm execute in core-sized waves.
        """
        graph, topology = build_mega_mhc_graph(2560, 7168, 20, token_tile=32)
        config = build_runtime_config(graph, topology, num_cube_cores=20)
        tile_count = 80
        vector_ids = tuple(config.vector_task_indices[: config.task_index_num[1]])
        expected_ids = list(range(tile_count))
        for wave_start in range(0, tile_count, 40):
            wave_end = wave_start + 40
            expected_ids.extend(range(tile_count + wave_start, tile_count + wave_end))
            expected_ids.extend(range(3 * tile_count + wave_start, 3 * tile_count + wave_end))
            expected_ids.extend(range(5 * tile_count + wave_start, 5 * tile_count + wave_end))
        expected_ids.extend(range(4 * tile_count, 5 * tile_count))
        self.assertEqual(vector_ids, tuple(expected_ids))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_defaults_to_measured_32_token_tile(self) -> None:
        """Validate the default token tile.

        Feature: HyperMegaMhc token tiling.
        Description: Build a graph without an explicit tile size.
        Expectation: The measured 32-token default is selected.
        """
        graph, _ = build_mega_mhc_graph(1024, 7168, 20)
        post = graph.get_op("mhc_post")
        self.assertEqual(post.task_num, 32)
        self.assertEqual(post.split_value, 32)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_enlarges_tile_to_keep_large_t_event_count_bounded(self) -> None:
        """Validate large-token event-capacity protection.

        Feature: HyperMegaMhc token tiling.
        Description: Resolve a tile for a shape that would exceed event capacity.
        Expectation: The tile grows and the resulting graph fits the event array.
        """
        token_tile = resolve_token_tile(65536, 128, 20)
        self.assertEqual(token_tile, 224)
        graph, _ = build_mega_mhc_graph(65536, 7168, 20, token_tile=128)
        tile_count = graph.get_op("mhc_post").task_num
        self.assertLessEqual(build_event_layout(tile_count).event_count, 1024)


if __name__ == "__main__":
    unittest.main()
