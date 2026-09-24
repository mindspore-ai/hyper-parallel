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
"""Unit tests for MegaMoe SwiGLU clamp task serialization."""

import math
import struct
import unittest

from hyper_parallel.core.multicore.modules.mega_moe.backward.graph import (
    build_backward_graph,
)
from hyper_parallel.core.multicore.modules.mega_moe.forward.graph import (
    build_forward_graph,
)
from hyper_parallel.core.multicore.scheduler.builder import build_runtime_config
from hyper_parallel.core.multicore.scheduler.config import TaskSplitValue, TaskType


class TestSwiGLUClamp(unittest.TestCase):
    """Keep the Python clamp option and device task descriptor synchronized."""

    def test_serializes_clamp_for_forward_and_backward_tasks(self) -> None:
        """Feature: propagate the clamp limit through both task graphs.

        Description: Build forward and backward runtime configurations with the
            same clamp value.
        Expectation: Every SwiGLU task stores the same float32 bit pattern.
        """
        expected_limit = 10.0
        expected_bits = struct.unpack("<I", struct.pack("<f", expected_limit))[0]
        for graph_builder, task_type in (
            (build_forward_graph, TaskType.TASK_SWI_GLU),
            (build_backward_graph, TaskType.TASK_SWI_GLU_GRAD),
        ):
            with self.subTest(task_type=task_type):
                values = TaskSplitValue(
                    tp=1,
                    ep=1,
                    seq_size=128,
                    all_expert_num=4,
                    top_k=2,
                )
                graph = graph_builder(
                    values,
                    hidden_size=16,
                    intermediate_size=8,
                    num_cube_cores=20,
                    swiglu_limit=expected_limit,
                )
                graph.propagate_splits(values)
                runtime_config = build_runtime_config(
                    graph,
                    values,
                    num_cube_cores=20,
                )
                clamp_tasks = [
                    task
                    for task in runtime_config.all_tasks[:runtime_config.task_num]
                    if task.task_type == task_type
                ]

                self.assertTrue(clamp_tasks)
                self.assertTrue(
                    all(task.extra_value_0 == expected_bits for task in clamp_tasks)
                )

    def test_preserves_zero_discriminator_for_legacy_tasks(self) -> None:
        """Feature: preserve the legacy unclamped task discriminator.

        Description: Build forward and backward runtime configurations without
            a clamp limit.
        Expectation: Every SwiGLU task leaves ``extra_value_0`` at zero.
        """
        for graph_builder, task_type in (
            (build_forward_graph, TaskType.TASK_SWI_GLU),
            (build_backward_graph, TaskType.TASK_SWI_GLU_GRAD),
        ):
            with self.subTest(task_type=task_type):
                values = TaskSplitValue(
                    tp=1,
                    ep=1,
                    seq_size=128,
                    all_expert_num=4,
                    top_k=2,
                )
                graph = graph_builder(
                    values,
                    hidden_size=16,
                    intermediate_size=8,
                    num_cube_cores=20,
                )
                graph.propagate_splits(values)
                runtime_config = build_runtime_config(
                    graph,
                    values,
                    num_cube_cores=20,
                )
                legacy_tasks = [
                    task
                    for task in runtime_config.all_tasks[:runtime_config.task_num]
                    if task.task_type == task_type
                ]

                self.assertTrue(legacy_tasks)
                self.assertTrue(all(task.extra_value_0 == 0 for task in legacy_tasks))

    def test_rejects_non_positive_or_non_finite_clamp_limit(self) -> None:
        """Feature: reject invalid device clamp descriptors.

        Description: Build tasks with non-positive, non-finite, underflowing,
            and overflowing clamp values.
        Expectation: Runtime configuration construction raises ``ValueError``.
        """
        for clamp_limit in (
            0.0,
            -1.0,
            1e-50,
            1e39,
            math.inf,
            math.nan,
            True,
            "10",
        ):
            with (
                self.subTest(clamp_limit=clamp_limit),
                self.assertRaisesRegex(ValueError, "clamp_limit"),
            ):
                values = TaskSplitValue(
                    tp=1,
                    ep=1,
                    seq_size=128,
                    all_expert_num=4,
                    top_k=2,
                )
                graph = build_forward_graph(
                    values,
                    hidden_size=16,
                    intermediate_size=8,
                    num_cube_cores=20,
                    swiglu_limit=clamp_limit,
                )
                graph.propagate_splits(values)
                build_runtime_config(graph, values, num_cube_cores=20)


if __name__ == "__main__":
    unittest.main()
