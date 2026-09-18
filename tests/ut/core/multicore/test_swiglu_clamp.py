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

import ctypes
import math
import struct
import unittest

from hyper_parallel.core.multicore.modules.mega_moe.backward.graph import (
    build_backward_graph,
)
from hyper_parallel.core.multicore.modules.mega_moe.backward.tiling_tables import (
    get_clipped_swiglu_grad_tiling_bytes,
)
from hyper_parallel.core.multicore.modules.mega_moe.forward.graph import (
    build_forward_graph,
)
from hyper_parallel.core.multicore.modules.mega_moe.forward.tiling_tables import (
    get_clipped_swiglu_tiling_bytes,
)
from hyper_parallel.core.multicore.scheduler.builder import build_runtime_config
from hyper_parallel.core.multicore.scheduler.config import (
    ClippedSwiGluTilingDataC,
    TaskSplitValue,
    TaskType,
)


class TestSwiGLUClamp(unittest.TestCase):
    """Keep the Python clamp option and device task descriptor synchronized."""

    def test_serializes_official_clipped_swiglu_tiling(self) -> None:
        """Feature: serialize the official clipped SwiGLU tiling ABI.

        Description: Build forward and backward tiling records for one expert
            tile and decode their first worker slot.
        Expectation: Both records use the official 80-byte layout and carry the
            configured dimensions and clamp attributes.
        """
        expected_size = ctypes.sizeof(ClippedSwiGluTilingDataC)
        self.assertEqual(expected_size, 80)
        for tiling_fn in (
            get_clipped_swiglu_tiling_bytes,
            get_clipped_swiglu_grad_tiling_bytes,
        ):
            with self.subTest(tiling_fn=tiling_fn.__name__):
                tiling_bytes = tiling_fn(
                    128,
                    intermediate_size=128,
                    clamp_limit=10.0,
                )
                self.assertEqual(len(tiling_bytes), expected_size * 49)
                tiling = ClippedSwiGluTilingDataC.from_buffer_copy(tiling_bytes)
                self.assertEqual(tiling.core_num_all, 1)
                self.assertEqual(tiling.dim_batch_size, 128)
                self.assertEqual(tiling.dim_2h, 256)
                self.assertEqual(tiling.ub_max_pair, 2048)
                self.assertEqual(tiling.is_long_h, 0)
                self.assertEqual(tiling.is_group, 0)
                self.assertEqual(tiling.is_interleaved, 0)
                self.assertEqual(tiling.alpha, 1.0)
                self.assertEqual(tiling.limit, 10.0)
                self.assertEqual(tiling.bias, 0.0)
                self.assertEqual(tiling.group_num, 0)

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
